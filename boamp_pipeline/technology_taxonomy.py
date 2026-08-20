"""Business technology taxonomy: the frozen classes and the annotated corpus.

BOAMP publishes an administrative vocabulary (CPV divisions, BOAMP
"descripteurs"), not the business-oriented technology class the internship
needs. This module is the data layer of the supervised classifier that learns
that class from procurement text:

* the frozen taxonomy and its ordering;
* light text normalisation that *keeps* French accents and technical acronyms;
* the leakage-preventing grouping of related notices;
* the annotated corpus, its audit, and its frozen cross-validation folds.

The modelling layer lives in :mod:`boamp_pipeline.technology_models` and the
deployment and reporting layer in :mod:`boamp_pipeline.technology_evidence`.
Every function here is importable, so
``notebooks/15_technology_taxonomy_classification.ipynb`` shows the real
construction rather than a description of it.

Nothing here reads or writes the frozen linkage, survival, or trend artifacts.
The only pipeline artifact consumed is ``episode_membership.parquet``, read to
*reuse* the canonical episode reconstruction as the grouping key rather than
inventing a second notion of "the same procurement".

Two deliberate departures from the linkage code are worth naming, because both
modules vectorise French procurement text and a reader could reasonably expect
them to share one helper:

1. :func:`boamp_pipeline.linkage.normalize_text` strips accents and every
   non-alphanumeric character. That is correct for *similarity ranking*, where
   "systeme" and "système" must collide. It is wrong here: the taxonomy is
   learned from words like ``cybersécurité``, ``télécom`` and ``métier``, and
   flattening them discards the orthography a French reader uses to tell the
   classes apart. :func:`normalize_objet` therefore only lowercases, repairs
   mojibake, and collapses whitespace.

2. Linkage uses a character analyser to survive typos across two documents.
   Classification searches word unigrams and unigrams-plus-bigrams, because the
   evidence for a class can be phrasal -- "intelligence artificielle", "logiciel
   métier", "business intelligence" -- and bigrams keep those phrases
   addressable if the inner cross-validation finds they help. On this corpus
   every fold selected unigrams alone; the bigram option is part of the search
   space, not of the deployed model.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from boamp_pipeline.episodes import DisjointSet
from boamp_pipeline.linkage import cpv_divisions, is_digital, normalize_text, parse_json_list
from boamp_pipeline.standardize import standardize_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data/processed/boamp"
TECHNOLOGY = PROCESSED / "technology"

#: The annotation file as delivered. The directory name is the one the corpus
#: arrived under; it is referenced from this single constant so the path never
#: has to be retyped anywhere else in the project.
ANNOTATION_FILE = (
    PROJECT_ROOT / "data/reference/tech classification"
    / "boamp_nlp_annotation_500_final_2015_2025.csv"
)

CORPUS_VERSION = "boamp_technology_corpus_v1.0"

#: Fewer words than this and ``objet`` cannot carry enough evidence for any
#: model; the rows are kept and flagged rather than dropped, because dropping
#: them would quietly improve every metric in the report.
SHORT_TEXT_WORDS = 4

TAXONOMY_VERSION = "boamp_technology_taxonomy_v1.0"

#: Seed used everywhere a random choice is made. One value, so a rerun of any
#: stage reproduces the same folds, the same subsamples, and the same solver
#: paths.
RANDOM_SEED = 20260819

#: Three folds, not five or ten. With 500 labelled notices, 11 classes, and
#: group isolation, five folds leave the rarest class with roughly one
#: observation per held-out fold, and its per-fold F1 becomes a coin flip
#: rather than a measurement.
N_SPLITS = 3

#: Substantive business classes, in the order used by every table and figure.
SUBSTANTIVE_CLASSES = (
    "CLOUD_HOSTING",
    "CYBERSECURITY",
    "NETWORK_TELECOM",
    "IT_INFRASTRUCTURE",
    "BUSINESS_SOFTWARE",
    "DATA_BI",
    "AI",
    "IT_SERVICES",
)

#: Fallback classes. They are real annotation decisions, not missing values:
#: MIXED marks a procurement with no dominant technology, OTHER_DIGITAL a
#: digital purchase outside the eight substantive classes, and OTHER a notice
#: that carries a digital CPV without being a technology procurement.
FALLBACK_CLASSES = ("MIXED", "OTHER_DIGITAL", "OTHER")

CLASS_ORDER = SUBSTANTIVE_CLASSES + FALLBACK_CLASSES

#: Below this per-class support, a per-class precision/recall/F1 is reported but
#: must not be read as an estimate of how well the class is predicted. The value
#: is a reporting convention fixed before any model ran, not a tuned cutoff.
MIN_RELIABLE_SUPPORT = 10

#: Character-level cosine at or above which two notices are treated as the same
#: procurement family for fold assignment. Fixed a priori at the value the
#: linkage code already uses for "the same text", and applied in the
#: conservative direction only: merging two genuinely distinct procurements
#: costs a little training signal, while splitting one across folds inflates
#: every metric in the report.
NEAR_DUPLICATE_THRESHOLD = 0.80

#: Natures that open a competition. Mirrors ``episodes.choose_origin`` so the
#: deployment text is drawn from the same notice the episode layer already
#: treats as the origin of the procurement.
COMPETITION_NATURES = ("APPEL_OFFRE", "PRE-INFORMATION", "PERIODIQUE")

#: Temporal robustness split. Fixed before any temporal metric was computed.
TEMPORAL_TRAIN_YEARS = (2015, 2022)
TEMPORAL_TEST_YEARS = (2023, 2025)

_WS_RE = re.compile(r"\s+")
#: Latin-1/CP1252 bytes that survived a bad decode upstream. ``\x9c`` is the
#: French ligature oe, which appears in "mise en œuvre" -- a phrase common
#: enough in this corpus that leaving it broken splits one token into two.
_MOJIBAKE = {
    "\x9c": "oe",
    "\x9d": "",
    "\x92": "'",
    "\x91": "'",
    "\x93": '"',
    "\x94": '"',
    "\x96": "-",
    "\x97": "-",
    "œ": "oe",
}


def normalize_objet(value: Any) -> str:
    """Lowercase and tidy procurement text without destroying its vocabulary.

    Accents, hyphens and apostrophes are preserved: the tokeniser used by the
    models splits on them, so they cost nothing here, and keeping them means a
    reader can check any training document against the published notice.
    """
    if value is None:
        return ""
    text = str(value)
    if text.strip().lower() in {"", "nan", "none", "null"}:
        return ""
    for bad, good in _MOJIBAKE.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch if ch.isprintable() or ch.isspace() else " " for ch in text)
    return _WS_RE.sub(" ", text).strip().lower()


def cpv_tokens(codes: Sequence[Any]) -> list[str]:
    """Hierarchical CPV tokens for the administrative benchmark.

    One CPV code yields its division, group, class and full code, so a linear
    model can back off to a coarser level when a full code is unseen. This is
    the entire feature engineering the benchmark gets: the benchmark exists to
    measure what the official vocabulary already carries, and tuning it into a
    strong model would answer a different question.
    """
    tokens: list[str] = []
    for code in codes:
        digits = re.sub(r"\D", "", str(code or ""))
        if len(digits) < 8:
            continue
        digits = digits[:8]
        tokens.extend([f"d{digits[:2]}", f"g{digits[:3]}", f"c{digits[:4]}", f"f{digits}"])
    return sorted(set(tokens))


def descriptor_tokens(values: Any) -> list[str]:
    """Normalised BOAMP descriptor labels, joined into single tokens."""
    tokens = []
    for value in parse_json_list(values):
        normalized = normalize_text(value).replace(" ", "_")
        if normalized:
            tokens.append(f"desc_{normalized}")
    return sorted(set(tokens))


def near_duplicate_pairs(
    texts: Sequence[str], threshold: float = NEAR_DUPLICATE_THRESHOLD
) -> list[tuple[int, int, float]]:
    """Index pairs whose texts are near-identical under a character analyser.

    The character analyser is used rather than the word analyser the classifier
    sees: republished notices differ by case, punctuation and small edits, and
    character n-grams are what detect that. Accents are stripped *here only*,
    for the same reason.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    stripped = [normalize_text(text) for text in texts]
    if not any(stripped):
        return []
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(stripped)
    similarity = (matrix @ matrix.T).toarray()
    np.fill_diagonal(similarity, 0.0)
    left, right = np.where(similarity >= threshold)
    return [
        (int(i), int(j), float(similarity[i, j]))
        for i, j in zip(left, right)
        if i < j
    ]


def build_group_ids(
    notice_ids: Sequence[str],
    episode_ids: Sequence[str],
    texts: Sequence[str],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> tuple[list[str], list[tuple[int, int, float]]]:
    """Assign every labelled notice to one procurement family.

    Two notices join the same family when the canonical episode reconstruction
    already placed them in one episode, or when their objects are near-identical
    under :func:`near_duplicate_pairs`. The second rule is required: the episode
    layer links notices by contract folder, declared links, and buyer-plus-
    reference, none of which fire when a buyer re-runs the same tender years
    later under a new reference. Those pairs are distinct procurements but not
    independent documents, and splitting one across folds would let a model be
    scored on text it had already read.

    Returns the group label per notice and the near-duplicate pairs that were
    merged, so the audit can report exactly what the second rule did.
    """
    disjoint = DisjointSet(list(notice_ids))
    by_episode: dict[str, list[str]] = {}
    for notice_id, episode in zip(notice_ids, episode_ids):
        if episode:
            by_episode.setdefault(str(episode), []).append(str(notice_id))
    for members in by_episode.values():
        for other in members[1:]:
            disjoint.union(members[0], other)

    pairs = near_duplicate_pairs(texts, threshold)
    for left, right, _ in pairs:
        disjoint.union(str(notice_ids[left]), str(notice_ids[right]))

    roots = {}
    groups: list[str] = []
    for notice_id in notice_ids:
        root = disjoint.find(str(notice_id))
        roots.setdefault(root, f"GRP-{len(roots):04d}")
        groups.append(roots[root])
    return groups, pairs


def text_pipeline(estimator: Any, **tfidf_options: Any):
    """TF-IDF word unigrams+bigrams feeding one linear estimator.

    Returned as a :class:`~sklearn.pipeline.Pipeline` so that the vocabulary,
    document frequencies and IDF weights are fitted inside each training fold.
    Vectorising before the split would let held-out documents influence their
    own features.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    options = {
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.9,
        "sublinear_tf": True,
        "strip_accents": None,
        "lowercase": True,
    }
    options.update(tfidf_options)
    return Pipeline([("tfidf", TfidfVectorizer(**options)), ("clf", estimator)])


def token_pipeline(estimator: Any, **tfidf_options: Any):
    """Administrative-token benchmark: pre-tokenised codes, no text analysis."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    options = {
        "analyzer": lambda tokens: tokens,
        "min_df": 1,
        "sublinear_tf": False,
    }
    options.update(tfidf_options)
    return Pipeline([("tfidf", TfidfVectorizer(**options)), ("clf", estimator)])


def class_support(labels: Iterable[str]) -> dict[str, int]:
    """Label counts in the frozen class order, including classes with zero."""
    counts = {label: 0 for label in CLASS_ORDER}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def reliable_classes(support: dict[str, int]) -> list[str]:
    """Classes whose support clears :data:`MIN_RELIABLE_SUPPORT`."""
    return [label for label in CLASS_ORDER if support.get(label, 0) >= MIN_RELIABLE_SUPPORT]


# ---------------------------------------------------------------------------
# The annotated corpus: audit, grouping, and frozen folds
# ---------------------------------------------------------------------------


def load_annotations() -> pd.DataFrame:
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"annotated technology corpus not found at {ANNOTATION_FILE}"
        )
    # utf-8-sig: the delivered file carries a byte-order mark, which would
    # otherwise become part of the first column name.
    frame = pd.read_csv(ANNOTATION_FILE, dtype=str, encoding="utf-8-sig")
    frame = frame.fillna("")
    required = {"idweb", "objet", "label", "dateparution"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"annotation file is missing required columns: {sorted(missing)}")
    return frame


def standardize_annotations(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse each annotated row with the project's canonical BOAMP parser.

    The annotation file preserves the raw ``donnees`` and ``gestion`` payloads,
    so the same parser that built the study cohort can extract CPV codes, notice
    text and declared links here. Re-deriving them means the corpus and the
    cohort cannot drift apart on the definition of "the CPV of a notice".
    """
    records = [standardize_record(row, ANNOTATION_FILE.name) for row in frame.to_dict("records")]
    parsed = pd.DataFrame(records)
    return parsed[
        [
            "idweb", "main_cpv", "all_cpvs_json", "cpv_divisions_json",
            "notice_text", "linked_notice_ids_json", "buyer_siren",
            "buyer_name_normalized", "buyer_region", "grand_ouest_flag",
            "publication_date", "nature", "type_avis",
        ]
    ]


def attach_episodes(frame: pd.DataFrame) -> pd.DataFrame:
    membership_path = PROCESSED / "episode_membership.parquet"
    if not membership_path.exists():
        raise FileNotFoundError(
            f"{membership_path} is required to reuse the canonical episode grouping"
        )
    membership = pd.read_parquet(membership_path, columns=["idweb", "episode_id"])
    merged = frame.merge(membership, on="idweb", how="left")
    merged["episode_id"] = merged["episode_id"].fillna("")
    return merged


def build_corpus() -> tuple[pd.DataFrame, list[tuple[int, int, float]]]:
    raw = load_annotations()
    parsed = standardize_annotations(raw)
    corpus = raw.merge(parsed, on="idweb", how="left", suffixes=("", "_parsed"))
    corpus = attach_episodes(corpus)

    corpus["text"] = corpus["objet"].map(normalize_objet)
    corpus["publication_date"] = pd.to_datetime(corpus["dateparution"], errors="coerce")
    corpus["year"] = corpus["publication_date"].dt.year
    corpus["cpv_codes_json"] = corpus["all_cpvs_json"]
    corpus["cpv_token_list"] = corpus["all_cpvs_json"].map(
        lambda value: cpv_tokens(parse_json_list(value))
    )
    corpus["descriptor_token_list"] = corpus["descripteur_libelle"].map(descriptor_tokens)
    corpus["cpv_main_division"] = corpus["main_cpv"].astype(str).str[:2]
    corpus["cpv_digital_flag"] = corpus["all_cpvs_json"].map(
        lambda value: is_digital(parse_json_list(value))
    )
    corpus["cpv_divisions_list"] = corpus["all_cpvs_json"].map(
        lambda value: sorted(cpv_divisions(parse_json_list(value)))
    )
    corpus["declared_link_count"] = corpus["linked_notice_ids_json"].map(
        lambda value: len(parse_json_list(value))
    )
    corpus["text_word_count"] = corpus["text"].str.split().map(len)
    corpus["text_char_count"] = corpus["text"].str.len()

    groups, pairs = build_group_ids(
        corpus["idweb"].tolist(),
        corpus["episode_id"].tolist(),
        corpus["objet"].tolist(),
        NEAR_DUPLICATE_THRESHOLD,
    )
    corpus["group_id"] = groups
    corpus["corpus_version"] = CORPUS_VERSION
    corpus["taxonomy_version"] = TAXONOMY_VERSION
    return corpus, pairs


def assign_folds(corpus: pd.DataFrame) -> pd.DataFrame:
    """Split groups into folds, stratified by class as far as groups allow.

    ``StratifiedGroupKFold`` keeps every group whole and balances the class
    distribution across folds as a secondary objective. Perfect stratification
    is unreachable here -- seven AI notices cannot be spread evenly over three
    folds while their groups stay intact -- so the realised per-fold support is
    written out alongside the assignment rather than assumed.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold = np.full(len(corpus), -1, dtype=int)
    for index, (_, test_index) in enumerate(
        splitter.split(corpus["text"], corpus["label"], corpus["group_id"])
    ):
        fold[test_index] = index
    if (fold < 0).any():
        raise RuntimeError("fold assignment left rows unassigned")

    folds = corpus[["idweb", "group_id", "label", "year", "episode_id"]].copy()
    folds["fold"] = fold
    folds["n_splits"] = N_SPLITS
    folds["random_seed"] = RANDOM_SEED
    folds["corpus_version"] = CORPUS_VERSION

    crossing = folds.groupby("group_id")["fold"].nunique()
    if int(crossing.max()) != 1:
        raise RuntimeError("a related-notice group was split across folds")
    return folds


def duplicate_report(corpus: pd.DataFrame, pairs: list[tuple[int, int, float]]) -> pd.DataFrame:
    rows = []
    for left, right, score in pairs:
        a, b = corpus.iloc[left], corpus.iloc[right]
        rows.append(
            {
                "left_idweb": a["idweb"],
                "right_idweb": b["idweb"],
                "similarity": round(score, 4),
                "same_episode": bool(a["episode_id"] and a["episode_id"] == b["episode_id"]),
                "same_buyer_name": bool(
                    a["buyer_name_normalized"]
                    and a["buyer_name_normalized"] == b["buyer_name_normalized"]
                ),
                "left_year": int(a["year"]) if pd.notna(a["year"]) else None,
                "right_year": int(b["year"]) if pd.notna(b["year"]) else None,
                "left_label": a["label"],
                "right_label": b["label"],
                "label_conflict": bool(a["label"] != b["label"]),
                "left_objet": a["objet"],
                "right_objet": b["objet"],
            }
        )
    return pd.DataFrame(rows).sort_values("similarity", ascending=False) if rows else pd.DataFrame(
        columns=["left_idweb", "right_idweb", "similarity"]
    )


def row_audit(corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per labelled notice with every flag the audit can raise."""
    group_sizes = corpus.groupby("group_id")["idweb"].transform("size")
    audit = pd.DataFrame(
        {
            "idweb": corpus["idweb"],
            "episode_id": corpus["episode_id"],
            "group_id": corpus["group_id"],
            "group_size": group_sizes,
            "year": corpus["year"],
            "label": corpus["label"],
            "nature": corpus["nature"],
            "buyer_region": corpus["buyer_region"],
            "grand_ouest_flag": corpus["grand_ouest_flag"],
            "main_cpv": corpus["main_cpv"],
            "cpv_main_division": corpus["cpv_main_division"],
            "cpv_digital_flag": corpus["cpv_digital_flag"],
            "cpv_token_count": corpus["cpv_token_list"].map(len),
            "descriptor_token_count": corpus["descriptor_token_list"].map(len),
            "declared_link_count": corpus["declared_link_count"],
            "text_word_count": corpus["text_word_count"],
            "text_char_count": corpus["text_char_count"],
            "objet": corpus["objet"],
        }
    )
    audit["flag_missing_label"] = corpus["label"].str.strip().eq("")
    audit["flag_unknown_label"] = ~corpus["label"].isin(CLASS_ORDER)
    audit["flag_missing_text"] = corpus["text"].str.strip().eq("")
    audit["flag_short_text"] = corpus["text_word_count"] < SHORT_TEXT_WORDS
    audit["flag_no_cpv"] = corpus["cpv_token_list"].map(len).eq(0)
    audit["flag_no_descriptor"] = corpus["descriptor_token_list"].map(len).eq(0)
    audit["flag_non_digital_cpv"] = ~corpus["cpv_digital_flag"]
    audit["flag_outside_grand_ouest"] = ~corpus["grand_ouest_flag"].astype(bool)
    audit["flag_rectificatif"] = corpus["nature"].eq("RECTIFICATIF")
    audit["flag_in_multi_notice_group"] = group_sizes > 1
    # A digital CPV on a notice labelled OTHER is the CPV vocabulary claiming a
    # purchase is digital when the annotator judged it is not. It is recorded
    # because it is evidence about CPV, not evidence of an annotation error.
    audit["flag_other_with_digital_cpv"] = corpus["label"].eq("OTHER") & corpus["cpv_digital_flag"]
    return audit


def summarize(
    corpus: pd.DataFrame, folds: pd.DataFrame, audit: pd.DataFrame, duplicates: pd.DataFrame
) -> dict[str, Any]:
    counts = corpus["label"].value_counts()
    ordered = {label: int(counts.get(label, 0)) for label in CLASS_ORDER}
    unexpected = sorted(set(corpus["label"]) - set(CLASS_ORDER))
    group_sizes = corpus.groupby("group_id").size()
    fold_support = (
        folds.pivot_table(index="label", columns="fold", values="idweb", aggfunc="count")
        .reindex(CLASS_ORDER)
        .fillna(0)
        .astype(int)
    )
    conflicting_groups = (
        corpus.groupby("group_id")["label"].nunique().pipe(lambda s: s[s > 1]).index.tolist()
    )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "corpus_version": CORPUS_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "annotation_file": str(ANNOTATION_FILE.relative_to(PROJECT_ROOT)),
        "rows": int(len(corpus)),
        "unique_idweb": int(corpus["idweb"].nunique()),
        "duplicate_idweb": int(len(corpus) - corpus["idweb"].nunique()),
        "labels": {
            "observed_classes": sorted(set(corpus["label"])),
            "unexpected_classes": unexpected,
            "counts": ordered,
            "proportions": {k: round(v / len(corpus), 4) for k, v in ordered.items()},
            "missing_label_rows": int(audit["flag_missing_label"].sum()),
        },
        "text": {
            "missing_objet_rows": int(audit["flag_missing_text"].sum()),
            "short_objet_rows": int(audit["flag_short_text"].sum()),
            "short_objet_threshold_words": SHORT_TEXT_WORDS,
            "word_count": {
                "min": int(corpus["text_word_count"].min()),
                "p05": int(corpus["text_word_count"].quantile(0.05)),
                "median": int(corpus["text_word_count"].median()),
                "mean": round(float(corpus["text_word_count"].mean()), 2),
                "p95": int(corpus["text_word_count"].quantile(0.95)),
                "max": int(corpus["text_word_count"].max()),
            },
            "char_count": {
                "min": int(corpus["text_char_count"].min()),
                "median": int(corpus["text_char_count"].median()),
                "max": int(corpus["text_char_count"].max()),
            },
            "exact_duplicate_text_rows": int(corpus["text"].duplicated(keep=False).sum()),
        },
        "years": {
            "min": int(corpus["year"].min()),
            "max": int(corpus["year"].max()),
            "counts": {str(k): int(v) for k, v in sorted(corpus["year"].value_counts().items())},
            "classes_missing_from_a_year": int(
                (pd.crosstab(corpus["label"], corpus["year"]) == 0).sum().sum()
            ),
        },
        "coverage": {
            "rows_with_cpv": int((~audit["flag_no_cpv"]).sum()),
            "rows_with_descriptor": int((~audit["flag_no_descriptor"]).sum()),
            "rows_with_digital_cpv": int(corpus["cpv_digital_flag"].sum()),
            "rows_in_grand_ouest": int(corpus["grand_ouest_flag"].astype(bool).sum()),
            "cpv_main_division_counts": {
                str(k): int(v) for k, v in corpus["cpv_main_division"].value_counts().items()
            },
            "nature_counts": {
                str(k): int(v) for k, v in corpus["nature"].value_counts().items()
            },
        },
        "grouping": {
            "rule": (
                "union of (a) shared canonical episode_id and (b) objet character "
                f"cosine >= {NEAR_DUPLICATE_THRESHOLD}"
            ),
            "groups": int(corpus["group_id"].nunique()),
            "singleton_groups": int((group_sizes == 1).sum()),
            "multi_notice_groups": int((group_sizes > 1).sum()),
            "largest_group": int(group_sizes.max()),
            "notices_in_multi_notice_groups": int(group_sizes[group_sizes > 1].sum()),
            "episode_groups_before_text_merge": int(
                corpus.loc[corpus["episode_id"] != "", "episode_id"].nunique()
                + int((corpus["episode_id"] == "").sum())
            ),
            "near_duplicate_pairs": int(len(duplicates)),
            "near_duplicate_pairs_across_episodes": int(
                (~duplicates["same_episode"]).sum()
            ) if len(duplicates) else 0,
            "near_duplicate_pairs_with_label_conflict": int(
                duplicates["label_conflict"].sum()
            ) if len(duplicates) else 0,
            "groups_with_conflicting_labels": len(conflicting_groups),
            "conflicting_group_ids": conflicting_groups,
            "notices_with_declared_links": int((corpus["declared_link_count"] > 0).sum()),
            "rectificatif_notices": int(audit["flag_rectificatif"].sum()),
        },
        "folds": {
            "n_splits": N_SPLITS,
            "random_seed": RANDOM_SEED,
            "rows_per_fold": {str(k): int(v) for k, v in folds["fold"].value_counts().sort_index().items()},
            "groups_per_fold": {
                str(k): int(v)
                for k, v in folds.groupby("fold")["group_id"].nunique().sort_index().items()
            },
            "class_support_per_fold": {
                str(label): {str(fold): int(value) for fold, value in row.items()}
                for label, row in fold_support.iterrows()
            },
            "classes_absent_from_some_fold": [
                str(label) for label, row in fold_support.iterrows() if (row == 0).any() and row.sum() > 0
            ],
        },
        "anomalies": {
            "other_label_with_digital_cpv": int(audit["flag_other_with_digital_cpv"].sum()),
            "rows_outside_grand_ouest": int(audit["flag_outside_grand_ouest"].sum()),
            "rows_without_digital_cpv": int(audit["flag_non_digital_cpv"].sum()),
        },
    }


def build_corpus_artifacts(force: bool = True) -> dict[str, Any]:
    corpus_path = TECHNOLOGY / "technology_corpus.parquet"
    if corpus_path.exists() and not force:
        raise FileExistsError(f"{corpus_path} already exists. Use --force to rebuild.")
    TECHNOLOGY.mkdir(parents=True, exist_ok=True)

    corpus, pairs = build_corpus()
    folds = assign_folds(corpus)
    audit = row_audit(corpus)
    duplicates = duplicate_report(corpus, pairs)
    summary = summarize(corpus, folds, audit, duplicates)

    stored = corpus.drop(columns=["cpv_token_list", "descriptor_token_list", "cpv_divisions_list"]).copy()
    stored["cpv_tokens_json"] = corpus["cpv_token_list"].map(json.dumps)
    stored["descriptor_tokens_json"] = corpus["descriptor_token_list"].map(json.dumps)
    stored = stored.astype({col: "string" for col in stored.columns if stored[col].dtype == object})
    stored["year"] = corpus["year"]
    stored["text_word_count"] = corpus["text_word_count"]
    stored["text_char_count"] = corpus["text_char_count"]
    stored["declared_link_count"] = corpus["declared_link_count"]
    stored["cpv_digital_flag"] = corpus["cpv_digital_flag"]
    stored["fold"] = folds["fold"].to_numpy()
    stored.to_parquet(corpus_path, index=False, compression="zstd")

    audit.to_csv(TECHNOLOGY / "annotation_audit.csv", index=False, encoding="utf-8")
    folds.to_csv(TECHNOLOGY / "nlp_cv_folds.csv", index=False, encoding="utf-8")
    duplicates.to_csv(TECHNOLOGY / "annotation_near_duplicates.csv", index=False, encoding="utf-8")

    class_summary = pd.DataFrame(
        {
            "technology": CLASS_ORDER,
            "n": [summary["labels"]["counts"][label] for label in CLASS_ORDER],
            "share": [summary["labels"]["proportions"][label] for label in CLASS_ORDER],
        }
    )
    class_summary.to_csv(TECHNOLOGY / "annotation_class_summary.csv", index=False, encoding="utf-8")
    (
        pd.crosstab(corpus["label"], corpus["year"])
        .reindex(CLASS_ORDER)
        .fillna(0)
        .astype(int)
        .to_csv(TECHNOLOGY / "annotation_class_by_year.csv", encoding="utf-8")
    )
    (TECHNOLOGY / "annotation_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
