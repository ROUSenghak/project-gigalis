"""Successor-linkage features and scoring at procurement-episode level.

This module ports the linkage design developed in notebooks 05 and 08 onto the
v2 episode layer. The design is unchanged in spirit: block candidates by buyer
and time, then score five interpretable components (buyer, text, CPV, time,
and -- historically -- geography).

Two deliberate changes from the notebook implementation:

1. ``notebooks/05`` scored the time component by *assuming a four-year contract*
   whenever the expected end date was unknown. That silently converts a missing
   value into a strong assumption. Here, evidence that does not exist is
   reported as :data:`MISSING` (NaN) instead.

2. The weighted score renormalises over the components that actually carry
   evidence (see :func:`weighted_score`). A pair with no CPV on either side is
   scored on buyer/text/time alone rather than being penalised as if the CPV
   codes disagreed. This implements "missing evidence is not negative
   evidence" without inventing a neutral constant.

The geography component from notebook 08 is dropped: the study cohort is
restricted to Grand Ouest, so within-cohort geography carries almost no signal
(this was already the stated justification for its 0.05 weight).
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MISSING = float("nan")

#: CPV divisions defining the digital study scope (guideline section 3.2.2):
#: 48 software, 72 IT services, 32 telecommunications equipment, 35 security.
DIGITAL_CPV_DIVISIONS = ("32", "35", "48", "72")

#: Longest gap between an anchor award and a successor tender that is still
#: considered plausible. Retained from notebook 05 (~8 years); the confirmed
#: benchmark links span 163-2,644 days, so this window excludes none of them.
MAX_GAP_DAYS = 2920

#: Shortest gap that can count as succession, reinstated from notebook 05.
#:
#: A procurement published by the same buyer within days or weeks of an award
#: is concurrent activity -- typically another lot of the same programme, or a
#: closely related parallel tender -- not a successor addressing a renewed
#: need. Dropping this floor produced an obvious pathology: 132 of 628 accepted
#: links fell inside three months, and every accepted link for a 2025 award sat
#: under twelve months (median 1.7), which no renewal cycle can explain.
#:
#: The value is set empirically rather than tuned: across the 23 manually
#: confirmed successor links that resolve onto the cohort, the shortest
#: award-to-successor gap is 139 days, so a 90-day floor discards none of them
#: while removing the concurrent-procurement band.
MIN_GAP_DAYS = 90

#: Strength of each kind of buyer evidence. A validated SIREN is the only
#: identifier that is verified (Luhn-checked upstream); an exact normalised
#: name is strong but not verified; fuzzy evidence is weaker still.
BUYER_SCORE_MAP: Mapping[str, float] = {
    "siren": 1.0,
    "normalized_name": 0.9,
    "fuzzy_name": 0.7,
    "token_overlap": 0.6,
    "none": 0.0,
}

#: Weights ported from the ``buyer_precision_v1`` configuration frozen in
#: ``data/processed/boamp_grand_ouest/high_precision_linkage_config.json``.
#: The geography weight (0.05) is redistributed onto CPV, which is the
#: component it was competing with for the same "functional continuity" role.
DEFAULT_WEIGHTS: Mapping[str, float] = {
    "buyer": 0.50,
    "text": 0.25,
    "cpv": 0.20,
    "time": 0.05,
}

_ACCENT_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")
_BUYER_STOPWORDS_RE = re.compile(
    r"\b(commune de|ville de|mairie de|departement de|departement|region|"
    r"conseil regional|conseil departemental|communaute de communes|"
    r"communaute d agglomeration|metropole)\b"
)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def parse_json_list(value: Any) -> list:
    """Parse a compact JSON list field, returning ``[]` for missing input."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else [parsed]


def normalize_text(value: Any) -> str:
    """Lowercase, strip accents, and reduce to single-spaced alphanumerics."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value).lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _WS_RE.sub(" ", _ACCENT_RE.sub(" ", stripped)).strip()


def normalize_buyer_for_blocking(value: Any) -> str:
    """Normalise a buyer name and drop generic French public-body prefixes.

    "Commune de Nantes" and "Nantes" block together, which matters because the
    same buyer is named inconsistently across BOAMP schemas.
    """
    return _WS_RE.sub(" ", _BUYER_STOPWORDS_RE.sub(" ", normalize_text(value))).strip()


def buyer_tokens(value: Any) -> set[str]:
    """Informative tokens of a buyer name, used for overlap-based blocking."""
    return {tok for tok in str(value or "").split() if len(tok) >= 4 and not tok.isdigit()}


def jaccard(left: Iterable, right: Iterable) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def buyer_similarity(left: str, right: str) -> float:
    """Fuzzy string ratio between two normalised buyer names."""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------


def classify_buyer_match(
    anchor_siren: str,
    candidate_siren: str,
    anchor_name: str,
    candidate_name: str,
    fuzzy_threshold: float = 0.82,
    token_threshold: float = 0.34,
) -> str:
    """Return the strongest kind of buyer evidence linking two episodes.

    Only validated SIRENs are compared as identifiers; an empty SIREN never
    matches another empty SIREN, because "both unknown" is not evidence of
    being the same organisation.
    """
    if anchor_siren and candidate_siren and anchor_siren == candidate_siren:
        return "siren"
    if anchor_name and candidate_name and anchor_name == candidate_name:
        return "normalized_name"
    if buyer_similarity(anchor_name, candidate_name) >= fuzzy_threshold:
        return "fuzzy_name"
    if jaccard(buyer_tokens(anchor_name), buyer_tokens(candidate_name)) >= token_threshold:
        return "token_overlap"
    return "none"


def buyer_score(match_type: str) -> float:
    return BUYER_SCORE_MAP.get(match_type, 0.0)


def cpv_divisions(codes: Sequence[Any]) -> set[str]:
    """Two-digit CPV divisions (e.g. ``72`` IT services)."""
    return {str(code)[:2] for code in codes if str(code).strip()}


def cpv_groups(codes: Sequence[Any]) -> set[str]:
    """Three-digit CPV groups, a finer notion of functional continuity."""
    return {str(code)[:3] for code in codes if len(str(code).strip()) >= 3}


def is_digital(codes: Sequence[Any]) -> bool:
    """True when any CPV code falls in the digital study scope."""
    return bool(cpv_divisions(codes) & set(DIGITAL_CPV_DIVISIONS))


def cpv_continuity_score(anchor_codes: Sequence[Any], candidate_codes: Sequence[Any]) -> float:
    """Functional continuity from CPV, or :data:`MISSING` when unobservable.

    Returns NaN -- not 0.0 -- when either side has no CPV at all. A missing
    CPV code is an absence of evidence, whereas 0.0 would assert that the two
    procurements are in unrelated categories.
    """
    anchor_codes = [c for c in anchor_codes if str(c).strip()]
    candidate_codes = [c for c in candidate_codes if str(c).strip()]
    if not anchor_codes or not candidate_codes:
        return MISSING
    group_overlap = jaccard(cpv_groups(anchor_codes), cpv_groups(candidate_codes))
    division_overlap = jaccard(cpv_divisions(anchor_codes), cpv_divisions(candidate_codes))
    # Group agreement is the sharper signal; division agreement is a weaker
    # fallback so that 72xxx vs 72yyy still scores above 72xxx vs 45xxx.
    return max(group_overlap, 0.5 * division_overlap)


def time_plausibility_score(gap_days: float, expected_duration_months: float | None) -> float:
    """Score how well a candidate's timing matches the anchor's expected end.

    Returns :data:`MISSING` when the anchor has no reliable declared duration.
    Notebook 05 substituted a four-year assumption here; that fabricates the
    single most influential timing input for 63.8% of the cohort, so it is
    deliberately not reproduced.
    """
    if expected_duration_months is None:
        return MISSING
    if isinstance(expected_duration_months, float) and math.isnan(expected_duration_months):
        return MISSING
    if expected_duration_months <= 0:
        return MISSING
    expected_gap = float(expected_duration_months) * 30.4375
    deviation = abs(float(gap_days) - expected_gap)
    # A re-procurement is typically launched within roughly six months either
    # side of contract end (guideline section 3.2.1); decay linearly to zero
    # one year out.
    return max(0.0, 1.0 - deviation / 365.0)


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------


def weighted_score(
    components: Mapping[str, float],
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    scale: float = 100.0,
) -> float:
    """Combine component scores, renormalising over available evidence.

    Components whose value is NaN are dropped from both the numerator and the
    denominator, so a pair lacking CPV evidence is judged on its remaining
    evidence rather than penalised. Returns NaN if nothing is observable.

    >>> round(weighted_score({"buyer": 1.0, "text": 0.0}, {"buyer": 0.5, "text": 0.5}), 1)
    50.0
    >>> round(weighted_score({"buyer": 1.0, "text": MISSING}, {"buyer": 0.5, "text": 0.5}), 1)
    100.0
    """
    total_weight = 0.0
    total = 0.0
    for name, weight in weights.items():
        value = components.get(name, MISSING)
        if value is None:
            continue
        value = float(value)
        if math.isnan(value):
            continue
        total += weight * min(max(value, 0.0), 1.0)
        total_weight += weight
    if total_weight <= 0:
        return MISSING
    return scale * total / total_weight


def evidence_completeness(components: Mapping[str, float]) -> int:
    """Number of components carrying actual evidence, used for gating."""
    return sum(
        1
        for value in components.values()
        if value is not None and not (isinstance(value, float) and math.isnan(float(value)))
    )


def tfidf_similarities(anchor_text: str, candidate_texts: Sequence[str], analyzer: str = "word") -> np.ndarray:
    """Cosine similarity of one anchor text against many candidate texts.

    The vectoriser is fitted per anchor block, as in notebook 05: IDF weights
    are then local to the buyer's own vocabulary, which keeps boilerplate
    common to a single buyer's notices from dominating the similarity.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [str(t or "") for t in candidate_texts]
    if not texts:
        return np.array([])
    if analyzer == "char":
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    else:
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    try:
        matrix = vectorizer.fit_transform([str(anchor_text or "")] + texts)
    except ValueError:
        # Raised when the whole block is empty or stop-word-only.
        return np.zeros(len(texts))
    return cosine_similarity(matrix[0], matrix[1:]).ravel()
