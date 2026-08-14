#!/usr/bin/env python3
"""Resolve which earlier procurement a declaring notice is talking about.

``mine_renewal_declarations.py`` finds notices asserting that a predecessor
contract exists. This script proposes *which* one, by matching the attributes
the notice states -- an expiry date, a reference number, an incumbent supplier
-- against earlier episodes of the same buyer.

The resolvers are ordered by how much they depend on the thing being evaluated:

``R1 explicit_expiry``   the notice states the predecessor's end date, matched
                         against a candidate's computed expected end. Strongest
                         and entirely independent of text similarity: a declared
                         term is close to a unique key inside one buyer's
                         history.
``R2 reference``         the notice names a procedure reference.
``R3 incumbent``         the declaring episode and the candidate awarded to the
                         same supplier. Structured, text-free.
``R4 text_fallback``     the only earlier episode whose object is similar and
                         whose term plausibly ends near the declaration.

R4 is the one resolver that uses text, so it is recorded separately and never
counted as strong. If enrichment positives came mostly from R4 they would be
text-similar by construction, which would flatter a text-ranking method -- so
the share is reported and the resolution method travels with every row.

.. warning::
   A resolved predecessor is **not a label**. It is a recruitment proposal.
   Every pair still goes through blind annotation and can be rejected there.
   Nothing downstream may treat this file as ground truth.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.linkage import jaccard, normalize_text, parse_json_list  # noqa: E402
from boamp_pipeline.renewal_language import (  # noqa: E402
    EXPIRY_DECLARATION,
    POSITIVE_CLASSES,
    RENEWAL_DECLARATION,
)
from boamp_pipeline.standardize import normalize_reference  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp/benchmark")
RESOLUTION_VERSION = "boamp_declared_predecessor_v1.0"

#: How far back a predecessor may sit. Generous: a declaration in 2024 can
#: legitimately refer to a contract awarded in 2015 under a long framework.
MAX_LOOKBACK_DAYS = 4380  # 12 years

#: A predecessor must have started before the declaration was published.
MIN_LOOKBACK_DAYS = 1

#: Tolerance on matching a declared expiry against a computed expected end.
#: An explicit end date recorded in structured fields is exact; a date derived
#: from a duration inherits that duration's imprecision; and a declaration
#: giving only a month or a year cannot be matched more tightly than that.
TOLERANCE_BY_SOURCE = {
    "explicit_end": 31,
    "start_plus_duration": 92,
    "award_plus_duration": 92,
    "unavailable": 0,
}
TOLERANCE_BY_PRECISION = {"day": 31, "month": 183, "year": 365}

#: R4 only fires on a clearly similar object; below this the pairing is a guess.
TEXT_FALLBACK_MIN_JACCARD = 0.5
TEXT_FALLBACK_WINDOW_BEFORE = 540
TEXT_FALLBACK_WINDOW_AFTER = 180

INDEX_COLUMNS = [
    "episode_id", "episode_origin_date", "award_date", "buyer_block_key",
    "buyer_name_raw", "buyer_department", "dept_group", "digital_flag",
    "digital_segment", "main_cpv", "all_cpvs_json", "expected_end_date",
    "expected_end_source", "duration_months_reliable", "framework_flag",
    "procedure_references_json", "titulaire_names_json", "notice_count",
]


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "resolve_declared_predecessors.log", encoding="utf-8"),
        ],
    )


def token_set(text: Any) -> set[str]:
    return {token for token in normalize_text(text).split() if len(token) >= 4}


def expiry_tolerance(expected_end_source: str, precision: str) -> int:
    """Days of slack allowed when matching a declared expiry to a computed one."""
    return max(
        TOLERANCE_BY_SOURCE.get(expected_end_source, 0),
        TOLERANCE_BY_PRECISION.get(precision, 0),
    )


# ---------------------------------------------------------------------------
# Declaring episodes
# ---------------------------------------------------------------------------


def load_declaring_episodes(output_dir: Path) -> pd.DataFrame:
    """One row per episode that declares a predecessor, with its best evidence.

    A notice may match several families; the episode keeps the strongest
    declared attributes and the evidence that produced them.
    """
    declarations = pd.read_parquet(output_dir / "renewal_declarations.parquet")
    positive = declarations.loc[
        declarations["excluded_reason"].fillna("").eq("")
        & declarations["pattern_class"].isin(POSITIVE_CLASSES)
        & declarations["episode_id"].fillna("").ne("")
    ].copy()
    logging.info("Positive declaration rows: %s", f"{len(positive):,}")

    rows: list[dict[str, Any]] = []
    for episode_id, group in positive.groupby("episode_id", sort=False):
        classes = sorted(set(group["pattern_class"]))
        dated = group.loc[group["declared_end_date"].notna()]
        referenced = group.loc[group["declared_reference"].fillna("").ne("")]
        # Prefer the evidence a human would find most checkable.
        lead = group.loc[group["pattern_class"].eq(RENEWAL_DECLARATION)]
        if lead.empty:
            lead = group.loc[group["pattern_class"].eq(EXPIRY_DECLARATION)]
        if lead.empty:
            lead = group
        lead_row = lead.iloc[0]
        rows.append({
            "successor_episode_id": episode_id,
            "declaring_idweb": lead_row["idweb"],
            "declaration_publication_date": group["publication_date"].min(),
            "declared_classes_json": json.dumps(classes, ensure_ascii=False),
            "declaration_strength": int(group["notice_declaration_strength"].max()),
            "declared_end_date": dated["declared_end_date"].iloc[0] if len(dated) else None,
            "declared_end_date_precision": (
                dated["declared_end_date_precision"].iloc[0] if len(dated) else ""
            ),
            "declared_end_date_snippet": (
                dated["declared_end_date_snippet"].iloc[0] if len(dated) else ""
            ),
            "declared_reference": (
                referenced["declared_reference"].iloc[0] if len(referenced) else ""
            ),
            "evidence_pattern": lead_row["pattern_name"],
            "evidence_field": lead_row["field"],
            "evidence_snippet": lead_row["evidence_snippet"],
        })
    frame = pd.DataFrame(rows)
    logging.info("Declaring episodes: %s", f"{len(frame):,}")
    return frame


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def candidate_predecessors(
    declaration_date: date,
    block_rows: list[dict[str, Any]],
    successor_episode_id: str,
) -> list[dict[str, Any]]:
    """Earlier contracts of the same buyer that could be the predecessor.

    The predecessor must have been **awarded** before the declaration was
    published, not merely tendered earlier. Requiring only an earlier tender
    start admitted procedures that were still running when the declaration
    appeared, which produced negative award-to-successor gaps -- a "predecessor"
    awarded after the contract said to be replacing it. A renewal replaces a
    contract that exists, so an award is what makes a candidate eligible.
    """
    earliest = declaration_date - timedelta(days=MAX_LOOKBACK_DAYS)
    latest = declaration_date - timedelta(days=MIN_LOOKBACK_DAYS)
    return [
        row for row in block_rows
        if row["episode_id"] != successor_episode_id
        and row["episode_origin_date"] is not None
        and earliest <= row["episode_origin_date"] <= latest
        and row["award_date"] is not None
        and row["award_date"] < declaration_date
    ]


def resolve_by_expiry(declaration: dict[str, Any],
                      candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """R1: the notice stated when the predecessor ends."""
    declared_end = declaration["declared_end_date"]
    if declared_end is None or pd.isna(declared_end):
        return []
    precision = declaration["declared_end_date_precision"] or "day"
    hits: list[dict[str, Any]] = []
    for candidate in candidates:
        expected = candidate["expected_end_date"]
        if expected is None:
            continue
        tolerance = expiry_tolerance(candidate["expected_end_source"], precision)
        if tolerance <= 0:
            continue
        delta = abs((expected - declared_end).days)
        if delta <= tolerance:
            hits.append({**candidate, "_delta": delta, "_tolerance": tolerance})
    return sorted(hits, key=lambda row: row["_delta"])


def resolve_by_reference(declaration: dict[str, Any],
                         candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """R2: the notice named a procedure reference."""
    declared = normalize_reference(declaration["declared_reference"])
    if not declared:
        return []
    hits = []
    for candidate in candidates:
        references = {
            normalize_reference(value)
            for value in parse_json_list(candidate["procedure_references_json"])
        }
        references.discard("")
        if declared in references:
            hits.append(candidate)
    return hits


#: Year tokens inside a reference. No trailing word boundary is required,
#: because the year is usually glued to the stem ("2018feux", "18nettoyage").
_YEAR_TOKEN = re.compile(
    r"(?:19|20)\d{2}"            # four-digit year, possibly glued to letters
    r"|(?<![0-9])\d{2}(?=[a-z])"  # two-digit year prefixing a stem
    r"|(?<=[a-z])\d{2}(?![0-9])"  # two-digit year suffixing a stem
)


def reference_stem(value: Any) -> str:
    """A procedure reference with its year token removed.

    Buyers number consecutive procurements of the same need on a stable stem:
    ``2018feux`` then ``2023feux``, ``18nettoyage`` then ``23nettoyage``.
    On its own the stem is weak evidence -- measured at 6% precision across
    7,343 same-buyer pairs, because the stem is usually a per-year sequence
    counter. Here it is only ever applied to a pair whose successor has already
    *declared* that a predecessor exists, which is what makes it usable.
    """
    normalised = normalize_reference(value)
    if not normalised:
        return ""
    stem = _YEAR_TOKEN.sub("", normalised)
    stem = re.sub(r"[^a-z0-9]+", "", stem)
    return stem if len(stem) >= 4 else ""


def resolve_by_reference_stem(declaration: dict[str, Any], successor_references: list[str],
                              candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """R2b: the successor's own reference shares a stem with a candidate's."""
    stems = {reference_stem(value) for value in successor_references}
    stems.discard("")
    if not stems:
        return []
    hits = []
    for candidate in candidates:
        candidate_stems = {
            reference_stem(value)
            for value in parse_json_list(candidate["procedure_references_json"])
        }
        candidate_stems.discard("")
        if stems & candidate_stems:
            hits.append(candidate)
    return hits


def resolve_by_sole_candidate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """R0: the buyer has exactly one earlier awarded contract to point at.

    Resolution by elimination, and graded **weak** for a reason: it assumes the
    declared predecessor is in the corpus at all. Many are not -- contracts
    below the publication threshold, or awarded before 2015, never appear here,
    so the single visible candidate can simply be the wrong contract. Measured
    behaviour bears that out: pairs resolved this way have a median
    award-to-successor gap of under a year, well short of the two-to-four-year
    cycle the reliably-resolved pairs show.

    Kept because a sole candidate is genuinely worth showing an annotator, but
    excluded from the strong/medium count and from the market-behaviour
    estimates.
    """
    return list(candidates) if len(candidates) == 1 else []


def resolve_by_incumbent(successor_suppliers: set[str],
                         candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """R3: the same supplier holds both contracts. Structured, text-free."""
    if not successor_suppliers:
        return []
    hits = []
    for candidate in candidates:
        suppliers = set(parse_json_list(candidate["titulaire_names_json"]))
        if suppliers & successor_suppliers:
            hits.append(candidate)
    return hits


def resolve_by_text(successor_tokens: set[str], declaration_date: date,
                    candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """R4: the only similar earlier object whose term plausibly ends nearby.

    This is the sole resolver that consults text, so pairs recruited by it are
    text-similar by construction and are reported separately.
    """
    if not successor_tokens:
        return []
    hits = []
    for candidate in candidates:
        expected = candidate["expected_end_date"]
        if expected is not None:
            offset = (declaration_date - expected).days
            if not -TEXT_FALLBACK_WINDOW_BEFORE <= offset <= TEXT_FALLBACK_WINDOW_AFTER:
                continue
        similarity = jaccard(successor_tokens, candidate["_tokens"])
        if similarity >= TEXT_FALLBACK_MIN_JACCARD:
            hits.append({**candidate, "_similarity": similarity})
    return sorted(hits, key=lambda row: -row["_similarity"])


def build(project_root: Path, output_dir: Path, force: bool,
          allow_text_fallback: bool = False) -> dict[str, Any]:
    links_path = output_dir / "declared_predecessor_links.parquet"
    summary_path = output_dir / "declared_predecessor_summary.json"
    if links_path.exists() and not force:
        raise FileExistsError(f"{links_path} already exists. Use --force to rebuild.")

    declaring = load_declaring_episodes(output_dir)
    if declaring.empty:
        raise RuntimeError("no declaring episodes found; check the mining stage")

    index = pd.read_parquet(output_dir / "national_episode_index.parquet", columns=INDEX_COLUMNS)
    logging.info("National index: %s episodes", f"{len(index):,}")

    by_episode = index.set_index("episode_id")
    declaring = declaring.loc[declaring["successor_episode_id"].isin(by_episode.index)].copy()
    logging.info("Declaring episodes present in the index: %s", f"{len(declaring):,}")

    # Only buyer blocks containing a declaration can produce a resolution, so
    # the pool is restricted before anything expensive happens.
    successor_blocks = set(
        by_episode.loc[declaring["successor_episode_id"], "buyer_block_key"].dropna()
    )
    successor_blocks.discard("")
    pool = index.loc[index["buyer_block_key"].isin(successor_blocks)].copy()
    logging.info(
        "Candidate pool: %s episodes across %s buyer blocks",
        f"{len(pool):,}", f"{len(successor_blocks):,}",
    )

    # Text is joined only for the pool, never for all 1.1M episodes.
    needed = set(pool["episode_id"]) | set(declaring["successor_episode_id"])
    texts = load_episode_texts(output_dir.parent, needed)
    logging.info("Loaded text for %s episodes", f"{len(texts):,}")

    block_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool.to_dict(orient="records"):
        row["_tokens"] = token_set(texts.get(row["episode_id"], ""))
        block_rows[row["buyer_block_key"]].append(row)

    method_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    resolved_rows: list[dict[str, Any]] = []

    for declaration in declaring.to_dict(orient="records"):
        successor_id = declaration["successor_episode_id"]
        successor = by_episode.loc[successor_id]
        block = str(successor["buyer_block_key"] or "")
        publication = declaration["declaration_publication_date"]
        if not block or publication is None:
            status_counts["no_block_or_date"] += 1
            continue
        candidates = candidate_predecessors(publication, block_rows.get(block, []), successor_id)
        if not candidates:
            status_counts["no_candidate_in_window"] += 1
            continue

        successor_suppliers = set(parse_json_list(successor["titulaire_names_json"]))
        successor_tokens = token_set(texts.get(successor_id, ""))

        attempts = [
            ("R1_explicit_expiry", "strong", resolve_by_expiry(declaration, candidates)),
            ("R2_reference", "strong", resolve_by_reference(declaration, candidates)),
            ("R2b_reference_stem", "medium",
             resolve_by_reference_stem(
                 declaration,
                 parse_json_list(successor["procedure_references_json"]),
                 candidates,
             )),
            ("R3_incumbent", "medium", resolve_by_incumbent(successor_suppliers, candidates)),
            ("R0_sole_candidate", "weak", resolve_by_sole_candidate(candidates)),
        ]
        if allow_text_fallback:
            attempts.append(
                ("R4_text_fallback", "weak",
                 resolve_by_text(successor_tokens, publication, candidates))
            )
        chosen = None
        for method, confidence, hits in attempts:
            if not hits:
                continue
            # A resolver that cannot single out one predecessor has not
            # resolved anything. Ties become multi-way annotation questions
            # rather than an arbitrary pick.
            if len(hits) > 1 and method != "R1_explicit_expiry":
                chosen = (method, confidence, hits[:5], "ambiguous")
                break
            if len(hits) > 1 and hits[0]["_delta"] == hits[1]["_delta"]:
                chosen = (method, confidence, hits[:5], "ambiguous")
                break
            chosen = (method, confidence, [hits[0]], "resolved")
            break

        if chosen is None:
            status_counts["unresolved"] += 1
            continue

        method, confidence, hits, status = chosen
        method_counts[f"{method}:{status}"] += 1
        status_counts[status] += 1
        for rank, candidate in enumerate(hits, start=1):
            resolved_rows.append({
                "successor_episode_id": successor_id,
                "predecessor_episode_id": candidate["episode_id"],
                "resolution_method": method,
                "resolution_confidence": confidence,
                "resolution_status": status,
                "candidate_rank": rank,
                "candidates_considered": len(candidates),
                "buyer_block_key": block,
                "buyer_name_raw": successor["buyer_name_raw"],
                "dept_group": successor["dept_group"],
                "successor_origin_date": successor["episode_origin_date"],
                "successor_digital_segment": successor["digital_segment"],
                "predecessor_origin_date": candidate["episode_origin_date"],
                "predecessor_award_date": candidate["award_date"],
                "predecessor_expected_end_date": candidate["expected_end_date"],
                "predecessor_expected_end_source": candidate["expected_end_source"],
                "predecessor_digital_segment": candidate["digital_segment"],
                "predecessor_framework_flag": bool(candidate["framework_flag"]),
                "declaration_publication_date": publication,
                "declared_end_date": declaration["declared_end_date"],
                "declared_end_date_precision": declaration["declared_end_date_precision"],
                "declared_reference": declaration["declared_reference"],
                "declared_classes_json": declaration["declared_classes_json"],
                "declaration_strength": declaration["declaration_strength"],
                "declaring_idweb": declaration["declaring_idweb"],
                "evidence_pattern": declaration["evidence_pattern"],
                "evidence_field": declaration["evidence_field"],
                "evidence_snippet": declaration["evidence_snippet"],
                "declared_end_date_snippet": declaration["declared_end_date_snippet"],
                "gap_days": (
                    (publication - candidate["award_date"]).days
                    if candidate["award_date"] is not None else None
                ),
                "days_relative_to_expected_end": (
                    (publication - candidate["expected_end_date"]).days
                    if candidate["expected_end_date"] is not None else None
                ),
                "resolution_version": RESOLUTION_VERSION,
            })

    links = pd.DataFrame(resolved_rows)
    links.to_parquet(links_path, index=False, compression="zstd")

    unique = links.loc[links["resolution_status"].eq("resolved")] if len(links) else links
    reliable = (
        unique.loc[unique["resolution_confidence"].isin(["strong", "medium"])]
        if len(unique) else unique
    )
    strong_medium = int(len(reliable))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "resolution_version": RESOLUTION_VERSION,
        "output_file": str(links_path),
        "role": (
            "Anchor recruitment and exposure input only. A resolved predecessor is "
            "NOT a label; every pair goes to blind annotation and may be rejected."
        ),
        "declaring_episodes": int(len(declaring)),
        "resolution_status_counts": dict(status_counts),
        "resolution_method_counts": dict(method_counts.most_common()),
        "uniquely_resolved": int(len(unique)),
        "uniquely_resolved_strong_or_medium": strong_medium,
        "text_fallback_enabled": allow_text_fallback,
        "text_recruited_share": (
            round(float(unique["resolution_method"].eq("R4_text_fallback").mean()), 4)
            if len(unique) else None
        ),
        # Estimated on strong/medium resolutions only. Including the weak
        # sole-candidate arm would mix in pairs whose predecessor is probably
        # not the one the notice meant.
        "market_behaviour": market_behaviour(reliable) if len(reliable) else {},
        "market_behaviour_basis": "resolution_confidence in {strong, medium}",
        "resolved_by_confidence": (
            {str(k): int(v) for k, v in unique["resolution_confidence"].value_counts().items()}
            if len(unique) else {}
        ),
        "digital_reliable_pairs": (
            int(reliable["predecessor_digital_segment"].ne("").sum()) if len(reliable) else 0
        ),
        # Gate G2: enough uniquely resolved, non-text-recruited predecessors.
        "gate_g2_min_strong_medium": 250,
        "gate_g2_passed": bool(strong_medium >= 250),
        "validation_passed": bool(len(links) > 0),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def load_episode_texts(processed_dir: Path, needed: set[str]) -> dict[str, str]:
    """Join episode text for a restricted set of episodes, in batches."""
    texts: dict[str, str] = {}
    parquet = pq.ParquetFile(processed_dir / "episodes_2015_2025.parquet")
    for batch in parquet.iter_batches(batch_size=50_000, columns=["episode_id", "episode_text"]):
        rows = batch.to_pydict()
        for episode_id, text in zip(rows["episode_id"], rows["episode_text"]):
            if episode_id in needed:
                texts[episode_id] = text or ""
    return texts


def market_behaviour(links: pd.DataFrame) -> dict[str, Any]:
    """Empirical renewal timing, from pairs the buyers themselves declared.

    This is the study of market behaviour the benchmark is meant to learn from.
    Two questions matter for candidate generation: how long after a contract
    starts its replacement appears, and whether replacements are published
    before the contract they replace has expired -- which, if common, means an
    expiry-based hard exclusion rule would discard real successors.
    """
    gaps = pd.to_numeric(links["gap_days"], errors="coerce").dropna()
    relative = pd.to_numeric(links["days_relative_to_expected_end"], errors="coerce").dropna()

    def quantiles(series: pd.Series) -> dict[str, float]:
        if series.empty:
            return {}
        return {
            "n": int(series.size),
            "min": float(series.min()),
            "p05": float(series.quantile(0.05)),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "p95": float(series.quantile(0.95)),
            "max": float(series.max()),
        }

    by_segment = {}
    for segment, group in links.groupby("predecessor_digital_segment"):
        series = pd.to_numeric(group["gap_days"], errors="coerce").dropna()
        if len(series) >= 5:
            by_segment[str(segment or "NON_DIGITAL")] = {
                "n": int(series.size), "median_gap_days": float(series.median()),
            }

    published_early = float((relative < 0).mean()) if len(relative) else None
    return {
        "award_to_successor_gap_days": quantiles(gaps),
        "gap_days_by_predecessor_segment": by_segment,
        "days_relative_to_expected_end": quantiles(relative),
        "share_published_before_expected_end": (
            round(published_early, 4) if published_early is not None else None
        ),
        "share_published_more_than_365d_before_expected_end": (
            round(float((relative < -365).mean()), 4) if len(relative) else None
        ),
        "framework_share": round(float(links["predecessor_framework_flag"].mean()), 4),
        "interpretation": (
            "Gaps are measured from the predecessor's award to the successor's first "
            "notice, on pairs the buyer declared. A large share published before the "
            "expected end would mean an expiry-based hard exclusion removes genuine "
            "successors."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-text-fallback",
        action="store_true",
        help="Enable R4, which selects a predecessor by object similarity. Off by "
             "default: pairs recruited that way are text-similar by construction, "
             "which would flatter a text-ranking method in the enrichment arm.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    configure_logging(project_root)
    summary = build(project_root, output_dir, args.force, args.allow_text_fallback)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["validation_passed"]:
        raise RuntimeError("Declared predecessor resolution validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
