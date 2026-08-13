#!/usr/bin/env python3
"""Prepare a blinded, stratified pair sample for independent specialist review."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data/processed/boamp"
BENCHMARK = PROCESSED / "benchmark"
OUTPUT_DIR = PROJECT_ROOT / "data/review"
RANDOM_SEED = 20260813
ROWS_PER_STRATUM = 20


def deterministic_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed).copy()


def pair_key(frame: pd.DataFrame) -> pd.Series:
    return frame["anchor_episode_id"].astype(str) + "|" + frame["candidate_episode_id"].astype(str)


def build_sample() -> pd.DataFrame:
    accepted = pd.read_parquet(PROCESSED / "accepted_successor_links.parquet").rename(
        columns={"text_component": "selection_text_component"}
    )
    accepted = accepted.assign(source_stratum="PRIMARY_ACCEPTED")
    accepted = (
        accepted.groupby(["anchor_digital_segment", "anchor_region"], group_keys=False)
        .apply(lambda group: deterministic_sample(group, max(1, round(ROWS_PER_STRATUM * len(group) / len(accepted))), RANDOM_SEED), include_groups=False)
        .reset_index(drop=True)
    )
    accepted = deterministic_sample(accepted, ROWS_PER_STRATUM, RANDOM_SEED)

    negatives = pd.read_parquet(BENCHMARK / "structural_negatives.parquet")
    negatives = negatives.loc[
        negatives["negative_source"].isin(["SAME_PROCUREMENT", "PARALLEL_LOT"])
    ].sort_values("text_component", ascending=False)
    negatives = deterministic_sample(negatives.head(300), ROWS_PER_STRATUM, RANDOM_SEED + 1)
    negatives = negatives.assign(
        source_stratum="HIGH_SIMILARITY_STRUCTURAL_NEGATIVE",
        selection_text_component=negatives["text_component"],
    )

    declared = pd.read_parquet(BENCHMARK / "declared_predecessor_links.parquet")
    declared = declared.loc[
        declared["resolution_status"].eq("resolved")
        & declared["resolution_confidence"].isin(["strong", "medium"])
    ].rename(
        columns={
            "predecessor_episode_id": "anchor_episode_id",
            "successor_episode_id": "candidate_episode_id",
        }
    )
    declared = deterministic_sample(declared, ROWS_PER_STRATUM, RANDOM_SEED + 2)
    declared = declared.assign(
        source_stratum="BUYER_DECLARED_RESOLVED",
        selection_text_component=np.nan,
    )

    keep = [
        "anchor_episode_id",
        "candidate_episode_id",
        "source_stratum",
        "selection_text_component",
    ]
    combined = pd.concat(
        [
            accepted.reindex(columns=keep),
            negatives.reindex(columns=keep),
            declared.reindex(columns=keep),
        ],
        ignore_index=True,
    )
    combined["pair_key"] = pair_key(combined)
    combined = combined.drop_duplicates("pair_key", keep="first")
    return combined.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def enrich_pairs(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes = pd.read_parquet(
        PROCESSED / "episodes_2015_2025.parquet",
        columns=[
            "episode_id",
            "episode_origin_date",
            "buyer_siren",
            "buyer_name_raw",
            "buyer_name_normalized",
            "buyer_region",
            "episode_text",
            "main_cpv",
            "all_cpvs_json",
            "duration_months",
            "duration_quality",
            "framework_flag",
        ],
    ).drop_duplicates("episode_id")
    anchor = episodes.add_prefix("anchor_").rename(columns={"anchor_episode_id": "anchor_episode_id"})
    candidate = episodes.add_prefix("candidate_").rename(
        columns={"candidate_episode_id": "candidate_episode_id"}
    )
    enriched = sample.merge(anchor, on="anchor_episode_id", how="left", validate="many_to_one")
    enriched = enriched.merge(candidate, on="candidate_episode_id", how="left", validate="many_to_one")
    if enriched["anchor_episode_text"].isna().any() or enriched["candidate_episode_text"].isna().any():
        raise RuntimeError("review sample could not be fully joined to episode text")

    enriched["gap_days"] = (
        pd.to_datetime(enriched["candidate_episode_origin_date"])
        - pd.to_datetime(enriched["anchor_episode_origin_date"])
    ).dt.days
    enriched.insert(0, "review_id", [f"R{index:03d}" for index in range(1, len(enriched) + 1)])

    reviewer_columns = [
        "review_id",
        "anchor_buyer_name_raw",
        "candidate_buyer_name_raw",
        "anchor_buyer_region",
        "candidate_buyer_region",
        "anchor_episode_origin_date",
        "candidate_episode_origin_date",
        "gap_days",
        "anchor_main_cpv",
        "candidate_main_cpv",
        "anchor_all_cpvs_json",
        "candidate_all_cpvs_json",
        "anchor_framework_flag",
        "anchor_duration_months",
        "anchor_duration_quality",
        "anchor_episode_text",
        "candidate_episode_text",
    ]
    review = enriched[reviewer_columns].copy()
    for column in [
        "same_legal_buyer_Y_N_UNCERTAIN",
        "relationship_label",
        "observable_successor_Y_N_UNCERTAIN",
        "confidence_HIGH_MEDIUM_LOW",
        "review_notes",
        "review_date",
    ]:
        review[column] = ""

    audit = enriched[
        [
            "review_id",
            "source_stratum",
            "anchor_episode_id",
            "candidate_episode_id",
            "pair_key",
            "selection_text_component",
        ]
    ].copy()
    return review, audit


def write_protocol(row_count: int, strata: dict[str, int]) -> Path:
    text = f"""# Independent Link Review Protocol

Generated: `{datetime.now().isoformat(timespec='seconds')}`

Pairs prepared: `{row_count}`

## Purpose

This review is the minimum independent check needed before reporting linkage
precision as externally validated. The reviewer file is blinded: it contains
descriptions, buyer fields, CPVs, and dates, but no bootstrap label, algorithm
decision, score, or sampling stratum.

## Sample Composition

| Hidden audit stratum | Rows |
|---|---:|
{chr(10).join(f'| {name} | {count} |' for name, count in sorted(strata.items()))}

The strata deliberately mix likely matches, difficult non-matches, and
buyer-declared relationships. They estimate performance on this challenge set,
not national prevalence. The audit key must remain hidden until labels are
finalised.

## Review Fields

For each row, decide:

1. `same_legal_buyer_Y_N_UNCERTAIN`: whether the notices concern the same legal
   buyer. A matching name is not enough when legal form or SIREN evidence conflicts.
2. `relationship_label`: one of `RENEWAL_OF_EXPIRING`,
   `NEXT_PHASE_OF_PROGRAMME`, `RETENDER_AFTER_FAILURE`, `PARALLEL_LOT`,
   `EXTENSION_SAME_CONTRACT`, `UNRELATED`, or `UNUSABLE`.
3. `observable_successor_Y_N_UNCERTAIN`: `Y` only for a later procurement that
   plausibly replaces or continues the same need. Use `UNCERTAIN` when the visible
   evidence cannot support a defensible decision.
4. Confidence and short evidence-based notes.

Timing alone must not decide the label. Early successor publication is possible,
and missing duration must not be replaced with an assumed four-year term.

## Files

- Reviewer file: `data/review/independent_link_review_sample.csv`
- Hidden audit key: `data/review/independent_link_review_audit_key.csv`
- Review provenance: `data/review/review_provenance.json`

Reviewer identity is not stored row by row. The review source and whether it is
independent human validation must instead be recorded truthfully in the
provenance file.

## Acceptance Rule

Freeze `M_B_text_ranking @ 0.70` while reviewing. After review, compute exact
binomial confidence intervals for accepted-link precision overall and by sample
stratum. Do not tune the threshold on these labels and then report the same rows
as validation.
"""
    path = PROJECT_ROOT / "INDEPENDENT_LINK_REVIEW_PROTOCOL.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = build_sample()
    review, audit = enrich_pairs(sample)
    review_path = OUTPUT_DIR / "independent_link_review_sample.csv"
    audit_path = OUTPUT_DIR / "independent_link_review_audit_key.csv"
    review.to_csv(review_path, index=False)
    audit.to_csv(audit_path, index=False)
    protocol = write_protocol(len(review), audit["source_stratum"].value_counts().to_dict())
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(review),
        "strata": {str(k): int(v) for k, v in audit["source_stratum"].value_counts().items()},
        "review_file": str(review_path),
        "audit_key": str(audit_path),
        "protocol": str(protocol),
        "blinding": {
            "bootstrap_labels_exposed": False,
            "algorithm_scores_exposed": False,
            "sample_stratum_exposed": False,
        },
        "review_schema": {
            "row_level_reviewer_id_stored": False,
            "provenance_record": str(OUTPUT_DIR / "review_provenance.json"),
        },
        "validation_passed": bool(
            len(review) == len(audit)
            and review["review_id"].is_unique
            and set(review["review_id"]) == set(audit["review_id"])
        ),
    }
    summary_path = OUTPUT_DIR / "independent_link_review_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
