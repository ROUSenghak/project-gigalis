#!/usr/bin/env python3
"""Build pair-level modeling tables from the regional reference and its exposure.

One row per reviewed anchor and exposed candidate. ``y_primary`` marks the pair
the reviewer named as the observable successor; every other candidate the
production blocking step proposed for that anchor is a negative.

That negative definition is corpus-relative and the tables say so: the reviewer
saw roughly 25 candidates per anchor, so a pair labelled 0 here means "not the
successor the reviewer identified", not "reviewed and rejected". The curves
built from these tables are score-ranking diagnostics, not accuracy claims.

Split roles: ``dev`` is the reference's own pilot stratum, kept for display;
``validation`` is its locked stratum and is where the frozen policy is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.fellegi_sunter_scoring import score_with_fitted_model  # noqa: E402
from boamp_pipeline.regional_benchmark_io import load_truth  # noqa: E402

DEFAULT_BENCHMARK_DIR = Path("data/processed/boamp/regional_benchmark")

#: Features carried through from the production candidate pool. This is a
#: shorter list than the retired national exposure carried, because the study's
#: own blocking step is what generates these pairs and it does not compute the
#: benchmark-only enrichment columns.
FEATURE_COLUMNS = [
    "gap_days",
    "buyer_name_similarity",
    "word_tfidf_similarity",
    "char_tfidf_similarity",
    "buyer_component",
    "text_component",
    "cpv_component",
    "time_component",
    "evidence_components",
    "linkage_score",
    "buyer_match_type",
    "candidate_is_digital",
    "fs_match_weight",
    "fs_match_probability",
]

NUMERIC_EVIDENCE = [
    "buyer_name_similarity", "word_tfidf_similarity", "char_tfidf_similarity",
    "buyer_component", "text_component", "cpv_component", "time_component",
    "linkage_score",
]

IDENTITY_COLUMNS = [
    "split", "frame", "stratum_id", "anchor_episode_id", "candidate_episode_id",
]

DESIGN_COLUMNS = ["inclusion_probability", "design_weight", "pool_size", "exposure_mode"]

LABEL_COLUMNS = [
    "label", "anchor_verdict", "negative_is_censored", "negative_verification",
]


def load_split(benchmark_dir: Path, split: str) -> pd.DataFrame:
    truth = load_truth(benchmark_dir, split, "primary")
    truth = truth.loc[truth["truth_usable"]].copy()
    exposure = pd.read_parquet(benchmark_dir / "exposure_full.parquet")
    exposure = score_with_fitted_model(
        exposure, benchmark_dir.parent / "fellegi_sunter_model.json"
    )

    positives = {
        (record.anchor_episode_id, successor)
        for record in truth.itertuples(index=False)
        for successor in record.true_successors
    }
    anchor_fields = truth[[
        "anchor_episode_id", "anchor_verdict", "stratum_id", "inclusion_probability",
        "design_weight", "pool_size", "exposure_mode", "negative_is_censored",
        "negative_verification", "has_successor_primary",
    ]]

    pairs = exposure.loc[
        exposure["anchor_episode_id"].isin(set(truth["anchor_episode_id"]))
    ].copy()
    merged = pairs.merge(
        anchor_fields, on="anchor_episode_id", how="inner", validate="many_to_one",
        suffixes=("", "_anchor"),
    )
    merged["split"] = split
    merged["frame"] = "PROBABILITY"
    merged["y_primary"] = [
        int(pair in positives)
        for pair in zip(merged["anchor_episode_id"], merged["candidate_episode_id"])
    ]
    merged["label"] = merged["y_primary"].map({1: "RENEWAL_OF_EXPIRING", 0: "UNRELATED"})
    merged["y_nonmatch"] = 1 - merged["y_primary"]
    merged["sample_weight"] = merged["design_weight"]

    keep = [
        *IDENTITY_COLUMNS, *DESIGN_COLUMNS, *LABEL_COLUMNS,
        "y_primary", "y_nonmatch", "has_successor_primary", "sample_weight",
        *FEATURE_COLUMNS,
    ]
    missing = [column for column in keep if column not in merged.columns]
    if missing:
        raise RuntimeError(f"{split} modeling table is missing columns: {missing}")
    return (
        merged[keep]
        .sort_values(["anchor_episode_id", "candidate_episode_id"])
        .reset_index(drop=True)
    )


def profile(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "anchors": int(frame["anchor_episode_id"].nunique()),
        "primary_positive_pairs": int(frame["y_primary"].sum()),
        "positive_anchors": int(
            frame.drop_duplicates("anchor_episode_id")["has_successor_primary"].sum()
        ),
        "feature_missing_rate": {
            column: round(float(frame[column].isna().mean()), 4)
            for column in FEATURE_COLUMNS
            if column in frame.columns
        },
        "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().items()},
    }


def build(benchmark_dir: Path, force: bool) -> dict[str, Any]:
    output_dir = benchmark_dir / "modeling"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "dev": output_dir / "modeling_dev.parquet",
        "validation": output_dir / "modeling_validation.parquet",
    }
    summary_path = output_dir / "modeling_summary.json"
    existing = [path for path in [*outputs.values(), summary_path] if path.exists()]
    if existing and not force:
        raise FileExistsError(f"modeling outputs already exist; use --force: {existing}")

    result: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": "regional_grand_ouest",
        "source_benchmark_dir": str(benchmark_dir),
        "role": {
            "dev": "pilot stratum of the reference; threshold display only",
            "validation": "locked stratum of the reference; held-out reading of the frozen policy",
        },
        "target_columns": {
            "y_primary": ["reviewed observable successor"],
            "y_nonmatch": ["every other exposed candidate of a reviewed anchor"],
        },
        "negative_definition": (
            "Corpus-relative. A zero means the pair is not the successor the reviewer "
            "named, not that the reviewer inspected and rejected it."
        ),
        "feature_columns": FEATURE_COLUMNS,
        "outputs": {},
    }

    for split, path in outputs.items():
        frame = load_split(benchmark_dir, split)
        if frame[["anchor_episode_id", "candidate_episode_id"]].duplicated().any():
            raise RuntimeError(f"{split} contains duplicate anchor-candidate rows")
        if frame[NUMERIC_EVIDENCE].isna().all(axis=1).any():
            raise RuntimeError(f"{split} contains a row with no numeric feature evidence")
        if frame["y_primary"].sum() == 0:
            raise RuntimeError(f"{split} has no positive pair; curves would be undefined")
        frame.to_parquet(path, index=False, compression="zstd")
        result["outputs"][split] = {"file": str(path), **profile(frame)}

    result["validation_passed"] = True
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_dir = args.benchmark_dir if args.benchmark_dir.is_absolute() else PROJECT_ROOT / args.benchmark_dir
    summary = build(benchmark_dir, args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
