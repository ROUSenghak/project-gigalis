#!/usr/bin/env python3
"""Build modeling-ready tables from the canonical benchmark and exposure features.

The output is intentionally split-aware: dev is for calibration, validation is
for method selection, and sealed test is not read here.
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

from boamp_pipeline.annotation_schema import EVENT_SETS  # noqa: E402
from boamp_pipeline.fellegi_sunter_scoring import score_with_fitted_model  # noqa: E402

DEFAULT_BENCHMARK_DIR = Path("data/processed/boamp/benchmark")

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
    "shared_supplier",
    "shared_reference_stem",
    "days_from_expected_end",
    "candidate_digital_segment",
    "candidate_main_cpv",
    "buyer_match_type",
    "gap_bucket",
    "is_random_tail",
    "tail_inclusion_probability",
    "fs_match_weight",
    "fs_match_probability",
]

IDENTITY_COLUMNS = [
    "split",
    "frame",
    "stratum_id",
    "anchor_episode_id",
    "candidate_episode_id",
    "buyer_block_key",
]

DESIGN_COLUMNS = [
    "inclusion_probability",
    "design_weight",
    "pool_size",
    "exposure_mode",
]

LABEL_COLUMNS = [
    "label",
    "confidence",
    "anchor_verdict",
    "negative_is_censored",
    "negative_verification",
]


def load_split(benchmark_dir: Path, split: str) -> pd.DataFrame:
    pairs = pd.read_parquet(benchmark_dir / f"benchmark_{split}_pairs.parquet")
    anchors = pd.read_parquet(benchmark_dir / f"benchmark_{split}.parquet")
    exposure = pd.read_parquet(benchmark_dir / "exposure_full.parquet")
    exposure = score_with_fitted_model(exposure, benchmark_dir.parent / "fellegi_sunter_model.json")

    anchor_fields = anchors[
        [
            "anchor_episode_id",
            "negative_is_censored",
            "negative_verification",
            "has_successor_strict",
            "has_successor_primary",
            "has_successor_broad",
        ]
    ]
    merged = (
        pairs.merge(anchor_fields, on="anchor_episode_id", how="left", validate="many_to_one")
        .merge(
            exposure[
                ["anchor_episode_id", "candidate_episode_id", *FEATURE_COLUMNS]
            ],
            on=["anchor_episode_id", "candidate_episode_id"],
            how="left",
            validate="one_to_one",
        )
    )

    for event_set, labels in EVENT_SETS.items():
        merged[f"y_{event_set}"] = merged["label"].isin(labels).astype("int8")
    merged["y_nonmatch"] = (merged["label"] == "UNRELATED").astype("int8")
    merged["sample_weight"] = merged["design_weight"].where(
        merged["frame"].eq("PROBABILITY"), pd.NA
    )

    keep = [
        *IDENTITY_COLUMNS,
        *DESIGN_COLUMNS,
        *LABEL_COLUMNS,
        "y_strict",
        "y_primary",
        "y_broad",
        "y_nonmatch",
        "has_successor_strict",
        "has_successor_primary",
        "has_successor_broad",
        "sample_weight",
        *FEATURE_COLUMNS,
    ]
    missing = [column for column in keep if column not in merged.columns]
    if missing:
        raise RuntimeError(f"{split} modeling table is missing columns: {missing}")
    return merged[keep].sort_values(["anchor_episode_id", "candidate_episode_id"]).reset_index(drop=True)


def profile(frame: pd.DataFrame) -> dict[str, Any]:
    probability = frame.loc[frame["frame"].eq("PROBABILITY")]
    return {
        "rows": int(len(frame)),
        "anchors": int(frame["anchor_episode_id"].nunique()),
        "probability_frame_anchors": int(probability["anchor_episode_id"].nunique()),
        "primary_positive_pairs": int(frame["y_primary"].sum()),
        "strict_positive_pairs": int(frame["y_strict"].sum()),
        "broad_positive_pairs": int(frame["y_broad"].sum()),
        "verified_negative_anchors": int(
            frame.drop_duplicates("anchor_episode_id")["negative_verification"]
            .eq("EXHAUSTIVE_POOL")
            .sum()
        ),
        "feature_missing_rate": {
            column: round(float(frame[column].isna().mean()), 4)
            for column in FEATURE_COLUMNS
            if column in frame.columns
        },
        "label_counts": {
            str(k): int(v) for k, v in frame["label"].value_counts().items()
        },
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
        "source_benchmark_dir": str(benchmark_dir),
        "role": {
            "dev": "calibration and threshold tuning",
            "validation": "method selection and adoption checks",
            "sealed_test": "not read by this script",
        },
        "target_columns": {
            "y_strict": sorted(EVENT_SETS["strict"]),
            "y_primary": sorted(EVENT_SETS["primary"]),
            "y_broad": sorted(EVENT_SETS["broad"]),
            "y_nonmatch": ["UNRELATED"],
        },
        "feature_columns": FEATURE_COLUMNS,
        "outputs": {},
    }

    for split, path in outputs.items():
        frame = load_split(benchmark_dir, split)
        if frame[["anchor_episode_id", "candidate_episode_id"]].duplicated().any():
            raise RuntimeError(f"{split} contains duplicate anchor-candidate rows")
        if frame[FEATURE_COLUMNS].drop(columns=["candidate_digital_segment", "candidate_main_cpv", "buyer_match_type", "gap_bucket"], errors="ignore").isna().all(axis=1).any():
            raise RuntimeError(f"{split} contains a row with no numeric feature evidence")
        frame.to_parquet(path, index=False, compression="zstd")
        result["outputs"][split] = {
            "file": str(path),
            **profile(frame),
        }

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
