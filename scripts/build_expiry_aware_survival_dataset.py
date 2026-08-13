#!/usr/bin/env python3
"""Build a parallel survival dataset from expiry-aware successor links."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_survival_dataset import (  # noqa: E402
    COHORT_COLUMNS,
    assemble,
    describe,
    linkage_bias,
    validate,
)


DEFAULT_OUTPUT_DIR = Path("data/processed/boamp")
SURVIVAL_VERSION = "boamp_survival_dataset_expiry_aware_v1.0"


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "build_expiry_aware_survival_dataset.log", encoding="utf-8"),
        ],
    )


def build(project_root: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    dataset_path = output_dir / "survival_dataset_expiry_aware.parquet"
    summary_path = output_dir / "survival_dataset_expiry_aware_summary.json"
    existing = [path for path in (dataset_path, summary_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"{existing[0]} already exists. Use --force to rebuild parallel artifacts.")

    links_path = output_dir / "accepted_successor_links_expiry_aware.parquet"
    linkage_summary_path = output_dir / "expiry_aware_linkage_summary.json"
    if not links_path.exists():
        raise FileNotFoundError(
            f"{links_path} does not exist; run scripts/evaluate_expiry_aware_linkage.py first"
        )

    cohort = pd.read_parquet(output_dir / "survival_cohort.parquet", columns=COHORT_COLUMNS)
    cohort["award_date"] = pd.to_datetime(cohort["award_date"])
    links = pd.read_parquet(links_path)
    frame = assemble(cohort, links)

    timing_columns = [
        "anchor_episode_id",
        "anchor_expected_end_date",
        "expected_end_source",
        "explicit_start_date",
        "explicit_end_date",
        "days_from_expected_end",
        "timing_class",
        "selection_method",
    ]
    timing = links[timing_columns].rename(columns={"anchor_episode_id": "episode_id"})
    frame = frame.merge(timing, on="episode_id", how="left", validate="one_to_one")
    frame["survival_version"] = SURVIVAL_VERSION
    frame.to_parquet(dataset_path, index=False, compression="zstd")

    checks = validate(frame)
    linkage_summary = (
        json.loads(linkage_summary_path.read_text(encoding="utf-8"))
        if linkage_summary_path.exists()
        else {}
    )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "survival_version": SURVIVAL_VERSION,
        "output_file": str(dataset_path),
        "accepted_links_file": str(links_path),
        "linkage_method": "M_E_expiry_aware_text",
        "time_zero": "award notice publication date",
        "event_definition": "first accepted observable successor episode, timed by its first notice",
        "censoring_definition": "no accepted observable successor before the study cutoff",
        "validation": checks,
        "description": describe(frame),
        "linkage_bias": linkage_bias(frame),
        "promotion_gate": linkage_summary.get("promotion_gate"),
        "event_interpretation": "observable successor procurement, not a confirmed legal renewal",
        "validation_passed": bool(
            checks["events_plus_censored_equals_rows"]
            and checks["negative_durations"] == 0
            and checks["duplicate_episodes"] == 0
            and checks["successor_equals_anchor"] == 0
            and links["anchor_episode_id"].is_unique
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    configure_logging(project_root)
    summary = build(project_root, output_dir, args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["validation_passed"]:
        raise RuntimeError("Expiry-aware survival dataset validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
