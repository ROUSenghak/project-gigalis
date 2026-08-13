#!/usr/bin/env python3
"""Turn accepted successor links into a right-censored survival dataset.

One row per cohort episode. Time zero is the award-notice date.

* **Event (``event = 1``)** -- an observable successor procurement was accepted
  by the frozen linkage policy. The duration runs from the anchor's award date
  to the successor's first notice, because a renewed purchasing need becomes
  visible when the new tender is published, not when it is later awarded.
* **Censored (``event = 0``)** -- no successor was accepted before the study
  cutoff. The duration runs from the award date to the cutoff.

A censored row is *not* a contract that was proven never to be renewed. It is
a contract for which no successor was identified in BOAMP before the cutoff,
which is a weaker and different statement. Keeping these rows censored rather
than deleting them is the whole reason survival analysis is used here.

The script emits the main dataset plus stricter and looser linkage variants,
so the downstream analysis can show whether conclusions depend on where the
acceptance threshold was drawn.
"""

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

from scripts.evaluate_linkage import predict  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp")
SURVIVAL_VERSION = "boamp_survival_dataset_v1.0"

DAYS_PER_MONTH = 30.4375

COHORT_COLUMNS = [
    "episode_id", "award_date", "award_year", "study_cutoff", "days_to_cutoff",
    "digital_segment", "buyer_region", "buyer_department", "framework_flag",
    "buyer_identifier_quality", "duration_months_reliable", "duration_quality",
    "procedure_type", "notice_count", "buyer_name_raw", "buyer_key", "main_cpv",
    "episode_text",
]


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "build_survival_dataset.log", encoding="utf-8"),
        ],
    )


def assemble(cohort: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Join accepted links onto the cohort and derive duration and event."""
    first_links = (
        links.sort_values(["anchor_episode_id", "candidate_origin_date"])
        .groupby("anchor_episode_id", as_index=False)
        .first()
    )
    frame = cohort.merge(
        first_links[
            ["anchor_episode_id", "candidate_episode_id", "candidate_origin_date", "linkage_score", "gap_days"]
        ].rename(
            columns={
                "anchor_episode_id": "episode_id",
                "candidate_episode_id": "successor_episode_id",
                "candidate_origin_date": "successor_origin_date",
                "linkage_score": "successor_linkage_score",
                "gap_days": "successor_gap_days",
            }
        ),
        on="episode_id",
        how="left",
    )
    frame["event"] = frame["successor_episode_id"].notna().astype(int)
    frame["duration_days"] = frame["successor_gap_days"].fillna(frame["days_to_cutoff"]).astype(int)
    frame["duration_months"] = (frame["duration_days"] / DAYS_PER_MONTH).round(2)
    frame["survival_version"] = SURVIVAL_VERSION
    return frame


def validate(frame: pd.DataFrame) -> dict[str, Any]:
    """Structural checks that must hold before any model is fitted."""
    return {
        "rows": int(len(frame)),
        "unique_episodes": int(frame["episode_id"].nunique()),
        "events": int(frame["event"].sum()),
        "censored": int((frame["event"] == 0).sum()),
        "events_plus_censored_equals_rows": bool(
            int(frame["event"].sum()) + int((frame["event"] == 0).sum()) == len(frame)
        ),
        "negative_durations": int((frame["duration_days"] < 0).sum()),
        "zero_durations": int((frame["duration_days"] == 0).sum()),
        "duplicate_episodes": int(len(frame) - frame["episode_id"].nunique()),
        "successor_equals_anchor": int((frame["successor_episode_id"] == frame["episode_id"]).sum()),
    }


def describe(frame: pd.DataFrame) -> dict[str, Any]:
    events = frame.loc[frame["event"] == 1]
    by_segment = (
        frame.groupby("digital_segment")
        .agg(rows=("episode_id", "size"), events=("event", "sum"))
        .assign(event_rate=lambda f: (f["events"] / f["rows"]).round(4))
    )
    by_region = (
        frame.groupby("buyer_region")
        .agg(rows=("episode_id", "size"), events=("event", "sum"))
        .assign(event_rate=lambda f: (f["events"] / f["rows"]).round(4))
    )
    return {
        "event_rate": round(float(frame["event"].mean()), 4),
        "median_time_to_successor_months": (
            round(float(events["duration_months"].median()), 2) if len(events) else None
        ),
        "by_digital_segment": by_segment.to_dict(orient="index"),
        "by_region": by_region.to_dict(orient="index"),
        "segments_meeting_30_event_minimum": [
            str(segment) for segment, row in by_segment.iterrows() if row["events"] >= 30
        ],
        "max_cox_covariates_at_10_events_each": int(frame["event"].sum() // 10),
    }


def linkage_bias(frame: pd.DataFrame) -> dict[str, Any]:
    """Are linked anchors systematically better documented than unlinked ones?

    If linkage succeeds mainly where evidence happens to be complete, the
    observed events are a biased sample and survival estimates inherit that
    bias. Reporting it is required for honest interpretation.
    """
    frame = frame.copy()
    frame["has_trusted_siren"] = frame["buyer_identifier_quality"].eq("trusted_siren")
    frame["has_reliable_duration"] = frame["duration_months_reliable"].notna()
    frame["is_framework"] = frame["framework_flag"].astype(bool)
    report: dict[str, Any] = {}
    for column in ["has_trusted_siren", "has_reliable_duration", "is_framework"]:
        linked = frame.loc[frame["event"] == 1, column].mean()
        unlinked = frame.loc[frame["event"] == 0, column].mean()
        report[column] = {
            "linked_share": round(float(linked), 4),
            "unlinked_share": round(float(unlinked), 4),
            "difference": round(float(linked - unlinked), 4),
        }
    report["link_rate_by_award_year"] = {
        str(year): round(float(value), 4)
        for year, value in frame.groupby("award_year")["event"].mean().items()
    }
    report["link_rate_by_segment"] = {
        str(segment): round(float(value), 4)
        for segment, value in frame.groupby("digital_segment")["event"].mean().items()
    }
    return report


def build(project_root: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    dataset_path = output_dir / "survival_dataset.parquet"
    summary_path = output_dir / "survival_dataset_summary.json"
    if dataset_path.exists() and not force:
        raise FileExistsError(f"{dataset_path} already exists. Use --force to rebuild.")

    cohort = pd.read_parquet(output_dir / "survival_cohort.parquet", columns=COHORT_COLUMNS)
    cohort["award_date"] = pd.to_datetime(cohort["award_date"])
    candidates = pd.read_parquet(output_dir / "linkage_candidates.parquet")
    config = json.loads((output_dir / "linkage_config.json").read_text(encoding="utf-8"))
    method = config["selected_method"]
    threshold = float(config["selected_threshold"])
    logging.info("Applying frozen policy %s at threshold %s", method, threshold)

    # Arms mirror the linkage sensitivity analysis: three strictness levels of
    # the frozen policy plus one deliberately higher-recall, lower-precision
    # contrast. Conclusions that hold across all four do not depend on where
    # the acceptance line was drawn.
    arms = [
        ("strict", method, threshold + 10.0),
        ("main", method, threshold),
        ("looser", method, threshold - 10.0),
        ("contrast_high_recall", config["contrast_method"], float(config["contrast_threshold"])),
    ]

    variants: dict[str, Any] = {}
    for label, arm_method, arm_threshold in arms:
        predictions = predict(candidates, arm_method, arm_threshold)
        links = predictions.loc[predictions["predicted_rank"] == 1] if len(predictions) else predictions
        frame = assemble(cohort, links)
        checks = validate(frame)
        variants[label] = {
            "method": arm_method,
            "threshold": arm_threshold,
            "validation": checks,
            "description": describe(frame),
        }
        if label == "main":
            frame.to_parquet(dataset_path, index=False, compression="zstd")
            variants[label]["linkage_bias"] = linkage_bias(frame)
            main_checks = checks
        else:
            frame.to_parquet(
                output_dir / f"survival_dataset_{label}.parquet", index=False, compression="zstd"
            )
        logging.info(
            "%s (%s @ %.1f) -> %s events / %s rows",
            label, arm_method, arm_threshold, checks["events"], checks["rows"],
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "survival_version": SURVIVAL_VERSION,
        "output_file": str(dataset_path),
        "linkage_method": method,
        "linkage_threshold": threshold,
        "study_cutoff": str(cohort["study_cutoff"].iloc[0].date()),
        "time_zero": "award notice publication date",
        "event_definition": "first accepted observable successor episode, timed by its first notice",
        "censoring_definition": "no accepted observable successor before the study cutoff",
        "variants": variants,
        "validation_passed": bool(
            main_checks["events_plus_censored_equals_rows"]
            and main_checks["negative_durations"] == 0
            and main_checks["duplicate_episodes"] == 0
            and main_checks["successor_equals_anchor"] == 0
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
        raise RuntimeError("Survival dataset validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
