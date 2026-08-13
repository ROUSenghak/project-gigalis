#!/usr/bin/env python3
"""Build the digital Grand Ouest survival cohort from the v2 episode layer.

A cohort row is one procurement episode that satisfies all three conditions:

* the contracting buyer's structured postal address is in Grand Ouest;
* at least one CPV code falls in a digital division (32, 35, 48, 72);
* the episode contains an award notice (``ATTRIBUTION``).

The award condition matters: it is what makes the row a *contract* rather than
a tender that may never have been awarded, and it supplies time zero for
survival analysis (the guideline defines the lifetime as running from the
notification of the contract). Time zero is the earliest award-notice
publication date among the episode's notices.

This script only selects and describes. It does not impute: variables that are
unobservable stay missing and are counted in the quality summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.linkage import (
    DIGITAL_CPV_DIVISIONS,
    cpv_divisions,
    is_digital,
    normalize_buyer_for_blocking,
    normalize_text,
    parse_json_list,
)

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp")
COHORT_VERSION = "boamp_survival_cohort_v1.0"

#: Observation cutoff. The corpus ends on 2025-12-31; every episode without an
#: accepted successor by this date is right-censored, not a confirmed negative.
STUDY_CUTOFF = date(2025, 12, 31)

#: Only a typed, internally consistent duration is treated as reliable enough
#: to derive an expected contract end date. "unresolved" and "conflict" stay
#: missing rather than being back-filled with a default contract length.
RELIABLE_DURATION_QUALITY = ("typed_consensus",)

EPISODE_COLUMNS = [
    "episode_id", "constituent_notice_ids_json", "notice_count",
    "episode_origin_date", "episode_last_notice_date", "episode_notice_natures_json",
    "buyer_siren", "buyer_key", "buyer_name_raw", "buyer_name_normalized", "buyer_id_quality",
    "buyer_department", "buyer_region", "grand_ouest_flag",
    "episode_text", "main_cpv", "all_cpvs_json",
    "procedure_type", "framework_flag", "duration_months", "duration_quality",
    "amount_candidates_json", "episode_reconstruction_method",
]


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "build_survival_cohort.log", encoding="utf-8"),
        ],
    )


def award_dates_by_episode(output_dir: Path) -> pd.Series:
    """Earliest award-notice publication date for every Grand Ouest episode."""
    membership = pd.read_parquet(
        output_dir / "episode_membership.parquet", columns=["idweb", "episode_id"]
    )
    notices = pd.read_parquet(
        output_dir / "notices_grand_ouest.parquet",
        columns=["idweb", "nature", "publication_date"],
    )
    awards = notices.loc[notices["nature"] == "ATTRIBUTION"]
    joined = awards.merge(membership, on="idweb", how="inner")
    return joined.groupby("episode_id")["publication_date"].min()


def missingness_table(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for column in columns:
        series = frame[column]
        if series.dtype == object:
            missing = series.isna() | (series.astype(str).str.strip().isin(["", "[]", "None", "nan"]))
        else:
            missing = series.isna()
        report[column] = {
            "missing": int(missing.sum()),
            "missing_rate": round(float(missing.mean()), 4),
            "present": int((~missing).sum()),
        }
    return report


def build(project_root: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    cohort_path = output_dir / "survival_cohort.parquet"
    summary_path = output_dir / "survival_cohort_summary.json"
    if cohort_path.exists() and not force:
        raise FileExistsError(f"{cohort_path} already exists. Use --force to rebuild.")

    episodes_path = output_dir / "episodes_grand_ouest.parquet"
    if not episodes_path.exists():
        raise FileNotFoundError(episodes_path)

    episodes = pd.read_parquet(episodes_path, columns=EPISODE_COLUMNS)
    logging.info("Loaded %s Grand Ouest episodes", f"{len(episodes):,}")

    episodes["cpv_codes"] = episodes["all_cpvs_json"].map(parse_json_list)
    episodes["digital_flag"] = episodes["cpv_codes"].map(is_digital)
    episodes["natures"] = episodes["episode_notice_natures_json"].map(parse_json_list)
    episodes["has_award_notice"] = episodes["natures"].map(lambda values: "ATTRIBUTION" in values)

    digital = episodes.loc[episodes["digital_flag"]].copy()
    cohort = digital.loc[digital["has_award_notice"]].copy()
    logging.info(
        "Digital episodes: %s | with award notice: %s", f"{len(digital):,}", f"{len(cohort):,}"
    )

    awards = award_dates_by_episode(output_dir)
    cohort["award_date"] = cohort["episode_id"].map(awards)
    unresolved_award = int(cohort["award_date"].isna().sum())
    if unresolved_award:
        logging.warning("Dropping %s episodes with no resolvable award date", unresolved_award)
        cohort = cohort.loc[cohort["award_date"].notna()].copy()

    cohort["award_date"] = pd.to_datetime(cohort["award_date"])
    cohort["episode_origin_date"] = pd.to_datetime(cohort["episode_origin_date"])
    cohort["award_year"] = cohort["award_date"].dt.year

    # Expected contract end, only where a typed duration was agreed by the
    # episode's notices. Elsewhere it stays NaT and no timing evidence is used.
    reliable = cohort["duration_quality"].isin(RELIABLE_DURATION_QUALITY) & cohort["duration_months"].gt(0)
    cohort["duration_months_reliable"] = cohort["duration_months"].where(reliable)
    cohort["expected_end_date"] = pd.NaT
    cohort.loc[reliable, "expected_end_date"] = cohort.loc[reliable, "award_date"] + pd.to_timedelta(
        cohort.loc[reliable, "duration_months"] * 30.4375, unit="D"
    )

    cohort["buyer_name_blocking"] = cohort["buyer_name_normalized"].map(normalize_buyer_for_blocking)
    cohort["cohort_text"] = cohort["episode_text"].map(normalize_text)
    cohort["cpv_divisions_json"] = cohort["cpv_codes"].map(
        lambda codes: json.dumps(sorted(cpv_divisions(codes)), ensure_ascii=False, separators=(",", ":"))
    )
    cohort["digital_divisions_json"] = cohort["cpv_codes"].map(
        lambda codes: json.dumps(
            sorted(cpv_divisions(codes) & set(DIGITAL_CPV_DIVISIONS)), ensure_ascii=False, separators=(",", ":")
        )
    )
    # Primary segment for stratified Kaplan-Meier: the lowest-numbered digital
    # division present, so each episode contributes to exactly one curve.
    cohort["digital_segment"] = cohort["cpv_codes"].map(
        lambda codes: "CPV-" + sorted(cpv_divisions(codes) & set(DIGITAL_CPV_DIVISIONS))[0]
    )
    cohort["buyer_identifier_quality"] = cohort["buyer_id_quality"]
    cohort["study_cutoff"] = pd.Timestamp(STUDY_CUTOFF)
    cohort["days_to_cutoff"] = (cohort["study_cutoff"] - cohort["award_date"]).dt.days
    cohort["cohort_version"] = COHORT_VERSION

    negative_followup = int((cohort["days_to_cutoff"] < 0).sum())
    if negative_followup:
        logging.warning("Dropping %s episodes awarded after the study cutoff", negative_followup)
        cohort = cohort.loc[cohort["days_to_cutoff"] >= 0].copy()

    cohort = cohort.drop(columns=["cpv_codes", "natures"]).reset_index(drop=True)
    cohort.to_parquet(cohort_path, index=False, compression="zstd")

    quality_columns = [
        "buyer_siren", "buyer_name_normalized", "main_cpv", "episode_text",
        "duration_months", "duration_months_reliable", "amount_candidates_json",
        "award_date", "procedure_type", "framework_flag",
    ]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cohort_version": COHORT_VERSION,
        "source_episodes": str(episodes_path),
        "output_file": str(cohort_path),
        "study_cutoff": STUDY_CUTOFF.isoformat(),
        "digital_cpv_divisions": list(DIGITAL_CPV_DIVISIONS),
        "selection_funnel": {
            "grand_ouest_episodes": int(len(episodes)),
            "digital_cpv": int(len(digital)),
            "digital_with_award_notice": int(digital["has_award_notice"].sum()),
            "dropped_unresolved_award_date": unresolved_award,
            "dropped_awarded_after_cutoff": negative_followup,
            "cohort_rows": int(len(cohort)),
        },
        "rows_by_award_year": {
            str(year): int(count) for year, count in sorted(cohort["award_year"].value_counts().items())
        },
        "rows_by_region": {str(k): int(v) for k, v in cohort["buyer_region"].value_counts().items()},
        "rows_by_digital_segment": {
            str(k): int(v) for k, v in cohort["digital_segment"].value_counts().items()
        },
        "rows_by_buyer_id_quality": {
            str(k): int(v) for k, v in cohort["buyer_identifier_quality"].value_counts().items()
        },
        "rows_by_duration_quality": {
            str(k): int(v) for k, v in cohort["duration_quality"].value_counts(dropna=False).items()
        },
        "missingness": missingness_table(cohort, quality_columns),
        "follow_up_days": {
            "min": int(cohort["days_to_cutoff"].min()),
            "median": int(cohort["days_to_cutoff"].median()),
            "max": int(cohort["days_to_cutoff"].max()),
        },
        "meets_guideline_minimum_800_contracts": bool(len(cohort) >= 800),
        "validation_passed": bool(
            len(cohort) > 0
            and cohort["episode_id"].is_unique
            and cohort["award_date"].notna().all()
            and (cohort["days_to_cutoff"] >= 0).all()
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
        raise RuntimeError("Survival cohort validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
