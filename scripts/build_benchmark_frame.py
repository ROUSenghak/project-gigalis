#!/usr/bin/env python3
"""Define the national sampling frame the v3 benchmark is drawn from.

The v1 benchmark was drawn from a frame of 1,616 episodes whose definition
exists nowhere in the repository: there is no sampling script, no manifest, and
no record of what "digital award episode" meant when it was built. Its
inclusion probabilities span 0.0148 to 1.000 -- a 68-fold range with four
strata sampled as censuses -- and none of the resulting weights were ever used
in a metric.

This script is the missing definition, executable. It writes every eligible
episode with its stratum and the planned allocation, so the population is a
file that can be inspected and re-derived rather than a number in a CSV.

Nothing is sampled here. ``sample_benchmark_anchors.py`` draws from this frame,
which keeps "who could have been chosen" separable from "who was".
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

from boamp_pipeline.benchmark_frame import (  # noqa: E402
    FRAME_SEED,
    FRAME_VERSION,
    MAX_AWARD_YEAR,
    MAX_STRATUM_SAMPLE,
    MIN_AWARD_YEAR,
    MIN_STRATUM_SAMPLE,
    add_design_variables,
    allocate,
    eligible_frame,
)

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp/benchmark")

#: Headline probability-sample size. Chosen against the annotation budget: at a
#: renewal base rate near a quarter this yields roughly 100 positive anchors,
#: against 22 in v1, while staying reviewable in two waves.
DEFAULT_PROBABILITY_SAMPLE = 400

FRAME_COLUMNS = [
    "episode_id", "has_award_notice", "award_date", "award_year", "episode_origin_date",
    "buyer_block_key", "buyer_block_basis", "buyer_block_episode_count",
    "buyer_size_class", "buyer_key", "buyer_siren", "buyer_name_raw",
    "buyer_name_normalized", "buyer_name_blocking", "buyer_id_quality",
    "buyer_department", "buyer_region_national", "dept_group", "grand_ouest_flag",
    "digital_flag", "digital_segment", "main_cpv", "all_cpvs_json",
    "duration_months", "duration_quality", "duration_months_reliable",
    "expected_end_date", "expected_end_source", "framework_flag",
    "titulaire_names_json", "titulaire_count", "notice_count",
    "procedure_references_json", "episode_reconstruction_quality",
]


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "build_benchmark_frame.log", encoding="utf-8"),
        ],
    )


def crosstab(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).items()}


def build(project_root: Path, output_dir: Path, force: bool,
          probability_sample: int, digital_only: bool) -> dict[str, Any]:
    index_path = output_dir / "national_episode_index.parquet"
    frame_path = output_dir / "frame_national.parquet"
    summary_path = output_dir / "frame_strata_summary.json"
    if frame_path.exists() and not force:
        raise FileExistsError(f"{frame_path} already exists. Use --force to rebuild.")
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} does not exist; run scripts/build_national_episode_index.py first"
        )

    index = pd.read_parquet(index_path, columns=FRAME_COLUMNS)
    logging.info("National index: %s episodes", f"{len(index):,}")

    eligible = eligible_frame(index)
    logging.info(
        "Award-bearing, blockable, awarded %s-%s: %s",
        MIN_AWARD_YEAR, MAX_AWARD_YEAR, f"{len(eligible):,}",
    )

    # The digital scope is the study's, defined by CPV division rather than by a
    # classifier. A non-digital control arm is kept addressable but is not part
    # of the headline frame; v1 discovered 20% of its sample was out of scope
    # only after review, so scope is settled here instead.
    if digital_only:
        frame = eligible.loc[eligible["digital_flag"].astype(bool)].copy()
    else:
        frame = eligible.copy()
    logging.info("Frame: %s episodes", f"{len(frame):,}")

    frame = add_design_variables(frame)
    populations = frame["stratum_id"].value_counts().to_dict()
    allocation = allocate(populations, probability_sample)

    strata = []
    for name in sorted(populations):
        population = int(populations[name])
        planned = int(allocation.get(name, 0))
        strata.append({
            "stratum_id": name,
            "population_n": population,
            "planned_sample_n": planned,
            "inclusion_probability": round(planned / population, 6) if population else None,
            "design_weight": round(population / planned, 4) if planned else None,
        })

    frame["frame_seed"] = FRAME_SEED
    frame.to_parquet(frame_path, index=False, compression="zstd")

    smallest = min((s["planned_sample_n"] for s in strata), default=0)
    probabilities = [s["inclusion_probability"] for s in strata if s["inclusion_probability"]]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frame_version": FRAME_VERSION,
        "frame_seed": FRAME_SEED,
        "output_file": str(frame_path),
        "source_index": str(index_path),
        "definition": {
            "has_award_notice": True,
            "buyer_block_key_non_empty": True,
            "award_year_between": [MIN_AWARD_YEAR, MAX_AWARD_YEAR],
            "digital_cpv_division_only": digital_only,
            "rationale_award_year_ceiling": (
                "The national gap between a contract and its re-procurement peaks at "
                "three to four years, so an anchor awarded after 2021 cannot yet show a "
                "typical successor and would enter the frame as a near-certain negative."
            ),
        },
        "funnel": {
            "national_episodes": int(len(index)),
            "eligible_award_blockable_in_window": int(len(eligible)),
            "frame_rows": int(len(frame)),
        },
        "planned_probability_sample": probability_sample,
        "strata": strata,
        "stratum_count": len(strata),
        "smallest_planned_stratum_sample": smallest,
        "inclusion_probability_range": (
            [min(probabilities), max(probabilities)] if probabilities else None
        ),
        "inclusion_probability_spread": (
            round(max(probabilities) / min(probabilities), 2) if probabilities else None
        ),
        "rows_by_dept_group": crosstab(frame, "dept_group"),
        "rows_by_digital_segment": crosstab(frame, "digital_segment"),
        "rows_by_buyer_size_class": crosstab(frame, "buyer_size_class"),
        "rows_by_award_period": crosstab(frame, "award_period"),
        "rows_by_duration_available": crosstab(frame, "duration_available"),
        "rows_by_buyer_block_basis": crosstab(frame, "buyer_block_basis"),
        "validation_passed": bool(
            len(frame) > 0
            and frame["episode_id"].is_unique
            and smallest >= MIN_STRATUM_SAMPLE
            and all(s["planned_sample_n"] <= MAX_STRATUM_SAMPLE for s in strata)
            and sum(s["planned_sample_n"] for s in strata) == probability_sample
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--probability-sample", type=int, default=DEFAULT_PROBABILITY_SAMPLE)
    parser.add_argument(
        "--include-non-digital",
        action="store_true",
        help="Keep non-digital award episodes in the frame as a control arm.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    configure_logging(project_root)
    summary = build(
        project_root, output_dir, args.force, args.probability_sample,
        digital_only=not args.include_non_digital,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["validation_passed"]:
        raise RuntimeError("Benchmark frame validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
