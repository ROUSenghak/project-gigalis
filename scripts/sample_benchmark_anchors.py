#!/usr/bin/env python3
"""Draw the benchmark anchors, in three frames that must never be pooled.

``PROBABILITY``
    A stratified systematic sample of the national frame. Every national
    estimate comes from here and from nowhere else, because these are the only
    anchors with a known inclusion probability.

``ENRICHMENT_DECLARED``
    Contracts whose successor announced itself in words. Purposively recruited,
    so they carry no inclusion probability and cannot enter a national
    estimate -- they over-represent buyers who write detailed notices, which is
    exactly the population a text-similarity method handles best. They exist to
    supply hard positives for error analysis.

``ENRICHMENT_EXHAUSTIVE_NEG``
    Contracts whose entire candidate pool is small enough to review completely.
    These are the only anchors that can yield a *verified* negative rather than
    "nothing in the list I was shown", which is the defect that made v1's 70
    negatives uninterpretable.

The separation is enforced downstream by
:func:`boamp_pipeline.benchmark_frame.assert_no_frame_mixing`, which raises
rather than silently averaging across designs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.benchmark_frame import (  # noqa: E402
    FRAME_SEED,
    FRAME_VERSION,
    add_design_variables,
    draw_probability_sample,
    stable_hash,
    weights_reconstruct_population,
)
from boamp_pipeline.exposure import EXHAUSTIVE_MAX, pool_for_anchor  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp_v2/benchmark_v3")
SAMPLE_VERSION = "boamp_benchmark_anchors_v3.0"

DEFAULT_PROBABILITY_SAMPLE = 400
DEFAULT_DECLARED_ENRICHMENT = 200
DEFAULT_EXHAUSTIVE_NEG = 100

#: Wave 1 is annotated first. Order is randomised *within stratum*, so if wave 2
#: is never annotated the inclusion probability is still exactly
#: ``n_annotated_h / N_h`` and the design stays valid -- a partial benchmark
#: remains a probability sample rather than a truncated one.
DEFAULT_WAVE_ONE_SHARE = 0.5

INDEX_COLUMNS = [
    "episode_id", "episode_origin_date", "award_date", "buyer_block_key",
    "buyer_name_raw", "buyer_department", "dept_group", "digital_flag",
    "digital_segment", "main_cpv", "all_cpvs_json", "expected_end_date",
    "expected_end_source", "duration_months_reliable", "framework_flag",
    "procedure_references_json", "titulaire_names_json", "notice_count",
    "buyer_size_class", "buyer_block_episode_count", "has_award_notice",
    "award_year", "grand_ouest_flag",
]


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "sample_benchmark_anchors.log", encoding="utf-8"),
        ],
    )


def block_index(index: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Group every national episode by its buyer blocking key."""
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index.to_dict(orient="records"):
        key = row.get("buyer_block_key") or ""
        if key:
            blocks[key].append(row)
    return blocks


def pool_sizes(anchors: pd.DataFrame, blocks: dict[str, list[dict[str, Any]]]) -> pd.Series:
    """Candidate-pool size for each anchor, which decides exposure mode."""
    sizes = []
    for anchor in anchors.itertuples(index=False):
        pool = pool_for_anchor(
            anchor.award_date, anchor.episode_id, blocks.get(anchor.buyer_block_key, [])
        )
        sizes.append(len(pool))
    return pd.Series(sizes, index=anchors.index, dtype=int)


def assign_waves(anchors: pd.DataFrame, share: float) -> pd.Series:
    """Randomise annotation order within stratum and cut into two waves."""
    waves = pd.Series(2, index=anchors.index, dtype=int)
    for _, group in anchors.groupby("stratum_id", sort=False):
        order = sorted(
            group.index, key=lambda i: stable_hash("wave", anchors.at[i, "episode_id"])
        )
        take = max(1, int(round(len(order) * share)))
        waves.loc[order[:take]] = 1
    return waves


def build(project_root: Path, output_dir: Path, force: bool, probability_n: int,
          declared_n: int, exhaustive_neg_n: int, wave_one_share: float) -> dict[str, Any]:
    anchors_path = output_dir / "anchors.parquet"
    summary_path = output_dir / "anchors_summary.json"
    if anchors_path.exists() and not force:
        raise FileExistsError(f"{anchors_path} already exists. Use --force to rebuild.")

    frame = pd.read_parquet(output_dir / "frame_national.parquet")
    index = pd.read_parquet(output_dir / "national_episode_index.parquet", columns=INDEX_COLUMNS)
    logging.info("Frame: %s | national index: %s", f"{len(frame):,}", f"{len(index):,}")

    frame = add_design_variables(frame) if "stratum_id" not in frame.columns else frame
    blocks = block_index(index)
    logging.info("Buyer blocks: %s", f"{len(blocks):,}")

    # --- probability frame ------------------------------------------------
    probability, design = draw_probability_sample(frame, probability_n)
    probability["frame"] = "PROBABILITY"
    logging.info("Probability sample: %s anchors", f"{len(probability):,}")
    if not weights_reconstruct_population(probability, len(frame)):
        raise RuntimeError(
            "design weights do not reconstruct the frame size; the sampling design is incoherent"
        )

    already = set(probability["episode_id"])

    # --- declared enrichment ----------------------------------------------
    declared = pd.read_parquet(output_dir / "declared_predecessor_links.parquet")
    reliable = declared.loc[
        declared["resolution_status"].eq("resolved")
        & declared["resolution_confidence"].isin(["strong", "medium"])
    ].copy()
    # The anchor is the *predecessor*: the contract whose successor announced
    # itself. It must be in the frame, or it is out of study scope.
    in_frame = set(frame["episode_id"])
    reliable = reliable.loc[reliable["predecessor_episode_id"].isin(in_frame)]
    reliable = reliable.loc[~reliable["predecessor_episode_id"].isin(already)]
    reliable = reliable.drop_duplicates("predecessor_episode_id")
    logging.info("Declared-predecessor anchors available in frame: %s", f"{len(reliable):,}")

    declared_anchors = frame.loc[
        frame["episode_id"].isin(set(reliable["predecessor_episode_id"]))
    ].copy()
    if len(declared_anchors) > declared_n:
        keep = sorted(
            declared_anchors["episode_id"],
            key=lambda e: stable_hash("declared", e),
        )[:declared_n]
        declared_anchors = declared_anchors.loc[declared_anchors["episode_id"].isin(set(keep))]
    declared_anchors["frame"] = "ENRICHMENT_DECLARED"
    declared_anchors["inclusion_probability"] = np.nan
    declared_anchors["design_weight"] = np.nan
    already |= set(declared_anchors["episode_id"])
    logging.info("Declared enrichment: %s anchors", f"{len(declared_anchors):,}")

    # --- exhaustive-negative enrichment -----------------------------------
    # Restricted to anchors whose whole pool can be reviewed, because only
    # those can produce a negative that means "verified against the pool".
    remaining = frame.loc[~frame["episode_id"].isin(already)].copy()
    remaining["pool_size"] = pool_sizes(remaining, blocks)
    small = remaining.loc[remaining["pool_size"].between(1, EXHAUSTIVE_MAX)].copy()
    logging.info(
        "Frame anchors with a fully reviewable pool (1-%s): %s",
        EXHAUSTIVE_MAX, f"{len(small):,}",
    )
    if len(small) > exhaustive_neg_n:
        keep = sorted(small["episode_id"], key=lambda e: stable_hash("exhaustive", e))
        small = small.loc[small["episode_id"].isin(set(keep[:exhaustive_neg_n]))]
    small["frame"] = "ENRICHMENT_EXHAUSTIVE_NEG"
    # Conditional on the small-pool restriction, so recorded but not usable as
    # a national weight.
    small["inclusion_probability"] = (
        len(small) / len(remaining.loc[remaining["pool_size"].between(1, EXHAUSTIVE_MAX)])
        if len(remaining) else np.nan
    )
    small["design_weight"] = np.nan
    logging.info("Exhaustive-negative enrichment: %s anchors", f"{len(small):,}")

    # --- assemble ---------------------------------------------------------
    anchors = pd.concat([probability, declared_anchors, small], ignore_index=True)
    anchors["pool_size"] = pool_sizes(anchors, blocks)
    anchors["wave"] = assign_waves(anchors, wave_one_share)
    anchors["sample_version"] = SAMPLE_VERSION
    anchors["frame_seed"] = FRAME_SEED
    anchors = anchors.sort_values(["frame", "stratum_id", "episode_id"]).reset_index(drop=True)
    anchors.to_parquet(anchors_path, index=False, compression="zstd")

    pool_distribution = anchors["pool_size"].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    exhaustive_share = float(anchors["pool_size"].le(EXHAUSTIVE_MAX).mean())

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_version": SAMPLE_VERSION,
        "frame_version": FRAME_VERSION,
        "frame_seed": FRAME_SEED,
        "output_file": str(anchors_path),
        "frame_separation": (
            "PROBABILITY carries inclusion probabilities and is the only frame "
            "permitted in a national estimate. ENRICHMENT_* frames are purposive and "
            "are used for error analysis and verified negatives only."
        ),
        "anchors_total": int(len(anchors)),
        "anchors_by_frame": {
            str(k): int(v) for k, v in anchors["frame"].value_counts().items()
        },
        "anchors_by_wave": {str(k): int(v) for k, v in anchors["wave"].value_counts().items()},
        "probability_sample": {
            "requested": probability_n,
            "drawn": int(len(probability)),
            "strata": len(design),
            "weights_reconstruct_frame": weights_reconstruct_population(probability, len(frame)),
            "estimated_frame_total": round(float(probability["design_weight"].sum()), 1),
            "actual_frame_total": int(len(frame)),
            "rows_by_dept_group": {
                str(k): int(v) for k, v in probability["dept_group"].value_counts().items()
            },
            "rows_by_digital_segment": {
                str(k): int(v) for k, v in probability["digital_segment"].value_counts().items()
            },
        },
        "declared_enrichment": {
            "available_in_frame": int(len(reliable)),
            "drawn": int(len(declared_anchors)),
            "note": (
                "Anchor is the predecessor contract; its successor declared the "
                "succession in words. Recruitment used text-free resolvers only."
            ),
        },
        "exhaustive_negative_enrichment": {
            "eligible_small_pool_anchors": int(
                len(remaining.loc[remaining["pool_size"].between(1, EXHAUSTIVE_MAX)])
            ),
            "drawn": int(len(small)),
            "pool_size_ceiling": EXHAUSTIVE_MAX,
        },
        "pool_size": {
            "min": int(pool_distribution["min"]),
            "p25": float(pool_distribution["25%"]),
            "median": float(pool_distribution["50%"]),
            "p75": float(pool_distribution["75%"]),
            "p90": float(pool_distribution["90%"]),
            "max": int(pool_distribution["max"]),
            "share_fully_reviewable": round(exhaustive_share, 4),
            "anchors_with_empty_pool": int((anchors["pool_size"] == 0).sum()),
        },
        "validation_passed": bool(
            len(anchors) > 0
            and anchors["episode_id"].is_unique
            and weights_reconstruct_population(probability, len(frame))
            and probability["inclusion_probability"].between(0, 1, inclusive="right").all()
            and declared_anchors["inclusion_probability"].isna().all()
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
    parser.add_argument("--declared-enrichment", type=int, default=DEFAULT_DECLARED_ENRICHMENT)
    parser.add_argument("--exhaustive-negatives", type=int, default=DEFAULT_EXHAUSTIVE_NEG)
    parser.add_argument("--wave-one-share", type=float, default=DEFAULT_WAVE_ONE_SHARE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    configure_logging(project_root)
    summary = build(
        project_root, output_dir, args.force, args.probability_sample,
        args.declared_enrichment, args.exhaustive_negatives, args.wave_one_share,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["validation_passed"]:
        raise RuntimeError("Benchmark anchor sampling validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
