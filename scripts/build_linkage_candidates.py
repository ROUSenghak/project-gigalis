#!/usr/bin/env python3
"""Generate and score successor candidates for the digital Grand Ouest cohort.

For every cohort episode (the *anchor*, a digital Grand Ouest contract with a
known award date), this script retrieves the later procurement episodes that
could plausibly be its successor and computes the linkage features.

Blocking rule -- a candidate is retrieved when all of the following hold:

* it is a Grand Ouest episode other than the anchor itself;
* its first notice is published at least :data:`MIN_GAP_DAYS` after the
  anchor's award date, so that concurrent lots of the same programme are not
  mistaken for succession;
* that gap is at most :data:`MAX_GAP_DAYS` (~8 years);
* it shares the anchor's buyer, by exact ``buyer_key`` or by exact normalised
  buyer name.

The candidate pool is deliberately *not* restricted to digital or award-bearing
episodes. Only 23 of the 28 manually confirmed successors carry a digital CPV
and only 20 carry an award notice, so filtering the pool the same way the
anchor cohort is filtered would cap achievable recall at 16/28. A renewed
purchasing need is announced by a tender, so a successor is timed by its first
notice.

Scoring produces four components -- buyer, text, CPV, time -- each of which may
be genuinely unobservable. Unobservable components are recorded as NaN and are
excluded from the combined score rather than counted as disagreement; see
``boamp_pipeline.linkage.weighted_score``.
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

from boamp_pipeline.linkage import (
    DEFAULT_WEIGHTS,
    MAX_GAP_DAYS,
    MIN_GAP_DAYS,
    buyer_score,
    buyer_similarity,
    classify_buyer_match,
    cpv_continuity_score,
    evidence_completeness,
    is_digital,
    normalize_buyer_for_blocking,
    normalize_text,
    parse_json_list,
    tfidf_similarities,
    time_plausibility_score,
    weighted_score,
)

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp_v2")
CANDIDATE_VERSION = "boamp_linkage_candidates_v1.0"

POOL_COLUMNS = [
    "episode_id", "episode_origin_date", "episode_notice_natures_json",
    "buyer_siren", "buyer_key", "buyer_name_raw", "buyer_name_normalized",
    "buyer_region", "episode_text", "main_cpv", "all_cpvs_json",
    "notice_count", "framework_flag",
]


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "build_linkage_candidates.log", encoding="utf-8"),
        ],
    )


def load_pool(output_dir: Path) -> pd.DataFrame:
    pool = pd.read_parquet(output_dir / "episodes_grand_ouest.parquet", columns=POOL_COLUMNS)
    pool["origin_date"] = pd.to_datetime(pool["episode_origin_date"])
    pool["buyer_name_blocking"] = pool["buyer_name_normalized"].map(normalize_buyer_for_blocking)
    pool["pool_text"] = pool["episode_text"].map(normalize_text)
    pool["cpv_codes"] = pool["all_cpvs_json"].map(parse_json_list)
    pool["siren_clean"] = pool["buyer_siren"].fillna("").astype(str).str.strip()
    pool["row_pos"] = np.arange(len(pool))
    return pool


def build_blocking_index(pool: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    by_key: dict[str, np.ndarray] = {
        key: group["row_pos"].to_numpy()
        for key, group in pool.groupby("buyer_key", sort=False)
        if str(key).strip()
    }
    by_name: dict[str, np.ndarray] = {
        key: group["row_pos"].to_numpy()
        for key, group in pool.groupby("buyer_name_blocking", sort=False)
        if str(key).strip()
    }
    return by_key, by_name


def build(project_root: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    candidates_path = output_dir / "linkage_candidates.parquet"
    summary_path = output_dir / "linkage_candidates_summary.json"
    if candidates_path.exists() and not force:
        raise FileExistsError(f"{candidates_path} already exists. Use --force to rebuild.")

    cohort = pd.read_parquet(output_dir / "survival_cohort.parquet")
    cohort["award_date"] = pd.to_datetime(cohort["award_date"])
    cohort["expected_end_date"] = pd.to_datetime(cohort["expected_end_date"])
    logging.info("Cohort anchors: %s", f"{len(cohort):,}")

    pool = load_pool(output_dir)
    by_key, by_name = build_blocking_index(pool)
    logging.info("Candidate pool: %s Grand Ouest episodes", f"{len(pool):,}")

    pool_origin = pool["origin_date"].to_numpy()
    pool_ids = pool["episode_id"].to_numpy()
    pool_texts = pool["pool_text"].to_numpy()
    pool_siren = pool["siren_clean"].to_numpy()
    pool_name = pool["buyer_name_blocking"].to_numpy()
    pool_cpv = pool["cpv_codes"].to_numpy()

    frames: list[pd.DataFrame] = []
    anchors_without_candidates = 0

    for position, anchor in enumerate(cohort.itertuples(index=False), start=1):
        award_date = np.datetime64(anchor.award_date)
        earliest = award_date + np.timedelta64(MIN_GAP_DAYS, "D")
        horizon = award_date + np.timedelta64(MAX_GAP_DAYS, "D")

        positions: set[int] = set()
        key_block = by_key.get(str(anchor.buyer_key or ""))
        if key_block is not None:
            positions.update(key_block.tolist())
        name_block = by_name.get(str(anchor.buyer_name_blocking or ""))
        if name_block is not None:
            positions.update(name_block.tolist())
        if not positions:
            anchors_without_candidates += 1
            continue

        index = np.fromiter(positions, dtype=np.int64, count=len(positions))
        origins = pool_origin[index]
        keep = (origins >= earliest) & (origins <= horizon) & (pool_ids[index] != anchor.episode_id)
        index = index[keep]
        if index.size == 0:
            anchors_without_candidates += 1
            continue

        anchor_siren = str(anchor.buyer_siren or "").strip()
        anchor_name = str(anchor.buyer_name_blocking or "")
        anchor_cpv = parse_json_list(anchor.all_cpvs_json)
        expected_duration = (
            float(anchor.duration_months_reliable)
            if pd.notna(anchor.duration_months_reliable)
            else None
        )

        candidate_texts = pool_texts[index].tolist()
        word_sim = tfidf_similarities(anchor.cohort_text, candidate_texts, analyzer="word")
        char_sim = tfidf_similarities(anchor.cohort_text, candidate_texts, analyzer="char")
        gaps = (pool_origin[index] - award_date).astype("timedelta64[D]").astype(int)

        match_types: list[str] = []
        name_sims: list[float] = []
        cpv_scores: list[float] = []
        time_scores: list[float] = []
        for offset, pool_row in enumerate(index):
            match_types.append(
                classify_buyer_match(anchor_siren, pool_siren[pool_row], anchor_name, pool_name[pool_row])
            )
            name_sims.append(buyer_similarity(anchor_name, pool_name[pool_row]))
            cpv_scores.append(cpv_continuity_score(anchor_cpv, pool_cpv[pool_row]))
            time_scores.append(time_plausibility_score(gaps[offset], expected_duration))

        text_score = np.maximum(word_sim, char_sim)
        components = [
            {
                "buyer": buyer_score(match_types[i]),
                "text": float(text_score[i]),
                "cpv": cpv_scores[i],
                "time": time_scores[i],
            }
            for i in range(index.size)
        ]

        frames.append(
            pd.DataFrame(
                {
                    "anchor_episode_id": anchor.episode_id,
                    "anchor_award_date": anchor.award_date,
                    "anchor_digital_segment": anchor.digital_segment,
                    "anchor_buyer_name": anchor.buyer_name_raw,
                    "anchor_region": anchor.buyer_region,
                    "candidate_episode_id": pool_ids[index],
                    "candidate_origin_date": pool_origin[index],
                    "candidate_buyer_name": pool["buyer_name_raw"].to_numpy()[index],
                    "gap_days": gaps,
                    "buyer_match_type": match_types,
                    "buyer_name_similarity": name_sims,
                    "word_tfidf_similarity": word_sim,
                    "char_tfidf_similarity": char_sim,
                    "buyer_component": [c["buyer"] for c in components],
                    "text_component": [c["text"] for c in components],
                    "cpv_component": [c["cpv"] for c in components],
                    "time_component": [c["time"] for c in components],
                    "evidence_components": [evidence_completeness(c) for c in components],
                    "linkage_score": [weighted_score(c, DEFAULT_WEIGHTS) for c in components],
                    "candidate_is_digital": [bool(is_digital(pool_cpv[i])) for i in index],
                }
            )
        )

        if position % 500 == 0:
            logging.info("Scored %s / %s anchors", f"{position:,}", f"{len(cohort):,}")

    candidates = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    candidates["candidate_rank"] = (
        candidates.groupby("anchor_episode_id")["linkage_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    candidates = candidates.sort_values(["anchor_episode_id", "candidate_rank"]).reset_index(drop=True)
    candidates["candidate_version"] = CANDIDATE_VERSION
    candidates.to_parquet(candidates_path, index=False, compression="zstd")

    per_anchor = candidates.groupby("anchor_episode_id").size()
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_version": CANDIDATE_VERSION,
        "output_file": str(candidates_path),
        "weights": dict(DEFAULT_WEIGHTS),
        "min_gap_days": MIN_GAP_DAYS,
        "max_gap_days": MAX_GAP_DAYS,
        "blocking_rule": (
            "same buyer_key OR same normalized buyer name; candidate origin between "
            f"{MIN_GAP_DAYS} and {MAX_GAP_DAYS} days after the anchor award date"
        ),
        "cohort_anchors": int(len(cohort)),
        "anchors_with_candidates": int(per_anchor.size),
        "anchors_without_candidates": int(anchors_without_candidates),
        "candidate_pairs": int(len(candidates)),
        "candidates_per_anchor": {
            "min": int(per_anchor.min()),
            "median": float(per_anchor.median()),
            "mean": round(float(per_anchor.mean()), 1),
            "max": int(per_anchor.max()),
        },
        "buyer_match_type_counts": {
            str(k): int(v) for k, v in candidates["buyer_match_type"].value_counts().items()
        },
        "component_missing_rates": {
            "cpv": round(float(candidates["cpv_component"].isna().mean()), 4),
            "time": round(float(candidates["time_component"].isna().mean()), 4),
        },
        "validation_passed": bool(
            len(candidates) > 0
            and (candidates["gap_days"] >= MIN_GAP_DAYS).all()
            and (candidates["gap_days"] <= MAX_GAP_DAYS).all()
            and (candidates["anchor_episode_id"] != candidates["candidate_episode_id"]).all()
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
        raise RuntimeError("Candidate generation validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
