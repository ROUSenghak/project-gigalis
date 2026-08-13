#!/usr/bin/env python3
"""Assemble the adjudicated labels into dev, validation and a sealed test split.

Splitting is on ``buyer_block_key``, not on the anchor. A buyer's notices share
boilerplate, procedure references and often suppliers, so an anchor-level split
would let a method tuned on dev memorise a buyer's vocabulary and meet it again
in test. Blocks are hashed within each stratum, so the three splits stay
balanced across geography and technological segment.

The three splits have distinct jobs, and the separation is the fix for the leak
the existing study admits to. Thresholds are chosen on dev. Method selection
and adoption tests happen on validation. The sealed test is opened once, at the
end, through an audited path.
"""

from __future__ import annotations

import argparse
import hashlib
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

from boamp_pipeline.annotation_schema import (  # noqa: E402
    EVENT_SETS,
    HAS_RENEWAL_SUCCESSOR,
    NO_RENEWAL_POOL_EXHAUSTIVE,
    SCHEMA_VERSION,
)
from boamp_pipeline.sealed_split import write_manifest  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp/benchmark")
# Schema identifiers may evolve without creating parallel project versions.
BENCHMARK_SCHEMA = "boamp_national_benchmark_schema_1.0"

SPLIT_SALT = "boamp-benchmark-v3-split"
SPLIT_SHARES = (("dev", 50), ("validation", 25), ("sealed_test", 25))

#: Study cutoff, inherited from the frozen cohort definition.
STUDY_CUTOFF = pd.Timestamp("2025-12-31")

#: A contract whose expected end falls near or after the cutoff cannot have had
#: time to show a successor, so counting it as a confirmed negative would reward
#: under-linking on recent anchors -- the same error as a corpus-relative
#: negatives, in a new disguise.
CENSORING_MARGIN_DAYS = 180


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "assemble_benchmark.log", encoding="utf-8"),
        ],
    )


def split_for_block(buyer_block_key: str) -> str:
    """Deterministic split assignment for a whole buyer block."""
    digest = hashlib.sha256(f"{SPLIT_SALT}|{buyer_block_key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    cumulative = 0
    for name, share in SPLIT_SHARES:
        cumulative += share
        if bucket < cumulative:
            return name
    return SPLIT_SHARES[-1][0]


def build(project_root: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    sealed_dir = output_dir / "sealed"
    manifest_path = output_dir / "benchmark_manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} already exists. Use --force to rebuild.")

    adjudicated = pd.read_parquet(output_dir / "labels_adjudicated.parquet")
    anchors = pd.read_parquet(output_dir / "anchors.parquet")
    index = pd.read_parquet(
        output_dir / "national_episode_index.parquet",
        columns=["episode_id", "award_date", "expected_end_date", "duration_months_reliable"],
    ).set_index("episode_id")
    logging.info("Adjudicated labels: %s", f"{len(adjudicated):,}")

    settled = adjudicated.loc[adjudicated["label"].notna()].copy()
    if settled.empty:
        raise RuntimeError("no settled labels; resolve the adjudication queue first")

    anchor_meta = anchors.set_index("episode_id")
    settled["buyer_block_key"] = settled["anchor_episode_id"].map(anchor_meta["buyer_block_key"])
    settled["frame"] = settled["anchor_episode_id"].map(anchor_meta["frame"])
    settled["stratum_id"] = settled["anchor_episode_id"].map(anchor_meta["stratum_id"])
    settled["inclusion_probability"] = settled["anchor_episode_id"].map(
        anchor_meta["inclusion_probability"]
    )
    settled["design_weight"] = settled["anchor_episode_id"].map(anchor_meta["design_weight"])
    settled["pool_size"] = settled["anchor_episode_id"].map(anchor_meta["pool_size"])
    settled["split"] = settled["buyer_block_key"].map(split_for_block)

    # Right-censoring: an anchor whose contract could still be running at the
    # cutoff is not a confirmed negative.
    def censored(episode_id: str) -> bool:
        if episode_id not in index.index:
            return False
        row = index.loc[episode_id]
        end = row["expected_end_date"]
        if end is None or pd.isna(end):
            return False
        return pd.Timestamp(end) + pd.Timedelta(days=CENSORING_MARGIN_DAYS) > STUDY_CUTOFF

    anchor_level = (
        settled.groupby("anchor_episode_id")
        .agg(
            anchor_verdict=("anchor_verdict", "first"),
            exposure_mode=("exposure_mode", "first"),
            split=("split", "first"),
            frame=("frame", "first"),
            stratum_id=("stratum_id", "first"),
            inclusion_probability=("inclusion_probability", "first"),
            design_weight=("design_weight", "first"),
            pool_size=("pool_size", "first"),
            labels_recorded=("label", "size"),
        )
        .reset_index()
    )
    anchor_level["negative_is_censored"] = anchor_level["anchor_episode_id"].map(censored)
    anchor_level["negative_verification"] = anchor_level["anchor_verdict"].map(
        lambda verdict: "EXHAUSTIVE_POOL" if verdict == NO_RENEWAL_POOL_EXHAUSTIVE else "AMONG_SHOWN"
    )
    for name, members in EVENT_SETS.items():
        positives = (
            settled.loc[settled["label"].isin(members)]
            .groupby("anchor_episode_id")["candidate_episode_id"].apply(list)
        )
        anchor_level[f"successors_{name}_json"] = anchor_level["anchor_episode_id"].map(
            lambda episode: json.dumps(positives.get(episode, []), ensure_ascii=False)
        )
        anchor_level[f"has_successor_{name}"] = anchor_level["anchor_episode_id"].map(
            lambda episode: len(positives.get(episode, [])) > 0
        )
    anchor_level["benchmark_schema"] = BENCHMARK_SCHEMA
    anchor_level["schema_version"] = SCHEMA_VERSION

    outputs: dict[str, Any] = {}
    sealed_dir.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "validation", "sealed_test"):
        subset = anchor_level.loc[anchor_level["split"] == split]
        pairs = settled.loc[settled["split"] == split]
        if split == "sealed_test":
            path = sealed_dir / "benchmark_test.parquet"
            pairs_path = sealed_dir / "benchmark_test_pairs.parquet"
        else:
            path = output_dir / f"benchmark_{split}.parquet"
            pairs_path = output_dir / f"benchmark_{split}_pairs.parquet"
        subset.to_parquet(path, index=False, compression="zstd")
        pairs.to_parquet(pairs_path, index=False, compression="zstd")
        outputs[split] = {
            "anchors": int(len(subset)),
            "labelled_pairs": int(len(pairs)),
            "positives_primary": int(subset["has_successor_primary"].sum()),
            "file": str(path),
        }
        logging.info("%s: %s anchors, %s pairs", split, len(subset), len(pairs))

    sealed_path = sealed_dir / "benchmark_test.parquet"
    sealed_manifest = write_manifest(
        sealed_path, project_root,
        {
            "benchmark_schema": BENCHMARK_SCHEMA,
            "split_salt": SPLIT_SALT,
            "anchors": outputs["sealed_test"]["anchors"],
            "note": (
                "Aggregate counts only. Per-anchor labels are deliberately absent so "
                "the manifest cannot be used to read the split without opening it."
            ),
        },
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_schema": BENCHMARK_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "split_rule": (
            "Assigned on buyer_block_key, not on the anchor: a buyer's notices share "
            "boilerplate and suppliers, so an anchor-level split would let a method "
            "tuned on dev meet the same buyer's vocabulary in test."
        ),
        "split_salt": SPLIT_SALT,
        "split_shares": dict(SPLIT_SHARES),
        "split_roles": {
            "dev": "threshold selection and method development",
            "validation": "method selection and adoption tests",
            "sealed_test": "opened once, through an audited path, for the final report",
        },
        "outputs": outputs,
        "sealed_manifest": str(sealed_manifest),
        "event_sets": {name: sorted(members) for name, members in EVENT_SETS.items()},
        "anchor_totals": {
            "anchors": int(len(anchor_level)),
            "labelled_pairs": int(len(settled)),
            "by_frame": {
                str(k): int(v) for k, v in anchor_level["frame"].value_counts().items()
            },
            "by_verdict": {
                str(k): int(v) for k, v in anchor_level["anchor_verdict"].value_counts().items()
            },
            "verified_negatives": int(
                (anchor_level["negative_verification"] == "EXHAUSTIVE_POOL").sum()
            ),
            "censored_anchors_excluded_from_fpr": int(
                anchor_level["negative_is_censored"].sum()
            ),
        },
        "positives_by_event_set": {
            name: int(anchor_level[f"has_successor_{name}"].sum()) for name in EVENT_SETS
        },
        "validation_passed": bool(len(anchor_level) > 0),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


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
        raise RuntimeError("Benchmark assembly validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
