"""Load the canonical Grand Ouest regional benchmark behind one truth contract.

This is the active reference. It replaced the France-level benchmark, whose
labels were emitted by deterministic rules built from the same text, CPV and
date evidence the linkage methods consume: that benchmark could only measure
how closely a method agreed with a hand-written rule.

The regional reference is a stratified review of 120 Grand Ouest anchors
carried out against real BOAMP notices before these methods existed. It is a
*reference sample*, not ground truth, and the constraints in
``data/processed/boamp/regional_benchmark/DATASHEET.md`` bind anything computed
from it.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_BENCHMARK_DIR = Path("data/processed/boamp/regional_benchmark")

#: The reference records one successor relationship per anchor and never
#: separates renewal from next-phase, so the strict/broad event sets the
#: national schema carried have no regional counterpart.
SUPPORTED_EVENT_SETS = ("primary",)

SPLITS = ("dev", "validation")

TRUTH_COLUMNS = (
    "anchor_episode_id",
    "true_successors",
    "has_successor",
    "truth_usable",
    "benchmark_split",
)


def load_truth(
    benchmark_dir: Path = DEFAULT_BENCHMARK_DIR,
    split: str = "validation",
    event_set: str = "primary",
) -> pd.DataFrame:
    """Load one regional split and map it onto the evaluator's contract."""
    if event_set not in SUPPORTED_EVENT_SETS:
        raise ValueError(
            f"the regional reference supports only {SUPPORTED_EVENT_SETS}; "
            f"got {event_set!r}. It does not distinguish renewal from next-phase."
        )
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}; got {split!r}")

    path = benchmark_dir / f"benchmark_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run scripts/build_regional_benchmark.py first"
        )

    truth = pd.read_parquet(path)
    truth["true_successors"] = truth[f"successors_{event_set}_json"].map(json.loads)
    truth["has_successor"] = truth[f"has_successor_{event_set}"].astype(bool)
    truth["truth_usable"] = truth["anchor_verdict"].ne("ANCHOR_UNUSABLE")
    truth["benchmark_split"] = split
    truth["event_set"] = event_set
    return truth


def load_manifest(benchmark_dir: Path = DEFAULT_BENCHMARK_DIR) -> dict[str, Any]:
    path = benchmark_dir / "regional_benchmark_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def wilson_interval(successes: int, trials: int, z: float = 1.959963985) -> list[float] | None:
    """Wilson score interval.

    Used rather than a bare proportion because on 72 held-out anchors, of which
    18 are positive, a point estimate on its own invites reading two digits of
    precision that are not there.
    """
    if trials <= 0:
        return None
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = z * sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]
