"""Load the canonical national benchmark behind one truth-column contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from boamp_pipeline.annotation_schema import EVENT_SETS
from boamp_pipeline.sealed_split import open_sealed

DEFAULT_BENCHMARK_DIR = Path("data/processed/boamp/benchmark")

TRUTH_COLUMNS = (
    "anchor_episode_id",
    "true_successors",
    "has_successor",
    "truth_usable",
    "benchmark_split",
)


def load_truth(
    benchmark_dir: Path = DEFAULT_BENCHMARK_DIR,
    split: str = "dev",
    event_set: str = "primary",
    *,
    project_root: Path | None = None,
    allow_sealed: bool = False,
    seal_reason: str = "",
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Load one benchmark split and map labels to the evaluator contract."""
    if event_set not in EVENT_SETS:
        raise ValueError(f"event_set must be one of {sorted(EVENT_SETS)}")

    access_record: dict[str, Any] | None = None
    if split == "sealed_test":
        sealed_path = benchmark_dir / "sealed" / "benchmark_test.parquet"
        frame, access_record = open_sealed(
            sealed_path,
            project_root or Path.cwd(),
            reason=seal_reason,
            allow=allow_sealed,
        )
    else:
        path = benchmark_dir / f"benchmark_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)

    truth = frame.copy()
    truth["true_successors"] = truth[f"successors_{event_set}_json"].map(json.loads)
    truth["has_successor"] = truth[f"has_successor_{event_set}"].astype(bool)
    truth["truth_usable"] = truth["anchor_verdict"].ne("ANCHOR_UNUSABLE")
    truth["benchmark_split"] = split
    truth["event_set"] = event_set
    return truth, access_record
