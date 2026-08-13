"""Load benchmark truth, from either version, behind one column contract.

``scripts/evaluate_linkage.py`` was written against the v1 remap CSVs. Rather
than rewrite its metric functions, ``load_truth_v3`` returns the same columns
v1 returned -- ``anchor_v2_episode_id``, ``true_successors``, ``has_successor``,
``truth_usable``, ``benchmark_split`` -- so ``candidate_recall`` and
``evaluate`` keep working untouched, and the v1 numbers stay reproducible for
comparison.

Everything v3 adds arrives as extra columns: the inclusion probability, the
stratum, whether a negative was verified against a whole pool or only against
what was shown, and whether the anchor is right-censored. The metric layer uses
them; the legacy functions ignore them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from boamp_pipeline.annotation_schema import EVENT_SETS
from boamp_pipeline.sealed_split import open_sealed

DEFAULT_V3_DIR = Path("data/processed/boamp_v2/benchmark_v3")

#: The column contract both loaders satisfy.
TRUTH_COLUMNS = (
    "anchor_v2_episode_id",
    "true_successors",
    "has_successor",
    "truth_usable",
    "benchmark_split",
)


def load_truth_v1(output_dir: Path) -> pd.DataFrame:
    """The v1 benchmark, unchanged.

    Moved here verbatim from ``evaluate_linkage.load_truth`` so that adding v3
    cannot alter the v1 result. A test asserts the v1 summary still reproduces
    field for field.
    """
    remap_dir = output_dir / "benchmark_remap"
    evaluation = pd.read_csv(remap_dir / "evaluation_subset_v2_remap.csv")
    evaluation = evaluation.loc[evaluation["anchor_v2_episode_id"].notna()].copy()
    evaluation["true_successors"] = evaluation["successor_v2_episode_ids_json"].map(json.loads)
    evaluation["has_successor"] = evaluation["final_outcome"].isin(
        ["OBSERVED_SUCCESSOR", "MULTIPLE_SUCCESSORS"]
    )
    evaluation["truth_usable"] = (
        ~evaluation["has_successor"] | evaluation["true_successors"].str.len().gt(0)
    )
    return evaluation


def load_truth_v3(
    benchmark_dir: Path,
    split: str = "dev",
    event_set: str = "primary",
    *,
    project_root: Path | None = None,
    allow_sealed: bool = False,
    seal_reason: str = "",
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """The v3 benchmark, for one split and one event definition.

    Returns the truth frame and, when the sealed split was opened, the access
    record that must be embedded in whatever summary quotes the numbers.
    """
    if event_set not in EVENT_SETS:
        raise ValueError(f"event_set must be one of {sorted(EVENT_SETS)}")

    access_record: dict[str, Any] | None = None
    if split == "sealed_test":
        sealed_path = benchmark_dir / "sealed" / "benchmark_v3_test.parquet"
        frame, access_record = open_sealed(
            sealed_path,
            project_root or benchmark_dir.parents[2],
            reason=seal_reason,
            allow=allow_sealed,
        )
    else:
        path = benchmark_dir / f"benchmark_v3_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)

    truth = frame.rename(columns={"anchor_episode_id": "anchor_v2_episode_id"}).copy()
    truth["true_successors"] = truth[f"successors_{event_set}_json"].map(json.loads)
    truth["has_successor"] = truth[f"has_successor_{event_set}"].astype(bool)
    # An anchor is scorable unless the annotator could not judge it at all.
    truth["truth_usable"] = truth["anchor_verdict"].ne("ANCHOR_UNUSABLE")
    truth["benchmark_split"] = split
    truth["event_set"] = event_set
    return truth, access_record


def load_truth(
    output_dir: Path,
    benchmark: str = "v1",
    split: str = "dev",
    event_set: str = "primary",
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Dispatch to a benchmark version. Defaults to v1 so nothing changes."""
    if benchmark == "v1":
        return load_truth_v1(output_dir), None
    if benchmark == "v3":
        return load_truth_v3(output_dir / "benchmark_v3", split, event_set, **kwargs)
    raise ValueError(f"unknown benchmark {benchmark!r}")
