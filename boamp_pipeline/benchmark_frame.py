"""Probability sampling design for the v3 benchmark frame.

The v1 benchmark recorded ``inclusion_probability`` and ``sampling_weight`` for
all 120 anchors and then used them in no metric: every reported precision and
recall is an unweighted sample quantity presented as if it described the
population. It also drew from a frame of 1,616 episodes whose definition and
sampling code are absent from the repository, so the design cannot be checked
or repeated.

This module is the design, in code. It defines strata, allocates the sample
across them, draws it reproducibly, and returns the inclusion probability that
the estimator layer later divides by. The arithmetic identity that makes the
weights real -- that the design weights sum back to the frame size -- is
asserted rather than assumed.

Two sampling choices deserve their reasons stated:

*Square-root allocation.* Proportional allocation would give the smallest
strata almost no sample, and per-stratum estimates are wanted, not just a
national total. ``n_h`` proportional to ``sqrt(N_h)`` is the standard
compromise between the two.

*Implicit stratification.* Buyer size, duration availability and award period
all matter, but crossing them with the primary strata would give 6x4x5x2x3 =
720 cells over a frame of ~20k, most of them empty. Instead the sample is
sorted on those variables inside each stratum and drawn systematically, which
balances them without creating cells -- and keeps ``pi = n_h / N_h`` exact.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

#: Frame identity, recorded on every sampled row.
FRAME_VERSION = "boamp_benchmark_frame_v3.0"

#: Fixed so the draw is reproducible. Any change to this value produces a
#: different sample and must be treated as a new frame version.
FRAME_SEED = 20260813

#: Anchors need enough follow-up to be able to show a successor at all. The
#: measured national gap distribution between a contract and its re-procurement
#: peaks at three to four years, so an anchor awarded in 2024 cannot yet
#: contribute a typical positive and would enter the frame as a near-certain
#: negative. 2021 leaves four full years to the 2025-12-31 corpus end.
MIN_AWARD_YEAR = 2015
MAX_AWARD_YEAR = 2021

#: Minimum sample per stratum. Below this a stratum cannot support its own
#: estimate and its variance contribution is unusable.
MIN_STRATUM_SAMPLE = 8

#: Ceiling per stratum, so one large cell cannot absorb the budget.
MAX_STRATUM_SAMPLE = 60

AWARD_PERIODS: tuple[tuple[int, int, str], ...] = (
    (2015, 2017, "2015-2017"),
    (2018, 2019, "2018-2019"),
    (2020, 2021, "2020-2021"),
)


def award_period(year: Any) -> str:
    """Coarse award era, controlling for the eForms schema break downstream."""
    try:
        value = int(year)
    except (TypeError, ValueError):
        return "unknown"
    for start, end, label in AWARD_PERIODS:
        if start <= value <= end:
            return label
    return "out_of_range"


def stable_hash(*parts: Any, seed: int = FRAME_SEED) -> int:
    """Deterministic 64-bit hash, stable across processes and Python versions.

    ``hash()`` is salted per interpreter run, so it cannot be used anywhere a
    sample must be reproducible.
    """
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def stratum_id(dept_group: Any, digital_segment: Any) -> str:
    """Primary stratum: geography group crossed with technological segment."""
    segment = str(digital_segment or "NON_DIGITAL")
    return f"{dept_group}|{segment}"


def eligible_frame(index: pd.DataFrame) -> pd.DataFrame:
    """Rows of the national index that may be sampled as anchors.

    An anchor must be a contract (it has an award notice, which is what proves
    a contract exists and supplies time zero), must be blockable (otherwise it
    has no candidate pool and a negative would be vacuous), and must have
    enough follow-up.
    """
    required = {"has_award_notice", "award_year", "buyer_block_key", "digital_flag"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"index is missing required columns: {sorted(missing)}")

    award_year = pd.to_numeric(index["award_year"], errors="coerce")
    keep = (
        index["has_award_notice"].fillna(False).astype(bool)
        & index["buyer_block_key"].fillna("").astype(str).str.len().gt(0)
        & award_year.between(MIN_AWARD_YEAR, MAX_AWARD_YEAR)
    )
    return index.loc[keep].copy()


def add_design_variables(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the stratum and the implicit-stratification sort keys."""
    frame = frame.copy()
    frame["stratum_id"] = [
        stratum_id(group, segment)
        for group, segment in zip(frame["dept_group"], frame["digital_segment"])
    ]
    frame["award_period"] = frame["award_year"].map(award_period)
    frame["duration_available"] = frame["duration_months_reliable"].notna()
    frame["frame_version"] = FRAME_VERSION
    return frame


def allocate(
    populations: Mapping[str, int],
    total: int,
    minimum: int = MIN_STRATUM_SAMPLE,
    maximum: int = MAX_STRATUM_SAMPLE,
) -> dict[str, int]:
    """Split a sample budget across strata, proportional to ``sqrt(N_h)``.

    Each stratum is clamped to ``[minimum, min(maximum, N_h)]`` and the
    remainder is distributed by largest fractional part, so the allocation sums
    to ``total`` exactly whenever that is arithmetically possible.

    >>> allocate({"a": 100, "b": 400}, total=30, minimum=1, maximum=100)
    {'a': 10, 'b': 20}
    """
    strata = {name: int(size) for name, size in populations.items() if int(size) > 0}
    if not strata:
        return {}
    feasible_min = sum(min(minimum, size) for size in strata.values())
    feasible_max = sum(min(maximum, size) for size in strata.values())
    if total < feasible_min:
        raise ValueError(
            f"budget {total} is below the per-stratum floor across {len(strata)} strata "
            f"(needs at least {feasible_min}); lower the floor or merge strata"
        )
    total = min(total, feasible_max)

    roots = {name: math.sqrt(size) for name, size in strata.items()}
    root_total = sum(roots.values())
    raw = {name: total * root / root_total for name, root in roots.items()}

    allocation = {
        name: int(min(max(math.floor(raw[name]), min(minimum, strata[name])), maximum, strata[name]))
        for name in strata
    }
    # Distribute what the flooring left over, preferring the largest fractional
    # parts, and never exceeding a stratum's own population.
    while sum(allocation.values()) < total:
        candidates = [
            name for name in strata
            if allocation[name] < min(maximum, strata[name])
        ]
        if not candidates:
            break
        chosen = max(candidates, key=lambda name: (raw[name] - allocation[name], roots[name]))
        allocation[chosen] += 1
    while sum(allocation.values()) > total:
        candidates = [
            name for name in strata
            if allocation[name] > min(minimum, strata[name])
        ]
        if not candidates:
            break
        chosen = min(candidates, key=lambda name: (raw[name] - allocation[name], roots[name]))
        allocation[chosen] -= 1
    return allocation


def systematic_sample(
    stratum: pd.DataFrame,
    size: int,
    sort_columns: Sequence[str],
    seed_parts: Iterable[Any] = (),
) -> pd.DataFrame:
    """Draw ``size`` rows systematically after an implicit-stratification sort.

    Sorting on the secondary design variables and then taking every ``N/n``-th
    row spreads the sample across them, while a random start keeps every unit's
    inclusion probability exactly ``n/N``. Ties inside the sort are broken by a
    stable hash so the draw is reproducible but not correlated with any
    substantive field.
    """
    population = len(stratum)
    if size <= 0 or population == 0:
        return stratum.iloc[0:0].copy()
    if size >= population:
        return stratum.copy()

    ordered = stratum.copy()
    ordered["_tiebreak"] = [
        stable_hash(episode_id, *seed_parts) for episode_id in ordered["episode_id"]
    ]
    ordered = ordered.sort_values(
        list(sort_columns) + ["_tiebreak"], kind="mergesort"
    ).reset_index(drop=True)

    interval = population / size
    generator = np.random.default_rng(stable_hash("start", *seed_parts) % (2**32))
    start = float(generator.random()) * interval
    positions = np.floor(start + interval * np.arange(size)).astype(int)
    positions = np.clip(positions, 0, population - 1)
    # A rounding collision would silently shrink the sample; walk duplicates
    # forward so exactly `size` distinct units are drawn.
    seen: set[int] = set()
    resolved: list[int] = []
    for position in positions:
        candidate = int(position)
        while candidate in seen:
            candidate = (candidate + 1) % population
        seen.add(candidate)
        resolved.append(candidate)

    drawn = ordered.iloc[sorted(resolved)].drop(columns="_tiebreak").copy()
    return drawn


def draw_probability_sample(
    frame: pd.DataFrame,
    total: int,
    sort_columns: Sequence[str] = (
        "buyer_size_class", "duration_available", "award_period",
    ),
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Draw the stratified probability sample and attach inclusion probabilities."""
    populations = frame["stratum_id"].value_counts().to_dict()
    allocation = allocate(populations, total)

    drawn: list[pd.DataFrame] = []
    design: dict[str, dict[str, int]] = {}
    for name, group in frame.groupby("stratum_id", sort=True):
        size = int(allocation.get(name, 0))
        design[name] = {"population": len(group), "sample": size}
        if size == 0:
            continue
        sample = systematic_sample(group, size, sort_columns, seed_parts=(name,))
        sample["stratum_population_n"] = len(group)
        sample["stratum_sample_n"] = size
        sample["inclusion_probability"] = size / len(group)
        sample["design_weight"] = len(group) / size
        drawn.append(sample)

    if not drawn:
        return frame.iloc[0:0].copy(), design
    return pd.concat(drawn, ignore_index=True), design


def weights_reconstruct_population(sample: pd.DataFrame, frame_size: int,
                                   tolerance: float = 0.01) -> bool:
    """Whether the design weights sum back to the frame size.

    This is the arithmetic check that the sampling design is coherent. It is
    also the check that v1 never had an opportunity to fail, because its
    weights were never used for anything.
    """
    if sample.empty or frame_size <= 0:
        return False
    estimated = float(sample["design_weight"].sum())
    return abs(estimated - frame_size) / frame_size < tolerance


def assert_no_frame_mixing(sample: pd.DataFrame) -> None:
    """Refuse to estimate from probability and purposive rows together.

    Enrichment anchors are recruited because they declared a renewal, so they
    over-represent buyers who write detailed notices -- exactly the buyers a
    text-similarity method handles best. They are essential for error analysis
    and must never enter a national estimate.
    """
    if "frame" not in sample.columns:
        raise ValueError("sample has no 'frame' column; cannot verify frame separation")
    frames = set(sample["frame"].dropna().unique())
    probability = {f for f in frames if f == "PROBABILITY"}
    enrichment = {f for f in frames if str(f).startswith("ENRICHMENT")}
    if probability and enrichment:
        raise ValueError(
            "refusing to mix frames in one estimate: "
            f"{sorted(probability)} with {sorted(enrichment)}. "
            "Enrichment anchors are purposively recruited and carry no inclusion "
            "probability; estimate on PROBABILITY rows only."
        )
