from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boamp_pipeline.evidence import (
    build_quarterly_panel,
    pelt_break_indices,
    recent_trend_signal,
    stable_breaks,
    stationarity_diagnostics,
)


def test_quarterly_panel_fills_missing_quarters_and_excludes_partial_start() -> None:
    cohort = pd.DataFrame(
        {
            "episode_id": ["a", "b", "c"],
            "award_date": pd.to_datetime(["2015-03-20", "2015-04-10", "2015-10-01"]),
            "digital_segment": ["CPV-48", "CPV-48", "CPV-48"],
            "duration_months_reliable": [12.0, np.nan, 24.0],
        }
    )
    panel = build_quarterly_panel(cohort, start_quarter="2015Q2", end_quarter="2015Q4")
    segment = panel.loc[panel["segment"].eq("CPV-48")]
    assert segment["quarter"].tolist() == ["2015Q2", "2015Q3", "2015Q4"]
    assert segment["episode_count"].tolist() == [1, 0, 1]
    assert segment["reliable_duration_count"].tolist() == [0, 0, 1]


def test_quarterly_panel_rejects_duplicate_episode_grain() -> None:
    cohort = pd.DataFrame(
        {
            "episode_id": ["a", "a"],
            "award_date": pd.to_datetime(["2016-01-01", "2016-02-01"]),
            "digital_segment": ["CPV-72", "CPV-72"],
            "duration_months_reliable": [12.0, 12.0],
        }
    )
    with pytest.raises(ValueError, match="episode_id must be unique"):
        build_quarterly_panel(cohort, start_quarter="2016Q1", end_quarter="2016Q2")


def test_pelt_detects_large_level_shift() -> None:
    values = [10.0] * 12 + [30.0] * 12
    breaks = pelt_break_indices(values, penalty_multiplier=1.0, min_size=4)
    assert breaks == [12]


def test_pelt_constant_series_has_no_break() -> None:
    assert pelt_break_indices([5.0] * 20) == []


def test_stable_breaks_requires_penalty_persistence() -> None:
    values = [10.0] * 12 + [30.0] * 12
    result = stable_breaks(values)
    assert result["central_break_indices"] == [12]
    assert result["stable_break_indices"] == [12]


def test_recent_trend_signal_distinguishes_direction() -> None:
    increasing = recent_trend_signal(range(12), quarters=12)
    decreasing = recent_trend_signal(reversed(range(12)), quarters=12)
    flat = recent_trend_signal([4] * 12, quarters=12)
    assert increasing["state"] == "increasing"
    assert decreasing["state"] == "decreasing"
    assert flat["state"] == "stable_or_uncertain"


def test_stationarity_diagnostics_handles_constant_series() -> None:
    result = stationarity_diagnostics([1.0] * 20)
    assert result["available"] is False
