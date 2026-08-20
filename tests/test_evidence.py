from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boamp_pipeline.evidence import (
    build_quarterly_panel,
    pelt_break_indices,
    recent_trend_signal,
    regime_diagnostics,
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


def test_regime_diagnostics_requires_minimum_length() -> None:
    result = regime_diagnostics(range(10))
    assert result["available"] is False
    assert "16" in result["reason"]


def test_regime_diagnostics_handles_constant_change_series() -> None:
    result = regime_diagnostics(range(20))
    assert result["available"] is False


def test_regime_diagnostics_returns_valid_regime_labels_and_probabilities() -> None:
    values = [20.0] * 10 + [20.0 + 10.0 * step for step in range(1, 11)]
    result = regime_diagnostics(values, min_observations=16)
    assert result["available"] is True
    means = result["mean_change_by_regime"]
    assert set(means) == {"decline", "plateau", "growth"}
    assert means["decline"] <= means["plateau"] <= means["growth"]
    assert result["current_regime"] in {"decline", "plateau", "growth"}
    assert 0.0 <= result["current_regime_probability"] <= 1.0
    matrix = result["transition_matrix_rows_from_columns_to"]
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    for row in matrix:
        assert pytest.approx(sum(row), abs=1e-6) == 1.0


def test_adjust_p_values_matches_the_textbook_procedures() -> None:
    """Both step-wise procedures, on a case worked by hand."""
    from boamp_pipeline.evidence import adjust_p_values

    raw = [0.01, 0.02, 0.03, 0.04, 0.05]
    holm = adjust_p_values(raw, "holm")
    # Holm multiplies the k-th smallest by (n - k + 1) and enforces monotonicity.
    assert holm[0] == pytest.approx(0.05)
    assert holm[1] == pytest.approx(0.08)
    assert list(holm) == sorted(holm)

    bh = adjust_p_values(raw, "bh")
    # BH scales the k-th smallest by n/k, then takes a running minimum downward
    # from the largest. Here every (n/k) * p_k equals 0.05, so all five do.
    assert list(bh) == pytest.approx([0.05] * 5)
    assert all(b <= h + 1e-9 for b, h in zip(bh, holm, strict=True))
    # Adjustment never makes a p-value more significant. An earlier BH
    # implementation reversed its rank multipliers, which returned the smallest
    # p-value unadjusted; this assertion is what catches that class of error.
    assert all(a >= r - 1e-9 for a, r in zip(bh, raw, strict=True))


def test_adjust_p_values_matches_statsmodels_on_random_families() -> None:
    """Cross-check both procedures against an independent implementation."""
    from statsmodels.stats.multitest import multipletests

    from boamp_pipeline.evidence import adjust_p_values

    generator = np.random.default_rng(20260820)
    for size in (1, 2, 3, 5, 7, 11, 20):
        raw = list(np.round(generator.random(size), 6))
        for method, reference_name in (("holm", "holm"), ("bh", "fdr_bh")):
            mine = adjust_p_values(raw, method)
            reference = np.round(multipletests(raw, method=reference_name)[1], 4)
            assert np.allclose(mine, reference, atol=1e-4), (method, raw)


def test_the_smallest_raw_p_value_is_always_adjusted_upward() -> None:
    """The most significant test is the one a broken procedure leaves alone."""
    from boamp_pipeline.evidence import adjust_p_values

    raw = [0.0317, 0.2854, 0.9214, 0.9227, 0.9888]
    for method in ("holm", "bh"):
        adjusted = adjust_p_values(raw, method)
        assert adjusted[0] > raw[0], method
        assert adjusted[0] == pytest.approx(0.1585, abs=1e-4), method


def test_adjust_p_values_rejects_an_unknown_method() -> None:
    from boamp_pipeline.evidence import adjust_p_values

    with pytest.raises(ValueError):
        adjust_p_values([0.1, 0.2], "bonferroni")


def test_cpv_trend_slopes_carry_the_same_multiplicity_correction_as_technology() -> None:
    """One standard for both families of simultaneous slope tests.

    The technology trend section adjusts its five class slopes; the CPV section
    used to report five segment slopes uncorrected, and the executive summary
    then promoted the smallest raw p-value as a finding.
    """
    from pathlib import Path

    import pandas as pd

    matrix_path = Path("data/processed/boamp/trend_signal_matrix.csv")
    if not matrix_path.exists():
        pytest.skip("trend evidence not materialised")
    matrix = pd.read_csv(matrix_path)

    assert {"p_holm", "p_bh", "n_tests", "multiplicity_status"} <= set(matrix.columns)
    assert (matrix["n_tests"] == len(matrix)).all()
    assert (matrix["p_holm"] >= matrix["p_value"] - 1e-9).all()
    assert (matrix["p_bh"] >= matrix["p_value"] - 1e-9).all()

    from boamp_pipeline.evidence import adjust_p_values

    assert list(matrix["p_holm"]) == list(adjust_p_values(matrix["p_value"], "holm"))
    assert list(matrix["p_bh"]) == list(adjust_p_values(matrix["p_value"], "bh"))

    # A directional label that does not survive adjustment must say so.
    nominal_only = matrix.loc[
        matrix["state"].ne("stable_or_uncertain") & ~matrix["survives_multiplicity"]
    ]
    for row in nominal_only.itertuples():
        assert "does not survive" in row.multiplicity_status, row.segment


def test_the_executive_summary_does_not_promote_an_unadjusted_trend() -> None:
    """The 'What Works' section may not claim a decline the correction removes."""
    import json
    from pathlib import Path

    summary_path = Path("EXECUTIVE_SUMMARY.md")
    trend_path = Path("data/processed/boamp/trend_analysis_summary.json")
    if not (summary_path.exists() and trend_path.exists()):
        pytest.skip("reader artifacts not materialised")

    multiplicity = json.loads(trend_path.read_text(encoding="utf-8"))["multiplicity"]
    works = summary_path.read_text(encoding="utf-8").split("## What Works")[1]
    works = works.split("## What Remains Uncertain")[0]

    if not multiplicity["segments_surviving_multiplicity"]:
        assert "statistically distinguishable recent decline" not in works
    for segment in multiplicity["segments_with_nominal_signal_only"]:
        if segment in works:
            assert "does not survive" in works or "exploratory" in works
