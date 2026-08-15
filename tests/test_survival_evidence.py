"""Checks for the materialised survival evidence contract."""

import json
from pathlib import Path

import pandas as pd

from scripts.build_survival_evidence import standardized_mean_difference


PROCESSED = Path("data/processed/boamp")


def test_standardized_mean_difference_is_zero_for_equal_groups() -> None:
    left = pd.Series([1.0, 2.0, 3.0])
    right = pd.Series([1.0, 2.0, 3.0])
    assert standardized_mean_difference(left, right) == 0.0


def test_materialized_survival_summary_matches_frozen_cohort() -> None:
    summary = json.loads((PROCESSED / "survival_analysis_summary.json").read_text())
    assert summary["cohort"]["contracts"] == 3800
    assert summary["cohort"]["events"] == 544
    assert summary["cohort"]["censored"] == 3256
    assert summary["cohort"]["event_rate"] == 544 / 3800
    assert summary["cohort"]["censoring_rate"] == 3256 / 3800
    assert summary["km"]["median_status"] == "not_reached"
    assert summary["cox"]["temporal_validation"]["test_c_index"] < 0.55
    assert summary["validation_passed"]


def test_temporal_validation_reports_the_guideline_window_as_primary() -> None:
    cox = json.loads((PROCESSED / "survival_analysis_summary.json").read_text())["cox"]
    primary = cox["temporal_validation"]
    extended = cox["temporal_validation_including_latest_cohort"]

    assert primary["train_years"] == "2015-2021"
    assert primary["test_years"] == "2022-2024"
    assert extended["test_years"] == "2022-2025"
    # One fit, two scoring windows: a differing train C-index would mean the
    # model was refitted per window, which the freeze forbids.
    assert primary["train_c_index"] == extended["train_c_index"]
    assert primary["test_contracts"] < extended["test_contracts"]
    for split in (primary, extended):
        assert split["test_events"] > 0
        assert 0.0 <= split["test_c_index"] <= 1.0


def test_borderline_check_removes_a_band_around_the_frozen_threshold() -> None:
    summary = json.loads((PROCESSED / "survival_analysis_summary.json").read_text())
    border = summary["borderline_link_sensitivity"]
    table = pd.read_csv(PROCESSED / "survival_borderline_link_sensitivity.csv")

    assert border["band"] == {"low": 0.65, "high": 0.75, "variable": "M_B text_component (0-1)"}
    assert border["contracts_removed"] == border["events_removed"] + border["censored_removed"]
    # Both event-positive and event-negative episodes must fall in the band;
    # excluding only accepted links would test the wrong thing.
    assert border["events_removed"] > 0 and border["censored_removed"] > 0
    assert set(table["analysis"]) == {"main", "excluding_borderline_links"}
    kept = border["excluding_borderline_links"]
    assert kept["contracts"] == summary["cohort"]["contracts"] - border["contracts_removed"]
    # The comparative claims the project makes are the ones this check defends.
    assert border["assessment"]["comparative_claims"] == "NOT_DRIVEN_BY_BORDERLINE_LINKS"
    assert (kept["cox_hr_cpv_35"] > 1) == (border["main"]["cox_hr_cpv_35"] > 1)
    assert (kept["cox_hr_framework"] > 1) == (border["main"]["cox_hr_framework"] > 1)


def test_parametric_model_is_not_the_operational_probability_source() -> None:
    parametric = json.loads(
        (PROCESSED / "survival_analysis_summary.json").read_text()
    )["parametric"]
    assert "Kaplan-Meier" in parametric["operational_probability_source"]


def test_selection_diagnostic_has_no_missing_smd() -> None:
    diagnostic = pd.read_csv(PROCESSED / "survival_selection_diagnostic.csv")
    assert diagnostic["standardized_mean_difference"].notna().all()
    assert set(diagnostic["variable"]) >= {
        "text_length_chars", "framework_flag", "has_validated_siren"
    }
