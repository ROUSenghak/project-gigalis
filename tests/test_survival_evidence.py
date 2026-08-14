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


def test_selection_diagnostic_has_no_missing_smd() -> None:
    diagnostic = pd.read_csv(PROCESSED / "survival_selection_diagnostic.csv")
    assert diagnostic["standardized_mean_difference"].notna().all()
    assert set(diagnostic["variable"]) >= {
        "text_length_chars", "framework_flag", "has_validated_siren"
    }
