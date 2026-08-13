from __future__ import annotations

import pandas as pd
import pytest

from scripts.evaluate_review_audit import exact_binomial, validate_review


def test_exact_binomial_contains_point_estimate() -> None:
    result = exact_binomial(14, 19)
    assert result["estimate"] == pytest.approx(14 / 19)
    assert result["ci_95"][0] < result["estimate"] < result["ci_95"][1]


def test_completed_review_is_schema_valid() -> None:
    blank = pd.read_csv(
        "data/review/independent_link_review_sample.csv", dtype=str, keep_default_na=False
    )
    reviewed = pd.read_csv(
        "data/review/independent_link_review_sample_reviewed.csv",
        dtype=str,
        keep_default_na=False,
    )
    result = validate_review(blank, reviewed)
    assert result["labels_complete_and_valid"] is True
    assert result["evidence_columns_unchanged"] is True


def test_review_validation_rejects_changed_evidence() -> None:
    blank = pd.read_csv(
        "data/review/independent_link_review_sample.csv", dtype=str, keep_default_na=False
    )
    reviewed = pd.read_csv(
        "data/review/independent_link_review_sample_reviewed.csv",
        dtype=str,
        keep_default_na=False,
    )
    reviewed.loc[0, "anchor_episode_text"] = "changed"
    with pytest.raises(ValueError, match="evidence columns changed"):
        validate_review(blank, reviewed)
