"""Checks for the materialised survival evidence contract."""

import json
from pathlib import Path

import pandas as pd
import pytest

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


def test_template_risk_check_recensors_rather_than_dropping_episodes() -> None:
    """The borderline check drops rows; this one must not.

    Its counterfactual is "the link was spurious", under which the anchor still
    contributes its full follow-up as censored exposure. Dropping the rows would
    silently discard that exposure and answer a different question.
    """
    summary = json.loads((PROCESSED / "survival_analysis_summary.json").read_text())
    template = summary["template_risk_sensitivity"]
    table = pd.read_csv(PROCESSED / "survival_template_risk_sensitivity.csv")
    recensored = template["recensoring_template_risk_links"]

    assert set(table["analysis"]) == {"main", "recensoring_template_risk_links"}
    assert recensored["contracts"] == summary["cohort"]["contracts"]
    assert recensored["events"] == summary["cohort"]["events"] - template["flagged_links"]
    # The group is defined by the two signatures the linkage audit publishes,
    # at the threshold it publishes. Neither is re-chosen here.
    assert template["carried_by_char_threshold"] == 0.50
    assert template["flagged_links"] <= (
        template["carried_by_char_similarity"]
        + template["successor_shared_with_another_anchor"]
    )
    # The check exists to defend the comparative claims, framework above all.
    assert template["assessment"]["comparative_claims"] == "NOT_DRIVEN_BY_TEMPLATE_RISK_LINKS"
    assert (recensored["cox_hr_cpv_35"] > 1) == (template["main"]["cox_hr_cpv_35"] > 1)
    assert (recensored["cox_hr_framework"] > 1) == (template["main"]["cox_hr_framework"] > 1)


def test_template_risk_group_matches_the_candidate_generation_audit() -> None:
    """Both artifacts count links carried by the character analyser. If the two
    counts diverge, one of them is stale and the report quotes a mix."""
    template = json.loads(
        (PROCESSED / "survival_analysis_summary.json").read_text()
    )["template_risk_sensitivity"]
    audit = json.loads((PROCESSED / "candidate_generation_audit.json").read_text())

    assert (
        template["carried_by_char_threshold"]
        == audit["cpv_continuity"]["low_word_similarity_threshold"]
    )
    assert (
        template["carried_by_char_similarity"]
        == audit["cpv_continuity"]["accepted_links_carried_by_char_similarity"]
    )
    assert template["accepted_links"] == audit["cpv_continuity"]["accepted_links"]


def test_conditional_probabilities_cover_the_reported_ages_with_intervals() -> None:
    """The operational table in the report is generated from this file."""
    conditional = pd.read_csv(PROCESSED / "survival_conditional_probabilities.csv")

    assert set(conditional["contract_age_months"]) == {0, 12, 24, 36, 48}
    assert set(conditional["horizon_months"]) == {12, 24}
    assert conditional["probability"].between(0, 1).all()
    assert (conditional["ci_95_low"] <= conditional["probability"]).all()
    assert (conditional["probability"] <= conditional["ci_95_high"]).all()


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


def test_candidate_pool_size_is_published_in_the_detectability_diagnostic() -> None:
    """The largest linked-vs-censored imbalance must be visible, not omitted.

    ``M_B`` accepts the maximum text score over an anchor's candidate block, so
    block size is a detectability variable. It was absent from the published
    diagnostic while sitting above every variable that was in it.
    """
    diagnostic = pd.read_csv(PROCESSED / "survival_selection_diagnostic.csv")
    published = set(diagnostic["variable"])
    assert {"candidate_pool_size", "log_candidate_pool_size"} <= published

    log_row = diagnostic.loc[diagnostic["variable"] == "log_candidate_pool_size"].iloc[0]
    assert log_row["linked_n"] + log_row["censored_n"] == 3800
    # It is the largest absolute imbalance; if that ever stops being true the
    # report's framing of it should be revisited deliberately, not silently.
    assert log_row["absolute_smd"] == diagnostic["absolute_smd"].max()


def test_candidate_pool_size_recomputes_from_the_candidate_table() -> None:
    candidates = pd.read_parquet(
        PROCESSED / "linkage_candidates_scored.parquet", columns=["anchor_episode_id"]
    )
    survival = pd.read_parquet(
        PROCESSED / "survival_dataset.parquet", columns=["episode_id", "event"]
    )
    pool = candidates.groupby("anchor_episode_id").size()
    sizes = survival["episode_id"].map(pool).fillna(0)

    diagnostic = pd.read_csv(PROCESSED / "survival_selection_diagnostic.csv").set_index(
        "variable"
    )
    linked = sizes[survival["event"].eq(1).to_numpy()]
    censored = sizes[survival["event"].eq(0).to_numpy()]
    assert diagnostic.loc["candidate_pool_size", "linked_mean"] == pytest.approx(
        float(linked.mean()), rel=1e-9
    )
    assert diagnostic.loc["candidate_pool_size", "censored_mean"] == pytest.approx(
        float(censored.mean()), rel=1e-9
    )


def test_detectability_cox_is_a_sensitivity_and_not_the_headline_model() -> None:
    """One extra model, reported beside the main one and never replacing it."""
    sensitivity = pd.read_csv(PROCESSED / "survival_cox_detectability_sensitivity.csv")
    main = pd.read_csv(PROCESSED / "survival_cox_results.csv")

    assert "log_candidate_pool_size" in set(sensitivity["covariate"])
    # The main Cox table is untouched by the sensitivity model.
    assert "log_candidate_pool_size" not in set(main["covariate"])
    assert set(main["covariate"]) <= set(sensitivity["covariate"])

    summary = json.loads((PROCESSED / "survival_analysis_summary.json").read_text())
    detectability = summary["detectability_cox"]
    assert detectability["role"].startswith("sensitivity only")

    # Main-model hazard ratios carried into the comparison must be the published
    # ones, so the two columns really are the same model before and after.
    published = main.set_index("covariate")["exp(coef)"]
    for row in sensitivity.itertuples():
        if row.covariate in published.index:
            assert row.hazard_ratio_main == pytest.approx(
                float(published.loc[row.covariate]), rel=1e-9
            )


def test_the_survival_report_publishes_the_pool_size_comparison() -> None:
    report = Path("SURVIVAL_ANALYSIS_REPORT.md").read_text(encoding="utf-8")
    sensitivity = pd.read_csv(
        PROCESSED / "survival_cox_detectability_sensitivity.csv"
    ).set_index("covariate")

    assert "Candidate-pool size is the largest imbalance" in report
    for covariate in ("framework_flag", "digital_segment_CPV-35"):
        adjusted = sensitivity.loc[covariate, "hazard_ratio_pool_adjusted"]
        assert f"{adjusted:.3f}" in report, covariate
    # No causal language attached to the added term.
    section = report.split("Candidate-pool size is the largest imbalance")[1]
    assert "Neither statement is causal" in section
