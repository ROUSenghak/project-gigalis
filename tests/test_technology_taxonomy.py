"""Contract checks for the technology taxonomy classification layer.

These tests guard the two things that quietly invalidate a text classifier: a
related notice leaking across a fold boundary, and a reported number drifting
away from the table it was computed from.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, f1_score

from boamp_pipeline.technology_taxonomy import (
    CLASS_ORDER,
    MIN_RELIABLE_SUPPORT,
    N_SPLITS,
    SUBSTANTIVE_CLASSES,
    build_group_ids,
    cpv_tokens,
    normalize_objet,
)

PROCESSED = Path("data/processed/boamp")
TECHNOLOGY = PROCESSED / "technology"

pytestmark = pytest.mark.skipif(
    not (TECHNOLOGY / "technology_corpus.parquet").exists(),
    reason="technology layer not materialised; run scripts/build_technology_corpus.py",
)


def load_json(name: str) -> dict:
    return json.loads((TECHNOLOGY / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Unit behaviour
# ---------------------------------------------------------------------------


def test_normalisation_preserves_french_accents_and_acronyms() -> None:
    """The taxonomy is learned from accented French; flattening it loses evidence."""
    assert normalize_objet("Cybersécurité et SIEM") == "cybersécurité et siem"
    assert normalize_objet("Marché de TÉLÉPHONIE") == "marché de téléphonie"
    assert "oe" in normalize_objet("mise en \x9cuvre")
    assert normalize_objet("  double   space  ") == "double space"
    assert normalize_objet(None) == ""


def test_cpv_tokens_back_off_through_the_hierarchy() -> None:
    tokens = cpv_tokens(["72222000"])
    assert tokens == ["c7222", "d72", "f72222000", "g722"]
    assert cpv_tokens(["not-a-code"]) == []


def test_grouping_merges_near_duplicates_across_episodes() -> None:
    """Two episodes, one wording: the pair must not be separable by a fold."""
    groups, pairs = build_group_ids(
        ["a", "b", "c"],
        ["EP-1", "EP-2", "EP-3"],
        [
            "Hebergement d'applications web et gestion des noms de domaines",
            "hébergement d'applications web et gestion des noms de domaines.",
            "Fourniture de vêtements de travail",
        ],
    )
    assert groups[0] == groups[1], "near-identical notices were left in different groups"
    assert groups[2] != groups[0]
    assert len(pairs) == 1


def test_grouping_merges_notices_sharing_an_episode() -> None:
    groups, _ = build_group_ids(
        ["a", "b"], ["EP-1", "EP-1"], ["acquisition de licences", "maintenance du reseau"]
    )
    assert groups[0] == groups[1]


# ---------------------------------------------------------------------------
# Frozen taxonomy
# ---------------------------------------------------------------------------


def test_taxonomy_is_eight_substantive_classes_plus_three_fallbacks() -> None:
    assert len(SUBSTANTIVE_CLASSES) == 8
    assert len(CLASS_ORDER) == 11
    assert len(set(CLASS_ORDER)) == 11
    assert "AI" in SUBSTANTIVE_CLASSES


def test_corpus_uses_only_frozen_classes() -> None:
    corpus = pd.read_parquet(TECHNOLOGY / "technology_corpus.parquet")
    assert set(corpus["label"]) <= set(CLASS_ORDER)
    assert len(corpus) == 500
    assert corpus["idweb"].is_unique
    assert (corpus["text"].str.strip() != "").all()


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_no_procurement_family_is_split_across_folds() -> None:
    """The single check that decides whether any reported score is meaningful."""
    folds = pd.read_csv(TECHNOLOGY / "nlp_cv_folds.csv", dtype={"idweb": str})
    assert folds.groupby("group_id")["fold"].nunique().max() == 1
    assert sorted(folds["fold"].unique()) == list(range(N_SPLITS))
    assert folds["idweb"].is_unique


def test_every_class_appears_in_every_fold() -> None:
    folds = pd.read_csv(TECHNOLOGY / "nlp_cv_folds.csv", dtype={"idweb": str})
    support = pd.crosstab(folds["label"], folds["fold"])
    assert (support > 0).all().all(), f"a class is missing from a fold:\n{support}"


def test_out_of_fold_predictions_cover_the_corpus_exactly_once() -> None:
    decision = load_json("model_selection_decision.json")
    oof = pd.read_csv(TECHNOLOGY / "oof_predictions.csv")
    for model in decision["models"]:
        rows = oof.loc[oof["model"] == model]
        assert len(rows) == 500
        assert rows["idweb"].is_unique


# ---------------------------------------------------------------------------
# Reported numbers match the stored predictions
# ---------------------------------------------------------------------------


def test_headline_macro_f1_matches_the_stored_out_of_fold_predictions() -> None:
    decision = load_json("model_selection_decision.json")
    oof = pd.read_csv(TECHNOLOGY / "oof_predictions.csv")
    rows = oof.loc[oof["model"] == decision["selected_model"]]
    recomputed = f1_score(
        rows["true_label"], rows["predicted_label"],
        average="macro", labels=CLASS_ORDER, zero_division=0,
    )
    assert recomputed == pytest.approx(
        decision["selected_model_metrics"]["oof_macro_f1"], abs=5e-4
    )


def test_confusion_matrix_totals_equal_the_corpus_and_the_class_supports() -> None:
    decision = load_json("model_selection_decision.json")
    stored = pd.read_csv(TECHNOLOGY / "confusion_matrix.csv", index_col=0)
    oof = pd.read_csv(TECHNOLOGY / "oof_predictions.csv")
    rows = oof.loc[oof["model"] == decision["selected_model"]]
    expected = confusion_matrix(
        rows["true_label"], rows["predicted_label"], labels=list(CLASS_ORDER)
    )
    assert stored.to_numpy().sum() == 500
    assert np.array_equal(stored.to_numpy(), expected)

    corpus = pd.read_parquet(TECHNOLOGY / "technology_corpus.parquet")
    counts = corpus["label"].value_counts()
    for label in CLASS_ORDER:
        assert stored.loc[label].sum() == counts.get(label, 0)


def test_per_class_support_matches_the_reference_labels() -> None:
    corpus = pd.read_parquet(TECHNOLOGY / "technology_corpus.parquet")
    counts = corpus["label"].value_counts()
    per_class = pd.read_csv(TECHNOLOGY / "per_class_metrics.csv")
    for model, block in per_class.groupby("model"):
        block = block.set_index("technology")
        for label in CLASS_ORDER:
            assert int(block.loc[label, "support"]) == int(counts.get(label, 0)), model


def test_rare_classes_are_flagged_rather_than_presented_as_measured() -> None:
    per_class = pd.read_csv(TECHNOLOGY / "per_class_metrics.csv")
    ai = per_class.loc[per_class["technology"] == "AI"]
    assert (ai["support"] < MIN_RELIABLE_SUPPORT).all()
    assert not ai["support_adequate"].any()


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def test_temporal_split_uses_the_declared_windows_and_leaks_no_family() -> None:
    metrics = pd.read_csv(TECHNOLOGY / "temporal_validation_metrics.csv").iloc[0]
    assert metrics["train_years"] == "2015-2022"
    assert metrics["test_years"] == "2023-2025"
    assert metrics["n_train"] + metrics["n_test"] == 500
    # Families straddling the boundary go to training, so the test set can only
    # shrink relative to a naive year cut.
    corpus = pd.read_parquet(TECHNOLOGY / "technology_corpus.parquet")
    naive_test = int((corpus["year"] >= 2023).sum())
    assert metrics["n_test"] <= naive_test


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


def test_every_cohort_episode_has_exactly_one_prediction() -> None:
    predictions = pd.read_csv(TECHNOLOGY / "episode_technology_predictions.csv")
    cohort = pd.read_parquet(PROCESSED / "survival_cohort.parquet", columns=["episode_id"])
    assert predictions["episode_id"].is_unique
    assert set(predictions["episode_id"]) == set(cohort["episode_id"])
    assert len(predictions) == len(cohort)


def test_no_prediction_was_silently_dropped_for_low_confidence() -> None:
    predictions = pd.read_csv(TECHNOLOGY / "episode_technology_predictions.csv")
    assert set(predictions["confidence_status"]) <= {"high", "low"}
    assert (predictions["confidence_status"] == "low").any()
    assert predictions["predicted_technology"].notna().all()


def test_confidence_values_are_probabilities() -> None:
    predictions = pd.read_csv(TECHNOLOGY / "episode_technology_predictions.csv")
    assert predictions["confidence"].between(0.0, 1.0).all()
    assert (predictions["confidence"] >= predictions["runner_up_confidence"]).all()
    probability_columns = [c for c in predictions.columns if c.startswith("p_")]
    assert len(probability_columns) == len(CLASS_ORDER)
    assert np.allclose(predictions[probability_columns].sum(axis=1), 1.0, atol=1e-6)


def test_deployment_used_no_label_and_no_forbidden_feature() -> None:
    config = load_json("final_model_config.json")
    assert config["input_field"] == "objet"
    assert "CPV codes" in config["excluded_features"]
    predictions = pd.read_csv(TECHNOLOGY / "episode_technology_predictions.csv")
    assert "label" not in predictions.columns
    assert "true_label" not in predictions.columns


def test_confidence_coverage_is_reported_for_every_award_year() -> None:
    coverage = pd.read_csv(TECHNOLOGY / "confidence_coverage_by_year.csv")
    predictions = pd.read_csv(TECHNOLOGY / "episode_technology_predictions.csv")
    assert sorted(coverage["award_year"]) == sorted(predictions["award_year"].unique())
    assert coverage["n"].sum() == len(predictions)
    assert coverage["coverage"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# Support gates on the downstream enrichment
# ---------------------------------------------------------------------------


def test_downstream_enrichment_covers_only_classes_that_cleared_their_gate() -> None:
    evidence = load_json("technology_evidence_summary.json")
    support = pd.read_csv(TECHNOLOGY / "technology_survival_support.csv")
    analysed = set(pd.read_csv(TECHNOLOGY / "technology_survival_summary.csv")["technology"])
    eligible = set(support.loc[support["meets_support_gate"], "technology"])
    assert analysed == eligible
    assert "AI" not in analysed
    assert set(evidence["survival"]["excluded_classes"]).isdisjoint(analysed)

    trend_support = pd.read_csv(TECHNOLOGY / "technology_trend_support.csv")
    trend = set(pd.read_csv(TECHNOLOGY / "technology_trend_summary.csv")["technology"])
    assert trend == set(trend_support.loc[trend_support["meets_support_gate"], "technology"])
    assert "AI" not in trend


def test_the_frozen_cpv_analysis_is_untouched_by_the_technology_layer() -> None:
    """The taxonomy enriches the study; it must not have rewritten it."""
    summary = json.loads((PROCESSED / "survival_analysis_summary.json").read_text())
    assert summary["cohort"]["contracts"] == 3800
    assert summary["cohort"]["events"] == 544
    cohort = pd.read_parquet(PROCESSED / "survival_cohort.parquet", columns=["digital_segment"])
    assert set(cohort["digital_segment"]) <= {"CPV-32", "CPV-35", "CPV-48", "CPV-72"}


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


def test_the_technology_layer_is_one_always_run_stage() -> None:
    from scripts.run_final_pipeline import technology_stages

    stages = technology_stages()
    assert [stage.name for stage in stages] == ["technology_taxonomy"]
    # It refreshes every run: a skipped stage would leave the report quoting the
    # previous run's metrics.
    assert all(stage.always_run for stage in stages)


def test_every_stage_of_the_layer_is_importable_library_code() -> None:
    """The notebook must be able to run the same code the pipeline runs."""
    from boamp_pipeline.technology_evidence import build_evidence, build_predictions
    from boamp_pipeline.technology_models import (
        build_evaluation_artifacts,
        run_all_specifications,
        specifications,
    )
    from boamp_pipeline.technology_taxonomy import build_corpus_artifacts, build_corpus
    from scripts.build_technology_taxonomy import STAGES

    assert set(STAGES) == {"corpus", "models", "predictions", "evidence"}
    assert STAGES["corpus"] is build_corpus_artifacts
    assert STAGES["models"] is build_evaluation_artifacts
    assert STAGES["predictions"] is build_predictions
    assert STAGES["evidence"] is build_evidence
    assert len(specifications()) == 6
    assert callable(run_all_specifications) and callable(build_corpus)


def test_the_technology_layer_writes_nothing_the_frozen_study_owns() -> None:
    """The layer may read the study's outputs; it may not produce them."""
    from scripts.run_final_pipeline import (
        evidence_stages,
        pipeline_stages,
        technology_stages,
    )

    frozen = {
        str(path)
        for stage in pipeline_stages() + evidence_stages()
        for path in stage.outputs
    }
    produced = {str(path) for stage in technology_stages() for path in stage.outputs}
    assert not (frozen & produced), sorted(frozen & produced)
    for path in produced:
        assert "technology" in path.lower(), path


# ---------------------------------------------------------------------------
# Uncertainty and fit diagnostics
# ---------------------------------------------------------------------------


def test_the_headline_contrast_carries_a_family_bootstrap_interval() -> None:
    """The central claim needs an interval, not a multiple of a fold SD."""
    per_model = pd.read_csv(TECHNOLOGY / "bootstrap_macro_f1_ci.csv")
    paired = pd.read_csv(TECHNOLOGY / "bootstrap_paired_differences.csv")
    decision = load_json("model_selection_decision.json")
    selected = decision["selected_model"]

    assert (per_model["resampled_unit"] == "procurement family").all()
    assert (per_model["replicates"] >= 1000).all()
    # The families resampled must be the leakage unit, not the notices.
    corpus = pd.read_parquet(TECHNOLOGY / "technology_corpus.parquet")
    assert (per_model["families"] == corpus["group_id"].nunique()).all()
    for row in per_model.itertuples():
        assert row.ci_lower <= row.macro_f1 <= row.ci_upper, row.model

    contrast = paired.loc[
        (paired["model_a"] == "M0b_cpv_descriptor") & (paired["model_b"] == selected)
    ]
    assert len(contrast) == 1
    assert bool(contrast.iloc[0]["excludes_zero"]), "text-vs-CPV interval must be reported"


def test_train_and_validation_scores_are_both_recorded() -> None:
    """Without a training score the bias/variance question cannot be answered."""
    folds = pd.read_csv(TECHNOLOGY / "model_cv_fold_results.csv")
    scored = folds.loc[folds["model"] != "M_majority"]
    assert scored["train_macro_f1"].notna().all()
    assert scored["train_validation_gap"].notna().all()
    assert np.allclose(
        scored["train_validation_gap"], scored["train_macro_f1"] - scored["macro_f1"], atol=1e-3
    )

    curve = pd.read_csv(TECHNOLOGY / "learning_curve.csv")
    assert "train_macro_f1_mean" in curve.columns
    # Validation must be measured on held-out folds, so it cannot beat training.
    assert (curve["train_macro_f1_mean"] >= curve["macro_f1_mean"]).all()


def test_every_searched_specification_is_registered() -> None:
    register = pd.read_csv(TECHNOLOGY / "specification_register.csv")
    decision = load_json("model_selection_decision.json")
    assert set(register["model"]) == set(decision["models"])
    # The benchmark must be searched over the same regularisation range as the
    # text models, or the comparison measures tuning effort rather than signal.
    benchmark = json.loads(
        register.loc[register["model"] == "M0b_cpv_descriptor", "grid"].iloc[0]
    )
    text = json.loads(register.loc[register["model"] == "M1_tfidf_logreg", "grid"].iloc[0])
    assert benchmark["clf__C"] == text["clf__C"]


# ---------------------------------------------------------------------------
# The two downstream gates
# ---------------------------------------------------------------------------


def test_downstream_inclusion_requires_classifier_evidence_and_statistical_support() -> None:
    gate = pd.read_csv(TECHNOLOGY / "technology_classifier_gate.csv")
    survival = pd.read_csv(TECHNOLOGY / "technology_survival_support.csv")

    # Gate A must reject every fallback class outright: they are operational
    # residuals, not technologies, and a "comparison across technologies" that
    # contains them is answering a different question.
    for label in ("MIXED", "OTHER_DIGITAL", "OTHER"):
        row = gate.loc[gate["technology"] == label].iloc[0]
        assert not row["passes_classifier_gate"], label
        assert "fallback" in row["classifier_gate_reason"]

    # AI is rejected on support, not silently included because it is substantive.
    ai = gate.loc[gate["technology"] == "AI"].iloc[0]
    assert not ai["passes_classifier_gate"]
    assert "support" in ai["classifier_gate_reason"]

    analysed = set(pd.read_csv(TECHNOLOGY / "technology_survival_summary.csv")["technology"])
    assert analysed <= set(SUBSTANTIVE_CLASSES), "a fallback class reached the survival analysis"
    both = survival.loc[survival["meets_support_gate"], "technology"]
    assert set(both) == analysed
    assert (
        survival.loc[survival["meets_support_gate"], "passes_classifier_gate"].all()
        and survival.loc[survival["meets_support_gate"], "passes_statistical_gate"].all()
    )


def test_simultaneous_trend_tests_are_multiplicity_adjusted() -> None:
    trend = pd.read_csv(TECHNOLOGY / "technology_trend_summary.csv")
    assert {"p_holm", "p_bh", "n_tests"} <= set(trend.columns)
    assert (trend["n_tests"] == len(trend)).all()
    # Adjustment can only make a p-value less significant.
    assert (trend["p_holm"] >= trend["slope_p_value"] - 1e-9).all()
    assert (trend["p_bh"] >= trend["slope_p_value"] - 1e-9).all()
    # No class may be described as trending on a raw p-value alone.
    nominal_only = trend.loc[(trend["slope_p_value"] < 0.05) & (trend["p_holm"] >= 0.05)]
    for row in nominal_only.itertuples():
        assert "does not survive" in row.direction, row.technology


def test_technology_trends_share_the_canonical_window_and_recent_slope() -> None:
    quarterly = pd.read_csv(TECHNOLOGY / "technology_quarterly_counts.csv")
    trend = pd.read_csv(TECHNOLOGY / "technology_trend_summary.csv")

    assert quarterly["quarter"].iloc[0] == "2015Q2"
    assert quarterly["quarter"].iloc[-1] == "2025Q4"
    assert len(quarterly) == 43
    assert (trend["observation_window_quarters"] == 43).all()
    assert (trend["quarters"] == 12).all()

    protocol = Path("PROJECT_WORK_PROTOCOL.md").read_text(encoding="utf-8")
    assert "technology series of §3.9 is fitted on `44` quarters" not in protocol
    assert "same `43`-quarter observation" in protocol


def test_the_confidence_threshold_trend_sensitivity_was_actually_run() -> None:
    sensitivity = pd.read_csv(TECHNOLOGY / "technology_trend_confidence_sensitivity.csv")
    assert set(sensitivity["arm"]) == {"all_predictions", "confidence_ge_0.70"}
    analysed = set(pd.read_csv(TECHNOLOGY / "technology_trend_summary.csv")["technology"])
    assert set(sensitivity["technology"]) == analysed
    # The published series must be the unfiltered one: the filtered arm is a
    # diagnostic, and adopting it would estimate classifier certainty over time.
    filtered = sensitivity.loc[sensitivity["arm"] == "confidence_ge_0.70"]
    unfiltered = sensitivity.loc[sensitivity["arm"] == "all_predictions"]
    assert filtered["episodes"].sum() < unfiltered["episodes"].sum()
    assert unfiltered["zero_quarters"].max() <= filtered["zero_quarters"].max()


# ---------------------------------------------------------------------------
# Deployed-confidence consistency
#
# The published section 12 once described a Platt-scaled score while the
# pipeline deployed the raw one, and every existing test passed: each artifact
# was internally consistent, and nothing compared them to each other. These
# tests tie the report, the notebook, the config, the run log and the deployment
# CSV to one declared variant, so the same drift cannot recur silently.
# ---------------------------------------------------------------------------

REPORT = Path("TECHNOLOGY_TAXONOMY_REPORT.md")
NOTEBOOK = Path("notebooks/15_technology_taxonomy_classification.ipynb")
BUILD_LOG = Path("logs/build_technology_taxonomy.log")


def deployed_variant() -> str:
    return load_json("final_model_config.json")["calibration"]["deployed_variant"]


def report_text() -> str:
    return REPORT.read_text(encoding="utf-8")


def test_the_deployed_variant_is_one_of_the_evaluated_variants() -> None:
    calibration = load_json("final_model_config.json")["calibration"]
    assert calibration["deployed_variant"] in calibration["variants"]
    # The adoption flag and the deployed variant are one decision, not two.
    expected = "calibrated" if calibration["adopted"] else "raw"
    assert calibration["deployed_variant"] == expected


def test_deployment_rows_declare_the_deployed_variant() -> None:
    calibration = load_json("final_model_config.json")["calibration"]
    predictions = pd.read_csv(TECHNOLOGY / "episode_technology_predictions.csv")
    declared = set(predictions["confidence_type"].dropna())
    assert len(declared) == 1, declared
    expected = (
        "calibrated_class_probability"
        if calibration["adopted"]
        else "uncalibrated_class_score"
    )
    assert declared == {expected}


def test_the_evidence_summary_records_the_deployed_variant() -> None:
    summary = load_json("technology_evidence_summary.json")["deployed_confidence"]
    config = load_json("final_model_config.json")["calibration"]
    predictions = pd.read_csv(TECHNOLOGY / "episode_technology_predictions.csv")
    assert summary["variant"] == config["deployed_variant"]
    assert summary["adopted"] == config["adopted"]
    assert summary["confidence_type"] == predictions["confidence_type"].iloc[0]
    assert summary["episodes"] == len(predictions)
    assert summary["max_confidence"] == pytest.approx(
        float(predictions["confidence"].max())
    )


def test_the_report_names_the_deployed_variant_and_not_the_other_one() -> None:
    variant = deployed_variant()
    text = report_text()
    section = text.split("## 12.")[1].split("## 13.")[0]
    assert f"the **{variant}** variant" in section

    if variant == "raw":
        # The rejected branch must be described as rejected, never as adopted.
        assert "**was evaluated and rejected**" in section
        assert "Calibration was adopted" not in text
        assert "calibrated model confidence\nscore" not in section
        assert "uncalibrated model confidence score" in section
    else:
        assert "**Calibration was adopted**" in section
        assert "calibrated model confidence score" in section


def test_the_published_reliability_table_is_the_deployed_variant() -> None:
    """Bin counts in the report must match the deployed variant's bins."""
    reliability = pd.read_csv(TECHNOLOGY / "confidence_reliability_oof.csv")
    deployed = reliability.loc[reliability["variant"] == deployed_variant()]
    other = reliability.loc[reliability["variant"] != deployed_variant()]
    section = report_text().split("## 12.")[1].split("### Result 5")[0]

    rows = [line for line in section.splitlines() if line.startswith("| [")]
    assert len(rows) == len(deployed), (len(rows), len(deployed))
    published_n = [int(line.split("|")[2].strip()) for line in rows]
    assert published_n == deployed["n"].tolist()
    # And it must not be the variant that did not ship, when the two differ.
    if published_n != other["n"].tolist():
        assert published_n != other["n"].tolist()


def test_the_reported_cutoff_behaviour_recomputes_from_deployed_predictions() -> None:
    predictions = pd.read_csv(TECHNOLOGY / "episode_technology_predictions.csv")
    sweep = pd.read_csv(TECHNOLOGY / "confidence_cutoff_sweep.csv")
    summary = load_json("technology_evidence_summary.json")["deployed_confidence"]
    scores = predictions["confidence"].astype(float)

    for cutoff in sweep["cutoff"]:
        recomputed = int((scores >= float(cutoff)).sum())
        assert int(sweep.loc[sweep["cutoff"] == cutoff, "retained"].iloc[0]) == recomputed
        assert summary["counts_at_or_above"][f"{float(cutoff):g}"] == recomputed

    # Any claim about an unusable cutoff must be true of the shipped scores.
    for cutoff in summary["cutoffs_with_no_predictions"]:
        assert int((scores >= float(cutoff)).sum()) == 0
    reachable = summary["highest_reachable_cutoff"]
    if reachable is not None:
        assert int((scores >= float(reachable)).sum()) > 0


def test_the_report_never_claims_an_unreachable_cutoff() -> None:
    predictions = pd.read_csv(TECHNOLOGY / "episode_technology_predictions.csv")
    section = report_text().split("## 12.")[1].split("## 13.")[0]
    maximum = float(predictions["confidence"].max())
    for cutoff in (0.80, 0.90):
        claim = f"no {deployed_variant()} prediction reaches `{cutoff:.2f}`"
        if int((predictions["confidence"] >= cutoff).sum()) > 0:
            assert claim.lower() not in section.lower()
    assert f"`{maximum:.4f}`" in section


def test_the_run_log_agrees_with_the_deployed_variant() -> None:
    if not BUILD_LOG.exists():
        pytest.skip("build log not present in this checkout")
    adopted = [
        line for line in BUILD_LOG.read_text(encoding="utf-8").splitlines()
        if "Confidence variant adopted:" in line
    ]
    assert adopted, "the build log records no confidence decision"
    assert adopted[-1].split("Confidence variant adopted:")[1].strip().split()[0] == (
        deployed_variant()
    )


def test_the_notebook_narrative_matches_the_deployed_variant() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    prose = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    if deployed_variant() == "raw":
        assert "**rejected**" in prose
        assert "Calibration was adopted under a" not in prose
        assert "The deployed model emits Platt-scaled class probabilities" not in prose
    else:
        assert "Platt-scaled" in prose
    # The reliability display must select the variant from the config, never a
    # literal, so it follows the decision instead of restating it.
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'config["calibration"]["deployed_variant"]' in code


def test_one_prediction_per_cohort_episode() -> None:
    predictions = pd.read_csv(TECHNOLOGY / "episode_technology_predictions.csv")
    cohort = pd.read_parquet(PROCESSED / "survival_cohort.parquet", columns=["episode_id"])
    assert predictions["episode_id"].is_unique
    assert set(predictions["episode_id"]) == set(cohort["episode_id"])
    assert len(predictions) == len(cohort)
