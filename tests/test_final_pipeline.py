from scripts.run_final_pipeline import evidence_stages, pipeline_stages


def test_final_pipeline_has_one_ordered_primary_path() -> None:
    stages = pipeline_stages()
    names = [stage.name for stage in stages]

    assert names == [
        "acquisition",
        "standardisation",
        "episodes",
        "cohort",
        "candidates",
        "fellegi_sunter_comparison",
        "primary_linkage",
        "primary_survival_dataset",
        "expiry_sensitivity_audit",
        "expiry_sensitivity_survival_dataset",
    ]
    assert names.index("primary_survival_dataset") < names.index("expiry_sensitivity_audit")


def test_primary_and_sensitivity_outputs_are_separate() -> None:
    stages = {stage.name: stage for stage in pipeline_stages()}
    primary = set(stages["primary_linkage"].outputs)
    sensitivity = set(stages["expiry_sensitivity_audit"].outputs)

    assert primary.isdisjoint(sensitivity)
    assert any(path.name == "accepted_successor_links.parquet" for path in primary)
    assert any(path.name == "expiry_link_review.csv" for path in sensitivity)


def test_one_national_benchmark_drives_all_accuracy_evidence() -> None:
    stages = {stage.name: stage for stage in evidence_stages()}

    assert stages["benchmark_evaluate_dev"].command[-2:] == ("--event-set", "primary")
    assert stages["benchmark_evaluate_validation"].command[-2:] == ("--event-set", "primary")
    all_outputs = {path for stage in stages.values() for path in stage.outputs}
    legacy_name = "benchmark_" + "v3"
    assert all(legacy_name not in str(path) for path in all_outputs)
    assert list(stages)[-1] == "canonical_state_validation"
