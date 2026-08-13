from scripts.run_final_pipeline import pipeline_stages


def test_final_pipeline_has_one_ordered_primary_path() -> None:
    stages = pipeline_stages()
    names = [stage.name for stage in stages]

    assert names == [
        "acquisition",
        "standardisation",
        "episodes",
        "benchmark_remap",
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
