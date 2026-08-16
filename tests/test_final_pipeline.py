from scripts.run_final_pipeline import (
    evidence_stages,
    generator_is_newer_than_outputs,
    pipeline_stages,
    source_dependencies,
)


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
        "regional_reference",
        "primary_linkage",
        "primary_survival_dataset",
    ]
    # The reference is cut from the scored candidate pool, so it cannot be built
    # before that pool exists.
    assert names.index("fellegi_sunter_comparison") < names.index("regional_reference")


def test_the_removed_expiry_arm_has_no_stage() -> None:
    """The duration-conditioned arm was removed from the repository in full.
    A stage reappearing here would resurrect outputs no report describes."""
    stages = pipeline_stages() + evidence_stages()
    for stage in stages:
        assert "expiry" not in stage.name
        assert all("expiry" not in str(path) for path in stage.outputs)
        assert all("expiry" not in part for part in stage.command)


def test_one_regional_reference_drives_all_accuracy_evidence() -> None:
    stages = {stage.name: stage for stage in evidence_stages()}

    assert stages["reference_evaluate_dev"].command[-2:] == ("--event-set", "primary")
    assert stages["reference_evaluate_validation"].command[-2:] == ("--event-set", "primary")
    all_outputs = {path for stage in stages.values() for path in stage.outputs}
    legacy_name = "benchmark_" + "v3"
    assert all(legacy_name not in str(path) for path in all_outputs)
    assert list(stages)[-1] == "canonical_state_validation"


def test_every_evidence_stage_runs_before_the_stages_that_quote_it() -> None:
    """Reader-facing stages must follow the stages whose artifacts they read.

    Nothing fails loudly when this order is wrong: the consumer finds last
    run's file on disk and republishes its numbers under this run's timestamp.
    Each pair below is a real read in the consuming script.
    """
    names = [stage.name for stage in evidence_stages()]
    required_order = (
        # DATA_QUALITY_REPORT.md quotes survival_analysis_summary.json.
        ("survival_evidence", "project_quality_and_trend_evidence"),
        # ... and buyer_blocking_legal_form_audit_summary.json.
        ("buyer_blocking_legal_form_audit", "project_quality_and_trend_evidence"),
        # The methodology chapter quotes all of these.
        ("candidate_generation_audit", "reader_artifact_refresh"),
        ("survival_evidence", "reader_artifact_refresh"),
        ("project_quality_and_trend_evidence", "reader_artifact_refresh"),
        ("buyer_blocking_legal_form_audit", "reader_artifact_refresh"),
        ("completed_review_diagnostic", "reader_artifact_refresh"),
        # The readiness artifact quotes the data-quality and trend profiles.
        ("project_quality_and_trend_evidence", "readiness_report_data"),
    )
    for producer, consumer in required_order:
        assert names.index(producer) < names.index(consumer), (
            f"{consumer} reads an artifact written by {producer}"
        )


def test_evidence_stages_never_skip_on_existing_outputs() -> None:
    """A stage that skips because its output file exists will happily leave a
    stale report behind after its generator changes. Everything downstream of
    the data is cheap to recompute, so it recomputes."""
    for stage in evidence_stages():
        assert stage.always_run, f"{stage.name} may go stale"
    # The expensive upstream stages keep their skip.
    assert not any(stage.always_run for stage in pipeline_stages())


def test_no_stage_touches_the_retired_national_benchmark() -> None:
    """The retired benchmark was deleted; no stage may read or write its paths."""
    stages = [*pipeline_stages(), *evidence_stages()]
    retired = "data/processed/boamp/benchmark/"

    for stage in stages:
        assert all(retired not in str(path) for path in stage.outputs), stage.name
        assert all("archive/" not in part for part in stage.command), stage.name


def test_stage_dependencies_include_first_party_modules() -> None:
    """A stage is stale when its library changed, not only its script."""
    from pathlib import Path

    deps = source_dependencies(Path("scripts/evaluate_linkage.py"))
    names = {path.name for path in deps}

    assert "evaluate_linkage.py" in names
    assert "linkage.py" in names, "boamp_pipeline imports must count as dependencies"
    assert "regional_benchmark_io.py" in names


def test_every_materialised_stage_is_current() -> None:
    """No stage may be serving outputs older than the code that writes them."""
    stale = {
        stage.name: str(source)
        for stage in (*pipeline_stages(), *evidence_stages())
        if (source := generator_is_newer_than_outputs(stage)) is not None
    }

    assert not stale, f"outputs predate their generator: {stale}"
