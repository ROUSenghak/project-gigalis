#!/usr/bin/env python3
"""Run the final BOAMP analysis in one reproducible, ordered workflow."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = Path("data/processed/boamp")


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    accepts_force: bool = False
    #: Cheap derivations of current inputs - evaluation, reports, figures,
    #: notebooks-facing artifacts. They must never be skipped because their
    #: outputs happen to exist: an edited generator would then leave a stale
    #: document behind, which is exactly the failure this pipeline exists to
    #: prevent. Only the expensive upstream stages earn a skip.
    always_run: bool = False


def regional_reference_stage() -> Stage:
    """Materialise the Grand Ouest regional reference onto current episode ids.

    The reference itself is fixed input under ``data/reference/``; this stage
    only re-resolves it onto the current episode reconstruction and cuts its
    exposure from the production candidate pool, so it must rerun whenever
    episodes or candidates are rebuilt.
    """
    reference = PROCESSED / "regional_benchmark"
    return Stage(
        "regional_reference",
        ("scripts/build_regional_benchmark.py",),
        (
            reference / "benchmark_dev.parquet",
            reference / "benchmark_validation.parquet",
            reference / "exposure_full.parquet",
            reference / "regional_benchmark_manifest.json",
        ),
        True,
    )


def evidence_stages() -> tuple[Stage, ...]:
    """Evaluate the frozen reference and refresh every reader-facing artifact."""
    bench = PROCESSED / "regional_benchmark"
    return (
        Stage(
            "reference_evaluate_dev",
            ("scripts/evaluate_linkage.py", "--evaluate-split", "dev", "--event-set", "primary"),
            (PROCESSED / "linkage_evaluation_dev.json",),
            always_run=True,
        ),
        Stage(
            "reference_evaluate_validation",
            ("scripts/evaluate_linkage.py", "--evaluate-split", "validation", "--event-set", "primary"),
            (PROCESSED / "linkage_evaluation_validation.json",),
            always_run=True,
        ),
        Stage(
            "reference_modeling_tables",
            ("scripts/build_benchmark_modeling_dataset.py",),
            (
                bench / "modeling" / "modeling_dev.parquet",
                bench / "modeling" / "modeling_validation.parquet",
                bench / "modeling" / "modeling_summary.json",
            ),
            True,
            always_run=True,
        ),
        Stage(
            "reference_quality_evidence",
            ("scripts/build_quality_evidence.py",),
            (
                PROCESSED / "quality_evidence" / "quality_evidence_summary.json",
                PROCESSED / "quality_evidence" / "validation_anchor_confusion.csv",
                PROCESSED / "quality_evidence" / "validation_pair_curve_metrics.csv",
                PROCESSED / "quality_evidence" / "dev_m_b_threshold_sweep.csv",
                PROCESSED / "quality_evidence" / "validation_m_b_threshold_sweep.csv",
                Path("QUALITY_EVIDENCE.md"),
                Path("reports/figures/benchmark_validation_confusion_matrices.png"),
                Path("reports/figures/benchmark_validation_pair_roc.png"),
                Path("reports/figures/benchmark_validation_pair_precision_recall.png"),
                Path("reports/figures/benchmark_validation_m_b_threshold_tradeoff.png"),
            ),
            always_run=True,
        ),
        # One ordering invariant governs the rest of this tuple: every stage that
        # writes an evidence artifact runs before every stage that reads it. The
        # reader-artifact refresh sits near the end because the methodology
        # chapter quotes the candidate-generation audit, the survival summary and
        # Cox table, the trend summary and signal matrix, the buyer-blocking
        # audit, and the completed review diagnostic. Running it earlier does not
        # fail loudly -- it silently republishes the previous run's numbers.
        Stage(
            "candidate_generation_audit",
            ("scripts/audit_candidate_generation.py",),
            (
                PROCESSED / "candidate_generation_audit.json",
                PROCESSED / "candidate_generation_unreachable.csv",
                PROCESSED / "candidate_generation_cpv_transitions.csv",
                Path("CANDIDATE_GENERATION_AUDIT.md"),
            ),
            True,
            always_run=True,
        ),
        # Survival evidence precedes the project evidence stage: the data-quality
        # report's event-validation section quotes the survival summary, so
        # generating it second would document the previous run's numbers.
        Stage(
            "survival_evidence",
            ("scripts/build_survival_evidence.py",),
            (
                PROCESSED / "survival_analysis_summary.json",
                PROCESSED / "survival_km_horizons.csv",
                PROCESSED / "survival_segment_summary.csv",
                PROCESSED / "survival_conditional_probabilities.csv",
                PROCESSED / "survival_selection_diagnostic.csv",
                PROCESSED / "survival_selection_by_category.csv",
                PROCESSED / "survival_cox_results.csv",
                PROCESSED / "survival_ph_diagnostics.csv",
                PROCESSED / "survival_linkage_sensitivity.csv",
                PROCESSED / "survival_cox_linkage_sensitivity.csv",
                PROCESSED / "survival_borderline_link_sensitivity.csv",
                PROCESSED / "survival_template_risk_sensitivity.csv",
                PROCESSED / "survival_parametric_comparison.csv",
                Path("SURVIVAL_ANALYSIS_REPORT.md"),
                Path("reports/figures/survival_kaplan_meier.png"),
                Path("reports/figures/survival_conditional_probabilities.png"),
            ),
            always_run=True,
        ),
        # Precedes the project evidence stage, which quotes the buyer-blocking
        # audit summary in the data-quality report.
        Stage(
            "buyer_blocking_legal_form_audit",
            ("scripts/audit_buyer_blocking_legal_forms.py",),
            (
                PROCESSED / "buyer_blocking_legal_form_audit.csv",
                PROCESSED / "buyer_blocking_legal_form_audit_summary.json",
            ),
            always_run=True,
        ),
        Stage(
            "project_quality_and_trend_evidence",
            ("scripts/build_project_evidence.py",),
            (
                PROCESSED / "data_quality_profile.json",
                PROCESSED / "trend_analysis_summary.json",
                PROCESSED / "trend_quarterly.csv",
                PROCESSED / "trend_breakpoints.csv",
                PROCESSED / "trend_signal_matrix.csv",
                Path("DATA_QUALITY_REPORT.md"),
                Path("TREND_ANALYSIS_REPORT.md"),
                Path("notebooks/14_data_quality_and_trend_analysis.ipynb"),
                Path("reports/figures/data_quality_key_missingness.png"),
                Path("reports/figures/trend_quarterly_episode_counts.png"),
                Path("reports/figures/trend_duration_completeness.png"),
            ),
            always_run=True,
        ),
        Stage(
            "completed_review_diagnostic",
            ("scripts/evaluate_review_audit.py",),
            (
                Path("data/review/review_audit_evaluation.json"),
                Path("data/review/review_audit_findings.csv"),
                Path("REVIEW_AUDIT_RESULTS.md"),
            ),
            always_run=True,
        ),
        # Last of the reader-facing stages: it consumes every evidence artifact
        # written above.
        Stage(
            "reader_artifact_refresh",
            ("scripts/refresh_reader_artifacts.py",),
            (
                Path("FINAL_PIPELINE.md"),
                Path("REGIONAL_BENCHMARK_REFERENCE.md"),
                Path("notebooks/12_successor_linkage_and_evaluation.ipynb"),
                Path("reports/boamp_methodology_chapter.tex"),
                Path("reports/boamp_methodology_chapter.pdf"),
                Path("reports/figures/benchmark_validation_method_metrics.png"),
            ),
            always_run=True,
        ),
        Stage(
            "readiness_report_data",
            ("scripts/build_readiness_report.py",),
            (
                Path("reports/current_project_readiness.db"),
                Path("reports/current_project_readiness_artifact.json"),
            ),
            always_run=True,
        ),
        Stage(
            "canonical_state_validation",
            ("scripts/validate_canonical_state.py",),
            (PROCESSED / "canonical_state_validation.json",),
            always_run=True,
        ),
    )


def technology_stages() -> tuple[Stage, ...]:
    """The supervised technology taxonomy layer, added on top of the study.

    One stage, four ordered sub-stages inside it. Every line of analysis lives in
    ``boamp_pipeline/technology_{taxonomy,models,evidence}.py`` so that
    ``notebooks/15`` imports and runs the same code, and this stage only invokes
    it. Splitting the sub-stages into separate pipeline entries would buy a
    finer skip granularity on a chain that takes a minute end to end, at the
    cost of a report that quotes the previous run's macro-F1 whenever one link
    is rebuilt and the next is skipped.

    It reads frozen artifacts and writes only under
    ``data/processed/boamp/technology/``, the technology figures, and its own
    report. It cannot alter the linkage, survival, or trend results, which is
    why it runs as its own group after the evidence stages.
    """
    technology = PROCESSED / "technology"
    return (
        Stage(
            "technology_taxonomy",
            ("scripts/build_technology_taxonomy.py", "--quiet"),
            (
                technology / "technology_corpus.parquet",
                technology / "annotation_audit.csv",
                technology / "annotation_audit_summary.json",
                technology / "nlp_cv_folds.csv",
                technology / "model_cv_results.csv",
                technology / "per_class_metrics.csv",
                technology / "oof_predictions.csv",
                technology / "confusion_matrix.csv",
                technology / "error_analysis.csv",
                technology / "learning_curve.csv",
                technology / "temporal_validation_metrics.csv",
                technology / "model_selection_decision.json",
                technology / "episode_technology_predictions.csv",
                technology / "final_model_config.json",
                technology / "technology_classifier.joblib",
                technology / "confidence_coverage_by_year.csv",
                technology / "technology_composition.csv",
                technology / "technology_cpv_crosswalk.csv",
                technology / "technology_survival_support.csv",
                technology / "technology_trend_summary.csv",
                technology / "technology_evidence_summary.json",
                Path("TECHNOLOGY_TAXONOMY_REPORT.md"),
                Path("reports/figures/technology_confusion_matrix.png"),
                Path("reports/figures/technology_learning_curve.png"),
                Path("reports/figures/technology_composition.png"),
                Path("reports/figures/technology_confidence_coverage.png"),
            ),
            True,
            always_run=True,
        ),
    )


def pipeline_stages() -> tuple[Stage, ...]:
    return (
        Stage(
            "acquisition",
            ("scripts/download_boamp.py",),
            tuple(Path(f"data/raw/boamp/boamp_{year}.jsonl") for year in range(2015, 2026)),
            True,
        ),
        Stage(
            "standardisation",
            ("scripts/build_standardized_notices.py",),
            (
                PROCESSED / "notices_standardized_2015_2025.parquet",
                PROCESSED / "notices_grand_ouest.parquet",
                PROCESSED / "standardized_notice_summary.json",
            ),
            True,
        ),
        Stage(
            "episodes",
            ("scripts/build_procurement_episodes.py",),
            (
                PROCESSED / "episodes_2015_2025.parquet",
                PROCESSED / "episodes_grand_ouest.parquet",
                PROCESSED / "episode_membership.parquet",
                PROCESSED / "episode_reconstruction_summary.json",
            ),
            True,
        ),
        Stage(
            "cohort",
            ("scripts/build_survival_cohort.py",),
            (PROCESSED / "survival_cohort.parquet", PROCESSED / "survival_cohort_summary.json"),
            True,
        ),
        Stage(
            "candidates",
            ("scripts/build_linkage_candidates.py",),
            (PROCESSED / "linkage_candidates.parquet", PROCESSED / "linkage_candidates_summary.json"),
            True,
        ),
        Stage(
            "fellegi_sunter_comparison",
            ("scripts/fit_fellegi_sunter.py",),
            (PROCESSED / "linkage_candidates_scored.parquet", PROCESSED / "fellegi_sunter_model.json"),
            True,
        ),
        regional_reference_stage(),
        Stage(
            "primary_linkage",
            ("scripts/evaluate_linkage.py",),
            (
                PROCESSED / "accepted_successor_links.parquet",
                PROCESSED / "linkage_config.json",
                PROCESSED / "linkage_application_summary.json",
            ),
        ),
        Stage(
            "primary_survival_dataset",
            ("scripts/build_survival_dataset.py",),
            (PROCESSED / "survival_dataset.parquet", PROCESSED / "survival_dataset_summary.json"),
            True,
        ),
    )


def source_dependencies(script: Path) -> list[Path]:
    """The stage's script plus every first-party module it imports."""
    seen: set[Path] = set()
    stack = [script]
    found = [script]
    while stack:
        current = stack.pop()
        try:
            tree = ast.parse(current.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            candidates: list[Path] = []
            if node.module.startswith("boamp_pipeline"):
                parts = node.module.split(".")
                names = [parts[1]] if len(parts) > 1 else [alias.name for alias in node.names]
                candidates = [PROJECT_ROOT / "boamp_pipeline" / f"{name}.py" for name in names]
            elif node.module.startswith("scripts"):
                candidates = [PROJECT_ROOT / "scripts" / f"{node.module.split('.')[-1]}.py"]
            for candidate in candidates:
                if candidate.exists() and candidate not in seen:
                    seen.add(candidate)
                    stack.append(candidate)
                    found.append(candidate)
    return found


def generator_is_newer_than_outputs(stage: Stage) -> Path | None:
    """Return the source file that post-dates the stage's oldest output.

    A stage that skips on "outputs exist" will keep serving results from a
    generator that has since been edited. That is how a report ends up
    describing a methodology the code no longer implements, so the skip is
    conditioned on the code being older than what it produced.
    """
    script = PROJECT_ROOT / stage.command[0]
    if not script.exists():
        return None
    outputs = [PROJECT_ROOT / path for path in stage.outputs if (PROJECT_ROOT / path).exists()]
    if not outputs:
        return None
    oldest_output = min(path.stat().st_mtime for path in outputs)
    newest_source = max(source_dependencies(script), key=lambda path: path.stat().st_mtime)
    return newest_source if newest_source.stat().st_mtime > oldest_output else None


def run_command(command: list[str], dry_run: bool) -> None:
    print("$", " ".join(command), flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def run_stage(stage: Stage, force: bool, dry_run: bool) -> str:
    resolved_outputs = [PROJECT_ROOT / path for path in stage.outputs]
    existing = [path for path in resolved_outputs if path.exists()]
    if stage.always_run:
        force = True
    stale_source = generator_is_newer_than_outputs(stage)
    if not force and stale_source is not None:
        print(
            f"REBUILD {stage.name}: {stale_source.relative_to(PROJECT_ROOT)} "
            "is newer than its outputs",
            flush=True,
        )
        force = True
    if not force and len(existing) == len(resolved_outputs):
        print(f"SKIP {stage.name}: outputs already complete", flush=True)
        return "skipped_complete"
    if not force and existing:
        missing = [str(path.relative_to(PROJECT_ROOT)) for path in resolved_outputs if not path.exists()]
        raise RuntimeError(
            f"Stage {stage.name} has partial outputs. Missing: {missing}. "
            "Rerun with --force to rebuild this workflow consistently."
        )

    command = [sys.executable, *stage.command]
    if force and stage.accepts_force:
        command.append("--force")
    run_command(command, dry_run)
    if not dry_run:
        missing = [path for path in resolved_outputs if not path.exists()]
        if missing:
            raise RuntimeError(f"Stage {stage.name} completed without expected outputs: {missing}")
    return "planned" if dry_run else "completed"


def run_notebooks(dry_run: bool) -> None:
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--execute",
        "--to",
        "notebook",
        "--inplace",
        "notebooks/10_standardized_notice_and_episode_evidence_audit.ipynb",
        "notebooks/11_cohort_and_data_quality.ipynb",
        "notebooks/12_successor_linkage_and_evaluation.ipynb",
        "notebooks/13_survival_analysis.ipynb",
        "notebooks/14_data_quality_and_trend_analysis.ipynb",
        "notebooks/15_technology_taxonomy_classification.ipynb",
        "--ExecutePreprocessor.timeout=0",
    ]
    run_command(command, dry_run)


def write_manifest(statuses: dict[str, str]) -> Path:
    path = PROJECT_ROOT / PROCESSED / "final_pipeline_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workflow": "canonical_boamp_pipeline",
        "primary_method": "M_B_text_ranking @ 0.70",
        "primary_links": str(PROCESSED / "accepted_successor_links.parquet"),
        "primary_survival_dataset": str(PROCESSED / "survival_dataset.parquet"),
        "event_definition_sensitivity_arms": [
            "M_B_text_ranking @ 0.80",
            "M_B_text_ranking @ 0.70",
            "M_B_text_ranking @ 0.60",
            "M_C_weighted_gated @ 0.70",
        ],
        "borderline_robustness": "M_B text_component band [0.65, 0.75] excluded",
        "removed_arm": "M_E_expiry_aware_text",
        "removed_arm_note": (
            "Built and evaluated, never promoted, and removed from the repository in "
            "full: its code, outputs, and tests. Reliable duration is missing for most "
            "of the cohort, so the rule could differentiate itself only on a minority "
            "of episodes. History remains in version control."
        ),
        "technology_taxonomy": "data/processed/boamp/technology/",
        "technology_taxonomy_report": "TECHNOLOGY_TAXONOMY_REPORT.md",
        "technology_taxonomy_status": (
            "supervised business technology classifier over procurement text; an "
            "enrichment layer that leaves the CPV-based cohort, linkage, survival, "
            "and trend results unchanged"
        ),
        "reference": "regional_grand_ouest",
        "evaluation_status": "regional reference sample; external accuracy not established",
        "independent_review_sample": "data/review/independent_link_review_sample.csv",
        "independent_review_protocol": "INDEPENDENT_LINK_REVIEW_PROTOCOL.md",
        "independent_review_status": (
            "frozen historical diagnostic; the sample was drawn once and reviewed, so "
            "the pipeline no longer regenerates it"
        ),
        "stage_status": statuses,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild every materialised stage.")
    parser.add_argument("--with-notebooks", action="store_true", help="Execute evidence notebooks 10-14.")
    parser.add_argument("--with-tests", action="store_true", help="Run the test suite after the pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Print work without executing commands.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    statuses: dict[str, str] = {}
    for stage in pipeline_stages():
        statuses[stage.name] = run_stage(stage, args.force, args.dry_run)

    for stage in evidence_stages():
        statuses[stage.name] = run_stage(stage, args.force, args.dry_run)

    for stage in technology_stages():
        statuses[stage.name] = run_stage(stage, args.force, args.dry_run)

    if args.with_notebooks:
        run_notebooks(args.dry_run)
        statuses["evidence_notebooks"] = "planned" if args.dry_run else "completed"
    if args.with_tests:
        run_command([sys.executable, "-m", "pytest", "-q"], args.dry_run)
        statuses["tests"] = "planned" if args.dry_run else "completed"

    if not args.dry_run:
        manifest = write_manifest(statuses)
        print(f"Manifest: {manifest.relative_to(PROJECT_ROOT)}", flush=True)
    print("Primary result: M_B_text_ranking @ 0.70", flush=True)
    print("Sensitivity: M_B @ 0.80/0.70/0.60, M_C @ 0.70, borderline band", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
