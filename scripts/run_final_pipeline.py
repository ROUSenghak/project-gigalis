#!/usr/bin/env python3
"""Run the final BOAMP analysis in one reproducible, ordered workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = Path("data/processed/boamp_v2")


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    accepts_force: bool = False


def benchmark_v3_stages() -> tuple[Stage, ...]:
    """Current national benchmark construction.

    Kept out of the default path on purpose: annotation and sealed-test handling
    are separate from the core Grand Ouest survival refresh.
    """
    bench = PROCESSED / "benchmark_v3"
    return (
        Stage("benchmark_national_index", ("scripts/build_national_episode_index.py",),
              (bench / "national_episode_index.parquet",), True),
        Stage("benchmark_frame", ("scripts/build_benchmark_frame.py",),
              (bench / "frame_national.parquet", bench / "frame_strata_summary.json"), True),
        Stage("benchmark_mine_declarations", ("scripts/mine_renewal_declarations.py",),
              (bench / "renewal_declarations.parquet",), True),
        Stage("benchmark_resolve_predecessors", ("scripts/resolve_declared_predecessors.py",),
              (bench / "declared_predecessor_links.parquet",), True),
        Stage("benchmark_sample_anchors", ("scripts/sample_benchmark_anchors.py",),
              (bench / "anchors.parquet",), True),
        Stage("benchmark_exposure", ("scripts/build_benchmark_exposure.py",),
              (bench / "exposure_full.parquet", bench / "pool_definition.json"), True),
        Stage("benchmark_structural_negatives", ("scripts/harvest_structural_negatives.py",),
              (bench / "structural_negatives.parquet",), True),
    )


def benchmark_v3_evaluation_stages() -> tuple[Stage, ...]:
    """Evaluate current benchmark splits and refresh reader-facing artifacts."""
    bench = PROCESSED / "benchmark_v3"
    return (
        Stage(
            "benchmark_evaluate_dev_primary",
            ("scripts/evaluate_linkage.py", "--benchmark", "v3", "--split", "dev", "--event-set", "primary"),
            (PROCESSED / "linkage_evaluation_summary_v3_dev_primary.json",),
        ),
        Stage(
            "benchmark_evaluate_validation_primary",
            ("scripts/evaluate_linkage.py", "--benchmark", "v3", "--split", "validation", "--event-set", "primary"),
            (PROCESSED / "linkage_evaluation_summary_v3_validation_primary.json",),
        ),
        Stage(
            "benchmark_modeling_tables",
            ("scripts/build_benchmark_modeling_dataset.py",),
            (
                bench / "modeling" / "benchmark_v3_modeling_dev.parquet",
                bench / "modeling" / "benchmark_v3_modeling_validation.parquet",
                bench / "modeling" / "benchmark_v3_modeling_summary.json",
            ),
            True,
        ),
        Stage(
            "reader_artifact_refresh",
            ("scripts/refresh_reader_artifacts.py",),
            (
                Path("FINAL_PIPELINE.md"),
                Path("NATIONAL_BENCHMARK_REFERENCE.md"),
                Path("notebooks/12_successor_linkage_and_evaluation.ipynb"),
                Path("reports/boamp_methodology_chapter.tex"),
                Path("reports/figures/benchmark_v3_validation_method_metrics.png"),
            ),
        ),
        Stage(
            "benchmark_quality_evidence",
            ("scripts/build_quality_evidence.py",),
            (
                PROCESSED / "quality_evidence" / "benchmark_v3_quality_evidence_summary.json",
                PROCESSED / "quality_evidence" / "benchmark_v3_validation_anchor_confusion.csv",
                PROCESSED / "quality_evidence" / "benchmark_v3_validation_pair_curve_metrics.csv",
                Path("QUALITY_EVIDENCE.md"),
                Path("reports/figures/benchmark_v3_validation_confusion_matrices.png"),
                Path("reports/figures/benchmark_v3_validation_pair_roc.png"),
                Path("reports/figures/benchmark_v3_validation_pair_precision_recall.png"),
            ),
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
        ),
        Stage(
            "independent_link_review_queue",
            ("scripts/prepare_independent_link_review.py",),
            (
                Path("data/review/independent_link_review_sample.csv"),
                Path("data/review/independent_link_review_audit_key.csv"),
                Path("data/review/independent_link_review_summary.json"),
                Path("INDEPENDENT_LINK_REVIEW_PROTOCOL.md"),
            ),
        ),
        Stage(
            "readiness_report_data",
            ("scripts/build_readiness_report.py",),
            (
                Path("reports/current_project_readiness.db"),
                Path("reports/current_project_readiness_artifact.json"),
            ),
        ),
    )


def pipeline_stages() -> tuple[Stage, ...]:
    benchmark = PROCESSED / "benchmark_remap"
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
            "benchmark_remap",
            ("scripts/remap_benchmark_to_v2_episodes.py",),
            (
                benchmark / "reference_120_v2_remap.csv",
                benchmark / "confirmed_successor_links_v2_remap.csv",
                benchmark / "evaluation_subset_v2_remap.csv",
                benchmark / "benchmark_v2_remap_summary.json",
            ),
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
        Stage(
            "primary_linkage",
            ("scripts/evaluate_linkage.py",),
            (
                PROCESSED / "accepted_successor_links.parquet",
                PROCESSED / "linkage_frozen_config.json",
                PROCESSED / "linkage_evaluation_summary.json",
            ),
        ),
        Stage(
            "primary_survival_dataset",
            ("scripts/build_survival_dataset.py",),
            (PROCESSED / "survival_dataset.parquet", PROCESSED / "survival_dataset_summary.json"),
            True,
        ),
        Stage(
            "expiry_sensitivity_audit",
            ("scripts/evaluate_expiry_aware_linkage.py",),
            (
                PROCESSED / "accepted_successor_links_expiry_aware.parquet",
                PROCESSED / "expiry_aware_linkage_summary.json",
                PROCESSED / "expiry_link_review.csv",
            ),
            True,
        ),
        Stage(
            "expiry_sensitivity_survival_dataset",
            ("scripts/build_expiry_aware_survival_dataset.py",),
            (
                PROCESSED / "survival_dataset_expiry_aware.parquet",
                PROCESSED / "survival_dataset_expiry_aware_summary.json",
            ),
            True,
        ),
    )


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
        "--ExecutePreprocessor.timeout=0",
    ]
    run_command(command, dry_run)


def write_manifest(statuses: dict[str, str]) -> Path:
    path = PROJECT_ROOT / PROCESSED / "final_pipeline_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workflow": "current_defensible_pipeline",
        "primary_method": "M_B_text_ranking @ 0.70",
        "primary_links": str(PROCESSED / "accepted_successor_links.parquet"),
        "primary_survival_dataset": str(PROCESSED / "survival_dataset.parquet"),
        "sensitivity_method": "M_E_expiry_aware_text",
        "sensitivity_role": "audit only; not promoted to the primary event definition",
        "manual_review": str(PROCESSED / "expiry_link_review.csv"),
        "evaluation_status": (
            "internal deterministic-bootstrap development reference; "
            "independent specialist review pending"
        ),
        "independent_review_sample": str(
            PROJECT_ROOT / "data/review/independent_link_review_sample.csv"
        ),
        "independent_review_protocol": str(
            PROJECT_ROOT / "INDEPENDENT_LINK_REVIEW_PROTOCOL.md"
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
    parser.add_argument(
        "--with-current-benchmark", "--with-benchmark-v3", action="store_true",
        dest="with_benchmark_v3",
        help="Build the current national benchmark. Annotation is a separate, manual step.",
    )
    parser.add_argument(
        "--with-current-benchmark-evaluation", "--with-benchmark-v3-evaluation", action="store_true",
        dest="with_benchmark_v3_evaluation",
        help="Evaluate labelled current benchmark dev/validation splits and refresh notebooks/reports.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print work without executing commands.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    statuses: dict[str, str] = {}
    for stage in pipeline_stages():
        statuses[stage.name] = run_stage(stage, args.force, args.dry_run)

    if args.with_benchmark_v3:
        for stage in benchmark_v3_stages():
            statuses[stage.name] = run_stage(stage, args.force, args.dry_run)

    if args.with_benchmark_v3_evaluation:
        for stage in benchmark_v3_evaluation_stages():
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
    print("Expiry-aware result: sensitivity audit only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
