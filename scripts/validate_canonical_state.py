#!/usr/bin/env python3
"""Validate that all final BOAMP artifacts belong to one consistent state."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data/processed/boamp"
BENCHMARK = PROCESSED / "benchmark"
OUTPUT = PROCESSED / "canonical_state_validation.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def build() -> dict[str, Any]:
    required = [
        PROCESSED / "accepted_successor_links.parquet",
        PROCESSED / "linkage_application_summary.json",
        PROCESSED / "linkage_config.json",
        PROCESSED / "survival_dataset.parquet",
        PROCESSED / "survival_dataset_summary.json",
        PROCESSED / "survival_analysis_summary.json",
        PROCESSED / "accepted_successor_links_expiry_aware.parquet",
        PROCESSED / "expiry_aware_linkage_summary.json",
        PROCESSED / "expiry_link_review.csv",
        PROCESSED / "survival_dataset_expiry_aware.parquet",
        PROCESSED / "survival_dataset_expiry_aware_summary.json",
        PROCESSED / "linkage_evaluation_dev.json",
        PROCESSED / "linkage_evaluation_validation.json",
        BENCHMARK / "benchmark_manifest.json",
        BENCHMARK / "benchmark_dev.parquet",
        BENCHMARK / "benchmark_validation.parquet",
        BENCHMARK / "modeling/modeling_summary.json",
        PROJECT_ROOT / "FINAL_PIPELINE.md",
        PROJECT_ROOT / "SURVIVAL_ANALYSIS_REPORT.md",
        PROJECT_ROOT / "reports/boamp_methodology_chapter.pdf",
    ]
    missing = [relative(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"canonical outputs are missing: {missing}")

    primary_links = pd.read_parquet(PROCESSED / "accepted_successor_links.parquet")
    primary_survival = pd.read_parquet(PROCESSED / "survival_dataset.parquet")
    application = load_json(PROCESSED / "linkage_application_summary.json")
    survival_summary = load_json(PROCESSED / "survival_dataset_summary.json")

    expiry_links = pd.read_parquet(PROCESSED / "accepted_successor_links_expiry_aware.parquet")
    expiry_survival = pd.read_parquet(PROCESSED / "survival_dataset_expiry_aware.parquet")
    expiry_review = pd.read_csv(PROCESSED / "expiry_link_review.csv")
    expiry_summary = load_json(PROCESSED / "expiry_aware_linkage_summary.json")
    expiry_survival_summary = load_json(
        PROCESSED / "survival_dataset_expiry_aware_summary.json"
    )

    manifest = load_json(BENCHMARK / "benchmark_manifest.json")
    dev_evaluation = load_json(PROCESSED / "linkage_evaluation_dev.json")
    validation_evaluation = load_json(PROCESSED / "linkage_evaluation_validation.json")
    modeling = load_json(BENCHMARK / "modeling/modeling_summary.json")

    metadata_files = [path for path in PROCESSED.rglob("*.json") if path != OUTPUT]
    # Version identifiers inside schemas are data-contract labels, not paths.
    # Only legacy directory references make an artifact non-canonical.
    legacy_tokens = (
        "data/processed/" + "boamp_v2",
        "/benchmark_" + "v3/",
    )
    stale_metadata = {
        relative(path): token
        for path in metadata_files
        for token in legacy_tokens
        if token in path.read_text(encoding="utf-8", errors="ignore")
    }

    evaluation_provenance = [
        evaluation.get("caveats", {}).get("annotation_source", "")
        for evaluation in (dev_evaluation, validation_evaluation)
    ]

    main_summary = survival_summary["variants"]["main"]
    checks = {
        "primary_link_count_matches_application": (
            len(primary_links) == application["cohort_application"]["accepted_links"]
        ),
        "primary_link_count_matches_survival_events": (
            len(primary_links) == int(primary_survival["event"].sum())
            == main_summary["validation"]["events"]
        ),
        "primary_links_are_unique_and_not_self_links": bool(
            primary_links["anchor_episode_id"].is_unique
            and primary_links["anchor_episode_id"]
            .ne(primary_links["candidate_episode_id"])
            .all()
        ),
        "expiry_link_count_matches_survival_events": (
            len(expiry_links) == int(expiry_survival["event"].sum())
            == expiry_survival_summary["validation"]["events"]
            == expiry_summary["cohort_comparison"]["expiry_aware"]["accepted_links"]
        ),
        "expiry_review_count_matches_changed_anchors": (
            len(expiry_review)
            == expiry_summary["cohort_comparison"]["changed_anchors_for_review"]
        ),
        "benchmark_evaluations_use_national_reference": (
            dev_evaluation["benchmark"] == "national"
            and dev_evaluation["split"] == "dev"
            and validation_evaluation["benchmark"] == "national"
            and validation_evaluation["split"] == "validation"
        ),
        "benchmark_counts_match_modeling_tables": (
            manifest["outputs"]["dev"]["anchors"]
            == modeling["outputs"]["dev"]["anchors"]
            and manifest["outputs"]["validation"]["anchors"]
            == modeling["outputs"]["validation"]["anchors"]
        ),
        "benchmark_paths_are_canonical": all(
            "/data/processed/boamp/benchmark/" in payload["file"]
            for payload in manifest["outputs"].values()
        ),
        "materialized_metadata_has_no_legacy_paths": not stale_metadata,
        "benchmark_provenance_is_truthful": all(
            "deterministic bootstrap rules" in note
            and "independent" in note
            and "language model" not in note
            for note in evaluation_provenance
        ),
        "no_legacy_processed_tree": not (
            PROJECT_ROOT / "data/processed" / ("boamp_" + "v2")
        ).exists(),
        "no_legacy_benchmark_" + "remap": not (
            PROCESSED / ("benchmark_" + "remap")
        ).exists(),
    }

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workflow": "canonical_boamp_pipeline",
        "processed_root": relative(PROCESSED),
        "benchmark_root": relative(BENCHMARK),
        "counts": {
            "cohort_rows": int(len(primary_survival)),
            "primary_links": int(len(primary_links)),
            "expiry_links": int(len(expiry_links)),
            "expiry_changed_anchors": int(len(expiry_review)),
            "benchmark_anchors": int(manifest["anchor_totals"]["anchors"]),
            "benchmark_labelled_pairs": int(manifest["anchor_totals"]["labelled_pairs"]),
        },
        "checks": checks,
        "stale_metadata": stale_metadata,
        "validation_passed": all(checks.values()),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["validation_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"canonical-state validation failed: {failed}")
    return result


def main() -> int:
    print(json.dumps(build(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
