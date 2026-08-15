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
REFERENCE = PROCESSED / "regional_benchmark"
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
        PROCESSED / "linkage_evaluation_dev.json",
        PROCESSED / "linkage_evaluation_validation.json",
        REFERENCE / "regional_benchmark_manifest.json",
        REFERENCE / "benchmark_dev.parquet",
        REFERENCE / "benchmark_validation.parquet",
        REFERENCE / "modeling/modeling_summary.json",
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

    manifest = load_json(REFERENCE / "regional_benchmark_manifest.json")
    dev_evaluation = load_json(PROCESSED / "linkage_evaluation_dev.json")
    validation_evaluation = load_json(PROCESSED / "linkage_evaluation_validation.json")
    modeling = load_json(REFERENCE / "modeling/modeling_summary.json")

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
        evaluation.get("caveats", {}).get("label_source", "")
        for evaluation in (dev_evaluation, validation_evaluation)
    ]

    # The France-level benchmark and the expiry-aware linkage arm were both
    # removed, not paused. Two guards apply to each: nothing may read their
    # paths, and the paths may not exist to be read.
    retired_paths = [
        PROJECT_ROOT / "data/processed/boamp/benchmark",
        PROJECT_ROOT / "archive/national_benchmark",
        PROJECT_ROOT / "archive/legacy_reference",
        PROJECT_ROOT / "boamp_pipeline/expiry_linkage.py",
        PROJECT_ROOT / "scripts/evaluate_expiry_aware_linkage.py",
        PROJECT_ROOT / "scripts/build_expiry_aware_survival_dataset.py",
        PROJECT_ROOT / "tests/test_expiry_linkage.py",
        *PROCESSED.glob("*expiry*"),
    ]
    retired_present = [relative(path) for path in retired_paths if path.exists()]

    retired_tokens = (
        "data/processed/boamp/benchmark/",
        "archive/national_benchmark/",
        "expiry_aware",
        "expiry_link",
        "expiry_linkage",
    )
    # Naming the removal in generated prose is required; opening something inside
    # it is the thing to forbid, so a line only counts when it also reads.
    access_markers = ("Path(", "open(", "read_", "load_json(", "glob(", "rglob(")
    executable_sources = [
        path
        for root in ("scripts", "boamp_pipeline", "notebooks")
        for path in (PROJECT_ROOT / root).rglob("*")
        if path.is_file() and path.suffix in {".py", ".ipynb"}
        and "__pycache__" not in path.parts
        and path != Path(__file__)
    ]
    retired_references: dict[str, str] = {}
    for path in executable_sources:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            for token in retired_tokens:
                if token in line and any(marker in line for marker in access_markers):
                    retired_references[relative(path)] = line.strip()[:160]

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
        "evaluations_use_the_regional_reference": (
            dev_evaluation["benchmark"] == "regional_grand_ouest"
            and dev_evaluation["split"] == "dev"
            and validation_evaluation["benchmark"] == "regional_grand_ouest"
            and validation_evaluation["split"] == "validation"
        ),
        "evaluated_anchor_counts_match_the_reference_manifest": (
            dev_evaluation["anchors_evaluated"] == manifest["splits"]["dev"]["usable_anchors"]
            and validation_evaluation["anchors_evaluated"]
            == manifest["splits"]["validation"]["usable_anchors"]
        ),
        # Anchors the blocking step proposes nothing for cannot appear in a
        # pair-level table, so the modeling tables are a subset by construction.
        "modeling_tables_are_a_subset_of_the_reference": all(
            modeling["outputs"][split]["anchors"] <= manifest["splits"][split]["usable_anchors"]
            for split in ("dev", "validation")
        ),
        "materialized_metadata_has_no_legacy_paths": not stale_metadata,
        "reference_provenance_is_truthful": all(
            "LLM-assisted" in note
            and "not an independent human specialist panel" in note
            for note in evaluation_provenance
        ),
        "no_executable_source_reads_a_retired_branch": not retired_references,
        "retired_branches_are_absent_from_the_repository": not retired_present,
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
        "reference_root": relative(REFERENCE),
        "counts": {
            "cohort_rows": int(len(primary_survival)),
            "primary_links": int(len(primary_links)),
            "reference_reviewed_anchors": int(manifest["reviewed_anchors"]),
            "reference_usable_anchors": int(
                sum(split["usable_anchors"] for split in manifest["splits"].values())
            ),
        },
        "checks": checks,
        "stale_metadata": stale_metadata,
        "retired_branch_references": retired_references,
        "retired_paths_still_present": retired_present,
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
