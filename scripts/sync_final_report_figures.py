#!/usr/bin/env python3
"""Synchronise canonical generated figures into the final report package.

The internship report deliberately uses reader-facing filenames. This script
records the mapping from those names to the active pipeline outputs so a report
rebuild cannot silently retain an older plot after the evidence pipeline runs.
Report-specific composites and crops are not overwritten here.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "figures"
DESTINATION = ROOT / "rapport" / "BOAMP_Report_EN_Overleaf" / "figures"

FIGURE_MAP = {
    "data_quality_key_missingness.png": "appB_missingness.png",
    "benchmark_validation_confusion_matrices.png": "appC_confusion_matrices.png",
    "benchmark_validation_pair_precision_recall.png": "appC_pair_pr.png",
    "benchmark_validation_pair_roc.png": "appC_pair_roc.png",
    "technology_learning_curve.png": "appE_learning_curve.png",
    "technology_confusion_matrix.png": "appE_tech_confusion.png",
    "technology_kaplan_meier.png": "appE_technology_km.png",
    "trend_duration_completeness.png": "appF_duration_completeness.png",
    "trend_quarterly_episode_counts.png": "appF_quarterly_counts.png",
    "survival_kaplan_meier.png": "fig02_km_overall_and_segment.png",
    "survival_conditional_probabilities.png": "fig03_conditional_probabilities.png",
    "benchmark_validation_m_b_threshold_tradeoff.png": "fig05_threshold_tradeoff.png",
    "technology_composition.png": "fig06_technology_composition.png",
}


def main() -> int:
    missing = [name for name in FIGURE_MAP if not (SOURCE / name).is_file()]
    if missing:
        raise FileNotFoundError(f"canonical report figures are missing: {missing}")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for source_name, destination_name in FIGURE_MAP.items():
        destination = DESTINATION / destination_name
        if destination.exists():
            destination.chmod(0o644)
        shutil.copy2(SOURCE / source_name, destination)
        print(f"{source_name} -> {destination_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
