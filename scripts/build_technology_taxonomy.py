#!/usr/bin/env python3
"""Run the technology taxonomy layer: corpus, models, deployment, evidence.

This script is orchestration only. Every line of analysis lives in
``boamp_pipeline/technology_taxonomy.py`` (the frozen classes, the annotated
corpus, the leakage-preventing grouping and the folds),
``boamp_pipeline/technology_models.py`` (the specifications, the nested grouped
cross-validation, the diagnostics and the selection rule) and
``boamp_pipeline/technology_evidence.py`` (deployment to the study cohort, the
composition and CPV crosswalk, the support-gated enrichment and the report).

That split exists so ``notebooks/15_technology_taxonomy_classification.ipynb``
can import the same functions and *show* the method rather than describe it. A
number in the notebook and the matching number in
``TECHNOLOGY_TAXONOMY_REPORT.md`` come from one code path, not two.

The four stages are ordered. ``--stage`` runs one of them when iterating; the
default runs all four, which is what the pipeline does.

Nothing here writes outside ``data/processed/boamp/technology/``, the report,
and the technology figures. The linkage, survival, and trend results are read
where needed and never modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This runner is headless; the evidence notebook imports the same library code
# under the inline backend. The choice belongs here, not in the library.
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from boamp_pipeline.technology_evidence import build_evidence, build_predictions  # noqa: E402
from boamp_pipeline.technology_models import build_evaluation_artifacts  # noqa: E402
from boamp_pipeline.technology_taxonomy import build_corpus_artifacts  # noqa: E402

STAGES = {
    "corpus": build_corpus_artifacts,
    "models": build_evaluation_artifacts,
    "predictions": build_predictions,
    "evidence": build_evidence,
}


def configure_logging() -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "build_technology_taxonomy.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["all", *STAGES],
        default="all",
        help="Run one stage instead of the whole layer. Stages are ordered.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs. Required to rebuild in place.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress the per-stage JSON summary."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    selected = list(STAGES) if args.stage == "all" else [args.stage]
    summaries = {}
    for name in selected:
        logging.info("technology stage: %s", name)
        summaries[name] = STAGES[name](args.force)
    if not args.quiet:
        print(json.dumps(summaries, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
