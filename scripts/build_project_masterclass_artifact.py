#!/usr/bin/env python3
"""Package the BOAMP teaching guide into the canonical report artifact shape."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "boamp_project_masterclass.md"
OUTPUT = ROOT / "reports" / "boamp_project_masterclass_artifact.json"
SURVIVAL_DATASET_SUMMARY = ROOT / "data" / "processed" / "boamp" / "survival_dataset_summary.json"

ARM_LABELS = {
    "strict": "M_B at 0.80",
    "main": "M_B at 0.70",
    "looser": "M_B at 0.60",
    "contrast_high_recall": "M_C at 0.70",
}


SOURCES = [
    {
        "id": "protocol",
        "label": "Current formal project protocol and source-of-truth register",
        "path": "PROJECT_WORK_PROTOCOL.md",
        "query": {
            "engine": "Local repository",
            "language": "markdown",
            "description": "Current canonical research definitions, workflow, results, permitted claims, and source hierarchy.",
            "tables_used": [
                "PROJECT_WORK_PROTOCOL.md",
                "FINAL_PIPELINE.md",
                "data/processed/boamp/canonical_state_validation.json",
            ],
            "filters": ["Active canonical workflow only; archived and retired methods excluded"],
            "metric_definitions": [
                "Observable-successor event means one candidate accepted under the frozen linkage rule, not legal renewal.",
                "Every percentage in the guide names or implies its project denominator.",
            ],
        },
    },
    {
        "id": "quality",
        "label": "Current data-quality profile",
        "path": "data/processed/boamp/data_quality_profile.json",
        "query": {
            "engine": "Local repository",
            "language": "json",
            "description": "Materialised grain, volume, integrity, missingness, cohort-scope, and detectability evidence.",
            "tables_used": [
                "data/processed/boamp/data_quality_profile.json",
                "DATA_QUALITY_REPORT.md",
            ],
            "filters": ["BOAMP notices through 31 December 2025"],
            "metric_definitions": [
                "Cohort missingness rates use 3,800 awarded Grand Ouest any-digital-CPV episodes.",
                "Candidate coverage is anchors with at least one candidate divided by 3,800 cohort anchors.",
            ],
        },
    },
    {
        "id": "evaluation",
        "label": "Locked regional linkage-reference evaluation",
        "path": "data/processed/boamp/linkage_evaluation_validation.json",
        "query": {
            "engine": "Local repository",
            "language": "json",
            "description": "Exact-successor linkage metrics on the locked Grand Ouest regional reference split.",
            "tables_used": [
                "data/processed/boamp/linkage_evaluation_validation.json",
                "REGIONAL_BENCHMARK_REFERENCE.md",
                "QUALITY_EVIDENCE.md",
            ],
            "filters": ["72 usable locked anchors; 18 reviewed positives; primary event set"],
            "metric_definitions": [
                "Precision = exact correct accepted successors divided by all accepted successors.",
                "Recall = exact correct successors divided by reviewed-positive anchors.",
                "Negative-anchor FPR excludes wrong-successor errors occurring on positive anchors.",
            ],
        },
    },
    {
        "id": "survival",
        "label": "Current survival analysis summary",
        "path": "data/processed/boamp/survival_dataset_summary.json",
        "query": {
            "engine": "Local repository",
            "language": "json",
            "description": "Kaplan–Meier, Cox, temporal validation, parametric comparison, detectability, and linkage sensitivities.",
            "tables_used": [
                "data/processed/boamp/survival_dataset_summary.json",
                "data/processed/boamp/survival_analysis_summary.json",
                "data/processed/boamp/survival_cox_results.csv",
                "data/processed/boamp/survival_conditional_probabilities.csv",
                "SURVIVAL_ANALYSIS_REPORT.md",
            ],
            "filters": ["3,800 cohort episodes; M_B text ranking at 0.70; cutoff 31 December 2025"],
            "metric_definitions": [
                "Event rate = 544 accepted observable successors divided by 3,800 episodes.",
                "Kaplan–Meier probabilities are linkage-conditioned and incorporate right censoring.",
            ],
        },
    },
    {
        "id": "trend",
        "label": "Current quarterly trend evidence",
        "path": "data/processed/boamp/trend_analysis_summary.json",
        "query": {
            "engine": "Local repository",
            "language": "json/csv",
            "description": "Quarterly episode counts, 12-quarter OLS slopes, PELT, stationarity tests, HMM regimes, and multiplicity adjustments.",
            "tables_used": [
                "data/processed/boamp/trend_analysis_summary.json",
                "data/processed/boamp/trend_signal_matrix.csv",
                "TREND_ANALYSIS_REPORT.md",
            ],
            "filters": ["CPV trend window 2015Q2–2025Q4; five simultaneous series"],
            "metric_definitions": ["Slope is change in awarded episode count per quarter over the latest 12 quarters."],
        },
    },
    {
        "id": "technology",
        "label": "Current technology-taxonomy evidence",
        "path": "data/processed/boamp/technology/technology_evidence_summary.json",
        "query": {
            "engine": "Local repository",
            "language": "json/csv",
            "description": "Annotation audit, grouped validation, family bootstrap, calibration decision, deployment, and downstream gates.",
            "tables_used": [
                "data/processed/boamp/technology/model_selection_decision.json",
                "data/processed/boamp/technology/technology_evidence_summary.json",
                "data/processed/boamp/technology/per_class_metrics.csv",
                "TECHNOLOGY_TAXONOMY_REPORT.md",
            ],
            "filters": ["500 labelled notices; 459 families; three grouped folds; 1,000 family-bootstrap draws"],
            "metric_definitions": [
                "Macro-F1 is the unweighted mean of 11 class-specific F1 values.",
                "High confidence uses the uncalibrated raw class score at or above 0.70.",
            ],
        },
    },
]


def split_blocks(markdown: str) -> list[dict[str, str]]:
    parts = re.split(r"(?=^## )", markdown, flags=re.MULTILINE)
    blocks = []
    for index, raw in enumerate(parts):
        body = raw.strip()
        if not body:
            continue
        source_match = re.search(r"\n<!-- source:([a-z-]+) -->\n", body)
        block: dict[str, str] = {
            "id": "title" if index == 0 else f"section-{len(blocks):02d}",
            "type": "markdown",
            "body": re.sub(r"\n<!-- source:[a-z-]+ -->\n", "\n", body, count=1),
        }
        if source_match:
            block["sourceId"] = source_match.group(1)
        blocks.append(block)
        if block["body"].startswith("## 11. Primary top-rank rule"):
            blocks.append(
                {
                    "id": "linkage-sensitivity-chart-block",
                    "type": "chart",
                    "chartId": "linkage-sensitivity-chart",
                }
            )
    return blocks


def main() -> int:
    markdown = INPUT.read_text(encoding="utf-8")
    survival_variants = json.loads(
        SURVIVAL_DATASET_SUMMARY.read_text(encoding="utf-8")
    )["variants"]
    linkage_sensitivity = [
        {
            "rule": ARM_LABELS[arm],
            "events": variant["validation"]["events"],
            "cohort": variant["validation"]["rows"],
            "event_rate": variant["description"]["event_rate"],
            "median_event_months": variant["description"][
                "median_time_to_successor_months"
            ],
        }
        for arm, variant in survival_variants.items()
    ]
    generated_at = datetime.now().isoformat(timespec="seconds")
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "BOAMP Observable-Successor Study: Complete Teaching and Defense Guide",
            "description": "An end-to-end, calculation-led explanation of the canonical BOAMP research pipeline, findings, assumptions, limitations, and oral defense.",
            "generatedAt": generated_at,
            "blocks": split_blocks(markdown),
            "cards": [],
            "charts": [
                {
                    "id": "linkage-sensitivity-chart",
                    "title": "Observed Event Rate by Linkage Rule",
                    "subtitle": "Same 3,800 cohort episodes; accepted observable-successor events under four retained definitions.",
                    "type": "horizontalBar",
                    "intent": "comparison",
                    "question": "How strongly does the observed event rate depend on the linkage definition?",
                    "rationale": "Four named rules share one denominator, so sorted horizontal bars make the event-definition sensitivity directly comparable.",
                    "dataset": "linkage_sensitivity",
                    "sourceId": "survival",
                    "encodings": {
                        "x": {"field": "rule", "type": "nominal", "label": "Linkage rule"},
                        "y": {"field": "event_rate", "type": "quantitative", "format": "percent", "label": "Observed event rate"},
                        "tooltip": [
                            {"field": "events", "type": "quantitative", "label": "Accepted events"},
                            {"field": "cohort", "type": "quantitative", "label": "Cohort denominator"},
                            {"field": "median_event_months", "type": "quantitative", "label": "Median among events, months"},
                        ],
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                }
            ],
            "tables": [],
            "sources": SOURCES,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "linkage_sensitivity": linkage_sensitivity
            },
        },
        "sources": SOURCES,
    }
    OUTPUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
