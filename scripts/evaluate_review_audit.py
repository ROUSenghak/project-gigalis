#!/usr/bin/env python3
"""Evaluate the completed blinded linkage challenge without reusing it for tuning."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import binomtest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / "data/review"
BLANK_PATH = REVIEW_DIR / "independent_link_review_sample.csv"
REVIEWED_PATH = REVIEW_DIR / "independent_link_review_sample_reviewed.csv"
AUDIT_PATH = REVIEW_DIR / "independent_link_review_audit_key.csv"
PROVENANCE_PATH = REVIEW_DIR / "review_provenance.json"
SUMMARY_PATH = REVIEW_DIR / "review_audit_evaluation.json"
FINDINGS_PATH = REVIEW_DIR / "review_audit_findings.csv"
REPORT_PATH = PROJECT_ROOT / "REVIEW_AUDIT_RESULTS.md"

LABEL_COLUMNS = [
    "same_legal_buyer_Y_N_UNCERTAIN",
    "relationship_label",
    "observable_successor_Y_N_UNCERTAIN",
    "confidence_HIGH_MEDIUM_LOW",
    "review_notes",
    "review_date",
]
ALLOWED = {
    "same_legal_buyer_Y_N_UNCERTAIN": {"Y", "N", "UNCERTAIN"},
    "relationship_label": {
        "RENEWAL_OF_EXPIRING",
        "NEXT_PHASE_OF_PROGRAMME",
        "RETENDER_AFTER_FAILURE",
        "PARALLEL_LOT",
        "EXTENSION_SAME_CONTRACT",
        "UNRELATED",
        "UNUSABLE",
    },
    "observable_successor_Y_N_UNCERTAIN": {"Y", "N", "UNCERTAIN"},
    "confidence_HIGH_MEDIUM_LOW": {"HIGH", "MEDIUM", "LOW"},
}
EVENT_BY_RELATIONSHIP = {
    "RENEWAL_OF_EXPIRING": "Y",
    "NEXT_PHASE_OF_PROGRAMME": "Y",
    "RETENDER_AFTER_FAILURE": "N",
    "PARALLEL_LOT": "N",
    "EXTENSION_SAME_CONTRACT": "N",
    "UNRELATED": "N",
    "UNUSABLE": "UNCERTAIN",
}


def exact_binomial(successes: int, trials: int) -> dict[str, Any]:
    if trials == 0:
        return {"successes": successes, "trials": trials, "estimate": None, "ci_95": None}
    interval = binomtest(successes, trials).proportion_ci(0.95, method="exact")
    return {
        "successes": int(successes),
        "trials": int(trials),
        "estimate": float(successes / trials),
        "ci_95": [float(interval.low), float(interval.high)],
        "interval_method": "Clopper-Pearson exact binomial",
    }


def validate_review(blank: pd.DataFrame, reviewed: pd.DataFrame) -> dict[str, Any]:
    if list(blank.columns) != list(reviewed.columns):
        raise ValueError("reviewed columns differ from the blinded template")
    if len(blank) != 60 or len(reviewed) != 60:
        raise ValueError("the audit requires the original 60 rows")
    if not reviewed["review_id"].is_unique:
        raise ValueError("review_id must be unique")
    if blank["review_id"].tolist() != reviewed["review_id"].tolist():
        raise ValueError("review rows were added, removed, or reordered")

    evidence_columns = [column for column in blank.columns if column not in LABEL_COLUMNS]
    changed = [column for column in evidence_columns if not blank[column].equals(reviewed[column])]
    if changed:
        raise ValueError(f"evidence columns changed: {changed}")
    for column in LABEL_COLUMNS:
        if reviewed[column].str.strip().eq("").any():
            raise ValueError(f"{column} contains blank values")
    for column, allowed in ALLOWED.items():
        invalid = sorted(set(reviewed[column]) - allowed)
        if invalid:
            raise ValueError(f"{column} contains invalid values: {invalid}")
    expected = reviewed["relationship_label"].map(EVENT_BY_RELATIONSHIP)
    inconsistent = reviewed.loc[
        expected.ne(reviewed["observable_successor_Y_N_UNCERTAIN"]), "review_id"
    ].tolist()
    if inconsistent:
        raise ValueError(f"relationship and event labels disagree: {inconsistent}")
    if pd.to_datetime(reviewed["review_date"], format="%Y-%m-%d", errors="coerce").isna().any():
        raise ValueError("review_date must use YYYY-MM-DD")
    return {
        "rows": 60,
        "evidence_columns_unchanged": True,
        "labels_complete_and_valid": True,
        "relationship_event_consistency": True,
    }


def event_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["observable_successor_Y_N_UNCERTAIN"].value_counts()
    return {label: int(counts.get(label, 0)) for label in ["Y", "N", "UNCERTAIN"]}


def evaluate(joined: pd.DataFrame, provenance: dict[str, Any]) -> dict[str, Any]:
    accepted = joined.loc[joined["source_stratum"].eq("PRIMARY_ACCEPTED")]
    structural = joined.loc[
        joined["source_stratum"].eq("HIGH_SIMILARITY_STRUCTURAL_NEGATIVE")
    ]
    declared = joined.loc[joined["source_stratum"].eq("BUYER_DECLARED_RESOLVED")]
    accepted_counts = event_counts(accepted)
    structural_counts = event_counts(structural)
    declared_counts = event_counts(declared)

    determinate_trials = accepted_counts["Y"] + accepted_counts["N"]
    structural_trials = structural_counts["Y"] + structural_counts["N"]
    declared_trials = declared_counts["Y"] + declared_counts["N"]
    conservative = exact_binomial(accepted_counts["Y"], len(accepted))
    point_target_met = bool(conservative["estimate"] >= 0.80)

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "model_assisted_diagnostic_complete_human_validation_pending",
        "provenance": provenance,
        "primary_accepted_links": {
            "rows": len(accepted),
            "review_counts": accepted_counts,
            "precision_excluding_uncertain": exact_binomial(
                accepted_counts["Y"], determinate_trials
            ),
            "precision_conservative": conservative,
            "target": 0.80,
            "point_estimate_target_met": point_target_met,
        },
        "structural_negative_challenge": {
            "rows": len(structural),
            "review_counts": structural_counts,
            "reviewed_negative_share": exact_binomial(structural_counts["N"], structural_trials),
            "contradiction_share": exact_binomial(structural_counts["Y"], structural_trials),
            "use": "failure-mode diagnostic only; not ground truth or a national FPR denominator",
        },
        "buyer_declared_challenge": {
            "rows": len(declared),
            "review_counts": declared_counts,
            "reviewed_successor_yield": exact_binomial(declared_counts["Y"], declared_trials),
            "use": "positive-case recruitment diagnostic only; not a recall denominator",
        },
        "decision": (
            "Keep M_B @ 0.70 frozen and provisional. The diagnostic accepted-link "
            "precision does not meet the 0.80 point target. Because these are not "
            "independent human labels, obtain human specialist review before "
            "recalibration or external accuracy claims."
        ),
    }


def write_findings(joined: pd.DataFrame) -> pd.DataFrame:
    findings = joined.loc[
        (
            joined["source_stratum"].eq("PRIMARY_ACCEPTED")
            & joined["observable_successor_Y_N_UNCERTAIN"].ne("Y")
        )
        | (
            joined["source_stratum"].eq("HIGH_SIMILARITY_STRUCTURAL_NEGATIVE")
            & joined["observable_successor_Y_N_UNCERTAIN"].ne("N")
        )
    ].copy()
    columns = [
        "review_id",
        "source_stratum",
        "anchor_episode_id",
        "candidate_episode_id",
        "anchor_buyer_name_raw",
        "candidate_buyer_name_raw",
        "gap_days",
        "relationship_label",
        "observable_successor_Y_N_UNCERTAIN",
        "confidence_HIGH_MEDIUM_LOW",
        "review_notes",
    ]
    findings[columns].to_csv(FINDINGS_PATH, index=False)
    return findings[columns]


def write_report(summary: dict[str, Any], finding_count: int) -> None:
    accepted = summary["primary_accepted_links"]
    determinate = accepted["precision_excluding_uncertain"]
    conservative = accepted["precision_conservative"]
    structural = summary["structural_negative_challenge"]
    declared = summary["buyer_declared_challenge"]
    text = f"""# Linkage Challenge Review Results

Generated: `{summary['created_at']}`

Status: **model-assisted diagnostic complete; independent human validation pending**

## Accepted-Link Diagnostic

Among 20 sampled links accepted by `M_B @ 0.70`:

- `Y`: `{accepted['review_counts']['Y']}`;
- `N`: `{accepted['review_counts']['N']}`;
- `UNCERTAIN`: `{accepted['review_counts']['UNCERTAIN']}`.

Precision excluding the uncertain row is `{determinate['estimate']:.4f}`
(`{determinate['successes']}/{determinate['trials']}`), with exact 95% CI
`[{determinate['ci_95'][0]:.4f}, {determinate['ci_95'][1]:.4f}]`.

Treating uncertainty conservatively as not confirmed gives precision
`{conservative['estimate']:.4f}` (`{conservative['successes']}/{conservative['trials']}`),
with exact 95% CI `[{conservative['ci_95'][0]:.4f}, {conservative['ci_95'][1]:.4f}]`.

The diagnostic point estimate does not meet the predefined `0.80` target.

## Challenge Diagnostics

- Structural-negative challenges: `{structural['review_counts']['N']}` reviewed
  `N`, `{structural['review_counts']['Y']}` reviewed `Y`, and
  `{structural['review_counts']['UNCERTAIN']}` uncertain.
- Buyer-declared relationships: `{declared['review_counts']['Y']}` reviewed `Y`,
  `{declared['review_counts']['N']}` reviewed `N`, and
  `{declared['review_counts']['UNCERTAIN']}` uncertain.

The structural contradiction demonstrates that rule-generated structural
negatives are challenge cases, not verified ground truth. Buyer declarations are
also recruitment evidence rather than automatic successor labels. Neither
stratum estimates national recall, prevalence, or false-positive rate.

## Findings Queue

`{finding_count}` accepted-link non-confirmations or structural contradictions
are listed in `data/review/review_audit_findings.csv`.

## Decision

Keep `M_B @ 0.70` frozen and provisional. These labels are documented as a
model-assisted diagnostic, so they cannot complete the independent validation
gate. An independent human procurement-domain reviewer should assess the
original blinded file before recalibration or final accuracy claims.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    blank = pd.read_csv(BLANK_PATH, dtype=str, keep_default_na=False)
    reviewed = pd.read_csv(REVIEWED_PATH, dtype=str, keep_default_na=False)
    validation = validate_review(blank, reviewed)
    audit = pd.read_csv(AUDIT_PATH, dtype=str, keep_default_na=False)
    joined = reviewed.merge(audit, on="review_id", how="left", validate="one_to_one")
    if joined["source_stratum"].isna().any():
        raise ValueError("review rows do not fully map to the audit key")
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    summary = evaluate(joined, provenance)
    summary["input_validation"] = validation
    findings = write_findings(joined)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, len(findings))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
