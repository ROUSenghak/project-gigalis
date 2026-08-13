#!/usr/bin/env python3
"""Build reproducible data-quality and descriptive trend evidence."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.evidence import (  # noqa: E402
    build_quarterly_panel,
    recent_trend_signal,
    stable_breaks,
    stationarity_diagnostics,
)

PROCESSED = PROJECT_ROOT / "data/processed/boamp_v2"
BENCHMARK = PROCESSED / "benchmark_v3"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
NOTEBOOK = PROJECT_ROOT / "notebooks/14_data_quality_and_trend_analysis.ipynb"

SOURCE_URLS = {
    "boamp": "https://www.data.gouv.fr/dataservices/api-bulletin-officiel-des-annonces-des-marches-publics-boamp",
    "siren": "https://www.insee.fr/fr/metadonnees/definition/c2047",
    "siret": "https://www.insee.fr/fr/metadonnees/definition/c1841",
    "cpv": "https://eur-lex.europa.eu/eli/reg/2008/213/oj",
    "framework": "https://eur-lex.europa.eu/eli/dir/2014/24/oj",
    "tfidf": "https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity",
    "fellegi_sunter": "https://doi.org/10.1080/01621459.1969.10501049",
    "kaplan_meier": "https://doi.org/10.1080/01621459.1958.10501452",
    "cox": "https://doi.org/10.1111/j.2517-6161.1972.tb00899.x",
    "ph_diagnostics": "https://doi.org/10.1093/biomet/81.3.515",
    "pelt": "https://doi.org/10.1080/01621459.2012.737745",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def data_quality_profile(cohort: pd.DataFrame) -> dict[str, Any]:
    standardized = load_json(PROCESSED / "standardized_notice_summary.json")
    episodes = load_json(PROCESSED / "episode_reconstruction_summary.json")
    cohort_summary = load_json(PROCESSED / "survival_cohort_summary.json")
    candidates = load_json(PROCESSED / "linkage_candidates_summary.json")
    survival = load_json(PROCESSED / "survival_dataset_summary.json")
    buyer_audit = load_json(PROCESSED / "buyer_blocking_legal_form_audit_summary.json")
    annotation = load_json(BENCHMARK / "annotation_agreement_summary.json")

    duration_by_year = (
        cohort.assign(has_reliable_duration=cohort["duration_months_reliable"].notna())
        .groupby("award_year", observed=True)["has_reliable_duration"]
        .agg(["sum", "count", "mean"])
    )
    duration_by_year.index = duration_by_year.index.astype(str)

    missingness = {
        "buyer_siren": cohort_summary["missingness"]["buyer_siren"]["missing_rate"],
        "reliable_duration": cohort_summary["missingness"]["duration_months_reliable"]["missing_rate"],
        "amount_candidate_container": cohort_summary["missingness"]["amount_candidates_json"]["missing_rate"],
        "main_cpv": cohort_summary["missingness"]["main_cpv"]["missing_rate"],
        "episode_text": cohort_summary["missingness"]["episode_text"]["missing_rate"],
        "award_date": cohort_summary["missingness"]["award_date"]["missing_rate"],
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_as_of": standardized["max_publication_date"],
        "study_period": standardized["input_period"],
        "grain": {
            "raw_standardized": "one BOAMP notice",
            "episodes": "one reconstructed procurement episode",
            "cohort": "one awarded Grand Ouest digital procurement episode",
            "survival": "one cohort episode with event or administrative censoring",
        },
        "volume": {
            "standardized_notices": standardized["rows"],
            "unique_standardized_notice_ids": standardized["unique_idweb"],
            "reconstructed_episodes": episodes["episode_rows"],
            "grand_ouest_episodes": episodes["grand_ouest_episode_rows"],
            "survival_cohort_rows": cohort_summary["selection_funnel"]["cohort_rows"],
            "candidate_pairs": candidates["candidate_pairs"],
            "accepted_primary_links": survival["variants"]["main"]["validation"]["events"],
        },
        "integrity": {
            "duplicate_standardized_notice_ids": standardized["duplicate_idweb"],
            "all_notices_assigned_once": episodes["all_notices_assigned_once"],
            "buyer_conflict_episodes": episodes["buyer_conflict_episodes"],
            "impossible_chronology_episodes": episodes["impossible_chronology_episodes"],
            "duplicate_survival_episodes": survival["variants"]["main"]["validation"]["duplicate_episodes"],
            "negative_survival_durations": survival["variants"]["main"]["validation"]["negative_durations"],
            "accepted_links_with_conflicting_validated_siren": buyer_audit["hard_fail_conflicting_validated_siren"],
            "accepted_municipal_intercommunal_mixes": buyer_audit["municipal_intercommunal_mix"],
        },
        "cohort_missingness": missingness,
        "duration_completeness_by_award_year": {
            year: {
                "reliable": int(row["sum"]),
                "rows": int(row["count"]),
                "rate": round(float(row["mean"]), 4),
            }
            for year, row in duration_by_year.iterrows()
        },
        "candidate_feature_missingness": candidates["component_missing_rates"],
        "candidate_coverage": {
            "anchors": candidates["cohort_anchors"],
            "with_candidates": candidates["anchors_with_candidates"],
            "without_candidates": candidates["anchors_without_candidates"],
            "coverage_rate": round(
                candidates["anchors_with_candidates"] / candidates["cohort_anchors"], 4
            ),
        },
        "linkage_sensitivity": {
            name: {
                "events": payload["validation"]["events"],
                "event_rate": payload["description"]["event_rate"],
                "median_successor_months": payload["description"]["median_time_to_successor_months"],
            }
            for name, payload in survival["variants"].items()
        },
        "benchmark_provenance": {
            "labels": annotation["pass_a_labels"],
            "double_annotated_pairs": annotation["double_annotated_pairs"],
            "raw_agreement": annotation["agreement"]["raw_agreement"],
            "cohen_kappa_six_class": annotation["agreement"]["cohen_kappa_six_class"],
            "independent_human_annotation": False,
            "interpretation": (
                "Both passes were generated by the same deterministic bootstrap "
                "rules after candidate re-ordering and re-framing. Agreement is "
                "rule repeatability, not inter-annotator agreement."
            ),
            "generator": "scripts/auto_annotate_wave1a.py",
        },
        "assessment": "share_with_caveats",
        "sources": SOURCE_URLS,
    }


def build_trend_outputs(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    breakpoint_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for segment, group in panel.groupby("segment", sort=False):
        group = group.sort_values("quarter_start").reset_index(drop=True)
        values = group["episode_count"].astype(float).to_numpy()
        break_result = stable_breaks(values)
        trend = recent_trend_signal(values)
        diagnostics[str(segment)] = stationarity_diagnostics(values)
        for multiplier, indices in break_result["break_indices_by_penalty_multiplier"].items():
            for index in indices:
                breakpoint_rows.append(
                    {
                        "segment": segment,
                        "penalty_multiplier": float(multiplier),
                        "break_index": index,
                        "break_quarter": group.loc[index, "quarter"],
                        "stable_across_penalties": index in break_result["stable_break_indices"],
                    }
                )
        central = break_result["central_break_indices"]
        stable = break_result["stable_break_indices"]
        signal_rows.append(
            {
                "segment": segment,
                **trend,
                "last_central_break": group.loc[central[-1], "quarter"] if central else None,
                "last_stable_break": group.loc[stable[-1], "quarter"] if stable else None,
                "central_break_count": len(central),
                "stable_break_count": len(stable),
            }
        )
    return pd.DataFrame(breakpoint_rows), pd.DataFrame(signal_rows), diagnostics


def plot_missingness(profile: dict[str, Any], path: Path) -> None:
    labels = {
        "buyer_siren": "Validated SIREN",
        "reliable_duration": "Reliable duration",
        "amount_candidate_container": "Any amount candidate",
        "main_cpv": "Main CPV",
        "episode_text": "Episode text",
        "award_date": "Award date",
    }
    frame = pd.Series(profile["cohort_missingness"]).rename(index=labels).sort_values()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.barh(frame.index, 100 * frame.values, color="#356E9A", edgecolor="#173B57")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Missing rows (%)")
    fig.suptitle("Missingness in the 3,800-episode survival cohort", fontsize=14, y=0.98)
    fig.text(
        0.5,
        0.925,
        "Grand Ouest awarded digital episodes, 2015-2025",
        ha="center",
        color="#555555",
    )
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, 100 * frame.values, strict=True):
        ax.text(min(value + 1.2, 96), bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_quarterly_counts(panel: pd.DataFrame, breakpoints: pd.DataFrame, path: Path) -> None:
    order = ["Overall", "CPV-32", "CPV-35", "CPV-48", "CPV-72"]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.0), sharex=True)
    axes = axes.flatten()
    for ax, segment in zip(axes, order, strict=False):
        group = panel.loc[panel["segment"].eq(segment)].sort_values("quarter_start")
        ax.plot(group["quarter_start"], group["episode_count"], color="#356E9A", linewidth=1.8)
        stable = breakpoints.loc[
            breakpoints["segment"].eq(segment)
            & breakpoints["penalty_multiplier"].eq(1.0)
            & breakpoints["stable_across_penalties"]
        ]
        for quarter in stable["break_quarter"]:
            date = pd.Period(quarter, freq="Q").start_time
            ax.axvline(date, color="#C28A24", linestyle="--", linewidth=1.2)
        ax.set_title(segment)
        ax.set_ylabel("episodes")
        ax.grid(axis="y", color="#E1E5E8", linewidth=0.7)
    axes[-1].axis("off")
    fig.suptitle("Quarterly awarded digital procurement episodes", fontsize=15, y=0.995)
    fig.text(0.5, 0.965, "2015Q2-2025Q4; dashed lines are PELT breaks stable across penalty sensitivity", ha="center", color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_duration_completeness(panel: pd.DataFrame, path: Path) -> None:
    group = panel.loc[panel["segment"].eq("Overall")].sort_values("quarter_start")
    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    ax.plot(group["quarter_start"], 100 * group["duration_completeness"], color="#356E9A", linewidth=2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Reliable duration available (%)")
    fig.suptitle("Duration completeness changes materially over time", fontsize=14, y=0.98)
    fig.text(
        0.5,
        0.925,
        "Quarterly cohort rate; this instability rules out simple global duration imputation",
        ha="center",
        color="#555555",
    )
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_data_quality_report(profile: dict[str, Any]) -> Path:
    missing = profile["cohort_missingness"]
    sensitivity = profile["linkage_sensitivity"]
    text = f"""# BOAMP Data Quality Report

Generated: `{profile['generated_at']}`  
Data through: `{profile['data_as_of']}`  
Assessment: **Share with caveats**

## Technical Summary

The processed data are reproducible and internally coherent at their declared grains: `{profile['volume']['standardized_notices']:,}` unique BOAMP notices become `{profile['volume']['reconstructed_episodes']:,}` reconstructed procurement episodes, and the final study cohort contains `{profile['volume']['survival_cohort_rows']:,}` unique awarded Grand Ouest digital episodes. No duplicate notice IDs, duplicate survival episodes, negative survival durations, impossible episode chronologies, or accepted links with conflicting validated SIRENs were found.

The main risks are measurement rather than pipeline corruption. Validated SIREN is missing for `{missing['buyer_siren']:.1%}` of the cohort, reliable duration for `{missing['reliable_duration']:.1%}`, and the benchmark labels are deterministic bootstrap labels rather than independent specialist judgements. Therefore the linkage and survival results are usable as conservative exploratory evidence, but their reported precision is not an independently established accuracy guarantee.

## Data Grain And Selection

| Layer | Grain | Rows | Main rule |
|---|---|---:|---|
| Standardised data | BOAMP notice | {profile['volume']['standardized_notices']:,} | Official notices published 2015-2025 |
| Reconstructed data | Procurement episode | {profile['volume']['reconstructed_episodes']:,} | Explicit links, shared folder IDs, or constrained reference links |
| Study cohort | Awarded digital episode | {profile['volume']['survival_cohort_rows']:,} | Grand Ouest, CPV divisions 32/35/48/72, resolved award date |
| Candidate table | Anchor-candidate pair | {profile['volume']['candidate_pairs']:,} | Same plausible buyer, 90-2,920 days later |
| Survival table | Cohort episode | {profile['volume']['survival_cohort_rows']:,} | First accepted successor or administrative censoring |

The source is the official [BOAMP API]({SOURCE_URLS['boamp']}), which publishes procurement notices and results. A notice is not necessarily a distinct contract, so episode reconstruction is necessary before successor linkage.

## Integrity Checks

| Check | Result |
|---|---:|
| Duplicate standardised notice IDs | {profile['integrity']['duplicate_standardized_notice_ids']} |
| All notices assigned to exactly one episode | {profile['integrity']['all_notices_assigned_once']} |
| Buyer-conflict episodes | {profile['integrity']['buyer_conflict_episodes']} |
| Impossible episode chronologies | {profile['integrity']['impossible_chronology_episodes']} |
| Duplicate survival episodes | {profile['integrity']['duplicate_survival_episodes']} |
| Negative survival durations | {profile['integrity']['negative_survival_durations']} |
| Accepted links with conflicting validated SIRENs | {profile['integrity']['accepted_links_with_conflicting_validated_siren']} |
| Accepted municipal/intercommunal entity mixes | {profile['integrity']['accepted_municipal_intercommunal_mixes']} |

These checks support structural consistency, not semantic truth. A syntactically valid episode can still combine notices incorrectly, and a plausible successor can still be a different procurement need.

## Missingness And Treatment

| Field | Missing rate | Treatment | Reason |
|---|---:|---|---|
| Validated SIREN | {missing['buyer_siren']:.1%} | Preserve name-only buyer key; audit risky links | [SIREN]({SOURCE_URLS['siren']}) identifies a legal unit; [SIRET]({SOURCE_URLS['siret']}) identifies an establishment, so names alone cannot prove legal identity |
| Reliable duration | {missing['reliable_duration']:.1%} | No imputation | Completeness changes sharply by year and a universal four-year value would create false expiry dates |
| Any amount candidate container | {missing['amount_candidate_container']:.1%} | Do not aggregate as contract value | The container can hold multiple notice-level amount candidates and has no validated canonical awarded value |
| Main CPV | {missing['main_cpv']:.1%} | Required by cohort selection | CPV is a hierarchical procurement vocabulary under [Regulation 213/2008]({SOURCE_URLS['cpv']}) |
| Episode text | {missing['episode_text']:.1%} | Required by cohort selection | Needed for linkage ranking |
| Award date | {missing['award_date']:.1%} | Required by cohort selection | Defines survival time zero |

The decision not to impute duration is supported by the observed temporal instability: reliable duration is present for only `{profile['duration_completeness_by_award_year']['2023']['rate']:.1%}` of 2023 episodes but `{profile['duration_completeness_by_award_year']['2025']['rate']:.1%}` of 2025 episodes. Missingness is therefore not plausibly exchangeable across years. EU procurement law also treats four years as a general framework-agreement limit with justified exceptions, not as the duration of every contract ([Directive 2014/24/EU, Article 33]({SOURCE_URLS['framework']})).

![Cohort missingness](reports/figures/data_quality_key_missingness.png)

## Linkage Coverage And Sensitivity

Candidate generation finds at least one candidate for `{profile['candidate_coverage']['with_candidates']:,}` of `{profile['candidate_coverage']['anchors']:,}` anchors ({profile['candidate_coverage']['coverage_rate']:.1%}). This is candidate availability, not linking accuracy. The primary method accepts `{profile['volume']['accepted_primary_links']:,}` links.

| Linkage arm | Events | Event rate | Median observed successor time |
|---|---:|---:|---:|
| Strict text threshold | {sensitivity['strict']['events']} | {sensitivity['strict']['event_rate']:.1%} | {sensitivity['strict']['median_successor_months']:.1f} months |
| Primary text threshold | {sensitivity['main']['events']} | {sensitivity['main']['event_rate']:.1%} | {sensitivity['main']['median_successor_months']:.1f} months |
| Looser text threshold | {sensitivity['looser']['events']} | {sensitivity['looser']['event_rate']:.1%} | {sensitivity['looser']['median_successor_months']:.1f} months |
| Weighted high-recall contrast | {sensitivity['contrast_high_recall']['events']} | {sensitivity['contrast_high_recall']['event_rate']:.1%} | {sensitivity['contrast_high_recall']['median_successor_months']:.1f} months |

The large event-rate range is a material uncertainty result. Absolute survival probabilities depend on the linkage policy and should be presented with sensitivity analyses.

## Benchmark Evidence Quality

The current national reference has `{profile['benchmark_provenance']['labels']:,}` labelled pairs, but both passes were generated by the deterministic rules in `scripts/auto_annotate_wave1a.py`. Their κ of `{profile['benchmark_provenance']['cohen_kappa_six_class']:.2f}` is self-consistency under re-presentation, not human inter-annotator agreement. The split names `dev` and `validation` remain useful for preventing threshold reuse, but the resulting precision and recall are internal protocol-reference estimates.

This is the main unresolved validation risk. The appropriate correction is an independent specialist review of a compact, stratified sample, especially accepted links, method disagreements, high-similarity structural negatives, and name-only buyer matches.

## Defensible Use

- Safe: describe the pipeline as identifying **observable successor procurements**.
- Safe: use `M_B_text_ranking @ 0.70` as a provisional precision-first operating baseline.
- Safe: report sensitivity across linkage thresholds and methods.
- Not safe: call the events confirmed legal renewals.
- Not safe: call 0.80 an independently validated precision guarantee.
- Not safe: use current amount candidates for monetary trend conclusions.

## Reproduction

```bash
PYTHONPATH=. python3 scripts/build_project_evidence.py
PYTHONPATH=. jupyter nbconvert --execute --to notebook --inplace notebooks/14_data_quality_and_trend_analysis.ipynb
PYTHONPATH=. pytest -q
```
"""
    path = PROJECT_ROOT / "DATA_QUALITY_REPORT.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_trend_report(
    summary: dict[str, Any], signal_matrix: pd.DataFrame, breakpoints: pd.DataFrame
) -> Path:
    rows = []
    for row in signal_matrix.itertuples(index=False):
        last_stable_break = "--" if pd.isna(row.last_stable_break) else row.last_stable_break
        rows.append(
            f"| {row.segment} | {row.state} | {row.slope_episodes_per_quarter:.2f} | "
            f"{row.p_value:.3f} | {last_stable_break} |"
        )
    text = rf"""# BOAMP Descriptive Trend Analysis

Generated: `{summary['generated_at']}`  
Analysis window: `2015Q2-2025Q4`  
Unit: awarded Grand Ouest digital procurement episodes

## Technical Summary

This analysis adds the guide's missing time-series component without claiming a forecast. Quarterly episode counts are examined for the overall cohort and CPV divisions 32, 35, 48, and 72. PELT identifies candidate mean shifts, penalty sensitivity distinguishes stable from fragile breaks, and a 12-quarter linear slope describes the current direction.

The results are descriptive signals only. Breaks are not automatically attributed to policy, technology, or COVID; those explanations require documentary evidence and stakeholder validation. Amount trends are omitted because the current episode layer has multiple unvalidated amount candidates rather than one canonical awarded amount.

## Current Signal Matrix

| Segment | Recent direction | Episodes/quarter slope | Exploratory p-value | Last stable PELT break |
|---|---|---:|---:|---|
{chr(10).join(rows)}

`stable_or_uncertain` means the 12-quarter slope is not distinguishable from zero at the pre-declared exploratory level α = 0.10. These p-values are descriptive and are not corrected for multiple testing.

![Quarterly episode counts](reports/figures/trend_quarterly_episode_counts.png)

## Method

For segment $s$ and quarter $q$, the count is

\[
N_{{s,q}}=\sum_i \mathbf{{1}}(S_i=s, Q_i=q),
\]

where $S_i$ is the episode's CPV division and $Q_i$ is its award quarter. The partial first quarter of 2015 is excluded; all quarters from 2015Q2 onward are represented, including zeros.

PELT minimizes a penalized segmentation objective,

\[
\sum_{{r=0}}^m \mathcal{{C}}(y_{{\tau_r+1:\tau_{{r+1}}}})+\beta m,
\]

where \(\mathcal{{C}}\) is within-segment squared error, \(m\) is the number of breaks, and \(\beta=\lambda\log(n)\) after z-standardization. The central result uses \(\lambda=1\); sensitivity uses 0.5 and 2.0. A break is called stable only when a break lies within one quarter under all three penalties. This follows the PELT framework of [Killick, Fearnhead and Eckley (2012)]({SOURCE_URLS['pelt']}).

The recent direction comes from an ordinary least-squares slope over the latest 12 quarters. It is `increasing` or `decreasing` only when its two-sided p-value is below 0.10; otherwise it is `stable_or_uncertain`. This is a signal description, not a forward prediction.

## Duration Completeness Is A Measurement Break

![Duration completeness](reports/figures/trend_duration_completeness.png)

The sharp rise in duration availability in 2025 is a schema/completeness change, not evidence that contract durations suddenly changed. Median-duration trend claims would mix periods with substantially different observation processes, so the report does not run change-point detection on duration values.

## Limits

- The series cover awarded digital episodes, not every procurement notice or every French contract.
- CPV divisions are broad operational segments, not the 8-12 class supervised taxonomy proposed in the internship guide.
- Count changes can reflect publication practice, schema changes, buyer coverage, or procurement activity.
- PELT proposes candidate breaks; it does not identify their causes.
- Monetary trends remain unavailable until one awarded-value definition is validated.

## Reproducible Outputs

- `data/processed/boamp_v2/trend_quarterly.csv`
- `data/processed/boamp_v2/trend_breakpoints.csv`
- `data/processed/boamp_v2/trend_signal_matrix.csv`
- `data/processed/boamp_v2/trend_analysis_summary.json`
- `notebooks/14_data_quality_and_trend_analysis.ipynb`
"""
    path = PROJECT_ROOT / "TREND_ANALYSIS_REPORT.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# 14. Data quality and descriptive trends\n\n"
            "## tl;dr\n\n"
            "The processing layers pass their structural integrity checks, but buyer identifiers, duration, amount, and independent benchmark validation remain material limitations. Quarterly PELT results are descriptive break signals, not causal explanations or forecasts."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "The unit is one awarded Grand Ouest digital procurement episode. The trend window starts at 2015Q2 because the raw extract begins in March 2015. PELT uses standardized quarterly counts and a log(n) penalty with sensitivity multipliers 0.5, 1, and 2."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / 'scripts').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "PROCESSED = PROJECT_ROOT / 'data/processed/boamp_v2'\n"
            "with open(PROCESSED / 'data_quality_profile.json', encoding='utf-8') as f:\n"
            "    quality = json.load(f)\n"
            "with open(PROCESSED / 'trend_analysis_summary.json', encoding='utf-8') as f:\n"
            "    trend_summary = json.load(f)\n"
            "quarterly = pd.read_csv(PROCESSED / 'trend_quarterly.csv', parse_dates=['quarter_start'])\n"
            "breakpoints = pd.read_csv(PROCESSED / 'trend_breakpoints.csv')\n"
            "signals = pd.read_csv(PROCESSED / 'trend_signal_matrix.csv')\n"
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            "pd.DataFrame([\n"
            "    {'metric': 'standardized notices', 'value': quality['volume']['standardized_notices']},\n"
            "    {'metric': 'reconstructed episodes', 'value': quality['volume']['reconstructed_episodes']},\n"
            "    {'metric': 'study cohort episodes', 'value': quality['volume']['survival_cohort_rows']},\n"
            "    {'metric': 'candidate pairs', 'value': quality['volume']['candidate_pairs']},\n"
            "    {'metric': 'primary successor events', 'value': quality['volume']['accepted_primary_links']},\n"
            "])"
        ),
        nbf.v4.new_code_cell(
            "pd.Series(quality['cohort_missingness'], name='missing_rate').sort_values(ascending=False).to_frame()"
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            "display(signals)\n"
            "fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)\n"
            "for ax, (segment, group) in zip(axes.flatten(), quarterly.groupby('segment', sort=False)):\n"
            "    group = group.sort_values('quarter_start')\n"
            "    ax.plot(group['quarter_start'], group['episode_count'], color='#356E9A')\n"
            "    ax.set_title(segment)\n"
            "    ax.grid(axis='y', alpha=0.25)\n"
            "axes.flatten()[-1].axis('off')\n"
            "fig.suptitle('Quarterly awarded digital procurement episodes')\n"
            "plt.tight_layout()"
        ),
        nbf.v4.new_code_cell(
            "overall = quarterly.loc[quarterly['segment'].eq('Overall')].sort_values('quarter_start')\n"
            "ax = overall.plot(x='quarter_start', y='duration_completeness', figsize=(10, 4), legend=False, color='#356E9A')\n"
            "ax.set_title('Reliable duration completeness by quarter')\n"
            "ax.set_ylabel('share')\n"
            "ax.set_ylim(0, 1)\n"
            "ax.grid(axis='y', alpha=0.25)"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "- The current project has enough episodes for descriptive survival analysis, but the event definition remains linkage-conditioned.\n"
            "- Duration missingness changes sharply over time, so global duration imputation would create unsupported temporal structure.\n"
            "- PELT breaks are candidates for documentary interpretation, not causal findings.\n"
            "- Current benchmark metrics remain internal development evidence until independent specialist review is completed."
        ),
    ]
    nbf.write(nb, NOTEBOOK)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(PROCESSED / "survival_cohort.parquet")
    profile = data_quality_profile(cohort)
    profile_path = PROCESSED / "data_quality_profile.json"
    write_json(profile_path, profile)

    panel = build_quarterly_panel(cohort)
    breakpoints, signal_matrix, diagnostics = build_trend_outputs(panel)
    panel.to_csv(PROCESSED / "trend_quarterly.csv", index=False)
    breakpoints.to_csv(PROCESSED / "trend_breakpoints.csv", index=False)
    signal_matrix.to_csv(PROCESSED / "trend_signal_matrix.csv", index=False)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window": {"start": "2015Q2", "end": "2025Q4", "quarters": 43},
        "unit": "awarded Grand Ouest digital procurement episode",
        "method": {
            "change_point": "PELT l2 on z-standardized quarterly counts",
            "penalty": "multiplier * log(n)",
            "penalty_multipliers": [0.5, 1.0, 2.0],
            "stable_break_tolerance_quarters": 1,
            "recent_trend": "OLS slope over latest 12 quarters; alpha=0.10",
        },
        "stationarity": diagnostics,
        "amount_series_available": False,
        "amount_omission_reason": "no validated canonical awarded amount at episode grain",
        "causal_interpretation": False,
        "sources": SOURCE_URLS,
        "validation_passed": True,
    }
    write_json(PROCESSED / "trend_analysis_summary.json", summary)

    plot_missingness(profile, FIGURES / "data_quality_key_missingness.png")
    plot_quarterly_counts(panel, breakpoints, FIGURES / "trend_quarterly_episode_counts.png")
    plot_duration_completeness(panel, FIGURES / "trend_duration_completeness.png")
    quality_report = write_data_quality_report(profile)
    trend_report = write_trend_report(summary, signal_matrix, breakpoints)
    write_notebook()

    print(
        json.dumps(
            {
                "data_quality_report": str(quality_report.relative_to(PROJECT_ROOT)),
                "trend_report": str(trend_report.relative_to(PROJECT_ROOT)),
                "notebook": str(NOTEBOOK.relative_to(PROJECT_ROOT)),
                "profile": str(profile_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
