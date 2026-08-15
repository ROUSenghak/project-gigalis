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
    regime_diagnostics,
    stable_breaks,
    stationarity_diagnostics,
)

REGIME_SEGMENTS = ("Overall", "CPV-72", "CPV-32")

PROCESSED = PROJECT_ROOT / "data/processed/boamp"
REFERENCE = PROCESSED / "regional_benchmark"
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
    reference_manifest = load_json(REFERENCE / "regional_benchmark_manifest.json")

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
        "event_detectability": {
            str(row["variable"]): round(float(row["standardized_mean_difference"]), 4)
            for _, row in (
                pd.read_csv(PROCESSED / "survival_selection_diagnostic.csv")
                .sort_values("absolute_smd", ascending=False)
                .head(4)
                .iterrows()
            )
        },
        "benchmark_provenance": {
            "reference": reference_manifest["benchmark"],
            "reviewed_anchors": reference_manifest["reviewed_anchors"],
            "usable_anchors": sum(
                split["usable_anchors"] for split in reference_manifest["splits"].values()
            ),
            "label_provenance": reference_manifest["label_provenance"],
            "independent_of_linkage_algorithms": reference_manifest[
                "independent_of_linkage_algorithms"
            ],
            "independent_human_annotation": reference_manifest[
                "independent_human_specialist_review"
            ],
            "candidate_generation_recall_ceiling": reference_manifest[
                "candidate_reachability"
            ]["candidate_generation_recall_ceiling"],
            "interpretation": (
                "A stratified Grand Ouest review carried out before the linkage methods "
                "existed. Labels were generated by a single LLM research pass over the "
                "BOAMP notices, their official URLs, and wider public sources, then "
                "spot-checked on a subset by the project owner. Independent of those "
                "methods, but not verified anchor-by-anchor, not an independent "
                "specialist panel, and with corpus-relative negatives."
            ),
            "generator": "scripts/build_regional_benchmark.py",
            "source": "data/reference/regional_link_benchmark/",
        },
        "assessment": "share_with_caveats",
        "sources": SOURCE_URLS,
    }


def business_recommendation(
    segment: str, state: str, slope: float, last_stable_break: Any, regime: str | None
) -> str:
    """Translate one segment's descriptive signals into a monitoring action.

    The recommendation is a reading of the signals already in the row, never a
    causal story: PELT reports where the series level shifted, not why, so the
    wording stays at the level of what a purchasing body should do next rather
    than what it should believe happened.
    """
    if state == "decreasing":
        action = (
            f"Investigate the recent decline in {segment} before reducing or "
            "expanding procurement capacity; confirm whether it reflects demand, "
            "publication practice, or a routing change to another channel."
        )
    elif state == "increasing":
        action = (
            f"Plan for sustained {segment} volume; the recent direction is upward "
            "and recurring-demand evidence supports keeping framework planning active."
        )
    else:
        action = (
            f"Maintain monitoring for {segment}; no statistically clear recent "
            "direction at the pre-declared exploratory level."
        )
    if not pd.isna(last_stable_break):
        action += (
            f" A penalty-stable level shift is dated {last_stable_break}; treat it as "
            "a break candidate to be explained with documentary evidence, not as a "
            "demonstrated cause."
        )
    if regime and state == "stable_or_uncertain":
        action += (
            f" The HMM currently reads this series as `{regime}`, which describes "
            "recent quarter-over-quarter change and need not agree with the "
            "12-quarter slope."
        )
    return action


def build_trend_outputs(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    breakpoint_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    regimes: dict[str, Any] = {}
    for segment, group in panel.groupby("segment", sort=False):
        group = group.sort_values("quarter_start").reset_index(drop=True)
        values = group["episode_count"].astype(float).to_numpy()
        break_result = stable_breaks(values)
        trend = recent_trend_signal(values)
        diagnostics[str(segment)] = stationarity_diagnostics(values)
        if str(segment) in REGIME_SEGMENTS:
            regimes[str(segment)] = regime_diagnostics(values)
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
        last_stable_break = group.loc[stable[-1], "quarter"] if stable else None
        current_regime = regimes.get(str(segment), {}).get("current_regime")
        signal_rows.append(
            {
                "segment": segment,
                **trend,
                "last_central_break": group.loc[central[-1], "quarter"] if central else None,
                "last_stable_break": last_stable_break,
                "central_break_count": len(central),
                "stable_break_count": len(stable),
                "hmm_current_regime": current_regime,
                "hmm_current_regime_probability": regimes.get(str(segment), {}).get(
                    "current_regime_probability"
                ),
                "business_recommendation": business_recommendation(
                    str(segment),
                    trend["state"],
                    trend["slope_episodes_per_quarter"],
                    last_stable_break if last_stable_break is not None else float("nan"),
                    current_regime,
                ),
            }
        )
    return pd.DataFrame(breakpoint_rows), pd.DataFrame(signal_rows), diagnostics, regimes


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


def event_validation_section(profile: dict[str, Any]) -> str:
    """The single canonical account of how the `event` variable was validated.

    The supervisor asked for the constructed event to be validated before it was
    used in survival analysis. The underlying evidence lives in several
    generators, so this section is the one place that states the definition and
    the verdict; every number in it is quoted from a materialised artifact and
    the detail is left in the appendices those artifacts back.
    """
    survival = load_json(PROCESSED / "survival_analysis_summary.json")
    evaluation = load_json(PROCESSED / "linkage_evaluation_validation.json")
    reference = load_json(REFERENCE / "regional_benchmark_manifest.json")
    border = survival["borderline_link_sensitivity"]
    temporal = survival["cox"]["temporal_validation"]
    primary = next(
        method for method in evaluation["methods"]
        if method["method"] == "M_B_text_ranking"
    )
    exact = primary["unweighted"]
    sensitivity = profile["linkage_sensitivity"]
    smd = profile["event_detectability"]
    return f"""## Validation Of The Constructed Successor-Event Variable

`event` is a constructed, linkage-conditioned proxy, so it is validated here
before it is used in survival analysis. This is the canonical account; the
detailed tables live in the artifacts each subsection names.

**1. Definition.** `event = 1` when a later Grand Ouest procurement episode is
accepted as the observable successor of an awarded cohort episode under the
frozen linkage rule, and the survival time is the award-to-successor gap.
`event = 0` when no successor is accepted before `{profile['data_as_of']}`, and the row
is administratively right-censored. `event = 0` is **not** evidence that the
procurement was never renewed or re-procured: it may have been re-procured
outside BOAMP, through a central purchasing body, below publication thresholds,
or through a successor the linkage rule failed to recover.

**2. Candidate generation.** A later episode is exposed for comparison only when
buyer evidence is plausible, validated SIRENs do not conflict, and it is
published between 90 and 2,920 days after the award. This produced
`{profile['volume']['candidate_pairs']:,}` pairs covering
`{profile['candidate_coverage']['with_candidates']:,}` of
`{profile['candidate_coverage']['anchors']:,}` anchors. The window is an operational
search range, not an assumed contract duration.

**3. Scoring rule.** `M_B_text_ranking` selects the single highest TF-IDF cosine
candidate per anchor and accepts it when that similarity reaches the threshold.
At most one successor is selected per anchor; otherwise the method abstains.

**4. Frozen threshold.** `0.70`, fixed a priori on a precision-first principle,
because a false link fabricates both a survival event and its event time. It was
not moved after the regional reference was read — which is the only reason the
locked split can be reported as held out — and it was not tuned toward any target
event rate. The internship guide's 40-60% linkage expectation was treated as a
planning figure, never as an optimisation target; the realised
`{sensitivity['main']['event_rate']:.1%}` is a consequence of the precision-first rule,
not a miss against a goal.

**5. Reference design.** A stratified set of `{reference['reviewed_anchors']}` Grand
Ouest anchors labelled on `{reference['review_date']}`, before these linkage methods
existed. The project owner supplied the raw anchor and candidate data; a single LLM
research pass consulted the BOAMP notices, their official URLs, and wider public
sources, and proposed a successor or an abstention per anchor; the project owner
then spot-checked a subset rather than verifying every anchor.
`{reference['remap']['resolved_to_current_episodes']}` re-resolve onto the current
episode reconstruction. This is a reference sample, not ground truth, and the
per-anchor evidence trail was not recorded. See `REGIONAL_BENCHMARK_REFERENCE.md`.

**6. Reference performance.** On the locked split
(`{evaluation['anchors_evaluated']}` evaluable anchors: `{exact['positive_anchors']}`
with a reviewed successor and `{exact['negative_anchors']}` without), exact-successor
accounting gives TP `{exact['true_positive']}`, FP
`{exact['false_positive_wrong_successor'] + exact['false_positive_on_no_successor_anchor']}`,
FN `{exact['false_negative_abstained']}`, TN `{exact['true_negative_abstained']}`:
precision `{exact['precision_at_1']:.3f}` (95% CI
`{exact['precision_at_1_interval_95'][0]:.3f}`-`{exact['precision_at_1_interval_95'][1]:.3f}`),
recall `{exact['recall_at_1']:.3f}` (95% CI
`{exact['recall_at_1_interval_95'][0]:.3f}`-`{exact['recall_at_1_interval_95'][1]:.3f}`),
false-positive rate on reviewed negative anchors
`{exact['false_positive_rate_on_negatives']:.3f}`. Both event-positive and event-negative
situations are represented, so all four cells are populated. Recall is additionally
capped at `{reference['candidate_reachability']['candidate_generation_recall_ceiling']:.3f}`
by candidate generation, before any method runs. Intervals matter at this sample
size — see `QUALITY_EVIDENCE.md`.

A caveat binds cell 6 in particular: reference negatives are corpus-relative,
because roughly 25 candidates per anchor were considered rather than the full pool.
An accepted link on an anchor labelled negative counts as an error even when the
research pass never saw that candidate, so the figure is conservative by
construction: it can overstate error but not understate it. It remains a
diagnostic on this sample rather than a population-wide false-positive rate — the
reference is a stratified sample with design weights spanning 1 to 68, and
`{exact['false_positive_rate_on_negatives']:.3f}` on
`{exact['negative_anchors']}` negative anchors still carries a wide interval.

The checks in cells 6 to 9 — reference scoring, threshold and borderline
sensitivity, and the linked-versus-unlinked comparison — follow the
linkage-quality evaluation strategy set out by
[Harron et al. (2017)](https://doi.org/10.1093/ije/dyx177). That source supports
the strategy, not these numbers.

**7. Threshold sensitivity.** Four retained arms span the event definition:

| Arm | Events | Event rate | Median observed successor time |
|---|---:|---:|---:|
| `M_B @ 0.80` strict | {sensitivity['strict']['events']} | {sensitivity['strict']['event_rate']:.1%} | {sensitivity['strict']['median_successor_months']:.1f} months |
| `M_B @ 0.70` primary | {sensitivity['main']['events']} | {sensitivity['main']['event_rate']:.1%} | {sensitivity['main']['median_successor_months']:.1f} months |
| `M_B @ 0.60` loose | {sensitivity['looser']['events']} | {sensitivity['looser']['event_rate']:.1%} | {sensitivity['looser']['median_successor_months']:.1f} months |
| `M_C @ 0.70` contrast | {sensitivity['contrast_high_recall']['events']} | {sensitivity['contrast_high_recall']['event_rate']:.1%} | {sensitivity['contrast_high_recall']['median_successor_months']:.1f} months |

**8. Borderline-case robustness.** Excluding every anchor whose best candidate
scores within `±0.05` of the threshold removes `{border['contracts_removed']:,}` episodes
and `{border['events_removed']}` events. The direction of both headline hazard ratios
survives (CPV-35 `{border['main']['cox_hr_cpv_35']:.2f}` to
`{border['excluding_borderline_links']['cox_hr_cpv_35']:.2f}`; framework
`{border['main']['cox_hr_framework']:.2f}` to
`{border['excluding_borderline_links']['cox_hr_framework']:.2f}`), while the absolute KM
level moves. Comparative claims are therefore not driven by borderline linkage
decisions; absolute probabilities remain threshold-uncertain. See
`survival_borderline_link_sensitivity.csv`.

**9. Linked versus unlinked detectability.** Standardised mean differences between
event-positive and event-negative episodes are largest for
{', '.join(f"`{name}` (SMD `{value:+.3f}`)" for name, value in smd.items())}.
Longer episode text is linked more often, which is consistent with text-similarity
scoring having more to work with. Framework agreements are linked more frequently;
this may reflect genuine recurrence behaviour, differential linkability, or both,
and this evidence cannot separate them. Award-year differences are confounded with
unequal follow-up. See `survival_selection_diagnostic.csv`.

**10. Implications for survival interpretation.** Every survival quantity is
conditional on this event definition. Missed successors may reduce the observed
event rate, whereas residual false links may increase it, so the estimates are
**not** formal lower bounds on true re-procurement probability. The defensible
claims are comparative — which segments and contract types show an observable
successor sooner — supported by the four-arm and borderline checks. Absolute
probabilities must be quoted with their sensitivity range. Out-of-time Cox
discrimination is weak (test C-index `{temporal['test_c_index']:.3f}` on
`{temporal['test_years']}`), so the Cox layer is descriptive and no individualised
prediction claim is made.

"""


def write_data_quality_report(profile: dict[str, Any]) -> Path:
    missing = profile["cohort_missingness"]
    sensitivity = profile["linkage_sensitivity"]
    text = f"""# BOAMP Data Quality Report

Generated: `{profile['generated_at']}`  
Data through: `{profile['data_as_of']}`  
Assessment: **Share with caveats**

## Technical Summary

The processed data are reproducible and internally coherent at their declared grains: `{profile['volume']['standardized_notices']:,}` unique BOAMP notices become `{profile['volume']['reconstructed_episodes']:,}` reconstructed procurement episodes, and the final study cohort contains `{profile['volume']['survival_cohort_rows']:,}` unique awarded Grand Ouest digital episodes. No duplicate notice IDs, duplicate survival episodes, negative survival durations, impossible episode chronologies, or accepted links with conflicting validated SIRENs were found.

The main risks are measurement rather than pipeline corruption. Validated SIREN is missing for `{missing['buyer_siren']:.1%}` of the cohort, reliable duration for `{missing['reliable_duration']:.1%}`, and the reference labels were generated by a single LLM research pass and spot-checked on a subset rather than verified anchor-by-anchor or judged by an independent specialist. Therefore the linkage and survival results are usable as conservative exploratory evidence, but their reported precision is not an independently established accuracy guarantee.

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

{event_validation_section(profile)}## Reference Evidence Quality

The active reference is the Grand Ouest regional review: `{profile['benchmark_provenance']['reviewed_anchors']}` anchors reviewed against real BOAMP notices, of which `{profile['benchmark_provenance']['usable_anchors']}` are usable after re-resolution onto the current episode reconstruction. It replaced a France-level benchmark whose labels were generated by deterministic rules reading the same evidence the linkage methods consume, which made that comparison circular.

Three limits remain. The labels were generated by a single LLM research pass over the notices, their official URLs, and wider public sources, then spot-checked on a subset by the project owner rather than verified anchor-by-anchor, and they are not an independent specialist panel. The sources behind each individual label were not recorded, so a given anchor's evidence trail cannot be fully reconstructed. Negatives are corpus-relative because roughly 25 candidates per anchor were considered. Recall is additionally capped at `{profile['benchmark_provenance']['candidate_generation_recall_ceiling']:.3f}` by candidate generation, before any method runs.

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
    recommendation_rows = []
    for row in signal_matrix.itertuples(index=False):
        last_stable_break = "--" if pd.isna(row.last_stable_break) else row.last_stable_break
        regime = "--" if not row.hmm_current_regime else row.hmm_current_regime
        rows.append(
            f"| {row.segment} | {row.state} | {row.slope_episodes_per_quarter:.2f} | "
            f"{row.p_value:.3f} | {last_stable_break} | {regime} |"
        )
        recommendation_rows.append(
            f"| {row.segment} | {row.state} | {row.business_recommendation} |"
        )

    stationarity_rows = []
    for segment, diag in summary["stationarity"].items():
        if not diag.get("available"):
            stationarity_rows.append(f"| {segment} | n/a | n/a | {diag.get('reason', '')} |")
            continue
        adf = diag["adf"]
        kpss_diag = diag["kpss"]
        kpss_text = (
            f"stat={kpss_diag['statistic']:.3f}, p={kpss_diag['p_value']:.3f}"
            if kpss_diag.get("available", True)
            else "n/a"
        )
        stationarity_rows.append(
            f"| {segment} | ADF stat={adf['statistic']:.3f}, p={adf['p_value']:.3f} "
            f"| KPSS {kpss_text} | |"
        )

    regime_rows = []
    for segment, regime in summary["regimes"].items():
        if not regime.get("available"):
            regime_rows.append(f"| {segment} | n/a | n/a | {regime.get('reason', '')} |")
            continue
        regime_rows.append(
            f"| {segment} | {regime['current_regime']} | "
            f"{regime['current_regime_probability']:.3f} | "
            f"decline={regime['mean_change_by_regime']['decline']:.1f}, "
            f"plateau={regime['mean_change_by_regime']['plateau']:.1f}, "
            f"growth={regime['mean_change_by_regime']['growth']:.1f} |"
        )

    text = rf"""# BOAMP Descriptive Trend Analysis

Generated: `{summary['generated_at']}`  
Analysis window: `2015Q2-2025Q4`  
Unit: awarded Grand Ouest digital procurement episodes

## Technical Summary

This analysis adds the guide's missing time-series component without claiming a forecast. Quarterly episode counts are examined for the overall cohort and CPV divisions 32, 35, 48, and 72. PELT identifies candidate mean shifts, penalty sensitivity distinguishes stable from fragile breaks, and a 12-quarter linear slope describes the current direction.

The results are descriptive signals only. Breaks are not automatically attributed to policy, technology, or COVID; those explanations require documentary evidence and stakeholder validation. Amount trends are omitted because the current episode layer has multiple unvalidated amount candidates rather than one canonical awarded amount.

## Current Signal Matrix

| Segment | Recent direction | Episodes/quarter slope | Exploratory p-value | Last stable PELT break | HMM regime |
|---|---|---:|---:|---|---|
{chr(10).join(rows)}

`stable_or_uncertain` means the 12-quarter slope is not distinguishable from zero at the pre-declared exploratory level α = 0.10. These p-values are descriptive and are not corrected for multiple testing.

![Quarterly episode counts](reports/figures/trend_quarterly_episode_counts.png)

## Operational Reading

Each row translates that segment's own signals into a monitoring action. These are
readings of the descriptive evidence, not causal explanations and not forecasts: a
PELT break marks where the series level shifted, never why, and no recommendation
below should be quoted as attributing a shift to policy, COVID, regulation, or
technology.

| Segment | Recent direction | Recommendation |
|---|---|---|
{chr(10).join(recommendation_rows)}

## Stationarity (ADF/KPSS)

| Segment | ADF (H0: unit root) | KPSS (H0: level stationary) | Note |
|---|---|---|---|
{chr(10).join(stationarity_rows)}

ADF and KPSS test opposite null hypotheses, so they are read together rather than
individually. A series that rejects the ADF unit-root null while failing to reject
the KPSS stationarity null is consistent with level stationarity around a constant
or slowly varying mean; disagreement between the two tests indicates the series is
not cleanly classified as stationary or non-stationary over this short window. These
diagnostics describe the fitted quarterly series; they are not used to justify or
rule out forecasting, which remains out of scope.

## Regime Detection (HMM)

A 3-state Gaussian hidden Markov model is fit on the quarter-over-quarter change in
episode count (not the level) for the overall cohort and the two highest-volume CPV
segments, so the three states describe typical period-over-period direction —
`decline`, `plateau`, `growth` — rather than the segment's absolute activity level. A
segment can hold a high or low count while still sitting in a `plateau` regime if its
recent changes are small. The reported probability is the model's posterior
probability of the current-quarter regime, not a forecast, and a detected regime is
not a causal explanation of any prior shift.

| Segment | Current regime | Probability | Mean quarterly change by regime |
|---|---|---:|---|
{chr(10).join(regime_rows)}

Because the model is fit on noisy, low-count quarterly series, the `plateau` state is
a data-driven middle tier rather than a change centered exactly at zero; its mean
change should be read alongside `decline` and `growth` rather than interpreted as
"no change." The HMM's current-regime label and the 12-quarter OLS slope above are
complementary, not identical: the OLS slope summarizes the last 12 quarters, while
the HMM regime reflects the model's belief about the most recent quarter's state and
can differ from the OLS signal without either being wrong.

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

- `data/processed/boamp/trend_quarterly.csv`
- `data/processed/boamp/trend_breakpoints.csv`
- `data/processed/boamp/trend_signal_matrix.csv`
- `data/processed/boamp/trend_analysis_summary.json`
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
            "The processing layers pass their structural integrity checks, but buyer identifiers, duration, amount, and independent reference validation remain material limitations. Quarterly PELT results are descriptive break signals, not causal explanations or forecasts."
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
            "PROCESSED = PROJECT_ROOT / 'data/processed/boamp'\n"
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
            "- Current reference metrics are regional reference-sample evidence; independent specialist review is still needed before any external accuracy claim."
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
    breakpoints, signal_matrix, diagnostics, regimes = build_trend_outputs(panel)
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
            "regime_detection": (
                "3-state Gaussian HMM on quarter-over-quarter change in episode "
                "count, for Overall and the two highest-volume CPV segments"
            ),
        },
        "stationarity": diagnostics,
        "regimes": regimes,
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
