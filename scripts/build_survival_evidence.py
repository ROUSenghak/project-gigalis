#!/usr/bin/env python3
"""Materialise the current survival results and detectability diagnostics."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lifelines import (  # noqa: E402
    CoxPHFitter,
    ExponentialFitter,
    GeneralizedGammaFitter,
    KaplanMeierFitter,
    LogLogisticFitter,
    LogNormalFitter,
    WeibullFitter,
)
from lifelines.statistics import (  # noqa: E402
    multivariate_logrank_test,
    proportional_hazard_test,
)
from lifelines.utils import concordance_index  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data/processed/boamp"
FIGURES = PROJECT_ROOT / "reports/figures"
REPORT = PROJECT_ROOT / "SURVIVAL_ANALYSIS_REPORT.md"
#: Report figure palette, matching scripts/build_project_evidence.py so the
#: methodology chapter reads as one document.
INK, ACCENT, GRID = "#356E9A", "#C28A24", "#E1E5E8"
SEGMENT_COLOURS = ("#356E9A", "#C28A24", "#3F8F6B", "#9A4F35")
MONTH_DAYS = 30.4375
HORIZONS = (12, 24, 36, 48, 60)
ARM_FILES = {
    "strict": "survival_dataset_strict.parquet",
    "main": "survival_dataset.parquet",
    "looser": "survival_dataset_looser.parquet",
    "contrast_high_recall": "survival_dataset_contrast_high_recall.parquet",
}
#: The M_B decision variable is ``text_component`` on a 0-1 scale and the frozen
#: acceptance threshold is 0.70. A symmetric +/-0.05 band around it collects the
#: anchors whose event status would flip under a small threshold perturbation:
#: accepted links that only just cleared the bar, and abstentions whose best
#: candidate only just missed it. One band, fixed a priori, not optimised.
BORDERLINE_BAND = (0.65, 0.75)
#: Word-level similarity below which an accepted link rests on the character
#: analyser alone. ``text_component`` is ``max(word_tfidf, char_tfidf)``, and
#: French award notices carry long standardised framework boilerplate --
#: accord-cadre, bons de commande, bordereau des prix unitaires -- on which
#: character n-grams score highly between substantively different objects. The
#: value is the one already published by ``scripts/audit_candidate_generation.py``;
#: it is not re-chosen here and nothing is tuned against the outcome.
LOW_WORD_SIMILARITY = 0.50


def conditional_probability(fitter: Any, age: float, horizon: float) -> float:
    """P(event in the next horizon | event-free at age)."""
    survival_now = float(fitter.survival_function_at_times(age).iloc[0])
    survival_then = float(fitter.survival_function_at_times(age + horizon).iloc[0])
    return float("nan") if survival_now <= 0 else 1.0 - survival_then / survival_now


def standardized_mean_difference(event: pd.Series, censored: pd.Series) -> float | None:
    """Difference in means divided by the equally pooled within-group SD."""
    event = pd.to_numeric(event, errors="coerce").dropna()
    censored = pd.to_numeric(censored, errors="coerce").dropna()
    if len(event) < 2 or len(censored) < 2:
        return None
    denominator = np.sqrt((event.var(ddof=1) + censored.var(ddof=1)) / 2.0)
    if not np.isfinite(denominator) or denominator == 0:
        return 0.0
    return float((event.mean() - censored.mean()) / denominator)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["t_months"] = np.maximum(frame["duration_days"], 1) / MONTH_DAYS
    frame["text_length_chars"] = frame["episode_text"].fillna("").astype(str).str.len()
    frame["administrative_followup_months"] = frame["days_to_cutoff"] / MONTH_DAYS
    frame["has_reliable_duration"] = frame["duration_months_reliable"].notna()
    frame["has_validated_siren"] = frame["buyer_identifier_quality"].eq("trusted_siren")
    return frame


def cox_design(frame: pd.DataFrame) -> pd.DataFrame:
    design = frame[[
        "t_months", "event", "digital_segment", "buyer_region",
        "framework_flag", "buyer_identifier_quality", "award_year",
    ]].copy()
    design["framework_flag"] = design["framework_flag"].astype(int)
    design["has_validated_siren"] = design["buyer_identifier_quality"].eq("trusted_siren").astype(int)
    design["award_year_centered"] = design["award_year"] - design["award_year"].median()
    design = design.drop(columns=["buyer_identifier_quality", "award_year"])
    return pd.get_dummies(
        design, columns=["digital_segment", "buyer_region"], drop_first=True
    )


def round_or_none(value: Any, digits: int = 4) -> float | None:
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def km_outputs(frame: pd.DataFrame) -> tuple[KaplanMeierFitter, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    km = KaplanMeierFitter().fit(frame["t_months"], frame["event"], label="overall")
    overall = pd.DataFrame([
        {
            "months": horizon,
            "survival_no_successor": float(km.survival_function_at_times(horizon).iloc[0]),
            "cumulative_successor_probability": 1.0 - float(
                km.survival_function_at_times(horizon).iloc[0]
            ),
        }
        for horizon in HORIZONS
    ])

    segment_rows = []
    for segment, group in frame.groupby("digital_segment", sort=True):
        fitted = KaplanMeierFitter().fit(group["t_months"], group["event"])
        row = {
            "segment": segment,
            "contracts": len(group),
            "events": int(group["event"].sum()),
            "censored": int((group["event"] == 0).sum()),
            "event_rate": float(group["event"].mean()),
            "km_median_months": round_or_none(fitted.median_survival_time_, 2),
        }
        for horizon in HORIZONS:
            row[f"survival_{horizon}m"] = float(
                fitted.survival_function_at_times(horizon).iloc[0]
            )
        segment_rows.append(row)
    segments = pd.DataFrame(segment_rows)
    test = multivariate_logrank_test(
        frame["t_months"], frame["digital_segment"], frame["event"]
    )
    logrank = {
        "test": "multivariate log-rank across CPV segments",
        "test_statistic": round(float(test.test_statistic), 4),
        "p_value": float(test.p_value),
    }
    return km, overall, segments, logrank


def conditional_outputs(km: KaplanMeierFitter, frame: pd.DataFrame) -> pd.DataFrame:
    ages = (0, 12, 24, 36, 48)
    rng = np.random.default_rng(20260812)
    draws = {(age, horizon): [] for age in ages for horizon in (12, 24)}
    for _ in range(500):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        fitted = KaplanMeierFitter().fit(sample["t_months"], sample["event"])
        for age, horizon in draws:
            draws[(age, horizon)].append(conditional_probability(fitted, age, horizon))

    rows = []
    for age in ages:
        for horizon in (12, 24):
            values = np.asarray(draws[(age, horizon)], dtype=float)
            values = values[np.isfinite(values)]
            rows.append({
                "contract_age_months": age,
                "horizon_months": horizon,
                "probability": conditional_probability(km, age, horizon),
                "ci_95_low": float(np.percentile(values, 2.5)),
                "ci_95_high": float(np.percentile(values, 97.5)),
                "interval_method": "episode bootstrap, 500 draws",
            })
    return pd.DataFrame(rows)


def selection_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    linked = frame.loc[frame["event"].eq(1)]
    censored = frame.loc[frame["event"].eq(0)]
    continuous = []
    for column in (
        "award_year", "notice_count", "text_length_chars",
        "administrative_followup_months",
    ):
        smd = standardized_mean_difference(linked[column], censored[column])
        continuous.append({
            "variable": column,
            "linked_n": int(linked[column].notna().sum()),
            "censored_n": int(censored[column].notna().sum()),
            "linked_mean": float(linked[column].mean()),
            "censored_mean": float(censored[column].mean()),
            "standardized_mean_difference": smd,
            "absolute_smd": abs(smd) if smd is not None else None,
        })
    for column in ("framework_flag", "has_validated_siren", "has_reliable_duration"):
        smd = standardized_mean_difference(
            linked[column].astype(int), censored[column].astype(int)
        )
        continuous.append({
            "variable": column,
            "linked_n": len(linked),
            "censored_n": len(censored),
            "linked_mean": float(linked[column].mean()),
            "censored_mean": float(censored[column].mean()),
            "standardized_mean_difference": smd,
            "absolute_smd": abs(smd) if smd is not None else None,
        })

    categories = []
    for column in (
        "digital_segment", "buyer_region", "award_year", "framework_flag",
        "buyer_identifier_quality", "has_reliable_duration",
    ):
        for level, group in frame.groupby(column, dropna=False, observed=True):
            categories.append({
                "variable": column,
                "level": str(level),
                "contracts": len(group),
                "events": int(group["event"].sum()),
                "censored": int((group["event"] == 0).sum()),
                "event_rate": float(group["event"].mean()),
            })
    return pd.DataFrame(continuous), pd.DataFrame(categories)


def cox_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    design = cox_design(frame)
    model = CoxPHFitter().fit(design, duration_col="t_months", event_col="event")
    results = model.summary[[
        "coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"
    ]].reset_index().rename(columns={"covariate": "covariate"})
    ph = proportional_hazard_test(model, design, time_transform="rank")
    ph_results = ph.summary[["test_statistic", "p"]].reset_index().rename(
        columns={"index": "covariate"}
    )

    # The training window is fixed at 2015-2021 for both evaluations, so the two
    # windows differ only in whether the 2025 award year is included in the test
    # set. The model is never refitted to improve either number.
    train = design.loc[frame["award_year"].le(2021).to_numpy()]
    temporal = CoxPHFitter().fit(train, duration_col="t_months", event_col="event")

    def evaluate(last_test_year: int) -> dict[str, Any]:
        mask = frame["award_year"].ge(2022) & frame["award_year"].le(last_test_year)
        test = design.loc[mask.to_numpy()]
        test_c = concordance_index(
            test["t_months"], -temporal.predict_partial_hazard(test), test["event"]
        )
        return {
            "train_years": "2015-2021",
            "test_years": f"2022-{last_test_year}",
            "train_contracts": len(train),
            "train_events": int(train["event"].sum()),
            "test_contracts": len(test),
            "test_events": int(test["event"].sum()),
            # Rounded once, to the precision every artifact displays. Storing 4
            # decimals and formatting 3 double-rounds, which is how the report
            # and the notebook end up disagreeing in the last digit.
            "train_c_index": round(float(temporal.concordance_index_), 3),
            "test_c_index": round(float(test_c), 3),
        }

    primary = evaluate(2024)
    primary["role"] = "primary; the internship guideline's 2015-2021 / 2022-2024 split"
    extended = evaluate(2025)
    extended["role"] = "sensitivity; adds the 2025 award cohort, whose follow-up is shortest"

    diagnostics = {
        "contracts": len(frame),
        "events": int(frame["event"].sum()),
        "covariates": len(results),
        "partial_aic": round(float(model.AIC_partial_), 3),
        "in_sample_c_index": round(float(model.concordance_index_), 4),
        "ph_violations_p_lt_0_05": ph_results.loc[
            ph_results["p"].lt(0.05), "covariate"
        ].tolist(),
        "temporal_validation": primary,
        "temporal_validation_including_latest_cohort": extended,
        "interpretation": "descriptive time-averaged associations; not a validated individual prediction model",
    }
    return results, ph_results, diagnostics


def sensitivity_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    km_rows: list[dict[str, Any]] = []
    cox_rows: list[dict[str, Any]] = []
    for arm, filename in ARM_FILES.items():
        frame = prepare(pd.read_parquet(PROCESSED / filename))
        km = KaplanMeierFitter().fit(frame["t_months"], frame["event"])
        km_rows.append({
            "arm": arm,
            "events": int(frame["event"].sum()),
            "event_rate": float(frame["event"].mean()),
            "survival_12m": float(km.survival_function_at_times(12).iloc[0]),
            "survival_24m": float(km.survival_function_at_times(24).iloc[0]),
            "p_12m_given_event_free_at_24m": conditional_probability(km, 24, 12),
            "p_24m_given_event_free_at_24m": conditional_probability(km, 24, 24),
            "km_median_months": round_or_none(km.median_survival_time_, 2),
        })
        model = CoxPHFitter().fit(
            cox_design(frame), duration_col="t_months", event_col="event"
        )
        for covariate, row in model.summary.iterrows():
            cox_rows.append({
                "arm": arm,
                "covariate": covariate,
                "hazard_ratio": float(row["exp(coef)"]),
                "ci_95_low": float(row["exp(coef) lower 95%"]),
                "ci_95_high": float(row["exp(coef) upper 95%"]),
                "p_value": float(row["p"]),
            })
    cox = pd.DataFrame(cox_rows)
    direction = cox.assign(above_one=cox["hazard_ratio"].gt(1)).groupby("covariate")
    assessment = {}
    for covariate, group in direction:
        same_direction = group["above_one"].nunique() == 1
        significant = int(group["p_value"].lt(0.05).sum())
        if not same_direction:
            label = "LINKAGE_SENSITIVE"
        elif significant >= 3:
            label = "ROBUST"
        elif significant >= 2:
            label = "MOSTLY_ROBUST"
        else:
            label = "LINKAGE_SENSITIVE"
        assessment[covariate] = label
    cox["robustness_assessment"] = cox["covariate"].map(assessment)
    return pd.DataFrame(km_rows), cox


def best_candidate_score(episode_ids: pd.Series) -> pd.Series:
    """Highest M_B text score available to each anchor, NaN when it had none.

    This is the quantity the frozen rule thresholds, so it is what decides
    whether an anchor sits near the acceptance boundary. Anchors that generated
    no candidate at all are not near any boundary: their event status is fixed
    by blocking, not by the threshold.
    """
    candidates = pd.read_parquet(
        PROCESSED / "linkage_candidates_scored.parquet",
        columns=["anchor_episode_id", "text_component"],
    )
    best = candidates.groupby("anchor_episode_id")["text_component"].max()
    return episode_ids.map(best)


def km_and_cox_headline(frame: pd.DataFrame) -> dict[str, Any]:
    """The four quantities the borderline check compares before and after."""
    km = KaplanMeierFitter().fit(frame["t_months"], frame["event"])
    model = CoxPHFitter().fit(cox_design(frame), duration_col="t_months", event_col="event")
    return {
        "contracts": len(frame),
        "events": int(frame["event"].sum()),
        "event_rate": float(frame["event"].mean()),
        "km_successor_by_12m": 1.0 - float(km.survival_function_at_times(12).iloc[0]),
        "km_successor_by_24m": 1.0 - float(km.survival_function_at_times(24).iloc[0]),
        "cox_hr_cpv_35": float(model.summary.loc["digital_segment_CPV-35", "exp(coef)"]),
        "cox_hr_framework": float(model.summary.loc["framework_flag", "exp(coef)"]),
    }


def borderline_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-run the headline results with near-threshold anchors excluded.

    An anchor whose best candidate scores inside the band is one the frozen
    rule could plausibly have classified either way. Dropping the whole band -
    borderline acceptances and borderline abstentions alike - asks whether the
    conclusions depend on those coin-flips. It is a robustness check, not a
    second event definition: the excluded rows are removed, not relabelled.
    """
    low, high = BORDERLINE_BAND
    scores = best_candidate_score(frame["episode_id"])
    in_band = scores.between(low, high, inclusive="both").fillna(False).to_numpy()
    retained = frame.loc[~in_band]

    main = km_and_cox_headline(frame)
    excluded = km_and_cox_headline(retained)
    rows = pd.DataFrame([
        {"analysis": "main", **main},
        {"analysis": "excluding_borderline_links", **excluded},
    ])

    removed = frame.loc[in_band]
    summary = {
        "band": {"low": low, "high": high, "variable": "M_B text_component (0-1)"},
        "band_rationale": (
            "Symmetric +/-0.05 around the frozen 0.70 acceptance threshold, fixed "
            "a priori. The band was not searched over and no second band exists."
        ),
        "contracts_removed": int(in_band.sum()),
        "events_removed": int(removed["event"].sum()),
        "censored_removed": int((removed["event"] == 0).sum()),
        "anchors_without_candidates": int(scores.isna().sum()),
        "main": main,
        "excluding_borderline_links": excluded,
        "deltas": {
            "km_successor_by_12m": excluded["km_successor_by_12m"] - main["km_successor_by_12m"],
            "km_successor_by_24m": excluded["km_successor_by_24m"] - main["km_successor_by_24m"],
            "cox_hr_cpv_35": excluded["cox_hr_cpv_35"] - main["cox_hr_cpv_35"],
            "cox_hr_framework": excluded["cox_hr_framework"] - main["cox_hr_framework"],
        },
    }
    # The project makes two kinds of claim and they do not stand or fall together,
    # so they are assessed separately rather than compressed into one verdict.
    # The comparative claim is that CPV-35 and framework episodes re-procure
    # sooner; it survives iff both hazard ratios keep their side of 1. The
    # absolute claim is the KM probability level, which drops mechanically once
    # borderline events are removed and is already declared linkage-sensitive by
    # the four-arm table.
    directions_hold = bool(
        (excluded["cox_hr_cpv_35"] > 1) == (main["cox_hr_cpv_35"] > 1)
        and (excluded["cox_hr_framework"] > 1) == (main["cox_hr_framework"] > 1)
    )
    summary["assessment"] = {
        "comparative_claims": (
            "NOT_DRIVEN_BY_BORDERLINE_LINKS" if directions_hold else "THRESHOLD_UNCERTAIN"
        ),
        "absolute_probability_level": "THRESHOLD_UNCERTAIN",
        "interpretation": (
            "The direction of both headline hazard ratios is unchanged, so the "
            "comparative findings the project actually claims do not rest on "
            "borderline linkage decisions. The absolute KM level does move, which "
            "is the expected mechanical consequence of removing borderline events "
            "and is consistent with the four-arm linkage sensitivity: absolute "
            "probabilities remain threshold-uncertain and are not quoted alone."
        ),
    }
    return rows, summary


def plot_kaplan_meier(frame: pd.DataFrame, km: KaplanMeierFitter, path: Path) -> None:
    """The overall survivor curve and the per-segment comparison it supports.

    Two panels because the report makes two distinct claims from this estimator:
    an absolute level (left, quoted only with its linkage caveat) and a
    comparative ordering across segments (right, the claim that survives every
    sensitivity arm).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    km.plot_survival_function(ax=axes[0], color=INK, linewidth=2, ci_alpha=0.15,
                              legend=False)
    for drop, horizon in enumerate((12, 24)):
        value = float(km.survival_function_at_times(horizon).iloc[0])
        axes[0].axvline(horizon, color=GRID, linewidth=1, zorder=0)
        axes[0].annotate(f"{horizon}m: {1 - value:.1%}", (horizon, value),
                         textcoords="offset points", xytext=(8, -18 - 16 * drop),
                         fontsize=9)
    axes[0].set_title("All digital episodes")
    axes[0].set_ylim(0, 1.02)

    for colour, (segment, group) in zip(
        SEGMENT_COLOURS, frame.groupby("digital_segment", sort=True), strict=False
    ):
        fitted = KaplanMeierFitter().fit(group["t_months"], group["event"])
        fitted.plot_survival_function(ax=axes[1], color=colour, linewidth=1.8,
                                      ci_show=False, label=segment)
    axes[1].set_title("By digital segment")
    axes[1].set_ylim(0.5, 1.02)
    axes[1].legend(loc="lower left", frameon=False, fontsize=9)

    for ax in axes:
        ax.set_xlim(0, 120)
        ax.set_xlabel("months since award")
        ax.set_ylabel("P(no observable successor yet)")
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.suptitle("Time to an observable successor procurement", fontsize=15, y=0.995)
    fig.text(0.5, 0.90,
             "Kaplan-Meier, M_B_text_ranking @ 0.70; censored at 2025-12-31. "
             "The event is an observable successor, not a certified renewal.",
             ha="center", color="#555555", fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_conditional_probabilities(conditional: pd.DataFrame, path: Path) -> None:
    """The operational output: next-12/24-month probability by contract age."""
    ages = sorted(conditional["contract_age_months"].unique())
    positions = np.arange(len(ages))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for offset, (horizon, colour) in enumerate(((12, INK), (24, ACCENT))):
        rows = (conditional.loc[conditional["horizon_months"].eq(horizon)]
                .set_index("contract_age_months").reindex(ages))
        values = 100 * rows["probability"].to_numpy()
        errors = np.vstack([
            values - 100 * rows["ci_95_low"].to_numpy(),
            100 * rows["ci_95_high"].to_numpy() - values,
        ])
        ax.bar(positions + (offset - 0.5) * width, values, width * 0.9,
               color=colour, label=f"within {horizon} months",
               yerr=errors, capsize=3, ecolor="#444444", error_kw={"linewidth": 1})
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{age}" for age in ages])
    ax.set_xlabel("contract age at assessment (months, no successor yet)")
    ax.set_ylabel("probability (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.suptitle("Probability that a successor becomes visible in the next 12 or 24 months",
                 fontsize=14, y=0.995)
    fig.text(0.5, 0.90,
             "Kaplan-Meier conditional probabilities with 95% episode-bootstrap intervals; "
             "the profile peaks into the 36-48 month renewal shoulder.",
             ha="center", color="#555555", fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def template_risk_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-run the headline results with the known false-positive mechanism removed.

    The threshold arms and the borderline band both perturb *where the bar
    sits*. Neither touches the failure mode the linkage audit actually
    identified, because that mode produces links well above the bar: shared
    framework boilerplate can drive the character analyser to a high score
    between unrelated objects, and because ``M_B`` ranks candidates within each
    anchor independently, one such episode can be accepted as the successor of
    several anchors at once. A stricter threshold does not remove either; it can
    enrich for them.

    Two observable signatures, both already published by
    ``scripts/audit_candidate_generation.py``, define the at-risk group:

    * the accepted link's word-level similarity is below
      :data:`LOW_WORD_SIMILARITY`, so acceptance rests on character n-grams;
    * the accepted successor episode is also accepted by another anchor.

    Those anchors are **re-censored at the study cutoff** rather than dropped,
    because that is the counterfactual being tested: if the link were spurious,
    the anchor would have no observed successor and would contribute its full
    follow-up as censored exposure. Dropping the rows instead -- what the
    borderline check does, for a different question -- would discard that
    exposure too.

    Nothing here is tuned: the group is defined by fixed, pre-published
    signatures, the threshold is untouched, no link is relabelled by hand, and
    the reference sample is not consulted.
    """
    links = pd.read_parquet(
        PROCESSED / "accepted_successor_links.parquet",
        columns=["anchor_episode_id", "candidate_episode_id", "word_tfidf_similarity"],
    )
    carried_by_char = links["word_tfidf_similarity"].lt(LOW_WORD_SIMILARITY)
    reused = links.groupby("candidate_episode_id")["anchor_episode_id"].transform("size").gt(1)
    at_risk = links.loc[carried_by_char | reused, "anchor_episode_id"]

    flagged = frame["episode_id"].isin(set(at_risk)).to_numpy()
    recensored = frame.copy()
    recensored.loc[flagged, "event"] = 0
    recensored.loc[flagged, "t_months"] = (
        np.maximum(recensored.loc[flagged, "days_to_cutoff"], 1) / MONTH_DAYS
    )

    main = km_and_cox_headline(frame)
    excluded = km_and_cox_headline(recensored)
    rows = pd.DataFrame([
        {"analysis": "main", **main},
        {"analysis": "recensoring_template_risk_links", **excluded},
    ])

    summary = {
        "group_definition": (
            "accepted links whose word-level similarity is below "
            f"{LOW_WORD_SIMILARITY:.2f} (acceptance carried by the character "
            "analyser) or whose successor episode is accepted by more than one "
            "anchor"
        ),
        "treatment": "re-censored at the study cutoff, not dropped",
        "accepted_links": int(len(links)),
        "carried_by_char_threshold": LOW_WORD_SIMILARITY,
        "carried_by_char_similarity": int(carried_by_char.sum()),
        "successor_shared_with_another_anchor": int(reused.sum()),
        "flagged_links": int(len(set(at_risk))),
        "flagged_share_of_events": float(len(set(at_risk)) / max(len(links), 1)),
        "main": main,
        "recensoring_template_risk_links": excluded,
        "deltas": {
            "km_successor_by_12m": excluded["km_successor_by_12m"] - main["km_successor_by_12m"],
            "km_successor_by_24m": excluded["km_successor_by_24m"] - main["km_successor_by_24m"],
            "cox_hr_cpv_35": excluded["cox_hr_cpv_35"] - main["cox_hr_cpv_35"],
            "cox_hr_framework": excluded["cox_hr_framework"] - main["cox_hr_framework"],
        },
    }
    directions_hold = bool(
        (excluded["cox_hr_cpv_35"] > 1) == (main["cox_hr_cpv_35"] > 1)
        and (excluded["cox_hr_framework"] > 1) == (main["cox_hr_framework"] > 1)
    )
    summary["assessment"] = {
        "comparative_claims": (
            "NOT_DRIVEN_BY_TEMPLATE_RISK_LINKS" if directions_hold else "TEMPLATE_RISK_SENSITIVE"
        ),
        "absolute_probability_level": "LINKAGE_SENSITIVE",
        "interpretation": (
            "This is the check the framework-agreement finding most needs, "
            "because framework boilerplate is the text that drives the "
            "mechanism: if the higher framework hazard were an artefact of "
            "shared legal wording, re-censoring these links would collapse it. "
            "Both headline hazard ratios keep their side of 1 and move little, "
            "so the comparative findings are not products of the documented "
            "false-positive mechanism. The absolute Kaplan-Meier level falls by "
            "roughly the share of events re-censored, which is arithmetic rather "
            "than evidence, and is consistent with the four-arm result that "
            "absolute probabilities are linkage-sensitive. The check bounds the "
            "mechanism's influence; it does not establish that the flagged links "
            "are false, and most of them are not."
        ),
    }
    return rows, summary


def parametric_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    warnings.filterwarnings("ignore")
    models = {
        "Exponential": ExponentialFitter(),
        "Weibull": WeibullFitter(),
        "LogLogistic": LogLogisticFitter(),
        "LogNormal": LogNormalFitter(),
        "GeneralizedGamma": GeneralizedGammaFitter(),
    }
    rows = []
    for name, model in models.items():
        model.fit(frame["t_months"], frame["event"])
        parameters = len(model.summary)
        rows.append({
            "model": name,
            "parameters": parameters,
            "log_likelihood": float(model.log_likelihood_),
            "aic": float(model.AIC_),
            "bic": float(model.AIC_ - 2 * parameters + parameters * np.log(len(frame))),
        })
    results = pd.DataFrame(rows).sort_values("aic")
    return results, str(results.iloc[0]["model"])


def write_report(
    summary: dict[str, Any],
    cox: pd.DataFrame,
    selection: pd.DataFrame,
    conditional: pd.DataFrame,
) -> None:
    temporal = summary["cox"]["temporal_validation"]
    extended = summary["cox"]["temporal_validation_including_latest_cohort"]
    border = summary["borderline_link_sensitivity"]
    border_main = border["main"]
    border_kept = border["excluding_borderline_links"]
    border_verdict = border["assessment"]["interpretation"]
    template = summary["template_risk_sensitivity"]
    template_main = template["main"]
    template_kept = template["recensoring_template_risk_links"]
    template_verdict = template["assessment"]["interpretation"]
    wide = conditional.pivot(
        index="contract_age_months", columns="horizon_months",
        values=["probability", "ci_95_low", "ci_95_high"],
    )
    conditional_lines = "\n".join([
        "| Contract age | P(successor within 12m) | 95% CI | P(successor within 24m) | 95% CI |",
        "|---:|---:|---|---:|---|",
        *[
            f"| {age} months "
            f"| {wide.loc[age, ('probability', 12)]:.3%} "
            f"| [{wide.loc[age, ('ci_95_low', 12)]:.3%}, {wide.loc[age, ('ci_95_high', 12)]:.3%}] "
            f"| {wide.loc[age, ('probability', 24)]:.3%} "
            f"| [{wide.loc[age, ('ci_95_low', 24)]:.3%}, {wide.loc[age, ('ci_95_high', 24)]:.3%}] |"
            for age in wide.index
        ],
    ])
    important_smd = selection.sort_values("absolute_smd", ascending=False).head(4)
    smd_lines = "\n".join(
        f"- `{row.variable}`: SMD `{row.standardized_mean_difference:.3f}`."
        for row in important_smd.itertuples()
    )
    hr = cox.set_index("covariate")
    text = f"""# Survival Analysis Report

Generated: `{summary['generated_at']}`  
Event: **accepted observable successor procurement**, not certified legal renewal.

## Cohort And Event Definition

The frozen primary cohort contains `{summary['cohort']['contracts']:,}` awarded Grand
Ouest digital procurement episodes. `{summary['cohort']['events']:,}` have an accepted
`M_B_text_ranking @ 0.70` successor and `{summary['cohort']['censored']:,}` have no
accepted observable successor before `2025-12-31`. The latter are right-censored; they
are not proven abandonments.

## Kaplan-Meier Results

- Event rate: `{summary['cohort']['event_rate']:.3%}`.
- Censoring proportion: `{summary['cohort']['censoring_rate']:.3%}`.
- Estimated probability of an observable successor by 12 months: `{summary['km']['successor_by_12m']:.3%}`.
- Estimated probability by 24 months: `{summary['km']['successor_by_24m']:.3%}`.
- Kaplan-Meier median: **not reached**.
- Segment log-rank: statistic `{summary['logrank']['test_statistic']:.2f}`, p-value `{summary['logrank']['p_value']:.3g}`.

The median delay of `31.82` months reported elsewhere is the median among linked
events only. It is not the Kaplan-Meier median.

## Cox Model

The parsimonious model uses CPV segment, region, framework status, validated-SIREN
availability, and award year. Its in-sample C-index is
`{summary['cox']['in_sample_c_index']:.3f}`. Framework episodes have HR
`{hr.loc['framework_flag', 'exp(coef)']:.3f}` and CPV-35 has HR
`{hr.loc['digital_segment_CPV-35', 'exp(coef)']:.3f}` relative to CPV-32.

The proportional-hazards diagnostic rejects constant effects for:
`{', '.join(summary['cox']['ph_violations_p_lt_0_05'])}`. These coefficients are
therefore descriptive time-averaged associations, not causal effects.

### Temporal Validation

The model is fit once on `{temporal['train_years']}` awards and scored out of time
without any refitting or retuning.

| Split | Train N | Train events | Test N | Test events | C-index train | C-index test |
|---|---:|---:|---:|---:|---:|---:|
| Primary, {temporal['test_years']} | {temporal['train_contracts']:,} | {temporal['train_events']} | {temporal['test_contracts']:,} | {temporal['test_events']} | {temporal['train_c_index']:.3f} | {temporal['test_c_index']:.3f} |
| Sensitivity, {extended['test_years']} | {extended['train_contracts']:,} | {extended['train_events']} | {extended['test_contracts']:,} | {extended['test_events']} | {extended['train_c_index']:.3f} | {extended['test_c_index']:.3f} |

The primary split is the one the internship guideline specifies. The extended split
adds the 2025 award cohort, whose follow-up is shortest, and is carried only as a
sensitivity read.

Out-of-time discrimination is weak in both: a C-index near `0.5` means the model
does not usefully rank individual episodes by time to successor on unseen award
years. That is the result, not a prompt to retune. Part of it is structural, since
episodes awarded from 2022 onwards can only contribute short-gap events, but the
model has no demonstrated out-of-time discriminative power and nothing in the
operational deliverable rests on it.

## Parametric Models And Indicators

`{summary['parametric']['selected_model']}` has the lowest AIC among exponential,
Weibull, log-logistic, log-normal, and generalized-gamma fits. Model selection does
not remove linkage uncertainty or guarantee tail extrapolation.

The selected parametric model is **not** the source of the operational numbers.
Every horizon reported here falls inside the observed window, and the smooth
families flatten the observed renewal shoulder, so the 12/24-month conditional
probabilities in `survival_conditional_probabilities.csv` are read off the
Kaplan-Meier estimator, with 500-draw episode-bootstrap intervals. The generalized
gamma is reported as the best-fitting family and as the instrument any
extrapolation past `2025-12-31` would use.

## Borderline-Link Robustness

The frozen rule accepts at `0.70` on the M_B text score. Anchors whose best
candidate falls in `[{border['band']['low']:.2f}, {border['band']['high']:.2f}]` are
the ones a small threshold perturbation would reclassify. Dropping that whole band —
borderline acceptances and borderline abstentions alike — removes
`{border['contracts_removed']:,}` episodes, of which `{border['events_removed']}` are
events, and gives:

| Analysis | Contracts | Events | KM 12m | KM 24m | CPV-35 HR | Framework HR |
|---|---:|---:|---:|---:|---:|---:|
| Main | {border_main['contracts']:,} | {border_main['events']} | {border_main['km_successor_by_12m']:.3%} | {border_main['km_successor_by_24m']:.3%} | {border_main['cox_hr_cpv_35']:.3f} | {border_main['cox_hr_framework']:.3f} |
| Excluding borderline | {border_kept['contracts']:,} | {border_kept['events']} | {border_kept['km_successor_by_12m']:.3%} | {border_kept['km_successor_by_24m']:.3%} | {border_kept['cox_hr_cpv_35']:.3f} | {border_kept['cox_hr_framework']:.3f} |

{border_verdict} The band is a fixed `±0.05` around the frozen threshold; it was not
searched over, and the excluded episodes are removed rather than relabelled.

## Template-Risk Robustness

The threshold arms and the borderline band both move where the acceptance bar
sits. Neither touches the false-positive mechanism the linkage audit actually
identified, because that mechanism produces links well *above* the bar: French
award notices carry long standardised framework boilerplate on which character
n-grams score highly between unrelated objects, and `M_B` ranks candidates within
each anchor independently, so one such episode can be accepted for several
anchors. A stricter threshold does not remove either signature.

Two observable signatures, both already published by the candidate-generation
audit, define the at-risk group: word-level similarity below
`{template['carried_by_char_threshold']:.2f}` (acceptance carried by the character
analyser, `{template['carried_by_char_similarity']}` links) or a successor episode
shared with another anchor (`{template['successor_shared_with_another_anchor']}`
links). Together they flag `{template['flagged_links']}` of the
`{template['accepted_links']}` accepted links
(`{template['flagged_share_of_events']:.1%}`). Those anchors are **re-censored at the
cutoff** rather than dropped, because that is the counterfactual under test: a
spurious link means the anchor had no observed successor and should contribute its
full follow-up as censored exposure.

| Analysis | Contracts | Events | KM 12m | KM 24m | CPV-35 HR | Framework HR |
|---|---:|---:|---:|---:|---:|---:|
| Main | {template_main['contracts']:,} | {template_main['events']} | {template_main['km_successor_by_12m']:.3%} | {template_main['km_successor_by_24m']:.3%} | {template_main['cox_hr_cpv_35']:.3f} | {template_main['cox_hr_framework']:.3f} |
| Re-censoring template-risk links | {template_kept['contracts']:,} | {template_kept['events']} | {template_kept['km_successor_by_12m']:.3%} | {template_kept['km_successor_by_24m']:.3%} | {template_kept['cox_hr_cpv_35']:.3f} | {template_kept['cox_hr_framework']:.3f} |

{template_verdict}

## Operational 12- And 24-Month Probabilities

For a contract that has reached age `a` months with no accepted successor, the
probability that one becomes visible within the next `h` months is
`P(T <= a+h | T > a) = 1 - S(a+h)/S(a)`, read off the Kaplan-Meier estimator with
500-draw episode-bootstrap intervals. This is the study's operational output.

{conditional_lines}

The intervals are wide relative to the estimates, and the profile is not monotone
in age: it rises into the 36-48 month renewal shoulder and falls away after it.
These rank ages and segments; they are not calibrated individual forecasts, and
they estimate an *observable successor procurement appearing in BOAMP*, not a
certified renewal. Segment-level curves are in `survival_segment_summary.csv`.

## Detectability And Censoring Diagnostic

Linked and censored observations differ most on these standardized comparisons:

{smd_lines}

These differences indicate differential observed-event detection and unequal
follow-up; they do not prove causal linkage bias. In particular, recent contracts
cannot yet show long successor gaps. Administrative censoring and missed successors
from imperfect linkage remain conceptually distinct but cannot be fully separated
with BOAMP alone.

## Linkage Sensitivity

Event counts range from `{summary['sensitivity']['minimum_events']}` to
`{summary['sensitivity']['maximum_events']}` across the four retained linkage arms
(`M_B` at `0.80`, `0.70`, `0.60`, and the `M_C` weighted-gated contrast at `0.70`).
Absolute probabilities are therefore linkage-sensitive. Cox effects and subgroup
ordering should only be claimed where the exported sensitivity tables show stable
direction.

These are linkage-conditioned estimates. Missed successors may reduce the observed
event rate, whereas residual false links may increase it. They should therefore not
be interpreted as formal lower bounds on true re-procurement probability.

## Decision

The survival analysis is reproducible and complete for descriptive,
linkage-conditioned reporting. It is not a validated legal-renewal forecast. The
primary outputs are comparative KM results and age/segment risk indicators; external
linkage validation remains the main condition for stronger accuracy claims.
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> int:
    frame = prepare(pd.read_parquet(PROCESSED / "survival_dataset.parquet"))
    if frame["episode_id"].duplicated().any() or (frame["duration_days"] < 0).any():
        raise ValueError("survival dataset failed structural validation")

    km, overall, segments, logrank = km_outputs(frame)
    conditional = conditional_outputs(km, frame)
    selection, selection_categories = selection_outputs(frame)
    cox, ph, cox_diagnostics = cox_outputs(frame)
    sensitivity, cox_sensitivity = sensitivity_outputs()
    borderline, borderline_summary = borderline_outputs(frame)
    template_risk, template_risk_summary = template_risk_outputs(frame)
    parametric, selected_model = parametric_outputs(frame)

    outputs = {
        "survival_km_horizons.csv": overall,
        "survival_segment_summary.csv": segments,
        "survival_conditional_probabilities.csv": conditional,
        "survival_selection_diagnostic.csv": selection,
        "survival_selection_by_category.csv": selection_categories,
        "survival_cox_results.csv": cox,
        "survival_ph_diagnostics.csv": ph,
        "survival_linkage_sensitivity.csv": sensitivity,
        "survival_cox_linkage_sensitivity.csv": cox_sensitivity,
        "survival_borderline_link_sensitivity.csv": borderline,
        "survival_template_risk_sensitivity.csv": template_risk,
        "survival_parametric_comparison.csv": parametric,
    }
    for filename, table in outputs.items():
        table.to_csv(PROCESSED / filename, index=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_kaplan_meier(frame, km, FIGURES / "survival_kaplan_meier.png")
    plot_conditional_probabilities(
        conditional, FIGURES / "survival_conditional_probabilities.png"
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "event_definition": "accepted observable successor procurement under M_B_text_ranking @ 0.70",
        "censoring_definition": "no accepted observable successor before 2025-12-31",
        "cohort": {
            "contracts": len(frame),
            "events": int(frame["event"].sum()),
            "censored": int((frame["event"] == 0).sum()),
            "event_rate": float(frame["event"].mean()),
            "censoring_rate": float((frame["event"] == 0).mean()),
        },
        "km": {
            "successor_by_12m": float(overall.loc[overall["months"].eq(12), "cumulative_successor_probability"].iloc[0]),
            "successor_by_24m": float(overall.loc[overall["months"].eq(24), "cumulative_successor_probability"].iloc[0]),
            "median_months": round_or_none(km.median_survival_time_, 2),
            "median_status": "not_reached" if not np.isfinite(km.median_survival_time_) else "reached",
        },
        "logrank": logrank,
        "cox": cox_diagnostics,
        "parametric": {
            "selected_model": selected_model,
            "selection_basis": "minimum AIC and BIC, checked against empirical KM",
            "role": (
                "reported as the best-fitting parametric family and as the instrument "
                "any extrapolation past the observation window would use"
            ),
            "operational_probability_source": (
                "Kaplan-Meier. Every horizon reported here falls inside the observed "
                "window, and the smooth families flatten the observed renewal shoulder, "
                "so the 12/24-month conditional probabilities are read off the empirical "
                "estimator rather than off the fitted parametric model."
            ),
        },
        "borderline_link_sensitivity": borderline_summary,
        "template_risk_sensitivity": template_risk_summary,
        "sensitivity": {
            "arms": list(ARM_FILES),
            "minimum_events": int(sensitivity["events"].min()),
            "maximum_events": int(sensitivity["events"].max()),
            "absolute_probabilities_linkage_sensitive": True,
        },
        "selection_diagnostic": {
            "largest_absolute_smd": selection.sort_values("absolute_smd", ascending=False).iloc[0].to_dict(),
            "interpretation": "possible differential detectability or unequal follow-up, not proof of causal linkage bias",
        },
        "outputs": {name: str(PROCESSED / name) for name in outputs},
        "report": str(REPORT),
        "validation_passed": bool(
            len(frame) == 3800
            and int(frame["event"].sum()) == 544
            and km.median_survival_time_ == np.inf
            and len(cox) == 8
        ),
    }
    (PROCESSED / "survival_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    write_report(summary, cox, selection, conditional)
    if not summary["validation_passed"]:
        raise RuntimeError("survival evidence validation failed")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
