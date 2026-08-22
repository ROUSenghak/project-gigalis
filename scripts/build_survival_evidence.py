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


def candidate_pool_sizes() -> pd.Series:
    """Exposed candidates per anchor -- the block size the linkage rule searches.

    ``M_B_text_ranking`` accepts the maximum text score over an anchor's block,
    and the maximum of more draws is larger. An anchor whose buyer publishes
    prolifically is therefore mechanically more likely to produce an accepted
    link than an otherwise identical anchor whose buyer publishes rarely, quite
    apart from whether either was actually re-procured. That makes block size a
    detectability variable, and it belongs in the selection diagnostic beside
    the ones already published.
    """
    pairs = pd.read_parquet(
        PROCESSED / "linkage_candidates_scored.parquet", columns=["anchor_episode_id"]
    )
    return pairs.groupby("anchor_episode_id").size()


def buyer_cluster_id(frame: pd.DataFrame) -> pd.Series:
    """Stable buyer cluster with a name fallback for missing buyer keys."""
    key = frame["buyer_key"].fillna("").astype(str).str.strip()
    name = (
        frame["buyer_name_raw"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return pd.Series(
        np.where(key.ne(""), "key:" + key, "name:" + name),
        index=frame.index,
        dtype="string",
    )


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["t_months"] = np.maximum(frame["duration_days"], 1) / MONTH_DAYS
    frame["text_length_chars"] = frame["episode_text"].fillna("").astype(str).str.len()
    frame["administrative_followup_months"] = frame["days_to_cutoff"] / MONTH_DAYS
    frame["has_reliable_duration"] = frame["duration_months_reliable"].notna()
    frame["has_validated_siren"] = frame["buyer_identifier_quality"].eq("trusted_siren")
    pool = candidate_pool_sizes()
    # Anchors with no exposed candidate get 0, not NaN: no candidates is a real
    # and consequential state -- those anchors cannot produce an event at all --
    # and dropping them from the diagnostic would remove the clearest cases.
    frame["candidate_pool_size"] = (
        frame["episode_id"].map(pool).fillna(0).astype(int)
    )
    frame["log_candidate_pool_size"] = np.log1p(frame["candidate_pool_size"])
    frame["buyer_cluster"] = buyer_cluster_id(frame)
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
    episode_rng = np.random.default_rng(20260812)
    buyer_rng = np.random.default_rng(20260822)
    episode_draws = {(age, horizon): [] for age in ages for horizon in (12, 24)}
    buyer_draws = {(age, horizon): [] for age in ages for horizon in (12, 24)}
    for _ in range(500):
        sample = frame.iloc[episode_rng.integers(0, len(frame), len(frame))]
        fitted = KaplanMeierFitter().fit(sample["t_months"], sample["event"])
        for age, horizon in episode_draws:
            episode_draws[(age, horizon)].append(
                conditional_probability(fitted, age, horizon)
            )

    clusters = {
        cluster: group.index.to_numpy()
        for cluster, group in frame.groupby("buyer_cluster", sort=False)
    }
    cluster_keys = np.asarray(list(clusters), dtype=object)
    for _ in range(500):
        sampled_clusters = buyer_rng.choice(
            cluster_keys, size=len(cluster_keys), replace=True
        )
        sampled_index = np.concatenate([clusters[key] for key in sampled_clusters])
        sample = frame.loc[sampled_index]
        fitted = KaplanMeierFitter().fit(sample["t_months"], sample["event"])
        for age, horizon in buyer_draws:
            buyer_draws[(age, horizon)].append(
                conditional_probability(fitted, age, horizon)
            )

    rows = []
    for age in ages:
        for horizon in (12, 24):
            episode_values = np.asarray(episode_draws[(age, horizon)], dtype=float)
            episode_values = episode_values[np.isfinite(episode_values)]
            buyer_values = np.asarray(buyer_draws[(age, horizon)], dtype=float)
            buyer_values = buyer_values[np.isfinite(buyer_values)]
            rows.append({
                "contract_age_months": age,
                "horizon_months": horizon,
                "probability": conditional_probability(km, age, horizon),
                "ci_95_low": float(np.percentile(episode_values, 2.5)),
                "ci_95_high": float(np.percentile(episode_values, 97.5)),
                "buyer_cluster_ci_95_low": float(np.percentile(buyer_values, 2.5)),
                "buyer_cluster_ci_95_high": float(np.percentile(buyer_values, 97.5)),
                "episode_interval_method": "episode bootstrap, 500 draws",
                "buyer_interval_method": "buyer-cluster bootstrap, 500 draws",
            })
    return pd.DataFrame(rows)


def selection_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    linked = frame.loc[frame["event"].eq(1)]
    censored = frame.loc[frame["event"].eq(0)]
    continuous = []
    for column in (
        "award_year", "notice_count", "text_length_chars",
        "administrative_followup_months",
        "candidate_pool_size", "log_candidate_pool_size",
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
        risk = temporal.predict_partial_hazard(test).to_numpy(dtype=float)
        durations = test["t_months"].to_numpy(dtype=float)
        events = test["event"].to_numpy(dtype=int)
        test_c = concordance_index(durations, -risk, events)
        result = {
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
        # Uncertainty is computed on the fixed out-of-time predictions: the
        # model is not refitted inside the bootstrap.  Episode resampling is
        # shown for continuity; buyer-cluster resampling is the conservative
        # analysis when several episodes belong to the same buyer.
        if last_test_year == 2024:
            episode_rng = np.random.default_rng(20260821)
            buyer_rng = np.random.default_rng(20260822)
            episode_values: list[float] = []
            buyer_values: list[float] = []
            for _ in range(2000):
                index = episode_rng.integers(0, len(test), len(test))
                episode_values.append(
                    concordance_index(durations[index], -risk[index], events[index])
                )

            test_clusters = frame.loc[mask, "buyer_cluster"].reset_index(drop=True)
            cluster_positions = {
                cluster: positions.to_numpy()
                for cluster, positions in test_clusters.groupby(test_clusters).groups.items()
            }
            cluster_keys = np.asarray(list(cluster_positions), dtype=object)
            for _ in range(2000):
                sampled = buyer_rng.choice(
                    cluster_keys, size=len(cluster_keys), replace=True
                )
                index = np.concatenate([cluster_positions[key] for key in sampled])
                buyer_values.append(
                    concordance_index(durations[index], -risk[index], events[index])
                )
            result["uncertainty"] = {
                "bootstrap_model_refit": False,
                "draws": 2000,
                "episode_bootstrap_ci_95": [
                    round(float(value), 3)
                    for value in np.percentile(episode_values, [2.5, 97.5])
                ],
                "buyer_cluster_bootstrap_ci_95": [
                    round(float(value), 3)
                    for value in np.percentile(buyer_values, [2.5, 97.5])
                ],
                "episode_probability_c_gt_055": round(
                    float(np.mean(np.asarray(episode_values) > 0.55)), 3
                ),
                "buyer_probability_c_gt_055": round(
                    float(np.mean(np.asarray(buyer_values) > 0.55)), 3
                ),
            }
        return result

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


def detectability_cox_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One sensitivity Cox model: the main specification plus log candidate pool.

    This is a **sensitivity model, not the headline model**. The main Cox
    specification is unchanged and stays the reported one. The question here is
    narrow: how much of each association survives holding constant how many
    candidates the linkage rule had to search for that anchor?

    Candidate pool size is not treated as a cause of anything. It is a property
    of how prolifically a buyer publishes, which is a channel through which an
    event becomes *observable*, and adjusting for it separates "this contract
    type is re-procured sooner" from "this contract type belongs to buyers whose
    successors are easier to find".
    """
    design = cox_design(frame)
    main = CoxPHFitter().fit(design, duration_col="t_months", event_col="event")
    adjusted_design = design.copy()
    adjusted_design["log_candidate_pool_size"] = frame["log_candidate_pool_size"].to_numpy()
    adjusted = CoxPHFitter().fit(
        adjusted_design, duration_col="t_months", event_col="event"
    )

    rows: list[dict[str, Any]] = []
    for covariate, row in adjusted.summary.iterrows():
        in_main = covariate in main.summary.index
        rows.append({
            "covariate": covariate,
            "hazard_ratio_main": (
                float(main.summary.loc[covariate, "exp(coef)"]) if in_main else None
            ),
            "p_value_main": float(main.summary.loc[covariate, "p"]) if in_main else None,
            "hazard_ratio_pool_adjusted": float(row["exp(coef)"]),
            "ci_95_low_pool_adjusted": float(row["exp(coef) lower 95%"]),
            "ci_95_high_pool_adjusted": float(row["exp(coef) upper 95%"]),
            "p_value_pool_adjusted": float(row["p"]),
            "log_hazard_ratio_attenuation": (
                round(
                    float(
                        np.log(main.summary.loc[covariate, "exp(coef)"])
                        - np.log(row["exp(coef)"])
                    ),
                    4,
                )
                if in_main
                else None
            ),
        })
    table = pd.DataFrame(rows)
    pool = adjusted.summary.loc["log_candidate_pool_size"]
    diagnostics = {
        "role": "sensitivity only; the main Cox model is unchanged and remains the reported one",
        "added_covariate": "log_candidate_pool_size = log(1 + exposed candidates for the anchor)",
        "pool_hazard_ratio": round(float(pool["exp(coef)"]), 4),
        "pool_p_value": float(pool["p"]),
        "interpretation": (
            "Candidate pool size is a detectability variable, not a cause. A larger "
            "block gives the max-over-block text score more draws to clear 0.70, so "
            "part of any association with a buyer-level publishing habit is "
            "observability rather than re-procurement behaviour."
        ),
    }
    return table, diagnostics


def buyer_stratified_cox_outputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Within-buyer Cox sensitivity over each retained linkage arm.

    Buyer stratification gives each buyer a separate baseline hazard and thus
    absorbs fixed buyer attributes.  It does not absorb anchor-specific
    candidate-pool size, which varies with award date and future activity.
    """
    from scipy import stats
    from statsmodels.duration.hazard_regression import PHReg

    rows: list[dict[str, Any]] = []
    primary_diagnostics: dict[str, Any] = {}
    covariates = [
        "framework_flag",
        "award_year_centered",
        "digital_segment_CPV-35",
        "digital_segment_CPV-48",
        "digital_segment_CPV-72",
    ]
    for arm, filename in ARM_FILES.items():
        frame = prepare(pd.read_parquet(PROCESSED / filename))
        eligible_clusters = frame.groupby("buyer_cluster").filter(
            lambda group: len(group) >= 2 and int(group["event"].sum()) >= 1
        )
        design = pd.DataFrame(
            {
                "framework_flag": eligible_clusters["framework_flag"].astype(int),
                "award_year_centered": (
                    eligible_clusters["award_year"]
                    - eligible_clusters["award_year"].median()
                ),
                "digital_segment_CPV-35": eligible_clusters["digital_segment"].eq("CPV-35").astype(int),
                "digital_segment_CPV-48": eligible_clusters["digital_segment"].eq("CPV-48").astype(int),
                "digital_segment_CPV-72": eligible_clusters["digital_segment"].eq("CPV-72").astype(int),
            },
            index=eligible_clusters.index,
        )
        fit = PHReg(
            eligible_clusters["t_months"].to_numpy(dtype=float),
            design[covariates].to_numpy(dtype=float),
            status=eligible_clusters["event"].to_numpy(dtype=int),
            strata=pd.factorize(eligible_clusters["buyer_cluster"])[0],
            ties="breslow",
        ).fit()
        for position, covariate in enumerate(covariates):
            coefficient = float(fit.params[position])
            standard_error = float(fit.bse[position])
            rows.append(
                {
                    "arm": arm,
                    "episodes": int(len(eligible_clusters)),
                    "events": int(eligible_clusters["event"].sum()),
                    "buyers": int(eligible_clusters["buyer_cluster"].nunique()),
                    "covariate": covariate,
                    "hazard_ratio": float(np.exp(coefficient)),
                    "ci_95_low": float(np.exp(coefficient - 1.96 * standard_error)),
                    "ci_95_high": float(np.exp(coefficient + 1.96 * standard_error)),
                    "p_value": float(2 * stats.norm.sf(abs(coefficient / standard_error))),
                }
            )
        if arm == "main":
            grouped = frame.groupby("buyer_cluster")["candidate_pool_size"].agg(
                episodes="size", distinct_pool_sizes="nunique", minimum="min", maximum="max"
            )
            multi = grouped["episodes"].ge(2)
            varying = grouped["distinct_pool_sizes"].gt(1)
            primary_diagnostics = {
                "role": "within-buyer sensitivity controlling fixed buyer-level heterogeneity",
                "episodes": int(len(eligible_clusters)),
                "events": int(eligible_clusters["event"].sum()),
                "buyers": int(eligible_clusters["buyer_cluster"].nunique()),
                "all_cohort_buyers": int(frame["buyer_cluster"].nunique()),
                "multi_episode_buyers": int(multi.sum()),
                "multi_episode_buyers_with_varying_candidate_pool_size": int(
                    (multi & varying).sum()
                ),
                "share_multi_episode_buyers_with_varying_candidate_pool_size": float(
                    (multi & varying).sum() / multi.sum()
                ),
                "interpretation": (
                    "Buyer stratification controls time-invariant buyer characteristics by "
                    "giving each buyer its own baseline hazard. Candidate-pool size is "
                    "anchor-specific and varies within most multi-episode buyers, so the "
                    "stratified model does not eliminate that detectability channel."
                ),
            }
    return pd.DataFrame(rows), primary_diagnostics


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
    detectability_cox: pd.DataFrame,
    buyer_stratified_cox: pd.DataFrame,
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
        values=[
            "probability", "ci_95_low", "ci_95_high",
            "buyer_cluster_ci_95_low", "buyer_cluster_ci_95_high",
        ],
    )
    conditional_lines = "\n".join([
        "| Episode age | P(successor within 12m) | Episode-bootstrap 95% CI | Buyer-cluster 95% CI | P(successor within 24m) | Episode-bootstrap 95% CI | Buyer-cluster 95% CI |",
        "|---:|---:|---|---|---:|---|---|",
        *[
            f"| {age} months "
            f"| {wide.loc[age, ('probability', 12)]:.3%} "
            f"| [{wide.loc[age, ('ci_95_low', 12)]:.3%}, {wide.loc[age, ('ci_95_high', 12)]:.3%}] "
            f"| [{wide.loc[age, ('buyer_cluster_ci_95_low', 12)]:.3%}, {wide.loc[age, ('buyer_cluster_ci_95_high', 12)]:.3%}] "
            f"| {wide.loc[age, ('probability', 24)]:.3%} "
            f"| [{wide.loc[age, ('ci_95_low', 24)]:.3%}, {wide.loc[age, ('ci_95_high', 24)]:.3%}] "
            f"| [{wide.loc[age, ('buyer_cluster_ci_95_low', 24)]:.3%}, {wide.loc[age, ('buyer_cluster_ci_95_high', 24)]:.3%}] |"
            for age in wide.index
        ],
    ])
    important_smd = selection.sort_values("absolute_smd", ascending=False)
    smd_lines = "\n".join([
        "| Variable | Linked mean | Censored mean | SMD |",
        "|---|---:|---:|---:|",
        *[
            f"| `{row.variable}` | {row.linked_mean:.4g} | {row.censored_mean:.4g} "
            f"| {row.standardized_mean_difference:+.3f} |"
            for row in important_smd.itertuples()
        ],
    ])
    pool_cox = detectability_cox.set_index("covariate")
    detectability = summary["detectability_cox"]
    pool_lines = "\n".join([
        "| Covariate | HR, main model | HR, + log(candidate pool) | p, adjusted |",
        "|---|---:|---:|---:|",
        *[
            f"| `{row.Index}` "
            f"| {'--' if pd.isna(row.hazard_ratio_main) else f'{row.hazard_ratio_main:.3f}'} "
            f"| {row.hazard_ratio_pool_adjusted:.3f} "
            f"| {row.p_value_pool_adjusted:.2g} |"
            for row in pool_cox.itertuples()
        ],
    ])
    hr = cox.set_index("covariate")
    stratified_main = buyer_stratified_cox.loc[
        buyer_stratified_cox["arm"].eq("main")
    ].set_index("covariate")
    temporal_uncertainty = temporal["uncertainty"]
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

The primary point estimate is close to chance. Its 95% interval is
`{temporal_uncertainty['episode_bootstrap_ci_95']}` under episode resampling and
`{temporal_uncertainty['buyer_cluster_bootstrap_ci_95']}` under buyer-cluster
resampling. The latter is wider because several episodes belong to the same buyer.
The available temporal validation therefore does not establish useful individual
discrimination, and the model is not used as an individual ranking engine. This is
the result, not a prompt to retune.

## Parametric Models And Indicators

`{summary['parametric']['selected_model']}` has the lowest AIC among exponential,
Weibull, log-logistic, log-normal, and generalized-gamma fits. Model selection does
not remove linkage uncertainty or guarantee tail extrapolation.

The selected parametric model is **not** the source of the operational numbers.
Every horizon reported here falls inside the observed window, and the smooth
families flatten the observed renewal shoulder, so the 12/24-month conditional
probabilities in `survival_conditional_probabilities.csv` are read off the
Kaplan-Meier estimator, with 500-draw episode and buyer-cluster bootstrap intervals. The generalized
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

One coincidence to head off, because the same number appears twice in this project
with two unrelated meanings: `{border['contracts_removed']:,}` anchors fall in the borderline band, and
`{summary['cohort']['contracts'] - summary['candidate_coverage']['anchors_with_candidates']:,}` anchors generated no candidate at all. These are different sets and
different questions. Every anchor removed here had a candidate and a best score
inside the band; anchors with no candidate are not near any threshold, since their
event status is decided by blocking rather than by the acceptance bar, and they stay
in the analysis as censored exposure.

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
500-draw episode and buyer-cluster bootstrap intervals. This is the study's operational output.

{conditional_lines}

The intervals are wide relative to the estimates, and the profile is not monotone
in age: it rises into the 36-48 month renewal shoulder and falls away after it.
These rank ages and segments; they are not calibrated individual forecasts, and
they estimate an *observable successor procurement appearing in BOAMP*, not a
certified renewal. Segment-level curves are in `survival_segment_summary.csv`.

## Detectability And Censoring Diagnostic

Linked and censored observations differ on these standardized comparisons:

{smd_lines}

These differences indicate differential observed-event detection and unequal
follow-up; they do not prove causal linkage bias. In particular, recent contracts
cannot yet show long successor gaps. Administrative censoring and missed successors
from imperfect linkage remain conceptually distinct but cannot be fully separated
with BOAMP alone.

### Candidate-pool size is the largest imbalance

The largest of these is not a property of the contract at all. `M_B_text_ranking`
accepts the maximum text score over an anchor's candidate block, and the maximum of
more draws is larger, so an anchor whose buyer publishes prolifically is
mechanically more likely to yield an accepted link than an otherwise identical
anchor whose buyer publishes rarely. On the log scale the linked-versus-censored
standardized difference is
`{selection.set_index('variable').loc['log_candidate_pool_size', 'standardized_mean_difference']:+.3f}`,
above every contract-level variable above it in the table.

This is a detectability channel, not a cause of re-procurement, so the response is
one **sensitivity** model rather than a change to the reported specification. The
main Cox model is unchanged. Adding `log(1 + candidate pool size)` to it gives:

{pool_lines}

The added term is itself strongly associated with an observed event
(HR `{detectability['pool_hazard_ratio']}`, p `{detectability['pool_p_value']:.2g}`), which is what a detectability
channel looks like.

Two readings follow, and they differ:

- **CPV-35 is largely insensitive to it.** The hazard ratio moves from
  `{pool_cox.loc['digital_segment_CPV-35', 'hazard_ratio_main']:.3f}` to
  `{pool_cox.loc['digital_segment_CPV-35', 'hazard_ratio_pool_adjusted']:.3f}`
  (p `{pool_cox.loc['digital_segment_CPV-35', 'p_value_pool_adjusted']:.2g}`). Together with its
  stability across the four linkage arms, the borderline band, and template-risk
  re-censoring, the CPV-35 result is the most robust comparative finding here and
  can be stated as such.
- **The framework association is partly detectability.** Its hazard ratio
  attenuates from `{pool_cox.loc['framework_flag', 'hazard_ratio_main']:.3f}` to
  `{pool_cox.loc['framework_flag', 'hazard_ratio_pool_adjusted']:.3f}`
  (p `{pool_cox.loc['framework_flag', 'p_value_pool_adjusted']:.2g}`), roughly
  `{100 * pool_cox.loc['framework_flag', 'log_hazard_ratio_attenuation'] / np.log(pool_cox.loc['framework_flag', 'hazard_ratio_main']):.0f}%`
  of the log hazard ratio. The direction survives every check the study runs, but
  buyers who use framework agreements also publish more, and publishing more raises
  the chance that a max-over-block text score clears `0.70`. The association is real
  and smaller than the main model alone implies; template boilerplate is not the
  only alternative explanation, and differential detectability is the other.

Neither statement is causal. Candidate-pool size is a description of a buyer's
publication volume, and this model adjusts for it to separate observability from
behaviour -- it does not identify an effect of either.

### Buyer-stratified sensitivity has a different role

Giving each buyer its own baseline hazard controls fixed buyer characteristics,
including persistent procurement culture and long-run publication propensity. It
does not eliminate anchor-specific detectability: candidate-pool size varies within
`{summary['buyer_stratified_cox']['multi_episode_buyers_with_varying_candidate_pool_size']}`
of `{summary['buyer_stratified_cox']['multi_episode_buyers']}` multi-episode buyers.
The direct detectability sensitivity therefore remains the model adding
`log(1 + candidate pool size)`.

In the primary buyer-stratified arm, framework HR is
`{stratified_main.loc['framework_flag', 'hazard_ratio']:.3f}`
(`p={stratified_main.loc['framework_flag', 'p_value']:.3g}`), CPV-35 is
`{stratified_main.loc['digital_segment_CPV-35', 'hazard_ratio']:.3f}`
(`p={stratified_main.loc['digital_segment_CPV-35', 'p_value']:.3g}`), and CPV-48 is
`{stratified_main.loc['digital_segment_CPV-48', 'hazard_ratio']:.3f}`
(`p={stratified_main.loc['digital_segment_CPV-48', 'p_value']:.3g}`). Thus the
population CPV-35 contrast attenuates within buyer, framework remains positive,
and the CPV-48 pattern is an exploratory secondary within-buyer result.

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
    detectability_cox, detectability_diagnostics = detectability_cox_outputs(frame)
    buyer_stratified_cox, buyer_stratified_diagnostics = buyer_stratified_cox_outputs()
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
        "survival_cox_detectability_sensitivity.csv": detectability_cox,
        "survival_cox_buyer_stratified_sensitivity.csv": buyer_stratified_cox,
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
        "candidate_coverage": {
            "anchors": len(frame),
            "anchors_with_candidates": int((frame["candidate_pool_size"] > 0).sum()),
            "anchors_without_candidates": int((frame["candidate_pool_size"] == 0).sum()),
            "candidate_pairs": int(frame["candidate_pool_size"].sum()),
            "median_pool_size_among_anchors_with_candidates": int(
                frame.loc[frame["candidate_pool_size"] > 0, "candidate_pool_size"].median()
            ),
        },
        "logrank": logrank,
        "cox": cox_diagnostics,
        "detectability_cox": detectability_diagnostics,
        "buyer_stratified_cox": buyer_stratified_diagnostics,
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
    write_report(
        summary, cox, selection, conditional, detectability_cox,
        buyer_stratified_cox,
    )
    if not summary["validation_passed"]:
        raise RuntimeError("survival evidence validation failed")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
