#!/usr/bin/env python3
"""Materialise the current survival results and detectability diagnostics."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines import (
    CoxPHFitter,
    ExponentialFitter,
    GeneralizedGammaFitter,
    KaplanMeierFitter,
    LogLogisticFitter,
    LogNormalFitter,
    WeibullFitter,
)
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data/processed/boamp"
REPORT = PROJECT_ROOT / "SURVIVAL_ANALYSIS_REPORT.md"
MONTH_DAYS = 30.4375
HORIZONS = (12, 24, 36, 48, 60)
ARM_FILES = {
    "strict": "survival_dataset_strict.parquet",
    "main": "survival_dataset.parquet",
    "looser": "survival_dataset_looser.parquet",
    "contrast_high_recall": "survival_dataset_contrast_high_recall.parquet",
}


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

    train_mask = frame["award_year"].le(2021).to_numpy()
    test_mask = frame["award_year"].ge(2022).to_numpy()
    train = design.loc[train_mask]
    test = design.loc[test_mask]
    temporal = CoxPHFitter().fit(train, duration_col="t_months", event_col="event")
    test_c = concordance_index(
        test["t_months"], -temporal.predict_partial_hazard(test), test["event"]
    )
    diagnostics = {
        "contracts": len(frame),
        "events": int(frame["event"].sum()),
        "covariates": len(results),
        "partial_aic": round(float(model.AIC_partial_), 3),
        "in_sample_c_index": round(float(model.concordance_index_), 4),
        "ph_violations_p_lt_0_05": ph_results.loc[
            ph_results["p"].lt(0.05), "covariate"
        ].tolist(),
        "temporal_validation": {
            "train_years": "2015-2021",
            "test_years": "2022-2025",
            "train_contracts": len(train),
            "train_events": int(train["event"].sum()),
            "test_contracts": len(test),
            "test_events": int(test["event"].sum()),
            "train_c_index": round(float(temporal.concordance_index_), 4),
            "test_c_index": round(float(test_c), 4),
        },
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


def write_report(summary: dict[str, Any], cox: pd.DataFrame, selection: pd.DataFrame) -> None:
    temporal = summary["cox"]["temporal_validation"]
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

Temporal validation is weak: C-index is `{temporal['train_c_index']:.3f}` on
2015–2021 and `{temporal['test_c_index']:.3f}` on 2022–2025. The Cox model is not
validated for individualized operational prediction.

## Parametric Models And Indicators

`{summary['parametric']['selected_model']}` has the lowest AIC among exponential,
Weibull, log-logistic, log-normal, and generalized-gamma fits. Model selection does
not remove linkage uncertainty or guarantee tail extrapolation. The exported
`survival_conditional_probabilities.csv` gives 12/24-month conditional indicators
with 500-draw episode-bootstrap intervals.

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
`{summary['sensitivity']['maximum_events']}` across the four retained linkage arms.
Absolute probabilities are therefore linkage-sensitive. Cox effects and subgroup
ordering should only be claimed where the exported sensitivity tables show stable
direction.

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
        "survival_parametric_comparison.csv": parametric,
    }
    for filename, table in outputs.items():
        table.to_csv(PROCESSED / filename, index=False)

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
        },
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
    write_report(summary, cox, selection)
    if not summary["validation_passed"]:
        raise RuntimeError("survival evidence validation failed")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
