"""Reusable data-quality and descriptive trend-analysis helpers."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import ruptures as rpt
from hmmlearn.hmm import GaussianHMM
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning


REQUIRED_COHORT_COLUMNS = {
    "episode_id",
    "award_date",
    "digital_segment",
    "duration_months_reliable",
}


def build_quarterly_panel(
    cohort: pd.DataFrame,
    *,
    start_quarter: str = "2015Q2",
    end_quarter: str = "2025Q4",
) -> pd.DataFrame:
    """Aggregate the awarded digital cohort to a complete quarterly panel.

    The first BOAMP extract begins in March 2015, so the default fitting period
    starts at 2015Q2 rather than treating the partial 2015Q1 as a full quarter.
    """

    missing = REQUIRED_COHORT_COLUMNS - set(cohort.columns)
    if missing:
        raise ValueError(f"cohort is missing required columns: {sorted(missing)}")

    data = cohort.loc[:, sorted(REQUIRED_COHORT_COLUMNS)].copy()
    data["award_date"] = pd.to_datetime(data["award_date"], errors="coerce")
    if data["award_date"].isna().any():
        raise ValueError("award_date contains missing or invalid values")
    if data["episode_id"].duplicated().any():
        raise ValueError("episode_id must be unique at cohort grain")

    data["quarter"] = data["award_date"].dt.to_period("Q")
    start = pd.Period(start_quarter, freq="Q")
    end = pd.Period(end_quarter, freq="Q")
    if start > end:
        raise ValueError("start_quarter must not be after end_quarter")
    data = data.loc[data["quarter"].between(start, end)].copy()

    segment_values = sorted(data["digital_segment"].dropna().astype(str).unique())
    quarters = pd.period_range(start, end, freq="Q")
    rows: list[dict[str, Any]] = []
    for segment in ["Overall", *segment_values]:
        segment_data = data if segment == "Overall" else data.loc[
            data["digital_segment"].astype(str).eq(segment)
        ]
        grouped = segment_data.groupby("quarter", observed=True)
        counts = grouped.size()
        reliable = grouped["duration_months_reliable"].count()
        medians = grouped["duration_months_reliable"].median()
        for quarter in quarters:
            count = int(counts.get(quarter, 0))
            reliable_count = int(reliable.get(quarter, 0))
            rows.append(
                {
                    "quarter": str(quarter),
                    "quarter_start": quarter.start_time,
                    "segment": segment,
                    "episode_count": count,
                    "reliable_duration_count": reliable_count,
                    "duration_completeness": (
                        reliable_count / count if count else math.nan
                    ),
                    "median_duration_months": (
                        float(medians.get(quarter)) if reliable_count else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def pelt_break_indices(
    values: Iterable[float],
    *,
    penalty_multiplier: float = 1.0,
    min_size: int = 4,
) -> list[int]:
    """Return PELT change indices for a standardized univariate series.

    The penalty is ``multiplier * log(n)`` after z-standardization. This gives a
    transparent BIC-like complexity penalty and makes sensitivity multipliers
    comparable across segments with different count scales.
    """

    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or len(array) < 2 * min_size:
        return []
    if not np.isfinite(array).all():
        raise ValueError("PELT values must all be finite")
    standard_deviation = float(array.std(ddof=0))
    if standard_deviation == 0:
        return []
    standardized = (array - array.mean()) / standard_deviation
    penalty = float(penalty_multiplier) * math.log(len(standardized))
    endpoints = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(
        standardized.reshape(-1, 1)
    ).predict(pen=penalty)
    return [int(index) for index in endpoints if int(index) < len(array)]


def recent_trend_signal(
    values: Iterable[float],
    *,
    quarters: int = 12,
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Describe the recent linear direction without claiming a forecast."""

    array = np.asarray(list(values), dtype=float)
    if len(array) < quarters:
        raise ValueError(f"at least {quarters} observations are required")
    recent = array[-quarters:]
    if not np.isfinite(recent).all():
        raise ValueError("trend values must all be finite")
    result = linregress(np.arange(quarters, dtype=float), recent)
    if result.pvalue < alpha and result.slope > 0:
        state = "increasing"
    elif result.pvalue < alpha and result.slope < 0:
        state = "decreasing"
    else:
        state = "stable_or_uncertain"
    mean_recent = float(recent.mean())
    return {
        "state": state,
        "quarters_used": quarters,
        "slope_episodes_per_quarter": float(result.slope),
        "annualized_slope_episodes": float(4 * result.slope),
        "recent_mean_episodes_per_quarter": mean_recent,
        "annualized_slope_relative_to_recent_mean": (
            float(4 * result.slope / mean_recent) if mean_recent else None
        ),
        "p_value": float(result.pvalue),
        "r_squared": float(result.rvalue**2),
        "alpha": alpha,
    }


def stationarity_diagnostics(values: Iterable[float]) -> dict[str, Any]:
    """Return ADF and KPSS diagnostics with explicit null hypotheses."""

    array = np.asarray(list(values), dtype=float)
    if len(array) < 12 or not np.isfinite(array).all() or array.std(ddof=0) == 0:
        return {
            "available": False,
            "reason": "requires at least 12 finite, non-constant observations",
        }
    adf = adfuller(array, autolag="AIC")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InterpolationWarning)
            kpss_result = kpss(array, regression="c", nlags="auto")
        kpss_payload = {
            "statistic": float(kpss_result[0]),
            "p_value": float(kpss_result[1]),
            "null": "level_stationary",
            "p_value_table_bound_warning": any(
                issubclass(item.category, InterpolationWarning) for item in caught
            ),
        }
    except (ValueError, OverflowError) as exc:
        kpss_payload = {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "adf": {
            "statistic": float(adf[0]),
            "p_value": float(adf[1]),
            "null": "unit_root_nonstationary",
        },
        "kpss": kpss_payload,
    }


def regime_diagnostics(
    values: Iterable[float],
    *,
    min_observations: int = 16,
    random_state: int = 0,
) -> dict[str, Any]:
    """Label growth/plateau/decline regimes with a 3-state Gaussian HMM.

    The model is fit on the quarter-over-quarter change, not the level, so the
    three states describe typical period-over-period direction rather than the
    segment's absolute activity level. A segment can sit at a high or low count
    while still being in a "plateau" regime if its recent changes are small. This
    is a descriptive regime label with a current-quarter posterior probability;
    it is not a forecast and a detected regime is not a causal explanation.
    """

    array = np.asarray(list(values), dtype=float)
    if len(array) < min_observations or not np.isfinite(array).all():
        return {
            "available": False,
            "reason": f"requires at least {min_observations} finite observations",
        }
    diffs = np.diff(array)
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=0))
    if std_diff == 0:
        return {
            "available": False,
            "reason": "quarter-over-quarter change is constant",
        }
    standardized = ((diffs - mean_diff) / std_diff).reshape(-1, 1)

    model = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=200,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(standardized)
        state_sequence = model.predict(standardized)
        posteriors = model.predict_proba(standardized)

    state_means_original = model.means_.reshape(-1) * std_diff + mean_diff
    order = np.argsort(state_means_original)
    regime_by_state = {
        int(order[0]): "decline",
        int(order[1]): "plateau",
        int(order[2]): "growth",
    }

    current_state = int(state_sequence[-1])
    current_regime = regime_by_state[current_state]
    current_probability = float(posteriors[-1, current_state])

    return {
        "available": True,
        "fit_on": "quarter_over_quarter_change_in_episode_count",
        "mean_change_by_regime": {
            regime_by_state[int(state)]: float(state_means_original[state])
            for state in range(3)
        },
        "transition_matrix_rows_from_columns_to": model.transmat_.tolist(),
        "current_regime": current_regime,
        "current_regime_probability": current_probability,
        "quarters_used": int(len(state_sequence)),
    }


def stable_breaks(
    values: Iterable[float],
    *,
    multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
    tolerance: int = 1,
    min_size: int = 4,
) -> dict[str, Any]:
    """Run PELT penalty sensitivity and identify central breaks that persist."""

    array = list(values)
    by_multiplier = {
        str(multiplier): pelt_break_indices(
            array,
            penalty_multiplier=multiplier,
            min_size=min_size,
        )
        for multiplier in multipliers
    }
    central = by_multiplier.get("1.0", [])
    stable = []
    for index in central:
        if all(
            any(abs(index - candidate) <= tolerance for candidate in indices)
            for indices in by_multiplier.values()
        ):
            stable.append(index)
    return {
        "break_indices_by_penalty_multiplier": by_multiplier,
        "central_break_indices": central,
        "stable_break_indices": stable,
        "stability_tolerance_quarters": tolerance,
        "min_segment_size_quarters": min_size,
    }
