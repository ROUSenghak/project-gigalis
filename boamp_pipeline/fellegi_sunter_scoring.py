"""Apply a fitted Fellegi-Sunter model to candidate-pair feature tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from boamp_pipeline.fellegi_sunter import (
    comparison_vectors,
    log_weights,
    match_probability,
    total_match_weight,
)

REQUIRED_FEATURE_COLUMNS = (
    "buyer_match_type",
    "text_component",
    "cpv_component",
    "time_component",
)


def score_with_fitted_model(
    frame: pd.DataFrame,
    model_path: Path,
    *,
    fit_key: str = "primary_fit_population_u",
) -> pd.DataFrame:
    """Return ``frame`` with Fellegi-Sunter weight and probability columns.

    The v3 benchmark exposure is generated independently of the full cohort
    candidate table, so it may not already carry ``fs_match_probability``.
    This function applies the same fitted model parameters used for the full
    candidate table without changing candidate generation or labels.
    """
    if "fs_match_probability" in frame.columns and "fs_match_weight" in frame.columns:
        return frame

    missing = [column for column in REQUIRED_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"cannot score Fellegi-Sunter model; missing columns: {missing}")
    if not model_path.exists():
        raise FileNotFoundError(f"Fellegi-Sunter model not found: {model_path}")

    model = json.loads(model_path.read_text(encoding="utf-8"))
    if fit_key not in model:
        raise RuntimeError(f"Fellegi-Sunter model missing fit block: {fit_key}")
    fit = model[fit_key]

    fields = comparison_vectors(
        buyer_match_type=frame["buyer_match_type"].tolist(),
        text_component=frame["text_component"].tolist(),
        cpv_component=frame["cpv_component"].tolist(),
        time_component=frame["time_component"].tolist(),
    )
    weights = log_weights(fit["m"], fit["u"])
    scored = frame.copy()
    scored["fs_match_weight"] = total_match_weight(fields, weights, fit["lambda"])
    scored["fs_match_probability"] = match_probability(scored["fs_match_weight"].to_numpy())
    scored["fs_model_version"] = model.get("model_version")
    scored["fs_fit_key"] = fit_key
    return scored
