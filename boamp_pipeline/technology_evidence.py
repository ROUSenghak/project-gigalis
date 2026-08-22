"""Deploy the frozen classifier and turn its predictions into business evidence.

Three things happen here and they must not be confused with each other.

**Deployment.** The specification chosen by
:mod:`boamp_pipeline.technology_models` is refitted on every labelled notice and
applied to the study cohort. This model has no honest in-sample score and none
is computed: its evidence is the grouped cross-validation and the temporal
split. The unit changes at that boundary -- the classifier was trained on
notices, the study analyses episodes -- so every episode is represented by the
``objet`` of the notice the episode layer already treats as the origin of the
procurement. Concatenating an episode's notices instead would feed the model
documents several times longer than anything it was trained on.

**Composition and crosswalk.** What the taxonomy says the cohort contains, and
what it adds to the CPV segmentation the study already uses. This is the first
downstream result and it stands on its own: descriptive, needing no survival
model, and the part a business reader will use.

**Support-gated enrichment.** Technology-specific survival and trend summaries
are produced *only* for classes with enough episodes and enough observed events
to carry an interpretation. The gates are fixed in this module before any curve
is fitted. Classes that fail them are listed with their counts and nothing is
estimated for them, because an unstable curve labelled "AI" would be read as a
finding about AI procurement rather than as a consequence of fourteen episodes.

Predictions are never discarded. A confidence below the operational cutoff sets
``confidence_status`` and nothing else, so a downstream analysis can filter,
weight, or ignore it, and the coverage tables show what any such filter would
cost by year and by class.

Nothing here modifies the CPV-based survival or trend analysis. The technology
taxonomy is an enrichment layer over that analysis, not a replacement for it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# No ``matplotlib.use`` here. Setting a global backend on import is the caller's
# decision, not the library's: ``scripts/build_technology_taxonomy.py`` selects
# Agg because it runs headless, while the evidence notebook needs the inline
# backend to render these same figures in place.
import matplotlib.pyplot as plt

import joblib  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.statistics import multivariate_logrank_test  # noqa: E402

from boamp_pipeline.evidence import adjust_p_values  # noqa: E402
from boamp_pipeline.linkage import parse_json_list  # noqa: E402
from boamp_pipeline.technology_taxonomy import (  # noqa: E402
    CLASS_ORDER,
    COMPETITION_NATURES,
    PROCESSED,
    PROJECT_ROOT,
    RANDOM_SEED,
    SUBSTANTIVE_CLASSES,
    TAXONOMY_VERSION,
    TECHNOLOGY,
    normalize_objet,
)
from boamp_pipeline.technology_models import (  # noqa: E402
    CALIBRATION_METHOD,
    CONFIDENCE_CUTOFF,
    build_estimator,
    calibrated,
    evaluate_confidence_variants,
    specifications,
)
from scripts.build_survival_evidence import conditional_probability  # noqa: E402

FIGURES = PROJECT_ROOT / "reports/figures"
REPORT = PROJECT_ROOT / "TECHNOLOGY_TAXONOMY_REPORT.md"
MODEL_VERSION = "boamp_technology_classifier_v1.0"
EVIDENCE_VERSION = "boamp_technology_evidence_v1.0"

INK, ACCENT, GRID = "#356E9A", "#C28A24", "#E1E5E8"
SEGMENT_COLOURS = (
    "#356E9A", "#C28A24", "#3F8F6B", "#9A4F35", "#6C5B9E", "#2F7D8C", "#8C6D3F",
)
MONTH_DAYS = 30.4375
HORIZONS = (12, 24, 36)
CONFIDENCE_SWEEP = (0.50, 0.60, 0.70, 0.80, 0.90)

#: Downstream inclusion needs two independent gates, and a class must clear both.
#:
#: Gate A asks whether the *label* means anything. A class the classifier cannot
#: separate produces a downstream group that is a mixture of several
#: technologies; a survival curve fitted to it estimates the mixture, not the
#: class, and no amount of episodes fixes that.
#:
#: Gate B asks whether the *sample* can carry an estimate. A perfectly
#: classified class with fourteen episodes and one event still cannot support a
#: curve.
#:
#: Both are fixed here, before any curve is fitted, and applied mechanically.

#: Gate A -- classifier evidence.
#:
#: Fallback classes are excluded outright. ``MIXED``, ``OTHER_DIGITAL`` and
#: ``OTHER`` are operational residuals, not technologies: ``OTHER_DIGITAL``
#: holds videosurveillance, RFID and web maintenance at once. Including them in
#: a "comparison across technologies" would put a heterogeneous bucket beside
#: cybersecurity and invite the reader to interpret the contrast as a technology
#: effect. They stay in the descriptive tables, where they belong.
SUBSTANTIVE_ONLY = True

#: Minimum out-of-fold F1 for a class to carry a downstream interpretation.
#: Below this, roughly a third of the class's predicted membership is wrong in
#: one direction or the other and the downstream group is substantially diluted.
CLASSIFIER_MIN_F1 = 0.65

#: Minimum annotated support behind that F1, so the gate is not opened by a
#: number estimated from a handful of notices.
CLASSIFIER_MIN_REFERENCE_SUPPORT = 10

#: Gate B -- survival support. Twenty observed successor events is the point
#: below which the curve is driven by individual episodes and its confidence
#: band spans most of the probability range; a hundred episodes keeps the risk
#: set populated far enough into follow-up for the 24-month reading to mean
#: anything.
SURVIVAL_MIN_EPISODES = 100
SURVIVAL_MIN_EVENTS = 20

#: Gate B -- trend support. Zero-count quarters are disqualifying rather than
#: interpolated: a class that disappears for a quarter is a class whose
#: quarterly counts are dominated by sampling, not by market movement.
TREND_MIN_EPISODES = 200
TREND_MAX_ZERO_QUARTERS = 0
TREND_MIN_MEDIAN_PER_QUARTER = 5


def _log(message: str, *args: Any) -> None:
    """Progress reporting. The runner configures logging; the library only emits."""
    import logging

    logging.getLogger(__name__).info(message, *args)


def rebuild_selected_estimator(decision: dict[str, Any]):
    """Reconstruct the selected specification with its frozen hyperparameters."""
    name = decision["selected_model"]
    spec = specifications()[name]
    if spec["features"] != "objet":
        raise RuntimeError(
            f"selected specification {name} does not use procurement text; "
            "deployment is only defined for the text classifier"
        )
    return name, spec, build_estimator(spec, decision["selected_model_params_modal"])


def canonical_episode_text() -> pd.DataFrame:
    """One representative ``objet`` per cohort episode.

    Selection mirrors ``boamp_pipeline.episodes.choose_origin``: the earliest
    notice that opens a competition, otherwise the earliest notice of any kind.
    Only notices carrying a non-empty ``objet`` are eligible, and every cohort
    episode has at least one, so no episode is dropped for lack of text.
    """
    cohort = pd.read_parquet(
        PROCESSED / "survival_cohort.parquet",
        columns=[
            "episode_id", "constituent_notice_ids_json", "digital_segment", "main_cpv",
            "all_cpvs_json", "award_date", "award_year", "buyer_region", "buyer_department",
            "notice_count", "framework_flag",
        ],
    )
    notices = pd.read_parquet(
        PROCESSED / "notices_grand_ouest.parquet",
        columns=["idweb", "objet", "nature", "publication_date"],
    )
    notices["objet"] = notices["objet"].fillna("").astype(str)
    notices = notices.loc[notices["objet"].str.strip() != ""].copy()
    notices["is_competition"] = notices["nature"].isin(COMPETITION_NATURES)

    membership = cohort[["episode_id", "constituent_notice_ids_json"]].copy()
    membership["idweb"] = membership["constituent_notice_ids_json"].map(parse_json_list)
    membership = membership.explode("idweb").dropna(subset=["idweb"])
    joined = membership.merge(notices, on="idweb", how="inner")

    # Competition notices first, then earliest publication, then notice id, so
    # the choice is total and independent of row order in the source files.
    joined = joined.sort_values(
        ["episode_id", "is_competition", "publication_date", "idweb"],
        ascending=[True, False, True, True],
    )
    chosen = joined.groupby("episode_id", as_index=False).first()
    chosen = chosen.rename(
        columns={
            "idweb": "objet_source_idweb",
            "objet": "objet_used_for_prediction",
            "nature": "objet_source_nature",
            "publication_date": "objet_source_date",
        }
    )
    frame = cohort.merge(
        chosen[
            [
                "episode_id", "objet_source_idweb", "objet_used_for_prediction",
                "objet_source_nature", "objet_source_date",
            ]
        ],
        on="episode_id",
        how="left",
    )
    unresolved = int(frame["objet_used_for_prediction"].isna().sum())
    if unresolved:
        raise RuntimeError(f"{unresolved} cohort episodes have no representative objet")
    frame["text"] = frame["objet_used_for_prediction"].map(normalize_objet)
    frame["text_word_count"] = frame["text"].str.split().map(len)
    return frame


def build_predictions(force: bool = True) -> dict[str, Any]:
    predictions_path = TECHNOLOGY / "episode_technology_predictions.csv"
    if predictions_path.exists() and not force:
        raise FileExistsError(f"{predictions_path} already exists. Use --force to rebuild.")
    decision_path = TECHNOLOGY / "model_selection_decision.json"
    if not decision_path.exists():
        raise FileNotFoundError(
            f"{decision_path} not found. Run evaluate_technology_models.py first."
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    corpus = pd.read_parquet(TECHNOLOGY / "technology_corpus.parquet")
    name, spec, base_estimator = rebuild_selected_estimator(decision)
    reliability, calibration = evaluate_confidence_variants(corpus, base_estimator)
    _log(
        "Confidence variant adopted: %s (expected calibration error %.4f -> %.4f)",
        calibration["deployed_variant"],
        calibration["variants"]["raw"]["expected_calibration_error"],
        calibration["variants"]["calibrated"]["expected_calibration_error"],
    )

    text = corpus["text"].to_numpy()
    labels = corpus["label"].to_numpy()
    groups = corpus["group_id"].to_numpy()
    _log("Retraining %s on all %s labelled notices", name, len(corpus))
    estimator = (
        calibrated(base_estimator, labels, groups, RANDOM_SEED)
        if calibration["adopted"]
        else base_estimator
    )
    estimator.fit(text, labels)
    classes = list(estimator.classes_)
    reliability_summary = calibration["variants"][calibration["deployed_variant"]]

    episodes = canonical_episode_text()
    _log("Predicting technology for %s cohort episodes", f"{len(episodes):,}")
    probabilities = estimator.predict_proba(episodes["text"].to_numpy())
    order = np.argsort(-probabilities, axis=1)
    top1 = probabilities[np.arange(len(probabilities)), order[:, 0]]
    top2 = probabilities[np.arange(len(probabilities)), order[:, 1]]

    predictions = pd.DataFrame(
        {
            "episode_id": episodes["episode_id"],
            "objet_used_for_prediction": episodes["objet_used_for_prediction"],
            "objet_source_idweb": episodes["objet_source_idweb"],
            "objet_source_nature": episodes["objet_source_nature"],
            "objet_word_count": episodes["text_word_count"],
            "existing_cpv_segment": episodes["digital_segment"],
            "main_cpv": episodes["main_cpv"],
            "award_year": episodes["award_year"],
            "buyer_region": episodes["buyer_region"],
            "framework_flag": episodes["framework_flag"],
            "predicted_technology": [classes[i] for i in order[:, 0]],
            "confidence": np.round(top1, 6),
            "runner_up_technology": [classes[i] for i in order[:, 1]],
            "runner_up_confidence": np.round(top2, 6),
            "confidence_margin": np.round(top1 - top2, 6),
        }
    )
    predictions["confidence_status"] = np.where(
        predictions["confidence"] >= CONFIDENCE_CUTOFF, "high", "low"
    )
    predictions["confidence_type"] = (
        "calibrated_class_probability" if calibration["adopted"] else "uncalibrated_class_score"
    )
    predictions["model_version"] = MODEL_VERSION
    predictions["taxonomy_version"] = TAXONOMY_VERSION
    for label in CLASS_ORDER:
        if label in classes:
            predictions[f"p_{label}"] = np.round(probabilities[:, classes.index(label)], 6)

    if not predictions["episode_id"].is_unique:
        raise RuntimeError("duplicate episode predictions")

    coverage_year = (
        predictions.assign(high=predictions["confidence_status"].eq("high"))
        .groupby("award_year")
        .agg(n=("episode_id", "size"), high_confidence_n=("high", "sum"))
        .reset_index()
    )
    coverage_year["coverage"] = (
        coverage_year["high_confidence_n"] / coverage_year["n"]
    ).round(4)
    coverage_year["mean_confidence"] = (
        predictions.groupby("award_year")["confidence"].mean().round(4).to_numpy()
    )
    coverage_year["cutoff"] = CONFIDENCE_CUTOFF

    coverage_class = (
        predictions.assign(high=predictions["confidence_status"].eq("high"))
        .groupby("predicted_technology")
        .agg(
            n=("episode_id", "size"),
            high_confidence_n=("high", "sum"),
            mean_confidence=("confidence", "mean"),
        )
        .reindex(CLASS_ORDER)
        .fillna(0)
        .reset_index()
        .rename(columns={"index": "predicted_technology"})
    )
    coverage_class["coverage"] = (
        coverage_class["high_confidence_n"] / coverage_class["n"].replace(0, np.nan)
    ).round(4)
    coverage_class["mean_confidence"] = coverage_class["mean_confidence"].round(4)
    coverage_class["n"] = coverage_class["n"].astype(int)
    coverage_class["high_confidence_n"] = coverage_class["high_confidence_n"].astype(int)

    coverage_region = (
        predictions.assign(high=predictions["confidence_status"].eq("high"))
        .groupby("buyer_region")
        .agg(n=("episode_id", "size"), high_confidence_n=("high", "sum"))
        .reset_index()
    )
    coverage_region["coverage"] = (
        coverage_region["high_confidence_n"] / coverage_region["n"]
    ).round(4)

    sweep = pd.DataFrame(
        {
            "cutoff": CONFIDENCE_SWEEP,
            "retained": [int((predictions["confidence"] >= c).sum()) for c in CONFIDENCE_SWEEP],
        }
    )
    sweep["rejected"] = len(predictions) - sweep["retained"]
    sweep["retention_rate"] = (sweep["retained"] / len(predictions)).round(4)
    sweep["min_year_retention"] = [
        round(
            float(
                predictions.assign(keep=predictions["confidence"] >= c)
                .groupby("award_year")["keep"]
                .mean()
                .min()
            ),
            4,
        )
        for c in CONFIDENCE_SWEEP
    ]

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "evaluation_version": decision["evaluation_version"],
        "specification": name,
        "description": spec["description"],
        "input_field": "objet",
        "input_normalisation": (
            "normalize_objet: mojibake repair, NFC, lowercase, whitespace collapse; "
            "accents and technical acronyms preserved"
        ),
        "vectoriser": {
            key: value
            for key, value in base_estimator.named_steps["tfidf"].get_params().items()
            if key in {"ngram_range", "min_df", "max_df", "sublinear_tf", "lowercase", "strip_accents", "analyzer"}
        },
        "classifier": {
            key: value
            for key, value in base_estimator.named_steps["clf"].get_params().items()
            if key in {"C", "class_weight", "max_iter", "random_state", "solver", "penalty"}
        },
        "calibration": calibration,
        "classes": classes,
        "excluded_features": [
            "buyer name", "SIREN/SIRET", "region", "department", "publication year or date",
            "award amount", "supplier", "procedure type", "framework status", "notice ids",
            "filename", "url", "successor linkage variables", "CPV codes",
        ],
        "excluded_features_reason": (
            "the classifier must learn what is being procured; CPV is held out of the "
            "text model so the text-versus-administrative comparison stays meaningful"
        ),
        "training_data": {
            "file": "data/processed/boamp/technology/technology_corpus.parquet",
            "rows": int(len(corpus)),
            "groups": int(corpus["group_id"].nunique()),
            "years": [int(corpus["year"].min()), int(corpus["year"].max())],
        },
        "validation_evidence": {
            "grouped_cv_macro_f1_mean": decision["selected_model_metrics"]["macro_f1_mean"],
            "grouped_cv_macro_f1_sd": decision["selected_model_metrics"]["macro_f1_sd"],
            "oof_macro_f1": decision["selected_model_metrics"]["oof_macro_f1"],
            "temporal_macro_f1": decision["temporal_validation"]["macro_f1"],
            "note": (
                "the deployment model is refitted on all labelled notices; it has no "
                "in-sample validation score and none is reported"
            ),
        },
        "confidence": {
            "type": "predicted_class_probability",
            "source": (
                f"{calibration['deployed_variant']} multinomial logistic regression "
                "predict_proba"
                + (
                    f", Platt-scaled by CalibratedClassifierCV(method='{CALIBRATION_METHOD}')"
                    if calibration["adopted"]
                    else " (uncalibrated: the calibration rule was not met)"
                )
            ),
            "operational_cutoff": CONFIDENCE_CUTOFF,
            "cutoff_status": "operational reporting convention, not a truth boundary",
            **reliability_summary,
        },
        "deployment": {
            "population": "data/processed/boamp/survival_cohort.parquet",
            "unit": "procurement episode",
            "text_rule": (
                "objet of the earliest competition notice in the episode, else the "
                "earliest notice of any kind; mirrors episodes.choose_origin"
            ),
            "episodes": int(len(predictions)),
        },
        "random_seed": RANDOM_SEED,
    }

    TECHNOLOGY.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    coverage_year.to_csv(
        TECHNOLOGY / "confidence_coverage_by_year.csv", index=False, encoding="utf-8"
    )
    coverage_class.to_csv(
        TECHNOLOGY / "confidence_coverage_by_class.csv", index=False, encoding="utf-8"
    )
    coverage_region.to_csv(
        TECHNOLOGY / "confidence_coverage_by_region.csv", index=False, encoding="utf-8"
    )
    sweep.to_csv(TECHNOLOGY / "confidence_cutoff_sweep.csv", index=False, encoding="utf-8")
    reliability.to_csv(
        TECHNOLOGY / "confidence_reliability_oof.csv", index=False, encoding="utf-8"
    )
    (TECHNOLOGY / "final_model_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    joblib.dump(estimator, TECHNOLOGY / "technology_classifier.joblib")

    summary = {
        **config,
        "prediction_summary": {
            "episodes": int(len(predictions)),
            "high_confidence": int((predictions["confidence_status"] == "high").sum()),
            "high_confidence_share": round(
                float((predictions["confidence_status"] == "high").mean()), 4
            ),
            "mean_confidence": round(float(predictions["confidence"].mean()), 4),
            "median_confidence": round(float(predictions["confidence"].median()), 4),
            "predicted_class_counts": {
                label: int((predictions["predicted_technology"] == label).sum())
                for label in CLASS_ORDER
            },
            "coverage_min_year": {
                "award_year": int(coverage_year.loc[coverage_year["coverage"].idxmin(), "award_year"]),
                "coverage": float(coverage_year["coverage"].min()),
            },
            "coverage_max_year": {
                "award_year": int(coverage_year.loc[coverage_year["coverage"].idxmax(), "award_year"]),
                "coverage": float(coverage_year["coverage"].max()),
            },
            "deployment_text_word_count_median": int(episodes["text_word_count"].median()),
            "training_text_word_count_median": int(corpus["text_word_count"].median()),
        },
    }
    return summary


def load_predictions() -> pd.DataFrame:
    path = TECHNOLOGY / "episode_technology_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_technology_predictions.py first.")
    predictions = pd.read_csv(path)
    if not predictions["episode_id"].is_unique:
        raise RuntimeError("duplicate episode predictions")
    return predictions


def deployed_confidence_facts(
    predictions: pd.DataFrame, config: dict[str, Any], sweep: pd.DataFrame
) -> dict[str, Any]:
    """Describe the confidence variant that actually scored the cohort.

    Section 12 is the published reading guide for an operational cutoff, so
    every number in it has to be read off the column that shipped. An earlier
    version of this generator hardcoded the calibrated branch -- the reliability
    table, the "calibration was adopted" sentence, and a claim that no
    prediction reaches 0.90 -- while the pipeline deployed the raw score. The
    two happen to coincide only when the calibration rule fires, so the branch
    is taken from ``calibration["deployed_variant"]`` and the cutoff facts are
    recomputed from ``episode_technology_predictions.csv`` rather than asserted.
    """
    calibration = config["calibration"]
    variant = str(calibration["deployed_variant"])
    if variant not in calibration["variants"]:
        raise RuntimeError(f"deployed variant {variant!r} has no reliability summary")
    declared = sorted(predictions["confidence_type"].dropna().astype(str).unique())
    if len(declared) != 1:
        raise RuntimeError(f"deployed predictions declare {len(declared)} confidence types: {declared}")
    scores = predictions["confidence"].astype(float)
    cutoffs = [float(value) for value in sweep["cutoff"]]
    reachable = [cutoff for cutoff in cutoffs if int((scores >= cutoff).sum()) > 0]
    return {
        "variant": variant,
        "adopted": bool(calibration["adopted"]),
        "confidence_type": declared[0],
        "episodes": int(len(scores)),
        "max_confidence": float(scores.max()),
        "expected_calibration_error": calibration["variants"][variant][
            "expected_calibration_error"
        ],
        "counts_at_or_above": {
            f"{cutoff:g}": int((scores >= cutoff).sum()) for cutoff in cutoffs
        },
        "highest_reachable_cutoff": max(reachable) if reachable else None,
        "cutoffs_with_no_predictions": [cutoff for cutoff in cutoffs if cutoff not in reachable],
    }


def composition(predictions: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Technology composition of the cohort, with the CPV segments it spans."""
    merged = predictions.merge(
        cohort[["episode_id", "all_cpvs_json"]], on="episode_id", how="left"
    )
    merged["divisions"] = merged["all_cpvs_json"].map(
        lambda value: sorted({str(code)[:2] for code in parse_json_list(value)})
    )
    rows = []
    for label in CLASS_ORDER:
        block = merged.loc[merged["predicted_technology"] == label]
        if block.empty:
            rows.append(
                {
                    "technology": label, "n": 0, "share": 0.0, "high_confidence_n": 0,
                    "high_confidence_share": None, "mean_confidence": None,
                    "first_award_year": None, "last_award_year": None, "years_represented": 0,
                    "cpv_segment_distribution": "", "dominant_cpv_segment": "",
                    "dominant_cpv_segment_share": None, "distinct_cpv_divisions": 0,
                }
            )
            continue
        high = block["confidence_status"].eq("high")
        segments = block["existing_cpv_segment"].value_counts()
        rows.append(
            {
                "technology": label,
                "n": int(len(block)),
                "share": round(len(block) / len(merged), 4),
                "high_confidence_n": int(high.sum()),
                "high_confidence_share": round(float(high.mean()), 4),
                "mean_confidence": round(float(block["confidence"].mean()), 4),
                "first_award_year": int(block["award_year"].min()),
                "last_award_year": int(block["award_year"].max()),
                "years_represented": int(block["award_year"].nunique()),
                "cpv_segment_distribution": "; ".join(
                    f"{segment}:{count}" for segment, count in segments.items()
                ),
                "dominant_cpv_segment": str(segments.index[0]),
                "dominant_cpv_segment_share": round(float(segments.iloc[0] / len(block)), 4),
                "distinct_cpv_divisions": int(
                    len({division for divisions in block.index.map(lambda i: merged.loc[i, "divisions"]) for division in divisions})
                ),
            }
        )
    return pd.DataFrame(rows)


def crosswalk(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """CPV segment against predicted technology, and how much each adds.

    The diagnostic quantities are deliberately simple. ``segment_purity`` is the
    share of a CPV segment falling in its own largest technology class: a
    segment at 0.9 is already a technology class under another name, one at 0.3
    is a container for several. The reverse quantity does the same for the
    technology classes.
    """
    table = pd.crosstab(predictions["existing_cpv_segment"], predictions["predicted_technology"])
    table = table.reindex(columns=[c for c in CLASS_ORDER if c in table.columns], fill_value=0)
    segment_purity = (table.max(axis=1) / table.sum(axis=1)).round(4)
    technology_purity = (table.max(axis=0) / table.sum(axis=0)).round(4)
    summary = {
        "cpv_segments": list(table.index),
        "segment_purity": segment_purity.to_dict(),
        "segment_dominant_technology": table.idxmax(axis=1).to_dict(),
        "segment_distinct_technologies": (table > 0).sum(axis=1).to_dict(),
        "technology_purity_in_cpv": technology_purity.to_dict(),
        "technology_dominant_segment": table.idxmax(axis=0).to_dict(),
        "technology_spans_all_segments": {
            str(column): bool((table[column] > 0).all()) for column in table.columns
        },
        "mean_segment_purity": round(float(segment_purity.mean()), 4),
        "mean_technology_purity": round(float(technology_purity.mean()), 4),
    }
    return table, summary


def classifier_gate() -> pd.DataFrame:
    """Gate A: can the classifier identify this class well enough to use it?

    Read from the out-of-fold per-class metrics of the selected model, so the
    gate is decided by held-out evidence rather than by how many deployment
    episodes happened to receive the label. A class can be numerous in
    deployment precisely *because* the classifier over-predicts it.
    """
    decision = json.loads(
        (TECHNOLOGY / "model_selection_decision.json").read_text(encoding="utf-8")
    )
    per_class = pd.read_csv(TECHNOLOGY / "per_class_metrics.csv")
    per_class = per_class.loc[per_class["model"] == decision["selected_model"]].copy()
    per_class["substantive"] = per_class["technology"].isin(SUBSTANTIVE_CLASSES)
    per_class["reference_support_adequate"] = (
        per_class["support"] >= CLASSIFIER_MIN_REFERENCE_SUPPORT
    )
    per_class["f1_adequate"] = per_class["f1"] >= CLASSIFIER_MIN_F1
    per_class["passes_classifier_gate"] = (
        per_class["substantive"]
        & per_class["reference_support_adequate"]
        & per_class["f1_adequate"]
    )
    per_class["classifier_gate_reason"] = np.where(
        ~per_class["substantive"],
        "fallback class, not a substantive technology",
        np.where(
            ~per_class["reference_support_adequate"],
            f"reference support below {CLASSIFIER_MIN_REFERENCE_SUPPORT}",
            np.where(
                ~per_class["f1_adequate"],
                f"out-of-fold F1 below {CLASSIFIER_MIN_F1}",
                "passes",
            ),
        ),
    )
    return per_class[
        [
            "technology", "support", "precision", "recall", "f1", "substantive",
            "reference_support_adequate", "f1_adequate", "passes_classifier_gate",
            "classifier_gate_reason",
        ]
    ].rename(columns={"support": "reference_support"})


def survival_support(predictions: pd.DataFrame, survival: pd.DataFrame) -> pd.DataFrame:
    merged = survival.merge(
        predictions[["episode_id", "predicted_technology", "confidence_status"]],
        on="episode_id",
        how="inner",
    )
    gate_a = classifier_gate().set_index("technology")
    rows = []
    for label in CLASS_ORDER:
        block = merged.loc[merged["predicted_technology"] == label]
        events = int(block["event"].sum()) if len(block) else 0
        high = block.loc[block["confidence_status"] == "high"]
        statistical = len(block) >= SURVIVAL_MIN_EPISODES and events >= SURVIVAL_MIN_EVENTS
        classifier = bool(gate_a.loc[label, "passes_classifier_gate"])
        eligible = statistical and classifier
        rows.append(
            {
                "technology": label,
                "reference_support": int(gate_a.loc[label, "reference_support"]),
                "cv_precision": float(gate_a.loc[label, "precision"]),
                "cv_recall": float(gate_a.loc[label, "recall"]),
                "cv_f1": float(gate_a.loc[label, "f1"]),
                "passes_classifier_gate": classifier,
                "classifier_gate_reason": str(gate_a.loc[label, "classifier_gate_reason"]),
                "episodes": int(len(block)),
                "events": events,
                "event_rate": round(events / len(block), 4) if len(block) else None,
                "high_confidence_episodes": int(len(high)),
                "high_confidence_events": int(high["event"].sum()) if len(high) else 0,
                "median_followup_months": round(
                    float(block["days_to_cutoff"].median() / MONTH_DAYS), 1
                ) if len(block) else None,
                "passes_statistical_gate": bool(statistical),
                "meets_support_gate": bool(eligible),
                "exclusion_reason": (
                    "" if eligible
                    else str(gate_a.loc[label, "classifier_gate_reason"]) if not classifier
                    else f"episodes < {SURVIVAL_MIN_EPISODES} or events < {SURVIVAL_MIN_EVENTS}"
                ),
                "gate": (
                    f"Gate A: substantive class, reference support >= "
                    f"{CLASSIFIER_MIN_REFERENCE_SUPPORT}, out-of-fold F1 >= {CLASSIFIER_MIN_F1}. "
                    f"Gate B: episodes >= {SURVIVAL_MIN_EPISODES} and events >= {SURVIVAL_MIN_EVENTS}"
                ),
            }
        )
    return pd.DataFrame(rows)


def technology_survival(
    predictions: pd.DataFrame, survival: pd.DataFrame, support: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Kaplan-Meier summaries for the classes that clear the support gate."""
    eligible = support.loc[support["meets_support_gate"], "technology"].tolist()
    merged = survival.merge(
        predictions[["episode_id", "predicted_technology"]], on="episode_id", how="inner"
    )
    merged["t_months"] = np.maximum(merged["duration_days"], 1) / MONTH_DAYS
    frame = merged.loc[merged["predicted_technology"].isin(eligible)].copy()

    horizon_rows, conditional_rows, curves = [], [], {}
    for label in eligible:
        block = frame.loc[frame["predicted_technology"] == label]
        fitter = KaplanMeierFitter().fit(block["t_months"], block["event"], label=label)
        curves[label] = fitter
        for horizon in HORIZONS:
            survival_at = float(fitter.survival_function_at_times(horizon).iloc[0])
            at_risk = int((block["t_months"] >= horizon).sum())
            horizon_rows.append(
                {
                    "technology": label,
                    "horizon_months": horizon,
                    "successor_probability": round(1.0 - survival_at, 4),
                    "survival_probability": round(survival_at, 4),
                    "episodes": int(len(block)),
                    "events": int(block["event"].sum()),
                    "at_risk_at_horizon": at_risk,
                }
            )
        for age in (12, 24, 36, 48):
            for horizon in (12, 24):
                at_risk = int((block["t_months"] >= age).sum())
                conditional_rows.append(
                    {
                        "technology": label,
                        "age_months": age,
                        "horizon_months": horizon,
                        "conditional_probability": round(
                            conditional_probability(fitter, age, horizon), 4
                        ),
                        "at_risk_at_age": at_risk,
                        "interpretable": at_risk >= 50,
                    }
                )

    test = multivariate_logrank_test(
        frame["t_months"], frame["predicted_technology"], frame["event"]
    )

    # What Gate A actually buys, measured rather than asserted. Re-running the
    # same test on every class that clears the statistical gate alone shows how
    # much of an apparent technology effect is carried by the residual buckets.
    statistical_only = support.loc[support["passes_statistical_gate"], "technology"].tolist()
    contrast_frame = merged.loc[merged["predicted_technology"].isin(statistical_only)]
    contrast_test = multivariate_logrank_test(
        contrast_frame["t_months"],
        contrast_frame["predicted_technology"],
        contrast_frame["event"],
    )

    diagnostics = {
        "eligible_classes": eligible,
        "gate_a_contrast": {
            "classes_statistical_gate_only": statistical_only,
            "k": len(statistical_only),
            "events": int(contrast_frame["event"].sum()),
            "logrank_p_value": round(float(contrast_test.p_value), 6),
            "logrank_statistic": round(float(contrast_test.test_statistic), 4),
            "note": (
                "same test over every class clearing the statistical gate alone, "
                "including the fallback residuals; reported to show what Gate A "
                "changes rather than to assert it"
            ),
        },
        "excluded_classes": support.loc[~support["meets_support_gate"], "technology"].tolist(),
        "logrank_statistic": round(float(test.test_statistic), 4),
        "logrank_p_value": round(float(test.p_value), 4),
        "logrank_degrees_of_freedom": int(len(eligible) - 1),
        "logrank_episodes": int(len(frame)),
        "logrank_events": int(frame["event"].sum()),
    }
    return pd.DataFrame(horizon_rows), pd.DataFrame(conditional_rows), diagnostics, curves


def trend_support(predictions: pd.DataFrame, survival: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = survival.merge(
        predictions[["episode_id", "predicted_technology"]], on="episode_id", how="inner"
    )
    merged["quarter"] = pd.PeriodIndex(merged["award_date"], freq="Q")
    # The BOAMP acquisition starts in March 2015, so 2015Q1 is only a partial
    # quarter.  The canonical CPV trend series excludes it; technology trends
    # must use the same observation window to avoid a known design mismatch.
    merged = merged.loc[merged["quarter"] >= pd.Period("2015Q2", freq="Q")]
    quarterly = (
        merged.pivot_table(
            index="quarter", columns="predicted_technology", values="episode_id", aggfunc="count"
        )
        .reindex(columns=[c for c in CLASS_ORDER if c in merged["predicted_technology"].unique()])
        .fillna(0)
        .astype(int)
    )
    gate_a = classifier_gate().set_index("technology")
    rows = []
    for label in CLASS_ORDER:
        classifier = bool(gate_a.loc[label, "passes_classifier_gate"])
        if label not in quarterly.columns:
            rows.append(
                {
                    "technology": label, "episodes": 0, "quarters": 0, "zero_quarters": 0,
                    "median_per_quarter": None, "cv_f1": float(gate_a.loc[label, "f1"]),
                    "passes_classifier_gate": classifier, "passes_statistical_gate": False,
                    "meets_support_gate": False,
                    "exclusion_reason": "no predicted episodes",
                }
            )
            continue
        series = quarterly[label]
        statistical = bool(
            series.sum() >= TREND_MIN_EPISODES
            and (series == 0).sum() <= TREND_MAX_ZERO_QUARTERS
            and series.median() >= TREND_MIN_MEDIAN_PER_QUARTER
        )
        eligible = statistical and classifier
        rows.append(
            {
                "technology": label,
                "episodes": int(series.sum()),
                "quarters": int(len(series)),
                "zero_quarters": int((series == 0).sum()),
                "median_per_quarter": float(series.median()),
                "cv_f1": float(gate_a.loc[label, "f1"]),
                "passes_classifier_gate": classifier,
                "passes_statistical_gate": statistical,
                "meets_support_gate": eligible,
                "exclusion_reason": (
                    "" if eligible
                    else str(gate_a.loc[label, "classifier_gate_reason"]) if not classifier
                    else "insufficient or intermittent quarterly volume"
                ),
            }
        )
    support = pd.DataFrame(rows)
    support["gate"] = (
        f"Gate A: substantive, reference support >= {CLASSIFIER_MIN_REFERENCE_SUPPORT}, "
        f"F1 >= {CLASSIFIER_MIN_F1}. Gate B: episodes >= {TREND_MIN_EPISODES}, "
        f"zero quarters <= {TREND_MAX_ZERO_QUARTERS}, "
        f"median per quarter >= {TREND_MIN_MEDIAN_PER_QUARTER}"
    )
    return support, quarterly


def technology_trend(quarterly: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    """Recent ordinary least squares slope per qualifying quarterly series.

    The CPV analysis defines "recent" as the last twelve quarters. Technology
    slopes use the same window and multiplicity correction. The breakpoint and regime machinery
    stays on the CPV reference series in ``TREND_ANALYSIS_REPORT.md``: running
    PELT and an HMM on eleven derived series would multiply the number of tests
    without adding a question anyone asked.
    """
    from scipy import stats

    eligible = support.loc[support["meets_support_gate"], "technology"].tolist()
    total = quarterly.sum(axis=1)
    rows = []
    for label in eligible:
        series = quarterly[label]
        recent = series.tail(12)
        recent_total = total.loc[recent.index]
        index = np.arange(len(recent), dtype=float)
        fit = stats.linregress(index, recent.to_numpy(dtype=float))
        share = (recent / recent_total).astype(float)
        share_fit = stats.linregress(index, share.to_numpy())
        first_three = series.iloc[:12].mean()
        last_three = series.iloc[-12:].mean()
        rows.append(
            {
                "technology": label,
                "episodes": int(series.sum()),
                "quarters": int(len(recent)),
                "observation_window_quarters": int(len(series)),
                "mean_per_quarter": round(float(series.mean()), 2),
                "slope_episodes_per_quarter": round(float(fit.slope), 4),
                "slope_p_value": round(float(fit.pvalue), 4),
                "slope_r_squared": round(float(fit.rvalue ** 2), 4),
                "share_slope_per_quarter": round(float(share_fit.slope), 6),
                "share_slope_p_value": round(float(share_fit.pvalue), 4),
                "mean_first_3_years": round(float(first_three), 2),
                "mean_last_3_years": round(float(last_three), 2),
                "change_first_to_last": round(float(last_three - first_three), 2),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # One slope per analysed class is one hypothesis test per analysed class.
    # Reading the smallest raw p-value as a finding, without saying how many
    # were computed, is the multiplicity error this correction exists to avoid.
    table["n_tests"] = len(table)
    table["p_holm"] = _adjust_p(table["slope_p_value"], "holm")
    table["p_bh"] = _adjust_p(table["slope_p_value"], "bh")
    table["direction"] = np.where(
        table["p_holm"] >= 0.05,
        np.where(
            table["slope_p_value"] < 0.05,
            "nominal signal only, does not survive multiplicity adjustment",
            "no linear trend detected",
        ),
        np.where(table["slope_episodes_per_quarter"] > 0, "increasing", "decreasing"),
    )
    return table


#: One implementation for both trend families. See
#: :func:`boamp_pipeline.evidence.adjust_p_values`.
_adjust_p = adjust_p_values


def trend_confidence_sensitivity(
    predictions: pd.DataFrame, survival: pd.DataFrame, support: pd.DataFrame
) -> pd.DataFrame:
    """Does restricting the trend series to high-confidence predictions change it?

    The original project brief contemplated dropping predictions below the 0.70
    cutoff before any trend estimation. That is only defensible if what remains
    is still a series -- a filter that removes five sixths of the observations
    changes what is being measured, and a slope fitted to the remainder
    describes the classifier's certainty as much as the market.

    Both specifications are therefore estimated and compared. Neither is
    presented as the answer; the comparison is the diagnostic.
    """
    from scipy import stats

    analysed = support.loc[support["meets_support_gate"], "technology"].tolist()
    merged = survival.merge(
        predictions[["episode_id", "predicted_technology", "confidence_status"]],
        on="episode_id",
        how="inner",
    )
    merged["quarter"] = pd.PeriodIndex(merged["award_date"], freq="Q")

    rows = []
    for label in analysed:
        for arm, subset_frame in (
            ("all_predictions", merged),
            ("confidence_ge_0.70", merged.loc[merged["confidence_status"] == "high"]),
        ):
            series = (
                subset_frame.loc[subset_frame["predicted_technology"] == label]
                .groupby("quarter")["episode_id"]
                .count()
                .reindex(merged["quarter"].sort_values().unique(), fill_value=0)
                .sort_index()
            )
            index = np.arange(len(series), dtype=float)
            fit = stats.linregress(index, series.to_numpy(dtype=float))
            rows.append(
                {
                    "technology": label,
                    "arm": arm,
                    "episodes": int(series.sum()),
                    "quarters": int(len(series)),
                    "zero_quarters": int((series == 0).sum()),
                    "median_per_quarter": float(series.median()),
                    "slope_episodes_per_quarter": round(float(fit.slope), 4),
                    "slope_p_value": round(float(fit.pvalue), 4),
                }
            )
    table = pd.DataFrame(rows)
    wide = table.pivot(index="technology", columns="arm", values="slope_episodes_per_quarter")
    table = table.merge(
        (np.sign(wide["all_predictions"]) == np.sign(wide["confidence_ge_0.70"]))
        .rename("slope_sign_agrees")
        .reset_index(),
        on="technology",
        how="left",
    )
    return table


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def composition_figure(table: pd.DataFrame, path: Path) -> None:
    frame = table.loc[table["n"] > 0].sort_values("n")
    figure, axes = plt.subplots(figsize=(8.0, 5.0))
    positions = np.arange(len(frame))
    axes.barh(positions, frame["n"], color=GRID, edgecolor=INK, linewidth=0.8, label="all predictions")
    axes.barh(positions, frame["high_confidence_n"], color=INK, label="confidence >= 0.70")
    axes.set_yticks(positions)
    axes.set_yticklabels(frame["technology"], fontsize=9)
    axes.set_xlabel("Awarded procurement episodes")
    axes.set_title(
        "Predicted technology composition of the study cohort\n"
        f"{int(table['n'].sum()):,} awarded Grand Ouest digital episodes, 2015-2025",
        fontsize=11,
    )
    axes.legend(frameon=False, fontsize=9, loc="lower right")
    axes.grid(axis="x", color=GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def coverage_figure(coverage: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7.2, 4.2))
    axes.plot(coverage["award_year"], coverage["coverage"], marker="o", color=INK, linewidth=1.8)
    axes.axhline(coverage["coverage"].mean(), color=ACCENT, linestyle="--", linewidth=1.2,
                 label=f"mean {coverage['coverage'].mean():.3f}")
    axes.set_ylim(0, max(0.35, coverage["coverage"].max() * 1.25))
    axes.set_xlabel("Award year")
    axes.set_ylabel("Share of predictions at confidence >= 0.70")
    axes.set_title(
        "Classifier confidence coverage by award year\n"
        "a flat line means class shares are not distorted by drifting confidence",
        fontsize=11,
    )
    axes.legend(frameon=False, fontsize=9)
    axes.grid(color=GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def survival_figure(curves: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(figsize=(8.0, 5.0))
    for index, (label, fitter) in enumerate(curves.items()):
        fitter.plot_survival_function(
            ax=axes, ci_show=False, color=SEGMENT_COLOURS[index % len(SEGMENT_COLOURS)], linewidth=1.7
        )
    axes.set_xlim(0, 96)
    axes.set_ylim(0.5, 1.0)
    axes.set_xlabel("Months since award")
    axes.set_ylabel("P(no observable successor yet)")
    axes.set_title(
        "Time to an observable successor by predicted technology\n"
        "classes clearing both the classifier and the support gates; confidence bands omitted for legibility",
        fontsize=11,
    )
    axes.legend(frameon=False, fontsize=8)
    axes.grid(color=GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Reader artifacts
# ---------------------------------------------------------------------------


def markdown_table(frame: pd.DataFrame, columns: dict[str, str], align: str = "") -> str:
    header = "| " + " | ".join(columns.values()) + " |"
    separator = "|" + "|".join(
        (align_char or "---") for align_char in (align.split(",") if align else ["---"] * len(columns))
    ) + "|"
    lines = [header, separator]
    for row in frame[list(columns)].itertuples(index=False):
        cells = []
        for value in row:
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                cells.append("--")
            elif isinstance(value, (bool, np.bool_)):
                cells.append("yes" if value else "no")
            elif isinstance(value, (int, np.integer)):
                cells.append(f"{int(value):,}")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.4f}".rstrip("0").rstrip("."))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(payload: dict[str, Any]) -> Path:
    audit = payload["audit"]
    decision = payload["decision"]
    config = payload["config"]
    comparison = payload["comparison"]
    per_class = payload["per_class"]
    curve = payload["learning_curve"]
    temporal = decision["temporal_validation"]
    composition_table = payload["composition"]
    cross = payload["crosswalk_summary"]
    survival_support_table = payload["survival_support"]
    horizons = payload["survival_horizons"]
    trend_summary = payload["trend_summary"]
    coverage_year = payload["coverage_year"]
    coverage_class = payload["coverage_class"]
    deployed = payload["deployed_confidence"]
    # The deployed variant, never a literal: the reliability table a reader
    # calibrates their expectations against must be the one that scored the
    # cohort. See deployed_confidence_facts.
    reliability = payload["reliability"].loc[
        payload["reliability"]["variant"] == deployed["variant"]
    ]
    errors = payload["error_triage"]
    selected = decision["selected_model"]
    grouping = audit["grouping"]
    metrics = decision["selected_model_metrics"]
    confidence = config["confidence"]

    reliable = [row["technology"] for row in per_class if row["support_adequate"] and row["f1"] >= 0.75]
    weak = [row["technology"] for row in per_class if row["support_adequate"] and row["f1"] < 0.65]
    unmeasurable = [row["technology"] for row in per_class if not row["support_adequate"]]
    leader = decision["selection"]["nominal_leader"]
    if leader == selected:
        selection_narrative = (
            f"`{selected}` has the highest mean grouped-CV macro-F1 of the four text "
            "specifications, so the pre-specified tie-break -- prefer a "
            "probability-emitting model within one paired standard error of the leader "
            "-- did not need to be invoked."
        )
    else:
        first = decision["selection"]["comparisons"][0]
        selection_narrative = (
            f"`{leader}` has the nominally highest macro-F1. The selected model is "
            f"**`{selected}`**, under a rule fixed before the numbers were read: where a "
            "probability-emitting model sits within one paired standard error of the "
            f"leader, it is preferred. The paired difference is "
            f"`{first['mean_difference_from_leader']}` against a standard error of "
            f"`{first['paired_standard_error']}`."
        )
    classifier_min_support = CLASSIFIER_MIN_REFERENCE_SUPPORT
    classifier_min_f1 = CLASSIFIER_MIN_F1
    stability_block = payload["stability"].loc[payload["stability"]["model"] == selected]
    stability_spread = float(
        stability_block["outer_macro_f1"].max() - stability_block["outer_macro_f1"].min()
    )
    comparison_selected_mean = next(
        row["macro_f1_mean"] for row in comparison if row["model"] == selected
    )
    paired = payload["bootstrap_paired"]
    admin_difference = paired.loc[
        (paired["model_a"] == "M0b_cpv_descriptor") & (paired["model_b"] == selected)
    ].iloc[0]
    pairs_frame = pd.DataFrame(decision["top_confusion_pairs"]).head(6)
    residual_pairs = int(
        pairs_frame.apply(
            lambda row: "OTHER_DIGITAL" in {row["true_label"], row["predicted_label"]}, axis=1
        ).sum()
    )

    # -- Section 12 narrative, branched on what was actually deployed ---------
    # Both branches must be reachable: the calibration rule is evaluated every
    # run and either fires or does not. Nothing below may name a variant, a
    # reliability table, or a cutoff that the deployed column does not carry.
    calibration = config["calibration"]
    if deployed["adopted"]:
        confidence_source_sentence = (
            "Confidence is the predicted class probability, Platt-scaled by\n"
            f"`CalibratedClassifierCV(method='{calibration['method']}')` fitted on labelled data only,\n"
            "inside the same grouped splits. **Calibration was adopted** under the pre-specified\n"
            f"rule -- it reduced the expected calibration error by `{calibration['expected_calibration_error_improvement']}`\n"
            f"at a macro-F1 cost of `{abs(calibration['macro_f1_change'])}`, inside the `0.02` budget. The table\n"
            "below and every confidence figure quoted in this report are the **calibrated**\n"
            "variant, which is the one that scored the cohort."
        )
        residual_mechanism_sentence = (
            "First, the residual miscalibration above: reweighting eleven classes with\n"
            "  `class_weight='balanced'` flattens the probability simplex and Platt scaling\n"
            "  only partly undoes it, so the values remain conservative and must be read\n"
            "  through the table."
        )
        honest_name = "calibrated model confidence score"
        scaling_clause = "The scaling was fitted where"
    else:
        confidence_source_sentence = (
            "Confidence is the **raw** predicted class probability of the refitted multinomial\n"
            "logistic regression -- `predict_proba`, with no post-hoc scaling. Platt scaling\n"
            f"(`CalibratedClassifierCV(method='{calibration['method']}')`, fitted on labelled data only inside\n"
            "the same grouped splits) **was evaluated and rejected** under the pre-specified\n"
            f"rule: adopt it only when it cuts the expected calibration error by at least `0.02`\n"
            f"and costs at most `0.02` macro-F1. It cut the error by `{calibration['expected_calibration_error_improvement']}`, which passes,\n"
            f"but cost `{abs(calibration['macro_f1_change'])}` macro-F1, which exceeds the budget. The rule was written\n"
            "before the numbers were read and was not relaxed to admit it. The table below\n"
            f"and every confidence figure in this report are therefore the **{deployed['variant']}** variant,\n"
            f"the one that scored the cohort; deployed rows carry `confidence_type = {deployed['confidence_type']}`."
        )
        residual_mechanism_sentence = (
            "First, the miscalibration above, which is not corrected: reweighting eleven\n"
            "  classes with `class_weight='balanced'` flattens the probability simplex, and no\n"
            "  post-hoc scaling was applied, so the values are conservative by a wide margin\n"
            "  and must be read through the table rather than at face value."
        )
        honest_name = "uncalibrated model confidence score"
        scaling_clause = "The classifier was fitted where"

    lowest_bin = reliability.iloc[0]
    highest_bin = reliability.iloc[-1]
    empty_cutoffs = deployed["cutoffs_with_no_predictions"]
    reachable_cutoff = deployed["highest_reachable_cutoff"]
    counts = deployed["counts_at_or_above"]
    if empty_cutoffs:
        cutoff_ceiling_sentence = (
            f"The highest {deployed['variant']} confidence in the deployed predictions is "
            f"`{deployed['max_confidence']:.4f}`, so cutoffs at "
            + ", ".join(f"`{value:g}`" for value in empty_cutoffs)
            + " retain nothing and are not usable."
        )
    else:
        cutoff_ceiling_sentence = (
            f"The highest {deployed['variant']} confidence in the deployed predictions is "
            f"`{deployed['max_confidence']:.4f}`; every cutoff in the published sweep still "
            f"retains episodes, down to `{counts[f'{reachable_cutoff:g}']:,}` at `{reachable_cutoff:g}`."
        )

    text = f"""# Technology Taxonomy Classification

Generated: `{payload['created_at']}`
Taxonomy: `{TAXONOMY_VERSION}` | Classifier: `{config['model_version']}`

## 1. Why This Component Exists

BOAMP publishes an administrative vocabulary, not a business one. The study
cohort is defined by CPV divisions `32`, `35`, `48` and `72`, which is
reproducible and auditable but coarse: it says a procurement is "digital"
without saying what technology was bought. Every business question a supplier
asks -- which segments are growing, which are re-procured soonest -- needs the
second thing.

This component learns that missing variable from procurement text:

```text
procurement object text  ->  supervised classifier  ->  business technology class
```

It does not replace the CPV segmentation. The existing survival and trend
results remain the reference analysis; the taxonomy is an enrichment layer over
them.

## 2. Taxonomy

Eight substantive classes -- {', '.join(f'`{c}`' for c in CLASS_ORDER[:8])} --
plus three fallback classes that are annotation decisions rather than missing
values: `MIXED` for a procurement with no dominant technology, `OTHER_DIGITAL`
for a digital purchase outside the eight, and `OTHER` for a notice that carries
a digital CPV without being a technology procurement at all.

The taxonomy was frozen before modelling and has not been changed since.

## 3. Annotated Reference Corpus

`{audit['rows']}` manually annotated BOAMP notices, `{audit['years']['min']}`-`{audit['years']['max']}`,
all with a label and a non-empty object text, `{audit['unique_idweb']}` distinct
notice identifiers, no duplicates. The input field is `objet`: a median of
`{audit['text']['word_count']['median']}` words, `{audit['text']['char_count']['median']}` characters.

{markdown_table(pd.DataFrame(payload['class_summary']), {'technology': 'Class', 'n': 'n', 'share': 'Share'}, '---,---:,---:')}

Two properties of this corpus constrain everything below.

**The sample is quota-stratified, not a random draw.** `BUSINESS_SOFTWARE`
appears almost exactly eight times per year and `NETWORK_TELECOM` eight to nine.
The class proportions above are a property of the annotation design, so they are
not an estimate of how common each technology is in the population, and the
predicted shares in section 13 must not be compared against them.

**AI is genuinely rare.** Seven notices across eleven years. No synthetic AI
examples were generated, no AI rows were duplicated, and no oversampling was
applied. The consequence is reported rather than engineered away: AI per-class
metrics are published with their support and marked as uninterpretable.

`{audit['coverage']['rows_in_grand_ouest']}` of the `{audit['rows']}` notices are in Grand Ouest and
`{audit['coverage']['rows_with_digital_cpv']}` carry a digital CPV. `{audit['coverage']['nature_counts'].get('APPEL_OFFRE', 0)}`
are tender notices, `{audit['coverage']['nature_counts'].get('ATTRIBUTION', 0)}` award notices,
`{audit['coverage']['nature_counts'].get('RECTIFICATIF', 0)}` corrections.

## 4. Leakage Prevention

BOAMP republishes one procurement many times, and buyers re-run the same tender
years later with almost the same wording. Scoring a model on a near-copy of a
document it trained on measures memorisation. Every labelled notice is therefore
assigned to a **procurement family**, and every family sits in exactly one fold.

A family is the union of two rules:

1. notices the canonical episode reconstruction already placed in one episode;
2. notices whose `objet` reaches character-level cosine `0.80` or above.

The second rule is not optional. Rule 1 alone gives `{grouping['episode_groups_before_text_merge']}` groups;
adding rule 2 merges `{grouping['near_duplicate_pairs']}` near-duplicate pairs, of which
`{grouping['near_duplicate_pairs_across_episodes']}` sit in *different* episodes and would otherwise have been
split across folds. The result is `{grouping['groups']}` families:
`{grouping['singleton_groups']}` singletons, `{grouping['multi_notice_groups']}` multi-notice families,
largest `{grouping['largest_group']}`.

Merging in the wrong direction is the safe direction: joining two genuinely
distinct procurements costs a little training signal, while splitting one
inflates every metric in this report.

**An annotation-consistency finding falls out of this.** `{grouping['groups_with_conflicting_labels']}` families
contain notices with near-identical text but different labels -- videoconference
services labelled `NETWORK_TELECOM` in 2017 and `OTHER_DIGITAL` in 2021, an
IaaS procurement labelled `MIXED` in 2017 and `CLOUD_HOSTING` in 2021. No label
was changed: there is no evidence deciding which reading is correct, and editing
labels after seeing model errors is how a corpus is fitted to its classifier.
They are recorded in `data/processed/boamp/technology/annotation_near_duplicates.csv`
as an empirical floor on attainable accuracy.

## 5. Text Representation

The input is `objet` alone. Normalisation is deliberately light: mojibake
repair, Unicode NFC, lowercasing, whitespace collapse. **Accents are preserved**
and no stemming is applied, because the classes are distinguished by words like
`cybersécurité`, `logiciel métier` and `intelligence artificielle`, and
flattening French orthography discards the evidence. Features are TF-IDF word
n-grams. Both unigrams and unigrams-plus-bigrams are offered to the grouped inner
cross-validation so phrases stay addressable if they earn it; the **search space**
includes bigrams and the **selected representation** is
`{config['vectoriser']['ngram_range'][0]}-{config['vectoriser']['ngram_range'][1]}` -- every fold chose unigrams alone, because with 500 short
documents the bigram vocabulary is too sparse to pay for itself.

The vectoriser lives inside a scikit-learn `Pipeline` and is fitted within each
training fold. No held-out document contributes to its own features.

**Excluded by design**: buyer name, SIREN/SIRET, region, department, publication
date or year, award amount, supplier, procedure type, framework status, notice
identifiers, filename, URL, and every successor-linkage variable. The classifier
must learn *what is being procured*, not who bought it, when, or whether it was
later re-procured. CPV is also excluded from the text models -- it is the
benchmark they are measured against, and mixing it in would dissolve the
comparison.

## 6. Models And Validation Design

All specifications share one frozen evaluation design: 3-fold group-aware
stratified cross-validation on the fold assignment saved in
`nlp_cv_folds.csv`, seed `{decision['random_seed']}`. Three folds rather than five
because `AI` has seven observations. Hyperparameters are chosen by a grouped
inner cross-validation *inside* each outer training fold, so no held-out notice
influences the configuration it is scored under.

{markdown_table(pd.DataFrame(comparison), {'model': 'Model', 'family': 'Family', 'macro_f1_mean': 'Macro-F1', 'macro_f1_sd': 'SD', 'weighted_f1_mean': 'Weighted F1', 'accuracy_mean': 'Accuracy'}, '---,---,---:,---:,---:,---:')}

### Result 1 -- procurement text carries substantially more than CPV

Uncertainty is estimated by resampling **procurement families**, not notices:
two notices in one family are near-copies, and treating them as independent
draws would shrink every interval by pretending the corpus holds more
information than it does. Both models are scored on the same resampled families
in each replicate, so the difference below is a *paired* interval.

{markdown_table(payload['bootstrap_per_model'], {'model': 'Model', 'macro_f1': 'Macro-F1', 'ci_lower': '95% CI lower', 'ci_upper': '95% CI upper', 'bootstrap_sd': 'Bootstrap SD'}, '---,---:,---:,---:,---:')}

{markdown_table(payload['bootstrap_paired'].loc[payload['bootstrap_paired']['model_b'] == selected], {'model_a': 'Against', 'observed_difference': 'Difference', 'ci_lower': '95% CI lower', 'ci_upper': '95% CI upper', 'excludes_zero': 'Excludes zero'}, '---,---:,---:,---:,---')}

* **Observation.** All three figures here are pooled out-of-fold macro-F1 --
  computed once over the union of the three held-out folds -- which is what the
  bootstrap resamples. The per-fold means in section 6 differ slightly from them
  (`{comparison_selected_mean}` against `{metrics['oof_macro_f1']}` for the selected model), because averaging
  three fold scores is not the same as scoring all 500 predictions together;
  neither is more correct and both are reported.
  The selected text model reaches
  `{metrics['oof_macro_f1']}`; the best administrative benchmark reaches
  `{payload['bootstrap_per_model'].set_index('model').loc['M0b_cpv_descriptor', 'macro_f1']}`. The paired difference is
  `{abs(admin_difference['observed_difference']):.4f}` macro-F1 with a 95% family-bootstrap interval of
  `[{abs(admin_difference['ci_upper']):.4f}, {abs(admin_difference['ci_lower']):.4f}]`, which excludes zero.
* **Confidence.** High. Both are measured on identical folds with identical
  group isolation and an identical metric, both were searched over the same
  regularisation range, and the interval is estimated at the level of the
  leakage unit.
* **What can be concluded.** The business technology class is genuinely present
  in procurement text and is genuinely *not* recoverable from the official
  classification codes alone. This is the empirical justification for the whole
  component.
* **What cannot be concluded.** That CPV is useless. On `OTHER` -- notices that
  are not technology procurement at all -- CPV outperforms text, because a
  clothing purchase is identifiable from its code and not from the word
  "fourniture". The two vocabularies are complementary.
* **Implication for Gigalis.** Segment reporting built on CPV divisions alone
  will merge cybersecurity, cloud, data and applications into one bucket. The
  text layer is what separates them.

### Final selection and the development budget

{selection_narrative}

Selection did not rest on the mean alone. `{selected}` also has the smallest
between-fold spread of the four text specifications (SD
`{metrics['macro_f1_sd']}`), the smallest train-validation gap (section 8), the
strongest temporal result (section 10), and it emits probabilities without a
second fitting step. Those are the criteria that would have overridden a small
mean advantage in the other direction.

The search budget was fixed before any outer-fold score was read, and every
specification in it is reported here rather than only the winner:

{markdown_table(payload['register'], {'model': 'Model', 'family': 'Family', 'features': 'Features', 'grid_points': 'Grid points'}, '---,---,---,---:')}

Grids are explored only by the *inner* grouped cross-validation, nested inside
each outer training fold, so widening them cannot inflate the outer estimate.
The administrative benchmark was searched over the same regularisation range as
the text models; giving it a narrower search would have made the headline
comparison a statement about how hard each side was tuned.

### Hyperparameter stability

{markdown_table(payload['stability'].loc[payload['stability']['model'] == selected], {'fold': 'Fold', 'ngram_range': 'n-grams', 'min_df': 'min_df', 'max_df': 'max_df', 'sublinear_tf': 'sublinear', 'C': 'C', 'outer_macro_f1': 'Outer macro-F1'}, '---:,---,---:,---:,---,---:,---:')}

Representation choices are stable: every fold selected the same n-gram range,
`min_df` and `max_df`. Only the regularisation strength and the sublinear term
weighting move, and the outer scores those configurations produce sit within
`{stability_spread:.3f}` macro-F1 of each other. Notably every fold preferred
**unigrams alone** over unigrams plus bigrams -- with 500 short documents the
bigram vocabulary is too sparse to pay for itself.

## 7. Per-Class Performance

Out-of-fold, `{selected}`, pooled over the three folds (n = `{audit['rows']}`).

{markdown_table(pd.DataFrame(per_class), {'technology': 'Class', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'support': 'Support', 'predicted_n': 'Predicted n', 'support_adequate': 'Support adequate'}, '---,---:,---:,---:,---:,---:,---')}

Headline: macro-F1 `{metrics['macro_f1_mean']}` (SD `{metrics['macro_f1_sd']}` across folds),
weighted F1 `{metrics['weighted_f1_mean']}`, accuracy `{metrics['accuracy_mean']}`.

### Result 2 -- some classes are reliable, others are not

* **Reliable.** {', '.join(f'`{c}`' for c in reliable) or 'none'} -- F1 at or above `0.75` on
  support of at least `10`.
* **Weak.** {', '.join(f'`{c}`' for c in weak) or 'none'} -- adequate support, F1 below `0.65`.
* **Not interpretable.** {', '.join(f'`{c}`' for c in unmeasurable)}, support `7`. Its observed F1 of
  `{[row['f1'] for row in per_class if row['technology'] == 'AI'][0]}` rests on three correct predictions.
  A high number here would be a small-sample artefact and a low number would be
  equally uninformative; the class is reported as a rare-class limitation, not
  as a measured capability.
* **What cannot be concluded.** That `CLOUD_HOSTING` recall of
  `{[row['recall'] for row in per_class if row['technology'] == 'CLOUD_HOSTING'][0]}` reflects an intrinsic limit. Section 8
  shows the errors are concentrated on website hosting, a boundary the
  annotation guidelines place inside `CLOUD_HOSTING` and the text places near
  `OTHER_DIGITAL`.

Confusion matrix: `data/processed/boamp/technology/confusion_matrix.csv` and
`reports/figures/technology_confusion_matrix.png`.

## 8. Error Analysis

Thirty representative out-of-fold errors were sampled from the largest confusion
pairs and triaged. Triage counts: {', '.join(f'{v} {k}' for k, v in errors.items())}.

The dominant confusion pairs are:

{markdown_table(pd.DataFrame(decision['top_confusion_pairs']).head(6), {'confusion_pair': 'Annotated -> predicted', 'n': 'n', 'share_of_errors': 'Share of errors', 'true_support': 'Support of annotated class'}, '---,---:,---:,---:')}

Reading the sampled texts rather than the triage labels, three patterns account
for most of the residual:

1. **`OTHER_DIGITAL` is a heterogeneous residual class.** It contains
   videosurveillance, RFID, videoconference and web maintenance. It borders on
   `NETWORK_TELECOM`, `BUSINESS_SOFTWARE` and `IT_SERVICES` simultaneously, and
   it is involved in `{residual_pairs}` of the six largest confusion pairs. This is a
   taxonomy-design consequence, not a modelling failure.
2. **Website hosting sits on the `CLOUD_HOSTING` boundary.** "Hébergement de
   sites internet" is annotated `CLOUD_HOSTING` and predicted `OTHER_DIGITAL`
   repeatedly. The classes are separable in principle but the object text is
   short and the distinction is definitional.
3. **`BUSINESS_SOFTWARE` versus `MIXED` is a genuine property of the
   procurement.** "Fourniture de matériels et logiciels informatiques" *is*
   mixed; whether it is filed as such is an annotation convention.

The pre-specified adjacency list under-covers the observed confusions -- it did
not name `OTHER_DIGITAL` pairs. It was not revised after the results were seen,
so the `model_error` bucket is a residual that includes boundary cases the list
missed. This matters only for reading the triage table; the CamemBERT decision
in section 11 does not turn on it.

No label was changed to improve any metric.

## 9. Fit Diagnosis: Bias, Variance, And What Limits Performance

Every model is scored on its own training fold as well as on the held-out fold.
The resubstitution score is not a performance estimate and is not reported as
one; its only use is the gap, which is what separates a model that has memorised
its training fold from one that is not expressive enough.

{markdown_table(pd.DataFrame([{'model': k, **v} for k, v in decision['fit_diagnostics'].items()]), {'model': 'Model', 'train_macro_f1': 'Train macro-F1', 'cv_macro_f1': 'Grouped-CV macro-F1', 'gap': 'Gap', 'fold_sd': 'Fold SD'}, '---,---:,---:,---:,---:')}

The learning curve carries the same two arms, subsampling **families** so a
subsample never holds half of a related-notice group:

{markdown_table(pd.DataFrame(curve).assign(mean_train_rows=lambda f: f["mean_train_rows"].round().astype(int)), {'fraction': 'Fraction', 'mean_train_rows': 'Notices', 'train_macro_f1_mean': 'Train macro-F1', 'macro_f1_mean': 'Validation macro-F1', 'macro_f1_sd': 'SD', 'train_validation_gap': 'Gap'}, '---:,---:,---:,---:,---:,---:')}

### Result 3 -- high variance that is resolving with data, not underfitting

* **Observation.** Training macro-F1 sits near `{curve[0]['train_macro_f1_mean']:.2f}` at every
  training size while validation climbs from `{curve[0]['macro_f1_mean']}` to
  `{curve[-1]['macro_f1_mean']}`. The gap closes monotonically from
  `{curve[0]['train_validation_gap']}` to `{curve[-1]['train_validation_gap']}`. Validation is still rising at the
  full corpus.
* **Confidence.** Moderate to high for the shape. Subsampling is over families
  and repeated five times per point; the final point is a single full-data
  evaluation per fold, so its spread is not directly comparable to the others.
* **What can be concluded.** This is the signature of **high variance**, not
  underfitting: the representation already separates the training folds almost
  perfectly at every size, so it does not lack capacity. The binding constraint
  is the number of independent labelled families, and the gap narrows as that
  number grows.
* **What cannot be concluded.** Where the plateau lies, or how much a larger
  corpus would buy. Five points do not identify an asymptote and no minimum
  sample size is implied by this curve.
* **Consequence for modelling.** A richer representation was considered and
  **not** tested. The candidate on the table was word TF-IDF combined with
  character n-grams, which adds capacity -- the opposite of what a high-variance
  diagnosis calls for. The regularisation path that *would* address variance was
  already searched: `C` spans two orders of magnitude in the inner
  cross-validation, and the selected values are interior to that range rather
  than at its edge, so the model is not starved of regularisation either.
* **Implication for Gigalis.** If the taxonomy is to be operationalised further,
  additional annotation is the lever with evidence behind it. A more elaborate
  model is not.

## 10. Temporal Robustness

Train `{temporal['train_years']}` (n = `{temporal['n_train']}`), test `{temporal['test_years']}`
(n = `{temporal['n_test']}`). `{temporal['groups_straddling_boundary_moved_to_train']}` families straddling the boundary were assigned to
training in full, which costs test observations and cannot flatter the result.

* Macro-F1 over all eleven classes: `{temporal['macro_f1']}`.
* Macro-F1 over the classes with test support of at least `10`: `{temporal['macro_f1_adequate_support_classes']}`.
* Weighted F1 `{temporal['weighted_f1']}`, accuracy `{temporal['accuracy']}`.

### Result 4 -- performance holds on recent notices, for the classes that can be measured

* **Observation.** The all-class macro-F1 falls to `{temporal['macro_f1']}` from
  `{metrics['macro_f1_mean']}`, but restricted to classes with adequate recent
  support it is `{temporal['macro_f1_adequate_support_classes']}` -- at or above the primary estimate.
* **Confidence.** Moderate. Six of eleven classes have test support below ten
  ({', '.join(f'`{c}`' for c in temporal['classes_with_support_below_threshold'])}), and the all-class figure is
  dominated by them.
* **What can be concluded.** The vocabulary of recent BOAMP notices has not
  drifted away from what the model learned on older ones for the high-volume
  classes.
* **What cannot be concluded.** Anything about recent `AI`, `OTHER` or `MIXED`
  performance. `OTHER` has one recent test observation and scores `0.000`; that
  is one notice, not a trend.

## 11. Was An Advanced Model Justified?

The gate was written before the classical results were read: a transformer is
tested only if the frozen classical model is materially inadequate (macro-F1
below `{payload['gate']['macro_f1_floor']}`) **and** fewer than half of its errors come from label ambiguity or
missing information, which no encoder can supply.

The selected model reaches `{payload['gate']['selected_model_macro_f1']}`. The first condition fails
decisively, so **CamemBERT was not tested**. It was not tested and then
discarded; it was not run, because the criterion for running it was not met.
Adding it would have added a large dependency, a GPU-shaped runtime, and an
opaque model to a component whose errors are concentrated on definitional
boundaries rather than semantics.

## 12. Frozen Classifier And Confidence

Specification `{selected}`, refitted on all `{audit['rows']}` labelled notices for
deployment. **That refit has no validation score and none is reported.** The
evidence for this model is the grouped cross-validation and the temporal split
above.

{confidence_source_sentence}

Out-of-fold reliability of the deployed (`{deployed['variant']}`) confidence score:

{markdown_table(reliability, {'bin': 'Stated confidence', 'n': 'n', 'observed_accuracy': 'Observed accuracy', 'mean_confidence': 'Mean stated', 'calibration_gap': 'Gap'}, '---,---:,---:,---:,---:')}

### Result 5 -- confidence ranks well but remains conservative

* **Observation.** Observed accuracy rises with stated confidence, from
  `{lowest_bin['observed_accuracy']}` in the `{lowest_bin['bin']}` bin to `{highest_bin['observed_accuracy']}` in the `{highest_bin['bin']}` bin. But the gap is
  positive in every bin above `0.3`: the score understates its own hit rate.
  Expected calibration error of the deployed variant is `{deployed['expected_calibration_error']}`.
* **Confidence.** High for the ranking, high for the direction of the residual
  miscalibration; it is measured out of fold on `{audit['rows']}` notices.
* **What can be concluded.** The score is a usable ordering and a usable filter.
  At the `{confidence['operational_cutoff']}` operational cutoff, out-of-fold accuracy is
  `{confidence['oof_accuracy_at_or_above_cutoff']}` on the `{confidence['oof_share_at_or_above_cutoff']:.0%}` of notices that clear it, against
  `{confidence['oof_accuracy_below_cutoff']}` below it.
* **What cannot be concluded.** That a stated confidence is the probability the
  deployment label is correct. Two separate reasons.

  {residual_mechanism_sentence}

  Second, and more fundamental: **the corpus is quota-stratified and the
  deployment population is not.** {scaling_clause} `AI` is 1.4% of
  observations by design; in the cohort it is a fraction of a percent. A
  confidence score read off the reference sample is therefore not a posterior
  probability in the deployment population, because the class prior it encodes
  is an artefact of the annotation design. The reliability table describes
  behaviour *on the reference distribution*. No prior correction is applied,
  because the deployment prior is exactly what the classifier is being used to
  estimate and assuming it would make the estimate circular.

  The honest name for the published value is an **{honest_name}**,
  useful for ranking and for selecting an operational subset, not a
  population probability.
* **Operational note.** `{confidence['operational_cutoff']}` is a reporting convention, not a truth
  boundary, and it is unrelated to the `0.70` linkage acceptance threshold, which
  scores an entirely different quantity. {cutoff_ceiling_sentence}

## 13. Propagation To The Study Cohort

The classifier was trained on notices; the study analyses episodes. Each of the
`{config['deployment']['episodes']:,}` cohort episodes is represented by the `objet` of its earliest
competition notice, or of its earliest notice when the episode is award-only --
the same origin rule the episode layer already uses. Concatenated episode text
would have been several times longer than any training document; the chosen
rule gives a deployment median of `{payload['deployment_word_median']}` words against a training median of
`{audit['text']['word_count']['median']}`.

Every episode receives exactly one prediction and none is discarded. Low
confidence sets a flag and nothing else.

{markdown_table(pd.DataFrame(composition_table), {'technology': 'Class', 'n': 'Episodes', 'share': 'Share', 'high_confidence_n': 'High-confidence n', 'high_confidence_share': 'High-confidence share', 'dominant_cpv_segment': 'Dominant CPV segment', 'dominant_cpv_segment_share': 'Its share'}, '---,---:,---:,---:,---:,---,---:')}

`{payload['high_confidence_total']:,}` of `{config['deployment']['episodes']:,}` predictions
(`{payload['high_confidence_share']:.1%}`) clear the `{confidence['operational_cutoff']}` cutoff.

### Result 6 -- coverage is flat over time, so composition shifts are not a confidence artefact

* **Observation.** High-confidence coverage by award year ranges from
  `{coverage_year['coverage'].min():.3f}` to `{coverage_year['coverage'].max():.3f}` with no monotone drift.
* **Confidence.** High; it is a direct count.
* **What can be concluded.** A change in a class's share across years is not
  produced by the classifier becoming less certain about recent notices.
* **What cannot be concluded.** That the class shares are unbiased. Only
  `{payload['high_confidence_share']:.1%}` of predictions clear the cutoff, so any analysis restricted
  to them works with roughly one episode in
  `{round(1 / max(payload['high_confidence_share'], 1e-9))}`, selected on a quantity that is itself correlated with
  class -- coverage ranges from `{coverage_class['coverage'].min():.3f}` to `{coverage_class['coverage'].max():.3f}` across predicted classes.
* **Operational consequence.** At this coverage the `0.70` cutoff is a tool for
  picking a small, high-precision worklist -- cases confident enough to act on
  without review -- and not a filter for population-level analysis. The full
  cutoff sweep is published in `confidence_cutoff_sweep.csv` so a different
  operating point can be chosen against its cost in coverage; none is
  recommended here, because choosing one after seeing these results would be
  selecting an operating point on the outcome.

### Result 7 -- the taxonomy cuts across the CPV segmentation

* **Observation.** Mean CPV-segment purity is `{cross['mean_segment_purity']}`: the largest
  technology class inside a CPV segment accounts for that share of it. The
  reverse, mean technology purity within CPV, is `{cross['mean_technology_purity']}`.
  {sum(1 for v in cross['technology_spans_all_segments'].values() if v)} of {len(cross['technology_spans_all_segments'])} technology classes appear in every CPV segment.
* **Confidence.** Moderate. The crosswalk inherits the classifier's error rate,
  and the counts are predictions, not annotations.
* **What can be concluded.** The two segmentations are not substitutes. CPV
  divisions are containers holding several business technologies each.
* **What cannot be concluded.** Exact class volumes. A class with `{[r['n'] for r in composition_table if r['technology'] == 'CYBERSECURITY'][0]:,}` predicted
  episodes and a cross-validated recall near `0.70` has a genuinely uncertain
  true volume, and no confidence interval on that volume is offered here.
* **Implication for Gigalis.** The taxonomy is the layer that makes
  "which technology market is moving" answerable at all. It should be used for
  segment framing, not for counting to the unit.

## 14. Technology-Level Enrichment, Behind Two Gates

Nothing was rerun mechanically for eleven classes. A class enters the downstream
analysis only by clearing **both** of two gates fixed before any curve was
fitted.

**Gate A -- classifier evidence.** Does the label mean anything? A class the
classifier cannot separate produces a downstream group that is a mixture of
several technologies, and a curve fitted to it estimates the mixture. The gate
requires a substantive technology class, annotated support of at least
`{classifier_min_support}`, and out-of-fold F1 of at least `{classifier_min_f1}`.

Fallback classes are excluded outright. `MIXED`, `OTHER_DIGITAL` and `OTHER` are
operational residuals, not technologies -- `OTHER_DIGITAL` holds
videosurveillance, RFID and web maintenance at once. Placing that bucket beside
cybersecurity in a "comparison across technologies" invites the reader to
interpret the contrast as a technology effect when part of it is the
heterogeneity of the bucket. They remain in the descriptive tables.

**Gate B -- statistical support.** Can the sample carry an estimate? A perfectly
classified class with fourteen episodes and one event still cannot support a
curve.

{markdown_table(pd.DataFrame(payload['gate_a']), {'technology': 'Class', 'reference_support': 'Reference n', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'passes_classifier_gate': 'Gate A', 'classifier_gate_reason': 'Reason'}, '---,---:,---:,---:,---:,---,---')}

### Survival

{markdown_table(pd.DataFrame(survival_support_table), {'technology': 'Class', 'cv_f1': 'CV F1', 'episodes': 'Episodes', 'events': 'Events', 'passes_classifier_gate': 'Gate A', 'passes_statistical_gate': 'Gate B', 'meets_support_gate': 'Analysed'}, '---,---:,---:,---:,---,---,---')}

{payload['survival_narrative']}

### Trend

Gate A applies here too. PELT breakpoints, HMM regimes and stationarity tests
stay on the CPV reference series in `TREND_ANALYSIS_REPORT.md`: running them
across derived technology series would multiply tests without answering a new
question.

One slope per analysed class is one hypothesis test per analysed class, so raw
p-values are reported beside Holm (family-wise) and Benjamini-Hochberg (false
discovery rate) adjustments. `TREND_ANALYSIS_REPORT.md` applies the same
correction to its own family of CPV segment slopes, using the same
implementation.

The technology and CPV series use the same window: 2015Q2--2025Q4. The partial
2015Q1 is excluded because the first BOAMP extract begins in March 2015.

{payload['trend_narrative']}

### Confidence-threshold sensitivity for the trend series

The original project brief contemplated dropping predictions below the `0.70`
cutoff before estimating any trend. That is only defensible if what remains is
still a series.

{markdown_table(payload['trend_sensitivity'], {'technology': 'Class', 'arm': 'Arm', 'episodes': 'Episodes', 'zero_quarters': 'Zero quarters', 'median_per_quarter': 'Median/quarter', 'slope_episodes_per_quarter': 'Slope', 'slope_p_value': 'Raw p'}, '---,---,---:,---:,---:,---:,---:')}

### Result 9 -- the high-confidence restriction is too selective for trend estimation

* **Observation.** Restricting to confidence `>= 0.70` leaves between
  `{int(payload['trend_sensitivity'].loc[payload['trend_sensitivity']['arm'] == 'confidence_ge_0.70', 'episodes'].min())}` and
  `{int(payload['trend_sensitivity'].loc[payload['trend_sensitivity']['arm'] == 'confidence_ge_0.70', 'episodes'].max())}` episodes per class across
  `{int(payload['trend_sensitivity']['quarters'].iloc[0])}` quarters, with
  `{int(payload['trend_sensitivity'].loc[payload['trend_sensitivity']['arm'] == 'confidence_ge_0.70', 'zero_quarters'].min())}` to
  `{int(payload['trend_sensitivity'].loc[payload['trend_sensitivity']['arm'] == 'confidence_ge_0.70', 'zero_quarters'].max())}` empty quarters. Median quarterly counts fall to zero or one.
* **What can be concluded.** The high-confidence subset is too sparse and too
  selective for stable technology trend estimation. The full prediction set is
  used for the trend series, with the classifier's error rate carried as a
  stated limitation rather than filtered away.
* **What this is not.** Evidence that the cutoff is useless. It is a useful
  operational filter for selecting cases to inspect (section 12); it is simply
  not a basis for a quarterly time series.
* **A caution the comparison itself supplies.** One class shows a nominally
  significant slope in the sparse arm that is absent in the full arm. That is
  what fitting a line to a mostly-empty series produces, and it is the reason
  the sparse arm is not adopted.

## 15. Limitations

1. **Training and deployment populations differ.** The corpus is a
   quota-stratified sample of notices spanning Grand Ouest and beyond, including
   procurements that were never awarded. The deployment population is
   `{config['deployment']['episodes']:,}` awarded Grand Ouest digital episodes. Only
   `{payload['annotated_in_cohort']}` annotated notices belong to cohort episodes. Performance measured
   on the corpus is evidence about the corpus.
2. **Class priors are not population priors.** The annotation quotas mean the
   corpus proportions cannot be read as prevalence, and a model trained under
   them carries that prior into its predictions.
3. **`AI` cannot be evaluated.** Seven annotated notices, `{[r['n'] for r in composition_table if r['technology'] == 'AI'][0]}` predicted cohort
   episodes, `{[r['events'] for r in survival_support_table if r['technology'] == 'AI'][0]}` observed successor event. Nothing about AI procurement is
   established here beyond its rarity in this corpus over 2015-2025.
4. **The deployed confidence score is the `{deployed['variant']}` variant and is conservative.**
   {'Platt scaling was adopted and the values below the diagonal remain' if deployed['adopted'] else 'Platt scaling was evaluated and rejected by the pre-specified rule, so no post-hoc correction is'}
   applied. See section 12.
5. **Fallback classes absorb ambiguity.** `OTHER_DIGITAL` carries
   `{[r['n'] for r in composition_table if r['technology'] == 'OTHER_DIGITAL'][0]:,}` predicted episodes and is definitionally heterogeneous.
6. **No inter-annotator agreement statistic exists.** The corpus was delivered
   as one labelled file with no annotator identifier and no second pass, so no
   Cohen's kappa can be computed and label reliability cannot be quantified. The
   internship guide's L2 design asks for two independent annotators; this corpus
   does not meet that design, and the `{grouping['groups_with_conflicting_labels']}` documented
   internal inconsistencies are the only direct evidence available about label
   stability. They are recorded and left uncorrected.
7. **Predictions are predictions.** Downstream technology-level survival and
   trend numbers inherit the classifier's error rate. They are not conditioned on
   verified labels and no uncertainty from the classification stage is propagated
   into their confidence intervals.

## 16. Boundaries

- Do not describe a predicted technology class as an observed attribute of the
  procurement.
- Do not read the corpus class proportions, or the predicted class shares, as
  market shares. The first are annotation quotas; the second are predictions
  carrying the error rate in section 7.
- Do not read a confidence value as a probability of correctness without the
  reliability table.
- Do not report survival figures for {', '.join(f'`{c}`' for c in payload['survival_excluded'])}
  or trend figures for {', '.join(f'`{c}`' for c in payload['trend_excluded'])}; they did not
  clear the support gates.
- Do not present technology-level comparisons as causal, or as adjusted for
  buyer type, contract size, or procedure.
- Do not replace the CPV-based cohort definition or the CPV-based survival and
  trend results with this layer.

## 17. Reproduction

```bash
# the whole layer: corpus, models, deployment, evidence -- about one minute
PYTHONPATH=. python3 scripts/build_technology_taxonomy.py --force

# one stage while iterating
PYTHONPATH=. python3 scripts/build_technology_taxonomy.py --stage models --force

PYTHONPATH=. jupyter nbconvert --execute --to notebook --inplace \\
    notebooks/15_technology_taxonomy_classification.ipynb
PYTHONPATH=. pytest -q tests/test_technology_taxonomy.py
```

All outputs land in `data/processed/boamp/technology/`.

Every line of analysis lives in `boamp_pipeline/technology_taxonomy.py`,
`boamp_pipeline/technology_models.py` and
`boamp_pipeline/technology_evidence.py`; the script is orchestration only. That
split exists so `notebooks/15_technology_taxonomy_classification.ipynb` imports
and runs the same functions -- it re-executes the full grouped cross-validation
and asserts that it reproduces the tables quoted above, rather than displaying
them. A number in this report and the matching number in the notebook come from
one code path.
"""
    REPORT.write_text(text, encoding="utf-8")
    return REPORT


def survival_narrative(horizons: pd.DataFrame, diagnostics: dict[str, Any]) -> str:
    if horizons.empty:
        return "No technology class cleared both gates."
    at_24 = horizons.loc[horizons["horizon_months"] == 24].sort_values(
        "successor_probability", ascending=False
    )
    highest, lowest = at_24.iloc[0], at_24.iloc[-1]
    p_value = diagnostics["logrank_p_value"]
    contrast = diagnostics["gate_a_contrast"]
    significant = p_value < 0.05
    heading = (
        "observable re-procurement timing differs across the analysed technology classes"
        if significant
        else "no difference in observable re-procurement timing was detected"
    )
    conclusion = (
        "A difference in the timing of observable successor procurement was detected "
        f"across the {len(diagnostics['eligible_classes'])} analysed substantive classes. The test is a single "
        "omnibus comparison, so it says the classes are not all alike; it does not "
        "identify which pair drives the result, and no pairwise comparison is offered "
        "here."
        if significant
        else "No statistically significant difference in observable-successor timing was "
        "detected among the sufficiently supported technology classes. This is an "
        "absence of detected difference at the available support, not evidence that "
        "the classes behave identically."
    )
    return f"""{markdown_table(at_24, {'technology': 'Class', 'episodes': 'Episodes', 'events': 'Events', 'successor_probability': 'P(successor by 24m)', 'at_risk_at_horizon': 'At risk at 24m'}, '---,---:,---:,---:,---:')}

#### Result 8 -- {heading}

* **Observation.** At 24 months the highest analysed class is `{highest['technology']}`
  (`{highest['successor_probability']:.4f}`) and the lowest is `{lowest['technology']}`
  (`{lowest['successor_probability']:.4f}`). A multivariate log-rank test across the
  `{len(diagnostics['eligible_classes'])}` classes clearing both gates gives p = `{p_value}`
  on `{diagnostics['logrank_events']}` observed events.
* **Confidence.** Moderate at best. Three things sit between this test and a
  statement about technology. The event is an *observable successor procurement*
  accepted by the frozen linkage policy, not a confirmed contract renewal. The
  class labels are predictions carrying the error rate in section 7, which
  blurs the groups being compared and generally works against detecting a
  difference. And the comparison is unadjusted for anything: buyer type,
  contract size and procedure differ across these classes and none is
  controlled.
* **What can be concluded.** {conclusion}
* **What cannot be concluded.** That any class has a different *contract
  duration*, or that technology causes the difference. Absolute levels inherit
  every caveat in `SURVIVAL_ANALYSIS_REPORT.md` -- they move with the linkage
  threshold and are not lower bounds -- and here they additionally inherit
  classification error.
* **What Gate A changes, measured.** Running the same test over every class
  that clears the *statistical* gate alone -- `{contrast['k']}` classes including the
  fallback residuals, `{contrast['events']}` events -- gives p = `{contrast['logrank_p_value']}`. Dropping the
  residual buckets makes the result **weaker**, not stronger. That is the
  direction that matters: `OTHER_DIGITAL`, `OTHER` and `MIXED` have distinctive
  timing because of what they contain, not because they are technologies, and
  including them manufactures part of an apparent technology effect. A reader
  shown only the eight-class number would over-read it. Gate A exists for this
  reason, and it costs significance rather than buying it.
* **Implication for Gigalis.** Useful for framing which segments generate
  visible re-tendering soonest; not usable for predicting when a named contract
  will be re-let.

Full curves: `technology_survival_summary.csv`, conditional probabilities in
`technology_conditional_probabilities.csv`, figure
`reports/figures/technology_kaplan_meier.png`."""


def trend_narrative(summary: pd.DataFrame, support: pd.DataFrame) -> str:
    if summary.empty:
        return "No technology class cleared both gates."
    excluded = support.loc[~support["meets_support_gate"], "technology"].tolist()
    quarters = int(summary["quarters"].iloc[0])
    nominal = summary.loc[summary["slope_p_value"] < 0.05]
    surviving = summary.loc[summary["p_holm"] < 0.05]
    if len(surviving):
        headline = f"{len(surviving)} technology series moves after multiplicity adjustment"
        conclusion = (
            ", ".join(f"`{row.technology}` {row.direction}" for row in surviving.itertuples())
            + " after Holm adjustment across the family of tests."
        )
    elif len(nominal):
        headline = "one nominal signal, none surviving multiplicity adjustment"
        conclusion = (
            ", ".join(
                f"`{row.technology}` has the smallest raw p-value (`{row.slope_p_value}`, "
                f"Holm-adjusted `{row.p_holm}`)"
                for row in nominal.itertuples()
            )
            + ". It should be treated as an investigation signal, not a confirmed trend."
        )
    else:
        headline = "no technology series shows a detectable linear trend"
        smallest = summary.loc[summary["slope_p_value"].idxmin()]
        conclusion = (
            f"The smallest raw p-value is `{smallest['technology']}` at "
            f"`{smallest['slope_p_value']}`, which does not reach the 5% level before any "
            f"adjustment and gives Holm-adjusted `{smallest['p_holm']}` across "
            f"`{int(summary['n_tests'].iloc[0])}` tests."
        )
    return f"""{markdown_table(summary, {'technology': 'Class', 'episodes': 'Episodes', 'mean_per_quarter': 'Mean/quarter', 'slope_episodes_per_quarter': 'Slope', 'slope_p_value': 'Raw p', 'p_holm': 'Holm p', 'p_bh': 'BH p', 'direction': 'Reading'}, '---,---:,---:,---:,---:,---:,---:,---')}

#### Result 10 -- {headline}

* **Observation.** {len(summary)} classes were tested simultaneously.
  {conclusion}
* **Confidence.** Low to moderate. These are counts of awarded episodes per
  quarter carrying predicted labels, over `{quarters}` quarters, with no
  adjustment for anything else that changed over the period.
* **What can be concluded.** No analysed technology class shows a linear
  movement in awarded Grand Ouest procurement volume that survives correction
  for the number of series examined.
* **What cannot be concluded.** Anything about
  {', '.join(f'`{c}`' for c in excluded) or 'any excluded class'}, which did not clear the gates; anything about
  market value, since these are notice counts and not euros; and absence of
  *any* change -- a linear slope is a weak instrument for detecting a
  non-monotone shift, which is precisely why PELT and the HMM remain on the CPV
  reference series.
* **Implication for Gigalis.** Segment prioritisation should rest on absolute
  volume and on the re-procurement timing above, not on a growth story these
  series do not support."""


def build_evidence(force: bool = True) -> dict[str, Any]:
    summary_path = TECHNOLOGY / "technology_evidence_summary.json"
    if summary_path.exists() and not force:
        raise FileExistsError(f"{summary_path} already exists. Use --force to rebuild.")

    predictions = load_predictions()
    cohort = pd.read_parquet(
        PROCESSED / "survival_cohort.parquet", columns=["episode_id", "all_cpvs_json"]
    )
    survival = pd.read_parquet(
        PROCESSED / "survival_dataset.parquet",
        columns=["episode_id", "event", "duration_days", "days_to_cutoff", "award_date"],
    )
    audit = json.loads((TECHNOLOGY / "annotation_audit_summary.json").read_text(encoding="utf-8"))
    decision = json.loads((TECHNOLOGY / "model_selection_decision.json").read_text(encoding="utf-8"))
    config = json.loads((TECHNOLOGY / "final_model_config.json").read_text(encoding="utf-8"))

    composition_table = composition(predictions, cohort)
    cross_table, cross_summary = crosswalk(predictions)
    support = survival_support(predictions, survival)
    horizons, conditional, diagnostics, curves = technology_survival(predictions, survival, support)
    trend_support_table, quarterly = trend_support(predictions, survival)
    trend_table = technology_trend(quarterly, trend_support_table)
    gate_a = classifier_gate()
    trend_sensitivity = trend_confidence_sensitivity(
        predictions, survival, trend_support_table
    )

    corpus = pd.read_parquet(TECHNOLOGY / "technology_corpus.parquet", columns=["episode_id"])
    annotated_in_cohort = int(corpus["episode_id"].isin(set(cohort["episode_id"])).sum())

    TECHNOLOGY.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    composition_table.to_csv(TECHNOLOGY / "technology_composition.csv", index=False, encoding="utf-8")
    cross_table.to_csv(TECHNOLOGY / "technology_cpv_crosswalk.csv", encoding="utf-8")
    support.to_csv(TECHNOLOGY / "technology_survival_support.csv", index=False, encoding="utf-8")
    horizons.to_csv(TECHNOLOGY / "technology_survival_summary.csv", index=False, encoding="utf-8")
    conditional.to_csv(
        TECHNOLOGY / "technology_conditional_probabilities.csv", index=False, encoding="utf-8"
    )
    trend_support_table.to_csv(
        TECHNOLOGY / "technology_trend_support.csv", index=False, encoding="utf-8"
    )
    trend_table.to_csv(TECHNOLOGY / "technology_trend_summary.csv", index=False, encoding="utf-8")
    gate_a.to_csv(TECHNOLOGY / "technology_classifier_gate.csv", index=False, encoding="utf-8")
    trend_sensitivity.to_csv(
        TECHNOLOGY / "technology_trend_confidence_sensitivity.csv", index=False, encoding="utf-8"
    )
    quarterly.to_csv(TECHNOLOGY / "technology_quarterly_counts.csv", encoding="utf-8")

    coverage_year = pd.read_csv(TECHNOLOGY / "confidence_coverage_by_year.csv")
    coverage_class = pd.read_csv(TECHNOLOGY / "confidence_coverage_by_class.csv").dropna(
        subset=["coverage"]
    )
    composition_figure(composition_table, FIGURES / "technology_composition.png")
    coverage_figure(coverage_year, FIGURES / "technology_confidence_coverage.png")
    if curves:
        survival_figure(curves, FIGURES / "technology_kaplan_meier.png")

    per_class = pd.read_csv(TECHNOLOGY / "per_class_metrics.csv")
    per_class = per_class.loc[per_class["model"] == decision["selected_model"]]
    cutoff_sweep = pd.read_csv(TECHNOLOGY / "confidence_cutoff_sweep.csv")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "audit": audit,
        "decision": decision,
        "config": config,
        "comparison": pd.read_csv(TECHNOLOGY / "model_cv_results.csv").to_dict("records"),
        "per_class": per_class.to_dict("records"),
        "class_summary": pd.read_csv(TECHNOLOGY / "annotation_class_summary.csv").to_dict("records"),
        "learning_curve": pd.read_csv(TECHNOLOGY / "learning_curve.csv").to_dict("records"),
        "composition": composition_table.to_dict("records"),
        "crosswalk_summary": cross_summary,
        "survival_support": support.to_dict("records"),
        "survival_horizons": horizons.to_dict("records"),
        "trend_summary": trend_table.to_dict("records"),
        "trend_sensitivity": trend_sensitivity,
        "classifier_gate": gate_a.to_dict("records"),
        "coverage_year": coverage_year,
        "coverage_class": coverage_class,
        "reliability": pd.read_csv(TECHNOLOGY / "confidence_reliability_oof.csv"),
        "cutoff_sweep": cutoff_sweep,
        "deployed_confidence": deployed_confidence_facts(predictions, config, cutoff_sweep),
        "error_triage": decision["error_triage_counts"],
        "gate": decision["camembert_gate"],
        "annotated_in_cohort": annotated_in_cohort,
        "deployment_word_median": int(
            pd.read_csv(TECHNOLOGY / "episode_technology_predictions.csv")["objet_word_count"].median()
        ),
        "high_confidence_total": int((predictions["confidence_status"] == "high").sum()),
        "high_confidence_share": float((predictions["confidence_status"] == "high").mean()),
        "survival_narrative": survival_narrative(horizons, diagnostics),
        "trend_narrative": trend_narrative(trend_table, trend_support_table),
        "trend_gate": trend_support_table["gate"].iloc[0],
        "bootstrap_per_model": pd.read_csv(TECHNOLOGY / "bootstrap_macro_f1_ci.csv"),
        "bootstrap_paired": pd.read_csv(TECHNOLOGY / "bootstrap_paired_differences.csv"),
        "stability": pd.read_csv(TECHNOLOGY / "hyperparameter_stability.csv"),
        "register": pd.read_csv(TECHNOLOGY / "specification_register.csv"),
        "gate_a": gate_a,
        "trend_sensitivity": trend_sensitivity,
        "survival_excluded": diagnostics["excluded_classes"],
        "trend_excluded": trend_support_table.loc[
            ~trend_support_table["meets_support_gate"], "technology"
        ].tolist(),
    }
    report = write_report(payload)

    summary = {
        "created_at": payload["created_at"],
        "evidence_version": EVIDENCE_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "model_version": config["model_version"],
        "cohort_episodes": int(len(predictions)),
        "annotated_notices_in_cohort_episodes": annotated_in_cohort,
        "composition": {
            row["technology"]: {"n": row["n"], "share": row["share"]}
            for row in composition_table.to_dict("records")
        },
        "high_confidence_share": round(payload["high_confidence_share"], 4),
        # Machine-readable statement of which confidence column shipped, so the
        # report, the notebook, the config, the log and the deployment CSV can
        # be checked against one artifact rather than against each other's prose.
        "deployed_confidence": payload["deployed_confidence"],
        "crosswalk": cross_summary,
        "survival": {
            "gate": {"min_episodes": SURVIVAL_MIN_EPISODES, "min_events": SURVIVAL_MIN_EVENTS},
            **diagnostics,
        },
        "trend": {
            "gate": {
                "min_episodes": TREND_MIN_EPISODES,
                "max_zero_quarters": TREND_MAX_ZERO_QUARTERS,
                "min_median_per_quarter": TREND_MIN_MEDIAN_PER_QUARTER,
            },
            "analysed_classes": trend_table["technology"].tolist(),
            "excluded_classes": trend_support_table.loc[
                ~trend_support_table["meets_support_gate"], "technology"
            ].tolist(),
            # Filter on the adjusted p-value, not on a prose string: matching
            # against wording is exactly how this summary silently reported every
            # class as trending when the wording changed.
            "classes_with_trend_surviving_multiplicity": (
                trend_table.loc[trend_table["p_holm"] < 0.05, "technology"].tolist()
                if len(trend_table) else []
            ),
            "classes_with_nominal_signal_only": (
                trend_table.loc[
                    (trend_table["slope_p_value"] < 0.05) & (trend_table["p_holm"] >= 0.05),
                    "technology",
                ].tolist()
                if len(trend_table) else []
            ),
            "smallest_raw_p_value": (
                float(trend_table["slope_p_value"].min()) if len(trend_table) else None
            ),
            "multiplicity_tests": int(len(trend_table)),
        },
        "outputs": {
            "report": str(report.relative_to(PROJECT_ROOT)),
            "notebook": "notebooks/15_technology_taxonomy_classification.ipynb",
            "figures": [
                "reports/figures/technology_composition.png",
                "reports/figures/technology_confidence_coverage.png",
                *(["reports/figures/technology_kaplan_meier.png"] if curves else []),
            ],
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary
