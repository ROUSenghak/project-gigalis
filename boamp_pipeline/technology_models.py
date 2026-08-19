"""Technology classifiers: specifications, grouped evaluation, and selection.

The question this module answers is narrow and stated before any model runs:
*does procurement text support a business technology taxonomy that the official
administrative vocabulary does not already provide?* Two administrative
benchmarks and four text specifications are therefore evaluated against each
other on identical folds:

===============================  =================================================
``M0_cpv``                       CPV codes only, hierarchical tokens, logistic
``M0b_cpv_descriptor``           CPV codes plus BOAMP descriptor labels, logistic
``M1_tfidf_logreg``              TF-IDF(objet) word n-grams, logistic
``M2_tfidf_logreg_balanced``     as M1, ``class_weight='balanced'``
``M3_tfidf_linearsvm``           TF-IDF(objet) word n-grams, linear SVM
``M4_tfidf_linearsvm_balanced``  as M3, ``class_weight='balanced'``
===============================  =================================================

``M_majority`` is reported alongside them as a floor, not as a candidate.

Every number is out-of-sample. Hyperparameters are chosen by a grouped inner
cross-validation *inside* each outer training fold, so no held-out notice
influences the configuration it is scored under, and the vectoriser is fitted
inside the pipeline so no held-out document contributes to its own features.

Rare classes are not oversampled. Seven AI notices are seven AI notices; the
weighted variants exist to show what reweighting does, and macro-F1 is the
headline so that the rare classes are not averaged away.

:func:`run_all_specifications` is the single entry point the evaluation stage
and the evidence notebook both call, so the notebook watches the real
cross-validation run rather than reading a table of its results.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import pandas as pd

# No ``matplotlib.use`` here. Setting a global backend on import is the caller's
# decision, not the library's: ``scripts/build_technology_taxonomy.py`` selects
# Agg because it runs headless, while the evidence notebook needs the inline
# backend to render these same figures in place.
import matplotlib.pyplot as plt

from sklearn.base import clone  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from boamp_pipeline.technology_taxonomy import (  # noqa: E402
    CLASS_ORDER,
    MIN_RELIABLE_SUPPORT,
    N_SPLITS,
    RANDOM_SEED,
    TAXONOMY_VERSION,
    TECHNOLOGY,
    TEMPORAL_TEST_YEARS,
    TEMPORAL_TRAIN_YEARS,
    class_support,
    reliable_classes,
    text_pipeline,
    token_pipeline,
)

from boamp_pipeline.technology_taxonomy import PROJECT_ROOT  # noqa: E402

FIGURES = PROJECT_ROOT / "reports/figures"
EVALUATION_VERSION = "boamp_technology_evaluation_v1.0"

#: Report palette, matching scripts/build_survival_evidence.py.
INK, ACCENT, GRID = "#356E9A", "#C28A24", "#E1E5E8"

#: Training fractions for the learning curve. Fractions of *groups*, not rows,
#: so a subsample never contains half of a related-notice family.
LEARNING_CURVE_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
LEARNING_CURVE_REPEATS = 5

#: Confusion pairs named in advance as taxonomy-boundary risks, from the class
#: definitions rather than from any observed result. Used only to label errors
#: in the triage table; nothing is tuned on them.
ADJACENT_PAIRS = frozenset(
    frozenset(pair)
    for pair in (
        ("CLOUD_HOSTING", "IT_SERVICES"),
        ("IT_INFRASTRUCTURE", "IT_SERVICES"),
        ("DATA_BI", "BUSINESS_SOFTWARE"),
        ("CYBERSECURITY", "NETWORK_TELECOM"),
        ("CLOUD_HOSTING", "IT_INFRASTRUCTURE"),
        ("BUSINESS_SOFTWARE", "IT_SERVICES"),
        ("OTHER_DIGITAL", "IT_SERVICES"),
    )
)

#: The model-development budget, fixed before any outer-fold result was read.
#:
#: Two distinct risks are being managed here and they pull in opposite
#: directions. Searching too narrowly leaves avoidable underfitting in place and
#: makes "text beats CPV" a claim about one arbitrary configuration. Searching
#: too widely -- or, worse, widening it *after* seeing an outer-fold score --
#: overfits the evaluation design itself.
#:
#: The resolution is that this grid is explored only by the *inner* grouped
#: cross-validation, nested inside each outer training fold. Widening it
#: therefore cannot inflate the outer estimate; only re-running the outer folds
#: with different specifications could, and the register in
#: :func:`specification_register` exists so every specification ever scored is
#: reported rather than only the winner.
TEXT_GRID = {
    "tfidf__ngram_range": ((1, 1), (1, 2)),
    "tfidf__min_df": (1, 2, 3),
    "tfidf__max_df": (0.9, 1.0),
    "tfidf__sublinear_tf": (True, False),
}

#: The administrative benchmark gets the same regularisation range as the text
#: models. Giving it a narrower search would make the headline comparison a
#: statement about how hard each side was tuned rather than about what each
#: representation carries.
TOKEN_GRID = {"tfidf__min_df": (1, 2)}

#: One compact logarithmic range, shared by every classifier.
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)


def specifications() -> dict[str, dict[str, Any]]:
    """Every evaluated specification, with its feature view and its grid."""
    return {
        "M0_cpv": {
            "family": "administrative_benchmark",
            "features": "cpv",
            "description": "CPV division/group/class/code tokens, multinomial logistic",
            "factory": lambda **params: token_pipeline(
                LogisticRegression(max_iter=5000, random_state=RANDOM_SEED, **params)
            ),
            "grid": {**TOKEN_GRID, "clf__C": C_GRID},
        },
        "M0b_cpv_descriptor": {
            "family": "administrative_benchmark",
            "features": "cpv_descriptor",
            "description": "CPV tokens plus BOAMP descriptor labels, multinomial logistic",
            "factory": lambda **params: token_pipeline(
                LogisticRegression(max_iter=5000, random_state=RANDOM_SEED, **params)
            ),
            "grid": {**TOKEN_GRID, "clf__C": C_GRID},
        },
        "M1_tfidf_logreg": {
            "family": "text",
            "features": "objet",
            "description": "TF-IDF(objet) word n-grams, range selected per fold, multinomial logistic",
            "factory": lambda **params: text_pipeline(
                LogisticRegression(max_iter=5000, random_state=RANDOM_SEED, **params)
            ),
            "grid": {**TEXT_GRID, "clf__C": C_GRID},
        },
        "M2_tfidf_logreg_balanced": {
            "family": "text",
            "features": "objet",
            "description": "as M1 with class_weight='balanced'",
            "factory": lambda **params: text_pipeline(
                LogisticRegression(
                    max_iter=5000, random_state=RANDOM_SEED, class_weight="balanced", **params
                )
            ),
            "grid": {**TEXT_GRID, "clf__C": C_GRID},
        },
        "M3_tfidf_linearsvm": {
            "family": "text",
            "features": "objet",
            "description": "TF-IDF(objet) word n-grams, range selected per fold, linear SVM",
            "factory": lambda **params: text_pipeline(
                LinearSVC(random_state=RANDOM_SEED, **params)
            ),
            "grid": {**TEXT_GRID, "clf__C": C_GRID},
        },
        "M4_tfidf_linearsvm_balanced": {
            "family": "text",
            "features": "objet",
            "description": "as M3 with class_weight='balanced'",
            "factory": lambda **params: text_pipeline(
                LinearSVC(random_state=RANDOM_SEED, class_weight="balanced", **params)
            ),
            "grid": {**TEXT_GRID, "clf__C": C_GRID},
        },
    }


def feature_view(corpus: pd.DataFrame, features: str) -> Any:
    """The input array a specification sees. No target-derived column enters it.

    Buyer, region, department, dates, amounts, procedure type, supplier, notice
    identifiers and every linkage variable are deliberately absent: the
    classifier must learn what is being bought, not who bought it or when.
    """
    if features == "objet":
        return corpus["text"].to_numpy()
    if features == "cpv":
        return corpus["cpv_tokens_json"].map(json.loads).tolist()
    if features == "cpv_descriptor":
        return [
            json.loads(cpv) + json.loads(desc)
            for cpv, desc in zip(corpus["cpv_tokens_json"], corpus["descriptor_tokens_json"])
        ]
    raise ValueError(f"unknown feature view: {features}")


def coerce_params(params: dict[str, Any]) -> dict[str, Any]:
    """Restore types JSON cannot round-trip.

    ``ngram_range`` is a tuple to scikit-learn and a list to JSON, and the
    selected configuration travels through ``model_selection_decision.json`` on
    its way to the learning curve and to deployment. Without this the frozen
    configuration cannot be rebuilt from its own record.
    """
    restored = dict(params)
    if isinstance(restored.get("tfidf__ngram_range"), list):
        restored["tfidf__ngram_range"] = tuple(restored["tfidf__ngram_range"])
    return restored


def grid_points(grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Deterministic expansion of a small parameter grid."""
    keys = sorted(grid)
    points = [{}]
    for key in keys:
        points = [{**point, key: value} for point in points for value in grid[key]]
    return points


def build_estimator(spec: dict[str, Any], params: dict[str, Any]):
    """Instantiate one specification at one point of its grid.

    Grid keys are ``step__parameter``. Classifier parameters go to the factory,
    which constructs the estimator; vectoriser parameters are applied to the
    assembled pipeline, since the factory owns the vectoriser's defaults.
    """
    params = coerce_params(params)
    classifier_params = {
        key.split("__", 1)[1]: value for key, value in params.items() if key.startswith("clf__")
    }
    pipeline = spec["factory"](**classifier_params)
    vectorizer_params = {
        key: value for key, value in params.items() if key.startswith("tfidf__")
    }
    if vectorizer_params:
        pipeline.set_params(**vectorizer_params)
    return pipeline


def subset(view: Any, index: np.ndarray) -> Any:
    if isinstance(view, np.ndarray):
        return view[index]
    return [view[int(i)] for i in index]


def select_hyperparameters(
    spec: dict[str, Any],
    view: Any,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], float]:
    """Grouped inner cross-validation over the specification's grid.

    Ties are broken by the deterministic grid order, so the same data always
    yields the same configuration.
    """
    inner = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    splits = list(inner.split(np.zeros(len(labels)), labels, groups))
    best_params, best_score = None, -np.inf
    for params in grid_points(spec["grid"]):
        scores = []
        for train_index, test_index in splits:
            estimator = build_estimator(spec, params)
            estimator.fit(subset(view, train_index), labels[train_index])
            predicted = estimator.predict(subset(view, test_index))
            scores.append(
                f1_score(
                    labels[test_index], predicted,
                    average="macro", labels=CLASS_ORDER, zero_division=0,
                )
            )
        score = float(np.mean(scores))
        if score > best_score:
            best_params, best_score = params, score
    return best_params or {}, best_score


def decision_scores(estimator: Any, view: Any) -> tuple[np.ndarray, list[str], str]:
    """Per-class scores from whichever interface the estimator exposes.

    Logistic regression returns calibrated-by-construction probabilities. A
    linear SVM returns signed distances to the hyperplane, which are *not*
    probabilities and are labelled as margins wherever they are stored.
    """
    classes = list(estimator.classes_) if hasattr(estimator, "classes_") else list(
        estimator[-1].classes_
    )
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(view), classes, "probability"
    return np.asarray(estimator.decision_function(view)), classes, "margin"


def top_two(scores: np.ndarray, classes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1)
    top1 = scores[np.arange(len(scores)), order[:, 0]]
    top2 = scores[np.arange(len(scores)), order[:, 1]]
    predicted = np.array([classes[i] for i in order[:, 0]])
    return predicted, top1, top1 - top2


# ---------------------------------------------------------------------------
# Outer evaluation
# ---------------------------------------------------------------------------


def run_specification(
    name: str,
    spec: dict[str, Any],
    corpus: pd.DataFrame,
    folds: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested grouped CV for one specification: fold rows and OOF predictions."""
    view = feature_view(corpus, spec["features"])
    labels = corpus["label"].to_numpy()
    groups = corpus["group_id"].to_numpy()

    fold_rows, oof_rows = [], []
    for fold in range(N_SPLITS):
        test_index = np.where(folds == fold)[0]
        train_index = np.where(folds != fold)[0]
        params, inner_score = select_hyperparameters(
            spec,
            subset(view, train_index),
            labels[train_index],
            groups[train_index],
            RANDOM_SEED + fold,
        )
        estimator = build_estimator(spec, params)
        estimator.fit(subset(view, train_index), labels[train_index])
        test_view = subset(view, test_index)
        scores, classes, score_type = decision_scores(estimator, test_view)
        predicted, top1, margin = top_two(scores, classes)
        truth = labels[test_index]
        # Resubstitution score on the fold's own training rows. It is not a
        # performance estimate and is never reported as one; its only use is the
        # gap against the held-out score, which is what separates a model that
        # has memorised its training fold from one that is simply not expressive
        # enough.
        train_predicted = estimator.predict(subset(view, train_index))
        train_macro_f1 = f1_score(
            labels[train_index], train_predicted,
            average="macro", labels=CLASS_ORDER, zero_division=0,
        )

        fold_rows.append(
            {
                "model": name,
                "fold": fold,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "train_groups": int(pd.Series(groups[train_index]).nunique()),
                "test_groups": int(pd.Series(groups[test_index]).nunique()),
                "selected_params": json.dumps(
                    {k: v for k, v in sorted(params.items())}, sort_keys=True
                ),
                "inner_macro_f1": round(inner_score, 4),
                "train_macro_f1": round(train_macro_f1, 4),
                "macro_f1": round(
                    f1_score(truth, predicted, average="macro", labels=CLASS_ORDER, zero_division=0), 4
                ),
                "weighted_f1": round(
                    f1_score(truth, predicted, average="weighted", labels=CLASS_ORDER, zero_division=0), 4
                ),
                "macro_precision": round(
                    precision_score(truth, predicted, average="macro", labels=CLASS_ORDER, zero_division=0), 4
                ),
                "macro_recall": round(
                    recall_score(truth, predicted, average="macro", labels=CLASS_ORDER, zero_division=0), 4
                ),
                "accuracy": round(accuracy_score(truth, predicted), 4),
                "train_validation_gap": round(
                    train_macro_f1
                    - f1_score(truth, predicted, average="macro",
                               labels=CLASS_ORDER, zero_division=0),
                    4,
                ),
            }
        )
        oof_rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "fold": fold,
                    "idweb": corpus["idweb"].to_numpy()[test_index],
                    "group_id": groups[test_index],
                    "year": corpus["year"].to_numpy()[test_index],
                    "true_label": truth,
                    "predicted_label": predicted,
                    "score_type": score_type,
                    "score_top1": np.round(top1, 6),
                    "score_margin": np.round(margin, 6),
                    "correct": truth == predicted,
                }
            )
        )
    return pd.DataFrame(fold_rows), pd.concat(oof_rows, ignore_index=True)


def majority_baseline(corpus: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    labels = corpus["label"].to_numpy()
    rows = []
    for fold in range(N_SPLITS):
        test_index = np.where(folds == fold)[0]
        train_index = np.where(folds != fold)[0]
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((len(train_index), 1)), labels[train_index])
        predicted = dummy.predict(np.zeros((len(test_index), 1)))
        truth = labels[test_index]
        rows.append(
            {
                "model": "M_majority",
                "fold": fold,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "train_groups": int(pd.Series(corpus["group_id"].to_numpy()[train_index]).nunique()),
                "test_groups": int(pd.Series(corpus["group_id"].to_numpy()[test_index]).nunique()),
                "selected_params": "{}",
                "inner_macro_f1": float("nan"),
                "train_macro_f1": float("nan"),
                "macro_f1": round(
                    f1_score(truth, predicted, average="macro", labels=CLASS_ORDER, zero_division=0), 4
                ),
                "weighted_f1": round(
                    f1_score(truth, predicted, average="weighted", labels=CLASS_ORDER, zero_division=0), 4
                ),
                "macro_precision": round(
                    precision_score(truth, predicted, average="macro", labels=CLASS_ORDER, zero_division=0), 4
                ),
                "macro_recall": round(
                    recall_score(truth, predicted, average="macro", labels=CLASS_ORDER, zero_division=0), 4
                ),
                "accuracy": round(accuracy_score(truth, predicted), 4),
                "train_validation_gap": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def pooled_metrics(oof: pd.DataFrame) -> dict[str, float]:
    truth, predicted = oof["true_label"], oof["predicted_label"]
    return {
        "oof_macro_f1": round(
            f1_score(truth, predicted, average="macro", labels=CLASS_ORDER, zero_division=0), 4
        ),
        "oof_weighted_f1": round(
            f1_score(truth, predicted, average="weighted", labels=CLASS_ORDER, zero_division=0), 4
        ),
        "oof_macro_precision": round(
            precision_score(truth, predicted, average="macro", labels=CLASS_ORDER, zero_division=0), 4
        ),
        "oof_macro_recall": round(
            recall_score(truth, predicted, average="macro", labels=CLASS_ORDER, zero_division=0), 4
        ),
        "oof_accuracy": round(accuracy_score(truth, predicted), 4),
    }


def per_class_table(oof: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = oof.loc[oof["model"] == model]
    precision, recall, f1, support = precision_recall_fscore_support(
        rows["true_label"], rows["predicted_label"], labels=CLASS_ORDER, zero_division=0
    )
    table = pd.DataFrame(
        {
            "model": model,
            "technology": CLASS_ORDER,
            "precision": np.round(precision, 4),
            "recall": np.round(recall, 4),
            "f1": np.round(f1, 4),
            "support": support.astype(int),
        }
    )
    table["predicted_n"] = [int((rows["predicted_label"] == label).sum()) for label in CLASS_ORDER]
    table["support_adequate"] = table["support"] >= MIN_RELIABLE_SUPPORT
    table["reliability_note"] = np.where(
        table["support_adequate"],
        "support adequate for interpretation",
        f"support below {MIN_RELIABLE_SUPPORT}; per-class metric is not an estimate of class performance",
    )
    return table


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def learning_curve(
    spec: dict[str, Any],
    name: str,
    corpus: pd.DataFrame,
    folds: np.ndarray,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Grouped learning curve for the selected specification.

    Subsampling is over groups, and the held-out fold is always the full outer
    test fold, so every point on the curve is measured on the same evaluation
    data and only the training size changes.
    """
    view = feature_view(corpus, spec["features"])
    labels = corpus["label"].to_numpy()
    groups = corpus["group_id"].to_numpy()
    rows = []
    for fraction in LEARNING_CURVE_FRACTIONS:
        for repeat in range(LEARNING_CURVE_REPEATS if fraction < 1.0 else 1):
            rng = np.random.default_rng(RANDOM_SEED + repeat)
            for fold in range(N_SPLITS):
                test_index = np.where(folds == fold)[0]
                train_index = np.where(folds != fold)[0]
                train_groups = pd.unique(groups[train_index])
                n_groups = max(2, int(round(fraction * len(train_groups))))
                chosen = set(rng.choice(train_groups, size=n_groups, replace=False))
                mask = np.array([groups[i] in chosen for i in train_index])
                selected = train_index[mask]
                if len(np.unique(labels[selected])) < 2:
                    continue
                estimator = build_estimator(spec, params)
                estimator.fit(subset(view, selected), labels[selected])
                predicted = estimator.predict(subset(view, test_index))
                train_predicted = estimator.predict(subset(view, selected))
                rows.append(
                    {
                        "model": name,
                        "fraction": fraction,
                        "repeat": repeat,
                        "fold": fold,
                        "n_train_rows": int(len(selected)),
                        "n_train_groups": int(n_groups),
                        "n_train_classes": int(len(np.unique(labels[selected]))),
                        "train_macro_f1": f1_score(
                            labels[selected], train_predicted,
                            average="macro", labels=CLASS_ORDER, zero_division=0,
                        ),
                        "macro_f1": f1_score(
                            labels[test_index], predicted,
                            average="macro", labels=CLASS_ORDER, zero_division=0,
                        ),
                        "weighted_f1": f1_score(
                            labels[test_index], predicted,
                            average="weighted", labels=CLASS_ORDER, zero_division=0,
                        ),
                    }
                )
    detail = pd.DataFrame(rows)
    curve = (
        detail.groupby("fraction")
        .agg(
            mean_train_rows=("n_train_rows", "mean"),
            mean_train_groups=("n_train_groups", "mean"),
            train_macro_f1_mean=("train_macro_f1", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            evaluations=("macro_f1", "size"),
        )
        .reset_index()
    )
    curve["train_validation_gap"] = curve["train_macro_f1_mean"] - curve["macro_f1_mean"]
    return curve.round(4)


def temporal_validation(
    spec: dict[str, Any], name: str, corpus: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train on 2015-2022, test on 2023-2025, with groups kept whole.

    A related-notice group that straddles the boundary is assigned to training
    in full. That costs test observations and never adds any, which is the only
    direction that cannot flatter the result.
    """
    view = feature_view(corpus, spec["features"])
    labels = corpus["label"].to_numpy()
    groups = corpus["group_id"].to_numpy()
    years = corpus["year"].to_numpy()

    is_test_year = years >= TEMPORAL_TEST_YEARS[0]
    by_group = pd.DataFrame({"group_id": groups, "test_year": is_test_year})
    straddling = by_group.groupby("group_id")["test_year"].nunique()
    straddling_groups = set(straddling[straddling > 1].index)
    test_mask = is_test_year & np.array([g not in straddling_groups for g in groups])
    train_mask = ~test_mask

    params, _ = select_hyperparameters(
        spec,
        subset(view, np.where(train_mask)[0]),
        labels[train_mask],
        groups[train_mask],
        RANDOM_SEED,
    )
    estimator = build_estimator(spec, params)
    estimator.fit(subset(view, np.where(train_mask)[0]), labels[train_mask])
    predicted = estimator.predict(subset(view, np.where(test_mask)[0]))
    truth = labels[test_mask]

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=CLASS_ORDER, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "model": name,
            "technology": CLASS_ORDER,
            "precision": np.round(precision, 4),
            "recall": np.round(recall, 4),
            "f1": np.round(f1, 4),
            "support": support.astype(int),
        }
    )
    per_class["support_adequate"] = per_class["support"] >= MIN_RELIABLE_SUPPORT
    adequate = per_class.loc[per_class["support_adequate"], "technology"].tolist()
    metrics = {
        "model": name,
        "train_years": f"{TEMPORAL_TRAIN_YEARS[0]}-{TEMPORAL_TRAIN_YEARS[1]}",
        "test_years": f"{TEMPORAL_TEST_YEARS[0]}-{TEMPORAL_TEST_YEARS[1]}",
        "selected_params": json.dumps({k: v for k, v in sorted(params.items())}, sort_keys=True),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "groups_straddling_boundary_moved_to_train": len(straddling_groups),
        "macro_f1": round(
            f1_score(truth, predicted, average="macro", labels=CLASS_ORDER, zero_division=0), 4
        ),
        "weighted_f1": round(
            f1_score(truth, predicted, average="weighted", labels=CLASS_ORDER, zero_division=0), 4
        ),
        "accuracy": round(accuracy_score(truth, predicted), 4),
        "macro_f1_adequate_support_classes": round(
            f1_score(truth, predicted, average="macro", labels=adequate, zero_division=0), 4
        ) if adequate else None,
        "adequate_support_classes": adequate,
        "classes_with_support_below_threshold": per_class.loc[
            ~per_class["support_adequate"], "technology"
        ].tolist(),
    }
    return metrics, per_class


def triage_reason(row: pd.Series, conflicting_groups: set[str], support: dict[str, int]) -> str:
    """Rule-based first pass over an error, applied in a fixed precedence.

    This is triage, not adjudication: it routes each error to the explanation
    the recorded evidence supports, and the report reads the sampled texts
    rather than trusting the label.
    """
    if row["group_id"] in conflicting_groups:
        return "annotation_inconsistency_in_related_notices"
    if row["text_word_count"] < 6:
        return "insufficient_information_in_objet"
    if "MIXED" in {row["true_label"], row["predicted_label"]}:
        return "genuinely_mixed_or_multi_technology_procurement"
    if frozenset({row["true_label"], row["predicted_label"]}) in ADJACENT_PAIRS:
        return "taxonomy_boundary_ambiguity"
    if support.get(row["true_label"], 0) < MIN_RELIABLE_SUPPORT:
        return "rare_class_too_few_training_examples"
    return "model_error"


def error_analysis(
    oof: pd.DataFrame, corpus: pd.DataFrame, model: str, conflicting_groups: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = oof.loc[oof["model"] == model].merge(
        corpus[["idweb", "objet", "text_word_count", "main_cpv", "nomacheteur", "episode_id"]],
        on="idweb",
        how="left",
    )
    errors = rows.loc[~rows["correct"]].copy()
    support = class_support(corpus["label"])
    errors["confusion_pair"] = errors["true_label"] + " -> " + errors["predicted_label"]
    errors["triage_reason"] = errors.apply(
        lambda row: triage_reason(row, conflicting_groups, support), axis=1
    )

    pairs = (
        errors.groupby("confusion_pair")
        .agg(n=("idweb", "size"))
        .sort_values("n", ascending=False)
        .reset_index()
    )
    pairs["share_of_errors"] = (pairs["n"] / len(errors)).round(4)
    pairs["true_label"] = pairs["confusion_pair"].str.split(" -> ").str[0]
    pairs["predicted_label"] = pairs["confusion_pair"].str.split(" -> ").str[1]
    pairs["true_support"] = pairs["true_label"].map(support)
    pairs["named_adjacent_pair"] = pairs.apply(
        lambda row: frozenset({row["true_label"], row["predicted_label"]}) in ADJACENT_PAIRS, axis=1
    )

    # Representative sample: the largest confusion pairs first, then up to four
    # examples each, ordered by model confidence so both confident and hesitant
    # mistakes appear.
    sampled = []
    for pair in pairs["confusion_pair"]:
        block = errors.loc[errors["confusion_pair"] == pair].sort_values(
            "score_margin", ascending=False
        )
        sampled.append(block.head(4))
        if sum(len(part) for part in sampled) >= 30:
            break
    sample = pd.concat(sampled, ignore_index=True).head(30)
    columns = [
        "idweb", "year", "group_id", "episode_id", "true_label", "predicted_label",
        "confusion_pair", "score_type", "score_top1", "score_margin",
        "text_word_count", "main_cpv", "triage_reason", "objet",
    ]
    return sample[columns], pairs


# ---------------------------------------------------------------------------
# Selection and figures
# ---------------------------------------------------------------------------


def select_model(fold_results: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, Any]:
    """Apply the selection rule, which is fixed before the numbers are read.

    Primary criterion is mean grouped-CV macro-F1. Where the leader's advantage
    over a simpler or probabilistic alternative is inside one standard error of
    the paired per-fold differences, the simpler alternative is preferred: at
    three folds that standard error is wide, and choosing the nominal maximum
    would be selecting on noise. Logistic regression counts as simpler than a
    calibrated SVM because it emits probabilities without a second fitting step.
    """
    candidates = comparison.loc[comparison["family"] == "text"].sort_values(
        "macro_f1_mean", ascending=False
    )
    leader = candidates.iloc[0]["model"]
    leader_folds = fold_results.loc[fold_results["model"] == leader].set_index("fold")["macro_f1"]

    rationale = []
    selected = leader
    for _, row in candidates.iterrows():
        model = row["model"]
        if model == leader:
            continue
        other = fold_results.loc[fold_results["model"] == model].set_index("fold")["macro_f1"]
        difference = (leader_folds - other).astype(float)
        standard_error = float(difference.std(ddof=1) / np.sqrt(len(difference)))
        within = bool(abs(float(difference.mean())) <= standard_error)
        emits_probability = "logreg" in model
        rationale.append(
            {
                "model": model,
                "macro_f1_mean": float(row["macro_f1_mean"]),
                "mean_difference_from_leader": round(float(difference.mean()), 4),
                "paired_standard_error": round(standard_error, 4),
                "within_one_standard_error": within,
                "emits_probabilities": emits_probability,
            }
        )
        if within and emits_probability and "logreg" not in selected:
            selected = model

    text_best = float(candidates["macro_f1_mean"].max())
    benchmark_best = float(
        comparison.loc[comparison["family"] == "administrative_benchmark", "macro_f1_mean"].max()
    )
    return {
        "nominal_leader": leader,
        "selected_model": selected,
        "selection_rule": (
            "highest mean grouped-CV macro-F1; a probability-emitting model within one "
            "paired standard error of the leader is preferred over a margin-only model"
        ),
        "comparisons": rationale,
        "text_best_macro_f1": round(text_best, 4),
        "administrative_benchmark_best_macro_f1": round(benchmark_best, 4),
        "text_gain_over_administrative_benchmark": round(text_best - benchmark_best, 4),
    }


def camembert_gate(comparison: pd.DataFrame, selection: dict[str, Any], errors: pd.DataFrame) -> dict[str, Any]:
    """Record the pre-specified decision on whether a transformer is justified.

    Three conditions were written down before the classical results were read.
    A transformer is only worth its cost when the classical model is materially
    inadequate *and* the residual errors look semantic rather than definitional.
    """
    selected = selection["selected_model"]
    macro = float(comparison.loc[comparison["model"] == selected, "macro_f1_mean"].iloc[0])
    reasons = errors["triage_reason"].value_counts(normalize=True).to_dict()
    label_driven = float(
        sum(
            share
            for reason, share in reasons.items()
            if reason
            in {
                "annotation_inconsistency_in_related_notices",
                "taxonomy_boundary_ambiguity",
                "genuinely_mixed_or_multi_technology_procurement",
                "insufficient_information_in_objet",
            }
        )
    )
    conditions = {
        "classical_macro_f1_below_0_55": macro < 0.55,
        "text_gain_over_administrative_benchmark_below_0_05": (
            selection["text_gain_over_administrative_benchmark"] < 0.05
        ),
        "majority_of_errors_are_not_label_or_information_limited": label_driven < 0.5,
    }
    justified = conditions["classical_macro_f1_below_0_55"] and conditions[
        "majority_of_errors_are_not_label_or_information_limited"
    ]
    return {
        "gate_specification": (
            "A transformer is tested only if the frozen classical model is materially "
            "inadequate (mean grouped-CV macro-F1 < 0.55) AND fewer than half of its "
            "errors are attributable to label ambiguity or missing information in the "
            "text, which no encoder can supply."
        ),
        "selected_model_macro_f1": round(macro, 4),
        "share_of_errors_label_or_information_limited": round(label_driven, 4),
        "conditions": conditions,
        "camembert_justified": bool(justified),
        "decision": (
            "test CamemBERT" if justified else "do not test CamemBERT; keep the classical model"
        ),
    }


def confusion_figure(oof: pd.DataFrame, model: str, path: Path) -> None:
    rows = oof.loc[oof["model"] == model]
    matrix = confusion_matrix(rows["true_label"], rows["predicted_label"], labels=list(CLASS_ORDER))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axes = plt.subplots(figsize=(9.0, 7.6))
    axes.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axes.set_xticks(range(len(CLASS_ORDER)))
    axes.set_yticks(range(len(CLASS_ORDER)))
    axes.set_xticklabels(CLASS_ORDER, rotation=45, ha="right", fontsize=8)
    axes.set_yticklabels(
        [f"{label} (n={matrix[i].sum()})" for i, label in enumerate(CLASS_ORDER)], fontsize=8
    )
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            if matrix[i, j]:
                axes.text(
                    j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if normalized[i, j] > 0.55 else "#20303C",
                )
    axes.set_xlabel("Predicted technology")
    axes.set_ylabel("Annotated technology (support)")
    axes.set_title(f"Out-of-fold confusion, {model}\ngrouped 3-fold CV, n={len(rows)}", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def learning_curve_figure(curve: pd.DataFrame, model: str, path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7.2, 4.4))
    axes.errorbar(
        curve["mean_train_rows"], curve["macro_f1_mean"], yerr=curve["macro_f1_sd"],
        marker="o", color=INK, ecolor=GRID, capsize=3, linewidth=1.6,
    )
    axes.set_xlabel("Labelled notices used for training (grouped subsample)")
    axes.set_ylabel("Grouped-CV macro-F1")
    axes.set_title(f"Learning curve, {model}", fontsize=11)
    axes.grid(color=GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


#: Platt scaling, not isotonic. With 500 labelled notices spread over eleven
#: classes, an isotonic fit per class would be estimating a free-form monotone
#: function from a few dozen points and would mostly reproduce its own noise.
CALIBRATION_METHOD = "sigmoid"

#: The calibrated model replaces the raw one for reporting confidence only when
#: it materially improves calibration without materially costing accuracy. Both
#: bounds are fixed here, before either number is computed.
MIN_CALIBRATION_GAP_IMPROVEMENT = 0.02
MAX_MACRO_F1_LOSS = 0.02

#: Operational confidence cutoff. It is a reporting convention for separating
#: predictions that can carry a business reading from those that cannot -- not a
#: truth boundary, and unrelated to the 0.70 linkage acceptance threshold, which
#: scores a different quantity entirely.
CONFIDENCE_CUTOFF = 0.70


def grouped_splits(labels, groups, seed: int):
    """Grouped, class-stratified splits used for calibration fitting."""
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels, groups))


def calibrated(estimator, labels, groups, seed: int):
    """Wrap the frozen specification in Platt scaling fitted on labelled data.

    ``CalibratedClassifierCV`` clones the pipeline, refits it on each training
    part, and fits the scaling on the corresponding held-out part. The splits
    are the same grouped, class-stratified splits used everywhere else, so no
    notice contributes to the scaling that was fitted to it, and no notice from
    the unlabelled cohort is involved at any point.
    """
    return CalibratedClassifierCV(
        clone(estimator), method=CALIBRATION_METHOD, cv=grouped_splits(labels, groups, seed)
    )


def reliability_table(scores: np.ndarray, correct: np.ndarray, variant: str) -> pd.DataFrame:
    """Observed accuracy against stated confidence, in fixed bins."""
    edges = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    frame = pd.DataFrame({"confidence": scores, "correct": correct})
    frame["bin"] = pd.cut(frame["confidence"], bins=edges, right=False, include_lowest=True)
    table = (
        frame.groupby("bin", observed=True)
        .agg(
            n=("correct", "size"),
            observed_accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    table["variant"] = variant
    table["bin"] = table["bin"].astype(str)
    table["share_of_predictions"] = (table["n"] / len(frame)).round(4)
    table["calibration_gap"] = (table["observed_accuracy"] - table["mean_confidence"]).round(4)
    return table.round(4)


def weighted_calibration_error(table: pd.DataFrame) -> float:
    """Expected calibration error: bin gaps weighted by how often each occurs."""
    return float((table["calibration_gap"].abs() * table["share_of_predictions"]).sum())


def evaluate_confidence_variants(
    corpus: pd.DataFrame, estimator
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare raw and calibrated confidence on the frozen out-of-fold splits.

    The raw multinomial probabilities of a ``class_weight='balanced'`` model
    over eleven classes are not calibrated: reweighting flattens the simplex, so
    a stated 0.35 is empirically worth far more than 0.35. That is measured here
    rather than assumed, and the calibrated variant is adopted only if it fixes
    the gap without costing accuracy.

    Calibration is fitted strictly inside each outer training fold, so the
    reported numbers for both variants are out-of-sample in the same sense.
    """
    folds = pd.read_csv(TECHNOLOGY / "nlp_cv_folds.csv", dtype={"idweb": str})
    if list(folds["idweb"]) != list(corpus["idweb"]):
        raise RuntimeError("fold file and corpus are not aligned")
    fold_index = folds["fold"].to_numpy()
    text = corpus["text"].to_numpy()
    labels = corpus["label"].to_numpy()
    groups = corpus["group_id"].to_numpy()

    collected: dict[str, dict[str, list]] = {
        "raw": {"confidence": [], "correct": [], "predicted": [], "truth": []},
        "calibrated": {"confidence": [], "correct": [], "predicted": [], "truth": []},
    }
    for fold in range(N_SPLITS):
        test_index = np.where(fold_index == fold)[0]
        train_index = np.where(fold_index != fold)[0]
        for variant in ("raw", "calibrated"):
            model = (
                clone(estimator)
                if variant == "raw"
                else calibrated(estimator, labels[train_index], groups[train_index], RANDOM_SEED + fold)
            )
            model.fit(text[train_index], labels[train_index])
            probabilities = model.predict_proba(text[test_index])
            classes = list(model.classes_)
            best = np.argmax(probabilities, axis=1)
            predicted = np.array([classes[i] for i in best])
            collected[variant]["confidence"].extend(probabilities[np.arange(len(best)), best])
            collected[variant]["predicted"].extend(predicted)
            collected[variant]["truth"].extend(labels[test_index])
            collected[variant]["correct"].extend(predicted == labels[test_index])

    tables, metrics = [], {}
    for variant, values in collected.items():
        table = reliability_table(
            np.asarray(values["confidence"]), np.asarray(values["correct"]), variant
        )
        tables.append(table)
        confidence = np.asarray(values["confidence"])
        correct = np.asarray(values["correct"])
        at_or_above = correct[confidence >= CONFIDENCE_CUTOFF]
        below = correct[confidence < CONFIDENCE_CUTOFF]
        metrics[variant] = {
            "oof_macro_f1": round(
                f1_score(
                    values["truth"], values["predicted"],
                    average="macro", labels=CLASS_ORDER, zero_division=0,
                ),
                4,
            ),
            "oof_accuracy": round(accuracy_score(values["truth"], values["predicted"]), 4),
            "expected_calibration_error": round(weighted_calibration_error(table), 4),
            "oof_share_at_or_above_cutoff": round(float((confidence >= CONFIDENCE_CUTOFF).mean()), 4),
            "oof_accuracy_at_or_above_cutoff": (
                round(float(at_or_above.mean()), 4) if len(at_or_above) else None
            ),
            "oof_accuracy_below_cutoff": round(float(below.mean()), 4) if len(below) else None,
            "oof_n_at_or_above_cutoff": int(len(at_or_above)),
            "oof_n_below_cutoff": int(len(below)),
        }

    gap_improvement = (
        metrics["raw"]["expected_calibration_error"]
        - metrics["calibrated"]["expected_calibration_error"]
    )
    macro_loss = metrics["raw"]["oof_macro_f1"] - metrics["calibrated"]["oof_macro_f1"]
    adopt = bool(
        gap_improvement >= MIN_CALIBRATION_GAP_IMPROVEMENT and macro_loss <= MAX_MACRO_F1_LOSS
    )
    summary = {
        "method": CALIBRATION_METHOD,
        "rule": (
            f"adopt calibration when it reduces the expected calibration error by at "
            f"least {MIN_CALIBRATION_GAP_IMPROVEMENT} and costs at most "
            f"{MAX_MACRO_F1_LOSS} macro-F1"
        ),
        "fitted_on": "labelled notices only, inside grouped cross-validation splits",
        "expected_calibration_error_improvement": round(gap_improvement, 4),
        "macro_f1_change": round(-macro_loss, 4),
        "adopted": adopt,
        "variants": metrics,
        "deployed_variant": "calibrated" if adopt else "raw",
    }
    return pd.concat(tables, ignore_index=True), summary


# ---------------------------------------------------------------------------
# One entry point for the stage and the notebook
# ---------------------------------------------------------------------------


def run_all_specifications(
    corpus: pd.DataFrame, fold_index: np.ndarray, log=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested grouped cross-validation for every specification plus the floor.

    Returns per-fold results and pooled out-of-fold predictions. This is what
    both ``scripts/build_technology_taxonomy.py`` and the evidence notebook
    call, so a number shown in the notebook is produced by the same code path
    that wrote the table on disk.
    """
    specs = specifications()
    fold_frames = [majority_baseline(corpus, fold_index)]
    oof_frames = []
    for name, spec in specs.items():
        if log is not None:
            log(f"Evaluating {name}")
        fold_result, oof = run_specification(name, spec, corpus, fold_index)
        fold_frames.append(fold_result)
        oof_frames.append(oof)
    return pd.concat(fold_frames, ignore_index=True), pd.concat(oof_frames, ignore_index=True)


def comparison_table(
    fold_results: pd.DataFrame, oof: pd.DataFrame
) -> pd.DataFrame:
    """One row per model: fold means, spread, and pooled out-of-fold metrics."""
    specs = specifications()
    rows = []
    for name in ["M_majority", *specs]:
        block = fold_results.loc[fold_results["model"] == name]
        entry = {
            "model": name,
            "family": specs.get(name, {}).get("family", "baseline"),
            "features": specs.get(name, {}).get("features", "none"),
            "description": specs.get(name, {}).get(
                "description", "predicts the most frequent training class"
            ),
            "macro_f1_mean": round(float(block["macro_f1"].mean()), 4),
            "macro_f1_sd": round(float(block["macro_f1"].std(ddof=1)), 4),
            "macro_f1_min": round(float(block["macro_f1"].min()), 4),
            "macro_f1_max": round(float(block["macro_f1"].max()), 4),
            "weighted_f1_mean": round(float(block["weighted_f1"].mean()), 4),
            "macro_precision_mean": round(float(block["macro_precision"].mean()), 4),
            "macro_recall_mean": round(float(block["macro_recall"].mean()), 4),
            "accuracy_mean": round(float(block["accuracy"].mean()), 4),
            "train_macro_f1_mean": round(float(block["train_macro_f1"].mean()), 4),
            "train_validation_gap_mean": round(
                float(block["train_validation_gap"].mean()), 4
            ),
        }
        if name in specs:
            entry.update(pooled_metrics(oof.loc[oof["model"] == name]))
            selected = block["selected_params"].tolist()
            entry["selected_params_per_fold"] = json.dumps(selected)
            entry["selected_params_stable"] = len(set(selected)) == 1
        rows.append(entry)
    return pd.DataFrame(rows)


def load_corpus_and_folds() -> tuple[pd.DataFrame, np.ndarray]:
    """The frozen corpus with its frozen fold assignment, checked for alignment."""
    corpus = pd.read_parquet(TECHNOLOGY / "technology_corpus.parquet")
    folds = pd.read_csv(TECHNOLOGY / "nlp_cv_folds.csv", dtype={"idweb": str})
    if list(folds["idweb"]) != list(corpus["idweb"]):
        raise RuntimeError("fold file and corpus are not aligned")
    return corpus, folds["fold"].to_numpy()


def _log(message: str) -> None:
    """Progress reporting. The runner configures logging; the library only emits."""
    import logging

    logging.getLogger(__name__).info(message)


def build_evaluation_artifacts(force: bool = True) -> dict[str, Any]:
    comparison_path = TECHNOLOGY / "model_cv_results.csv"
    if comparison_path.exists() and not force:
        raise FileExistsError(f"{comparison_path} already exists. Use --force to rebuild.")
    corpus_path = TECHNOLOGY / "technology_corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError(f"{corpus_path} not found. Run build_technology_corpus.py first.")

    corpus, fold_index = load_corpus_and_folds()
    specs = specifications()
    fold_results, oof = run_all_specifications(corpus, fold_index, log=_log)

    comparison = comparison_table(fold_results, oof)

    selection = select_model(fold_results, comparison)
    decision = selection
    selected_model = selection["selected_model"]
    selected_spec = specs[selected_model]

    per_class = pd.concat(
        [per_class_table(oof, name) for name in specs], ignore_index=True
    )
    matrix = confusion_matrix(
        oof.loc[oof["model"] == selected_model, "true_label"],
        oof.loc[oof["model"] == selected_model, "predicted_label"],
        labels=list(CLASS_ORDER),
    )
    confusion_frame = pd.DataFrame(matrix, index=CLASS_ORDER, columns=CLASS_ORDER)
    confusion_frame.index.name = "true_label"

    audit_summary = json.loads(
        (TECHNOLOGY / "annotation_audit_summary.json").read_text(encoding="utf-8")
    )
    conflicting_groups = set(audit_summary["grouping"]["conflicting_group_ids"])
    error_sample, confusion_pairs = error_analysis(oof, corpus, selected_model, conflicting_groups)

    modal_params = json.loads(
        fold_results.loc[fold_results["model"] == selected_model, "selected_params"]
        .value_counts()
        .index[0]
    )
    curve = learning_curve(selected_spec, selected_model, corpus, fold_index, modal_params)
    temporal, temporal_per_class = temporal_validation(selected_spec, selected_model, corpus)

    # Uncertainty on the central claim. The benchmark contrast is the comparison
    # the component exists to make, so it is the one that gets a paired interval.
    benchmark = comparison.loc[
        comparison["family"] == "administrative_benchmark", "macro_f1_mean"
    ].idxmax()
    benchmark_model = comparison.loc[benchmark, "model"]
    bootstrap_models = sorted(
        {selected_model, decision["nominal_leader"], benchmark_model, "M1_tfidf_logreg"}
    )
    bootstrap_models = [m for m in specs if m in bootstrap_models]
    per_model_ci, paired_ci = family_bootstrap(oof, bootstrap_models)
    stability = hyperparameter_stability(fold_results)
    register = specification_register()
    gate = camembert_gate(comparison, selection, error_sample)

    TECHNOLOGY.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(TECHNOLOGY / "model_cv_fold_results.csv", index=False, encoding="utf-8")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    per_class.to_csv(TECHNOLOGY / "per_class_metrics.csv", index=False, encoding="utf-8")
    oof.to_csv(TECHNOLOGY / "oof_predictions.csv", index=False, encoding="utf-8")
    confusion_frame.to_csv(TECHNOLOGY / "confusion_matrix.csv", encoding="utf-8")
    confusion_pairs.to_csv(TECHNOLOGY / "confusion_pairs.csv", index=False, encoding="utf-8")
    error_sample.to_csv(TECHNOLOGY / "error_analysis.csv", index=False, encoding="utf-8")
    curve.to_csv(TECHNOLOGY / "learning_curve.csv", index=False, encoding="utf-8")
    per_model_ci.to_csv(TECHNOLOGY / "bootstrap_macro_f1_ci.csv", index=False, encoding="utf-8")
    paired_ci.to_csv(TECHNOLOGY / "bootstrap_paired_differences.csv", index=False, encoding="utf-8")
    stability.to_csv(TECHNOLOGY / "hyperparameter_stability.csv", index=False, encoding="utf-8")
    register.to_csv(TECHNOLOGY / "specification_register.csv", index=False, encoding="utf-8")
    pd.DataFrame([temporal]).to_csv(
        TECHNOLOGY / "temporal_validation_metrics.csv", index=False, encoding="utf-8"
    )
    temporal_per_class.to_csv(
        TECHNOLOGY / "temporal_validation_per_class.csv", index=False, encoding="utf-8"
    )
    confusion_figure(oof, selected_model, FIGURES / "technology_confusion_matrix.png")
    learning_curve_figure(curve, selected_model, FIGURES / "technology_learning_curve.png")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_version": EVALUATION_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "corpus_rows": int(len(corpus)),
        "groups": int(corpus["group_id"].nunique()),
        "n_splits": N_SPLITS,
        "random_seed": RANDOM_SEED,
        "models": list(specs),
        "selection": selection,
        "selected_model": selected_model,
        "selected_model_params_modal": modal_params,
        "selected_model_metrics": {
            **{
                key: float(comparison.loc[comparison["model"] == selected_model, key].iloc[0])
                for key in ("macro_f1_mean", "macro_f1_sd", "weighted_f1_mean", "accuracy_mean")
            },
            **pooled_metrics(oof.loc[oof["model"] == selected_model]),
        },
        "reliable_classes": reliable_classes(class_support(corpus["label"])),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "resampled_unit": "procurement family",
            "per_model": per_model_ci.to_dict("records"),
            "paired_differences": paired_ci.to_dict("records"),
            "headline_contrast": {
                "text_model": selected_model,
                "administrative_model": benchmark_model,
            },
        },
        "hyperparameter_stability": {
            str(model): bool(block["stable"].all())
            for model, block in stability.groupby("model")
        },
        "fit_diagnostics": {
            str(row["model"]): {
                "train_macro_f1": row["train_macro_f1_mean"],
                "cv_macro_f1": row["macro_f1_mean"],
                "gap": row["train_validation_gap_mean"],
                "fold_sd": row["macro_f1_sd"],
            }
            for row in comparison.loc[comparison["family"] != "baseline"].to_dict("records")
        },
        "learning_curve_tail_gain": round(
            float(curve["macro_f1_mean"].iloc[-1] - curve["macro_f1_mean"].iloc[-2]), 4
        ),
        "temporal_validation": temporal,
        "top_confusion_pairs": confusion_pairs.head(8).to_dict("records"),
        "error_triage_counts": error_sample["triage_reason"].value_counts().to_dict(),
        "camembert_gate": gate,
    }
    (TECHNOLOGY / "model_selection_decision.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# Uncertainty: the family bootstrap
# ---------------------------------------------------------------------------

#: Bootstrap replicates. Deterministic given ``RANDOM_SEED``; a thousand is
#: enough for a percentile interval reported to three decimals and costs
#: seconds, because each replicate only re-scores stored predictions.
BOOTSTRAP_REPLICATES = 1000


def _macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    return f1_score(truth, predicted, average="macro", labels=CLASS_ORDER, zero_division=0)


def family_bootstrap(
    oof: pd.DataFrame,
    models: Sequence[str],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Percentile intervals for macro-F1, resampling procurement families.

    Resampling *families* rather than notices is the whole point. Two notices in
    one family are near-copies; treating them as independent draws would shrink
    the interval by pretending the corpus holds more information than it does.

    Every model is scored on the **same** resampled families in each replicate,
    so the paired difference between two models is computed within replicate and
    its interval reflects the correlation between them. An unpaired comparison
    of two separately-bootstrapped intervals would be far more conservative and
    would answer a question nobody asked.

    Returns per-model intervals and pairwise paired differences.
    """
    rows = oof.loc[oof["model"].isin(models)]
    families = np.array(sorted(rows["group_id"].unique()))
    by_family = {
        name: {
            family: (block["true_label"].to_numpy(), block["predicted_label"].to_numpy())
            for family, block in rows.loc[rows["model"] == name].groupby("group_id")
        }
        for name in models
    }

    generator = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in models}
    for _ in range(replicates):
        sampled = generator.choice(families, size=len(families), replace=True)
        for name in models:
            truth = np.concatenate([by_family[name][family][0] for family in sampled])
            predicted = np.concatenate([by_family[name][family][1] for family in sampled])
            draws[name].append(_macro_f1(truth, predicted))

    observed = {
        name: _macro_f1(
            rows.loc[rows["model"] == name, "true_label"].to_numpy(),
            rows.loc[rows["model"] == name, "predicted_label"].to_numpy(),
        )
        for name in models
    }
    per_model = pd.DataFrame(
        [
            {
                "model": name,
                "macro_f1": round(observed[name], 4),
                "ci_lower": round(float(np.percentile(draws[name], 2.5)), 4),
                "ci_upper": round(float(np.percentile(draws[name], 97.5)), 4),
                "bootstrap_sd": round(float(np.std(draws[name], ddof=1)), 4),
                "replicates": replicates,
                "resampled_unit": "procurement family",
                "families": len(families),
            }
            for name in models
        ]
    )

    pairs = []
    for index, left in enumerate(models):
        for right in models[index + 1:]:
            difference = np.array(draws[left]) - np.array(draws[right])
            lower = float(np.percentile(difference, 2.5))
            upper = float(np.percentile(difference, 97.5))
            pairs.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "observed_difference": round(observed[left] - observed[right], 4),
                    "ci_lower": round(lower, 4),
                    "ci_upper": round(upper, 4),
                    "excludes_zero": bool(lower > 0 or upper < 0),
                    "share_of_replicates_a_higher": round(float((difference > 0).mean()), 4),
                    "replicates": replicates,
                }
            )
    return per_model, pd.DataFrame(pairs)


def hyperparameter_stability(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Which configuration each outer fold selected, and whether it agreed.

    A specification that wins with a different configuration in every fold is
    telling you the inner selection is reading noise, and its outer score should
    be trusted less than its mean suggests.
    """
    rows = []
    for model, block in fold_results.groupby("model"):
        selected = block["selected_params"].tolist()
        distinct = sorted(set(selected))
        for row in block.itertuples(index=False):
            parameters = json.loads(row.selected_params)
            rows.append(
                {
                    "model": model,
                    "fold": row.fold,
                    **{key.replace("clf__", "").replace("tfidf__", ""): str(value)
                       for key, value in sorted(parameters.items())},
                    "inner_macro_f1": row.inner_macro_f1,
                    "outer_macro_f1": row.macro_f1,
                    "distinct_configurations_across_folds": len(distinct),
                    "stable": len(distinct) == 1,
                }
            )
    return pd.DataFrame(rows)


def specification_register() -> pd.DataFrame:
    """Every specification in the development budget, winner or not.

    Reporting only the selected model would hide how much searching produced it.
    This table is written on every run so the budget is auditable rather than
    asserted.
    """
    return pd.DataFrame(
        [
            {
                "model": name,
                "family": spec["family"],
                "features": spec["features"],
                "description": spec["description"],
                "grid_points": len(grid_points(spec["grid"])),
                "grid": json.dumps(
                    {key: [str(v) for v in values] for key, values in sorted(spec["grid"].items())}
                ),
                "searched_by": "grouped inner CV nested in each outer training fold",
            }
            for name, spec in specifications().items()
        ]
    )
