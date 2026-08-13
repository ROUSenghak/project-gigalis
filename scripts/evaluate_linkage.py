#!/usr/bin/env python3
"""Evaluate successor linkage against the manually reviewed benchmark.

Four methods are compared, then one is frozen:

``M_A_deterministic``
    Accept only on verified SIREN identity plus CPV continuity plus a text
    floor. Deliberately rigid; establishes what exact evidence alone achieves.

``M_B_text_ranking``
    Rank same-buyer candidates by TF-IDF text similarity alone. Tests whether
    the structured components earn their place.

``M_C_weighted_gated``
    The weighted score with independent acceptance gates, ported from
    ``notebooks/08``. Requires evidence from several dimensions at once.

``M_D_fellegi_sunter``
    Classical probabilistic record linkage. Unlike the three above, its
    weights are *estimated* from the candidate pairs by expectation
    maximisation rather than chosen by hand, and it needs no labels to fit.
    See ``scripts/fit_fellegi_sunter.py``.

Every method may abstain: an anchor with no candidate clearing its rule
returns *no automatic successor* rather than its best guess.

Two honesty constraints are enforced in the reporting:

* Thresholds are selected on ``PILOT_DEVELOPMENT`` only.
* ``LOCKED_TEST`` is reported but is **not** a pristine holdout. Notebook 07
  inspected locked-test error counts and feature medians before the gated
  method was designed in notebook 08, so its numbers are labelled
  already-inspected evaluation evidence.

A further limitation is recorded in the output rather than hidden: the human
reviewers only ever saw the *old* pipeline's top-25 candidates per anchor, so
a true successor the old ranker buried was never available to be labelled.
Recall measured here is therefore optimistic, and "no observed successor" is
relative to that review, not to BOAMP as a whole.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.benchmark_io import load_truth_v1, load_truth_v3
from boamp_pipeline.benchmark_metrics import (
    annotator_bias_report,
    hard_negative_suite_metrics,
    weighted_evaluate,
)
from boamp_pipeline.fellegi_sunter_scoring import score_with_fitted_model
from boamp_pipeline.linkage import DEFAULT_WEIGHTS

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp_v2")
EVALUATION_VERSION = "boamp_linkage_evaluation_v1.0"

RECALL_AT_K = (1, 5, 10, 20)

#: Threshold grid explored on the pilot split only. Kept deliberately coarse:
#: with 5 positive pilot anchors, one prediction flip moves precision by 0.2,
#: so a fine grid would be fitting noise.
THRESHOLD_GRID = (50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0)

STRONG_BUYER_MATCHES = ("siren", "normalized_name")
FUZZY_BUYER_MIN_SIMILARITY = 0.82

#: Frozen linkage policy for the survival analysis.
#:
#: This is a documented judgement, not an automatic selection, and the
#: reasoning is recorded here because it must survive into the write-up.
#:
#: The pilot split holds 5 positive anchors, so a single prediction change
#: moves its precision by 0.2. It cannot discriminate between the three
#: methods, and an automatic "best pilot precision" rule is fitting noise --
#: it initially selected ``M_C_weighted_gated`` only because an arbitrary
#: coverage floor of 0.25 excluded the more precise but lower-coverage
#: methods.
#:
#: With event volume not binding, the stated a-priori principle governs: a
#: false link fabricates both a survival event and its event time, so precision
#: and the false-positive rate on no-successor anchors take priority over
#: coverage. The current computed metrics live in the JSON summaries; this
#: comment deliberately avoids frozen numeric claims.
#:
#: Honesty constraint: this choice draws on the locked split, which was
#: already inspected in notebooks/07 before the present work and therefore
#: never functioned as an untouched holdout. Every metric reported here is
#: development evidence, and the true out-of-sample precision is likely
#: lower than the figures quoted.
PRIMARY_METHOD = "M_B_text_ranking"
PRIMARY_THRESHOLD = 70.0

#: Contrast arm carried through the survival sensitivity analysis: a much
#: higher-recall, lower-precision policy. If the survival conclusions hold
#: under both, they do not rest on where the acceptance line was drawn.
CONTRAST_METHOD = "M_C_weighted_gated"
CONTRAST_THRESHOLD = 70.0

M_C_DEFAULTS: dict[str, float] = {
    "min_text_similarity": 0.25,
    "text_override_for_missing_cpv": 0.55,
    "min_text_for_fuzzy_buyer": 0.40,
}


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "evaluate_linkage.log", encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# Benchmark truth
# ---------------------------------------------------------------------------


def load_truth(output_dir: Path) -> pd.DataFrame:
    """One row per evaluable benchmark anchor, mapped onto v2 episode IDs.

    The v1 body now lives in :func:`boamp_pipeline.benchmark_io.load_truth_v1`,
    moved verbatim so that adding the v3 benchmark cannot change a v1 number.
    A test asserts the v1 summary still reproduces field for field.
    """
    return load_truth_v1(output_dir)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def strong_buyer(frame: pd.DataFrame) -> pd.Series:
    return frame["buyer_match_type"].isin(STRONG_BUYER_MATCHES) | (
        frame["buyer_name_similarity"].fillna(0).ge(FUZZY_BUYER_MIN_SIMILARITY)
    )


def method_a_deterministic(frame: pd.DataFrame, threshold: float) -> pd.Series:
    """Verified SIREN identity + CPV continuity + a text floor."""
    return (
        frame["buyer_match_type"].eq("siren")
        & frame["cpv_component"].fillna(0).gt(0)
        & frame["text_component"].fillna(0).ge(0.25)
    )


def method_b_text(frame: pd.DataFrame, threshold: float) -> pd.Series:
    """Text similarity alone, on the same-buyer candidate pool."""
    return frame["text_component"].fillna(0).ge(threshold / 100.0)


def method_d_fellegi_sunter(frame: pd.DataFrame, threshold: float) -> pd.Series:
    """Posterior match probability from the fitted Fellegi-Sunter model.

    The threshold is expressed on the same ``threshold / 100`` convention as
    the other methods, so 60 means a posterior of 0.60. The fitted posterior
    tops out near 0.73 rather than approaching 1, because the unsupervised
    mixture's prior is not calibrated to the true successor rate; the ranking
    it induces is unaffected by that, which is what this method is judged on.
    """
    if "fs_match_probability" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["fs_match_probability"].fillna(0).ge(threshold / 100.0)


def method_c_weighted_gated(frame: pd.DataFrame, threshold: float) -> pd.Series:
    """Weighted score plus independent buyer / text / CPV gates."""
    text = frame["text_component"].fillna(0)
    cpv_present = frame["cpv_component"].notna() & frame["cpv_component"].gt(0)

    buyer_gate = strong_buyer(frame)
    # Missing CPV is not a mismatch, but it removes a dimension of evidence,
    # so text must carry more weight before the pair is accepted.
    cpv_gate = cpv_present | text.ge(M_C_DEFAULTS["text_override_for_missing_cpv"])
    text_gate = text.ge(M_C_DEFAULTS["min_text_similarity"])
    fuzzy_gate = frame["buyer_match_type"].isin(STRONG_BUYER_MATCHES) | (
        text.ge(M_C_DEFAULTS["min_text_for_fuzzy_buyer"]) & cpv_present
    )
    return buyer_gate & cpv_gate & text_gate & fuzzy_gate & frame["linkage_score"].fillna(0).ge(threshold)


METHODS: dict[str, Callable[[pd.DataFrame, float], pd.Series]] = {
    "M_A_deterministic": method_a_deterministic,
    "M_B_text_ranking": method_b_text,
    "M_C_weighted_gated": method_c_weighted_gated,
    "M_D_fellegi_sunter": method_d_fellegi_sunter,
}

#: Methods whose threshold lives on a different scale need their own grid.
#: The Fellegi-Sunter posterior peaks near 0.73, so the 50-80 grid used by the
#: score-based methods would reject everything above 75.
METHOD_THRESHOLD_GRID: dict[str, tuple[float, ...]] = {
    "M_D_fellegi_sunter": (20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0),
}

#: Column each method ranks its accepted candidates by.
RANK_COLUMN = {
    "M_A_deterministic": "linkage_score",
    "M_B_text_ranking": "text_component",
    "M_C_weighted_gated": "linkage_score",
    "M_D_fellegi_sunter": "fs_match_probability",
}


def predict(candidates: pd.DataFrame, method: str, threshold: float) -> pd.DataFrame:
    """Top-1 accepted successor per anchor, or no row when the method abstains."""
    accepted = candidates.loc[METHODS[method](candidates, threshold)].copy()
    if accepted.empty:
        return accepted.assign(predicted_rank=pd.Series(dtype=int))
    accepted = accepted.sort_values(
        ["anchor_episode_id", RANK_COLUMN[method]], ascending=[True, False]
    )
    accepted["predicted_rank"] = accepted.groupby("anchor_episode_id").cumcount() + 1
    return accepted


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def candidate_recall(candidates: pd.DataFrame, truth: pd.DataFrame, ks: tuple[int, ...]) -> dict[str, Any]:
    """Does the ranked candidate list contain the manually confirmed successor?"""
    positives = truth.loc[truth["has_successor"] & truth["truth_usable"]]
    by_anchor = {
        anchor: group.sort_values("linkage_score", ascending=False)["candidate_episode_id"].tolist()
        for anchor, group in candidates.groupby("anchor_episode_id")
    }
    result: dict[str, Any] = {"positive_anchors_evaluated": 0, "positive_anchors_without_candidates": 0}
    hits = {k: 0 for k in ks}
    ranks: list[int] = []
    for row in positives.itertuples(index=False):
        ranked = by_anchor.get(row.anchor_v2_episode_id)
        if not ranked:
            result["positive_anchors_without_candidates"] += 1
            continue
        result["positive_anchors_evaluated"] += 1
        true_set = set(row.true_successors)
        found = [i + 1 for i, cid in enumerate(ranked) if cid in true_set]
        if found:
            ranks.append(min(found))
        for k in ks:
            if any(r <= k for r in found):
                hits[k] += 1
    evaluated = result["positive_anchors_evaluated"]
    result["recall_at_k"] = {
        f"recall@{k}": round(hits[k] / evaluated, 4) if evaluated else None for k in ks
    }
    result["true_successor_rank"] = {
        "median": float(np.median(ranks)) if ranks else None,
        "max": int(max(ranks)) if ranks else None,
        "retrieved_anywhere": len(ranks),
    }
    return result


def evaluate(predictions: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    """Precision / recall / FPR / coverage for one method at one threshold."""
    top1 = (
        predictions.loc[predictions["predicted_rank"] == 1]
        .set_index("anchor_episode_id")["candidate_episode_id"]
        .to_dict()
        if len(predictions)
        else {}
    )
    top5 = (
        predictions.loc[predictions["predicted_rank"] <= 5]
        .groupby("anchor_episode_id")["candidate_episode_id"]
        .apply(list)
        .to_dict()
        if len(predictions)
        else {}
    )

    tp = fp_wrong = fn_no_link = 0
    fp_on_negative = tn_no_link = 0
    hits5 = 0
    positives = negatives = 0
    accepted = 0

    for row in truth.itertuples(index=False):
        anchor = row.anchor_v2_episode_id
        prediction = top1.get(anchor)
        if prediction is not None:
            accepted += 1
        if row.has_successor:
            positives += 1
            true_set = set(row.true_successors)
            if prediction is None:
                fn_no_link += 1
            elif prediction in true_set:
                tp += 1
            else:
                fp_wrong += 1
            if true_set & set(top5.get(anchor, [])):
                hits5 += 1
        else:
            negatives += 1
            if prediction is None:
                tn_no_link += 1
            else:
                fp_on_negative += 1

    total = positives + negatives
    return {
        "anchors_evaluated": total,
        "positive_anchors": positives,
        "negative_anchors": negatives,
        "accepted_links": accepted,
        "true_positive": tp,
        "false_positive_wrong_successor": fp_wrong,
        "false_positive_on_no_successor_anchor": fp_on_negative,
        "false_negative_abstained": fn_no_link,
        "true_negative_abstained": tn_no_link,
        "precision_at_1": round(tp / accepted, 4) if accepted else None,
        "recall_at_1": round(tp / positives, 4) if positives else None,
        "recall_at_5": round(hits5 / positives, 4) if positives else None,
        "false_positive_rate_on_negatives": round(fp_on_negative / negatives, 4) if negatives else None,
        "no_link_accuracy": round(tn_no_link / negatives, 4) if negatives else None,
        "coverage": round(accepted / total, 4) if total else None,
    }


def build(project_root: Path, output_dir: Path) -> dict[str, Any]:
    # Prefer the Fellegi-Sunter-scored candidates when they exist, so that
    # M_D can be evaluated alongside the rule-based methods on identical pairs.
    scored_path = output_dir / "linkage_candidates_scored.parquet"
    candidates_path = scored_path if scored_path.exists() else output_dir / "linkage_candidates.parquet"
    candidates = pd.read_parquet(candidates_path)
    logging.info("Candidates: %s (%s)", f"{len(candidates):,}", candidates_path.name)
    truth = load_truth(output_dir)

    scored_anchors = set(candidates["anchor_episode_id"].unique())
    truth["anchor_in_cohort"] = truth["anchor_v2_episode_id"].isin(scored_anchors)
    evaluable = truth.loc[truth["anchor_in_cohort"] & truth["truth_usable"]].copy()
    logging.info(
        "Benchmark anchors: %s total, %s scored in cohort", len(truth), len(evaluable)
    )

    pilot = evaluable.loc[evaluable["benchmark_split"] == "PILOT_DEVELOPMENT"]
    locked = evaluable.loc[evaluable["benchmark_split"] == "LOCKED_TEST"]

    eval_candidates = candidates.loc[
        candidates["anchor_episode_id"].isin(set(evaluable["anchor_v2_episode_id"]))
    ].copy()

    retrieval = {
        "all_evaluable": candidate_recall(eval_candidates, evaluable, RECALL_AT_K),
        "pilot": candidate_recall(eval_candidates, pilot, RECALL_AT_K),
        "locked_test": candidate_recall(eval_candidates, locked, RECALL_AT_K),
    }
    logging.info("Candidate retrieval (all evaluable): %s", retrieval["all_evaluable"]["recall_at_k"])

    # --- threshold selection on the pilot split only -----------------------
    grid: list[dict[str, Any]] = []
    for method in METHODS:
        if method == "M_D_fellegi_sunter" and "fs_match_probability" not in candidates.columns:
            continue
        for threshold in METHOD_THRESHOLD_GRID.get(method, THRESHOLD_GRID):
            predictions = predict(eval_candidates, method, threshold)
            pilot_metrics = evaluate(
                predictions.loc[predictions["anchor_episode_id"].isin(set(pilot["anchor_v2_episode_id"]))],
                pilot,
            )
            grid.append({"method": method, "threshold": threshold, "split": "PILOT_DEVELOPMENT", **pilot_metrics})
    grid_frame = pd.DataFrame(grid)
    grid_frame.to_csv(output_dir / "linkage_threshold_grid_pilot.csv", index=False)

    # The frozen policy is a documented judgement (see PRIMARY_METHOD), not an
    # automatic pick: with 5 positive pilot anchors, any argmax over pilot
    # precision is fitting noise.
    selected_method = PRIMARY_METHOD
    selected_threshold = PRIMARY_THRESHOLD
    logging.info("Frozen policy: %s at threshold %s", selected_method, selected_threshold)

    # --- report every method once on the locked split ----------------------
    comparison: list[dict[str, Any]] = []
    for method in METHODS:
        if method == "M_D_fellegi_sunter" and "fs_match_probability" not in candidates.columns:
            continue
        # The three score-scale methods are compared at the frozen operating
        # point so the headline table stays anchored to the policy in force.
        # M_D lives on a probability scale that caps near 0.73, so a threshold
        # of 70 means something different for it; it gets its own sweep below.
        method_threshold = (
            METHOD_THRESHOLD_GRID["M_D_fellegi_sunter"][-2]
            if method == "M_D_fellegi_sunter" else selected_threshold
        )
        predictions = predict(eval_candidates, method, method_threshold)
        for label, split_truth in (("PILOT_DEVELOPMENT", pilot), ("LOCKED_TEST", locked), ("ALL", evaluable)):
            subset = predictions.loc[
                predictions["anchor_episode_id"].isin(set(split_truth["anchor_v2_episode_id"]))
            ]
            comparison.append(
                {"method": method, "threshold": method_threshold, "split": label, **evaluate(subset, split_truth)}
            )
    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(output_dir / "linkage_method_comparison.csv", index=False)

    # --- freeze and apply the selected policy to the whole cohort ----------
    final_predictions = predict(candidates, selected_method, selected_threshold)
    accepted_links = final_predictions.loc[final_predictions["predicted_rank"] == 1].copy()
    accepted_links.to_parquet(output_dir / "accepted_successor_links.parquet", index=False, compression="zstd")

    anchors_scored = candidates["anchor_episode_id"].nunique()
    sensitivity = {}
    arms = (
        ("strict", selected_method, selected_threshold + 10.0),
        ("main", selected_method, selected_threshold),
        ("looser", selected_method, selected_threshold - 10.0),
        ("contrast_high_recall", CONTRAST_METHOD, CONTRAST_THRESHOLD),
    )
    for label, arm_method, arm_threshold in arms:
        variant = predict(candidates, arm_method, arm_threshold)
        variant_top1 = variant.loc[variant["predicted_rank"] == 1] if len(variant) else variant
        arm_predictions = predict(eval_candidates, arm_method, arm_threshold)
        locked_subset = arm_predictions.loc[
            arm_predictions["anchor_episode_id"].isin(set(locked["anchor_v2_episode_id"]))
        ] if len(arm_predictions) else arm_predictions
        sensitivity[label] = {
            "method": arm_method,
            "threshold": arm_threshold,
            "accepted_links": int(len(variant_top1)),
            "cohort_link_rate": round(len(variant_top1) / anchors_scored, 4) if anchors_scored else None,
            "locked_test_metrics": evaluate(locked_subset, locked),
        }

    # --- pre-registered adoption test for the probabilistic model -----------
    # Fixed before the model was fitted: Fellegi-Sunter replaces the incumbent
    # only if it weakly dominates on the locked split (precision >=, recall >=,
    # false-positive rate <=). It is swept across its whole grid so the test is
    # applied to its best available operating point, not an arbitrary one.
    adoption: dict[str, Any] | None = None
    if "fs_match_probability" in candidates.columns:
        incumbent_predictions = predict(eval_candidates, selected_method, selected_threshold)
        incumbent = evaluate(
            incumbent_predictions.loc[
                incumbent_predictions["anchor_episode_id"].isin(set(locked["anchor_v2_episode_id"]))
            ],
            locked,
        )
        sweep = []
        for threshold in METHOD_THRESHOLD_GRID["M_D_fellegi_sunter"]:
            challenger_predictions = predict(eval_candidates, "M_D_fellegi_sunter", threshold)
            subset = challenger_predictions.loc[
                challenger_predictions["anchor_episode_id"].isin(set(locked["anchor_v2_episode_id"]))
            ] if len(challenger_predictions) else challenger_predictions
            metrics = evaluate(subset, locked)
            metrics["threshold"] = threshold
            metrics["weakly_dominates_incumbent"] = bool(
                (metrics["precision_at_1"] or 0) >= (incumbent["precision_at_1"] or 0)
                and (metrics["recall_at_1"] or 0) >= (incumbent["recall_at_1"] or 0)
                and (metrics["false_positive_rate_on_negatives"] or 0)
                <= (incumbent["false_positive_rate_on_negatives"] or 0)
            )
            sweep.append(metrics)
        adoption = {
            "rule": (
                "Fellegi-Sunter replaces the incumbent only if it weakly dominates on the "
                "locked split: precision >=, recall >=, and false-positive rate <=. Fixed "
                "before the model was fitted."
            ),
            "incumbent": {"method": selected_method, "threshold": selected_threshold, **incumbent},
            "challenger_locked_sweep": sweep,
            "any_operating_point_dominates": any(row["weakly_dominates_incumbent"] for row in sweep),
            "outcome": (
                "adopted" if any(row["weakly_dominates_incumbent"] for row in sweep) else "incumbent_retained"
            ),
        }
        logging.info("Fellegi-Sunter adoption outcome: %s", adoption["outcome"])

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_version": EVALUATION_VERSION,
        "selected_method": selected_method,
        "selected_threshold": selected_threshold,
        "contrast_method": CONTRAST_METHOD,
        "contrast_threshold": CONTRAST_THRESHOLD,
        "weights": dict(DEFAULT_WEIGHTS),
        "gate_parameters": M_C_DEFAULTS,
        "selection_rule": (
            "Documented judgement, not an automatic argmax. The pilot split has 5 positive "
            "anchors, so its precision moves by 0.2 per prediction and cannot discriminate "
            "between methods. The a-priori precision-first principle governs: a false "
            "link fabricates both a survival event and its event time. Current precision, "
            "recall, false-positive-rate, and event-volume values are reported in the "
            "structured summary fields generated in the same run. This choice draws on "
            "the locked split, which was already inspected in notebooks/07 and never "
            "functioned as an untouched holdout."
        ),
        "threshold_grid": list(THRESHOLD_GRID),
    }
    (output_dir / "linkage_frozen_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        **config,
        "benchmark": {
            "anchors_in_benchmark": int(len(truth)),
            "anchors_scored_in_cohort": int(len(evaluable)),
            "anchors_not_in_cohort": int((~truth["anchor_in_cohort"]).sum()),
            "pilot_anchors": int(len(pilot)),
            "locked_test_anchors": int(len(locked)),
            "pilot_positive_anchors": int(pilot["has_successor"].sum()),
            "locked_test_positive_anchors": int(locked["has_successor"].sum()),
        },
        "candidate_retrieval": retrieval,
        "method_comparison": comparison,
        "fellegi_sunter_adoption": adoption,
        "cohort_application": {
            "cohort_anchors_with_candidates": int(candidates["anchor_episode_id"].nunique()),
            "accepted_links": int(len(accepted_links)),
            "link_rate": round(len(accepted_links) / candidates["anchor_episode_id"].nunique(), 4),
        },
        "threshold_sensitivity": sensitivity,
        "caveats": {
            "locked_test_is_not_a_clean_holdout": (
                "notebooks/07 inspected locked-test error counts and per-error-type feature "
                "medians before the gated method was designed in notebooks/08. Locked-test "
                "numbers are already-inspected evaluation evidence, not an untouched holdout."
            ),
            "benchmark_conditioned_on_previous_ranker": (
                "Reviewers saw only the previous pipeline's top-25 candidates per anchor "
                "(review_candidates_exported_n = 25 for 97 of 120 anchors, against a broad "
                "pool with median 146). Successors that ranker buried were never available "
                "to label, so recall here is optimistic and 'no observed successor' is "
                "relative to that review rather than to BOAMP."
            ),
            "small_benchmark": (
                "With a handful of positive pilot anchors, a single prediction change moves "
                "precision substantially. Metrics rule out clearly bad operating points; "
                "they do not identify an optimum."
            ),
        },
    }
    (output_dir / "linkage_evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_v3(project_root: Path, output_dir: Path, split: str, event_set: str,
             open_sealed: bool, seal_reason: str) -> dict[str, Any]:
    """Evaluate the frozen methods against the national v3 benchmark.

    Reported alongside, never instead of, the v1 result. Three things the v1
    path could not do happen here: estimates are design-weighted with
    intervals, a false-positive rate is computed only over anchors whose whole
    pool was reviewed, and every method is additionally scored on a suite of
    pairs that are provably not renewals.
    """
    benchmark_dir = output_dir / "benchmark_v3"
    exposure = pd.read_parquet(benchmark_dir / "exposure_full.parquet")
    exposure = score_with_fitted_model(exposure, output_dir / "fellegi_sunter_model.json")
    truth, access_record = load_truth_v3(
        benchmark_dir, split, event_set,
        project_root=project_root, allow_sealed=open_sealed, seal_reason=seal_reason,
    )
    logging.info(
        "v3 benchmark: split=%s event_set=%s anchors=%s", split, event_set, len(truth)
    )

    # The benchmark's own universe: a prediction outside it is out of scope,
    # not wrong.
    pool_membership: dict[str, set[str]] = {}
    for anchor, group in exposure.groupby("anchor_episode_id"):
        pool_membership[anchor] = set(group["candidate_episode_id"])

    scored = exposure.rename(columns={"anchor_episode_id": "anchor_episode_id"}).copy()
    evaluable = scored.loc[scored["anchor_episode_id"].isin(set(truth["anchor_v2_episode_id"]))]

    negatives_path = benchmark_dir / "structural_negatives.parquet"
    negatives = pd.read_parquet(negatives_path) if negatives_path.exists() else pd.DataFrame()

    results: list[dict[str, Any]] = []
    for method in METHODS:
        if method == "M_D_fellegi_sunter" and "fs_match_probability" not in evaluable.columns:
            continue
        threshold = (
            METHOD_THRESHOLD_GRID["M_D_fellegi_sunter"][-2]
            if method == "M_D_fellegi_sunter" else PRIMARY_THRESHOLD
        )
        predictions = predict(evaluable, method, threshold)
        top1 = predictions.loc[predictions["predicted_rank"] == 1] if len(predictions) else predictions
        entry = {
            "method": method,
            "threshold": threshold,
            "unweighted_all_frames": evaluate(predictions, truth),
        }
        # Design-weighted estimation is only meaningful on the probability
        # frame. Enrichment anchors were purposively recruited and carry no
        # inclusion probability, so they are reported unweighted and separately
        # rather than averaged into a national figure.
        probability = truth.loc[truth.get("frame", "PROBABILITY") == "PROBABILITY"]
        enrichment = truth.loc[truth.get("frame", "PROBABILITY") != "PROBABILITY"]
        if len(probability):
            entry["weighted_national"] = weighted_evaluate(
                predictions, probability, pool_membership=pool_membership
            )
        else:
            entry["weighted_national"] = {
                "anchors": 0,
                "note": "no probability-frame anchors labelled yet; no national estimate possible",
            }
        if len(enrichment):
            entry["unweighted_enrichment"] = evaluate(predictions, enrichment)
        if len(negatives):
            entry["hard_negatives"] = hard_negative_suite_metrics(top1, negatives)
        results.append(entry)

    labels_path = benchmark_dir / "labels_adjudicated.parquet"
    bias = {}
    if labels_path.exists():
        labels = pd.read_parquet(labels_path)
        settled = labels.loc[labels["label"].notna()]
        bias = annotator_bias_report(settled, exposure)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": "v3",
        "split": split,
        "event_set": event_set,
        "scoring_sources": {
            "exposure": str(benchmark_dir / "exposure_full.parquet"),
            "fellegi_sunter_model": str(output_dir / "fellegi_sunter_model.json"),
        },
        "anchors_evaluated": int(len(truth)),
        "methods": results,
        "annotator_bias": bias,
        "sealed_access_record": access_record,
        "caveats": {
            "annotation_source": (
                "Labels were produced by a language model under a written protocol with "
                "double annotation, adjudication and enforced verbatim evidence. Kappa "
                "measures self-consistency, not inter-annotator agreement."
            ),
            "exposure_universe": (
                "A negative means no successor inside the benchmark's candidate pool. "
                "Predictions outside that pool are reported separately."
            ),
        },
        "validation_passed": bool(results),
    }
    suffix = f"_{split}_{event_set}"
    (output_dir / f"linkage_evaluation_summary_v3{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--benchmark", choices=["v1", "v3"], default="v1",
        help="Which benchmark to evaluate against. Defaults to v1 so the frozen "
             "workflow reproduces byte for byte.",
    )
    parser.add_argument(
        "--split", choices=["dev", "validation", "sealed_test"], default="dev",
        help="v3 only. dev tunes thresholds, validation selects methods, "
             "sealed_test is opened once through an audited path.",
    )
    parser.add_argument(
        "--event-set", choices=["strict", "primary", "broad"], default="primary",
        help="v3 only. Which label classes count as an event.",
    )
    parser.add_argument(
        "--open-sealed-test", action="store_true",
        help="Authorise reading the sealed split. Every opening is logged.",
    )
    parser.add_argument(
        "--seal-reason", default="",
        help="Why the sealed split is being opened. Required with --open-sealed-test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    configure_logging(project_root)
    if args.benchmark == "v3":
        summary = build_v3(
            project_root, output_dir, args.split, args.event_set,
            args.open_sealed_test, args.seal_reason,
        )
    else:
        summary = build(project_root, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
