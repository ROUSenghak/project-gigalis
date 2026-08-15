#!/usr/bin/env python3
"""Apply and evaluate successor linkage under one canonical workflow.

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

Evaluation runs against the Grand Ouest regional reference sample
(``scripts/build_regional_benchmark.py``). Its labels were established by
review of real BOAMP notices before these methods existed, so unlike the
retired France-level benchmark it does not score a method against a rule built
from the method's own evidence. ``dev`` is the reference's own pilot stratum
and exists for threshold display; ``validation`` is its locked stratum, and the
operating point below was frozen before either was consulted.

Without ``--evaluate-split``, the script applies the frozen production policy
to the complete study cohort.
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

from boamp_pipeline.benchmark_metrics import annotator_bias_report, weighted_evaluate
from boamp_pipeline.fellegi_sunter_scoring import score_with_fitted_model
from boamp_pipeline.linkage import DEFAULT_WEIGHTS
from boamp_pipeline.regional_benchmark_io import load_manifest, load_truth, wilson_interval

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp")
EVALUATION_SCHEMA = "boamp_linkage_evaluation_schema_1.0"

RECALL_AT_K = (1, 5, 10, 20)

#: Coarse grid used for transparent threshold sensitivity analysis.
THRESHOLD_GRID = (50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0)

STRONG_BUYER_MATCHES = ("siren", "normalized_name")
FUZZY_BUYER_MIN_SIMILARITY = 0.82

#: Frozen linkage policy for the survival analysis.
#:
#: This is a documented judgement, not an automatic selection, and the
#: reasoning is recorded here because it must survive into the write-up.
#:
#: With event volume not binding, the stated a-priori principle governs: a
#: false link fabricates both a survival event and its event time, so precision
#: and the false-positive rate on no-successor anchors take priority over
#: coverage. The current computed metrics live in the JSON summaries; this
#: comment deliberately avoids frozen numeric claims.
#:
#: The choice remains a frozen conservative baseline rather than a claim that
#: 0.70 is universally optimal, and 0.60 is carried as a required sensitivity
#: arm. The operating point was fixed before the regional reference was
#: consulted and was not moved afterwards, which is what allows the regional
#: locked split to be read as held-out rather than as a tuning set.
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
    sort_columns = ["anchor_episode_id", RANK_COLUMN[method]]
    ascending = [True, False]
    for column in ("candidate_origin_date", "candidate_episode_id"):
        if column in accepted.columns and column not in sort_columns:
            sort_columns.append(column)
            ascending.append(True)
    accepted = accepted.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    )
    accepted["predicted_rank"] = accepted.groupby("anchor_episode_id").cumcount() + 1
    return accepted


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def candidate_recall(candidates: pd.DataFrame, truth: pd.DataFrame, ks: tuple[int, ...]) -> dict[str, Any]:
    """Does the ranked candidate list contain the reference-labelled successor?"""
    positives = truth.loc[truth["has_successor"] & truth["truth_usable"]]
    by_anchor = {
        anchor: group.sort_values("linkage_score", ascending=False)["candidate_episode_id"].tolist()
        for anchor, group in candidates.groupby("anchor_episode_id")
    }
    result: dict[str, Any] = {"positive_anchors_evaluated": 0, "positive_anchors_without_candidates": 0}
    hits = {k: 0 for k in ks}
    ranks: list[int] = []
    for row in positives.itertuples(index=False):
        ranked = by_anchor.get(row.anchor_episode_id)
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
        anchor = row.anchor_episode_id
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
        "precision_at_1_interval_95": wilson_interval(tp, accepted),
        "recall_at_1_interval_95": wilson_interval(tp, positives),
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


def apply_primary_linkage(output_dir: Path) -> dict[str, Any]:
    """Apply the frozen policy and materialise production linkage artifacts."""
    scored_path = output_dir / "linkage_candidates_scored.parquet"
    candidates_path = scored_path if scored_path.exists() else output_dir / "linkage_candidates.parquet"
    candidates = pd.read_parquet(candidates_path)

    predictions = predict(candidates, PRIMARY_METHOD, PRIMARY_THRESHOLD)
    accepted = predictions.loc[predictions["predicted_rank"].eq(1)].copy()
    accepted.to_parquet(
        output_dir / "accepted_successor_links.parquet",
        index=False,
        compression="zstd",
    )

    anchors = int(candidates["anchor_episode_id"].nunique())
    arms = (
        ("strict", PRIMARY_METHOD, PRIMARY_THRESHOLD + 10.0),
        ("main", PRIMARY_METHOD, PRIMARY_THRESHOLD),
        ("looser", PRIMARY_METHOD, PRIMARY_THRESHOLD - 10.0),
        ("contrast_high_recall", CONTRAST_METHOD, CONTRAST_THRESHOLD),
    )
    sensitivity: dict[str, Any] = {}
    for label, method, threshold in arms:
        arm = predict(candidates, method, threshold)
        top1 = arm.loc[arm["predicted_rank"].eq(1)] if len(arm) else arm
        sensitivity[label] = {
            "method": method,
            "threshold": threshold,
            "accepted_links": int(len(top1)),
            "cohort_link_rate": round(len(top1) / anchors, 4) if anchors else None,
        }

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_schema": EVALUATION_SCHEMA,
        "selected_method": PRIMARY_METHOD,
        "selected_threshold": PRIMARY_THRESHOLD,
        "contrast_method": CONTRAST_METHOD,
        "contrast_threshold": CONTRAST_THRESHOLD,
        "weights": dict(DEFAULT_WEIGHTS),
        "gate_parameters": M_C_DEFAULTS,
        "reference": "regional_grand_ouest",
        "selection_rule": (
            "Frozen conservative baseline, not an optimal-threshold claim. 0.70 was fixed "
            "a priori on a precision-first principle - a false link fabricates both a "
            "survival event and its event time - and was not moved after the regional "
            "reference was consulted. 0.60 remains a required sensitivity arm."
        ),
    }
    (output_dir / "linkage_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        **config,
        "candidate_source": str(candidates_path),
        "cohort_application": {
            "anchors_with_candidates": anchors,
            "accepted_links": int(len(accepted)),
            "link_rate": round(len(accepted) / anchors, 4) if anchors else None,
        },
        "threshold_sensitivity": sensitivity,
        "validation_passed": bool(
            not accepted["anchor_episode_id"].duplicated().any()
            and not accepted["anchor_episode_id"].eq(accepted["candidate_episode_id"]).any()
        ),
    }
    (output_dir / "linkage_application_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary



def reference_label_frame(truth: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    """Pair-level labels for the anchors in one split.

    Every exposed pair of a reviewed anchor is a label: the reviewed successor
    is positive, everything else the reviewer had in front of them is negative.
    Feeding this to :func:`annotator_bias_report` answers the question the
    retired benchmark could not - whether the labels track the incumbent text
    score - with an answer that means something, because these labels were not
    produced from that score.
    """
    positives = {
        (record.anchor_episode_id, successor)
        for record in truth.itertuples(index=False)
        for successor in record.true_successors
    }
    labels = exposure.loc[
        exposure["anchor_episode_id"].isin(set(truth["anchor_episode_id"])),
        ["anchor_episode_id", "candidate_episode_id"],
    ].copy()
    labels["label"] = [
        "RENEWAL_OF_EXPIRING" if pair in positives else "UNRELATED"
        for pair in zip(labels["anchor_episode_id"], labels["candidate_episode_id"])
    ]
    return labels


def evaluate_benchmark(
    project_root: Path,
    output_dir: Path,
    split: str,
    event_set: str,
) -> dict[str, Any]:
    """Evaluate all frozen methods against one regional reference split."""
    benchmark_dir = output_dir / "regional_benchmark"
    exposure = pd.read_parquet(benchmark_dir / "exposure_full.parquet")
    exposure = score_with_fitted_model(exposure, output_dir / "fellegi_sunter_model.json")
    truth = load_truth(benchmark_dir, split, event_set)
    truth = truth.loc[truth["truth_usable"]].copy()
    logging.info(
        "benchmark: split=%s event_set=%s anchors=%s", split, event_set, len(truth)
    )

    # The benchmark's own universe: a prediction outside it is out of scope,
    # not wrong.
    pool_membership: dict[str, set[str]] = {}
    for anchor, group in exposure.groupby("anchor_episode_id"):
        pool_membership[anchor] = set(group["candidate_episode_id"])

    scored = exposure.copy()
    evaluable = scored.loc[scored["anchor_episode_id"].isin(set(truth["anchor_episode_id"]))]

    results: list[dict[str, Any]] = []
    for method in METHODS:
        if method == "M_D_fellegi_sunter" and "fs_match_probability" not in evaluable.columns:
            continue
        threshold = (
            METHOD_THRESHOLD_GRID["M_D_fellegi_sunter"][-2]
            if method == "M_D_fellegi_sunter" else PRIMARY_THRESHOLD
        )
        predictions = predict(evaluable, method, threshold)
        entry = {
            "method": method,
            "threshold": threshold,
            "unweighted": evaluate(predictions, truth),
        }
        # The reference is a stratified probability sample of the Grand Ouest
        # cohort, so it also supports a design-weighted estimate. It is reported
        # beside the unweighted figure, not instead of it: the stratum
        # populations behind the weights were computed on the earlier episode
        # reconstruction, which the unweighted sample quantity does not depend on.
        entry["design_weighted"] = weighted_evaluate(
            predictions, truth, pool_membership=pool_membership
        )
        entry["design_weighted"]["interpretation"] = (
            "Design weights span 1 to 68, so this answers a different question from the "
            "unweighted figure: it re-weights the reviewed sample back to the v1 Grand "
            "Ouest frame, in which the well-identified strata this method links inside "
            "were deliberately oversampled. Where a stratum contributes fewer than two "
            "accepted rows the linearised variance is zero and the interval collapses; "
            "a collapsed interval is missing information, not precision."
        )
        results.append(entry)

    manifest = load_manifest(benchmark_dir)
    bias = annotator_bias_report(reference_label_frame(truth, evaluable), evaluable)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": "regional_grand_ouest",
        "reference_version": manifest["reference_version"],
        "split": split,
        "event_set": event_set,
        "scoring_sources": {
            "exposure": str(benchmark_dir / "exposure_full.parquet"),
            "fellegi_sunter_model": str(output_dir / "fellegi_sunter_model.json"),
        },
        "anchors_evaluated": int(len(truth)),
        "positive_anchors": int(truth["has_successor"].sum()),
        "candidate_reachability": manifest["candidate_reachability"],
        "methods": results,
        "label_score_association": bias,
        "caveats": {
            "label_source": (
                "Labels come from a single-pass LLM-assisted evidence review of real "
                "BOAMP notices by the project owner, dated 2026-08-11. They are "
                "independent of every linkage method evaluated here, but they are not "
                "an independent human specialist panel and not legal renewal truth."
            ),
            "negative_definition": (
                "A negative means no successor among the roughly 25 candidates the "
                "reviewer was shown, not no successor in the pool. The false-positive "
                "rate on these anchors is therefore an upper bound."
            ),
            "sample_size": (
                "Fewer than 100 usable anchors. Read every interval; a difference "
                "between two methods that both fit inside one interval is not a result."
            ),
            "recall_ceiling": (
                "Recall is bounded by candidate generation: see "
                "candidate_reachability.candidate_generation_recall_ceiling."
            ),
        },
        "validation_passed": bool(results),
    }
    (output_dir / f"linkage_evaluation_{split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--evaluate-split", choices=["dev", "validation"],
        help="Evaluate one regional reference split. Omit to apply the primary "
             "method to the complete study cohort.",
    )
    parser.add_argument(
        "--event-set", choices=["primary"], default="primary",
        help="Which reference label classes count as an event. The regional "
             "reference records one relationship per anchor, so only the primary "
             "event set exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    configure_logging(project_root)
    if args.evaluate_split:
        summary = evaluate_benchmark(
            project_root, output_dir, args.evaluate_split, args.event_set,
        )
    else:
        summary = apply_primary_linkage(output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
