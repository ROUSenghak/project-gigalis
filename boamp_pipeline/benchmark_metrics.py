"""Design-weighted estimation for a stratified linkage reference.

The Grand Ouest reference drew its 120 anchors under a stratified design and
recorded an inclusion probability and a sampling weight for every one of them,
but computed no metric from either: its own precision and recall are unweighted
sample quantities presented as though they described a population. Its strata
vary 68-fold in inclusion probability, so the unweighted and weighted answers
were never going to agree, and both are now reported side by side.

This module supplies the layer that was missing. Precision, recall, the
false-positive rate and coverage are all ratios of two survey totals, so each
is estimated as a ratio estimator with a linearised stratified variance, and
each comes back with an interval rather than a bare point.

Three rules are enforced rather than documented:

* a probability estimate refuses to run on purposively recruited anchors;
* a false-positive rate refuses to count anchors whose contract could still be
  running at the cutoff, because those are censored, not negative;
* a prediction outside the benchmark's own candidate pool is bucketed as
  out-of-universe rather than scored as a false positive, since the benchmark
  cannot speak about candidates it never considered.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def assert_no_frame_mixing(sample: pd.DataFrame) -> None:
    """Refuse to estimate from probability and purposive rows together.

    Purposively recruited anchors carry no inclusion probability, so averaging
    them into a design-weighted estimate silently invents one. The regional
    reference is a pure probability sample and this never fires on it; the guard
    stays so that adding an enrichment arm later cannot quietly corrupt an
    estimate.
    """
    if "frame" not in sample.columns:
        raise ValueError("sample has no 'frame' column; cannot verify frame separation")
    frames = set(sample["frame"].dropna().unique())
    probability = {f for f in frames if f == "PROBABILITY"}
    enrichment = {f for f in frames if str(f).startswith("ENRICHMENT")}
    if probability and enrichment:
        raise ValueError(
            "refusing to mix frames in one estimate: "
            f"{sorted(probability)} with {sorted(enrichment)}. "
            "Purposively recruited anchors carry no inclusion probability; "
            "estimate on PROBABILITY rows only."
        )


def stratified_ratio_estimate(
    frame: pd.DataFrame,
    numerator: str,
    denominator: str,
    *,
    stratum: str = "stratum_id",
    weight: str = "design_weight",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Estimate a ratio of survey totals, with a linearised variance.

    Precision, recall, coverage and the false-positive rate are all of the form
    ``sum(y) / sum(x)`` over a stratified sample, so they share one estimator::

        R = sum_h sum_i w_i y_i / sum_h sum_i w_i x_i

    with the standard linearised variance on the residual ``e = y - R x``. The
    interval is what makes these numbers quotable: on a sample this size a point
    estimate alone invites exactly the over-reading the reference's own
    unweighted figures invite.
    """
    if frame.empty:
        return {"estimate": None, "standard_error": None, "n": 0}
    assert_no_frame_mixing(frame)

    weights = pd.to_numeric(frame[weight], errors="coerce")
    if weights.isna().any():
        raise ValueError(
            "weighted estimation requires a design weight on every row; purposive "
            "anchors have none and must not be estimated from"
        )
    y = pd.to_numeric(frame[numerator], errors="coerce").fillna(0.0)
    x = pd.to_numeric(frame[denominator], errors="coerce").fillna(0.0)

    total_y = float((weights * y).sum())
    total_x = float((weights * x).sum())
    if total_x <= 0:
        return {"estimate": None, "standard_error": None, "n": int(len(frame)),
                "note": "denominator total is zero"}
    ratio = total_y / total_x

    variance = 0.0
    for _, group in frame.groupby(stratum, sort=False):
        n_h = len(group)
        if n_h < 2:
            continue
        w_h = pd.to_numeric(group[weight], errors="coerce")
        e_h = (
            pd.to_numeric(group[numerator], errors="coerce").fillna(0.0)
            - ratio * pd.to_numeric(group[denominator], errors="coerce").fillna(0.0)
        )
        # N_h is recovered from the design weights, which is exactly what makes
        # the finite-population correction available.
        N_h = float(w_h.sum())
        f_h = n_h / N_h if N_h > 0 else 0.0
        s2_h = float(((e_h - e_h.mean()) ** 2).sum() / (n_h - 1))
        variance += (N_h ** 2) * (1 - f_h) * s2_h / n_h

    standard_error = math.sqrt(variance) / total_x if variance > 0 else 0.0
    z = 1.959963985 if confidence == 0.95 else 1.644853627
    return {
        "estimate": round(ratio, 4),
        "standard_error": round(standard_error, 4),
        "confidence_interval": [
            round(max(0.0, ratio - z * standard_error), 4),
            round(min(1.0, ratio + z * standard_error), 4),
        ],
        "n": int(len(frame)),
        "estimated_population_total": round(total_x, 1),
    }


def weighted_evaluate(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    pool_membership: Mapping[str, set[str]] | None = None,
    exclude_censored: bool = True,
) -> dict[str, Any]:
    """Design-weighted precision, recall, false-positive rate and coverage.

    ``predictions`` is the top-1 accepted link per anchor, as
    ``evaluate_linkage.predict`` produces it.
    """
    top1 = (
        predictions.loc[predictions["predicted_rank"] == 1]
        .set_index("anchor_episode_id")["candidate_episode_id"].to_dict()
        if len(predictions) else {}
    )

    rows: list[dict[str, Any]] = []
    out_of_universe = 0
    for record in truth.itertuples(index=False):
        anchor = record.anchor_episode_id
        prediction = top1.get(anchor)
        truths = set(record.true_successors)
        in_pool = True
        if prediction is not None and pool_membership is not None:
            in_pool = prediction in pool_membership.get(anchor, set())
            if not in_pool:
                out_of_universe += 1
        accepted = int(prediction is not None and in_pool)
        correct = int(accepted and prediction in truths)
        rows.append({
            "anchor_episode_id": anchor,
            "stratum_id": getattr(record, "stratum_id", "ALL"),
            "design_weight": getattr(record, "design_weight", np.nan),
            "frame": getattr(record, "frame", "PROBABILITY"),
            "has_successor": int(bool(record.has_successor)),
            "accepted": accepted,
            "correct": correct,
            "one": 1,
            "negative_is_censored": bool(getattr(record, "negative_is_censored", False)),
            "negative_verification": getattr(record, "negative_verification", "AMONG_SHOWN"),
        })
    scored = pd.DataFrame(rows)
    if scored.empty:
        return {"anchors": 0}

    positives = scored.loc[scored["has_successor"] == 1]
    accepted_rows = scored.loc[scored["accepted"] == 1]
    negatives = scored.loc[scored["has_successor"] == 0]
    if exclude_censored:
        negatives = negatives.loc[~negatives["negative_is_censored"]]
    # Only an anchor whose whole pool was reviewed can contribute a false
    # positive: elsewhere "no successor" means "none among those shown".
    verified_negatives = negatives.loc[negatives["negative_verification"] == "EXHAUSTIVE_POOL"]

    return {
        "anchors": int(len(scored)),
        "positive_anchors": int(len(positives)),
        "negative_anchors": int(len(negatives)),
        "verified_negative_anchors": int(len(verified_negatives)),
        "accepted_links": int(len(accepted_rows)),
        "out_of_universe_predictions": out_of_universe,
        "precision_at_1": stratified_ratio_estimate(accepted_rows, "correct", "one"),
        "recall_at_1": stratified_ratio_estimate(positives, "correct", "one"),
        "false_positive_rate_on_verified_negatives": stratified_ratio_estimate(
            verified_negatives, "accepted", "one"
        ),
        "coverage": stratified_ratio_estimate(scored, "accepted", "one"),
        "censored_negatives_excluded": int(
            scored.loc[scored["has_successor"] == 0, "negative_is_censored"].sum()
        ),
        "notes": {
            "out_of_universe": (
                "Predictions outside the benchmark's candidate pool are counted here "
                "and excluded from precision, because the benchmark never considered "
                "them and cannot call them wrong."
            ),
            "censoring": (
                "Anchors whose contract could still be running at the study cutoff are "
                "excluded from the false-positive rate; counting them as negatives "
                "would reward under-linking on recent contracts."
            ),
        },
    }


def hard_negative_suite_metrics(
    predictions: pd.DataFrame, negatives: pd.DataFrame
) -> dict[str, Any]:
    """How often a method accepts a pair that is provably not a renewal.

    Unweighted by design: this suite has no inclusion probability.

    Retired with the France-level benchmark, which supplied the rule-generated
    negative suite this scores. It existed to give that benchmark one stress
    test its own label generator had not written; the regional reference does
    not need the workaround, because its labels were never generated from the
    methods' evidence in the first place.
    """
    accepted_pairs = set(
        zip(predictions["anchor_episode_id"], predictions["candidate_episode_id"])
    ) if len(predictions) else set()
    # Only pairs whose anchor the method actually ran on were ever at risk of
    # being accepted. Scoring the rest would report a flattering zero for pairs
    # that were never tested.
    anchors_at_risk = set(predictions["anchor_episode_id"]) if len(predictions) else set()

    results: dict[str, Any] = {}
    for source, group in negatives.groupby("negative_source"):
        at_risk = group.loc[group["anchor_episode_id"].isin(anchors_at_risk)]
        pairs = set(zip(at_risk["anchor_episode_id"], at_risk["candidate_episode_id"]))
        accepted = len(pairs & accepted_pairs)
        similarity = pd.to_numeric(group["text_component"], errors="coerce")
        results[str(source)] = {
            "pairs_in_suite": int(len(group)),
            "pairs_at_risk": len(pairs),
            "accepted": accepted,
            "false_acceptance_rate": (
                round(accepted / len(pairs), 4) if pairs else None
            ),
            "median_text_similarity": (
                round(float(similarity.median()), 4) if similarity.notna().any() else None
            ),
            "note": (
                None if pairs else
                "no pair in this suite shares an anchor with the evaluated split, so "
                "the method was never at risk of accepting one; not a zero-error result"
            ),
        }
    return results


def annotator_bias_report(
    labels: pd.DataFrame, exposure: pd.DataFrame
) -> dict[str, Any]:
    """Correlation between the annotator's labels and the incumbent text score.

    Published as a number rather than argued away. Deterministic bootstrap rules
    that inspect the same text used by a linkage method can favor that method;
    the size of this association helps a reader discount the benchmark.
    """
    merged = labels.merge(
        exposure[["anchor_episode_id", "candidate_episode_id", "text_component",
                  "linkage_score"]],
        on=["anchor_episode_id", "candidate_episode_id"],
        how="inner",
    )
    if merged.empty:
        return {"pairs": 0}
    positive = merged["label"].eq("RENEWAL_OF_EXPIRING").astype(float)
    text = pd.to_numeric(merged["text_component"], errors="coerce")
    score = pd.to_numeric(merged["linkage_score"], errors="coerce")
    usable = text.notna() & score.notna()
    return {
        "pairs": int(usable.sum()),
        "point_biserial_label_vs_text_component": (
            round(float(positive[usable].corr(text[usable])), 4)
            if usable.sum() > 2 and positive[usable].nunique() > 1 else None
        ),
        "point_biserial_label_vs_linkage_score": (
            round(float(positive[usable].corr(score[usable])), 4)
            if usable.sum() > 2 and positive[usable].nunique() > 1 else None
        ),
        "median_text_component_positive": (
            round(float(text[usable & positive.astype(bool)].median()), 4)
            if (usable & positive.astype(bool)).any() else None
        ),
        "median_text_component_negative": (
            round(float(text[usable & ~positive.astype(bool)].median()), 4)
            if (usable & ~positive.astype(bool)).any() else None
        ),
        "interpretation": (
            "Association between the reference labels and the incumbent text score. "
            "Under the retired rule-generated benchmark a high value was a warning, "
            "because the labels were computed from that evidence. Under the regional "
            "reference, whose labels were established by reading notices, a positive "
            "association is expected and is evidence that the score tracks something "
            "real; it is the size of the gap between the positive and negative medians "
            "that says how separable the two classes are."
        ),
    }
