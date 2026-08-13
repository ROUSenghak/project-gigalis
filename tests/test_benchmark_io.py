"""Benchmark I/O and weighted metric safeguards.

The v1 benchmark evidence must keep loading the same truth rows while the
current linkage summaries reflect intentional methodology fixes, such as the
buyer-identity correction that excludes conflicting validated SIRENs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boamp_pipeline.annotation_schema import EVENT_SETS
from boamp_pipeline.benchmark_io import TRUTH_COLUMNS, load_truth_v1
from boamp_pipeline.benchmark_metrics import (
    hard_negative_suite_metrics,
    stratified_ratio_estimate,
    weighted_evaluate,
)

PROCESSED = Path("data/processed/boamp_v2")
SUMMARY = PROCESSED / "linkage_evaluation_summary.json"


# ---------------------------------------------------------------------------
# v1/current summary checks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SUMMARY.exists(), reason="v1 evaluation summary not built")
def test_current_headline_numbers_match_corrected_buyer_identity_rule() -> None:
    """The primary summary reflects the current, defensible linkage rule."""
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert summary["selected_method"] == "M_B_text_ranking"
    assert summary["selected_threshold"] == 70.0
    assert summary["cohort_application"]["accepted_links"] == 544
    main = summary["threshold_sensitivity"]["main"]["locked_test_metrics"]
    assert main["precision_at_1"] == 0.875
    assert main["recall_at_1"] == 0.4118
    assert main["false_positive_rate_on_negatives"] == 0.0
    assert summary["benchmark"]["anchors_scored_in_cohort"] == 85


@pytest.mark.skipif(
    not (PROCESSED / "benchmark_remap" / "evaluation_subset_v2_remap.csv").exists(),
    reason="v1 benchmark remap not built",
)
def test_v1_loader_still_yields_22_positive_anchors() -> None:
    """The evidence base the whole study rests on, and the reason v3 exists."""
    truth = load_truth_v1(PROCESSED)

    assert set(TRUTH_COLUMNS) - {"benchmark_split"} <= set(truth.columns)
    evaluable = truth.loc[truth["truth_usable"]]
    assert int(evaluable["has_successor"].sum()) == 24  # before cohort restriction


# ---------------------------------------------------------------------------
# Weighted estimation
# ---------------------------------------------------------------------------


def sample_frame(n: int = 60, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strata = rng.choice(["A", "B", "C"], n)
    return pd.DataFrame({
        "anchor_v2_episode_id": [f"EP-{i}" for i in range(n)],
        "stratum_id": strata,
        "design_weight": np.where(strata == "A", 10.0, np.where(strata == "B", 20.0, 40.0)),
        "frame": "PROBABILITY",
        "correct": rng.integers(0, 2, n),
        "accepted": 1,
        "one": 1,
    })


def test_a_ratio_estimate_comes_with_an_interval() -> None:
    """A point estimate alone invites the over-reading v1 suffered."""
    result = stratified_ratio_estimate(sample_frame(), "correct", "one")

    assert 0.0 <= result["estimate"] <= 1.0
    assert result["standard_error"] > 0
    low, high = result["confidence_interval"]
    assert low <= result["estimate"] <= high


def test_weighted_estimation_refuses_purposive_anchors() -> None:
    """Enrichment anchors were recruited because their successor announced
    itself, so they over-represent buyers who write detailed notices."""
    frame = sample_frame(20)
    frame["design_weight"] = np.nan

    with pytest.raises(ValueError, match="requires a design weight"):
        stratified_ratio_estimate(frame, "correct", "one")


def test_weighted_estimation_refuses_to_mix_frames() -> None:
    frame = sample_frame(20)
    frame.loc[frame.index[:5], "frame"] = "ENRICHMENT_DECLARED"

    with pytest.raises(ValueError, match="refusing to mix frames"):
        stratified_ratio_estimate(frame, "correct", "one")


def test_weights_recover_the_population_total() -> None:
    frame = sample_frame(60)

    result = stratified_ratio_estimate(frame, "correct", "one")

    assert result["estimated_population_total"] == pytest.approx(
        float(frame["design_weight"].sum()), rel=1e-6
    )


# ---------------------------------------------------------------------------
# What a negative is allowed to count for
# ---------------------------------------------------------------------------


def truth_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "anchor_v2_episode_id": ["EP-1", "EP-2", "EP-3", "EP-4"],
        "true_successors": [["EP-S1"], [], [], []],
        "has_successor": [True, False, False, False],
        "stratum_id": ["A", "A", "A", "A"],
        "design_weight": [10.0, 10.0, 10.0, 10.0],
        "frame": ["PROBABILITY"] * 4,
        "negative_is_censored": [False, False, True, False],
        "negative_verification": [
            "EXHAUSTIVE_POOL", "EXHAUSTIVE_POOL", "EXHAUSTIVE_POOL", "AMONG_SHOWN",
        ],
    })


def predictions_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "anchor_episode_id": ["EP-1", "EP-2"],
        "candidate_episode_id": ["EP-S1", "EP-X"],
        "predicted_rank": [1, 1],
    })


def test_censored_anchors_are_excluded_from_the_false_positive_rate() -> None:
    """An anchor whose contract could still be running is not a confirmed
    negative; counting it would reward under-linking on recent contracts."""
    result = weighted_evaluate(predictions_frame(), truth_frame())

    assert result["censored_negatives_excluded"] == 1
    assert result["negative_anchors"] == 2  # EP-2 and EP-4, not the censored EP-3


def test_only_exhaustively_reviewed_negatives_can_be_false_positives() -> None:
    """v1's negatives all meant 'not in the 25 I was shown'."""
    result = weighted_evaluate(predictions_frame(), truth_frame())

    assert result["verified_negative_anchors"] == 1  # EP-2 only


def test_predictions_outside_the_pool_are_bucketed_not_scored() -> None:
    """A method whose blocking rule reaches outside the benchmark's universe is
    being judged on candidates the benchmark never considered."""
    result = weighted_evaluate(
        predictions_frame(), truth_frame(),
        pool_membership={"EP-1": {"EP-S1"}, "EP-2": set()},
    )

    assert result["out_of_universe_predictions"] == 1
    assert result["accepted_links"] == 1


# ---------------------------------------------------------------------------
# Hard negatives
# ---------------------------------------------------------------------------


def test_untested_hard_negatives_report_none_not_zero() -> None:
    """Reporting a zero error rate for pairs never at risk would flatter every
    method for free."""
    negatives = pd.DataFrame({
        "anchor_episode_id": ["EP-99"],
        "candidate_episode_id": ["EP-98"],
        "negative_source": ["SAME_PROCUREMENT"],
        "text_component": [1.0],
    })

    result = hard_negative_suite_metrics(predictions_frame(), negatives)

    assert result["SAME_PROCUREMENT"]["pairs_at_risk"] == 0
    assert result["SAME_PROCUREMENT"]["false_acceptance_rate"] is None
    assert "not a zero-error result" in result["SAME_PROCUREMENT"]["note"]


def test_a_hard_negative_actually_accepted_is_counted() -> None:
    negatives = pd.DataFrame({
        "anchor_episode_id": ["EP-1"],
        "candidate_episode_id": ["EP-S1"],
        "negative_source": ["PARALLEL_LOT"],
        "text_component": [0.9],
    })

    result = hard_negative_suite_metrics(predictions_frame(), negatives)

    assert result["PARALLEL_LOT"]["pairs_at_risk"] == 1
    assert result["PARALLEL_LOT"]["false_acceptance_rate"] == 1.0


def test_event_sets_are_selectable_and_nested() -> None:
    assert EVENT_SETS["strict"] < EVENT_SETS["broad"]
