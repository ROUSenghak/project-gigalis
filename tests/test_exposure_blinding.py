"""Exposure neutrality: what reaches the annotator, and what provably does not.

The v1 benchmark showed reviewers the top 25 of a single ranker. Every test
here defends against some version of that: a ranking leaking through a column,
through presentation order, or through a slot identifier that survives between
annotation passes.

The blinding tests are deliberately written so that they fail when someone adds
a new score column upstream without deciding it is safe to show.
"""

import json

import numpy as np
import pytest

from boamp_pipeline.exposure import (
    ALLOWED_DOSSIER_KEYS,
    EXHAUSTIVE_MAX,
    GAP_BUCKETS,
    MAX_PER_RETRIEVER,
    NEAR_EXHAUSTIVE_MAX,
    RETRIEVERS,
    TAIL_FLOOR,
    assert_no_forbidden_keys,
    assign_slots,
    blind_view,
    exposure_mode,
    gap_bucket,
    interleave,
    pool_size_bucket,
    presentation_order_correlation,
    retriever_coverage,
    stratified_tail_sample,
    two_stage_positive_estimate,
)


def make_pool(n: int, first_gap: int = 100, step: int = 30) -> list[dict]:
    return [
        {"candidate_episode_id": f"EP-{i:04d}", "gap_days": first_gap + i * step}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# The allow-list
# ---------------------------------------------------------------------------


def test_blind_view_withholds_anything_not_explicitly_allowed() -> None:
    record = {
        "slot_id": "C07",
        "objet": "Maintenance des installations thermiques",
        "award_date": "2019-04-28",
        # None of the following may ever be shown.
        "linkage_score": 88.4,
        "text_component": 0.91,
        "fs_match_probability": 0.62,
        "retrievers_json": '["R_TFIDF_WORD"]',
        "is_random_tail": False,
        "gap_days": 1461,
    }

    shown = blind_view(record)

    assert set(shown) == {"slot_id", "objet", "award_date"}
    assert "linkage_score" not in shown


def test_a_newly_added_score_column_does_not_leak_by_default() -> None:
    """The regression that an allow-list exists to prevent: someone adds a
    column upstream and it silently appears in dossiers."""
    record = {"slot_id": "C01", "objet": "x", "brand_new_model_output": 0.99}

    assert "brand_new_model_output" not in blind_view(record)


def test_forbidden_key_detection_is_substring_based() -> None:
    for key in [
        "linkage_score", "text_component", "word_tfidf_similarity",
        "fs_match_probability", "retrievers_json", "candidate_rank",
        "tail_inclusion_probability", "gap_days", "predicted_label",
    ]:
        with pytest.raises(ValueError, match="forbidden key"):
            assert_no_forbidden_keys({key: 1})


def test_allowed_keys_are_all_descriptive_not_derived() -> None:
    """A guard on the allow-list itself: nothing score-like may be added to it."""
    for key in ALLOWED_DOSSIER_KEYS:
        assert_no_forbidden_keys({key: None})


# ---------------------------------------------------------------------------
# Presentation order
# ---------------------------------------------------------------------------


def test_slots_are_assigned_after_shuffling_so_order_encodes_nothing() -> None:
    ordered_by_score = [f"EP-{i:04d}" for i in range(40)]

    slots = assign_slots(ordered_by_score, "EP-ANCHOR", "A")
    positions = [int(slot.lstrip("C")) for slot, _ in slots]
    scores = [40 - ordered_by_score.index(candidate) for _, candidate in slots]

    assert positions == sorted(positions)
    assert abs(presentation_order_correlation(positions, scores)) < 0.45


def test_the_two_passes_use_different_orders_and_different_slot_ids() -> None:
    """Pass A slot identifiers must be worthless in pass B, so that a leaked
    dossier cannot be aligned across passes."""
    candidates = [f"EP-{i:04d}" for i in range(30)]

    a = dict((candidate, slot) for slot, candidate in assign_slots(candidates, "EP-X", "A"))
    b = dict((candidate, slot) for slot, candidate in assign_slots(candidates, "EP-X", "B"))

    assert a != b
    assert sum(a[c] == b[c] for c in candidates) < len(candidates) // 2


def test_slot_assignment_is_reproducible() -> None:
    candidates = [f"EP-{i:04d}" for i in range(20)]

    assert assign_slots(candidates, "EP-X", "A") == assign_slots(candidates, "EP-X", "A")
    assert assign_slots(candidates, "EP-X", "A") != assign_slots(candidates, "EP-Y", "A")


def test_order_correlation_returns_zero_when_undefined() -> None:
    assert presentation_order_correlation([1, 2], [0.5, 0.5]) == 0.0
    assert presentation_order_correlation([], []) == 0.0


# ---------------------------------------------------------------------------
# Exposure sizing: what a negative is allowed to mean
# ---------------------------------------------------------------------------


def test_exposure_mode_decides_whether_a_negative_is_verified() -> None:
    assert exposure_mode(1) == "EXHAUSTIVE"
    assert exposure_mode(EXHAUSTIVE_MAX) == "EXHAUSTIVE"
    assert exposure_mode(EXHAUSTIVE_MAX + 1) == "NEAR_EXHAUSTIVE"
    assert exposure_mode(NEAR_EXHAUSTIVE_MAX) == "NEAR_EXHAUSTIVE"
    assert exposure_mode(NEAR_EXHAUSTIVE_MAX + 1) == "SAMPLED"


def test_pool_size_is_disclosed_only_as_a_bucket() -> None:
    """The annotator must know how completely the pool was shown, but an exact
    size would reveal how selective the exposure was per candidate."""
    assert pool_size_bucket(12) == "1-30"
    assert pool_size_bucket(200) == "121-400"
    assert pool_size_bucket(5000) == "400+"


# ---------------------------------------------------------------------------
# The random tail
# ---------------------------------------------------------------------------


def test_tail_sampling_fills_its_slots_when_material_exists() -> None:
    """A plain even split under-fills whenever a stratum is smaller than its
    share, which would silently weaken the only unbiased view of the pool."""
    pool = make_pool(400, first_gap=100, step=7)
    rng = np.random.default_rng(0)

    drawn, probabilities = stratified_tail_sample(pool, set(), TAIL_FLOOR, rng)

    assert len(drawn) == TAIL_FLOOR
    assert all(0 < p <= 1 for p in probabilities.values())
    assert all(candidate["tail_stratum"] for candidate in drawn)


def test_tail_sampling_spans_several_gap_strata() -> None:
    """Successors concentrate at three to four years, so a uniform tail would
    spend most draws where they are rare."""
    pool = make_pool(300, first_gap=100, step=9)
    rng = np.random.default_rng(1)

    drawn, probabilities = stratified_tail_sample(pool, set(), 20, rng)

    assert len(probabilities) >= 3
    assert {c["tail_stratum"] for c in drawn} == set(probabilities)


def test_tail_never_redraws_an_already_exposed_candidate() -> None:
    pool = make_pool(100)
    already = {"EP-0000", "EP-0001", "EP-0002"}
    rng = np.random.default_rng(2)

    drawn, _ = stratified_tail_sample(pool, already, 20, rng)

    assert not ({c["candidate_episode_id"] for c in drawn} & already)


def test_tail_takes_everything_when_the_remainder_is_small() -> None:
    pool = make_pool(5)
    rng = np.random.default_rng(3)

    drawn, probabilities = stratified_tail_sample(pool, set(), 20, rng)

    assert len(drawn) == 5
    assert all(p == 1.0 for p in probabilities.values())


def test_the_last_gap_bucket_is_closed_so_no_candidate_is_unsamplable() -> None:
    """A candidate on the window's final day still needs a stratum, or it would
    have no sampling probability and could never enter the estimate."""
    assert gap_bucket(2920) == GAP_BUCKETS[-1][2]
    assert gap_bucket(2921) == ""
    assert gap_bucket(89) == ""
    assert gap_bucket(90) == GAP_BUCKETS[0][2]


# ---------------------------------------------------------------------------
# Union construction
# ---------------------------------------------------------------------------


def test_interleave_prevents_any_retriever_from_dominating() -> None:
    ranked = {
        "R_TFIDF_WORD": [f"W{i}" for i in range(20)],
        "R_CPV_TIME": [f"C{i}" for i in range(20)],
        "R_INCUMBENT_SUPPLIER": [f"I{i}" for i in range(20)],
    }

    taken, provenance = interleave(ranked, limit=18, max_per_retriever=MAX_PER_RETRIEVER)

    assert len(taken) == 18
    for name in ranked:
        contributed = sum(1 for c in taken if name in provenance[c])
        assert contributed <= MAX_PER_RETRIEVER


def test_interleave_records_every_retriever_that_found_a_candidate() -> None:
    ranked = {"R_TFIDF_WORD": ["EP-1", "EP-2"], "R_CPV_TIME": ["EP-1", "EP-3"]}

    taken, provenance = interleave(ranked, limit=10)

    assert set(taken) == {"EP-1", "EP-2", "EP-3"}
    assert set(provenance["EP-1"]) == {"R_TFIDF_WORD", "R_CPV_TIME"}


def test_at_least_three_retrievers_are_text_free() -> None:
    """If every retriever ranked by text, the exposure would only ever surface
    text-similar candidates and a text method would score perfectly by
    construction."""
    text_free = [r.name for r in RETRIEVERS if r.text_free]

    assert len(text_free) >= 3
    assert "R_INCUMBENT_SUPPLIER" in text_free
    assert "R_CPV_TIME" in text_free


# ---------------------------------------------------------------------------
# Two-stage estimation
# ---------------------------------------------------------------------------


def test_tail_positives_are_scaled_by_their_sampling_probability() -> None:
    """The honest replacement for assuming a truncated list held everything."""
    estimate = two_stage_positive_estimate(
        union_positives=1,
        tail_positives_by_stratum={"3-4y": 1},
        tail_probabilities={"3-4y": 0.25},
    )

    assert estimate["union_positives"] == 1
    assert estimate["estimated_tail_positives"] == pytest.approx(4.0)
    assert estimate["estimated_total_positives"] == pytest.approx(5.0)
    assert estimate["tail_standard_error"] > 0


def test_a_census_of_the_tail_contributes_no_extra_uncertainty() -> None:
    estimate = two_stage_positive_estimate(2, {"0-2y": 3}, {"0-2y": 1.0})

    assert estimate["estimated_total_positives"] == pytest.approx(5.0)
    assert estimate["tail_standard_error"] == pytest.approx(0.0)


def test_retriever_coverage_reports_who_actually_found_the_positives() -> None:
    provenance = {
        "EP-1": ["R_TFIDF_WORD", "R_CPV_TIME"],
        "EP-2": ["R_RANDOM_TAIL"],
        "EP-3": ["R_TFIDF_WORD"],
    }

    coverage = retriever_coverage(provenance, ["EP-1", "EP-2"])

    assert coverage["R_TFIDF_WORD"]["positives_surfaced"] == 1
    assert coverage["R_RANDOM_TAIL"]["positives_surfaced"] == 1
    assert coverage["R_CPV_TIME"]["share_of_positives"] == 0.5
