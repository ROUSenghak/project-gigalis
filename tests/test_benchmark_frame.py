"""The sampling design, checked as arithmetic rather than as intention.

v1 recorded inclusion probabilities and weights and then used them in no
metric, so nothing ever had to add up. The central test here is that the design
weights sum back to the frame size -- the identity that makes a weighted
estimate meaningful, and the one v1 was never in a position to fail.
"""

import numpy as np
import pandas as pd
import pytest

from boamp_pipeline.benchmark_frame import (
    FRAME_SEED,
    MAX_AWARD_YEAR,
    MIN_AWARD_YEAR,
    MIN_STRATUM_SAMPLE,
    add_design_variables,
    allocate,
    assert_no_frame_mixing,
    award_period,
    draw_probability_sample,
    eligible_frame,
    stable_hash,
    stratum_id,
    systematic_sample,
    weights_reconstruct_population,
)


def make_index(rows: int = 4000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = ["GRAND_OUEST", "ILE_DE_FRANCE", "NORD_EST", "SUD_EST", "SUD_OUEST", "CVL_DOM_OTHER"]
    segments = ["CPV-32", "CPV-35", "CPV-48", "CPV-72"]
    return pd.DataFrame({
        "episode_id": [f"EP-{i:08d}" for i in range(rows)],
        "has_award_notice": True,
        "award_year": rng.integers(MIN_AWARD_YEAR, MAX_AWARD_YEAR + 1, rows),
        "buyer_block_key": [f"K|siren:{i % 400:09d}" for i in range(rows)],
        "digital_flag": True,
        "dept_group": rng.choice(groups, rows),
        "digital_segment": rng.choice(segments, rows),
        "buyer_size_class": rng.choice(["SMALL", "MEDIUM", "LARGE", "MEGA"], rows),
        "duration_months_reliable": np.where(rng.random(rows) < 0.25, 48.0, np.nan),
    })


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def test_frame_requires_award_blockability_and_follow_up() -> None:
    index = make_index(200)
    index.loc[0, "has_award_notice"] = False          # a tender, not a contract
    index.loc[1, "buyer_block_key"] = ""              # no pool, so no meaningful negative
    index.loc[2, "award_year"] = MAX_AWARD_YEAR + 2   # cannot yet show a successor
    index.loc[3, "award_year"] = MIN_AWARD_YEAR - 1

    eligible = eligible_frame(index)

    assert set(eligible["episode_id"]) == set(index["episode_id"][4:])


def test_eligible_frame_rejects_an_index_missing_design_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        eligible_frame(pd.DataFrame({"episode_id": ["EP-1"]}))


def test_award_period_buckets_the_eforms_schema_break() -> None:
    assert award_period(2015) == "2015-2017"
    assert award_period(2019) == "2018-2019"
    assert award_period(2021) == "2020-2021"
    assert award_period(2024) == "out_of_range"
    assert award_period(None) == "unknown"


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def test_allocation_hits_the_budget_exactly_and_respects_the_floor() -> None:
    populations = {f"s{i}": n for i, n in enumerate([6000, 3000, 1500, 900, 400, 120])}

    allocation = allocate(populations, total=200)

    assert sum(allocation.values()) == 200
    assert min(allocation.values()) >= MIN_STRATUM_SAMPLE


def test_allocation_is_square_root_not_proportional() -> None:
    """Proportional allocation starves small strata; per-stratum estimates are
    wanted here, so a 50x population gap must not become a 50x sample gap."""
    allocation = allocate({"big": 10_000, "small": 200}, total=120, minimum=1, maximum=1000)

    ratio = allocation["big"] / allocation["small"]
    assert 5 < ratio < 9  # sqrt(50) is about 7.07


def test_allocation_never_exceeds_a_stratum_population() -> None:
    allocation = allocate({"tiny": 5, "big": 5000}, total=400, minimum=1, maximum=1000)

    assert allocation["tiny"] <= 5


def test_allocation_refuses_a_budget_below_the_floor() -> None:
    with pytest.raises(ValueError, match="below the per-stratum floor"):
        allocate({f"s{i}": 500 for i in range(24)}, total=50)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def test_systematic_sample_draws_exactly_the_requested_size() -> None:
    frame = add_design_variables(eligible_frame(make_index(1000)))

    for size in (1, 8, 37, 100):
        drawn = systematic_sample(
            frame, size, ["buyer_size_class", "duration_available", "award_period"], ("s",)
        )
        assert len(drawn) == size
        assert drawn["episode_id"].is_unique


def test_systematic_sample_is_reproducible_and_seed_dependent() -> None:
    frame = add_design_variables(eligible_frame(make_index(500)))
    columns = ["buyer_size_class", "duration_available", "award_period"]

    first = systematic_sample(frame, 40, columns, ("stratum-a",))
    again = systematic_sample(frame, 40, columns, ("stratum-a",))
    other = systematic_sample(frame, 40, columns, ("stratum-b",))

    assert list(first["episode_id"]) == list(again["episode_id"])
    assert list(first["episode_id"]) != list(other["episode_id"])


def test_weights_reconstruct_the_population() -> None:
    """The identity that makes a Horvitz-Thompson estimate valid."""
    frame = add_design_variables(eligible_frame(make_index(4000)))

    sample, design = draw_probability_sample(frame, total=400)

    assert len(sample) == 400
    assert weights_reconstruct_population(sample, len(frame))
    # And per stratum, pi is exactly n_h / N_h.
    for stratum, group in sample.groupby("stratum_id"):
        assert group["inclusion_probability"].nunique() == 1
        expected = design[stratum]["sample"] / design[stratum]["population"]
        assert group["inclusion_probability"].iloc[0] == pytest.approx(expected)


def test_implicit_stratification_balances_the_secondary_variables() -> None:
    """Sorting on buyer size before a systematic draw should track the frame's
    composition far more closely than an arbitrary slice would."""
    frame = add_design_variables(eligible_frame(make_index(4000)))

    sample, _ = draw_probability_sample(frame, total=400)

    frame_share = frame["buyer_size_class"].value_counts(normalize=True)
    sample_share = sample["buyer_size_class"].value_counts(normalize=True)
    for label, share in frame_share.items():
        assert abs(sample_share.get(label, 0.0) - share) < 0.05, label


def test_stable_hash_is_reproducible_across_processes() -> None:
    """Python's hash() is salted per run and cannot back a reproducible sample."""
    assert stable_hash("EP-1", seed=FRAME_SEED) == stable_hash("EP-1", seed=FRAME_SEED)
    assert stable_hash("EP-1", seed=1) != stable_hash("EP-1", seed=2)


def test_stratum_id_labels_non_digital_explicitly() -> None:
    assert stratum_id("GRAND_OUEST", "CPV-72") == "GRAND_OUEST|CPV-72"
    assert stratum_id("SUD_EST", "") == "SUD_EST|NON_DIGITAL"


# ---------------------------------------------------------------------------
# Frame separation
# ---------------------------------------------------------------------------


def test_probability_and_enrichment_frames_cannot_be_estimated_together() -> None:
    """Declaration-recruited anchors over-represent buyers who write detailed
    notices -- exactly where a text-similarity method looks best. They must
    never enter a national estimate."""
    mixed = pd.DataFrame({"frame": ["PROBABILITY", "ENRICHMENT_DECLARED"]})

    with pytest.raises(ValueError, match="refusing to mix frames"):
        assert_no_frame_mixing(mixed)


def test_single_frame_estimates_are_allowed() -> None:
    assert_no_frame_mixing(pd.DataFrame({"frame": ["PROBABILITY"] * 5}))
    assert_no_frame_mixing(pd.DataFrame({"frame": ["ENRICHMENT_DECLARED"] * 3}))


def test_frame_column_is_required_before_estimating() -> None:
    with pytest.raises(ValueError, match="no 'frame' column"):
        assert_no_frame_mixing(pd.DataFrame({"episode_id": ["EP-1"]}))
