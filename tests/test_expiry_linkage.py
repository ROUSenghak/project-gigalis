import math

import pandas as pd

from boamp_pipeline.expiry_linkage import (
    EARLY_TEXT_THRESHOLD,
    NORMAL_TEXT_THRESHOLD,
    expiry_aware_eligible,
    resolve_expected_end,
    select_expiry_aware,
    timing_class,
)


def test_expected_end_precedence_and_source_labels() -> None:
    award = pd.Timestamp("2020-01-01")

    explicit, source = resolve_expected_end(
        award,
        48.0,
        start_dates=["2020-02-01"],
        end_dates=["2023-12-31"],
    )
    assert explicit == pd.Timestamp("2023-12-31")
    assert source == "explicit_end"

    from_start, source = resolve_expected_end(award, 12.0, start_dates=["2020-02-01"])
    assert from_start > pd.Timestamp("2021-01-31")
    assert source == "start_plus_duration"

    from_award, source = resolve_expected_end(award, 12.0)
    assert from_award > pd.Timestamp("2020-12-31")
    assert source == "award_plus_duration"


def test_conflicting_dates_are_not_used_and_missing_duration_is_not_imputed() -> None:
    award = pd.Timestamp("2020-01-01")
    fallback, source = resolve_expected_end(
        award,
        12.0,
        end_dates=["2020-12-01", "2021-01-01"],
    )
    assert fallback > award
    assert source == "award_plus_duration"

    unavailable, source = resolve_expected_end(
        award,
        None,
        start_dates=["2020-02-01"],
        end_dates=["2019-12-31"],
    )
    assert pd.isna(unavailable)
    assert source == "unavailable"


def test_timing_boundaries_are_exact() -> None:
    assert timing_class(None) == "expiry_unknown"
    assert timing_class(float("nan")) == "expiry_unknown"
    assert timing_class(-366) == "early_more_than_1y"
    assert timing_class(-365) == "pre_expiry_within_1y"
    assert timing_class(-1) == "pre_expiry_within_1y"
    assert timing_class(0) == "post_expiry_within_1y"
    assert timing_class(365) == "post_expiry_within_1y"
    assert timing_class(366) == "post_expiry_more_than_1y"


def test_very_early_candidate_requires_strong_text_and_positive_cpv() -> None:
    frame = pd.DataFrame(
        {
            "timing_class": [
                "early_more_than_1y",
                "early_more_than_1y",
                "early_more_than_1y",
                "pre_expiry_within_1y",
                "expiry_unknown",
            ],
            "text_component": [
                EARLY_TEXT_THRESHOLD,
                EARLY_TEXT_THRESHOLD - 0.01,
                EARLY_TEXT_THRESHOLD,
                NORMAL_TEXT_THRESHOLD,
                NORMAL_TEXT_THRESHOLD,
            ],
            "cpv_component": [0.5, 0.5, float("nan"), 0.0, float("nan")],
        }
    )

    assert expiry_aware_eligible(frame).tolist() == [True, False, False, True, True]


def test_selection_is_deterministic_and_returns_one_top_candidate() -> None:
    candidates = pd.DataFrame(
        {
            "anchor_episode_id": ["a", "a", "a", "b"],
            "candidate_episode_id": ["c3", "c2", "c1", "c4"],
            "candidate_origin_date": pd.to_datetime(
                ["2024-01-03", "2024-01-02", "2024-01-02", "2024-02-01"]
            ),
            "text_component": [0.80, 0.80, 0.80, 0.69],
            "linkage_score": [80.0, 81.0, 81.0, 99.0],
            "cpv_component": [0.0, 0.0, 0.0, 1.0],
            "timing_class": ["expiry_unknown"] * 4,
        }
    )

    selected = select_expiry_aware(candidates)
    top = selected.loc[selected["predicted_rank"] == 1]

    assert len(top) == 1
    assert top.iloc[0]["anchor_episode_id"] == "a"
    assert top.iloc[0]["candidate_episode_id"] == "c1"
    assert selected.loc[selected["anchor_episode_id"] == "a", "predicted_rank"].tolist() == [1, 2, 3]
    assert math.isclose(float(top.iloc[0]["text_component"]), 0.80)
