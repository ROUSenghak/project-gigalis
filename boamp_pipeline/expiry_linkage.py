"""Expiry-aware successor selection built on the existing candidate pool.

The broad same-buyer/time blocking stage remains unchanged. This module only
adds anchor expiry evidence and applies a stricter rule to candidates appearing
more than one year before a reliable expected contract end.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


DAYS_PER_MONTH = 30.4375
NORMAL_TEXT_THRESHOLD = 0.70
EARLY_TEXT_THRESHOLD = 0.85
EARLY_BOUNDARY_DAYS = 365
METHOD_NAME = "M_E_expiry_aware_text"

EXPIRY_SOURCES = (
    "explicit_end",
    "start_plus_duration",
    "award_plus_duration",
    "unavailable",
)

TIMING_CLASSES = (
    "expiry_unknown",
    "early_more_than_1y",
    "pre_expiry_within_1y",
    "post_expiry_within_1y",
    "post_expiry_more_than_1y",
)


def _unique_dates(values: Iterable[Any]) -> list[pd.Timestamp]:
    parsed = pd.to_datetime(pd.Series(list(values), dtype="object"), errors="coerce").dropna()
    return sorted(pd.Timestamp(value) for value in parsed.unique())


def resolve_expected_end(
    award_date: Any,
    reliable_duration_months: float | None,
    start_dates: Iterable[Any] = (),
    end_dates: Iterable[Any] = (),
) -> tuple[pd.Timestamp | pd.NaT, str]:
    """Resolve expected end using explicit end, start+duration, then award+duration.

    Date evidence must be unambiguous at the episode level. Invalid explicit
    ends are ignored, and missing duration is never replaced by a default.
    """
    award = pd.to_datetime(award_date, errors="coerce")
    starts = _unique_dates(start_dates)
    ends = _unique_dates(end_dates)

    if pd.notna(award) and len(ends) == 1 and ends[0] > award:
        return ends[0], "explicit_end"

    duration = pd.to_numeric(reliable_duration_months, errors="coerce")
    if pd.notna(duration) and float(duration) > 0:
        offset = pd.to_timedelta(float(duration) * DAYS_PER_MONTH, unit="D")
        if len(starts) == 1:
            calculated = starts[0] + offset
            if pd.isna(award) or calculated > award:
                return calculated, "start_plus_duration"
        if pd.notna(award):
            return award + offset, "award_plus_duration"

    return pd.NaT, "unavailable"


def build_expiry_features(
    cohort: pd.DataFrame,
    membership: pd.DataFrame,
    notices: pd.DataFrame,
) -> pd.DataFrame:
    """Build one expiry-evidence row per cohort episode."""
    required_cohort = {"episode_id", "award_date", "duration_months_reliable"}
    required_membership = {"idweb", "episode_id"}
    required_notices = {"idweb", "contract_start_date", "contract_end_date"}
    for label, frame, required in (
        ("cohort", cohort, required_cohort),
        ("membership", membership, required_membership),
        ("notices", notices, required_notices),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing required columns: {sorted(missing)}")

    cohort_ids = set(cohort["episode_id"])
    evidence = membership.loc[membership["episode_id"].isin(cohort_ids)].merge(
        notices[list(required_notices)], on="idweb", how="left", validate="one_to_one"
    )
    by_episode = {
        episode_id: (
            _unique_dates(group["contract_start_date"]),
            _unique_dates(group["contract_end_date"]),
        )
        for episode_id, group in evidence.groupby("episode_id", sort=False)
    }

    rows: list[dict[str, Any]] = []
    for anchor in cohort.itertuples(index=False):
        starts, ends = by_episode.get(anchor.episode_id, ([], []))
        expected_end, source = resolve_expected_end(
            anchor.award_date,
            anchor.duration_months_reliable,
            start_dates=starts,
            end_dates=ends,
        )
        rows.append(
            {
                "episode_id": anchor.episode_id,
                "anchor_expected_end_date": expected_end,
                "expected_end_source": source,
                "explicit_start_date": starts[0] if len(starts) == 1 else pd.NaT,
                "explicit_end_date": ends[0] if len(ends) == 1 else pd.NaT,
                "explicit_start_date_count": len(starts),
                "explicit_end_date_count": len(ends),
            }
        )
    return pd.DataFrame(rows)


def timing_class(days_from_expected_end: float | int | None) -> str:
    """Classify a candidate publication relative to expected expiry."""
    if days_from_expected_end is None or pd.isna(days_from_expected_end):
        return "expiry_unknown"
    days = float(days_from_expected_end)
    if days < -EARLY_BOUNDARY_DAYS:
        return "early_more_than_1y"
    if days < 0:
        return "pre_expiry_within_1y"
    if days <= EARLY_BOUNDARY_DAYS:
        return "post_expiry_within_1y"
    return "post_expiry_more_than_1y"


def add_expiry_timing(candidates: pd.DataFrame, expiry_features: pd.DataFrame) -> pd.DataFrame:
    """Attach anchor expiry evidence and timing class to every candidate."""
    feature_columns = [
        "episode_id",
        "anchor_expected_end_date",
        "expected_end_source",
        "explicit_start_date",
        "explicit_end_date",
        "explicit_start_date_count",
        "explicit_end_date_count",
    ]
    enriched = candidates.merge(
        expiry_features[feature_columns],
        left_on="anchor_episode_id",
        right_on="episode_id",
        how="left",
        validate="many_to_one",
    ).drop(columns="episode_id")
    enriched["candidate_origin_date"] = pd.to_datetime(enriched["candidate_origin_date"])
    enriched["anchor_expected_end_date"] = pd.to_datetime(enriched["anchor_expected_end_date"])
    enriched["days_from_expected_end"] = (
        enriched["candidate_origin_date"] - enriched["anchor_expected_end_date"]
    ).dt.days
    enriched["timing_class"] = enriched["days_from_expected_end"].map(timing_class)
    return enriched


def expiry_aware_eligible(frame: pd.DataFrame) -> pd.Series:
    """Eligibility mask for the fixed expiry-aware operating point."""
    text = frame["text_component"].fillna(0)
    regular_text = text.ge(NORMAL_TEXT_THRESHOLD)
    unusually_early = frame["timing_class"].eq("early_more_than_1y")
    strong_early_evidence = text.ge(EARLY_TEXT_THRESHOLD) & frame["cpv_component"].fillna(0).gt(0)
    return regular_text & (~unusually_early | strong_early_evidence)


def select_expiry_aware(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return all eligible candidates with deterministic per-anchor ranks."""
    accepted = candidates.loc[expiry_aware_eligible(candidates)].copy()
    if accepted.empty:
        accepted["predicted_rank"] = pd.Series(dtype=int)
        return accepted
    accepted = accepted.sort_values(
        [
            "anchor_episode_id",
            "text_component",
            "linkage_score",
            "candidate_origin_date",
            "candidate_episode_id",
        ],
        ascending=[True, False, False, True, True],
        na_position="last",
        kind="mergesort",
    )
    accepted["predicted_rank"] = accepted.groupby("anchor_episode_id").cumcount() + 1
    return accepted
