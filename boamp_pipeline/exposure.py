"""Candidate exposure: what the annotator is allowed to see, and why.

The v1 benchmark's central defect was here. Reviewers saw the top 25 candidates
produced by one ranker, drawn from pools with a median size of 146, and 77 of
94 eligible anchors were truncated. A successor that ranker buried at rank 40
was never on screen, and its anchor was recorded as having none. Measured
recall was therefore optimistic by construction, every negative meant "not
found in that list", and the ranker itself is not even preserved in the
repository.

Two mechanisms replace it.

**A union of diverse retrievers.** Seven retrievers each contribute their own
top-k, and the union is what gets shown. No single ranking decides visibility,
so each retriever's recall becomes measurable against the others rather than
assumed. Three of the seven consult no text at all, which matters because the
incumbent method is a text ranker.

**A random tail.** Whatever the retrievers agree to ignore is still sampled,
uniformly within gap strata, at a known probability. That turns "what did every
retriever miss?" from an unanswerable question into an estimate with an
interval -- see :func:`two_stage_positive_estimate`.

Blinding is structural rather than promised. :func:`blind_view` filters through
an explicit allow-list, so a score column added to the exposure table later
cannot leak into a dossier by default. Candidates are shuffled per pass and
given opaque slot identifiers *after* shuffling, so presentation order carries
no ranking and pass A's identifiers are meaningless in pass B.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

#: Retrievers, with the evidence each one uses. The `text_free` flag is what
#: keeps the union from being a restatement of the incumbent method: if every
#: retriever ranked by text, exposure would only ever surface text-similar
#: candidates and any text method would score perfectly by construction.
@dataclass(frozen=True)
class Retriever:
    name: str
    top_k: int
    text_free: bool


RETRIEVERS: tuple[Retriever, ...] = (
    Retriever("R_TFIDF_WORD", 8, text_free=False),
    Retriever("R_TFIDF_CHAR", 8, text_free=False),
    Retriever("R_CPV_TIME", 8, text_free=True),
    Retriever("R_BUYER_EXPIRY_WINDOW", 8, text_free=True),
    Retriever("R_INCUMBENT_SUPPLIER", 8, text_free=True),
    Retriever("R_REFERENCE_STEM", 6, text_free=True),
)

RANDOM_TAIL = "R_RANDOM_TAIL"

#: Pools at or below this size are shown in full, which is what makes a
#: negative mean "verified against the whole pool" rather than "not in the list
#: I was shown". Nationally the median buyer has three episodes, so a large
#: share of anchors qualify.
EXHAUSTIVE_MAX = 30

#: Above the exhaustive limit but still small enough to show completely.
NEAR_EXHAUSTIVE_MAX = 60

#: Slots offered when a pool is too large to show in full.
SAMPLED_SLOTS = 36

#: Slots reserved for the random tail regardless of what the retrievers want.
#: Without a floor, a large union would crowd out the only unbiased view of
#: what the retrievers miss.
TAIL_FLOOR = 12

#: No retriever may occupy more than this many slots, so one cannot dominate.
MAX_PER_RETRIEVER = 6

#: Gap strata for tail sampling, in days from the anchor's award. Successors
#: concentrate at three to four years, so sampling the tail uniformly would put
#: most draws where successors are rare; stratifying collapses the variance.
GAP_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (90, 730, "0-2y"),
    (730, 1095, "2-3y"),
    (1095, 1460, "3-4y"),
    (1460, 2190, "4-6y"),
    (2190, 2920, "6-8y"),
)

EXPOSURE_SEED = 20260813

#: Exactly the fields a dossier may carry. Anything not named here is withheld,
#: so a column added to the exposure table later cannot leak by default.
ALLOWED_DOSSIER_KEYS: frozenset[str] = frozenset({
    "slot_id", "buyer_name", "buyer_department", "first_notice_date", "award_date",
    "declared_duration_months", "declared_contract_start", "declared_contract_end",
    "cpv_codes", "main_cpv", "procedure_type", "framework_flag", "objet", "full_text",
    "suppliers", "notice_ids", "notice_natures", "notice_urls", "notice_count",
})

#: Substrings that must never appear as a dossier key. Checked in addition to
#: the allow-list, so an accidental rename is still caught.
FORBIDDEN_KEY_SUBSTRINGS: tuple[str, ...] = (
    "score", "similarity", "component", "probability", "retriev", "rank",
    "tail", "gap_days", "weight", "linkage", "fs_match", "predicted", "label",
)


#: Succession window, reused verbatim from the frozen study configuration so
#: that the benchmark's universe matches the one the linkage methods search.
#: Kept as an import rather than a copy: if the study widens its window, the
#: benchmark pool must widen with it or negatives would stop being comparable.
from boamp_pipeline.linkage import MAX_GAP_DAYS, MIN_GAP_DAYS  # noqa: E402


def pool_for_anchor(
    award_date: Any,
    anchor_episode_id: str,
    block_rows: Sequence[Mapping[str, Any]],
    excluded_episode_ids: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Every episode that could be this contract's successor.

    The pool is what a negative is measured against, so it is defined once here
    and shipped as ``pool_definition.json``. An anchor's own episode and any
    episode joined to it by an explicit notice link are removed: an explicit
    link marks the *same procurement*, never a succession.
    """
    if award_date is None:
        return []
    earliest = award_date + timedelta(days=MIN_GAP_DAYS)
    latest = award_date + timedelta(days=MAX_GAP_DAYS)
    pool: list[dict[str, Any]] = []
    for row in block_rows:
        episode_id = row["episode_id"]
        origin = row["episode_origin_date"]
        if episode_id == anchor_episode_id or episode_id in excluded_episode_ids:
            continue
        if origin is None or not (earliest <= origin <= latest):
            continue
        candidate = dict(row)
        candidate["candidate_episode_id"] = episode_id
        candidate["gap_days"] = (origin - award_date).days
        pool.append(candidate)
    return pool


def stable_seed(*parts: Any) -> int:
    payload = "|".join([str(EXPOSURE_SEED), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def gap_bucket(gap_days: Any) -> str:
    """Gap stratum for tail sampling, or ``""`` outside the succession window."""
    try:
        days = int(gap_days)
    except (TypeError, ValueError):
        return ""
    for index, (low, high, label) in enumerate(GAP_BUCKETS):
        # The final bucket is closed at the top so that a candidate sitting
        # exactly on the succession window's last day still has a stratum, and
        # therefore still has a sampling probability.
        last = index == len(GAP_BUCKETS) - 1
        if low <= days <= high if last else low <= days < high:
            return label
    return ""


def exposure_mode(pool_size: int) -> str:
    """How completely a pool can be shown, which decides what a negative means."""
    if pool_size <= EXHAUSTIVE_MAX:
        return "EXHAUSTIVE"
    if pool_size <= NEAR_EXHAUSTIVE_MAX:
        return "NEAR_EXHAUSTIVE"
    return "SAMPLED"


def pool_size_bucket(pool_size: int) -> str:
    """Coarse pool size, disclosed so the annotator knows what "none" means.

    The exact size is withheld: it would reveal how selective the exposure was
    for each candidate, which is a form of ranking information.
    """
    if pool_size <= EXHAUSTIVE_MAX:
        return f"1-{EXHAUSTIVE_MAX}"
    if pool_size <= NEAR_EXHAUSTIVE_MAX:
        return f"{EXHAUSTIVE_MAX + 1}-{NEAR_EXHAUSTIVE_MAX}"
    if pool_size <= 120:
        return f"{NEAR_EXHAUSTIVE_MAX + 1}-120"
    if pool_size <= 400:
        return "121-400"
    return "400+"


# ---------------------------------------------------------------------------
# Tail sampling
# ---------------------------------------------------------------------------


def stratified_tail_sample(
    pool: Sequence[Mapping[str, Any]],
    already_selected: set[str],
    slots: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Sample what the retrievers ignored, uniformly within gap strata.

    Returns the drawn candidates and the per-stratum inclusion probability that
    the two-stage estimator later divides by. Every drawn candidate carries the
    probability of the stratum it came from, because those probabilities differ
    and a single pooled rate would bias the estimate.
    """
    remaining: dict[str, list[Mapping[str, Any]]] = {label: [] for _, _, label in GAP_BUCKETS}
    for candidate in pool:
        if candidate["candidate_episode_id"] in already_selected:
            continue
        bucket = gap_bucket(candidate.get("gap_days"))
        if bucket:
            remaining[bucket].append(candidate)

    non_empty = [label for label, items in remaining.items() if items]
    if not non_empty or slots <= 0:
        return [], {}

    # Slots are handed out one at a time, round-robin over the occupied strata.
    # A plain even split silently under-fills whenever a stratum holds fewer
    # candidates than its share, which would leave the tail below its floor and
    # weaken the only unbiased view of what the retrievers missed.
    shuffled = {label: rng.permutation(len(remaining[label])).tolist() for label in non_empty}
    taken = {label: 0 for label in non_empty}
    total = 0
    while total < slots:
        progressed = False
        for label in non_empty:
            if total >= slots:
                break
            if taken[label] < len(remaining[label]):
                taken[label] += 1
                total += 1
                progressed = True
        if not progressed:
            break

    drawn: list[dict[str, Any]] = []
    probabilities: dict[str, float] = {}
    for label in non_empty:
        count = taken[label]
        if count <= 0:
            continue
        items = remaining[label]
        probability = count / len(items)
        probabilities[label] = probability
        for position in shuffled[label][:count]:
            candidate = dict(items[int(position)])
            candidate["tail_stratum"] = label
            candidate["tail_inclusion_probability"] = probability
            drawn.append(candidate)
    return drawn, probabilities


def interleave(
    ranked_by_retriever: Mapping[str, Sequence[str]],
    limit: int,
    max_per_retriever: int = MAX_PER_RETRIEVER,
) -> tuple[list[str], dict[str, list[str]]]:
    """Merge retriever outputs round-robin by rank.

    Round-robin rather than concatenation, so that no retriever's whole list is
    admitted before another's is considered. Provenance is returned separately;
    it is recorded but never shown.
    """
    provenance: dict[str, list[str]] = {}
    for name, ranked in ranked_by_retriever.items():
        for episode_id in ranked:
            provenance.setdefault(episode_id, []).append(name)

    taken: list[str] = []
    seen: set[str] = set()
    counts: dict[str, int] = {name: 0 for name in ranked_by_retriever}
    depth = 0
    max_depth = max((len(v) for v in ranked_by_retriever.values()), default=0)
    while len(taken) < limit and depth < max_depth:
        for name, ranked in ranked_by_retriever.items():
            if len(taken) >= limit:
                break
            if depth >= len(ranked) or counts[name] >= max_per_retriever:
                continue
            episode_id = ranked[depth]
            if episode_id in seen:
                continue
            seen.add(episode_id)
            taken.append(episode_id)
            counts[name] += 1
        depth += 1
    return taken, provenance


# ---------------------------------------------------------------------------
# Blinding
# ---------------------------------------------------------------------------


def assert_no_forbidden_keys(payload: Mapping[str, Any], where: str = "dossier") -> None:
    """Fail loudly if anything score-like reached a blind artifact."""
    for key in payload:
        lowered = str(key).lower()
        for banned in FORBIDDEN_KEY_SUBSTRINGS:
            if banned in lowered:
                raise ValueError(
                    f"{where} carries forbidden key {key!r} (matched {banned!r}); "
                    "exposure provenance and model scores must stay out of blind artifacts"
                )


def blind_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one episode record down to what an annotator may see.

    An allow-list rather than a deny-list: a column added upstream is withheld
    until someone decides it is safe, instead of leaking until someone notices.
    """
    projected = {key: value for key, value in record.items() if key in ALLOWED_DOSSIER_KEYS}
    assert_no_forbidden_keys(projected)
    return projected


def assign_slots(
    candidate_ids: Sequence[str], anchor_id: str, pass_id: str
) -> list[tuple[str, str]]:
    """Shuffle candidates and give them opaque identifiers.

    Slots are assigned *after* the shuffle, and the shuffle is keyed on the
    pass, so position encodes nothing and a leaked pass-A dossier cannot be
    aligned with pass B.
    """
    rng = np.random.default_rng(stable_seed(anchor_id, pass_id))
    order = rng.permutation(len(candidate_ids))
    return [(f"C{position + 1:02d}", candidate_ids[int(index)])
            for position, index in enumerate(order)]


def presentation_order_correlation(
    slot_positions: Sequence[int], scores: Sequence[float]
) -> float:
    """Spearman correlation between slot position and a model score.

    The measurable form of "presentation order carries no ranking". Computed
    over all exposed candidates pooled; a non-trivial value means the shuffle
    failed and the exposure is unusable.
    """
    positions = pd.Series(slot_positions, dtype=float)
    values = pd.Series(scores, dtype=float)
    usable = positions.notna() & values.notna()
    if usable.sum() < 3 or values[usable].nunique() < 2:
        return 0.0
    return float(positions[usable].corr(values[usable], method="spearman"))


# ---------------------------------------------------------------------------
# Two-stage estimation
# ---------------------------------------------------------------------------


def two_stage_positive_estimate(
    union_positives: int,
    tail_positives_by_stratum: Mapping[str, int],
    tail_probabilities: Mapping[str, float],
) -> dict[str, float]:
    """Estimate an anchor's true positive count from a partial review.

    The retriever union is reviewed exhaustively, so its positives are a
    census. The rest of the pool is only sampled, so its positives are scaled
    up by the sampling probability of the stratum they came from::

        n_hat = union_positives + sum_b tail_positives_b / pi_b

    This is what replaces v1's assumption that a truncated list contained
    everything. The variance term is reported alongside because a dozen tail
    slots against a pool of hundreds is a noisy estimate, and quoting the point
    value alone would repeat v1's mistake in a new form.
    """
    estimated_tail = 0.0
    variance = 0.0
    for stratum, positives in tail_positives_by_stratum.items():
        probability = tail_probabilities.get(stratum, 0.0)
        if probability <= 0:
            continue
        estimated_tail += positives / probability
        # Horvitz-Thompson variance for a without-replacement stratum sample.
        variance += positives * (1.0 - probability) / (probability ** 2)
    return {
        "union_positives": float(union_positives),
        "estimated_tail_positives": estimated_tail,
        "estimated_total_positives": union_positives + estimated_tail,
        "tail_standard_error": float(np.sqrt(variance)),
    }


def retriever_coverage(
    provenance_by_candidate: Mapping[str, Iterable[str]],
    positive_candidates: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Which retrievers actually surfaced the confirmed positives.

    Reported rather than gated. If one retriever surfaces nearly everything the
    union is still fair, but the imbalance belongs on the record -- especially
    when that retriever shares its mechanism with the method under evaluation.
    """
    positives = list(positive_candidates)
    exposed_counts: dict[str, int] = {}
    positive_counts: dict[str, int] = {}
    for candidate, retrievers in provenance_by_candidate.items():
        for name in retrievers:
            exposed_counts[name] = exposed_counts.get(name, 0) + 1
            if candidate in positives:
                positive_counts[name] = positive_counts.get(name, 0) + 1
    total_positive = len(positives)
    return {
        name: {
            "exposed": exposed_counts.get(name, 0),
            "positives_surfaced": positive_counts.get(name, 0),
            "share_of_positives": (
                round(positive_counts.get(name, 0) / total_positive, 4)
                if total_positive else None
            ),
        }
        for name in sorted(set(exposed_counts) | set(positive_counts))
    }
