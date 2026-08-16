#!/usr/bin/env python3
"""Two descriptive audits of the candidate-generation (blocking) stage.

Candidate generation is deliberately recall-oriented: it imposes buyer-identity
plausibility and a broad future-time window, and it does *not* impose CPV
continuity as a hard blocking rule. Both halves of that design need evidence,
and neither is a question the linkage scorer can answer, because a pair the
blocking step never proposes is invisible to every method downstream.

``blocking_loss``
    The regional reference already reports an aggregate recall ceiling -- the
    share of reviewed successors the candidate pool exposes. That number says
    how much truth is unreachable but not *why*, so a ceiling below 1.0 cannot
    be told apart from a bug. This audit re-derives the ceiling and attributes
    every unreachable case to the specific blocking condition that rejected it,
    evaluated in pipeline order.

``cpv_continuity``
    Relaxing hard same-CPV blocking is a design decision that needs a
    descriptive check, not an assertion. This audit measures how often accepted
    links stay inside one CPV division, and -- the decisive part -- measures the
    same quantity on the reviewed reference successors, which were labelled
    without any knowledge of the linkage methods. If the reviewed truth itself
    crosses divisions at a similar rate, then a hard same-division block would
    discard genuine successors rather than noise, and the relaxation is
    justified by the reference rather than by convenience.

Both audits are descriptive. Neither tunes a threshold, and neither is wired
into the acceptance rule; they document the blocking stage that the accepted
links and the reported recall figures already depend on.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.linkage import (  # noqa: E402
    MAX_GAP_DAYS,
    MIN_GAP_DAYS,
    compatible_buyer_identifiers,
    cpv_divisions,
    normalize_buyer_for_blocking,
    parse_json_list,
)

PROCESSED = Path("data/processed/boamp")
REFERENCE = PROCESSED / "regional_benchmark"
AUDIT_VERSION = "boamp_candidate_generation_audit_v1.0"

#: Blocking conditions in the order ``build_linkage_candidates.py`` applies
#: them. The first one a reviewed pair fails is the reason it is unreachable;
#: reporting a later one would misattribute the loss.
#: Word-level similarity below which an accepted link rests essentially on the
#: character-level analyser alone. Character n-grams are what let the matcher
#: recognise reworded titles that word overlap misses, so a low word score is
#: usually benign; it is also the precondition for the boilerplate failure mode,
#: where shared framework-agreement legal phrasing carries the score. Reported as
#: an exposure count, not used as an acceptance rule.
LOW_WORD_SIMILARITY = 0.50

BLOCKING_STAGES = (
    "ANCHOR_ABSENT_FROM_COHORT",
    "SUCCESSOR_ABSENT_FROM_CANDIDATE_POOL",
    "BUYER_BLOCKING_MISS",
    "CONFLICTING_VALIDATED_SIREN",
    "OUTSIDE_TIME_WINDOW",
)


def load_truth() -> pd.DataFrame:
    parts = [pd.read_parquet(PROJECT_ROOT / REFERENCE / f"benchmark_{s}.parquet") for s in ("dev", "validation")]
    return pd.concat(parts, ignore_index=True)


def load_episodes() -> pd.DataFrame:
    episodes = pd.read_parquet(
        PROJECT_ROOT / PROCESSED / "episodes_grand_ouest.parquet",
        columns=[
            "episode_id", "episode_origin_date", "buyer_siren", "buyer_key",
            "buyer_name_raw", "buyer_name_normalized", "main_cpv", "all_cpvs_json",
            "episode_text",
        ],
    )
    episodes["buyer_name_blocking"] = episodes["buyer_name_normalized"].map(normalize_buyer_for_blocking)
    episodes["cpv_division"] = episodes["main_cpv"].astype(str).str.strip().str[:2]
    episodes["cpv_division"] = episodes["cpv_division"].where(episodes["cpv_division"].str.len().eq(2))
    episodes["cpv_division_set"] = episodes["all_cpvs_json"].map(parse_json_list).map(cpv_divisions)
    return episodes.set_index("episode_id")


def cohort_absence_reason(episode_id: str, episodes: pd.DataFrame) -> str:
    """Why a Grand Ouest episode is not a cohort anchor.

    The cohort requires a digital CPV, an award notice, and an award date
    resolvable from the Grand Ouest notice table. The third is the one that can
    fail silently for an episode that looks eligible, so it is named explicitly.
    """
    if episode_id not in episodes.index:
        return "anchor episode is not in the Grand Ouest episode table"
    membership = pd.read_parquet(
        PROJECT_ROOT / PROCESSED / "episode_membership.parquet", columns=["idweb", "episode_id"]
    )
    notices = pd.read_parquet(
        PROJECT_ROOT / PROCESSED / "notices_grand_ouest.parquet",
        columns=["idweb", "nature", "publication_date"],
    )
    owned = membership.loc[membership["episode_id"].eq(episode_id), "idweb"]
    present = notices.loc[notices["idweb"].isin(owned)]
    if present.loc[present["nature"].eq("ATTRIBUTION")].empty:
        return (
            "episode carries an award notice, but that notice has no structured "
            "Grand Ouest address and is therefore absent from the regional notice "
            "table, so no award date could be resolved and the episode was dropped "
            "from the cohort"
        )
    return "anchor did not meet a cohort condition (digital CPV, award notice, award date, cutoff)"


def diagnose_pair(
    anchor_id: str,
    successor_id: str,
    cohort: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, Any]:
    """Attribute one unreachable reviewed pair to its first failing condition."""
    record: dict[str, Any] = {
        "anchor_episode_id": anchor_id,
        "reviewed_successor_id": successor_id,
        "blocking_stage": None,
        "blocking_reason": None,
        "anchor_buyer": None,
        "successor_buyer": None,
        "buyer_evidence": None,
        "gap_days": None,
        "anchor_cpv": None,
        "successor_cpv": None,
    }
    successor = episodes.loc[successor_id] if successor_id in episodes.index else None
    if successor is not None:
        record["successor_buyer"] = str(successor["buyer_name_raw"])
        record["successor_cpv"] = str(successor["main_cpv"])

    if anchor_id not in cohort.index:
        record["blocking_stage"] = "ANCHOR_ABSENT_FROM_COHORT"
        record["blocking_reason"] = cohort_absence_reason(anchor_id, episodes)
        if anchor_id in episodes.index:
            anchor_episode = episodes.loc[anchor_id]
            record["anchor_buyer"] = str(anchor_episode["buyer_name_raw"])
            record["anchor_cpv"] = str(anchor_episode["main_cpv"])
            record["buyer_evidence"] = "anchor never reached the blocking step"
            if successor is not None:
                gap = (
                    pd.to_datetime(successor["episode_origin_date"])
                    - pd.to_datetime(anchor_episode["episode_origin_date"])
                ).days
                record["gap_days"] = int(gap)
        return record

    anchor = cohort.loc[anchor_id]
    record["anchor_buyer"] = str(anchor["buyer_name_raw"])
    record["anchor_cpv"] = str(anchor["main_cpv"])

    if successor is None:
        record["blocking_stage"] = "SUCCESSOR_ABSENT_FROM_CANDIDATE_POOL"
        record["blocking_reason"] = "reviewed successor is not a Grand Ouest episode"
        return record

    anchor_siren = str(anchor["buyer_siren"] or "").strip()
    successor_siren = str(successor["buyer_siren"] or "").strip()
    anchor_key = str(anchor["buyer_key"] or "").strip()
    successor_key = str(successor["buyer_key"] or "").strip()
    anchor_name = str(anchor["buyer_name_blocking"] or "").strip()
    successor_name = str(successor["buyer_name_blocking"] or "").strip()
    gap = (pd.to_datetime(successor["episode_origin_date"]) - pd.to_datetime(anchor["award_date"])).days
    record["gap_days"] = int(gap)
    record["buyer_evidence"] = (
        f"anchor key={anchor_key or 'none'} siren={anchor_siren or 'none'} name={anchor_name or 'none'}; "
        f"successor key={successor_key or 'none'} siren={successor_siren or 'none'} name={successor_name or 'none'}"
    )

    same_key = bool(anchor_key) and anchor_key == successor_key
    same_name = bool(anchor_name) and anchor_name == successor_name
    if not (same_key or same_name):
        record["blocking_stage"] = "BUYER_BLOCKING_MISS"
        record["blocking_reason"] = (
            "neither the buyer key nor the normalised buyer name matches, so the "
            "successor was never proposed for this anchor"
        )
        return record
    if not compatible_buyer_identifiers(anchor_siren, successor_siren):
        record["blocking_stage"] = "CONFLICTING_VALIDATED_SIREN"
        record["blocking_reason"] = "anchor and successor carry different validated SIRENs"
        return record
    if not (MIN_GAP_DAYS <= gap <= MAX_GAP_DAYS):
        record["blocking_stage"] = "OUTSIDE_TIME_WINDOW"
        record["blocking_reason"] = (
            f"gap of {gap} days falls outside the {MIN_GAP_DAYS}-{MAX_GAP_DAYS} day window"
        )
        return record

    record["blocking_stage"] = "UNEXPLAINED"
    record["blocking_reason"] = "pair satisfies every blocking condition but is absent from the pool"
    return record


def audit_blocking_loss(
    truth: pd.DataFrame, exposure: pd.DataFrame, cohort: pd.DataFrame, episodes: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    pairs = set(zip(exposure["anchor_episode_id"], exposure["candidate_episode_id"]))
    usable = truth.loc[truth["anchor_verdict"].ne("ANCHOR_UNUSABLE")]
    positives = usable.loc[usable["has_successor_primary"]]

    unreachable: list[dict[str, Any]] = []
    reachable = 0
    for record in positives.itertuples(index=False):
        successors = json.loads(record.successors_primary_json)
        if any((record.anchor_episode_id, successor) in pairs for successor in successors):
            reachable += 1
            continue
        for successor in successors:
            row = diagnose_pair(record.anchor_episode_id, successor, cohort, episodes)
            row["sample_id"] = record.sample_id
            row["split"] = record.split
            unreachable.append(row)

    frame = pd.DataFrame(unreachable)
    if not frame.empty:
        frame = frame[[
            "sample_id", "split", "anchor_episode_id", "reviewed_successor_id",
            "blocking_stage", "blocking_reason", "anchor_buyer", "successor_buyer",
            "buyer_evidence", "gap_days", "anchor_cpv", "successor_cpv",
        ]]
    summary = {
        "reviewed_positive_anchors": int(len(positives)),
        "reachable_anchors": int(reachable),
        "unreachable_anchors": int(len(positives) - reachable),
        "pairs_completeness": round(reachable / len(positives), 4) if len(positives) else None,
        "blocking_stage_counts": (
            {str(k): int(v) for k, v in frame["blocking_stage"].value_counts().items()}
            if not frame.empty else {}
        ),
        "unexplained_cases": (
            int(frame["blocking_stage"].eq("UNEXPLAINED").sum()) if not frame.empty else 0
        ),
    }
    return summary, frame


def audit_cpv_continuity(
    links: pd.DataFrame, truth: pd.DataFrame, episodes: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    def divisions(frame: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
        out = frame.copy()
        out["anchor_division"] = out[left].map(episodes["cpv_division"])
        out["successor_division"] = out[right].map(episodes["cpv_division"])
        out["anchor_division_set"] = out[left].map(episodes["cpv_division_set"])
        out["successor_division_set"] = out[right].map(episodes["cpv_division_set"])
        return out

    scored = divisions(links, "anchor_episode_id", "candidate_episode_id")
    observed = scored.loc[scored["anchor_division"].notna() & scored["successor_division"].notna()].copy()
    observed["same_division"] = observed["anchor_division"].eq(observed["successor_division"])
    observed["shares_any_division"] = [
        bool(a & b) for a, b in zip(observed["anchor_division_set"], observed["successor_division_set"])
    ]

    # The reviewed successors were labelled without any knowledge of the linkage
    # methods, so their own cross-division rate is the benchmark against which
    # the accepted links' rate has to be read.
    usable = truth.loc[truth["anchor_verdict"].ne("ANCHOR_UNUSABLE")]
    reviewed_rows: list[dict[str, Any]] = []
    for record in usable.loc[usable["has_successor_primary"]].itertuples(index=False):
        for successor in json.loads(record.successors_primary_json):
            reviewed_rows.append(
                {"anchor_episode_id": record.anchor_episode_id, "candidate_episode_id": successor}
            )
    reviewed = divisions(pd.DataFrame(reviewed_rows), "anchor_episode_id", "candidate_episode_id")
    reviewed = reviewed.loc[reviewed["anchor_division"].notna() & reviewed["successor_division"].notna()].copy()
    reviewed["same_division"] = reviewed["anchor_division"].eq(reviewed["successor_division"])
    reviewed["shares_any_division"] = [
        bool(a & b) for a, b in zip(reviewed["anchor_division_set"], reviewed["successor_division_set"])
    ]

    # Successor reuse. M_B ranks candidates per anchor independently, with no
    # one-to-one constraint, so one episode may be accepted as the successor of
    # several anchors. Multi-lot programmes make that legitimate, but a heavily
    # reused successor is also the signature of the boilerplate failure mode:
    # notice text dominated by standard framework-agreement phrasing can clear a
    # character-level threshold on shared legal wording alone.
    reuse = links["candidate_episode_id"].value_counts()

    transitions = (
        observed.groupby(["anchor_division", "successor_division"]).size().reset_index(name="count")
    )
    transitions["share"] = (transitions["count"] / len(observed)).round(4)
    transitions = transitions.sort_values("count", ascending=False).reset_index(drop=True)

    summary = {
        "accepted_links": int(len(links)),
        "accepted_links_with_both_divisions_observed": int(len(observed)),
        "same_cpv2_count": int(observed["same_division"].sum()),
        "same_cpv2_share": round(float(observed["same_division"].mean()), 4),
        "cross_cpv2_count": int((~observed["same_division"]).sum()),
        "cross_cpv2_share": round(float((~observed["same_division"]).mean()), 4),
        "shares_any_division_share": round(float(observed["shares_any_division"].mean()), 4),
        "reviewed_reference_pairs": int(len(reviewed)),
        "reviewed_same_cpv2_count": int(reviewed["same_division"].sum()),
        "reviewed_same_cpv2_share": round(float(reviewed["same_division"].mean()), 4),
        "reviewed_cross_cpv2_count": int((~reviewed["same_division"]).sum()),
        "reviewed_cross_cpv2_share": round(float((~reviewed["same_division"]).mean()), 4),
        "reviewed_shares_any_division_share": round(float(reviewed["shares_any_division"].mean()), 4),
        # What a hard same-division block would cost, priced on reviewed truth
        # rather than on the method's own output.
        "reviewed_successors_lost_to_hard_same_division_block": int((~reviewed["same_division"]).sum()),
        "recall_ceiling_under_hard_same_division_block": round(
            float(reviewed["same_division"].mean()), 4
        ),
        "reviewed_successors_lost_to_hard_shared_division_block": int(
            (~reviewed["shares_any_division"]).sum()
        ),
        "accepted_links_carried_by_char_similarity": int(
            (links["word_tfidf_similarity"] < LOW_WORD_SIMILARITY).sum()
        ),
        "low_word_similarity_threshold": LOW_WORD_SIMILARITY,
        "distinct_successors": int(reuse.size),
        "successors_accepted_by_multiple_anchors": int((reuse >= 2).sum()),
        "max_anchors_per_successor": int(reuse.max()) if reuse.size else 0,
        "most_reused_successor_episode_id": str(reuse.index[0]) if reuse.size else None,
        "top_cross_division_flows": [
            {
                "anchor_division": str(row.anchor_division),
                "successor_division": str(row.successor_division),
                "count": int(row.count),
            }
            for row in transitions.loc[
                transitions["anchor_division"].ne(transitions["successor_division"])
            ].head(8).itertuples(index=False)
        ],
    }
    return summary, transitions


def write_report(summary: dict[str, Any]) -> Path:
    blocking = summary["blocking_loss"]
    cpv = summary["cpv_continuity"]
    stages = "\n".join(
        f"- `{stage}`: {count}" for stage, count in blocking["blocking_stage_counts"].items()
    ) or "- none"
    flows = "\n".join(
        f"- CPV-{flow['anchor_division']} to CPV-{flow['successor_division']}: {flow['count']}"
        for flow in cpv["top_cross_division_flows"]
    ) or "- none"
    text = f"""# Candidate-Generation Audit

Generated by `scripts/audit_candidate_generation.py`. Both sections are
descriptive checks on the blocking stage. Neither changes the acceptance rule.

## 1. Blocking Loss On The Regional Reference

Candidate generation exposed `{blocking['reachable_anchors']}` of the
`{blocking['reviewed_positive_anchors']}` reviewed successors in the Grand Ouest
regional reference, a pairs completeness of
`{blocking['pairs_completeness']:.4f}`. This is a property of the reference
sample, not a population recall estimate.

Every unreachable case is attributed below to the first blocking condition it
fails, evaluated in the order `build_linkage_candidates.py` applies them:

{stages}

Case detail is in `data/processed/boamp/candidate_generation_unreachable.csv`.
Unexplained cases: `{blocking['unexplained_cases']}`. A non-zero count there
would indicate an implementation defect rather than a blocking trade-off.

## 2. CPV Continuity Of Accepted Links

Of `{cpv['accepted_links']}` accepted primary links,
`{cpv['accepted_links_with_both_divisions_observed']}` have a CPV division
observed on both sides.

- same CPV division: `{cpv['same_cpv2_count']}` (`{cpv['same_cpv2_share']:.4f}`)
- cross CPV division: `{cpv['cross_cpv2_count']}` (`{cpv['cross_cpv2_share']:.4f}`)
- share sharing at least one division across all listed CPV codes:
  `{cpv['shares_any_division_share']:.4f}`

The reviewed reference successors, labelled with no knowledge of any linkage
method, cross divisions at a comparable rate:
`{cpv['reviewed_cross_cpv2_count']}` of `{cpv['reviewed_reference_pairs']}`
reviewed pairs (`{cpv['reviewed_cross_cpv2_share']:.4f}`) sit in different CPV
divisions, against `{cpv['cross_cpv2_share']:.4f}` among accepted links.

Imposing hard same-division blocking would therefore discard
`{cpv['reviewed_successors_lost_to_hard_same_division_block']}` of the
`{cpv['reviewed_reference_pairs']}` reviewed successors and cut the attainable
recall ceiling to `{cpv['recall_ceiling_under_hard_same_division_block']:.4f}`.
The relaxation of hard CPV blocking is supported by the reference rather than by
convenience.

Largest cross-division flows among accepted links:

{flows}

The full transition table is in
`data/processed/boamp/candidate_generation_cpv_transitions.csv`.
"""
    path = PROJECT_ROOT / "CANDIDATE_GENERATION_AUDIT.md"
    path.write_text(text, encoding="utf-8")
    return path


def build() -> dict[str, Any]:
    truth = load_truth()
    episodes = load_episodes()
    exposure = pd.read_parquet(
        PROJECT_ROOT / REFERENCE / "exposure_full.parquet",
        columns=["anchor_episode_id", "candidate_episode_id"],
    )
    cohort = pd.read_parquet(
        PROJECT_ROOT / PROCESSED / "survival_cohort.parquet",
        columns=["episode_id", "award_date", "buyer_siren", "buyer_key",
                 "buyer_name_raw", "buyer_name_blocking", "main_cpv"],
    ).set_index("episode_id")
    links = pd.read_parquet(
        PROJECT_ROOT / PROCESSED / "accepted_successor_links.parquet",
        columns=["anchor_episode_id", "candidate_episode_id", "word_tfidf_similarity"],
    )

    blocking_summary, unreachable = audit_blocking_loss(truth, exposure, cohort, episodes)
    cpv_summary, transitions = audit_cpv_continuity(links, truth, episodes)

    out = PROJECT_ROOT / PROCESSED
    unreachable.to_csv(out / "candidate_generation_unreachable.csv", index=False)
    transitions.to_csv(out / "candidate_generation_cpv_transitions.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "audit_version": AUDIT_VERSION,
        "blocking_rule": (
            f"same buyer_key OR same normalised buyer name, conflicting validated SIRENs "
            f"excluded, candidate origin {MIN_GAP_DAYS}-{MAX_GAP_DAYS} days after the anchor award"
        ),
        "blocking_loss": blocking_summary,
        "cpv_continuity": cpv_summary,
    }
    (out / "candidate_generation_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Accepted for pipeline symmetry.")
    return parser.parse_args()


def main() -> int:
    parse_args()
    print(json.dumps(build(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
