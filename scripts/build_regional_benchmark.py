#!/usr/bin/env python3
"""Materialise the canonical Grand Ouest regional linkage benchmark.

The reference itself is not produced here. It is the stratified 120-anchor
Grand Ouest review carried out on 2026-08-11 and stored under
``data/reference/regional_link_benchmark/``. That review was conducted against
real BOAMP notices and official notice URLs, before the linkage methods in this
repository existed, and it is *not* a product of any scoring code in this
repository. That independence from the linkage feature set is the entire reason
it replaced the earlier France-level reference, whose labels were emitted by
deterministic rules built from the same text/CPV/date evidence the methods use.

What this script does is the mechanical part: the review recorded episode ids
from an earlier episode reconstruction, and those ids no longer exist. Every
anchor and successor is therefore re-resolved through its BOAMP notice ids
(``idweb``) against the current ``episodes_grand_ouest.parquet``. Anchors that
cannot be resolved to exactly one current episode are dropped and counted, not
guessed.

Outputs mirror the truth contract the evaluator already speaks, so the
evaluation code does not need a second dialect:

``benchmark_dev.parquet``
    The reference's own ``PILOT_DEVELOPMENT`` stratum. Used for threshold
    *display* only.
``benchmark_validation.parquet``
    The reference's own ``LOCKED_TEST`` stratum. The held-out split. The
    operating point was frozen before this reference was consulted, so these
    anchors were never used to choose it.
``exposure_full.parquet``
    The production candidate pairs for the benchmark anchors, so the benchmark
    measures the linkage stack as the study actually runs it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED = Path("data/processed/boamp")
REFERENCE_DIR = Path("data/reference/regional_link_benchmark")
REFERENCE_CSV = REFERENCE_DIR / "BOAMP_Internship_Reference_120.csv"
CONFIRMED_CSV = REFERENCE_DIR / "BOAMP_Internship_Confirmed_Successor_Links.csv"
OUTPUT_DIR = PROCESSED / "regional_benchmark"

BENCHMARK_SCHEMA = "boamp_regional_benchmark_schema_1.0"
REFERENCE_VERSION = "internship_v1.0"

#: Same observation cutoff as the survival cohort; a negative anchor whose
#: contract could still be running at the cutoff is censored, not a negative.
STUDY_CUTOFF = date(2025, 12, 31)

IDWEB_RE = re.compile(r"idweb:([0-9]{2}-[0-9]+)")

#: The reference's split names, mapped onto the evaluator's vocabulary. The
#: mapping is a rename, not a re-split: no anchor moves between them.
SPLIT_MAP = {"PILOT_DEVELOPMENT": "dev", "LOCKED_TEST": "validation"}

#: Review outcome -> anchor verdict. ``OUTSIDE_SCOPE`` and
#: ``INSUFFICIENT_INFORMATION`` are unusable rather than negative: the reviewer
#: declined to decide, which is not evidence of no successor.
VERDICT_MAP = {
    "OBSERVED_SUCCESSOR": "HAS_RENEWAL_SUCCESSOR",
    "MULTIPLE_SUCCESSORS": "HAS_RENEWAL_SUCCESSOR",
    "NO_OBSERVED_SUCCESSOR_IN_SCOPE": "NO_RENEWAL_AMONG_SHOWN",
    "OUTSIDE_SCOPE": "ANCHOR_UNUSABLE",
    "INSUFFICIENT_INFORMATION": "ANCHOR_UNUSABLE",
}


def notice_to_episode(episodes: pd.DataFrame) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for episode_id, payload in zip(
        episodes["episode_id"], episodes["constituent_notice_ids_json"]
    ):
        for notice_id in json.loads(payload):
            index.setdefault(str(notice_id), set()).add(episode_id)
    return index


def episodes_for(notice_ids: list[str], index: dict[str, set[str]]) -> set[str]:
    resolved: set[str] = set()
    for notice_id in notice_ids:
        resolved |= index.get(str(notice_id), set())
    return resolved


def parse_json_list(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        return [str(item) for item in json.loads(value)]
    except json.JSONDecodeError:
        return []


def notice_ids_from_urls(value: Any) -> list[str]:
    return [match for url in parse_json_list(value) for match in IDWEB_RE.findall(url)]


def resolve_anchors(
    reference: pd.DataFrame, index: dict[str, set[str]]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Re-resolve every reviewed anchor onto the current episode reconstruction."""
    rows: list[dict[str, Any]] = []
    unresolved = ambiguous = 0
    for record in reference.itertuples(index=False):
        candidates = episodes_for(parse_json_list(record.anchor_notice_ids_json), index)
        if len(candidates) == 1:
            status = "RESOLVED"
            episode_id = next(iter(candidates))
        else:
            status = "UNRESOLVED_NO_CURRENT_EPISODE" if not candidates else "AMBIGUOUS_MULTIPLE_EPISODES"
            episode_id = None
            unresolved += not candidates
            ambiguous += len(candidates) > 1
        rows.append({
            "sample_id": record.sample_id,
            "anchor_episode_id": episode_id,
            "remap_status": status,
            "remap_candidate_count": len(candidates),
            "legacy_anchor_episode_id": record.anchor_episode_id,
            "final_outcome": record.final_outcome,
            "primary_evaluation_eligible": bool(record.primary_evaluation_eligible),
            "benchmark_split_source": record.benchmark_split,
            "stratum_id": record.sampling_stratum,
            "inclusion_probability": record.inclusion_probability,
            "design_weight": record.sampling_weight,
            "pool_size": record.broad_candidate_pool_n,
            "labels_recorded": record.review_candidates_exported_n,
            "anchor_award_notice_date": record.anchor_award_notice_date,
            "anchor_expected_end_date": record.anchor_expected_end_date,
            "anchor_theme": record.anchor_theme,
            "final_confidence": record.final_confidence,
            "evidence_notice_ids": notice_ids_from_urls(record.final_evidence_urls),
        })
    return pd.DataFrame(rows), {"unresolved": unresolved, "ambiguous": ambiguous}


def resolve_successors(
    anchors: pd.DataFrame, confirmed: pd.DataFrame, index: dict[str, set[str]]
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Resolve the reviewed successor of each positive anchor.

    Two independent routes are used and required to agree: the pair-level
    confirmed-links file, and the anchor-level evidence URLs minus the anchor's
    own episode. Disagreement is reported rather than silently preferred.
    """
    anchor_episode = dict(zip(anchors["sample_id"], anchors["anchor_episode_id"]))
    successors: dict[str, list[str]] = {}
    for sample_id, group in confirmed.groupby("sample_id"):
        anchor = anchor_episode.get(sample_id)
        if anchor is None:
            continue
        resolved: set[str] = set()
        for value in group["candidate_urls_json"]:
            resolved |= episodes_for(notice_ids_from_urls(value), index)
        successors[sample_id] = sorted(resolved - {anchor})

    disagreements = 0
    for record in anchors.itertuples(index=False):
        if record.anchor_episode_id is None or record.sample_id not in successors:
            continue
        from_evidence = episodes_for(record.evidence_notice_ids, index) - {record.anchor_episode_id}
        if from_evidence != set(successors[record.sample_id]):
            disagreements += 1
    return successors, {"route_disagreements": disagreements}


def build_truth(
    anchors: pd.DataFrame, successors: dict[str, list[str]]
) -> pd.DataFrame:
    frame = anchors.loc[anchors["anchor_episode_id"].notna()].copy()
    frame["anchor_verdict"] = frame["final_outcome"].map(VERDICT_MAP)
    frame["successors"] = frame["sample_id"].map(successors).apply(
        lambda value: value if isinstance(value, list) else []
    )

    # An anchor the reviewer called positive but whose successor no longer
    # resolves cannot be scored either way: it would punish a method for
    # missing a target the benchmark can no longer name.
    positive = frame["anchor_verdict"].eq("HAS_RENEWAL_SUCCESSOR")
    frame.loc[positive & frame["successors"].apply(len).eq(0), "anchor_verdict"] = "ANCHOR_UNUSABLE"
    frame.loc[~frame["primary_evaluation_eligible"], "anchor_verdict"] = "ANCHOR_UNUSABLE"

    has_successor = frame["anchor_verdict"].eq("HAS_RENEWAL_SUCCESSOR")
    # An unusable anchor carries no truth at all. Leaving a successor on it
    # would let a later consumer read a target the evaluator never scores.
    frame.loc[~has_successor, "successors"] = frame.loc[~has_successor, "successors"].apply(lambda _: [])
    expected_end = pd.to_datetime(frame["anchor_expected_end_date"], errors="coerce")
    frame["successors_primary_json"] = frame["successors"].apply(json.dumps)
    frame["has_successor_primary"] = has_successor
    frame["split"] = frame["benchmark_split_source"].map(SPLIT_MAP)
    frame["frame"] = "PROBABILITY"
    frame["exposure_mode"] = "SAMPLED"
    # "No successor among the ~25 candidates the reviewer was shown" is not
    # "no successor in the pool"; the pool ran to several hundred episodes.
    frame["negative_verification"] = "AMONG_SHOWN"
    frame["negative_is_censored"] = (~has_successor) & expected_end.gt(pd.Timestamp(STUDY_CUTOFF))
    frame["benchmark_schema"] = BENCHMARK_SCHEMA
    frame["reference_version"] = REFERENCE_VERSION
    return frame


TRUTH_OUTPUT_COLUMNS = [
    "anchor_episode_id", "sample_id", "anchor_verdict", "exposure_mode", "split",
    "frame", "stratum_id", "inclusion_probability", "design_weight", "pool_size",
    "labels_recorded", "negative_is_censored", "negative_verification",
    "successors_primary_json", "has_successor_primary", "anchor_theme",
    "final_confidence", "benchmark_schema", "reference_version",
]


def build_exposure(truth: pd.DataFrame, candidates_path: Path) -> pd.DataFrame:
    candidates = pd.read_parquet(candidates_path)
    anchors = set(truth["anchor_episode_id"])
    exposure = candidates.loc[candidates["anchor_episode_id"].isin(anchors)].copy()
    exposure["frame"] = "PROBABILITY"
    exposure = exposure.merge(
        truth[["anchor_episode_id", "stratum_id", "split"]],
        on="anchor_episode_id", how="left",
    )
    exposure["exposure_mode"] = "PRODUCTION_CANDIDATE_POOL"
    exposure["exposure_version"] = BENCHMARK_SCHEMA
    return exposure


def candidate_reachability(truth: pd.DataFrame, exposure: pd.DataFrame) -> dict[str, Any]:
    """How much of the reviewed truth the candidate generator can even reach.

    A successor the blocking step never proposes is unreachable by every method
    downstream, so this is the recall ceiling and belongs next to any recall
    number the benchmark produces.
    """
    pairs = set(zip(exposure["anchor_episode_id"], exposure["candidate_episode_id"]))
    anchors_with_candidates = set(exposure["anchor_episode_id"])
    usable = truth.loc[truth["anchor_verdict"].ne("ANCHOR_UNUSABLE")]
    positives = usable.loc[usable["has_successor_primary"]]
    reachable = sum(
        any((record.anchor_episode_id, successor) in pairs
            for successor in json.loads(record.successors_primary_json))
        for record in positives.itertuples(index=False)
    )
    return {
        "usable_anchors": int(len(usable)),
        "anchors_with_at_least_one_candidate": int(
            usable["anchor_episode_id"].isin(anchors_with_candidates).sum()
        ),
        "positive_anchors": int(len(positives)),
        "positive_anchors_with_reviewed_successor_in_pool": int(reachable),
        "candidate_generation_recall_ceiling": (
            round(reachable / len(positives), 4) if len(positives) else None
        ),
    }


def build(force: bool) -> dict[str, Any]:
    output_dir = PROJECT_ROOT / OUTPUT_DIR
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{OUTPUT_DIR} already populated; use --force")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(PROJECT_ROOT / REFERENCE_CSV)
    confirmed = pd.read_csv(PROJECT_ROOT / CONFIRMED_CSV)
    episodes = pd.read_parquet(
        PROJECT_ROOT / PROCESSED / "episodes_grand_ouest.parquet",
        columns=["episode_id", "constituent_notice_ids_json"],
    )
    index = notice_to_episode(episodes)

    anchors, remap_counts = resolve_anchors(reference, index)
    successors, successor_counts = resolve_successors(anchors, confirmed, index)
    truth = build_truth(anchors, successors)

    scored = PROJECT_ROOT / PROCESSED / "linkage_candidates_scored.parquet"
    candidates_path = scored if scored.exists() else PROJECT_ROOT / PROCESSED / "linkage_candidates.parquet"
    exposure = build_exposure(truth, candidates_path)

    anchors.drop(columns=["evidence_notice_ids"]).to_parquet(
        output_dir / "regional_reference_anchors.parquet", index=False, compression="zstd"
    )
    exposure.to_parquet(output_dir / "exposure_full.parquet", index=False, compression="zstd")
    split_counts: dict[str, Any] = {}
    for split in ("dev", "validation"):
        part = truth.loc[truth["split"].eq(split), TRUTH_OUTPUT_COLUMNS]
        part.to_parquet(output_dir / f"benchmark_{split}.parquet", index=False, compression="zstd")
        usable = part.loc[part["anchor_verdict"].ne("ANCHOR_UNUSABLE")]
        split_counts[split] = {
            "anchors": int(len(part)),
            "usable_anchors": int(len(usable)),
            "positive_anchors": int(usable["has_successor_primary"].sum()),
            "negative_anchors": int((~usable["has_successor_primary"]).sum()),
            "censored_negatives": int(usable["negative_is_censored"].sum()),
        }

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": "regional_grand_ouest",
        "benchmark_schema": BENCHMARK_SCHEMA,
        "reference_version": REFERENCE_VERSION,
        "reference_source": str(REFERENCE_CSV),
        "confirmed_links_source": str(CONFIRMED_CSV),
        "candidate_source": str(candidates_path.relative_to(PROJECT_ROOT)),
        "label_provenance": (
            "LLM-generated labels, spot-checked by the project owner. The project "
            "owner supplied the raw anchor and candidate data; a single research "
            "pass by a large language model consulted the BOAMP notices and "
            "official notice URLs together with wider public sources, and proposed "
            "a successor or an abstention per anchor. The project owner then "
            "spot-checked a subset rather than verifying every anchor. No linkage "
            "method existed or was consulted when the labels were produced."
        ),
        "independent_of_linkage_algorithms": True,
        "independent_human_specialist_review": False,
        "review_date": "2026-08-11",
        "geographical_scope": "Grand Ouest (Bretagne, Pays de la Loire, Normandie)",
        "temporal_scope": {
            "anchor_award_dates": [
                str(reference["anchor_award_notice_date"].min()),
                str(reference["anchor_award_notice_date"].max()),
            ],
            "study_cutoff": STUDY_CUTOFF.isoformat(),
        },
        "reviewed_anchors": int(len(reference)),
        "remap": {
            "resolved_to_current_episodes": int((anchors["remap_status"] == "RESOLVED").sum()),
            **remap_counts,
            **successor_counts,
        },
        "splits": split_counts,
        "event_sets_supported": ["primary"],
        "candidate_reachability": candidate_reachability(truth, exposure),
        "known_limitations": [
            "Labels were generated by a single LLM research pass and spot-checked on a subset by the project owner, not verified anchor-by-anchor and not reviewed by an independent multi-reviewer human panel. Anchors outside the spot-checked subset carry the model's judgement as recorded.",
            "The research pass consulted sources beyond BOAMP, but the specific sources behind each individual label were not recorded, so the evidence trail for a given anchor cannot be fully reconstructed or independently re-executed.",
            "Negatives are corpus-relative: roughly 25 candidates per anchor were considered, not the full pool, so a negative means 'no successor identified among those considered, using the sources consulted'.",
            "Labels rest partly on judging whether later notice text continues an earlier need. That judgement is not rule-identical to any method scored here, but it draws on text, CPV, and date evidence that the text-ranking method also uses, so the two are not fully evidence-independent.",
            "The reference records one successor relationship per anchor and does not distinguish renewal from next-phase, so only the primary event set exists.",
            "Design weights come from the v1 sampling frame, whose stratum populations were computed on the earlier episode reconstruction.",
            "Sample size is small; every metric needs its interval read alongside it.",
        ],
    }
    (output_dir / "regional_benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    manifest = build(parse_args().force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
