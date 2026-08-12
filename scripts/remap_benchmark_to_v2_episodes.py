#!/usr/bin/env python3
"""Remap the manual benchmark's stale episode IDs onto the v2 episode layer.

The benchmark (data/reference/BOAMP_Internship_Reference_120.csv,
BOAMP_Internship_Evaluation_Subset.csv, BOAMP_Internship_Confirmed_Successor_Links.csv)
was built against a superseded Grand Ouest episode reconstruction. Its
``anchor_episode_id`` / ``successor_episode_id`` values do not exist in the
current data/processed/boamp_v2 episode layer.

The benchmark's ground truth is not stale, though: it was assigned by human
reviewers to notice IDs (idweb), and idweb is stable across pipeline
versions. This script re-derives the v2 episode_id for every anchor and
confirmed successor by joining on idweb through episode_membership.parquet,
rather than trusting any old episode_id. No prediction is made here; a
benchmark row whose anchor notices land in more than one v2 episode is
flagged, not silently resolved, because that disagreement is itself
evidence about linkage/episode quality worth reviewing by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PROCESSED_DIR = Path("data/processed/boamp_v2")
DEFAULT_REFERENCE_DIR = Path("data/reference")
DEFAULT_OUTPUT_DIR = Path("data/processed/boamp_v2/benchmark_remap")

IDWEB_URL_RE = re.compile(r"idweb:([\w-]+)")


def parse_json_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    return [str(item) for item in json.loads(text)]


def idweb_from_urls(value: Any) -> list[str]:
    urls = parse_json_list(value)
    return [match.group(1) for url in urls for match in [IDWEB_URL_RE.search(url)] if match]


def load_membership(processed_dir: Path) -> dict[str, str]:
    membership = pd.read_parquet(
        processed_dir / "episode_membership.parquet", columns=["idweb", "episode_id"]
    )
    return dict(zip(membership["idweb"], membership["episode_id"]))


def remap_notice_ids(notice_ids: list[str], idweb_to_episode: dict[str, str]) -> dict[str, Any]:
    found = {nid: idweb_to_episode[nid] for nid in notice_ids if nid in idweb_to_episode}
    missing = [nid for nid in notice_ids if nid not in idweb_to_episode]
    episode_ids = sorted(set(found.values()))
    if not notice_ids:
        status = "no_notice_ids"
    elif missing:
        status = "missing_notices" if not episode_ids else "partial_missing_notices"
    elif len(episode_ids) == 1:
        status = "single_episode"
    else:
        status = "split_across_episodes"
    return {
        "v2_episode_ids": episode_ids,
        "v2_episode_id": episode_ids[0] if len(episode_ids) == 1 else None,
        "v2_episode_id_count": len(episode_ids),
        "missing_notice_ids": missing,
        "remap_status": status,
    }


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def remap_reference(reference_dir: Path, idweb_to_episode: dict[str, str]) -> pd.DataFrame:
    reference = pd.read_csv(reference_dir / "BOAMP_Internship_Reference_120.csv")
    records = []
    for row in reference.to_dict(orient="records"):
        notice_ids = parse_json_list(row.get("anchor_notice_ids_json"))
        remap = remap_notice_ids(notice_ids, idweb_to_episode)
        records.append({
            "sample_id": row["sample_id"],
            "benchmark_split": row["benchmark_split"],
            "primary_evaluation_eligible": row.get("primary_evaluation_eligible"),
            "final_outcome": row.get("final_outcome"),
            "anchor_episode_id_v1_stale": row.get("anchor_episode_id"),
            "anchor_notice_ids_json": json_compact(notice_ids),
            "anchor_buyer_name": row.get("anchor_buyer_name"),
            "anchor_theme": row.get("anchor_theme"),
            "anchor_v2_episode_id": remap["v2_episode_id"],
            "anchor_v2_episode_ids_json": json_compact(remap["v2_episode_ids"]),
            "anchor_v2_episode_id_count": remap["v2_episode_id_count"],
            "anchor_missing_notice_ids_json": json_compact(remap["missing_notice_ids"]),
            "anchor_remap_status": remap["remap_status"],
        })
    return pd.DataFrame.from_records(records)


def remap_confirmed_successors(
    reference_dir: Path,
    idweb_to_episode: dict[str, str],
    anchor_remap: pd.DataFrame,
) -> pd.DataFrame:
    links = pd.read_csv(reference_dir / "BOAMP_Internship_Confirmed_Successor_Links.csv")
    anchor_lookup = anchor_remap.set_index("sample_id")
    records = []
    for row in links.to_dict(orient="records"):
        sample_id = row["sample_id"]
        anchor_row = anchor_lookup.loc[sample_id] if sample_id in anchor_lookup.index else None
        successor_notice_ids = idweb_from_urls(row.get("candidate_urls_json"))
        successor_remap = remap_notice_ids(successor_notice_ids, idweb_to_episode)
        records.append({
            "sample_id": sample_id,
            "anchor_episode_id_v1_stale": row.get("anchor_episode_id"),
            "successor_episode_id_v1_stale": row.get("successor_episode_id"),
            "final_outcome": row.get("final_outcome"),
            "final_confidence": row.get("final_confidence"),
            "publication_gap_days": row.get("publication_gap_days"),
            "anchor_buyer_name": row.get("anchor_buyer_name"),
            "candidate_buyer_name": row.get("candidate_buyer_name"),
            "anchor_v2_episode_id": anchor_row["anchor_v2_episode_id"] if anchor_row is not None else None,
            "anchor_v2_episode_ids_json": anchor_row["anchor_v2_episode_ids_json"] if anchor_row is not None else json_compact([]),
            "anchor_remap_status": anchor_row["anchor_remap_status"] if anchor_row is not None else "sample_id_not_in_reference_120",
            "successor_notice_ids_json": json_compact(successor_notice_ids),
            "successor_v2_episode_id": successor_remap["v2_episode_id"],
            "successor_v2_episode_ids_json": json_compact(successor_remap["v2_episode_ids"]),
            "successor_v2_episode_id_count": successor_remap["v2_episode_id_count"],
            "successor_missing_notice_ids_json": json_compact(successor_remap["missing_notice_ids"]),
            "successor_remap_status": successor_remap["remap_status"],
        })
    return pd.DataFrame.from_records(records)


def remap_evaluation_subset(
    reference_dir: Path,
    anchor_remap: pd.DataFrame,
    successor_remap: pd.DataFrame,
) -> pd.DataFrame:
    evaluation = pd.read_csv(reference_dir / "BOAMP_Internship_Evaluation_Subset.csv")
    anchor_lookup = anchor_remap.set_index("sample_id")
    successor_by_sample: dict[str, list[str]] = {}
    for sample_id, group in successor_remap.groupby("sample_id"):
        episode_ids: set[str] = set()
        for ids_json in group["successor_v2_episode_ids_json"]:
            episode_ids.update(json.loads(ids_json))
        successor_by_sample[sample_id] = sorted(episode_ids)

    records = []
    for row in evaluation.to_dict(orient="records"):
        sample_id = row["sample_id"]
        anchor_row = anchor_lookup.loc[sample_id] if sample_id in anchor_lookup.index else None
        successor_v2_ids = successor_by_sample.get(sample_id, [])
        records.append({
            "sample_id": sample_id,
            "benchmark_split": row["benchmark_split"],
            "final_outcome": row.get("final_outcome"),
            "anchor_episode_id_v1_stale": row.get("anchor_episode_id"),
            "final_successor_episode_ids_v1_stale": row.get("final_successor_episode_ids"),
            "anchor_v2_episode_id": anchor_row["anchor_v2_episode_id"] if anchor_row is not None else None,
            "anchor_v2_episode_ids_json": anchor_row["anchor_v2_episode_ids_json"] if anchor_row is not None else json_compact([]),
            "anchor_remap_status": anchor_row["anchor_remap_status"] if anchor_row is not None else "sample_id_not_in_reference_120",
            "successor_v2_episode_ids_json": json_compact(successor_v2_ids),
            "successor_v2_episode_id_count": len(successor_v2_ids),
        })
    return pd.DataFrame.from_records(records)


def summarize(anchor_remap: pd.DataFrame, successor_remap: pd.DataFrame, evaluation_remap: pd.DataFrame) -> dict[str, Any]:
    anchor_status = anchor_remap["anchor_remap_status"].value_counts().to_dict()
    successor_status = successor_remap["successor_remap_status"].value_counts().to_dict()
    outcome_consistency = evaluation_remap.loc[
        evaluation_remap["final_outcome"].isin(["OBSERVED_SUCCESSOR", "MULTIPLE_SUCCESSORS"])
    ]
    outcome_without_v2_successor = int((outcome_consistency["successor_v2_episode_id_count"] == 0).sum())
    return {
        "anchor_rows": int(len(anchor_remap)),
        "anchor_remap_status_counts": {str(k): int(v) for k, v in anchor_status.items()},
        "confirmed_successor_rows": int(len(successor_remap)),
        "successor_remap_status_counts": {str(k): int(v) for k, v in successor_status.items()},
        "evaluation_rows": int(len(evaluation_remap)),
        "evaluation_rows_with_observed_successor_but_no_v2_successor": outcome_without_v2_successor,
        "all_anchor_notice_ids_resolved": bool(
            anchor_status.get("single_episode", 0) + anchor_status.get("split_across_episodes", 0)
            == len(anchor_remap)
        ),
    }


def build(project_root: Path, processed_dir: Path, reference_dir: Path, output_dir: Path) -> dict[str, Any]:
    idweb_to_episode = load_membership(processed_dir)

    anchor_remap = remap_reference(reference_dir, idweb_to_episode)
    successor_remap = remap_confirmed_successors(reference_dir, idweb_to_episode, anchor_remap)
    evaluation_remap = remap_evaluation_subset(reference_dir, anchor_remap, successor_remap)

    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_remap.to_csv(output_dir / "reference_120_v2_remap.csv", index=False)
    successor_remap.to_csv(output_dir / "confirmed_successor_links_v2_remap.csv", index=False)
    evaluation_remap.to_csv(output_dir / "evaluation_subset_v2_remap.csv", index=False)

    summary = summarize(anchor_remap, successor_remap, evaluation_remap)
    summary["created_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    summary["episode_membership_source"] = str(processed_dir / "episode_membership.parquet")
    summary["outputs"] = {
        "anchor_remap": str(output_dir / "reference_120_v2_remap.csv"),
        "successor_remap": str(output_dir / "confirmed_successor_links_v2_remap.csv"),
        "evaluation_remap": str(output_dir / "evaluation_subset_v2_remap.csv"),
    }
    (output_dir / "benchmark_v2_remap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    processed_dir = resolve(args.processed_dir, project_root)
    reference_dir = resolve(args.reference_dir, project_root)
    output_dir = resolve(args.output_dir, project_root)

    summary = build(project_root, processed_dir, reference_dir, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_anchor_notice_ids_resolved"]:
        raise RuntimeError(
            "Some benchmark anchors have notice IDs missing from episode_membership.parquet; "
            "investigate before using the remap for candidate generation."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
