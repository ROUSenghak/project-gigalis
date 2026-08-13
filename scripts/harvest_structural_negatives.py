#!/usr/bin/env python3
"""Harvest pairs that are provably not renewals, without spending annotation.

Some non-renewals can be established from structure alone, and they are the
ones that matter most: pairs with high text similarity that a text-ranking
method will happily accept. Because they need no human judgement, they can be
generated in bulk and scored as a separate suite -- one that penalises a
pure-text method without any annotator involvement, which is a useful
counterweight given that the annotator here is a language model reading the
same text the methods read.

Five sources, each carrying how strongly it is established:

``SAME_PROCUREMENT``            two episodes of one procurement. Definitional.
``RETENDER_AFTER_FAILURE``      a procedure relaunched after being declared
                                unsuccessful. Same need, but the first
                                procedure never produced a contract.
``PARALLEL_LOT``                concurrent lots of one programme.
``SAME_BUYER_DIFFERENT_DOMAIN`` the confusion a buyer creates by publishing
                                unrelated contracts while one is running.
``HOMONYM_BUYER``               identically-named buyers in different
                                departments with conflicting SIRENs.

.. warning::
   ``PARALLEL_LOT`` must not be selected on a short gap. The frozen policy
   already discards anything under 90 days, so a suite built from sub-90-day
   pairs would hand that policy a free hundred percent. Gap is recorded as an
   attribute and gate G4 requires a substantial share of the suite to sit
   *outside* the window the policy filters on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.linkage import (  # noqa: E402
    MIN_GAP_DAYS,
    cpv_divisions,
    jaccard,
    parse_json_list,
)
from boamp_pipeline.renewal_language import RETENDER_AFTER_FAILURE  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp_v2/benchmark_v3")
NEGATIVES_VERSION = "boamp_structural_negatives_v3.0"

SAME_PROCUREMENT = "SAME_PROCUREMENT"
PARALLEL_LOT = "PARALLEL_LOT"
SAME_BUYER_DIFFERENT_DOMAIN = "SAME_BUYER_DIFFERENT_DOMAIN"
HOMONYM_BUYER = "HOMONYM_BUYER"

#: How similar two objects must look before a pair is worth keeping as a hard
#: negative. Below this the pair is easy and teaches nothing.
HARD_NEGATIVE_MIN_TEXT = 0.35

#: A different-domain pair must be genuinely unrelated.
DIFFERENT_DOMAIN_MAX_TEXT = 0.10

#: Where renewals actually concentrate, so a different-domain pair sitting in
#: this band is the most confusable kind.
RENEWAL_BAND = (1095, 1825)

#: Gate G4: the parallel-lot suite must not simply restate the 90-day floor.
MIN_SHARE_OUTSIDE_GAP_FLOOR = 0.15


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "harvest_structural_negatives.log", encoding="utf-8"),
        ],
    )


def load_texts(processed_dir: Path, needed: set[str]) -> dict[str, str]:
    """Episode text for exactly the episodes involved, read in batches."""
    import pyarrow.parquet as pq

    texts: dict[str, str] = {}
    parquet = pq.ParquetFile(processed_dir / "episodes_2015_2025.parquet")
    for batch in parquet.iter_batches(batch_size=50_000, columns=["episode_id", "episode_text"]):
        rows = batch.to_pydict()
        for episode_id, text in zip(rows["episode_id"], rows["episode_text"]):
            if episode_id in needed:
                texts[episode_id] = text or ""
    return texts


def rejected_edge_pairs(processed_dir: Path, membership: dict[str, str]) -> list[dict[str, Any]]:
    """Episode pairs an edge tried to merge but a buyer conflict refused.

    These are the only cases where an explicit same-procurement link survives
    across two distinct episodes: everywhere else the episode layer already
    absorbed the link, which is why the explicit-link signal is a negative
    constraint rather than a renewal signal.
    """
    edges = pd.read_parquet(
        processed_dir / "episode_edges.parquet",
        columns=["left", "right", "method", "accepted", "decision_reason"],
    )
    rejected = edges.loc[~edges["accepted"].astype(bool)]
    pairs: list[dict[str, Any]] = []
    for row in rejected.itertuples(index=False):
        left = membership.get(str(row.left))
        right = membership.get(str(row.right))
        if left and right and left != right:
            pairs.append({
                "anchor_episode_id": left,
                "candidate_episode_id": right,
                "negative_source": SAME_PROCUREMENT,
                "verification": "DEFINITIONAL",
                "evidence": f"{row.method}:{row.decision_reason}",
            })
    return pairs


def harvest_from_exposure(exposure: pd.DataFrame, index: pd.DataFrame,
                          declarations: pd.DataFrame) -> list[dict[str, Any]]:
    """Hard negatives found among pairs already scored for the benchmark."""
    by_episode = index.set_index("episode_id")
    retender_episodes = set(
        declarations.loc[
            declarations["pattern_class"].eq(RETENDER_AFTER_FAILURE)
            & declarations["excluded_reason"].fillna("").eq(""),
            "episode_id",
        ].dropna()
    )

    rows: list[dict[str, Any]] = []
    for pair in exposure.itertuples(index=False):
        anchor_id = pair.anchor_episode_id
        candidate_id = pair.candidate_episode_id
        if anchor_id not in by_episode.index or candidate_id not in by_episode.index:
            continue
        anchor = by_episode.loc[anchor_id]
        candidate = by_episode.loc[candidate_id]
        text = float(pair.text_component or 0.0)
        gap = int(pair.gap_days)

        anchor_references = set(parse_json_list(anchor["procedure_references_json"]))
        candidate_references = set(parse_json_list(candidate["procedure_references_json"]))
        shared_reference = bool(anchor_references & candidate_references - {""})

        anchor_divisions = cpv_divisions(parse_json_list(anchor["all_cpvs_json"]))
        candidate_divisions = cpv_divisions(parse_json_list(candidate["all_cpvs_json"]))

        # A shared, identical procedure reference means one procedure split into
        # lots, not a succession. Gap is recorded, never used to select.
        if shared_reference and text >= HARD_NEGATIVE_MIN_TEXT:
            rows.append({
                "anchor_episode_id": anchor_id,
                "candidate_episode_id": candidate_id,
                "negative_source": PARALLEL_LOT,
                "verification": "STRUCTURAL",
                "evidence": "identical procedure reference on both sides",
                "gap_days": gap,
                "text_component": text,
            })
            continue

        if candidate_id in retender_episodes and gap <= 365 and text >= 0.5:
            rows.append({
                "anchor_episode_id": anchor_id,
                "candidate_episode_id": candidate_id,
                "negative_source": RETENDER_AFTER_FAILURE,
                "verification": "DECLARED",
                "evidence": "candidate declares a relaunch after an unsuccessful procedure",
                "gap_days": gap,
                "text_component": text,
            })
            continue

        # The confusion the market itself creates: a buyer publishing unrelated
        # contracts while an existing one runs, right where renewals cluster.
        if (
            RENEWAL_BAND[0] <= gap <= RENEWAL_BAND[1]
            and text <= DIFFERENT_DOMAIN_MAX_TEXT
            and anchor_divisions
            and candidate_divisions
            and not (anchor_divisions & candidate_divisions)
        ):
            rows.append({
                "anchor_episode_id": anchor_id,
                "candidate_episode_id": candidate_id,
                "negative_source": SAME_BUYER_DIFFERENT_DOMAIN,
                "verification": "STRUCTURAL",
                "evidence": "disjoint CPV divisions and unrelated object, in the renewal band",
                "gap_days": gap,
                "text_component": text,
            })
    return rows


def harvest_parallel_lots(index: pd.DataFrame, texts: dict[str, str],
                          limit: int) -> list[dict[str, Any]]:
    """Concurrent lots of one programme, harvested from the index not the pool.

    The candidate pool starts 90 days after the award, so concurrent lots are
    invisible there by construction -- which is why searching the exposure for
    them returned nothing. They have to be found in the index directly.

    Selection is on shared reference stem plus near-identical objects. The gap
    is recorded but never used to select, because selecting on a short gap
    would restate the frozen 90-day floor and hand any method that filters
    there a free hundred percent.
    """
    lots: list[dict[str, Any]] = []
    usable = index.loc[index["buyer_block_key"].fillna("").ne("")]
    for _, group in usable.groupby("buyer_block_key", sort=False):
        if len(group) < 2 or len(group) > 60:
            continue
        rows = group.to_dict(orient="records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                left, right = rows[i], rows[j]
                left_text = token_set(texts.get(left["episode_id"], ""))
                right_text = token_set(texts.get(right["episode_id"], ""))
                if not left_text or not right_text:
                    continue
                similarity = jaccard(left_text, right_text)
                if similarity < 0.6:
                    continue
                left_lot = "lot" in texts.get(left["episode_id"], "").lower()
                right_lot = "lot" in texts.get(right["episode_id"], "").lower()
                if not (left_lot and right_lot):
                    continue
                gap = None
                if left["episode_origin_date"] and right["episode_origin_date"]:
                    gap = abs((right["episode_origin_date"] - left["episode_origin_date"]).days)
                lots.append({
                    "anchor_episode_id": left["episode_id"],
                    "candidate_episode_id": right["episode_id"],
                    "negative_source": PARALLEL_LOT,
                    "verification": "STRUCTURAL",
                    "evidence": "near-identical object, both naming lots, same buyer",
                    "gap_days": gap,
                    "text_component": round(similarity, 4),
                })
                if len(lots) >= limit:
                    return lots
    return lots


def token_set(text: Any) -> set[str]:
    from boamp_pipeline.linkage import normalize_text

    return {token for token in normalize_text(text).split() if len(token) >= 4}


def harvest_homonyms(index: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    """Identically-named buyers in different departments with different SIRENs.

    National blocking on names alone would merge these. They are kept as a
    suite so the cost of the alternative blocking rule is measurable rather
    than asserted.
    """
    usable = index.loc[
        index["buyer_name_blocking"].fillna("").ne("")
        & index["buyer_siren"].fillna("").ne("")
        & index["buyer_department"].fillna("").ne("UNRESOLVED")
    ]
    rows: list[dict[str, Any]] = []
    for name, group in usable.groupby("buyer_name_blocking", sort=False):
        if group["buyer_department"].nunique() < 2 or group["buyer_siren"].nunique() < 2:
            continue
        representatives = group.drop_duplicates("buyer_siren").head(2)
        if len(representatives) < 2:
            continue
        left, right = representatives.iloc[0], representatives.iloc[1]
        if left["buyer_department"] == right["buyer_department"]:
            continue
        rows.append({
            "anchor_episode_id": left["episode_id"],
            "candidate_episode_id": right["episode_id"],
            "negative_source": HOMONYM_BUYER,
            "verification": "STRUCTURAL",
            "evidence": (
                f"buyer name '{name}' in departments {left['buyer_department']} and "
                f"{right['buyer_department']} with SIRENs {left['buyer_siren']} "
                f"and {right['buyer_siren']}"
            ),
            "gap_days": None,
            "text_component": None,
        })
        if len(rows) >= limit:
            break
    return rows


def build(project_root: Path, output_dir: Path, force: bool, homonym_limit: int,
          parallel_lot_limit: int = 800) -> dict[str, Any]:
    negatives_path = output_dir / "structural_negatives.parquet"
    summary_path = output_dir / "structural_negatives_summary.json"
    if negatives_path.exists() and not force:
        raise FileExistsError(f"{negatives_path} already exists. Use --force to rebuild.")

    processed_dir = output_dir.parent
    membership = pd.read_parquet(
        processed_dir / "episode_membership.parquet", columns=["idweb", "episode_id"]
    )
    membership_map = dict(zip(membership["idweb"], membership["episode_id"]))
    index = pd.read_parquet(
        output_dir / "national_episode_index.parquet",
        columns=[
            "episode_id", "buyer_name_blocking", "buyer_siren", "buyer_department",
            "procedure_references_json", "all_cpvs_json", "digital_flag",
            "buyer_block_key", "episode_origin_date",
        ],
    )
    exposure = pd.read_parquet(
        output_dir / "exposure_full.parquet",
        columns=["anchor_episode_id", "candidate_episode_id", "text_component", "gap_days"],
    )
    declarations = pd.read_parquet(
        output_dir / "renewal_declarations.parquet",
        columns=["episode_id", "pattern_class", "excluded_reason"],
    )
    logging.info("Exposure pairs: %s", f"{len(exposure):,}")

    rows: list[dict[str, Any]] = []
    same_procurement = rejected_edge_pairs(processed_dir, membership_map)
    logging.info("Same-procurement pairs from rejected edges: %s", f"{len(same_procurement):,}")
    rows.extend(same_procurement)
    rows.extend(harvest_from_exposure(exposure, index, declarations))
    rows.extend(harvest_homonyms(index, homonym_limit))

    # Parallel lots live inside the 90-day floor, so they are invisible in the
    # candidate pool and must be found in the index. Text is loaded once for
    # every episode any source touched, so each suite reports how hard it is.
    blocks_needed = set(index.loc[index["buyer_block_key"].fillna("").ne(""), "buyer_block_key"])
    lot_index = index.loc[index["buyer_block_key"].isin(blocks_needed)]
    needed = set(lot_index["episode_id"]) | {r["anchor_episode_id"] for r in rows} | {
        r["candidate_episode_id"] for r in rows
    }
    logging.info("Loading text for %s episodes", f"{len(needed):,}")
    texts = load_texts(processed_dir, needed)
    rows.extend(harvest_parallel_lots(lot_index, texts, parallel_lot_limit))

    negatives = pd.DataFrame(rows)
    if not negatives.empty:
        negatives = negatives.drop_duplicates(["anchor_episode_id", "candidate_episode_id"])
        # Difficulty is what makes a negative worth having: a suite of easy
        # pairs cannot penalise a text-ranking method.
        missing = negatives["text_component"].isna()
        negatives.loc[missing, "text_component"] = [
            round(jaccard(token_set(texts.get(a, "")), token_set(texts.get(c, ""))), 4)
            for a, c in zip(
                negatives.loc[missing, "anchor_episode_id"],
                negatives.loc[missing, "candidate_episode_id"],
            )
        ]
        negatives["negatives_version"] = NEGATIVES_VERSION
    negatives.to_parquet(negatives_path, index=False, compression="zstd")

    by_source = (
        {str(k): int(v) for k, v in negatives["negative_source"].value_counts().items()}
        if len(negatives) else {}
    )
    parallel = negatives.loc[negatives["negative_source"].eq(PARALLEL_LOT)] if len(negatives) else negatives
    outside_floor = (
        float(pd.to_numeric(parallel["gap_days"], errors="coerce").ge(MIN_GAP_DAYS).mean())
        if len(parallel) else 0.0
    )
    gate_g4 = bool(len(parallel) == 0 or outside_floor >= MIN_SHARE_OUTSIDE_GAP_FLOOR)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "negatives_version": NEGATIVES_VERSION,
        "output_file": str(negatives_path),
        "role": (
            "A separately-scored hard-negative suite. It has no inclusion probability "
            "and is never pooled into the national estimate."
        ),
        "pairs": int(len(negatives)),
        "pairs_by_source": by_source,
        "pairs_by_verification": (
            {str(k): int(v) for k, v in negatives["verification"].value_counts().items()}
            if len(negatives) else {}
        ),
        "text_similarity_by_source": (
            {
                str(source): {
                    "n": int(len(group)),
                    "median": round(float(pd.to_numeric(group["text_component"],
                                                        errors="coerce").median()), 4),
                    "p90": round(float(pd.to_numeric(group["text_component"],
                                                     errors="coerce").quantile(0.9)), 4),
                }
                for source, group in negatives.groupby("negative_source")
            } if len(negatives) else {}
        ),
        "difficulty_note": (
            "A hard negative is one a text-ranking method would accept. "
            "SAME_PROCUREMENT and PARALLEL_LOT are the difficult suites; "
            "SAME_BUYER_DIFFERENT_DOMAIN is deliberately easy and acts as a control."
        ),
        "gate_g4_parallel_lot_not_circular": {
            "rule": (
                "The parallel-lot suite must not simply restate the frozen 90-day gap "
                "floor, or any method filtering there would score perfectly for free."
            ),
            "share_with_gap_at_or_above_floor": round(outside_floor, 4),
            "threshold": MIN_SHARE_OUTSIDE_GAP_FLOOR,
            "passed": gate_g4,
        },
        "validation_passed": bool(len(negatives) > 0 and gate_g4),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--homonym-limit", type=int, default=400)
    parser.add_argument("--parallel-lot-limit", type=int, default=800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    configure_logging(project_root)
    summary = build(project_root, output_dir, args.force, args.homonym_limit,
                    args.parallel_lot_limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["validation_passed"]:
        raise RuntimeError("Structural negative harvesting validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
