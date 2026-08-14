#!/usr/bin/env python3
"""Decide what an annotator sees for each anchor, and prove it favours nobody.

This is the stage that replaces v1's fatal defect. There, reviewers saw the top
25 candidates of a single ranker drawn from pools with a median size of 146;
77 of 94 eligible anchors were truncated, so a successor that ranker buried was
never on screen and its anchor was recorded as having none.

Here, seven retrievers each contribute their own top-k and the union is shown.
Three of the seven consult no text at all, which matters because the incumbent
method is a text ranker. Whatever every retriever ignores is still sampled --
uniformly within gap strata, at a recorded probability -- so "what did they all
miss?" becomes an estimate with an interval instead of an assumption.

Every model score is computed and stored here, and none of it reaches the
annotator: :func:`boamp_pipeline.exposure.blind_view` filters through an
allow-list, candidates are shuffled per pass, and slot identifiers are assigned
after shuffling. The claim that order carries no ranking is then *tested*
rather than asserted, by correlating slot position against the incumbent score.

The stored scores are also what the evaluation stage needs later, so this file
doubles as the national candidate scoring for the benchmark anchors.
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

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.exposure import (  # noqa: E402
    EXPOSURE_SEED,
    MAX_PER_RETRIEVER,
    RANDOM_TAIL,
    RETRIEVERS,
    SAMPLED_SLOTS,
    TAIL_FLOOR,
    assign_slots,
    exposure_mode,
    gap_bucket,
    interleave,
    pool_for_anchor,
    pool_size_bucket,
    presentation_order_correlation,
    stable_seed,
    stratified_tail_sample,
)
from boamp_pipeline.linkage import (  # noqa: E402
    DEFAULT_WEIGHTS,
    MAX_GAP_DAYS,
    MIN_GAP_DAYS,
    buyer_score,
    buyer_similarity,
    classify_buyer_match,
    cpv_continuity_score,
    evidence_completeness,
    normalize_text,
    parse_json_list,
    tfidf_similarities,
    time_plausibility_score,
    weighted_score,
)

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp/benchmark")
EXPOSURE_VERSION = "boamp_benchmark_exposure_v3.0"

#: Ideal gap for the CPV+time retriever, in days. Four years is the French
#: framework ceiling and the measured mode of declared renewals.
IDEAL_GAP_DAYS = 1461

INDEX_COLUMNS = [
    "episode_id", "episode_origin_date", "award_date", "buyer_block_key",
    "buyer_name_raw", "buyer_name_blocking", "buyer_siren", "buyer_department",
    "dept_group", "digital_flag", "digital_segment", "main_cpv", "all_cpvs_json",
    "expected_end_date", "expected_end_source", "duration_months_reliable",
    "framework_flag", "procedure_references_json", "titulaire_names_json",
    "notice_count",
]


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "build_benchmark_exposure.log", encoding="utf-8"),
        ],
    )


def load_texts(processed_dir: Path, needed: set[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    parquet = pq.ParquetFile(processed_dir / "episodes_2015_2025.parquet")
    for batch in parquet.iter_batches(batch_size=50_000, columns=["episode_id", "episode_text"]):
        rows = batch.to_pydict()
        for episode_id, text in zip(rows["episode_id"], rows["episode_text"]):
            if episode_id in needed:
                texts[episode_id] = text or ""
    return texts


def reference_stems(value: Any) -> set[str]:
    from scripts.resolve_declared_predecessors import reference_stem

    stems = {reference_stem(item) for item in parse_json_list(value)}
    stems.discard("")
    return stems


def score_pool(anchor: Any, pool: list[dict[str, Any]], texts: dict[str, str]) -> pd.DataFrame:
    """Compute every comparison component for one anchor's whole pool.

    Scored exhaustively, not only for exposed candidates, because the retrievers
    rank against the full pool and the tail estimator needs the pool defined
    independently of what was shown.
    """
    if not pool:
        return pd.DataFrame()

    anchor_text = normalize_text(texts.get(anchor.episode_id, ""))
    candidate_texts = [normalize_text(texts.get(row["episode_id"], "")) for row in pool]
    word = tfidf_similarities(anchor_text, candidate_texts, analyzer="word")
    char = tfidf_similarities(anchor_text, candidate_texts, analyzer="char")
    text_component = np.maximum(word, char)

    anchor_cpv = parse_json_list(anchor.all_cpvs_json)
    anchor_siren = str(anchor.buyer_siren or "").strip()
    anchor_name = str(anchor.buyer_name_blocking or "")
    anchor_suppliers = set(parse_json_list(anchor.titulaire_names_json))
    anchor_stems = reference_stems(anchor.procedure_references_json)
    expected_duration = (
        float(anchor.duration_months_reliable)
        if pd.notna(anchor.duration_months_reliable) else None
    )

    rows: list[dict[str, Any]] = []
    for position, candidate in enumerate(pool):
        gap = int(candidate["gap_days"])
        match_type = classify_buyer_match(
            anchor_siren, str(candidate.get("buyer_siren") or "").strip(),
            anchor_name, str(candidate.get("buyer_name_blocking") or ""),
        )
        cpv = cpv_continuity_score(anchor_cpv, parse_json_list(candidate["all_cpvs_json"]))
        timing = time_plausibility_score(gap, expected_duration)
        components = {
            "buyer": buyer_score(match_type),
            "text": float(text_component[position]),
            "cpv": cpv,
            "time": timing,
        }
        suppliers = set(parse_json_list(candidate["titulaire_names_json"]))
        stems = reference_stems(candidate["procedure_references_json"])
        rows.append({
            "candidate_episode_id": candidate["episode_id"],
            "candidate_origin_date": candidate["episode_origin_date"],
            "candidate_award_date": candidate["award_date"],
            "candidate_digital_segment": candidate["digital_segment"],
            "candidate_main_cpv": candidate["main_cpv"],
            "gap_days": gap,
            "gap_bucket": gap_bucket(gap),
            "buyer_match_type": match_type,
            "buyer_name_similarity": buyer_similarity(
                anchor_name, str(candidate.get("buyer_name_blocking") or "")
            ),
            "word_tfidf_similarity": float(word[position]),
            "char_tfidf_similarity": float(char[position]),
            "buyer_component": components["buyer"],
            "text_component": components["text"],
            "cpv_component": cpv,
            "time_component": timing,
            "evidence_components": evidence_completeness(components),
            "linkage_score": weighted_score(components, DEFAULT_WEIGHTS),
            "shared_supplier": bool(anchor_suppliers & suppliers),
            "shared_reference_stem": bool(anchor_stems & stems),
            "days_from_expected_end": (
                (candidate["episode_origin_date"] - anchor.expected_end_date).days
                if anchor.expected_end_date is not None
                and candidate["episode_origin_date"] is not None
                else None
            ),
        })
    return pd.DataFrame(rows)


def rank_retrievers(scored: pd.DataFrame) -> dict[str, list[str]]:
    """Each retriever's own ordered shortlist over the pool.

    The three text-free retrievers are what keep the union from being a
    restatement of the method under evaluation.
    """
    ranked: dict[str, list[str]] = {}
    by_name = {retriever.name: retriever for retriever in RETRIEVERS}

    def top(frame: pd.DataFrame, name: str) -> list[str]:
        return frame["candidate_episode_id"].head(by_name[name].top_k).tolist()

    ranked["R_TFIDF_WORD"] = top(
        scored.sort_values("word_tfidf_similarity", ascending=False), "R_TFIDF_WORD"
    )
    ranked["R_TFIDF_CHAR"] = top(
        scored.sort_values("char_tfidf_similarity", ascending=False), "R_TFIDF_CHAR"
    )
    # Text-free: functional continuity, then nearness to the four-year mode.
    cpv_time = scored.assign(
        _cpv=scored["cpv_component"].fillna(-1.0),
        _gap=(scored["gap_days"] - IDEAL_GAP_DAYS).abs(),
    ).sort_values(["_cpv", "_gap"], ascending=[False, True])
    ranked["R_CPV_TIME"] = top(cpv_time, "R_CPV_TIME")

    # Text-free: candidates closest to the anchor's own expected end.
    expiry = scored.loc[scored["days_from_expected_end"].notna()]
    if len(expiry):
        expiry = expiry.assign(_d=expiry["days_from_expected_end"].abs()).sort_values("_d")
        ranked["R_BUYER_EXPIRY_WINDOW"] = top(expiry, "R_BUYER_EXPIRY_WINDOW")
    else:
        ranked["R_BUYER_EXPIRY_WINDOW"] = []

    # Text-free: the incumbent supplier won again.
    incumbent = scored.loc[scored["shared_supplier"]].sort_values("gap_days")
    ranked["R_INCUMBENT_SUPPLIER"] = top(incumbent, "R_INCUMBENT_SUPPLIER")

    # Text-free: same reference stem, different year.
    stem = scored.loc[scored["shared_reference_stem"]].sort_values("gap_days")
    ranked["R_REFERENCE_STEM"] = top(stem, "R_REFERENCE_STEM")
    return ranked


def build(project_root: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    exposure_path = output_dir / "exposure_full.parquet"
    summary_path = output_dir / "exposure_summary.json"
    pool_definition_path = output_dir / "pool_definition.json"
    if exposure_path.exists() and not force:
        raise FileExistsError(f"{exposure_path} already exists. Use --force to rebuild.")

    anchors = pd.read_parquet(output_dir / "anchors.parquet")
    index = pd.read_parquet(output_dir / "national_episode_index.parquet", columns=INDEX_COLUMNS)
    logging.info("Anchors: %s | index: %s", f"{len(anchors):,}", f"{len(index):,}")

    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index.to_dict(orient="records"):
        key = row.get("buyer_block_key") or ""
        if key:
            blocks[key].append(row)

    # Pools first, so text is joined for exactly the episodes involved.
    pools: dict[str, list[dict[str, Any]]] = {}
    needed: set[str] = set(anchors["episode_id"])
    for anchor in anchors.itertuples(index=False):
        pool = pool_for_anchor(
            anchor.award_date, anchor.episode_id, blocks.get(anchor.buyer_block_key, [])
        )
        pools[anchor.episode_id] = pool
        needed.update(row["episode_id"] for row in pool)
    logging.info("Episodes needing text: %s", f"{len(needed):,}")
    texts = load_texts(output_dir.parent, needed)

    frames: list[pd.DataFrame] = []
    mode_counts: dict[str, int] = defaultdict(int)
    retriever_exposed: dict[str, int] = defaultdict(int)
    tail_probabilities_seen: list[float] = []
    anchors_without_pool = 0

    for position, anchor in enumerate(anchors.itertuples(index=False), start=1):
        pool = pools[anchor.episode_id]
        if not pool:
            anchors_without_pool += 1
            mode_counts["EMPTY_POOL"] += 1
            continue

        scored = score_pool(anchor, pool, texts)
        mode = exposure_mode(len(pool))
        mode_counts[mode] += 1

        if mode in {"EXHAUSTIVE", "NEAR_EXHAUSTIVE"}:
            exposed_ids = scored["candidate_episode_id"].tolist()
            provenance: dict[str, list[str]] = {cid: ["R_FULL_POOL"] for cid in exposed_ids}
            tail_records: list[dict[str, Any]] = []
            tail_probability: dict[str, float] = {}
        else:
            ranked = rank_retrievers(scored)
            union_limit = SAMPLED_SLOTS - TAIL_FLOOR
            union_ids, provenance = interleave(ranked, union_limit, MAX_PER_RETRIEVER)
            rng = np.random.default_rng(stable_seed("tail", anchor.episode_id))
            tail_records, tail_probability = stratified_tail_sample(
                scored.to_dict(orient="records"),
                set(union_ids),
                SAMPLED_SLOTS - len(union_ids),
                rng,
            )
            for record in tail_records:
                provenance.setdefault(record["candidate_episode_id"], []).append(RANDOM_TAIL)
            exposed_ids = union_ids + [r["candidate_episode_id"] for r in tail_records]
            tail_probabilities_seen.extend(tail_probability.values())

        tail_lookup = {r["candidate_episode_id"]: r for r in tail_records}
        slots_a = dict(
            (candidate, slot) for slot, candidate in assign_slots(exposed_ids, anchor.episode_id, "A")
        )
        slots_b = dict(
            (candidate, slot) for slot, candidate in assign_slots(exposed_ids, anchor.episode_id, "B")
        )

        exposed = scored.loc[scored["candidate_episode_id"].isin(set(exposed_ids))].copy()
        exposed["anchor_episode_id"] = anchor.episode_id
        exposed["frame"] = anchor.frame
        exposed["wave"] = anchor.wave
        exposed["stratum_id"] = anchor.stratum_id
        exposed["pool_size"] = len(pool)
        exposed["exposure_mode"] = mode
        exposed["retrievers_json"] = exposed["candidate_episode_id"].map(
            lambda cid: json.dumps(sorted(provenance.get(cid, [])), ensure_ascii=False)
        )
        exposed["is_random_tail"] = exposed["candidate_episode_id"].isin(tail_lookup)
        exposed["tail_stratum"] = exposed["candidate_episode_id"].map(
            lambda cid: tail_lookup.get(cid, {}).get("tail_stratum", "")
        )
        exposed["tail_inclusion_probability"] = exposed["candidate_episode_id"].map(
            lambda cid: tail_lookup.get(cid, {}).get("tail_inclusion_probability", np.nan)
        )
        exposed["slot_id_pass_A"] = exposed["candidate_episode_id"].map(slots_a)
        exposed["slot_id_pass_B"] = exposed["candidate_episode_id"].map(slots_b)
        exposed["exposure_version"] = EXPOSURE_VERSION
        frames.append(exposed)

        for cid, names in provenance.items():
            for name in names:
                retriever_exposed[name] += 1

        if position % 100 == 0:
            logging.info("Exposed %s / %s anchors", position, len(anchors))

    exposure = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    exposure.to_parquet(exposure_path, index=False, compression="zstd")
    logging.info("Exposure rows: %s", f"{len(exposure):,}")

    # --- gates ------------------------------------------------------------
    slot_positions = (
        exposure["slot_id_pass_A"].str.lstrip("C").astype(int).tolist() if len(exposure) else []
    )
    spearman = presentation_order_correlation(
        slot_positions, exposure["linkage_score"].tolist() if len(exposure) else []
    )
    sampled = exposure.loc[exposure["exposure_mode"] == "SAMPLED"] if len(exposure) else exposure
    retrievers_per_anchor = (
        sampled.groupby("anchor_episode_id")["retrievers_json"]
        .apply(lambda values: len({r for v in values for r in json.loads(v)}))
        if len(sampled) else pd.Series(dtype=int)
    )
    tail_per_anchor = (
        sampled.groupby("anchor_episode_id")["is_random_tail"].sum()
        if len(sampled) else pd.Series(dtype=int)
    )

    gate_g3b = abs(spearman) < 0.05
    gate_g3c = bool(retrievers_per_anchor.ge(3).all()) if len(retrievers_per_anchor) else True
    gate_g3d = bool(tail_per_anchor.ge(TAIL_FLOOR).all()) if len(tail_per_anchor) else True

    pool_definition = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exposure_version": EXPOSURE_VERSION,
        "definition": (
            "P(A) = episodes sharing the anchor's buyer_block_key whose first notice falls "
            f"between award+{MIN_GAP_DAYS} and award+{MAX_GAP_DAYS} days, excluding the "
            "anchor itself."
        ),
        "blocking": (
            "A validated SIREN blocks nationally. A buyer name blocks only within its own "
            "department, because normalize_buyer_for_blocking strips generic prefixes and "
            "would otherwise merge every identically-named commune in France."
        ),
        "why_it_matters": (
            "A negative means 'no successor inside P(A)'. A method whose own blocking rule "
            "reaches outside P(A) is being scored on a universe it did not choose, so its "
            "out-of-pool predictions are bucketed separately rather than counted as false "
            "positives."
        ),
        "min_gap_days": MIN_GAP_DAYS,
        "max_gap_days": MAX_GAP_DAYS,
    }
    pool_definition_path.write_text(
        json.dumps(pool_definition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exposure_version": EXPOSURE_VERSION,
        "exposure_seed": EXPOSURE_SEED,
        "output_file": str(exposure_path),
        "restricted": (
            "Contains model scores and retriever provenance. Never hand this file to an "
            "annotator; dossiers are produced through exposure.blind_view()."
        ),
        "anchors": int(len(anchors)),
        "anchors_without_pool": anchors_without_pool,
        "exposure_rows": int(len(exposure)),
        "exposure_mode_counts": dict(sorted(mode_counts.items())),
        "candidates_shown_per_anchor": (
            {
                "min": int(exposure.groupby("anchor_episode_id").size().min()),
                "median": float(exposure.groupby("anchor_episode_id").size().median()),
                "max": int(exposure.groupby("anchor_episode_id").size().max()),
            } if len(exposure) else {}
        ),
        "retriever_exposed_counts": dict(sorted(retriever_exposed.items())),
        "tail_inclusion_probability": (
            {
                "min": round(float(np.min(tail_probabilities_seen)), 6),
                "median": round(float(np.median(tail_probabilities_seen)), 6),
                "max": round(float(np.max(tail_probabilities_seen)), 6),
            } if tail_probabilities_seen else {}
        ),
        "gates": {
            "g3b_presentation_order_carries_no_ranking": {
                "spearman_slot_vs_linkage_score": round(spearman, 5),
                "threshold": 0.05,
                "passed": gate_g3b,
            },
            "g3c_at_least_three_retrievers_per_sampled_anchor": {
                "min_observed": int(retrievers_per_anchor.min()) if len(retrievers_per_anchor) else None,
                "passed": gate_g3c,
            },
            "g3d_random_tail_floor": {
                "floor": TAIL_FLOOR,
                "min_observed": int(tail_per_anchor.min()) if len(tail_per_anchor) else None,
                "passed": gate_g3d,
            },
        },
        "validation_passed": bool(len(exposure) > 0 and gate_g3b and gate_g3c and gate_g3d),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    configure_logging(project_root)
    summary = build(project_root, output_dir, args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["validation_passed"]:
        raise RuntimeError("Benchmark exposure validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
