#!/usr/bin/env python3
"""Build the national episode index that benchmark v3 is sampled from.

The existing study reads ``episodes_grand_ouest.parquet`` and stratifies on
``buyer_region``. Neither works nationally: ``buyer_region`` is populated only
for the fourteen Grand Ouest departments by construction
(``standardize.py:20-34``), so it is blank on 88% of national episodes, and the
episode layer carries no award date, no expected end, and no incumbent
supplier -- all of which the benchmark needs.

This script produces one lean row per national episode with the fields a
benchmark frame, a candidate pool and a predecessor resolver require. Two
design choices are deliberate:

*It excludes ``episode_text``.* Text is the bulk of the 279 MB episode file and
is needed only for sampled anchors and their pools, which are joined back from
``episodes_2015_2025.parquet`` later. Carrying 1.1M full texts through every
downstream stage would dominate memory for no benefit.

*It adds ``buyer_block_key``.*
:func:`boamp_pipeline.linkage.normalize_buyer_for_blocking` strips generic
prefixes, so "Commune de Saint-Martin" normalises identically in every
department that has one. In a single-region study that was harmless; nationally
it silently merges distinct buyers. Name-based blocking is therefore scoped by
department, while a validated SIREN blocks nationally as before.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.expiry_linkage import resolve_expected_end  # noqa: E402
from boamp_pipeline.geography import (  # noqa: E402
    department_group,
    normalize_department,
    region_for_department,
)
from boamp_pipeline.linkage import (  # noqa: E402
    DIGITAL_CPV_DIVISIONS,
    cpv_divisions,
    normalize_buyer_for_blocking,
    parse_json_list,
)
from boamp_pipeline.standardize import normalize_name  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp/benchmark")
INDEX_VERSION = "boamp_national_episode_index_v1.0"

AWARD_NATURE = "ATTRIBUTION"
NOTICE_BATCH = 200_000
EPISODE_BATCH = 50_000

#: Mirrors ``build_survival_cohort.RELIABLE_DURATION_QUALITY``. A duration is
#: only trusted when the episode's notices agreed on a typed value; "unresolved"
#: and "conflict" stay missing rather than being back-filled.
RELIABLE_DURATION_QUALITY = ("typed_consensus",)

#: Buyer-size classes cut at the measured national quantiles of episodes per
#: buyer (median 3, p90 41, p99 283). Large buyers are where linkage fails, so
#: the frame has to balance on this rather than let it fall where it may.
BUYER_SIZE_BREAKS = ((4, "SMALL"), (40, "MEDIUM"), (282, "LARGE"))
BUYER_SIZE_MEGA = "MEGA"
BUYER_SIZE_UNKNOWN = "UNKNOWN"

#: Legal forms stripped before comparing supplier names. ``titulaire`` carries
#: essentially no identifier -- 86 of 348,842 values contain any digit -- so
#: incumbent matching is name-based and needs the forms removed.
_LEGAL_FORMS = {
    "sas", "sarl", "sa", "eurl", "sasu", "snc", "scop", "sci", "scic", "scp",
    "groupe", "ets", "etablissements", "societe", "ste", "cie", "et", "fils",
    "france", "sam", "gie", "eirl", "spa", "srl", "ltd", "gmbh", "bv", "nv",
}

INDEX_SCHEMA = pa.schema([
    pa.field("episode_id", pa.string()),
    pa.field("notice_count", pa.int32()),
    pa.field("episode_origin_date", pa.date32()),
    pa.field("episode_last_notice_date", pa.date32()),
    pa.field("award_date", pa.date32()),
    pa.field("has_award_notice", pa.bool_()),
    pa.field("award_year", pa.int32()),
    pa.field("buyer_siren", pa.string()),
    pa.field("buyer_key", pa.string()),
    pa.field("buyer_name_raw", pa.string()),
    pa.field("buyer_name_normalized", pa.string()),
    pa.field("buyer_name_blocking", pa.string()),
    pa.field("buyer_block_key", pa.string()),
    pa.field("buyer_block_basis", pa.string()),
    pa.field("buyer_block_episode_count", pa.int32()),
    pa.field("buyer_size_class", pa.string()),
    pa.field("buyer_id_quality", pa.string()),
    pa.field("buyer_department", pa.string()),
    pa.field("buyer_region_national", pa.string()),
    pa.field("dept_group", pa.string()),
    pa.field("grand_ouest_flag", pa.bool_()),
    pa.field("main_cpv", pa.string()),
    pa.field("all_cpvs_json", pa.string()),
    pa.field("cpv_divisions_json", pa.string()),
    pa.field("digital_flag", pa.bool_()),
    pa.field("digital_segment", pa.string()),
    pa.field("duration_months", pa.float64()),
    pa.field("duration_quality", pa.string()),
    pa.field("duration_months_reliable", pa.float64()),
    pa.field("expected_end_date", pa.date32()),
    pa.field("expected_end_source", pa.string()),
    pa.field("explicit_start_date", pa.date32()),
    pa.field("explicit_end_date", pa.date32()),
    pa.field("explicit_start_date_count", pa.int32()),
    pa.field("explicit_end_date_count", pa.int32()),
    pa.field("framework_flag", pa.bool_()),
    pa.field("procedure_references_json", pa.string()),
    pa.field("titulaire_names_json", pa.string()),
    pa.field("titulaire_count", pa.int32()),
    pa.field("episode_reconstruction_method", pa.string()),
    pa.field("episode_reconstruction_quality", pa.string()),
    pa.field("buyer_conflict_flag", pa.bool_()),
    pa.field("index_version", pa.string()),
])

EPISODE_COLUMNS = [
    "episode_id", "notice_count", "episode_origin_date", "episode_last_notice_date",
    "episode_notice_natures_json", "buyer_siren", "buyer_key", "buyer_name_raw",
    "buyer_name_normalized", "buyer_id_quality", "buyer_department", "grand_ouest_flag",
    "main_cpv", "all_cpvs_json", "procedure_references_json", "framework_flag",
    "duration_months", "duration_quality", "episode_reconstruction_method",
    "episode_reconstruction_quality", "buyer_conflict_flag",
]


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "build_national_episode_index.log", encoding="utf-8"),
        ],
    )


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Supplier names
# ---------------------------------------------------------------------------


def normalize_supplier(value: Any) -> str:
    """Normalise a supplier name for incumbent matching.

    Legal forms are dropped so that "SAS DUCROCQ TP" and "Ducrocq TP" compare
    equal. Names reducing to nothing but legal forms return ``""``.
    """
    tokens = [token for token in normalize_name(value).split() if token not in _LEGAL_FORMS]
    return " ".join(tokens)


def parse_titulaire(value: Any) -> list[str]:
    """Extract normalised supplier names from a raw ``titulaire`` cell."""
    if value is None:
        return []
    text = str(value).strip()
    if not text or text in {"[]", "{}", "None", "nan"}:
        return []
    candidates: list[Any]
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        candidates = [text]
    else:
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict):
            candidates = list(parsed.values())
        else:
            candidates = [parsed]
    names: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            candidate = candidate.get("#text") or candidate.get("DENOMINATION") or ""
        name = normalize_supplier(candidate)
        if name and name not in names:
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Buyer blocking
# ---------------------------------------------------------------------------


def buyer_block_key(buyer_key: Any, name_blocking: str, department: str) -> tuple[str, str]:
    """Blocking key for an episode, and the basis it was derived from.

    A validated SIREN identifies a buyer nationally, so it blocks on its own.
    A name does not: ``normalize_buyer_for_blocking`` maps "Commune de Nantes"
    and "Nantes" together on purpose, which also maps every "Commune de
    Saint-Martin" in France together. Name blocking is therefore confined to a
    single department. An episode with neither is unblockable and is excluded
    from the frame rather than being given a degenerate pool.
    """
    key = str(buyer_key or "").strip()
    if key:
        return f"K|{key}", "buyer_key"
    if name_blocking and department and department != "UNRESOLVED":
        return f"N|{name_blocking}|{department}", "name_department"
    return "", "unblockable"


def buyer_size_class(episode_count: int | None) -> str:
    if episode_count is None or episode_count <= 0:
        return BUYER_SIZE_UNKNOWN
    for threshold, label in BUYER_SIZE_BREAKS:
        if episode_count <= threshold:
            return label
    return BUYER_SIZE_MEGA


# ---------------------------------------------------------------------------
# Pass 1: notice-level evidence
# ---------------------------------------------------------------------------


def collect_notice_evidence(processed_dir: Path) -> dict[str, Any]:
    """Aggregate award dates, contract dates and suppliers onto episodes.

    An episode's notices are scattered across the 2.8 GB notice file, so this
    is a single batched pass that folds each notice into per-episode
    accumulators keyed by a compact integer code.
    """
    membership = pd.read_parquet(
        processed_dir / "episode_membership.parquet", columns=["idweb", "episode_id"]
    )
    codes, episode_ids = pd.factorize(membership["episode_id"], sort=False)
    code_by_idweb = dict(zip(membership["idweb"].tolist(), codes.tolist()))
    size = len(episode_ids)
    logging.info("Membership: %s notices across %s episodes", f"{len(membership):,}", f"{size:,}")
    del membership, codes

    award_ordinal = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
    start_dates: dict[int, set[date]] = defaultdict(set)
    end_dates: dict[int, set[date]] = defaultdict(set)
    suppliers: dict[int, set[str]] = defaultdict(set)

    parquet = pq.ParquetFile(processed_dir / "notices_standardized_2015_2025.parquet")
    columns = ["idweb", "nature", "publication_date", "contract_start_date",
               "contract_end_date", "titulaire"]
    scanned = 0
    unmatched = 0
    for batch in parquet.iter_batches(batch_size=NOTICE_BATCH, columns=columns):
        rows = batch.to_pydict()
        for index in range(batch.num_rows):
            code = code_by_idweb.get(rows["idweb"][index])
            if code is None:
                unmatched += 1
                continue
            if rows["nature"][index] == AWARD_NATURE:
                publication = rows["publication_date"][index]
                if publication is not None:
                    ordinal = publication.toordinal()
                    if ordinal < award_ordinal[code]:
                        award_ordinal[code] = ordinal
                names = parse_titulaire(rows["titulaire"][index])
                if names:
                    suppliers[code].update(names)
            start = rows["contract_start_date"][index]
            if start is not None:
                start_dates[code].add(start)
            finish = rows["contract_end_date"][index]
            if finish is not None:
                end_dates[code].add(finish)
        scanned += batch.num_rows
        if scanned % 400_000 < NOTICE_BATCH:
            logging.info("Scanned %s notices", f"{scanned:,}")

    logging.info(
        "Notice evidence: %s episodes with an award date, %s with contract dates, "
        "%s with a supplier (%s notices unmatched)",
        f"{int((award_ordinal != np.iinfo(np.int64).max).sum()):,}",
        f"{len(set(start_dates) | set(end_dates)):,}",
        f"{len(suppliers):,}",
        f"{unmatched:,}",
    )
    return {
        "code_by_episode": {episode: code for code, episode in enumerate(episode_ids)},
        "award_ordinal": award_ordinal,
        "start_dates": start_dates,
        "end_dates": end_dates,
        "suppliers": suppliers,
        "notices_unmatched": unmatched,
    }


# ---------------------------------------------------------------------------
# Pass 2: episode rows
# ---------------------------------------------------------------------------


def _sorted_unique(values: set[date]) -> list[date]:
    return sorted(values)


def build_row(
    episode: dict[str, Any],
    award: date | None,
    starts: list[date],
    ends: list[date],
    supplier_names: list[str],
    buyer_episode_counts: dict[str, int],
) -> dict[str, Any]:
    codes = parse_json_list(episode["all_cpvs_json"])
    divisions = sorted(cpv_divisions(codes))
    digital = sorted(set(divisions) & set(DIGITAL_CPV_DIVISIONS))

    department = normalize_department(episode["buyer_department"])
    name_blocking = normalize_buyer_for_blocking(episode["buyer_name_normalized"])
    block_key, block_basis = buyer_block_key(episode["buyer_key"], name_blocking, department)
    block_episodes = int(buyer_episode_counts.get(block_key, 0)) if block_key else 0

    duration = episode["duration_months"]
    reliable = (
        float(duration)
        if episode["duration_quality"] in RELIABLE_DURATION_QUALITY
        and duration is not None
        and float(duration) > 0
        else None
    )

    # Fast path: with no contract dates and no trusted duration there is nothing
    # to resolve, and this covers most episodes (only 93,882 notices carry a
    # start date and 56,126 an end date nationally).
    if not starts and not ends and reliable is None:
        expected_end, expected_source = None, "unavailable"
    else:
        resolved, expected_source = resolve_expected_end(
            award, reliable, start_dates=starts, end_dates=ends
        )
        expected_end = resolved.date() if pd.notna(resolved) else None

    return {
        "episode_id": episode["episode_id"],
        "notice_count": int(episode["notice_count"] or 0),
        "episode_origin_date": episode["episode_origin_date"],
        "episode_last_notice_date": episode["episode_last_notice_date"],
        "award_date": award,
        "has_award_notice": award is not None,
        "award_year": int(award.year) if award is not None else None,
        "buyer_siren": episode["buyer_siren"] or "",
        "buyer_key": episode["buyer_key"] or "",
        "buyer_name_raw": episode["buyer_name_raw"] or "",
        "buyer_name_normalized": episode["buyer_name_normalized"] or "",
        "buyer_name_blocking": name_blocking,
        "buyer_block_key": block_key,
        "buyer_block_basis": block_basis,
        "buyer_block_episode_count": block_episodes,
        "buyer_size_class": buyer_size_class(block_episodes if block_key else None),
        "buyer_id_quality": episode["buyer_id_quality"] or "",
        "buyer_department": department,
        "buyer_region_national": region_for_department(department),
        "dept_group": department_group(department),
        "grand_ouest_flag": bool(episode["grand_ouest_flag"]),
        "main_cpv": episode["main_cpv"] or "",
        "all_cpvs_json": episode["all_cpvs_json"] or "[]",
        "cpv_divisions_json": json_compact(divisions),
        "digital_flag": bool(digital),
        "digital_segment": f"CPV-{digital[0]}" if digital else "",
        "duration_months": float(duration) if duration is not None else None,
        "duration_quality": episode["duration_quality"] or "",
        "duration_months_reliable": reliable,
        "expected_end_date": expected_end,
        "expected_end_source": expected_source,
        "explicit_start_date": starts[0] if len(starts) == 1 else None,
        "explicit_end_date": ends[0] if len(ends) == 1 else None,
        "explicit_start_date_count": len(starts),
        "explicit_end_date_count": len(ends),
        "framework_flag": bool(episode["framework_flag"]),
        "procedure_references_json": episode["procedure_references_json"] or "[]",
        "titulaire_names_json": json_compact(supplier_names),
        "titulaire_count": len(supplier_names),
        "episode_reconstruction_method": episode["episode_reconstruction_method"] or "",
        "episode_reconstruction_quality": episode["episode_reconstruction_quality"] or "",
        "buyer_conflict_flag": bool(episode["buyer_conflict_flag"]),
        "index_version": INDEX_VERSION,
    }


def scan_buyer_blocks(processed_dir: Path) -> tuple[dict[str, int], dict[str, int]]:
    """One pass giving episodes per blocking key and the homonym exposure.

    The homonym figure quantifies the risk that department-scoped blocking
    removes: blocking names shared by several departments, and among those the
    ones whose episodes carry conflicting validated SIRENs -- provably distinct
    buyers that a single national name block would have merged.
    """
    counts: dict[str, int] = defaultdict(int)
    departments_by_name: dict[str, set[str]] = defaultdict(set)
    sirens_by_name: dict[str, set[str]] = defaultdict(set)

    parquet = pq.ParquetFile(processed_dir / "episodes_2015_2025.parquet")
    columns = ["buyer_key", "buyer_name_normalized", "buyer_department", "buyer_siren"]
    for batch in parquet.iter_batches(batch_size=EPISODE_BATCH, columns=columns):
        rows = batch.to_pydict()
        for index in range(batch.num_rows):
            name = normalize_buyer_for_blocking(rows["buyer_name_normalized"][index])
            department = normalize_department(rows["buyer_department"][index])
            key, _ = buyer_block_key(rows["buyer_key"][index], name, department)
            if key:
                counts[key] += 1
            if not name:
                continue
            if department != "UNRESOLVED":
                departments_by_name[name].add(department)
            siren = str(rows["buyer_siren"][index] or "").strip()
            if siren:
                sirens_by_name[name].add(siren)

    multi_department = {n for n, depts in departments_by_name.items() if len(depts) > 1}
    conflicting = {n for n in multi_department if len(sirens_by_name.get(n, ())) > 1}
    homonym = {
        "blocking_names": len(departments_by_name),
        "names_spanning_multiple_departments": len(multi_department),
        "names_spanning_multiple_departments_with_conflicting_siren": len(conflicting),
    }
    return dict(counts), homonym


def build(project_root: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    processed_dir = output_dir.parent
    index_path = output_dir / "national_episode_index.parquet"
    summary_path = output_dir / "national_episode_index_summary.json"
    if index_path.exists() and not force:
        raise FileExistsError(f"{index_path} already exists. Use --force to rebuild.")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Counting episodes per buyer block")
    buyer_episode_counts, homonym = scan_buyer_blocks(processed_dir)
    logging.info(
        "Distinct buyer blocks: %s | blocking names spanning >1 department with "
        "conflicting SIREN: %s",
        f"{len(buyer_episode_counts):,}",
        f"{homonym['names_spanning_multiple_departments_with_conflicting_siren']:,}",
    )

    evidence = collect_notice_evidence(processed_dir)
    code_by_episode = evidence["code_by_episode"]
    award_ordinal = evidence["award_ordinal"]
    sentinel = np.iinfo(np.int64).max

    temp_path = index_path.with_suffix(".parquet.part")
    if temp_path.exists():
        temp_path.unlink()
    writer = pq.ParquetWriter(temp_path, INDEX_SCHEMA, compression="zstd", use_dictionary=True)

    counters: dict[str, int] = defaultdict(int)
    block_basis_counts: dict[str, int] = defaultdict(int)
    dept_group_counts: dict[str, int] = defaultdict(int)
    expected_source_counts: dict[str, int] = defaultdict(int)
    rows_written = 0
    buffer: list[dict[str, Any]] = []

    parquet = pq.ParquetFile(processed_dir / "episodes_2015_2025.parquet")
    try:
        for batch in parquet.iter_batches(batch_size=EPISODE_BATCH, columns=EPISODE_COLUMNS):
            episodes = batch.to_pylist()
            for episode in episodes:
                code = code_by_episode.get(episode["episode_id"])
                award = None
                if code is not None and award_ordinal[code] != sentinel:
                    award = date.fromordinal(int(award_ordinal[code]))
                starts = _sorted_unique(evidence["start_dates"].get(code, set()))
                ends = _sorted_unique(evidence["end_dates"].get(code, set()))
                suppliers = sorted(evidence["suppliers"].get(code, set()))

                row = build_row(episode, award, starts, ends, suppliers, buyer_episode_counts)
                buffer.append(row)
                rows_written += 1
                counters["has_award_notice"] += int(row["has_award_notice"])
                counters["digital"] += int(row["digital_flag"])
                counters["digital_with_award"] += int(row["digital_flag"] and row["has_award_notice"])
                counters["has_expected_end"] += int(row["expected_end_date"] is not None)
                counters["has_supplier"] += int(row["titulaire_count"] > 0)
                counters["blockable"] += int(bool(row["buyer_block_key"]))
                block_basis_counts[row["buyer_block_basis"]] += 1
                dept_group_counts[row["dept_group"]] += 1
                expected_source_counts[row["expected_end_source"]] += 1

            if len(buffer) >= EPISODE_BATCH:
                writer.write_table(pa.Table.from_pylist(buffer, schema=INDEX_SCHEMA))
                buffer.clear()
                logging.info("Indexed %s episodes", f"{rows_written:,}")
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=INDEX_SCHEMA))
    finally:
        writer.close()
    temp_path.replace(index_path)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "index_version": INDEX_VERSION,
        "output_file": str(index_path),
        "source_episodes": str(processed_dir / "episodes_2015_2025.parquet"),
        "episodes": rows_written,
        "episodes_with_award_notice": counters["has_award_notice"],
        "digital_episodes": counters["digital"],
        "digital_with_award_notice": counters["digital_with_award"],
        "episodes_with_expected_end": counters["has_expected_end"],
        "episodes_with_supplier": counters["has_supplier"],
        "blockable_episodes": counters["blockable"],
        "blockable_rate": round(counters["blockable"] / rows_written, 4) if rows_written else 0.0,
        "buyer_block_basis_counts": dict(sorted(block_basis_counts.items())),
        "dept_group_counts": dict(sorted(dept_group_counts.items())),
        "expected_end_source_counts": dict(sorted(expected_source_counts.items())),
        "distinct_buyer_blocks": len(buyer_episode_counts),
        "notices_unmatched_to_episode": evidence["notices_unmatched"],
        "homonym_exposure": homonym,
        "validation_passed": bool(
            rows_written > 0
            and evidence["notices_unmatched"] == 0
            and counters["blockable"] / rows_written >= 0.95
        ),
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
        raise RuntimeError("National episode index validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
