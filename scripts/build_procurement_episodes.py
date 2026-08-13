#!/usr/bin/env python3
"""Build conservative deterministic procurement episodes from BOAMP notices.

The edge index is stored temporarily in SQLite and union-find state is held in
compact NumPy arrays. This avoids materializing the full notice corpus as
Python dictionaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.episodes import (
    EPISODE_VERSION,
    REFERENCE_MAX_SPAN_DAYS,
    IndexedDisjointSet,
    aggregate_episode,
    episode_id,
    parse_json_list,
)


DEFAULT_OUTPUT_DIR = Path("data/processed/boamp")
BATCH_SIZE = 5_000
METHOD_BITS = {"contractfolderid": 1, "explicit_linked_notice": 2, "exact_reference_same_buyer": 4}

MEMBERSHIP_SCHEMA = pa.schema([
    pa.field("idweb", pa.string()),
    pa.field("episode_id", pa.string()),
    pa.field("notice_edge_methods_json", pa.string()),
    pa.field("episode_version", pa.string()),
])

EDGE_SCHEMA = pa.schema([
    pa.field("left", pa.string()),
    pa.field("right", pa.string()),
    pa.field("method", pa.string()),
    pa.field("evidence", pa.string()),
    pa.field("accepted", pa.bool_()),
    pa.field("decision_reason", pa.string()),
])

EPISODE_SCHEMA = pa.schema([
    pa.field("episode_id", pa.string()),
    pa.field("constituent_notice_ids_json", pa.string()),
    pa.field("notice_count", pa.int32()),
    pa.field("episode_origin_date", pa.date32()),
    pa.field("episode_origin_date_source", pa.string()),
    pa.field("episode_last_notice_date", pa.date32()),
    pa.field("episode_span_days", pa.int32()),
    pa.field("episode_notice_natures_json", pa.string()),
    pa.field("episode_notice_types_json", pa.string()),
    pa.field("buyer_siren", pa.string()),
    pa.field("buyer_key", pa.string()),
    pa.field("buyer_name_raw", pa.string()),
    pa.field("buyer_name_normalized", pa.string()),
    pa.field("buyer_id_quality", pa.string()),
    pa.field("buyer_department", pa.string()),
    pa.field("buyer_region", pa.string()),
    pa.field("grand_ouest_flag", pa.bool_()),
    pa.field("episode_text", pa.string()),
    pa.field("main_cpv", pa.string()),
    pa.field("all_cpvs_json", pa.string()),
    pa.field("procedure_references_json", pa.string()),
    pa.field("procedure_type", pa.string()),
    pa.field("framework_flag", pa.bool_()),
    pa.field("duration_months", pa.float64()),
    pa.field("duration_quality", pa.string()),
    pa.field("amount_candidates_json", pa.string()),
    pa.field("episode_reconstruction_method", pa.string()),
    pa.field("episode_reconstruction_methods_json", pa.string()),
    pa.field("episode_reconstruction_quality", pa.string()),
    pa.field("buyer_conflict_flag", pa.bool_()),
    pa.field("procedure_reference_conflict_flag", pa.bool_()),
    pa.field("duration_conflict_flag", pa.bool_()),
    pa.field("impossible_chronology_flag", pa.bool_()),
    pa.field("episode_version", pa.string()),
])

AGGREGATION_COLUMNS = [
    "idweb", "buyer_siren", "buyer_key", "buyer_name_raw", "buyer_name_normalized",
    "buyer_department", "buyer_region", "grand_ouest_flag", "publication_date", "nature",
    "type_avis", "notice_text", "main_cpv", "all_cpvs_json", "procedure_reference",
    "procedure_reference_source",
    "procedure_libelle", "type_procedure", "framework_flag", "duration_months",
    "amount_candidates_json",
]


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "build_procurement_episodes.log", encoding="utf-8")],
    )


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def methods_from_mask(mask: int) -> list[str]:
    return [method for method, bit in METHOD_BITS.items() if mask & bit]


def write_rows(writer: pq.ParquetWriter, schema: pa.Schema, rows: list[dict[str, Any]]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def make_sqlite_index(notice_path: Path, database_path: Path) -> tuple[sqlite3.Connection, int]:
    if database_path.exists():
        database_path.unlink()
    connection = sqlite3.connect(database_path)
    connection.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE notices (
            idx INTEGER PRIMARY KEY,
            idweb TEXT NOT NULL UNIQUE,
            contractfolderid TEXT NOT NULL,
            procedure_reference TEXT NOT NULL,
            procedure_reference_source TEXT NOT NULL,
            buyer_siren TEXT NOT NULL,
            buyer_key TEXT NOT NULL,
            buyer_name_normalized TEXT NOT NULL,
            publication_date TEXT NOT NULL
        );
        CREATE TABLE explicit_links (right_idx INTEGER NOT NULL, target_idweb TEXT NOT NULL);
    """)
    parquet = pq.ParquetFile(notice_path)
    columns = ["idweb", "contractfolderid", "linked_notice_ids_json", "procedure_reference", "procedure_reference_source", "buyer_siren", "buyer_key", "buyer_name_normalized", "publication_date"]
    index = 0
    for batch in parquet.iter_batches(batch_size=25_000, columns=columns):
        notice_rows = []
        link_rows = []
        for row in batch.to_pylist():
            notice_rows.append((
                index,
                str(row.get("idweb") or ""),
                str(row.get("contractfolderid") or ""),
                str(row.get("procedure_reference") or ""),
                str(row.get("procedure_reference_source") or ""),
                str(row.get("buyer_siren") or ""),
                str(row.get("buyer_key") or ""),
                str(row.get("buyer_name_normalized") or ""),
                str(row.get("publication_date") or ""),
            ))
            for target in parse_json_list(row.get("linked_notice_ids_json")):
                link_rows.append((index, target))
            index += 1
        connection.executemany("INSERT INTO notices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", notice_rows)
        if link_rows:
            connection.executemany("INSERT INTO explicit_links VALUES (?, ?)", link_rows)
        if index % 100_000 < len(notice_rows):
            logging.info("Indexed %s notices for episode reconstruction", f"{index:,}")
    connection.commit()
    connection.executescript("""
        CREATE INDEX idx_notices_folder ON notices(contractfolderid, idx);
        CREATE INDEX idx_notices_reference ON notices(buyer_key, procedure_reference, publication_date, idx);
        CREATE INDEX idx_explicit_target ON explicit_links(target_idweb);
    """)
    return connection, index


def accept_edge(
    dsu: IndexedDisjointSet,
    notice_masks: np.ndarray,
    edge_writer: pq.ParquetWriter,
    edge_buffer: list[dict[str, Any]],
    counters: Counter[str],
    left_idx: int,
    left_id: str,
    right_idx: int,
    right_id: str,
    method: str,
    evidence: str,
) -> None:
    accepted, reason = dsu.union(left_idx, right_idx)
    if accepted:
        bit = METHOD_BITS[method]
        notice_masks[left_idx] |= bit
        notice_masks[right_idx] |= bit
        counters[f"accepted_{method}"] += 1
    else:
        counters[f"rejected_{reason}"] += 1
    edge_buffer.append({
        "left": left_id,
        "right": right_id,
        "method": method,
        "evidence": evidence,
        "accepted": accepted,
        "decision_reason": reason,
    })
    if len(edge_buffer) >= BATCH_SIZE:
        write_rows(edge_writer, EDGE_SCHEMA, edge_buffer)


def build_edges(
    connection: sqlite3.Connection,
    size: int,
    edge_writer: pq.ParquetWriter,
) -> tuple[IndexedDisjointSet, np.ndarray, Counter[str]]:
    dsu = IndexedDisjointSet(size)
    for index, siren in connection.execute("SELECT idx, buyer_siren FROM notices WHERE buyer_siren != ''"):
        dsu.set_trusted_buyer(index, siren)
    masks = np.zeros(size, dtype=np.uint8)
    counters: Counter[str] = Counter()
    buffer: list[dict[str, Any]] = []

    current_key = None
    anchor = None
    for idx, idweb, key in connection.execute(
        "SELECT idx, idweb, contractfolderid FROM notices WHERE contractfolderid != '' ORDER BY contractfolderid, idx"
    ):
        if key != current_key:
            current_key = key
            anchor = (idx, idweb)
            continue
        accept_edge(dsu, masks, edge_writer, buffer, counters, anchor[0], anchor[1], idx, idweb, "contractfolderid", key)

    explicit_query = """
        SELECT target.idx, target.idweb, source.idx, source.idweb, links.target_idweb
        FROM explicit_links links
        JOIN notices source ON source.idx = links.right_idx
        JOIN notices target ON target.idweb = links.target_idweb
        WHERE target.idx != source.idx
        ORDER BY target.idweb, source.idweb
    """
    for left_idx, left_id, right_idx, right_id, evidence in connection.execute(explicit_query):
        accept_edge(dsu, masks, edge_writer, buffer, counters, left_idx, left_id, right_idx, right_id, "explicit_linked_notice", evidence)

    previous = None
    reference_query = """
        SELECT idx, idweb, buyer_key, buyer_name_normalized, procedure_reference, publication_date
        FROM notices
        WHERE buyer_key != '' AND buyer_name_normalized != '' AND procedure_reference != ''
          AND procedure_reference_source != 'eforms:cbc:ContractFolderID'
          AND publication_date != ''
        ORDER BY buyer_key, buyer_name_normalized, procedure_reference, publication_date, idx
    """
    for idx, idweb, buyer_key, buyer_name, reference, publication_date in connection.execute(reference_query):
        group = (buyer_key, buyer_name, reference)
        if previous and previous[0] == group:
            prior_date = datetime.fromisoformat(previous[3]).date()
            current_date = datetime.fromisoformat(publication_date).date()
            span = (current_date - prior_date).days
            if span <= REFERENCE_MAX_SPAN_DAYS:
                accept_edge(
                    dsu, masks, edge_writer, buffer, counters,
                    previous[1], previous[2], idx, idweb,
                    "exact_reference_same_buyer", f"{buyer_key}|{reference}",
                )
            else:
                counters["rejected_reference_span"] += 1
        previous = (group, idx, idweb, publication_date)
    write_rows(edge_writer, EDGE_SCHEMA, buffer)
    return dsu, masks, counters


def episode_ids_for_components(
    connection: sqlite3.Connection,
    roots: np.ndarray,
    sizes: np.ndarray,
) -> tuple[dict[int, str], set[int]]:
    multi_roots = set(np.flatnonzero(sizes > 1).tolist())
    identifiers: dict[int, list[str]] = defaultdict(list)
    for index, idweb in connection.execute("SELECT idx, idweb FROM notices ORDER BY idx"):
        root = int(roots[index])
        if root in multi_roots:
            identifiers[root].append(idweb)
    return {root: episode_id(ids) for root, ids in identifiers.items()}, multi_roots


def update_episode_summary(summary: dict[str, Any], row: dict[str, Any]) -> None:
    summary["episode_rows"] += 1
    if row["grand_ouest_flag"]:
        summary["grand_ouest_episode_rows"] += 1
    count = int(row["notice_count"])
    bucket = str(count) if count <= 4 else "5_plus"
    summary["notices_per_episode"][bucket] += 1
    summary["reconstruction_method_counts"][row["episode_reconstruction_method"]] += 1
    summary["buyer_conflict_episodes"] += int(row["buyer_conflict_flag"])
    summary["reference_conflict_episodes"] += int(row["procedure_reference_conflict_flag"])
    summary["impossible_chronology_episodes"] += int(row["impossible_chronology_flag"])


def stream_outputs(
    notice_path: Path,
    roots: np.ndarray,
    sizes: np.ndarray,
    notice_masks: np.ndarray,
    component_masks: np.ndarray,
    multi_episode_ids: dict[int, str],
    membership_writer: pq.ParquetWriter,
    episode_writer: pq.ParquetWriter,
    grand_ouest_writer: pq.ParquetWriter,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "membership_rows": 0,
        "episode_rows": 0,
        "grand_ouest_episode_rows": 0,
        "notices_per_episode": Counter(),
        "reconstruction_method_counts": Counter(),
        "buyer_conflict_episodes": 0,
        "reference_conflict_episodes": 0,
        "impossible_chronology_episodes": 0,
    }
    membership_buffer: list[dict[str, Any]] = []
    singleton_buffer: list[dict[str, Any]] = []
    singleton_go_buffer: list[dict[str, Any]] = []
    multi_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    index = 0
    parquet = pq.ParquetFile(notice_path)
    for batch in parquet.iter_batches(batch_size=10_000, columns=AGGREGATION_COLUMNS):
        for row in batch.to_pylist():
            root = int(roots[index])
            methods = methods_from_mask(int(notice_masks[index]))
            current_episode_id = multi_episode_ids.get(root)
            if current_episode_id is None:
                current_episode_id = "EP-" + hashlib.sha256(str(row["idweb"]).encode("utf-8")).hexdigest()[:16]
            membership_buffer.append({
                "idweb": str(row["idweb"]),
                "episode_id": current_episode_id,
                "notice_edge_methods_json": json_compact(methods),
                "episode_version": EPISODE_VERSION,
            })
            summary["membership_rows"] += 1
            if sizes[root] == 1:
                episode = aggregate_episode([row], methods_from_mask(int(component_masks[root])))
                singleton_buffer.append(episode)
                if episode["grand_ouest_flag"]:
                    singleton_go_buffer.append(episode)
                update_episode_summary(summary, episode)
            else:
                multi_rows[root].append(row)
            index += 1
            if len(membership_buffer) >= BATCH_SIZE:
                write_rows(membership_writer, MEMBERSHIP_SCHEMA, membership_buffer)
            if len(singleton_buffer) >= BATCH_SIZE:
                write_rows(episode_writer, EPISODE_SCHEMA, singleton_buffer)
            if len(singleton_go_buffer) >= BATCH_SIZE:
                write_rows(grand_ouest_writer, EPISODE_SCHEMA, singleton_go_buffer)
        if index % 100_000 < batch.num_rows:
            logging.info("Aggregated %s notice memberships", f"{index:,}")

    write_rows(membership_writer, MEMBERSHIP_SCHEMA, membership_buffer)
    write_rows(episode_writer, EPISODE_SCHEMA, singleton_buffer)
    write_rows(grand_ouest_writer, EPISODE_SCHEMA, singleton_go_buffer)
    multi_buffer: list[dict[str, Any]] = []
    multi_go_buffer: list[dict[str, Any]] = []
    for root, rows in multi_rows.items():
        episode = aggregate_episode(rows, methods_from_mask(int(component_masks[root])))
        multi_buffer.append(episode)
        if episode["grand_ouest_flag"]:
            multi_go_buffer.append(episode)
        update_episode_summary(summary, episode)
        if len(multi_buffer) >= BATCH_SIZE:
            write_rows(episode_writer, EPISODE_SCHEMA, multi_buffer)
        if len(multi_go_buffer) >= BATCH_SIZE:
            write_rows(grand_ouest_writer, EPISODE_SCHEMA, multi_go_buffer)
    write_rows(episode_writer, EPISODE_SCHEMA, multi_buffer)
    write_rows(grand_ouest_writer, EPISODE_SCHEMA, multi_go_buffer)
    return summary


def build(project_root: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    notice_path = output_dir / "notices_standardized_2015_2025.parquet"
    membership_path = output_dir / "episode_membership.parquet"
    edge_path = output_dir / "episode_edges.parquet"
    episode_path = output_dir / "episodes_2015_2025.parquet"
    grand_ouest_path = output_dir / "episodes_grand_ouest.parquet"
    summary_path = output_dir / "episode_reconstruction_summary.json"
    database_path = output_dir / ".episode_index.sqlite"
    outputs = [membership_path, edge_path, episode_path, grand_ouest_path, summary_path]
    if not notice_path.exists():
        raise FileNotFoundError(notice_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(f"Outputs already exist: {existing}. Use --force to rebuild.")
    temporary = {path: path.with_suffix(path.suffix + ".part") for path in outputs[:-1]}
    for path in [*temporary.values(), database_path]:
        if path.exists():
            path.unlink()

    connection, notice_count = make_sqlite_index(notice_path, database_path)
    edge_writer = pq.ParquetWriter(temporary[edge_path], EDGE_SCHEMA, compression="zstd")
    try:
        dsu, notice_masks, edge_counts = build_edges(connection, notice_count, edge_writer)
    finally:
        edge_writer.close()
    roots = dsu.roots()
    sizes = np.bincount(roots, minlength=notice_count)
    component_masks = np.zeros(notice_count, dtype=np.uint8)
    np.bitwise_or.at(component_masks, roots, notice_masks)
    multi_episode_ids, _ = episode_ids_for_components(connection, roots, sizes)

    membership_writer = pq.ParquetWriter(temporary[membership_path], MEMBERSHIP_SCHEMA, compression="zstd")
    episode_writer = pq.ParquetWriter(temporary[episode_path], EPISODE_SCHEMA, compression="zstd")
    go_writer = pq.ParquetWriter(temporary[grand_ouest_path], EPISODE_SCHEMA, compression="zstd")
    try:
        streamed = stream_outputs(
            notice_path, roots, sizes, notice_masks, component_masks, multi_episode_ids,
            membership_writer, episode_writer, go_writer,
        )
    finally:
        membership_writer.close()
        episode_writer.close()
        go_writer.close()
        connection.close()

    for destination, part in temporary.items():
        part.replace(destination)
    if database_path.exists():
        database_path.unlink()
    singleton_count = int(streamed["notices_per_episode"].get("1", 0))
    episode_count = int(streamed["episode_rows"])
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_notices": str(notice_path),
        "outputs": {
            "membership": str(membership_path), "edges": str(edge_path),
            "episodes": str(episode_path), "grand_ouest_episodes": str(grand_ouest_path),
        },
        "notice_rows": notice_count,
        "membership_rows": int(streamed["membership_rows"]),
        "unique_membership_notices": int(streamed["membership_rows"]),
        "episode_rows": episode_count,
        "grand_ouest_episode_rows": int(streamed["grand_ouest_episode_rows"]),
        "singleton_share": singleton_count / episode_count if episode_count else 0.0,
        "notices_per_episode": {key: int(streamed["notices_per_episode"].get(key, 0)) for key in ["1", "2", "3", "4", "5_plus"]},
        "reconstruction_method_counts": {str(key): int(value) for key, value in streamed["reconstruction_method_counts"].items()},
        "edge_counts": {str(key): int(value) for key, value in edge_counts.items()},
        "buyer_conflict_episodes": int(streamed["buyer_conflict_episodes"]),
        "reference_conflict_episodes": int(streamed["reference_conflict_episodes"]),
        "impossible_chronology_episodes": int(streamed["impossible_chronology_episodes"]),
        "all_notices_assigned_once": bool(streamed["membership_rows"] == notice_count),
        "validation_passed": bool(
            streamed["membership_rows"] == notice_count
            and streamed["buyer_conflict_episodes"] == 0
            and streamed["impossible_chronology_episodes"] == 0
        ),
        "implementation": "disk_backed_sqlite_index_and_numpy_union_find",
        "episode_version": EPISODE_VERSION,
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
        raise RuntimeError("Episode reconstruction validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
