#!/usr/bin/env python3
"""Build the schema-aware BOAMP standardized notice layer."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.standardize import PARSER_VERSION, RAW_FIELD_NAMES, standardize_record


START_YEAR = 2015
END_YEAR = 2025
DEFAULT_OUTPUT_DIR = Path("data/processed/boamp")
BATCH_SIZE = 2_000


STANDARDIZED_SCHEMA = pa.schema(
    [*(pa.field(name, pa.string()) for name in RAW_FIELD_NAMES),
     pa.field("buyer_name_raw", pa.string()),
     pa.field("buyer_name_normalized", pa.string()),
     pa.field("buyer_siret", pa.string()),
     pa.field("buyer_siren", pa.string()),
     pa.field("buyer_id", pa.string()),
     pa.field("buyer_id_source", pa.string()),
     pa.field("buyer_id_quality", pa.string()),
     pa.field("buyer_key", pa.string()),
     pa.field("buyer_identifier_candidates_json", pa.string()),
     pa.field("buyer_multi_organization_flag", pa.bool_()),
     pa.field("buyer_postcode", pa.string()),
     pa.field("buyer_postcode_candidates_json", pa.string()),
     pa.field("buyer_city", pa.string()),
     pa.field("buyer_department", pa.string()),
     pa.field("buyer_region", pa.string()),
     pa.field("geography_source", pa.string()),
     pa.field("geography_quality", pa.string()),
     pa.field("grand_ouest_flag", pa.bool_()),
     pa.field("publication_date", pa.date32()),
     pa.field("diffusion_end_date", pa.date32()),
     pa.field("response_deadline", pa.timestamp("us", tz="UTC")),
     pa.field("deadline_before_publication", pa.bool_()),
     pa.field("contract_end_before_start", pa.bool_()),
     pa.field("date_outside_study", pa.bool_()),
     pa.field("main_cpv", pa.string()),
     pa.field("all_cpvs_json", pa.string()),
     pa.field("cpv_divisions_json", pa.string()),
     pa.field("cpv_groups_json", pa.string()),
     pa.field("cpv_classes_json", pa.string()),
     pa.field("cpv_categories_json", pa.string()),
     pa.field("has_any_cpv", pa.bool_()),
     pa.field("notice_titles_json", pa.string()),
     pa.field("notice_descriptions_json", pa.string()),
     pa.field("notice_text", pa.string()),
     pa.field("has_episode_text", pa.bool_()),
     pa.field("procedure_reference", pa.string()),
     pa.field("procedure_reference_source", pa.string()),
     pa.field("linked_notice_ids_json", pa.string()),
     pa.field("framework_flag", pa.bool_()),
     pa.field("duration_raw", pa.string()),
     pa.field("duration_months", pa.float64()),
     pa.field("duration_source", pa.string()),
     pa.field("duration_quality", pa.string()),
     pa.field("contract_start_date", pa.date32()),
     pa.field("contract_end_date", pa.date32()),
     pa.field("amount_candidates_json", pa.string()),
     pa.field("amount_candidate_count", pa.int32()),
     pa.field("parser_version", pa.string()),
     pa.field("raw_source_file", pa.string()),
     pa.field("raw_record_sha256", pa.string())]
)


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "build_standardized_notices.log", encoding="utf-8")],
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {exc}") from exc


def write_batch(writer: pq.ParquetWriter, rows: list[dict[str, Any]]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=STANDARDIZED_SCHEMA))


def build(project_root: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    raw_dir = project_root / "data/raw/boamp"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "notices_standardized_2015_2025.parquet"
    grand_ouest_path = output_dir / "notices_grand_ouest.parquet"
    summary_path = output_dir / "standardized_notice_summary.json"
    targets = [output_path, grand_ouest_path, summary_path]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(f"Outputs already exist: {existing}. Use --force to rebuild.")
    temp_output = output_path.with_suffix(".parquet.part")
    temp_go = grand_ouest_path.with_suffix(".parquet.part")
    for path in [temp_output, temp_go]:
        if path.exists():
            path.unlink()

    writer = pq.ParquetWriter(temp_output, STANDARDIZED_SCHEMA, compression="zstd", use_dictionary=True)
    go_writer = pq.ParquetWriter(temp_go, STANDARDIZED_SCHEMA, compression="zstd", use_dictionary=True)
    batch: list[dict[str, Any]] = []
    go_batch: list[dict[str, Any]] = []
    count = 0
    go_count = 0
    invalid_ids = 0
    duplicate_ids = 0
    seen_ids: set[str] = set()
    year_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    buyer_source_counts: Counter[str] = Counter()
    geography_source_counts: Counter[str] = Counter()
    min_date = None
    max_date = None
    started = time.monotonic()
    try:
        for year in range(START_YEAR, END_YEAR + 1):
            source_path = raw_dir / f"boamp_{year}.jsonl"
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            logging.info("Standardizing %s", source_path.name)
            for record in iter_jsonl(source_path):
                row = standardize_record(record, str(source_path.relative_to(project_root)))
                count += 1
                idweb = row["idweb"]
                if not idweb:
                    invalid_ids += 1
                elif idweb in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(idweb)
                publication = row["publication_date"]
                if publication:
                    year_counts[str(publication.year)] += 1
                    min_date = publication if min_date is None else min(min_date, publication)
                    max_date = publication if max_date is None else max(max_date, publication)
                source_counts[row["source_schema"] or "missing"] += 1
                buyer_source_counts[row["buyer_id_source"]] += 1
                geography_source_counts[row["geography_source"]] += 1
                batch.append(row)
                if row["grand_ouest_flag"]:
                    go_batch.append(row)
                    go_count += 1
                if len(batch) >= BATCH_SIZE:
                    write_batch(writer, batch)
                    batch.clear()
                if len(go_batch) >= BATCH_SIZE:
                    write_batch(go_writer, go_batch)
                    go_batch.clear()
                if count % 100_000 == 0:
                    logging.info("Standardized %s notices in %.1f minutes", f"{count:,}", (time.monotonic() - started) / 60)
        write_batch(writer, batch)
        write_batch(go_writer, go_batch)
    finally:
        writer.close()
        go_writer.close()

    temp_output.replace(output_path)
    temp_go.replace(grand_ouest_path)
    expected = 1_620_712
    summary = {
        "created_at": datetime_now(),
        "input_period": {"start": "2015-01-01", "end_exclusive": "2026-01-01"},
        "output_file": str(output_path),
        "grand_ouest_output_file": str(grand_ouest_path),
        "rows": count,
        "expected_rows": expected,
        "rows_match_expected": count == expected,
        "grand_ouest_rows": go_count,
        "unique_idweb": len(seen_ids),
        "missing_idweb": invalid_ids,
        "duplicate_idweb": duplicate_ids,
        "min_publication_date": str(min_date) if min_date else None,
        "max_publication_date": str(max_date) if max_date else None,
        "rows_by_year": dict(sorted(year_counts.items())),
        "source_schema_counts": dict(source_counts.most_common()),
        "buyer_id_source_counts": dict(buyer_source_counts.most_common()),
        "geography_source_counts": dict(geography_source_counts.most_common()),
        "schema_field_count": len(STANDARDIZED_SCHEMA),
        "parser_version": PARSER_VERSION,
        "validation_passed": bool(count == expected and len(seen_ids) == expected and not invalid_ids and not duplicate_ids and min_date and min_date >= date_from_text("2015-01-01") and max_date and max_date < date_from_text("2026-01-01")),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def datetime_now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def date_from_text(value: str):
    from datetime import date
    return date.fromisoformat(value)


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
        raise RuntimeError("Standardized notice validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
