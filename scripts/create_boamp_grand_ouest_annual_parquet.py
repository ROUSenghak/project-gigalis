#!/usr/bin/env python3
"""Split the raw BOAMP Grand Ouest CSV into annual Parquet files.

The output keeps all source columns as strings. The only operation is a yearly
partition by dateparution so the files remain raw acquisition artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


START_YEAR = 2015
END_YEAR = 2025
RAW_RELATIVE_DIR = Path("data/raw/boamp")
METADATA_RELATIVE_DIR = Path("data/metadata")
DEFAULT_INPUT_NAME = "boamp_2015_2025_grand_ouest_raw.csv"
DEFAULT_OUTPUT_DIR = Path("data/raw/boamp/grand_ouest_annual_parquet")
CHUNKSIZE = 25_000
PROGRESS_INTERVAL_SECONDS = 30


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "create_boamp_grand_ouest_annual_parquet.log", encoding="utf-8"),
        ],
    )


def annual_path(output_dir: Path, year: int) -> Path:
    return output_dir / f"boamp_grand_ouest_{year}_raw.parquet"


def chunk_to_table(frame: pd.DataFrame, schema: pa.Schema | None = None) -> pa.Table:
    frame = frame.astype("string").fillna("")
    return pa.Table.from_pandas(frame, schema=schema, preserve_index=False)


def create_annual_parquet(input_path: Path, output_dir: Path, chunksize: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_paths = {year: annual_path(output_dir, year).with_suffix(".parquet.part") for year in range(START_YEAR, END_YEAR + 1)}
    final_paths = {year: annual_path(output_dir, year) for year in range(START_YEAR, END_YEAR + 1)}

    for path in temp_paths.values():
        if path.exists():
            path.unlink()

    writers: dict[int, pq.ParquetWriter] = {}
    schema: pa.Schema | None = None
    source_rows = 0
    written_rows = 0
    rows_by_year = {str(year): 0 for year in range(START_YEAR, END_YEAR + 1)}
    outside_historical_range = 0
    missing_dateparution = 0
    min_dateparution: str | None = None
    max_dateparution: str | None = None
    started = time.monotonic()
    last_log = started

    try:
        for chunk in pd.read_csv(input_path, chunksize=chunksize, dtype=str, keep_default_na=False):
            source_rows += len(chunk)
            if schema is None:
                schema = chunk_to_table(chunk.iloc[:0]).schema

            dates = pd.to_datetime(chunk["dateparution"], errors="coerce")
            missing_dateparution += int(dates.isna().sum())

            valid_date_strings = chunk.loc[dates.notna(), "dateparution"].astype(str)
            if len(valid_date_strings):
                chunk_min = valid_date_strings.min()
                chunk_max = valid_date_strings.max()
                min_dateparution = chunk_min if min_dateparution is None else min(min_dateparution, chunk_min)
                max_dateparution = chunk_max if max_dateparution is None else max(max_dateparution, chunk_max)

            for year in range(START_YEAR, END_YEAR + 1):
                mask = dates.dt.year.eq(year)
                if not mask.any():
                    continue

                yearly_chunk = chunk.loc[mask]
                table = chunk_to_table(yearly_chunk, schema=schema)
                if year not in writers:
                    writers[year] = pq.ParquetWriter(
                        temp_paths[year],
                        schema,
                        compression="snappy",
                        use_dictionary=True,
                    )
                writers[year].write_table(table)
                rows_by_year[str(year)] += len(yearly_chunk)
                written_rows += len(yearly_chunk)

            expected_mask = dates.ge(pd.Timestamp(f"{START_YEAR}-01-01")) & dates.lt(pd.Timestamp(f"{END_YEAR + 1}-01-01"))
            outside_historical_range += int((~expected_mask).sum())

            now = time.monotonic()
            if now - last_log >= PROGRESS_INTERVAL_SECONDS:
                logging.info(
                    "Scanned %s rows, wrote %s annual Parquet rows in %.1f minutes",
                    f"{source_rows:,}",
                    f"{written_rows:,}",
                    (now - started) / 60,
                )
                last_log = now
    finally:
        for writer in writers.values():
            writer.close()

    for year, temp_path in temp_paths.items():
        final_path = final_paths[year]
        if temp_path.exists():
            temp_path.replace(final_path)
        elif not final_path.exists():
            empty_table = pa.Table.from_pydict(
                {field.name: pa.array([], type=field.type) for field in schema or pa.schema([])}
            )
            pq.write_table(empty_table, final_path, compression="snappy", use_dictionary=True)

    files = {
        str(year): {
            "file": str(final_paths[year]),
            "rows": rows_by_year[str(year)],
            "file_size_bytes": final_paths[year].stat().st_size,
        }
        for year in range(START_YEAR, END_YEAR + 1)
    }

    return {
        "created_at": date.today().isoformat(),
        "source_file": str(input_path),
        "output_dir": str(output_dir),
        "format": "Parquet",
        "compression": "snappy",
        "raw_columns_preserved": True,
        "columns_as_strings": True,
        "cleaning_or_normalization_applied": False,
        "partitioning": "one file per dateparution year",
        "period": {
            "start": f"{START_YEAR}-01-01",
            "end_exclusive": f"{END_YEAR + 1}-01-01",
        },
        "source_rows": source_rows,
        "written_rows": written_rows,
        "rows_by_year": rows_by_year,
        "missing_dateparution": missing_dateparution,
        "outside_historical_range": outside_historical_range,
        "min_dateparution": min_dateparution,
        "max_dateparution": max_dateparution,
        "field_count": len(schema.names) if schema else 0,
        "files": files,
    }


def validate_parquet_files(summary: dict[str, Any]) -> dict[str, Any]:
    parquet_rows_by_year = {}
    parquet_total_rows = 0
    schema_names: list[str] | None = None
    schema_mismatches: list[str] = []

    for year, payload in summary["files"].items():
        path = Path(payload["file"])
        parquet_file = pq.ParquetFile(path)
        parquet_rows_by_year[year] = parquet_file.metadata.num_rows
        parquet_total_rows += parquet_file.metadata.num_rows
        names = parquet_file.schema_arrow.names
        if schema_names is None:
            schema_names = names
        elif names != schema_names:
            schema_mismatches.append(year)

    return {
        "parquet_rows_by_year": parquet_rows_by_year,
        "parquet_total_rows": parquet_total_rows,
        "matches_written_rows": parquet_total_rows == summary["written_rows"],
        "schema_mismatch_years": schema_mismatches,
    }


def save_summary(metadata_dir: Path, summary: dict[str, Any]) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / "boamp_grand_ouest_annual_parquet_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create annual Parquet files from the BOAMP Grand Ouest raw CSV.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV. Defaults to data/raw/boamp/boamp_2015_2025_grand_ouest_raw.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to data/raw/boamp/grand_ouest_annual_parquet.",
    )
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE, help="Rows per CSV chunk.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing annual Parquet files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    configure_logging(project_root / "logs")

    input_path = args.input or project_root / RAW_RELATIVE_DIR / DEFAULT_INPUT_NAME
    output_dir = args.output_dir or project_root / DEFAULT_OUTPUT_DIR
    if not input_path.is_absolute():
        input_path = project_root / input_path
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    existing = [annual_path(output_dir, year) for year in range(START_YEAR, END_YEAR + 1) if annual_path(output_dir, year).exists()]
    if existing and not args.force:
        raise FileExistsError("Annual Parquet files already exist. Use --force to rebuild them.")

    summary = create_annual_parquet(input_path, output_dir, args.chunksize)
    summary["validation"] = validate_parquet_files(summary)
    save_summary(project_root / METADATA_RELATIVE_DIR, summary)

    print()
    print("BOAMP Grand Ouest annual Parquet")
    print(f"Output dir: {summary['output_dir']}")
    print(f"Source rows: {summary['source_rows']:,}")
    print(f"Written rows: {summary['written_rows']:,}")
    print(f"Fields: {summary['field_count']}")
    print(f"Min dateparution: {summary['min_dateparution']}")
    print(f"Max dateparution: {summary['max_dateparution']}")
    print(f"Outside historical range: {summary['outside_historical_range']:,}")
    print(f"Parquet row validation passed: {summary['validation']['matches_written_rows']}")
    for year, rows in summary["rows_by_year"].items():
        print(f"{year}: {rows:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
