#!/usr/bin/env python3
"""Build one canonical raw CSV from yearly BOAMP JSONL files.

This script does not clean, deduplicate, normalize, or filter the records. It
only serializes the already acquired raw JSONL rows into one CSV with a stable
field order from BOAMP metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Iterable


START_YEAR = 2015
END_YEAR = 2025
RAW_RELATIVE_DIR = Path("data/raw/boamp")
METADATA_RELATIVE_DIR = Path("data/metadata")
DEFAULT_OUTPUT_NAME = "boamp_2015_2025_raw.csv"
PROGRESS_INTERVAL_SECONDS = 30


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "build_boamp_raw_csv.log", encoding="utf-8"),
        ],
    )


def load_fieldnames(metadata_dir: Path) -> list[str]:
    fields_path = metadata_dir / "boamp_fields.json"
    fields = json.loads(fields_path.read_text(encoding="utf-8"))
    fieldnames = [field["name"] for field in fields]
    if not fieldnames:
        raise ValueError(f"No fields found in {fields_path}")
    return fieldnames


def jsonl_files(raw_dir: Path, start_year: int, end_year: int) -> list[Path]:
    files = [raw_dir / f"boamp_{year}.jsonl" for year in range(start_year, end_year + 1)]
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing BOAMP JSONL files: " + ", ".join(str(p) for p in missing))
    return files


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def scan_for_unexpected_fields(files: list[Path], fieldnames: list[str]) -> dict[str, int]:
    expected = set(fieldnames)
    unexpected: dict[str, int] = {}
    for path in files:
        logging.info("Scanning fields in %s", path)
        for record in iter_jsonl(path):
            for key in record:
                if key not in expected:
                    unexpected[key] = unexpected.get(key, 0) + 1
    return unexpected


def write_csv(
    files: list[Path],
    output_path: Path,
    fieldnames: list[str],
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    total_rows = 0
    min_dateparution: str | None = None
    max_dateparution: str | None = None
    rows_outside_historical_range = 0
    rows_missing_dateparution = 0
    rows_by_year: dict[str, int] = {str(year): 0 for year in range(START_YEAR, END_YEAR + 1)}
    started = time.monotonic()
    last_log = started

    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()

        for path in files:
            logging.info("Writing rows from %s", path)
            year = path.stem.rsplit("_", 1)[-1]
            for record in iter_jsonl(path):
                dateparution = record.get("dateparution")
                if dateparution is None:
                    rows_missing_dateparution += 1
                    rows_outside_historical_range += 1
                else:
                    if min_dateparution is None or dateparution < min_dateparution:
                        min_dateparution = dateparution
                    if max_dateparution is None or dateparution > max_dateparution:
                        max_dateparution = dateparution
                    if not ("2015-01-01" <= dateparution < "2026-01-01"):
                        rows_outside_historical_range += 1

                writer.writerow({name: csv_cell(record.get(name)) for name in fieldnames})
                total_rows += 1
                rows_by_year[year] = rows_by_year.get(year, 0) + 1

                now = time.monotonic()
                if now - last_log >= PROGRESS_INTERVAL_SECONDS:
                    elapsed = now - started
                    logging.info(
                        "Wrote %s rows in %.1f minutes to %s",
                        total_rows,
                        elapsed / 60,
                        output_path,
                    )
                    last_log = now

    temp_path.replace(output_path)
    return {
        "output_file": str(output_path),
        "rows": total_rows,
        "rows_by_year": rows_by_year,
        "field_count": len(fieldnames),
        "fields": fieldnames,
        "min_dateparution": min_dateparution,
        "max_dateparution": max_dateparution,
        "rows_missing_dateparution": rows_missing_dateparution,
        "rows_outside_historical_range": rows_outside_historical_range,
        "file_size_bytes": output_path.stat().st_size,
    }


def save_summary(metadata_dir: Path, summary: dict[str, Any], source_files: list[Path]) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": date.today().isoformat(),
        "source_files": [str(path) for path in source_files],
        "serialization": {
            "format": "CSV UTF-8",
            "one_row_per_raw_boamp_record": True,
            "null_values": "empty CSV cell",
            "list_or_object_values": "compact JSON string inside CSV cell",
            "cleaning_or_normalization_applied": False,
        },
        **summary,
    }
    (metadata_dir / "boamp_raw_csv_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine BOAMP yearly raw JSONL files into one canonical raw CSV."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Project/output root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to data/raw/boamp/boamp_2015_2025_raw.csv.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the CSV if it already exists.",
    )
    parser.add_argument(
        "--skip-field-scan",
        action="store_true",
        help="Skip the preflight unexpected-field scan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.output_root.resolve()
    raw_dir = project_root / RAW_RELATIVE_DIR
    metadata_dir = project_root / METADATA_RELATIVE_DIR
    configure_logging(project_root / "logs")

    output_path = args.output or raw_dir / DEFAULT_OUTPUT_NAME
    if not output_path.is_absolute():
        output_path = project_root / output_path
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} already exists. Use --force to rebuild it.")

    fieldnames = load_fieldnames(metadata_dir)
    files = jsonl_files(raw_dir, START_YEAR, END_YEAR)

    if not args.skip_field_scan:
        unexpected = scan_for_unexpected_fields(files, fieldnames)
        if unexpected:
            raise ValueError(f"Unexpected fields found; refusing to drop raw fields: {unexpected}")

    summary = write_csv(files, output_path, fieldnames)
    save_summary(metadata_dir, summary, files)

    print()
    print("BOAMP canonical raw CSV")
    print(f"Output: {summary['output_file']}")
    print(f"Rows: {summary['rows']}")
    print(f"Fields: {summary['field_count']}")
    print(f"Min dateparution: {summary['min_dateparution']}")
    print(f"Max dateparution: {summary['max_dateparution']}")
    print(f"Outside historical range: {summary['rows_outside_historical_range']}")
    print(f"File size: {summary['file_size_bytes'] / 1024 / 1024 / 1024:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
