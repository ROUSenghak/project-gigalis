#!/usr/bin/env python3
"""Create a raw BOAMP Grand Ouest CSV from the canonical raw BOAMP CSV.

This script keeps every original source column. It only filters rows whose
publication or prestation department belongs to Grand Ouest:
Bretagne, Pays de la Loire, or Normandie.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


RAW_RELATIVE_DIR = Path("data/raw/boamp")
METADATA_RELATIVE_DIR = Path("data/metadata")
DEFAULT_INPUT_NAME = "boamp_2015_2025_raw.csv"
DEFAULT_OUTPUT_NAME = "boamp_2015_2025_grand_ouest_raw.csv"
CHUNKSIZE = 50_000
PROGRESS_INTERVAL_SECONDS = 30

GRAND_OUEST_REGION_BY_DEPARTMENT = {
    "22": "Bretagne",
    "29": "Bretagne",
    "35": "Bretagne",
    "56": "Bretagne",
    "44": "Pays de la Loire",
    "49": "Pays de la Loire",
    "53": "Pays de la Loire",
    "72": "Pays de la Loire",
    "85": "Pays de la Loire",
    "14": "Normandie",
    "27": "Normandie",
    "50": "Normandie",
    "61": "Normandie",
    "76": "Normandie",
}
GRAND_OUEST_DEPARTMENTS = set(GRAND_OUEST_REGION_BY_DEPARTMENT)
GRAND_OUEST_LABEL = "Grand Ouest: Bretagne + Pays de la Loire + Normandie"


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "create_boamp_grand_ouest_raw_csv.log", encoding="utf-8"),
        ],
    )


def normalize_department_code(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None

    upper_text = text.upper()
    if upper_text in {"2A", "2B"}:
        return upper_text

    match = re.search(r"\d+", text)
    if not match:
        return None

    number = int(match.group(0))
    return f"{number:02d}" if number < 100 else str(number)


def departments_from_publication_cell(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        values = parsed
    else:
        values = re.findall(r"2A|2B|\d+", text.upper())

    departments: list[str] = []
    for item in values:
        department = normalize_department_code(item)
        if department:
            departments.append(department)
    return departments


def first_grand_ouest_department(prestation_value: Any, publication_value: Any) -> str | None:
    prestation = normalize_department_code(prestation_value)
    if prestation in GRAND_OUEST_DEPARTMENTS:
        return prestation

    for department in departments_from_publication_cell(publication_value):
        if department in GRAND_OUEST_DEPARTMENTS:
            return department
    return None


def build_grand_ouest_mask(chunk: pd.DataFrame) -> pd.Series:
    department_series = pd.Series(
        [
            first_grand_ouest_department(prestation, publication)
            for prestation, publication in zip(
                chunk["code_departement_prestation"],
                chunk["code_departement"],
                strict=False,
            )
        ],
        index=chunk.index,
        dtype="string",
    )
    return department_series.notna()


def create_extract(input_path: Path, output_path: Path, chunksize: int) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    source_rows = 0
    output_rows = 0
    chunk_count = 0
    header_written = False
    started = time.monotonic()
    last_log = started

    yearly_counts: dict[str, int] = {}
    region_counts: dict[str, int] = {}
    department_counts: dict[str, int] = {}
    idweb_seen: set[str] = set()
    duplicate_idweb = 0
    missing_idweb = 0
    min_dateparution: str | None = None
    max_dateparution: str | None = None
    outside_historical_range = 0

    for chunk in pd.read_csv(input_path, chunksize=chunksize, dtype=str, keep_default_na=False):
        chunk_count += 1
        source_rows += len(chunk)
        mask = build_grand_ouest_mask(chunk)
        filtered = chunk.loc[mask]

        if not filtered.empty:
            filtered.to_csv(
                temp_path,
                mode="a",
                index=False,
                header=not header_written,
                encoding="utf-8",
                lineterminator="\n",
            )
            header_written = True
            output_rows += len(filtered)

            departments = [
                first_grand_ouest_department(prestation, publication)
                for prestation, publication in zip(
                    filtered["code_departement_prestation"],
                    filtered["code_departement"],
                    strict=False,
                )
            ]
            for department in departments:
                if not department:
                    continue
                department_counts[department] = department_counts.get(department, 0) + 1
                region = GRAND_OUEST_REGION_BY_DEPARTMENT[department]
                region_counts[region] = region_counts.get(region, 0) + 1

            for raw_idweb in filtered["idweb"]:
                idweb = str(raw_idweb).strip()
                if not idweb:
                    missing_idweb += 1
                elif idweb in idweb_seen:
                    duplicate_idweb += 1
                else:
                    idweb_seen.add(idweb)

            dates = filtered["dateparution"].astype(str)
            for dateparution in dates:
                if not dateparution:
                    outside_historical_range += 1
                    continue
                if min_dateparution is None or dateparution < min_dateparution:
                    min_dateparution = dateparution
                if max_dateparution is None or dateparution > max_dateparution:
                    max_dateparution = dateparution
                if not ("2015-01-01" <= dateparution < "2026-01-01"):
                    outside_historical_range += 1
                year = dateparution[:4]
                yearly_counts[year] = yearly_counts.get(year, 0) + 1

        now = time.monotonic()
        if now - last_log >= PROGRESS_INTERVAL_SECONDS:
            logging.info(
                "Scanned %s source rows, wrote %s Grand Ouest rows in %.1f minutes",
                f"{source_rows:,}",
                f"{output_rows:,}",
                (now - started) / 60,
            )
            last_log = now

    if not header_written:
        pd.read_csv(input_path, nrows=0).to_csv(temp_path, index=False, encoding="utf-8", lineterminator="\n")

    temp_path.replace(output_path)
    return {
        "created_at": date.today().isoformat(),
        "definition": GRAND_OUEST_LABEL,
        "departments": sorted(GRAND_OUEST_DEPARTMENTS),
        "source_file": str(input_path),
        "output_file": str(output_path),
        "source_rows_scanned": source_rows,
        "rows": output_rows,
        "rows_by_year": dict(sorted(yearly_counts.items())),
        "rows_by_region": dict(sorted(region_counts.items())),
        "rows_by_department": dict(sorted(department_counts.items())),
        "unique_idweb": len(idweb_seen),
        "duplicate_idweb": duplicate_idweb,
        "missing_idweb": missing_idweb,
        "min_dateparution": min_dateparution,
        "max_dateparution": max_dateparution,
        "outside_historical_range": outside_historical_range,
        "field_count": len(pd.read_csv(output_path, nrows=0).columns),
        "file_size_bytes": output_path.stat().st_size,
        "filter": {
            "kept_all_source_columns": True,
            "cleaning_or_normalization_applied": False,
            "row_selection": "code_departement_prestation in Grand Ouest OR code_departement contains a Grand Ouest department",
        },
    }


def save_summary(metadata_dir: Path, summary: dict[str, Any]) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metadata_dir / "boamp_grand_ouest_raw_csv_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the BOAMP Grand Ouest raw CSV extract.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input canonical raw CSV. Defaults to data/raw/boamp/boamp_2015_2025_raw.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV. Defaults to data/raw/boamp/boamp_2015_2025_grand_ouest_raw.csv.",
    )
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE, help="Rows per read_csv chunk.")
    parser.add_argument("--force", action="store_true", help="Overwrite the output if it already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    configure_logging(project_root / "logs")

    input_path = args.input or project_root / RAW_RELATIVE_DIR / DEFAULT_INPUT_NAME
    output_path = args.output or project_root / RAW_RELATIVE_DIR / DEFAULT_OUTPUT_NAME
    if not input_path.is_absolute():
        input_path = project_root / input_path
    if not output_path.is_absolute():
        output_path = project_root / output_path

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} already exists. Use --force to rebuild it.")

    summary = create_extract(input_path, output_path, args.chunksize)
    save_summary(project_root / METADATA_RELATIVE_DIR, summary)

    print()
    print("BOAMP Grand Ouest raw CSV")
    print(f"Output: {summary['output_file']}")
    print(f"Rows: {summary['rows']:,}")
    print(f"Fields: {summary['field_count']}")
    print(f"Unique idweb: {summary['unique_idweb']:,}")
    print(f"Duplicate idweb: {summary['duplicate_idweb']:,}")
    print(f"Missing idweb: {summary['missing_idweb']:,}")
    print(f"Min dateparution: {summary['min_dateparution']}")
    print(f"Max dateparution: {summary['max_dateparution']}")
    print(f"Outside historical range: {summary['outside_historical_range']:,}")
    print(f"File size: {summary['file_size_bytes'] / 1024 / 1024 / 1024:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
