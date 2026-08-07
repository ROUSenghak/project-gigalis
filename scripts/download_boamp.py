#!/usr/bin/env python3
"""Download raw BOAMP notices from Opendatasoft, partitioned by publication year.

This is an acquisition-only script. It preserves source records exactly as
returned by the BOAMP JSONL export endpoint and filters only on dateparution.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from requests.exceptions import ChunkedEncodingError, ContentDecodingError
from urllib3.util.retry import Retry


BASE_URL = "https://boamp-datadila.opendatasoft.com/api/explore/v2.1"
DATASET_ID = "boamp"
START_YEAR = 2015
END_YEAR = 2025
CURRENT_PARTIAL_YEAR = 2026
REQUEST_TIMEOUT = (10, 120)
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "project-gigalis-boamp-downloader/1.0"

RAW_RELATIVE_DIR = Path("data/raw/boamp")
METADATA_RELATIVE_DIR = Path("data/metadata")


@dataclass
class YearSummary:
    year: int | str
    start_date: str
    end_exclusive: str | None
    partial: bool
    api_count: int
    downloaded_count: int
    unique_idweb: int
    duplicates: int
    missing_idweb: int
    min_dateparution: str | None
    max_dateparution: str | None
    outside_expected_range: int
    file_size_bytes: int
    file: str
    completed_marker: str
    validation_passed: bool


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "download_boamp.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def build_session(max_retries: int) -> requests.Session:
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": USER_AGENT,
        }
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    max_attempts: int = 5,
) -> dict[str, Any]:
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code in {429, 500, 502, 503, 504}:
                sleep_for = retry_sleep_seconds(response, attempt)
                logging.warning(
                    "HTTP %s from %s; retrying in %.1fs",
                    response.status_code,
                    response.url,
                    sleep_for,
                )
                time.sleep(sleep_for)
                continue
            response.raise_for_status()
            return response.json()
        except (
            requests.Timeout,
            requests.ConnectionError,
            ChunkedEncodingError,
            ContentDecodingError,
        ) as exc:
            if attempt == max_attempts:
                raise
            sleep_for = min(2**attempt, 60)
            logging.warning("%s; retrying in %.1fs", exc, sleep_for)
            time.sleep(sleep_for)
    raise RuntimeError(f"Failed to retrieve JSON from {url}")


def retry_sleep_seconds(response: Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), 300.0)
        except ValueError:
            pass
    return min(2**attempt, 300.0)


def dataset_url(*parts: str) -> str:
    joined = "/".join(part.strip("/") for part in parts)
    return f"{BASE_URL}/catalog/datasets/{DATASET_ID}/{joined}" if joined else f"{BASE_URL}/catalog/datasets/{DATASET_ID}"


def date_where(start_date: str, end_exclusive: str | None) -> str:
    if end_exclusive is None:
        return f"dateparution >= date'{start_date}'"
    return f"dateparution >= date'{start_date}' AND dateparution < date'{end_exclusive}'"


def api_count(session: requests.Session, start_date: str, end_exclusive: str | None) -> int:
    payload = request_json(
        session,
        dataset_url("records"),
        params={"limit": 0, "where": date_where(start_date, end_exclusive)},
    )
    return int(payload["total_count"])


def save_metadata(session: requests.Session, metadata_dir: Path) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = request_json(session, dataset_url())
    exports = request_json(session, dataset_url("exports"))

    (metadata_dir / "boamp_metadata.json").write_text(
        json.dumps(
            {
                "retrieved_at": date.today().isoformat(),
                "base_url": BASE_URL,
                "dataset_id": DATASET_ID,
                "dataset": metadata,
                "exports": exports,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (metadata_dir / "boamp_fields.json").write_text(
        json.dumps(metadata.get("fields", []), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def stream_download(
    session: requests.Session,
    output_path: Path,
    start_date: str,
    end_exclusive: str | None,
    max_attempts: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    params = {
        "where": date_where(start_date, end_exclusive),
        "order_by": "dateparution asc, idweb asc",
    }
    export_url = f"{dataset_url('exports', 'jsonl')}?{urlencode(params)}"

    for attempt in range(1, max_attempts + 1):
        try:
            with session.get(export_url, stream=True, timeout=REQUEST_TIMEOUT) as response:
                if response.status_code in {429, 500, 502, 503, 504}:
                    sleep_for = retry_sleep_seconds(response, attempt)
                    logging.warning(
                        "HTTP %s while streaming %s; retrying in %.1fs",
                        response.status_code,
                        output_path.name,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    continue
                response.raise_for_status()

                bytes_written = 0
                records_seen = 0
                last_log = time.monotonic()
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        bytes_written += len(chunk)
                        records_seen += chunk.count(b"\n")
                        now = time.monotonic()
                        if now - last_log >= 30:
                            logging.info(
                                "%s: streamed %s records, %.1f MB",
                                output_path.name,
                                records_seen,
                                bytes_written / 1024 / 1024,
                            )
                            last_log = now

                temp_path.replace(output_path)
                logging.info(
                    "%s: download complete, %.1f MB",
                    output_path.name,
                    output_path.stat().st_size / 1024 / 1024,
                )
                return
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == max_attempts:
                raise
            sleep_for = min(2**attempt, 300)
            logging.warning("%s while streaming; retrying in %.1fs", exc, sleep_for)
            time.sleep(sleep_for)

    raise RuntimeError(f"Failed to download {output_path}")


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


def validate_file(
    path: Path,
    year_label: int | str,
    start_date: str,
    end_exclusive: str | None,
    partial: bool,
    expected_count: int,
    marker_path: Path,
) -> YearSummary:
    downloaded_count = 0
    missing_idweb = 0
    outside_expected_range = 0
    min_dateparution: str | None = None
    max_dateparution: str | None = None
    seen_idweb: set[str] = set()
    duplicate_idweb = 0

    for record in iter_jsonl(path):
        downloaded_count += 1
        idweb = record.get("idweb")
        if idweb is None or idweb == "":
            missing_idweb += 1
        elif idweb in seen_idweb:
            duplicate_idweb += 1
        else:
            seen_idweb.add(idweb)

        dateparution = record.get("dateparution")
        if dateparution is None:
            outside_expected_range += 1
            continue

        if min_dateparution is None or dateparution < min_dateparution:
            min_dateparution = dateparution
        if max_dateparution is None or dateparution > max_dateparution:
            max_dateparution = dateparution

        in_range = dateparution >= start_date and (
            end_exclusive is None or dateparution < end_exclusive
        )
        if not in_range:
            outside_expected_range += 1

    validation_passed = (
        downloaded_count == expected_count and outside_expected_range == 0
    )
    summary = YearSummary(
        year=year_label,
        start_date=start_date,
        end_exclusive=end_exclusive,
        partial=partial,
        api_count=expected_count,
        downloaded_count=downloaded_count,
        unique_idweb=len(seen_idweb),
        duplicates=duplicate_idweb,
        missing_idweb=missing_idweb,
        min_dateparution=min_dateparution,
        max_dateparution=max_dateparution,
        outside_expected_range=outside_expected_range,
        file_size_bytes=path.stat().st_size,
        file=str(path),
        completed_marker=str(marker_path),
        validation_passed=validation_passed,
    )
    print_year_summary(summary)
    return summary


def print_year_summary(summary: YearSummary) -> None:
    size_mb = summary.file_size_bytes / 1024 / 1024
    print()
    print(f"BOAMP {summary.year}")
    print()
    print(f"API expected records: {summary.api_count}")
    print(f"Downloaded records:   {summary.downloaded_count}")
    print(f"Unique idweb:         {summary.unique_idweb}")
    print(f"Duplicate idweb:      {summary.duplicates}")
    print(f"Missing idweb:        {summary.missing_idweb}")
    print(f"Min dateparution:     {summary.min_dateparution}")
    print(f"Max dateparution:     {summary.max_dateparution}")
    print(f"Outside range:        {summary.outside_expected_range}")
    print(f"File size:            {size_mb:.1f} MB")
    print()
    print(f"VALIDATION: {'PASSED' if summary.validation_passed else 'FAILED'}")


def marker_for(raw_dir: Path, year_label: int | str) -> Path:
    return raw_dir / f".completed_{year_label}"


def output_for(raw_dir: Path, year: int, partial: bool = False) -> Path:
    suffix = "_partial" if partial else ""
    return raw_dir / f"boamp_{year}{suffix}.jsonl"


def process_year(
    session: requests.Session,
    raw_dir: Path,
    year: int,
    start_date: str,
    end_exclusive: str | None,
    partial: bool,
    force: bool,
    max_attempts: int,
) -> YearSummary:
    year_label: int | str = f"{year}_partial" if partial else year
    output_path = output_for(raw_dir, year, partial=partial)
    marker_path = marker_for(raw_dir, year_label)
    expected_count = api_count(session, start_date, end_exclusive)

    if marker_path.exists() and output_path.exists() and not force:
        logging.info("%s already completed; validating existing file", year_label)
    else:
        if force:
            logging.info("%s: force enabled; redownloading", year_label)
        else:
            logging.info("%s: downloading %s", year_label, output_path)
        stream_download(
            session,
            output_path,
            start_date=start_date,
            end_exclusive=end_exclusive,
            max_attempts=max_attempts,
        )

    summary = validate_file(
        output_path,
        year_label=year_label,
        start_date=start_date,
        end_exclusive=end_exclusive,
        partial=partial,
        expected_count=expected_count,
        marker_path=marker_path,
    )
    if not summary.validation_passed:
        if marker_path.exists():
            marker_path.unlink()
        raise RuntimeError(f"Validation failed for BOAMP {year_label}")

    marker_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_jobs(include_2026: bool) -> list[tuple[int, str, str | None, bool]]:
    jobs: list[tuple[int, str, str | None, bool]] = []
    for year in range(START_YEAR, END_YEAR + 1):
        jobs.append((year, f"{year}-01-01", f"{year + 1}-01-01", False))
    if include_2026:
        jobs.append((CURRENT_PARTIAL_YEAR, "2026-01-01", None, True))
    return jobs


def save_summary(
    metadata_dir: Path,
    summaries: list[YearSummary],
    include_2026: bool,
) -> None:
    historical = [s for s in summaries if not s.partial]
    total_api = sum(s.api_count for s in historical)
    total_downloaded = sum(s.downloaded_count for s in historical)
    global_min = min((s.min_dateparution for s in historical if s.min_dateparution), default=None)
    global_max = max((s.max_dateparution for s in historical if s.max_dateparution), default=None)

    payload = {
        "dataset_id": DATASET_ID,
        "source_api": BASE_URL,
        "endpoint": dataset_url("exports", "jsonl"),
        "retrieved_at": date.today().isoformat(),
        "period": {
            "start": "2015-01-01",
            "end_exclusive": "2026-01-01",
        },
        "include_2026_partial": include_2026,
        "global_validation": {
            "historical_api_count": total_api,
            "historical_downloaded_count": total_downloaded,
            "counts_match": total_api == total_downloaded,
            "min_dateparution": global_min,
            "max_dateparution": global_max,
            "minimum_date_ok": global_min is None or global_min >= "2015-01-01",
            "maximum_date_ok": global_max is None or global_max < "2026-01-01",
            "outside_expected_range": sum(s.outside_expected_range for s in historical),
            "historical_validation_passed": all(s.validation_passed for s in historical)
            and total_api == total_downloaded
            and (global_min is None or global_min >= "2015-01-01")
            and (global_max is None or global_max < "2026-01-01"),
        },
        "years": {str(s.year): asdict(s) for s in summaries},
    }
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "boamp_download_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_global_summary(summaries: list[YearSummary]) -> None:
    historical = [s for s in summaries if not s.partial]
    print()
    print("BOAMP historical summary")
    print()
    print(f"{'Year':<8}{'API count':>12}{'Downloaded':>14}{'Unique IDs':>14}{'Duplicates':>14}")
    for summary in historical:
        print(
            f"{summary.year:<8}{summary.api_count:>12}"
            f"{summary.downloaded_count:>14}"
            f"{summary.unique_idweb:>14}"
            f"{summary.duplicates:>14}"
        )
    print(
        f"{'TOTAL':<8}{sum(s.api_count for s in historical):>12}"
        f"{sum(s.downloaded_count for s in historical):>14}"
        f"{sum(s.unique_idweb for s in historical):>14}"
        f"{sum(s.duplicates for s in historical):>14}"
    )

    global_min = min((s.min_dateparution for s in historical if s.min_dateparution), default=None)
    global_max = max((s.max_dateparution for s in historical if s.max_dateparution), default=None)
    print()
    print(f"Historical min dateparution: {global_min}")
    print(f"Historical max dateparution: {global_max}")
    print("Historical interval check: 2015-01-01 <= dateparution < 2026-01-01")
    print(
        "Historical validation: "
        + (
            "PASSED"
            if global_min is not None
            and global_min >= "2015-01-01"
            and global_max is not None
            and global_max < "2026-01-01"
            and all(s.validation_passed for s in historical)
            else "FAILED"
        )
    )


def inspect_api(session: requests.Session) -> dict[str, Any]:
    metadata = request_json(session, dataset_url())
    exports = request_json(session, dataset_url("exports"))
    sample = request_json(
        session,
        dataset_url("records"),
        params={
            "limit": 10,
            "where": date_where("2015-01-01", "2026-01-01"),
            "order_by": "dateparution asc",
        },
    )
    counts = {
        str(year): api_count(session, f"{year}-01-01", f"{year + 1}-01-01")
        for year in range(START_YEAR, END_YEAR + 1)
    }
    return {
        "dataset_id": DATASET_ID,
        "metadata_records_count": metadata.get("metas", {})
        .get("default", {})
        .get("records_count"),
        "fields": metadata.get("fields", []),
        "dateparution_type": next(
            (field.get("type") for field in metadata.get("fields", []) if field.get("name") == "dateparution"),
            None,
        ),
        "exports": exports,
        "sample_total_count": sample.get("total_count"),
        "sample_result_count": len(sample.get("results", [])),
        "sample_dates": [record.get("dateparution") for record in sample.get("results", [])],
        "year_counts": counts,
        "historical_count": sum(counts.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw BOAMP notices for 2015-2025 by publication year."
    )
    parser.add_argument(
        "--include-2026",
        action="store_true",
        help="Also download 2026 as data/raw/boamp/boamp_2026_partial.jsonl.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload years even when completed markers already exist.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Project/output root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts for transient HTTP/network failures.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect metadata, exports, sample records, and counts without downloading raw files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.output_root.resolve()
    raw_dir = project_root / RAW_RELATIVE_DIR
    metadata_dir = project_root / METADATA_RELATIVE_DIR
    configure_logging(project_root / "logs")

    session = build_session(args.max_retries)
    logging.info("Inspecting BOAMP metadata and export endpoints")
    save_metadata(session, metadata_dir)

    if args.inspect_only:
        inspection = inspect_api(session)
        (metadata_dir / "boamp_api_inspection.json").write_text(
            json.dumps(inspection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        return 0

    summaries: list[YearSummary] = []
    for year, start_date, end_exclusive, partial in build_jobs(args.include_2026):
        summaries.append(
            process_year(
                session,
                raw_dir=raw_dir,
                year=year,
                start_date=start_date,
                end_exclusive=end_exclusive,
                partial=partial,
                force=args.force,
                max_attempts=args.max_retries,
            )
        )

    print_global_summary(summaries)
    save_summary(metadata_dir, summaries, include_2026=args.include_2026)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
