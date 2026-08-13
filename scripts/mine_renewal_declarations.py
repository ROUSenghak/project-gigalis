#!/usr/bin/env python3
"""Mine notices that declare, in their own words, that they replace a contract.

BOAMP has no renewal field, so the most credible evidence available is a buyer
stating the fact. Crucially that statement sits on the *successor* notice
("le marche actuel arrive a echeance le 15 juillet 2024"), which is why this
runs successor-to-predecessor -- the opposite direction to the study's
anchor-to-successor linkage, and the reason the resulting labels are not a
restatement of what the linkage scorer already computes.

The scan is a single batched pass over all 1,620,712 standardised notices. A
cheap stem prefilter runs first; the full pattern battery only runs on notices
that survive it.

What this stage does **not** do is label anything. It records that a notice
claims a predecessor exists, with the exact text that says so. Resolving
*which* earlier procurement is meant happens in
``resolve_declared_predecessors.py``, and even that is only a recruitment
proposal -- every pair still goes to blind annotation and can be rejected.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import replace  # noqa: E402

from boamp_pipeline.renewal_language import (  # noqa: E402
    CONTRACT_HEAD_WINDOW,
    EXCLUDED_ADVISORY_MISSION,
    EXPIRY_DECLARATION,
    POSITIVE_CLASSES,
    RETENDER_AFTER_FAILURE,
    declaration_strength,
    extract_dates,
    extract_references,
    find_matches,
    objet_is_advisory,
)

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp_v2/benchmark_v3")
MINING_VERSION = "boamp_renewal_declarations_v1.0"

NOTICE_BATCH = 100_000
SEARCH_FIELDS = ("objet", "notice_text")

#: Cheap stem filter. Anything matching goes to the full battery; anything not
#: matching cannot possibly hit a declaration pattern. Expect roughly a 5% pass
#: rate, which is what makes a full-corpus scan affordable.
PREFILTER = re.compile(
    r"renouvel|[ée]ch[ée]ance|arrivant\s+[àa]|arrive\s+[àa]|prend\s+fin|prendra\s+fin"
    r"|expire|[ée]chu|sortant|pr[ée]c[ée]dent|ancien|actuel|en\s+cours|en\s+place"
    r"|reprise\s+d[ue]\s+personnel|relanc|infructueu|sans\s+suite|remise\s+en\s+concurrence"
    r"|nouvelle\s+consultation|nouveau\s+march|continuit[ée]",
    re.IGNORECASE,
)

DECLARATION_SCHEMA = pa.schema([
    pa.field("idweb", pa.string()),
    pa.field("episode_id", pa.string()),
    pa.field("publication_date", pa.date32()),
    pa.field("nature", pa.string()),
    pa.field("source_schema", pa.string()),
    pa.field("buyer_key", pa.string()),
    pa.field("buyer_name_normalized", pa.string()),
    pa.field("buyer_department", pa.string()),
    pa.field("all_cpvs_json", pa.string()),
    pa.field("procedure_reference", pa.string()),
    pa.field("pattern_name", pa.string()),
    pa.field("pattern_class", pa.string()),
    pa.field("field", pa.string()),
    pa.field("match_start", pa.int32()),
    pa.field("match_end", pa.int32()),
    pa.field("matched_text", pa.string()),
    pa.field("evidence_snippet", pa.string()),
    pa.field("excluded_reason", pa.string()),
    pa.field("declared_end_date", pa.date32()),
    pa.field("declared_end_date_precision", pa.string()),
    pa.field("declared_end_date_snippet", pa.string()),
    pa.field("declared_reference", pa.string()),
    pa.field("notice_declaration_strength", pa.int32()),
    pa.field("mining_version", pa.string()),
])

NOTICE_COLUMNS = [
    "idweb", "publication_date", "nature", "source_schema", "buyer_key",
    "buyer_name_normalized", "buyer_department", "all_cpvs_json",
    "procedure_reference", "objet", "notice_text",
]


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "mine_renewal_declarations.log", encoding="utf-8"),
        ],
    )


def load_episode_by_idweb(processed_dir: Path) -> dict[str, str]:
    membership = pd.read_parquet(
        processed_dir / "episode_membership.parquet", columns=["idweb", "episode_id"]
    )
    return dict(zip(membership["idweb"].tolist(), membership["episode_id"].tolist()))


def declaration_rows(
    notice: dict[str, Any],
    episode_id: str,
    window: int,
    counters: Counter[str],
) -> list[dict[str, Any]]:
    """All declaration matches in one notice, with predecessor attributes."""
    matches = []
    texts: dict[str, str] = {}
    for field in SEARCH_FIELDS:
        text = notice.get(field)
        if not text:
            continue
        texts[field] = str(text)
        matches.extend(find_matches(texts[field], field, window=window))
    if not matches:
        return []

    # A notice whose title shows it is an advisory mission has every match
    # reattributed, because the per-match guard only sees one sentence and a
    # trigger further down the body would otherwise survive on its own.
    if objet_is_advisory(notice.get("objet")):
        matches = [
            match if match.is_excluded
            else replace(match, excluded_reason=EXCLUDED_ADVISORY_MISSION)
            for match in matches
        ]
        counters["notice_level_advisory_reattribution"] += 1

    kept = [match for match in matches if not match.is_excluded]
    for match in matches:
        if match.is_excluded:
            counters[f"excluded_{match.excluded_reason}"] += 1
    strength = declaration_strength(kept)

    rows: list[dict[str, Any]] = []
    for match in matches:
        text = texts[match.field]
        declared_end = None
        precision = ""
        date_snippet = ""
        # A date is only bound to an expiry claim, and only when it sits inside
        # the same sentence near the trigger. An unbound date elsewhere in a
        # notice says nothing about the predecessor's term.
        if match.klass == EXPIRY_DECLARATION and not match.is_excluded:
            dates = extract_dates(text, near=match.end)
            if dates:
                chosen = dates[0]
                declared_end = chosen.value
                precision = chosen.precision
                date_snippet = chosen.snippet
        references = extract_references(text, near=match.end)
        rows.append({
            "idweb": notice["idweb"],
            "episode_id": episode_id,
            "publication_date": notice["publication_date"],
            "nature": notice["nature"] or "",
            "source_schema": notice["source_schema"] or "",
            "buyer_key": notice["buyer_key"] or "",
            "buyer_name_normalized": notice["buyer_name_normalized"] or "",
            "buyer_department": notice["buyer_department"] or "",
            "all_cpvs_json": notice["all_cpvs_json"] or "[]",
            "procedure_reference": notice["procedure_reference"] or "",
            "pattern_name": match.pattern,
            "pattern_class": match.klass,
            "field": match.field,
            "match_start": int(match.start),
            "match_end": int(match.end),
            "matched_text": match.matched_text,
            "evidence_snippet": match.snippet,
            "excluded_reason": match.excluded_reason,
            "declared_end_date": declared_end,
            "declared_end_date_precision": precision,
            "declared_end_date_snippet": date_snippet,
            "declared_reference": references[0][0] if references else "",
            "notice_declaration_strength": int(strength),
            "mining_version": MINING_VERSION,
        })
    return rows


def build(project_root: Path, output_dir: Path, force: bool, window: int) -> dict[str, Any]:
    processed_dir = output_dir.parent
    declarations_path = output_dir / "renewal_declarations.parquet"
    summary_path = output_dir / "renewal_declaration_summary.json"
    if declarations_path.exists() and not force:
        raise FileExistsError(f"{declarations_path} already exists. Use --force to rebuild.")
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_by_idweb = load_episode_by_idweb(processed_dir)
    logging.info("Loaded %s notice-to-episode mappings", f"{len(episode_by_idweb):,}")

    temp_path = declarations_path.with_suffix(".parquet.part")
    if temp_path.exists():
        temp_path.unlink()
    writer = pq.ParquetWriter(temp_path, DECLARATION_SCHEMA, compression="zstd", use_dictionary=True)

    counters: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    class_notices: dict[str, set[str]] = defaultdict(set)
    field_counts: Counter[str] = Counter()
    scanned = prefiltered = 0
    rows_written = 0
    buffer: list[dict[str, Any]] = []

    parquet = pq.ParquetFile(processed_dir / "notices_standardized_2015_2025.parquet")
    try:
        for batch in parquet.iter_batches(batch_size=NOTICE_BATCH, columns=NOTICE_COLUMNS):
            notices = batch.to_pylist()
            for notice in notices:
                scanned += 1
                blob = f"{notice.get('objet') or ''}\n{notice.get('notice_text') or ''}"
                if not PREFILTER.search(blob):
                    continue
                prefiltered += 1
                episode_id = episode_by_idweb.get(notice["idweb"], "")
                rows = declaration_rows(notice, episode_id, window, counters)
                for row in rows:
                    pattern_counts[row["pattern_name"]] += int(not row["excluded_reason"])
                    if not row["excluded_reason"]:
                        class_notices[row["pattern_class"]].add(row["idweb"])
                        field_counts[row["field"]] += 1
                        if row["declared_end_date"] is not None:
                            counters["declared_end_dates"] += 1
                        if row["declared_reference"]:
                            counters["declared_references"] += 1
                buffer.extend(rows)
                rows_written += len(rows)
            if len(buffer) >= 20_000:
                writer.write_table(pa.Table.from_pylist(buffer, schema=DECLARATION_SCHEMA))
                buffer.clear()
            if scanned % 200_000 < NOTICE_BATCH:
                logging.info(
                    "Scanned %s notices | prefiltered %s | declaration rows %s",
                    f"{scanned:,}", f"{prefiltered:,}", f"{rows_written:,}",
                )
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=DECLARATION_SCHEMA))
    finally:
        writer.close()
    temp_path.replace(declarations_path)

    positive_notices = set().union(*(class_notices[k] for k in POSITIVE_CLASSES if k in class_notices)) \
        if any(k in class_notices for k in POSITIVE_CLASSES) else set()
    recruitable = len(
        class_notices.get("RENEWAL_DECLARATION", set())
        | class_notices.get(EXPIRY_DECLARATION, set())
    )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mining_version": MINING_VERSION,
        "output_file": str(declarations_path),
        "contract_head_window": window,
        "notices_scanned": scanned,
        "notices_passing_prefilter": prefiltered,
        "prefilter_pass_rate": round(prefiltered / scanned, 4) if scanned else 0.0,
        "declaration_rows": rows_written,
        "notices_by_class": {k: len(v) for k, v in sorted(class_notices.items())},
        "notices_with_any_positive_class": len(positive_notices),
        "strong_declaration_notices": recruitable,
        "matches_by_pattern": dict(pattern_counts.most_common()),
        "matches_by_field": dict(field_counts.most_common()),
        "exclusions": {k: v for k, v in sorted(counters.items()) if k.startswith("excluded_")},
        "notices_with_declared_end_date": counters.get("declared_end_dates", 0),
        "notices_with_declared_reference": counters.get("declared_references", 0),
        "retender_contaminant_notices": len(class_notices.get(RETENDER_AFTER_FAILURE, set())),
        # Gate G1: enough self-declared successions to recruit anchors from.
        "gate_g1_min_strong_declarations": 1200,
        "gate_g1_passed": bool(recruitable >= 1200),
        "validation_passed": bool(rows_written > 0 and scanned == 1_620_712),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--contract-head-window",
        type=int,
        default=CONTRACT_HEAD_WINDOW,
        help="Characters after a renouvellement trigger in which a contract noun "
             "must appear. Widen only if gate G1 under-delivers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    configure_logging(project_root)
    summary = build(project_root, output_dir, args.force, args.contract_head_window)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["validation_passed"]:
        raise RuntimeError("Renewal declaration mining validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
