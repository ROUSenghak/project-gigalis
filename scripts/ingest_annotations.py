#!/usr/bin/env python3
"""Validate and record one annotation pass, rejecting anything that fails.

Rejection is the point. The v1 benchmark recorded 120 rows whose reviewer and
adjudication columns were entirely empty, whose 70 negatives shared a single
reason string, and whose confidence was a deterministic function of the label.
Nothing rejected them because nothing checked. Here a batch that fails any
check is not written, and the failure is reported with the record that caused
it.

Labels arrive as JSON, one file per batch, mirroring the dossier batches. Each
label references candidates by their opaque slot identifier; the mapping back
to episode identifiers happens here, against the restricted exposure table, so
the annotator never needs -- and never has -- the real identity of a candidate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.annotation_schema import (  # noqa: E402
    SCHEMA_VERSION,
    validate_anchor_record,
    validate_batch,
)

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp_v2/benchmark_v3")


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "ingest_annotations.log", encoding="utf-8"),
        ],
    )


def load_dossiers(output_dir: Path, pass_id: str) -> dict[str, dict[str, Any]]:
    dossier_dir = output_dir / "dossiers" / f"pass_{pass_id}"
    dossiers: dict[str, dict[str, Any]] = {}
    for path in sorted(dossier_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for dossier in payload["dossiers"]:
            dossiers[dossier["dossier_id"]] = dossier
    return dossiers


def validate_submission_coverage(
    submissions: list[dict[str, Any]], dossiers: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Require exactly one submitted anchor record per prepared dossier."""
    counts = Counter(str(record.get("dossier_id", "")) for record in submissions)
    problems: list[dict[str, Any]] = []
    for dossier_id in sorted(set(dossiers) - set(counts)):
        problems.append({"dossier_id": dossier_id, "problems": ["missing submission"]})
    for dossier_id in sorted(set(counts) - set(dossiers)):
        problems.append({"dossier_id": dossier_id, "problems": ["unknown dossier_id"]})
    for dossier_id, count in sorted(counts.items()):
        if dossier_id in dossiers and count > 1:
            problems.append({
                "dossier_id": dossier_id,
                "problems": [f"submitted {count} times; expected exactly once"],
            })
    return problems


def build(project_root: Path, output_dir: Path, force: bool, pass_id: str,
          labels_dir: Path) -> dict[str, Any]:
    pass_id = pass_id.upper()
    labels_path = output_dir / f"labels_pass_{pass_id}.parquet"
    summary_path = output_dir / f"labels_pass_{pass_id}_summary.json"
    if labels_path.exists() and not force:
        raise FileExistsError(f"{labels_path} already exists. Use --force to rebuild.")

    dossiers = load_dossiers(output_dir, pass_id)
    logging.info("Dossiers for pass %s: %s", pass_id, f"{len(dossiers):,}")
    if not dossiers:
        raise RuntimeError(f"no dossiers found for pass {pass_id}")

    exposure = pd.read_parquet(
        output_dir / "exposure_full.parquet",
        columns=["anchor_episode_id", "candidate_episode_id",
                 f"slot_id_pass_{pass_id}", "frame", "wave", "exposure_mode"],
    )
    slot_to_candidate = {
        (row.anchor_episode_id, getattr(row, f"slot_id_pass_{pass_id}")): row.candidate_episode_id
        for row in exposure.itertuples(index=False)
    }

    submissions: list[dict[str, Any]] = []
    for path in sorted(labels_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        submissions.extend(payload.get("annotations", []))
    logging.info("Anchor records submitted: %s", f"{len(submissions):,}")
    if not submissions:
        raise RuntimeError(f"no annotation files found in {labels_dir}")

    problems = validate_submission_coverage(submissions, dossiers)
    rows: list[dict[str, Any]] = []
    contexts: set[str] = set()

    for record in submissions:
        dossier_id = record.get("dossier_id")
        dossier = dossiers.get(dossier_id)
        if dossier is None:
            continue
        found = validate_anchor_record(record, dossier)
        if found:
            problems.append({"dossier_id": dossier_id, "problems": found})
            continue

        anchor_id = dossier["anchor_episode_id"]
        context = record.get("annotator_context_id", "")
        if context:
            contexts.add(context)
        for item in record.get("labels", []):
            slot = item["candidate_slot"]
            candidate_id = slot_to_candidate.get((anchor_id, slot))
            if candidate_id is None:
                problems.append({
                    "dossier_id": dossier_id,
                    "problems": [f"slot {slot} does not map to a candidate for this anchor"],
                })
                continue
            rows.append({
                "anchor_episode_id": anchor_id,
                "candidate_episode_id": candidate_id,
                "dossier_id": dossier_id,
                "pass_id": pass_id,
                "annotator_context_id": context,
                "candidate_slot": slot,
                "label": item["label"],
                "confidence": item["confidence"],
                "reasoning": item["reasoning"],
                "evidence_json": json.dumps(item.get("evidence", []), ensure_ascii=False),
                "anchor_verdict": record["anchor_verdict"],
                "exposure_mode": dossier["pool_disclosure"]["exposure_mode"],
                "schema_version": SCHEMA_VERSION,
            })

    batch_report = validate_batch([
        {"label": r["label"], "confidence": r["confidence"], "reasoning": r["reasoning"]}
        for r in rows
    ])

    if problems:
        logging.error("Rejected %s anchor records", len(problems))
        for problem in problems[:10]:
            logging.error("  %s: %s", problem["dossier_id"], problem["problems"])

    labels = pd.DataFrame(rows)
    if not labels.empty and not problems and batch_report["passed"]:
        labels.to_parquet(labels_path, index=False, compression="zstd")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "pass_id": pass_id,
        "output_file": str(labels_path),
        "anchor_records_submitted": len(submissions),
        "anchor_records_rejected": len(problems),
        "rejections": problems[:50],
        "candidate_labels": int(len(labels)),
        "distinct_annotator_contexts": len(contexts),
        "label_counts": (
            {str(k): int(v) for k, v in labels["label"].value_counts().items()}
            if len(labels) else {}
        ),
        "anchor_verdicts": (
            {str(k): int(v) for k, v in
             labels.drop_duplicates("dossier_id")["anchor_verdict"].value_counts().items()}
            if len(labels) else {}
        ),
        "batch_quality": batch_report,
        "validation_passed": bool(len(labels) > 0 and not problems and batch_report["passed"]),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pass", dest="pass_id", default="A", choices=["A", "B", "a", "b"])
    parser.add_argument("--labels-dir", type=Path, required=True,
                        help="Directory of submitted annotation JSON files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    labels_dir = args.labels_dir if args.labels_dir.is_absolute() else project_root / args.labels_dir
    configure_logging(project_root)
    summary = build(project_root, output_dir, args.force, args.pass_id, labels_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["validation_passed"]:
        raise RuntimeError("Annotation ingest validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
