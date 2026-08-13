#!/usr/bin/env python3
"""Compare the two annotation passes, measure agreement, and adjudicate.

The v1 benchmark shipped reviewer and adjudication columns that were empty on
every one of its 120 rows. This stage exists so that those columns cannot be
empty here: a pair labelled by both passes either agrees, or it is written to a
disagreement queue that a third pass must resolve before the benchmark can be
assembled.

The kappa reported here carries a caveat that must travel with it wherever it
is quoted. Both passes come from the same model, presented with a different
candidate order, different slot identifiers and a different framing -- pass B
is asked to argue against a relation before deciding. That measures
self-consistency under re-presentation. It is not agreement between independent
intelligences, and describing it as inter-annotator agreement would overstate
what was done.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.annotation_schema import (  # noqa: E402
    RENEWAL_OF_EXPIRING,
    SCHEMA_VERSION,
    agreement_report,
    cohen_kappa,
)

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp_v2/benchmark_v3")

#: Gate G5. Below these the class definitions are not reproducible enough to
#: label at scale, and the instructions need revising before the full run.
MIN_KAPPA_SIX_CLASS = 0.60
MIN_KAPPA_BINARY = 0.70


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "adjudicate_annotations.log", encoding="utf-8"),
        ],
    )


def build(project_root: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    adjudicated_path = output_dir / "labels_adjudicated.parquet"
    queue_path = output_dir / "adjudication_queue.csv"
    summary_path = output_dir / "annotation_agreement_summary.json"
    if adjudicated_path.exists() and not force:
        raise FileExistsError(f"{adjudicated_path} already exists. Use --force to rebuild.")

    path_a = output_dir / "labels_pass_A.parquet"
    path_b = output_dir / "labels_pass_B.parquet"
    if not path_a.exists():
        raise FileNotFoundError(path_a)
    labels_a = pd.read_parquet(path_a)
    labels_b = pd.read_parquet(path_b) if path_b.exists() else pd.DataFrame()
    logging.info("Pass A labels: %s | pass B labels: %s", len(labels_a), len(labels_b))

    # Independence is auditable, not asserted: a context that annotated both
    # passes would mean the second pass saw the first.
    contexts_a = set(labels_a["annotator_context_id"].dropna())
    contexts_b = set(labels_b["annotator_context_id"].dropna()) if len(labels_b) else set()
    shared_contexts = sorted(contexts_a & contexts_b)
    if shared_contexts:
        raise RuntimeError(
            f"annotator context(s) {shared_contexts} appear in both passes; the passes "
            "are not independent and cannot be adjudicated"
        )

    key = ["anchor_episode_id", "candidate_episode_id"]
    map_a = {tuple(r[k] for k in key): r["label"] for _, r in labels_a.iterrows()}
    map_b = {tuple(r[k] for k in key): r["label"] for _, r in labels_b.iterrows()} if len(labels_b) else {}
    report = agreement_report(map_a, map_b)

    overlap = sorted(set(map_a) & set(map_b))
    disagreements = [pair for pair in overlap if map_a[pair] != map_b[pair]]

    # Adjudication: agreed pairs are settled; disagreements go to a queue that a
    # third pass must resolve. Nothing silently picks a side.
    merged = labels_a.set_index(key)
    rows: list[dict[str, Any]] = []
    for pair, label in map_a.items():
        both = pair in map_b
        agreed = both and map_b[pair] == label
        record = merged.loc[pair]
        rows.append({
            "anchor_episode_id": pair[0],
            "candidate_episode_id": pair[1],
            "label_pass_a": label,
            "label_pass_b": map_b.get(pair),
            "double_annotated": both,
            "agreed": agreed if both else None,
            "adjudication_required": bool(both and not agreed),
            "label": label if (not both or agreed) else None,
            "confidence": record["confidence"],
            "reasoning": record["reasoning"],
            "evidence_json": record["evidence_json"],
            "anchor_verdict": record["anchor_verdict"],
            "exposure_mode": record["exposure_mode"],
            "schema_version": SCHEMA_VERSION,
        })
    adjudicated = pd.DataFrame(rows)
    adjudicated.to_parquet(adjudicated_path, index=False, compression="zstd")

    queue = adjudicated.loc[adjudicated["adjudication_required"]]
    if len(queue):
        queue.to_csv(queue_path, index=False)
        logging.warning("%s pairs need third-pass adjudication -> %s", len(queue), queue_path)

    kappa_six = report.get("cohen_kappa_six_class")
    kappa_binary = report.get("cohen_kappa_renewal_binary")
    gate_passed = bool(
        overlap
        and kappa_six is not None
        and kappa_six >= MIN_KAPPA_SIX_CLASS
        and kappa_binary is not None
        and kappa_binary >= MIN_KAPPA_BINARY
    )

    anchors_a = labels_a.drop_duplicates("dossier_id")[["dossier_id", "anchor_verdict"]]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "output_file": str(adjudicated_path),
        "pass_a_labels": int(len(labels_a)),
        "pass_b_labels": int(len(labels_b)),
        "double_annotated_pairs": len(overlap),
        "double_annotation_share": (
            round(len(overlap) / len(map_a), 4) if map_a else None
        ),
        "agreement": report,
        "disagreements_requiring_adjudication": len(disagreements),
        "adjudication_queue": str(queue_path) if len(queue) else None,
        "settled_labels": int(adjudicated["label"].notna().sum()),
        "anchor_verdicts_pass_a": {
            str(k): int(v) for k, v in anchors_a["anchor_verdict"].value_counts().items()
        },
        "gate_g5_pilot_agreement": {
            "min_kappa_six_class": MIN_KAPPA_SIX_CLASS,
            "min_kappa_renewal_binary": MIN_KAPPA_BINARY,
            "observed_six_class": kappa_six,
            "observed_renewal_binary": kappa_binary,
            "passed": gate_passed,
        },
        "kappa_caveat": (
            "Both passes are the same model under a different candidate order, "
            "different slot identifiers and an adversarial framing in pass B. This is "
            "self-consistency under re-presentation, not inter-annotator agreement, "
            "and must be described that way wherever it is quoted."
        ),
        "validation_passed": bool(len(adjudicated) > 0 and not shared_contexts),
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
        raise RuntimeError("Annotation adjudication validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
