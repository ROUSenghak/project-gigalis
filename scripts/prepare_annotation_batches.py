#!/usr/bin/env python3
"""Write the blind dossiers an annotator works from.

This is the only path from the scored exposure table to anything an annotator
sees, and it is deliberately narrow. Every field passes through
:func:`boamp_pipeline.exposure.blind_view`, which filters against an explicit
allow-list, so a score column added upstream is withheld until somebody decides
it is safe rather than leaking until somebody notices.

What the dossier *does* disclose is how completely the pool was shown. An
annotator has to know whether "none of these" can mean "none exist", because
that is precisely the distinction v1 could not make: its reviewers saw 25 of a
median 146 candidates and their "no successor" was recorded as if the pool had
been searched. Pool size is given as a bucket, since the exact number would
reveal how selective the exposure was for each candidate.

Passes A and B get separate directories, separate shuffles and separate slot
identifiers, so a pass-A dossier cannot be aligned against pass B.
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
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.annotation_schema import (  # noqa: E402
    ANCHOR_VERDICTS,
    CONFIDENCE_LEVELS,
    INSTRUCTIONS_VERSION,
    LABELS,
)
from boamp_pipeline.exposure import (  # noqa: E402
    assert_no_forbidden_keys,
    blind_view,
    pool_size_bucket,
)
from boamp_pipeline.linkage import parse_json_list  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp_v2/benchmark_v3")
DOSSIER_VERSION = "boamp_benchmark_v3_dossier_1.0"
BATCH_SIZE = 10

NOTICE_URL = "https://www.boamp.fr/pages/avis/?q=idweb:{}"

EPISODE_COLUMNS = [
    "episode_id", "episode_text", "constituent_notice_ids_json",
    "episode_notice_natures_json", "procedure_type",
]

INDEX_COLUMNS = [
    "episode_id", "buyer_name_raw", "buyer_department", "episode_origin_date",
    "award_date", "duration_months_reliable", "expected_end_date", "main_cpv",
    "all_cpvs_json", "framework_flag", "notice_count", "titulaire_names_json",
]


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "prepare_annotation_batches.log", encoding="utf-8"),
        ],
    )


def load_episode_detail(processed_dir: Path, needed: set[str]) -> dict[str, dict[str, Any]]:
    detail: dict[str, dict[str, Any]] = {}
    parquet = pq.ParquetFile(processed_dir / "episodes_2015_2025.parquet")
    for batch in parquet.iter_batches(batch_size=50_000, columns=EPISODE_COLUMNS):
        for row in batch.to_pylist():
            if row["episode_id"] in needed:
                detail[row["episode_id"]] = row
    return detail


def as_date(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(pd.Timestamp(value).date()) if not isinstance(value, str) else value


def episode_view(episode_id: str, slot_id: str, index_row: Any,
                 detail: dict[str, Any]) -> dict[str, Any]:
    """One procurement as the annotator sees it: description and dates only."""
    notice_ids = parse_json_list(detail.get("constituent_notice_ids_json"))
    text = detail.get("episode_text") or ""
    record = {
        "slot_id": slot_id,
        "buyer_name": index_row["buyer_name_raw"],
        "buyer_department": index_row["buyer_department"],
        "first_notice_date": as_date(index_row["episode_origin_date"]),
        "award_date": as_date(index_row["award_date"]),
        "declared_duration_months": (
            float(index_row["duration_months_reliable"])
            if pd.notna(index_row["duration_months_reliable"]) else None
        ),
        "declared_contract_end": as_date(index_row["expected_end_date"]),
        "main_cpv": index_row["main_cpv"],
        "cpv_codes": parse_json_list(index_row["all_cpvs_json"]),
        "procedure_type": detail.get("procedure_type") or "",
        "framework_flag": bool(index_row["framework_flag"]),
        "notice_count": int(index_row["notice_count"] or 0),
        "suppliers": parse_json_list(index_row["titulaire_names_json"]),
        "objet": text.split("\n")[0] if text else "",
        "full_text": text,
        "notice_ids": notice_ids,
        "notice_natures": parse_json_list(detail.get("episode_notice_natures_json")),
        "notice_urls": [NOTICE_URL.format(nid) for nid in notice_ids[:6]],
    }
    return blind_view(record)


def instructions() -> dict[str, Any]:
    """The labelling rules, shipped inside every batch so they are auditable."""
    return {
        "instructions_version": INSTRUCTIONS_VERSION,
        "task": (
            "For each candidate, decide its relation to the anchor contract. The "
            "anchor is a public contract that has been awarded; the candidates are "
            "later procurements by the same buyer."
        ),
        "labels": {
            "RENEWAL_OF_EXPIRING": (
                "The candidate takes over the need the anchor served, as the anchor "
                "ends or is about to end. The typical case: the same service is "
                "re-procured for a new term."
            ),
            "NEXT_PHASE_OF_PROGRAMME": (
                "The candidate continues a programme the anchor began, but extends "
                "its scope rather than replacing it -- a further deployment phase, a "
                "new geographic area."
            ),
            "RETENDER_AFTER_FAILURE": (
                "The candidate relaunches a procedure that produced no contract "
                "(declared unsuccessful or abandoned). Same need, but the anchor was "
                "never a running contract to replace."
            ),
            "PARALLEL_LOT": (
                "The candidate is another lot or component of the same programme, "
                "running alongside the anchor rather than after it."
            ),
            "EXTENSION_SAME_CONTRACT": (
                "The candidate is the same contract continued -- an amendment, a "
                "tacit reconduction, a modification notice -- not a new procurement."
            ),
            "UNRELATED": "No functional relation beyond sharing a buyer.",
        },
        "anchor_verdicts": {
            "HAS_RENEWAL_SUCCESSOR": "At least one candidate is RENEWAL_OF_EXPIRING.",
            "NO_RENEWAL_POOL_EXHAUSTIVE": (
                "No candidate is a renewal, AND the disclosure says the whole pool was "
                "shown. Only then may you claim no successor exists."
            ),
            "NO_RENEWAL_AMONG_SHOWN": (
                "No candidate is a renewal, but only part of the pool was shown. This "
                "is the honest verdict whenever exposure_mode is SAMPLED."
            ),
            "ANCHOR_UNUSABLE": "The anchor cannot be judged (empty or unintelligible).",
        },
        "evidence_rules": [
            "Every label must quote text that appears verbatim in the field cited.",
            "Any label other than UNRELATED needs a quote from the anchor AND from "
            "the candidate.",
            "Reasoning must be specific to this pair. Reused or templated reasoning "
            "is rejected by the schema.",
            "Confidence must reflect the evidence, not the label. Grading every "
            "positive HIGH is rejected.",
        ],
        "cautions": [
            "Similar wording is not sufficient. Two lots of one programme, and a "
            "contract with its own amendment, both read as near-identical text.",
            "'reconduction' and 'renouvellement' in a contract's own duration clause "
            "describe its extension options, not a new procurement.",
            "A renewal is often published well before the previous contract expires: "
            "roughly 70% of declared renewals appear before the expected end date, so "
            "early timing is not evidence against a renewal.",
        ],
        "valid_labels": list(LABELS),
        "valid_confidence": list(CONFIDENCE_LEVELS),
        "valid_anchor_verdicts": list(ANCHOR_VERDICTS),
    }


def require_complete_pass_a(output_dir: Path) -> None:
    """Block pass B until pass A covers every currently prepared dossier."""
    labels_path = output_dir / "labels_pass_A.parquet"
    summary_path = output_dir / "labels_pass_A_summary.json"
    if not labels_path.exists() or not summary_path.exists():
        raise RuntimeError(
            "refusing to prepare pass B before pass A labels and its validation "
            "summary exist"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("validation_passed"):
        raise RuntimeError("refusing to prepare pass B: pass A ingest did not validate")

    prepared_anchors: set[str] = set()
    dossier_dir = output_dir / "dossiers" / "pass_A"
    for path in sorted(dossier_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        prepared_anchors.update(
            str(dossier["anchor_episode_id"]) for dossier in payload.get("dossiers", [])
        )
    if not prepared_anchors:
        raise RuntimeError("refusing to prepare pass B: no pass A dossiers are prepared")

    labelled_anchors = set(
        pd.read_parquet(labels_path, columns=["anchor_episode_id"])[
            "anchor_episode_id"
        ].astype(str)
    )
    if labelled_anchors != prepared_anchors:
        missing = len(prepared_anchors - labelled_anchors)
        unexpected = len(labelled_anchors - prepared_anchors)
        raise RuntimeError(
            "refusing to prepare pass B: pass A coverage does not match the current "
            f"dossiers ({missing} missing, {unexpected} unexpected anchors)"
        )


def build(project_root: Path, output_dir: Path, force: bool, pass_id: str,
          wave: int | None, limit: int | None) -> dict[str, Any]:
    pass_id = pass_id.upper()
    if pass_id not in {"A", "B"}:
        raise ValueError("pass must be A or B")
    dossier_dir = output_dir / "dossiers" / f"pass_{pass_id}"
    if dossier_dir.exists() and any(dossier_dir.iterdir()) and not force:
        raise FileExistsError(f"{dossier_dir} already has batches. Use --force to rebuild.")
    # Pass B must not be prepared before pass A has been annotated, or the two
    # passes are not independent in any meaningful sense.
    if pass_id == "B":
        require_complete_pass_a(output_dir)
    dossier_dir.mkdir(parents=True, exist_ok=True)

    exposure = pd.read_parquet(output_dir / "exposure_full.parquet")
    anchors = pd.read_parquet(output_dir / "anchors.parquet")
    index = pd.read_parquet(output_dir / "national_episode_index.parquet", columns=INDEX_COLUMNS)

    if wave is not None:
        anchors = anchors.loc[anchors["wave"] == wave]
    anchors = anchors.loc[anchors["episode_id"].isin(set(exposure["anchor_episode_id"]))]
    if limit:
        anchors = anchors.head(limit)
    logging.info("Anchors to prepare: %s (pass %s)", f"{len(anchors):,}", pass_id)

    needed = set(anchors["episode_id"]) | set(
        exposure.loc[exposure["anchor_episode_id"].isin(set(anchors["episode_id"])),
                     "candidate_episode_id"]
    )
    detail = load_episode_detail(output_dir.parent, needed)
    by_episode = index.set_index("episode_id")
    slot_column = f"slot_id_pass_{pass_id}"

    dossiers: list[dict[str, Any]] = []
    for position, anchor in enumerate(anchors.itertuples(index=False), start=1):
        pairs = exposure.loc[exposure["anchor_episode_id"] == anchor.episode_id]
        if pairs.empty:
            continue
        mode = str(pairs["exposure_mode"].iloc[0])
        candidates = []
        for pair in pairs.sort_values(slot_column).itertuples(index=False):
            candidate_id = pair.candidate_episode_id
            if candidate_id not in by_episode.index:
                continue
            candidates.append(episode_view(
                candidate_id, getattr(pair, slot_column),
                by_episode.loc[candidate_id], detail.get(candidate_id, {}),
            ))
        dossier = {
            "dossier_version": DOSSIER_VERSION,
            "dossier_id": f"D-{position:06d}",
            "anchor_episode_id": anchor.episode_id,
            "pass_id": pass_id,
            "instructions_version": INSTRUCTIONS_VERSION,
            "anchor": episode_view(
                anchor.episode_id, "ANCHOR",
                by_episode.loc[anchor.episode_id], detail.get(anchor.episode_id, {}),
            ),
            "candidates": candidates,
            "pool_disclosure": {
                "exposure_mode": mode,
                "candidates_shown": len(candidates),
                "pool_size_bucket": pool_size_bucket(int(pairs["pool_size"].iloc[0])),
                "meaning": (
                    "The whole pool is shown; 'none' means none exist in this buyer's "
                    "window." if mode in {"EXHAUSTIVE", "NEAR_EXHAUSTIVE"} else
                    "Only part of the pool is shown; 'none' means none among these."
                ),
            },
        }
        dossiers.append(dossier)

    # Gate G3a: nothing score-like may reach a blind artifact.
    for dossier in dossiers:
        assert_no_forbidden_keys(dossier["anchor"], "dossier anchor")
        for candidate in dossier["candidates"]:
            assert_no_forbidden_keys(candidate, "dossier candidate")

    batches = 0
    for start in range(0, len(dossiers), BATCH_SIZE):
        batches += 1
        payload = {
            "batch_id": f"batch_{batches:04d}",
            "pass_id": pass_id,
            "instructions": instructions(),
            "dossiers": dossiers[start: start + BATCH_SIZE],
        }
        (dossier_dir / f"batch_{batches:04d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    summary_path = output_dir / f"annotation_batches_pass_{pass_id}_summary.json"
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dossier_version": DOSSIER_VERSION,
        "instructions_version": INSTRUCTIONS_VERSION,
        "pass_id": pass_id,
        "wave": wave,
        "directory": str(dossier_dir),
        "dossiers": len(dossiers),
        "batches": batches,
        "candidates_total": sum(len(d["candidates"]) for d in dossiers),
        "exposure_modes": {
            mode: sum(1 for d in dossiers if d["pool_disclosure"]["exposure_mode"] == mode)
            for mode in sorted({d["pool_disclosure"]["exposure_mode"] for d in dossiers})
        },
        "gate_g3a_no_forbidden_keys_in_dossiers": True,
        "blinding": (
            "Fields pass through exposure.blind_view(), an allow-list. Candidate order "
            "and slot identifiers differ between passes, so order carries no ranking "
            "and pass-A identifiers are meaningless in pass B."
        ),
        "validation_passed": bool(len(dossiers) > 0),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pass", dest="pass_id", default="A", choices=["A", "B", "a", "b"])
    parser.add_argument("--wave", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    configure_logging(project_root)
    summary = build(project_root, output_dir, args.force, args.pass_id, args.wave, args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["validation_passed"]:
        raise RuntimeError("Annotation batch preparation validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
