#!/usr/bin/env python3
"""Repack canonical annotation dossiers into bounded worker assignments.

This utility changes only workload size. It never splits an anchor dossier and
copies the blinded dossier payload and instructions without modification.
Worker submissions belong in a separate directory and remain compatible with
``scripts/ingest_annotations.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = Path(
    "data/processed/boamp_v2/benchmark_v3/dossiers/pass_A"
)
DEFAULT_OUTPUT_DIR = Path("scratchpad/w1a/assignments")


def candidate_count(dossier: dict[str, Any]) -> int:
    return len(dossier.get("candidates", []))


def pack_dossiers(
    dossiers: Sequence[dict[str, Any]], max_candidates: int
) -> list[list[dict[str, Any]]]:
    """Pack complete dossiers with deterministic first-fit decreasing."""
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")

    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for dossier in dossiers:
        dossier_id = str(dossier.get("dossier_id", ""))
        if not dossier_id:
            raise ValueError("every dossier must have a dossier_id")
        if dossier_id in seen:
            raise ValueError(f"duplicate dossier_id: {dossier_id}")
        seen.add(dossier_id)
        count = candidate_count(dossier)
        if count > max_candidates:
            raise ValueError(
                f"{dossier_id} has {count} candidates, above the assignment cap "
                f"of {max_candidates}; dossiers are never split"
            )
        ordered.append(dossier)

    ordered.sort(key=lambda dossier: (-candidate_count(dossier), dossier["dossier_id"]))
    assignments: list[list[dict[str, Any]]] = []
    assignment_sizes: list[int] = []
    for dossier in ordered:
        count = candidate_count(dossier)
        for index, size in enumerate(assignment_sizes):
            if size + count <= max_candidates:
                assignments[index].append(dossier)
                assignment_sizes[index] += count
                break
        else:
            assignments.append([dossier])
            assignment_sizes.append(count)

    for assignment in assignments:
        assignment.sort(key=lambda dossier: dossier["dossier_id"])
    return assignments


def load_dossiers(source_dir: Path) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[Path]]:
    paths = sorted(source_dir.glob("batch_*.json"))
    if not paths:
        raise FileNotFoundError(f"no canonical dossier batches found in {source_dir}")

    pass_ids: set[str] = set()
    instruction_digests: set[str] = set()
    instructions: dict[str, Any] | None = None
    dossiers: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pass_ids.add(str(payload.get("pass_id", "")))
        current_instructions = payload.get("instructions") or {}
        digest = hashlib.sha256(
            json.dumps(current_instructions, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        instruction_digests.add(digest)
        instructions = current_instructions if instructions is None else instructions
        dossiers.extend(payload.get("dossiers", []))

    if len(pass_ids) != 1 or "" in pass_ids:
        raise ValueError(f"source batches do not share one pass_id: {sorted(pass_ids)}")
    if len(instruction_digests) != 1:
        raise ValueError("source batches do not share identical annotation instructions")
    return pass_ids.pop(), instructions or {}, dossiers, paths


def build(
    source_dir: Path,
    output_dir: Path,
    prefix: str,
    max_candidates: int,
    force: bool,
) -> dict[str, Any]:
    pass_id, instructions, dossiers, source_paths = load_dossiers(source_dir)
    assignments = pack_dossiers(dossiers, max_candidates)

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob(f"{prefix}[0-9][0-9][0-9].json"))
    if existing and not force:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing)} assignments; use --force"
        )
    if force:
        for path in existing:
            path.unlink()

    assignment_sizes: list[int] = []
    for index, assignment in enumerate(assignments, start=1):
        assignment_id = f"{prefix}{index:03d}"
        size = sum(candidate_count(dossier) for dossier in assignment)
        assignment_sizes.append(size)
        payload = {
            "assignment_id": assignment_id,
            "pass_id": pass_id,
            "instructions": instructions,
            "dossiers": assignment,
        }
        (output_dir / f"{assignment_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    flattened = [
        dossier["dossier_id"] for assignment in assignments for dossier in assignment
    ]
    validation_passed = bool(
        len(flattened) == len(dossiers)
        and len(flattened) == len(set(flattened))
        and max(assignment_sizes, default=0) <= max_candidates
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_directory": str(source_dir),
        "source_batches": len(source_paths),
        "output_directory": str(output_dir),
        "pass_id": pass_id,
        "assignment_prefix": prefix,
        "max_candidates_per_assignment": max_candidates,
        "assignments": len(assignments),
        "dossiers": len(dossiers),
        "candidates": sum(candidate_count(dossier) for dossier in dossiers),
        "assignment_candidate_min": min(assignment_sizes, default=0),
        "assignment_candidate_max": max(assignment_sizes, default=0),
        "assignment_candidate_mean": (
            round(sum(assignment_sizes) / len(assignment_sizes), 2)
            if assignment_sizes else 0
        ),
        "dossiers_preserved_whole": True,
        "validation_passed": validation_passed,
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not validation_passed:
        raise RuntimeError("assignment packing validation failed")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="A")
    parser.add_argument("--max-candidates", type=int, default=90)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    source_dir = args.source_dir if args.source_dir.is_absolute() else project_root / args.source_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    manifest = build(
        source_dir=source_dir,
        output_dir=output_dir,
        prefix=args.prefix,
        max_candidates=args.max_candidates,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
