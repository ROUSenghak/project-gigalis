"""Make the final benchmark split hard to open accidentally or silently.

This module makes the sealed split a mechanism rather than an intention:

* reading it requires an explicit flag, and the default path never touches it;
* every opening appends to an audit log recording who opened it, why, from
  which line of which file, and at which commit;
* any summary quoting a sealed metric must embed its access record, so a sealed
  number cannot appear in a report without its provenance;
* a checksum manifest detects the file changing underneath the manifest.

The budget is three openings for the life of the benchmark. Spending them is a
decision, and the log is what makes that decision visible afterwards.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAME = "SEALED_MANIFEST.json"
ACCESS_LOG = Path("logs/sealed_test_access.log")

#: Openings allowed for the life of the benchmark. Not enforced as a hard stop
#: -- an enforced limit would just be worked around -- but reported, so the
#: write-up has to state how many times the seal was broken.
OPENING_BUDGET = 3


class SealedSplitError(RuntimeError):
    """Raised when the sealed split is read without explicit authorisation."""


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def write_manifest(sealed_path: Path, project_root: Path, extra: dict[str, Any]) -> Path:
    """Record what was sealed, without recording anything that reveals content.

    Only aggregate counts go in. A manifest that listed per-anchor labels would
    let the split be read without opening it.
    """
    manifest_path = sealed_path.parent / MANIFEST_NAME
    payload = {
        "sealed_file": sealed_path.name,
        "sha256": file_digest(sealed_path),
        "bytes": sealed_path.stat().st_size,
        "sealed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": git_head(project_root),
        "opening_budget": OPENING_BUDGET,
        "rule": (
            "Reading this file requires evaluate_linkage.py --open-sealed-test with a "
            "stated reason. Every opening is appended to logs/sealed_test_access.log, "
            "and any summary quoting a sealed metric must embed its access record."
        ),
        **extra,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def read_access_log(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / ACCESS_LOG
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def record_opening(project_root: Path, sealed_path: Path, reason: str) -> dict[str, Any]:
    """Append one opening to the audit log and return the record."""
    caller = inspect.stack()[2] if len(inspect.stack()) > 2 else inspect.stack()[-1]
    record = {
        "opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": reason,
        "called_from": f"{Path(caller.filename).name}:{caller.lineno}",
        "git_head": git_head(project_root),
        "sha256": file_digest(sealed_path),
        "opening_number": len(read_access_log(project_root)) + 1,
    }
    log_path = project_root / ACCESS_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def open_sealed(
    sealed_path: Path,
    project_root: Path,
    *,
    reason: str = "",
    allow: bool = False,
):
    """Read the sealed split, refusing unless explicitly authorised.

    Returns ``(frame, access_record)``. The access record is meant to be
    embedded in whatever summary quotes the resulting numbers.
    """
    import pandas as pd

    if not allow:
        raise SealedSplitError(
            f"{sealed_path.name} is sealed. Pass allow=True with a stated reason "
            "(evaluate_linkage.py --open-sealed-test --seal-reason '...'). Every "
            "opening is logged and counted against a budget of "
            f"{OPENING_BUDGET}."
        )
    if not reason.strip():
        raise SealedSplitError("opening the sealed split requires a stated reason")
    if not sealed_path.exists():
        raise FileNotFoundError(sealed_path)

    manifest_path = sealed_path.parent / MANIFEST_NAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = file_digest(sealed_path)
        if manifest.get("sha256") != current:
            raise SealedSplitError(
                f"{sealed_path.name} does not match its manifest checksum; the sealed "
                "split has changed since it was sealed and cannot be trusted"
            )

    record = record_opening(project_root, sealed_path, reason)
    return pd.read_parquet(sealed_path), record


def verify(sealed_path: Path, project_root: Path) -> dict[str, Any]:
    """Check the seal without breaking it: checksum plus opening history."""
    manifest_path = sealed_path.parent / MANIFEST_NAME
    result: dict[str, Any] = {
        "sealed_file": str(sealed_path),
        "manifest_present": manifest_path.exists(),
        "file_present": sealed_path.exists(),
    }
    if manifest_path.exists() and sealed_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = file_digest(sealed_path)
        result["checksum_matches"] = manifest.get("sha256") == current
        result["sealed_at"] = manifest.get("sealed_at")
    openings = read_access_log(project_root)
    result["openings"] = len(openings)
    result["opening_budget"] = OPENING_BUDGET
    result["within_budget"] = len(openings) <= OPENING_BUDGET
    result["opening_history"] = openings
    result["passed"] = bool(
        result.get("manifest_present")
        and result.get("file_present")
        and result.get("checksum_matches", False)
        and result["within_budget"]
    )
    return result
