#!/usr/bin/env python3
"""Audit buyer-identity risks in generated successor links.

The linkage pipeline deliberately tolerates name-only buyer evidence when no
validated SIREN is available, because BOAMP notices do not always expose a
usable identifier. This audit makes the residual risk explicit:

* hard failures: both sides have validated SIRENs and they conflict;
* legal-form review cases: municipal and intercommunal labels are mixed;
* homonym review cases: exact name-only matches cross departments.

The script writes a compact CSV of accepted links needing review plus a JSON
summary suitable for methodology reporting.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.linkage import normalize_buyer_for_blocking

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp_v2")

MUNICIPAL_RE = re.compile(r"\b(commune|ville|mairie)\b", re.IGNORECASE)
INTERCOMMUNAL_RE = re.compile(
    r"\b(metropole|métropole|communaute|communauté|agglomeration|agglomération|epci)\b",
    re.IGNORECASE,
)


def legal_family(value: Any) -> str:
    text = str(value or "")
    municipal = bool(MUNICIPAL_RE.search(text))
    intercommunal = bool(INTERCOMMUNAL_RE.search(text))
    if municipal and intercommunal:
        return "mixed_label"
    if municipal:
        return "municipal"
    if intercommunal:
        return "intercommunal"
    return "other"


def load_episode_metadata(output_dir: Path) -> pd.DataFrame:
    columns = [
        "episode_id",
        "buyer_siren",
        "buyer_key",
        "buyer_name_raw",
        "buyer_name_normalized",
        "buyer_department",
        "buyer_region",
    ]
    episodes = pd.read_parquet(output_dir / "episodes_grand_ouest.parquet", columns=columns)
    episodes["buyer_name_blocking"] = episodes["buyer_name_normalized"].map(normalize_buyer_for_blocking)
    episodes["buyer_legal_family"] = episodes["buyer_name_raw"].map(legal_family)
    return episodes


def audit(output_dir: Path) -> dict[str, Any]:
    links_path = output_dir / "accepted_successor_links.parquet"
    links = pd.read_parquet(links_path)
    episodes = load_episode_metadata(output_dir)

    enriched = (
        links.merge(
            episodes.add_prefix("anchor_"),
            left_on="anchor_episode_id",
            right_on="anchor_episode_id",
            how="left",
        )
        .merge(
            episodes.add_prefix("candidate_"),
            left_on="candidate_episode_id",
            right_on="candidate_episode_id",
            how="left",
        )
    )

    enriched["anchor_siren_present"] = enriched["anchor_buyer_siren"].fillna("").astype(str).str.strip().ne("")
    enriched["candidate_siren_present"] = (
        enriched["candidate_buyer_siren"].fillna("").astype(str).str.strip().ne("")
    )
    enriched["both_siren_present"] = enriched["anchor_siren_present"] & enriched["candidate_siren_present"]
    enriched["conflicting_validated_siren"] = enriched["both_siren_present"] & (
        enriched["anchor_buyer_siren"].astype(str) != enriched["candidate_buyer_siren"].astype(str)
    )
    enriched["department_diff"] = (
        enriched["anchor_buyer_department"].astype(str) != enriched["candidate_buyer_department"].astype(str)
    )
    enriched["legal_family_diff"] = (
        enriched["anchor_buyer_legal_family"] != enriched["candidate_buyer_legal_family"]
    )
    enriched["municipal_intercommunal_mix"] = [
        frozenset(pair) == frozenset({"municipal", "intercommunal"})
        for pair in zip(enriched["anchor_buyer_legal_family"], enriched["candidate_buyer_legal_family"])
    ]
    enriched["name_only_cross_department"] = (
        enriched["buyer_match_type"].eq("normalized_name")
        & ~enriched["anchor_siren_present"]
        & ~enriched["candidate_siren_present"]
        & enriched["department_diff"]
    )

    reasons: list[str] = []
    for row in enriched.itertuples(index=False):
        row_reasons: list[str] = []
        if row.conflicting_validated_siren:
            row_reasons.append("conflicting_validated_siren")
        if row.municipal_intercommunal_mix:
            row_reasons.append("municipal_intercommunal_mix")
        if row.name_only_cross_department:
            row_reasons.append("name_only_cross_department")
        if row.legal_family_diff and not row.municipal_intercommunal_mix:
            row_reasons.append("legal_family_diff_review")
        reasons.append(";".join(row_reasons))
    enriched["audit_reason"] = reasons

    review = enriched.loc[enriched["audit_reason"].ne("")].copy()
    review_columns = [
        "audit_reason",
        "anchor_episode_id",
        "candidate_episode_id",
        "anchor_buyer_name_raw",
        "candidate_buyer_name_raw",
        "anchor_buyer_siren",
        "candidate_buyer_siren",
        "anchor_buyer_department",
        "candidate_buyer_department",
        "anchor_buyer_legal_family",
        "candidate_buyer_legal_family",
        "buyer_match_type",
        "text_component",
        "linkage_score",
        "gap_days",
    ]
    review_path = output_dir / "buyer_blocking_legal_form_audit.csv"
    summary_path = output_dir / "buyer_blocking_legal_form_audit_summary.json"
    review[review_columns].sort_values(
        ["audit_reason", "text_component", "anchor_episode_id"],
        ascending=[True, False, True],
    ).to_csv(review_path, index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(links_path),
        "review_file": str(review_path),
        "accepted_links": int(len(enriched)),
        "hard_fail_conflicting_validated_siren": int(enriched["conflicting_validated_siren"].sum()),
        "municipal_intercommunal_mix": int(enriched["municipal_intercommunal_mix"].sum()),
        "name_only_cross_department": int(enriched["name_only_cross_department"].sum()),
        "legal_family_diff_review": int(
            (enriched["legal_family_diff"] & ~enriched["municipal_intercommunal_mix"]).sum()
        ),
        "review_rows": int(len(review)),
        "validation_passed": bool(enriched["conflicting_validated_siren"].sum() == 0),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    summary = audit(output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["validation_passed"]:
        raise RuntimeError("Buyer blocking legal-form audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
