from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.audit_candidate_generation import BLOCKING_STAGES

AUDIT = Path("data/processed/boamp/candidate_generation_audit.json")
UNREACHABLE = Path("data/processed/boamp/candidate_generation_unreachable.csv")
MANIFEST = Path("data/processed/boamp/regional_benchmark/regional_benchmark_manifest.json")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_audit_pairs_completeness_matches_the_reference_manifest() -> None:
    """The audit must re-derive the ceiling, not restate it.

    It recomputes reachability from the exposure table independently of
    ``build_regional_benchmark.py``. If the two ever disagree, one of them is
    reading a stale artifact.
    """
    audit = load(AUDIT)["blocking_loss"]
    ceiling = load(MANIFEST)["candidate_reachability"]
    assert audit["reviewed_positive_anchors"] == ceiling["positive_anchors"]
    assert audit["reachable_anchors"] == ceiling["positive_anchors_with_reviewed_successor_in_pool"]
    assert audit["pairs_completeness"] == ceiling["candidate_generation_recall_ceiling"]


def test_every_unreachable_case_has_a_named_blocking_stage() -> None:
    """An unattributed loss is indistinguishable from a bug, so none may remain."""
    audit = load(AUDIT)["blocking_loss"]
    assert audit["unexplained_cases"] == 0
    frame = pd.read_csv(UNREACHABLE)
    assert len(frame) == audit["unreachable_anchors"]
    assert set(frame["blocking_stage"]).issubset(set(BLOCKING_STAGES))
    assert frame["blocking_reason"].notna().all()


def test_cross_cpv_shares_are_complementary_and_counted() -> None:
    cpv = load(AUDIT)["cpv_continuity"]
    assert cpv["same_cpv2_count"] + cpv["cross_cpv2_count"] == (
        cpv["accepted_links_with_both_divisions_observed"]
    )
    assert cpv["same_cpv2_share"] + cpv["cross_cpv2_share"] == 1.0
    assert cpv["accepted_links_with_both_divisions_observed"] <= cpv["accepted_links"]


def test_hard_same_cpv_blocking_would_cost_reviewed_successors() -> None:
    """The evidence that keeps hard CPV blocking out of the candidate generator.

    The reviewed successors are labelled independently of every linkage method,
    so their own cross-division rate prices the counterfactual. If this ever
    reached zero, hard same-CPV blocking would become defensible and the
    documented justification would need revisiting.
    """
    cpv = load(AUDIT)["cpv_continuity"]
    assert cpv["reviewed_successors_lost_to_hard_same_division_block"] > 0
    assert cpv["recall_ceiling_under_hard_same_division_block"] < 1.0
    assert cpv["reviewed_cross_cpv2_count"] + cpv["reviewed_same_cpv2_count"] == (
        cpv["reviewed_reference_pairs"]
    )
