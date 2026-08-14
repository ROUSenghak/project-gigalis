from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.ingest_annotations import validate_submission_coverage
from scripts.pack_annotation_assignments import candidate_count, pack_dossiers
from scripts.prepare_annotation_batches import require_complete_pass_a


def make_dossier(dossier_id: str, candidates: int) -> dict:
    return {
        "dossier_id": dossier_id,
        "candidates": [{"slot_id": f"S-{index}"} for index in range(candidates)],
    }


def test_pack_dossiers_preserves_whole_dossiers_within_cap() -> None:
    dossiers = [
        make_dossier("D-1", 59),
        make_dossier("D-2", 36),
        make_dossier("D-3", 36),
        make_dossier("D-4", 20),
    ]

    assignments = pack_dossiers(dossiers, max_candidates=90)

    assert sorted(
        dossier["dossier_id"] for group in assignments for dossier in group
    ) == ["D-1", "D-2", "D-3", "D-4"]
    assert all(sum(candidate_count(dossier) for dossier in group) <= 90
               for group in assignments)
    assert {tuple(dossier["dossier_id"] for dossier in group) for group in assignments} == {
        ("D-1", "D-4"),
        ("D-2", "D-3"),
    }


def test_pack_dossiers_rejects_an_oversized_dossier() -> None:
    with pytest.raises(ValueError, match="dossiers are never split"):
        pack_dossiers([make_dossier("D-1", 91)], max_candidates=90)


def test_pack_dossiers_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate dossier_id"):
        pack_dossiers(
            [make_dossier("D-1", 10), make_dossier("D-1", 20)],
            max_candidates=90,
        )


def test_ingest_requires_exact_dossier_coverage() -> None:
    dossiers = {"D-1": {}, "D-2": {}}
    submissions = [
        {"dossier_id": "D-1"},
        {"dossier_id": "D-1"},
        {"dossier_id": "D-3"},
    ]

    problems = validate_submission_coverage(submissions, dossiers)

    assert {problem["dossier_id"] for problem in problems} == {"D-1", "D-2", "D-3"}


def test_pass_b_gate_rejects_stale_partial_pass_a(tmp_path) -> None:
    dossier_dir = tmp_path / "dossiers" / "pass_A"
    dossier_dir.mkdir(parents=True)
    (dossier_dir / "batch_0001.json").write_text(
        json.dumps({
            "dossiers": [
                {"anchor_episode_id": "E-1"},
                {"anchor_episode_id": "E-2"},
            ]
        }),
        encoding="utf-8",
    )
    pd.DataFrame({"anchor_episode_id": ["E-1"]}).to_parquet(
        tmp_path / "labels_pass_A.parquet", index=False
    )
    (tmp_path / "labels_pass_A_summary.json").write_text(
        json.dumps({"validation_passed": True}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="1 missing"):
        require_complete_pass_a(tmp_path)
