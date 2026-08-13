from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_quality_evidence import threshold_sweep


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_threshold_sweep_uses_top_one_anchor_decisions() -> None:
    candidates = pd.DataFrame(
        [
            {"anchor_episode_id": "a", "candidate_episode_id": "a_true", "text_component": 0.80},
            {"anchor_episode_id": "a", "candidate_episode_id": "a_wrong", "text_component": 0.90},
            {"anchor_episode_id": "b", "candidate_episode_id": "b_true", "text_component": 0.65},
            {"anchor_episode_id": "c", "candidate_episode_id": "c_wrong", "text_component": 0.75},
        ]
    )
    truth = pd.DataFrame(
        [
            {"anchor_episode_id": "a", "has_successor": True, "true_successors": ["a_true"]},
            {"anchor_episode_id": "b", "has_successor": True, "true_successors": ["b_true"]},
            {"anchor_episode_id": "c", "has_successor": False, "true_successors": []},
        ]
    )

    sweep = threshold_sweep(candidates, truth, thresholds=(60.0, 70.0, 95.0))

    at_60 = sweep.loc[sweep["threshold_percent"].eq(60.0)].iloc[0]
    assert at_60["accepted_links"] == 3
    assert at_60["true_positive"] == 1
    assert at_60["precision"] == 0.3333
    assert at_60["recall"] == 0.5
    assert at_60["false_positive_rate"] == 1.0

    at_95 = sweep.loc[sweep["threshold_percent"].eq(95.0)].iloc[0]
    assert at_95["accepted_links"] == 0
    assert pd.isna(at_95["precision"])
    assert at_95["recall"] == 0.0


def test_threshold_sweep_acceptance_is_nonincreasing() -> None:
    candidates = pd.DataFrame(
        [
            {"anchor_episode_id": "a", "candidate_episode_id": "a1", "text_component": 0.80},
            {"anchor_episode_id": "b", "candidate_episode_id": "b1", "text_component": 0.60},
        ]
    )
    truth = pd.DataFrame(
        [
            {"anchor_episode_id": "a", "has_successor": True, "true_successors": ["a1"]},
            {"anchor_episode_id": "b", "has_successor": False, "true_successors": []},
        ]
    )

    sweep = threshold_sweep(candidates, truth, thresholds=(50.0, 70.0, 90.0))

    assert sweep["accepted_links"].tolist() == [2, 1, 0]


def test_academic_precision_recall_sources_and_boundary_are_recorded() -> None:
    references = (PROJECT_ROOT / "METHODOLOGICAL_REFERENCES.md").read_text(
        encoding="utf-8"
    )
    generator = (PROJECT_ROOT / "scripts/refresh_reader_artifacts.py").read_text(
        encoding="utf-8"
    )

    for doi in ["10.1145/1143844.1143874", "10.1371/journal.pone.0118432"]:
        assert doi in references
        assert doi in generator
    assert "Generic web illustrations" in references
    assert "not as academic" in generator
