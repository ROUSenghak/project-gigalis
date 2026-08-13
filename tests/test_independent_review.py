from pathlib import Path

import pandas as pd

from scripts.prepare_independent_link_review import deterministic_sample, pair_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_deterministic_sample_is_reproducible() -> None:
    frame = pd.DataFrame({"value": range(20)})
    first = deterministic_sample(frame, 6, 17)
    second = deterministic_sample(frame, 6, 17)
    assert first["value"].tolist() == second["value"].tolist()
    assert len(first) == 6


def test_pair_key_is_directional_and_unique_for_distinct_pairs() -> None:
    frame = pd.DataFrame(
        {
            "anchor_episode_id": ["A", "B"],
            "candidate_episode_id": ["B", "A"],
        }
    )
    keys = pair_key(frame)
    assert keys.tolist() == ["A|B", "B|A"]
    assert keys.is_unique


def test_generated_review_sample_is_blinded_and_balanced() -> None:
    review = pd.read_csv(PROJECT_ROOT / "data/review/independent_link_review_sample.csv")
    audit = pd.read_csv(PROJECT_ROOT / "data/review/independent_link_review_audit_key.csv")

    assert len(review) == 60
    assert review["review_id"].is_unique
    assert set(review["review_id"]) == set(audit["review_id"])
    assert audit["source_stratum"].value_counts().to_dict() == {
        "PRIMARY_ACCEPTED": 20,
        "HIGH_SIMILARITY_STRUCTURAL_NEGATIVE": 20,
        "BUYER_DECLARED_RESOLVED": 20,
    }
    assert "source_stratum" not in review.columns
    assert "selection_text_component" not in review.columns
    for column in [
        "same_legal_buyer_Y_N_UNCERTAIN",
        "relationship_label",
        "observable_successor_Y_N_UNCERTAIN",
    ]:
        assert review[column].isna().all()
