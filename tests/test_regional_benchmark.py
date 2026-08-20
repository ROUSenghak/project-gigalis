"""Guards on how the regional reference is re-resolved onto current episodes.

The review recorded episode identifiers from an earlier reconstruction, so every
anchor has to be re-resolved through its BOAMP notice identifiers. That step is
where a reference can quietly acquire wrong labels, so it is tested rather than
trusted.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_regional_benchmark import (
    VERDICT_MAP,
    build_truth,
    episodes_for,
    notice_to_episode,
    resolve_anchors,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = PROJECT_ROOT / "data/processed/boamp/regional_benchmark"
SOURCE = PROJECT_ROOT / "data/reference/regional_link_benchmark/BOAMP_Internship_Reference_120.csv"


def test_notice_index_maps_every_notice_of_an_episode() -> None:
    episodes = pd.DataFrame({
        "episode_id": ["EP-1", "EP-2"],
        "constituent_notice_ids_json": ['["18-1", "19-2"]', '["20-3"]'],
    })

    index = notice_to_episode(episodes)

    assert index["18-1"] == {"EP-1"}
    assert index["19-2"] == {"EP-1"}
    assert episodes_for(["18-1", "20-3"], index) == {"EP-1", "EP-2"}
    assert episodes_for(["99-9"], index) == set()


def test_an_anchor_spanning_two_episodes_is_dropped_not_guessed() -> None:
    """Episode reconstruction changed since the review. When a reviewed anchor
    now spans two episodes there is no way to know which one carries the label,
    and picking one would fabricate a reference row."""
    reference = pd.DataFrame([{
        "sample_id": "RV-999",
        "anchor_notice_ids_json": '["18-1", "20-3"]',
        "anchor_episode_id": "EP-old",
        "final_outcome": "NO_OBSERVED_SUCCESSOR_IN_SCOPE",
        "primary_evaluation_eligible": True,
        "benchmark_split": "LOCKED_TEST",
        "sampling_stratum": "S", "inclusion_probability": 0.5, "sampling_weight": 2.0,
        "broad_candidate_pool_n": 10, "review_candidates_exported_n": 5,
        "anchor_award_notice_date": "2019-01-01", "anchor_expected_end_date": None,
        "anchor_theme": "CPV-72", "final_confidence": "MEDIUM", "final_evidence_urls": "[]",
    }])
    index = {"18-1": {"EP-1"}, "20-3": {"EP-2"}}

    anchors, counts = resolve_anchors(reference, index)

    assert anchors.loc[0, "anchor_episode_id"] is None
    assert anchors.loc[0, "remap_status"] == "AMBIGUOUS_MULTIPLE_EPISODES"
    assert counts["ambiguous"] == 1


def test_a_positive_whose_successor_no_longer_resolves_becomes_unusable() -> None:
    """Scoring a positive anchor whose target the reference can no longer name
    would count every method as missing a successor that is not in the data."""
    anchors = pd.DataFrame([{
        "sample_id": "RV-1", "anchor_episode_id": "EP-a", "remap_status": "RESOLVED",
        "final_outcome": "OBSERVED_SUCCESSOR", "primary_evaluation_eligible": True,
        "benchmark_split_source": "LOCKED_TEST", "stratum_id": "S",
        "inclusion_probability": 0.5, "design_weight": 2.0, "pool_size": 10,
        "labels_recorded": 5, "anchor_award_notice_date": "2019-01-01",
        "anchor_expected_end_date": None, "anchor_theme": "CPV-72",
        "final_confidence": "HIGH",
    }])

    truth = build_truth(anchors, successors={})

    assert truth.loc[0, "anchor_verdict"] == "ANCHOR_UNUSABLE"
    assert not truth.loc[0, "has_successor_primary"]


def test_declined_reviews_never_become_negatives() -> None:
    assert VERDICT_MAP["OUTSIDE_SCOPE"] == "ANCHOR_UNUSABLE"
    assert VERDICT_MAP["INSUFFICIENT_INFORMATION"] == "ANCHOR_UNUSABLE"
    assert VERDICT_MAP["NO_OBSERVED_SUCCESSOR_IN_SCOPE"] == "NO_RENEWAL_AMONG_SHOWN"


@pytest.mark.skipif(not REFERENCE.exists(), reason="regional reference not built")
def test_the_materialised_reference_agrees_with_its_source_csv() -> None:
    manifest = json.loads(
        (REFERENCE / "regional_benchmark_manifest.json").read_text(encoding="utf-8")
    )
    source = pd.read_csv(SOURCE)

    assert manifest["reviewed_anchors"] == len(source)
    assert manifest["remap"]["route_disagreements"] == 0
    assert manifest["independent_of_linkage_algorithms"] is True
    assert manifest["independent_human_specialist_review"] is False

    splits = [pd.read_parquet(REFERENCE / f"benchmark_{name}.parquet") for name in ("dev", "validation")]
    combined = pd.concat(splits)
    assert combined["anchor_episode_id"].is_unique
    assert set(combined["sample_id"]) <= set(source["sample_id"])
    # No anchor may be its own successor, and no successor may be missing.
    for record in combined.itertuples(index=False):
        successors = json.loads(record.successors_primary_json)
        assert record.anchor_episode_id not in successors
        assert bool(successors) == bool(record.has_successor_primary)


@pytest.mark.skipif(not REFERENCE.exists(), reason="regional reference not built")
def test_the_two_splits_never_share_an_anchor() -> None:
    dev = pd.read_parquet(REFERENCE / "benchmark_dev.parquet")
    validation = pd.read_parquet(REFERENCE / "benchmark_validation.parquet")

    assert set(dev["anchor_episode_id"]).isdisjoint(set(validation["anchor_episode_id"]))


def test_the_manifest_separates_label_from_candidate_surfacing_independence() -> None:
    """Two different claims, and only one of them holds.

    The labels were produced before any linkage method existed, which is a real
    independence property. Which candidates the reviewer was shown is a
    different question, and the export rule is not recorded anywhere in this
    repository. Reporting only the first overstates the reference's standing on
    recall, so both are carried explicitly.
    """
    import json
    from pathlib import Path

    manifest_path = Path(
        "data/processed/boamp/regional_benchmark/regional_benchmark_manifest.json"
    )
    if not manifest_path.exists():
        import pytest

        pytest.skip("regional benchmark not materialised")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["label_independence"]["holds"] is True
    surfacing = manifest["candidate_surfacing_independence"]
    assert surfacing["holds"] is False
    assert surfacing["status"] == "not recoverable"
    assert surfacing["review_candidates_cap"] > 0
    # Precision must be explicitly exempted; it is not affected by the gap.
    assert "Precision" in surfacing["basis"]


def test_reader_artifacts_do_not_claim_full_candidate_independence() -> None:
    """No active document may say the reference is independent of every method
    without separating the label claim from the candidate-pool claim."""
    from pathlib import Path

    import pytest

    for name in ("DATA_QUALITY_REPORT.md", "FINAL_PIPELINE.md", "REGIONAL_BENCHMARK_REFERENCE.md"):
        path = Path(name)
        if not path.exists():
            pytest.skip(f"{name} not materialised")
        text = path.read_text(encoding="utf-8")
        if "independent of every method" in text:
            assert "candidate" in text.lower() and (
                "not recorded" in text or "not recoverable" in text
            ), f"{name} claims independence without disclosing the candidate-export gap"
