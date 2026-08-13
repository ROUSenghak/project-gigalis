"""The sealed split must be hard to open by accident and impossible to open quietly.

The existing study admits it has no clean holdout: notebook 07 inspected
locked-test errors before notebook 08 designed against them, and the final
method choice drew on the locked split too. The split was called locked and
nothing locked it. These tests check that this one is mechanically different.
"""

import json

import pandas as pd
import pytest

from boamp_pipeline.sealed_split import (
    MANIFEST_NAME,
    OPENING_BUDGET,
    SealedSplitError,
    file_digest,
    open_sealed,
    read_access_log,
    verify,
    write_manifest,
)


@pytest.fixture()
def sealed(tmp_path):
    project_root = tmp_path
    (project_root / "logs").mkdir()
    sealed_dir = project_root / "bench" / "sealed"
    sealed_dir.mkdir(parents=True)
    path = sealed_dir / "benchmark_v3_test.parquet"
    pd.DataFrame({"anchor_episode_id": ["EP-1", "EP-2"], "has_successor_primary": [True, False]}).to_parquet(path)
    write_manifest(path, project_root, {"anchors": 2})
    return project_root, path


def test_reading_is_refused_by_default(sealed) -> None:
    project_root, path = sealed

    with pytest.raises(SealedSplitError, match="is sealed"):
        open_sealed(path, project_root)


def test_a_default_read_attempt_leaves_no_trace_because_it_never_happened(sealed) -> None:
    """A refused attempt must not be logged as an opening; the log counts real
    openings only, so the budget stays meaningful."""
    project_root, path = sealed

    with pytest.raises(SealedSplitError):
        open_sealed(path, project_root)

    assert read_access_log(project_root) == []


def test_opening_requires_a_stated_reason(sealed) -> None:
    project_root, path = sealed

    with pytest.raises(SealedSplitError, match="requires a stated reason"):
        open_sealed(path, project_root, allow=True, reason="   ")


def test_an_authorised_opening_is_recorded_with_its_provenance(sealed) -> None:
    project_root, path = sealed

    frame, record = open_sealed(
        path, project_root, allow=True, reason="final report, single opening"
    )

    assert len(frame) == 2
    assert record["reason"] == "final report, single opening"
    assert record["opening_number"] == 1
    assert record["sha256"] == file_digest(path)
    assert "test_sealed_split.py" in record["called_from"]
    assert len(read_access_log(project_root)) == 1


def test_repeated_openings_are_counted(sealed) -> None:
    project_root, path = sealed

    for i in range(3):
        _, record = open_sealed(path, project_root, allow=True, reason=f"opening {i}")

    assert record["opening_number"] == 3
    assert verify(path, project_root)["openings"] == 3


def test_exceeding_the_budget_is_reported_not_hidden(sealed) -> None:
    project_root, path = sealed
    for i in range(OPENING_BUDGET + 1):
        open_sealed(path, project_root, allow=True, reason=f"opening {i}")

    result = verify(path, project_root)

    assert result["openings"] == OPENING_BUDGET + 1
    assert not result["within_budget"]
    assert not result["passed"]


def test_a_changed_sealed_file_is_detected(sealed) -> None:
    """If the split is rebuilt after sealing, the numbers are not the ones the
    manifest attests to."""
    project_root, path = sealed
    pd.DataFrame({"anchor_episode_id": ["EP-9"], "has_successor_primary": [True]}).to_parquet(path)

    with pytest.raises(SealedSplitError, match="does not match its manifest checksum"):
        open_sealed(path, project_root, allow=True, reason="after tampering")

    assert not verify(path, project_root)["checksum_matches"]


def test_the_manifest_does_not_leak_the_labels(sealed) -> None:
    """A manifest listing per-anchor outcomes would let the split be read
    without opening it."""
    project_root, path = sealed
    manifest = json.loads((path.parent / MANIFEST_NAME).read_text())

    assert "anchors" in manifest
    for forbidden in ("labels", "successors", "has_successor_primary", "anchor_episode_id"):
        assert forbidden not in json.dumps(manifest)


def test_verify_passes_on_an_untouched_seal(sealed) -> None:
    project_root, path = sealed

    result = verify(path, project_root)

    assert result["passed"]
    assert result["openings"] == 0
    assert result["checksum_matches"]
