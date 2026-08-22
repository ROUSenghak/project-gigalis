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


def test_linkage_reference_is_not_overclaimed_as_untouched_holdout() -> None:
    """Project history shows that reference evidence informed the retained rule.

    The reference labels still predate the methods, but that is different from
    saying the final operating policy was never informed by the reference.
    """
    paths = [
        PROJECT_ROOT / "scripts/evaluate_linkage.py",
        PROJECT_ROOT / "scripts/build_regional_benchmark.py",
        PROJECT_ROOT / "scripts/build_quality_evidence.py",
        PROJECT_ROOT / "scripts/build_project_evidence.py",
        PROJECT_ROOT / "scripts/refresh_reader_artifacts.py",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "PROJECT_WORK_PROTOCOL.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "internal validation" in text.lower()
    for unsupported in (
        "operating point was frozen before",
        "fixed before this reference was read",
        "fixed before the regional reference was consulted",
        "only reason the locked split can be reported as held out",
    ):
        assert unsupported not in text.lower(), unsupported


def test_a_threshold_free_comparator_is_not_shown_with_a_threshold() -> None:
    """``M_A_deterministic`` is a conjunction of fixed gates and reads no
    threshold, so printing one advertises an operating point that does not
    exist."""
    import json

    import pytest

    from scripts.evaluate_linkage import THRESHOLD_FREE_METHODS

    assert "M_A_deterministic" in THRESHOLD_FREE_METHODS

    evaluation = PROJECT_ROOT / "data/processed/boamp/linkage_evaluation_validation.json"
    if not evaluation.exists():
        pytest.skip("linkage evaluation not materialised")
    methods = json.loads(evaluation.read_text(encoding="utf-8"))["methods"]
    for method in methods:
        if method["method"] in THRESHOLD_FREE_METHODS:
            assert method["threshold"] is None, method["method"]
        else:
            assert method["threshold"] is not None, method["method"]

    # Only tables that actually carry a threshold column are relevant; the pair
    # ROC/average-precision table lists the same method with no threshold at all.
    for name in ("QUALITY_EVIDENCE.md", "REGIONAL_BENCHMARK_REFERENCE.md"):
        path = PROJECT_ROOT / name
        if not path.exists():
            continue
        threshold_column: int | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("|"):
                threshold_column = None
                continue
            cells = [cell.strip().strip("`").lower() for cell in line.split("|")[1:-1]]
            if "threshold" in cells:
                threshold_column = cells.index("threshold")
                continue
            if threshold_column is None or "m_a_deterministic" not in cells:
                continue
            assert cells[threshold_column] == "n/a", line


def test_the_data_quality_report_states_the_any_code_cohort_rule() -> None:
    """"CPV divisions 32/35/48/72" reads as a main-CPV rule and is not one."""
    import json

    import pytest

    profile_path = PROJECT_ROOT / "data/processed/boamp/data_quality_profile.json"
    report_path = PROJECT_ROOT / "DATA_QUALITY_REPORT.md"
    if not (profile_path.exists() and report_path.exists()):
        pytest.skip("data quality evidence not materialised")

    scope = json.loads(profile_path.read_text(encoding="utf-8"))["cohort_scope"]
    report = report_path.read_text(encoding="utf-8")

    assert "any-code rule at episode level" in report
    assert "lowest-numbered digital division present" in report
    assert f"{scope['main_cpv_outside_digital_set']:,}" in report
    assert f"{scope['main_cpv_outside_digital_set_share']:.1%}" in report
    # The measured share must be recomputable, not asserted.
    assert 0 < scope["main_cpv_outside_digital_set_share"] < 1
    assert (
        scope["main_cpv_inside_digital_set"] + scope["main_cpv_outside_digital_set"]
        == scope["cohort_rows"]
    )


def test_the_integrity_table_reports_its_non_zero_counts() -> None:
    """A table of nothing-but-zeros hides the counts worth looking at."""
    import json

    import pytest

    profile_path = PROJECT_ROOT / "data/processed/boamp/data_quality_profile.json"
    report_path = PROJECT_ROOT / "DATA_QUALITY_REPORT.md"
    if not (profile_path.exists() and report_path.exists()):
        pytest.skip("data quality evidence not materialised")

    integrity = json.loads(profile_path.read_text(encoding="utf-8"))["integrity"]
    report = report_path.read_text(encoding="utf-8")
    for key in ("reference_conflict_episodes", "suspicious_review_cases"):
        assert integrity[key] > 0, key
        assert f"{integrity[key]:,}" in report, key
    assert "Not an error" in report
