"""Prevent legacy project versions from re-entering the active workflow."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PROCESSED = PROJECT_ROOT / "data" / "processed" / "boamp"


def test_only_the_canonical_processed_tree_is_active() -> None:
    legacy_processed = PROJECT_ROOT / "data" / "processed" / ("boamp_" + "v2")
    legacy_remap = CANONICAL_PROCESSED / ("benchmark_" + "remap")

    assert CANONICAL_PROCESSED.is_dir()
    assert not legacy_processed.exists()
    assert not legacy_remap.exists()


def test_active_sources_do_not_route_to_legacy_artifacts() -> None:
    forbidden = (
        "data/processed/" + "boamp_v2",
        "benchmark_" + "v3",
        "benchmark_" + "remap",
        "linkage_" + "frozen_config",
        "remap_benchmark_to_" + "v2",
    )
    roots = [
        PROJECT_ROOT / "boamp_pipeline",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "tests",
    ]
    top_level = list(PROJECT_ROOT.glob("*.md"))
    files = top_level + [
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".md", ".ipynb"}
    ]

    violations: list[str] = []
    for path in files:
        if path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token in text or token in path.name:
                violations.append(f"{path.relative_to(PROJECT_ROOT)}: {token}")
    assert not violations, "legacy artifact references remain:\n" + "\n".join(violations)


def test_materialized_json_metadata_uses_canonical_paths() -> None:
    forbidden = ("data/processed/boamp_v2", "/benchmark_v3/")
    violations = []
    for path in CANONICAL_PROCESSED.rglob("*.json"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(PROJECT_ROOT)}: {token}")
    assert not violations, "legacy metadata references remain:\n" + "\n".join(violations)


def test_hand_maintained_documents_quote_the_canonical_numbers() -> None:
    """README, the protocol and the compliance file are written by hand.

    Everything generated is regenerated every run and cannot drift. These three
    are not, so the headline figures they restate are checked against the
    artifacts they came from. A literal that no longer matches its source is the
    exact failure mode this test exists to catch.
    """
    import json
    from pathlib import Path

    import pandas as pd
    import pytest

    processed = Path("data/processed/boamp")
    technology = processed / "technology"
    if not (processed / "survival_analysis_summary.json").exists():
        pytest.skip("pipeline outputs not materialised")

    survival = json.loads((processed / "survival_analysis_summary.json").read_text())
    config = json.loads((technology / "final_model_config.json").read_text())
    bootstrap = pd.read_csv(technology / "bootstrap_macro_f1_ci.csv").set_index("model")
    paired = pd.read_csv(technology / "bootstrap_paired_differences.csv")
    selected = config["specification"]
    gain = paired.loc[
        (paired["model_a"] == "M0b_cpv_descriptor") & (paired["model_b"] == selected)
    ].iloc[0]
    detectability = pd.read_csv(
        processed / "survival_cox_detectability_sensitivity.csv"
    ).set_index("covariate")
    scope = json.loads((processed / "data_quality_profile.json").read_text())["cohort_scope"]

    expected = {
        f"{survival['cohort']['contracts']:,}",
        f"{survival['cohort']['events']:,}",
        f"{bootstrap.loc[selected, 'macro_f1']:.3f}",
        f"{bootstrap.loc['M0b_cpv_descriptor', 'macro_f1']:.3f}",
        f"{abs(gain['observed_difference']):.3f}",
    }
    readme = Path("README.md").read_text(encoding="utf-8")
    protocol = Path("PROJECT_WORK_PROTOCOL.md").read_text(encoding="utf-8")
    compliance = Path("INTERNSHIP_GUIDE_COMPLIANCE.md").read_text(encoding="utf-8")

    for value in expected:
        assert value in readme, f"README.md no longer quotes {value}"
    for value in expected:
        assert value in protocol or value in compliance, (
            f"neither the protocol nor the compliance file quotes {value}"
        )

    # The cohort-scope numbers and the detectability comparison are new claims
    # in these files; they must match what the pipeline measured.
    assert f"{scope['main_cpv_outside_digital_set']:,}" in readme
    assert f"{scope['main_cpv_outside_digital_set_share']:.1%}" in readme
    for covariate in ("framework_flag", "digital_segment_CPV-35"):
        row = detectability.loc[covariate]
        assert f"{row['hazard_ratio_main']:.3f}" in readme, covariate
        assert f"{row['hazard_ratio_pool_adjusted']:.3f}" in readme, covariate

    # The deployed confidence variant must be named, and the rejected one must
    # not be presented as what shipped.
    variant = config["calibration"]["deployed_variant"]
    assert variant in readme
    if not config["calibration"]["adopted"]:
        assert "rejected" in readme
        assert "Platt scaling was evaluated and rejected" in compliance
