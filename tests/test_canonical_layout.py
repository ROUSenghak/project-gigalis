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
