#!/usr/bin/env python3
"""Build and evaluate the parallel expiry-aware successor policy.

This script does not change candidate generation or the primary linkage output.
It enriches the existing candidate pool with the best observable anchor expiry
evidence, applies ``M_E_expiry_aware_text``, and writes separate audit artifacts.
Accuracy comparisons use the same national dev and validation benchmark as the
four primary linkage methods.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.expiry_linkage import (  # noqa: E402
    DAYS_PER_MONTH,
    EARLY_BOUNDARY_DAYS,
    EARLY_TEXT_THRESHOLD,
    METHOD_NAME,
    NORMAL_TEXT_THRESHOLD,
    add_expiry_timing,
    build_expiry_features,
    expiry_aware_eligible,
    select_expiry_aware,
)
from boamp_pipeline.benchmark_io import load_truth  # noqa: E402
from scripts.evaluate_linkage import evaluate, predict  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data/processed/boamp")
EXPIRY_POLICY_SCHEMA = "boamp_expiry_aware_linkage_schema_1.0"


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "evaluate_expiry_aware_linkage.log", encoding="utf-8"),
        ],
    )


def _national_expiry_features(index: pd.DataFrame) -> pd.DataFrame:
    """Adapt the national index's resolved dates to the expiry-policy contract."""
    columns = [
        "episode_id",
        "expected_end_date",
        "expected_end_source",
        "explicit_start_date",
        "explicit_end_date",
        "explicit_start_date_count",
        "explicit_end_date_count",
    ]
    missing = set(columns) - set(index.columns)
    if missing:
        raise RuntimeError(f"national episode index is missing expiry fields: {sorted(missing)}")
    features = index[columns].copy()
    features = features.rename(columns={"expected_end_date": "anchor_expected_end_date"})
    features["expected_end_source"] = features["expected_end_source"].fillna("unavailable")
    return features


def benchmark_metrics(project_root: Path, output_dir: Path) -> dict[str, Any]:
    """Compare primary and expiry policies on the canonical national benchmark."""
    benchmark_dir = output_dir / "benchmark"
    exposure = pd.read_parquet(benchmark_dir / "exposure_full.parquet")
    national_index = pd.read_parquet(benchmark_dir / "national_episode_index.parquet")
    enriched = add_expiry_timing(exposure, _national_expiry_features(national_index))
    baseline_predictions = predict(enriched, "M_B_text_ranking", 70.0)
    expiry_predictions = select_expiry_aware(enriched)

    result: dict[str, Any] = {}
    for split_name in ("dev", "validation"):
        truth, _ = load_truth(
            benchmark_dir,
            split_name,
            "primary",
            project_root=project_root,
        )
        truth = truth.loc[truth["truth_usable"]].copy()
        anchors = set(truth["anchor_episode_id"])
        result[split_name] = {
            "M_B_text_ranking_70": evaluate(
                baseline_predictions.loc[
                    baseline_predictions["anchor_episode_id"].isin(anchors)
                ],
                truth,
            ),
            METHOD_NAME: evaluate(
                expiry_predictions.loc[expiry_predictions["anchor_episode_id"].isin(anchors)],
                truth,
            ),
        }
    return result


def cohort_description(cohort: pd.DataFrame, links: pd.DataFrame) -> dict[str, Any]:
    linked_ids = set(links["anchor_episode_id"])
    frame = cohort.copy()
    frame["event"] = frame["episode_id"].isin(linked_ids).astype(int)

    def grouped(column: str) -> dict[str, dict[str, Any]]:
        grouped_frame = frame.groupby(column, dropna=False).agg(rows=("episode_id", "size"), events=("event", "sum"))
        grouped_frame["event_rate"] = (grouped_frame["events"] / grouped_frame["rows"]).round(4)
        return {str(key): value for key, value in grouped_frame.to_dict(orient="index").items()}

    return {
        "cohort_rows": int(len(frame)),
        "accepted_links": int(len(links)),
        "event_rate": round(float(frame["event"].mean()), 4),
        "median_time_to_successor_months": (
            round(float(links["gap_days"].median() / DAYS_PER_MONTH), 2) if len(links) else None
        ),
        "by_digital_segment": grouped("digital_segment"),
        "by_region": grouped("buyer_region"),
    }


def _candidate_lookup(candidates: pd.DataFrame) -> pd.DataFrame:
    return candidates.drop_duplicates(["anchor_episode_id", "candidate_episode_id"]).set_index(
        ["anchor_episode_id", "candidate_episode_id"]
    )


def _episode_value(episodes: pd.DataFrame, episode_id: Any, column: str) -> Any:
    if pd.isna(episode_id) or episode_id not in episodes.index:
        return None
    return episodes.at[episode_id, column]


def build_review(
    baseline_links: pd.DataFrame,
    expiry_links: pd.DataFrame,
    candidates: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Describe every primary link removed or replaced by the expiry-aware rule."""
    old_top = baseline_links.set_index("anchor_episode_id")["candidate_episode_id"]
    new_top = expiry_links.set_index("anchor_episode_id")["candidate_episode_id"]
    changed = [anchor for anchor, candidate in old_top.items() if new_top.get(anchor) != candidate]
    lookup = _candidate_lookup(candidates)
    episodes = episodes.set_index("episode_id")

    rows: list[dict[str, Any]] = []
    for anchor_id in changed:
        old_id = old_top[anchor_id]
        new_id = new_top.get(anchor_id)
        old = lookup.loc[(anchor_id, old_id)]
        new = lookup.loc[(anchor_id, new_id)] if pd.notna(new_id) else None
        old_eligible = bool(expiry_aware_eligible(pd.DataFrame([old])).iloc[0])

        reasons: list[str] = []
        if old["timing_class"] == "early_more_than_1y":
            if float(old.get("text_component", 0) or 0) < EARLY_TEXT_THRESHOLD:
                reasons.append("text_below_0.85")
            if pd.isna(old.get("cpv_component")) or float(old.get("cpv_component") or 0) <= 0:
                reasons.append("no_positive_cpv_continuity")
        if old_eligible:
            reasons.append("deterministic_ranking_changed")

        row = {
            "change_type": "replaced" if pd.notna(new_id) else "removed",
            "rejection_reason": ";".join(reasons) or "not_eligible",
            "anchor_episode_id": anchor_id,
            "anchor_buyer_name": old.get("anchor_buyer_name"),
            "anchor_award_date": old.get("anchor_award_date"),
            "anchor_expected_end_date": old.get("anchor_expected_end_date"),
            "expected_end_source": old.get("expected_end_source"),
            "anchor_main_cpv": _episode_value(episodes, anchor_id, "main_cpv"),
            "anchor_all_cpvs_json": _episode_value(episodes, anchor_id, "all_cpvs_json"),
            "anchor_text": _episode_value(episodes, anchor_id, "episode_text"),
            "old_candidate_episode_id": old_id,
            "old_candidate_origin_date": old.get("candidate_origin_date"),
            "old_timing_class": old.get("timing_class"),
            "old_days_from_expected_end": old.get("days_from_expected_end"),
            "old_text_component": old.get("text_component"),
            "old_cpv_component": old.get("cpv_component"),
            "old_linkage_score": old.get("linkage_score"),
            "old_candidate_main_cpv": _episode_value(episodes, old_id, "main_cpv"),
            "old_candidate_all_cpvs_json": _episode_value(episodes, old_id, "all_cpvs_json"),
            "old_candidate_text": _episode_value(episodes, old_id, "episode_text"),
            "new_candidate_episode_id": new_id,
            "new_candidate_origin_date": new.get("candidate_origin_date") if new is not None else None,
            "new_timing_class": new.get("timing_class") if new is not None else None,
            "new_days_from_expected_end": new.get("days_from_expected_end") if new is not None else None,
            "new_text_component": new.get("text_component") if new is not None else None,
            "new_cpv_component": new.get("cpv_component") if new is not None else None,
            "new_linkage_score": new.get("linkage_score") if new is not None else None,
            "new_candidate_main_cpv": _episode_value(episodes, new_id, "main_cpv"),
            "new_candidate_all_cpvs_json": _episode_value(episodes, new_id, "all_cpvs_json"),
            "new_candidate_text": _episode_value(episodes, new_id, "episode_text"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def promotion_gate(benchmark: dict[str, Any]) -> dict[str, Any]:
    baseline = benchmark["validation"]["M_B_text_ranking_70"]
    challenger = benchmark["validation"][METHOD_NAME]
    positive_anchors = int(baseline["positive_anchors"])
    allowed_recall_loss = 1 / positive_anchors if positive_anchors else 0.0
    checks = {
        "precision_at_least_0_80": bool((challenger["precision_at_1"] or 0) >= 0.80),
        "false_positive_rate_no_higher_than_primary": bool(
            (challenger["false_positive_rate_on_negatives"] or 0)
            <= (baseline["false_positive_rate_on_negatives"] or 0)
        ),
        "recall_loses_no_more_than_one_positive": bool(
            (challenger["recall_at_1"] or 0) >= (baseline["recall_at_1"] or 0) - allowed_recall_loss
        ),
    }
    return {
        "criteria": checks,
        "passed": all(checks.values()),
        "recommendation": "review_then_consider_promotion" if all(checks.values()) else "retain_primary",
        "manual_review_required_before_promotion": True,
    }


def build(project_root: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    links_path = output_dir / "accepted_successor_links_expiry_aware.parquet"
    summary_path = output_dir / "expiry_aware_linkage_summary.json"
    review_path = output_dir / "expiry_link_review.csv"
    targets = (links_path, summary_path, review_path)
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(f"{existing[0]} already exists. Use --force to rebuild parallel artifacts.")

    scored_path = output_dir / "linkage_candidates_scored.parquet"
    candidates_path = scored_path if scored_path.exists() else output_dir / "linkage_candidates.parquet"
    candidates = pd.read_parquet(candidates_path)
    cohort = pd.read_parquet(
        output_dir / "survival_cohort.parquet",
        columns=[
            "episode_id",
            "award_date",
            "duration_months_reliable",
            "digital_segment",
            "buyer_region",
        ],
    )
    membership = pd.read_parquet(output_dir / "episode_membership.parquet", columns=["idweb", "episode_id"])
    notices = pd.read_parquet(
        output_dir / "notices_grand_ouest.parquet",
        columns=["idweb", "contract_start_date", "contract_end_date"],
    )

    logging.info("Building expiry evidence for %s cohort contracts", f"{len(cohort):,}")
    expiry_features = build_expiry_features(cohort, membership, notices)
    enriched = add_expiry_timing(candidates, expiry_features)
    predictions = select_expiry_aware(enriched)
    links = predictions.loc[predictions["predicted_rank"] == 1].copy()
    links["selection_method"] = METHOD_NAME
    links["expiry_policy_schema"] = EXPIRY_POLICY_SCHEMA
    links.to_parquet(links_path, index=False, compression="zstd")

    baseline_links = pd.read_parquet(output_dir / "accepted_successor_links.parquet")
    episodes = pd.read_parquet(
        output_dir / "episodes_grand_ouest.parquet",
        columns=["episode_id", "episode_text", "main_cpv", "all_cpvs_json"],
    )
    review = build_review(baseline_links, links, enriched, episodes)
    review.to_csv(review_path, index=False)

    benchmark = benchmark_metrics(project_root, output_dir)
    baseline_description = cohort_description(cohort, baseline_links)
    expiry_description = cohort_description(cohort, links)
    old_pairs = set(zip(baseline_links["anchor_episode_id"], baseline_links["candidate_episode_id"]))
    new_pairs = set(zip(links["anchor_episode_id"], links["candidate_episode_id"]))
    old_by_anchor = baseline_links.set_index("anchor_episode_id")["candidate_episode_id"].to_dict()
    new_by_anchor = links.set_index("anchor_episode_id")["candidate_episode_id"].to_dict()
    changed_primary_anchors = {
        anchor for anchor, candidate in old_by_anchor.items() if new_by_anchor.get(anchor) != candidate
    }
    timing_counts = links["timing_class"].value_counts(dropna=False)
    source_counts = expiry_features["expected_end_source"].value_counts(dropna=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "expiry_policy_schema": EXPIRY_POLICY_SCHEMA,
        "method": METHOD_NAME,
        "candidate_source": str(candidates_path),
        "candidate_generation_changed": False,
        "candidate_pairs": int(len(candidates)),
        "policy": {
            "normal_text_threshold": NORMAL_TEXT_THRESHOLD,
            "early_boundary_days": EARLY_BOUNDARY_DAYS,
            "early_text_threshold": EARLY_TEXT_THRESHOLD,
            "early_requires_positive_cpv_continuity": True,
            "ranking": [
                "text_component desc",
                "linkage_score desc",
                "candidate_origin_date asc",
                "candidate_episode_id asc",
            ],
        },
        "expiry_source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "accepted_timing_class_counts": {str(key): int(value) for key, value in timing_counts.items()},
        "cohort_comparison": {
            "primary": baseline_description,
            "expiry_aware": expiry_description,
            "retained_pairs": len(old_pairs & new_pairs),
            "removed_pairs": len(old_pairs - new_pairs),
            "new_pairs": len(new_pairs - old_pairs),
            "changed_anchors_for_review": int(len(review)),
        },
        "benchmark_comparison": benchmark,
        "promotion_gate": promotion_gate(benchmark),
        "outputs": {
            "accepted_links": str(links_path),
            "review_csv": str(review_path),
        },
        "event_interpretation": "observable successor procurement, not a confirmed legal renewal",
        "validation": {
            "unique_anchor_links": bool(links["anchor_episode_id"].is_unique),
            "no_self_links": bool((links["anchor_episode_id"] != links["candidate_episode_id"]).all()),
            "review_covers_every_changed_primary_anchor": bool(
                set(review["anchor_episode_id"]) == changed_primary_anchors
            ),
        },
    }
    summary["validation_passed"] = all(summary["validation"].values())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    configure_logging(project_root)
    summary = build(project_root, output_dir, args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["validation_passed"]:
        raise RuntimeError("Expiry-aware linkage validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
