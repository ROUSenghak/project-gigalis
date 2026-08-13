#!/usr/bin/env python3
"""Build quality-evidence tables and plots for the national linkage benchmark."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_linkage import (  # noqa: E402
    evaluate,
    method_a_deterministic,
    method_c_weighted_gated,
    predict,
)
from boamp_pipeline.benchmark_io import load_truth  # noqa: E402

PROCESSED = PROJECT_ROOT / "data/processed/boamp"
BENCHMARK = PROCESSED / "benchmark"
OUTPUT_DIR = PROCESSED / "quality_evidence"
FIGURE_DIR = PROJECT_ROOT / "reports/figures"

METHOD_LABELS = {
    "M_A_deterministic": "M_A deterministic",
    "M_B_text_ranking": "M_B text ranking",
    "M_C_weighted_gated": "M_C weighted gated",
    "M_D_fellegi_sunter": "M_D Fellegi-Sunter",
}

M_B_THRESHOLD_GRID = tuple(float(value) for value in range(0, 101))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def anchor_confusion(summary: dict[str, Any]) -> pd.DataFrame:
    """Return anchor-level event and exact-successor error accounting.

    ``event`` treats any accepted link on an anchor with a labelled successor as
    a positive event detection, even if the chosen candidate is the wrong
    successor. ``exact_successor`` follows the project precision/recall
    definition: a wrong successor is both a false accepted link and a missed
    true successor, so that accounting is not a conventional matrix whose cells
    sum to anchors.
    """
    rows: list[dict[str, Any]] = []
    for method in summary["methods"]:
        metrics = method["unweighted_all_frames"]
        exact_tp = int(metrics["true_positive"])
        wrong_successor = int(metrics["false_positive_wrong_successor"])
        fp_no_successor = int(metrics["false_positive_on_no_successor_anchor"])
        fn_abstained = int(metrics["false_negative_abstained"])
        tn = int(metrics["true_negative_abstained"])
        positives = int(metrics["positive_anchors"])
        negatives = int(metrics["negative_anchors"])

        rows.append(
            {
                "matrix_type": "event_detection",
                "method": method["method"],
                "threshold": method["threshold"],
                "tp": exact_tp + wrong_successor,
                "fp": fp_no_successor,
                "fn": fn_abstained,
                "tn": tn,
                "positive_anchors": positives,
                "negative_anchors": negatives,
                "note": "wrong successor on a positive anchor counts as event detected but candidate incorrect",
            }
        )
        rows.append(
            {
                "matrix_type": "exact_successor_accounting",
                "method": method["method"],
                "threshold": method["threshold"],
                "tp": exact_tp,
                "fp": wrong_successor + fp_no_successor,
                "fn": positives - exact_tp,
                "tn": tn,
                "positive_anchors": positives,
                "negative_anchors": negatives,
                "note": "wrong successor counts as both FP and FN; cells need not sum to anchors",
            }
        )
    return pd.DataFrame(rows)


def method_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    m_c_gate = method_c_weighted_gated(frame, 0.0)
    return {
        "M_A_deterministic": method_a_deterministic(frame, 70.0).astype(float).to_numpy(),
        "M_B_text_ranking": frame["text_component"].fillna(0).to_numpy(dtype=float),
        "M_C_weighted_gated": np.where(
            m_c_gate.to_numpy(),
            frame["linkage_score"].fillna(0).to_numpy(dtype=float) / 100.0,
            -1.0,
        ),
        "M_D_fellegi_sunter": frame["fs_match_probability"].fillna(0).to_numpy(dtype=float),
    }


def curve_points(y_true: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-score, kind="mergesort")
    ranked_y = y_true[order]
    ranked_score = score[order]
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        raise RuntimeError("ROC/PR curves require both positive and negative pairs")

    distinct = np.r_[np.where(np.diff(ranked_score) != 0)[0], len(ranked_score) - 1]
    tp = np.cumsum(ranked_y)[distinct]
    fp = (1 + distinct) - tp
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    fpr = fp / negatives

    roc_fpr = np.r_[0.0, fpr, 1.0]
    roc_tpr = np.r_[0.0, recall, 1.0]
    roc_auc = float(np.trapezoid(roc_tpr, roc_fpr))

    pr_recall = np.r_[0.0, recall]
    pr_precision = np.r_[1.0, precision]
    average_precision = float(np.sum(np.diff(pr_recall) * pr_precision[1:]))

    return {
        "thresholds": ranked_score[distinct],
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
        "roc_auc": roc_auc,
        "pr_recall": pr_recall,
        "pr_precision": pr_precision,
        "average_precision": average_precision,
        "positive_pairs": positives,
        "negative_pairs": negatives,
    }


def pair_curve_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    y_true = frame["y_primary"].to_numpy(dtype=int)
    curves: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for method, score in method_scores(frame).items():
        curve = curve_points(y_true, score)
        curves[method] = curve
        rows.append(
            {
                "method": method,
                "pair_roc_auc": round(curve["roc_auc"], 4),
                "pair_average_precision": round(curve["average_precision"], 4),
                "positive_pairs": curve["positive_pairs"],
                "negative_pairs": curve["negative_pairs"],
                "diagnostic_level": "pair-level score diagnostic, not top-1 anchor selection",
            }
        )
    return pd.DataFrame(rows), curves


def threshold_sweep(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    thresholds: tuple[float, ...] = M_B_THRESHOLD_GRID,
) -> pd.DataFrame:
    """Evaluate the actual top-1 M_B decision over a threshold grid."""
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = evaluate(
            predict(candidates, "M_B_text_ranking", threshold),
            truth,
        )
        rows.append(
            {
                "threshold": threshold / 100.0,
                "threshold_percent": threshold,
                "accepted_links": metrics["accepted_links"],
                "true_positive": metrics["true_positive"],
                "false_positive_wrong_successor": metrics[
                    "false_positive_wrong_successor"
                ],
                "false_positive_on_no_successor_anchor": metrics[
                    "false_positive_on_no_successor_anchor"
                ],
                "precision": metrics["precision_at_1"],
                "recall": metrics["recall_at_1"],
                "false_positive_rate": metrics[
                    "false_positive_rate_on_negatives"
                ],
                "coverage": metrics["coverage"],
                "positive_anchors": metrics["positive_anchors"],
                "negative_anchors": metrics["negative_anchors"],
            }
        )
    return pd.DataFrame(rows)


def assert_frozen_m_b_point(sweep: pd.DataFrame, summary: dict[str, Any]) -> None:
    """Fail if the plotted 0.70 point diverges from the official evaluator."""
    frozen = sweep.loc[sweep["threshold_percent"].eq(70.0)]
    if len(frozen) != 1:
        raise RuntimeError("threshold sweep must contain exactly one 0.70 row")
    plotted = frozen.iloc[0]
    method = next(
        entry for entry in summary["methods"]
        if entry["method"] == "M_B_text_ranking"
    )
    official = method["unweighted_all_frames"]
    comparisons = {
        "accepted_links": official["accepted_links"],
        "precision": official["precision_at_1"],
        "recall": official["recall_at_1"],
        "false_positive_rate": official["false_positive_rate_on_negatives"],
    }
    for column, expected in comparisons.items():
        actual = plotted[column]
        if pd.isna(actual) and expected is None:
            continue
        if not np.isclose(float(actual), float(expected)):
            raise RuntimeError(
                f"threshold plot mismatch at 0.70 for {column}: "
                f"plotted={actual}, official={expected}"
            )


def plot_confusion(confusion: pd.DataFrame, path: Path) -> None:
    event = confusion.loc[confusion["matrix_type"].eq("event_detection")].reset_index(drop=True)
    fig, axes = plt.subplots(1, len(event), figsize=(13.2, 3.4), constrained_layout=True)
    vmax = int(event[["tp", "fp", "fn", "tn"]].to_numpy().max())
    for ax, row in zip(axes, event.itertuples(index=False)):
        matrix = np.array([[row.tp, row.fn], [row.fp, row.tn]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(METHOD_LABELS[row.method], fontsize=10)
        ax.set_xticks([0, 1], ["Pred link", "Pred abstain"], rotation=28, ha="right")
        ax.set_yticks([0, 1], ["Actual successor", "Actual no successor"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="#1f2933")
    fig.suptitle("Held-out internal anchor-level event-detection matrices", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_curves(curves: dict[str, dict[str, Any]], path_roc: Path, path_pr: Path) -> None:
    colors = {
        "M_A_deterministic": "#69727D",
        "M_B_text_ranking": "#2F6B9A",
        "M_C_weighted_gated": "#C28B2C",
        "M_D_fellegi_sunter": "#3B8178",
    }

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.plot([0, 1], [0, 1], color="#9AA3AD", linestyle="--", linewidth=1, label="Random")
    for method, curve in curves.items():
        ax.plot(
            curve["roc_fpr"],
            curve["roc_tpr"],
            label=f"{METHOD_LABELS[method]} (AUC {curve['roc_auc']:.3f})",
            color=colors[method],
            linewidth=2,
        )
    ax.set_title("Held-out internal pair-level ROC curves")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path_roc.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_roc, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    base_rate = next(iter(curves.values()))["positive_pairs"] / (
        next(iter(curves.values()))["positive_pairs"]
        + next(iter(curves.values()))["negative_pairs"]
    )
    ax.axhline(
        base_rate,
        color="#9AA3AD",
        linestyle="--",
        linewidth=1,
        label=f"Base rate {base_rate:.3f}",
    )
    for method, curve in curves.items():
        ax.plot(
            curve["pr_recall"],
            curve["pr_precision"],
            label=f"{METHOD_LABELS[method]} (AP {curve['average_precision']:.3f})",
            color=colors[method],
            linewidth=2,
        )
    ax.set_title("Held-out internal pair-level precision-recall curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path_pr.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pr, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_threshold_tradeoff(sweep: pd.DataFrame, path: Path) -> None:
    """Plot the top-1 threshold trade-off without interpolation or smoothing."""
    visible = sweep.loc[sweep["threshold"].between(0.10, 0.90)].copy()
    operating = visible.loc[visible["threshold_percent"].isin([50, 60, 70, 75, 80])]

    colors = {
        "precision": "#2F6B9A",
        "recall": "#C07A24",
        "false_positive_rate": "#3B8178",
        "accepted": "#69727D",
        "correct": "#2F6B9A",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=True)

    ax = axes[0]
    for metric, label in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("false_positive_rate", "FPR"),
    ]:
        ax.step(
            visible["threshold"],
            visible[metric],
            where="post",
            linewidth=2,
            color=colors[metric],
            label=label,
        )
    ax.axvline(0.70, color="#20262E", linestyle="--", linewidth=1.3)
    ax.text(0.705, 0.04, "Frozen 0.70", rotation=90, va="bottom", fontsize=8)
    ax.set_title("Metrics by acceptance threshold", fontsize=11)
    ax.set_xlabel("Minimum text similarity")
    ax.set_ylabel("Rate")
    ax.set_xlim(0.10, 0.90)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    valid = visible.dropna(subset=["precision"])
    ax.plot(
        valid["recall"],
        valid["precision"],
        color="#69727D",
        linewidth=1.5,
        marker="o",
        markersize=2.5,
    )
    for row in operating.itertuples(index=False):
        if pd.isna(row.precision):
            continue
        ax.scatter(row.recall, row.precision, color="#C07A24", s=28, zorder=3)
        ax.annotate(
            f"{row.threshold:.2f}",
            (row.recall, row.precision),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    frozen = sweep.loc[sweep["threshold_percent"].eq(70)].iloc[0]
    ax.scatter(
        frozen["recall"], frozen["precision"],
        facecolor="white", edgecolor="#20262E", linewidth=1.5, s=80, zorder=4,
    )
    ax.set_title("Precision-recall operating points", fontsize=11)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.22)

    ax = axes[2]
    ax.step(
        visible["threshold"], visible["accepted_links"], where="post",
        color=colors["accepted"], linewidth=2, label="Accepted links",
    )
    ax.step(
        visible["threshold"], visible["true_positive"], where="post",
        color=colors["correct"], linewidth=2, label="Correct successors",
    )
    ax.axvline(0.70, color="#20262E", linestyle="--", linewidth=1.3)
    ax.set_title("Decision volume by threshold", fontsize=11)
    ax.set_xlabel("Minimum text similarity")
    ax.set_ylabel("Validation anchors")
    ax.set_xlim(0.10, 0.90)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "M_B top-1 threshold trade-off on the internal validation reference",
        fontsize=13,
    )
    fig.text(
        0.5,
        -0.035,
        "60 anchors (22 with a labelled successor). Empirical steps only; no smoothing. "
        "Bootstrap labels are development evidence, not independent validation.",
        ha="center",
        fontsize=8.5,
        color="#4A535E",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in view.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_quality_markdown(
    validation_confusion: pd.DataFrame,
    validation_pair_metrics: pd.DataFrame,
    validation_threshold_sweep: pd.DataFrame,
) -> Path:
    event = validation_confusion.loc[
        validation_confusion["matrix_type"].eq("event_detection")
    ].copy()
    exact = validation_confusion.loc[
        validation_confusion["matrix_type"].eq("exact_successor_accounting")
    ].copy()
    event = event[["method", "threshold", "tp", "fp", "fn", "tn"]]
    exact = exact[["method", "threshold", "tp", "fp", "fn", "tn"]]
    pair = validation_pair_metrics[
        ["method", "pair_roc_auc", "pair_average_precision", "positive_pairs", "negative_pairs"]
    ].copy()
    operating = validation_threshold_sweep.loc[
        validation_threshold_sweep["threshold_percent"].isin([50, 55, 60, 65, 70, 75, 80]),
        [
            "threshold", "accepted_links", "true_positive", "precision", "recall",
            "false_positive_rate", "coverage",
        ],
    ].copy()

    path = PROJECT_ROOT / "QUALITY_EVIDENCE.md"
    path.write_text(
        "# Linkage Quality Evidence\n\n"
        "Generated from the held-out split of the current national development reference. These labels come from deterministic bootstrap rules, not independent specialist review.\n\n"
        "## Academic Basis And Evidence Boundary\n\n"
        "The metric choice is supported by [Davis and Goadrich (2006)](https://doi.org/10.1145/1143844.1143874), who analyse the relationship between ROC and precision-recall curves for skewed binary decisions, and [Saito and Rehmsmeier (2015)](https://doi.org/10.1371/journal.pone.0118432), who show why precision-recall analysis is more informative for imbalanced data. The classical probabilistic linkage comparator follows [Fellegi and Sunter (1969)](https://doi.org/10.1080/01621459.1969.10501049). These sources justify methods and diagnostics; they do not validate this project's labels, numerical results, or `0.70` threshold.\n\n"
        "The figures below are generated from this project's data and code. Generic web or presentation illustrations are explanatory aids only and are not used as academic evidence.\n\n"
        "## What Each Diagnostic Means\n\n"
        "- **Confusion matrix:** anchor-level evidence, matching the actual pipeline decision: one accepted successor or abstention.\n"
        "- **Exact-successor accounting:** stricter project metric; a wrong successor is both a false accepted link and a missed true successor.\n"
        "- **ROC curve:** pair-level score-ranking diagnostic over exposed candidate pairs. Useful, but less important than precision-recall because positives are rare.\n"
        "- **Precision-recall curve:** pair-level score-ranking diagnostic. This is the better curve for this project because validation has only 62 positive pairs out of 1,763 candidate pairs.\n"
        "- **Threshold trade-off:** anchor-level sweep of the actual `M_B` top-1 decision. It shows how strict acceptance changes precision, recall, false-positive rate, and link volume.\n\n"
        "## Held-Out Internal Event-Detection Confusion Matrix\n\n"
        "Rows are actual anchor status; columns are predicted link/abstention. Here, a wrong candidate on a positive anchor still counts as detecting that the anchor has a successor.\n\n"
        f"{markdown_table(event, ['method', 'threshold', 'tp', 'fp', 'fn', 'tn'])}\n\n"
        "## Held-Out Internal Exact-Successor Accounting\n\n"
        "This is the stricter accounting behind project precision and recall. Cells do not necessarily sum to the number of anchors because a wrong successor contributes one FP and one FN.\n\n"
        f"{markdown_table(exact, ['method', 'threshold', 'tp', 'fp', 'fn', 'tn'])}\n\n"
        "## Held-Out Internal Pair-Level ROC and Precision-Recall Metrics\n\n"
        f"{markdown_table(pair, ['method', 'pair_roc_auc', 'pair_average_precision', 'positive_pairs', 'negative_pairs'])}\n\n"
        "## M_B Anchor-Level Threshold Trade-Off\n\n"
        f"{markdown_table(operating, ['threshold', 'accepted_links', 'true_positive', 'precision', 'recall', 'false_positive_rate', 'coverage'])}\n\n"
        "At `0.60`, the internal validation split has 8 correct successors among 9 accepted links (precision `0.8889`) and recall `0.3636`. At the frozen `0.70`, it has 4 correct successors among 5 accepted links (precision `0.8000`) and recall `0.1818`. Thus `0.60` empirically dominates `0.70` on this particular validation sample. The development evidence points the other way: `0.70` has precision `0.8750` and FPR `0.0345`, compared with precision `0.8000` and FPR `0.0575` at `0.60`. The production-link diagnostic at `0.70` also confirmed only 14 of 20 links conservatively. Therefore `0.70` is retained as the previously frozen conservative baseline, not described as optimal; `0.60` remains a required sensitivity arm. Promoting the unreviewed lower threshold after observing four favourable incremental validation cases would be post-hoc tuning.\n\n"
        "## Interpretation\n\n"
        "`M_B_text_ranking @ 0.70` is the frozen conservative operating baseline, not a claim of threshold optimality. Its role is to provide one reproducible primary event definition while `0.60`, `0.80`, `M_C`, and expiry-aware variants quantify sensitivity. The bootstrap labels support development comparisons but cannot establish external accuracy.\n\n"
        "## Plot Files\n\n"
        "- `reports/figures/benchmark_validation_confusion_matrices.png`\n"
        "- `reports/figures/benchmark_validation_pair_roc.png`\n"
        "- `reports/figures/benchmark_validation_pair_precision_recall.png`\n"
        "- `reports/figures/benchmark_validation_m_b_threshold_tradeoff.png`\n",
        encoding="utf-8",
    )
    return path


def build(split: str) -> dict[str, Any]:
    if split not in {"dev", "validation"}:
        raise ValueError("split must be dev or validation")
    summary = load_json(PROCESSED / f"linkage_evaluation_{split}.json")
    modeling = pd.read_parquet(BENCHMARK / "modeling" / f"modeling_{split}.parquet")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    confusion = anchor_confusion(summary)
    pair_metrics, curves = pair_curve_tables(modeling)
    truth, _ = load_truth(
        BENCHMARK,
        split,
        "primary",
        project_root=PROJECT_ROOT,
        allow_sealed=False,
        seal_reason="",
    )
    sweep = threshold_sweep(modeling, truth)
    assert_frozen_m_b_point(sweep, summary)
    confusion_path = OUTPUT_DIR / f"{split}_anchor_confusion.csv"
    pair_metrics_path = OUTPUT_DIR / f"{split}_pair_curve_metrics.csv"
    sweep_path = OUTPUT_DIR / f"{split}_m_b_threshold_sweep.csv"
    confusion.to_csv(confusion_path, index=False)
    pair_metrics.to_csv(pair_metrics_path, index=False)
    sweep.to_csv(sweep_path, index=False)

    if split == "validation":
        plot_confusion(confusion, FIGURE_DIR / "benchmark_validation_confusion_matrices.png")
        plot_curves(
            curves,
            FIGURE_DIR / "benchmark_validation_pair_roc.png",
            FIGURE_DIR / "benchmark_validation_pair_precision_recall.png",
        )
        plot_threshold_tradeoff(
            sweep,
            FIGURE_DIR / "benchmark_validation_m_b_threshold_tradeoff.png",
        )

    return {
        "split": split,
        "confusion_csv": str(confusion_path),
        "pair_curve_metrics_csv": str(pair_metrics_path),
        "m_b_threshold_sweep_csv": str(sweep_path),
        "pair_curve_metrics": pair_metrics.to_dict(orient="records"),
    }


def main() -> int:
    dev_output = build("dev")
    validation_output = build("validation")
    validation_confusion = pd.read_csv(OUTPUT_DIR / "validation_anchor_confusion.csv")
    validation_pair_metrics = pd.read_csv(OUTPUT_DIR / "validation_pair_curve_metrics.csv")
    validation_threshold_sweep = pd.read_csv(
        OUTPUT_DIR / "validation_m_b_threshold_sweep.csv"
    )
    markdown_path = write_quality_markdown(
        validation_confusion,
        validation_pair_metrics,
        validation_threshold_sweep,
    )
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": "national",
        "event_set": "primary",
        "notes": {
            "confusion_matrix": (
                "Anchor-level event-detection matrix; exact-successor accounting is "
                "also emitted because wrong successor selections are both FP and FN "
                "under project precision/recall."
            ),
            "roc_pr": (
                "Pair-level score diagnostics over exposed candidate pairs. They do "
                "not replace top-1 anchor-level precision, recall, and FPR."
            ),
            "threshold_tradeoff": (
                "Anchor-level top-1 M_B sweep. The 0.70 row is asserted against the "
                "frozen validation summary; points are empirical and unsmoothed."
            ),
        },
        "outputs": [dev_output, validation_output],
        "figures": [
            str(FIGURE_DIR / "benchmark_validation_confusion_matrices.png"),
            str(FIGURE_DIR / "benchmark_validation_pair_roc.png"),
            str(FIGURE_DIR / "benchmark_validation_pair_precision_recall.png"),
            str(FIGURE_DIR / "benchmark_validation_m_b_threshold_tradeoff.png"),
        ],
        "markdown_report": str(markdown_path),
        "validation_passed": True,
    }
    summary_path = OUTPUT_DIR / "quality_evidence_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
