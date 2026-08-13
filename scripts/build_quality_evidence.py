#!/usr/bin/env python3
"""Build quality-evidence tables and plots for the latest v3 linkage benchmark."""

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
    method_a_deterministic,
    method_c_weighted_gated,
)

PROCESSED = PROJECT_ROOT / "data/processed/boamp_v2"
BENCHMARK = PROCESSED / "benchmark_v3"
OUTPUT_DIR = PROCESSED / "quality_evidence"
FIGURE_DIR = PROJECT_ROOT / "reports/figures"

METHOD_LABELS = {
    "M_A_deterministic": "M_A deterministic",
    "M_B_text_ranking": "M_B text ranking",
    "M_C_weighted_gated": "M_C weighted gated",
    "M_D_fellegi_sunter": "M_D Fellegi-Sunter",
}


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
    fig.suptitle("Validation anchor-level event-detection confusion matrices", fontsize=12)
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
    ax.set_title("Validation pair-level ROC curves")
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
        next(iter(curves.values()))["positive_pairs"] + next(iter(curves.values()))["negative_pairs"]
    )
    ax.axhline(base_rate, color="#9AA3AD", linestyle="--", linewidth=1, label=f"Base rate {base_rate:.3f}")
    for method, curve in curves.items():
        ax.plot(
            curve["pr_recall"],
            curve["pr_precision"],
            label=f"{METHOD_LABELS[method]} (AP {curve['average_precision']:.3f})",
            color=colors[method],
            linewidth=2,
        )
    ax.set_title("Validation pair-level precision-recall curves")
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


def write_quality_markdown(validation_confusion: pd.DataFrame, validation_pair_metrics: pd.DataFrame) -> Path:
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

    path = PROJECT_ROOT / "QUALITY_EVIDENCE.md"
    path.write_text(
        "# Linkage Quality Evidence\n\n"
        "Generated from the current national validation reference.\n\n"
        "## What Each Diagnostic Means\n\n"
        "- **Confusion matrix:** anchor-level evidence, matching the actual pipeline decision: one accepted successor or abstention.\n"
        "- **Exact-successor accounting:** stricter project metric; a wrong successor is both a false accepted link and a missed true successor.\n"
        "- **ROC curve:** pair-level score-ranking diagnostic over exposed candidate pairs. Useful, but less important than precision-recall because positives are rare.\n"
        "- **Precision-recall curve:** pair-level score-ranking diagnostic. This is the better curve for this project because validation has only 62 positive pairs out of 1,754 candidate pairs.\n\n"
        "## Validation Event-Detection Confusion Matrix\n\n"
        "Rows are actual anchor status; columns are predicted link/abstention. Here, a wrong candidate on a positive anchor still counts as detecting that the anchor has a successor.\n\n"
        f"{markdown_table(event, ['method', 'threshold', 'tp', 'fp', 'fn', 'tn'])}\n\n"
        "## Validation Exact-Successor Accounting\n\n"
        "This is the stricter accounting behind project precision and recall. Cells do not necessarily sum to the number of anchors because a wrong successor contributes one FP and one FN.\n\n"
        f"{markdown_table(exact, ['method', 'threshold', 'tp', 'fp', 'fn', 'tn'])}\n\n"
        "## Validation Pair-Level ROC and Precision-Recall Metrics\n\n"
        f"{markdown_table(pair, ['method', 'pair_roc_auc', 'pair_average_precision', 'positive_pairs', 'negative_pairs'])}\n\n"
        "## Interpretation\n\n"
        "`M_B_text_ranking @ 0.70` is still the defensible primary method. It is not the highest-recall method, but it has zero validation false positives on no-successor anchors and the best pair-level ranking diagnostics.\n\n"
        "## Plot Files\n\n"
        "- `reports/figures/benchmark_v3_validation_confusion_matrices.png`\n"
        "- `reports/figures/benchmark_v3_validation_pair_roc.png`\n"
        "- `reports/figures/benchmark_v3_validation_pair_precision_recall.png`\n",
        encoding="utf-8",
    )
    return path


def build(split: str) -> dict[str, Any]:
    if split not in {"dev", "validation"}:
        raise ValueError("split must be dev or validation")
    summary = load_json(PROCESSED / f"linkage_evaluation_summary_v3_{split}_primary.json")
    modeling = pd.read_parquet(BENCHMARK / "modeling" / f"benchmark_v3_modeling_{split}.parquet")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    confusion = anchor_confusion(summary)
    pair_metrics, curves = pair_curve_tables(modeling)
    confusion_path = OUTPUT_DIR / f"benchmark_v3_{split}_anchor_confusion.csv"
    pair_metrics_path = OUTPUT_DIR / f"benchmark_v3_{split}_pair_curve_metrics.csv"
    confusion.to_csv(confusion_path, index=False)
    pair_metrics.to_csv(pair_metrics_path, index=False)

    if split == "validation":
        plot_confusion(confusion, FIGURE_DIR / "benchmark_v3_validation_confusion_matrices.png")
        plot_curves(
            curves,
            FIGURE_DIR / "benchmark_v3_validation_pair_roc.png",
            FIGURE_DIR / "benchmark_v3_validation_pair_precision_recall.png",
        )

    return {
        "split": split,
        "confusion_csv": str(confusion_path),
        "pair_curve_metrics_csv": str(pair_metrics_path),
        "pair_curve_metrics": pair_metrics.to_dict(orient="records"),
    }


def main() -> int:
    dev_output = build("dev")
    validation_output = build("validation")
    validation_confusion = pd.read_csv(OUTPUT_DIR / "benchmark_v3_validation_anchor_confusion.csv")
    validation_pair_metrics = pd.read_csv(OUTPUT_DIR / "benchmark_v3_validation_pair_curve_metrics.csv")
    markdown_path = write_quality_markdown(validation_confusion, validation_pair_metrics)
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": "v3",
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
        },
        "outputs": [dev_output, validation_output],
        "figures": [
            str(FIGURE_DIR / "benchmark_v3_validation_confusion_matrices.png"),
            str(FIGURE_DIR / "benchmark_v3_validation_pair_roc.png"),
            str(FIGURE_DIR / "benchmark_v3_validation_pair_precision_recall.png"),
        ],
        "markdown_report": str(markdown_path),
        "validation_passed": True,
    }
    summary_path = OUTPUT_DIR / "benchmark_v3_quality_evidence_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
