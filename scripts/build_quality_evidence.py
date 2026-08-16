#!/usr/bin/env python3
"""Build quality-evidence tables and plots for the regional linkage reference."""

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
from boamp_pipeline.regional_benchmark_io import load_truth, wilson_interval  # noqa: E402

PROCESSED = PROJECT_ROOT / "data/processed/boamp"
BENCHMARK = PROCESSED / "regional_benchmark"
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
        metrics = method["unweighted"]
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
        precision_interval = wilson_interval(
            metrics["true_positive"], metrics["accepted_links"]
        ) or [None, None]
        recall_interval = wilson_interval(
            metrics["true_positive"], metrics["positive_anchors"]
        ) or [None, None]
        rows[-1].update({
            "precision_ci_low": precision_interval[0],
            "precision_ci_high": precision_interval[1],
            "recall_ci_low": recall_interval[0],
            "recall_ci_high": recall_interval[1],
        })
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
    official = method["unweighted"]
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
    """Plot the top-1 threshold trade-off without interpolation or smoothing.

    The steps are deliberately left as steps. Each riser is one anchor's decision
    flipping, and at eight accepted links a fitted curve would draw precision
    values between thresholds where nothing in the data changed. What the eye
    reads as noise is instead given its proper name: the 95% Wilson band shows
    that most of the jaggedness sits inside sampling uncertainty.
    """
    visible = sweep.loc[sweep["threshold"].between(0.10, 0.90)].copy()
    operating = visible.loc[visible["threshold_percent"].isin([50, 60, 80])]

    # Categorical slots 1-3 of the validated palette; the previous blue and teal
    # sat below the chroma floor and read gray beside each other.
    colors = {
        "precision": "#2a78d6",
        "recall": "#eb6834",
        "false_positive_rate": "#1baf7a",
        "accepted": "#898781",
        "correct": "#2a78d6",
    }
    grid = "#e1e0d9"
    muted = "#898781"
    ink = "#0b0b0b"
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=True)
    for ax in axes:
        ax.grid(color=grid, linewidth=0.8, alpha=1.0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)

    ax = axes[0]
    for metric, label, low, high in [
        ("precision", "Precision", "precision_ci_low", "precision_ci_high"),
        ("recall", "Recall", "recall_ci_low", "recall_ci_high"),
        ("false_positive_rate", "FPR", None, None),
    ]:
        if low is not None:
            ax.fill_between(
                visible["threshold"], visible[low], visible[high],
                step="post", color=colors[metric], alpha=0.13, linewidth=0,
            )
        ax.step(
            visible["threshold"], visible[metric], where="post",
            linewidth=2, color=colors[metric], label=label,
        )
    ax.axvline(0.70, color=ink, linestyle="--", linewidth=1.2)
    ax.text(0.695, 0.055, "Frozen 0.70", rotation=90, va="bottom", ha="right",
            fontsize=8, color=muted)
    # Direct labels at the right edge: identity without reading a legend box,
    # and the relief the aqua slot's contrast warning requires.
    right = visible.iloc[-1]
    for metric, label in [("precision", "Precision"), ("recall", "Recall"),
                          ("false_positive_rate", "FPR")]:
        ax.annotate(
            label, (0.90, right[metric]), xytext=(4, 0), textcoords="offset points",
            fontsize=8.5, color=colors[metric], va="center",
        )
    ax.set_title("Metrics by acceptance threshold", fontsize=11, color=ink)
    ax.set_xlabel("Minimum text similarity", fontsize=9, color=muted)
    ax.set_ylabel("Rate", fontsize=9, color=muted)
    ax.set_xlim(0.10, 0.98)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=muted)

    ax = axes[1]
    valid = visible.dropna(subset=["precision"])
    ax.plot(valid["recall"], valid["precision"], color=muted, linewidth=1,
            alpha=0.55, zorder=1)
    for row in operating.itertuples(index=False):
        if pd.isna(row.precision):
            continue
        ax.scatter(row.recall, row.precision, color=colors["recall"], s=30,
                   zorder=3, edgecolor="#fcfcfb", linewidth=2)
        ax.annotate(
            f"{row.threshold:.2f}", (row.recall, row.precision),
            xytext=(5, 6), textcoords="offset points", fontsize=8.5, color=muted,
        )
    frozen = sweep.loc[sweep["threshold_percent"].eq(70)].iloc[0]
    ax.errorbar(
        frozen["recall"], frozen["precision"],
        yerr=[[frozen["precision"] - frozen["precision_ci_low"]],
              [frozen["precision_ci_high"] - frozen["precision"]]],
        xerr=[[frozen["recall"] - frozen["recall_ci_low"]],
              [frozen["recall_ci_high"] - frozen["recall"]]],
        fmt="o", markersize=9, markerfacecolor="#fcfcfb", markeredgecolor=ink,
        markeredgewidth=1.5, ecolor=ink, elinewidth=1.1, capsize=3, alpha=0.85,
        zorder=4,
    )
    ax.annotate("Frozen 0.70", (frozen["recall"], frozen["precision"]),
                xytext=(12, 4), textcoords="offset points", fontsize=8.5, color=ink)
    ax.set_title("Precision-recall operating points", fontsize=11, color=ink)
    ax.set_xlabel("Recall", fontsize=9, color=muted)
    ax.set_ylabel("Precision", fontsize=9, color=muted)
    ax.set_xlim(0.15, 0.80)
    ax.set_ylim(0.15, 1.05)

    ax = axes[2]
    ax.step(visible["threshold"], visible["accepted_links"], where="post",
            color=colors["accepted"], linewidth=2, label="Accepted links")
    ax.step(visible["threshold"], visible["true_positive"], where="post",
            color=colors["correct"], linewidth=2, label="Correct successors")
    ax.axvline(0.70, color=ink, linestyle="--", linewidth=1.2)
    ax.set_title("Decision volume by threshold", fontsize=11, color=ink)
    ax.set_xlabel("Minimum text similarity", fontsize=9, color=muted)
    ax.set_ylabel("Locked-split anchors", fontsize=9, color=muted)
    ax.set_xlim(0.10, 0.90)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="center right", labelcolor=muted)

    anchors = int(sweep["positive_anchors"].iloc[0] + sweep["negative_anchors"].iloc[0])
    positives = int(sweep["positive_anchors"].iloc[0])
    fig.suptitle(
        "M_B top-1 threshold trade-off on the locked regional-reference split",
        fontsize=13,
    )
    fig.text(
        0.5,
        -0.035,
        f"{anchors} anchors ({positives} with a reviewed successor). Empirical steps only; "
        "no smoothing - each riser is one anchor changing decision. Shaded bands and "
        "whiskers are 95% Wilson intervals. An LLM-generated, subset-spot-checked "
        "reference sample, not independent validation.",
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


def sweep_row(sweep: pd.DataFrame, threshold_percent: float) -> pd.Series:
    match = sweep.loc[sweep["threshold_percent"].eq(threshold_percent)]
    if len(match) != 1:
        raise RuntimeError(f"threshold sweep has no unique row at {threshold_percent}")
    return match.iloc[0]


def describe_point(row: pd.Series) -> str:
    """One threshold row in words, straight from the row."""
    return (
        f"{int(row['true_positive'])} correct of {int(row['accepted_links'])} accepted "
        f"(precision `{row['precision']:.4f}`), recall `{row['recall']:.4f}`, "
        f"FPR `{row['false_positive_rate']:.4f}`"
    )


def threshold_narrative(
    validation_sweep: pd.DataFrame, dev_sweep: pd.DataFrame
) -> str:
    """Write the threshold paragraph from the sweeps rather than from memory.

    Every number here is read out of the current sweep tables, so this paragraph
    cannot drift away from the figures beside it the way a hand-typed one does.
    """
    frozen_validation = sweep_row(validation_sweep, 70.0)
    lower_validation = sweep_row(validation_sweep, 60.0)
    frozen_dev = sweep_row(dev_sweep, 70.0)
    lower_dev = sweep_row(dev_sweep, 60.0)
    return (
        f"On the locked split the frozen `0.70` gives {describe_point(frozen_validation)}; "
        f"`0.60` gives {describe_point(lower_validation)}. On the pilot split `0.70` gives "
        f"{describe_point(frozen_dev)} against {describe_point(lower_dev)} at `0.60`. "
        "`0.70` was fixed before this reference was read and has not been moved since, "
        "which is the only reason the locked split can be reported as held out. Selecting "
        "a threshold now, from these rows, would convert the locked split into a tuning "
        "set and forfeit that. `0.60` therefore stays a sensitivity arm whatever these "
        "rows say."
    )


def write_quality_markdown(
    validation_confusion: pd.DataFrame,
    validation_pair_metrics: pd.DataFrame,
    validation_threshold_sweep: pd.DataFrame,
    dev_threshold_sweep: pd.DataFrame,
    manifest: dict[str, Any],
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

    positive_pairs = int(pair["positive_pairs"].iloc[0])
    negative_pairs = int(pair["negative_pairs"].iloc[0])
    splits = manifest["splits"]["validation"]
    ceiling = manifest["candidate_reachability"]

    path = PROJECT_ROOT / "QUALITY_EVIDENCE.md"
    path.write_text(
        "# Linkage Quality Evidence\n\n"
        f"Generated from the locked split of the Grand Ouest regional reference: "
        f"`{splits['usable_anchors']}` usable anchors, of which `{splits['positive_anchors']}` "
        "have a labelled observable successor. The labels were generated by a single "
        "LLM research pass over real BOAMP notices, their official URLs, and wider "
        "public sources on 2026-08-11, before these linkage methods existed, then "
        "spot-checked on a subset by the project owner rather than verified "
        "anchor-by-anchor. They are independent of every algorithm scored below, but "
        "they are neither an independent human specialist panel nor legal renewal "
        "truth.\n\n"
        f"Recall is capped before any method runs: candidate generation reaches "
        f"`{ceiling['positive_anchors_with_reviewed_successor_in_pool']}` of "
        f"`{ceiling['positive_anchors']}` reviewed successors, a ceiling of "
        f"`{ceiling['candidate_generation_recall_ceiling']:.4f}`. No method below can "
        "exceed it, and the gap is a blocking-stage limitation rather than a scoring one.\n\n"
        "## Academic Basis And Evidence Boundary\n\n"
        "The metric choice is supported by [Davis and Goadrich (2006)](https://doi.org/10.1145/1143844.1143874), who analyse the relationship between ROC and precision-recall curves for skewed binary decisions, and [Saito and Rehmsmeier (2015)](https://doi.org/10.1371/journal.pone.0118432), who show why precision-recall analysis is more informative for imbalanced data. The classical probabilistic linkage comparator follows [Fellegi and Sunter (1969)](https://doi.org/10.1080/01621459.1969.10501049). These sources justify methods and diagnostics; they do not validate this project's labels, numerical results, or `0.70` threshold.\n\n"
        "The figures below are generated from this project's data and code. Generic web or presentation illustrations are explanatory aids only and are not used as academic evidence.\n\n"
        "## What Each Diagnostic Means\n\n"
        "- **Confusion matrix:** anchor-level evidence, matching the actual pipeline decision: one accepted successor or abstention.\n"
        "- **Exact-successor accounting:** stricter project metric; a wrong successor is both a false accepted link and a missed true successor.\n"
        "- **ROC curve:** pair-level score-ranking diagnostic over exposed candidate pairs. Useful, but less important than precision-recall because positives are rare.\n"
        f"- **Precision-recall curve:** pair-level score-ranking diagnostic. This is the better curve for this project because the locked split has only {positive_pairs} positive pairs among {positive_pairs + negative_pairs} candidate pairs.\n"
        "- **Threshold trade-off:** anchor-level sweep of the actual `M_B` top-1 decision. It shows how strict acceptance changes precision, recall, false-positive rate, and link volume.\n\n"
        "## Held-Out Event-Detection Confusion Matrix\n\n"
        "Rows are actual anchor status; columns are predicted link/abstention. Here, a wrong candidate on a positive anchor still counts as detecting that the anchor has a successor.\n\n"
        f"{markdown_table(event, ['method', 'threshold', 'tp', 'fp', 'fn', 'tn'])}\n\n"
        "## Held-Out Exact-Successor Accounting\n\n"
        "This is the stricter accounting behind project precision and recall. Cells do not necessarily sum to the number of anchors because a wrong successor contributes one FP and one FN.\n\n"
        f"{markdown_table(exact, ['method', 'threshold', 'tp', 'fp', 'fn', 'tn'])}\n\n"
        "## Held-Out Pair-Level ROC and Precision-Recall Metrics\n\n"
        f"{markdown_table(pair, ['method', 'pair_roc_auc', 'pair_average_precision', 'positive_pairs', 'negative_pairs'])}\n\n"
        "A negative pair here is any exposed candidate that is not the reviewed successor. The reviewer inspected roughly 25 candidates per anchor, so most negatives were never individually rejected; these curves rank scores and do not measure accuracy.\n\n"
        "## M_B Anchor-Level Threshold Trade-Off\n\n"
        f"{markdown_table(operating, ['threshold', 'accepted_links', 'true_positive', 'precision', 'recall', 'false_positive_rate', 'coverage'])}\n\n"
        f"{threshold_narrative(validation_threshold_sweep, dev_threshold_sweep)}\n\n"
        "## Interpretation\n\n"
        "`M_B_text_ranking @ 0.70` is the frozen conservative operating baseline, not a claim of threshold optimality. Its role is to provide one reproducible primary event definition while `0.60`, `0.80`, and `M_C @ 0.70` quantify event-definition sensitivity and a fixed borderline band around the threshold tests how much rests on near-boundary decisions. On a reference of this size every figure above should be read with its interval: two methods whose intervals overlap have not been separated by this evidence.\n\n"
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
    truth = load_truth(BENCHMARK, split, "primary")
    truth = truth.loc[truth["truth_usable"]].copy()
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
    dev_threshold_sweep = pd.read_csv(OUTPUT_DIR / "dev_m_b_threshold_sweep.csv")
    manifest = load_json(BENCHMARK / "regional_benchmark_manifest.json")
    markdown_path = write_quality_markdown(
        validation_confusion,
        validation_pair_metrics,
        validation_threshold_sweep,
        dev_threshold_sweep,
        manifest,
    )
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": "regional_grand_ouest",
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
