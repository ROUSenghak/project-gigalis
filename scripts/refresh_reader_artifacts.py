#!/usr/bin/env python3
"""Refresh reader-facing notebooks, figures, and reports from current outputs."""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data/processed/boamp_v2"
BENCHMARK = PROCESSED / "benchmark_v3"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
NOTEBOOK = PROJECT_ROOT / "notebooks/12_successor_linkage_and_evaluation.ipynb"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def method_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in summary["methods"]:
        all_frames = method["unweighted_all_frames"]
        weighted = method.get("weighted_national", {})
        rows.append(
            {
                "method": method["method"],
                "threshold": method["threshold"],
                "accepted_links": all_frames["accepted_links"],
                "precision": all_frames["precision_at_1"],
                "recall": all_frames["recall_at_1"],
                "fpr": all_frames["false_positive_rate_on_negatives"],
                "coverage": all_frames["coverage"],
                "weighted_precision": weighted.get("precision_at_1", {}).get("estimate"),
                "weighted_recall": weighted.get("recall_at_1", {}).get("estimate"),
                "weighted_fpr": weighted.get(
                    "false_positive_rate_on_verified_negatives", {}
                ).get("estimate"),
            }
        )
    return pd.DataFrame(rows)


def plot_method_metrics(frame: pd.DataFrame, title: str, path: Path) -> None:
    plot = frame.set_index("method")[["precision", "recall", "fpr"]]
    ax = plot.plot(kind="bar", figsize=(9.2, 4.6), width=0.72)
    ax.set_title(title)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.legend(["precision@1", "recall@1", "FPR on negatives"], frameon=False)
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_modeling_counts(modeling: dict[str, Any], path: Path) -> None:
    rows = []
    for split, payload in modeling["outputs"].items():
        rows.append(
            {
                "split": split,
                "anchors": payload["anchors"],
                "rows": payload["rows"],
                "primary positives": payload["primary_positive_pairs"],
                "verified negative anchors": payload["verified_negative_anchors"],
            }
        )
    frame = pd.DataFrame(rows).set_index("split")
    ax = frame[["anchors", "primary positives", "verified negative anchors"]].plot(
        kind="bar", figsize=(8.2, 4.2), width=0.72
    )
    ax.set_title("Benchmark v3 modeling tables")
    ax.set_ylabel("count")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def format_metric(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.3f}"


def latex_method_rows(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            " & ".join(
                [
                    latex_escape(row.method),
                    format_metric(row.threshold),
                    str(int(row.accepted_links)),
                    format_metric(row.precision),
                    format_metric(row.recall),
                    format_metric(row.fpr),
                    format_metric(row.coverage),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def write_methodology_report(
    dev: dict[str, Any],
    validation: dict[str, Any],
    modeling: dict[str, Any],
    manifest: dict[str, Any],
    generated_at: str,
) -> Path:
    dev_frame = method_frame(dev)
    validation_frame = method_frame(validation)
    labelled_anchors = manifest["anchor_totals"]["anchors"]
    labelled_pairs = manifest["anchor_totals"]["labelled_pairs"]
    m_b = validation_frame.loc[validation_frame["method"].eq("M_B_text_ranking")].iloc[0]
    m_c = validation_frame.loc[validation_frame["method"].eq("M_C_weighted_gated")].iloc[0]
    m_d = validation_frame.loc[validation_frame["method"].eq("M_D_fellegi_sunter")].iloc[0]

    tex = rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage{{geometry}}
\geometry{{left=2.5cm,right=2.5cm,top=2.6cm,bottom=2.6cm}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs,tabularx}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\hypersetup{{colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!60!black}}
\newcommand{{\code}}[1]{{\texttt{{#1}}}}
\newcommand{{\pathcode}}[1]{{\texttt{{\small #1}}}}
\title{{BOAMP Successor Linkage Methodology\\\large Current v3 Benchmark State}}
\author{{BOAMP Data Science Internship Project}}
\date{{{generated_at}}}
\begin{{document}}
\sloppy
\maketitle

\section*{{Technical Summary}}
This report is regenerated from the current v3 benchmark artifacts, not from
static notebook output. The latest labelled reference contains
{labelled_anchors} labelled anchors and {labelled_pairs:,}
anchor-candidate labels. The dev split has
{modeling["outputs"]["dev"]["anchors"]} anchors and
{modeling["outputs"]["dev"]["rows"]:,} pair rows; the validation split has
{modeling["outputs"]["validation"]["anchors"]} anchors and
{modeling["outputs"]["validation"]["rows"]:,} pair rows.

All four linkage algorithms have now been evaluated on the latest v3 reference:
\code{{M\_A\_deterministic}}, \code{{M\_B\_text\_ranking}},
\code{{M\_C\_weighted\_gated}}, and \code{{M\_D\_fellegi\_sunter}}.
On validation, \code{{M\_B\_text\_ranking @ 0.70}} remains the strongest
precision-first method: precision {m_b.precision:.3f}, recall {m_b.recall:.3f},
and false-positive rate {m_b.fpr:.3f}. \code{{M\_C}} has higher recall
({m_c.recall:.3f}) but also more false positives ({m_c.fpr:.3f}). \code{{M\_D}}
does not currently challenge the incumbent: precision {m_d.precision:.3f} and
recall {m_d.recall:.3f}.

\section{{Pipeline State}}
The defensible project pipeline is:
\[
\begin{{aligned}}
&\text{{BOAMP notices}}
\rightarrow \text{{standardised notices}}
\rightarrow \text{{procurement episodes}} \\
&\rightarrow \text{{candidate pairs}}
\rightarrow \text{{linkage scoring}}
\rightarrow \text{{one successor or abstention}}
\rightarrow \text{{survival dataset}} .
\end{{aligned}}
\]
The candidate-generation rule is unchanged. For an anchor contract with award
date \(A\) and a candidate with publication date \(C\), the candidate is eligible
only when
\[
A+90 \leq C \leq A+2920.
\]
The new v3 benchmark does not replace this candidate pool; it evaluates how well
the selection algorithms choose from exposed candidate pairs.

\section{{Benchmark v3 Reference}}
The v3 benchmark is national rather than Grand Ouest-only. It samples awarded
digital procurement episodes across French region groups and CPV divisions.
Only the probability frame supports national weighted estimates; enrichment
frames are used for stress cases and diagnostics.

The current annotation caveat remains important: labels are protocol-generated
development evidence with double-pass validation and adjudication. They are not
official legal renewal ground truth and should not be described as human
inter-annotator ground truth.

\section{{Algorithms}}
\paragraph{{\(M_A\): deterministic evidence.}}
Accepts only strong buyer identity, positive CPV continuity, and a minimum text
signal. It is conservative but misses many true successors.

\paragraph{{\(M_B\): text ranking.}}
Ranks same-buyer candidates by TF-IDF cosine similarity and accepts the top
candidate only if similarity is at least 0.70. It is transparent and currently
best aligned with a precision-first survival event definition.

\paragraph{{\(M_C\): weighted gated score.}}
Combines buyer, text, CPV, timing, and evidence components, then applies
independent gates. It recovers more positives but admits more false links.

\paragraph{{\(M_D\): Fellegi--Sunter probabilistic linkage.}}
Discretises buyer, text, CPV, and time comparisons, estimates match and
non-match level distributions by expectation maximisation, and ranks by
posterior match probability. The latest refresh applies the fitted model to the
v3 exposure table so this method is no longer skipped.

\section{{Dev Results}}
\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{lrrrrrr}}
\toprule
Method & Threshold & Accepted & Precision & Recall & FPR & Coverage \\
\midrule
{latex_method_rows(dev_frame)}
\bottomrule
\end{{tabularx}}
\caption{{Unweighted dev metrics on all labelled v3 frames.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{figures/benchmark_v3_dev_method_metrics.png}}
\caption{{Dev split method comparison generated from the current v3 evaluation JSON.}}
\end{{figure}}

\section{{Validation Results}}
\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{lrrrrrr}}
\toprule
Method & Threshold & Accepted & Precision & Recall & FPR & Coverage \\
\midrule
{latex_method_rows(validation_frame)}
\bottomrule
\end{{tabularx}}
\caption{{Unweighted validation metrics on all labelled v3 frames.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{figures/benchmark_v3_validation_method_metrics.png}}
\caption{{Validation split method comparison generated from the current v3 evaluation JSON.}}
\end{{figure}}

\section{{Modeling Tables}}
The modeling-ready tables include strict, primary, broad, and non-match target
columns, plus observable features. They now include the Fellegi--Sunter columns
\code{{fs\_match\_weight}} and \code{{fs\_match\_probability}}, so modeling and
evaluation use the same feature state.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\textwidth]{{figures/benchmark_v3_modeling_counts.png}}
\caption{{Current v3 modeling table sizes and label support.}}
\end{{figure}}

\section{{Decision}}
The current defensible decision is to keep \code{{M\_B\_text\_ranking @ 0.70}}
as the primary observable-successor method. It is not the highest-recall method,
but the survival analysis is more sensitive to false positives than to
abstention: a false positive fabricates both an event and an event time, while an
abstention becomes right-censoring.

\section{{Current Source Files}}
\begin{{itemize}}
\item \pathcode{{data/processed/boamp\_v2/linkage\_evaluation\_summary\_v3\_dev\_primary.json}}
\item \pathcode{{data/processed/boamp\_v2/linkage\_evaluation\_summary\_v3\_validation\_primary.json}}
\item \pathcode{{data/processed/boamp\_v2/benchmark\_v3/modeling/benchmark\_v3\_modeling\_summary.json}}
\item \pathcode{{data/processed/boamp\_v2/benchmark\_v3/benchmark\_v3\_manifest.json}}
\end{{itemize}}

\end{{document}}
"""
    path = REPORTS / "boamp_methodology_chapter.tex"
    path.write_text(textwrap.dedent(tex).lstrip(), encoding="utf-8")
    return path


def write_status_files(
    dev: dict[str, Any],
    validation: dict[str, Any],
    modeling: dict[str, Any],
    manifest: dict[str, Any],
    generated_at: str,
) -> None:
    validation_frame = method_frame(validation)
    labelled_anchors = manifest["anchor_totals"]["anchors"]
    labelled_pairs = manifest["anchor_totals"]["labelled_pairs"]
    m_b = validation_frame.loc[validation_frame["method"].eq("M_B_text_ranking")].iloc[0]
    final_pipeline = f"""# Final Defensible Pipeline

Generated: `{generated_at}`

## Current Decision

The current primary method remains `M_B_text_ranking @ 0.70`.

On the latest v3 validation reference it gives:

- precision@1: `{m_b.precision:.3f}`;
- recall@1: `{m_b.recall:.3f}`;
- false-positive rate on negative anchors: `{m_b.fpr:.3f}`;
- accepted validation links: `{int(m_b.accepted_links)}`.

`M_C_weighted_gated` has higher recall but also higher false-positive risk.
`M_D_fellegi_sunter` is now evaluated on v3, but it does not outperform `M_B`.

## End-to-End Workflow

```text
Official BOAMP API, 2015-2025
  -> schema-aware standardisation
  -> procurement episode reconstruction
  -> Grand Ouest digital study cohort
  -> broad same-buyer candidate generation
  -> four linkage algorithms compared on v3
  -> M_B primary successor selection
  -> survival dataset and expiry-aware sensitivity audit
```

The event remains an **observable successor procurement**, not a confirmed legal
renewal.

## Latest Benchmark State

- labelled anchors: `{labelled_anchors}`;
- labelled pairs: `{labelled_pairs:,}`;
- dev: `{modeling["outputs"]["dev"]["anchors"]}` anchors and `{modeling["outputs"]["dev"]["rows"]:,}` pair rows;
- validation: `{modeling["outputs"]["validation"]["anchors"]}` anchors and `{modeling["outputs"]["validation"]["rows"]:,}` pair rows;
- sealed test: not used for method selection.

## Current Source of Truth

- `data/processed/boamp_v2/linkage_evaluation_summary_v3_dev_primary.json`
- `data/processed/boamp_v2/linkage_evaluation_summary_v3_validation_primary.json`
- `data/processed/boamp_v2/benchmark_v3/modeling/benchmark_v3_modeling_summary.json`
- `reports/boamp_methodology_chapter.pdf`
- `notebooks/12_successor_linkage_and_evaluation.ipynb`

## Refresh Command

```bash
PYTHONPATH=. python3 scripts/run_final_pipeline.py --with-benchmark-v3-evaluation --force
```
"""
    (PROJECT_ROOT / "FINAL_PIPELINE.md").write_text(final_pipeline, encoding="utf-8")

    national = f"""# National Benchmark Reference

Generated: `{generated_at}`

## Purpose

The v3 benchmark is the current France-wide reference for calibrating,
validating, and evaluating observable-successor linkage algorithms.

It does not prove legal renewal. It tests whether a method can identify a
plausible later successor procurement from the same buyer.

## Current Materialised State

- labelled anchors: `{labelled_anchors}`;
- labelled pairs: `{labelled_pairs:,}`;
- dev split: `{modeling["outputs"]["dev"]["anchors"]}` anchors, `{modeling["outputs"]["dev"]["rows"]:,}` pair rows, `{modeling["outputs"]["dev"]["primary_positive_pairs"]}` primary-positive pairs;
- validation split: `{modeling["outputs"]["validation"]["anchors"]}` anchors, `{modeling["outputs"]["validation"]["rows"]:,}` pair rows, `{modeling["outputs"]["validation"]["primary_positive_pairs"]}` primary-positive pairs;
- sealed test: closed for method selection.

## Current Method Comparison

The latest v3 evaluation includes all four methods. `M_D_fellegi_sunter` is no
longer skipped because `fs_match_probability` is now computed from the fitted
Fellegi-Sunter model when v3 exposure is evaluated.

| Method | Threshold | Validation precision | Validation recall | Validation FPR | Accepted |
|---|---:|---:|---:|---:|---:|
"""
    for row in validation_frame.itertuples(index=False):
        national += (
            f"| `{row.method}` | {row.threshold:.1f} | {row.precision:.3f} | "
            f"{row.recall:.3f} | {row.fpr:.3f} | {int(row.accepted_links)} |\n"
        )
    national += """
## Decision Rule

The incumbent `M_B_text_ranking @ 0.70` remains the primary method because it
has the strongest precision-first validation profile. A replacement should only
be promoted if it preserves or improves precision and false-positive control
without an unacceptable recall loss.

## Caveat

The current labels are protocol-generated development evidence with double-pass
validation and adjudication. They should not be described as official legal
renewal truth or independent human inter-annotator ground truth.
"""
    (PROJECT_ROOT / "NATIONAL_BENCHMARK_REFERENCE.md").write_text(national, encoding="utf-8")


def write_notebook(generated_at: str) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    cells = [
        nbf.v4.new_markdown_cell(
            "# 12. Successor linkage and latest v3 evaluation\n\n"
            f"Generated: `{generated_at}`\n\n"
            "This notebook is regenerated from the current script outputs. It is the "
            "reader-facing linkage/evaluation notebook for the latest v3 state."
        ),
        nbf.v4.new_markdown_cell(
            "## tl;dr\n\n"
            "`M_B_text_ranking @ 0.70` remains the primary method. The latest "
            "validation comparison includes all four algorithms, including "
            "`M_D_fellegi_sunter`, which is now scored from the fitted model."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / 'scripts').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "PROCESSED = PROJECT_ROOT / 'data/processed/boamp_v2'\n"
            "BENCHMARK = PROCESSED / 'benchmark_v3'\n\n"
            "def load_json(path):\n"
            "    with open(path, 'r', encoding='utf-8') as f:\n"
            "        return json.load(f)\n\n"
            "dev = load_json(PROCESSED / 'linkage_evaluation_summary_v3_dev_primary.json')\n"
            "validation = load_json(PROCESSED / 'linkage_evaluation_summary_v3_validation_primary.json')\n"
            "modeling = load_json(BENCHMARK / 'modeling/benchmark_v3_modeling_summary.json')\n"
            "manifest = load_json(BENCHMARK / 'benchmark_v3_manifest.json')\n"
        ),
        nbf.v4.new_code_cell(
            "def method_frame(summary):\n"
            "    rows = []\n"
            "    for method in summary['methods']:\n"
            "        metrics = method['unweighted_all_frames']\n"
            "        weighted = method.get('weighted_national', {})\n"
            "        rows.append({\n"
            "            'method': method['method'],\n"
            "            'threshold': method['threshold'],\n"
            "            'accepted_links': metrics['accepted_links'],\n"
            "            'precision': metrics['precision_at_1'],\n"
            "            'recall': metrics['recall_at_1'],\n"
            "            'fpr': metrics['false_positive_rate_on_negatives'],\n"
            "            'coverage': metrics['coverage'],\n"
            "            'weighted_precision': weighted.get('precision_at_1', {}).get('estimate'),\n"
            "            'weighted_recall': weighted.get('recall_at_1', {}).get('estimate'),\n"
            "            'weighted_fpr': weighted.get('false_positive_rate_on_verified_negatives', {}).get('estimate'),\n"
            "        })\n"
            "    return pd.DataFrame(rows)\n\n"
            "dev_methods = method_frame(dev)\n"
            "validation_methods = method_frame(validation)\n"
            "validation_methods\n"
        ),
        nbf.v4.new_markdown_cell("## Benchmark State"),
        nbf.v4.new_code_cell(
            "pd.DataFrame([\n"
            "    {'item': 'labelled anchors', 'value': manifest['anchor_totals']['anchors']},\n"
            "    {'item': 'labelled pairs', 'value': manifest['anchor_totals']['labelled_pairs']},\n"
            "    {'item': 'dev anchors', 'value': modeling['outputs']['dev']['anchors']},\n"
            "    {'item': 'dev rows', 'value': modeling['outputs']['dev']['rows']},\n"
            "    {'item': 'validation anchors', 'value': modeling['outputs']['validation']['anchors']},\n"
            "    {'item': 'validation rows', 'value': modeling['outputs']['validation']['rows']},\n"
            "])"
        ),
        nbf.v4.new_markdown_cell("## Validation Method Comparison"),
        nbf.v4.new_code_cell(
            "display(validation_methods[['method', 'threshold', 'accepted_links', 'precision', 'recall', 'fpr', 'coverage']])\n\n"
            "ax = validation_methods.set_index('method')[['precision', 'recall', 'fpr']].plot(\n"
            "    kind='bar', figsize=(9, 4.5), width=0.72\n"
            ")\n"
            "ax.set_title('Latest v3 validation metrics')\n"
            "ax.set_ylabel('rate')\n"
            "ax.set_ylim(0, 1)\n"
            "ax.set_xlabel('')\n"
            "ax.legend(['precision@1', 'recall@1', 'FPR on negatives'], frameon=False)\n"
            "ax.tick_params(axis='x', rotation=28)\n"
            "ax.grid(axis='y', alpha=0.25)\n"
            "plt.tight_layout()"
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretation\n\n"
            "`M_C_weighted_gated` recovers more true successors, but its false-positive "
            "rate is materially higher. For survival analysis, a false link is more "
            "damaging than an abstention because it fabricates both an event and an "
            "event time. This is why `M_B_text_ranking @ 0.70` remains the primary "
            "defensible method."
        ),
        nbf.v4.new_markdown_cell("## Modeling-Ready Tables"),
        nbf.v4.new_code_cell(
            "pd.DataFrame(modeling['outputs']).T[[\n"
            "    'rows', 'anchors', 'probability_frame_anchors', 'primary_positive_pairs',\n"
            "    'strict_positive_pairs', 'broad_positive_pairs', 'verified_negative_anchors'\n"
            "]]"
        ),
        nbf.v4.new_code_cell(
            "feature_columns = pd.Series(modeling['feature_columns'], name='feature')\n"
            "display(feature_columns.to_frame())\n"
            "assert 'fs_match_probability' in set(modeling['feature_columns'])\n"
        ),
        nbf.v4.new_markdown_cell(
            "## Caveat\n\n"
            "The current labels are protocol-generated development evidence with "
            "double-pass validation and adjudication. They are not official legal "
            "renewal ground truth."
        ),
    ]
    nb["cells"] = cells
    nbf.write(nb, NOTEBOOK)


def main() -> int:
    generated_at = datetime.now().isoformat(timespec="seconds")
    dev = load_json(PROCESSED / "linkage_evaluation_summary_v3_dev_primary.json")
    validation = load_json(PROCESSED / "linkage_evaluation_summary_v3_validation_primary.json")
    modeling = load_json(BENCHMARK / "modeling/benchmark_v3_modeling_summary.json")
    manifest = load_json(BENCHMARK / "benchmark_v3_manifest.json")

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_method_metrics(
        method_frame(dev),
        "Latest v3 dev metrics",
        FIGURES / "benchmark_v3_dev_method_metrics.png",
    )
    plot_method_metrics(
        method_frame(validation),
        "Latest v3 validation metrics",
        FIGURES / "benchmark_v3_validation_method_metrics.png",
    )
    plot_modeling_counts(modeling, FIGURES / "benchmark_v3_modeling_counts.png")
    report_path = write_methodology_report(dev, validation, modeling, manifest, generated_at)
    write_status_files(dev, validation, modeling, manifest, generated_at)
    write_notebook(generated_at)

    print(
        json.dumps(
            {
                "generated_at": generated_at,
                "report": str(report_path.relative_to(PROJECT_ROOT)),
                "notebook": str(NOTEBOOK.relative_to(PROJECT_ROOT)),
                "figures": [
                    str((FIGURES / "benchmark_v3_dev_method_metrics.png").relative_to(PROJECT_ROOT)),
                    str((FIGURES / "benchmark_v3_validation_method_metrics.png").relative_to(PROJECT_ROOT)),
                    str((FIGURES / "benchmark_v3_modeling_counts.png").relative_to(PROJECT_ROOT)),
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
