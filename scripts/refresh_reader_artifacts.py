#!/usr/bin/env python3
"""Refresh reader-facing notebooks, figures, and reports from current outputs."""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data/processed/boamp"
BENCHMARK = PROCESSED / "regional_benchmark"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
NOTEBOOK = PROJECT_ROOT / "notebooks/12_successor_linkage_and_evaluation.ipynb"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compile_methodology_pdf(tex_path: Path) -> Path:
    """Compile the generated report so the PDF cannot lag behind the source."""
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError("latexmk is required to refresh the methodology PDF")
    subprocess.run(
        [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError(f"latexmk completed without creating {pdf_path}")
    return pdf_path


def method_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in summary["methods"]:
        all_frames = method["unweighted"]
        weighted = method.get("design_weighted", {})
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
                "precision_low": (all_frames.get("precision_at_1_interval_95") or [None, None])[0],
                "precision_high": (all_frames.get("precision_at_1_interval_95") or [None, None])[1],
                "recall_low": (all_frames.get("recall_at_1_interval_95") or [None, None])[0],
                "recall_high": (all_frames.get("recall_at_1_interval_95") or [None, None])[1],
            }
        )
    return pd.DataFrame(rows)


def plot_method_metrics(frame: pd.DataFrame, title: str, path: Path) -> None:
    plot = frame.set_index("method")[["precision", "recall", "fpr"]]
    ax = plot.plot(kind="bar", figsize=(9.2, 4.6), width=0.72)
    ax.set_title(title)
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    # Named by the conditional question each one answers, so the figure carries
    # the same reading as the metric definitions in the report and notebook 12.
    ax.legend(
        ["precision  P(C=1 | A=1)", "recall  P(C=1 | P=1)", "FPR  P(A=1 | P=0)"],
        frameon=False,
    )
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
                "positive anchors": payload["positive_anchors"],
            }
        )
    frame = pd.DataFrame(rows).set_index("split")
    ax = frame[["anchors", "primary positives", "positive anchors"]].plot(
        kind="bar", figsize=(8.2, 4.2), width=0.72
    )
    ax.set_title("Grand Ouest regional reference tables")
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


def latex_pvalue(value: float, digits: int = 3) -> str:
    """Render a p-value as maths rather than as Python's ``3.26e-05``.

    Inside ``\\(...\\)`` the exponent form typesets as an italic ``e`` next to a
    minus sign, which reads as a variable. Only the small values need the
    scientific form, so ordinary ones are left alone.
    """
    number = float(value)
    if number == 0.0:
        return "p<10^{-16}"
    if 1e-4 <= number < 1:
        return f"p={number:.{digits}g}"
    mantissa, exponent = f"{number:.{digits - 1}e}".split("e")
    return rf"p={mantissa}\times 10^{{{int(exponent)}}}"


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


def latex_cox_rows(frame: pd.DataFrame) -> str:
    rows = []
    for record in frame.to_dict("records"):
        rows.append(
            " & ".join(
                [
                    latex_escape(record["covariate"]),
                    f"{record['exp(coef)']:.3f}",
                    f"[{record['exp(coef) lower 95%']:.3f}, {record['exp(coef) upper 95%']:.3f}]",
                    f"{record['p']:.3g}",
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def latex_ph_rows(frame: pd.DataFrame) -> str:
    rows = []
    for record in frame.to_dict("records"):
        flag = r"\textbf{violated}" if record["p"] < 0.05 else "ok"
        rows.append(
            " & ".join(
                [
                    latex_escape(record["covariate"]),
                    f"{record['test_statistic']:.3f}",
                    f"{record['p']:.3g}",
                    flag,
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def latex_parametric_rows(frame: pd.DataFrame) -> str:
    rows = []
    for record in frame.to_dict("records"):
        rows.append(
            " & ".join(
                [
                    latex_escape(record["model"]),
                    str(int(record["parameters"])),
                    f"{record['log_likelihood']:.1f}",
                    f"{record['aic']:.1f}",
                    f"{record['bic']:.1f}",
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def latex_conditional_rows(frame: pd.DataFrame) -> str:
    """One row per episode age, both horizons side by side with their intervals."""
    wide = frame.pivot(
        index="contract_age_months", columns="horizon_months",
        values=["probability", "ci_95_low", "ci_95_high"],
    )
    return "\n".join(
        f"{age} months & "
        f"{wide.loc[age, ('probability', 12)] * 100:.2f}\\% & "
        f"[{wide.loc[age, ('ci_95_low', 12)] * 100:.2f}, {wide.loc[age, ('ci_95_high', 12)] * 100:.2f}] & "
        f"{wide.loc[age, ('probability', 24)] * 100:.2f}\\% & "
        f"[{wide.loc[age, ('ci_95_low', 24)] * 100:.2f}, {wide.loc[age, ('ci_95_high', 24)] * 100:.2f}] \\\\"
        for age in wide.index
    )


def latex_cox_sensitivity_rows(frame: pd.DataFrame) -> str:
    pivot_hr = frame.pivot(index="covariate", columns="arm", values="hazard_ratio")
    pivot_robust = frame.pivot(index="covariate", columns="arm", values="robustness_assessment")
    rows = []
    for covariate in pivot_hr.index:
        robustness = pivot_robust.loc[covariate, "main"]
        rows.append(
            " & ".join(
                [
                    latex_escape(covariate),
                    f"{pivot_hr.loc[covariate, 'strict']:.3f}",
                    f"{pivot_hr.loc[covariate, 'main']:.3f}",
                    f"{pivot_hr.loc[covariate, 'looser']:.3f}",
                    f"{pivot_hr.loc[covariate, 'contrast_high_recall']:.3f}",
                    latex_escape(robustness.replace("_", " ").title()),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def latex_trend_recommendation_items(signal_matrix: pd.DataFrame) -> str:
    """The operational reading already carried in ``trend_signal_matrix.csv``.

    The recommendation column is computed once by the trend generator; this only
    surfaces it in the report so the PDF and the CSV cannot drift apart. The
    wording is deliberately monitoring-oriented: a PELT break dates a level
    shift, it does not explain one.
    """
    items = []
    for row in signal_matrix.itertuples(index=False):
        items.append(
            r"\item \textbf{" + latex_escape(row.segment) + "} ("
            + latex_escape(row.state) + "): "
            + latex_escape(str(row.business_recommendation))
        )
    return "\n".join(items)


def latex_trend_signal_rows(signal_matrix: pd.DataFrame) -> str:
    rows = []
    for row in signal_matrix.itertuples(index=False):
        last_stable = "--" if pd.isna(row.last_stable_break) else latex_escape(str(row.last_stable_break))
        regime = (
            "--"
            if pd.isna(getattr(row, "hmm_current_regime", None))
            else latex_escape(str(row.hmm_current_regime))
        )
        rows.append(
            " & ".join(
                [
                    latex_escape(row.segment),
                    latex_escape(row.state),
                    f"{row.slope_episodes_per_quarter:.2f}",
                    f"{row.p_value:.3f}",
                    last_stable,
                    regime,
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def latex_stationarity_rows(diagnostics: dict[str, Any]) -> str:
    rows = []
    for segment, diag in diagnostics.items():
        if not diag.get("available"):
            rows.append(f"{latex_escape(segment)} & -- & -- & {latex_escape(diag.get('reason', ''))}" + r" \\")
            continue
        adf = diag["adf"]
        kpss_diag = diag["kpss"]
        kpss_text = (
            f"stat={kpss_diag['statistic']:.3f}, p={kpss_diag['p_value']:.3f}"
            if kpss_diag.get("available", True)
            else "n/a"
        )
        rows.append(
            " & ".join(
                [
                    latex_escape(segment),
                    f"stat={adf['statistic']:.3f}, p={adf['p_value']:.3f}",
                    kpss_text,
                    "",
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
    reviewed_anchors = manifest["reviewed_anchors"]
    usable_anchors = sum(split["usable_anchors"] for split in manifest["splits"].values())
    locked = manifest["splits"]["validation"]
    pilot = manifest["splits"]["dev"]
    ceiling = manifest["candidate_reachability"]
    candidates = load_json(PROCESSED / "linkage_candidates_summary.json")
    candidate_audit = load_json(PROCESSED / "candidate_generation_audit.json")
    blocking_loss = candidate_audit["blocking_loss"]
    cpv_continuity = candidate_audit["cpv_continuity"]
    survival_cohort = load_json(PROCESSED / "survival_cohort_summary.json")
    survival = load_json(PROCESSED / "survival_dataset_summary.json")
    buyer_audit = load_json(PROCESSED / "buyer_blocking_legal_form_audit_summary.json")
    review_audit = load_json(PROJECT_ROOT / "data/review/review_audit_evaluation.json")
    survival_main = survival["variants"]["main"]
    survival_strict = survival["variants"]["strict"]
    survival_looser = survival["variants"]["looser"]
    survival_contrast = survival["variants"]["contrast_high_recall"]
    m_b = validation_frame.loc[validation_frame["method"].eq("M_B_text_ranking")].iloc[0]
    m_c = validation_frame.loc[validation_frame["method"].eq("M_C_weighted_gated")].iloc[0]
    m_d = validation_frame.loc[validation_frame["method"].eq("M_D_fellegi_sunter")].iloc[0]
    # The raw anchor-level cell counts behind M_B's locked-split rates. They are
    # quoted rather than the rates alone so a reader can see that the two kinds
    # of miss -- abstention and wrong acceptance -- are counted separately.
    m_b_cells = next(
        method["unweighted"]
        for method in validation["methods"]
        if method["method"] == "M_B_text_ranking"
    )

    survival_summary = load_json(PROCESSED / "survival_analysis_summary.json")
    cox_results = pd.read_csv(PROCESSED / "survival_cox_results.csv")
    ph_diagnostics = pd.read_csv(PROCESSED / "survival_ph_diagnostics.csv")
    parametric_comparison = pd.read_csv(PROCESSED / "survival_parametric_comparison.csv")
    cox_sensitivity = pd.read_csv(PROCESSED / "survival_cox_linkage_sensitivity.csv")
    km_horizons = pd.read_csv(PROCESSED / "survival_km_horizons.csv").set_index("months")
    conditional_probabilities = pd.read_csv(PROCESSED / "survival_conditional_probabilities.csv")
    template_risk = survival_summary["template_risk_sensitivity"]
    template_main = template_risk["main"]
    template_kept = template_risk["recensoring_template_risk_links"]
    trend_summary = load_json(PROCESSED / "trend_analysis_summary.json")
    trend_signal_matrix = pd.read_csv(PROCESSED / "trend_signal_matrix.csv")
    validation_sweep = pd.read_csv(PROCESSED / "quality_evidence/validation_m_b_threshold_sweep.csv")
    dev_sweep = pd.read_csv(PROCESSED / "quality_evidence/dev_m_b_threshold_sweep.csv")
    locked_70 = validation_sweep.loc[validation_sweep["threshold_percent"].eq(70.0)].iloc[0]
    locked_60 = validation_sweep.loc[validation_sweep["threshold_percent"].eq(60.0)].iloc[0]
    pilot_70 = dev_sweep.loc[dev_sweep["threshold_percent"].eq(70.0)].iloc[0]
    pilot_60 = dev_sweep.loc[dev_sweep["threshold_percent"].eq(60.0)].iloc[0]
    ph_violations = survival_summary["cox"]["ph_violations_p_lt_0_05"]
    temporal = survival_summary["cox"]["temporal_validation"]
    extended = survival_summary["cox"]["temporal_validation_including_latest_cohort"]
    borderline = survival_summary["borderline_link_sensitivity"]

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
\title{{BOAMP Digital Procurement Study\\\large Successor Linkage, Survival, and Trend Methodology}}
\author{{BOAMP Data Science Internship Project}}
\date{{{generated_at}}}
\begin{{document}}
\sloppy
\maketitle
\tableofcontents
\newpage

\section*{{Technical Summary}}
This report states the BOAMP successor-linkage, survival, and trend methodology
and the results it produces. The study does not claim to certify legal contract
renewals. Its event is an \emph{{observable successor procurement}}: a later BOAMP
procurement episode from the same buyer that is sufficiently similar to an
earlier awarded digital procurement episode.

The Grand Ouest survival cohort contains
{survival_main["validation"]["rows"]:,} awarded digital procurement episodes.
The main linkage rule accepts {survival_main["validation"]["events"]:,}
successor links, giving an observed event rate of
{survival_main["description"]["event_rate"]:.3f} and a median observed successor
time of {survival_main["description"]["median_time_to_successor_months"]:.2f}
months among linked events. The candidate pool is broad:
{candidates["candidate_pairs"]:,} candidate pairs from
{candidates["anchors_with_candidates"]:,} anchors with at least one candidate.

Linkage accuracy is read against the Grand Ouest regional reference: a
stratified review of {reviewed_anchors} anchors of which {usable_anchors} are
usable, carried out against real BOAMP notices before these methods existed. On
its locked split of {locked["usable_anchors"]} anchors
({locked["positive_anchors"]} with a reviewed successor),
\code{{M\_B\_text\_ranking @ 0.70}} has precision {m_b.precision:.3f}
(95\% CI {m_b.precision_low:.3f}--{m_b.precision_high:.3f}), recall
{m_b.recall:.3f} (95\% CI {m_b.recall_low:.3f}--{m_b.recall_high:.3f}),
false-positive rate {m_b.fpr:.3f}, and {int(m_b.accepted_links)} accepted links.
Recall is capped at {ceiling["candidate_generation_recall_ceiling"]:.3f} by
candidate generation, which reaches
{ceiling["positive_anchors_with_reviewed_successor_in_pool"]} of the
{ceiling["positive_anchors"]} reviewed successors.
These are reference-sample estimates on fewer than a hundred anchors, not
independently validated accuracy: the labels were generated by a single LLM
research pass and spot-checked on a subset by the project owner, not verified
anchor-by-anchor and not reviewed by an independent specialist panel.
\code{{M\_C\_weighted\_gated}} recovers more positives
(recall {m_c.recall:.3f}) but admits more false positives
(FPR {m_c.fpr:.3f}). This trade-off is important because a false positive in a
survival dataset creates both a false event and a false event time, while an
abstained link is handled conservatively as right-censoring.

\section{{Context}}
Gigalis is the digital central purchasing body (\emph{{centrale d'achat
num\'erique}}) for public institutions in the Pays de la Loire region and beyond.
It designs pooled framework agreements in cloud, cybersecurity, digital
infrastructure, and artificial intelligence, and its strategic relevance depends
on anticipating members' future needs. The internship's central question is:
using historical public digital procurement data, how can the probability that a
contract or technological segment generates an identifiable purchasing need
within the next 12--24 months be estimated, and how can that estimate inform
Gigalis's central purchasing strategy?

This decomposes into three sub-problems: a \emph{{lifetime problem}} (survival
analysis: how long between a digital procurement episode and its successor?), a
\emph{{trend problem}} (change-point detection: which technological segments are
growing, plateauing, or declining?), and a \emph{{text-signal problem}} (NLP:
extracting a contract's technological theme automatically). This project answers
the first two in depth; the NLP sub-problem is addressed with a reproducible
coarse substitute rather than a trained classifier (\S\ref{{sec:nlp-scope}}).

BOAMP does not encode legal contract renewal as an explicit field, so the study
cannot certify legal renewals. It instead constructs and evaluates a proxy: an
\emph{{observable successor procurement}}, a later BOAMP episode from the same
buyer that is sufficiently similar to, and temporally plausible relative to, an
earlier awarded digital procurement episode. Every downstream result --
survival curves, hazard ratios, trend segments -- is conditioned on this proxy,
not on certified legal ground truth. The strongest defensible framing is:
because legal renewal status is not directly observed, the study measures
observable successor procurements under a precision-first linkage rule, then
assesses how sensitive its descriptive conclusions are to that rule.

\section{{What The Pipeline Measures}}
Let \(i\) index an earlier awarded procurement episode (the \emph{{anchor}}) and
\(j\) index a later candidate episode. The object of interest is not a legal
renewal certificate, because BOAMP notices do not provide that ground truth
consistently. Instead, the pipeline estimates whether the data show a later
procurement episode that is credible enough to be treated as a successor:
\[
Y_i =
\begin{{cases}}
1, & \text{{if one accepted observable successor is found before the cutoff,}}\\
0, & \text{{otherwise.}}
\end{{cases}}
\]
The event time is
\[
\tau_i = v_{{\hat{{j}}}} - u_i,
\]
where \(u_i\) is the award-date origin of episode \(i\), \(v_{{\hat{{j}}}}\) is
the first-publication date of the accepted successor, and \(\hat{{j}}\) is the
selected candidate. If no accepted successor is found before 2025-12-31, the
episode is right-censored.

\subsection{{Notation}}
\label{{sec:notation}}
Three layers are chained -- candidate generation, linkage decision, survival --
and each answers a different conditional question. One symbol table serves the
whole report so that no quantity below is undefined at the point it is used.

\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{llX}}
\toprule
Layer & Symbol & Meaning \\
\midrule
Data      & \(u_i\)                & award-date origin of anchor \(i\) (calendar date) \\
Data      & \(v_j\)                & first-publication date of candidate \(j\) (calendar date) \\
Data      & \(J_i\)                & set of candidates for anchor \(i\) surviving blocking \\
Data      & \(T_{{ij}}, B_{{ij}}, K_{{ij}}, G_{{ij}}\) & pairwise text, buyer, CPV-continuity and timing evidence \\
\midrule
Reference & \(R_i \in J_i\cup\{{\varnothing\}}\) & reviewed successor identity, \(\varnothing\) if the review found none \\
Reference & \(P_i=\mathbf{{1}}\{{R_i\neq\varnothing\}}\) & the reference identifies a successor for anchor \(i\) \\
Reference & \(E_i=\mathbf{{1}}\{{R_i\in J_i\}}\) & the reviewed successor survived candidate generation \\
\midrule
Decision  & \(\hat{{R}}_i \in J_i\cup\{{\varnothing\}}\) & successor accepted by the linkage rule, \(\varnothing\) if it abstains \\
Decision  & \(A_i=\mathbf{{1}}\{{\hat{{R}}_i\neq\varnothing\}}\) & the method accepted some successor \\
Decision  & \(C_i=\mathbf{{1}}\{{\hat{{R}}_i=R_i\neq\varnothing\}}\) & the accepted successor is exactly the reviewed one \\
Decision  & \(\mathcal{{L}}_m\)      & one linkage definition (method and threshold), \(m\) indexing the arms \\
\midrule
Survival  & \(T_i\)                & months from award to an observable successor (latent) \\
Survival  & \(L_i\)                & follow-up months from award to the 2025-12-31 cutoff \\
Survival  & \(Y_i=\mathbf{{1}}\{{T_i\le L_i\}}\) & an accepted observable successor was seen before the cutoff \\
Survival  & \(S(t), F(t), \lambda(t)\) & survivor, cumulative-event and hazard functions of \(T\) \\
Survival  & \(a, h\)               & current episode age and forward horizon, in months \\
\bottomrule
\end{{tabularx}}
\caption{{Symbols used throughout, grouped by the layer they belong to. Reading
across the three middle blocks gives the logic of the study: blocking decides
whether \(E_i=1\), the linkage rule decides \(A_i\) and hence \(C_i\), and the
survival layer turns \(A_i\) into the event indicator \(Y_i\).}}
\end{{table}}

Two conventions are worth stating because they prevent the two collisions a
reader would otherwise hit. Calendar dates are lower-case (\(u_i, v_j\)) so that
the upper-case \(A_i\) and \(C_i\) can carry their usual meaning as decision
indicators. The hazard is written \(\lambda(t)\) rather than the equally common
\(h(t)\), because \(h\) is already the forward horizon in the operational
quantity \(P(T\le a+h\mid T>a)\), which is the report's headline number. Pairwise
evidence always carries two indices (\(T_{{ij}}\) is a text similarity); a single
index means an episode-level quantity (\(T_i\) is a survival time).

\subsection{{One Chain Of Questions}}
The methods below are not a collection of separate exercises. Each stage hands
the next one its input, and each is best remembered by the question it answers.

\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{rlX}}
\toprule
Step & Quantity & Question it answers \\
\midrule
1 & \(P(E=1\mid P=1)\)          & If a reviewed successor exists, did blocking keep it reachable? \\
2 & \(P(C=1\mid A=1)\)          & If the method accepts a link, is the accepted candidate the reviewed one? \\
2 & \(P(C=1\mid P=1)\)          & If a reviewed successor exists, is it recovered exactly? \\
3 & \(Y_i=\mathbf{{1}}\{{A_i=1\}}\) & Does this episode contribute an event or censored exposure? \\
4 & \(S(t)=P(T>t)\)             & How long do episodes go without an observable successor? \\
5 & \(P(T\le a+h\mid T>a)\)     & Given none by age \(a\), how likely is one in the next \(h\) months? \\
6 & \(\lambda(t\mid X)=\lambda_0(t)e^{{\beta^{{\top}}X}}\) & Which episode characteristics go with a sooner successor? \\
7 & \(P_m(T\le t)\), \(\mathrm{{HR}}_{{k,m}}\) & Which of those conclusions survive a change of linkage definition? \\
8 & PELT, HMM, OLS              & When did the market's level shift, and where is it now? \\
\bottomrule
\end{{tabularx}}
\caption{{The study as one chain. Steps 1--2 measure the instrument, step 3
builds the data, steps 4--6 estimate, step 7 stress-tests, step 8 describes the
surrounding market.}}
\end{{table}}

\section{{End-to-End Pipeline}}
The implemented pipeline is:
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

\paragraph{{Standardisation and feature engineering.}}
The preprocessing stage standardises dates, buyer identifiers, buyer names, CPV
codes, text fields, procedure information, framework flags, and duration fields
where they are explicitly available. It does not impute missing duration or
invent expected expiry dates. Buyer standardisation is deliberately conservative:
municipal variants such as commune/ville/mairie can be harmonised, but
intercommunal legal forms are preserved as distinct entities, and conflicting
validated SIRENs block a buyer match. The legal-form audit reports
{buyer_audit["hard_fail_conflicting_validated_siren"]} accepted links with
conflicting validated SIREN evidence and {buyer_audit["municipal_intercommunal_mix"]}
accepted municipal/intercommunal mixes.

\paragraph{{Episode reconstruction.}}
BOAMP is notice-level data, but the analysis needs procurement-level events.
Therefore notices belonging to the same procurement are reconstructed into an
episode. This avoids treating a consultation notice, correction notice, award
notice, or lot-level administrative notice as separate renewals.

\paragraph{{Candidate generation.}}
Candidate generation is intentionally broad and is not the final model. For an
anchor episode with award date \(u_i\) and candidate publication date \(v_j\),
the candidate enters the exposed set \(J_i\) only if
\[
u_i+90 \leq v_j \leq u_i+2920 .
\]
The lower bound removes very near follow-up notices and parallel administrative
activity. The upper bound is approximately eight years; it keeps the candidate
pool inclusive enough for long public-procurement cycles. Precision is then
controlled by the selection stage rather than by a narrow time window. The
blocking rule generates {candidates["candidate_pairs"]:,} candidate
pairs, with a median of {candidates["candidates_per_anchor"]["median"]:.0f}
candidates per anchor.

The question this stage must answer is \emph{{not}} how often it is right. It is
whether a successor that genuinely exists can still be found downstream, which
is the conditional probability
\[
P(E_i=1 \mid P_i=1)
\quad\text{{estimated by}}\quad
\hat{{P}}(E=1\mid P=1)=\frac{{{ceiling["positive_anchors_with_reviewed_successor_in_pool"]}}}{{{ceiling["positive_anchors"]}}}={ceiling["candidate_generation_recall_ceiling"]:.3f} .
\]
In words: given that the reference identifies a successor for an anchor,
blocking retained that successor in
{ceiling["positive_anchors_with_reviewed_successor_in_pool"]} of the
{ceiling["positive_anchors"]} reviewed cases. This is candidate-generation
reachability measured on this reference sample -- in record-linkage terms
\emph{{pairs completeness}} -- and it is not an estimate of population recall.
It is a hard ceiling: a successor discarded here can never be recovered by any
scorer, which is why the two stages are given opposite objectives. Candidate
generation maximises \(P(E=1\mid P=1)\); the linkage rule then maximises
\(P(C=1\mid A=1)\) (\S\ref{{sec:locked-results}}). High recall first, high
precision later.

That objective is why exact CPV continuity is \emph{{not}} imposed as a hard
blocking rule, even though it would raise precision: CPV coding is incomplete,
often generic, and assigned by the contracting authority rather than validated,
so a same-division requirement would remove reviewed successors from \(J_i\) and
lower \(P(E=1\mid P=1)\) permanently.
\href{{https://doi.org/10.1007/978-3-540-44918-8_6}}{{Christen and Goiser (2007)}}
identify pairs completeness as a confounder that must be published alongside any
linkage-quality figure rather than folded into it, which is why it is reported
here as its own conditional probability rather than absorbed into recall.

Relaxing hard same-CPV blocking is checked against the reference rather than
asserted. Among the {cpv_continuity["reviewed_reference_pairs"]} reviewed
successor pairs -- labelled with no knowledge of any linkage method --
{cpv_continuity["reviewed_cross_cpv2_count"]}
({cpv_continuity["reviewed_cross_cpv2_share"] * 100:.1f}\%) connect episodes in
\emph{{different}} CPV divisions. A hard same-division block would therefore
discard those {cpv_continuity["reviewed_successors_lost_to_hard_same_division_block"]}
reviewed successors outright and cut the attainable recall ceiling to
{cpv_continuity["recall_ceiling_under_hard_same_division_block"]:.3f}. The
discarded cases are substantive continuations rather than noise: equipment
purchases followed by their maintenance contracts (CPV-32 or CPV-35 to CPV-50)
and software purchases followed by the corresponding service contracts (CPV-48
to CPV-72). This is the ordinary goods/services split built into the CPV
hierarchy itself, compounded by assignment error that
\href{{https://aclanthology.org/2023.clicit-1.47}}{{Siciliani et al. (2023)}}
report is frequent even for human experts given the size of the vocabulary.

\section{{Grand Ouest Regional Reference}}
\label{{sec:linkage-caveat}}
Linkage evaluation uses a Grand Ouest regional reference sample drawn from the
study population itself: {reviewed_anchors} awarded digital procurement anchors
stratified by CPV theme, buyer-identifier quality, and duration availability
across Bretagne, Pays de la Loire, and Normandie. Each anchor was reviewed on
2026-08-11 against the notices and official BOAMP URLs of its candidates. The
review is a single LLM research pass over those notices, their official URLs, and
wider public sources, spot-checked on a subset by the project owner; it proposes
one successor or an abstention per anchor. No linkage method existed or was
consulted when the labels were produced, so the labels are independent of every
method scored against them.

Reference anchors are resolved onto the procurement-episode table through their
BOAMP notice identifiers.
{reviewed_anchors - manifest["remap"]["resolved_to_current_episodes"]} anchors did
not resolve to exactly one episode and were dropped rather than guessed, and
anchors the review declined to decide are excluded rather than counted as
negatives, leaving {usable_anchors} evaluable anchors.

The sample is split into a pilot part and a locked part. The acceptance threshold
was fixed before the locked part was read, which is what allows locked-split
figures to be reported as held out. The pilot split carries
{pilot["usable_anchors"]} evaluable anchors
({pilot["positive_anchors"]} with a reviewed successor) and
{modeling["outputs"]["dev"]["rows"]:,} pair rows; the locked split carries
{locked["usable_anchors"]} evaluable anchors
({locked["positive_anchors"]} with a reviewed successor) and
{modeling["outputs"]["validation"]["rows"]:,} pair rows.

Two anchor counts appear for the locked split and the difference is by
construction, not a discrepancy: {locked["usable_anchors"]} anchors
are evaluable at anchor level, of which
{modeling["outputs"]["validation"]["anchors"]} also have at least one exposed
candidate pair and so appear in the pair-level table. The remaining
{locked["usable_anchors"] - modeling["outputs"]["validation"]["anchors"]}
generated no candidate at all, which is a blocking-stage loss counted against
recall rather than a scoring failure. Anchor-level metrics use
{locked["usable_anchors"]}; pair-level ROC and precision-recall
curves use {modeling["outputs"]["validation"]["anchors"]}.

Candidate generation, not scoring, sets the ceiling on recall against this
reference: \(P(C=1\mid P=1)\le P(E=1\mid P=1)={ceiling["candidate_generation_recall_ceiling"]:.3f}\),
as quantified above. No method evaluated here can exceed that, and the gap is a
blocking-stage limitation rather than a scoring failure.

Both unreachable cases were attributed to the specific blocking condition that
rejected them, evaluated in the order the generator applies them, and neither is
an implementation defect. In the first, the reviewed anchor never reached the
blocking step at all: its episode carries an award notice, but that notice has no
structured Grand Ouest address, so it is absent from the regional notice table,
no award date could be resolved, and the episode was dropped from the cohort --
one of the {survival_cohort["selection_funnel"]["dropped_unresolved_award_date"]}
episodes the cohort funnel already counts under that heading. In the second, the
buyer changed legal form between the two procurements, from a communal social
action centre (CCAS) to its intercommunal successor (CIAS); the anchor carries no
SIREN, so no identifier could bridge the transition, and the normalised names
differ. Buyer blocking rejected the pair by design, because the pipeline
deliberately preserves intercommunal legal forms as distinct entities rather than
merging them on name similarity.

That second case is the known weak point of buyer identification in French
procurement notices rather than a defect specific to this pipeline:
\href{{https://doi.org/10.1038/s41597-023-02213-z}}{{Potin et al. (2023)}} report
missing agent identification as the most serious quality problem in the French
data, with buyer SIRETs populated on a minority of lots, and note that the same
authority appears under alternative and former names. Recovering this particular
link would require an external legal-succession register, not a change to the
blocking rule. Both losses are accepted and documented rather than repaired;
neither is repaired by loosening the rule, and the audit reports
{blocking_loss["unexplained_cases"]} unexplained cases, which is the check that
distinguishes a blocking trade-off from a bug. The attribution is regenerated by
\texttt{{scripts/audit\_candidate\_generation.py}}.

Five limitations bind everything computed from the regional reference. The
labels were spot-checked on a subset rather than verified anchor-by-anchor;
anchors outside that subset carry the model's judgement as recorded, so this is a
reference sample and not ground truth, and it is not an independent human
specialist panel. The sources behind each individual label were not recorded, so
a given anchor's evidence trail cannot be fully reconstructed or independently
re-executed. Negatives are corpus-relative: roughly 25 candidates per anchor were
considered rather than the full pool, so a false-positive rate computed on them
is conservative by construction and is a diagnostic on this sample rather than a
population-wide rate. Judging whether later notice text continues an earlier need
draws on the same text, CPV, and date evidence the text-ranking method scores, so
the labels are method-independent without being fully evidence-independent. And
the sample is small enough that every point estimate needs its interval read
beside it.

\section{{Linkage Algorithms}}
All methods operate on the same exposed candidate set. This is crucial: the
primary method is not comparing text over the whole BOAMP universe. Buyer and
time plausibility are imposed before text ranking.

\paragraph{{\(M_A\): deterministic evidence.}}
Define \(B_{{ij}}\) as buyer-identity support, \(K_{{ij}}\) as CPV continuity, and
\(T_{{ij}}\) as text similarity. The deterministic rule accepts a link only when
strong buyer evidence is present, CPV continuity is positive, and a minimum text
signal is present:
\[
B_{{ij}}=1,\quad K_{{ij}}>0,\quad T_{{ij}}\geq t_A .
\]
It is interpretable, but it loses recall when CPV or buyer identifiers are
missing or noisy.

\paragraph{{\(M_B\): text ranking.}}
For each episode, the text fields are converted to TF--IDF vectors \(x_i\) and
\(x_j\). Text similarity is cosine similarity:
\[
T_{{ij}}=\cos(x_i,x_j)=\frac{{x_i\cdot x_j}}{{\|x_i\|\|x_j\|}}.
\]
Within the exposed candidate set \(J_i\), the method selects
\[
\hat{{j}}_i=\arg\max_{{j\in J_i}} T_{{ij}},
\]
and accepts that candidate as \(\hat{{R}}_i\) only if
\[
T_{{i\hat{{j}}_i}}\geq 0.70,
\qquad\text{{otherwise}}\quad \hat{{R}}_i=\varnothing .
\]
The abstention branch is not a formality: it is what makes \(A_i\) a genuine
decision and what turns a non-accepted anchor into a censored observation rather
than a negative one.
This is the primary method because it is simple, reproducible, auditable, and
best matches the precision-first objective.

\paragraph{{\(M_C\): weighted gated score.}}
This method combines evidence components into a score:
\[
S_{{ij}}=0.50B_{{ij}}+0.25T_{{ij}}+0.20K_{{ij}}+0.05G_{{ij}},
\]
where \(G_{{ij}}\) is a timing-plausibility score. Missing components are handled
by the implemented score-normalisation logic rather than imputation. The method
then applies gates and ranks by \(S_{{ij}}\). It is useful as a higher-recall
contrast, but the validation results show a materially higher false-positive
rate.

\paragraph{{\(M_D\): Fellegi--Sunter probabilistic linkage.}}
Let \(\gamma_{{ij}}\) be the discretised comparison vector for buyer, text, CPV,
and timing evidence. Fellegi--Sunter estimates the likelihood of observing
\(\gamma_{{ij}}\) under a match class \(M\) and a non-match class \(U\), then uses
the log-likelihood ratio
\[
w_{{ij}}=\log\frac{{P(\gamma_{{ij}}\mid M)}}{{P(\gamma_{{ij}}\mid U)}}.
\]
The fitted model provides \code{{fs\_match\_weight}} and
\code{{fs\_match\_probability}}. On this cohort it does not outperform the
simple text-ranking rule, likely because reviewed successors are rare and
same-buyer procurement activity contains many non-renewal lookalikes.

\paragraph{{Declared duration.}} Declared contract duration is not imposed as a
mandatory successor-linkage condition. Reliable duration is unavailable for most
episodes, so a duration-conditioned rule could differentiate itself only on a
minority of the cohort; and where the evidence does exist, the observed
relationship between declared duration and successor timing is too weak to
support a hard expected-expiry rule -- many declared successors are published
well before the declared end date. Duration is therefore retained as a
descriptive diagnostic about data reliability, not as an event-definition
criterion, and the descriptive comparison between declared duration and observed
successor delay remains part of the evidence on that basis.

\paragraph{{Sensitivity framework.}} Event-definition sensitivity is carried by
four threshold and method arms (\(M_B@0.80\), \(M_B@0.70\), \(M_B@0.60\),
\(M_C@0.70\)), which vary the decision rule along the dimension that actually
moves the event set, together with the borderline-band check described in
\S\ref{{sec:survival-methods}}.

\section{{Survival Modeling}}
\label{{sec:survival-methods}}
\paragraph{{Event and censoring.}} The accepted-link decision from \(M_B\)
defines the survival event. Let \(T_i\) be the months from award to an
observable successor and \(L_i\) the months from award to the 2025-12-31 cutoff.
The pipeline observes
\[
\tilde{{T}}_i=\min(T_i,L_i),
\qquad
Y_i=\mathbf{{1}}\{{T_i\le L_i\}}=\mathbf{{1}}\{{A_i=1\}} ,
\]
so \(\tau_i=\tilde{{T}}_i\) is the follow-up time actually stored and \(Y_i\)
the event indicator stored beside it. For a censored episode the data assert
\(T_i>L_i\) and nothing more: the eventual \(T_i\) is unobserved, and it is
\emph{{not}} the case that \(T_i=L_i\). Writing the follow-up limit as \(L_i\)
rather than the conventional \(C_i\) keeps it clear of the linkage correctness
indicator of \S\ref{{sec:notation}}.

\(Y_i=0\) means no accepted observable successor was found before the cutoff.
It does not mean ``not renewed'': the contract may have been re-procured through
a central purchasing body, below publication thresholds, or under a link the
precision-first rule declined to accept.

\paragraph{{Kaplan--Meier.}} The survivor function and its complement are
\[
S(t)=P(T>t),
\qquad
F(t)=P(T\le t)=1-S(t) ,
\]
read as ``the probability that an episode is still without an observable
successor at age \(t\)'' and ``the probability that one has appeared by age
\(t\)''. Both are estimated non-parametrically by
\href{{https://doi.org/10.1080/01621459.1958.10501452}}{{Kaplan and Meier
(1958)}}'s product-limit estimator
\[
\hat{{S}}(t)=\prod_{{t_k\le t}}\left(1-\frac{{d_k}}{{n_k}}\right),
\]
where \(d_k\) is the number of observable-successor events at event time
\(t_k\) and \(n_k\) the number of episodes still at risk immediately before it.
At each observed event time the estimator multiplies in the probability of
getting past that instant among the episodes still under observation, which is
what lets censored episodes contribute their exposure up to the moment they
leave. It is reported overall and stratified by CPV segment, the only
stratification this study estimates; group differences use a multivariate
log-rank test across those segments, whose hypotheses are
\[
\begin{{aligned}}
H_0&:\; S_{{32}}(t)=S_{{35}}(t)=S_{{48}}(t)=S_{{72}}(t)
\quad\text{{over the observation window}},\\
H_1&:\;\text{{at least one segment survivor function differs}}.
\end{{aligned}}
\]
This is an omnibus test: rejecting \(H_0\) says the four curves are not all
equal, not that every pair of segments differs. Region and framework status
enter the Cox model as covariates but are not estimated as separate
Kaplan--Meier strata.

\paragraph{{Hazard and the Cox model.}} The hazard is the conditional event rate
among episodes that have survived to \(t\):
\[
\lambda(t)=\lim_{{\Delta t\to 0}}
\frac{{P(t\le T<t+\Delta t\mid T\ge t)}}{{\Delta t}} .
\]
It answers ``among episodes that have reached age \(t\) with no observable
successor, how fast are successors appearing right now?'' A hazard is a rate,
not a probability: it is not bounded by 1, and a hazard ratio is consequently
not a risk ratio. The semi-parametric model is
\[
\lambda(t\mid X)=\lambda_0(t)\exp(\beta_1X_1+\cdots+\beta_pX_p),
\qquad
\mathrm{{HR}}_k=\exp(\beta_k),
\]
which leaves the baseline \(\lambda_0(t)\) unspecified and reads each covariate
as a multiplicative shift of the hazard. \(\mathrm{{HR}}_k>1\) means a higher
instantaneous observable-successor hazard at every \(t\), holding the other
included covariates fixed; \(\mathrm{{HR}}_k<1\) a lower one. Covariates are
selected for substantive relevance and data quality rather than automated
search: CPV digital segment, buyer region, framework-agreement flag,
validated-SIREN availability, and centered award year. The rule of one
covariate per ten observed events (Van Belle et al., 2002) is respected with
{survival_summary["cox"]["events"]:,} events supporting
{survival_summary["cox"]["covariates"]} covariates.

\paragraph{{Proportional-hazards diagnostic.}} The model above assumes
\(\mathrm{{HR}}_k(t)=\exp(\beta_k)\) does not vary with \(t\). The
Schoenfeld-residual test (Grambsch and Therneau, 1994) takes
\[
H_0:\;\text{{the hazard ratio for covariate }} k \text{{ is constant in }} t,
\]
and a \(p<0.05\) is evidence against it. Where it fails, the coefficient is
still a well-defined summary -- a time-averaged association -- and is reported
as such rather than silently dropped or used to discard the model.

\paragraph{{Parametric models.}} Exponential, Weibull, log-logistic, log-normal,
and generalized-gamma models are compared by log-likelihood \(\ell\) and the
penalised criteria
\[
\mathrm{{AIC}}=2k-2\ell,
\qquad
\mathrm{{BIC}}=k\log n-2\ell ,
\]
with \(k\) the number of fitted parameters and \(n\) the number of episodes.
Lower is better \emph{{within the compared set}}: these criteria rank the five
families against one another on a fit-versus-complexity trade-off and say
nothing about whether the winner fits in absolute terms, which is why the choice
is also checked graphically against the Kaplan--Meier curve. Their role is to
identify the best-fitting family and to provide the instrument any extrapolation
past the observation window would require. They are \emph{{not}} the source of
the reported 12/24-month probabilities: every horizon quoted in this report
falls inside the observed window, and the smooth families flatten the empirical
renewal shoulder, so the operational conditional probabilities are read off the
Kaplan--Meier estimator, which imposes no shape.

\paragraph{{Temporal validation.}} The model is fit once on episodes awarded
2015--2021 and scored out of time without refitting. The primary evaluation
window is 2022--2024, as specified by the internship guideline; 2022--2025 is
carried as a sensitivity read that adds the shortest-follow-up award cohort.
Harrell's concordance index is a probability about \emph{{pairs}}, not a
classification accuracy:
\[
C\approx P\big(\text{{the model ranks the earlier-event episode as higher risk}}
\ \big|\ \text{{the pair is comparable}}\big),
\]
where a pair is comparable when censoring allows their orderings to be
determined. \(C=0.5\) is ordering no better than chance and \(C=1\) is perfect
ordering. It is reported on each split to assess discrimination and out-of-time
stability, not to target a specific value.

\paragraph{{Interval estimation.}} The operational conditional probabilities are
ratios of two points on the same fitted Kaplan--Meier curve, so their sampling
error is not the sum of two independent errors. For a statistic
\(\hat{{\theta}}\) the pipeline therefore resamples episodes with replacement,
refits Kaplan--Meier on each resample, and reports the empirical percentiles of
the replicates \(\hat{{\theta}}^{{(1)}},\dots,\hat{{\theta}}^{{(B)}}\) with
\(B=500\). Resampling whole episodes carries the dependence between \(S(a)\) and
\(S(a+h)\) through without a delta-method derivation or further distributional
assumptions.

\paragraph{{Evaluation strategy.}} The three checks applied to the linkage --
scoring it against a reference subset, comparing linked with unlinked episodes,
and re-running the analysis under alternative linkage rules -- follow the guidance
of \href{{https://doi.org/10.1093/ije/dyx177}}{{Harron et al. (2017)}} for
evaluating linkage quality in linked-data analyses. That source supports the
strategy; it does not validate the precision, recall, or threshold reported here.

\paragraph{{Borderline-link robustness.}} Threshold sensitivity swaps one event
definition for another; a separate check asks how much rests on the anchors the
frozen rule nearly classified the other way. Anchors whose best candidate scores
within \(\pm0.05\) of the \(0.70\) acceptance bar are dropped entirely --
borderline acceptances and borderline abstentions alike -- and the headline
Kaplan--Meier levels and hazard ratios are recomputed on the remainder. The band
is fixed a priori and is not searched over.

\paragraph{{Linkage sensitivity.}} Because the event is linkage-conditioned, the
same Kaplan--Meier and Cox analyses are repeated under the strict
(\(M_B@0.80\)), main (\(M_B@0.70\)), looser (\(M_B@0.60\)), and high-recall
contrast (\(M_C@0.70\)) event definitions. Writing \(\mathcal{{L}}_m\) for
linkage definition \(m\) makes the dependence explicit: every survival quantity
in this report is really
\[
P_m(T\le t)
\quad\text{{and}}\quad
\mathrm{{HR}}_{{k,m}},
\]
that is, a probability and a hazard ratio computed \emph{{under}}
\(\mathcal{{L}}_m\). The subscript is dropped elsewhere only because
\(\mathcal{{L}}_{{\text{{main}}}}\) is fixed. The arms are deterministic
pre-specified scenarios, not draws from a distribution over linkage rules, so
varying \(m\) bounds a sensitivity rather than estimating an uncertainty. A
conclusion is reported as robust only when it is stable in sign and approximate
magnitude across these arms; the recurring finding is that \(P_m(T\le t)\) moves
a great deal with \(m\) while the sign and rough size of the leading
\(\mathrm{{HR}}_{{k,m}}\) do not.

\section{{Trend And Change-Point Detection}}
Quarterly awarded-episode counts \(N_{{s,q}}\) are built for the overall cohort
and each CPV digital segment from 2015Q2 (the first complete quarter) through
2025Q4, including zero-count quarters.

These are time-series methods and are stated in their natural forms; forcing
them into conditional-probability notation would obscure rather than clarify.
The one exception is the hidden Markov model, whose output genuinely is a
conditional probability and is written as one below. Greek letters in this
section are local to it: the penalty multiplier \(\lambda\) is unrelated to the
hazard \(\lambda(t)\) of \S\ref{{sec:survival-methods}}.

\paragraph{{Change-point detection (PELT).}}
\href{{https://doi.org/10.1080/01621459.2012.737745}}{{Killick, Fearnhead and
Eckley (2012)}}'s PELT algorithm minimizes, over the number of change points
\(K\) and their positions \(\tau_1<\cdots<\tau_K\),
\[
\sum_{{k=0}}^{{K}}\mathcal{{C}}\!\left(y_{{\tau_k+1:\tau_{{k+1}}}}\right)+K\beta,
\qquad
\beta=\lambda\log(n),
\]
on the z-standardized series, where \(\mathcal{{C}}\) is within-segment squared
error. The first term rewards fitting each segment well; the second charges a
fixed price \(\beta\) per break, which is what stops the optimum from putting a
break between every pair of quarters. The central result uses \(\lambda=1\) with
sensitivity at \(0.5\) and \(2.0\), and a break is called stable only if it lies
within one quarter under all three multipliers. PELT answers \emph{{when}} the
level of the series shifted; it never answers \emph{{why}}.

\paragraph{{Stationarity.}} Augmented Dickey--Fuller and KPSS are run on each
segment's series with deliberately opposite nulls,
\[
H_0^{{\mathrm{{ADF}}}}:\ \text{{the series has a unit root (non-stationary)}},
\qquad
H_0^{{\mathrm{{KPSS}}}}:\ \text{{the series is level-stationary}},
\]
so they are read jointly. Rejecting the ADF null while failing to reject the
KPSS null is coherent evidence of stationarity; when the two disagree the
correct report is ambiguity over this short window, not a forced binary label.

\paragraph{{Regime detection (HMM).}} A 3-state Gaussian hidden Markov model is
fit on the quarter-over-quarter \emph{{change}}
\(\Delta N_t=N_t-N_{{t-1}}\) -- not the level -- for the overall cohort and the
two highest-volume CPV segments. The hidden state
\(Z_t\in\{{\text{{decline}},\text{{plateau}},\text{{growth}}\}}\) is governed by
transition probabilities \(P(Z_t=k\mid Z_{{t-1}}=l)\), and the reported quantity
is the filtered posterior
\[
P(Z_t=k \mid \Delta N_1,\dots,\Delta N_t),
\]
read as: given the observed sequence of quarterly changes and the fitted model,
how probable is regime \(k\) in the current quarter? States therefore describe
typical period-over-period direction rather than absolute activity level. This
probability is conditional on the fitted model, not an observed property of the
market: a high posterior on \texttt{{growth}} says the model finds that regime
most consistent with recent changes, not that the market is demonstrably
growing. It complements PELT, which finds discrete historical breaks, with a
continuously updated regime read; the two are not required to agree, and
disagreement is reported honestly rather than reconciled.

\paragraph{{Recent direction.}} The recent direction comes from an ordinary
least-squares fit over the latest 12 quarters,
\[
N_t=\alpha+\beta t+\varepsilon_t,
\]
where \(\hat{{\beta}}\) is the estimated change in awarded episodes per quarter
over that window. A segment is labelled \texttt{{increasing}} or
\texttt{{decreasing}} only when \(\hat{{\beta}}\)'s two-sided \(p\)-value falls
below the pre-declared exploratory \(\alpha=0.10\), uncorrected for multiple
testing; otherwise it is \texttt{{stable\_or\_uncertain}}. \(\hat{{\beta}}\)
describes the last 12 quarters. It is not a forecast, and no value of \(N_t\)
beyond the window is implied. None of these methods forecasts future values or
identifies the cause of a break; causal attribution requires documentary or
stakeholder evidence not available in BOAMP alone. Monetary and duration trend
series are omitted because no canonical awarded-amount field is validated at
episode grain, and 2025's duration-field completeness jump is a measurement
change, not a genuine shift in contract durations.

\section{{Technology Segmentation}}
\label{{sec:nlp-scope}}
The internship guide's L2 deliverable specifies a supervised technology
taxonomy: 300--500 manually annotated contracts across 8--12 classes, a
TF--IDF+logistic-regression/SVM baseline evaluated by macro-F1 and confusion
matrix, and an optional CamemBERT comparison. \textbf{{This is outside the scope
of the reproducible analytical branch reported here.}} A defensible L2 corpus
requires two independent annotators and a Cohen's kappa agreement statistic, and
a second qualified annotator was not available; single-pass AI-assisted labels
whose agreement was called "kappa" would carry exactly the self-consistency
problem this project documents for its own reference labels
(\S\ref{{sec:linkage-caveat}}) rather than avoid it.

A finer technology-classification effort exists in parallel, outside this
pipeline: \pathcode{{data/reference/technology\_classification/}} holds a 945-row
export dated 2026-08-12 carrying a \code{{Domaine}} label per notice. It is read
by no stage of this pipeline. It covers present-day national opportunities rather
than the 2015--2025 Grand Ouest study cohort, and it arrives without the training
corpus, the annotation guidelines, or the validation artifacts that would let its
labels be audited or propagated over that historical cohort. This is a statement
about what can be validated and applied historically here, not a judgement on the
parallel work.

Every technology segment referenced in this report -- cohort selection, Cox
covariates, quarterly trend series -- therefore uses CPV divisions 32
(telecommunications equipment), 35 (security), 48 (software), and 72 (IT
services) as a reproducible substitute. This is a real scope reduction, not a
disguised classifier: CPV divisions are official, zero-missingness EU
categories under Regulation 213/2008, but they are coarser than a learned
taxonomy and cannot distinguish sub-themes such as cloud versus on-premise
infrastructure within a division. Completing L2 would require a second qualified
annotator and a blinded double-annotation design of the kind recorded in
\pathcode{{INDEPENDENT\_LINK\_REVIEW\_PROTOCOL.md}}.

\section{{Pilot Reference Results}}
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
\caption{{Anchor-level metrics on the pilot split, on which the threshold was
frozen. The pilot carries {pilot["positive_anchors"]} reviewed-positive anchors,
so these rates are read for direction only.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{figures/benchmark_dev_method_metrics.png}}
\caption{{The pilot split as conditional probabilities. It is shown for
completeness; the held-out reading is the locked split below.}}
\end{{figure}}

\section{{Locked Reference Results}}
\label{{sec:locked-results}}
The table below is an internal method-comparison diagnostic. It must not be
presented as external validation.

\paragraph{{What is being scored.}} The decision is not a yes/no call on a
pre-formed pair. For each anchor \(i\), the reference names a successor identity
\(R_i\in J_i\cup\{{\varnothing\}}\) and the method returns an identity
\(\hat{{R}}_i\in J_i\cup\{{\varnothing\}}\), so an anchor with a reviewed
successor can still be scored wrong by accepting the \emph{{wrong}} candidate.
That is why the three indicators of \S\ref{{sec:notation}} are needed rather than
a single ``correct/incorrect'' bit:
\[
P_i=\mathbf{{1}}\{{R_i\neq\varnothing\}},\qquad
A_i=\mathbf{{1}}\{{\hat{{R}}_i\neq\varnothing\}},\qquad
C_i=\mathbf{{1}}\{{\hat{{R}}_i=R_i\neq\varnothing\}} .
\]
The familiar counts follow from them: \(TP=\sum_i C_i\); a false positive is
either \(A_i=1, P_i=0\) (accepted where the reference has nothing) or
\(A_i=1, P_i=1, C_i=0\) (accepted the wrong candidate); a false negative is
either an abstention on a positive anchor or that same wrong acceptance. Note
that a wrong acceptance is counted on both sides, which is exactly right and is
invisible in the \(TP/(TP+FP)\) shorthand.

\paragraph{{The four metrics as conditional probabilities.}} Each answers a
different question, distinguished by what is conditioned on:
\[
\underbrace{{P(C_i=1\mid A_i=1)}}_{{\text{{precision}}}},\qquad
\underbrace{{P(C_i=1\mid P_i=1)}}_{{\text{{recall}}}},\qquad
\underbrace{{P(A_i=1\mid P_i=0)}}_{{\text{{false-positive rate}}}},\qquad
\underbrace{{P(A_i=1)}}_{{\text{{coverage}}}} .
\]
Precision and recall are the \emph{{same event under reversed conditioning}}, and
this is the single most useful thing for a reader to hold on to. Precision asks:
given that the method committed to a successor, how often was it the reviewed
one? Recall asks: given that a reviewed successor exists, how often did the
method recover it? \(P(C=1\mid P=1)\) and \(P(C=1\mid A=1)\) are not
interchangeable and need not be close, because the conditioning sets --
{m_b_cells["positive_anchors"]} reviewed-positive anchors and
{m_b_cells["accepted_links"]} accepted links -- are different populations. The
false-positive rate conditions on the \emph{{third}} population, the
{m_b_cells["negative_anchors"]} anchors the reference found nothing for, so it is
not \(1-\text{{precision}}\); the two share no denominator.

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
\caption{{Anchor-level linkage performance on the locked split of the regional
reference. Precision \(P(C=1\mid A=1)\) is the probability that an accepted
successor is exactly the reviewed one; recall \(P(C=1\mid P=1)\) the probability
of recovering the reviewed successor when one exists; FPR \(P(A=1\mid P=0)\) the
probability of accepting anything where the reference found nothing; coverage
\(P(A=1)\) the share of anchors that receive a link rather than an abstention.
Unweighted sample estimates on {m_b_cells["anchors_evaluated"]} anchors.}}
\end{{table}}

\paragraph{{The frozen rule, read cell by cell.}} On the locked split
\code{{M\_B\_text\_ranking @ 0.70}} accepts
{m_b_cells["accepted_links"]} successors across
{m_b_cells["anchors_evaluated"]} anchors, of which
{m_b_cells["true_positive"]} are the reviewed successor and
{m_b_cells["false_positive_wrong_successor"]} is a wrong candidate on an anchor
that does have one. It abstains on {m_b_cells["false_negative_abstained"]}
reviewed-positive anchors and on all
{m_b_cells["true_negative_abstained"]} reviewed-negative anchors. Hence
\[
\hat{{P}}(C=1\mid A=1)=\frac{{{m_b_cells["true_positive"]}}}{{{m_b_cells["accepted_links"]}}}={m_b.precision:.3f},
\qquad
\hat{{P}}(C=1\mid P=1)=\frac{{{m_b_cells["true_positive"]}}}{{{m_b_cells["positive_anchors"]}}}={m_b.recall:.3f},
\]
\[
\hat{{P}}(A=1\mid P=0)=\frac{{{m_b_cells["false_positive_on_no_successor_anchor"]}}}{{{m_b_cells["negative_anchors"]}}}={m_b.fpr:.3f},
\qquad
\hat{{P}}(A=1)=\frac{{{m_b_cells["accepted_links"]}}}{{{m_b_cells["anchors_evaluated"]}}}={m_b.coverage:.3f} .
\]
Read them as follows. Among the {m_b_cells["accepted_links"]} links the method
committed to, {m_b_cells["true_positive"]} matched the reviewed successor. Of
the {m_b_cells["positive_anchors"]} anchors the reference says have a successor,
the method recovered {m_b_cells["true_positive"]}; the
{m_b_cells["positive_anchors"] - m_b_cells["true_positive"]} it did not are
{m_b_cells["false_negative_abstained"]} abstentions and
{m_b_cells["false_positive_wrong_successor"]} wrong acceptance, and both count
identically against recall even though only the second is a fabricated link.

Four caveats travel with these four numbers and none of them is optional.
The precision estimate rests on {m_b_cells["accepted_links"]} accepted links, so
its 95\% interval is [{m_b.precision_low:.3f}, {m_b.precision_high:.3f}]: one
changed decision moves it materially. Recall's ceiling is not 1 but
{ceiling["candidate_generation_recall_ceiling"]:.3f}, because
\(P(C=1\mid P=1)\le P(E=1\mid P=1)\) -- a successor that blocking discarded
cannot be recovered by any scorer. An FPR of {m_b.fpr:.3f} is a diagnostic on
{m_b_cells["negative_anchors"]} corpus-relative negatives on this small sample,
not evidence that the population false-positive rate is zero. And coverage is
deliberately low: it is a consequence of prioritising \(P(C=1\mid A=1)\), not a
target to raise, because in a survival dataset a false link fabricates both an
event and an event time whereas an abstention is handled honestly as
right-censoring.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{figures/benchmark_validation_method_metrics.png}}
\caption{{The same three conditional probabilities per method. \(M_C\) buys
recall \(P(C=1\mid P=1)\) with a materially higher \(P(A=1\mid P=0)\); \(M_B\)
takes the opposite trade, which is the one this study needs.}}
\end{{figure}}

\section{{Quality Evidence And Interpretation}}
The held-out result supports retaining a conservative baseline for continued
work; it does not establish final accuracy. Within this reference the direction
is coherent, with \code{{M\_B}} best on \(P(C=1\mid A=1)\) and lowest on
\(P(A=1\mid P=0)\) among useful methods and \code{{M\_C}} showing the expected
trade-off, but the intervals in \S\ref{{sec:locked-results}} overlap heavily and
must be read before separating any two methods.

The pair-level ROC and precision-recall curves should be read as score-ranking
diagnostics, not as the final operating decision. The final pipeline decision is
anchor-level: choose one successor or abstain. The ROC curve is stepped rather
than smooth because the held-out reference contains a finite number of
labelled pairs and many tied or near-tied scores.

The emphasis on precision-recall for rare positive decisions is supported by
\href{{https://doi.org/10.1145/1143844.1143874}}{{Davis and Goadrich (2006)}}
and
\href{{https://doi.org/10.1371/journal.pone.0118432}}{{Saito and Rehmsmeier
(2015)}}. These sources justify the diagnostic choice; they do not validate the
reference labels, numerical results, or selected threshold. Generic web
illustrations are therefore treated as presentation aids only, not as academic
evidence.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.99\textwidth]{{figures/benchmark_validation_m_b_threshold_tradeoff.png}}
\caption{{Project-specific, unsmoothed anchor-level threshold trade-off for
\code{{M\_B}}. The figure is empirical evidence from the locked split of the
regional reference, not an idealised illustration.}}
\end{{figure}}

On the locked split, $0.70$ accepts {int(locked_70.accepted_links)} links of
which {int(locked_70.true_positive)} are correct (precision
{locked_70.precision:.3f}, recall {locked_70.recall:.3f}, FPR
{locked_70.false_positive_rate:.3f}), while $0.60$ accepts
{int(locked_60.accepted_links)} of which {int(locked_60.true_positive)} are
correct (precision {locked_60.precision:.3f}, recall {locked_60.recall:.3f},
FPR {locked_60.false_positive_rate:.3f}). On the pilot split the same two points
give precision {pilot_70.precision:.3f} and {pilot_60.precision:.3f}
respectively. The threshold was fixed before this reference was read, which is
the only reason the locked split can be reported as held out; selecting one from
these rows now would convert it into a tuning set. The lower threshold is therefore
reported as a sensitivity arm rather than promoted after inspecting validation.

\section{{Model-Assisted Challenge Review}}
A separate blinded challenge review sampled 20 accepted links, 20
high-similarity structural-negative candidates, and 20 buyer-declared
relationships. Its provenance is model-assisted rather than independent human
specialist review. Among the 20 accepted links,
{review_audit["primary_accepted_links"]["review_counts"]["Y"]} were confirmed,
{review_audit["primary_accepted_links"]["review_counts"]["N"]} were rejected,
and {review_audit["primary_accepted_links"]["review_counts"]["UNCERTAIN"]} was
uncertain. Conservative reviewed precision was
{review_audit["primary_accepted_links"]["precision_conservative"]["estimate"]:.3f},
with exact 95\% interval
[{review_audit["primary_accepted_links"]["precision_conservative"]["ci_95"][0]:.3f},
{review_audit["primary_accepted_links"]["precision_conservative"]["ci_95"][1]:.3f}].
This diagnostic falls below the 0.80 point target and exposes useful failure
modes. It does not establish independent human validation, but it is sufficient
to justify keeping the reported claim narrow and the lower threshold unpromoted.

\section{{Modeling Tables}}
The modeling-ready tables include strict, primary, broad, and non-match target
columns, plus observable features. They include the Fellegi--Sunter columns
\code{{fs\_match\_weight}} and \code{{fs\_match\_probability}}, so modeling and
evaluation use the same feature state.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\textwidth]{{figures/benchmark_modeling_counts.png}}
\caption{{Row and label counts behind each modelling table, so a reader can
see how thin the positive support is before reading any metric computed on it.}}
\end{{figure}}

\section{{Survival Results}}
The survival dataset uses the main accepted links as events and all other
episodes as right-censored observations at the study cutoff. The main variant
contains {survival_main["validation"]["events"]:,} events and
{survival_main["validation"]["censored"]:,} censored observations
({survival_summary["cohort"]["event_rate"] * 100:.3f}\% event rate). Segment
event rates are {survival_main["description"]["by_digital_segment"]["CPV-32"]["event_rate"]:.3f}
for CPV-32, {survival_main["description"]["by_digital_segment"]["CPV-35"]["event_rate"]:.3f}
for CPV-35, {survival_main["description"]["by_digital_segment"]["CPV-48"]["event_rate"]:.3f}
for CPV-48, and {survival_main["description"]["by_digital_segment"]["CPV-72"]["event_rate"]:.3f}
for CPV-72. CPV-35 has the highest observed successor rate in the main arm.

\paragraph{{Kaplan--Meier.}} The cumulative event probability
\(F(t)=1-S(t)\) is estimated at
\[
\hat{{F}}(12)={km_horizons.loc[12, 'cumulative_successor_probability']:.5f},
\qquad
\hat{{F}}(24)={km_horizons.loc[24, 'cumulative_successor_probability']:.5f},
\qquad
\hat{{F}}(60)={km_horizons.loc[60, 'cumulative_successor_probability']:.5f},
\]
that is {km_horizons.loc[12, 'cumulative_successor_probability'] * 100:.3f}\% by
12 months, {km_horizons.loc[24, 'cumulative_successor_probability'] * 100:.3f}\%
by 24 months and
{km_horizons.loc[60, 'cumulative_successor_probability'] * 100:.3f}\% by 60
months. These are linkage-conditioned probabilities that an observable successor
procurement becomes visible in BOAMP for an episode of the
{survival_main["validation"]["rows"]:,}-episode study cohort under
\(M_B@0.70\). They are not certified renewal probabilities, and because missed
successors push the level down while residual false links push it up, they are
not one-sided bounds either.

The Kaplan--Meier median survival time is
\textbf{{{survival_summary["km"]["median_status"].replace('_', ' ')}}}: \(\hat{{S}}(t)\)
never falls below 0.5 within the observation window, so no median is reported.
This is distinct from the {survival_main["description"]["median_time_to_successor_months"]:.2f}-month
median delay \emph{{among linked events only}}, which conditions on the event
having occurred. A multivariate log-rank test of
\(H_0: S_{{32}}=S_{{35}}=S_{{48}}=S_{{72}}\) across CPV segments gives statistic
{survival_summary["logrank"]["test_statistic"]:.2f}
(\({latex_pvalue(survival_summary["logrank"]["p_value"])}\)), so the data reject
equality of the four segment curves. Being an omnibus test, it does not
establish that every pair of segments differs.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.99\textwidth]{{figures/survival_kaplan_meier.png}}
\caption{{Estimated \(\hat{{S}}(t)=\hat{{P}}(T>t)\), the probability that an
episode is still without an observable successor at age \(t\). The left panel
carries the absolute level, which is linkage-conditioned and quoted only with
that caveat; the right panel carries the segment ordering, which survives every
sensitivity arm.}}
\end{{figure}}

\paragraph{{Cox model.}} The parsimonious model
({survival_summary["cox"]["covariates"]} covariates,
{survival_summary["cox"]["events"]:,} events) gives:
\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{lrlr}}
\toprule
Covariate & HR & 95\% CI & p \\
\midrule
{latex_cox_rows(cox_results)}
\bottomrule
\end{{tabularx}}
\caption{{Cox hazard ratios \(\mathrm{{HR}}_k=\exp(\beta_k)\) for the
observable-successor hazard, main linkage arm. \(\mathrm{{HR}}_k>1\) means the
covariate is associated with a successor appearing sooner. In-sample C-index
{survival_summary["cox"]["in_sample_c_index"]:.3f}.}}
\end{{table}}
Framework-agreement episodes and CPV-35 carry the largest hazard ratios among
substantively interpretable covariates. The framework coefficient reads:
conditional on an episode still having no observable successor at time \(t\),
and holding the other included covariates fixed, framework episodes have an
estimated hazard {cox_results.loc[cox_results["covariate"].eq("framework_flag"), "exp(coef)"].iloc[0]:.3f}
times that of non-framework episodes -- roughly
{(cox_results.loc[cox_results["covariate"].eq("framework_flag"), "exp(coef)"].iloc[0] - 1) * 100:.1f}\%
higher. CPV-35 reads the same way at
{cox_results.loc[cox_results["covariate"].eq("digital_segment_CPV-35"), "exp(coef)"].iloc[0]:.3f}
against the CPV-32 reference segment. Three qualifications apply to both. These
are descriptive associations with the observed hazard, not causal effects. A
hazard ratio is not a risk ratio, so neither number says an episode is
{(cox_results.loc[cox_results["covariate"].eq("framework_flag"), "exp(coef)"].iloc[0] - 1) * 100:.1f}\%
more likely to be renewed. And the proportional-hazards assumption is rejected
for \code{{framework\_flag}} below, so its ratio is a time-averaged association
rather than a constant multiplier.

\paragraph{{Proportional-hazards diagnostic.}}
\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{lrrl}}
\toprule
Covariate & Test statistic & p & PH assumption \\
\midrule
{latex_ph_rows(ph_diagnostics)}
\bottomrule
\end{{tabularx}}
\caption{{Schoenfeld-residual tests of \(H_0\): the hazard ratio for this
covariate is constant over time. A small \(p\) is evidence against
proportionality, not evidence that the covariate does not matter.}}
\end{{table}}
The assumption is rejected (\(p<0.05\)) for
\code{{{latex_escape(', '.join(ph_violations))}}}. These coefficients are
therefore reported as time-averaged associations rather than constant hazard
ratios; stratification or time-interaction terms would be the next refinement
if individualized prediction were required. Nothing in the operational
deliverable depends on proportionality, because the 12/24-month probabilities
come from the Kaplan--Meier estimator, which makes no such assumption.

\paragraph{{Temporal validation.}} Training on
{temporal["train_years"]} ({temporal["train_contracts"]:,} episodes,
{temporal["train_events"]:,} events) and evaluating on the guideline-aligned
{temporal["test_years"]} window ({temporal["test_contracts"]:,} episodes,
{temporal["test_events"]:,} events) gives C-index
{temporal["train_c_index"]:.3f} in-sample versus
{temporal["test_c_index"]:.3f} out-of-time. Extending the test window to
{extended["test_years"]} ({extended["test_contracts"]:,} episodes,
{extended["test_events"]:,} events), without refitting, gives
{extended["test_c_index"]:.3f}. Read these as pairwise ranking probabilities:
among comparable pairs of episodes drawn from the test window, the model puts
the one that gets its successor first at higher risk about
{temporal["test_c_index"] * 100:.1f}\% of the time, where 50\% is chance. This is
\emph{{not}} a classification accuracy of
{temporal["test_c_index"] * 100:.1f}\%; no episode is being classified. Both
figures sit close to the \(0.5\) chance line, so individualized out-of-time
discrimination is weak. The two windows also differ in
follow-up, and \href{{https://doi.org/10.1002/sim.4154}}{{Uno et al. (2011)}} show
that the concordance statistic for right-censored data converges to a quantity
depending on the censoring distribution, so the gap between the two figures should
not be read as a change in discriminative ability. That is a result, not a
prompt to retune -- the model is retained as a descriptive risk-factor summary
and nothing in the operational deliverable rests on it. Part of the gap is
structural, because episodes awarded from 2022 onwards can contribute only
short-gap events.

\paragraph{{Borderline-link robustness.}} Dropping the
{borderline["contracts_removed"]:,} episodes whose best candidate scores within
\(\pm0.05\) of the acceptance threshold removes
{borderline["events_removed"]:,} events and leaves the direction of both headline
hazard ratios unchanged (CPV-35 {borderline["main"]["cox_hr_cpv_35"]:.2f} to
{borderline["excluding_borderline_links"]["cox_hr_cpv_35"]:.2f}; framework
{borderline["main"]["cox_hr_framework"]:.2f} to
{borderline["excluding_borderline_links"]["cox_hr_framework"]:.2f}). The absolute
Kaplan--Meier level falls, from
{borderline["main"]["km_successor_by_12m"] * 100:.1f}\% to
{borderline["excluding_borderline_links"]["km_successor_by_12m"] * 100:.1f}\% at 12 months,
which is the mechanical consequence of removing borderline events. The
comparative conclusions therefore do not rest on borderline linkage decisions,
while absolute probabilities remain threshold-uncertain.

\paragraph{{Parametric models.}}
\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{lrrrr}}
\toprule
Model & Parameters & Log-likelihood & AIC & BIC \\
\midrule
{latex_parametric_rows(parametric_comparison)}
\bottomrule
\end{{tabularx}}
\caption{{Parametric survival model comparison by \(\mathrm{{AIC}}=2k-2\ell\) and
\(\mathrm{{BIC}}=k\log n-2\ell\). Lower is better \emph{{among these five
families}}; neither criterion establishes that the winner fits the data.}}
\end{{table}}
\code{{{survival_summary["parametric"]["selected_model"]}}} has the lowest AIC
and BIC and was checked graphically against the Kaplan--Meier curve. It is
reported as the best-fitting family and as the instrument any extrapolation past
the 2025-12-31 cutoff would use. It is \emph{{not}} the source of the exported
12/24-month conditional successor probabilities
(\pathcode{{survival\_conditional\_probabilities.csv}}): those come from the
Kaplan--Meier estimator, each with a 500-draw episode-bootstrap interval. The
reason is visible in the fit itself -- around age 36 months, just before the
observed renewal shoulder, every parametric family including the selected one
reports roughly 0.02--0.03 for the next twelve months where Kaplan--Meier reports
near 0.075. Since every horizon this report quotes falls inside the observed
window, the extrapolation that would justify accepting that smoothing is not
needed. These are estimated probabilities of an \emph{{identifiable observable
successor procurement}}, not certified renewal probabilities, and no active
Gigalis portfolio was available to score, so they cover the study cohort rather
than a live deployment set.

\paragraph{{Operational 12- and 24-month probabilities.}} This is the quantity
the business question actually asks for, and it is the clearest mathematical
statement of the internship's operational problem. For an episode that has
reached age \(a\) months with no accepted observable successor, the probability
that one becomes visible within the next \(h\) months is
\[
P(T\le a+h \mid T>a)
= P(a<T\le a+h \mid T>a)
= 1-\frac{{S(a+h)}}{{S(a)}} ,
\]
the two left-hand forms being identical because the conditioning event
\(\{{T>a\}}\) already excludes \(T\le a\). Here \(a\) is the episode's current
age, \(h\) the forward horizon, and \(\{{T>a\}}\) the statement that no
observable successor has appeared yet. The quantity is read off the
Kaplan--Meier estimator with the {int(conditional_probabilities["interval_method"].iloc[0].split()[-2])}-draw
episode-bootstrap intervals described in \S\ref{{sec:survival-methods}}.
\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{lrlrl}}
\toprule
Age \(a\) at assessment & \(P(T\le a{{+}}12\mid T{{>}}a)\) & 95\% CI & \(P(T\le a{{+}}24\mid T{{>}}a)\) & 95\% CI \\
\midrule
{latex_conditional_rows(conditional_probabilities)}
\bottomrule
\end{{tabularx}}
\caption{{Given that an episode has reached age \(a\) with no accepted
observable successor, the estimated probability that one becomes visible in the
following 12 or 24 months. Kaplan--Meier on the
{survival_main["validation"]["rows"]:,}-episode cohort under \(M_B@0.70\).}}
\end{{table}}
Worked example, the row a purchasing body would use most. At
\(a=36\) months and \(h=12\),
\[
\hat{{P}}(T\le 48\mid T>36)={float(conditional_probabilities.loc[(conditional_probabilities["contract_age_months"] == 36) & (conditional_probabilities["horizon_months"] == 12), "probability"].iloc[0]):.4f},
\qquad
\text{{95\% CI }}[{float(conditional_probabilities.loc[(conditional_probabilities["contract_age_months"] == 36) & (conditional_probabilities["horizon_months"] == 12), "ci_95_low"].iloc[0]):.4f},\,
{float(conditional_probabilities.loc[(conditional_probabilities["contract_age_months"] == 36) & (conditional_probabilities["horizon_months"] == 12), "ci_95_high"].iloc[0]):.4f}].
\]
In words: conditional on an episode reaching 36 months with no accepted
observable successor, the estimated probability that one becomes visible in the
following 12 months is
{float(conditional_probabilities.loc[(conditional_probabilities["contract_age_months"] == 36) & (conditional_probabilities["horizon_months"] == 12), "probability"].iloc[0]) * 100:.2f}\%.
\begin{{figure}}[H]
\centering
\includegraphics[width=0.94\textwidth]{{figures/survival_conditional_probabilities.png}}
\caption{{The same quantities with their bootstrap intervals. The profile is not
monotone in \(a\): it rises into the 36--48 month renewal shoulder and falls
away after it, which is the empirical feature every parametric family smooths
out.}}
\end{{figure}}
The intervals are wide relative to the estimates, and these numbers rank ages
and segments rather than calibrating individual forecasts. They estimate an
observable successor procurement appearing in BOAMP, not a certified renewal,
and they are conditional on the linkage rule.

\paragraph{{Template-risk robustness.}} The four linkage arms and the borderline
band both perturb where the acceptance bar sits. Neither addresses the
false-positive mechanism identified above, because that mechanism produces links
well \emph{{above}} the bar: shared framework boilerplate can drive the character
analyser high between unrelated objects, and because \(M_B\) ranks candidates
within each anchor independently, one such episode can be accepted for several
anchors. A stricter threshold does not remove either signature and can enrich for
them. Two observable signatures, both published by the candidate-generation
audit, define a risk flag on each accepted link:
\[
Q_i=\mathbf{{1}}\{{\text{{word-level similarity}}<{template_risk["carried_by_char_threshold"]:.2f}
\ \ \text{{or}}\ \ \text{{successor reused across anchors}}\}} ,
\]
carrying {template_risk["carried_by_char_similarity"]} and
{template_risk["successor_shared_with_another_anchor"]} links respectively and
{template_risk["flagged_links"]} in union, so that
\[
\hat{{P}}(Q=1\mid A=1)=\frac{{{template_risk["flagged_links"]}}}{{{template_risk["accepted_links"]}}}
={template_risk["flagged_share_of_events"]:.3f} .
\]
\textbf{{This is the share of accepted production links carrying a conservative
risk signature. It is emphatically not
\(P(\text{{false link}}\mid A=1)={template_risk["flagged_share_of_events"]:.3f}\).}}
A flag is a reason to stress-test a link, not a finding that it is wrong;
inspection says most flagged links are legitimate rewordings and multi-lot
programmes. Those anchors are re-censored at the cutoff rather than dropped,
because if a link were spurious the anchor had no observed successor and should
still contribute its full follow-up as censored exposure.
\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{Xrrrrrr}}
\toprule
Analysis & Episodes & Events & KM 12m & KM 24m & CPV-35 HR & Framework HR \\
\midrule
Main & {template_main["contracts"]:,} & {template_main["events"]} & {template_main["km_successor_by_12m"] * 100:.2f}\% & {template_main["km_successor_by_24m"] * 100:.2f}\% & {template_main["cox_hr_cpv_35"]:.3f} & {template_main["cox_hr_framework"]:.3f} \\
Re-censoring template risk & {template_kept["contracts"]:,} & {template_kept["events"]} & {template_kept["km_successor_by_12m"] * 100:.2f}\% & {template_kept["km_successor_by_24m"] * 100:.2f}\% & {template_kept["cox_hr_cpv_35"]:.3f} & {template_kept["cox_hr_framework"]:.3f} \\
\bottomrule
\end{{tabularx}}
\caption{{Headline results with the \(Q_i=1\) links re-censored. The absolute
event level moves strongly; the relative associations barely move. That
contrast, not either row on its own, is the finding.}}
\end{{table}}
This is the check the framework-agreement finding most needs, since framework
boilerplate is the text driving the mechanism: were the higher framework hazard
an artefact of shared legal wording, re-censoring would collapse it. It does
not. Events fall from {template_main["events"]} to {template_kept["events"]} and
\(\hat{{F}}(12)\) from {template_main["km_successor_by_12m"] * 100:.2f}\% to
{template_kept["km_successor_by_12m"] * 100:.2f}\%, while CPV-35 moves
{template_main["cox_hr_cpv_35"]:.3f} to {template_kept["cox_hr_cpv_35"]:.3f} and
framework {template_main["cox_hr_framework"]:.3f} to
{template_kept["cox_hr_framework"]:.3f}. Both hazard ratios keep their side of 1,
so the comparative findings are not products of the documented false-positive
mechanism. The fall in the absolute Kaplan--Meier level is roughly the share of
events re-censored, which is arithmetic rather than evidence. The check
\emph{{bounds}} how much the mechanism could be moving the results; it does not
establish that the flagged links are false, and most of them are not.

\paragraph{{Detectability and selection diagnostic.}} If links were found mainly
where records happen to be well documented, the event set would be a biased
sample and the survival estimates would inherit that bias. Linked
(\(Y_i=1\)) and censored (\(Y_i=0\)) episodes are therefore compared on the
standardized mean difference
\[
\mathrm{{SMD}}=\frac{{\bar{{X}}_1-\bar{{X}}_0}}{{\sqrt{{(s_1^2+s_0^2)/2}}}},
\]
which measures the gap between the two groups in pooled within-group standard
deviations and so is comparable across variables on different scales. The
largest observed gap is
\code{{{latex_escape(survival_summary["selection_diagnostic"]["largest_absolute_smd"]["variable"])}}}
at {survival_summary["selection_diagnostic"]["largest_absolute_smd"]["absolute_smd"]:.3f}.
This indicates possible differential detectability -- longer notice text gives
the text scorer more to match on -- not proof of causal linkage bias, and it
cannot be fully separated from genuine heterogeneity in renewal behaviour using
BOAMP alone.

\paragraph{{Linkage sensitivity.}} Event counts range from
{survival_summary["sensitivity"]["minimum_events"]:,} to
{survival_summary["sensitivity"]["maximum_events"]:,} across the four retained
linkage arms, so \(P_m(T\le t)\) is linkage-sensitive by construction: a looser
\(\mathcal{{L}}_m\) manufactures more events. The question the table below
answers is whether \(\mathrm{{HR}}_{{k,m}}\) is equally sensitive.
\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{lrrrrl}}
\toprule
Covariate & Strict & Main & Looser & High-recall & Robustness \\
\midrule
{latex_cox_sensitivity_rows(cox_sensitivity)}
\bottomrule
\end{{tabularx}}
\caption{{\(\mathrm{{HR}}_{{k,m}}\) for each covariate \(k\) under each linkage
definition \(\mathcal{{L}}_m\). Reading a row across is the robustness test: a
covariate whose ratio keeps its side of 1 across a fourfold change in event
count is not an artefact of where the acceptance bar was placed.}}
\end{{table}}
Framework flag, CPV-35, and centered award year are the most robust
associations; buyer-region and CPV-72 effects are linkage-sensitive and should
not be over-interpreted. The overall pattern is the one the notation was
introduced to name: \(P_m(T\le t)\) varies considerably with \(m\), while the
direction and rough size of the leading \(\mathrm{{HR}}_{{k,m}}\) do not.

\section{{Trend Results}}
\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{lrrrll}}
\toprule
Segment & Direction & Slope/qtr & p & Last stable PELT break & HMM regime \\
\midrule
{latex_trend_signal_rows(trend_signal_matrix)}
\bottomrule
\end{{tabularx}}
\caption{{Trend signal matrix: OLS 12-quarter slope, PELT breaks, and
HMM current regime (Overall and top-2 segments only).}}
\end{{table}}
CPV-48 is the only segment with a statistically distinguishable 12-quarter
decline at the exploratory \(\alpha=0.10\) level; the rest are
\code{{stable\_or\_uncertain}} by this signal.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.99\textwidth]{{figures/trend_quarterly_episode_counts.png}}
\caption{{Quarterly awarded digital procurement episodes, 2015Q2--2025Q4. Dashed
lines mark PELT breaks that survive all three penalty multipliers. A break dates
a level shift; it does not explain one.}}
\end{{figure}}

\paragraph{{Operational reading.}} Each segment's signals translate into a
monitoring action, carried in the \code{{business\_recommendation}} column of
\pathcode{{trend\_signal\_matrix.csv}} so that this text and the materialised
table cannot drift apart. These are readings of descriptive evidence, not
forecasts and not causal explanations: a PELT break dates a level shift without
explaining it, and none of the statements below should be quoted as attributing a
shift to policy, COVID, regulation, or technology.
\begin{{itemize}}
{latex_trend_recommendation_items(trend_signal_matrix)}
\end{{itemize}}

\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{lXXl}}
\toprule
Segment & ADF (H0: unit root) & KPSS (H0: level stationary) & Note \\
\midrule
{latex_stationarity_rows(trend_summary["stationarity"])}
\bottomrule
\end{{tabularx}}
\caption{{Stationarity diagnostics per segment.}}
\end{{table}}
ADF and KPSS test opposite null hypotheses and are read jointly; disagreement
between them means the series is not cleanly classified as stationary or
non-stationary over this 43-quarter window.

The HMM regime column above reports each covered segment's current-quarter
regime and posterior probability, fit on the quarter-over-quarter change in
episode count rather than its level. Because the model is fit on noisy,
low-count quarterly data, the \code{{plateau}} state is a data-driven middle
tier rather than a change centered exactly at zero, and it is read alongside
\code{{decline}}/\code{{growth}} rather than as "no change." The HMM's
current-quarter regime and the 12-quarter OLS slope are complementary
diagnostics computed over different windows; they are not required to agree,
and this report does not adjust either to match the other.

\section{{Final Linkage Specification}}
The final event definition is \code{{M\_B\_text\_ranking @ 0.70}}, a frozen
conservative observable-successor baseline, with the stricter
(\(M_B@0.80\)), looser (\(M_B@0.60\)), and weighted-gated (\(M_C@0.70\)) variants
as the required sensitivity analyses, together with the borderline-band check.
The threshold is not claimed to be optimal; it is claimed to be
pre-specified. It was fixed before the regional reference was consulted and has
not been moved since, so the locked-split figures are held out rather than
fitted. Moreover, the completed
model-assisted review confirmed only 14 of 20 sampled production links at
$0.70$ conservatively, so lowering an unreviewed threshold would not support a
stronger accuracy claim. This is a precision-first design. Low recall is
accepted as an explicit trade-off, but the
observed event rate is not a mathematical lower bound: missed successors push it
downward while residual false links can push it upward.
\href{{https://doi.org/10.1093/ije/dyz203}}{{Doidge and Harron (2019)}} make the
general point that missed and false links misclassify in opposite directions, which
is why no one-sided bound follows from a linkage-conditioned event rate. It is therefore a
linkage-conditioned indicator whose absolute value must be read with the strict,
looser, and weighted-gated sensitivity results and the borderline-band check.
Independent specialist review is required only before claiming externally
validated precision or promoting a new threshold, not to complete this
linkage-conditioned descriptive study.
If that review shows weaker precision, the claim should be narrowed further
rather than forcing a more complex model.

\paragraph{{Segment continuity of the accepted links.}}
Because hard same-CPV blocking is not imposed, the accepted links are checked
after the fact for segment drift. Of the {cpv_continuity["accepted_links"]}
accepted primary links,
{cpv_continuity["accepted_links_with_both_divisions_observed"]} have a CPV
division observed on both sides;
{cpv_continuity["same_cpv2_count"]}
({cpv_continuity["same_cpv2_share"] * 100:.1f}\%) stay inside one division and
{cpv_continuity["cross_cpv2_count"]}
({cpv_continuity["cross_cpv2_share"] * 100:.1f}\%) cross divisions, while
{cpv_continuity["shares_any_division_share"] * 100:.1f}\% share at least one division
across their full CPV lists. The comparison that interprets this is the reviewed
reference, which crosses divisions at
{cpv_continuity["reviewed_cross_cpv2_share"] * 100:.1f}\% -- close to the
{cpv_continuity["cross_cpv2_share"] * 100:.1f}\% observed among accepted links. Relaxed
CPV blocking therefore reproduces roughly the segment-continuity behaviour the
reviewed successors already display, rather than introducing drift of its own.
The largest cross-division flows are the goods-to-services pairs named above.
Full transition counts are in
\code{{data/processed/boamp/candidate\_generation\_cpv\_transitions.csv}}.

Inspecting the cross-division links by hand also identifies the dominant
false-positive mechanism, which is textual rather than categorical. French
award notices carry long standardised framework-agreement boilerplate --
\emph{{accord-cadre mixte}}, \emph{{bons de commande}}, \emph{{bordereau des prix
unitaires}}, \emph{{marchés subséquents}} -- and for buyers whose notices are
dominated by that template, character-level similarity can clear the threshold on
shared legal phrasing alone.
{cpv_continuity["accepted_links_carried_by_char_similarity"]} accepted links
carry word-level similarity below
{cpv_continuity["low_word_similarity_threshold"]:.2f} and so rest on the
character analyser; inspection shows most are legitimate rewordings of the same
object, but this is also the band in which the failure mode lives. Because
\(M_B\) ranks candidates within each anchor independently, with no one-to-one
constraint, one episode can be accepted for several anchors: the most reused
successor here is accepted by
{cpv_continuity["max_anchors_per_successor"]} anchors, and
{cpv_continuity["successors_accepted_by_multiple_anchors"]} of the
{cpv_continuity["distinct_successors"]} distinct successors serve more than one.
Multi-lot programmes make much of that legitimate, but the clearest spurious
instance found by inspection is a single catering framework accepted as the
successor of nine unrelated digital anchors from one regional authority, whose
notices are almost entirely template text.
A hard same-CPV block would suppress that particular case but would not fix the
mechanism, which also operates within a division, and would cost the
{cpv_continuity["reviewed_successors_lost_to_hard_same_division_block"]} genuine
reviewed successors quantified above. The mechanism is therefore documented as a
known contributor to the measured imprecision rather than patched, and it is one
reason the absolute event level is reported as linkage-sensitive.

\section{{Limitations And Robustness}}
\begin{{itemize}}
\item BOAMP does not consistently encode legal renewal status; accepted links are
observable successor procurements, not legal renewal proof.
\item The reference labels are a spot-checked LLM research pass, not
anchor-by-anchor verification and not an independent specialist panel, and its
negatives are corpus-relative, so the reported false-positive rate is a
sample diagnostic rather than a population-wide rate
(\S\ref{{sec:linkage-caveat}}).
\item The held-out reference is small: {locked["positive_anchors"]}
positive anchors and {int(m_b.accepted_links)} accepted \code{{M\_B}} links.
The selected threshold should not be over-tuned and its accuracy remains provisional.
\item Buyer standardisation is improved but remains an important risk area. The
legal-form audit is retained to catch name-only and cross-legal-form cases.
\item Missing duration and expiry fields are not imputed. This avoids creating
false timing certainty.
\item The 3-state HMM regime labels are fit on noisy, short (43-quarter) count
series; state-count and window choices are not exhaustively validated, and a
`plateau` label is a relative middle tier rather than a change centered at
zero.
\item Technology segmentation uses CPV divisions, not a learned taxonomy
(\S\ref{{sec:nlp-scope}}); this is a real scope reduction, disclosed rather
than concealed.
\end{{itemize}}

\section{{Perspectives}}
\subsection{{Causal Inference (Outline Only)}}
\label{{sec:causal-inference}}
The internship guide frames a distinct causal question, separate from this
report's descriptive linkage/survival/trend results: does a Gigalis pooled
framework agreement actually change member purchasing behaviour (amounts,
frequency, segments), or does Gigalis simply capture demand that would have
existed anyway? \textbf{{This question is not answered here, and no causal
estimate is computed in this repository.}} Answering it requires data this
project's BOAMP-only corpus does not contain: which buyers are Gigalis
members, and when each joined. Attempting a substitute causal claim from
BOAMP alone -- for example treating CPV-segment membership or region as a
proxy treatment -- would confound procurement domain and geography with
Gigalis adoption, and this project has consistently avoided exactly this kind
of unsupported inferential leap elsewhere (survival hazard ratios are reported
as associations, not effects, for the same reason).

If Gigalis-internal membership and adoption-date data were joined to this
corpus, the natural identification strategy is a \textbf{{staggered-adoption
difference-in-differences}} design
(\href{{https://doi.org/10.1016/j.jeconom.2020.12.001}}{{Callaway and
Sant'Anna, 2021}}), because members join the central purchasing body at
different calendar dates rather than simultaneously -- the specific setting
that motivates that estimator over classical two-period DiD. The parallel-trends
assumption (comparable purchasing trajectories absent Gigalis membership)
would need to be assessed with pre-adoption trend plots per member cohort.
Propensity-score matching
(\href{{https://www.jstor.org/stable/j.ctvcm4j72}}{{Angrist and Pischke,
2009}}) is a secondary candidate if adoption is not sharply timed but depends
on observable buyer characteristics (size, budget, sector); regression
discontinuity would apply only if a hard eligibility threshold (e.g. a
population cutoff) determines Gigalis access, which is not currently known to
be the case. Any of these designs should state its identifying assumptions as
an explicit causal graph
(\href{{https://www.wiley.com/en-us/Causal+Inference+in+Statistics\%3A+A+Primer-p-9781119186847}}{{Pearl,
Glymour and Jewell, 2016}}) before estimation, not after.

This section is a design outline for future work conditional on Gigalis
supplying membership data, not a promise that such data will become available
within this internship.

\section{{Interpretation And Scope Of Claims}}
The technical report should defend the project as a reproducible measurement
pipeline, not as a black-box prediction system. The strongest statement is:
\begin{{quote}}
Because legal renewal status is not directly observed in BOAMP, the study
estimates observable successor procurements. Candidate generation restricts
comparison to the same buyer and a plausible future interval; the selected
method then accepts only the strongest textually similar candidate above a fixed
threshold. This prioritises precision over recall, which is appropriate because
false positive links would create artificial survival events.
\end{{quote}}

\section{{Reproducibility Artifacts}}
\begin{{itemize}}
\item \pathcode{{data/processed/boamp/linkage\_evaluation\_dev.json}}
\item \pathcode{{data/processed/boamp/linkage\_evaluation\_validation.json}}
\item \pathcode{{data/processed/boamp/regional\_benchmark/modeling/modeling\_summary.json}}
\item \pathcode{{data/processed/boamp/regional\_benchmark/regional\_benchmark\_manifest.json}}
\item \pathcode{{data/reference/regional\_link\_benchmark/}}
\item \pathcode{{data/processed/boamp/survival\_dataset\_summary.json}}
\item \pathcode{{data/processed/boamp/linkage\_candidates\_summary.json}}
\item \pathcode{{data/processed/boamp/survival\_analysis\_summary.json}}
\item \pathcode{{data/processed/boamp/survival\_cox\_results.csv}}
\item \pathcode{{data/processed/boamp/survival\_ph\_diagnostics.csv}}
\item \pathcode{{data/processed/boamp/survival\_parametric\_comparison.csv}}
\item \pathcode{{data/processed/boamp/survival\_cox\_linkage\_sensitivity.csv}}
\item \pathcode{{data/processed/boamp/trend\_analysis\_summary.json}}
\item \pathcode{{data/processed/boamp/trend\_signal\_matrix.csv}}
\item \pathcode{{DATA\_QUALITY\_REPORT.md}}
\item \pathcode{{TREND\_ANALYSIS\_REPORT.md}}
\item \pathcode{{SURVIVAL\_ANALYSIS\_REPORT.md}}
\item \pathcode{{PROJECT\_WORK\_PROTOCOL.md}}
\item \pathcode{{REVIEW\_AUDIT\_RESULTS.md}}
\item \pathcode{{EXECUTIVE\_SUMMARY.md}}
\end{{itemize}}

\section{{Methodological References}}
The source and identity definitions follow the official BOAMP and INSEE
documentation. CPV interpretation follows Commission Regulation (EC) No
213/2008. TF--IDF cosine similarity follows the standard vector-space
definition. Record linkage, survival estimation, proportional-hazards
diagnostics, and change-point detection are supported by the original
Fellegi--Sunter (1969), Davis--Goadrich (2006), Saito--Rehmsmeier (2015),
Kaplan--Meier (1958), Cox (1972), Grambsch--Therneau (1994), and PELT
(Killick et al., 2012) methods. Regime detection follows the Markov-switching
framework of Hamilton (1989). The causal-inference outline in
\S\ref{{sec:causal-inference}} draws on Angrist and Pischke (2009), Pearl, Glymour
and Jewell (2016), Athey and Imbens (2017), and Callaway and Sant'Anna (2021).
None of these sources validates this project's own labels, numerical results,
or thresholds; they justify method choices only. Full URLs and the specific
design implications are recorded in \pathcode{{METHODOLOGICAL\_REFERENCES.md}}.
\begin{{itemize}}
\item \href{{https://www.data.gouv.fr/dataservices/api-bulletin-officiel-des-annonces-des-marches-publics-boamp}}{{Official BOAMP API (DILA)}}.
\item \href{{https://www.insee.fr/fr/metadonnees/definition/c2047}}{{INSEE SIREN definition}} and
\href{{https://www.insee.fr/fr/metadonnees/definition/c1841}}{{INSEE SIRET definition}}.
\item \href{{https://eur-lex.europa.eu/eli/reg/2008/213/oj}}{{Commission Regulation (EC) No 213/2008 on CPV}} and
\href{{https://eur-lex.europa.eu/eli/dir/2014/24/oj}}{{Directive 2014/24/EU, Article 33}}.
\item \href{{https://scikit-learn.org/stable/modules/metrics.html\#cosine-similarity}}{{Cosine similarity documentation}}.
\item \href{{https://doi.org/10.1145/1143844.1143874}}{{Davis--Goadrich (2006)}} and
\href{{https://doi.org/10.1371/journal.pone.0118432}}{{Saito--Rehmsmeier (2015)}} on ROC and precision-recall evaluation for skewed binary decisions.
\item \href{{https://doi.org/10.1080/01621459.1969.10501049}}{{Fellegi--Sunter (1969)}},
\href{{https://doi.org/10.1080/01621459.1958.10501452}}{{Kaplan--Meier (1958)}}, and
\href{{https://doi.org/10.1111/j.2517-6161.1972.tb00899.x}}{{Cox (1972)}}.
\item \href{{https://doi.org/10.1093/biomet/81.3.515}}{{Grambsch--Therneau (1994)}} and
\href{{https://doi.org/10.1080/01621459.2012.737745}}{{Killick et al. (2012)}}.
\item \href{{https://doi.org/10.2307/1912559}}{{Hamilton (1989)}} on Markov-switching
regime models, the framework underlying the trend HMM.
\item \href{{https://www.jstor.org/stable/j.ctvcm4j72}}{{Angrist and Pischke
(2009)}}, \href{{https://www.wiley.com/en-us/Causal+Inference+in+Statistics\%3A+A+Primer-p-9781119186847}}{{Pearl,
Glymour and Jewell (2016)}}, \href{{https://doi.org/10.1257/jep.31.2.3}}{{Athey
and Imbens (2017)}}, and
\href{{https://doi.org/10.1016/j.jeconom.2020.12.001}}{{Callaway and Sant'Anna
(2021)}} on causal identification, cited only to support the outlined-not-executed
causal-inference perspective.
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
    locked = manifest["splits"]["validation"]
    pilot = manifest["splits"]["dev"]
    ceiling = manifest["candidate_reachability"]
    application = load_json(PROCESSED / "linkage_application_summary.json")
    survival = load_json(PROCESSED / "survival_dataset_summary.json")["variants"]["main"]
    m_b = validation_frame.loc[validation_frame["method"].eq("M_B_text_ranking")].iloc[0]
    final_pipeline = f"""# Final Defensible Pipeline

Generated: `{generated_at}`

## Linkage Decision

The final primary event definition is `M_B_text_ranking @ 0.70`. It is a frozen
conservative baseline, not a claim that `0.70` is the optimal threshold.

On the locked split of the Grand Ouest regional reference it gives:

- precision@1: `{m_b.precision:.3f}` (95% CI `{m_b.precision_low:.3f}`-`{m_b.precision_high:.3f}`);
- recall@1: `{m_b.recall:.3f}` (95% CI `{m_b.recall_low:.3f}`-`{m_b.recall_high:.3f}`);
- false-positive rate on negative anchors: `{m_b.fpr:.3f}`;
- accepted links: `{int(m_b.accepted_links)}` on `{locked["usable_anchors"]}` usable anchors.

Recall cannot exceed `{ceiling["candidate_generation_recall_ceiling"]:.3f}`: candidate
generation reaches `{ceiling["positive_anchors_with_reviewed_successor_in_pool"]}` of the
`{ceiling["positive_anchors"]}` reviewed successors, so the remainder is a blocking-stage
loss rather than a scoring one.

These are reference-sample estimates, not independent validation. The labels were
generated by a single LLM research pass over real BOAMP notices, their official
URLs, and wider public sources, dated 2026-08-11, then spot-checked on a subset by
the project owner. They are independent of every method scored, but they were not
verified anchor-by-anchor and are not an independent specialist panel.

The threshold was frozen before this reference was consulted and has not been moved
since. `0.60` remains a required survival sensitivity arm. The completed
production-link diagnostic at `0.70` confirmed `14/20` sampled links conservatively.

`M_C_weighted_gated` has higher recall but also higher false-positive risk.
`M_D_fellegi_sunter` is evaluated on the same reference and does not outperform `M_B`.

## End-to-End Workflow

```text
Official BOAMP API, 2015-2025
  -> schema-aware standardisation
  -> procurement episode reconstruction
  -> Grand Ouest digital study cohort
  -> broad same-buyer candidate generation
  -> four linkage algorithms compared on the Grand Ouest regional reference
  -> M_B primary successor selection
  -> survival dataset, threshold sensitivity, and borderline-band robustness
```

The event remains an **observable successor procurement**, not a confirmed legal
renewal.

## Reference State

- reviewed anchors: `{manifest["reviewed_anchors"]}`, of which `{manifest["remap"]["resolved_to_current_episodes"]}` resolve to exactly one procurement episode;
- pilot split: `{pilot["usable_anchors"]}` usable anchors, `{pilot["positive_anchors"]}` with a reviewed successor;
- locked split: `{locked["usable_anchors"]}` usable anchors, `{locked["positive_anchors"]}` with a reviewed successor;
- pair rows: `{modeling["outputs"]["dev"]["rows"]:,}` pilot and `{modeling["outputs"]["validation"]["rows"]:,}` locked.

## Study State

- cohort episodes: `{survival["validation"]["rows"]:,}`;
- candidate pairs: `{load_json(PROCESSED / "linkage_candidates_summary.json")["candidate_pairs"]:,}`;
- primary accepted links: `{application["cohort_application"]["accepted_links"]}`;
- primary cohort event rate: `{survival["description"]["event_rate"]:.4f}`;

## Canonical Outputs

- `data/processed/boamp/`
- `data/processed/boamp/regional_benchmark/`
- `data/processed/boamp/linkage_evaluation_dev.json`
- `data/processed/boamp/linkage_evaluation_validation.json`
- `data/processed/boamp/regional_benchmark/modeling/modeling_summary.json`
- `data/processed/boamp/survival_analysis_summary.json`
- `SURVIVAL_ANALYSIS_REPORT.md`
- `reports/boamp_methodology_chapter.pdf`
- `notebooks/12_successor_linkage_and_evaluation.ipynb`
- `DATA_QUALITY_REPORT.md`
- `TREND_ANALYSIS_REPORT.md`
- `INTERNSHIP_GUIDE_COMPLIANCE.md`
- `INDEPENDENT_LINK_REVIEW_PROTOCOL.md`
- `PROJECT_WORK_PROTOCOL.md`

## Refresh Command

```bash
PYTHONPATH=. python3 scripts/run_final_pipeline.py --with-notebooks --with-tests
```

Use `--force` only when intentionally rebuilding all materialised stages.
"""
    (PROJECT_ROOT / "FINAL_PIPELINE.md").write_text(final_pipeline, encoding="utf-8")

    reference = f"""# Regional Reference Datasheet

Generated: `{generated_at}`

## What This Reference Is

The reference for successor linkage is a stratified review of
`{manifest["reviewed_anchors"]}` awarded digital procurement anchors drawn from the
study region itself: {manifest["geographical_scope"]}. Each anchor was reviewed
against the real BOAMP notices and official notice URLs of its candidates on
`{manifest["review_date"]}`, before the linkage methods compared below existed.

- source: `{manifest["reference_source"]}`;
- version: `{manifest["reference_version"]}`;
- construction: {manifest["label_provenance"]};
- anchor award dates: `{manifest["temporal_scope"]["anchor_award_dates"][0]}` to `{manifest["temporal_scope"]["anchor_award_dates"][1]}`;
- observation cutoff: `{manifest["temporal_scope"]["study_cutoff"]}`;
- independent of the linkage algorithms: `{manifest["independent_of_linkage_algorithms"]}`;
- independent human specialist review: `{manifest["independent_human_specialist_review"]}`.

It is a **regional reference sample**, not ground truth, and not proof of legal
renewal.

## Materialised State

- reviewed anchors: `{manifest["reviewed_anchors"]}`;
- resolved onto exactly one procurement episode: `{manifest["remap"]["resolved_to_current_episodes"]}`;
- pilot split: `{pilot["usable_anchors"]}` usable anchors, `{pilot["positive_anchors"]}` positive, `{pilot["negative_anchors"]}` negative;
- locked split: `{locked["usable_anchors"]}` usable anchors, `{locked["positive_anchors"]}` positive, `{locked["negative_anchors"]}` negative;
- pair rows: `{modeling["outputs"]["dev"]["rows"]:,}` pilot, `{modeling["outputs"]["validation"]["rows"]:,}` locked;
- candidate-generation recall ceiling: `{ceiling["candidate_generation_recall_ceiling"]:.4f}`.

## Label Definitions

- `OBSERVED_SUCCESSOR`: a later procurement in the reviewed candidate set that
  plausibly replaces or continues the anchor's need.
- `NO_OBSERVED_SUCCESSOR_IN_SCOPE`: none among the candidates considered in the
  research pass.
  This is corpus-relative, not proof that no renewal occurred.
- `OUTSIDE_SCOPE` / `INSUFFICIENT_INFORMATION`: the research pass declined to decide;
  these anchors are excluded from evaluation rather than counted as negatives.

## Method Comparison On The Locked Split

| Method | Threshold | Precision | 95% CI | Recall | 95% CI | FPR | Accepted |
|---|---:|---:|---|---:|---|---:|---:|
"""
    for row in validation_frame.itertuples(index=False):
        reference += (
            f"| `{row.method}` | {row.threshold:.1f} | {row.precision:.3f} | "
            f"{row.precision_low:.3f}-{row.precision_high:.3f} | {row.recall:.3f} | "
            f"{row.recall_low:.3f}-{row.recall_high:.3f} | {row.fpr:.3f} | "
            f"{int(row.accepted_links)} |\n"
        )
    reference += """
Intervals are Wilson score intervals. They overlap heavily: this reference
separates the methods only coarsely, and any claim that one method beats another
must survive that overlap.

## Decision Rule

`M_B_text_ranking @ 0.70` remains the frozen primary event definition. It was
fixed before this reference was consulted and has not been moved since, which is
what allows the locked split to be reported as held out. Choosing a threshold
from these rows now would convert the locked split into a tuning set. A
replacement requires a pre-specified selection rule, direct review of the
incremental links, and fresh evidence.

## Known Limitations

"""
    for limitation in manifest["known_limitations"]:
        reference += f"- {limitation}\n"
    reference += """
## What It May Legitimately Be Used For

Comparing linkage methods on the same exposed candidate pairs, reading the
frozen operating point held out, and bounding recall through candidate
generation. It may not be used to claim externally validated accuracy, national
prevalence, or legal renewal status.
"""
    (PROJECT_ROOT / "REGIONAL_BENCHMARK_REFERENCE.md").write_text(reference, encoding="utf-8")


def write_executive_summary(
    dev: dict[str, Any],
    validation: dict[str, Any],
    modeling: dict[str, Any],
    manifest: dict[str, Any],
    generated_at: str,
) -> Path:
    validation_frame = method_frame(validation)
    m_b = validation_frame.loc[validation_frame["method"].eq("M_B_text_ranking")].iloc[0]
    application = load_json(PROCESSED / "linkage_application_summary.json")
    survival = load_json(PROCESSED / "survival_dataset_summary.json")["variants"]["main"]
    survival_summary = load_json(PROCESSED / "survival_analysis_summary.json")
    km_horizons = pd.read_csv(PROCESSED / "survival_km_horizons.csv").set_index("months")
    trend_signal = pd.read_csv(PROCESSED / "trend_signal_matrix.csv")
    decreasing_segments = trend_signal.loc[trend_signal["state"].eq("decreasing"), "segment"].tolist()
    standardized_notices = load_json(PROCESSED / "standardized_notice_summary.json")["rows"]
    temporal = survival_summary["cox"]["temporal_validation"]
    extended = survival_summary["cox"]["temporal_validation_including_latest_cohort"]

    text = f"""# Executive Summary

Generated: `{generated_at}`
Audience: Gigalis Data & AI Hub management

## What This Project Does

Analyzes official BOAMP public digital procurement notices (2015-2025, Grand
Ouest) to identify **observable successor procurements** -- later BOAMP
episodes from the same buyer that plausibly continue an earlier awarded
digital contract -- and studies time-to-successor with survival analysis and
segment activity with change-point/regime detection. BOAMP does not encode
legal contract renewal directly, so this measures a data proxy, not certified
legal renewal.

## What Was Done

- Standardised {standardized_notices:,} BOAMP notices into
  reconstructed procurement episodes and an awarded Grand Ouest digital study
  cohort of `{survival["validation"]["rows"]:,}` episodes.
- Compared four linkage algorithms on a `{manifest["reviewed_anchors"]}`-anchor Grand
  Ouest regional reference, labelled by an LLM research pass over BOAMP notices,
  official URLs, and wider public sources before those algorithms existed and
  spot-checked on a subset, and kept the pre-frozen `M_B_text_ranking @ 0.70` as the
  primary, precision-first rule (precision `{m_b.precision:.3f}`, recall
  `{m_b.recall:.3f}` on its locked split of
  `{manifest["splits"]["validation"]["usable_anchors"]}` anchors).
- Applied it to the full cohort: `{application["cohort_application"]["accepted_links"]}`
  accepted links, `{survival["description"]["event_rate"]:.1%}` event rate.
- Built a full survival pipeline: Kaplan-Meier, log-rank, Cox (with PH
  diagnostics and temporal validation), parametric models, and 12/24-month
  conditional successor-probability estimates, each cross-checked across four
  linkage-definition sensitivity arms.
- Built a descriptive trend pipeline: quarterly series by CPV segment, PELT
  change-point detection with penalty sensitivity, ADF/KPSS stationarity
  tests, and a 3-state HMM regime model for the overall series and the two
  highest-volume segments.
- Ran a model-assisted (not independent-human) blinded challenge review of 20
  accepted links, 20 structural negatives, and 20 buyer-declared relationships.
- Documented every provenance caveat honestly: an LLM-assisted single-pass
  reference sample, a model-assisted review, and a CPV-division substitute
  where the guide asks for a supervised technology classifier.

## What Works

- The pipeline is reproducible end to end (`scripts/run_final_pipeline.py`),
  with the automated test suite passing and internal consistency checks
  (`data/processed/boamp/canonical_state_validation.json`) all green.
- Kaplan-Meier shows a clear, well-powered separation across CPV segments
  (log-rank statistic `{survival_summary["logrank"]["test_statistic"]:.2f}`,
  `p={survival_summary["logrank"]["p_value"]:.2g}`); estimated successor
  probability is `{km_horizons.loc[12, 'cumulative_successor_probability']:.1%}`
  by 12 months and `{km_horizons.loc[24, 'cumulative_successor_probability']:.1%}`
  by 24 months.
- Framework-agreement status and CPV-35 are the most linkage-robust Cox
  covariates across all four sensitivity arms.
- CPV-48 shows a statistically distinguishable recent decline
  (segments: {", ".join(decreasing_segments) if decreasing_segments else "none"}); other
  segments are stable or uncertain by the 12-quarter signal.

## What Remains Uncertain

- The reference's labels were generated by a single LLM research pass and
  spot-checked on a subset rather than verified anchor-by-anchor, so they are not
  independent human annotation, and its negatives are corpus-relative; the model-assisted
  60-pair review found `70.0%` conservative precision among accepted links,
  below the `80%` target -- independent human review is still needed before
  claiming validated accuracy.
- Absolute event rates and probabilities are linkage-sensitive: event counts
  range from `{survival_summary["sensitivity"]["minimum_events"]:,}` to
  `{survival_summary["sensitivity"]["maximum_events"]:,}` across retained arms.
- Cox temporal validation is weak (C-index `{temporal["train_c_index"]:.3f}` in-sample
  vs `{temporal["test_c_index"]:.3f}` out-of-time on the guideline-aligned
  {temporal["test_years"]} window, `{extended["test_c_index"]:.3f}` on {extended["test_years"]}); the
  model is not validated for individualized operational prediction, and no
  active Gigalis portfolio was available to score.
- The guide's supervised technology-classification deliverable (L2) was not
  built inside this reproducible branch. A parallel classification effort exists
  as an un-integrated export
  (`data/reference/technology_classification/`, 945 rows dated 2026-08-12) that
  covers present-day national opportunities rather than the historical study cohort
  and arrives without training corpus or validation artifacts, so CPV divisions
  are used as the coarser, reproducible substitute.
- The guide's causal-inference question (does a Gigalis framework change
  member behaviour?) is outlined methodologically but not answered -- it
  needs Gigalis-internal membership/adoption-date data not present in BOAMP.

## Recommended Next Steps

1. Commission an independent human procurement-domain reviewer to label the
   prepared blinded 60-pair sample (`INDEPENDENT_LINK_REVIEW_PROTOCOL.md`)
   before any external accuracy claim or threshold change.
2. If the technology classifier remains a priority, recruit a second qualified
   annotator and build a real 300-500 example corpus with genuine Cohen's kappa,
   following the blinded double-annotation design in
   `INDEPENDENT_LINK_REVIEW_PROTOCOL.md`. Supplying the parallel classification
   work's training corpus and validation artifacts would be the cheaper route,
   if they can be obtained.
3. If a Gigalis-membership causal analysis is wanted, supply member identity
   and adoption-date data so the outlined staggered-adoption
   difference-in-differences design can actually be estimated.
4. Treat the linkage, survival, and trend components as frozen; do
   not reopen them without new evidence, per `PROJECT_WORK_PROTOCOL.md`.

## Full Documentation

`README.md`, `FINAL_PIPELINE.md`, `reports/boamp_methodology_chapter.pdf`,
`SURVIVAL_ANALYSIS_REPORT.md`, `TREND_ANALYSIS_REPORT.md`,
`DATA_QUALITY_REPORT.md`, `INTERNSHIP_GUIDE_COMPLIANCE.md`.
"""
    path = PROJECT_ROOT / "EXECUTIVE_SUMMARY.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_notebook(
    validation: dict[str, Any], manifest: dict[str, Any], generated_at: str
) -> None:
    ceiling = manifest["candidate_reachability"]
    # Anchor-level cell counts for the frozen rule, so the notebook can show the
    # arithmetic behind each conditional probability rather than only its value.
    cells_m_b = next(
        method["unweighted"]
        for method in validation["methods"]
        if method["method"] == "M_B_text_ranking"
    )
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
            "# 12. Successor linkage and regional-reference evaluation\n\n"
            f"Generated: `{generated_at}`\n\n"
            "This notebook is regenerated from the pipeline outputs. It is the "
            "reader-facing linkage and evaluation notebook."
        ),
        nbf.v4.new_markdown_cell(
            "## tl;dr\n\n"
            "`M_B_text_ranking @ 0.70` is the frozen conservative primary event "
            "definition, not a claim of threshold optimality. It was fixed before "
            "the Grand Ouest regional reference was consulted, which is why the "
            "locked split below can be read as held out. All four algorithms are "
            "compared, including `M_D_fellegi_sunter`, scored from the fitted model."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / 'scripts').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "PROCESSED = PROJECT_ROOT / 'data/processed/boamp'\n"
            "BENCHMARK = PROCESSED / 'regional_benchmark'\n\n"
            "def load_json(path):\n"
            "    with open(path, 'r', encoding='utf-8') as f:\n"
            "        return json.load(f)\n\n"
            "dev = load_json(PROCESSED / 'linkage_evaluation_dev.json')\n"
            "validation = load_json(PROCESSED / 'linkage_evaluation_validation.json')\n"
            "modeling = load_json(BENCHMARK / 'modeling/modeling_summary.json')\n"
            "manifest = load_json(BENCHMARK / 'regional_benchmark_manifest.json')\n"
        ),
        nbf.v4.new_code_cell(
            "def method_frame(summary):\n"
            "    rows = []\n"
            "    for method in summary['methods']:\n"
            "        metrics = method['unweighted']\n"
            "        weighted = method.get('design_weighted', {})\n"
            "        rows.append({\n"
            "            'method': method['method'],\n"
            "            'threshold': method['threshold'],\n"
            "            'accepted_links': metrics['accepted_links'],\n"
            "            'precision': metrics['precision_at_1'],\n"
            "            'recall': metrics['recall_at_1'],\n"
            "            'fpr': metrics['false_positive_rate_on_negatives'],\n"
            "            'coverage': metrics['coverage'],\n"
            "            'precision_ci': metrics.get('precision_at_1_interval_95'),\n"
            "            'recall_ci': metrics.get('recall_at_1_interval_95'),\n"
            "            'weighted_precision': weighted.get('precision_at_1', {}).get('estimate'),\n"
            "            'weighted_recall': weighted.get('recall_at_1', {}).get('estimate'),\n"
            "            'weighted_fpr': weighted.get('false_positive_rate_on_verified_negatives', {}).get('estimate'),\n"
            "        })\n"
            "    return pd.DataFrame(rows)\n\n"
            "dev_methods = method_frame(dev)\n"
            "validation_methods = method_frame(validation)\n"
            "validation_methods\n"
        ),
        nbf.v4.new_markdown_cell(
            "## What Is Actually Being Scored\n\n"
            "The decision here is **not** a yes/no call on a pre-formed pair, so an "
            "ordinary binary confusion matrix would hide the failure mode that matters "
            "most. For each anchor $i$ the reference names a *successor identity* and "
            "the method returns one, and the method can be wrong by naming the wrong "
            "candidate rather than by naming one at all.\n\n"
            "Let $J_i$ be the candidate set that survived blocking for anchor $i$. Then\n\n"
            "$$R_i \\in J_i \\cup \\{\\varnothing\\}, \\qquad \\hat R_i \\in J_i \\cup \\{\\varnothing\\},$$\n\n"
            "where $R_i$ is the **reviewed successor** the reference identifies "
            "($\\varnothing$ if it found none) and $\\hat R_i$ is the successor the "
            "linkage rule **accepts** ($\\varnothing$ if it abstains). Three indicators "
            "follow, and they are what every metric below is built from:\n\n"
            "$$P_i = \\mathbf{1}\\{R_i \\neq \\varnothing\\}, \\qquad "
            "A_i = \\mathbf{1}\\{\\hat R_i \\neq \\varnothing\\}, \\qquad "
            "C_i = \\mathbf{1}\\{\\hat R_i = R_i \\neq \\varnothing\\}.$$\n\n"
            "In words: $P_i$ says the reference found a successor, $A_i$ says the method "
            "committed to one, and $C_i$ says the one it committed to is exactly the "
            "reviewed one. A fourth indicator belongs to the stage *before* linkage:\n\n"
            "$$E_i = \\mathbf{1}\\{R_i \\in J_i\\},$$\n\n"
            "the reviewed successor survived candidate generation. An anchor with "
            "$E_i = 0$ is unrecoverable by any scorer, however good.\n\n"
            "The reason this notation earns its place: an anchor can have $P_i = 1$ and "
            "$A_i = 1$ and still have $C_i = 0$, because the method accepted the wrong "
            "candidate. That single case is counted against precision *and* against "
            "recall, and it is invisible in the $TP/(TP+FP)$ shorthand."
        ),
        nbf.v4.new_markdown_cell(
            "## The Metrics As Conditional Probabilities\n\n"
            "Each metric conditions on a different population, which is why they can "
            "move in opposite directions.\n\n"
            "| Quantity | Form | Question it answers |\n"
            "|---|---|---|\n"
            "| Candidate reachability | $P(E=1 \\mid P=1)$ | If a reviewed successor exists, did blocking keep it? |\n"
            "| Precision | $P(C=1 \\mid A=1)$ | If the method accepts, is the accepted candidate the reviewed one? |\n"
            "| Recall | $P(C=1 \\mid P=1)$ | If a reviewed successor exists, is it recovered exactly? |\n"
            "| False-positive rate | $P(A=1 \\mid P=0)$ | If no reviewed successor exists, did the method accept anyway? |\n"
            "| Coverage | $P(A=1)$ | What share of anchors get a link rather than an abstention? |\n\n"
            "**Precision and recall are the same event under reversed conditioning.** "
            "$P(C=1 \\mid P=1)$ conditions on the reference; $P(C=1 \\mid A=1)$ conditions "
            "on the algorithm. They are not interchangeable and need not be close, "
            "because the two conditioning sets are different populations. The "
            "false-positive rate conditions on a *third* population -- anchors the "
            "reference found nothing for -- so it is **not** $1 - \\text{precision}$; the "
            "two share no denominator.\n\n"
            "The two stages are deliberately given opposite objectives. Candidate "
            "generation maximises $P(E=1 \\mid P=1)$, because a successor it discards is "
            "gone for good; the linkage rule then maximises $P(C=1 \\mid A=1)$, because a "
            "false link fabricates both a survival event and an event time. High recall "
            "first, high precision later. This also fixes the ceiling: since a correct "
            "acceptance requires the successor to be in $J_i$ at all,\n\n"
            "$$P(C=1 \\mid P=1) \\le P(E=1 \\mid P=1).$$"
        ),
        nbf.v4.new_markdown_cell("## Reference State"),
        nbf.v4.new_code_cell(
            "pd.DataFrame([\n"
            "    {'item': 'reviewed anchors', 'value': manifest['reviewed_anchors']},\n"
            "    {'item': 'resolved to one episode', 'value': manifest['remap']['resolved_to_current_episodes']},\n"
            "    {'item': 'pilot usable anchors', 'value': manifest['splits']['dev']['usable_anchors']},\n"
            "    {'item': 'pilot positive anchors', 'value': manifest['splits']['dev']['positive_anchors']},\n"
            "    {'item': 'locked usable anchors', 'value': manifest['splits']['validation']['usable_anchors']},\n"
            "    {'item': 'locked positive anchors', 'value': manifest['splits']['validation']['positive_anchors']},\n"
            "    {'item': 'reviewed successors in the reference', 'value': manifest['candidate_reachability']['positive_anchors']},\n"
            "    {'item': 'of those, reachable after blocking', 'value': manifest['candidate_reachability']['positive_anchors_with_reviewed_successor_in_pool']},\n"
            "    {'item': 'P(E=1 | P=1), the recall ceiling', 'value': manifest['candidate_reachability']['candidate_generation_recall_ceiling']},\n"
            "])"
        ),
        nbf.v4.new_markdown_cell(
            "Candidate generation retained the reviewed successor in "
            f"`{ceiling['positive_anchors_with_reviewed_successor_in_pool']}` of the "
            f"`{ceiling['positive_anchors']}` reviewed cases, so "
            f"$\\hat P(E=1 \\mid P=1) = {ceiling['positive_anchors_with_reviewed_successor_in_pool']}/"
            f"{ceiling['positive_anchors']} = "
            f"{ceiling['candidate_generation_recall_ceiling']:.3f}$. This is "
            "candidate-generation reachability measured on this reference sample -- "
            "*pairs completeness* in record-linkage terms -- and **not** population "
            "recall. Both unreachable cases are attributed to a named blocking "
            "condition in `CANDIDATE_GENERATION_AUDIT.md`; neither is an "
            "implementation defect."
        ),
        nbf.v4.new_markdown_cell("## Held-Out Method Comparison On The Locked Split"),
        nbf.v4.new_code_cell(
            "display(validation_methods[['method', 'threshold', 'accepted_links', 'precision', 'precision_ci', 'recall', 'recall_ci', 'fpr', 'coverage']])\n\n"
            "ax = validation_methods.set_index('method')[['precision', 'recall', 'fpr']].plot(\n"
            "    kind='bar', figsize=(9, 4.5), width=0.72\n"
            ")\n"
            "ax.set_title('Locked split: the same three conditional probabilities per method')\n"
            "ax.set_ylabel('probability')\n"
            "ax.set_ylim(0, 1)\n"
            "ax.set_xlabel('')\n"
            "ax.legend(['precision  P(C=1 | A=1)', 'recall  P(C=1 | P=1)',\n"
            "           'FPR  P(A=1 | P=0)'], frameon=False)\n"
            "ax.tick_params(axis='x', rotation=28)\n"
            "ax.grid(axis='y', alpha=0.25)\n"
            "plt.tight_layout()"
        ),
        nbf.v4.new_code_cell(
            "# The frozen rule read cell by cell, so each rate above can be traced to\n"
            "# the anchors that produced it.\n"
            "m_b = next(m['unweighted'] for m in validation['methods']\n"
            "           if m['method'] == 'M_B_text_ranking')\n"
            "display(pd.DataFrame([\n"
            "    {'cell': 'C=1 (accepted the reviewed successor)', 'anchors': m_b['true_positive']},\n"
            "    {'cell': 'A=1, P=1, C=0 (accepted the wrong candidate)', 'anchors': m_b['false_positive_wrong_successor']},\n"
            "    {'cell': 'A=0, P=1 (abstained on a positive anchor)', 'anchors': m_b['false_negative_abstained']},\n"
            "    {'cell': 'A=1, P=0 (accepted where the reference has none)', 'anchors': m_b['false_positive_on_no_successor_anchor']},\n"
            "    {'cell': 'A=0, P=0 (abstained on a negative anchor)', 'anchors': m_b['true_negative_abstained']},\n"
            "]).set_index('cell'))\n\n"
            "print(f\"P(C=1 | A=1) = {m_b['true_positive']}/{m_b['accepted_links']}\"\n"
            "      f\" = {m_b['precision_at_1']:.3f}   precision\")\n"
            "print(f\"P(C=1 | P=1) = {m_b['true_positive']}/{m_b['positive_anchors']}\"\n"
            "      f\" = {m_b['recall_at_1']:.3f}   recall\")\n"
            "print(f\"P(A=1 | P=0) = {m_b['false_positive_on_no_successor_anchor']}/{m_b['negative_anchors']}\"\n"
            "      f\" = {m_b['false_positive_rate_on_negatives']:.3f}   false-positive rate\")\n"
            "print(f\"P(A=1)       = {m_b['accepted_links']}/{m_b['anchors_evaluated']}\"\n"
            "      f\" = {m_b['coverage']:.3f}   coverage\")\n"
        ),
        nbf.v4.new_markdown_cell(
            "### Reading those four numbers\n\n"
            f"**Precision, $\\hat P(C=1 \\mid A=1) = {cells_m_b['true_positive']}/"
            f"{cells_m_b['accepted_links']} = {cells_m_b['precision_at_1']:.3f}$.** Among the "
            f"`{cells_m_b['accepted_links']}` links `M_B` accepted on the locked reference, "
            f"`{cells_m_b['true_positive']}` matched the reviewed successor. This is a "
            f"reference-sample estimate on `{cells_m_b['accepted_links']}` accepted links with a wide "
            f"interval (95% CI "
            f"`{cells_m_b['precision_at_1_interval_95'][0]:.3f}`-`{cells_m_b['precision_at_1_interval_95'][1]:.3f}`): "
            "one changed decision moves it materially. It is not population accuracy and "
            "not independent specialist validation.\n\n"
            f"**Recall, $\\hat P(C=1 \\mid P=1) = {cells_m_b['true_positive']}/"
            f"{cells_m_b['positive_anchors']} = {cells_m_b['recall_at_1']:.3f}$.** Of the "
            f"`{cells_m_b['positive_anchors']}` anchors the reference says have a successor, the rule "
            f"recovered `{cells_m_b['true_positive']}`. The "
            f"`{cells_m_b['positive_anchors'] - cells_m_b['true_positive']}` misses are of two kinds and "
            f"count identically here: `{cells_m_b['false_negative_abstained']}` abstentions and "
            f"`{cells_m_b['false_positive_wrong_successor']}` wrong acceptance. Only the second "
            "fabricates a link. The ceiling is not 1 but "
            f"`{ceiling['candidate_generation_recall_ceiling']:.3f}`, set by blocking.\n\n"
            f"**False-positive rate, $\\hat P(A=1 \\mid P=0) = "
            f"{cells_m_b['false_positive_on_no_successor_anchor']}/{cells_m_b['negative_anchors']} = "
            f"{cells_m_b['false_positive_rate_on_negatives']:.3f}$.** On the "
            f"`{cells_m_b['negative_anchors']}` anchors the reference found nothing for, the rule "
            "accepted nothing either. This is a diagnostic on a small set of "
            "corpus-relative negatives, **not** evidence that the population "
            "false-positive rate is literally zero.\n\n"
            f"**Coverage, $\\hat P(A=1) = {cells_m_b['accepted_links']}/"
            f"{cells_m_b['anchors_evaluated']} = {cells_m_b['coverage']:.3f}$.** Roughly one anchor in "
            "nine receives a link. Low coverage is not a defect to be fixed here: it is "
            "the direct consequence of prioritising $P(C=1 \\mid A=1)$, and in a survival "
            "dataset an abstention becomes honest right-censoring while a false link "
            "becomes a fabricated event at a fabricated time."
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretation\n\n"
            "`M_C_weighted_gated` recovers more reviewed successors, but its false-positive "
            "rate is materially higher. For survival analysis, a false link is more "
            "damaging than an abstention because it fabricates both an event and an "
            "event time. Thresholds other than `0.70` are carried as sensitivity arms "
            "rather than selected from these rows: choosing one now would turn the "
            "locked split into a tuning set. On a reference this small the intervals "
            "overlap heavily, so read them before separating any two methods. The use of "
            "precision-recall evidence for this rare-positive "
            "decision follows [Davis and Goadrich (2006)](https://doi.org/10.1145/1143844.1143874) "
            "and [Saito and Rehmsmeier (2015)](https://doi.org/10.1371/journal.pone.0118432). "
            "Those papers support the diagnostic choice, not this project's numerical results."
        ),
        nbf.v4.new_markdown_cell("## Modeling-Ready Tables"),
        nbf.v4.new_code_cell(
            "pd.DataFrame(modeling['outputs']).T[[\n"
            "    'rows', 'anchors', 'primary_positive_pairs', 'positive_anchors'\n"
            "]]"
        ),
        nbf.v4.new_code_cell(
            "feature_columns = pd.Series(modeling['feature_columns'], name='feature')\n"
            "display(feature_columns.to_frame())\n"
            "assert 'fs_match_probability' in set(modeling['feature_columns'])\n"
        ),
        nbf.v4.new_markdown_cell(
            "## Caveat\n\n"
            "The labels were generated by a single LLM research pass over real BOAMP "
            "notices, their official URLs, and wider public sources, dated 2026-08-11, "
            "then spot-checked on a subset by the project owner. They are independent "
            "of every method scored here, but they were not verified "
            "anchor-by-anchor and are not an independent specialist panel. "
            "Negatives are corpus-relative: roughly 25 candidates per anchor were "
            "considered, so the false-positive rate is conservative by construction "
            "rather than a population-wide rate. These are "
            "reference-sample estimates, not validated legal-renewal accuracy."
        ),
    ]
    nb["cells"] = cells
    nbf.write(nb, NOTEBOOK)


def main() -> int:
    generated_at = datetime.now().isoformat(timespec="seconds")
    dev = load_json(PROCESSED / "linkage_evaluation_dev.json")
    validation = load_json(PROCESSED / "linkage_evaluation_validation.json")
    modeling = load_json(BENCHMARK / "modeling/modeling_summary.json")
    manifest = load_json(BENCHMARK / "regional_benchmark_manifest.json")

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_method_metrics(
        method_frame(dev),
        "Regional reference: pilot split",
        FIGURES / "benchmark_dev_method_metrics.png",
    )
    plot_method_metrics(
        method_frame(validation),
        "Regional reference: locked split",
        FIGURES / "benchmark_validation_method_metrics.png",
    )
    plot_modeling_counts(modeling, FIGURES / "benchmark_modeling_counts.png")
    report_path = write_methodology_report(dev, validation, modeling, manifest, generated_at)
    pdf_path = compile_methodology_pdf(report_path)
    write_status_files(dev, validation, modeling, manifest, generated_at)
    write_notebook(validation, manifest, generated_at)
    summary_path = write_executive_summary(dev, validation, modeling, manifest, generated_at)

    print(
        json.dumps(
            {
                "generated_at": generated_at,
                "report": str(report_path.relative_to(PROJECT_ROOT)),
                "report_pdf": str(pdf_path.relative_to(PROJECT_ROOT)),
                "notebook": str(NOTEBOOK.relative_to(PROJECT_ROOT)),
                "executive_summary": str(summary_path.relative_to(PROJECT_ROOT)),
                "figures": [
                    str((FIGURES / "benchmark_dev_method_metrics.png").relative_to(PROJECT_ROOT)),
                    str((FIGURES / "benchmark_validation_method_metrics.png").relative_to(PROJECT_ROOT)),
                    str((FIGURES / "benchmark_modeling_counts.png").relative_to(PROJECT_ROOT)),
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
