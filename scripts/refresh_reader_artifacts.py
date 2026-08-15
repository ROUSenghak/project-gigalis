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
    ceiling = manifest["candidate_reachability"]
    candidates = load_json(PROCESSED / "linkage_candidates_summary.json")
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

    survival_summary = load_json(PROCESSED / "survival_analysis_summary.json")
    cox_results = pd.read_csv(PROCESSED / "survival_cox_results.csv")
    ph_diagnostics = pd.read_csv(PROCESSED / "survival_ph_diagnostics.csv")
    parametric_comparison = pd.read_csv(PROCESSED / "survival_parametric_comparison.csv")
    cox_sensitivity = pd.read_csv(PROCESSED / "survival_cox_linkage_sensitivity.csv")
    km_horizons = pd.read_csv(PROCESSED / "survival_km_horizons.csv").set_index("months")
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
This report states the current defensible BOAMP successor-linkage pipeline. The
study does not claim to certify legal contract renewals. Its
event is an \emph{{observable successor procurement}}: a later BOAMP procurement
episode from the same buyer that is sufficiently similar to an earlier awarded
digital procurement episode.

The current Grand Ouest survival cohort contains
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
Let \(i\) index an earlier awarded procurement episode and \(j\) index a later
candidate episode. The object of interest is not a legal renewal certificate,
because BOAMP notices do not provide that ground truth consistently. Instead,
the pipeline estimates whether the data show a later procurement episode that is
credible enough to be treated as a successor:
\[
Y_i =
\begin{{cases}}
1, & \text{{if one accepted observable successor is found before the cutoff,}}\\
0, & \text{{otherwise.}}
\end{{cases}}
\]
The event time is
\[
\tau_i = C_{{\hat{{j}}}} - A_i,
\]
where \(A_i\) is the award-date origin of episode \(i\), \(C_{{\hat{{j}}}}\) is
the first-publication date of the accepted successor, and \(\hat{{j}}\) is the
selected candidate. If no accepted successor is found before 2025-12-31, the
episode is right-censored.

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
validated SIRENs block a buyer match. The current legal-form audit reports
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
anchor episode with award date \(A_i\) and candidate publication date \(C_j\),
the candidate is exposed only if
\[
A_i+90 \leq C_j \leq A_i+2920.
\]
The lower bound removes very near follow-up notices and parallel administrative
activity. The upper bound is approximately eight years; it keeps the candidate
pool inclusive enough for long public-procurement cycles. Precision is then
controlled by the selection stage rather than by a narrow time window. The
current blocking rule generated {candidates["candidate_pairs"]:,} candidate
pairs, with a median of {candidates["candidates_per_anchor"]["median"]:.0f}
candidates per anchor.

\section{{Grand Ouest Regional Reference}}
\label{{sec:linkage-caveat}}
Evaluation uses a regional reference sample drawn from the study region itself:
{reviewed_anchors} awarded digital procurement anchors stratified by CPV theme,
buyer-identifier quality, and duration availability across Bretagne, Pays de la
Loire, and Normandie, reviewed on 2026-08-11 against the notices and official
BOAMP URLs of each candidate. The pilot split carries
{modeling["outputs"]["dev"]["anchors"]} anchors and
{modeling["outputs"]["dev"]["rows"]:,} pair rows; the locked split carries
{modeling["outputs"]["validation"]["anchors"]} anchors and
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

This reference replaced an earlier France-level benchmark whose labels were
emitted by deterministic rules reading the same text, CPV, and date evidence the
linkage methods consume. That construction made the comparison circular: a
method could only score well by agreeing with a hand-written rule built from its
own features, and the resulting numbers measured rule agreement rather than
correctness. It has been removed from the repository in full -- its data, its
construction scripts, and its annotation tooling -- so nothing here can descend
from it. Its history remains in version control.

Five limitations bind everything computed from the regional reference. The
labels were produced by a single LLM research pass over the supplied notices,
their official URLs, and wider public sources, then spot-checked on a subset by
the project owner rather than verified anchor-by-anchor; anchors outside that
subset carry the model's judgement as recorded, so this is a reference sample and
not ground truth. The sources behind each individual label were not recorded, so
a given anchor's evidence trail cannot be fully reconstructed or independently
re-executed. Negatives are corpus-relative: roughly 25 candidates per anchor were
considered rather than the full pool, so a false-positive rate computed on them
is an upper bound. And while no linkage method existed or was consulted when the
labels were made, judging whether later notice text continues an earlier need
draws on the same text, CPV, and date evidence the text-ranking method scores, so
the labels are method-independent without being fully evidence-independent. The sample is small enough that every point estimate needs its
interval read beside it. And the anchors were re-resolved onto the current
episode reconstruction through their BOAMP notice identifiers, because the
review recorded episode identifiers from an earlier reconstruction;
{reviewed_anchors - manifest["remap"]["resolved_to_current_episodes"]} anchors
did not resolve to exactly one current episode and were dropped rather than
guessed.

\section{{Linkage Algorithms}}
All methods operate on the same exposed candidate set. This is crucial: the
primary method is not comparing text over the whole BOAMP universe. Buyer and
time plausibility are imposed before text ranking.

\paragraph{{\(M_A\): deterministic evidence.}}
Define \(B_{{ij}}\) as buyer-identity support, \(P_{{ij}}\) as CPV continuity, and
\(T_{{ij}}\) as text similarity. The deterministic rule accepts a link only when
strong buyer evidence is present, CPV continuity is positive, and a minimum text
signal is present:
\[
B_{{ij}}=1,\quad P_{{ij}}>0,\quad T_{{ij}}\geq t_A .
\]
It is interpretable, but it loses recall when CPV or buyer identifiers are
missing or noisy.

\paragraph{{\(M_B\): text ranking.}}
For each episode, the text fields are converted to TF--IDF vectors \(x_i\) and
\(x_j\). Text similarity is cosine similarity:
\[
T_{{ij}}=\cos(x_i,x_j)=\frac{{x_i\cdot x_j}}{{\|x_i\|\|x_j\|}}.
\]
Within the same-buyer, plausible-time candidate set, the method selects
\[
\hat{{j}}_i=\arg\max_j T_{{ij}},
\]
and accepts the selected candidate only if
\[
T_{{i\hat{{j}}_i}}\geq 0.70.
\]
This is the primary method because it is simple, reproducible, auditable, and
best matches the precision-first objective.

\paragraph{{\(M_C\): weighted gated score.}}
This method combines evidence components into a score:
\[
S_{{ij}}=0.50B_{{ij}}+0.25T_{{ij}}+0.20P_{{ij}}+0.05G_{{ij}},
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
\code{{fs\_match\_probability}}. In the current data it does not outperform the
simple text-ranking rule, likely because true successors are rare and
same-buyer procurement activity contains many non-renewal lookalikes.

\paragraph{{A removed arm.}} A duration-conditioned variant was built and
evaluated during development, then removed from the repository in full. Reliable
duration is missing for most of the cohort, so such a rule can differentiate
itself only on a minority of episodes, and where the evidence does exist the
observed data show many declared successors published well before the declared
end date. The event-definition sensitivity framework is therefore the four
threshold and method arms (\(M_B@0.80\), \(M_B@0.70\), \(M_B@0.60\),
\(M_C@0.70\)), which vary the decision rule along the dimension that actually
moves the event set, together with the borderline-band check. The removed arm's
history remains in version control.

The separate descriptive comparison between declared contract duration and
observed successor delay is a different thing and remains part of the evidence:
it is a diagnostic about duration reliability, not a linkage algorithm.

\section{{Survival Modeling}}
The accepted-link decision from \(M_B\) defines the survival event. For episode
\(i\), \(\tau_i\) is the time from award date \(A_i\) to the accepted successor's
publication date if \(Y_i=1\), or to the study cutoff (2025-12-31) if
\(Y_i=0\) (administrative right-censoring). \(Y_i=0\) means no accepted
observable successor was found before the cutoff; it is not proof of
abandonment.

\paragraph{{Kaplan--Meier.}} The non-parametric survivor function
\(S(t)=P(T>t)\) is estimated by
\href{{https://doi.org/10.1080/01621459.1958.10501452}}{{Kaplan and Meier
(1958)}}'s product-limit estimator, stratified by CPV segment, buyer type, and
other covariates of interest. Group differences use the log-rank test.

\paragraph{{Cox proportional hazards.}} The semi-parametric model is
\[
h(t\mid X)=h_0(t)\exp(\beta_1X_1+\cdots+\beta_pX_p),
\]
with covariates selected for substantive relevance and data quality rather than
automated search: CPV digital segment, buyer region, framework-agreement flag,
validated-SIREN availability, and centered award year. The rule of one
covariate per ten observed events (Van Belle et al., 2002) is respected with
{survival_summary["cox"]["events"]:,} events supporting
{survival_summary["cox"]["covariates"]} covariates. The proportional-hazards
assumption is tested with Schoenfeld residuals (Grambsch and Therneau, 1994);
violations are reported and interpreted descriptively rather than silently
dropped or used to discard the model.

\paragraph{{Parametric models.}} Exponential, Weibull, log-logistic, log-normal,
and generalized-gamma models are compared by log-likelihood, AIC, and BIC and
checked graphically against the Kaplan--Meier curve. Their role is to identify
the best-fitting family and to provide the instrument any extrapolation past the
observation window would require. They are \emph{{not}} the source of the reported
12/24-month probabilities: every horizon quoted in this report falls inside the
observed window, and the smooth families flatten the empirical renewal shoulder,
so the operational conditional probabilities are read off the Kaplan--Meier
estimator, which imposes no shape.

\paragraph{{Temporal validation.}} The model is fit once on episodes awarded
2015--2021 and scored out of time without refitting. The primary evaluation
window is 2022--2024, as specified by the internship guideline; 2022--2025 is
carried as a sensitivity read that adds the shortest-follow-up award cohort.
Harrell's concordance index \(C\) is reported on each split to assess
discrimination and out-of-time stability, not to target a specific value.

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
contrast (\(M_C@0.70\)) event definitions. A conclusion is reported as robust
only when it is stable in sign and approximate magnitude across these arms.

\section{{Trend And Change-Point Detection}}
Quarterly awarded-episode counts \(N_{{s,q}}\) are built for the overall cohort
and each CPV digital segment from 2015Q2 (the first complete quarter) through
2025Q4, including zero-count quarters.

\paragraph{{Change-point detection (PELT).}}
\href{{https://doi.org/10.1080/01621459.2012.737745}}{{Killick, Fearnhead and
Eckley (2012)}}'s PELT algorithm minimizes a penalized segmentation objective on
the z-standardized series with penalty \(\beta=\lambda\log(n)\); the central
result uses \(\lambda=1\) with sensitivity at \(0.5\) and \(2.0\), and a break
is called stable only if it lies within one quarter under all three penalties.

\paragraph{{Stationarity.}} Augmented Dickey--Fuller (unit-root null) and KPSS
(level-stationary null) tests are run on each segment's series. The two tests
have opposite null hypotheses and are read together, not individually.

\paragraph{{Regime detection (HMM).}} A 3-state Gaussian hidden Markov model is
fit on the quarter-over-quarter \emph{{change}} in episode count -- not the
level -- for the overall cohort and the two highest-volume CPV segments, so
states describe typical period-over-period direction (decline, plateau,
growth) rather than absolute activity level. The Viterbi-decoded state
sequence gives the current-quarter regime and its posterior probability. This
complements PELT (which finds discrete historical breaks) with a continuously
updated regime read; the two are not required to agree, and disagreement is
reported honestly rather than reconciled.

\paragraph{{Recent direction.}} An OLS slope over the latest 12 quarters gives
an \texttt{{increasing}}/\texttt{{decreasing}}/\texttt{{stable\_or\_uncertain}}
label at exploratory \(\alpha=0.10\), uncorrected for multiple testing. None of
these methods forecasts future values or identifies the cause of a break;
causal attribution requires documentary or stakeholder evidence not available
in BOAMP alone. Monetary and duration trend series are omitted because no
canonical awarded-amount field is validated at episode grain, and 2025's
duration-field completeness jump is a measurement change, not a genuine shift
in contract durations.

\section{{Technology Segmentation}}
\label{{sec:nlp-scope}}
The internship guide's L2 deliverable specifies a supervised technology
taxonomy: 300--500 manually annotated contracts across 8--12 classes, a
TF--IDF+logistic-regression/SVM baseline evaluated by macro-F1 and confusion
matrix, and an optional CamemBERT comparison. \textbf{{This was not built inside
the reproducible analytical branch.}} Two independent annotators with a Cohen's
kappa agreement statistic are required for a defensible L2 corpus, and no second
qualified annotator was available within this session's scope; producing
single-pass AI-assisted labels and calling their agreement "kappa" would repeat
exactly the self-consistency problem this project has documented for the
successor-linkage benchmark (\S\ref{{sec:linkage-caveat}}) instead of avoiding it.

A finer technology-classification effort was carried out in parallel, outside
this pipeline: \pathcode{{data/reference/technology\_classification/}} holds a
945-row export dated 2026-08-12 carrying a \code{{Domaine}} label per notice. It
could not be integrated at the analytical freeze and is read by no stage of this
pipeline. It covers current national opportunities rather than the 2015--2025
Grand Ouest study cohort, and it arrives without the training corpus, the
annotation guidelines, or the validation artifacts that would let its labels be
audited or applied historically. The present reproducible branch therefore
retains CPV divisions as the primary technological segmentation. That is a
statement about what could be integrated and verified here, not a judgement on
the parallel work.

Every technology segment referenced in this report -- cohort selection, Cox
covariates, quarterly trend series -- therefore uses CPV divisions 32
(telecommunications equipment), 35 (security), 48 (software), and 72 (IT
services) as a reproducible substitute. This is a real scope reduction, not a
disguised classifier: CPV divisions are official, zero-missingness EU
categories under Regulation 213/2008, but they are coarser than a learned
taxonomy and cannot distinguish sub-themes such as cloud versus on-premise
infrastructure within a division. A future session with a second qualified
annotator could complete L2 by reusing the blinded-review design recorded in
\pathcode{{INDEPENDENT\_LINK\_REVIEW\_PROTOCOL.md}}; the scripts that once
implemented it were removed with the retired France-level benchmark and remain
recoverable from version control.

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
\caption{{Unweighted dev metrics on all labelled current benchmark frames.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{figures/benchmark_dev_method_metrics.png}}
\caption{{Dev split method comparison generated from the current benchmark evaluation JSON.}}
\end{{figure}}

\section{{Internal Held-Out Reference Results}}
The table below is an internal method-comparison diagnostic. It must not be
presented as external validation. The primary comparison is
anchor-level exact-successor performance: each anchor can produce one accepted
successor or abstain.
\[
\text{{precision}}=\frac{{TP}}{{TP+FP}},\quad
\text{{recall}}=\frac{{TP}}{{TP+FN}},\quad
\text{{FPR}}=\frac{{FP}}{{FP+TN}}.
\]
For this project, false-positive control is prioritised because false links
fabricate survival events.

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
\caption{{Unweighted held-out metrics on the locked split of the regional reference.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{figures/benchmark_validation_method_metrics.png}}
\caption{{Held-out method comparison generated from the regional-reference evaluation JSON.}}
\end{{figure}}

\section{{Quality Evidence And Interpretation}}
The held-out internal result supports retaining a conservative baseline for
continued work; it does not establish final accuracy. \code{{M\_B}} has
only {int(m_b.accepted_links)} accepted held-out links, so its precision
estimate is necessarily sample-sensitive; one changed decision would move the
number materially. Within this reference, the direction is coherent:
\code{{M\_B}} gives the best precision and the lowest false-positive
rate among useful methods, while \code{{M\_C}} shows the expected precision-recall
trade-off.

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
A separate blinded challenge review sampled 20 current accepted links, 20
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
to justify keeping the current claim narrow and the lower threshold unpromoted.

\section{{Modeling Tables}}
The modeling-ready tables include strict, primary, broad, and non-match target
columns, plus observable features. They now include the Fellegi--Sunter columns
\code{{fs\_match\_weight}} and \code{{fs\_match\_probability}}, so modeling and
evaluation use the same feature state.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\textwidth]{{figures/benchmark_modeling_counts.png}}
\caption{{Current benchmark modeling table sizes and label support.}}
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

\paragraph{{Kaplan--Meier.}} The estimated probability of an observable
successor is {km_horizons.loc[12, 'cumulative_successor_probability'] * 100:.3f}\%
by 12 months,
{km_horizons.loc[24, 'cumulative_successor_probability'] * 100:.3f}\% by 24
months, and
{km_horizons.loc[60, 'cumulative_successor_probability'] * 100:.3f}\% by 60
months. The Kaplan--Meier median survival time is
\textbf{{{survival_summary["km"]["median_status"].replace('_', ' ')}}}: the curve
never falls below 0.5 within the observation window, so no median is reported.
This is distinct from the {survival_main["description"]["median_time_to_successor_months"]:.2f}-month
median delay \emph{{among linked events only}}, which conditions on the event
having occurred. A multivariate log-rank test across CPV segments gives
statistic {survival_summary["logrank"]["test_statistic"]:.2f}
(\(p={survival_summary["logrank"]["p_value"]:.3g}\)).

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
\caption{{Cox hazard ratios, main linkage arm. In-sample C-index
{survival_summary["cox"]["in_sample_c_index"]:.3f}.}}
\end{{table}}
Framework-agreement episodes and CPV-35 carry the largest hazard ratios among
substantively interpretable covariates. These are descriptive associations with
the observed hazard, not causal effects.

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
\caption{{Schoenfeld-residual proportional-hazards tests.}}
\end{{table}}
The assumption is rejected (\(p<0.05\)) for
\code{{{latex_escape(', '.join(ph_violations))}}}. These coefficients are
therefore reported as time-averaged associations rather than constant hazard
ratios; stratification or time-interaction terms would be the next refinement
if individualized prediction were required.

\paragraph{{Temporal validation.}} Training on
{temporal["train_years"]} ({temporal["train_contracts"]:,} episodes,
{temporal["train_events"]:,} events) and evaluating on the guideline-aligned
{temporal["test_years"]} window ({temporal["test_contracts"]:,} episodes,
{temporal["test_events"]:,} events) gives C-index
{temporal["train_c_index"]:.3f} in-sample versus
{temporal["test_c_index"]:.3f} out-of-time. Extending the test window to
{extended["test_years"]} ({extended["test_contracts"]:,} episodes,
{extended["test_events"]:,} events), without refitting, gives
{extended["test_c_index"]:.3f}. Both are close to the \(0.5\) chance line:
individualized out-of-time discrimination is weak. That is a result, not a
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
\caption{{Parametric survival model comparison.}}
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

\paragraph{{Detectability and selection diagnostic.}} Comparing linked and
censored episodes on standardized mean differences, the largest gap is
\code{{{latex_escape(survival_summary["selection_diagnostic"]["largest_absolute_smd"]["variable"])}}}
(SMD {survival_summary["selection_diagnostic"]["largest_absolute_smd"]["absolute_smd"]:.3f}).
This indicates possible differential detectability across episode
characteristics, not proof of causal linkage bias; it cannot be fully separated
from genuine heterogeneity in renewal behaviour using BOAMP alone.

\paragraph{{Linkage sensitivity.}} Event counts range from
{survival_summary["sensitivity"]["minimum_events"]:,} to
{survival_summary["sensitivity"]["maximum_events"]:,} across the four retained
linkage arms, so absolute probabilities are linkage-sensitive by construction.
Cox effects under the same four arms:
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
\caption{{Cox hazard ratios across linkage-definition sensitivity arms.}}
\end{{table}}
Framework flag, CPV-35, and centered award year are the most robust
associations; buyer-region and CPV-72 effects are linkage-sensitive and should
not be over-interpreted.

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
\caption{{Current trend signal matrix: OLS 12-quarter slope, PELT breaks, and
HMM current regime (Overall and top-2 segments only).}}
\end{{table}}
CPV-48 is the only segment with a statistically distinguishable 12-quarter
decline at the exploratory \(\alpha=0.10\) level; the rest are
\code{{stable\_or\_uncertain}} by this signal.

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

\section{{Defensible Decision}}
The final project decision is to keep \code{{M\_B\_text\_ranking @ 0.70}}
as the frozen conservative observable-successor baseline and use the stricter
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
downward while residual false links can push it upward. It is therefore a
linkage-conditioned indicator whose absolute value must be read with the strict,
looser, and weighted-gated sensitivity results and the borderline-band check.
Independent specialist review is required only before claiming externally
validated precision or promoting a new threshold, not to complete this
linkage-conditioned descriptive study.
If that review shows weaker precision, the claim should be narrowed further
rather than forcing a more complex model.

\section{{Limitations And Robustness}}
\begin{{itemize}}
\item BOAMP does not consistently encode legal renewal status; accepted links are
observable successor procurements, not legal renewal proof.
\item The reference labels were generated by a single LLM research pass and
spot-checked on a subset by the project owner, not verified anchor-by-anchor and
not reviewed by an independent specialist panel. Its negatives are
corpus-relative: roughly 25 candidates per anchor were considered, so the
reported false-positive rate is an upper bound.
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

\section{{Recommended Reporting Position}}
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

\section{{Current Source Files}}
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

## Current Decision

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

## Latest Reference State

- reviewed anchors: `{manifest["reviewed_anchors"]}`, of which `{manifest["remap"]["resolved_to_current_episodes"]}` resolve to a current episode;
- pilot split: `{pilot["usable_anchors"]}` usable anchors, `{pilot["positive_anchors"]}` with a reviewed successor;
- locked split: `{locked["usable_anchors"]}` usable anchors, `{locked["positive_anchors"]}` with a reviewed successor;
- pair rows: `{modeling["outputs"]["dev"]["rows"]:,}` pilot and `{modeling["outputs"]["validation"]["rows"]:,}` locked.

## Current Study State

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

The active reference for successor linkage is a stratified review of
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

## What It Replaced And Why

It replaced a France-level benchmark whose two annotation passes were both
emitted by deterministic rules in a single script, built from the same text,
CPV, and date evidence the linkage methods consume. A method could score well
there only by agreeing with that rule, so the numbers measured rule agreement
rather than correctness. Those artifacts have been removed from the repository
in full; their history remains in version control.

## Current Materialised State

- reviewed anchors: `{manifest["reviewed_anchors"]}`;
- resolved onto the current episode reconstruction: `{manifest["remap"]["resolved_to_current_episodes"]}`;
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

## Current Method Comparison On The Locked Split

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
- Retired an earlier France-level benchmark whose labels were generated by
  deterministic rules reading the same evidence the linkage methods use, which
  made its method comparison circular. It is archived and read by nothing.
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
  segments are stable or uncertain by the current 12-quarter signal.

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
  covers current national opportunities rather than the historical study cohort
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
   following the blinded-review design in
   `INDEPENDENT_LINK_REVIEW_PROTOCOL.md`. The scripts that once implemented that
   workflow were removed with the retired France-level benchmark and are
   recoverable from version control. Supplying the parallel classification
   work's training corpus and validation artifacts would be the cheaper route,
   if they can be obtained.
3. If a Gigalis-membership causal analysis is wanted, supply member identity
   and adoption-date data so the outlined staggered-adoption
   difference-in-differences design can actually be estimated.
4. Treat the current linkage, survival, and trend components as frozen; do
   not reopen them without new evidence, per `PROJECT_WORK_PROTOCOL.md`.

## Full Documentation

`README.md`, `FINAL_PIPELINE.md`, `reports/boamp_methodology_chapter.pdf`,
`SURVIVAL_ANALYSIS_REPORT.md`, `TREND_ANALYSIS_REPORT.md`,
`DATA_QUALITY_REPORT.md`, `INTERNSHIP_GUIDE_COMPLIANCE.md`.
"""
    path = PROJECT_ROOT / "EXECUTIVE_SUMMARY.md"
    path.write_text(text, encoding="utf-8")
    return path


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
            "# 12. Successor linkage and regional-reference evaluation\n\n"
            f"Generated: `{generated_at}`\n\n"
            "This notebook is regenerated from the current script outputs. It is the "
            "reader-facing linkage/evaluation notebook for the current evidence state."
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
        nbf.v4.new_markdown_cell("## Reference State"),
        nbf.v4.new_code_cell(
            "pd.DataFrame([\n"
            "    {'item': 'reviewed anchors', 'value': manifest['reviewed_anchors']},\n"
            "    {'item': 'resolved to current episodes', 'value': manifest['remap']['resolved_to_current_episodes']},\n"
            "    {'item': 'pilot usable anchors', 'value': manifest['splits']['dev']['usable_anchors']},\n"
            "    {'item': 'pilot positive anchors', 'value': manifest['splits']['dev']['positive_anchors']},\n"
            "    {'item': 'locked usable anchors', 'value': manifest['splits']['validation']['usable_anchors']},\n"
            "    {'item': 'locked positive anchors', 'value': manifest['splits']['validation']['positive_anchors']},\n"
            "    {'item': 'candidate recall ceiling', 'value': manifest['candidate_reachability']['candidate_generation_recall_ceiling']},\n"
            "])"
        ),
        nbf.v4.new_markdown_cell("## Held-Out Method Comparison On The Locked Split"),
        nbf.v4.new_code_cell(
            "display(validation_methods[['method', 'threshold', 'accepted_links', 'precision', 'precision_ci', 'recall', 'recall_ci', 'fpr', 'coverage']])\n\n"
            "ax = validation_methods.set_index('method')[['precision', 'recall', 'fpr']].plot(\n"
            "    kind='bar', figsize=(9, 4.5), width=0.72\n"
            ")\n"
            "ax.set_title('Regional reference: locked split')\n"
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
            "of every method scored here, which the retired France-level benchmark's "
            "rule-generated labels were not, but they were not verified "
            "anchor-by-anchor and are not an independent specialist panel. "
            "Negatives are corpus-relative: roughly 25 candidates per anchor were "
            "considered, so the false-positive rate is an upper bound. These are "
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
    write_notebook(generated_at)
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
