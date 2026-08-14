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
BENCHMARK = PROCESSED / "benchmark"
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
    ax.set_title("Current national development-reference tables")
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
    candidates = load_json(PROCESSED / "linkage_candidates_summary.json")
    survival = load_json(PROCESSED / "survival_dataset_summary.json")
    buyer_audit = load_json(PROCESSED / "buyer_blocking_legal_form_audit_summary.json")
    expiry = load_json(PROCESSED / "expiry_aware_linkage_summary.json")
    review_audit = load_json(PROJECT_ROOT / "data/review/review_audit_evaluation.json")
    survival_main = survival["variants"]["main"]
    survival_strict = survival["variants"]["strict"]
    survival_looser = survival["variants"]["looser"]
    survival_contrast = survival["variants"]["contrast_high_recall"]
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
\title{{BOAMP Successor Linkage Methodology\\\large Current Evidence State}}
\author{{BOAMP Data Science Internship Project}}
\date{{{generated_at}}}
\begin{{document}}
\sloppy
\maketitle

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

The current national development reference contains {labelled_anchors}
bootstrap-labelled anchors and {labelled_pairs:,} labelled anchor-candidate
rows. On its held-out internal split, \code{{M\_B\_text\_ranking @ 0.70}}
has precision {m_b.precision:.3f},
recall {m_b.recall:.3f}, false-positive rate {m_b.fpr:.3f}, and
{int(m_b.accepted_links)} accepted links. These are internal protocol-reference
estimates, not independently validated accuracy estimates, because both label
passes were generated by the deterministic bootstrap rules in
\pathcode{{scripts/auto\_annotate\_wave1a.py}}.
\code{{M\_C\_weighted\_gated}} recovers more positives
(recall {m_c.recall:.3f}) but admits more false positives
(FPR {m_c.fpr:.3f}). This trade-off is important because a false positive in a
survival dataset creates both a false event and a false event time, while an
abstained link is handled conservatively as right-censoring.

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

\section{{Current National Development Reference}}
The current reference is national rather than Grand Ouest-only. It samples awarded
digital procurement episodes across French region groups and CPV divisions.
The dev split has {modeling["outputs"]["dev"]["anchors"]} anchors and
{modeling["outputs"]["dev"]["rows"]:,} pair rows; the validation split has
{modeling["outputs"]["validation"]["anchors"]} anchors and
{modeling["outputs"]["validation"]["rows"]:,} pair rows. Only the probability
frame supports national weighted estimates; enrichment frames are used for
stress cases and diagnostics.

The annotation caveat is decisive: labels were produced by deterministic
bootstrap rules. Pass A and pass B are re-presentations of the same rules, so
their agreement measures repeatability rather than independent annotator
agreement. The reference is useful for development, stress testing, and split
discipline, but it is not official legal renewal truth or independent specialist
ground truth.

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

\paragraph{{\(M_E\): expiry-aware audit arm.}}
The expiry-aware method runs in parallel with the primary method. It estimates
an expected end date \(E_i\) only from explicit end-date or duration evidence.
It does not assume a four-year duration. Candidate timing is
\[
R_{{ij}}=C_j-E_i.
\]
Very early candidates require stronger text evidence and CPV continuity. This
arm accepts {expiry["cohort_comparison"]["expiry_aware"]["accepted_links"]:,} links, compared with
{survival_main["validation"]["events"]:,} under the main method. It is retained
as an audit/sensitivity check, not promoted, because observed market behaviour
shows many declared successors can be published before expected expiry.

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
\caption{{Unweighted held-out metrics on all bootstrap-labelled reference frames.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{figures/benchmark_validation_method_metrics.png}}
\caption{{Held-out internal method comparison generated from the development-reference JSON.}}
\end{{figure}}

\section{{Quality Evidence And Interpretation}}
The held-out internal result supports retaining a conservative baseline for
continued work; it does not establish final accuracy. \code{{M\_B}} has
only {int(m_b.accepted_links)} accepted held-out links, so its precision
estimate is necessarily sample-sensitive; one changed decision would move the
number materially. Within this bootstrap reference, the direction is coherent:
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
bootstrap labels, numerical results, or selected threshold. Generic web
illustrations are therefore treated as presentation aids only, not as academic
evidence.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.99\textwidth]{{figures/benchmark_validation_m_b_threshold_tradeoff.png}}
\caption{{Project-specific, unsmoothed anchor-level threshold trade-off for
\code{{M\_B}}. The figure is empirical evidence from the internal validation
reference, not an idealised illustration.}}
\end{{figure}}

At threshold $0.60$, this validation sample contains 8 correct successors
among 9 accepted links, compared with 4 among 5 at $0.70$. Thus $0.60$
performs better on this particular sample. Development evidence points in the
opposite direction: precision is 0.800 and FPR is 0.058 at $0.60$, compared
with precision 0.875 and FPR 0.035 at $0.70$. The lower threshold is therefore
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
{survival_main["validation"]["censored"]:,} censored observations. Segment
event rates are {survival_main["description"]["by_digital_segment"]["CPV-32"]["event_rate"]:.3f}
for CPV-32, {survival_main["description"]["by_digital_segment"]["CPV-35"]["event_rate"]:.3f}
for CPV-35, {survival_main["description"]["by_digital_segment"]["CPV-48"]["event_rate"]:.3f}
for CPV-48, and {survival_main["description"]["by_digital_segment"]["CPV-72"]["event_rate"]:.3f}
for CPV-72. CPV-35 has the highest observed successor rate in the main arm.

Sensitivity checks show that absolute event rates depend strongly on the
linkage rule: {survival_strict["validation"]["events"]:,} events under
\code{{M\_B @ 0.80}}, {survival_main["validation"]["events"]:,} under
\code{{M\_B @ 0.70}}, {survival_looser["validation"]["events"]:,} under
\code{{M\_B @ 0.60}}, and {survival_contrast["validation"]["events"]:,} under
\code{{M\_C @ 0.70}}. Therefore the survival estimates should be interpreted as
linkage-conditioned descriptive evidence, not as exact renewal probabilities.

\section{{Defensible Decision}}
The final project decision is to keep \code{{M\_B\_text\_ranking @ 0.70}}
as the frozen conservative observable-successor baseline and use the stricter,
looser, weighted-gated, and expiry-aware variants as required sensitivity
analyses. The threshold is not claimed to be optimal: $0.60$ performs better
on the small bootstrap validation split, while $0.70$ has better precision and
false-positive control on development evidence. Moreover, the completed
model-assisted review confirmed only 14 of 20 sampled production links at
$0.70$ conservatively, so lowering an unreviewed threshold would not support a
stronger accuracy claim. This is a precision-first design. Low recall is
accepted as an explicit trade-off, but the
observed event rate is not a mathematical lower bound: missed successors push it
downward while residual false links can push it upward. It is therefore a
linkage-conditioned indicator whose absolute value must be read with the strict,
looser, weighted-gated, and expiry-aware sensitivity results.
Independent specialist review is required only before claiming externally
validated precision or promoting a new threshold, not to complete this
linkage-conditioned descriptive study.
If that review shows weaker precision, the claim should be narrowed further
rather than forcing a more complex model.

\section{{Limitations And Robustness}}
\begin{{itemize}}
\item BOAMP does not consistently encode legal renewal status; accepted links are
observable successor procurements, not legal renewal proof.
\item The current reference labels come from deterministic bootstrap rules;
pass agreement is self-consistency, not independent human agreement.
\item The held-out reference is small:
{validation["methods"][1]["unweighted_all_frames"]["positive_anchors"]}
positive anchors and {int(m_b.accepted_links)} accepted \code{{M\_B}} links.
The selected threshold should not be over-tuned and its accuracy remains provisional.
\item Buyer standardisation is improved but remains an important risk area. The
legal-form audit is retained to catch name-only and cross-legal-form cases.
\item Missing duration and expiry fields are not imputed. This avoids creating
false timing certainty.
\end{{itemize}}

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
\item \pathcode{{data/processed/boamp/benchmark/modeling/modeling\_summary.json}}
\item \pathcode{{data/processed/boamp/benchmark/benchmark\_manifest.json}}
\item \pathcode{{data/processed/boamp/survival\_dataset\_summary.json}}
\item \pathcode{{data/processed/boamp/linkage\_candidates\_summary.json}}
\item \pathcode{{DATA\_QUALITY\_REPORT.md}}
\item \pathcode{{TREND\_ANALYSIS\_REPORT.md}}
\item \pathcode{{PROJECT\_WORK\_PROTOCOL.md}}
\item \pathcode{{REVIEW\_AUDIT\_RESULTS.md}}
\end{{itemize}}

\section{{Methodological References}}
The source and identity definitions follow the official BOAMP and INSEE
documentation. CPV interpretation follows Commission Regulation (EC) No
213/2008. TF--IDF cosine similarity follows the standard vector-space
definition. Record linkage, survival estimation, proportional-hazards
diagnostics, and change-point detection are supported by the original
Fellegi--Sunter (1969), Davis--Goadrich (2006), Saito--Rehmsmeier (2015),
Kaplan--Meier (1958), Cox (1972), Grambsch--Therneau (1994), and PELT
(Killick et al., 2012) methods. Full URLs and the specific
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
    application = load_json(PROCESSED / "linkage_application_summary.json")
    survival = load_json(PROCESSED / "survival_dataset_summary.json")["variants"]["main"]
    expiry = load_json(PROCESSED / "expiry_aware_linkage_summary.json")
    m_b = validation_frame.loc[validation_frame["method"].eq("M_B_text_ranking")].iloc[0]
    final_pipeline = f"""# Final Defensible Pipeline

Generated: `{generated_at}`

## Current Decision

The final primary event definition is `M_B_text_ranking @ 0.70`. It is a frozen
conservative baseline, not a claim that `0.70` is the optimal threshold.

On the held-out split of the current national development reference it gives:

- precision@1: `{m_b.precision:.3f}`;
- recall@1: `{m_b.recall:.3f}`;
- false-positive rate on negative anchors: `{m_b.fpr:.3f}`;
- accepted links: `{int(m_b.accepted_links)}`.

These are internal bootstrap-reference estimates, not independent validation
results. Both annotation passes were generated by the deterministic rules in
`scripts/auto_annotate_wave1a.py`.

Threshold `0.60` performs better on this particular validation sample, but
development evidence favours `0.70` for precision and false-positive control.
The completed production-link diagnostic at `0.70` confirmed only `14/20`
sampled links conservatively, so the unreviewed lower threshold is not promoted
post hoc. It remains a required survival sensitivity arm.

`M_C_weighted_gated` has higher recall but also higher false-positive risk.
`M_D_fellegi_sunter` is evaluated on the current benchmark, but it does not outperform `M_B`.

## End-to-End Workflow

```text
Official BOAMP API, 2015-2025
  -> schema-aware standardisation
  -> procurement episode reconstruction
  -> Grand Ouest digital study cohort
  -> broad same-buyer candidate generation
  -> four linkage algorithms compared on the national development reference
  -> M_B primary successor selection
  -> survival dataset and expiry-aware sensitivity audit
```

The event remains an **observable successor procurement**, not a confirmed legal
renewal.

## Latest Benchmark State

- bootstrap-labelled anchors: `{labelled_anchors}`;
- bootstrap-labelled pairs: `{labelled_pairs:,}`;
- dev: `{modeling["outputs"]["dev"]["anchors"]}` anchors and `{modeling["outputs"]["dev"]["rows"]:,}` pair rows;
- validation: `{modeling["outputs"]["validation"]["anchors"]}` anchors and `{modeling["outputs"]["validation"]["rows"]:,}` pair rows;
- sealed test: not used for method selection.

## Current Study State

- cohort episodes: `{survival["validation"]["rows"]:,}`;
- candidate pairs: `{load_json(PROCESSED / "linkage_candidates_summary.json")["candidate_pairs"]:,}`;
- primary accepted links: `{application["cohort_application"]["accepted_links"]}`;
- primary cohort event rate: `{survival["description"]["event_rate"]:.4f}`;
- expiry-aware accepted links: `{expiry["cohort_comparison"]["expiry_aware"]["accepted_links"]}`;
- expiry-aware changed primary anchors: `{expiry["cohort_comparison"]["changed_anchors_for_review"]}`.

## Canonical Outputs

- `data/processed/boamp/`
- `data/processed/boamp/benchmark/`
- `data/processed/boamp/linkage_evaluation_dev.json`
- `data/processed/boamp/linkage_evaluation_validation.json`
- `data/processed/boamp/benchmark/modeling/modeling_summary.json`
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

    national = f"""# National Development Reference

Generated: `{generated_at}`

## Purpose

The current national reference is a France-wide bootstrap dataset for developing,
debugging, and provisionally comparing observable-successor linkage algorithms.

It does not prove legal renewal and it is not independent ground truth. Both
annotation passes were generated by the deterministic rules in
`scripts/auto_annotate_wave1a.py`.

## Current Materialised State

- labelled anchors: `{labelled_anchors}`;
- labelled pairs: `{labelled_pairs:,}`;
- dev split: `{modeling["outputs"]["dev"]["anchors"]}` anchors, `{modeling["outputs"]["dev"]["rows"]:,}` pair rows, `{modeling["outputs"]["dev"]["primary_positive_pairs"]}` primary-positive pairs;
- validation split: `{modeling["outputs"]["validation"]["anchors"]}` anchors, `{modeling["outputs"]["validation"]["rows"]:,}` pair rows, `{modeling["outputs"]["validation"]["primary_positive_pairs"]}` primary-positive pairs;
- sealed test: closed for method selection.

## Current Method Comparison

The current internal comparison includes all four methods. `M_D_fellegi_sunter` is no
longer skipped because `fs_match_probability` is now computed from the fitted
Fellegi-Sunter model when the current benchmark exposure is evaluated.

| Method | Threshold | Internal precision | Internal recall | Internal FPR | Accepted |
|---|---:|---:|---:|---:|---:|
"""
    for row in validation_frame.itertuples(index=False):
        national += (
            f"| `{row.method}` | {row.threshold:.1f} | {row.precision:.3f} | "
            f"{row.recall:.3f} | {row.fpr:.3f} | {int(row.accepted_links)} |\n"
        )
    national += """
## Decision Rule

The incumbent `M_B_text_ranking @ 0.70` remains the frozen conservative primary
event definition, not an empirically optimal threshold. `0.60` performs better
on this small validation split but worse on development precision and
false-positive control. A replacement requires a pre-specified selection rule,
direct review of the incremental links, and fresh evaluation evidence.

## Caveat

The current labels are deterministic bootstrap development evidence. Pass A and
pass B are not independent annotations; their agreement is self-consistency.
The resulting metrics must not be described as official legal renewal truth,
independent specialist validation, or human inter-annotator agreement.
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
            "# 12. Successor linkage and current development-reference evaluation\n\n"
            f"Generated: `{generated_at}`\n\n"
            "This notebook is regenerated from the current script outputs. It is the "
            "reader-facing linkage/evaluation notebook for the current evidence state."
        ),
        nbf.v4.new_markdown_cell(
            "## tl;dr\n\n"
            "`M_B_text_ranking @ 0.70` is the frozen conservative primary event "
            "definition, not a claim of threshold optimality. The latest "
            "internal held-out comparison includes all four algorithms, including "
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
            "PROCESSED = PROJECT_ROOT / 'data/processed/boamp'\n"
            "BENCHMARK = PROCESSED / 'benchmark'\n\n"
            "def load_json(path):\n"
            "    with open(path, 'r', encoding='utf-8') as f:\n"
            "        return json.load(f)\n\n"
            "dev = load_json(PROCESSED / 'linkage_evaluation_dev.json')\n"
            "validation = load_json(PROCESSED / 'linkage_evaluation_validation.json')\n"
            "modeling = load_json(BENCHMARK / 'modeling/modeling_summary.json')\n"
            "manifest = load_json(BENCHMARK / 'benchmark_manifest.json')\n"
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
        nbf.v4.new_markdown_cell("## Internal Held-Out Method Comparison"),
        nbf.v4.new_code_cell(
            "display(validation_methods[['method', 'threshold', 'accepted_links', 'precision', 'recall', 'fpr', 'coverage']])\n\n"
            "ax = validation_methods.set_index('method')[['precision', 'recall', 'fpr']].plot(\n"
            "    kind='bar', figsize=(9, 4.5), width=0.72\n"
            ")\n"
            "ax.set_title('Current internal reference metrics')\n"
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
            "event time. Threshold `0.60` performs better on the small bootstrap "
            "validation split but worse on development precision and FPR, so it remains "
            "a sensitivity arm rather than being promoted post hoc. The use of "
            "precision-recall evidence for this rare-positive "
            "decision follows [Davis and Goadrich (2006)](https://doi.org/10.1145/1143844.1143874) "
            "and [Saito and Rehmsmeier (2015)](https://doi.org/10.1371/journal.pone.0118432). "
            "Those papers support the diagnostic choice, not this project's numerical results."
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
            "The current labels were generated by deterministic bootstrap rules. "
            "The two passes are not independent annotations, so their agreement is "
            "self-consistency rather than specialist inter-annotator agreement. "
            "These metrics are development evidence, not validated legal-renewal accuracy."
        ),
    ]
    nb["cells"] = cells
    nbf.write(nb, NOTEBOOK)


def main() -> int:
    generated_at = datetime.now().isoformat(timespec="seconds")
    dev = load_json(PROCESSED / "linkage_evaluation_dev.json")
    validation = load_json(PROCESSED / "linkage_evaluation_validation.json")
    modeling = load_json(BENCHMARK / "modeling/modeling_summary.json")
    manifest = load_json(BENCHMARK / "benchmark_manifest.json")

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_method_metrics(
        method_frame(dev),
        "Current dev metrics",
        FIGURES / "benchmark_dev_method_metrics.png",
    )
    plot_method_metrics(
        method_frame(validation),
        "Current internal reference metrics",
        FIGURES / "benchmark_validation_method_metrics.png",
    )
    plot_modeling_counts(modeling, FIGURES / "benchmark_modeling_counts.png")
    report_path = write_methodology_report(dev, validation, modeling, manifest, generated_at)
    pdf_path = compile_methodology_pdf(report_path)
    write_status_files(dev, validation, modeling, manifest, generated_at)
    write_notebook(generated_at)

    print(
        json.dumps(
            {
                "generated_at": generated_at,
                "report": str(report_path.relative_to(PROJECT_ROOT)),
                "report_pdf": str(pdf_path.relative_to(PROJECT_ROOT)),
                "notebook": str(NOTEBOOK.relative_to(PROJECT_ROOT)),
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
