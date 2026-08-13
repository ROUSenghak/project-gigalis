# BOAMP Successor Procurement Study: Formal Work Protocol

Protocol date: `2026-08-13`  
Data cutoff: `2025-12-31`  
Current status: **complete for linkage-conditioned descriptive reporting; external accuracy validation not established**

## 1. Research Objective

The project studies whether an awarded digital public-procurement episode is
followed by a later, observable procurement episode that plausibly continues or
replaces the same need.

The primary research question is:

> Among awarded digital procurement episodes in Grand Ouest, how long is it
> until an observable successor procurement is published in BOAMP?

The project does not attempt to certify legal contract renewal. The observable
event is a procurement-data proxy because BOAMP does not consistently provide a
complete legal-renewal label.

## 2. Formal Units And Notation

Let:

- $n$ denote a BOAMP notice;
- $i$ denote an earlier awarded procurement episode, called the anchor;
- $j$ denote a later candidate episode;
- $A_i$ denote the anchor award-date origin;
- $C_j$ denote the candidate first-publication date;
- $T_{ij}$ denote TF-IDF cosine text similarity;
- $Y_i=1$ denote an accepted observable successor and $Y_i=0$ an abstention;
- $\tau_i$ denote observed or censored time from $A_i$.

The analytical grains are fixed:

| Layer | One row represents |
|---|---|
| Standardised notices | One official BOAMP notice |
| Procurement episodes | One reconstructed procurement process |
| Study cohort | One awarded Grand Ouest digital episode |
| Candidate table | One anchor-candidate pair |
| Survival table | One cohort episode with an event or censoring time |

## 3. Fixed End-To-End Workflow

```text
Official BOAMP notices, 2015-2025
  -> schema-aware standardisation
  -> procurement episode reconstruction
  -> Grand Ouest awarded digital cohort
  -> same-buyer, future-time candidate generation
  -> four-method development comparison
  -> one selected successor or abstention
  -> expiry-aware sensitivity audit
  -> right-censored survival dataset
  -> survival and descriptive trend analyses
```

### 3.1 Official Source And Standardisation

The input is the official BOAMP API. Raw values and source lineage are
preserved. Dates, CPVs, buyer identifiers, buyer names, text, framework flags,
and explicit duration fields are standardised with schema-aware parsers.

Rules:

- no assumed four-year duration;
- no global duration imputation;
- validated SIREN evidence outranks name similarity;
- conflicting validated SIRENs cannot be treated as the same buyer;
- municipality and intercommunal authority legal forms are not automatically
  merged.

### 3.2 Procurement Episode Reconstruction

Multiple notices can describe one procurement process, such as a consultation,
correction, award, or lot-level notice. These notices are grouped into an
episode before successor linkage so administrative republication is not counted
as a new procurement event.

Episode reconstruction is an inferred transformation, not a BOAMP-provided
identifier. Structural integrity tests are complete; a compact semantic
spot-check remains desirable.

### 3.3 Study Cohort

The cohort contains `3,800` awarded Grand Ouest digital procurement episodes.
Digital scope is defined reproducibly through CPV divisions `32`, `35`, `48`,
and `72`. This is a coarse procurement-domain segmentation, not a trained
technology taxonomy classifier.

### 3.4 Candidate Generation

A candidate is exposed only when buyer evidence is plausible, validated SIRENs
do not conflict, and:

\[
A_i+90\text{ days} \le C_j \le A_i+2920\text{ days}.
\]

The 90-2,920 day interval is an operational search window, not an assumed
contract duration or statutory rule. It produces `763,417` candidate pairs for
`3,520` of the `3,800` anchors.

### 3.5 Linkage Algorithms

Four methods are compared on the same exposed candidate set:

| Method | Role | Main idea |
|---|---|---|
| `M_A_deterministic` | Conservative comparator | Requires explicit buyer, CPV, and minimum text evidence |
| `M_B_text_ranking` | Provisional primary method | Selects the highest TF-IDF cosine candidate and accepts at `T >= 0.70` |
| `M_C_weighted_gated` | Higher-recall comparator | Combines buyer, text, CPV, and timing evidence with gates |
| `M_D_fellegi_sunter` | Probabilistic comparator | Uses match/non-match comparison likelihood ratios |

The fixed primary decision is:

\[
\hat{j}_i=\arg\max_j T_{ij},
\qquad
Y_i=\mathbf{1}\!\left(T_{i\hat{j}_i}\ge0.70\right).
\]

At most one candidate is selected for each anchor. If no candidate qualifies,
the method abstains.

### 3.6 Expiry-Aware Sensitivity

Expected expiry is calculated only from valid explicit end-date or duration
evidence. Missing duration remains missing. Very early candidates require text
similarity of at least `0.85` plus positive CPV continuity; other candidates use
the `0.70` text rule. This method is an audit arm and is not the primary event
definition.

### 3.7 Survival Construction

For an accepted candidate $\hat{j}_i$:

\[
\tau_i=C_{\hat{j}_i}-A_i,
\qquad Y_i=1.
\]

If no successor is accepted by `2025-12-31`, the row is administratively
right-censored:

\[
\tau_i=\text{2025-12-31}-A_i,
\qquad Y_i=0.
\]

Kaplan-Meier estimates describe the linkage-conditioned survivor function. Cox
models describe covariate associations with the observed event hazard; they are
not causal effects. Strict, main, loose, weighted-gated, and expiry-aware event
definitions are retained as sensitivity analyses.

### 3.8 Descriptive Trend Analysis

Quarterly awarded-episode counts are analysed from `2015Q2` through `2025Q4`.
The partial `2015Q1` extract is excluded. PELT identifies candidate change
points under penalty sensitivity, while 12-quarter linear slopes describe
recent direction. Neither method supplies a causal explanation. Monetary trend
analysis is omitted because no canonical awarded-amount field has been
validated at episode grain.

## 4. Current Materialised Results

| Result | Current value | Interpretation |
|---|---:|---|
| Standardised notices | `1,620,712` | Unique official notice records |
| Reconstructed episodes | `1,103,632` | Inferred procurement processes |
| Study cohort | `3,800` | Awarded Grand Ouest digital episodes |
| Candidate pairs | `763,417` | Broad exposed comparison set |
| Main accepted successors | `544` | `M_B @ 0.70` observable events |
| Main observed event rate | `14.32%` | Linkage-conditioned, not legal renewal prevalence |
| Median successor time | `31.82 months` | Median among accepted events only |
| Expiry-aware accepted links | `504` | Sensitivity result |
| Expiry-aware changed anchors | `42` | Existing audit queue |

Event-definition sensitivity is material:

| Rule | Events | Event rate | Median accepted-event time |
|---|---:|---:|---:|
| `M_B @ 0.80` | `296` | `7.79%` | `35.71` months |
| `M_B @ 0.70` | `544` | `14.32%` | `31.82` months |
| `M_B @ 0.60` | `853` | `22.45%` | `26.58` months |
| `M_C @ 0.70` | `1,332` | `35.05%` | `26.09` months |

Therefore, no single event rate is treated as exact truth or as a formal lower
bound.

## 5. Evaluation Status

The national development reference contains `252` anchors and `7,031` labelled
pairs. Its two passes were generated by the same deterministic bootstrap rules.
Consequently, their agreement measures rule repeatability rather than human
inter-annotator agreement.

On its held-out internal split, `M_B @ 0.70` has:

- exact-successor true positives: `4`;
- accepted links: `5`;
- precision: `0.800`;
- recall: `0.1818`;
- negative-anchor false-positive rate: `0.000`.

These are development diagnostics. They are not independently validated
accuracy estimates, and the five accepted decisions make precision
sample-sensitive.

## 6. Review Audit Protocol

A blinded `60`-pair challenge set was prepared:

- `20` current `M_B @ 0.70` accepted links;
- `20` high-similarity structural negatives;
- `20` resolved buyer-declared relationships.

The reviewer sees buyer fields, CPVs, descriptions, and dates, but not model
scores, bootstrap labels, or sampling strata. A model-assisted diagnostic review
has now been completed. It found 14 confirmed successors, 5 non-successors, and
1 uncertain case among the 20 sampled accepted links. Precision was `14/19 =
0.7368` excluding uncertainty (exact 95% CI `[0.4880, 0.9085]`) and `14/20 =
0.7000` conservatively (exact 95% CI `[0.4572, 0.8811]`). This does not meet the
`0.80` point target and does not replace independent human validation.

### 6.1 Optional Stronger Validation Path

1. Freeze `M_B @ 0.70` and the current sample.
2. Have a procurement-domain reviewer label all `60` pairs using
   `INDEPENDENT_LINK_REVIEW_PROTOCOL.md`.
3. Keep the audit key hidden until all labels and notes are final.
4. Compute accepted-stratum precision and an exact 95% binomial confidence
   interval from the `20` accepted-link rows.
5. Report structural-negative and buyer-declared results separately. The mixed
   challenge set does not estimate national prevalence or national FPR.
6. Inspect every error for buyer-identity, same-procurement, parallel-lot,
   different-domain, text, CPV, and timing failure modes.

### 6.2 Optional Double-Review Path

Have a second independent reviewer label a balanced overlap of at least `15`
pairs. Report raw agreement and Cohen's kappa for the binary observable-successor
decision, with `UNCERTAIN` reported separately. This is preferable but not
required to complete the minimum specialist audit.

### 6.3 Decision Rule After Review

- Retain `M_B @ 0.70` when reviewed accepted-link precision is at least `0.80`
  and no systematic high-risk identity failure is found.
- Retain it only as provisional when the point estimate is acceptable but its
  confidence interval is too wide for a strong claim.
- Recalibrate when precision is below `0.80` or errors reveal a systematic
  failure mode.
- Do not tune on these `60` rows and report performance on the same rows as new
  validation. Any revised method requires separate development and fresh
  holdout evidence.

## 7. Quality Gates

| Gate | State |
|---|---|
| Raw-to-standardised row integrity | Passed |
| Unique notice identifiers | Passed |
| Episode membership integrity | Passed |
| Candidate chronology and no self-links | Passed |
| Conflicting validated SIREN exclusion | Passed |
| Maximum one selected successor per anchor | Passed |
| Survival rows = events + censored | Passed |
| No negative survival durations | Passed |
| Benchmark buyer-level split isolation | Passed |
| Sealed-split protection | Passed |
| Evidence notebooks execute | Passed |
| Automated test suite | `161 passed` |
| Model-assisted linkage diagnostic | Complete; `14/20` conservatively confirmed |
| Independent specialist linkage validity | Not established; required for stronger external accuracy claims, not for the current descriptive scope |

Passing software tests establishes implementation consistency. It does not
establish real-world linkage accuracy.

## 8. Permitted Final Claims

The final report may state that:

- the pipeline is reproducible and structurally validated;
- it identifies observable successor procurements rather than legal renewals;
- the main operating rule accepts `544` successors in a `3,800`-episode cohort;
- survival estimates are conditional on the linkage rule;
- the current internal reference supports `M_B @ 0.70` as a provisional,
  precision-first baseline;
- `M_B @ 0.60` performs better on the small bootstrap validation split but is
  retained as sensitivity evidence because development and production-review
  evidence do not justify post-hoc promotion;
- trend findings are descriptive and non-causal.

The report must not state that:

- the `544` links are confirmed legal renewals;
- `0.800` is independently validated precision;
- the bootstrap passes are independent human annotations;
- missing contracts are four years long;
- observed statistical breaks have known causes;
- the project implements a supervised technology taxonomy classifier;
- Cox hazard ratios are causal or provide validated individual forecasts.

## 9. Formal Completion Definition

The current computational work is complete when the pipeline, tests, notebooks,
and reports reproduce from the documented commands. That gate is passed.

The linkage study is ready for defensible final reporting now, provided its
claim remains linkage-conditioned and descriptive. The model-assisted audit
estimates, intervals, and failure classes are already reported. Independent
specialist review is future work required before claiming externally validated
accuracy or recalibrating the threshold, not a condition for completing the
current internship analysis.

## 10. Current Sources Of Truth

- `FINAL_PIPELINE.md`
- `reports/current_project_readiness_report.html`
- `reports/boamp_methodology_chapter.pdf`
- `DATA_QUALITY_REPORT.md`
- `QUALITY_EVIDENCE.md`
- `TREND_ANALYSIS_REPORT.md`
- `INTERNSHIP_GUIDE_COMPLIANCE.md`
- `METHODOLOGICAL_REFERENCES.md`
- `INDEPENDENT_LINK_REVIEW_PROTOCOL.md`
- `data/processed/boamp/final_pipeline_manifest.json`
