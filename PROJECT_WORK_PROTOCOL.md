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

### 3.6 The Removed Duration-Conditioned Arm

A duration-conditioned linkage variant was built and evaluated during
development. It derived an expected end date from explicit end-date or duration
evidence only, never assuming a four-year duration, and required unusually early
candidates to clear stronger text and CPV-continuity evidence.

It was never the primary event definition, and it has now been **removed from the
repository in full**: its module, its two scripts, its tests, and all five of its
materialised outputs. Reliable duration is missing for `74.9%` of the cohort, so
the rule could differentiate itself only on a minority of episodes, and where the
evidence does exist the observed data show many declared successors published
well before the declared end date. Varying the acceptance threshold and the
scoring method moves the event set along the dimension that matters; varying a
duration assumption that is absent three times in four does not.
`scripts/validate_canonical_state.py` fails the build if any of it reappears. The
history remains in version control.

This removal does not touch the separate descriptive comparison between declared
duration and observed successor delay, which remains part of the survival
evidence (notebook 13). That diagnostic measures duration reliability; it is not
a linkage algorithm.

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

Kaplan-Meier estimates describe the linkage-conditioned survivor function and
are the source of the operational 12/24-month conditional probabilities, because
every reported horizon falls inside the observed window. Parametric families are
compared on AIC/BIC and the generalized gamma is reported as the best fit, but it
is the instrument for extrapolation beyond the window, not the source of the
operational numbers. Cox models describe covariate associations with the observed
event hazard; they are not causal effects.

Strict (`M_B @ 0.80`), main (`M_B @ 0.70`), loose (`M_B @ 0.60`), and
weighted-gated (`M_C @ 0.70`) event definitions are the retained sensitivity
arms. A fixed `±0.05` borderline band around the frozen threshold is additionally
excluded as a robustness check.

Neither the observed event rate nor any survival probability is a formal lower
bound on true re-procurement: missed successors push the measured level down and
residual false links push it up, so the net direction is not identified. They are
linkage-conditioned estimates.

### 3.8 Descriptive Trend Analysis

Quarterly awarded-episode counts are analysed from `2015Q2` through `2025Q4`.
The partial `2015Q1` extract is excluded. PELT identifies candidate change
points under penalty sensitivity, while 12-quarter linear slopes describe
recent direction. ADF/KPSS stationarity diagnostics are computed per segment,
and a 3-state Gaussian HMM fit on the quarter-over-quarter change in episode
count gives a current growth/plateau/decline regime label with posterior
probability for the overall series and the two highest-volume CPV segments.
Neither PELT, the HMM, nor the OLS slope supplies a causal explanation, and the
HMM's current-regime read is not forced to agree with the PELT/OLS signals.
Monetary trend analysis is omitted because no canonical awarded-amount field
has been validated at episode grain.

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
| Borderline-band episodes excluded | `280` | `133` events, `147` censored |
| Cox C-index, 2022-2024 out-of-time | `0.479` | Guideline-aligned primary validation |
| Cox C-index, 2022-2025 out-of-time | `0.518` | Sensitivity, adds the 2025 cohort |

Event-definition sensitivity is material:

| Rule | Events | Event rate | Median accepted-event time |
|---|---:|---:|---:|
| `M_B @ 0.80` | `296` | `7.79%` | `35.71` months |
| `M_B @ 0.70` | `544` | `14.32%` | `31.82` months |
| `M_B @ 0.60` | `853` | `22.45%` | `26.55` months |
| `M_C @ 0.70` | `1,332` | `35.05%` | `26.09` months |

Therefore, no single event rate is treated as exact truth or as a formal lower
bound.

Excluding the `280` episodes inside the borderline band leaves both headline
hazard ratios on the same side of 1 (CPV-35 `1.55` to `1.78`; framework `1.75` to
`1.62`) while the absolute KM level falls. The comparative claims are therefore
robust to near-threshold linkage decisions; the absolute level is not, which the
four-arm table already establishes.

## 5. Evaluation Status

The active reference is the Grand Ouest regional review: `120` anchors reviewed
against real BOAMP notices on 2026-08-11, of which `112` re-resolve onto the
current episode reconstruction and `88` are usable. It replaced a France-level
benchmark whose two annotation passes were both generated by deterministic rules
reading the same text, CPV, and date evidence the linkage methods consume; that
construction made the method comparison circular, and its artifacts have been
removed from the repository in full.

Two anchor counts appear for the locked split and the difference is by
construction: `72` anchors are evaluable at anchor level, of which `69` also have
at least one exposed candidate pair and so appear in the pair-level table. The
other `3` generated no candidate, which is a blocking-stage loss counted against
recall. Anchor-level metrics use `72`; pair-level ROC and precision-recall curves
use `69`.

On the locked split (`72` usable anchors, `18` with a reviewed successor),
`M_B @ 0.70` has:

- exact-successor true positives: `7`;
- accepted links: `8`;
- precision: `0.875` (95% CI `[0.529, 0.978]`);
- recall: `0.389` (95% CI `[0.203, 0.614]`);
- negative-anchor false-positive rate: `0.000`.

Recall is capped at `0.913` by candidate generation, which reaches `21` of the
`23` reviewed successors. These labels are independent of every method scored,
but they were generated by a single LLM research pass and spot-checked on a
subset rather than verified anchor-by-anchor or judged by an independent
specialist panel, and the negatives are corpus-relative, so the false-positive
rate is an upper bound. Eight accepted decisions make precision
sample-sensitive.

## 6. Review Audit Protocol

A blinded `60`-pair challenge set was prepared:

- `20` current `M_B @ 0.70` accepted links;
- `20` high-similarity structural negatives;
- `20` resolved buyer-declared relationships.

The reviewer sees buyer fields, CPVs, descriptions, and dates, but not model
scores, reference labels, or sampling strata. This sample is frozen: it was
drawn once, under a sampling frame whose structural-negative and
buyer-declared strata came from the now-removed France-level benchmark, and
the pipeline no longer regenerates it. Only the `20` accepted-link rows, drawn
from production links, are quoted as active evidence. A model-assisted diagnostic review
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
| Regional reference split isolation | Passed |
| Reference anchors re-resolved by notice id, ambiguities dropped | Passed |
| Retired benchmark and duration-conditioned arm absent from the repository | Passed |
| Evidence notebooks execute | Passed |
| Automated test suite | `72 passed` |
| Model-assisted linkage diagnostic | Complete; `14/20` conservatively confirmed |
| Guideline-aligned temporal validation (2022-2024) | Complete; weak, `C = 0.479` |
| Borderline-band robustness | Complete; comparative claims hold |
| Independent specialist linkage validity | Not established; required for stronger external accuracy claims, not for the current descriptive scope |

Passing software tests establishes implementation consistency. It does not
establish real-world linkage accuracy.

## 8. Permitted Final Claims

The final report may state that:

- the pipeline is reproducible and structurally validated;
- it identifies observable successor procurements rather than legal renewals;
- the main operating rule accepts `544` successors in a `3,800`-episode cohort;
- survival estimates are conditional on the linkage rule;
- the regional reference supports `M_B @ 0.70` as a provisional,
  precision-first baseline, held out because the threshold was frozen before
  that reference was consulted;
- lower thresholds trade precision for recall on the locked split and are
  retained as sensitivity arms rather than promoted, because selecting one from
  those rows would convert the locked split into a tuning set;
- trend findings are descriptive and non-causal.

The report must not state that:

- the `544` links are confirmed legal renewals;
- the locked-split precision is independently validated;
- the reference labels are independent human specialist annotations;
- a negative reference anchor proves that no successor exists;
- missing contracts are four years long;
- observed statistical breaks have known causes;
- the project implements a supervised technology taxonomy classifier;
- Cox hazard ratios are causal or provide validated individual forecasts;
- the survival probabilities are lower bounds on true re-procurement;
- a duration-conditioned linkage arm is part of the current sensitivity
  framework.

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

- `EXECUTIVE_SUMMARY.md`
- `FINAL_PIPELINE.md`
- `reports/current_project_readiness_report.html`
- `reports/boamp_methodology_chapter.pdf`
- `DATA_QUALITY_REPORT.md`
- `QUALITY_EVIDENCE.md`
- `TREND_ANALYSIS_REPORT.md`
- `SURVIVAL_ANALYSIS_REPORT.md`
- `INTERNSHIP_GUIDE_COMPLIANCE.md`
- `METHODOLOGICAL_REFERENCES.md`
- `INDEPENDENT_LINK_REVIEW_PROTOCOL.md`
- `data/processed/boamp/final_pipeline_manifest.json`
