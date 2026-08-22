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
- $u_i$ denote the anchor award-date origin;
- $v_j$ denote the candidate first-publication date;
- $J_i$ denote the candidate set that survives blocking for anchor $i$;
- $T_{ij}$ denote TF-IDF cosine text similarity;
- $\hat{R}_i\in J_i\cup\{\varnothing\}$ denote the accepted successor, $\varnothing$ on abstention;
- $Y_i=1$ denote an accepted observable successor and $Y_i=0$ an abstention;
- $\tau_i$ denote observed or censored time from $u_i$.

Calendar dates are lower-case so that the upper-case indicators $A_i$
(acceptance), $C_i$ (exact-match correctness), $P_i$ (the reference identifies a
successor) and $E_i$ (the reviewed successor survived blocking) can carry their
standard meaning in the linkage-evaluation layer. Those four are defined in
`reports/boamp_methodology_chapter.pdf` §Notation and in notebook 12, which is
where they are used.

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

The cohort contains `3,800` awarded Grand Ouest procurement episodes carrying at
least one CPV code in divisions `32`, `35`, `48`, or `72`. The rule is
**episode-level any-code inclusion**, not a rule about the episode's main CPV:
formally, episode $e$ is in scope iff
$\exists\, c \in \mathrm{CPV}(e)$ with $\lfloor c/10^{6} \rfloor \in \{32,35,48,72\}$.
`1,176` of the `3,800` (`30.9%`) are multi-lot procurements whose main CPV falls
outside the set; they enter on one digital lot. This is a coarse
procurement-domain segmentation and a deliberately inclusive one.

`digital_segment`, the stratifying variable for the segment Kaplan-Meier curves
and the Cox model, is the lowest-numbered digital division present, so each
episode contributes to exactly one curve. The tie-break binds on the `412`
episodes (`10.8%`) carrying more than one digital division. Its impact was
measured, not assumed: among episodes whose main CPV is itself digital the
assigned segment agrees with the main division `94.7%` of the time, and event
rates by assigned segment track those by main-CPV division closely. The rule is
documented rather than changed.

The trained technology taxonomy of §3.9 is layered on top of this and does not
alter it: CPV remains the cohort definition, the Cox covariate, and the trend
series.

### 3.4 Candidate Generation

A candidate is exposed only when buyer evidence is plausible, validated SIRENs
do not conflict, and:

\[
u_i+90\text{ days} \le v_j \le u_i+2920\text{ days}.
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
\tau_i=v_{\hat{j}_i}-u_i,
\qquad Y_i=1.
\]

If no successor is accepted by `2025-12-31`, the row is administratively
right-censored:

\[
\tau_i=\text{2025-12-31}-u_i,
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

A third robustness check covers what the first two cannot. Both of them move the
acceptance bar; the false-positive mechanism the linkage audit identified
produces links well *above* it, because standardised framework boilerplate
scores highly under the character analyser and because `M_B` imposes no
one-to-one constraint. Accepted links whose word-level similarity falls below
`0.50` or whose successor episode is shared with another anchor — `173` of the
`544`, on signatures already published by the candidate-generation audit — are
therefore **re-censored at the cutoff**, which is the counterfactual a spurious
link implies. Both headline hazard ratios keep their side of 1 (CPV-35 `1.55` to
`1.54`; framework `1.75` to `1.69`) while the KM level falls to `2.64%` at 12
months. Nothing is tuned, no link is relabelled, and the check bounds the
mechanism's influence rather than asserting the flagged links are false.

Neither the observed event rate nor any survival probability is a formal lower
bound on true re-procurement: missed successors push the measured level down and
residual false links push it up, so the net direction is not identified. They are
linkage-conditioned estimates.

The one operational artifact derived from these curves is
`data/processed/boamp/segment_watch_km.csv` (renamed on 2026-08-21 from
`renewal_watchlist_top20.csv`, a filename that implied an individual ranking the study
deliberately does not produce). It lists, for each of the four CPV segments, the five
still-unlinked episodes awarded in `2021` or later with the highest conditional probability of
showing a successor in the next twelve months. It is read off the **segment-stratified
Kaplan-Meier curves**, not off the Cox model, and it is stratified by segment on purpose: because
the conditional probability is a function of segment and age alone, an unstratified ranking would
simply return the highest-hazard segment at the age closest to the renewal shoulder, which would
suggest an individual-level precision the out-of-time C-index of `0.479` does not support. It is a
cohort-level monitoring aid, not a prediction for any single contract.

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

The CPV and technology trend families use the same `43`-quarter observation
window (`2015Q2`-`2025Q4`), excluding the partial `2015Q1` extract. In both
families, the published OLS slope describes only the latest `12` quarters. This
alignment prevents an avoidable difference in time support from being mistaken
for a substantive difference between CPV and predicted-technology results.

Five slopes are fitted and read together, so the raw p-values are reported beside
Holm and Benjamini-Hochberg adjustments across those five series -- the same
standard §3.9's technology trend section applies to its own family. A segment
whose raw p clears the pre-declared exploratory `α = 0.10` but whose Holm p does
not is reported as a nominal signal to monitor, not as a finding. On the current
data that applies to `CPV-48`: raw `p = 0.032`, Holm and BH `p = 0.159`.

### 3.9 Supervised Technology Taxonomy

A separate annotated corpus of `500` BOAMP notices, `2015`-`2025`, labelled into
eight substantive business technology classes plus `MIXED`, `OTHER_DIGITAL` and
`OTHER`, trains a supervised classifier over the notice object text.

Let $X_i$ be the object text of notice $i$ and $Y_i \in \mathcal{Y}$ its
annotated class. The estimand is a classifier $f: X \mapsto \mathcal{Y}$
evaluated by macro-F1 under group-aware 3-fold cross-validation, where the group
is a *procurement family*: the union of notices sharing a reconstructed episode
and notices whose objects reach character cosine `0.80`. Every family lies in
exactly one fold.

The frozen specification is TF-IDF word unigrams with a class-weighted
multinomial logistic regression. Both unigrams and unigrams-plus-bigrams were in
the **search space**; every fold selected unigrams alone, and that is the
**deployed representation**. The deployed confidence is the *raw* class score:
Platt scaling was evaluated inside the same grouped splits and rejected by the
pre-specified rule, because its macro-F1 cost exceeded the `0.02` budget the rule
allows even though its calibration gain cleared the `0.02` requirement. Hyperparameters come from a pre-specified
compact grid explored only by the inner cross-validation; every specification in
the budget is recorded in `specification_register.csv` rather than only the
winner. Selection used mean macro-F1 together with fold spread, the
train-validation gap, temporal behaviour and probability output. It is refit on all `500` labels for deployment and
applied to every cohort episode through the episode's origin-notice object text.

Buyer identity, geography, dates, amounts, procedure type, framework status,
notice identifiers and every linkage variable are excluded from the features, as
is CPV, which serves as the benchmark the text is measured against.

Downstream technology-level survival and trend analysis is gated twice: on
classifier evidence (substantive class, reference support `>= 10`, out-of-fold
F1 `>= 0.65`) and on statistical support. The first gate is not cosmetic --
including the fallback residuals moves the technology log-rank result from
`p = 0.036` to `p = 0.0001`.

The corpus has no second annotation pass, so no inter-annotator agreement
statistic exists. `AI` has `7` labelled notices and is reported as a rare-class
limitation. Confidence is the **raw** class probability -- Platt scaling was
evaluated and rejected -- and it remains conservative by a wide margin; the
`0.70` cutoff is an operational reporting convention and is unrelated to the
`0.70` linkage acceptance threshold.

## 4. Current Materialised Results

| Result | Current value | Interpretation |
|---|---:|---|
| Standardised notices | `1,620,712` | Unique official notice records |
| Reconstructed episodes | `1,103,632` | Inferred procurement processes |
| Study cohort | `3,800` | Awarded Grand Ouest episodes with at least one CPV code in 32/35/48/72 |
| Candidate pairs | `763,417` | Broad exposed comparison set |
| Main accepted successors | `544` | `M_B @ 0.70` observable events |
| Main observed event rate | `14.32%` | Linkage-conditioned, not legal renewal prevalence |
| Median successor time | `31.82 months` | Median among accepted events only |
| Borderline-band episodes excluded | `280` | `133` events, `147` censored; a separate `280` anchors have no candidate at all, which is a different set |
| SMD, log candidate-pool size | `+0.470` | Largest linked-vs-censored imbalance; a detectability variable |
| Framework HR, main / pool-adjusted | `1.751` / `1.617` | Partly detectability; sensitivity model only |
| CPV-35 HR, main / pool-adjusted | `1.553` / `1.512` | Largely insensitive to detectability |
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
rate is conservative by construction rather than a population-wide rate. The
candidate-export rule behind those roughly 25 reviewed candidates per anchor was
not recorded, so this recall figure and the `0.913` ceiling are not fully
independent of the text ranking; precision is not affected by that gap. Eight
accepted decisions make precision
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
| Automated test suite | See `final_pipeline_manifest.json` for the current run's status |
| Deployed confidence variant consistent across config, CSV, report, notebook and log | Passed |
| Simultaneous trend slopes multiplicity-adjusted in both families | Passed |
| Model-assisted linkage diagnostic | Complete; `14/20` conservatively confirmed |
| Guideline-aligned temporal validation (2022-2024) | Complete; weak, `C = 0.479` |
| Borderline-band robustness | Complete; comparative claims hold |
| Template-risk robustness | Complete; comparative claims hold under re-censoring |
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
  precision-first baseline; because that evidence informed the retained policy,
  it is internal validation rather than an untouched holdout;
- procurement text supports a business technology taxonomy that CPV codes do
  not carry, with an out-of-fold macro-F1 of `0.744` (95% family-bootstrap CI
  `0.682`-`0.791`) against `0.473` (`0.413`-`0.526`) for a CPV/descriptor
  benchmark on identical folds and the same regularisation range, a paired
  difference of `0.271` (`0.201`-`0.340`) excluding zero;
- among the five substantive technology classes the classifier separates well
  enough to analyse, a difference in observable-successor timing is detected
  (log-rank `p = 0.036`), unadjusted for buyer, size or procedure;
- no technology quarterly series shows a linear trend surviving Holm adjustment
  across the family of tests, and no CPV segment series does either -- `CPV-48`
  carries the clearest nominal recent decline (raw `p = 0.032`) but does not
  survive correction for the five segments tested, so it is exploratory;
- CPV-35 shows the highest observable-successor hazard among the segments
  (`1.553`, `p = 0.0004`) and remains essentially unmoved (`1.512`) when the
  candidate-pool detectability variable is added, so it is the most robust
  comparative finding here;
- framework agreements are associated with an earlier observable successor, but
  the association is **partly differential detectability**: adding
  `log(candidate pool size)` attenuates the hazard ratio from `1.751` to `1.617`;
- lower thresholds trade precision for recall on the recorded locked stratum
  and are retained as sensitivity arms; replacing the frozen post-development
  policy requires fresh independent evidence;
- trend findings are descriptive and non-causal.

The report must not state that:

- the `544` links are confirmed legal renewals;
- the locked-split precision is independently validated;
- the reference labels are independent human specialist annotations;
- the reference's recall figures or its `0.913` candidate-generation ceiling are
  independent of the text score they evaluate: the labels are independent, but
  the rule that selected the candidates shown to the reviewer was not recorded.
  Precision is unaffected by that gap;
- candidate-pool size causes re-procurement, or that the pool-adjusted Cox model
  is the headline model;
- a negative reference anchor proves that no successor exists;
- missing contracts are four years long;
- observed statistical breaks have known causes;
- the technology corpus has a measured inter-annotator agreement, or that its
  class counts estimate market prevalence;
- the classifier's `AI` performance has been measured;
- a predicted technology class is an observed attribute of a procurement, or
  that its confidence value is a probability of correctness read at face value;
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
- `reports/current_project_readiness_artifact.json`
- `reports/boamp_methodology_chapter.pdf`
- `DATA_QUALITY_REPORT.md`
- `QUALITY_EVIDENCE.md`
- `TREND_ANALYSIS_REPORT.md`
- `SURVIVAL_ANALYSIS_REPORT.md`
- `TECHNOLOGY_TAXONOMY_REPORT.md`
- `INTERNSHIP_GUIDE_COMPLIANCE.md`
- `METHODOLOGICAL_REFERENCES.md`
- `INDEPENDENT_LINK_REVIEW_PROTOCOL.md`
- `data/processed/boamp/final_pipeline_manifest.json`
