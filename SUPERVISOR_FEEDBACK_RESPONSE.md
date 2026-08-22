# Response to supervisor feedback — supporting evidence

Date: 21 August 2026
Purpose: point-by-point reply, each remark tied to what the work already contains and to a verifiable number
Sources: `project-gigalis` repository (reports, materialised tables in `data/processed/boamp/`, notebooks 13/14/15)

---

## Summary table

| # | Supervisor's remark | Status | Main evidence |
|---|---|---|---|
| 1 | Title and subtitle to be sharpened | To apply (wording) | — |
| 2 | "Mon interprétation" box to professionalise | To apply (wording) | — |
| 3 | "Not yet certain of the right depth" → methodological trade-offs | To apply (wording) | — |
| 4 | Document the Pays de la Loire → Grand Ouest logic | **Quantified here**, to fold into the report | recomputation § 4 |
| 5 | Document the inclusion of 2025 and its completeness | **Quantified here**, to fold into the report | recomputation § 5 |
| 6 | Good reframing: the "observable successor" | Already in place | `DATA_QUALITY_REPORT.md`, event definition |
| 7 | Linkage architecture (broad → conservative, no hard CPV/duration filter) | In place and empirically justified | `CANDIDATE_GENERATION_AUDIT.md`, notebook 13 § 9 |
| 8 | Vigilance on linkage validation; pre-specified independent manual review | **Protocol already written, sample prepared**; independent human reviewer missing | `INDEPENDENT_LINK_REVIEW_PROTOCOL.md`, `REVIEW_AUDIT_RESULTS.md` |
| 9 | Kaplan-Meier correctly interpreted, conditional probabilities useful | Already in place | `SURVIVAL_ANALYSIS_REPORT.md` |
| 10 | Cox: weak temporal validation, no artificial "top 20" | Already in place — but one file to rename | § 10 below |
| 11 | Parametric models: keep KM for observed horizons | Already in place | `survival_parametric_comparison.csv` |
| 12 | Robustness = strong point, keep it in the final deliverable | Already in place (4 arms + 2 counter-tests) | `survival_linkage_sensitivity.csv`, template-risk section |
| 13 | Trends: don't stack models, CPV-48 = investigation lead | Already applied (multiplicity correction) | `TREND_ANALYSIS_REPORT.md` |
| 14 | NLP / technology taxonomy under-covered | **Delivered since the note**: 11 classes, TF-IDF + logistic regression | `TECHNOLOGY_TAXONOMY_REPORT.md` |
| 15 | Individual prediction → separate, pre-specified experiment | Agreed; protocol proposed § 15 | — |
| 16 | Message hierarchy, de-duplicate the limitations sections | To do (writing) | — |
| 17 | One-page executive summary answering five questions | **Already exists**, to be reordered onto those five questions | `EXECUTIVE_SUMMARY.md` |

Three remarks genuinely call for new work: writing up the scope decisions (4, 5), the independent human review (8), and the hierarchical rewrite (16, 17). The rest is either wording or already instrumented in the repository.

---

## 1-3. Wording remarks

No objection; I adopt the proposed formulations.

- Title: **"Analysis and modelling of digital public procurement using BOAMP data"**, subtitle unchanged for an interim note.
- Box renamed **"Document status and interpretation caveats"**.
- Hesitation reframed as a trade-off: the note will list the remaining methodological trade-offs explicitly — depth of linkage validation, granularity of the business segmentation, scope of the trend section — rather than an uncertainty about "the right level of depth".

The result / interpretation / uncertainty separation is already structural in the repository: every result in the technical reports is written as *Observation → Confidence → What can be concluded → What cannot be concluded* (see `TECHNOLOGY_TAXONOMY_REPORT.md`, Results 2 to 8). Remark 16 is a request to generalise that grid to the whole note.

---

## 4. Why Grand Ouest rather than Pays de la Loire alone

The requested chain (PdL → insufficient volume → extension → consequences) is exactly quantifiable. Recomputed on `survival_dataset.parquet` (frozen cohort):

| Scope | Episodes | Events | Rate | P(successor ≤ 12 m) | 95% CI | CI width |
|---|---:|---:|---:|---:|---|---:|
| Grand Ouest | 3,800 | 544 | 14.3% | 4.62% | [3.94; 5.30] | 1.37 pt |
| Pays de la Loire only | 1,452 | 219 | 15.1% | 5.36% | [4.17; 6.55] | 2.38 pt |

**What actually motivates the extension.** Not overall volume — 1,452 episodes remain analysable — but the **granularity** the research questions require:

- the 12-month confidence interval is **1.7× wider** for PdL alone (2.38 pt vs 1.37 pt), and 1.74× at 24 months;
- the segment carrying the study's most robust comparative finding, **CPV-35, has only 34 events in PdL alone against 115 across Grand Ouest**;
- the Cox model has 8 covariates: 68 events per parameter on Grand Ouest against **27 for PdL alone**, below the usual prudence threshold for multi-covariate survival models;
- the between-segment log-rank stays significant for PdL alone (statistic 14.68, p = 0.0021) but with a 1.6× weaker statistic: the signal exists, it is simply no longer resolved enough to support the segment × age breakdowns that constitute the operational deliverable.

**Consequences for interpretation, to be stated plainly.** The extension is not neutral: region is a model covariate and the three regions do not behave identically. Observed event rates: Bretagne 15.5%, Pays de la Loire 15.1%, Normandie 12.5%. In the Cox model, **Pays de la Loire vs Bretagne HR = 1.003 (p = 0.81)** — no measurable difference — but **Normandie HR = 0.800 (p = 0.046)**. Pooling is therefore defensible for the two closest regions and **explicitly adjusted for** Normandie; aggregate results must be read as "Grand Ouest", not as a proxy for Pays de la Loire, and any Gigalis-centred reading uses the PdL row of the model rather than the regional average.

---

## 5. Why 2025, and with what caveat

**Complete in volume: yes.** The 2025 awards sit within the corpus norm: 93 / 73 / 73 / 87 episodes across the four quarters of 2025, against 72 / 57 / 72 / 85 in 2024. There is no publication gap; the observation cut-off is 31/12/2025 and 2025 accounts for 8.6% of the cohort (326 episodes).

**Complete in follow-up: no — and this is where explicitness matters.** A contract awarded in 2025 has a **median follow-up of 5.9 months** against 59.9 months for 2015-2024. It can only contribute short-gap successors (longest observed gap: 11.4 months), and Q4 2025 contains **no** events by construction.

**Measured impact on the temporal analyses.** Recomputed with 2025 awards removed:

| Cohort | n | Events | P(≤ 12 m) | P(≤ 24 m) |
|---|---:|---:|---:|---:|
| 2015-2025 (retained) | 3,800 | 544 | 4.62% | 6.73% |
| 2015-2024 | 3,474 | 499 | 3.54% | 5.68% |

The gap is not nil: including 2025 raises the 12-month estimate by roughly 1.1 points, because 45 short-gap events enter with very little time at risk. **This is therefore not a neutral choice and will be presented as a sensitivity, not a detail.** Two protections are already in place:

1. the **primary** Cox temporal validation is the window aligned with the original framing, training 2015-2021 → test **2022-2024**; the 2022-2025 variant is carried only as a sensitivity read (C-index 0.479 vs 0.518);
2. the 2025 measurement break is already documented: duration-field completeness jumps from 30.4% (2024) to **84.4%** (2025). That is a schema change, not a change in purchasing behaviour; `TREND_ANALYSIS_REPORT.md` refuses change-point detection on duration values for exactly this reason.

**Justification retained:** 2025 is kept because it adds 326 episodes of censored exposure that improve estimation at short horizons and because the 31/12/2025 cut-off is clean; it is flagged as partial in follow-up, excluded from the primary validation window, and accompanied by the sensitivity table above.

---

## 6. The "observable successor" as the measured object

This is indeed the pivot of the work, and it is locked into the event definition:

> `event = 1` when a later episode from the same buyer is **accepted** by the frozen linkage rule; `event = 0` when no successor is accepted before 31/12/2025, the row then being right-censored — which **does not prove abandonment**.

BOAMP does not encode legal renewal, and the whole documentation chain is consistent with that (`INTERNSHIP_GUIDE_COMPLIANCE.md` § "Claims not yet allowed" explicitly forbids the sentence "the 544 links are confirmed legal renewals").

---

## 7. Linkage architecture: broad generation, conservative acceptance

**Broad generation.** Blocking on plausible buyer + a 90–2,920 day window, with no expiry assumption.

**No hard CPV filter — empirical argument, not convenience.** Among the 544 accepted links, 34.8% cross a CPV division. But the reference successors, labelled **with no knowledge of any linkage method**, cross a division in 39.1% of cases (9 of 23). Imposing strict CPV blocking would therefore destroy 9 of the 23 reference successors and cut the attainable recall ceiling from **0.913 to 0.609**.

**No hard declared-duration filter — also empirical.** Reliable duration is missing for 74.9% of the cohort, and more importantly: among contracts with both a reliable duration and an accepted successor, **only 21.8% are re-procured within six months of their declared end**, with a median absolute discrepancy of **21.1 months**. Declared duration is a poor predictor of actual timing; it is kept as a descriptive diagnostic, not as a constraint.

**Precision over recall.** The argument you make (a false link creates both a false event *and* a false event time) is exactly the one adopted. On the locked split, `M_B_text_ranking @ 0.70` shows precision 0.875 and a **false-positive rate of 0.000**, against 0.522 precision for `M_C`, which buys recall (0.667 vs 0.389).

---

## 8. Linkage validation: what exists, what is missing

**What I concede without reservation** — and which is already written in the repository:

- the regional reference (120 anchors, 112 resolved, locked split of 72) has **labels produced by a single LLM research pass, spot-checked** rather than verified anchor by anchor: it is not independent human ground truth;
- the 0.875 precision rests on **8 accepted links**, 95% CI [0.529; 0.978] — an interval that forbids any performance claim;
- worse, and documented in `REGIONAL_BENCHMARK_REFERENCE.md`: the rule that selected the ~25 candidates exported per anchor (from pools of up to 3,258) **was not recorded**, and every retrievable successor ranks near the top of the production text score. **Recall** and the 0.913 ceiling are therefore **not independent of the score they evaluate**. Precision is unaffected: a false positive is a false positive however the candidate list was assembled;
- the 60-pair challenge review is **model-assisted, not independent-human**: of the 20 accepted links, 14 confirmed, 5 rejected, 1 uncertain → conservative precision **0.700** (CI [0.457; 0.881]), **below the stated 0.80 target**. It is written exactly that way in `REVIEW_AUDIT_RESULTS.md`, with no favourable rounding.

**Your recommendation is already instrumented.** The protocol you describe exists: `INDEPENDENT_LINK_REVIEW_PROTOCOL.md` defines a **blinded** 60-pair sample (20 production-accepted links, 20 high-similarity structural negatives, 20 buyer-declared relationships), a reviewer file with no label, score, or stratum, a separate audit key, a four-field decision schema, and a **pre-specified acceptance rule**: freeze `M_B @ 0.70` during review, compute exact binomial intervals, and no tuning the threshold on those rows and then reporting them as validation.

**What is missing is a person, not a method.** I fully endorse your trade-off: an additional independent human review is worth more than a fifth linkage method. Stated objective: **qualify the confidence we can place in the current rule, without recalibrating it**. Two caveats to carry into a relaunch: the original draw is no longer reproducible (two of its three strata came from the national benchmark removed on 15/08), and the only stratum usable as active evidence is `PRIMARY_ACCEPTED`. A fresh draw must therefore be re-specified and documented before execution.

---

## 9. Kaplan-Meier and conditional probabilities

The reading is the one you validate:

- probability of an **observable successor** — never "probability of renewal";
- KM median **not reached**; the 31.8 months quoted elsewhere is the median **among linked events only**, and the report says so explicitly to head off the confusion;
- well-powered separation between segments: log-rank 23.45, p = 3.3 × 10⁻⁵; CPV-35 at 20.4% event rate against 11.6% for CPV-48.

The operational translation you mention already exists as a conditional table, with 500-draw bootstrap intervals:

| Contract age | P(successor ≤ 12 m) | 95% CI | P(≤ 24 m) | 95% CI |
|---:|---:|---|---:|---|
| 0 months | 4.62% | [3.91; 5.24] | 6.73% | [5.91; 7.58] |
| 12 months | 2.22% | [1.70; 2.73] | 4.26% | [3.55; 4.98] |
| 24 months | 2.09% | [1.58; 2.60] | 9.40% | [8.33; 10.68] |
| 36 months | 7.46% | [6.52; 8.59] | 9.69% | [8.58; 11.12] |
| 48 months | 2.41% | [1.77; 3.07] | 2.89% | [2.15; 3.72] |

The profile is not monotone: it rises into the 36-48 month renewal shoulder and falls away after it. That is precisely a **cohort-level watch logic** (segment × age), not an individual score.

---

## 10. Cox and the refusal of a "top 20"

**Temporal validation.** Fitted once on 2015-2021, scored out of time with no refitting: C-index 0.606 in training, **0.479** on the 2022-2024 window (0.518 including 2025). That is indistinguishable from chance, and the report states it as a **result**, not as a prompt to retune.

**PH assumption.** Rejected for `award_year_centered`, `framework_flag` and `has_validated_siren`. The coefficients are therefore presented as time-averaged descriptive associations, never as effects.

**Positioning to keep in the final version**, as you put it: Cox = descriptive associations at population level; **not** an individual scoring engine.

**One point of honesty to flag.** The repository contains `data/processed/boamp/renewal_watchlist_top20.csv`. Its **name is misleading** and I will rename it, because it is not a Cox-derived top 20 of contracts "most likely to be renewed". It is a **segment-stratified** list (the 5 nearest-horizon contracts in each of the 4 CPV segments), built from the **segment-stratified Kaplan-Meier** curves rather than from Cox, restricted to awards from 2021 onwards. Notebook 13 documents explicitly why an unstratified top 20 would be degenerate: since the probability is a function of segment and age alone, a global sort simply returns the highest-hazard segment at the age closest to the peak — creating a false impression of individual granularity. The logic is the one you recommend; only the filename is not.

---

## 11. Parametric models

`GeneralizedGamma` has the lowest AIC among exponential, Weibull, log-logistic, log-normal and generalized gamma. It is **not** the source of the operational numbers: every smooth family flattens the empirical renewal shoulder, and every published horizon lies inside the observed window. The 12- and 24-month probabilities are therefore read off Kaplan-Meier; the parametric model is kept as the best-fitting family and as the instrument any extrapolation beyond 31/12/2025 would use. That is exactly the trade-off you judge more reasonable than mechanical AIC/BIC selection.

---

## 12. Robustness (which you identify as a strong point)

The result you ask me to preserve — unstable absolute levels, more stable relative associations — is materialised across three independent tests.

**a) Four linkage arms.**

| Arm | Events | Rate | P(≤ 12 m) | P(≤ 24 m) |
|---|---:|---:|---:|---:|
| strict (`M_B @ 0.80`) | 296 | 7.8% | 2.37% | 3.23% |
| primary (`M_B @ 0.70`) | 544 | 14.3% | 4.62% | 6.73% |
| looser (`M_B @ 0.60`) | 853 | 22.4% | 8.00% | 11.47% |
| high-recall contrast (`M_C @ 0.70`) | 1,332 | 35.1% | 12.21% | 17.98% |

A factor of 4.5 on the event count: no absolute level can be quoted on its own.

**b) Borderline band (±0.05 around the threshold).** Removing the 280 borderline episodes (133 of them events): KM 12 m 4.62% → 3.72%; **CPV-35 HR 1.553 → 1.780**; framework 1.751 → 1.616. Direction unchanged.

**c) Template-risk re-censoring.** The false positives the audit actually identified come not from the threshold but from **framework-agreement legal boilerplate**, on which character n-grams score high between unrelated objects. 173 of 544 links (31.8%) carry one of the two observable signatures; they are **re-censored** (not dropped): KM 12 m falls to 2.64%, but **CPV-35 stays at 1.541 and framework at 1.692**. This is the counter-test the framework finding most needed, since that boilerplate is the very text driving the mechanism.

**d) Detectability — the point I consider the most important in the work.** The largest linked-vs-censored imbalance is not a property of the contract at all: it is **candidate-pool size** (SMD +0.470 on the log scale). A buyer who publishes prolifically mechanically produces a higher maximum score. As a sensitivity, adding log(1 + pool size) to the Cox model:

| Covariate | HR, main model | HR, + log(pool) | adjusted p |
|---|---:|---:|---:|
| `framework_flag` | 1.751 | **1.617** | 2.6 × 10⁻⁶ |
| `digital_segment_CPV-35` | 1.553 | **1.512** | 8.5 × 10⁻⁴ |
| `log_candidate_pool_size` | — | 1.184 | 6 × 10⁻⁹ |

Reading: **CPV-35 is the study's most robust comparative finding** (stable across the four arms, the borderline band, template-risk re-censoring, and the detectability adjustment). **The framework association is partly detectability**, not behaviour alone — roughly 14% of the log hazard ratio evaporates. That nuance is carried in the executive summary, not buried in an appendix.

---

## 13. Trends: caution already applied

Agreed on not investing further. The current state is deliberately restrained and **multiplicity correction is already in place**:

| Segment | Slope / quarter | Raw p | Holm p | BH p | Reading |
|---|---:|---:|---:|---:|---|
| Overall | −0.11 | 0.921 | 1.000 | 0.989 | no signal |
| CPV-32 | −0.01 | 0.989 | 1.000 | 0.989 | no signal |
| CPV-35 | +0.03 | 0.923 | 1.000 | 0.989 | no signal |
| **CPV-48** | **−0.84** | **0.032** | 0.159 | 0.159 | **nominal signal only** |
| CPV-72 | +0.70 | 0.285 | 1.000 | 0.714 | no signal |

CPV-48 is therefore already treated as an **investigation lead**, not a forecast: the report explicitly recommends watching it "for another few quarters before acting on it". No PELT break is attributed to a cause (policy, COVID, technology) without documentary evidence, and the HMM is presented as a regime reading, not a forecast. Monetary series remain excluded for want of a validated awarded-amount definition.

---

## 14. NLP / technology taxonomy — delivered since the note

This is where your reading is behind the repository: **the taxonomy you describe has been built**, and along exactly the pragmatic path you recommend.

**Taxonomy**: 8 substantive classes — `CLOUD_HOSTING`, `CYBERSECURITY`, `NETWORK_TELECOM`, `IT_INFRASTRUCTURE`, `BUSINESS_SOFTWARE`, `DATA_BI`, `AI`, `IT_SERVICES` — plus 3 acknowledged fallback classes (`MIXED`, `OTHER_DIGITAL`, `OTHER`). It covers the list in your message. It was **frozen before modelling**.

**Corpus**: 500 manually annotated notices, 2015-2025, written annotation rules, input field = `objet` (median 14 words).

**Baseline exactly as you propose**: word TF-IDF + class-weighted logistic regression (`class_weight='balanced'`), linear SVM variant tested, hyperparameters selected by inner CV **inside each training fold**.

**Leakage control**: BOAMP republishes, and buyers re-run near-identical tenders. Every notice is assigned to a **procurement family** (same reconstructed episode, or character cosine ≥ 0.80), and every family sits in exactly one fold: 459 families, 0 spanning two folds.

**Results** (group-aware 3-fold cross-validation, out of fold):

| Model | Out-of-fold macro-F1 | 95% CI (family bootstrap) |
|---|---:|---|
| TF-IDF + logistic regression | **0.744** | [0.682; 0.790] |
| CPV / descriptor benchmark (same folds, same budget) | 0.473 | [0.413; 0.526] |
| **Paired difference** | **+0.271** | [0.201; 0.340] — excludes zero |

This is the quantitative answer to "the CPV segmentation remains very broad": **text carries the business information CPV does not**. Mean CPV-segment purity against the taxonomy is 0.34 — each CPV division holds several distinct business technologies.

**No complexity for its own sake.** The rule for using a transformer was written *before* the classical results were read: CamemBERT is tested only if the classical model is materially inadequate (macro-F1 < 0.55) **and** fewer than half its errors come from label ambiguity. The model reaches 0.741: the condition fails, so **CamemBERT was never run**. It was not tested and discarded; the criterion for running it was not met.

**Temporal robustness**: training 2015-2022 → test 2023-2025. Macro-F1 on classes with test support ≥ 10: **0.815**. Recent notice vocabulary has not drifted for the high-volume classes.

**Downstream use behind two explicit gates**: a class feeds a survival curve only if (A) it is substantive, support ≥ 10 and F1 ≥ 0.65, and (B) it has enough events. 5 of 11 classes pass. Result: observable re-procurement timing differs across technologies, log-rank p = 0.036 on 416 events — `BUSINESS_SOFTWARE` at 8.6% by 24 months against `IT_INFRASTRUCTURE` at 6.1%.

**Limits I carry myself**: single-pass annotation, **no Cohen's kappa** available (the L2 design asked for two annotators); annotation quotas ⇒ proportions are not prevalence; `AI` has 7 notices and is declared unevaluable; Platt scaling was **evaluated and rejected** by the pre-specified rule (ECE gain 0.140 passes, but a 0.036 macro-F1 cost exceeds the 0.02 budget), so the published score is a **raw, uncalibrated confidence score** — conservative, usable for ranking and filtering, not as a probability.

**Best remaining investment on this component**: the learning curve is still rising at n = 500. A second annotation pass (kappa + volume) beats a more sophisticated model — the same trade-off you apply to linkage.

---

## 15. Individual prediction: a separate, pre-specified experiment

Full agreement on the principle: **do not progressively tweak the current Cox model until it produces a better number**. The current model stays frozen as it is, with its out-of-time C-index of 0.479.

Proposed protocol if this line is prioritised, to be written **before** running anything:

1. **Award-time variables only**: object text (predicted technology class + confidence, already available for all 3,800 episodes), buyer history (prior publication volume, seniority, recurrence in the segment), procedure type, framework status.
2. **Pre-declared temporal validation**: training 2015-2021, test 2022-2024, one pass, no refitting.
3. **Trap to name explicitly**: candidate-pool size (HR 1.184, p = 6 × 10⁻⁹) is a **detectability** channel, not purchasing behaviour. Including it would raise the C-index by learning to predict *who is detectable*, not *who renews*. It must be handled as a sensitivity, never in the main specification — otherwise the experiment concludes positively for the wrong reason.
4. **Success criterion announced in advance** (e.g. out-of-time C-index ≥ 0.60), and publication of the result even if negative.

That allows, as you write, a clean conclusion on what these variables actually add.

---

## 16-17. Message hierarchy and executive summary

**The one-pager already exists** (`EXECUTIVE_SUMMARY.md`) but is organised as "what the project does / what was done / what works / what remains uncertain / next steps". I am reordering it onto your five questions:

| Question | Content |
|---|---|
| **What we know** | 1,620,712 standardised notices → 3,800 awarded Grand Ouest digital episodes 2015-2025; end-to-end reproducible pipeline; BOAMP does not encode legal renewal, so what is measured is an **observable successor** |
| **What the models indicate** | 14.3% of contracts receive an observable successor; 4.6% by 12 months, 6.7% by 24; renewal shoulder at 36-48 months; CPV-35 re-procures fastest (HR 1.553), stable across every test; text carries a business segmentation CPV does not (+0.271 macro-F1) |
| **What remains uncertain** | Absolute levels depend on the linkage rule (296 to 1,332 events); linkage precision is not human-validated (0.70 in the assisted review, 0.80 target); Cox does not discriminate out of time (0.479); no kappa on the annotation corpus; the framework effect is partly detectability |
| **What this can mean for Gigalis** | A framework for **measurement and cohort-level watch** (segment × age), not a per-contract score: prioritise monitoring on CPV-35 and the 36-48 month window; use an 8-class business segmentation where CPV is too broad; track CPV-48 as a signal to confirm |
| **What we recommend next** | (1) independent human review of the 60 blinded pairs; (2) a second annotation pass to obtain a kappa; (3) supply Gigalis membership data if the causal question is wanted; (4) freeze linkage / survival / trends |

**De-duplication.** You are right that four sections currently say similar things: "limitations", "formulations to avoid", "discussion points" and "my interpretation". I merge them into **two** blocks: (a) *Document status and interpretation caveats* up front, (b) *Limitations and claim boundaries* at the end, the latter reusing the list already formalised in `INTERNSHIP_GUIDE_COMPLIANCE.md` ("claims allowed" / "claims not yet allowed"), which is exactly the can/cannot-conclude grid.

**Grid to apply to each result** (your five-point list): it is already the native format of the technical reports (Observation → Confidence → What can be concluded → What cannot be concluded); the fifth line, **"what Gigalis can do with it"**, is the one missing, and I am adding it systematically.

---

## Resulting work plan

| Priority | Action | Rationale |
|---|---|---|
| 1 | Hierarchical rewrite + five-question summary + merge the limitations sections | Low cost, immediate readability gain |
| 2 | Written scope framing (PdL → Grand Ouest, inclusion of 2025) with the tables above | Explicit request, numbers already computed |
| 3 | Independent human review of a re-drawn, pre-specified blinded sample | The only gate before any precision claim; more useful than a 5th method |
| 4 | Rename `renewal_watchlist_top20.csv` to `segment_watch_km.csv` | The name suggests the individual top 20 the work deliberately refuses to produce |
| 5 | If time allows: second technology annotation pass (kappa) | Learning curve still rising at n = 500 |
| 6 | Not a priority: trends, individual prediction | Short, noisy series; individual prediction to be handled as a separate pre-specified experiment |

---

### Files cited

`EXECUTIVE_SUMMARY.md` · `SURVIVAL_ANALYSIS_REPORT.md` · `TREND_ANALYSIS_REPORT.md` · `TECHNOLOGY_TAXONOMY_REPORT.md` · `DATA_QUALITY_REPORT.md` · `REGIONAL_BENCHMARK_REFERENCE.md` · `CANDIDATE_GENERATION_AUDIT.md` · `REVIEW_AUDIT_RESULTS.md` · `INDEPENDENT_LINK_REVIEW_PROTOCOL.md` · `INTERNSHIP_GUIDE_COMPLIANCE.md` · `notebooks/13_survival_analysis.ipynb` · `notebooks/14_data_quality_and_trend_analysis.ipynb` · `notebooks/15_technology_taxonomy_classification.ipynb`

The figures in sections 4 and 5 (Pays de la Loire alone, effect of 2025) were recomputed on 21/08/2026 directly from `data/processed/boamp/survival_dataset.parquet`; all others are taken from the repository's frozen tables and reports.
