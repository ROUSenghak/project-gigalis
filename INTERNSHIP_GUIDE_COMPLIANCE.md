# Internship Guide Compliance

Assessment date: `2026-08-20`
Overall status: **ready for final reporting within the narrowed descriptive scope; not full guide compliance**

## Scope Decision

The defensible final scope is:

> A reproducible BOAMP pipeline for detecting observable successor procurements,
> estimating linkage-conditioned time to visible reprocurement, and describing
> quarterly digital-procurement trends in Grand Ouest.

The project must not be presented as a complete legal-renewal registry, a fully
validated predictive system, or a technology-taxonomy model trained on a
double-annotated corpus.

## Deliverable Matrix

| Guide deliverable | Current status | Evidence now available | Remaining gap |
|---|---|---|---|
| L1 Data quality report | Ready with caveats | `DATA_QUALITY_REPORT.md`, notebooks 10/11/14, structural tests | Independent semantic audit of a sample of reconstructed episodes and buyer matches would strengthen it |
| L2 Annotated corpus + NLP classifier | Implemented, single-annotator corpus | `TECHNOLOGY_TAXONOMY_REPORT.md`, notebook 15, `data/processed/boamp/technology/`. 500 manually annotated notices 2015-2025 across 11 classes; TF-IDF word n-grams (unigrams and unigrams+bigrams both searched; every fold selected unigrams alone) with class-weighted logistic regression and linear SVM variants, plus CPV and CPV+descriptor benchmarks, all on identical group-aware 3-fold folds that keep related notices together; out-of-fold macro-F1 0.744 (95% family-bootstrap CI 0.682-0.791) against 0.473 (0.413-0.526) for the best administrative benchmark, paired difference 0.271 (0.201-0.340) excluding zero; per-class precision/recall/F1 with support, confusion matrix, train-vs-validation learning curve, 30-error triage, 2015-2022 to 2023-2025 temporal check, frozen config, and one prediction per cohort episode carrying the deployed raw (uncalibrated) model confidence score | No second annotation pass, so no Cohen's kappa and no quantified label reliability; the corpus is quota-stratified so its class counts are not prevalence; AI has 7 labelled notices and cannot be evaluated; Platt scaling was evaluated and rejected by the pre-specified rule, so the deployed confidence is the raw class score and is conservative rather than calibrated; CamemBERT was gated out rather than tested, so no transformer comparison exists |
| L3 Survival analysis | Ready within descriptive scope | `SURVIVAL_ANALYSIS_REPORT.md`, notebook 13, materialised KM/Cox/PH/parametric/sensitivity tables, guideline-aligned 2015-2021 → 2022-2024 temporal validation, operational 12/24-month conditional probabilities with bootstrap intervals, borderline-band and template-risk robustness checks | Current events are linkage-conditioned; no active Gigalis portfolio, and out-of-time C-index is 0.479 on the guideline window (0.518 including 2025), so individualized prediction is not validated |
| L4 Trend report | Ready with caveats | `TREND_ANALYSIS_REPORT.md`, notebook 14, quarterly counts, PELT sensitivity, ADF/KPSS stationarity, 3-state HMM regime model for Overall/CPV-72/CPV-32, signal matrix carrying a per-segment operational recommendation | No validated monetary series (no canonical awarded-amount field); breaks and regimes are statistical candidates, not stakeholder-confirmed causes |
| L5 Final methodological report | Ready with caveats | `reports/boamp_methodology_chapter.pdf` (21 pages, 7 figures) structured as context/data/methods-per-family/results/limitations/perspectives; real Methods+Results depth for linkage, survival (KM curves, Cox HR table, PH diagnostics, temporal validation, parametric comparison, operational 12/24-month probability table and figure, linkage/borderline/template-risk sensitivity), and trend (quarterly series with PELT breaks, stationarity, HMM); technology-taxonomy section reporting the classifier, its benchmark comparison, and its annotation limits; causal-inference outline (`EXECUTIVE_SUMMARY.md` also added as a companion one-pager) | Final presentation should retain the same claim boundaries; the report is technical-depth-complete but shorter than the guide's 40-60 page target because it is not padded with literature review already covered by the guide itself |
| L6 Documented reproducible pipeline | Ready | scripts, tests, README, requirements, final pipeline runner | Environment pinning could be made stricter with a lock file, but this is not a current blocker |

## Methodological Readiness

| Component | Assessment | Reason |
|---|---|---|
| Official source ingestion | Defensible | The source is the official [BOAMP API](https://www.data.gouv.fr/dataservices/api-bulletin-officiel-des-annonces-des-marches-publics-boamp), with row reconciliation and source hashes |
| Notice standardisation | Defensible with known missingness | Schema-aware extraction preserves raw fields and records parser/source metadata |
| Episode reconstruction | Defensible as a heuristic transformation | It prevents notice duplication, but still needs semantic spot-checking because an episode is inferred rather than supplied by BOAMP |
| Buyer resolution | Conservative but incomplete | [SIREN](https://www.insee.fr/fr/metadonnees/definition/c2047) is preferred for legal-unit identity; name-only blocking remains necessary for 66.3% of the cohort |
| Digital segmentation | Reproducible, coarse; enriched | CPV divisions are official hierarchical categories under [Regulation 213/2008](https://eur-lex.europa.eu/eli/reg/2008/213/oj) and remain the cohort definition. They are demonstrably coarser than the business taxonomy: on identical folds and the same regularisation range a CPV/descriptor classifier reaches macro-F1 0.473 against 0.744 from text (paired difference 0.271, 95% CI 0.201-0.340), and mean CPV-segment purity against the predicted taxonomy is 0.34 |
| Technology classification | Defensible within a single-annotator corpus | Group-aware folds prevent related-notice leakage, hyperparameters are selected by inner CV inside each training fold, the CPV benchmark shares the folds and the search budget, uncertainty is a family-level bootstrap, and downstream use is gated on classifier quality as well as sample size. Label reliability is unquantified and rare classes are reported rather than engineered |
| Candidate generation | Reasonable broad blocking | Same plausible buyer plus 90-2,920 days controls computation while avoiding a hard expected-expiry assumption |
| Primary text ranking | Reasonable provisional baseline | TF-IDF cosine is standard document similarity; the 0.70 operating threshold is simple and auditable |
| Linkage evaluation | Regional reference sample plus model-assisted diagnostic | Locked-split precision is 0.875 (95% CI 0.529-0.978) on 8 accepted links; the separate 20-link accepted-stratum diagnostic confirmed 14, and neither is independent human validation |
| Declared duration in the event definition | Excluded by design | Reliable duration is missing for 74.9% of the cohort, so a duration-conditioned rule could differentiate itself only on a minority of episodes and would impose unsupported timing certainty on the rest; duration is kept as a descriptive diagnostic, and event-definition sensitivity is carried by the four threshold/method arms and the borderline band |
| Survival analysis | Descriptively useful | Kaplan-Meier handles administrative censoring; Cox results are qualified by PH violations and weak out-of-time discrimination |
| Trend analysis | Descriptively useful | PELT candidate breaks are sensitivity-tested, ADF/KPSS stationarity is reported per segment, and a 3-state HMM regime model complements PELT for the highest-volume segments; none are given unsupported causal explanations |
| Causal inference (guide's optional fourth dimension) | Outline only, as the guide allows ("if time allows... a research perspective and outline") | `reports/boamp_methodology_chapter.pdf` \S18.1 frames the Gigalis-membership causal question and the staggered-adoption DiD design that would answer it, but computes no estimate because it needs Gigalis-internal membership/adoption-date data this BOAMP-only corpus does not contain |

## Claims Allowed In The Final Report

- The study analyzes official BOAMP notices published from 2015 through 2025.
- It reconstructs procurement episodes and identifies observable successor procurements.
- The primary operating baseline accepts 544 links among 3,800 cohort episodes.
- The primary linkage-conditioned event rate is 14.3%, with a median observed successor time of 31.8 months among accepted events.
- CPV-35 has the highest observed successor rate in the primary arm.
- Absolute survival results are sensitive to the linkage rule.
- Quarterly CPV-48 episode counts show a recent exploratory decrease; other current segment slopes are stable or uncertain.
- Procurement object text supports a supervised 11-class business technology taxonomy at an out-of-fold macro-F1 of 0.744 (95% family-bootstrap CI 0.682-0.791), against 0.473 for the best CPV/descriptor benchmark on identical folds; the paired difference of 0.271 has a 95% interval of 0.201-0.340 excluding zero.
- The taxonomy cuts across the CPV segmentation rather than reproducing it: mean CPV-segment purity against the predicted classes is 0.34.
- Among the five substantive classes the classifier separates well enough to analyse, a difference in observable-successor timing is detected (log-rank p = 0.036); no technology series shows a linear volume trend surviving multiplicity adjustment.
- Every cohort episode carries exactly one predicted technology class and a confidence value; none is discarded.

## Claims Not Yet Allowed

- “The method has independently validated precision of 80%.”
- “The 544 links are confirmed legal renewals.”
- “The regional reference provides human inter-annotator agreement.”
- “Missing contracts have a four-year duration.”
- “Detected trend breaks were caused by a named policy or external event.”
- “The technology corpus has measured inter-annotator agreement.”
- “The classifier's AI performance has been measured.”
- “Predicted technology class counts are market shares.”
- “The survival model is ready for accurate individual forecasts.”

## Submission Position

There is no remaining computational blocker for submitting the narrowed
descriptive study. The final report must explicitly state that the supervised
technology-classification deliverable rests on a single-pass annotation with no
inter-annotator agreement statistic, that its predicted labels carry a measured
error rate into every downstream technology-level figure, and that operational
12/24-month individual prediction remains future work because no active Gigalis
portfolio with adequate temporal validation is available.

The guide's indicative 40-60% linkage rate was treated as a planning expectation,
never as an optimisation target. The realised 14.3% follows from the retained
precision-first threshold. Project history shows that the operating point and
candidate-generation choices drew on the reviewed reference evidence, so the
reported sweep is internal validation and sensitivity analysis, not a fully
held-out test.

Independent human specialist review remains necessary before claiming
independently validated linkage accuracy, but that stronger claim is outside the
final scope rather than a hidden unfinished task.
