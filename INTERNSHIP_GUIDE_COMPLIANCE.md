# Internship Guide Compliance

Assessment date: `2026-08-14`
Overall status: **ready for final reporting within the narrowed descriptive scope; not full guide compliance**

## Scope Decision

The defensible final scope is:

> A reproducible BOAMP pipeline for detecting observable successor procurements,
> estimating linkage-conditioned time to visible reprocurement, and describing
> quarterly digital-procurement trends in Grand Ouest.

The project must not be presented as a complete legal-renewal registry, a fully
validated predictive system, or a completed supervised technology-taxonomy model.

## Deliverable Matrix

| Guide deliverable | Current status | Evidence now available | Remaining gap |
|---|---|---|---|
| L1 Data quality report | Ready with caveats | `DATA_QUALITY_REPORT.md`, notebooks 10/11/14, structural tests | Independent semantic audit of a sample of reconstructed episodes and buyer matches would strengthen it |
| L2 Annotated corpus + NLP classifier | Not implemented | CPV divisions define digital scope; TF-IDF is used for successor linkage | No 300-500 independently annotated technology corpus, taxonomy classifier, macro-F1, or classifier confusion matrix |
| L3 Survival analysis | Ready within descriptive scope | `SURVIVAL_ANALYSIS_REPORT.md`, notebook 13, materialised KM/Cox/PH/parametric/sensitivity tables | Current events are linkage-conditioned; no active Gigalis portfolio and temporal C-index is only 0.518, so individualized prediction is not validated |
| L4 Trend report | Partial but usable | `TREND_ANALYSIS_REPORT.md`, notebook 14, quarterly counts, PELT sensitivity, signal matrix | No validated monetary series, HMM regime model, or stakeholder-confirmed explanations for breaks |
| L5 Final methodological report | Ready with caveats | `reports/boamp_methodology_chapter.pdf` integrates data, linkage, evaluation, survival, trends, limitations, and references | Final presentation should retain the same claim boundaries |
| L6 Documented reproducible pipeline | Ready | scripts, tests, README, requirements, final pipeline runner | Environment pinning could be made stricter with a lock file, but this is not a current blocker |

## Methodological Readiness

| Component | Assessment | Reason |
|---|---|---|
| Official source ingestion | Defensible | The source is the official [BOAMP API](https://www.data.gouv.fr/dataservices/api-bulletin-officiel-des-annonces-des-marches-publics-boamp), with row reconciliation and source hashes |
| Notice standardisation | Defensible with known missingness | Schema-aware extraction preserves raw fields and records parser/source metadata |
| Episode reconstruction | Defensible as a heuristic transformation | It prevents notice duplication, but still needs semantic spot-checking because an episode is inferred rather than supplied by BOAMP |
| Buyer resolution | Conservative but incomplete | [SIREN](https://www.insee.fr/fr/metadonnees/definition/c2047) is preferred for legal-unit identity; name-only blocking remains necessary for 66.3% of the cohort |
| Digital segmentation | Reproducible, coarse | CPV divisions are official hierarchical categories under [Regulation 213/2008](https://eur-lex.europa.eu/eli/reg/2008/213/oj), but they are not a learned 8-12 class taxonomy |
| Candidate generation | Reasonable broad blocking | Same plausible buyer plus 90-2,920 days controls computation while avoiding a hard expected-expiry assumption |
| Primary text ranking | Reasonable provisional baseline | TF-IDF cosine is standard document similarity; the 0.70 operating threshold is simple and auditable |
| Linkage evaluation | Development evidence plus model-assisted diagnostic | Bootstrap-reference precision is 0.80; the separate 20-link accepted-stratum diagnostic confirmed 14 links, but neither is independent human validation |
| Expiry-aware arm | Correctly retained as sensitivity | Missing duration is not imputed; unusually early candidates require stronger evidence |
| Survival analysis | Descriptively useful | Kaplan-Meier handles administrative censoring; Cox results are qualified by PH violations and weak out-of-time discrimination |
| Trend analysis | Descriptively useful | PELT candidate breaks are sensitivity-tested and are not given unsupported causal explanations |

## Claims Allowed In The Final Report

- The study analyzes official BOAMP notices published from 2015 through 2025.
- It reconstructs procurement episodes and identifies observable successor procurements.
- The primary operating baseline accepts 544 links among 3,800 cohort episodes.
- The primary linkage-conditioned event rate is 14.3%, with a median observed successor time of 31.8 months among accepted events.
- CPV-35 has the highest observed successor rate in the primary arm.
- Absolute survival results are sensitive to the linkage rule.
- Quarterly CPV-48 episode counts show a recent exploratory decrease; other current segment slopes are stable or uncertain.

## Claims Not Yet Allowed

- “The method has independently validated precision of 80%.”
- “The 544 links are confirmed legal renewals.”
- “The two annotation passes provide human inter-annotator agreement.”
- “Missing contracts have a four-year duration.”
- “Detected trend breaks were caused by a named policy or external event.”
- “The project implements the guide's supervised technology classifier.”
- “The survival model is ready for accurate individual forecasts.”

## Submission Position

There is no remaining computational blocker for submitting the narrowed
descriptive study. The final report must explicitly state that the supervised
technology-classification deliverable was not implemented and that operational
12/24-month individual prediction remains future work because no active Gigalis
portfolio with adequate temporal validation is available.

Independent human specialist review remains necessary before claiming
independently validated linkage accuracy, but that stronger claim is outside the
final scope rather than a hidden unfinished task.
