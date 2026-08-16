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
| L2 Annotated corpus + NLP classifier | Not implemented in this branch | CPV divisions define digital scope; TF-IDF is used for successor linkage. A parallel classification effort exists as an un-integrated export (`data/reference/technology_classification/`, 945 rows dated 2026-08-12, one `Domaine` label per notice) | That export covers current national opportunities rather than the 2015-2025 Grand Ouest cohort and arrives without training corpus, annotation guidelines, or validation artifacts, so it could not be integrated or audited at the freeze. No 300-500 independently annotated technology corpus, taxonomy classifier, macro-F1, or classifier confusion matrix exists here |
| L3 Survival analysis | Ready within descriptive scope | `SURVIVAL_ANALYSIS_REPORT.md`, notebook 13, materialised KM/Cox/PH/parametric/sensitivity tables, guideline-aligned 2015-2021 → 2022-2024 temporal validation, operational 12/24-month conditional probabilities with bootstrap intervals, borderline-band and template-risk robustness checks | Current events are linkage-conditioned; no active Gigalis portfolio, and out-of-time C-index is 0.479 on the guideline window (0.518 including 2025), so individualized prediction is not validated |
| L4 Trend report | Ready with caveats | `TREND_ANALYSIS_REPORT.md`, notebook 14, quarterly counts, PELT sensitivity, ADF/KPSS stationarity, 3-state HMM regime model for Overall/CPV-72/CPV-32, signal matrix carrying a per-segment operational recommendation | No validated monetary series (no canonical awarded-amount field); breaks and regimes are statistical candidates, not stakeholder-confirmed causes |
| L5 Final methodological report | Ready with caveats | `reports/boamp_methodology_chapter.pdf` (21 pages, 7 figures) structured as context/data/methods-per-family/results/limitations/perspectives; real Methods+Results depth for linkage, survival (KM curves, Cox HR table, PH diagnostics, temporal validation, parametric comparison, operational 12/24-month probability table and figure, linkage/borderline/template-risk sensitivity), and trend (quarterly series with PELT breaks, stationarity, HMM); explicit NLP-scope-decision section; causal-inference outline (`EXECUTIVE_SUMMARY.md` also added as a companion one-pager) | Final presentation should retain the same claim boundaries; the report is technical-depth-complete but shorter than the guide's 40-60 page target because it is not padded with literature review already covered by the guide itself |
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

## Claims Not Yet Allowed

- “The method has independently validated precision of 80%.”
- “The 544 links are confirmed legal renewals.”
- “The regional reference provides human inter-annotator agreement.”
- “Missing contracts have a four-year duration.”
- “Detected trend breaks were caused by a named policy or external event.”
- “The project implements the guide's supervised technology classifier.”
- “The survival model is ready for accurate individual forecasts.”

## Submission Position

There is no remaining computational blocker for submitting the narrowed
descriptive study. The final report must explicitly state that the supervised
technology-classification deliverable was not implemented inside this
reproducible branch — the parallel effort's export could not be integrated or
validated here — and that operational 12/24-month individual prediction remains
future work because no active Gigalis portfolio with adequate temporal validation
is available.

The guide's indicative 40-60% linkage rate was treated throughout as a planning
expectation, never as an optimisation target. The realised 14.3% follows from the
precision-first threshold that was frozen before the reference was consulted; it
is a consequence of the chosen operating point, not a miss against a goal.

Independent human specialist review remains necessary before claiming
independently validated linkage accuracy, but that stronger claim is outside the
final scope rather than a hidden unfinished task.
