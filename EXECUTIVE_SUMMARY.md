# Executive Summary

Generated: `2026-08-14T11:16:53`
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

- Standardised 1,620,712 BOAMP notices into
  reconstructed procurement episodes and an awarded Grand Ouest digital study
  cohort of `3,800` episodes.
- Compared four linkage algorithms on a `252`-anchor,
  `7,031`-pair national development reference and
  froze `M_B_text_ranking @ 0.70` as the primary, precision-first rule
  (precision `0.800`, recall `0.182` on the internal
  held-out split).
- Applied it to the full cohort: `544`
  accepted links, `14.3%` event rate.
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
- Documented every provenance caveat honestly: bootstrap-labelled benchmark,
  model-assisted review, and a CPV-division substitute where the guide asks
  for a supervised technology classifier.

## What Works

- The pipeline is reproducible end to end (`scripts/run_final_pipeline.py`),
  with `171` automated tests passing and internal consistency checks
  (`data/processed/boamp/canonical_state_validation.json`) all green.
- Kaplan-Meier shows a clear, well-powered separation across CPV segments
  (log-rank statistic `23.45`,
  `p=3.3e-05`); estimated successor
  probability is `4.6%`
  by 12 months and `6.7%`
  by 24 months.
- Framework-agreement status and CPV-35 are the most linkage-robust Cox
  covariates across all four sensitivity arms.
- CPV-48 shows a statistically distinguishable recent decline
  (segments: CPV-48); other
  segments are stable or uncertain by the current 12-quarter signal.

## What Remains Uncertain

- The benchmark's labels come from deterministic rules, not independent human
  annotation; the model-assisted 60-pair review found `70.0%` conservative
  precision among accepted links, below the `80%` target -- independent
  human review is still needed before claiming validated accuracy.
- Absolute event rates and probabilities are linkage-sensitive: event counts
  range from `296` to
  `1,332` across retained arms.
- Cox temporal validation is weak (C-index
  `0.607` in-sample
  vs `0.518` out-of-time); the
  model is not validated for individualized operational prediction, and no
  active Gigalis portfolio was available to score.
- The guide's supervised technology-classification deliverable (L2) was not
  built; CPV divisions are used as a coarser, reproducible substitute.
- The guide's causal-inference question (does a Gigalis framework change
  member behaviour?) is outlined methodologically but not answered -- it
  needs Gigalis-internal membership/adoption-date data not present in BOAMP.

## Recommended Next Steps

1. Commission an independent human procurement-domain reviewer to label the
   prepared blinded 60-pair sample (`INDEPENDENT_LINK_REVIEW_PROTOCOL.md`)
   before any external accuracy claim or threshold change.
2. If the technology classifier remains a priority, recruit a second
   qualified annotator and reuse the existing blinded-sample-plus-adjudication
   tooling (`scripts/prepare_independent_link_review.py`,
   `scripts/ingest_annotations.py`, `scripts/adjudicate_annotations.py`) for a
   real 300-500 example corpus with genuine Cohen's kappa.
3. If a Gigalis-membership causal analysis is wanted, supply member identity
   and adoption-date data so the outlined staggered-adoption
   difference-in-differences design can actually be estimated.
4. Treat the current linkage, survival, and trend components as frozen; do
   not reopen them without new evidence, per `PROJECT_WORK_PROTOCOL.md`.

## Full Documentation

`README.md`, `FINAL_PIPELINE.md`, `reports/boamp_methodology_chapter.pdf`,
`SURVIVAL_ANALYSIS_REPORT.md`, `TREND_ANALYSIS_REPORT.md`,
`DATA_QUALITY_REPORT.md`, `INTERNSHIP_GUIDE_COMPLIANCE.md`.
