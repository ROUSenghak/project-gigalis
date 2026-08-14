# Survival Analysis Report

Generated: `2026-08-14T10:14:42`  
Event: **accepted observable successor procurement**, not certified legal renewal.

## Cohort And Event Definition

The frozen primary cohort contains `3,800` awarded Grand
Ouest digital procurement episodes. `544` have an accepted
`M_B_text_ranking @ 0.70` successor and `3,256` have no
accepted observable successor before `2025-12-31`. The latter are right-censored; they
are not proven abandonments.

## Kaplan-Meier Results

- Event rate: `14.316%`.
- Censoring proportion: `85.684%`.
- Estimated probability of an observable successor by 12 months: `4.621%`.
- Estimated probability by 24 months: `6.733%`.
- Kaplan-Meier median: **not reached**.
- Segment log-rank: statistic `23.45`, p-value `3.26e-05`.

The median delay of `31.82` months reported elsewhere is the median among linked
events only. It is not the Kaplan-Meier median.

## Cox Model

The parsimonious model uses CPV segment, region, framework status, validated-SIREN
availability, and award year. Its in-sample C-index is
`0.626`. Framework episodes have HR
`1.751` and CPV-35 has HR
`1.553` relative to CPV-32.

The proportional-hazards diagnostic rejects constant effects for:
`award_year_centered, framework_flag, has_validated_siren`. These coefficients are
therefore descriptive time-averaged associations, not causal effects.

Temporal validation is weak: C-index is `0.607` on
2015–2021 and `0.518` on 2022–2025. The Cox model is not
validated for individualized operational prediction.

## Parametric Models And Indicators

`GeneralizedGamma` has the lowest AIC among exponential,
Weibull, log-logistic, log-normal, and generalized-gamma fits. Model selection does
not remove linkage uncertainty or guarantee tail extrapolation. The exported
`survival_conditional_probabilities.csv` gives 12/24-month conditional indicators
with 500-draw episode-bootstrap intervals.

## Detectability And Censoring Diagnostic

Linked and censored observations differ most on these standardized comparisons:

- `text_length_chars`: SMD `0.262`.
- `framework_flag`: SMD `0.187`.
- `administrative_followup_months`: SMD `0.146`.
- `award_year`: SMD `-0.137`.

These differences indicate differential observed-event detection and unequal
follow-up; they do not prove causal linkage bias. In particular, recent contracts
cannot yet show long successor gaps. Administrative censoring and missed successors
from imperfect linkage remain conceptually distinct but cannot be fully separated
with BOAMP alone.

## Linkage Sensitivity

Event counts range from `296` to
`1332` across the four retained linkage arms.
Absolute probabilities are therefore linkage-sensitive. Cox effects and subgroup
ordering should only be claimed where the exported sensitivity tables show stable
direction.

## Decision

The survival analysis is reproducible and complete for descriptive,
linkage-conditioned reporting. It is not a validated legal-renewal forecast. The
primary outputs are comparative KM results and age/segment risk indicators; external
linkage validation remains the main condition for stronger accuracy claims.
