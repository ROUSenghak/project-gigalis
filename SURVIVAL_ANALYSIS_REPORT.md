# Survival Analysis Report

Generated: `2026-08-20T11:54:58`  
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

### Temporal Validation

The model is fit once on `2015-2021` awards and scored out of time
without any refitting or retuning.

| Split | Train N | Train events | Test N | Test events | C-index train | C-index test |
|---|---:|---:|---:|---:|---:|---:|
| Primary, 2022-2024 | 2,470 | 392 | 1,004 | 107 | 0.606 | 0.479 |
| Sensitivity, 2022-2025 | 2,470 | 392 | 1,330 | 152 | 0.606 | 0.518 |

The primary split is the one the internship guideline specifies. The extended split
adds the 2025 award cohort, whose follow-up is shortest, and is carried only as a
sensitivity read.

Out-of-time discrimination is weak in both: a C-index near `0.5` means the model
does not usefully rank individual episodes by time to successor on unseen award
years. That is the result, not a prompt to retune. Part of it is structural, since
episodes awarded from 2022 onwards can only contribute short-gap events, but the
model has no demonstrated out-of-time discriminative power and nothing in the
operational deliverable rests on it.

## Parametric Models And Indicators

`GeneralizedGamma` has the lowest AIC among exponential,
Weibull, log-logistic, log-normal, and generalized-gamma fits. Model selection does
not remove linkage uncertainty or guarantee tail extrapolation.

The selected parametric model is **not** the source of the operational numbers.
Every horizon reported here falls inside the observed window, and the smooth
families flatten the observed renewal shoulder, so the 12/24-month conditional
probabilities in `survival_conditional_probabilities.csv` are read off the
Kaplan-Meier estimator, with 500-draw episode-bootstrap intervals. The generalized
gamma is reported as the best-fitting family and as the instrument any
extrapolation past `2025-12-31` would use.

## Borderline-Link Robustness

The frozen rule accepts at `0.70` on the M_B text score. Anchors whose best
candidate falls in `[0.65, 0.75]` are
the ones a small threshold perturbation would reclassify. Dropping that whole band —
borderline acceptances and borderline abstentions alike — removes
`280` episodes, of which `133` are
events, and gives:

| Analysis | Contracts | Events | KM 12m | KM 24m | CPV-35 HR | Framework HR |
|---|---:|---:|---:|---:|---:|---:|
| Main | 3,800 | 544 | 4.621% | 6.733% | 1.553 | 1.751 |
| Excluding borderline | 3,520 | 411 | 3.721% | 5.277% | 1.780 | 1.616 |

The direction of both headline hazard ratios is unchanged, so the comparative findings the project actually claims do not rest on borderline linkage decisions. The absolute KM level does move, which is the expected mechanical consequence of removing borderline events and is consistent with the four-arm linkage sensitivity: absolute probabilities remain threshold-uncertain and are not quoted alone. The band is a fixed `±0.05` around the frozen threshold; it was not
searched over, and the excluded episodes are removed rather than relabelled.

One coincidence to head off, because the same number appears twice in this project
with two unrelated meanings: `280` anchors fall in the borderline band, and
`280` anchors generated no candidate at all. These are different sets and
different questions. Every anchor removed here had a candidate and a best score
inside the band; anchors with no candidate are not near any threshold, since their
event status is decided by blocking rather than by the acceptance bar, and they stay
in the analysis as censored exposure.

## Template-Risk Robustness

The threshold arms and the borderline band both move where the acceptance bar
sits. Neither touches the false-positive mechanism the linkage audit actually
identified, because that mechanism produces links well *above* the bar: French
award notices carry long standardised framework boilerplate on which character
n-grams score highly between unrelated objects, and `M_B` ranks candidates within
each anchor independently, so one such episode can be accepted for several
anchors. A stricter threshold does not remove either signature.

Two observable signatures, both already published by the candidate-generation
audit, define the at-risk group: word-level similarity below
`0.50` (acceptance carried by the character
analyser, `65` links) or a successor episode
shared with another anchor (`127`
links). Together they flag `173` of the
`544` accepted links
(`31.8%`). Those anchors are **re-censored at the
cutoff** rather than dropped, because that is the counterfactual under test: a
spurious link means the anchor had no observed successor and should contribute its
full follow-up as censored exposure.

| Analysis | Contracts | Events | KM 12m | KM 24m | CPV-35 HR | Framework HR |
|---|---:|---:|---:|---:|---:|---:|
| Main | 3,800 | 544 | 4.621% | 6.733% | 1.553 | 1.751 |
| Re-censoring template-risk links | 3,800 | 371 | 2.639% | 3.884% | 1.541 | 1.692 |

This is the check the framework-agreement finding most needs, because framework boilerplate is the text that drives the mechanism: if the higher framework hazard were an artefact of shared legal wording, re-censoring these links would collapse it. Both headline hazard ratios keep their side of 1 and move little, so the comparative findings are not products of the documented false-positive mechanism. The absolute Kaplan-Meier level falls by roughly the share of events re-censored, which is arithmetic rather than evidence, and is consistent with the four-arm result that absolute probabilities are linkage-sensitive. The check bounds the mechanism's influence; it does not establish that the flagged links are false, and most of them are not.

## Operational 12- And 24-Month Probabilities

For a contract that has reached age `a` months with no accepted successor, the
probability that one becomes visible within the next `h` months is
`P(T <= a+h | T > a) = 1 - S(a+h)/S(a)`, read off the Kaplan-Meier estimator with
500-draw episode-bootstrap intervals. This is the study's operational output.

| Contract age | P(successor within 12m) | 95% CI | P(successor within 24m) | 95% CI |
|---:|---:|---|---:|---|
| 0 months | 4.621% | [3.912%, 5.239%] | 6.733% | [5.914%, 7.577%] |
| 12 months | 2.215% | [1.704%, 2.726%] | 4.260% | [3.554%, 4.980%] |
| 24 months | 2.091% | [1.575%, 2.598%] | 9.399% | [8.330%, 10.676%] |
| 36 months | 7.464% | [6.522%, 8.587%] | 9.693% | [8.579%, 11.117%] |
| 48 months | 2.409% | [1.765%, 3.069%] | 2.893% | [2.154%, 3.719%] |

The intervals are wide relative to the estimates, and the profile is not monotone
in age: it rises into the 36-48 month renewal shoulder and falls away after it.
These rank ages and segments; they are not calibrated individual forecasts, and
they estimate an *observable successor procurement appearing in BOAMP*, not a
certified renewal. Segment-level curves are in `survival_segment_summary.csv`.

## Detectability And Censoring Diagnostic

Linked and censored observations differ on these standardized comparisons:

| Variable | Linked mean | Censored mean | SMD |
|---|---:|---:|---:|
| `log_candidate_pool_size` | 4.787 | 3.972 | +0.470 |
| `candidate_pool_size` | 274.3 | 188.6 | +0.285 |
| `text_length_chars` | 1611 | 1087 | +0.262 |
| `framework_flag` | 0.2647 | 0.1867 | +0.187 |
| `administrative_followup_months` | 70.65 | 65.38 | +0.146 |
| `award_year` | 2020 | 2020 | -0.137 |
| `has_validated_siren` | 0.3107 | 0.3409 | -0.065 |
| `notice_count` | 1.978 | 2.045 | -0.058 |
| `has_reliable_duration` | 0.2445 | 0.2518 | -0.017 |

These differences indicate differential observed-event detection and unequal
follow-up; they do not prove causal linkage bias. In particular, recent contracts
cannot yet show long successor gaps. Administrative censoring and missed successors
from imperfect linkage remain conceptually distinct but cannot be fully separated
with BOAMP alone.

### Candidate-pool size is the largest imbalance

The largest of these is not a property of the contract at all. `M_B_text_ranking`
accepts the maximum text score over an anchor's candidate block, and the maximum of
more draws is larger, so an anchor whose buyer publishes prolifically is
mechanically more likely to yield an accepted link than an otherwise identical
anchor whose buyer publishes rarely. On the log scale the linked-versus-censored
standardized difference is
`+0.470`,
above every contract-level variable above it in the table.

This is a detectability channel, not a cause of re-procurement, so the response is
one **sensitivity** model rather than a change to the reported specification. The
main Cox model is unchanged. Adding `log(1 + candidate pool size)` to it gives:

| Covariate | HR, main model | HR, + log(candidate pool) | p, adjusted |
|---|---:|---:|---:|
| `framework_flag` | 1.751 | 1.617 | 2.6e-06 |
| `has_validated_siren` | 1.082 | 0.983 | 0.87 |
| `award_year_centered` | 1.107 | 1.145 | 4.1e-11 |
| `digital_segment_CPV-35` | 1.553 | 1.512 | 0.00085 |
| `digital_segment_CPV-48` | 0.828 | 0.769 | 0.048 |
| `digital_segment_CPV-72` | 1.056 | 0.977 | 0.83 |
| `buyer_region_Normandie` | 0.800 | 0.796 | 0.046 |
| `buyer_region_Pays de la Loire` | 1.003 | 0.975 | 0.81 |
| `log_candidate_pool_size` | -- | 1.184 | 6e-09 |

The added term is itself strongly associated with an observed event
(HR `1.1837`, p `6e-09`), which is what a detectability
channel looks like.

Two readings follow, and they differ:

- **CPV-35 is largely insensitive to it.** The hazard ratio moves from
  `1.553` to
  `1.512`
  (p `0.00085`). Together with its
  stability across the four linkage arms, the borderline band, and template-risk
  re-censoring, the CPV-35 result is the most robust comparative finding here and
  can be stated as such.
- **The framework association is partly detectability.** Its hazard ratio
  attenuates from `1.751` to
  `1.617`
  (p `2.6e-06`), roughly
  `14%`
  of the log hazard ratio. The direction survives every check the study runs, but
  buyers who use framework agreements also publish more, and publishing more raises
  the chance that a max-over-block text score clears `0.70`. The association is real
  and smaller than the main model alone implies; template boilerplate is not the
  only alternative explanation, and differential detectability is the other.

Neither statement is causal. Candidate-pool size is a description of a buyer's
publication volume, and this model adjusts for it to separate observability from
behaviour -- it does not identify an effect of either.

## Linkage Sensitivity

Event counts range from `296` to
`1332` across the four retained linkage arms
(`M_B` at `0.80`, `0.70`, `0.60`, and the `M_C` weighted-gated contrast at `0.70`).
Absolute probabilities are therefore linkage-sensitive. Cox effects and subgroup
ordering should only be claimed where the exported sensitivity tables show stable
direction.

These are linkage-conditioned estimates. Missed successors may reduce the observed
event rate, whereas residual false links may increase it. They should therefore not
be interpreted as formal lower bounds on true re-procurement probability.

## Decision

The survival analysis is reproducible and complete for descriptive,
linkage-conditioned reporting. It is not a validated legal-renewal forecast. The
primary outputs are comparative KM results and age/segment risk indicators; external
linkage validation remains the main condition for stronger accuracy claims.
