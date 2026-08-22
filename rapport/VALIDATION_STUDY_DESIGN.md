# Research design: independent validation of the observable-successor event definition

Prepared 21 August 2026. Sampling frames computed from the frozen artifacts
(`accepted_successor_links.parquet`, `linkage_candidates_scored.parquet`, `survival_dataset.parquet`).
Nothing in the pipeline was modified.

---

## 1. Research question

> Under the frozen rule `M_B_text_ranking @ 0.70`, what is the measurement error of the observable-successor
> event, and what does that error imply for the reported survival estimates?

Two estimands, not one:

$$\pi = \Pr(\text{an accepted link is a genuine observable successor}) \quad \text{— precision}$$
$$\beta = \Pr(\text{a censored anchor actually has an observable successor}) \quad \text{— missed-link rate}$$

## 2. Why $\beta$ matters more than $\pi$ — the design-driving calculation

Let $L = 544$ accepted links, $U = 3{,}256$ censored anchors, $N = 3{,}800$. The bias-corrected event count is

$$N_{\text{true}} \;\approx\; \pi L + \beta U$$

Therefore the sensitivities are

$$\frac{\partial N_{\text{true}}}{\partial \pi} = L = 544
\qquad\text{versus}\qquad
\frac{\partial N_{\text{true}}}{\partial \beta} = U = 3{,}256$$

**$\beta$ carries six times the leverage of $\pi$.** A 10-point error in $\pi$ moves the event rate by 1.4 pp;
a 10-point error in $\beta$ moves it by 8.6 pp.

| $\beta$ | $\pi=0.70$ | $\pi=0.80$ | $\pi=0.90$ |
|---:|---|---|---|
| 0.02 | 446 (11.7 %) | 500 (13.2 %) | 555 (14.6 %) |
| 0.05 | 544 (14.3 %) | 598 (15.7 %) | 652 (17.2 %) |
| 0.10 | 706 (18.6 %) | 761 (20.0 %) | 815 (21.5 %) |
| 0.15 | 869 (22.9 %) | 924 (24.3 %) | 978 (25.7 %) |

The published 14.3 % is exactly the cell $(\pi = 0.70,\ \beta = 0.05)$. The plausible range spans
**11.7 % to 25.7 %**, and almost all of that width comes from $\beta$.

> This is the argument for changing the existing protocol: it samples accepted links only, so it would
> estimate the *less* consequential parameter and leave the dominant one unidentified.

---

## 3. Recommended design

**Stratified two-frame validation study with blinded independent human adjudication.**

Cross-sectional, retrospective, measurement-validation design. Not a new data collection on the market —
a measurement study on an existing instrument.

### Frame A — precision (judge a proposed pair)

Stratified by acceptance score band × risk signature. Cell sizes are exact:

| | clean | risk-flagged | total |
|---|---:|---:|---:|
| text_component 0.70–0.75 (borderline) | 68 | 65 | 133 |
| 0.75–0.85 (middle) | 139 | 63 | 202 |
| 0.85–1.00 (high) | 164 | 45 | 209 |
| **total** | 371 | 173 | **544** |

"Risk-flagged" = word-similarity < 0.50 (65 links) or successor shared with another anchor (127 links);
173 links carry at least one signature.

### Frame B — missed links (search an anchor)

Stratified by the anchor's *best* candidate score, which is the natural importance-sampling variable:

| Stratum | best text_component | $N_h$ | share of censored |
|---|---|---:|---:|
| B1 near-miss | 0.62 – 0.70 | 234 | 7.2 % |
| B2 | 0.55 – 0.62 | 271 | 8.3 % |
| B3 | 0.40 – 0.55 | 792 | 24.3 % |
| B4 remote | < 0.40 | 1,679 | 51.6 % |
| B0 no candidate | — | 280 | 8.6 % |

B0 needs no review: those anchors were excluded by blocking, and the blocking ceiling (0.913) already
describes that loss.

## 4. Inclusion and exclusion criteria

**Include.** Any cohort episode in `survival_dataset.parquet` (awarded, Grand Ouest, ≥1 digital CPV,
resolvable award date, awarded 2015–2025).

**Exclude.**

- anchors in stratum B0 (no candidate generated) — nothing for a reviewer to judge;
- any anchor already reviewed in the 120-anchor regional reference or the 60-pair diagnostic, to keep the
  new estimate independent of the material that produced the frozen rule;
- links whose anchor and candidate texts are both empty after normalisation (none observed, but state the rule).

## 5. Sample size and justification

Wilson 95 % half-widths:

```
  n   | p=0.70  p=0.75  p=0.85
 ---- | ------  ------  ------
   20 |  0.187   0.178   0.154     <- current evidence
  100 |  0.088   0.084   0.070
  120 |  0.081   0.077   0.064     <- recommended
  200 |  0.063   0.060   0.049
```

| Tier | Frame A | Frame B | Reviewer time | What it buys |
|---|---:|---:|---:|---|
| Minimum | 60 | 40 | ~11 h | halves the interval on $\pi$; a crude $\beta$ |
| **Recommended** | **120** | **100** | **~25 h** | $\pi$ to ±0.08; $\beta$ to ±0.03–0.05; a defensible corrected rate |
| Ideal | 200 | 160 | ~40 h | $\pi$ to ±0.06; stratum-specific $\pi_h$ usable to re-weight |

Assumes ~4 minutes to judge a proposed pair and ~10 minutes to search an anchor. The recommended tier is
about three working days of one reviewer.

**Allocation.** Frame A proportional to cell size, with the borderline × risk-flagged cell oversampled
(it is where the decision-relevant uncertainty lives). Frame B by Neyman allocation:

```
stratum                   N_h   assumed beta_h   n_h (of 100)
B1 near-miss 0.62-0.70    234        0.35            20
B2           0.55-0.62    271        0.15            18
B3           0.40-0.55    792        0.05            31
B4 remote      < 0.40   1,679        0.01            30
```

> The assumed $\beta_h$ are used **only** to allocate effort. The estimate itself uses the observed labels.
> Pre-register the allocation so the choice cannot be revisited after seeing results.

## 6. Data-collection procedure

1. **Draw** the sample with a recorded seed and commit the frame, the strata and the allocation to git
   **before** any file is exported.
2. **Export a blinded reviewer file.** Anchor and candidate object text, buyer name, CPV codes, dates,
   procedure type. **No** score, no stratum, no algorithm decision, no indication of which frame a row
   came from. Frames A and B are interleaved in one shuffled file so the reviewer cannot infer the task.
3. **Frame A task:** *does this later procurement plausibly continue or replace the anchor's need?*
   Decision in `{Y, N, UNCERTAIN}` plus a one-line evidence note.
4. **Frame B task:** the reviewer receives the anchor plus its top 25 candidates by date order (not by
   score) and answers *is any of these a successor, and which?* Recording the ordering rule this time
   removes the circularity that damaged the historical reference.
5. **Hold the audit key** in a separate file until every row is labelled.
6. **Second reviewer** on a 25 % subsample of both frames, for a Cohen's $\kappa$.
7. **Adjudicate** disagreements by discussion, recording the reason; never by algorithm score.

## 7. Analysis

**Stratified estimators.** With $N_h$ the stratum size and $W_h = N_h/\sum_h N_h$:

$$\hat\pi = \sum_h W_h \hat\pi_h, \qquad
\widehat{\operatorname{Var}}(\hat\pi) = \sum_h W_h^2 \frac{\hat\pi_h(1-\hat\pi_h)}{n_h}\Big(1-\tfrac{n_h}{N_h}\Big)$$

The finite-population correction matters here: $n_h/N_h$ reaches 0.3 in the small strata.

**Bias-corrected event count and rate**, with a bootstrap interval over reviewer-labelled units:

$$\hat N_{\text{true}} = \hat\pi L + \hat\beta U, \qquad \widehat{\text{rate}} = \hat N_{\text{true}}/N$$

**Propagation into the survival estimates.** Do not simply rescale $\hat S(t)$. Instead re-run the frozen
survival pipeline under two re-labelled datasets:

- *lower arm*: drop the accepted links the reviewer rejected, re-censoring those anchors;
- *upper arm*: additionally convert reviewer-identified missed successors into events at their observed dates.

Report the resulting band as a **fifth sensitivity arm**, alongside the four existing ones. It is the only
arm anchored in human judgement rather than in a threshold choice.

**Do not** retune the 0.70 threshold on these labels. That would convert the study into a tuning set and
destroy the pre-registration property that makes the locked split readable as held out.

## 8. Validity and trustworthiness measures

| Threat | Measure |
|---|---|
| Reviewer anchoring on the algorithm | Full blinding: no scores, no decisions, no strata in the reviewer file |
| Post-hoc rationalisation | Frame, allocation, estimators and stopping rule committed to git before export |
| Reviewer unreliability | 25 % double-reviewed; report Cohen's $\kappa$; adjudicate by evidence, not by score |
| Circular candidate exposure | Frame B candidates ordered by **date**, and the ordering rule recorded — the specific failure of the historical reference |
| Selective reporting | Publish per-stratum $\hat\pi_h$ and $\hat\beta_h$, including the strata that look bad |
| Instrument drift | Re-run against the frozen `survival_dataset.parquet`, SHA-256 recorded in the manifest |

## 9. Ethical considerations

- **Data.** BOAMP notices are open public records; no personal data. Buyer names identify legal persons,
  not natural ones, so GDPR obligations are minimal — but individual named contacts occasionally appear in
  notice text and should be stripped from the reviewer file.
- **Reviewer.** Obtain explicit agreement on time commitment; ~25 hours is a real cost. Credit them in the
  report.
- **Conflict of interest.** A Gigalis employee may recognise buyers and hold prior beliefs about them.
  Record the reviewer's role and relationship to the buyers, and prefer someone outside the team that
  built the pipeline.
- **Reputational care.** Per-buyer judgments must not be published in a form that appears to audit a named
  organisation's procurement conduct. Report aggregates only.
- **Honest provenance.** If any part is model-assisted, record it as the project already does in
  `review_provenance.json`. Do not describe an assisted review as independent human validation.

## 10. Likely limitations

1. **The reviewer is not ground truth either.** A domain expert judging from notice text is a better
   instrument than an LLM pass, not a perfect one. Report $\kappa$ so the reader can see the instrument's
   own reliability.
2. **$\hat\beta$ is bounded below by search effort.** A reviewer who sees 25 candidates cannot find a
   successor outside that set, so $\hat\beta$ estimates *missed within the exposed pool*, not missed overall.
   Blocking loss remains separately described by the 0.913 ceiling.
3. **Frame B strata rest on the score being validated.** Importance sampling by best-candidate score is
   efficient but score-dependent; the estimator is unbiased under the stratum weights, yet the *precision*
   of $\hat\beta$ depends on the score being informative.
4. **No retrospective repair.** This study cannot fix the unrecorded export rule of the historical
   120-anchor reference. It replaces that evidence going forward; it does not rehabilitate it.
5. **Generalisability.** Estimates apply to Grand Ouest digital procurement 2015–2025 under this rule.

## 11. Why each choice fits the question

| Choice | Why |
|---|---|
| Two frames, not one | $\beta$ has 6× the leverage of $\pi$ on the headline number |
| Stratification by score band | Precision varies with score; strata make $\hat\pi_h$ usable to re-weight or to justify dropping the borderline band |
| Importance sampling in Frame B | A uniform sample of 3,256 censored anchors would spend half its effort on cases with best score < 0.40 |
| Blinding | The reviewer must not reconstruct the algorithm's decision |
| Pre-registration before export | It is the property that already makes the frozen threshold defensible; keep it |
| Fifth sensitivity arm, not a replacement | Preserves the project's central message — levels are conditional, comparisons are stable |

## 12. One alternative design, with trade-offs

**Capture–recapture (dual-system estimation), zero human labour.**

Treat two linkage methods as two independent "captures" of the same population of true successors. With
$n_1$ links from `M_B`, $n_2$ from `M_C`, and $m$ in both, the Lincoln–Petersen estimator gives

$$\hat N_{\text{true}} \approx \frac{n_1 n_2}{m}$$

and the missed count follows.

**Advantages.** Free, immediate, computable tonight from artifacts already in the repository, and it uses a
comparator arm you have already built.

**Trade-offs, and why it is not the primary design.**

- The independence assumption **fails by construction**: `M_B` and `M_C` share the same text component, so
  captures are positively dependent. That biases $\hat N_{\text{true}}$ **downward** and therefore
  *understates* the missed-link rate.
- The bias direction is known but its magnitude is not, so the estimate cannot be corrected.
- It says nothing about $\pi$: two methods agreeing on a wrong successor is a shared error, not evidence.

**Recommended use.** Run it as a cheap consistency check to see whether its (downward-biased) missed-link
estimate is at least as large as the human-based $\hat\beta$. If capture–recapture implies *more* missed
links than the review found, that is an informative contradiction worth reporting.

---

## 13. Note on the second study

The technology-annotation reliability study ($\kappa$ on a re-annotated sample) follows the same logic with
a smaller frame: stratify the 500 annotated notices by predicted class and by classifier confidence,
oversample the boundary classes (`MIXED`, `OTHER_DIGITAL`, `CLOUD_HOSTING`) where the error triage located
7 of 30 errors, and target $n \approx 150$ for a $\kappa$ with a usable interval. It is the second priority,
not the first, because the classifier is an enrichment layer while the event definition is load-bearing.
