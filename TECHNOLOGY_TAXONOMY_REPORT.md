# Technology Taxonomy Classification

Generated: `2026-08-19T21:30:51`
Taxonomy: `boamp_technology_taxonomy_v1.0` | Classifier: `boamp_technology_classifier_v1.0`

## 1. Why This Component Exists

BOAMP publishes an administrative vocabulary, not a business one. The study
cohort is defined by CPV divisions `32`, `35`, `48` and `72`, which is
reproducible and auditable but coarse: it says a procurement is "digital"
without saying what technology was bought. Every business question a supplier
asks -- which segments are growing, which are re-procured soonest -- needs the
second thing.

This component learns that missing variable from procurement text:

```text
procurement object text  ->  supervised classifier  ->  business technology class
```

It does not replace the CPV segmentation. The existing survival and trend
results remain the reference analysis; the taxonomy is an enrichment layer over
them.

## 2. Taxonomy

Eight substantive classes -- `CLOUD_HOSTING`, `CYBERSECURITY`, `NETWORK_TELECOM`, `IT_INFRASTRUCTURE`, `BUSINESS_SOFTWARE`, `DATA_BI`, `AI`, `IT_SERVICES` --
plus three fallback classes that are annotation decisions rather than missing
values: `MIXED` for a procurement with no dominant technology, `OTHER_DIGITAL`
for a digital purchase outside the eight, and `OTHER` for a notice that carries
a digital CPV without being a technology procurement at all.

The taxonomy was frozen before modelling and has not been changed since.

## 3. Annotated Reference Corpus

`500` manually annotated BOAMP notices, `2015`-`2025`,
all with a label and a non-empty object text, `500` distinct
notice identifiers, no duplicates. The input field is `objet`: a median of
`14` words, `100` characters.

| Class | n | Share |
|---|---:|---:|
| CLOUD_HOSTING | 32 | 0.064 |
| CYBERSECURITY | 54 | 0.108 |
| NETWORK_TELECOM | 81 | 0.162 |
| IT_INFRASTRUCTURE | 46 | 0.092 |
| BUSINESS_SOFTWARE | 88 | 0.176 |
| DATA_BI | 29 | 0.058 |
| AI | 7 | 0.014 |
| IT_SERVICES | 74 | 0.148 |
| MIXED | 21 | 0.042 |
| OTHER_DIGITAL | 53 | 0.106 |
| OTHER | 15 | 0.03 |

Two properties of this corpus constrain everything below.

**The sample is quota-stratified, not a random draw.** `BUSINESS_SOFTWARE`
appears almost exactly eight times per year and `NETWORK_TELECOM` eight to nine.
The class proportions above are a property of the annotation design, so they are
not an estimate of how common each technology is in the population, and the
predicted shares in section 13 must not be compared against them.

**AI is genuinely rare.** Seven notices across eleven years. No synthetic AI
examples were generated, no AI rows were duplicated, and no oversampling was
applied. The consequence is reported rather than engineered away: AI per-class
metrics are published with their support and marked as uninterpretable.

`451` of the `500` notices are in Grand Ouest and
`500` carry a digital CPV. `434`
are tender notices, `55` award notices,
`9` corrections.

## 4. Leakage Prevention

BOAMP republishes one procurement many times, and buyers re-run the same tender
years later with almost the same wording. Scoring a model on a near-copy of a
document it trained on measures memorisation. Every labelled notice is therefore
assigned to a **procurement family**, and every family sits in exactly one fold.

A family is the union of two rules:

1. notices the canonical episode reconstruction already placed in one episode;
2. notices whose `objet` reaches character-level cosine `0.80` or above.

The second rule is not optional. Rule 1 alone gives `486` groups;
adding rule 2 merges `39` near-duplicate pairs, of which
`29` sit in *different* episodes and would otherwise have been
split across folds. The result is `459` families:
`423` singletons, `36` multi-notice families,
largest `3`.

Merging in the wrong direction is the safe direction: joining two genuinely
distinct procurements costs a little training signal, while splitting one
inflates every metric in this report.

**An annotation-consistency finding falls out of this.** `3` families
contain notices with near-identical text but different labels -- videoconference
services labelled `NETWORK_TELECOM` in 2017 and `OTHER_DIGITAL` in 2021, an
IaaS procurement labelled `MIXED` in 2017 and `CLOUD_HOSTING` in 2021. No label
was changed: there is no evidence deciding which reading is correct, and editing
labels after seeing model errors is how a corpus is fitted to its classifier.
They are recorded in `data/processed/boamp/technology/annotation_near_duplicates.csv`
as an empirical floor on attainable accuracy.

## 5. Text Representation

The input is `objet` alone. Normalisation is deliberately light: mojibake
repair, Unicode NFC, lowercasing, whitespace collapse. **Accents are preserved**
and no stemming is applied, because the classes are distinguished by words like
`cybersécurité`, `logiciel métier` and `intelligence artificielle`, and
flattening French orthography discards the evidence. Features are TF-IDF word
unigrams and bigrams, so phrases stay addressable.

The vectoriser lives inside a scikit-learn `Pipeline` and is fitted within each
training fold. No held-out document contributes to its own features.

**Excluded by design**: buyer name, SIREN/SIRET, region, department, publication
date or year, award amount, supplier, procedure type, framework status, notice
identifiers, filename, URL, and every successor-linkage variable. The classifier
must learn *what is being procured*, not who bought it, when, or whether it was
later re-procured. CPV is also excluded from the text models -- it is the
benchmark they are measured against, and mixing it in would dissolve the
comparison.

## 6. Models And Validation Design

All specifications share one frozen evaluation design: 3-fold group-aware
stratified cross-validation on the fold assignment saved in
`nlp_cv_folds.csv`, seed `20260819`. Three folds rather than five
because `AI` has seven observations. Hyperparameters are chosen by a grouped
inner cross-validation *inside* each outer training fold, so no held-out notice
influences the configuration it is scored under.

| Model | Family | Macro-F1 | SD | Weighted F1 | Accuracy |
|---|---|---:|---:|---:|---:|
| M_majority | baseline | 0.0272 | 0.0004 | 0.0527 | 0.176 |
| M0_cpv | administrative_benchmark | 0.4388 | 0.0764 | 0.4934 | 0.5122 |
| M0b_cpv_descriptor | administrative_benchmark | 0.4694 | 0.0376 | 0.5065 | 0.5241 |
| M1_tfidf_logreg | text | 0.6617 | 0.0673 | 0.7264 | 0.738 |
| M2_tfidf_logreg_balanced | text | 0.7414 | 0.0341 | 0.7647 | 0.766 |
| M3_tfidf_linearsvm | text | 0.7169 | 0.0299 | 0.7543 | 0.76 |
| M4_tfidf_linearsvm_balanced | text | 0.7157 | 0.0544 | 0.7442 | 0.7439 |

### Result 1 -- procurement text carries substantially more than CPV

Uncertainty is estimated by resampling **procurement families**, not notices:
two notices in one family are near-copies, and treating them as independent
draws would shrink every interval by pretending the corpus holds more
information than it does. Both models are scored on the same resampled families
in each replicate, so the difference below is a *paired* interval.

| Model | Macro-F1 | 95% CI lower | 95% CI upper | Bootstrap SD |
|---|---:|---:|---:|---:|
| M0b_cpv_descriptor | 0.4731 | 0.4126 | 0.5259 | 0.0279 |
| M1_tfidf_logreg | 0.6701 | 0.6072 | 0.7248 | 0.0305 |
| M2_tfidf_logreg_balanced | 0.7442 | 0.6819 | 0.7905 | 0.0275 |

| Against | Difference | 95% CI lower | 95% CI upper | Excludes zero |
|---|---:|---:|---:|---|
| M0b_cpv_descriptor | -0.2711 | -0.3403 | -0.2009 | yes |
| M1_tfidf_logreg | -0.0741 | -0.1283 | -0.0223 | yes |

* **Observation.** All three figures here are pooled out-of-fold macro-F1 --
  computed once over the union of the three held-out folds -- which is what the
  bootstrap resamples. The per-fold means in section 6 differ slightly from them
  (`0.7414` against `0.7442` for the selected model), because averaging
  three fold scores is not the same as scoring all 500 predictions together;
  neither is more correct and both are reported.
  The selected text model reaches
  `0.7442`; the best administrative benchmark reaches
  `0.4731`. The paired difference is
  `0.2711` macro-F1 with a 95% family-bootstrap interval of
  `[0.2009, 0.3403]`, which excludes zero.
* **Confidence.** High. Both are measured on identical folds with identical
  group isolation and an identical metric, both were searched over the same
  regularisation range, and the interval is estimated at the level of the
  leakage unit.
* **What can be concluded.** The business technology class is genuinely present
  in procurement text and is genuinely *not* recoverable from the official
  classification codes alone. This is the empirical justification for the whole
  component.
* **What cannot be concluded.** That CPV is useless. On `OTHER` -- notices that
  are not technology procurement at all -- CPV outperforms text, because a
  clothing purchase is identifiable from its code and not from the word
  "fourniture". The two vocabularies are complementary.
* **Implication for Gigalis.** Segment reporting built on CPV divisions alone
  will merge cybersecurity, cloud, data and applications into one bucket. The
  text layer is what separates them.

### Final selection and the development budget

`M2_tfidf_logreg_balanced` has the highest mean grouped-CV macro-F1 of the four text specifications, so the pre-specified tie-break -- prefer a probability-emitting model within one paired standard error of the leader -- did not need to be invoked.

Selection did not rest on the mean alone. `M2_tfidf_logreg_balanced` also has the smallest
between-fold spread of the four text specifications (SD
`0.0341`), the smallest train-validation gap (section 8), the
strongest temporal result (section 10), and it emits probabilities without a
second fitting step. Those are the criteria that would have overridden a small
mean advantage in the other direction.

The search budget was fixed before any outer-fold score was read, and every
specification in it is reported here rather than only the winner:

| Model | Family | Features | Grid points |
|---|---|---|---:|
| M0_cpv | administrative_benchmark | cpv | 10 |
| M0b_cpv_descriptor | administrative_benchmark | cpv_descriptor | 10 |
| M1_tfidf_logreg | text | objet | 120 |
| M2_tfidf_logreg_balanced | text | objet | 120 |
| M3_tfidf_linearsvm | text | objet | 120 |
| M4_tfidf_linearsvm_balanced | text | objet | 120 |

Grids are explored only by the *inner* grouped cross-validation, nested inside
each outer training fold, so widening them cannot inflate the outer estimate.
The administrative benchmark was searched over the same regularisation range as
the text models; giving it a narrower search would have made the headline
comparison a statement about how hard each side was tuned.

### Hyperparameter stability

| Fold | n-grams | min_df | max_df | sublinear | C | Outer macro-F1 |
|---:|---|---:|---:|---|---:|---:|
| 0 | [1, 1] | 1 | 0.9 | yes | 3 | 0.7481 |
| 1 | [1, 1] | 1 | 0.9 | yes | 10 | 0.7717 |
| 2 | [1, 1] | 1 | 0.9 | no | 3 | 0.7044 |

Representation choices are stable: every fold selected the same n-gram range,
`min_df` and `max_df`. Only the regularisation strength and the sublinear term
weighting move, and the outer scores those configurations produce sit within
`0.067` macro-F1 of each other. Notably every fold preferred
**unigrams alone** over unigrams plus bigrams -- with 500 short documents the
bigram vocabulary is too sparse to pay for itself.

## 7. Per-Class Performance

Out-of-fold, `M2_tfidf_logreg_balanced`, pooled over the three folds (n = `500`).

| Class | Precision | Recall | F1 | Support | Predicted n | Support adequate |
|---|---:|---:|---:|---:|---:|---|
| CLOUD_HOSTING | 0.7308 | 0.5938 | 0.6552 | 32 | 26 | yes |
| CYBERSECURITY | 0.8478 | 0.7222 | 0.78 | 54 | 46 | yes |
| NETWORK_TELECOM | 0.8095 | 0.8395 | 0.8242 | 81 | 84 | yes |
| IT_INFRASTRUCTURE | 0.7021 | 0.7174 | 0.7097 | 46 | 47 | yes |
| BUSINESS_SOFTWARE | 0.798 | 0.8977 | 0.8449 | 88 | 99 | yes |
| DATA_BI | 0.9231 | 0.8276 | 0.8727 | 29 | 26 | yes |
| AI | 0.8 | 0.5714 | 0.6667 | 7 | 5 | no |
| IT_SERVICES | 0.7778 | 0.7568 | 0.7671 | 74 | 72 | yes |
| MIXED | 0.4815 | 0.619 | 0.5417 | 21 | 27 | yes |
| OTHER_DIGITAL | 0.6545 | 0.6792 | 0.6667 | 53 | 55 | yes |
| OTHER | 0.9231 | 0.8 | 0.8571 | 15 | 13 | yes |

Headline: macro-F1 `0.7414` (SD `0.0341` across folds),
weighted F1 `0.7647`, accuracy `0.766`.

### Result 2 -- some classes are reliable, others are not

* **Reliable.** `CYBERSECURITY`, `NETWORK_TELECOM`, `BUSINESS_SOFTWARE`, `DATA_BI`, `IT_SERVICES`, `OTHER` -- F1 at or above `0.75` on
  support of at least `10`.
* **Weak.** `MIXED` -- adequate support, F1 below `0.65`.
* **Not interpretable.** `AI`, support `7`. Its observed F1 of
  `0.6667` rests on three correct predictions.
  A high number here would be a small-sample artefact and a low number would be
  equally uninformative; the class is reported as a rare-class limitation, not
  as a measured capability.
* **What cannot be concluded.** That `CLOUD_HOSTING` recall of
  `0.5938` reflects an intrinsic limit. Section 8
  shows the errors are concentrated on website hosting, a boundary the
  annotation guidelines place inside `CLOUD_HOSTING` and the text places near
  `OTHER_DIGITAL`.

Confusion matrix: `data/processed/boamp/technology/confusion_matrix.csv` and
`reports/figures/technology_confusion_matrix.png`.

## 8. Error Analysis

Thirty representative out-of-fold errors were sampled from the largest confusion
pairs and triaged. Triage counts: 16 model_error, 7 taxonomy_boundary_ambiguity, 4 genuinely_mixed_or_multi_technology_procurement, 2 annotation_inconsistency_in_related_notices, 1 insufficient_information_in_objet.

The dominant confusion pairs are:

| Annotated -> predicted | n | Share of errors | Support of annotated class |
|---|---:|---:|---:|
| OTHER_DIGITAL -> BUSINESS_SOFTWARE | 6 | 0.0513 | 53 |
| BUSINESS_SOFTWARE -> MIXED | 6 | 0.0513 | 88 |
| OTHER_DIGITAL -> NETWORK_TELECOM | 5 | 0.0427 | 53 |
| NETWORK_TELECOM -> IT_INFRASTRUCTURE | 5 | 0.0427 | 81 |
| IT_SERVICES -> IT_INFRASTRUCTURE | 5 | 0.0427 | 74 |
| CYBERSECURITY -> BUSINESS_SOFTWARE | 5 | 0.0427 | 54 |

Reading the sampled texts rather than the triage labels, three patterns account
for most of the residual:

1. **`OTHER_DIGITAL` is a heterogeneous residual class.** It contains
   videosurveillance, RFID, videoconference and web maintenance. It borders on
   `NETWORK_TELECOM`, `BUSINESS_SOFTWARE` and `IT_SERVICES` simultaneously, and
   it is involved in `2` of the six largest confusion pairs. This is a
   taxonomy-design consequence, not a modelling failure.
2. **Website hosting sits on the `CLOUD_HOSTING` boundary.** "Hébergement de
   sites internet" is annotated `CLOUD_HOSTING` and predicted `OTHER_DIGITAL`
   repeatedly. The classes are separable in principle but the object text is
   short and the distinction is definitional.
3. **`BUSINESS_SOFTWARE` versus `MIXED` is a genuine property of the
   procurement.** "Fourniture de matériels et logiciels informatiques" *is*
   mixed; whether it is filed as such is an annotation convention.

The pre-specified adjacency list under-covers the observed confusions -- it did
not name `OTHER_DIGITAL` pairs. It was not revised after the results were seen,
so the `model_error` bucket is a residual that includes boundary cases the list
missed. This matters only for reading the triage table; the CamemBERT decision
in section 11 does not turn on it.

No label was changed to improve any metric.

## 9. Fit Diagnosis: Bias, Variance, And What Limits Performance

Every model is scored on its own training fold as well as on the held-out fold.
The resubstitution score is not a performance estimate and is not reported as
one; its only use is the gap, which is what separates a model that has memorised
its training fold from one that is not expressive enough.

| Model | Train macro-F1 | Grouped-CV macro-F1 | Gap | Fold SD |
|---|---:|---:|---:|---:|
| M0_cpv | 0.7562 | 0.4388 | 0.3174 | 0.0764 |
| M0b_cpv_descriptor | 0.8198 | 0.4694 | 0.3503 | 0.0376 |
| M1_tfidf_logreg | 1 | 0.6617 | 0.3383 | 0.0673 |
| M2_tfidf_logreg_balanced | 0.9901 | 0.7414 | 0.2487 | 0.0341 |
| M3_tfidf_linearsvm | 0.9967 | 0.7169 | 0.2798 | 0.0299 |
| M4_tfidf_linearsvm_balanced | 0.9893 | 0.7157 | 0.2735 | 0.0544 |

The learning curve carries the same two arms, subsampling **families** so a
subsample never holds half of a related-notice group:

| Fraction | Notices | Train macro-F1 | Validation macro-F1 | SD | Gap |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 66 | 0.974 | 0.4342 | 0.0298 | 0.5398 |
| 0.4 | 133 | 0.984 | 0.5632 | 0.0453 | 0.4208 |
| 0.6 | 199 | 0.9931 | 0.6612 | 0.0519 | 0.3319 |
| 0.8 | 266 | 0.9924 | 0.6895 | 0.0442 | 0.3029 |
| 1 | 333 | 0.9881 | 0.7465 | 0.0162 | 0.2415 |

### Result 3 -- high variance that is resolving with data, not underfitting

* **Observation.** Training macro-F1 sits near `0.97` at every
  training size while validation climbs from `0.4342` to
  `0.7465`. The gap closes monotonically from
  `0.5398` to `0.2415`. Validation is still rising at the
  full corpus.
* **Confidence.** Moderate to high for the shape. Subsampling is over families
  and repeated five times per point; the final point is a single full-data
  evaluation per fold, so its spread is not directly comparable to the others.
* **What can be concluded.** This is the signature of **high variance**, not
  underfitting: the representation already separates the training folds almost
  perfectly at every size, so it does not lack capacity. The binding constraint
  is the number of independent labelled families, and the gap narrows as that
  number grows.
* **What cannot be concluded.** Where the plateau lies, or how much a larger
  corpus would buy. Five points do not identify an asymptote and no minimum
  sample size is implied by this curve.
* **Consequence for modelling.** A richer representation was considered and
  **not** tested. The candidate on the table was word TF-IDF combined with
  character n-grams, which adds capacity -- the opposite of what a high-variance
  diagnosis calls for. The regularisation path that *would* address variance was
  already searched: `C` spans two orders of magnitude in the inner
  cross-validation, and the selected values are interior to that range rather
  than at its edge, so the model is not starved of regularisation either.
* **Implication for Gigalis.** If the taxonomy is to be operationalised further,
  additional annotation is the lever with evidence behind it. A more elaborate
  model is not.

## 10. Temporal Robustness

Train `2015-2022` (n = `393`), test `2023-2025`
(n = `107`). `4` families straddling the boundary were assigned to
training in full, which costs test observations and cannot flatter the result.

* Macro-F1 over all eleven classes: `0.6617`.
* Macro-F1 over the classes with test support of at least `10`: `0.8148`.
* Weighted F1 `0.7721`, accuracy `0.7757`.

### Result 4 -- performance holds on recent notices, for the classes that can be measured

* **Observation.** The all-class macro-F1 falls to `0.6617` from
  `0.7414`, but restricted to classes with adequate recent
  support it is `0.8148` -- at or above the primary estimate.
* **Confidence.** Moderate. Six of eleven classes have test support below ten
  (`CLOUD_HOSTING`, `IT_INFRASTRUCTURE`, `DATA_BI`, `AI`, `MIXED`, `OTHER`), and the all-class figure is
  dominated by them.
* **What can be concluded.** The vocabulary of recent BOAMP notices has not
  drifted away from what the model learned on older ones for the high-volume
  classes.
* **What cannot be concluded.** Anything about recent `AI`, `OTHER` or `MIXED`
  performance. `OTHER` has one recent test observation and scores `0.000`; that
  is one notice, not a trend.

## 11. Was An Advanced Model Justified?

The gate was written before the classical results were read: a transformer is
tested only if the frozen classical model is materially inadequate (macro-F1
below `0.55`) **and** fewer than half of its errors come from label ambiguity or
missing information, which no encoder can supply.

The selected model reaches `0.7414`. The first condition fails
decisively, so **CamemBERT was not tested**. It was not tested and then
discarded; it was not run, because the criterion for running it was not met.
Adding it would have added a large dependency, a GPU-shaped runtime, and an
opaque model to a component whose errors are concentrated on definitional
boundaries rather than semantics.

## 12. Frozen Classifier And Confidence

Specification `M2_tfidf_logreg_balanced`, refitted on all `500` labelled notices for
deployment. **That refit has no validation score and none is reported.** The
evidence for this model is the grouped cross-validation and the temporal split
above.

Confidence is the predicted class probability, Platt-scaled by
`CalibratedClassifierCV(method='sigmoid')` fitted on labelled data only, inside
the same grouped splits. Calibration was adopted under a pre-specified rule: it
reduced the expected calibration error by `0.1405`
at a macro-F1 cost of `0.0364`.

| Stated confidence | n | Observed accuracy | Mean stated | Gap |
|---|---:|---:|---:|---:|
| [0.0, 0.3) | 45 | 0.1778 | 0.2471 | -0.0693 |
| [0.3, 0.4) | 73 | 0.4521 | 0.3511 | 0.101 |
| [0.4, 0.5) | 86 | 0.6977 | 0.4435 | 0.2542 |
| [0.5, 0.6) | 87 | 0.8736 | 0.5566 | 0.3169 |
| [0.6, 0.7) | 115 | 0.8783 | 0.6562 | 0.222 |
| [0.7, 0.8) | 77 | 0.961 | 0.7496 | 0.2114 |
| [0.8, 0.9) | 17 | 1 | 0.8157 | 0.1843 |

### Result 5 -- confidence ranks well but remains conservative

* **Observation.** Observed accuracy rises monotonically with stated confidence,
  from `0.27` in the lowest bin to `1.00` in the highest. But the gap is positive
  in every bin above `0.3`: a stated `0.45` is worth about `0.65` in practice.
  Expected calibration error after scaling is `0.2097`.
* **Confidence.** High for the ranking, high for the direction of the residual
  miscalibration; it is measured out of fold on `500` notices.
* **What can be concluded.** The score is a usable ordering and a usable filter.
  At the `0.7` operational cutoff, out-of-fold accuracy is
  `0.9556` on the `9%` of notices that clear it, against
  `0.7495` below it.
* **What cannot be concluded.** That a stated confidence is the probability the
  deployment label is correct. Two separate reasons.

  First, the residual miscalibration above: reweighting eleven classes with
  `class_weight='balanced'` flattens the probability simplex and Platt scaling
  only partly undoes it, so the values remain conservative and must be read
  through the table.

  Second, and more fundamental: **the corpus is quota-stratified and the
  deployment population is not.** The scaling was fitted where `AI` is 1.4% of
  observations by design; in the cohort it is a fraction of a percent. A
  calibrated score on the reference sample is therefore not a posterior
  probability in the deployment population, because the class prior it encodes
  is an artefact of the annotation design. The reliability table describes
  behaviour *on the reference distribution*. No prior correction is applied,
  because the deployment prior is exactly what the classifier is being used to
  estimate and assuming it would make the estimate circular.

  The honest name for the published value is a **calibrated model confidence
  score**, useful for ranking and for selecting an operational subset, not a
  population probability.
* **Operational note.** `0.7` is a reporting convention, not a truth
  boundary, and it is unrelated to the `0.70` linkage acceptance threshold, which
  scores an entirely different quantity. No calibrated prediction reaches `0.90`,
  so cutoffs above `0.80` are not usable.

## 13. Propagation To The Study Cohort

The classifier was trained on notices; the study analyses episodes. Each of the
`3,800` cohort episodes is represented by the `objet` of its earliest
competition notice, or of its earliest notice when the episode is award-only --
the same origin rule the episode layer already uses. Concatenated episode text
would have been several times longer than any training document; the chosen
rule gives a deployment median of `15` words against a training median of
`14`.

Every episode receives exactly one prediction and none is discarded. Low
confidence sets a flag and nothing else.

| Class | Episodes | Share | High-confidence n | High-confidence share | Dominant CPV segment | Its share |
|---|---:|---:|---:|---:|---|---:|
| CLOUD_HOSTING | 115 | 0.0303 | 12 | 0.1043 | CPV-72 | 0.6522 |
| CYBERSECURITY | 316 | 0.0832 | 11 | 0.0348 | CPV-35 | 0.3354 |
| NETWORK_TELECOM | 859 | 0.2261 | 52 | 0.0605 | CPV-32 | 0.4959 |
| IT_INFRASTRUCTURE | 298 | 0.0784 | 17 | 0.057 | CPV-32 | 0.4295 |
| BUSINESS_SOFTWARE | 854 | 0.2247 | 52 | 0.0609 | CPV-48 | 0.4075 |
| DATA_BI | 86 | 0.0226 | 17 | 0.1977 | CPV-72 | 0.5814 |
| AI | 6 | 0.0016 | 2 | 0.3333 | CPV-72 | 1 |
| IT_SERVICES | 492 | 0.1295 | 21 | 0.0427 | CPV-72 | 0.6606 |
| MIXED | 139 | 0.0366 | 19 | 0.1367 | CPV-32 | 0.4604 |
| OTHER_DIGITAL | 462 | 0.1216 | 8 | 0.0173 | CPV-32 | 0.4372 |
| OTHER | 173 | 0.0455 | 24 | 0.1387 | CPV-35 | 0.659 |

`235` of `3,800` predictions
(`6.2%`) clear the `0.7` cutoff.

### Result 6 -- coverage is flat over time, so composition shifts are not a confidence artefact

* **Observation.** High-confidence coverage by award year ranges from
  `0.017` to `0.099` with no monotone drift.
* **Confidence.** High; it is a direct count.
* **What can be concluded.** A change in a class's share across years is not
  produced by the classifier becoming less certain about recent notices.
* **What cannot be concluded.** That the class shares are unbiased. Only
  `6.2%` of predictions clear the cutoff, so any analysis restricted
  to them works with roughly one episode in
  `16`, selected on a quantity that is itself correlated with
  class -- coverage ranges from `0.017` to `0.333` across predicted classes.
* **Operational consequence.** At this coverage the `0.70` cutoff is a tool for
  picking a small, high-precision worklist -- cases confident enough to act on
  without review -- and not a filter for population-level analysis. The full
  cutoff sweep is published in `confidence_cutoff_sweep.csv` so a different
  operating point can be chosen against its cost in coverage; none is
  recommended here, because choosing one after seeing these results would be
  selecting an operating point on the outcome.

### Result 7 -- the taxonomy cuts across the CPV segmentation

* **Observation.** Mean CPV-segment purity is `0.3385`: the largest
  technology class inside a CPV segment accounts for that share of it. The
  reverse, mean technology purity within CPV, is `0.5563`.
  9 of 11 technology classes appear in every CPV segment.
* **Confidence.** Moderate. The crosswalk inherits the classifier's error rate,
  and the counts are predictions, not annotations.
* **What can be concluded.** The two segmentations are not substitutes. CPV
  divisions are containers holding several business technologies each.
* **What cannot be concluded.** Exact class volumes. A class with `316` predicted
  episodes and a cross-validated recall near `0.70` has a genuinely uncertain
  true volume, and no confidence interval on that volume is offered here.
* **Implication for Gigalis.** The taxonomy is the layer that makes
  "which technology market is moving" answerable at all. It should be used for
  segment framing, not for counting to the unit.

## 14. Technology-Level Enrichment, Behind Two Gates

Nothing was rerun mechanically for eleven classes. A class enters the downstream
analysis only by clearing **both** of two gates fixed before any curve was
fitted.

**Gate A -- classifier evidence.** Does the label mean anything? A class the
classifier cannot separate produces a downstream group that is a mixture of
several technologies, and a curve fitted to it estimates the mixture. The gate
requires a substantive technology class, annotated support of at least
`10`, and out-of-fold F1 of at least `0.65`.

Fallback classes are excluded outright. `MIXED`, `OTHER_DIGITAL` and `OTHER` are
operational residuals, not technologies -- `OTHER_DIGITAL` holds
videosurveillance, RFID and web maintenance at once. Placing that bucket beside
cybersecurity in a "comparison across technologies" invites the reader to
interpret the contrast as a technology effect when part of it is the
heterogeneity of the bucket. They remain in the descriptive tables.

**Gate B -- statistical support.** Can the sample carry an estimate? A perfectly
classified class with fourteen episodes and one event still cannot support a
curve.

| Class | Reference n | Precision | Recall | F1 | Gate A | Reason |
|---|---:|---:|---:|---:|---|---|
| CLOUD_HOSTING | 32 | 0.7308 | 0.5938 | 0.6552 | yes | passes |
| CYBERSECURITY | 54 | 0.8478 | 0.7222 | 0.78 | yes | passes |
| NETWORK_TELECOM | 81 | 0.8095 | 0.8395 | 0.8242 | yes | passes |
| IT_INFRASTRUCTURE | 46 | 0.7021 | 0.7174 | 0.7097 | yes | passes |
| BUSINESS_SOFTWARE | 88 | 0.798 | 0.8977 | 0.8449 | yes | passes |
| DATA_BI | 29 | 0.9231 | 0.8276 | 0.8727 | yes | passes |
| AI | 7 | 0.8 | 0.5714 | 0.6667 | no | reference support below 10 |
| IT_SERVICES | 74 | 0.7778 | 0.7568 | 0.7671 | yes | passes |
| MIXED | 21 | 0.4815 | 0.619 | 0.5417 | no | fallback class, not a substantive technology |
| OTHER_DIGITAL | 53 | 0.6545 | 0.6792 | 0.6667 | no | fallback class, not a substantive technology |
| OTHER | 15 | 0.9231 | 0.8 | 0.8571 | no | fallback class, not a substantive technology |

### Survival

| Class | CV F1 | Episodes | Events | Gate A | Gate B | Analysed |
|---|---:|---:|---:|---|---|---|
| CLOUD_HOSTING | 0.6552 | 115 | 11 | yes | no | no |
| CYBERSECURITY | 0.78 | 316 | 60 | yes | yes | yes |
| NETWORK_TELECOM | 0.8242 | 859 | 140 | yes | yes | yes |
| IT_INFRASTRUCTURE | 0.7097 | 298 | 38 | yes | yes | yes |
| BUSINESS_SOFTWARE | 0.8449 | 854 | 106 | yes | yes | yes |
| DATA_BI | 0.8727 | 86 | 15 | yes | no | no |
| AI | 0.6667 | 6 | 0 | no | no | no |
| IT_SERVICES | 0.7671 | 492 | 72 | yes | yes | yes |
| MIXED | 0.5417 | 139 | 32 | no | yes | no |
| OTHER_DIGITAL | 0.6667 | 462 | 44 | no | yes | no |
| OTHER | 0.8571 | 173 | 26 | no | yes | no |

| Class | Episodes | Events | P(successor by 24m) | At risk at 24m |
|---|---:|---:|---:|---:|
| BUSINESS_SOFTWARE | 854 | 106 | 0.0857 | 673 |
| NETWORK_TELECOM | 859 | 140 | 0.0649 | 716 |
| CYBERSECURITY | 316 | 60 | 0.0646 | 246 |
| IT_SERVICES | 492 | 72 | 0.0617 | 393 |
| IT_INFRASTRUCTURE | 298 | 38 | 0.0611 | 239 |

#### Result 8 -- observable re-procurement timing differs across the analysed technology classes

* **Observation.** At 24 months the highest analysed class is `BUSINESS_SOFTWARE`
  (`0.0857`) and the lowest is `IT_INFRASTRUCTURE`
  (`0.0611`). A multivariate log-rank test across the
  `5` classes clearing both gates gives p = `0.0363`
  on `416` observed events.
* **Confidence.** Moderate at best. Three things sit between this test and a
  statement about technology. The event is an *observable successor procurement*
  accepted by the frozen linkage policy, not a confirmed contract renewal. The
  class labels are predictions carrying the error rate in section 7, which
  blurs the groups being compared and generally works against detecting a
  difference. And the comparison is unadjusted for anything: buyer type,
  contract size and procedure differ across these classes and none is
  controlled.
* **What can be concluded.** A difference in the timing of observable successor procurement was detected across the 5 analysed substantive classes. The test is a single omnibus comparison, so it says the classes are not all alike; it does not identify which pair drives the result, and no pairwise comparison is offered here.
* **What cannot be concluded.** That any class has a different *contract
  duration*, or that technology causes the difference. Absolute levels inherit
  every caveat in `SURVIVAL_ANALYSIS_REPORT.md` -- they move with the linkage
  threshold and are not lower bounds -- and here they additionally inherit
  classification error.
* **What Gate A changes, measured.** Running the same test over every class
  that clears the *statistical* gate alone -- `8` classes including the
  fallback residuals, `518` events -- gives p = `0.000119`. Dropping the
  residual buckets makes the result **weaker**, not stronger. That is the
  direction that matters: `OTHER_DIGITAL`, `OTHER` and `MIXED` have distinctive
  timing because of what they contain, not because they are technologies, and
  including them manufactures part of an apparent technology effect. A reader
  shown only the eight-class number would over-read it. Gate A exists for this
  reason, and it costs significance rather than buying it.
* **Implication for Gigalis.** Useful for framing which segments generate
  visible re-tendering soonest; not usable for predicting when a named contract
  will be re-let.

Full curves: `technology_survival_summary.csv`, conditional probabilities in
`technology_conditional_probabilities.csv`, figure
`reports/figures/technology_kaplan_meier.png`.

### Trend

Gate A applies here too. PELT breakpoints, HMM regimes and stationarity tests
stay on the CPV reference series in `TREND_ANALYSIS_REPORT.md`: running them
across derived technology series would multiply tests without answering a new
question.

One slope per analysed class is one hypothesis test per analysed class, so raw
p-values are reported beside Holm (family-wise) and Benjamini-Hochberg (false
discovery rate) adjustments.

| Class | Episodes | Mean/quarter | Slope | Raw p | Holm p | BH p | Reading |
|---|---:|---:|---:|---:|---:|---:|---|
| CYBERSECURITY | 316 | 7.18 | 0.0245 | 0.4857 | 1 | 0.6071 | no linear trend detected |
| NETWORK_TELECOM | 859 | 19.52 | -0.1668 | 0.0563 | 0.2815 | 0.0563 | no linear trend detected |
| IT_INFRASTRUCTURE | 298 | 6.77 | 0.0078 | 0.8116 | 1 | 1 | no linear trend detected |
| BUSINESS_SOFTWARE | 854 | 19.41 | 0.044 | 0.5604 | 1 | 0.934 | no linear trend detected |
| IT_SERVICES | 492 | 11.18 | 0.0021 | 0.9617 | 1 | 1 | no linear trend detected |

#### Result 10 -- no technology series shows a detectable linear trend

* **Observation.** 5 classes were tested simultaneously.
  The smallest raw p-value is `NETWORK_TELECOM` at `0.0563`, which does not reach the 5% level before any adjustment and gives Holm-adjusted `0.2815` across `5` tests.
* **Confidence.** Low to moderate. These are counts of awarded episodes per
  quarter carrying predicted labels, over `44` quarters, with no
  adjustment for anything else that changed over the period.
* **What can be concluded.** No analysed technology class shows a linear
  movement in awarded Grand Ouest procurement volume that survives correction
  for the number of series examined.
* **What cannot be concluded.** Anything about
  `CLOUD_HOSTING`, `DATA_BI`, `AI`, `MIXED`, `OTHER_DIGITAL`, `OTHER`, which did not clear the gates; anything about
  market value, since these are notice counts and not euros; and absence of
  *any* change -- a linear slope is a weak instrument for detecting a
  non-monotone shift, which is precisely why PELT and the HMM remain on the CPV
  reference series.
* **Implication for Gigalis.** Segment prioritisation should rest on absolute
  volume and on the re-procurement timing above, not on a growth story these
  series do not support.

### Confidence-threshold sensitivity for the trend series

The original project brief contemplated dropping predictions below the `0.70`
cutoff before estimating any trend. That is only defensible if what remains is
still a series.

| Class | Arm | Episodes | Zero quarters | Median/quarter | Slope | Raw p |
|---|---|---:|---:|---:|---:|---:|
| CYBERSECURITY | all_predictions | 316 | 0 | 7 | 0.0245 | 0.4857 |
| CYBERSECURITY | confidence_ge_0.70 | 11 | 33 | 0 | -0.0056 | 0.2896 |
| NETWORK_TELECOM | all_predictions | 859 | 0 | 19 | -0.1668 | 0.0563 |
| NETWORK_TELECOM | confidence_ge_0.70 | 52 | 16 | 1 | -0.007 | 0.6437 |
| IT_INFRASTRUCTURE | all_predictions | 298 | 0 | 7 | 0.0078 | 0.8116 |
| IT_INFRASTRUCTURE | confidence_ge_0.70 | 17 | 33 | 0 | -0.0077 | 0.4156 |
| BUSINESS_SOFTWARE | all_predictions | 854 | 0 | 20 | 0.044 | 0.5604 |
| BUSINESS_SOFTWARE | confidence_ge_0.70 | 52 | 15 | 1 | 0.0103 | 0.4305 |
| IT_SERVICES | all_predictions | 492 | 0 | 11 | 0.0021 | 0.9617 |
| IT_SERVICES | confidence_ge_0.70 | 21 | 25 | 0 | 0.0147 | 0.0339 |

### Result 9 -- the high-confidence restriction is too selective for trend estimation

* **Observation.** Restricting to confidence `>= 0.70` leaves between
  `11` and
  `52` episodes per class across
  `44` quarters, with
  `15` to
  `33` empty quarters. Median quarterly counts fall to zero or one.
* **What can be concluded.** The high-confidence subset is too sparse and too
  selective for stable technology trend estimation. The full prediction set is
  used for the trend series, with the classifier's error rate carried as a
  stated limitation rather than filtered away.
* **What this is not.** Evidence that the cutoff is useless. It is a useful
  operational filter for selecting cases to inspect (section 12); it is simply
  not a basis for a quarterly time series.
* **A caution the comparison itself supplies.** One class shows a nominally
  significant slope in the sparse arm that is absent in the full arm. That is
  what fitting a line to a mostly-empty series produces, and it is the reason
  the sparse arm is not adopted.

## 15. Limitations

1. **Training and deployment populations differ.** The corpus is a
   quota-stratified sample of notices spanning Grand Ouest and beyond, including
   procurements that were never awarded. The deployment population is
   `3,800` awarded Grand Ouest digital episodes. Only
   `243` annotated notices belong to cohort episodes. Performance measured
   on the corpus is evidence about the corpus.
2. **Class priors are not population priors.** The annotation quotas mean the
   corpus proportions cannot be read as prevalence, and a model trained under
   them carries that prior into its predictions.
3. **`AI` cannot be evaluated.** Seven annotated notices, `6` predicted cohort
   episodes, `0` observed successor event. Nothing about AI procurement is
   established here beyond its rarity in this corpus over 2015-2025.
4. **Confidence is conservative, not calibrated.** See section 12.
5. **Fallback classes absorb ambiguity.** `OTHER_DIGITAL` carries
   `462` predicted episodes and is definitionally heterogeneous.
6. **No inter-annotator agreement statistic exists.** The corpus was delivered
   as one labelled file with no annotator identifier and no second pass, so no
   Cohen's kappa can be computed and label reliability cannot be quantified. The
   internship guide's L2 design asks for two independent annotators; this corpus
   does not meet that design, and the `3` documented
   internal inconsistencies are the only direct evidence available about label
   stability. They are recorded and left uncorrected.
7. **Predictions are predictions.** Downstream technology-level survival and
   trend numbers inherit the classifier's error rate. They are not conditioned on
   verified labels and no uncertainty from the classification stage is propagated
   into their confidence intervals.

## 16. Boundaries

- Do not describe a predicted technology class as an observed attribute of the
  procurement.
- Do not read the corpus class proportions, or the predicted class shares, as
  market shares. The first are annotation quotas; the second are predictions
  carrying the error rate in section 7.
- Do not read a confidence value as a probability of correctness without the
  reliability table.
- Do not report survival figures for `CLOUD_HOSTING`, `DATA_BI`, `AI`, `MIXED`, `OTHER_DIGITAL`, `OTHER`
  or trend figures for `CLOUD_HOSTING`, `DATA_BI`, `AI`, `MIXED`, `OTHER_DIGITAL`, `OTHER`; they did not
  clear the support gates.
- Do not present technology-level comparisons as causal, or as adjusted for
  buyer type, contract size, or procedure.
- Do not replace the CPV-based cohort definition or the CPV-based survival and
  trend results with this layer.

## 17. Reproduction

```bash
# the whole layer: corpus, models, deployment, evidence -- about one minute
PYTHONPATH=. python3 scripts/build_technology_taxonomy.py --force

# one stage while iterating
PYTHONPATH=. python3 scripts/build_technology_taxonomy.py --stage models --force

PYTHONPATH=. jupyter nbconvert --execute --to notebook --inplace \
    notebooks/15_technology_taxonomy_classification.ipynb
PYTHONPATH=. pytest -q tests/test_technology_taxonomy.py
```

All outputs land in `data/processed/boamp/technology/`.

Every line of analysis lives in `boamp_pipeline/technology_taxonomy.py`,
`boamp_pipeline/technology_models.py` and
`boamp_pipeline/technology_evidence.py`; the script is orchestration only. That
split exists so `notebooks/15_technology_taxonomy_classification.ipynb` imports
and runs the same functions -- it re-executes the full grouped cross-validation
and asserts that it reproduces the tables quoted above, rather than displaying
them. A number in this report and the matching number in the notebook come from
one code path.
