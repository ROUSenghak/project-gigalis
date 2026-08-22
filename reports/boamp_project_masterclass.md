# BOAMP Observable-Successor Study: Complete Teaching and Defense Guide

## Technical summary
<!-- source:protocol -->

This project does **not** observe legal contract renewal. It reconstructs procurement processes from BOAMP notices, identifies later procurements that look like continuations of the same need, treats those algorithmically accepted links as **observable-successor events**, and studies their timing with survival analysis. A separate text classifier predicts business-technology classes for enrichment; it never defines a successor.

The canonical pipeline is internally consistent as of 22 August 2026. It standardises **1,620,712 unique notices**, reconstructs **1,103,632 episodes**, selects **3,800 awarded Grand Ouest episodes containing at least one CPV code in divisions 32/35/48/72**, exposes **763,417 anchor-candidate pairs**, and accepts **544 successors** under the frozen `M_B_text_ranking @ 0.70` rule. The resulting **14.32%** event rate and all Kaplan–Meier levels are linkage-conditioned, not renewal prevalence. The strongest comparative result is that CPV-35 has a higher observed-successor hazard than CPV-32 (HR **1.553**, 95% CI **1.218–1.981**), remaining **1.512** after candidate-pool adjustment. The Cox model is useful associationally but weak predictively: out-of-time C-index **0.479** for 2022–2024.

The technology model uses group-aware three-fold validation on **500 labelled notices in 459 procurement families**. Balanced multinomial logistic regression on word-TF-IDF achieves out-of-fold macro-F1 **0.744** (95% family-bootstrap CI **0.682–0.791**), versus **0.473** (**0.413–0.526**) for the best CPV/descriptor benchmark; paired improvement **0.271** (**0.201–0.340**). These are classification estimates on the annotation corpus, not proof that predicted class shares are market shares.

The evidence hierarchy is uneven. Structural pipeline integrity is strong. Comparative CPV-35 survival evidence is relatively robust. Absolute successor probabilities are sensitivity-dependent. Linkage accuracy is provisional because the reference labels are a single LLM research pass, subset spot-checked rather than independently specialist-reviewed. This project is descriptive and associational; **it does not establish causality or validated individual forecasting**.

## 1. The research problem before any model
<!-- source:protocol -->

### Problem and why it matters

BOAMP is France's official bulletin for public-procurement notices. A row describes a publication: for example a call for competition, a correction, an award, or another procedural notice. It can carry a notice identifier, publication and deadline dates, buyer name and sometimes SIREN/SIRET, postal geography, CPV codes, object and description text, procedure and framework wording, declared durations, lots, and amount candidates. It does **not** provide a reliable field saying “this new procurement replaces that earlier contract.”

The real-world phenomenon of interest is recurrent demand: after a public buyer procures a digital need, when does a later procurement appear that plausibly continues or replaces it? The ideal dataset would contain a stable contract identifier, buyer and supplier legal identities, signed and expiry dates, option exercises, amendments, termination, and an explicit predecessor/successor or renewal relationship. BOAMP instead observes publications. Legal renewal can occur without a new BOAMP notice, and a new notice can be a re-tender, replacement, extension, new phase, parallel lot, or unrelated purchase.

### Idea and why it makes sense

The project constructs an observable proxy:

> An **observable successor procurement** is a later BOAMP procurement episode from the same plausible buyer that appears to continue or replace the same need and is accepted by a frozen linkage rule.

“Observable” says the event must leave a BOAMP trace. “Successor” avoids claiming a particular legal mechanism. “Procurement” says the matched object is a later buying process, not necessarily an amendment to the same legal contract.

The concepts are distinct:

| Concept | Meaning | Directly observed here? |
|---|---|---|
| Legal renewal | Exercise or continuation under contract law | No |
| Procurement continuation | The same substantive need continues | Not directly |
| Successor procurement | A later buying process replaces/continues a need | Inferred |
| Observable successor in BOAMP | A later episode accepted by the documented linkage rule | Constructed |

The legitimate claim is about **time to an accepted observable successor under a specified rule**. The study cannot claim that all 544 links are legal renewals, that censored episodes were never re-procured, or that hazard-ratio differences are causal.

The whole logic in one sentence is:

> Because we cannot observe **legal renewal** directly, we construct **observable-successor procurements** using **standardised BOAMP episodes, buyer/chronology blocking, and frozen text-based linkage**, and then analyse **right-censored time to that constructed event**.

### Quantity and goal labels

Observed quantities come from BOAMP or annotation: notice date, raw buyer identifier, CPV, text, reviewed label. Constructed quantities are deterministic or algorithmic: standardised field, episode, candidate set, accepted link, censoring row. Estimated quantities come from statistical procedures: Kaplan–Meier probability, hazard ratio, confidence interval, trend slope. Predicted quantities come from machine learning: technology class and confidence score. Keeping these labels separate prevents a prediction from being retold as an observation and a proxy from being retold as legal truth.

## 2. Complete project map and why every arrow exists
<!-- source:protocol -->

```text
Official BOAMP JSONL, 2015–2025
  ↓ schemas differ; raw fields are not comparable
Schema-aware notice standardisation
  ↓ one procurement can publish several notices
Procurement episode reconstruction
  ↓ the research scope is awarded Grand Ouest digital demand
Awarded Grand Ouest any-digital-CPV cohort
  ↓ a successor must be searched for among later episodes
Buyer-and-time candidate generation
  ↓ blocking only creates plausible comparisons; it does not decide
Four linkage methods on the same pairs
  ↓ survival requires one event or an abstention per anchor
M_B top-text candidate accepted at 0.70, otherwise abstain
  ↓ episodes without an accepted link still contain follow-up information
Right-censored survival dataset
  ↓ censored observations invalidate ordinary event-only averages
Kaplan–Meier, Cox, parametric and sensitivity analyses
```

Parallel enrichment branch:

```text
500 labelled procurement-object texts
  ↓ related notices can leak across ordinary random folds
459 procurement families and grouped cross-validation
  ↓ text must become numeric evidence
word TF-IDF + balanced multinomial logistic regression
  ↓ predictions are uncertain and some classes are weak or small
out-of-fold validation, bootstrap, calibration decision and quality gates
  ↓ only defensible predicted groups should be analysed
technology-specific survival and quarterly trend enrichment
```

The trend branch asks a different time question:

```text
awarded episodes → award quarter → quarterly counts → OLS / PELT / ADF-KPSS / HMM
```

Survival time is age since an anchor award; trend time is calendar quarter. Mixing them would confuse “when a need reappears after award” with “how total procurement activity changes through history.”

## 3. Raw notices: why one publication is not one procurement
<!-- source:protocol -->

### Problem

Suppose a city launches a two-lot IT tender in March, corrects the response deadline in April, publishes an award in July, and publishes separate lot information. Those four publications are administrative views of one underlying process, not four independent purchasing needs.

### Practical example

```text
Notice A, 10 March: call for tenders — finance software + training
Notice B, 2 April: correction — deadline changed
Notice C, 20 July: award result
Notice D, 21 July: award details for lot 2
```

At notice grain, B appears 23 days after A and C one hundred days later. A notice-level successor algorithm could link A→B or B→C and invent very short “renewal” times. It could also count lots separately and let an award notice link to its own consultation.

### Output, interpretation, limitation, next step

Raw standardisation keeps one row per official notice. The project has **1,620,712** such unique rows from 2 March 2015 through 31 December 2025, including **209,502** geolocated to Grand Ouest. This is an observed publication layer, not yet a procurement-process layer. The next step must reconstruct episodes so repeated administrative observations do not become demand events.

## 4. Standardisation: making heterogeneous schemas comparable
<!-- source:quality -->

### Core idea

The raw BOAMP format changes across legacy XML-derived schemas, simplified forms and eForms. Standardisation uses field-specific parsers and keeps both the raw value and its lineage. “Unresolved” is preferred to a generic JSON guess because a wrong buyer, date or amount becomes strong false evidence downstream.

| Field | Raw problem | Rule and mechanism | Example | Error prevented |
|---|---|---|---|---|
| Dates | strings, timestamps, invalid dates | parse an explicit `YYYY-MM-DD` pattern; normalise datetimes to UTC; flag impossible order | `2025/04/03` → `2025-04-03` | negative durations, wrong chronology |
| SIREN/SIRET | punctuation, wrong length, arbitrary IDs | keep digits; validate 9- or 14-digit identifiers with Luhn; derive SIREN from valid SIRET | `123 456 789 00012` → validated SIRET + first 9 digits as SIREN | false legal-entity match |
| Buyer name | case, accents, punctuation, schema variants | Unicode normalisation, case folding, punctuation/space collapse; legal terms retained at standardisation | `VILLE  DE  NANTES` → `ville de nantes` | missed same-buyer comparison |
| CPV | main and lot codes live in different paths | find eight-digit CPV patterns across schema-aware locations; preserve all unique codes and identify a preferred main code | main `45000000`, lot `72000000` → both retained | excluding a digital lot |
| Text | object/title/description scattered | collect object, titles and descriptions; normalise whitespace; deduplicate | repeated title appears once | noisy or empty similarity input |
| Geography | buyer vs service-location fields differ | use contracting-buyer structured postcode, derive department and Grand Ouest region | `44000` → department 44, Pays de la Loire | wrong regional cohort |
| Framework | inconsistent explicit fields | search normalised procurement text for `accord cadre` / `framework agreement` | phrase found → flag 1 | missing a Cox covariate |
| Duration | days, months, years, start/end dates; conflicts | convert typed measures to months; require plausible 0–240; consensus or conflict; never global-impute | 730 days → 23.98 months | fabricated expiry dates |
| Missing values | empty strings can resemble agreement | encode absence as empty/NaN; in weighted scoring, missing evidence drops out | both SIRENs missing ≠ match | “unknown = same” error |
| Duplicates | repeated file extraction | enforce unique official `idweb` | duplicate ID count = 0 | double counting |
| Lineage | cleaning can hide source | retain `raw_source_file`, raw fields, parser version and record SHA-256 | transformed value traceable to raw record | unauditable corrections |

Validated identifiers dominate names because they identify legal units. If two validated SIRENs conflict, similar names must not override them: “Ville de X” and “Métropole de X” can share tokens while being legally distinct. The canonical checks find **zero accepted links with conflicting validated SIRENs** and **zero municipality/intercommunal mixes**. The limitation is that validated SIREN is missing for **2,521/3,800 = 66.34%** of cohort episodes, so name-based identity remains necessary and weaker.

Reliable duration is present for only **953/3,800 = 25.08%** and missing for **2,847/3,800 = 74.92%**. Its completeness rises from 11.8% in 2023 to 84.4% in 2025, proving that missingness changes with schema/time. A universal four-year imputation would encode an assumption as data.

## 5. Episode reconstruction: the first constructed research object
<!-- source:protocol -->

### Problem and idea

The award notice alone may omit original text, consultation timing, complete CPVs, corrected buyer details or lot structure. The idea is to treat notices as nodes in a graph and trustworthy relationships as edges. Every connected component becomes an inferred procurement episode.

### Method

Edges are considered in decreasing reliability:

1. shared `contractFolderID`;
2. explicit BOAMP links to earlier notices;
3. exact normalised procedure reference for the same buyer within 730 days.

Before joining two components, a validated-buyer conflict vetoes the merge. A union-find algorithm joins accepted edges efficiently. The episode then aggregates all notice IDs, earliest substantive competition date (or earliest notice), latest publication date, natures, CPVs, deduplicated text, buyer evidence, region, framework flag and consistent duration.

### Worked example

```text
A consultation ─explicit link─ B correction ─same folder─ C award
                                      └same reference/buyer─ D lot result
```

All four become episode E1. A later 2023 consultation E2 remains a separate component because it has a new folder/reference and is years later. E2 may be a successor; B cannot be, because B is already inside E1.

### Actual calculation and output

**1,620,712 notices** become **1,103,632 episodes**, a reduction of **517,080 publications (31.9%)** at the change of grain. Episode sizes are 662,798 singletons, 379,750 with two notices, 50,423 with three, 8,387 with four, and 2,274 with five or more. Accepted reconstruction evidence includes 45,885 folder edges, 584,896 explicit-link edges and 195,043 exact-reference/same-buyer edges. All notices are assigned once; there are zero buyer-conflict episodes and zero impossible chronologies.

This prevents duplicate events, self-linkage, artificially short gaps and administrative republication being counted as renewed demand. It does **not** prove semantic perfection: episode reconstruction is inferred, and 3,344 unusual/conflicted cases are exported for review.

## 6. Cohort construction and its selection effects
<!-- source:quality -->

### Filtering chain

| Sequential filter | Episodes remaining | Removed at step |
|---|---:|---:|
| Grand Ouest reconstructed episodes | 144,269 | — |
| at least one CPV in 32/35/48/72 | 7,376 | 136,893 |
| carrying an award notice | 3,826 | 3,550 |
| resolvable award date by cutoff | **3,800** | 26 |

The unit is one awarded episode. Grand Ouest supplies a coherent operational territory; award status provides a defensible time origin; the CPV rule narrows to digital-related demand; a valid award date is required to measure elapsed time. Each restriction may select procurements with better reporting and excludes needs purchased or published differently.

For episode `i`, let `CPV_i` be all its codes. The indicator is:

```text
D_i = 1 if there exists c in CPV_i whose two-digit division is 32, 35, 48 or 72;
D_i = 0 otherwise.
```

“Any-code” differs from “main-code-only.” Example: a construction framework has main CPV 45000000, lot 1 electrical work, and lot 2 network integration CPV 72000000. Main-only exclusion calls the whole episode non-digital; any-code inclusion retains the digital lot. In the actual cohort, **1,176/3,800 = 30.95%** enter although their main CPV is outside the digital set. Therefore the precise name is “awarded Grand Ouest episodes containing at least one qualifying digital CPV,” not simply “3,800 digital contracts.”

For mutually exclusive segment curves, `digital_segment` is the lowest qualifying division. It affects **412/3,800 = 10.84%** with multiple digital divisions. Among episodes with a digital main CPV, it agrees with the main division 94.7%; the measured sensitivity is small, but the tie-break is still arbitrary.

## 7. Successor notation and candidate blocking
<!-- source:protocol -->

Let `i` be the awarded anchor episode; `j` a later episode; `u_i` the anchor award date; `v_j` the candidate's first-publication date; and `J_i` the candidates surviving blocking. The problem is to choose one `j` in `J_i`, or choose none, that plausibly continues the same need.

This is record linkage/entity resolution because there is no labelled outcome column to predict directly; two records must be judged as representing a substantive predecessor/successor relationship. Abstention is essential because many anchors have no plausible observed successor.

### Why blocking exists

Exhaustive comparison would create billions of obvious non-matches and many chances for an unrelated text to score highly. Blocking asks only “worth comparing?”, never “is the successor?” Candidates require:

- same buyer key or same normalised buyer name;
- no conflict between two validated SIRENs;
- candidate not equal to anchor;
- `u_i + 90 days ≤ v_j ≤ u_i + 2,920 days`.

Municipal prefixes such as “commune de” may be removed for blocking, but intercommunal legal forms are preserved. Geography is already constrained by the Grand Ouest pool and is not an acceptance feature. CPV is deliberately **not** a hard block: **9/23 reviewed successors cross CPV divisions**; hard same-division blocking would reduce the attainable ceiling from 0.913 to 0.609.

The 90-day lower bound removes parallel lots and concurrent programmes; the shortest reviewed successor gap is 139 days, so it removes none of the 23 reviewed positives. The 2,920-day upper bound bounds computation and implausibly remote matches and excludes none of those reviewed successors. It is not an eight-year contract assumption.

Actual output: **763,417 pairs** for **3,520/3,800 anchors (92.63%)**; **280** anchors have no candidate. Among covered anchors, pool size ranges 1–1,713, median 89, mean 216.9. A large block raises both computation and the chance that a maximum similarity clears the threshold.

## 8. Candidate-generation recall and its ceiling
<!-- source:evaluation -->

There are two gates at which a true successor can disappear:

```text
true successor
  ↓ was it exposed by blocking?
candidate set
  ↓ did scoring rank and accept the correct row?
accepted successor
```

Blocking recall is:

```text
known successors present in their candidate sets / known successors
= 21 / 23
= 0.9130 = 91.3%.
```

No scoring model can recover the two unreachable successors, so 91.3% is a reference-sample recall ceiling. One case never entered the cohort because the award lacked structured Grand Ouest geography; one buyer changed legal form from CCAS to CIAS without a shared SIREN. These are explained operational losses, not scoring bugs.

The limitation is important: the reference reviewer saw roughly 25 surfaced candidates per anchor, but the rule that produced that review list is not recorded. Labels are independent of the later linkage methods, but candidate-surfacing independence is not established. Therefore the 21/23 ceiling and recall estimates are less independent of text ranking than precision is.

## 9. TF-IDF and cosine similarity from first principles
<!-- source:protocol -->

### Problem and idea

Raw sentences cannot be compared numerically. TF-IDF builds one coordinate per term or n-gram. A document receives a high coordinate when it uses that feature often but the feature is uncommon across the documents being compared.

For term `t` and document `d`, term frequency `TF(t,d)` counts or transforms occurrences. Document frequency `df(t)` counts documents containing `t`. With `N` documents, a basic inverse-document frequency is:

```text
IDF(t) = log(N / df(t))
TFIDF(t,d) = TF(t,d) × IDF(t).
```

The implementation uses scikit-learn's smoothed conventions, but the intuition is unchanged: ubiquitous words receive little identifying weight.

### Tiny text example

Documents are `maintenance logiciel finance`, `maintenance logiciel RH`, and `construction route`. Here `maintenance` and `logiciel` appear in 2/3 documents, while `finance`, `RH`, `construction`, and `route` appear in 1/3. The rarer business terms get larger IDF weights. The first two vectors share maintenance/software coordinates; the road document points elsewhere.

For successor linkage, the vectoriser is fitted **inside each anchor's buyer block**. Word features are 1–2 grams. Character features are within-word-boundary 3–5 grams. The pair score is the maximum of word and character cosine. Character fragments help spelling variants, abbreviations, morphology and formatting, but can also make standard framework boilerplate look deceptively similar.

For the technology classifier, the design differs: word unigrams only were selected within grouped folds; character features are not deployed. Do not conflate the two TF-IDF uses.

### Cosine mechanism and hand calculation

For vectors `x` and `y`:

```text
cos(x,y) = (x·y) / (||x|| ||y||).
```

The dot product sums shared weighted features. Each norm measures total vector length. Division removes document-length scale, so the score depends mainly on direction.

With `x=(1,1,0)` and `y=(1,0.8,0)`, dot product is `1×1 + 1×0.8 = 1.8`; norms are `√2 = 1.4142` and `√1.64 = 1.2806`. Cosine is `1.8/(1.4142×1.2806)=0.9939`. The texts have almost the same feature pattern. That is strong similarity evidence, not proof of succession: two framework notices can share language without continuing the same need.

## 10. The four active linkage methods
<!-- source:evaluation -->

All methods see the same exposed pairs so differences reflect decision rules, not easier candidate pools.

| Method | Mechanism | Acceptance | Strength | Weakness | Locked result |
|---|---|---|---|---|---|
| M_A deterministic | verified SIREN AND CPV continuity >0 AND text ≥0.25; rank by weighted score | fixed gates | auditable strong evidence | misses name-only true links; still accepts generic similar work | 15 accepted, precision .533, recall .444 |
| M_B text ranking | rank by max word/character TF-IDF cosine | top score ≥0.70 | simple, conservative, no unreliable duration | boilerplate/template risk; ignores CPV in decision | 8 accepted, precision .875, recall .389 |
| M_C weighted-gated | buyer .50 + text .25 + CPV .20 + time .05, renormalised over observed components; independent gates | weighted score ≥70 plus buyer/text/CPV rules | combines evidence, higher recall | hand-chosen weights and more false events | 23 accepted, precision .522, recall .667 |
| M_D Fellegi–Sunter | discretise buyer/text/CPV/time; estimate match and non-match level probabilities by EM; sum log likelihood ratios | posterior ≥.65, rank by posterior | data-estimated evidence weights, missing is its own level | conditional-independence and calibration problems | 5 accepted, precision .200, recall .056 |

For M_C, if observed components are set `O`, its score is:

```text
100 × Σ(k in O) w_k c_k / Σ(k in O) w_k.
```

If buyer=1, text=.8, CPV=.5 and time is missing, the score is:

```
100 × (.50×1 + .25×.8 + .20×.5) / (.50 + .25 + .20) = 84.21
```

Missing time is not treated as time disagreement.

Fellegi–Sunter estimates `m_kl=P(level l | match)` and `u_kl=P(level l | non-match)`. A level contributes `log2(m_kl/u_kl)`. Evidence common among matches but rare among non-matches is positive; equally common evidence contributes zero. The model was retained as a comparator and did not outperform M_B.

## 11. Primary top-rank rule, abstention and threshold choice
<!-- source:evaluation -->

For each anchor:

```text
j_hat_i = argmax over j in J_i of T_ij
Y_i = 1 if T_i,j_hat_i ≥ 0.70, else 0.
```

`argmax` means “choose the candidate with the largest score.” Example: B=.43, C=.68, D=.76 → D ranks first and is accepted. If B=.41 and C=.66 → C ranks first but is rejected, so the algorithm abstains.

A threshold is needed because every nonempty candidate set has a maximum even when all candidates are bad. Raising the threshold usually decreases accepted volume and recall while increasing precision; lowering it does the reverse. The project retains:

| Rule | Events / 3,800 | Event rate | Median among events |
|---|---:|---:|---:|
| M_B @ .80 | 296 | 7.79% | 35.71 months |
| M_B @ .70 | 544 | 14.32% | 31.82 months |
| M_B @ .60 | 853 | 22.45% | 26.55 months |
| M_C @ .70 | 1,332 | 35.05% | 26.09 months |

The .70 point is a frozen post-development operating convention, not mathematical truth or an optimiser output. Project history shows that the regional reference informed the retained policy, so the recorded locked stratum is internal validation rather than an untouched holdout. Replacing the policy requires fresh independent evidence.

### Why precision-first is defensible

A false positive invents an event **and a date**, producing an unjustified downward step in survival. A false negative leaves a true event censored, pushing survival upward. Both matter. Precision-first is defensible because fabricated exact event times are especially damaging, but it costs recall and observed event count. Because the biases oppose each other, the measured probability is not automatically a lower bound.

## 12. Reference sample, confusion matrix and uncertainty
<!-- source:evaluation -->

The regional reference reviewed 120 anchors on 11 August 2026 before current linkage methods existed. Of these, 112 re-resolve uniquely, 88 are usable, 16 form a pilot split (5 reviewed successors), and 72 form the locked split (18 reviewed successors). The pair tables contain 5,221 pilot and 20,917 locked rows; only 69 locked anchors have an exposed pair, while anchor metrics correctly retain all 72.

Labels came from one LLM research pass over BOAMP notices, official URLs and wider public sources, then subset spot-checking by the owner. They are method-independent reference labels, not legal ground truth or independent specialist validation.

### Exact-successor confusion accounting

- TP: accepted candidate equals the reviewed successor.
- FP: an accepted candidate is wrong. A wrong choice on a positive anchor is an FP.
- FN: a reviewed successor is not correctly recovered. A wrong choice on a positive anchor is also an FN.
- TN: a reviewed negative anchor receives no accepted link.

For M_B @ .70: TP=7, accepted=8, so FP=1; 18 positives imply FN=11; all 54 negative anchors are abstained, so TN=54 and negative-anchor FPR=0/54. The materialised JSON separately names **10 abstained positive anchors** and **1 wrong-successor positive anchor**. One narrative report prints FN=10 while also printing recall 7/18; the exact-successor FN is 11. This guide uses the arithmetic and the JSON field definitions rather than silently reconciling the discrepancy.

### Metric calculations

```text
Precision = TP/(TP+FP) = 7/8 = 0.875 = 87.5%.
Recall = TP/(TP+FN) = 7/18 = 0.3889 = 38.9%.
FPR on negative anchors = FP_negative/(FP_negative+TN) = 0/(0+54) = 0%.
Coverage = accepted/anchors = 8/72 = 11.1%.
```

Precision asks whether accepted links are usually correct. Recall asks whether reviewed successors are recovered. A cautious system can have high precision and low recall. Zero observed FPR on 54 negatives is not population-zero FPR; the negatives are corpus-relative and the sample is small.

### Wilson confidence intervals

The project uses Wilson intervals for locked precision and recall. With proportion `p_hat=x/n` and 95% normal quantile `z=1.96`:

```text
centre = (p_hat + z²/(2n)) / (1+z²/n)
margin = z√(p_hat(1-p_hat)/n + z²/(4n²)) / (1+z²/n).
```

For 7/8, the result is **0.529–0.978**; for 7/18, **0.203–0.614**. Wilson pulls the centre away from an unstable boundary and remains inside [0,1]. The wide precision interval reflects only eight accepted decisions. A 95% confidence procedure covers the fixed population parameter in 95% of repeated comparable samples; it does not mean a 95% probability that this one fixed interval contains the truth.

The separate 20-link model-assisted production diagnostic found 14 yes, 5 no and 1 uncertain: conservative precision **14/20=.700**, exact Clopper–Pearson CI **.457–.881**, below the .80 target. It is diagnostic, not independent human validation.

## 13. Turning links into right-censored survival observations
<!-- source:survival -->

### Problem, idea and method

An ordinary average of the 544 linked delays drops 3,256 episodes with no accepted event and therefore conditions on experiencing the event. Survival analysis preserves what those episodes reveal: each remained event-free for at least its observed follow-up.

For an accepted successor:

```text
tau_i = v_j_hat - u_i; Y_i=1.
```

For no accepted successor by cutoff `C=2025-12-31`:

```text
tau_i = C - u_i; Y_i=0.
```

`Y=0` means right-censored, not “never re-procured.” We know only that no accepted BOAMP successor was observed by the cutoff.

### Three-row hand example

| Episode | Award | Accepted successor/cutoff | Months | Event |
|---|---|---|---:|---:|
| A | Jan 2020 | successor Jan 2022 | 24 | 1 |
| B | Jan 2021 | no link by Dec 2025 | 59 | 0 |
| C | Jan 2024 | successor Jul 2025 | 18 | 1 |

The event-only average is `(24+18)/2=21` months and says nothing about B's 59 event-free months. Survival methods use all three.

Actual output is one row for each of **3,800** cohort episodes: **544 events** and **3,256 censored**. No duplicate episode and no negative duration exists.

## 14. Kaplan–Meier and conditional future probabilities
<!-- source:survival -->

### Problem and estimator

The question is: what proportion remains without an accepted observable successor as procurement age grows? Define `S(t)=P(T>t)`. At each event time `t_k`, let `n_k` be episodes at risk just before it and `d_k` events at that time. Kaplan–Meier multiplies conditional survival steps:

```text
S_hat(t) = product over t_k≤t of (1 - d_k/n_k).
```

### Five-episode example

Suppose times/status are `(2,event)`, `(3,censored)`, `(4,event)`, `(5,event)`, `(6,censored)` months.

- At month 2, `n=5,d=1`: survival becomes `1×(1-1/5)=0.8`.
- Month 3 censoring creates no step; it reduces the later risk set.
- At month 4, three remain at risk, `d=1`: `0.8×(1-1/3)=0.5333`.
- At month 5, two remain, `d=1`: `0.5333×(1-1/2)=0.2667`.

Censoring removes future exposure after its censor time without pretending an event occurred.

### Actual results

| Age | No-successor survival `S(t)` | Cumulative successor probability `1-S(t)` |
|---:|---:|---:|
| 12 months | .953791 | **4.621%** |
| 24 months | .932669 | **6.733%** |
| 36 months | .913163 | 8.684% |
| 48 months | .845003 | 15.500% |
| 60 months | .824647 | 17.535% |

The Kaplan–Meier median is the first time `S_hat(t)≤.5`; it is **not reached**. The reported **31.82 months** is the median delay among the 544 accepted events only. It can exist even while well over half the full cohort remains event-free/censored.

### Conditional probability

For an episode event-free at age `t`, the probability of an event within the next `h` is:

```text
P(T≤t+h | T>t) = 1 - S(t+h)/S(t).
```

Example at age 36 for the next 12 months: `1 - .845003/.913163 = .07464 = 7.464%`. This is different from the unconditional 48-month cumulative probability of 15.500% because it conditions on having reached month 36 without an event.

Actual age-36 estimates are **7.464%** in the next 12 months (95% episode-bootstrap CI **6.522–8.587%**) and **9.693%** in the next 24 (**8.579–11.117%**), based on 500 bootstrap draws. They estimate an observable successor under the primary linkage rule, not a named contract's legal renewal probability.

## 15. Cox proportional hazards: association, not probability or cause
<!-- source:survival -->

### Problem and intuition

Kaplan–Meier describes groups but cannot simultaneously compare CPV, region, framework status, identifier availability and award year. The hazard is the instantaneous event tendency at age `t` among episodes still event-free. Cox specifies:

```text
h(t|X) = h0(t) exp(X' beta); hazard ratio HR=exp(beta).
```

`h0(t)` is an unspecified baseline shape; `X` contains covariates; `beta` contains log-hazard coefficients. Exponentiation turns an additive coefficient into a multiplicative hazard ratio.

If `HR=1.55`, then at the same procurement age and conditional on model covariates, episodes still at risk have an estimated event hazard about 55% higher than the reference. This does **not** mean 55% greater renewal probability, 55% more contracts, 55% shorter duration, or a causal effect.

### Reference categories and actual estimates

CPV-32, Bretagne, non-framework, no validated SIREN and centred award year form the relevant baselines. Main results on 3,800 episodes/544 events:

| Covariate | HR | 95% CI | p-value | Reading |
|---|---:|---:|---:|---|
| Framework agreement | 1.751 | 1.435–2.136 | 3.38e-8 | higher observed-successor hazard; partly detectability |
| Validated SIREN | 1.082 | .885–1.323 | .443 | no clear association |
| Award year +1 | 1.107 | 1.066–1.149 | 1.21e-7 | time/cohort effect; PH violated |
| CPV-35 vs 32 | 1.553 | 1.218–1.981 | .000383 | most robust comparative result |
| CPV-48 vs 32 | .828 | .638–1.073 | .153 | not statistically clear |
| CPV-72 vs 32 | 1.056 | .850–1.310 | .624 | not statistically clear |
| Normandie vs Bretagne | .800 | .640–1.000 | .0501 | borderline |
| Pays de la Loire vs Bretagne | 1.003 | .815–1.234 | .979 | no difference |

A p-value tests compatibility with `HR=1` under the model; the confidence interval shows effect uncertainty. Statistical significance is not practical importance or causality.

Proportional-hazards tests flag award year, framework and validated SIREN (`p<.05`). Their reported HRs are time-averaged descriptive associations. The model's in-sample C-index is .626, but out-of-time discrimination is the more honest predictive test.

## 16. Detectability, C-index and why Cox is not a forecasting model
<!-- source:survival -->

### Candidate-pool detectability

An anchor with 500 candidates has 500 chances to generate a large maximum score; one with three has only three. Thus pool size can affect whether an event is **detectable** even with identical underlying re-procurement behaviour.

The standardised mean difference is:

```text
SMD = (mean_linked - mean_censored) / pooled_SD.
```

It measures difference in standard-deviation units, not statistical significance. For log candidate-pool size, linked mean=4.787, censored mean=3.972 and SMD=**+0.470**, the largest measured imbalance.

Adding `log(1+pool size)` to a sensitivity Cox model gives HR **1.184** (`p=6.0e-9`). CPV-35 moves only **1.553→1.512**; framework moves **1.751→1.617**, about 14% attenuation on the log-HR scale. This suggests the CPV-35 comparison is largely insensitive to detectability, while part of the framework association reflects buyers with more publication opportunities. The covariate is not causal, so this sensitivity model does not automatically replace the pre-specified primary model.

### C-index

For comparable pairs, the C-index asks whether the episode assigned higher risk experiences the event sooner. A predicted-high A with event at 20 months versus predicted-low B at 50 is concordant; reversing the predictions is discordant. `C=.5` is approximately random ranking and `C=1` perfect.

Train 2015–2021: 2,470 episodes/392 events, C=.606. Test 2022–2024: 1,004/107, **C=.479**. The sensitivity test through 2025 is 1,330/152, **C=.518**. Therefore the covariates may describe population-level associations while failing to rank individuals on unseen years. A reliable forecast would need stronger external temporal performance, calibration at useful horizons, model updating, richer predictors and independent outcome validation.

## 17. Parametric survival and robustness of the constructed event
<!-- source:survival -->

### Parametric families

Parametric models assume a distributional family for event time. The project fits exponential (constant hazard), Weibull (monotone hazard), log-logistic, log-normal and generalized gamma (flexible shapes). Maximum likelihood finds parameters that make observed events and censoring most plausible.

```text
AIC = 2k - 2 log L
BIC = k log n - 2 log L.
```

`k` is parameter count and `L` maximised likelihood. Smaller balances fit against complexity. Results: generalized gamma AIC **7520.99**, then log-normal **7573.24**, log-logistic **7618.57**, Weibull **7633.35**, exponential **7661.31**. Best AIC/BIC does not make generalized gamma literally true. KM supplies all reported within-window operational probabilities; generalized gamma is only the preferred instrument for any cautious extrapolation beyond observation.

### Event-definition sensitivity

Different linkage rules create different event sets, hence different survival data. Under the four arms, 12-month successor probabilities are 2.368%, 4.621%, 8.005% and 12.214%; 24-month probabilities are 3.235%, 6.733%, 11.470% and 17.984%. Absolute levels are therefore highly linkage-sensitive.

Two additional checks isolate different mechanisms:

- **Borderline exclusion:** remove 280 anchors whose best score is .65–.75 (133 events, 147 censored). Remaining 3,520/411 give KM 12m **3.721%**, CPV-35 HR **1.780**, framework HR **1.616**.
- **Template/shared-successor re-censoring:** re-censor 173/544 accepted links with word similarity <.50 or a successor reused by another anchor. Events fall to 371 and KM 12m **2.639%**, but CPV-35 HR **1.541** and framework HR **1.692**.

These checks support comparative directions more strongly than absolute levels. They do not prove flagged links false; they bound sensitivity to known signatures.

## 18. Calendar-time trends: OLS, PELT, stationarity and HMM
<!-- source:trend -->

### Quarterly counts and OLS

Each awarded episode contributes once to its award quarter. Episode counts avoid changes driven merely by corrections or repeated notices. The CPV series uses 43 complete quarters, 2015Q2–2025Q4; partial 2015Q1 is excluded.

For the latest 12 quarters:

```text
Y_t = alpha + beta t + error_t.
```

`beta` is episodes per quarter per quarter. Overall beta=-.108 (`p=.921`); CPV-48 beta=-.836 (`p=.0317`); other raw p-values exceed .10. This is a local recent-direction question, not full-history growth or a forecast.

### PELT

A straight line can hide level shifts. PELT minimises within-segment squared error plus `penalty × number of changes`. Small penalty over-segments; large penalty suppresses breaks. The central penalty is `log(n)` after z-standardisation, with .5× and 2× sensitivity. A break is stable if all penalties place one within a quarter. Stable candidates are CPV-32 2020Q2, CPV-48 2024Q1 and CPV-72 2021Q1. PELT locates statistical shifts, not their economic causes.

### ADF and KPSS

ADF null: unit root/non-stationarity. KPSS null: level stationarity.

| ADF | KPSS | Reading |
|---|---|---|
| reject | do not reject | supports stationarity |
| do not reject | reject | supports non-stationarity |
| reject | reject | conflicting evidence |
| do not reject | do not reject | inconclusive |

Overall rejects ADF (`p≈2.9e-7`) and does not reject KPSS at reported bound .10, supporting stationarity. CPV-32 and CPV-72 reject both, so evidence conflicts. CPV-48 fails to reject ADF (`p=.406`) and rejects KPSS (`p=.0316`), supporting non-stationarity. Short, noisy series make disagreement unsurprising.

### HMM

A three-state Gaussian hidden Markov model is fit to quarter-over-quarter count changes. Hidden state is decline/plateau/growth; the observed emission is the numerical change; transition probabilities express persistence/movement; posterior probability quantifies current-state uncertainty. Current regimes: overall growth probability .750, CPV-32 growth .992, CPV-72 growth .594.

HMM, PELT and OLS can disagree without contradiction: HMM describes the most recent latent change regime, PELT partitions historical levels, and OLS averages a 12-quarter slope.

## 19. Multiple testing and the actual trend verdict
<!-- source:trend -->

Testing five segments at 5% creates multiple opportunities for a chance small p-value. Under 20 independent nulls, probability of at least one false positive is `1-.95^20=64.2%`.

Holm controls family-wise error: sort p-values; compare the smallest with `alpha/m`, then progressively less strict thresholds. Adjusted Holm p-values can be computed as running maxima of `(m-rank+1)×p_sorted`, capped at 1. Benjamini–Hochberg controls expected false-discovery proportion: sort p-values, compare `p_(k)` with `k alpha/m`, and enforce monotone adjusted values.

For five CPV trend slopes, CPV-48 raw `p=.0317` becomes Holm and BH **.1585**. It is a nominal monitoring signal, not a multiplicity-robust finding. No CPV segment survives correction.

## 20. Why technology classification is a separate enrichment problem
<!-- source:technology -->

CPV divisions are broad administrative containers. Cybersecurity, cloud, ERP/business software, telecommunications, AI, data/BI and IT services can share CPV divisions. The classifier predicts a finer business taxonomy from object text; it does **not** create or validate successor links.

The labelled corpus contains 500 unique notices, 2015–2025, in 11 classes: eight substantive technologies plus MIXED, OTHER_DIGITAL and OTHER. Supports are Cloud 32, Cybersecurity 54, Network/Telecom 81, IT Infrastructure 46, Business Software 88, Data/BI 29, AI 7, IT Services 74, Mixed 21, Other Digital 53 and Other 15. Class counts are annotation quotas, not market prevalence. The file has no annotator identifier or second pass, so no inter-annotator agreement exists.

### Leakage and grouped folds

Random notice splitting could train on “Maintenance plateforme RH 2022” and validate on its near-identical rectification. Families are unions of shared canonical episode and object character cosine ≥.80. This produces 459 groups: 423 singleton, 36 multi-notice, largest 3; 39 near-duplicate pairs are merged, 29 across different reconstructed episodes. Every family stays in one of three stratified folds. Grouping can sacrifice independent signal by merging two true procurements, but that is safer than leakage-inflated accuracy.

## 21. Multiclass logistic regression, class weighting and macro-F1
<!-- source:technology -->

### Mechanism

Word-TF-IDF converts each object into vector `x`. Each class `k` has weights `beta_k`; score `beta_k'x` becomes a probability through softmax:

```text
P(Y=k|x) = exp(beta_k'x) / Σ_l exp(beta_l'x).
```

The largest probability supplies the predicted class. Positive feature weights raise a class score; negative weights lower it. L2-style regularisation constrains large coefficients in the high-dimensional sparse vocabulary. `C=3` controls inverse regularisation strength in the deployed model.

Class weighting is not feature scaling. Scaling changes numeric feature magnitudes; class weighting gives mistakes on rare classes more loss influence; resampling changes the training rows. With Cloud 150 and AI 7 in a fictional corpus, an unweighted optimiser can improve total loss mostly by serving Cloud. Balanced weights counter that dominance without duplicating AI rows.

For class `k`, `F1_k=2 P_k R_k/(P_k+R_k)`. Macro-F1 is the unweighted average over 11 classes. A model predicting a 90%-majority class can score 90% accuracy but nearly zero F1 on minorities; macro-F1 exposes that failure.

### Model comparison

| Model | Representation | Mean grouped-CV macro-F1 | Fold SD |
|---|---|---:|---:|
| Majority | constant | .027 | .0004 |
| CPV | administrative | .439 | .076 |
| CPV+descriptor | administrative | .469 | .038 |
| logistic | word TF-IDF | .662 | .067 |
| balanced logistic | word TF-IDF | **.741** | .034 |
| linear SVM | word TF-IDF | .717 | .030 |
| balanced SVM | word TF-IDF | .716 | .054 |

Balanced logistic is selected because it leads mean macro-F1, has the smallest train-validation gap among text models (.249), stable representation choices, good temporal behaviour and native probability output. Every fold chose word unigrams rather than the offered 1–2 gram representation. CamemBERT was not run because the pre-specified gate required classical macro-F1 below .55; observed .741 exceeded it.

CPV is deliberately excluded from text features so the benchmark asks a scientific question: can text add a business taxonomy beyond official administrative coding?

## 22. Bootstrap, calibration, confidence and technology quality gates
<!-- source:technology -->

### Family bootstrap and paired comparison

To approximate sampling variation, sample 459 procurement families with replacement, rebuild the evaluation rows, calculate macro-F1, and repeat 1,000 times. Families, not notices, are resampled because related notices are not independent.

Selected model macro-F1 is **.7442**, bootstrap SD **.0275**, 95% percentile CI **.6819–.7905**. CPV/descriptor is **.4731**, CI **.4126–.5259**. In each replicate compute `D_b=F1_text,b-F1_CPV,b`; observed difference **.2711**, CI **.2009–.3403**, excluding zero. Pairing isolates the within-sample performance difference better than comparing two independent intervals.

### Calibration decision

Accuracy and calibration differ. A model can rank correctly while its .70 score does not correspond to 70% empirical correctness. Platt/sigmoid calibration was evaluated inside grouped splits. Pre-specified adoption required ECE improvement ≥.02 and macro-F1 cost ≤.02. ECE improved by .1405, but macro-F1 fell by .0364, so calibration was rejected. The deployed score is raw, uncalibrated class probability.

At raw score ≥.70, out-of-fold accuracy is **43/45=95.56%** on the annotation distribution, versus 74.95% below. In production, **235/3,800=6.18%** episodes clear .70. The score is useful for ordering/filtering but is not automatically the probability a deployed prediction is correct because it is miscalibrated and class priors differ between quota corpus and population. The technology .70 and linkage .70 are numerically coincident but solve unrelated problems.

### Gates

```text
Predicted class
  ↓ substantive technology, labelled support ≥10, OOF F1 ≥.65?
Classifier Gate A
  ↓ survival: ≥100 episodes and ≥20 events?
Statistical Gate B
  ↓
Include in technology survival / exclude with reason
```

Five classes pass both: Cybersecurity (316 episodes/60 events), Network/Telecom (859/140), IT Infrastructure (298/38), Business Software (854/106), IT Services (492/72). Cloud and Data/BI fail downstream sample/event support; AI fails labelled support; residual classes are not substantive technologies.

## 23. Technology survival and trend results
<!-- source:technology -->

The predicted episode class becomes a constructed covariate. Technology-specific KM curves compare time to the already-defined observable successor. The log-rank null is that all included survival functions are equal over time.

Among five gated classes (2,819 episodes, 416 events), log-rank statistic **10.258**, `p=.0363`. At 24 months, Business Software has 8.57% estimated successor probability and IT Infrastructure 6.11%. This omnibus result says not all five curves are alike; it does not identify a causal technology effect or a specific pair, and it is unadjusted for buyer/procedure/size. Including statistically large but substantively residual classes would produce `p=.000119`; the gate prevents that heterogeneous fallback contrast from masquerading as a stronger technology finding.

Technology trends use all predictions because the ≥.70 subset leaves mostly empty quarterly series. They use the same 43-quarter observation window as the CPV series (`2015Q2`–`2025Q4`), with the reported OLS slopes fitted over the latest 12 quarters. Five gated classes are tested: the smallest raw p is IT Services **.1944** (slope **+.4476** episodes per quarter; Holm **.9720**), so no class is even nominally significant. This is a negative result: there is no corrected evidence of a recent linear technology-count trend. Classification uncertainty is not propagated into trend or survival intervals, so both branches inherit additional measurement error.

## 24. What could threaten the analysis?
<!-- source:quality -->

| Threat | Evidence / calculation | Why it matters | Decision | Residual limitation |
|---|---|---|---|---|
| Missing duration | 2,847/3,800=74.9%; 2023 11.8% vs 2025 84.4% complete | expiry/timing rule would be schema-driven | no global imputation; removed duration-conditioned arm | cannot study true contract duration broadly |
| Missing validated identity | 2,521/3,800=66.3% | name-only buyer errors | keep names but veto conflicting validated SIRENs | legal-form changes can be missed |
| Post-award information / unequal follow-up | recent cohorts have short follow-up; 2025 Q4 cannot show ≥90-day event | censoring and award year intertwine | administrative censoring; 2025 excluded from primary temporal test | independent censoring may be imperfect |
| Text-length regime | linked mean 1,611 chars vs censored 1,087; SMD .262 | more text improves matchability | report detectability diagnostics | content and measurement cannot be separated fully |
| Multi-lot episodes | 1,176/3,800 enter on digital lot; 412 multi-digital | broad episode mixes domains | any-code rule + explicit segment tie-break | segment can simplify mixed demand |
| Unresolved award date | 26/3,826 awarded digital episodes dropped | no defensible time zero | drop rather than guess | selection toward better records |
| Candidate-pool imbalance | log pool SMD .470 | maximum-over-more-draws bias | pool-adjusted Cox sensitivity | residual buyer publication behaviour |
| Shared successors | 127 links share successor; 173 total template-risk flags | no one-to-one constraint may reuse generic notice | re-censor sensitivity | flagged ≠ false; exact rate unknown |
| Boilerplate similarity | 65 links have word score <.50 but character score carries acceptance | framework text can look alike | template-risk re-censoring | semantic false positives remain possible |
| Rare technology classes | AI n=7; no interpretable performance | unstable per-class metrics | fail classifier gate | AI conclusions unavailable |
| Small linkage validation denominator | precision 7/8, CI .529–.978 | point estimate unstable | report Wilson interval and production diagnostic | independent specialist accuracy absent |
| Annotation inconsistency | 3 near-duplicate families have conflicting labels | ceiling on classification | leave labels unchanged and document | no inter-annotator reliability |
| Amount ambiguity | 15.7% no container; multiple unvalidated candidates | false monetary trends | omit monetary analysis | spend/value question unanswered |

An upstream error propagates:

```text
wrong episode → wrong anchor → wrong block → wrong link → wrong event time → distorted KM/Cox
blocking miss → model cannot select → censored row → event level pushed down
false accepted link → invented event/date → event level pushed up
```

Because the last two mechanisms point in opposite directions, observed successor probabilities are not guaranteed lower bounds.

## 25. Metric and calculation dictionary
<!-- source:protocol -->

| Quantity | Type | Formula/algorithm | Actual value and denominator | Interpretation / limitation |
|---|---|---|---|---|
| Standardised notices | observed→standardised | unique official ID | 1,620,712/1,620,712 unique | publications, not contracts |
| Episodes | constructed | graph connected components | 1,103,632 from 1,620,712 notices | inferred processes |
| Cohort | constructed | region+award+any digital CPV+date | 3,800; 26 unresolved-date drops | selected study population |
| Digital-lot/main-outside share | constructed | 1,176/3,800 | 30.95% | why “any code” matters |
| Candidate coverage | constructed | 3,520/3,800 | 92.63% | any candidate, not correct candidate |
| Candidate pairs | constructed | anchor-candidate rows | 763,417; median 89 per covered anchor | exposed comparisons |
| Blocking ceiling | estimated on reference | 21/23 | 91.30% | not fully surfacing-independent |
| Primary events | constructed | M_B top score ≥.70 | 544/3,800 | accepted observable successors |
| Event rate | descriptive | 544/3,800 | 14.3158% | linkage-conditioned |
| Observed-event median | descriptive | median of 544 event delays | 31.82 months | not KM median |
| Precision | estimated on reference | 7/8 | 87.5%, CI .529–.978 | accepted-link correctness in locked sample |
| Recall | estimated on reference | 7/18 | 38.89%, CI .203–.614 | recovered reviewed successors |
| Negative-anchor FPR | estimated on reference | 0/54 | 0%; small corpus-relative denominator | not population zero |
| KM 12m | estimated | 1-product risk steps | 4.621%, 3,800/544 | event probability under primary linkage |
| KM 24m | estimated | same | 6.733% | linkage-sensitive |
| KM median | estimated | first S≤.5 | not reached | differs from event-only median |
| CPV-35 HR | estimated/associational | exp Cox coefficient | 1.553, CI 1.218–1.981 | vs CPV-32, non-causal |
| Framework HR | estimated/associational | exp Cox coefficient | 1.751, CI 1.435–2.136 | partly detectability |
| Pool SMD | descriptive diagnostic | mean difference/pooled SD | +.470 log pool | detectability imbalance, not p-value |
| Pool-adjusted HRs | robustness | Cox + log pool | CPV-35 1.512; framework 1.617 | sensitivity, not new primary |
| Out-of-time C-index | predictive | concordant comparable pairs | .479 on 1,004/107 | no useful individual ranking |
| Parametric winner | estimated | minimum AIC/BIC | generalized gamma AIC 7520.99 | preferred fit, not truth |
| Trend CPV-48 | descriptive/inferential | 12-quarter OLS | slope -.836; raw p .0317; Holm .1585 | nominal only |
| Labelled notices/families | observed/constructed | annotations/grouping | 500/459 | quota corpus |
| Technology macro-F1 | predictive estimate | mean class F1 | .7442; 1,000-family bootstrap CI .6819–.7905 | corpus performance |
| CPV benchmark | predictive estimate | same folds/metric | .4731; CI .4126–.5259 | administrative baseline |
| Paired gain | predictive comparison | F1_text−F1_CPV | .2711; CI .2009–.3403 | text adds signal |
| High-confidence predictions | predicted | raw score ≥.70 | 235/3,800=6.18% | operational subset, not observed truth |
| Technology survival | associational enrichment | gated log-rank | 5 classes; 2,819/416; p .0363 | predicted groups, unadjusted |
| Technology trend | robustness/negative | 5 slopes + Holm/BH | none survives; min Holm .2815 | no supported linear trend |

Complex numbers are not reducible to one fraction. KM iterates risk sets; Cox maximises partial likelihood; C-index enumerates comparable pairs; bootstrap resamples families; AIC/BIC use maximised likelihood; PELT optimises penalised partitions; ADF/KPSS fit time-series regressions with opposite nulls; HMM estimates transition/emission parameters and posteriors; Holm/BH reorder a family of p-values.

## 26. Why each method was chosen, and what happens if it is removed
<!-- source:protocol -->

| Problem | Alternative | Chosen solution | Why / protection | Trade-off; without it |
|---|---|---|---|---|
| many notices per process | notice-level analysis | episode reconstruction | avoids administrative duplicates | inferred merges; without it false short events |
| huge pair space | exhaustive comparison | buyer/time blocking | computation and false-opportunity control | ceiling <1; without it noisy billions |
| no-match anchors | always choose top | threshold + abstention | avoids forced false events | lower recall; without it every covered anchor “renews” |
| text representation | raw string rules / embeddings | TF-IDF word+character for linkage | transparent, local vocabulary, variants | boilerplate risk; embeddings were not active/evaluated |
| event-time censoring | event-only mean | Kaplan–Meier | uses incomplete follow-up | independent-censoring assumption |
| adjusted association | separate group means | Cox | simultaneous covariates, unspecified baseline | PH assumptions; not prediction/causality |
| tail model | KM extrapolation | generalized-gamma comparison | explicit tail family | extrapolation uncertainty |
| related annotation rows | random CV | family-grouped CV | blocks near-copy leakage | fewer independent units |
| class imbalance | accuracy/unweighted loss | balanced logistic + macro-F1 | minority classes matter equally | probability calibration distorted |
| small-sample uncertainty | normal approximation | family bootstrap; Wilson for linkage | respects clustering/bounds | still depends on sample representativeness |
| many trend tests | raw p-values | Holm and BH | false-positive control | less power |
| weak predicted groups | analyse all labels | classifier/statistical gates | prevents residual buckets driving conclusions | excludes potentially interesting small classes |

Some rationales are explicit project decisions: frozen precision-first threshold, any-code cohort, group-aware CV, no duration imputation and gates. Standard explanations of why KM, Cox, bootstrap, Holm and BH behave as described are statistical theory used to teach the documented choice, not invented project history.

## 27. Assumptions and consequences if they fail
<!-- source:protocol -->

| Assumption | Where | Need | Failure consequence | Tested? |
|---|---|---|---|---|
| episode edges join one process | reconstruction | define analysis unit | duplicated/merged needs | structural tests; semantic spot-check incomplete |
| buyer/time block contains true links | candidates | feasible search | unrecoverable false negatives | 21/23 reference reachability |
| high text similarity tracks same need | M_B | select successor | boilerplate false links | reference + template-risk sensitivity |
| administrative censoring is non-informative conditional on design | KM/Cox | use incomplete follow-up | biased curves if linkability/follow-up differs | diagnostics, not fully resolved |
| proportional hazards | Cox | constant HR interpretation | time-averaged/misleading coefficient | violations found for 3 covariates |
| chosen covariates address relevant confounding for association | Cox | adjusted comparisons | residual confounding | no causal design |
| parametric family approximates tail | extrapolation | beyond-window estimates | wrong tail probabilities | AIC/BIC/KM fit comparison only |
| quarterly counts are comparable | trends | interpret change | schema/publication shifts mimic demand | duration break documented; no causal attribution |
| OLS errors and form support slope inference | trend | p-values | unreliable significance | descriptive only |
| PELT cost/penalty reflect meaningful level shifts | change points | parsimonious breaks | over/under-detection | three penalties |
| HMM Gaussian state model is adequate | regimes | latent-state summary | unstable/mislabelled states | posterior reported; limited series |
| annotations reflect frozen taxonomy | classifier | supervised target | model learns inconsistent boundaries | three conflicting families documented |
| families are independent resampling units | CV/bootstrap | uncertainty/leakage | intervals too narrow if dependency remains | grouping audit |
| future text resembles labelled years | deployment | generalisation | temporal drift | 2023–25 holdout, limited rare-class support |

## 28. Which conclusions are strongest and weakest?
<!-- source:protocol -->

### Stronger / relatively robust

- The software/data pipeline is reproducible and structurally coherent: canonical validation passed; all notices map once; counts reconcile.
- BOAMP notices must be converted to episodes before successor analysis; the 31.9% publication-to-episode reduction demonstrates the grain problem.
- Text contains technology information beyond CPV: paired macro-F1 gain .271 with CI excluding zero on identical grouped folds.
- CPV-35 has a higher observable-successor hazard than CPV-32: HR direction survives threshold arms, borderline removal, template-risk re-censoring and pool adjustment.

### Moderate / descriptive

- Under M_B @ .70, 544/3,800 episodes have accepted observable successors and KM 12m is 4.621%. Exact levels are well computed for that event rule but not stable across reasonable linkage rules.
- Five gated technology curves differ omnibus (`p=.0363`), but groups are predicted and the test is unadjusted.
- PELT and HMM provide monitoring signals, not causes or forecasts.

### Fragile / sensitivity-dependent

- Framework HR magnitude: direction is stable, but detectability attenuates 1.751→1.617 and PH is violated.
- Absolute successor probability and event rate: 7.79%–35.05% event rate across retained arms.
- Locked precision .875: only eight accepted links and CI .529–.978; separate diagnostic gives .700 conservatively.
- CPV-48 recent decline: raw p .032 but adjusted .159.

### Not established

- Confirmed legal renewals, population linkage accuracy, causal effects, individual contract forecasts, market shares from predicted classes, AI classifier performance, awarded-amount trends, and causes of change points.

## 29. The entire project as one connected story
<!-- source:protocol -->

We begin with BOAMP because it records official procurement publications. But BOAMP does not identify renewals and one procurement can appear through several notices. We first standardise schema-specific fields while keeping raw lineage, then reconstruct notices into episodes so administrative repeats do not become new demand.

We restrict episodes to awarded Grand Ouest procurement containing at least one qualifying digital CPV code and a valid award date. Each selected award becomes an anchor. We search forward 90–2,920 days among plausible same-buyer episodes, vetoing conflicting validated SIRENs. This blocking step creates comparisons but can permanently lose true successors.

For every anchor block, word and character TF-IDF turn descriptions into vectors and cosine similarity measures their alignment. Four linkage methods are evaluated on the same candidates. The frozen primary rule chooses the most text-similar candidate only if its score reaches .70; otherwise it abstains. The output is an observable-successor proxy, never a legal-renewal label.

Accepted links provide event dates. Abstentions provide follow-up until 31 December 2025 and are right-censored. Kaplan–Meier estimates linkage-conditioned survival; Cox describes adjusted associations; parametric families characterise possible tails. Threshold, borderline, template-risk and candidate-pool checks show that comparative directions—especially CPV-35—are more stable than absolute probability levels. Out-of-time C-index shows the Cox model should not be used for individual prediction.

Calendar-quarter aggregation answers a separate question about activity through history. OLS summarises recent direction, PELT flags candidate structural breaks, ADF/KPSS assess stationarity evidence, and an HMM describes current latent change regimes. Multiplicity correction turns CPV-48 into a nominal monitoring signal rather than a finding.

Finally, a labelled text corpus supplies a finer business-technology taxonomy than CPV. Grouped folds prevent related-notice leakage; balanced multinomial logistic regression wins a pre-specified comparison; family bootstrap quantifies uncertainty; calibration is evaluated and rejected by rule. Predictions enrich all episodes, but quality and sample-size gates restrict downstream survival/trend analysis. The final story is scientifically useful because it owns every transformation and limitation: it measures visible re-procurement patterns under a documented observation process, not legal renewal, causality or individual destiny.

## 30. Four levels of understanding
<!-- source:protocol -->

### Level 0 — one sentence

I built and validated a BOAMP pipeline that reconstructs procurement episodes, links plausible later successor procurements, analyses their censored timing, and enriches them with predicted technology classes—while keeping the results explicitly distinct from legal renewal and causal prediction.

### Level 1 — 60 seconds

BOAMP publishes notices, not clean contracts or renewals. I standardised 1.62 million notices and reconstructed 1.10 million procurement episodes, then selected 3,800 awarded Grand Ouest episodes containing a digital CPV. For each award I generated later same-buyer candidates and used a frozen, conservative TF-IDF text-ranking rule to accept at most one observable successor. That produced 544 events; the rest were right-censored, so I used Kaplan–Meier and Cox rather than averaging only linked cases. The main comparative finding is a higher observed-successor hazard for CPV-35, but absolute probabilities depend strongly on the linkage rule and the Cox model has weak out-of-time ranking. Separately, a grouped-CV technology classifier substantially outperformed CPV as a business taxonomy. The project is descriptive and associational, not legal-renewal proof or causal forecasting.

### Level 2 — five minutes

Explain the chain in Section 2, the 3,800 cohort funnel, the 90–2,920-day block, local word/character TF-IDF cosine, top-1 .70 abstention, reference precision 7/8 and recall 7/18 with wide Wilson intervals, 544/3,800 events, right censoring, KM 12m 4.621% and median not reached, CPV-35 HR 1.553 with pool-adjusted 1.512, out-of-time C=.479, four event-definition arms, and the separate 500-notice/459-family technology classifier with macro-F1 .744 vs .473 CPV baseline.

### Level 3 — technical defense

Use Sections 3–28: define each analytical grain and symbol; derive TF-IDF/cosine, Wilson, KM, conditional probability, Cox HR, SMD, C-index, AIC/BIC, OLS, PELT objective, ADF/KPSS nulls, HMM state posterior, softmax, macro-F1, bootstrap and multiplicity corrections; then state the exact denominators, assumptions, PH violations, reference provenance, calibration decision, quality gates, contradiction note and robustness results.

## 31. Oral-defense questions and evidence-backed answers
<!-- source:protocol -->

| Challenge | Simple answer | Technical answer | Numerical evidence |
|---|---|---|---|
| Why not notice level? | One procurement publishes several notices. | Notice-level linkage creates self/duplicate and short-gap events; connected components define episodes. | 1,620,712 notices → 1,103,632 episodes. |
| Why episodes? | They approximate one procurement process. | Union-find over folder, explicit links and constrained same-reference edges, with SIREN conflict veto. | 517,080 fewer units; zero buyer-conflict episodes. |
| Why these CPVs? | They define a broad auditable digital scope. | Episode enters if any code division is 32/35/48/72. | 7,376 digital-scope Grand Ouest episodes before award filter. |
| Why any-code inclusion? | Digital work can be one lot of a broader tender. | Main-only would exclude mixed episodes carrying a qualifying lot. | 1,176/3,800 (30.9%) have non-digital main CPV. |
| Why 90 days? | Shorter gaps looked like concurrent work. | Lower floor removes parallel lots; shortest reviewed successor is 139 days. | Earlier pathology: 132/628 links under 3 months. |
| Why eight years? | It bounds the search, not contract duration. | 2,920-day operational horizon excludes no reviewed successor. | reviewed gaps up to 2,644 days. |
| Why text similarity? | Similar needs often reuse discriminative wording. | Local word 1–2 gram and char 3–5 gram TF-IDF cosine rank same-buyer candidates. | M_B locked precision 7/8. |
| Why not embeddings? | They were not an active evaluated method. | The project selected transparent sparse features; no canonical comparison supports an embeddings claim. | Four active linkage methods only. |
| Why .70? | Conservative frozen abstention point. | Set before reference inspection to prioritise false-event control; .60 remains sensitivity. | 296/.544/.853 events at .80/.70/.60. |
| Why precision-first? | False links invent event dates. | FP moves KM at a false time; FN censors a true event. | M_C recall .667 but precision .522 vs M_B .389/.875. |
| What does 87.5% precision mean? | Seven of eight accepted locked links matched review. | Reference-sample exact-successor precision with Wilson uncertainty. | .875, CI .529–.978, n=8. |
| Why is recall low? | The rule abstains deliberately and blocking misses some. | TP/18=.389; maximum reference recall 21/23=.913 before scoring. | 7 recovered; 10 abstentions + 1 wrong choice. |
| What is censoring? | Follow-up ends before an event is observed. | Y=0 contributes risk time to cutoff, not a no-renewal label. | 3,256/3,800 censored. |
| Why KM? | It keeps censored episodes. | Product-limit estimator updates at events and removes censoring only from future risk sets. | KM 12m 4.621%; median not reached. |
| Why Cox? | To compare characteristics jointly. | Semiparametric proportional-hazards model estimates adjusted HRs without specifying baseline hazard. | CPV-35 HR 1.553. |
| Why isn't HR a probability? | It is a rate ratio among those still at risk. | Probability depends on baseline hazard and time integration as well as HR. | CPV-35 1.553; KM probabilities are separate. |
| Significant HR but weak C-index? | Group association need not rank individuals. | Coefficients can shift average hazard while overlapping individual risk scores. | CPV-35 p=.00038; test C=.479. |
| Why not individual prediction? | It fails out-of-time ranking. | Prediction requires discrimination and calibration on unseen cohorts. | 2022–24 C=.479; 2022–25 C=.518. |
| Why bootstrap? | Sampling a different labelled corpus could change F1. | Resample whole procurement families to preserve within-family dependence. | 1,000 draws; .744 CI .682–.791. |
| Why grouped CV? | Near-duplicate notices would leak. | All episode/≥.80-character-similarity family members stay in one fold. | 459 families; 29 near-duplicate pairs across episodes. |
| Why macro-F1? | Rare classes count equally. | Average 11 class-specific harmonic means, not observation-weighted accuracy. | accuracy .766; macro-F1 .744. |
| Why not trust classifier probability? | Raw scores are miscalibrated and priors differ. | Calibration improved ECE but violated the macro-F1 loss budget; quota priors do not equal deployment priors. | ECE gain .1405; macro-F1 change −.0364. |
| Why adjust p-values? | Several chances can create a lucky signal. | Holm controls family-wise error; BH controls FDR. | CPV-48 raw .0317 → .1585. |
| Why can PELT/HMM/OLS disagree? | They answer different time questions. | historical level partitions vs current latent change vs 12-quarter average slope. | overall OLS p=.921 while HMM growth=.750. |
| Biggest weakness? | The successor event lacks independent specialist validation. | Reference is method-independent but single-pass LLM/subset-checked, with unknown review-candidate surfacing. | precision n=8; diagnostic 14/20. |
| Strongest result? | Text adds technology signal; CPV-35 comparison is robust. | paired grouped-family bootstrap and multi-arm survival sensitivity respectively. | gain .271 CI excludes 0; CPV-35 1.553→1.512. |
| Most cautious result? | Absolute successor probability and CPV-48 trend. | event rule changes levels; multiplicity removes trend significance. | event rate 7.79–35.05%; .0317→.1585. |

## 32. Recommended next steps and further questions
<!-- source:protocol -->

1. Obtain blinded procurement-specialist review of the frozen accepted-link sample, keeping the audit key hidden until labels are final; do not tune and evaluate on the same rows.
2. Run a compact semantic episode-reconstruction review, prioritising long-span, high-notice-count and reference-conflict episodes.
3. If recalibrating linkage after human review, create a fresh holdout and compare one-to-one/shared-successor controls plus template-resistant text features.
4. Add labelled procurement families, especially AI, Cloud/Data and documented taxonomy boundaries; repeat the same grouped temporal evaluation.
5. Before monetary analysis, define and validate one canonical awarded-value estimand across schemas and lots.
6. Monitor whether CPV-48's nominal decline persists and seek documentary evidence before proposing a cause.

Open questions that could change interpretation are: how much independent review lowers or raises production precision; whether missing successors concentrate among legal-form changes or purchasing-centralisation; whether PH violations reflect genuine time-varying associations; and how classification uncertainty should be propagated into technology survival curves.
