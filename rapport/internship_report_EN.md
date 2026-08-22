# Cover page

> **Archived drafting source.** The authoritative English report is the LaTeX
> project under `rapport/BOAMP_Report_EN_Overleaf/`; use
> `rapport/BOAMP_Report_EN_Final.pdf` for delivery. This Markdown draft is
> retained for provenance and is not kept numerically synchronized.

> **To be typeset using the mandatory ENSAE template (annex 1 of the instructions).**
> Top left: **{SURNAME First name}** — Top right: **ENSAE 2nd year**, *Stage d'application*, *Academic year 2025-2026*
> Boxed title:
>
> ### Analysis and modelling of digital public procurement using BOAMP data
> *Constructing an observable successor, survival analysis, and technology segmentation from text*
>
> Bottom left: **Gigalis** — {City}
> Bottom right: **Supervisor: {First name SURNAME}** — {start date} to {end date}

---

# Contents

1. Introduction
2. Context and reformulation of the problem
3. Data, units of analysis, and scope
4. Constructing the outcome: successor linkage
5. Survival analysis
6. Text signal: a supervised technology taxonomy
7. Downstream use of the predictions
8. Trends and change-point detection
9. Discussion
10. Limitations
11. Conclusion

Bibliography
Annexes A to G
Note de synthèse (French)
Executive summary (English)

---

# 1. Introduction

Gigalis is the public-interest group responsible for digital services in the Pays de la Loire region. Among its roles it acts as a central purchasing body: it builds pooled framework agreements in cloud, cybersecurity, networks and artificial intelligence, which its members may use but are never obliged to. The strategic value of those agreements therefore depends on Gigalis anticipating its members' purchasing needs rather than observing them after the fact. The question put to this internship was: using historical data on digital public procurement, can we estimate the probability that a contract or a technology segment generates an identifiable purchasing need within the next twelve to twenty-four months?

The work was carried out entirely on open data — the notices published in the *Bulletin officiel des annonces des marchés publics* (BOAMP), 1,620,712 of them between 2015 and 2025, reduced to a study cohort of 3,800 awarded digital procurements in the Grand Ouest. The means were those of a standard workstation: Python, scikit-learn, lifelines, statsmodels and ruptures, with an end-to-end reproducible pipeline.

It became clear early on that the main difficulty was not estimating a duration but defining what is being measured. The BOAMP carries no field indicating that one contract renews another. The outcome variable therefore had to be constructed, and that construction governs everything downstream. This report presents the construction, the survival analysis it enables, the technology segmentation learnt from the text of the notices, the descriptive trend analysis, and a critical reading of what these results support — and of what they do not.

---

# 2. Context and reformulation of the problem

## 2.1 The business problem

A central purchasing body pools demand. Its value is twofold: lower prices through volume, and one procedure instead of many. Both depend on opening the right framework agreement at the right time. Open one too early and it ties up a contract nobody uses; open it too late and members have already run their own procurements. The operational question is therefore a question about timing: when will a need that has already been met reappear?

The internship brief decomposed that question into three sub-problems, each attached to a family of methods: the lifetime problem (survival analysis), the trend problem (change-point detection in time series), and the text-signal problem (natural language processing). A fourth, causal dimension was mentioned as a research perspective if time allowed.

## 2.2 What the BOAMP contains, and what it does not

The BOAMP publishes notices: calls for competition, corrections, award notices. It does not publish contracts, and above all it publishes **no renewal identifier**. Nothing indicates that a contract awarded in 2023 replaces one awarded in 2019. There is therefore no ground truth for the event the brief called "renewal".

This is not a data-quality detail: it makes it impossible to pose the problem in its apparently natural form, "estimate P(renewal within 12 months)", because no renewal observation exists in the data from which to estimate anything. Two routes remain. The first is to *assume* renewal — for instance by declaring that a four-year framework agreement is renewed after four years. It was rejected: it would fabricate expiry dates for three quarters of the cohort, whose declared duration is missing (§ 3.4), and it would measure the assumption rather than the data. The second is to construct an observable and to own the gap between that observable and the legal concept. That is the route taken.

## 2.3 The estimand: the observable successor

> **Definition.** An *observable successor* of an awarded digital procurement is a later procurement episode, published in the BOAMP by the same buyer, that plausibly continues or replaces the same need, and that is **accepted** as such by a frozen, documented linkage rule.

Three consequences follow, and the report carries them throughout.

First, the event is *conditional on a rule*. Changing the rule changes the number of events: from 296 to 1,332 across the retained arms (§ 5.7). No absolute level can be quoted on its own.

Second, the event is *observable*, not legal. A contract can be renewed without that being visible in the BOAMP — through a central purchasing body, below publication thresholds, or through a successor the rule failed to recover. The absence of an accepted successor is **censoring**, never evidence of abandonment.

Third, linkage errors do not cancel out. A missed link removes an event and artificially extends a censored exposure; a false link fabricates both an event and its date. The two biases run in opposite directions, which forbids presenting the observed rate as a lower bound on the true re-procurement rate — a point the record-linkage literature is explicit about (Doidge and Harron, 2019).

## 2.4 The questions as actually addressed

**Q1.** Can an observable-successor variable of measurable quality be constructed from the BOAMP alone? (§ 4)
**Q2.** How long does it take for an awarded digital procurement to show an observable successor, and does that delay differ by segment and contract type? (§ 5)
**Q3.** Does the text of the notices carry a business technology segmentation that the administrative CPV vocabulary does not? (§ 6)
**Q4.** Does that learnt segmentation support readings the CPV segmentation cannot support? (§ 7)
**Q5.** Do quarterly volumes by segment show usable trends or breaks? (§ 8)
**Q6.** Does the model support individual, contract-level prediction? (§ 5.5 — and the answer is no.)

---

# 3. Data, units of analysis, and scope

## 3.1 Source and acquisition

The source is the official BOAMP API, operated by DILA and published on data.gouv.fr. Every notice published between 1 January 2015 and 31 December 2025 was downloaded in JSONL form and standardised by schema-aware parsers: raw values and parser lineage are preserved alongside normalised values, so that any cleaning decision remains traceable and reversible.

The extraction yields **1,620,712 unique notices**, with no duplicate identifier.

## 3.2 From notice to episode: the first change of grain

A notice is not a contract. The BOAMP republishes the same procedure several times — call for competition, correction, award, sometimes lot by lot. Analysing at notice level would count the same procedure repeatedly and mechanically inflate any event rate.

Notices are therefore grouped into **procurement episodes** by connected-component search (union-find) over three edge types, in decreasing order of reliability:

1. shared `contractFolderID` (45,885 edges accepted, 737 rejected for validated-buyer conflict);
2. explicit declared link between notices (584,896 edges);
3. identical procedure reference for the same buyer within a 730-day window (195,043 accepted, 861 rejected for exceeding the window).

The result is **1,103,632 episodes**, 60.1 % of them singletons. Reconstruction is an *inference*, not an identifier supplied by the BOAMP; it is the first uncertainty the pipeline introduces, and it is measured: zero buyer-conflict episodes, zero impossible chronologies, every notice assigned to exactly one episode. 3,274 episodes carry more than one distinct procedure reference — expected for multi-lot or republished procurements — and are exported for inspection rather than silently corrected.

## 3.3 The study cohort

The selection funnel reconciles completely:

| Step | Episodes |
|---|---:|
| Grand Ouest episodes | 144,269 |
| with at least one CPV code in divisions 32, 35, 48, 72 | 7,376 |
| carrying an award notice | 3,826 |
| with a resolvable award date | **3,800** |

The 26 episodes lost at the last step have no usable award date and are dropped rather than dated by default.

**One precision that matters.** The digital filter is an **any-code rule at episode level**: formally, episode $e$ qualifies if $\exists\, c \in \mathrm{CPV}(e)$ with $\lfloor c/10^{6}\rfloor \in \{32,35,48,72\}$. It is not a rule about the main CPV. As a result, **1,176 of the 3,800 episodes (30.9 %) have a main CPV outside those divisions**: multi-lot tenders that enter on a single digital lot. The cohort is therefore exactly "awarded Grand Ouest procurement episodes containing at least one digital lot", not "3,800 digital procurements". The first phrasing is used throughout.

The stratifying variable `digital_segment` is the lowest-numbered digital division present, so each episode feeds exactly one curve. That tie-break is arbitrary and binds on the 412 episodes (10.8 %) carrying more than one digital division. Its effect was measured rather than assumed: among the 2,624 episodes whose main CPV is itself digital, the assigned segment agrees with the main division 94.7 % of the time, and event rates by assigned segment track those by main-CPV division closely (CPV-35: 0.2039 vs 0.1863; CPV-32: 0.1319 vs 0.1241). The rule is documented and kept.

Cohort composition: 1,452 episodes in Pays de la Loire, 1,282 in Normandie, 1,066 in Bretagne; 1,294 in CPV-72, 1,152 in CPV-32, 790 in CPV-48, 564 in CPV-35; median follow-up 2,015 days.

## 3.4 Data quality and missing values

| Field | Missing | Decision | Reason |
|---|---:|---|---|
| Validated SIREN | 66.3 % | Keep a name-only buyer key; audit risky links | A SIREN identifies a legal unit; a similar name does not prove legal identity |
| Reliable duration | 74.9 % | No imputation | Completeness moves from 11.8 % in 2023 to 84.4 % in 2025: missingness is not exchangeable over time |
| Amount container | 15.7 % | Excluded from any monetary claim | The container holds several notice-level amount candidates with no validated canonical awarded value |
| Main CPV, text, award date | 0.0 % | Required by selection | — |

Two of these decisions close routes the brief had envisaged and must be defended explicitly.

**Not imputing duration.** The brief proposed using declared duration to bound the renewal window ("a four-year contract is renewed within ±6 months of its end"). That route was tested and then abandoned on evidence rather than convenience: among contracts holding both a reliable duration and an accepted successor, **only 21.8 % are re-procured within six months of the declared end**, with a median absolute discrepancy of **21.1 months**. Declared duration is a poor predictor of the actual timetable. EU law itself treats four years as a general limit subject to justified exceptions (Directive 2014/24/EU, Art. 33), not as the duration of every contract.

It is worth being precise about what "excluding duration" means, because the word misleads. Declared duration is
used **neither as a hard filter nor in the primary rule**: `M_B` reads text alone. It does enter as *graded
evidence* in the time component of `M_C`'s weighted score, the contrast arm: where a reliable duration exists, the
component equals 1 at the declared end date and decays linearly to zero one year later; where it does not, the
component is `NaN` and drops out of both numerator **and** denominator rather than being scored zero. That is the
difference between "I have no information" and "the information is unfavourable". The project's initial design
substituted a four-year duration whenever the end date was unknown; that substitution was explicitly removed,
because it fabricated the single most influential timing input for most of the cohort.

**Not aggregating amounts.** No monetary analysis is produced. That is a real loss — the brief expected amount series — but a value reconstructed from an unvalidated container would produce series nobody could interpret.

## 3.5 Scope decisions

Two departures from the brief must be made explicit, because neither is neutral.

**Pays de la Loire → Grand Ouest.** The brief gave priority to Pays de la Loire, with extension to the Grand Ouest if volume proved insufficient. Volume alone did not force the extension: Pays de la Loire holds 1,452 episodes and 219 events, above the brief's own 800-contract minimum. What forces it is the **granularity** the research questions require:

| Scope | Episodes | Events | P(successor ≤ 12 m) | 95 % CI | Width |
|---|---:|---:|---:|---|---:|
| Grand Ouest | 3,800 | 544 | 4.62 % | [3.94; 5.30] | 1.37 pt |
| Pays de la Loire only | 1,452 | 219 | 5.36 % | [4.17; 6.55] | 2.38 pt |

The twelve-month interval is 1.7 times wider on Pays de la Loire alone; CPV-35, which carries the study's most robust comparative finding, falls to 34 events there against 115; and the Cox model, with eight covariates, would drop from 68 to 27 events per parameter, below the usual prudence threshold. The extension is not neutral for all that: region is a model covariate, and while Pays de la Loire is indistinguishable from Bretagne (HR 1.003, p = 0.81), Normandie shows a lower hazard (HR 0.800, p = 0.050). Aggregate results therefore read as "Grand Ouest"; a Pays-de-la-Loire reading goes through the regional row of the model, not through the average.

**2015-2024 → 2015-2025.** The brief planned ten years to 2024. 2025 was kept, for two reasons and under one caveat. It is complete in volume: 93, 73, 73 and 87 episodes across the four quarters, against 72, 57, 72 and 85 in 2024; there is no publication gap before the 31 December 2025 cut-off. It adds 326 episodes of censored exposure that improve estimation at short horizons. But it is **incomplete in follow-up**: median follow-up for a 2025 award is 5.9 months against 59.9 months for 2015-2024, and the fourth quarter of 2025 can contain no event by construction. The effect is measurable: removing the 2025 awards moves the twelve-month estimate from 4.62 % to 3.54 %. 2025 is therefore kept, flagged as partial in follow-up, and **excluded from the primary temporal-validation window** (§ 5.5), where it appears only as a sensitivity read.

## 3.6 Summary of grains

| Layer | One row represents | Count |
|---|---|---:|
| Standardised notices | one official BOAMP notice | 1,620,712 |
| Episodes | one reconstructed procurement procedure | 1,103,632 |
| Study cohort | one awarded Grand Ouest digital episode | 3,800 |
| Candidate table | one anchor-candidate pair | 763,417 |
| Survival table | one cohort episode with an observed or censored time | 3,800 |

Each arrow between these layers is a change of statistical unit, and each introduces uncertainty that propagates to the final results. That is why the report treats them as methodological objects rather than preparation steps.

---

# 4. Constructing the outcome: successor linkage

## 4.1 Why this step exists

Survival analysis needs two numbers per contract: a delay and an event indicator. The BOAMP supplies neither. This section builds both, and measures the quality of that construction — because an error here is corrected nowhere downstream.

## 4.2 Candidate generation

For each **anchor** $i$ (an awarded cohort episode with origin $u_i$), a later episode $j$ with origin $v_j$ is exposed for comparison if and only if:

- buyer identifiers are compatible — same buyer key or same normalised name, with two different validated SIRENs disqualifying;
- and the chronology satisfies
$$u_i + 90\ \text{days} \le v_j \le u_i + 2{,}920\ \text{days}.$$

The 90-day to 8-year window is an **operational search range**, not an assumed contract duration. Both bounds are set empirically, and the code records why. Without a floor, a pathology appeared: `132` of the `628` links then accepted fell within three months of the award, and every link for a 2025 award sat under twelve months, with a median of 1.7 — which no renewal cycle explains. Those were concurrent procurements, typically another lot of the same programme. The floor was then calibrated against the reference data rather than tuned: among the 23 confirmed successors that resolve onto the cohort, the shortest award-to-successor gap is **139 days**, so a 90-day floor discards none of them while removing the concurrent-procurement band. The eight-year ceiling excludes no confirmed successor.

The result is **763,417 pairs** covering **3,520 of the 3,800 anchors**. The 280 anchors with no candidate at all stay in the analysis as censored exposure: their status is decided by blocking, not by the acceptance bar.

**No hard CPV filter, and the argument is empirical.** Requiring the successor to share the anchor's CPV division would be tempting. The reference forbids it: of the 23 reviewed successors, **9 (39.1 %) cross a division**. Hard blocking would destroy them and cut the attainable recall ceiling from 0.913 to 0.609. The literature points the same way: CPV assignment is documented as error-prone even for experts (Siciliani et al., 2023). CPV is therefore used as continuity evidence, never as a hard constraint.

## 4.3 Four methods on the same pairs

| Method | Role | Principle |
|---|---|---|
| `M_A_deterministic` | Conservative comparator | Requires buyer, CPV and a minimum text-similarity floor |
| `M_B_text_ranking` | **Primary method** | Takes the highest TF-IDF cosine candidate and accepts it above the threshold |
| `M_C_weighted_gated` | High-recall comparator | Weighted score, buyer 0.50 / text 0.25 / CPV 0.20 / time 0.05, renormalised over observed evidence |
| `M_D_fellegi_sunter` | Probabilistic comparator | Match / non-match likelihood ratios (Fellegi and Sunter, 1969) |

The primary decision rule is
$$\hat{j}_i = \arg\max_{j \in J_i} T_{ij}, \qquad Y_i = \mathbf{1}\!\left(T_{i\hat{j}_i} \ge 0.70\right),$$
where $T_{ij}$ is the maximum of a word-level and a character-level TF-IDF cosine similarity. **At most one successor per anchor**; otherwise the method abstains.

One implementation detail has consequences analysed in § 5.7: the vectoriser is fitted *per anchor block*. That choice has a reason — the IDF weights become local to the buyer's own vocabulary, which stops the administrative boilerplate common to all of a buyer's notices from dominating the similarity — and it has a cost: the score becomes block-relative, a fixed 0.70 does not mean quite the same thing for every anchor, and the maximum over a larger block is mechanically higher. § 5.7d measures that cost rather than assuming it negligible.

## 4.4 Why the threshold was frozen before evaluation

The 0.70 threshold follows a precision-first principle: a false link fabricates both an event and its date, whereas a missed link fabricates nothing. Project history shows that reference evidence informed the retained policy, so the split evaluation is **internal validation rather than an untouched holdout**. The threshold sweep (Annex C) shows 0.60 would buy recall; it remains a sensitivity arm, and replacing the frozen post-development policy requires fresh independent evidence.

The brief anticipated a linkage rate of 40-60 %. The realised rate is **14.3 %**. This is not a shortfall against a target: the brief's figure was treated as a planning assumption, never as an optimisation objective, and 14.3 % is the arithmetic consequence of a precision-first rule combined with an event defined as *observable in the BOAMP*.

## 4.5 The regional reference, and what it is worth

Evaluation rests on a stratified review of **120 Grand Ouest anchors**, carried out on 11 August 2026 against the real BOAMP notices and their official URLs, **before the linkage methods in this project existed**. 112 anchors re-resolve onto exactly one episode and 88 are usable; the pilot (16 anchors) / locked (72 anchors) split is recorded in the reference file itself and is not post hoc.

Four limits bound its reach, and they belong in the body of the report rather than in a footnote:

1. **The labels come from a single large-language-model research pass**, spot-checked by the project owner rather than verified anchor by anchor, and not judged by a specialist panel. They are independent of every method scored — none existed at the time — but they are not human ground truth.
2. **The per-anchor evidence trail was not recorded**: the sources behind a given decision cannot be reconstructed.
3. **Negatives are corpus-relative**: roughly 25 candidates per anchor were considered, not the full pool. A negative means "no successor identified among those examined".
4. **Candidate-surfacing independence does not hold.** The rule that selected those ~25 candidates (from pools of up to 3,258) is recorded nowhere, and all 16 retrievable locked-split successors rank within the top thirteen by the very text score being evaluated. Consequently **recall and the 0.913 ceiling are not fully independent of the score they bound**. Precision is unaffected: a false positive is a false positive however the list was assembled.

## 4.6 Results on the locked split

Two accountings coexist and do not answer the same question. *Anchor-level* accounting asks whether the system detected that a successor exists. *Exact-successor* accounting, which is stricter, asks whether it named the right one; a wrong successor on a positive anchor then counts as one false positive **and** one false negative. The project's headline figures are the latter.

| Method | Threshold | TP | FP | FN | TN | Precision | 95 % CI | Recall | FPR |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| `M_A_deterministic` | n/a | 8 | 7 | 10 | 47 | 0.533 | 0.301-0.752 | 0.444 | 0.130 |
| **`M_B_text_ranking`** | **0.70** | **7** | **1** | **11** | **54** | **0.875** | **0.529-0.978** | **0.389** | **0.000** |
| `M_C_weighted_gated` | 0.70 | 12 | 11 | 6 | 44 | 0.522 | 0.330-0.708 | 0.667 | 0.185 |
| `M_D_fellegi_sunter` | 0.65 | 1 | 4 | 17 | 53 | 0.200 | 0.036-0.625 | 0.056 | 0.018 |

Three readings, in order of importance.

**The intervals overlap heavily.** A precision of 0.875 resting on **eight accepted links** has an interval running from "coin flip" to "near perfect". The figure must never be quoted without it, and this reference separates the methods only coarsely.

**Recall is capped before any scoring.** Candidate generation exposes only 21 of the 23 reviewed successors, a pairs completeness of 0.913 (Christen and Goiser, 2007); both unreachable cases are attributed to a named blocking condition — one anchor never entered the cohort for want of a structured address, one buyer changed legal form from CCAS to CIAS with no shared SIREN. No unexplained case, which is what distinguishes an owned ceiling from an implementation defect.

**The precision-recall trade is deliberate.** `M_C` buys recall (0.667) at half the precision (0.522). For survival analysis that is not a neutral exchange: its 11 false positives would introduce eleven events that never happened, with eleven dates that never happened.

## 4.7 Challenge review and linkage audit

A blinded 60-pair sample was prepared (20 production-accepted links, 20 high-similarity structural negatives, 20 buyer-declared relationships), with a separate audit key and a pre-specified acceptance rule. The review that was carried out is **model-assisted, not independent-human** — its provenance is recorded explicitly in the repository. Of the 20 accepted links: 14 confirmed, 5 rejected, 1 uncertain, giving a precision of **0.700** on a conservative reading (95 % CI [0.457; 0.881]), **below the 0.80 target** the project set itself.

The result is reported as it stands. It does not close the validation gate; it makes closing it necessary. Independent human review is the first recommendation in § 9.F.

## 4.8 What linkage produces

**544 accepted links** over 3,800 anchors, an event rate of **14.32 %**. CPV continuity: 351 of the 538 links with both divisions observed stay within one division (65.2 %), comparable to the reviewed reference successors (14 of 23, 60.9 %). 461 distinct successors for 544 links: 44 episodes are accepted by more than one anchor, the most reused by eleven — a false-positive signature exploited in § 5.7. Zero accepted links with conflicting validated SIRENs; zero municipality / inter-municipal mixes.

---

# 5. Survival analysis

## 5.1 Time and censoring

For an anchor $i$ whose successor $\hat{j}_i$ is accepted:
$$\tau_i = v_{\hat{j}_i} - u_i, \qquad Y_i = 1.$$
If no successor is accepted before 31 December 2025:
$$\tau_i = \text{2025-12-31} - u_i, \qquad Y_i = 0,$$
and the row is **administratively right-censored**. Time zero is the award date.

The cohort therefore holds **3,800 rows, 544 events and 3,256 censored observations** (85.7 %). That high censoring rate is precisely why survival analysis is used rather than a regression on the linked contracts alone: ignoring censoring would restrict the analysis to contracts whose successor has been seen, that is, condition the analysis on the outcome.

## 5.2 Non-parametric estimation

The Kaplan-Meier estimator (Kaplan and Meier, 1958),
$$\hat S(t) = \prod_{t_k \le t}\left(1 - \frac{d_k}{n_k}\right),$$
with $d_k$ events at $t_k$ and $n_k$ at risk, assumes no distributional form and handles administrative censoring without bias as long as censoring is independent of the hazard.

| Horizon | P(observable successor) | Survival |
|---|---:|---:|
| 12 months | **4.62 %** | 0.9538 |
| 24 months | **6.73 %** | 0.9327 |
| 36 months | 8.68 % | 0.9132 |
| 48 months | 15.50 % | 0.8450 |
| 60 months | 17.54 % | 0.8246 |

**The Kaplan-Meier median is not reached**: the curve never falls below 0.5 within the observation window. The 31.8 months quoted elsewhere is the median *among linked events only*; the two quantities share a word and measure different things, the second being conditional on the event having occurred.

The jump between 36 and 48 months — 8.7 % to 15.5 % — is the **renewal shoulder**, consistent with the general four-year limit on framework agreements. It is the most interesting part of the curve for Gigalis and also the most fragile: at 48 months the risk set contains only pre-2022 awards, so cohort composition changes along the curve. It should not be quoted without the at-risk counts (Annex D).

**Between-segment comparison.** The multivariate log-rank test across the four CPV segments gives a statistic of **23.45** for $p = 3.3\times10^{-5}$: the segments do not behave alike. Raw event rates run from 11.6 % (CPV-48) to 20.4 % (CPV-35).

## 5.3 Conditional probabilities: the operational output

For a contract that has reached age $a$ with no observed successor, the probability that one appears within the next $h$ months is
$$P(T \le a+h \mid T > a) = 1 - \frac{S(a+h)}{S(a)},$$
read off the Kaplan-Meier estimator, with 500-draw episode-bootstrap intervals.

| Contract age | P(successor ≤ 12 m) | 95 % CI | P(≤ 24 m) | 95 % CI |
|---:|---:|---|---:|---|
| 0 months | 4.62 % | [3.91; 5.24] | 6.73 % | [5.91; 7.58] |
| 12 months | 2.22 % | [1.70; 2.73] | 4.26 % | [3.55; 4.98] |
| 24 months | 2.09 % | [1.58; 2.60] | 9.40 % | [8.33; 10.68] |
| **36 months** | **7.46 %** | [6.52; 8.59] | **9.69 %** | [8.58; 11.12] |
| 48 months | 2.41 % | [1.77; 3.07] | 2.89 % | [2.15; 3.72] |

The profile is not monotone in age: it rises into the 36-48 month shoulder and falls away after it. That is what a portfolio of multi-year contracts should look like, and it is what makes the table usable as a **cohort-level watch logic**: it ranks ages and segments, it does not predict a contract.

## 5.4 Cox model: comparative associations

The proportional-hazards model (Cox, 1972),
$$h(t \mid X) = h_0(t)\exp(\beta_1X_1 + \cdots + \beta_pX_p),$$
estimates covariate effects without specifying the baseline hazard. Five covariates are retained, consistent with the one-parameter-per-ten-events rule: CPV segment, region, framework status, validated-SIREN availability, and centred award year — eight parameters for 544 events.

| Covariate | HR | 95 % CI | p |
|---|---:|---|---:|
| Framework agreement | 1.751 | [1.435; 2.136] | 3.4 × 10⁻⁸ |
| **CPV-35** (ref. CPV-32) | **1.553** | [1.218; 1.981] | **3.8 × 10⁻⁴** |
| CPV-48 | 0.828 | [0.638; 1.073] | 0.153 |
| CPV-72 | 1.056 | [0.850; 1.310] | 0.624 |
| Normandie (ref. Bretagne) | 0.800 | [0.640; 1.000] | 0.050 |
| Pays de la Loire | 1.003 | [0.815; 1.234] | 0.979 |
| Validated SIREN | 1.082 | [0.885; 1.323] | 0.443 |
| Award year (centred) | 1.107 | [1.066; 1.149] | 1.2 × 10⁻⁷ |

In-sample concordance: 0.626.

**The proportional-hazards diagnostic rejects constant effects for three covariates**: award year (statistic 70.7, $p = 4\times10^{-17}$), framework status (7.04, $p = 0.008$) and validated SIREN (6.76, $p = 0.009$). The award-year violation is severe and expected: award year is structurally confounded with follow-up length. These coefficients are therefore **time-averaged descriptive associations, not effects**. Stratifying on award-year bands would be cleaner; it was judged optional because the coefficients of interest barely move and nothing operational rests on this model.

## 5.5 Temporal validation: a negative result

The model is fitted once on 2015-2021 awards and scored out of time, with no refitting and no retuning.

| Window | Train N | Train events | Test N | Test events | Train C | **Test C** |
|---|---:|---:|---:|---:|---:|---:|
| Primary, 2022-2024 | 2,470 | 392 | 1,004 | 107 | 0.606 | **0.479** |
| Sensitivity, 2022-2025 | 2,470 | 392 | 1,330 | 152 | 0.606 | **0.518** |

A concordance index of 0.479 is **indistinguishable from chance**. The model does not usefully rank individual contracts by time to successor on unseen award years. Part of that is structural — contracts awarded from 2022 can only contribute short gaps, and Harrell's C depends on the censoring distribution (Uno et al., 2011), so two windows with unequal follow-up are not strictly comparable.

The result is published as it stands, and it drives a decision: **no individual prediction is produced**. The brief expected a table of the twenty contracts most likely to be renewed; producing it would convey a precision the validation does not support. The operational deliverable is therefore the conditional table of § 5.3 and, in the annex, a **segment-stratified watch list** — five contracts per CPV segment among those awarded since 2021 — read off the segment-stratified Kaplan-Meier curves rather than the Cox model. The stratification is deliberate: since the conditional probability is a function of segment and age alone, a global ranking would mechanically return the highest-hazard segment at the age closest to the shoulder, implying an individual-level granularity that does not exist.

## 5.6 Parametric models

Five families were fitted and compared on AIC and BIC:

| Model | Parameters | Log-likelihood | AIC | BIC |
|---|---:|---:|---:|---:|
| **Generalised gamma** | 3 | −3,757.5 | **7,521.0** | **7,539.7** |
| Log-normal | 2 | −3,784.6 | 7,573.2 | 7,585.7 |
| Log-logistic | 2 | −3,807.3 | 7,618.6 | 7,631.1 |
| Weibull | 2 | −3,814.7 | 7,633.4 | 7,645.8 |
| Exponential | 1 | −3,829.7 | 7,661.3 | 7,667.6 |

The generalised gamma wins on both criteria. It is **not** the source of the operational figures, and that choice deserves defending: every one of these families is smooth and flattens the empirical 36-48 month shoulder, which is precisely the business object; and every published horizon (12, 24 months) lies inside the observed window, where the empirical estimator needs no shape assumption. The parametric model is therefore kept as the best-fitting family and as the instrument any extrapolation beyond 31 December 2025 would use. Mechanically taking the lowest AIC to produce operational numbers would have traded a slightly better global fit for a worse rendering of the only region the user cares about.

## 5.7 Robustness: four independent tests

This is the most important section of the report for interpretation, because it separates what moves from what holds.

**(a) Four event definitions.**

| Arm | Events | Rate | P(≤ 12 m) | P(≤ 24 m) | Median observed gap |
|---|---:|---:|---:|---:|---:|
| `M_B @ 0.80` strict | 296 | 7.8 % | 2.37 % | 3.23 % | 35.7 months |
| **`M_B @ 0.70` primary** | **544** | **14.3 %** | **4.62 %** | **6.73 %** | **31.8 months** |
| `M_B @ 0.60` loose | 853 | 22.4 % | 8.00 % | 11.47 % | 26.6 months |
| `M_C @ 0.70` contrast | 1,332 | 35.1 % | 12.21 % | 17.98 % | 26.1 months |

A factor of 4.5 on the event count. **No absolute probability can stand alone.** One observation cuts the other way, however: plotted together, the four survival curves shift vertically but keep the same **shape**, with the drop between 40 and 48 months visible in all four arms. The level depends on the rule; the renewal shoulder is not an artefact of where the acceptance line was drawn.

**(b) Borderline band.** The most fragile decisions are those whose best candidate sits near the bar. Removing the 280 episodes scoring in $[0.65; 0.75]$ — a symmetric band fixed a priori and never searched over — drops 133 events and gives: P(≤ 12 m) 4.62 % → 3.72 %, CPV-35 HR 1.553 → **1.780**, framework HR 1.751 → 1.616. Both headline hazard ratios keep their direction.

**(c) Template-risk re-censoring.** The first two tests move the acceptance bar. But the false-positive mechanism the linkage audit identified produces links well **above** it: French award notices carry long standardised framework boilerplate on which character n-grams score highly between unrelated objects, and `M_B` ranks candidates independently per anchor, so one episode can be accepted for several anchors. Two observable signatures define the at-risk group: word-level similarity below 0.50 (65 links), or a successor shared with another anchor (127 links) — together **173 of the 544 links (31.8 %)**. Those anchors are **re-censored at the cut-off** rather than deleted, because that is the counterfactual a spurious link implies: the anchor had no observed successor and should contribute its full follow-up as censored exposure. Result: P(≤ 12 m) falls to 2.64 %, but CPV-35 HR = **1.541** and framework HR = **1.692**. This is the test the framework finding most needed, since that boilerplate is the very text driving the mechanism.

**(d) Detectability.** The largest imbalance between linked and censored episodes is not a property of the contract at all:

| Variable | Linked mean | Censored mean | SMD |
|---|---:|---:|---:|
| **log(candidate pool size)** | 4.787 | 3.972 | **+0.470** |
| candidate pool size | 274.3 | 188.6 | +0.285 |
| text length | 1,611 | 1,087 | +0.262 |
| framework flag | 0.265 | 0.187 | +0.187 |

`M_B` takes the maximum text score over the candidate block, and the maximum of more draws is larger: a buyer who publishes prolifically is mechanically more likely to yield an accepted link. This is an **observability** channel, not a cause of re-procurement. A sensitivity model adds $\log(1 + \text{pool size})$:

| Covariate | HR, main model | HR, + log(pool) | adjusted p |
|---|---:|---:|---:|
| Framework agreement | 1.751 | **1.617** | 2.6 × 10⁻⁶ |
| CPV-35 | 1.553 | **1.512** | 8.5 × 10⁻⁴ |
| log(pool size) | — | 1.184 | 6.0 × 10⁻⁹ |

Two readings follow, and they differ. **CPV-35 is insensitive to it**: the hazard ratio moves from 1.553 to 1.512. Together with its stability across the four arms, the borderline band and template-risk re-censoring, it is the study's most robust comparative finding. **The framework association is partly detectability**: roughly 14 % of the log hazard ratio evaporates. The direction survives every test, but the association is smaller than the main model alone implies, and the alternative explanation is not only boilerplate — it is also publication volume.

The main model is unchanged: this column is a sensitivity, not a new reference specification.

## 5.8 What the survival analysis establishes

Under the retained event definition: the probability that an awarded Grand Ouest digital procurement shows an observable successor is estimated at 4.6 % by twelve months and 6.7 % by twenty-four, with a marked shoulder between 36 and 48 months; segments differ clearly, CPV-35 being fastest; framework agreements show a successor earlier, partly because they are more detectable; absolute levels vary by a factor of four and a half with the linkage rule, while relative comparisons survive all four tests. The model does not discriminate out of time and is therefore not a tool for individual prediction.

---

# 6. Text signal: a supervised technology taxonomy

## 6.1 Why CPV is not enough

The cohort is defined by four CPV divisions. That is reproducible and auditable — and coarse: CPV says a procurement is digital, not what was bought. Yet every business question Gigalis asks lives at the second level: which segments are growing, which are re-procured fastest, where to open a framework agreement. This section learns that missing variable from the text of the contract object:

$$\text{procurement object text} \longrightarrow \text{supervised classifier} \longrightarrow \text{business technology class}$$

It is an **enrichment layer**. It does not replace the CPV segmentation, which remains the cohort definition, the Cox covariate, and the axis of the time series.

## 6.2 Taxonomy and corpus

Eight substantive classes — `CLOUD_HOSTING`, `CYBERSECURITY`, `NETWORK_TELECOM`, `IT_INFRASTRUCTURE`, `BUSINESS_SOFTWARE`, `DATA_BI`, `AI`, `IT_SERVICES` — plus three fallback classes that are **annotation decisions**, not missing values: `MIXED` (no dominant technology), `OTHER_DIGITAL` (a digital purchase outside the eight), `OTHER` (a digital CPV on something that is not a technology purchase). The taxonomy was frozen before any modelling.

The corpus holds **500 manually annotated notices**, 2015-2025, all with a non-empty object text (median 14 words). Two properties constrain how it may be read. The sample is **quota-stratified**: the class proportions are a property of the annotation design, not an estimate of prevalence. And the `AI` class holds only **7 notices** across eleven years: no synthetic examples were created, no rows duplicated, no oversampling applied. The consequence is published rather than engineered away.

## 6.3 Leakage prevention

The BOAMP republishes procedures, and buyers re-run near-identical consultations a few years apart. Scoring a model on a near-copy of a document it trained on measures memorisation. Every notice is therefore assigned to a **procurement family**, defined as the union of two rules: notices already grouped into one reconstructed episode, and notices whose objects reach a character-level cosine similarity of 0.80. Each family belongs to exactly one fold.

The second rule is not redundant: the first alone gives 486 groups; adding the second merges 39 near-duplicate pairs, of which **29 sit in different episodes** and would otherwise have been split across folds. The result is **459 families**, none spanning two folds.

One side effect deserves mention: **3 families contain notices with near-identical text but different labels** — videoconference services labelled `NETWORK_TELECOM` in 2017 and `OTHER_DIGITAL` in 2021; an infrastructure-as-a-service procurement labelled `MIXED` in 2017 and `CLOUD_HOSTING` in 2021. No label was changed: no evidence decides which reading is right, and editing labels after seeing the model's errors is how a corpus is fitted to its classifier. They are recorded as an empirical floor on attainable accuracy.

## 6.4 Representation and models

The input is the `objet` field alone. Normalisation is deliberately light — mojibake repair, NFC, lowercasing, whitespace — and **accents are preserved**, with no stemming: the classes are distinguished by words such as *cybersécurité*, *logiciel métier* and *intelligence artificielle*, and flattening French orthography would discard the evidence. Features are TF-IDF word n-grams; both unigrams and unigrams-plus-bigrams were in the search space, and **every fold selected unigrams alone**, the bigram vocabulary being too sparse across 500 short documents.

Excluded from the features by design: buyer identity, geography, dates, amounts, procedure type, framework status, notice identifiers, every linkage variable, and CPV — which serves as the comparator.

Six specifications were compared on identical folds, with hyperparameters selected by inner cross-validation *inside each training fold*:

| Model | Features | Out-of-fold macro-F1 |
|---|---|---:|
| Majority class | — | 0.027 |
| CPV only | CPV codes | 0.441 |
| CPV + BOAMP descriptors | administrative | 0.473 |
| TF-IDF + logistic regression | text | 0.670 |
| **TF-IDF + class-weighted logistic regression** | text | **0.744** |
| TF-IDF + linear SVM | text | 0.718 |
| TF-IDF + class-weighted linear SVM | text | 0.715 |

## 6.5 The headline result

| Comparison | Macro-F1 | 95 % CI (family bootstrap) |
|---|---:|---|
| TF-IDF + class-weighted logistic regression | **0.7442** | [0.682; 0.791] |
| Best CPV / descriptor comparator | 0.4731 | [0.413; 0.526] |
| **Paired difference** | **+0.2711** | **[0.201; 0.340]** |

The interval on the difference excludes zero. Three precautions make this result solid rather than trivial: both sides use **the same folds** and **the same hyperparameter search budget**; the difference is **paired** fold by fold; and the bootstrap resamples **procurement families**, the unit at which dependence exists, rather than notices.

In other words, the text of the notices carries business information the administrative vocabulary does not. The crosswalk makes this concrete:

| CPV division | Dominant class | Dominant count | Purity | Other classes ≥ 100 episodes |
|---|---|---:|---:|---|
| CPV-32 (telecoms) | NETWORK_TELECOM | 426 / 1,152 | 0.370 | OTHER_DIGITAL 202, BUSINESS_SOFTWARE 146, IT_INFRASTRUCTURE 128 |
| CPV-35 (security) | NETWORK_TELECOM | 165 / 564 | 0.293 | OTHER 114, CYBERSECURITY 106 |
| CPV-48 (software) | BUSINESS_SOFTWARE | 348 / 790 | 0.441 | — |
| CPV-72 (IT services) | IT_SERVICES | 325 / 1,294 | 0.251 | BUSINESS_SOFTWARE 310, NETWORK_TELECOM 204, OTHER_DIGITAL 165 |

Two readings. The mean purity of a CPV segment against the learnt taxonomy is **0.34**: no division corresponds to a technology. And the largest division, CPV-72, is the least pure of the four (0.251): what the administrative vocabulary calls "IT services" in fact spans four business families of comparable size. One telling detail: CPV-32 and CPV-35, two distinct divisions of the vocabulary, share the **same** dominant predicted class.

## 6.6 What the classifier can and cannot do

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| DATA_BI | 0.923 | 0.828 | **0.873** | 29 |
| OTHER | 0.923 | 0.800 | 0.857 | 15 |
| BUSINESS_SOFTWARE | 0.798 | 0.898 | 0.845 | 88 |
| NETWORK_TELECOM | 0.810 | 0.840 | 0.824 | 81 |
| CYBERSECURITY | 0.848 | 0.722 | 0.780 | 54 |
| IT_SERVICES | 0.778 | 0.757 | 0.767 | 74 |
| IT_INFRASTRUCTURE | 0.702 | 0.717 | 0.710 | 46 |
| OTHER_DIGITAL | 0.655 | 0.679 | 0.667 | 53 |
| AI | 0.800 | 0.571 | *0.667* | **7** |
| CLOUD_HOSTING | 0.731 | 0.594 | 0.655 | 32 |
| MIXED | 0.482 | 0.619 | 0.542 | 21 |

Macro-F1 0.741 (fold standard deviation 0.034), weighted F1 0.765, accuracy 0.766. Two aggregations coexist and should be kept apart: 0.741 is the **mean across the three folds**, whereas the headline figure of § 6.5, 0.744, is computed **out of fold on the pooled 500 predictions**. The gap is the expected one between an average of per-fold scores and a single score over all predictions.

Three readings. The reliable classes are `CYBERSECURITY`, `NETWORK_TELECOM`, `BUSINESS_SOFTWARE`, `DATA_BI`, `IT_SERVICES` and `OTHER`. `MIXED` is weak, which is consistent with its definition as the bucket for procurements without a dominant technology. And **`AI` is not interpretable**: its F1 of 0.667 rests on three correct predictions; a high number would be a small-sample artefact and a low one equally uninformative.

Error analysis on thirty representative cases gives: 16 model errors, 7 taxonomy-boundary ambiguities, 4 genuinely multi-technology procurements, 2 annotation inconsistencies, 1 object text too poor to decide. Close to half the errors therefore come from class definitions or from missing information rather than from model capacity.

**Temporal robustness.** Training 2015-2022 (n = 393), test 2023-2025 (n = 107), with the 4 boundary-straddling families assigned to training — which costs test observations and cannot flatter the result. All-class macro-F1 0.662, but **0.815 on classes with test support of at least 10**. The vocabulary of recent notices has not drifted for the high-volume classes; nothing can be concluded for the six classes with thin recent support.

## 6.7 Two negative decisions, taken against pre-written criteria

**The transformer was not run.** The rule was written before the classical results were read: test CamemBERT only if the classical model is materially inadequate (macro-F1 < 0.55) **and** fewer than half its errors come from label ambiguity or missing information, which no encoder can supply. The model reaches 0.741: the first condition fails decisively. CamemBERT was **not tested and then discarded; it was not run**, because the criterion for running it was not met. The learning-curve diagnosis confirms this was the right call: training F1 sits near 0.99 at every sample size while validation F1 climbs from 0.434 to 0.747 — a **variance** problem, for which added capacity is the wrong medicine. The brief itself allowed keeping the baseline if the gain were under five F1 points.

**Calibration was evaluated and rejected.** The published confidence is the raw predicted probability of the logistic regression. Platt scaling was estimated inside the same grouped splits and judged against a pre-specified rule: adopt it only if it cuts expected calibration error by at least 0.02 **and** costs at most 0.02 macro-F1. It cut the error by 0.1405 — condition met — but cost 0.0364 macro-F1, exceeding the budget. The rule was not relaxed to admit it.

The deployed score is therefore an **uncalibrated confidence score**. Its out-of-fold reliability is measured: observed accuracy rises from 0.52 in the [0, 0.3) bin to 1.00 in the [0.9, 1.0) bin, but the gap between stated confidence and observed accuracy is positive in every bin above 0.3 — the score **understates** its own hit rate, with an expected calibration error of 0.350. It is therefore usable as a **ranking and a filter** — at the 0.70 operational cutoff, out-of-fold accuracy is 0.956 on the 9 % of notices that clear it, against 0.750 below — but a confidence value is not the probability that the label is correct. Two reasons: the miscalibration above, and the fact that the corpus is quota-stratified while the deployment population is not, so the class prior the classifier encodes is an artefact of the annotation design.

## 6.8 Deployment

The model is refitted on all 500 labels and applied to the **3,800 cohort episodes** through the object text of each episode's origin notice. That refit has no validation score and none is reported: the evidence for the model is the grouped cross-validation and the temporal check above. Every episode receives exactly one predicted class and one confidence value; none is discarded. **235 episodes (6.2 %)** clear the 0.70 cutoff.

Predicted composition: `NETWORK_TELECOM` 859, `BUSINESS_SOFTWARE` 854, `IT_SERVICES` 492, `OTHER_DIGITAL` 462, `CYBERSECURITY` 316, `IT_INFRASTRUCTURE` 298, `OTHER` 173, `MIXED` 139, `CLOUD_HOSTING` 115, `DATA_BI` 86, `AI` 6. **These counts are not market shares**: they are predictions carrying the error rate of § 6.6, over a cohort defined by an inclusive CPV rule.

---

# 7. Downstream use of the predictions

## 7.1 Two gates before any use

Nothing was re-run mechanically for eleven classes. A class enters downstream analysis only by clearing **both** gates, fixed before any curve was fitted.

**Gate A — classifier evidence.** Does the label mean anything? A class the model cannot separate produces a group that mixes several technologies, and a curve fitted to it estimates the mixture. The gate requires a substantive class, annotated support of at least 10, and out-of-fold F1 of at least 0.65. The three fallback classes are excluded outright: `OTHER_DIGITAL` simultaneously holds video surveillance, RFID and website maintenance; placing that bucket beside cybersecurity in a "comparison across technologies" would invite reading the heterogeneity of the bucket as a technology effect.

**Gate B — statistical support.** At least 100 episodes and 20 events. A perfectly classified class with fourteen episodes and one event cannot carry a curve.

Five classes of eleven clear both: `CYBERSECURITY`, `NETWORK_TELECOM`, `IT_INFRASTRUCTURE`, `BUSINESS_SOFTWARE`, `IT_SERVICES`. `CLOUD_HOSTING` (115 episodes, 11 events) and `DATA_BI` (86, 15) fail Gate B; `AI` fails both.

**What Gate A costs.** The same log-rank test over every class clearing the statistical gate alone — that is, with the residual classes reinstated — gives $p = 0.000119$ against $p = 0.0363$ with the gate. The gate **weakens** the result, and it was kept. That is published in the technical report, and it is the strongest available evidence that the analysis was not steered toward a result.

## 7.2 Survival by technology

| Class | Episodes | Events | P(successor ≤ 24 m) | At risk at 24 m |
|---|---:|---:|---:|---:|
| BUSINESS_SOFTWARE | 854 | 106 | 8.57 % | 673 |
| NETWORK_TELECOM | 859 | 140 | 6.49 % | 716 |
| CYBERSECURITY | 316 | 60 | 6.46 % | 246 |
| IT_SERVICES | 492 | 72 | 6.17 % | 393 |
| IT_INFRASTRUCTURE | 298 | 38 | 6.11 % | 239 |

The multivariate log-rank test over these five classes gives a statistic of 10.26 for $p = 0.0363$ on 416 events: **the timing of the observable successor differs across the analysed technologies**. The spread between extremes is 2.5 points at 24 months.

Confidence in this result is moderate, for three cumulative reasons: the event is an accepted observable successor under the frozen rule, not a renewal; the class labels are **predictions** carrying the error rate of § 6.6, which does not propagate into the published intervals; and the comparison is adjusted for neither buyer, size, nor procedure.

## 7.3 Trends by technology

Five quarterly series are fitted on the retained classes. **None shows a linear trend**, before or after multiplicity correction: the smallest raw p-value is 0.0563 (`NETWORK_TELECOM`, slope −0.17 episodes per quarter), which becomes 0.28 under Holm across the five simultaneous tests. The reading is therefore: over the observed window and at this noise level, **no technology trend is established**.

## 7.4 The three inherited uncertainties

Every technology-level figure inherits, in this order: **linkage uncertainty** (the event itself depends on the rule, § 5.7), **classifier uncertainty** (macro-F1 0.744, with errors concentrated on definitional boundaries), and **sampling uncertainty** (the published intervals). Only the third appears in the confidence intervals. That is acceptable for a layer explicitly presented as enrichment; it would not be for a reference analysis.

---

# 8. Trends and change-point detection

## 8.1 Constructing the series

For a segment $s$ and quarter $q$, the count is
$$N_{s,q} = \sum_i \mathbf{1}(S_i = s,\ Q_i = q).$$
The series span **43 quarters** (2015Q2 to 2025Q4); the partial first quarter of 2015 is excluded and every subsequent quarter is represented, zeros included. Amount series are absent, for want of a validated canonical awarded value (§ 3.4).

The objective, as the brief states, is not to forecast: it is to **date and qualify changes**.

## 8.2 Three instruments, three different questions

**PELT** (Killick, Fearnhead and Eckley, 2012) minimises
$$\sum_{r=0}^{m}\mathcal{C}\big(y_{\tau_r+1:\tau_{r+1}}\big) + \beta m,$$
where $\mathcal{C}$ is within-segment squared error and $\beta = \lambda\log(n)$ after standardisation. The first term rewards fit; the second charges a fixed price per break, which is what stops the optimum from placing a break between every pair of quarters. The central result uses $\lambda = 1$, with sensitivity at 0.5 and 2.0, and **a break is declared stable only if it appears within one quarter under all three penalties**. That requirement eliminates half the candidate breaks.

**ADF and KPSS** test opposite null hypotheses — unit root for the first, level stationarity for the second — and are therefore read together. They are not forced to agree: disagreement means the series is not cleanly classified over so short a window, and that is the useful information.

**The hidden Markov model** with three states is fitted on the **quarter-over-quarter change** $\Delta N_t$, not on the level. Its states therefore describe a typical direction of change — decline, plateau, growth — not a segment's activity level. The reported figure is a posterior probability of the current state, not a forecast.

## 8.3 Signal matrix

| Segment | Recent direction | Slope (episodes/quarter) | Raw p | Holm p | BH p | Reading | Last stable break |
|---|---|---:|---:|---:|---:|---|---|
| Overall | stable or uncertain | −0.11 | 0.921 | 1.000 | 0.989 | no signal | — |
| CPV-32 | stable or uncertain | −0.01 | 0.989 | 1.000 | 0.989 | no signal | 2020Q2 |
| CPV-35 | stable or uncertain | +0.03 | 0.923 | 1.000 | 0.989 | no signal | — |
| **CPV-48** | **decreasing** | **−0.84** | **0.032** | 0.159 | 0.159 | **nominal signal only** | 2024Q1 |
| CPV-72 | stable or uncertain | +0.70 | 0.285 | 1.000 | 0.714 | no signal | 2021Q1 |

Five slopes are fitted and read at once, so the raw p-values are reported beside Holm (family-wise) and Benjamini-Hochberg (false discovery rate) adjustments. A segment whose raw p clears the pre-declared exploratory level of 0.10 but whose Holm p does not is **a nominal signal to monitor, not a finding**.

That is the case of CPV-48, and it is the one row in the table requiring an editorial decision. It is presented as it stands: the recent decline in CPV-48 procurement is the clearest signal in the panel, it does not survive correction for the five segments tested, and it justifies watching for a few more quarters — not a conclusion, still less a forecast.

## 8.4 Breaks and regimes

Only three breaks are stable under all three penalties: CPV-32 in 2020Q2, CPV-48 in 2024Q1 and CPV-72 in 2021Q1. Neither the overall series nor CPV-35 carries one: their candidate breaks disappear as soon as the penalty changes. **None is attributed to a cause.** A PELT break dates a change in level; it does not say why. Attributing the 2020Q2 break to the pandemic would be plausible and undemonstrated, and the report abstains: such attributions require documentary evidence and stakeholder validation.

The hidden Markov model places the overall series, CPV-32 and CPV-72 in a growth regime in the final quarter (posterior probabilities 0.750, 0.992 and 0.594). That label may look inconsistent with the null twelve-quarter slopes; it is not. The slope summarises twelve quarters, the regime describes the most recent change. The two are complementary, and the report does not reconcile them artificially.

## 8.5 What the trend analysis establishes

Over the observed window, **no CPV segment shows a recent linear trend surviving multiplicity correction**, and neither does any technology series. Three level shifts are stably dated. The operational reading is therefore a monitoring arrangement: keep watching all segments, and follow CPV-48 as a lead to investigate. This is a weaker result than the brief hoped for, and it is consistent with the nature of the data: forty-three quarters of low counts do not support robust trend inference.

---

# 9. Discussion

## A. What the work establishes with reasonable confidence

**The text of the notices carries a business segmentation that CPV does not.** This is the best-established result of the internship: a paired difference of +0.271 macro-F1, interval [0.201; 0.340], estimated at the correct level of aggregation, on identical folds with an identical search budget. It answers the brief's text-signal problem directly, and it has an immediate practical consequence: a technology-level reading is possible where the administrative vocabulary does not allow one.

**CPV segments differ clearly in time to observable successor**, with CPV-35 fastest. The result survives four independent perturbations of the event definition and the detectability adjustment. It can be stated firmly.

**The pipeline is reproducible and structurally coherent.** Every integrity check passes, each stage replays from a single command, and a cross-artifact validator checks that the published documents and the configurations say the same thing.

## B. What the models indicate, conditionally

**The probability levels.** 4.6 % at twelve months and 6.7 % at twenty-four, with a shoulder at 36-48 months, **under the retained event definition**. These figures vary by a factor of four and a half with the linkage rule; they must never be quoted alone.

**The framework effect.** Framework agreements show a successor earlier (HR 1.751), but the association is **partly detectability**: it falls to 1.617 once the buyer's publication volume is adjusted for. The direction holds; the magnitude does not.

**Technologies differ in timing** ($p = 0.036$ over five classes and 416 events), with `BUSINESS_SOFTWARE` fastest. Unadjusted, and carrying predicted labels.

## C. What remains uncertain

In order of importance: **linkage precision** is not independently validated (0.875 on eight links, CI [0.529; 0.978]; 0.700 in the model-assisted review, below the 0.80 target); **absolute levels** depend on the rule; the Cox model has **no out-of-time discrimination**; the annotation corpus has **no inter-annotator agreement**; and the reference's **recall** is not independent of the score it evaluates.

## D. What Gigalis can use now

Three uses are supported by the available evidence.

**Cohort-level monitoring, not contract-level.** The conditional probability table by age and segment (§ 5.3) identifies which *groups* of contracts are entering the window where a successor becomes likely. The 36-48 month shoulder is where attention pays off most, and CPV-35 is the fastest-moving segment. This is a rule for prioritising human attention, not a score.

**A usable business segmentation.** The eight technology classes give a reading of the regional portfolio that the four CPV divisions do not — with a mean CPV-segment purity of 0.34, each division mixes several business families. Used at aggregate level, with the confidence filter for demanding uses, it can map activity by technology.

**A reproducible measurement instrument.** The pipeline can be re-run each quarter on updated data. That may be the most durable contribution: Gigalis holds a documented instrument whose limits are known, rather than a number.

## E. What should not be operationalised

A **per-contract renewal score**, in any form: the out-of-time validation forbids it. An **individual ranking of contracts to watch** derived from the Cox model, for the same reason. Reading a **predicted technology class as an observed attribute** of a procurement, or its counts as market shares. A **trend forecast** by segment. And any **external communication** quoting the linkage precision as validated performance.

## F. What would add the most next

1. **An independent human review** of a blinded, re-drawn and pre-specified sample of accepted links. It is the only gate before any precision claim, and it is worth more than a fifth linkage method.
2. **A second technology annotation pass** by a different annotator, enabling a Cohen's κ. The learning curve is still rising at n = 500: more annotation beats a more complex model.
3. **Gigalis membership data** (member identity, adoption dates), without which the brief's causal question — does opening a pooled framework agreement change members' purchasing behaviour? — remains a staggered-adoption difference-in-differences design (Callaway and Sant'Anna, 2021), described but not estimated.
4. **A separate experiment on individual prediction**, if that line is prioritised: award-time variables only (object text, buyer history), a temporal validation declared in advance, a success criterion announced beforehand, and publication of the result even if negative. The trap is named: candidate-pool size would raise concordance by learning to predict *who is detectable*, not *who renews*.

---

# 10. Limitations

Ranked by what they can do to the results, not by category.

**1. Linkage validation is not independent.** *Mechanism*: the reference labels come from a language-model pass spot-checked by sampling, and the 60-pair review is likewise model-assisted. The stated precision could therefore be over- or under-estimated with nothing to signal it. *Effect*: any precision figure is provisional; the conservative review in fact gives 0.700, below target. *Mitigation in place*: the threshold was frozen before the reference was read, strict exact-successor accounting, intervals published everywhere, and an independent review protocol written with its sample prepared.

**2. Linkage errors distort both status and timing.** *Mechanism*: a false link creates an event and its date; a missed link creates censoring. *Effect*: absolute levels are not a lower bound, contrary to what a quick reading would suggest. *Mitigation*: four event arms, the borderline band, template-risk re-censoring, and an editorial rule — never quote a level without its range.

**3. Detectability is not uniform across buyers.** *Mechanism*: the score is a maximum over a block, and blocks vary greatly in size. *Effect*: prolific buyers produce more links, inflating any association with a variable correlated with publication volume — framework status first among them. *Mitigation*: the diagnostic is published (SMD +0.470) and a sensitivity model quantifies the attenuation.

**4. Technology labels are predictions.** *Mechanism*: macro-F1 of 0.744, with errors concentrated on definitional boundaries. *Effect*: every technology-level figure carries a measurement error that does not enter its intervals. *Mitigation*: two gates before any downstream use, an editorial rule forbidding treatment of a predicted class as an observed attribute, and exclusion of the fallback classes.

**5. The annotation corpus has no reliability measure.** *Mechanism*: a single pass, no annotator identifier, no second reading. *Effect*: no Cohen's κ; the brief's L2 design, which called for two annotators, is not met. *Mitigation*: three internal inconsistencies documented and left uncorrected as an empirical floor, annotation quotas flagged, `AI` declared unevaluable.

**6. The Cox model violates proportional hazards.** *Mechanism*: three covariates, award year severely so, structurally confounded with follow-up length. *Effect*: the coefficients are descriptive time averages. *Mitigation*: diagnostic published, interpretation restricted, no predictive use.

**7. The cohort is broader than its name.** *Mechanism*: an any-CPV-code rule at episode level. *Effect*: 30.9 % of episodes have a main CPV outside the perimeter. *Mitigation*: explicit phrasing, and measurement of the segment tie-break effect (94.7 % agreement).

**8. The time series are short and noisy.** *Mechanism*: 43 quarters, low counts, a documented completeness break in 2025. *Effect*: no trend survives correction; change-point detection proposes candidates without explaining them. *Mitigation*: break stability required under three penalties, multiplicity correction in both test families, and an explicit refusal to forecast.

**9. Geographic and temporal reach.** The results concern the Grand Ouest, 2015-2025, for contracts published in the BOAMP. They do not extend to France as a whole, to below-threshold purchases, or to procurements routed through a central purchasing body without their own publication.

---

# 11. Conclusion

The internship was meant to estimate the probability that a digital procurement generates an identifiable purchasing need within twelve to twenty-four months. It produced something else, and that difference is the main result.

**What was built.** A reproducible chain from 1,620,712 official BOAMP notices to a cohort of 3,800 awarded Grand Ouest digital procurements, with an episode reconstruction that prevents counting one procedure several times, an observable-successor variable whose quality is measured rather than assumed, and a cross-artifact validator that checks the published documents and the code agree.

**What was learnt about timing.** The probability that a contract shows an observable successor is estimated at 4.6 % within twelve months and 6.7 % within twenty-four, with a clear acceleration between 36 and 48 months consistent with the usual duration of framework agreements. Segments differ, CPV-35 being fastest, and framework agreements show a successor earlier — partly because they are more visible.

**What survived the robustness tests.** Absolute levels vary by a factor of four and a half with the event definition; relative comparisons hold, CPV-35 keeping its position under four independent perturbations. **This asymmetry — fragile levels, stable comparisons — is the central scientific message of the work**, and it is worth more than a single number presented as certain.

**What the text workstream added.** A supervised classifier on the contract object alone separates eleven business technology classes at an out-of-fold macro-F1 of 0.744, against 0.473 for the best CPV-based comparator on identical folds and budget: a paired difference of +0.271 whose interval excludes zero. Text therefore carries business information the administrative vocabulary does not. Its predictions, filtered through two explicit gates, reveal a difference in timing across technologies ($p = 0.036$).

**What the trend analysis showed.** Little, and that is a result: no segment slope survives correction for multiple testing. Three level shifts are stably dated, with no cause attributed to any of them.

**What individual prediction did not achieve.** The Cox model's out-of-time concordance is 0.479 — indistinguishable from chance. No per-contract score is therefore produced, and the table of twenty priority contracts foreseen in the brief was replaced by a cohort-level monitoring logic, owned as such. A model that does not discriminate out of time remains useful for describing a population; it is not useful for ranking a contract.

**What could not be undertaken.** The causal question — does opening a pooled framework agreement change the purchasing behaviour of Gigalis members? — requires internal membership data absent from the BOAMP; the identification design is described, no estimate is produced. Inter-annotator agreement on the technology corpus requires a second annotator. And external validation of the linkage requires an independent human review, whose protocol and sample are ready.

**The value for Gigalis.** It does not lie in a prediction. It lies in a measurement framework: an explicit definition of what is observed, a reproducible instrument to measure it, an honest quantification of what is solid and what is not, and a business segmentation that did not exist in the source data. Moving from a descriptive to a predictive logic presupposed a ground truth the source does not provide; this work shows precisely where it is missing, what would be needed to produce it, and what can already be decided without it.

---

# Bibliography

**Survival analysis**

1. Kaplan, E. L. and Meier, P. (1958). *Nonparametric Estimation from Incomplete Observations*. Journal of the American Statistical Association, 53(282), 457-481.
2. Cox, D. R. (1972). *Regression Models and Life-Tables*. Journal of the Royal Statistical Society, Series B, 34(2), 187-220.
3. Grambsch, P. M. and Therneau, T. M. (1994). *Proportional Hazards Tests and Diagnostics Based on Weighted Residuals*. Biometrika, 81(3), 515-526.
4. Therneau, T. M. and Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*. Springer.
5. Kalbfleisch, J. D. and Prentice, R. L. (2002). *The Statistical Analysis of Failure Time Data*, 2nd ed. Wiley.
6. Uno, H., Cai, T., Pencina, M. J., D'Agostino, R. B. and Wei, L. J. (2011). *On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data*. Statistics in Medicine, 30(10), 1105-1117.

**Record linkage**

7. Fellegi, I. P. and Sunter, A. B. (1969). *A Theory for Record Linkage*. Journal of the American Statistical Association, 64(328), 1183-1210.
8. Christen, P. and Goiser, K. (2007). *Quality and Complexity Measures for Data Linkage and Deduplication*. In: Quality Measures in Data Mining, Springer, 127-151.
9. Harron, K., Doidge, J., Knight, H., Gilbert, R., Goldstein, H., Cromwell, D. and van der Meulen, J. (2017). *A guide to evaluating linkage quality for the analysis of linked data*. International Journal of Epidemiology, 46(5), 1699-1710.
10. Doidge, J. C. and Harron, K. L. (2019). *Reflections on modern methods: linkage error bias*. International Journal of Epidemiology, 48(6), 2050-2060.

**Evaluation under class imbalance**

11. Davis, J. and Goadrich, M. (2006). *The Relationship Between Precision-Recall and ROC Curves*. ICML 2006, 233-240.
12. Saito, T. and Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLoS ONE, 10(3), e0118432.

**Multiple testing**

13. Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics, 6(2), 65-70.
14. Benjamini, Y. and Hochberg, Y. (1995). *Controlling the False Discovery Rate*. Journal of the Royal Statistical Society, Series B, 57(1), 289-300.

**Time series and change points**

15. Killick, R., Fearnhead, P. and Eckley, I. A. (2012). *Optimal Detection of Changepoints With a Linear Computational Cost*. Journal of the American Statistical Association, 107(500), 1590-1598.
16. Truong, C., Oudre, L. and Vayatis, N. (2020). *Selective review of offline change point detection methods*. Signal Processing, 167, 107299.
17. Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*. Econometrica, 57(2), 357-384.

**Natural language processing**

18. Blei, D. M., Ng, A. Y. and Jordan, M. I. (2003). *Latent Dirichlet Allocation*. Journal of Machine Learning Research, 3, 993-1022.
19. Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019, 4171-4186.
20. Martin, L. et al. (2020). *CamemBERT: a Tasty French Language Model*. ACL 2020, 7203-7219.

**Public procurement and data quality**

21. Fazekas, M. and Kocsis, G. (2020). *Uncovering High-Level Corruption: Cross-National Objective Corruption Risk Indicators Using Public Procurement Data*. Political Research Quarterly, 73(1), 155-177.
22. Potin, L., Labatut, V., Morand, P.-H. and Largeron, C. (2023). *FOPPA: A database of French Open Public Procurement Award notices*. Scientific Data, 10, 303.
23. Siciliani, L., Tanzi, G., Basile, P. and Lops, P. (2023). *Automatic CPV Code Classification for Italian Public Tenders*. CLiC-it 2023.

**Regulatory framework and official sources**

24. Official BOAMP API, DILA, published on data.gouv.fr.
25. Commission Regulation (EC) No 213/2008 on the Common Procurement Vocabulary (CPV).
26. Directive 2014/24/EU, Article 33 (framework agreements).
27. INSEE, definitions of SIREN and SIRET.

**Causal inference (perspective only)**

28. Angrist, J. D. and Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
29. Callaway, B. and Sant'Anna, P. H. C. (2021). *Difference-in-differences with multiple time periods*. Journal of Econometrics, 225(2), 200-230.

---

# Annexes

The annexes are accessory: the report reads without them. Each is referenced from the body.

**Annex A — Notation and formal definitions.** Units and notation ($n$, $i$, $j$, $u_i$, $v_j$, $J_i$, $T_{ij}$, $Y_i$, $\tau_i$); the blocking rule; the decision rule; the censoring definition; the procurement-family definition. *(Referenced from § 4.2, § 4.3, § 5.1, § 6.3.)*

**Annex B — Data quality.** Full completeness table by field and year; eleven integrity checks with their expected values; episode reconstruction method and edge counts; duration-completeness figure by year. *(Referenced from § 3.2, § 3.4.)*

**Annex C — Linkage evaluation.** Anchor-level and exact-successor confusion matrices for the four methods; pair-level ROC and precision-recall curves; the full 0.50-0.80 threshold sweep; the regional reference datasheet and its four limits; the buyer-blocking audit. *(Referenced from § 4.4, § 4.5, § 4.6.)*

**Annex D — Survival detail.** Kaplan-Meier curve with at-risk counts; curves by segment; the full Cox table with intervals; proportional-hazards diagnostics; parametric comparison and linearised goodness-of-fit; four sensitivity tables; the full selection diagnostic; the segment-stratified watch list. *(Referenced from § 5.2, § 5.4, § 5.5, § 5.7.)*

**Annex E — Text workstream detail.** Definitions of the eleven classes and the annotation rules; per-class metrics with support; confusion matrix; learning curve; the thirty-error triage; the register of specifications searched; the confidence reliability table and cutoff sweep. *(Referenced from § 6.2, § 6.6, § 6.7.)*

**Annex F — Trend detail.** The full signal matrix; PELT breaks under the three penalties; ADF and KPSS tests by segment; hidden Markov model parameters; the quarterly series. *(Referenced from § 8.3, § 8.4.)*

**Annex G — Reproducibility.** The single pipeline command; the ordered stage list with outputs; environment versions; SHA-256 digests of reference inputs and canonical outputs; the automated test inventory; artifact locations. *(Referenced from § 1, § 9.D.)*
