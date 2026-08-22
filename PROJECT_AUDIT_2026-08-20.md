# Independent Methodological Audit — BOAMP Observable Successor Procurement Study

Audit date: 2026-08-20
Auditor scope: full repository at `~/Desktop/project-gigalis`, working tree as of 2026-08-19 21:39
Method: documentation read in full; pipeline source read; **all headline results independently
recomputed from the materialised data** rather than taken from the reports.

> **Status as of 2026-08-22 — READ THIS FIRST.**
> This audit describes the working tree of **2026-08-19 21:39**. A corrective pass was run on
> 2026-08-20 (pipeline run 11:23–11:57). The following items in this document are **resolved and no
> longer describe the repository**: Critical **C1** (the calibrated-confidence narrative — §12 of the
> technology report now reports the deployed *raw* variant and the validator enforces the agreement),
> **I1** (candidate-pool detectability is now published as a diagnostic row and a Cox sensitivity
> column), **I3** (`technology_stages()` now runs before `evidence_stages()`, and
> `final_pipeline_manifest.json` records the technology stages), **I4** (Holm and Benjamini-Hochberg
> are now applied to the five CPV trend slopes and the executive summary reports CPV-48 as a nominal
> signal that does not survive correction), and Minor **M1-M6**. **I2** (the unrecorded
> candidate-export rule of the regional reference) cannot be repaired and is instead **disclosed** in
> `REGIONAL_BENCHMARK_REFERENCE.md`, `README.md`, `DATA_QUALITY_REPORT.md` and `QUALITY_EVIDENCE.md`.
> A later history check also overturned this audit's claim that the retained
> threshold predated all reference inspection. The current project correctly
> describes the regional split as **internal validation, not an untouched
> holdout**. The held-out claims in this historical audit are superseded.
> Sections **M** (what not to change), **N** and **O** remain current.
> Do not quote this document's defect list as the current state of the project.

---

## A. Executive Assessment

**Status: 🟡 Yellow — fundamentally sound, with a small number of targeted corrections needed.**

This is a substantially better-than-average piece of applied statistical work. The core claims are
reproducible, the epistemic discipline is unusually strong, and most of the things an auditor
normally has to argue for are already documented as limitations by the project itself.

What I verified by independent recomputation (not by reading):

| Claim | Source of truth | My recomputation | Match |
|---|---|---|---|
| KM P(successor ≤ 12m) = 4.621% | `survival_km_horizons.csv` | 0.046209 from `survival_dataset.parquet` | exact |
| KM P(successor ≤ 24m) = 6.733% | same | 0.067331 | exact |
| Median observed successor gap 31.82 months | `README.md` | 31.82 | exact |
| Event rate 0.1432 (544/3800) | `survival_analysis_summary.json` | 0.1431578947 | exact |
| Cox HR framework 1.751, CPV-35 1.553, award-year 1.107 | `survival_cox_results.csv` | 1.7508 / 1.5534 / 1.1069 (Breslow PHReg) | exact to 4 dp |
| Technology out-of-fold macro-F1 0.7442 | `bootstrap_macro_f1_ci.csv` | 0.7442 from `oof_predictions.csv` | exact |
| All 11 per-class precision/recall/F1 | `per_class_metrics.csv` | reproduced from `oof_predictions.csv` | exact |
| Group-aware folds contain no split family | §4 of the technology report | 0 of 459 families span >1 fold | confirmed |
| Borderline band removes 280 episodes / 133 events | `survival_borderline_link_sensitivity.csv` | 280 anchors with best text ∈ [0.65, 0.75], 133 events | exact |
| 127 accepted links share a successor with another anchor | `candidate_generation_audit.json` | 544 links / 461 distinct successors ⇒ 127 | exact |

**Are the methods appropriate?** Yes, and at the right level of sophistication. Nothing here is
over-engineered, and the two places where a fancier method was declined (CamemBERT; a
duration-conditioned linkage arm) were declined for stated, defensible, pre-registered reasons.

**Are the results credible?** The comparative results are. The absolute survival probabilities are
credible *as linkage-conditioned quantities*, which is exactly how the project presents them.

**Are the conclusions supported?** Almost entirely. The project is, if anything, more conservative
than it needs to be. Three exceptions are listed below.

**What threatens validity?** Nothing that invalidates the study. The one item I would call a genuine
defect rather than a limitation is the confidence-calibration narrative in the technology report
(§K, Critical-1) — it is a documentation defect about a frozen artifact, and it contradicts the
project's own log file and config.

**Should you keep modifying it or stop?** Stop, apart from a short fix list. The linkage, survival
and trend components should be treated as closed. The corrections below are hours of work, not days,
and none of them requires re-estimating anything.

---

## B. Reconstructed Project Pipeline

```
Official BOAMP API, 2015-01-01 → 2025-12-31
  → schema-aware notice standardisation           1,620,712 notices, 0 duplicate idweb
  → procurement episode reconstruction            1,103,632 episodes (union-find over
                                                   contractFolderID, explicit linked notices,
                                                   and same-buyer procedure references ≤730 d)
  → Grand Ouest subset                            144,269 episodes
  → digital CPV filter (32/35/48/72, any code)    7,376
  → require an award notice                       3,826
  → require a resolvable award date               3,800  ← STUDY COHORT
  → same-buyer candidate blocking, 90–2,920 d     763,417 pairs over 3,520 anchors
  → four linkage methods scored on the same pairs
  → frozen primary rule M_B_text_ranking @ 0.70   544 accepted links
  → right-censored survival dataset at 2025-12-31 3,800 rows, 544 events, 3,256 censored
  → KM / log-rank / Cox / PH diagnostics / parametric comparison
     + 4 linkage arms + borderline band + template-risk re-censoring
  → quarterly CPV trend series: PELT + ADF/KPSS + 3-state HMM
  → [enrichment layer] 500 annotated notices → 11-class TF-IDF + balanced logistic regression
     → predicted class for all 3,800 episodes → gated technology-level survival & trend
```

**Unit of analysis:** the reconstructed procurement *episode*, not the notice. Correct choice — BOAMP
republishes one procurement many times, and notice-level analysis would double-count.

**Event:** "observable successor procurement", explicitly not a legal renewal. The project is
consistent about this in every document I read, including the abstract-level claims.

**Time zero:** award date. **Censoring:** administrative, at 2025-12-31.

This reconstruction matches the code, not just the prose. I traced each arrow to the script that
produces it.

---

## C. File and Version Consistency

### Which files are current

The active pipeline is `scripts/run_final_pipeline.py` → `boamp_pipeline/*` → `data/processed/boamp/`.
`notebooks/10`–`15` are all fully executed with zero error outputs and embedded asserts (I checked
every code cell's `execution_count` and `outputs`). `notebooks/archive/` and `archive/` are isolated
and unreferenced by the active path; `pytest.ini` correctly restricts collection to `tests/`.

`git status` is clean apart from one file, and `git diff` on that file shows **only a timestamp line
changed**. That is a strong reproducibility signal: the last technology rebuild reproduced every
number bit-for-bit.

### The one real staleness mechanism (Important)

There is a stage-ordering defect in `scripts/run_final_pipeline.py::main`:

```python
for stage in pipeline_stages():   ...
for stage in evidence_stages():   ...   # ← includes reader_artifact_refresh,
                                        #   readiness_report_data, canonical_state_validation
for stage in technology_stages(): ...   # ← runs AFTER
```

`reader_artifact_refresh` regenerates `README.md`, `EXECUTIVE_SUMMARY.md`, `FINAL_PIPELINE.md`,
`REGIONAL_BENCHMARK_REFERENCE.md` and `reports/boamp_methodology_chapter.tex` — **all of which quote
technology-layer numbers** (0.744, 0.473, 0.271, 235, 6.2%, p = 0.036 vs 0.0001, purity 0.34). It
runs *before* the technology layer is rebuilt. `canonical_state_validation` has the same problem.

The filesystem shows the consequence:

| Artifact | Timestamp |
|---|---|
| `README.md`, `EXECUTIVE_SUMMARY.md`, `boamp_methodology_chapter.tex` | 2026-08-19 **21:15:39** |
| `notebooks/15` executed | 2026-08-19 **21:20** |
| `canonical_state_validation.json` | 2026-08-19 **21:20:08** |
| technology outputs + `TECHNOLOGY_TAXONOMY_REPORT.md` | 2026-08-19 **21:39:5x** |

So the validation gate certified a state that was subsequently overwritten, and every reader-facing
document quotes technology numbers from one run earlier. **Nothing is currently wrong** — the diff
proves the rebuild changed nothing — but the guarantee the pipeline advertises does not hold.

Separately, `final_pipeline_manifest.json` is dated **2026-08-16T16:29:37** and its `stage_status`
block lists **no technology stages at all**. The last full `run_final_pipeline.py` invocation
predates the technology layer; the layer has since been run standalone. The manifest is therefore not
a truthful record of how the current outputs were produced.

### Contradictory versions

None found in the data. The `data/processed/` tree is internally consistent at every grain I checked
(see §J).

---

## D. Problem Definition Assessment

**Correct.** The research question — "how long until an awarded digital procurement in Grand Ouest is
followed by an observable successor procurement in BOAMP?" — is well-posed, matches the unit of
analysis, and is answerable with the data. The distinction between *observable successor* and *legal
renewal* is stated in the README, the executive summary, the data-quality report, the survival
report, the manifest, and the code comments. This is the single hardest thing to get right in a study
like this, and it is right.

The epistemic register is also correct throughout: the project claims **association and description**,
never causation, and explicitly says the Cox coefficients are "descriptive time-averaged
associations, not causal effects" given the PH violations. The trend section refuses to attribute
breaks to COVID or policy. The technology section refuses to call predicted class shares market
shares.

**Questionable — cohort scope wording (Minor, documentation only).** The cohort rule is documented as
"Grand Ouest, CPV divisions 32/35/48/72". In code (`build_survival_cohort.py:125`,
`linkage.is_digital`) it is an **any-code** rule at episode level: an episode qualifies if *any* of
its CPV codes (median 2 per episode, max 71) is in the digital set. I measured the consequence:

- **1,176 of 3,800 (30.9%) cohort episodes have a *main* CPV outside 32/35/48/72.**
- The three largest cohort episodes by notice count are, by their own text, work clothing/PPE, a
  building rehabilitation, and office furniture — multi-lot tenders that included one digital lot.

So the cohort is really "Grand Ouest procurement episodes containing at least one digital lot". That
is a *defensible* definition, and the technology classifier's `OTHER` class (173 predicted episodes,
"carries a digital CPV without being a technology procurement") already partly measures it. But it
should be stated in one sentence, because a reader will otherwise take "3,800 digital procurement
episodes" more literally than the rule warrants.

**I checked whether this actually distorts the results, and it does not much** — see §F.

---

## E. Data Assessment

### Quality and integrity — strong

Every structural check I could re-run held: 1,620,712 unique notice IDs with zero duplicates, all
notices assigned to exactly one episode, zero buyer-conflict episodes, zero impossible chronologies,
zero negative durations, no accepted link with conflicting validated SIRENs. The selection funnel
(144,269 → 7,376 → 3,826 → 3,800) is fully reconciled and the 26 dropped rows are attributed.

### Missingness — honestly handled

- **Validated SIREN missing for 66.3%.** Preserved as a name-only buyer key rather than imputed. The
  `compatible_buyer_identifiers` rule is well designed: "both unknown" is explicitly *not* treated as
  evidence of identity. Municipal prefixes are stripped for blocking, but intercommunal legal forms
  ("Rennes Métropole" vs "Ville de Rennes") are deliberately preserved. This is the right call.
- **Reliable duration missing for 74.9%.** No imputation. The refusal is evidenced, not asserted:
  completeness is 11.8% in 2023 and 84.4% in 2025 (`data_quality_profile.json`), so missingness is
  clearly not exchangeable across years. **Keep this decision.**
- Amount container missing 15.7% and explicitly excluded from any monetary claim. Correct.

### Selection and detectability bias — one gap (Important)

The published diagnostic (`survival_selection_diagnostic.csv`) reports standardised mean differences
between linked and censored episodes for seven variables, the largest being `text_length_chars` at
+0.262. **It omits the variable with the largest imbalance of all.**

I computed the number of exposed candidates per anchor (the "block size", i.e. how prolific that
buyer is in Grand Ouest BOAMP) and compared linked vs censored episodes:

| Variable | SMD (linked vs censored) |
|---|---:|
| **log(candidate pool size)** | **+0.470** |
| candidate pool size (raw) | +0.285 |
| text_length_chars *(largest currently published)* | +0.262 |
| framework_flag | +0.187 |
| administrative_followup_months | +0.146 |
| award_year | −0.137 |

Acceptance rate rises monotonically with pool size:

| Pool-size sextile | median candidates | accept rate |
|---|---:|---:|
| 1–15 | 7 | 9.7% |
| 16–41 | 26 | 14.6% |
| 42–89 | 61 | 14.7% |
| 90–202 | 132 | 14.3% |
| 203–424 | 299 | 18.6% |
| 425–1713 | 654 | **21.1%** |

This is mechanically expected — `M_B` takes the maximum text score over the block, and the maximum of
more draws is larger — but it is a real detectability confound, and it is not in the diagnostic. I
tested whether it changes the headline conclusions (see §F/§I). Short answer: **CPV-35 survives
cleanly; the framework hazard ratio is partly detectability.**

### The regional reference — a documented gap plus an undocumented one (Important)

The documented limitations are excellent and I have nothing to add to them: single LLM research pass,
spot-checked rather than verified anchor-by-anchor, corpus-relative negatives, no per-anchor evidence
trail, design weights approximate, n small.

The **undocumented** limitation is how the ~25 reviewed candidates per anchor were chosen. The
reference file records `broad_candidate_pool_n` (mean 357, max 3,258) and
`review_candidates_exported_n` (capped at exactly 25), so a selection rule existed — but that rule is
recorded nowhere in the repository, and `build_regional_benchmark.py` explicitly says the reference
"is not produced here".

This matters, so I tested it. For the 18 positive anchors in the locked split, I found where each
reviewed successor sits in the *production* candidate ranking:

| Ranked by | reviewed successors found in pool | rank ≤ 25 | max rank |
|---|---:|---:|---:|
| `linkage_score` | 16 / 18 | 15 | 27 |
| `text_component` (the M_B score) | 16 / 18 | **16 / 16** | **13** |

Every retrievable reviewed successor is in the **top 13** by the very score being evaluated, with 12
of 16 at rank 1, out of pools of 35–681 candidates. Under genuinely independent labelling that is
extraordinarily unlikely. The pattern is what you would see if the 25 exported candidates were
ranked by text similarity.

**I cannot verify the export rule from the available evidence, and I am not asserting it was
score-ranked.** But if it was, then:

- The reference's **recall** figures (0.389 for M_B) and the **0.913 candidate-generation ceiling**
  are partly circular — the reviewer could only name a successor the scorer had already surfaced.
- The reference's **precision** figure (0.875) is *not* affected. A false positive is a false
  positive regardless of how the candidate list was built.
- `M_D_fellegi_sunter`, which ranks on a different scale, would be structurally disadvantaged.

The README's phrase "independent of every method it scores" is defensible for the *labels* and
overstated for the *candidate pool*.

### Class imbalance

Handled correctly in both places it arises. Linkage: 16 positive pairs among 20,917 → the project
leads with precision-recall rather than ROC and cites Davis & Goadrich and Saito & Rehmsmeier for
exactly that reason. Technology: `class_weight='balanced'`, macro-F1 as the headline, per-class
support published, and `AI` (n=7) explicitly marked uninterpretable rather than quietly averaged in.

### Leakage

I looked hard and found none in the technology layer: 0 of 459 procurement families span more than
one fold; the vectoriser is inside the `Pipeline` and fitted per training fold; hyperparameters come
from a grouped *inner* CV nested in each outer training fold; CPV, buyer, date, region, amount and
every linkage variable are excluded by design; the temporal split moves the 4 boundary-straddling
families into *training*, which costs test observations rather than flattering the result. This is
better leakage hygiene than most published work.

---

## F. Methodology Assessment

| Method | Purpose | Appropriate? | Main concern | Recommendation |
|---|---|---|---|---|
| Union-find episode reconstruction (contractFolderID → explicit links → same-buyer reference ≤730 d) | Collapse notice duplication before linkage | **Yes** | Pairwise chaining on shared references is transitively unbounded (max episode span 3,342 d repo-wide) | **Keep unchanged.** I checked the cohort specifically: median span 126 d, only 28/3,800 span >730 d, only **21/3,800** use the risky `exact_reference_same_buyer` route, and only 37 have any notice >90 d after award. The risk is real in principle and negligible here. |
| Same-buyer + 90–2,920 day blocking | Candidate generation | **Yes** | Recall ceiling 0.913 | **Keep.** The 90-day floor is empirically set (min confirmed gap 139 d) and documented against a named pathology it fixed. The 8-year ceiling excludes no confirmed link. |
| `M_B_text_ranking` = max(word TF-IDF cos, char_wb 3–5 TF-IDF cos), top-1 per anchor, accept ≥0.70 | Primary event definition | **Yes, as a precision-first baseline** | Vectoriser is fitted *per anchor block*, so the score is block-relative and a fixed 0.70 does not mean the same thing across anchors; this is the mechanism behind the pool-size confound in §E | **Keep the rule.** Add pool size to the detectability diagnostic and to one Cox sensitivity column. Do not re-tune. |
| `M_C_weighted_gated` (buyer 0.50 / text 0.25 / CPV 0.20 / time 0.05, renormalised over observed evidence) | High-recall contrast arm | **Yes** | — | **Keep.** The renormalise-over-available-evidence design (`weighted_score`) is a genuinely good idea: missing CPV is dropped from numerator *and* denominator rather than scored 0. |
| `M_A_deterministic` | SIREN + CPV + text floor comparator | Yes | Its `threshold` argument is accepted and **never used**, yet it is displayed as "threshold 70.0" in `QUALITY_EVIDENCE.md` and `REGIONAL_BENCHMARK_REFERENCE.md` | **Minor fix:** show `n/a`. |
| `M_D_fellegi_sunter` | Classical probabilistic-linkage comparator | Yes | Posterior tops out near 0.73 (unsupervised mixture prior uncalibrated); comprehensively beaten | **Keep as one comparator row.** Do not invest further. |
| Kaplan–Meier + log-rank | Primary survival estimator | **Yes** | — | **Keep.** Reproduced exactly. |
| Cox PH (segment, region, framework, SIREN validity, award year) | Comparative associations | **Yes, as descriptive** | `award_year_centered` PH test statistic **70.7, p = 4×10⁻¹⁷** — a severe violation, and award year is structurally confounded with follow-up length | **Keep, with the existing caveat.** The project already declares these time-averaged and not causal. Consider stratifying on award year rather than adjusting — but this is optional, not required. |
| Parametric AFT comparison (GeneralizedGamma lowest AIC) | Family selection / extrapolation instrument | Yes | — | **Keep, and keep it out of the headline.** The decision to read the 12/24-month numbers off KM rather than the parametric fit is correct and well argued. |
| PELT change-point + penalty sensitivity | Trend break candidates | **Yes** | — | **Keep.** Requiring a break to survive λ ∈ {0.5, 1, 2} within one quarter is a sound stability rule. |
| ADF + KPSS | Stationarity description | Yes | Reported as complementary, correctly | **Keep.** |
| 3-state Gaussian HMM on ΔN | Regime description | Borderline — but honestly framed | Fitted on 43 noisy low-count quarterly differences; the report itself says `plateau` is "a data-driven middle tier rather than a change centered exactly at zero" | **Keep, or drop.** It is the one component that adds the least per unit of complexity. If you want to simplify, this is the candidate. |
| TF-IDF unigrams + class-weighted multinomial logistic regression, nested grouped 3-fold CV | Technology taxonomy | **Yes** | — | **Keep.** Reproduced exactly. |
| Family-level bootstrap for the paired CPV-vs-text difference | Uncertainty at the leakage unit | **Yes — this is the right unit** | — | **Keep.** Resampling families rather than notices is exactly correct and is the kind of thing most projects get wrong. |
| CamemBERT **not** run | — | **Correct** | — | **Do not run it.** The gate was pre-registered (macro-F1 < 0.55), the realised 0.7414 fails it decisively, and the learning-curve diagnosis is *high variance*, for which added capacity is the wrong medicine. |
| Gate A (classifier evidence) + Gate B (statistical support) before technology-level enrichment | Prevent mixture-class artefacts | **Yes, and unusually well motivated** | — | **Keep.** That Gate A *weakens* the headline (p 0.000119 → 0.0363) and was kept anyway is the strongest single piece of evidence in this repository that the analysis was not tuned toward a result. |

---

## G. Validation Assessment

**Linkage.** The pilot (16 usable anchors) / locked (72 usable anchors) split is recorded *in the
reference file itself* under the reference's own `benchmark_split` column, dated 2026-08-11 — it is
not a post-hoc split. `build_regional_benchmark.py` only renames the strata and states explicitly
that "no anchor moves between them". A later history check found that designated
reference evidence informed the retained operating policy. The current project
therefore treats both strata as internal validation. The 0.70 policy remains a
frozen post-development convention; replacing it requires fresh independent evidence.

**Technology.** Group-aware nested CV, verified clean. The temporal split (2015–2022 → 2023–2025)
with boundary families forced into training is a genuine out-of-time check. The learning curve
subsamples *families*, not notices. The specification register (`specification_register.csv`) lists
every model searched with its grid size, so the search budget is auditable and the winner is not
being reported out of a hidden set.

**Survival.** The Cox temporal validation is honest to the point of being self-damaging: train on
2015–2021, score 2022–2024 with no refitting, C-index **0.479** — below chance. The report says
plainly "That is the result, not a prompt to retune." I agree with both the number and the refusal.

**Weaknesses of the validation design:**

1. The 60-pair "independent" review is model-assisted, so it cannot close the validation gate. The
   project says so itself, in the file name of the protocol and in its own decision line. Correct.
2. Reference negatives are corpus-relative, making FPR = 0.000 an upper-bound-flavoured statistic
   rather than a population rate. Documented.
3. No pair-level independence adjustment: 20,917 pairs come from 69 anchors, so the pair-level
   ROC-AUC (0.9915) and average precision (0.4777) treat non-independent observations as independent.
   The project already flags these curves as "rank scores and do not measure accuracy", which mostly
   covers it.

---

## H. Metrics Assessment

| Metric | Current value | What it means here | Appropriate? | Concern |
|---|---:|---|---|---|
| Precision@1 (M_B @ 0.70, locked) | 0.875 (95% CI 0.529–0.978) | 7 of 8 accepted links were the reviewed successor | **Yes** — the right primary metric for a precision-first rule | n = 8. The interval spans from "coin flip" to "near perfect". Never quote 0.875 without it. |
| Recall@1 | 0.389 (0.203–0.614) | 7 of 18 positive anchors linked | Yes | Partly circular if the reference candidate export was score-ranked (§E) |
| FPR on reviewed negatives | 0.000 on 54 anchors | No accepted link on a reviewed-negative anchor | Yes | Corpus-relative; 0/54 still carries a wide interval |
| Candidate recall ceiling | 0.913 (21/23) | Blocking-stage bound before any scoring | Yes — good practice to separate this | Same circularity caveat |
| Pair ROC-AUC / average precision | 0.9915 / 0.4777 | Score ranking over exposed pairs | AP yes, ROC secondary — correctly ordered | Non-independent pairs; correctly de-emphasised |
| Event rate | 0.1432 (544/3,800) | Linkage-conditioned | Yes | See naming issue below |
| KM P(≤12m) / P(≤24m) | 4.621% / 6.733% | Probability an observable successor appears | Yes | Linkage-sensitive: 296–1,332 events across arms |
| Log-rank across CPV segments | χ² = 23.45, p = 3.26e-05 | Segments differ in successor timing | Yes | Well powered |
| Cox HRs | framework 1.751, CPV-35 1.553 | Time-averaged associations | Yes, as descriptive | PH violated for framework, SIREN, award year |
| Cox C-index (out-of-time) | 0.479 (2022–24), 0.518 (2022–25) | No usable individual discrimination | Yes | Reported honestly; nothing operational rests on it |
| Conditional P(successor in next h months) | 0.021–0.097 depending on age | Operational deliverable | Yes | Non-monotone in age (rises into a 36–48 month shoulder); the report says so |
| Technology macro-F1 (grouped OOF) | 0.7442 (family-bootstrap 0.6819–0.7905) | 11-class separation from object text | **Yes** — macro is right for 11 imbalanced classes | — |
| Paired difference vs CPV benchmark | 0.2711 (0.2009–0.3403), excludes 0 | Text carries what CPV cannot | **Yes** — paired, same folds, same search budget | The most solid single result in the project |
| Technology log-rank (5 gated classes) | p = 0.0363 on 416 events | Timing differs across technologies | Yes | Unadjusted for buyer type, size, procedure — stated |
| Confidence / reliability table | ECE 0.2097 quoted | Reading guide for the operational cutoff | **No — wrong variant is published** | See §K Critical-1 |

**Naming issue (Minor).** `linkage_application_summary.json` reports `cohort_link_rate: 0.1545`,
which is 544/**3,520** (anchors *with candidates*), not 544/3,800. The reports correctly use 0.1432
everywhere, so nothing downstream is wrong, but the field name invites a future misquote. Same for
the strict arm (0.0841 vs 0.0779).

---

## I. Results Assessment

### Strong / reliable findings — lead with these

1. **Procurement text carries the business technology class that CPV does not.** Paired difference
   0.2711 macro-F1, 95% family-bootstrap CI [0.2009, 0.3403], excluding zero, on identical folds with
   an identical search budget for both sides. I reproduced the point estimate exactly. This is
   methodologically the cleanest result in the project and the interval is estimated at the correct
   unit. Emphasise it more than you currently do.

2. **CPV segments differ in observable successor timing** (log-rank χ² = 23.45, p = 3.3e-05), with
   **CPV-35 the highest** (HR 1.553 vs CPV-32, p = 0.00038). I stress-tested this against the
   detectability confound in §E — adding log(candidate pool size) to the Cox model moves CPV-35 only
   from 1.553 to **1.512** (p = 0.00085). It also survives the borderline band (1.78) and
   template-risk re-censoring (1.541). **This finding is robust and you can state it more firmly
   than you do.**

3. **The classifier's confidence score ranks well.** Out-of-fold accuracy rises monotonically across
   confidence bins in both variants. The ranking claim holds regardless of the calibration confusion.

4. **Gate A costs significance rather than buying it** (p 0.000119 → 0.0363 when residual buckets are
   dropped). Publishing this is strong evidence of methodological good faith and is worth keeping
   prominent.

5. **Structural data integrity.** Every integrity check reproduced.

### Findings that need cautious interpretation

6. **Framework agreements have a higher observable-successor hazard (HR 1.751).** Direction survives
   every check the project runs. But it is **partly a detectability artefact**: adding
   log(candidate pool size) attenuates it from 1.751 to **1.617**, roughly a quarter of the log
   hazard ratio, while the pool-size term itself is highly significant (HR 1.184, p = 6.1e-09).
   Framework buyers publish more, and publishing more mechanically raises the chance that the
   max-over-block text score clears 0.70. The association is real but smaller than reported, and the
   report should say the detectability channel is *partly* responsible rather than only that
   template boilerplate is not responsible.

7. **Absolute survival probabilities** (4.6% / 6.7%). Correct as computed, but conditional on the
   linkage policy — 296 to 1,332 events across arms. The project already refuses to call them lower
   bounds, correctly, since missed successors and false links push in opposite directions.

8. **The 36–48 month renewal shoulder.** S(36) = 0.913 → S(48) = 0.845 is a big drop, driving
   P(successor within 12m | age 36) = 7.5% against ~2% at other ages. At 48 months the risk set is
   restricted to pre-2022 awards, so composition changes across the curve. Plausible as a real
   four-year framework effect, but it is the one part of the curve I would not quote without the
   at-risk counts beside it.

### Findings that may be unreliable

9. **"CPV-48 shows a statistically distinguishable recent decline."** This appears in
   `EXECUTIVE_SUMMARY.md` under **"What Works"**. It rests on a 12-quarter OLS slope with
   **p = 0.032, uncorrected, across 5 segments tested simultaneously**. The project's own technology
   trend section applies Holm and Benjamini–Hochberg to exactly the same situation (5 simultaneous
   slope tests). Applying the same standard here gives Holm p ≈ 0.16, which does not clear the
   project's own pre-declared α = 0.10. `TREND_ANALYSIS_REPORT.md` does disclose that the p-values
   are uncorrected — but the executive summary then promotes the finding anyway. **This is the one
   place where the project claims more than its own methodology supports.**

10. **`AI` results.** 7 annotated notices, 6 predicted episodes, 0 events. Correctly marked
    uninterpretable everywhere. No action.

11. **`MIXED` class** (F1 0.5417) and `OTHER_DIGITAL` (0.6667). Correctly gated out of enrichment.

---

## J. Internal Consistency Check

I cross-checked every headline number that appears in more than one place. **The arithmetic is
consistent throughout** — 544/3,800 = 0.1432; 7/8 = 0.875; 7/18 = 0.389; 21/23 = 0.913; 351/538 =
0.6524; 235/3,800 = 0.0618; 14/23 = 0.6087; 544 links − 461 distinct successors = 127; 60+140+38+106+72
= 416 events; +MIXED/OTHER_DIGITAL/OTHER = 518. Gate B thresholds in
`technology_evidence_summary.json` (≥100 episodes, ≥20 events) exactly explain every yes/no in the
published gate table. All of it holds.

**Material inconsistencies found — three:**

| # | Where | Contradiction |
|---|---|---|
| 1 | `TECHNOLOGY_TAXONOMY_REPORT.md` §12 and `notebooks/15` cells 1 & 45 vs `final_model_config.json`, `logs/build_technology_taxonomy.log`, `episode_technology_predictions.csv` | The report and notebook say confidence is Platt-scaled and "Calibration was adopted under a pre-specified rule". The config says `"adopted": false`, `"deployed_variant": "raw"`; the log says `Confidence variant adopted: raw`; every deployed row carries `confidence_type = uncalibrated_class_score`. See §K Critical-1. |
| 2 | Same section | The reliability table shown is the **calibrated** variant (n = 45/73/86/87/115/77/17) while the accompanying cutoff statistics (`0.9556` accuracy on `9%` of notices) come from the **raw** variant (n = 26/16/3 above 0.70 = 45 = 9%). Two variants are interleaved in one narrative without labels. |
| 3 | Same section | "No calibrated prediction reaches `0.90`, so cutoffs above `0.80` are not usable." The deployed predictions include **28 episodes at confidence ≥ 0.90** (max 0.9707), and `confidence_cutoff_sweep.csv` publishes a 0.9 row. The statement is false for the deployed score. |

**Non-material inconsistencies:**

- "Observed accuracy rises... from `0.27` in the lowest bin" — `0.27` is a hardcoded literal in
  `technology_evidence.py:1478` and matches neither table (raw 0.5232, calibrated 0.1778).
- Trend quarter counts: CPV series uses **43** quarters (2015Q2–2025Q4, partial Q1 excluded);
  technology series uses **44**. Neither is wrong; the difference is unexplained.
- `boamp_methodology_chapter.tex` says the representation is "TF–IDF word unigrams and bigrams" in
  one sentence and "word unigrams" in the next. Every fold selected unigrams alone.
- `INTERNSHIP_GUIDE_COMPLIANCE.md` header says "Assessment date: 2026-08-14" but the body describes
  work done on 2026-08-19.
- `run_final_pipeline.py --with-notebooks` help text says "notebooks 10-14"; notebook 15 is included.
- `linkage.py` docstring cites a legacy path `data/processed/boamp_grand_ouest/high_precision_linkage_config.json`.
- `INTERNSHIP_GUIDE_COMPLIANCE.md` L2 row says "one calibrated prediction per cohort episode" — same
  error as Critical-1.

**A coincidence worth a footnote, not a fix:** the number 280 appears twice with different meanings —
280 anchors have *no* candidates, and 280 anchors have a best score in the borderline band
[0.65, 0.75]. I verified these are two disjoint-in-meaning sets and the borderline analysis removes
the right 280 (with exactly 133 events, as reported). One clarifying clause would prevent a reader
concluding the borderline analysis simply dropped the no-candidate anchors.

**Not found:** no stale figure, no notebook that fails to execute, no metric that changes between
documents, no percentage that fails to sum, no confidence interval inconsistent with its point
estimate, no result traceable to a retired pipeline version.

---

## K. Issues by Severity

### 🔴 Critical (1)

**C1 — The technology report and notebook 15 describe a calibrated confidence score that is not what
was deployed.**

Evidence chain:
- `logs/build_technology_taxonomy.log`: `Confidence variant adopted: raw (expected calibration error 0.3502 -> 0.2097)`
- `final_model_config.json`: `"adopted": false`, `"deployed_variant": "raw"`,
  `"source": "raw multinomial logistic regression predict_proba (uncalibrated: the calibration rule was not met)"`
- `episode_technology_predictions.csv`: all 3,800 rows carry `confidence_type = uncalibrated_class_score`; max confidence 0.9707; 28 rows ≥ 0.90.
- The rule itself rejected calibration correctly: ECE improved 0.1405 (≥ 0.02 ✓) but macro-F1 cost was 0.0364, exceeding the 0.02 budget.
- Root cause: `boamp_pipeline/technology_evidence.py:1073` hardcodes
  `reliability["variant"] == "calibrated"`, and the prose at lines ~1467–1510 hardcodes the
  "Calibration was adopted" narrative, the "no calibrated prediction reaches 0.90" claim, and the
  literal `0.27`. The *pipeline* handles both branches correctly (lines 284, 412–416); only the
  report generator does not.

Why Critical rather than Important: §12 is the published **reading guide for an operational
deliverable**. A user applying the 0.70 worklist cutoff would calibrate their expectations against a
table for a variant that does not exist in the shipped predictions, and would believe cutoffs above
0.80 are unusable when 123 episodes clear 0.80 and 28 clear 0.90. It does not invalidate the
classifier's macro-F1, per-class metrics, or the CPV comparison — all of which I reproduced exactly.

### 🟠 Important (4)

**I1 — Candidate-pool size is the largest linked-vs-censored imbalance and is neither published nor
adjusted for.** SMD +0.470 on the log scale, above every variable currently in
`survival_selection_diagnostic.csv`. Adding it to the Cox model attenuates the framework HR
1.751 → 1.617; the pool-size term is HR 1.184 (p = 6.1e-09). CPV-35 is essentially unmoved
(1.553 → 1.512), which is a *reassuring* result worth publishing.

**I2 — The regional reference's candidate-export rule is unrecorded, and the rank evidence is
consistent with a text-score-ranked export.** All 16 retrievable locked-split successors sit in the
top 13 by `text_component`. If score-ranked, the recall figure and the 0.913 ceiling are partly
circular. Precision is unaffected. I cannot confirm the rule from the repository.

**I3 — Pipeline stage ordering makes the validation gate and every reader artifact lag the technology
layer by one run.** `technology_stages()` runs after `evidence_stages()` (which contains
`reader_artifact_refresh`, `readiness_report_data`, `canonical_state_validation`). Currently
harmless — the diff proves it — but the guarantee is not real. `final_pipeline_manifest.json` is also
three days stale and lists no technology stages.

**I4 — Multiplicity standard is applied inconsistently, and the executive summary promotes the
uncorrected result.** Technology trends: 5 tests, Holm + BH applied, correctly reported as null. CPV
trends: 5 tests, no correction, and CPV-48 (raw p = 0.032, Holm ≈ 0.16) appears under "What Works".

### 🟡 Minor (7)

**M1** — Cohort scope: "CPV divisions 32/35/48/72" is an any-code episode rule; 30.9% have a main CPV
outside the set. One sentence of documentation.

**M2** — `digital_segment` is assigned as the *lowest-numbered* digital division present, an
arbitrary tie-break affecting 10.8% of episodes. **I measured the impact and it is small**: among the
2,624 episodes whose main CPV is digital, the assigned segment matches the main division 94.7% of the
time, and event rates by assigned vs main-CPV segment are nearly identical (CPV-35: 0.1886 vs 0.1863;
CPV-32: 0.1207 vs 0.1241). Document the rule; do not change it.

**M3** — `reference_conflict_episodes = 3,274` and `suspicious_review_cases = 3,344` are computed and
exported but absent from `DATA_QUALITY_REPORT.md`'s integrity table, which shows only the zero-valued
checks. Add two rows so the table doesn't read as "all clear".

**M4** — `cohort_link_rate` in `linkage_application_summary.json` divides by 3,520, not 3,800. Rename
to `link_rate_among_anchors_with_candidates`.

**M5** — `M_A_deterministic` is displayed with "threshold 70.0" though the function ignores its
threshold argument. Show `n/a`.

**M6** — The two unrelated 280s (§J) deserve one clarifying clause in the survival report.

**M7** — Assorted doc drift: chapter unigram/bigram sentence, compliance-file date header,
`--with-notebooks` help text, legacy path in `linkage.py` docstring, 43-vs-44 quarters.

### ⚪ Optional (3)

**O1** — Cox `award_year_centered` PH violation is extreme (statistic 70.7). Stratifying on award-year
band instead of adjusting would be cleaner. Coefficient interpretation would barely change.
**O2** — The 3-state HMM on 43 noisy quarterly differences is the lowest-value component per unit of
complexity.
**O3** — `requirements.txt` uses `>=` throughout. A lock file would harden reproducibility. Already
acknowledged in the compliance file as a non-blocker.

---

## L. What I Should Change

| Priority | Action | Why | Expected benefit | Effort |
|---|---|---|---|---|
| **Must do** | Fix `technology_evidence.py`: select the reliability table by `calibration["deployed_variant"]` instead of the literal `"calibrated"`; make the §12 prose branch on `calibration["adopted"]`; replace the hardcoded `0.27` and the "no prediction reaches 0.90" sentence with values read from the deployed variant. Then regenerate the report and re-execute notebook 15. Fix the same sentence in `notebooks/15` cells 1 & 45 and the L2 row of `INTERNSHIP_GUIDE_COMPLIANCE.md`. | The published reading guide contradicts the shipped artifact, the config, and the run log | Removes the only defect that could mislead an operational user | ~1 hour |
| **Must do** | Add a test asserting the report's confidence narrative matches `final_model_config.json["calibration"]["deployed_variant"]`, and that `episode_technology_predictions.csv["confidence_type"]` agrees | 30 technology tests exist and none caught this; the gap is structural, not accidental | Prevents recurrence | 20 min |
| **Must do** | Move `technology_stages()` **before** `evidence_stages()` in `run_final_pipeline.py::main`, then do one clean full run so `final_pipeline_manifest.json` and `canonical_state_validation.json` describe the actual current state | The gate currently certifies a superseded state | Restores the reproducibility guarantee the README advertises | 30 min + run time |
| **Worth doing** | Add `log(candidate_pool_size)` as a row in `survival_selection_diagnostic.csv` and as one extra column in the Cox sensitivity table; state in `SURVIVAL_ANALYSIS_REPORT.md` that framework HR moves 1.751 → 1.617 under it while CPV-35 moves 1.553 → 1.512 | It is the largest detectability imbalance and is currently invisible | Strengthens the CPV-35 claim, correctly qualifies the framework claim, closes the most obvious hole a reviewer would find | 1–2 hours |
| **Worth doing** | Either record how the ~25 reviewed candidates per anchor were selected, or add one sentence to `REGIONAL_BENCHMARK_REFERENCE.md` §Known Limitations stating the rule was not recorded and that recall is therefore not fully independent of the text score, while precision is | The independence claim is currently stronger than the evidence supports | Protects the reference's credibility under external review | 15 min (disclosure) |
| **Worth doing** | Apply Holm/BH to the 5 CPV trend slopes, exactly as the technology trend section already does, and soften the CPV-48 line in `EXECUTIVE_SUMMARY.md` from "statistically distinguishable decline" to "nominal decline that does not survive correction for the five segments tested" | The project already holds itself to this standard elsewhere | Removes the only over-claim I found; costs one bullet | 30 min |
| **Worth doing** | Add one sentence to `DATA_QUALITY_REPORT.md` §Data Grain: the digital filter is an any-CPV-code rule at episode level and 30.9% of the cohort has a main CPV outside 32/35/48/72; note that `digital_segment` is the lowest-numbered digital division present and that event rates are near-identical under a main-CPV assignment | Prevents an over-literal reading; the measured robustness is a selling point | Pre-empts the most likely reviewer objection to the cohort | 30 min |
| **Optional** | Add `reference_conflict_episodes` and `suspicious_review_cases` rows to the integrity table; rename `cohort_link_rate`; show `n/a` for M_A's threshold; the §J doc-drift list | Presentation and future-misquote hygiene | Small | 1 hour total |
| **Optional** | Stratify the Cox model on award-year band instead of adjusting | Cleanest response to a 4×10⁻¹⁷ PH violation | Marginal — coefficients barely move | 2 hours |
| **Do not bother** | Re-tuning the 0.70 threshold on the same reference | The policy is frozen after development and the current evidence is internal validation | Would add post-hoc optimisation without independent evidence | — |
| **Do not bother** | Running CamemBERT | Pre-registered gate not met; diagnosis is high variance, for which more capacity is the wrong fix | Negative | — |
| **Do not bother** | Reworking episode reconstruction over the transitive-merge concern | I measured it: only 21/3,800 cohort episodes use the risky route, median span 126 days, 37 episodes with any notice >90 d post-award | None | — |
| **Do not bother** | Imputing missing durations | Completeness swings 11.8% → 84.4% across years; imputation would fabricate expiry dates | Negative | — |
| **Do not bother** | Adding character n-grams or a richer text representation | Learning curve shows high variance, not underfitting; training F1 ≈ 0.97 at every size | Negative | — |
| **Do not bother** | Building a fifth linkage arm | Four arms plus a borderline band plus template-risk re-censoring already bracket the event definition | None | — |
| **Do not bother** | Padding the methodology chapter toward the guide's 40–60 pages | Technical depth is complete; padding adds no evidence | None | — |

---

## M. What I Should NOT Change

This section matters as much as the fix list. The following are already correct and further
optimisation would produce nothing.

1. **`M_B_text_ranking @ 0.70`, frozen after development.** It is not optimal and the project never
   claims it is. Reference evidence informed the retained policy, so the split is internal validation,
   not an untouched holdout. `QUALITY_EVIDENCE.md` publishes the sweep showing 0.60 has better recall
   and retains it as a sensitivity arm. **Keep the policy unless fresh independent evidence supports a change.**

2. **The four-arm sensitivity + borderline band + template-risk re-censoring.** Three independent
   robustness families for one event definition is already generous. The template-risk analysis in
   particular — re-censoring rather than dropping the 173 at-risk links, because "a spurious link
   means the anchor had no observed successor and should contribute its full follow-up as censored
   exposure" — is exactly the right counterfactual. **Keep all three; do not add a fourth.**

3. **Refusing to impute duration.** Evidenced by measured temporal instability, not asserted. **Keep.**

4. **Reading the 12/24-month probabilities off Kaplan–Meier rather than the GeneralizedGamma fit,**
   despite the parametric model having the lowest AIC. The stated reason — smooth families flatten
   the observed renewal shoulder and every reported horizon is inside the observed window — is
   correct. **Keep.**

5. **Gate A + Gate B on technology enrichment.** Keep, and keep publishing that Gate A *costs*
   significance. That paragraph is the most persuasive thing in the technology report.

6. **Not running CamemBERT, and saying so with the pre-registered threshold.** Keep the exact
   wording: "It was not tested and then discarded; it was not run."

7. **Leaving the 3 near-duplicate annotation inconsistencies uncorrected**, with the stated reason
   that "editing labels after seeing model errors is how a corpus is fitted to its classifier."
   Keep. Correcting them would be worse.

8. **Reporting `AI` (n = 7) with its support and marking it uninterpretable** rather than
   oversampling, synthesising examples, or dropping the class. Keep.

9. **The 90–2,920 day candidate window.** Empirically justified in the code comment against a named
   pathology (132 of 628 links falling inside three months). Keep.

10. **Not imposing same-CPV-division blocking.** Evidenced by the reference itself — 9 of 23 reviewed
    successors cross divisions, and hard blocking would cut the ceiling to 0.609. Keep.

11. **Publishing the Cox out-of-time C-index of 0.479 and refusing to retune.** Keep. Deleting or
    re-tuning this would be the single worst change available.

12. **The `weighted_score` renormalise-over-available-evidence design.** Missing CPV is dropped from
    numerator and denominator rather than scored zero. Keep.

13. **The whole "Boundaries" section of the README.** Nine explicit prohibitions on how the results
    may be used. Keep verbatim.

---

## N. Missing Analysis

### Required to support current claims

Only two, and both are small:

1. **Publish the pool-size detectability row and the framework-HR attenuation** (§L, Worth doing).
   Without it, the framework finding is stated more strongly than the data support.
2. **Apply the project's own multiplicity standard to the CPV trend slopes**, or withdraw CPV-48
   from the executive summary's "What Works". Currently the same situation is handled to two
   different standards in two documents.

(Plus the Critical-1 documentation fix, which is a correction rather than a missing analysis.)

### Interesting but unnecessary

- **Cohen's kappa on the technology corpus.** Genuinely valuable, but it needs a second annotator you
  do not have. The project's own recommendation — a second pass over a *sample* — is the right ask,
  and it correctly notes the learning curve is still rising at n = 500, so more annotation beats a
  bigger model. This is future work, not a gap in the current deliverable.
- **Independent human specialist review of the 60-pair sample.** Prepared, protocol written, cannot
  be completed in-house by definition. Correctly scoped as a precondition for a *stronger* claim, not
  for the current one.
- **The Gigalis-membership causal DiD.** Outlined, not estimated, because it needs membership and
  adoption-date data BOAMP does not contain. The guide allows an outline. Correct.
- **Propagating classification uncertainty into technology-level survival intervals.** Would be
  methodologically nice; the current treatment (state the error rate as a limitation) is adequate for
  an enrichment layer that is explicitly not the reference analysis.
- **Pair-level clustered inference for the ROC/PR curves.** The curves are already de-emphasised as
  ranking diagnostics.

---

## O. Final Recommendation

### ▶ Option 2 — Keep the overall approach and make a few targeted corrections.

Nothing in the methodology needs substantial revision. Every headline number I recomputed matched
exactly. The design choices I checked were either correct or correct-and-better-documented-than-usual.
The three things I found that the project had not already found itself are one report-generation
defect, one omitted covariate in a diagnostic, and one unrecorded provenance detail — none of which
changes a conclusion, and one of which (the CPV-35 robustness under pool-size adjustment) is good
news you should publish.

The main risk to this project is not methodological. It is that it keeps getting revised. The
documentation already carries more caveats than most published papers, and adding more would start to
obscure the findings rather than qualify them.

### Next actions, in order

1. **Fix the calibration narrative** in `technology_evidence.py` (branch on `deployed_variant` and
   `adopted`), regenerate `TECHNOLOGY_TAXONOMY_REPORT.md`, re-execute notebook 15, and fix the L2 row
   in `INTERNSHIP_GUIDE_COMPLIANCE.md`. *(~1 hour — this is the only item I would call blocking.)*
2. **Add the regression test** that ties the published confidence narrative to
   `final_model_config.json`. *(20 min)*
3. **Reorder** `technology_stages()` before `evidence_stages()` in `run_final_pipeline.py`, then run
   `PYTHONPATH=. python3 scripts/run_final_pipeline.py --with-notebooks --with-tests` once cleanly so
   the manifest and the validation gate describe reality. *(30 min + run)*
4. **Publish the pool-size diagnostic row and the Cox sensitivity column**, with the note that CPV-35
   is stable and framework attenuates to 1.617. *(1–2 hours)*
5. **Correct the CPV-48 claim** in `EXECUTIVE_SUMMARY.md` and apply Holm/BH to the CPV trend slopes.
   *(30 min)*
6. **Add the two documentation sentences**: the any-CPV-code cohort rule with its measured
   robustness, and the reference candidate-export disclosure. *(45 min)*
7. **Sweep the Minor list** (§K M3–M7) in one pass. *(1 hour)*
8. **Stop.** Freeze the repository and write the final report against it.

---

## Evidence Index

Claims in this audit trace to: `README.md`; `EXECUTIVE_SUMMARY.md`; `FINAL_PIPELINE.md`;
`DATA_QUALITY_REPORT.md`; `SURVIVAL_ANALYSIS_REPORT.md`; `TREND_ANALYSIS_REPORT.md`;
`TECHNOLOGY_TAXONOMY_REPORT.md` §7, §9, §12, §13, §14; `QUALITY_EVIDENCE.md`;
`REGIONAL_BENCHMARK_REFERENCE.md`; `CANDIDATE_GENERATION_AUDIT.md`; `INTERNSHIP_GUIDE_COMPLIANCE.md`;
`reports/boamp_methodology_chapter.tex`; `boamp_pipeline/linkage.py`, `episodes.py`,
`technology_evidence.py` (lines 1073, 1467–1510), `technology_models.py` (lines 960–1045);
`scripts/run_final_pipeline.py` (`main`, `technology_stages`), `build_linkage_candidates.py`
(lines 143–249), `build_survival_cohort.py` (lines 125–168), `evaluate_linkage.py` (lines 88–213),
`build_regional_benchmark.py`; `tests/test_technology_taxonomy.py`;
`data/processed/boamp/{survival_dataset,survival_cohort,linkage_candidates_scored,accepted_successor_links}.parquet`;
`data/processed/boamp/{survival_*,linkage_*,candidate_generation_audit,data_quality_profile,episode_reconstruction_summary,standardized_notice_summary,canonical_state_validation,final_pipeline_manifest,trend_analysis_summary}.{json,csv}`;
`data/processed/boamp/technology/*`; `data/processed/boamp/regional_benchmark/*`;
`data/reference/regional_link_benchmark/BOAMP_Internship_Reference_120.csv`;
`logs/build_technology_taxonomy.log`; `git log`, `git status`, `git diff`.

Recomputations were performed in an isolated environment using pandas 3.0.2, scikit-learn 1.8.0,
statsmodels (Breslow-tie `PHReg`) and a hand-implemented Kaplan–Meier estimator, reading only the
project's materialised outputs.
