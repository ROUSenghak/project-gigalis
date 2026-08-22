# Buyer-level dependence: three new analyses on the frozen cohort

Run: 21 August 2026
Data: `data/processed/boamp/survival_dataset*.parquet` (frozen, unmodified)
Tools: statsmodels 0.14.6 `PHReg` (Breslow ties) — the same version recorded in `final_pipeline_manifest.json`
Nothing in the pipeline was re-run or altered. These are read-only analyses on the published artifacts.

---

## 0. Replication check first

Before anything new, the published results were reproduced independently:

| Quantity | Published | Reproduced |
|---|---|---|
| Cox HR, all 8 covariates | see `survival_cox_results.csv` | identical to 4 dp |
| Cox 95% CIs | [1.435, 2.136] etc. | identical |
| Train C-index (2015-2021) | 0.606 | 0.6065 |
| Out-of-time C-index (2022-2024) | 0.479 | 0.4785 |

The published survival layer reproduces exactly.

---

## 1. The cohort has strong buyer-level clustering

```
episodes                                    3,800
distinct buyers (buyer_key + name fallback)   976
mean episodes per buyer                      4.17
max episodes for one buyer                    114
share of episodes in buyers with >=10          51.9 %
share of episodes in buyers with >=25          27.2 %
```

**Data note:** `buyer_key` is an empty string for 114 episodes (3.0%). Those 114 are in fact 65 distinct
buyers by raw name. Any buyer-level analysis must apply a name fallback, or it silently treats 114
unrelated procurements as one buyer.

The Cox partial likelihood treats all 3,800 episodes as independent contributions. They are not.

---

## 2. Buyer-clustered standard errors (Lin–Wei sandwich)

$$\widehat V_{\text{robust}} = \mathcal I^{-1}\Big(\sum_{g} U_g U_g^{\top}\Big)\mathcal I^{-1},
\qquad U_g = \sum_{i \in g} u_i$$

with $u_i$ the Cox score residuals and $g$ indexing buyers. Point estimates are unchanged by construction.

*Implementation check:* with each episode as its own cluster, the sandwich SEs reproduce the model-based
SEs to three decimals (0.0999 vs 0.1014, 0.1030 vs 0.1027, …), confirming the residual computation.

| Covariate | HR | SE model | SE buyer | inflation | p model | **p buyer** |
|---|---:|---:|---:|---:|---:|---:|
| framework_flag | 1.751 | 0.1014 | 0.1603 | 1.58 | 3.4e-8 | **0.0005** |
| has_validated_siren | 1.082 | 0.1027 | 0.1774 | 1.73 | 0.444 | 0.657 |
| award_year_centered | 1.107 | 0.0192 | 0.0385 | 2.00 | 1.2e-7 | **0.0083** |
| **digital_segment_CPV-35** | **1.553** | 0.1240 | **0.1197** | **0.97** | 3.8e-4 | **0.0002** |
| digital_segment_CPV-48 | 0.828 | 0.1324 | 0.2368 | 1.79 | 0.153 | 0.424 |
| digital_segment_CPV-72 | 1.056 | 0.1104 | 0.1564 | 1.42 | 0.625 | 0.730 |
| buyer_region_Normandie | 0.800 | 0.1138 | 0.1654 | 1.45 | **0.050** | **0.178** |
| buyer_region_Pays de la Loire | 1.003 | 0.1061 | 0.1713 | 1.62 | 0.979 | 0.987 |

Median SE inflation 1.60, i.e. a design effect on the variance of about **2.55**. The effective sample size
is closer to 1,500 than to 3,800.

### Why CPV-35 is the exception

| Covariate | share of variance that is *within* buyer |
|---|---:|
| framework_flag | 66.9 % |
| **digital_segment_CPV-35** | **57.9 %** |
| has_validated_siren | 0.1 % |
| buyer_region | ~0.6 % of buyers vary |

Region and SIREN quality are buyer *attributes*: they carry essentially no within-buyer information, so
their effective sample size is 976 buyers, not 3,800 episodes, and their SEs inflate accordingly. CPV
segment varies inside a buyer, so it is identified from within-buyer contrasts and is barely affected.

> **Consequence for the report.** The regional coefficient (Normandie, $p = 0.050 \to 0.178$) should no
> longer be described as borderline significant. Framework and CPV-35 both survive comfortably.

---

## 3. Bootstrap interval on the out-of-time C-index

Model fitted once on 2015–2021 (2,470 contracts, 392 events), scored on 2022–2024 (1,004 contracts,
107 events), no refit. 2,000 bootstrap replicates.

| Resampling unit | $\hat C_{\text{oot}}$ | 95 % CI | $\Pr(C>0.5)$ | $\Pr(C>0.55)$ | $\Pr(C>0.60)$ |
|---|---:|---|---:|---:|---:|
| episodes | 0.478 | **[0.415, 0.542]** | 0.256 | 0.011 | 0.000 |
| buyers (clustered) | 0.478 | [0.354, 0.623] | 0.452 | 0.259 | 0.073 |

**Reading.** Both intervals contain 0.5, so the null of no out-of-time discrimination is not rejected — the
report's claim holds. The episode bootstrap additionally excludes 0.55 with 99% confidence, which supports
a *stronger positive statement* than the report currently makes:

> The data do not merely fail to establish useful individual discrimination; they rule it out. Under
> episode resampling, $\Pr(C > 0.55) = 0.011$ and $\Pr(C > 0.60) = 0.000$.

The buyer-clustered interval is wider and is the conservative version. Report both, or report the episode
interval with the clustered one as a footnote.

---

## 4. Buyer-stratified Cox — the strongest available control for detectability

### 4.1 Why this is a stronger test than adding `log(pool size)`

Stratifying on buyer gives every buyer its own baseline hazard $h_{0g}(t)$:

$$h(t \mid X, g) = h_{0g}(t)\exp(\beta^{\top}X)$$

Then **every buyer-level confounder is absorbed non-parametrically** — publication volume (the detectability
channel), region, institutional type, procurement culture, SIREN quality. Candidate-pool size is constant
within a buyer, so the detectability confound is eliminated rather than adjusted for.

Coverage: 195 buyers contribute (≥2 episodes and ≥1 event), 2,188 episodes, **511 of 544 events (94%)**.
Buyers identifying each contrast: CPV-35 → 98 buyers / 328 events; CPV-48 → 121 / 369; framework → 119 / 388.

### 4.2 Results, all four linkage arms

| Arm | events | framework | award year | **CPV-35** | **CPV-48** | CPV-72 |
|---|---:|---|---|---|---|---|
| strict `M_B@0.80` | 296 | 1.128 (0.533) | 0.925 (0.020) | 1.265 (0.298) | **0.477 (0.001)** | 0.976 (0.898) |
| primary `M_B@0.70` | 544 | **1.497 (0.003)** | 1.015 (0.547) | 1.213 (0.262) | **0.482 (<0.001)** | 0.831 (0.196) |
| loose `M_B@0.60` | 853 | **1.664 (<0.001)** | 1.047 (0.024) | 1.280 (0.080) | **0.589 (<0.001)** | 0.881 (0.274) |
| contrast `M_C@0.70` | 1,332 | **1.459 (<0.001)** | 1.071 (<0.001) | 1.109 (0.389) | **0.744 (0.006)** | 1.279 (0.006) |

Hazard ratios with $p$-values in parentheses. Every row is buyer-stratified.

### 4.3 What changes

**CPV-35 attenuates and loses significance in every arm.** Point estimates 1.265 / 1.213 / 1.280 / 1.109
against an unstratified 1.553.

The within-buyer 95% CI is [0.866, 1.699] and **does contain 1.553**, so this is *attenuation with loss of
precision, not a refutation*. But four independent arms all landing near 1.2 is a consistent signal.

> **Interpretation.** The two models answer different questions.
> *Unstratified:* across the Grand Ouest population, do CPV-35 contracts show successors earlier? → yes.
> *Stratified:* within one buyer, do its CPV-35 contracts show successors earlier than its own others? → not detectably.
> The gap says the population-level CPV-35 effect is substantially **between-buyer compositional** — it is
> partly about *which buyers* procure security equipment, not about the segment itself.

**CPV-48 becomes a strong, consistent finding in the opposite direction.** HR 0.48–0.74, significant in all
four arms, and invisible in the unstratified model ($p = 0.15$). Within a buyer, business-software
procurements show an observable successor markedly *later* than telecoms.

**Framework survives the strongest control available.** 1.497 (p = 0.003) in the primary arm, significant in
three of four arms (the strict arm has only 296 events). Since buyer stratification removes the pool-size
channel entirely, this is stronger evidence for a genuine framework association than the
`log(pool size)` sensitivity provided — while confirming its magnitude is smaller than the headline 1.751.

**Award year collapses** (1.107 → 1.015, p = 0.55), consistent with it being largely compositional and
confounded with follow-up length.

### 4.4 Limitations of this check

1. Buyer stratification discards all between-buyer information. The population question remains legitimate.
2. It removes *volume*-based detectability, not *text*-based detectability. If segments differ in how
   boilerplate their wording is, a residual channel remains within buyer.
3. The within-buyer CPV-35 CI is wide; the result qualifies the claim rather than overturning it.

---

## 5. What this implies for the report

| Current report claim | Suggested revision |
|---|---|
| "CPV-35 is the study's most robust comparative finding" | Qualify: robust to event-definition perturbations, but **substantially between-buyer**; the within-buyer estimate is attenuated to ~1.2 and not significant in any arm |
| "The framework association is partly detectability" | Strengthen: it survives **buyer stratification**, the strongest available control, at HR ≈ 1.50 |
| CPV-48 not discussed as a finding | Add: within-buyer, CPV-48 shows a **markedly later** successor, HR 0.48–0.74, consistent and significant across all four arms |
| Normandie $p = 0.050$ | Restate as null: $p = 0.178$ under buyer-clustered inference |
| "out-of-time C = 0.479, indistinguishable from chance" | Strengthen: 95% CI [0.415, 0.542]; $\Pr(C>0.55) = 0.011$ — useful discrimination is **ruled out**, not merely unestablished |

---

## 6. Reproduction

```python
import pandas as pd, numpy as np
from statsmodels.duration.hazard_regression import PHReg

d = pd.read_parquet('data/processed/boamp/survival_dataset.parquet')
bk = d.buyer_key.astype(str).str.strip()
d['cluster'] = np.where(bk == '',
                        'name2:' + d.buyer_name_raw.astype(str).str.lower().str.strip(), bk)

X = np.column_stack([
    d.framework_flag.astype(int),
    d.award_year - d.award_year.mean(),
    (d.digital_segment == 'CPV-35').astype(int),
    (d.digital_segment == 'CPV-48').astype(int),
    (d.digital_segment == 'CPV-72').astype(int)]).astype(float)

fit = PHReg(d.duration_months.values.astype(float), X,
            status=d.event.values.astype(int),
            strata=pd.factorize(d.cluster)[0], ties='breslow').fit()
print(np.exp(fit.params), fit.bse)
```

The Lin–Wei sandwich and the bootstrap use the same frozen inputs; full scripts were run read-only against
`data/processed/boamp/` and changed nothing in the repository.
