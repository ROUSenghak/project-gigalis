# BOAMP Data Quality Report

Generated: `2026-08-13T22:14:19`  
Data through: `2025-12-31`  
Assessment: **Share with caveats**

## Technical Summary

The processed data are reproducible and internally coherent at their declared grains: `1,620,712` unique BOAMP notices become `1,103,632` reconstructed procurement episodes, and the final study cohort contains `3,800` unique awarded Grand Ouest digital episodes. No duplicate notice IDs, duplicate survival episodes, negative survival durations, impossible episode chronologies, or accepted links with conflicting validated SIRENs were found.

The main risks are measurement rather than pipeline corruption. Validated SIREN is missing for `66.3%` of the cohort, reliable duration for `74.9%`, and the benchmark labels are deterministic bootstrap labels rather than independent specialist judgements. Therefore the linkage and survival results are usable as conservative exploratory evidence, but their reported precision is not an independently established accuracy guarantee.

## Data Grain And Selection

| Layer | Grain | Rows | Main rule |
|---|---|---:|---|
| Standardised data | BOAMP notice | 1,620,712 | Official notices published 2015-2025 |
| Reconstructed data | Procurement episode | 1,103,632 | Explicit links, shared folder IDs, or constrained reference links |
| Study cohort | Awarded digital episode | 3,800 | Grand Ouest, CPV divisions 32/35/48/72, resolved award date |
| Candidate table | Anchor-candidate pair | 763,417 | Same plausible buyer, 90-2,920 days later |
| Survival table | Cohort episode | 3,800 | First accepted successor or administrative censoring |

The source is the official [BOAMP API](https://www.data.gouv.fr/dataservices/api-bulletin-officiel-des-annonces-des-marches-publics-boamp), which publishes procurement notices and results. A notice is not necessarily a distinct contract, so episode reconstruction is necessary before successor linkage.

## Integrity Checks

| Check | Result |
|---|---:|
| Duplicate standardised notice IDs | 0 |
| All notices assigned to exactly one episode | True |
| Buyer-conflict episodes | 0 |
| Impossible episode chronologies | 0 |
| Duplicate survival episodes | 0 |
| Negative survival durations | 0 |
| Accepted links with conflicting validated SIRENs | 0 |
| Accepted municipal/intercommunal entity mixes | 0 |

These checks support structural consistency, not semantic truth. A syntactically valid episode can still combine notices incorrectly, and a plausible successor can still be a different procurement need.

## Missingness And Treatment

| Field | Missing rate | Treatment | Reason |
|---|---:|---|---|
| Validated SIREN | 66.3% | Preserve name-only buyer key; audit risky links | [SIREN](https://www.insee.fr/fr/metadonnees/definition/c2047) identifies a legal unit; [SIRET](https://www.insee.fr/fr/metadonnees/definition/c1841) identifies an establishment, so names alone cannot prove legal identity |
| Reliable duration | 74.9% | No imputation | Completeness changes sharply by year and a universal four-year value would create false expiry dates |
| Any amount candidate container | 15.7% | Do not aggregate as contract value | The container can hold multiple notice-level amount candidates and has no validated canonical awarded value |
| Main CPV | 0.0% | Required by cohort selection | CPV is a hierarchical procurement vocabulary under [Regulation 213/2008](https://eur-lex.europa.eu/eli/reg/2008/213/oj) |
| Episode text | 0.0% | Required by cohort selection | Needed for linkage ranking |
| Award date | 0.0% | Required by cohort selection | Defines survival time zero |

The decision not to impute duration is supported by the observed temporal instability: reliable duration is present for only `11.8%` of 2023 episodes but `84.4%` of 2025 episodes. Missingness is therefore not plausibly exchangeable across years. EU procurement law also treats four years as a general framework-agreement limit with justified exceptions, not as the duration of every contract ([Directive 2014/24/EU, Article 33](https://eur-lex.europa.eu/eli/dir/2014/24/oj)).

![Cohort missingness](reports/figures/data_quality_key_missingness.png)

## Linkage Coverage And Sensitivity

Candidate generation finds at least one candidate for `3,520` of `3,800` anchors (92.6%). This is candidate availability, not linking accuracy. The primary method accepts `544` links.

| Linkage arm | Events | Event rate | Median observed successor time |
|---|---:|---:|---:|
| Strict text threshold | 296 | 7.8% | 35.7 months |
| Primary text threshold | 544 | 14.3% | 31.8 months |
| Looser text threshold | 853 | 22.4% | 26.6 months |
| Weighted high-recall contrast | 1332 | 35.0% | 26.1 months |

The large event-rate range is a material uncertainty result. Absolute survival probabilities depend on the linkage policy and should be presented with sensitivity analyses.

## Benchmark Evidence Quality

The current national reference has `7,031` labelled pairs, but both passes were generated by the deterministic rules in `scripts/auto_annotate_wave1a.py`. Their κ of `1.00` is self-consistency under re-presentation, not human inter-annotator agreement. The split names `dev` and `validation` remain useful for preventing threshold reuse, but the resulting precision and recall are internal protocol-reference estimates.

This is the main unresolved validation risk. The appropriate correction is an independent specialist review of a compact, stratified sample, especially accepted links, method disagreements, high-similarity structural negatives, and name-only buyer matches.

## Defensible Use

- Safe: describe the pipeline as identifying **observable successor procurements**.
- Safe: use `M_B_text_ranking @ 0.70` as a provisional precision-first operating baseline.
- Safe: report sensitivity across linkage thresholds and methods.
- Not safe: call the events confirmed legal renewals.
- Not safe: call 0.80 an independently validated precision guarantee.
- Not safe: use current amount candidates for monetary trend conclusions.

## Reproduction

```bash
PYTHONPATH=. python3 scripts/build_project_evidence.py
PYTHONPATH=. jupyter nbconvert --execute --to notebook --inplace notebooks/14_data_quality_and_trend_analysis.ipynb
PYTHONPATH=. pytest -q
```
