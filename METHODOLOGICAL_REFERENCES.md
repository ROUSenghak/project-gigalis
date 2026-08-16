# Methodological References And Design Implications

Assessment date: `2026-08-13`

This file records the external support for the implemented choices. A reference
supports a definition or method; it does not validate this project's empirical
results.

| Source | What it supports here | What it does not justify |
|---|---|---|
| [Official BOAMP API, DILA](https://www.data.gouv.fr/dataservices/api-bulletin-officiel-des-annonces-des-marches-publics-boamp) | BOAMP notices are the official source for calls and award/result notices used by the pipeline | BOAMP does not provide a complete legal-renewal label for every contract |
| [INSEE SIREN definition](https://www.insee.fr/fr/metadonnees/definition/c2047) | A valid SIREN identifies a legal unit and should outrank buyer-name similarity | Similar names alone do not prove legal identity |
| [INSEE SIRET definition](https://www.insee.fr/fr/metadonnees/definition/c1841) | A SIRET identifies an establishment and contains the legal unit's SIREN | Different establishments must not automatically be treated as different legal buyers |
| [Commission Regulation (EC) No 213/2008](https://eur-lex.europa.eu/eli/reg/2008/213/oj) | CPV is a hierarchical procurement vocabulary; division continuity is a reproducible coarse domain feature | CPV equality does not prove renewal, and four CPV divisions are not a learned technology taxonomy |
| [Directive 2014/24/EU, Article 33](https://eur-lex.europa.eu/eli/dir/2014/24/oj) | Framework duration is generally limited to four years, subject to justified exceptions | It does not justify imputing four years for contracts with missing duration |
| [scikit-learn cosine similarity](https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity) | The implemented TF-IDF cosine score is a standard normalized document-similarity measure | A high cosine score does not independently prove a successor relationship |
| [Davis and Goadrich (2006)](https://doi.org/10.1145/1143844.1143874) | Precision-recall curves are appropriate for highly skewed binary decisions and must not be interpreted through invalid linear interpolation between operating points | The paper does not validate this project's labels, scores, or chosen threshold |
| [Saito and Rehmsmeier (2015)](https://doi.org/10.1371/journal.pone.0118432) | Precision-recall analysis can reveal performance differences hidden by ROC analysis when positives are rare | It does not establish that `0.70`, or any other threshold, generalises beyond this reference sample |
| [Christen and Goiser (2007)](https://doi.org/10.1007/978-3-540-44918-8_6) | Blocking recall has a name and a standard treatment: *pairs completeness*, the share of true matches surviving the blocking step, "analogous to recall". The chapter states that blocking is a confounding factor in linkage-quality measurement and that the blocking measures and the blocking rule should be published alongside any quality figure. This is why the candidate-generation ceiling is reported next to every recall number and why the blocking rule is stated as an explicit formula | It does not set an acceptable pairs-completeness level, and it does not license reading this reference sample's ceiling as a population recall ceiling |
| [Potin, Labatut, Morand and Largeron (2023)](https://doi.org/10.1038/s41597-023-02213-z) | Missing agent identification is documented as the most serious data-quality problem in French public procurement notices, with buyer SIRETs populated on a minority of lots, and the same authority appearing under alternative and former names, so identifier-first buyer blocking with a conservative name fallback is the appropriate design and a legal-form transition without a shared SIREN is a known unresolvable case | It does not supply the SIRENE-based entity-resolution layer that paper builds, and it does not establish that this project's buyer blocking achieves comparable coverage |
| [Siciliani, Tanzi, Basile and Lops (2023)](https://aclanthology.org/2023.clicit-1.47) | CPV assignment is error-prone even for human experts: the size of the vocabulary "frequently leads to errors in the CPV assignment like typos or wrong interpretation", with skewed usage concentrating on a few well-known codes. This supports treating CPV as soft continuity evidence rather than a hard blocking constraint | It is an Italian-tender classification study; it does not measure CPV error rates in BOAMP, and it does not justify the specific cross-division links accepted here |
| [Fellegi and Sunter (1969)](https://doi.org/10.1080/01621459.1969.10501049) | `M_D` is a recognized probabilistic record-linkage approach based on match/non-match comparison likelihoods | Its assumptions do not guarantee good performance on rare successor events with dependent candidate pairs |
| [Harron, Doidge, Knight, Gilbert, Goldstein, Cromwell and van der Meulen (2017)](https://doi.org/10.1093/ije/dyx177) | The three-part strategy this project uses to evaluate linkage quality: apply the algorithm to a reference subset with known status, compare linked with unlinked records using standardized differences, and test whether conclusions survive changes to the linkage procedure | It endorses the strategy, not this project's precision, recall, or threshold; a reference subset this small still yields wide intervals |
| [Doidge and Harron (2019)](https://doi.org/10.1093/ije/dyz203) | Missed links and false links cause misclassification in opposite directions, so a linkage-conditioned event rate is not a one-sided bound; linkage error surfaces as information bias or selection bias depending on how it correlates with the analysis variables | It does not quantify this project's error rates, and it does not license a correction that would require knowing them |
| [Uno, Cai, Pencina, D'Agostino and Wei (2011)](https://doi.org/10.1002/sim.4154) | Harrell's C for right-censored data converges to a quantity that depends on the censoring distribution, so concordance figures computed on windows with different follow-up are not strictly comparable | It does not rescue a weak C-index, and this project reports the standard estimator rather than implementing the paper's IPCW alternative |
| [Kaplan and Meier (1958)](https://doi.org/10.1080/01621459.1958.10501452) | Kaplan-Meier estimation handles administrative right-censoring | It cannot correct event misclassification created by linkage errors |
| [Cox (1972)](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x) | The Cox model estimates covariate associations with the event hazard without specifying a baseline hazard | Hazard ratios are not causal effects without stronger identification assumptions |
| [Grambsch and Therneau (1994)](https://doi.org/10.1093/biomet/81.3.515) | Schoenfeld-residual diagnostics test the proportional-hazards assumption | Passing a diagnostic would not validate linkage or causal interpretation |
| [Killick, Fearnhead and Eckley (2012)](https://doi.org/10.1080/01621459.2012.737745) | PELT provides an efficient multiple-change-point procedure | A statistical break does not identify its cause; the report treats breaks as descriptive candidates |
| [Hamilton (1989)](https://doi.org/10.2307/1912559) | Markov-switching regime models are the general framework behind the trend HMM's growth/plateau/decline states | Regime-switching theory does not validate a specific 3-state fit on 43 noisy quarterly observations, and a detected regime is not a forecast |
| [Angrist and Pischke (2009)](https://www.jstor.org/stable/j.ctvcm4j72) | Difference-in-differences and instrumental-variables identification are standard tools for the causal question the internship guide poses about Gigalis membership | The textbook does not supply Gigalis-internal membership data, without which no DiD estimate can be computed here |
| [Pearl, Glymour and Jewell (2016)](https://www.wiley.com/en-us/Causal+Inference+in+Statistics%3A+A+Primer-p-9781119186847) | The DAG framework is the right way to state causal assumptions explicitly before any causal estimation is attempted | It does not by itself identify an effect; assumptions still require domain knowledge and appropriate data |
| [Athey and Imbens (2017)](https://doi.org/10.1257/jep.31.2.3) | Surveys modern causal-ML methods relevant if a Gigalis-membership causal analysis is undertaken later | It is a survey, not evidence that this project's BOAMP-only data supports causal identification |
| [Callaway and Sant'Anna (2021)](https://doi.org/10.1016/j.jeconom.2020.12.001) | Staggered-adoption DiD is the natural design if Gigalis members join at different dates | Requires member-level adoption-date data not present in this repository; cited as a design reference only |

## Consequences For The Current Pipeline

1. Preserve raw BOAMP values and record parser lineage before standardisation.
2. Prefer validated SIREN evidence; keep legal forms such as municipality and
   intercommunal authority distinct when identity is not established.
3. Use CPV and text similarity as continuity evidence, not legal proof. Do not
   impose CPV equality as a hard blocking rule: the reviewed reference
   successors themselves cross CPV divisions in 9 of 23 cases, so a hard
   same-division block would discard genuine successors, and CPV assignment is
   independently documented as error-prone.
4. Report the candidate-generation ceiling (pairs completeness) next to every
   recall figure, state the blocking rule explicitly, and attribute unreachable
   reference cases to a named blocking condition, so a ceiling below 1.0 can be
   distinguished from a defect.
5. Keep the 90-2,920 day interval as an operational broad candidate window, not
   a statutory contract-duration rule.
6. Do not impute missing duration or assume every framework lasts four years.
7. Treat linkage metrics from an LLM-generated, subset-spot-checked reference sample as provisional
   diagnostics until the blinded specialist review is completed.
8. Use the empirical, unsmoothed anchor-level precision-recall threshold sweep
   as project evidence. Generic web illustrations may explain intuition in a
   presentation but are not academic evidence.
9. Report survival estimates as linkage-conditioned and non-causal, with event
   definition sensitivity. Because missed and false links misclassify in opposite
   directions, never state a one-sided bound on true re-procurement.
10. Read the out-of-time concordance as weak discrimination, and do not treat the
    difference between the 2022-2024 and 2022-2025 figures as a change in model
    quality: the two windows have different censoring distributions.
11. Treat PELT breaks and recent slopes as exploratory signals requiring domain
    corroboration.
12. Treat HMM regime labels as a descriptive complement to PELT, not a
    forecast; report disagreement between the two honestly rather than
    reconciling it.
13. Keep the causal-inference discussion of Gigalis membership effects as a
    design outline only until Gigalis-internal membership and adoption-date
    data are available; do not substitute a BOAMP-only proxy treatment.
