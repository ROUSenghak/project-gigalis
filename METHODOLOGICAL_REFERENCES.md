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
| [Fellegi and Sunter (1969)](https://doi.org/10.1080/01621459.1969.10501049) | `M_D` is a recognized probabilistic record-linkage approach based on match/non-match comparison likelihoods | Its assumptions do not guarantee good performance on rare successor events with dependent candidate pairs |
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
3. Use CPV and text similarity as continuity evidence, not legal proof.
4. Keep the 90-2,920 day interval as an operational broad candidate window, not
   a statutory contract-duration rule.
5. Do not impute missing duration or assume every framework lasts four years.
6. Treat linkage metrics from deterministic bootstrap labels as development
   diagnostics until the blinded specialist review is completed.
7. Use the empirical, unsmoothed anchor-level precision-recall threshold sweep
   as project evidence. Generic web illustrations may explain intuition in a
   presentation but are not academic evidence.
8. Report survival estimates as linkage-conditioned and non-causal, with event
   definition sensitivity.
9. Treat PELT breaks and recent slopes as exploratory signals requiring domain
   corroboration.
10. Treat HMM regime labels as a descriptive complement to PELT, not a
    forecast; report disagreement between the two honestly rather than
    reconciling it.
11. Keep the causal-inference discussion of Gigalis membership effects as a
    design outline only until Gigalis-internal membership and adoption-date
    data are available; do not substitute a BOAMP-only proxy treatment.
