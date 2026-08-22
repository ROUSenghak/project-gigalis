# Audit de cohérence interne et audit des revendications
*Internal consistency audit and claim audit — applies to both the French and the English report*

Date : 21 août 2026
Portée : rapport français, rapport anglais, note de synthèse, executive summary
Méthode : chaque nombre publié a été retracé jusqu'à un artefact du dépôt (table matérialisée, JSON de résumé, configuration ou recalcul direct sur `survival_dataset.parquet`). Aucun chiffre n'a été repris d'un rapport antérieur lorsqu'un fichier canonique existait.

---

## 1. Vérification chiffre par chiffre

### 1.1 Données et cohorte

| Chiffre publié | Source | État |
|---|---|---|
| 1 620 712 avis | `episode_reconstruction_summary.json` → `notice_rows` | ✓ |
| 1 103 632 épisodes | idem → `episode_rows` | ✓ |
| 60,1 % de singletons | 662 798 / 1 103 632 | ✓ recalculé |
| Arêtes 45 885 / 737 / 584 896 / 195 043 / 861 | idem → `edge_counts` | ✓ |
| 3 274 épisodes à référence conflictuelle | idem | ✓ |
| Entonnoir 144 269 → 7 376 → 3 826 → 3 800, 26 écartés | `survival_cohort_summary.json` → `selection_funnel` | ✓ |
| 1 452 / 1 282 / 1 066 par région | idem → `rows_by_region` | ✓ |
| 1 294 / 1 152 / 790 / 564 par segment | idem → `rows_by_digital_segment` | ✓ |
| Suivi médian 2 015 jours | idem → `follow_up_days.median` | ✓ |
| SIREN 66,3 %, durée 74,9 %, montants 15,7 % | idem → `missingness` | ✓ |
| 11,8 % (2023) → 84,4 % (2025) de durées fiables | `data_quality_profile.json` ; recalculé sur la cohorte | ✓ |
| 1 176 (30,9 %) hors CPV principal ; 412 (10,8 %) multi-divisions ; 94,7 % de concordance | `DATA_QUALITY_REPORT.md` § grain, mesures publiées | ✓ |
| 2015 : 93 / 73 / 73 / 87 par trimestre ; 2024 : 72 / 57 / 72 / 85 | recalcul sur `survival_dataset.parquet` | ✓ recalculé |
| Suivi médian 2025 : 5,9 mois ; 2015-2024 : 59,9 mois | recalcul | ✓ recalculé |
| KM 12 m sans 2025 : 3,54 % contre 4,62 % | recalcul (KM implémenté à la main) | ✓ recalculé |
| PdL seuls : 1 452 épisodes, 219 événements, IC [4,17 ; 6,55] | recalcul, IC de Greenwood | ✓ recalculé |
| CPV-35 en PdL : 34 événements contre 115 | recalcul | ✓ recalculé |

### 1.2 Appariement

| Chiffre publié | Source | État |
|---|---|---|
| 763 417 couples, 3 520 ancres, 280 sans candidat | `survival_analysis_summary.json` → `candidate_coverage` | ✓ |
| 544 liens, taux 14,32 % | `final_pipeline_manifest.json` → `canonical_facts` | ✓ |
| M_B : VP 7, FP 1, FN 11, VN 54 ; précision 0,875 [0,529 ; 0,978] ; rappel 0,389 ; TFP 0,000 | `QUALITY_EVIDENCE.md`, comptabilité successeur exact | ✓ |
| M_A 0,533 / M_C 0,522 / M_D 0,200 | `REGIONAL_BENCHMARK_REFERENCE.md` ; recoupé avec les matrices de confusion | ✓ |
| Plafond 0,913 (21 / 23) ; deux cas attribués | `candidate_generation_audit.json` | ✓ |
| 9 / 23 successeurs changent de division (39,1 %) ; plafond 0,609 sous blocage strict | idem → `cpv_continuity` | ✓ |
| 351 / 538 (65,2 %) liens intra-division | idem | ✓ |
| 461 successeurs distincts ; 44 partagés ; maximum 11 ancres ; 127 liens concernés | idem ; 461 + 83 = 544 vérifié | ✓ recalculé |
| Revue : 14 / 5 / 1 ; précision 0,700 [0,457 ; 0,881] | `REVIEW_AUDIT_RESULTS.md`, `review_audit_evaluation.json` | ✓ |
| 21,8 % remis en concurrence à ±6 mois ; écart médian 21,1 mois | notebook 13, section Évidence / Décision | ✓ |

### 1.3 Survie

| Chiffre publié | Source | État |
|---|---|---|
| KM 4,62 / 6,73 / 8,68 / 15,50 / 17,54 % | `survival_km_horizons.csv` | ✓ |
| Médiane KM non atteinte ; 31,82 mois parmi les événements | `survival_analysis_summary.json` | ✓ |
| Log-rank 23,448 ; p = 3,26 × 10⁻⁵ | idem → `logrank` | ✓ |
| Table de Cox complète (8 lignes, RR, IC, p) | `survival_cox_results.csv` | ✓ ligne à ligne |
| Concordance en échantillon 0,626 | `survival_analysis_summary.json` → `cox.in_sample_c_index` (0,6257) | ✓ |
| Diagnostics PH : 70,73 / 7,04 / 6,76 | `survival_ph_diagnostics.csv` | ✓ |
| Validation temporelle 0,606 → 0,479 ; 0,518 en sensibilité | `survival_analysis_summary.json` → `cox.temporal_validation` | ✓ |
| AIC/BIC des cinq familles | `survival_parametric_comparison.csv` | ✓ |
| Quatre bras : 296 / 544 / 853 / 1 332 ; taux et KM | `survival_linkage_sensitivity.csv` | ✓ |
| Bande limite : 280 retirés, 133 événements ; RR 1,780 et 1,616 | `survival_borderline_link_sensitivity.csv` | ✓ |
| Risque « modèle type » : 173 liens (65 + 127) ; KM 2,64 % ; RR 1,541 et 1,692 | `survival_template_risk_sensitivity.csv` | ✓ |
| Détectabilité : SMD +0,470 / +0,285 / +0,262 / +0,187 et moyennes | `survival_selection_diagnostic.csv` | ✓ |
| Modèle de sensibilité : 1,617 / 1,512 / 1,184 (p = 6,0 × 10⁻⁹) | `survival_cox_detectability_sensitivity.csv` | ✓ |
| Probabilités conditionnelles (5 âges × 2 horizons + IC) | `survival_conditional_probabilities.csv` | ✓ |

### 1.4 Volet textuel

| Chiffre publié | Source | État |
|---|---|---|
| 500 avis, 11 classes, quotas, `AI` = 7 | `annotation_class_summary.csv`, `TECHNOLOGY_TAXONOMY_REPORT.md` § 3 | ✓ |
| 459 familles ; 486 groupes sans la règle 2 ; 39 paires fusionnées dont 29 inter-épisodes | idem § 4 ; `nlp_cv_folds.csv` | ✓ |
| Table des six spécifications (0,027 → 0,744) | `model_cv_results.csv`, colonne `oof_macro_f1` | ✓ |
| 0,7442 [0,682 ; 0,791] contre 0,4731 [0,413 ; 0,526] ; différence 0,2711 [0,201 ; 0,340] | `bootstrap_macro_f1_ci.csv`, `bootstrap_paired_differences.csv` | ✓ |
| Métriques par classe (11 lignes) | `per_class_metrics.csv` | ✓ ligne à ligne |
| Triage des erreurs 16 / 7 / 4 / 2 / 1 | `error_analysis.csv` | ✓ |
| Temporel : 393 / 107, 4 familles, 0,6617 et 0,8148 | `temporal_validation_metrics.csv` | ✓ |
| Courbe d'apprentissage 0,434 → 0,747, entraînement ≈ 0,99 | `learning_curve.csv` | ✓ |
| Platt : gain ECE 0,1405, coût 0,0364, budget 0,02 ; ECE déployé 0,3502 | `final_model_config.json`, `logs/build_technology_taxonomy.log` | ✓ |
| Seuils 648 / 412 / 235 / 123 / 28 ; 6,2 % ; maximum 0,97065 | `confidence_cutoff_sweep.csv`, `technology_evidence_summary.json` | ✓ |
| Composition prédite (11 classes) | `technology_composition.csv` | ✓ |
| Pureté moyenne de segment 0,3385 | `technology_evidence_summary.json` → `crosswalk` | ✓ |

### 1.5 Aval et tendances

| Chiffre publié | Source | État |
|---|---|---|
| Barrières : F1 ≥ 0,65, support ≥ 10 ; ≥ 100 épisodes, ≥ 20 événements | `technology_evidence_summary.json` → `survival.gate` | ✓ |
| 5 classes retenues ; log-rank 10,2579 ; p = 0,0363 ; 416 événements | idem | ✓ |
| Contraste sans barrière A : p = 0,000119 sur 8 classes et 518 événements | idem → `gate_a_contrast` | ✓ |
| Survie par technologie à 24 mois (5 lignes) | `technology_survival_summary.csv` | ✓ |
| Tendances technologiques : plus petit p brut 0,0563, Holm 0,2815 | `technology_trend_summary.csv` | ✓ |
| Matrice de signaux CPV (5 lignes, p bruts, Holm, BH) | `trend_signal_matrix.csv` | ✓ |
| Ruptures stables : CPV-32 2020T2, CPV-48 2024T1, CPV-72 2021T1 | `trend_breakpoints.csv`, colonne `stable_across_penalties` | ✓ |
| HMM : 0,750 / 0,992 / 0,594 | `trend_analysis_summary.json` | ✓ |
| 43 trimestres, 2015T2-2025T4 | `trend_quarterly.csv` (recompté) | ✓ recalculé |

---

## 2. Incohérences trouvées dans les brouillons, et leur traitement

| # | Incohérence | Traitement |
|---|---|---|
| 1 | § 8.4 annonçait « quatre ruptures stables » puis en listait trois. | **Corrigé** dans les deux versions : « trois ruptures », avec la mention explicite que ni la série d'ensemble ni CPV-35 n'en portent. Le § 8.5 et la conclusion disaient déjà trois : la cohérence interne est rétablie. |
| 2 | Deux valeurs de macro-F1 coexistent, 0,744 et 0,741, sans explication. | **Clarifié** au § 6.6 : 0,741 est la moyenne des trois plis, 0,744 le score hors pli sur les 500 prédictions mises en commun. Les deux sont exacts et proviennent de fichiers différents ; l'écart est désormais nommé. |
| 3 | Le taux d'appariement de 14,3 % pouvait se lire comme un échec face aux 40-60 % du cadrage. | **Traité au fond** au § 4.4 : le chiffre du cadrage est une hypothèse de planification, jamais une cible ; 14,3 % est la conséquence arithmétique d'une règle de précision d'abord et d'un événement défini comme observable. |
| 4 | Le rapport parle de « 3 800 marchés numériques » dans l'introduction alors que la règle est plus large. | **Traité** : l'introduction conserve la formule courte, mais le § 3.3 énonce la règle exacte et le chiffre de 30,9 %, et toutes les occurrences ultérieures parlent d'« épisodes comportant au moins un lot numérique ». |

Aucune autre divergence n'a été trouvée entre les deux versions linguistiques : les tableaux sont identiques ligne à ligne, et les arrondis suivent la même règle (trois décimales pour les rapports de risques, deux pour les pourcentages, notation scientifique en dessous de 10⁻³).

---

## 3. Audit des revendications

Chaque affirmation substantielle du rapport a été classée. Le tableau ci-dessous donne les catégories et vérifie que la formulation correspond au statut.

| Statut | Exemple de formulation retenue | Contrôle |
|---|---|---|
| **OBSERVÉ** | « la cohorte contient 3 800 épisodes » ; « le SIREN validé manque pour 66,3 % » | Aucune de ces phrases n'emploie « estimé » ou « prédit » ✓ |
| **ESTIMÉ** | « estimée à 4,62 %, IC [3,91 ; 5,24] » | Aucune probabilité n'est publiée sans son intervalle ou sans sa plage de sensibilité ✓ |
| **PRÉDIT** | « classe technologique prédite » ; « le classifieur attribue » | Aucune occurrence de « la technologie du marché est » ✓ |
| **INFÉRÉ** | « accepté comme successeur observable » ; « sous la définition d'événement retenue » | Les 544 liens ne sont jamais appelés renouvellements ✓ |
| **APPUYÉ PAR LA LITTÉRATURE** | « complétude des couples, au sens de Christen et Goiser (2007) » | Chaque référence appuie une méthode, jamais un résultat de ce projet ✓ |
| **DÉCISION DE PROJET** | « le seuil de 0,70 a été fixé a priori » ; « la fenêtre 90-2 920 jours » | Ces choix sont présentés comme des décisions justifiées, non comme des optima ✓ |
| **EXPLORATOIRE** | « signal nominal à surveiller » ; « ne survit pas à la correction » | CPV-48 n'apparaît jamais sans cette qualification ✓ |

### Les six conversions interdites, vérifiées une par une

| Conversion à ne jamais faire | Vérification |
|---|---|
| successeur observable → renouvellement juridique | Recherche plein texte de « renouvellement » : toutes les occurrences sont soit dans le rappel du cadrage initial, soit explicitement niées, soit dans l'expression « épaule de renouvellement » qui décrit une forme de courbe. ✓ |
| classe prédite → attribut observé | Toutes les mentions de classes technologiques au niveau cohorte portent « prédite » ou « prédiction ». Le § 6.8 rappelle explicitement que les effectifs ne sont pas des parts de marché. ✓ |
| rapport de risques → effet causal | Le § 5.4 déclare les coefficients « associations descriptives moyennées dans le temps » ; le § 5.7d précise qu'aucune des deux lectures de détectabilité n'est causale. ✓ |
| p brut → tendance robuste | Le § 8.3 publie systématiquement Holm et BH à côté du p brut, et le § 7.3 fait de même pour la famille technologique. ✓ |
| C ≈ 0,5 → prédiction utilisable | Le § 5.5 conclut à l'absence de prédiction individuelle et explique le remplacement par une logique de cohorte ; la conclusion le répète. ✓ |
| score brut → probabilité calibrée | Le § 6.7 nomme le score « non calibré », donne l'erreur de calibration (0,350) et interdit la lecture au premier degré. ✓ |

### Revendications explicitement refusées dans le rapport

Le rapport affirme, en toutes lettres, qu'il **ne peut pas** conclure : que les 544 liens sont des renouvellements confirmés ; que la précision de 0,875 est validée de manière indépendante ; que les étiquettes de la référence sont une annotation humaine spécialisée ; que le rappel de la référence est indépendant du score qu'il évalue ; que les probabilités de survie sont des bornes inférieures ; que les ruptures détectées ont une cause identifiée ; que le corpus technologique dispose d'un accord inter-annotateurs ; que la performance de la classe `AI` a été mesurée ; que le modèle de Cox fournit des prévisions individuelles validées.

---

## 4. Risques résiduels du rapport lui-même

Trois points où un lecteur pressé pourrait sur-lire, et ce qui est en place pour l'en empêcher :

1. **Le chiffre de 4,6 %.** C'est le nombre le plus citable du rapport et le plus conditionnel. Protection : il n'apparaît jamais sans la mention de la définition d'événement, et le tableau des quatre bras figure dans le corps du texte, pas en annexe.
2. **La précision de 0,875.** Protection : l'intervalle l'accompagne partout, le nombre de liens (huit) est donné dans la même phrase au § 4.6, et la revue conservatrice à 0,700 est publiée juste après.
3. **La différence de +0,271 en faveur du texte.** C'est le résultat le plus solide, et le risque est inverse — le sous-vendre. Protection : les trois précautions méthodologiques (mêmes plis, même budget, bootstrap de familles) sont énoncées avec le chiffre, de sorte que la solidité soit visible.

---

## 5. Éléments non vérifiables depuis le dépôt

Trois affirmations du rapport ne peuvent pas être tracées vers un artefact et relèvent de la rédaction :

- la description du rôle de Gigalis et de la centrale d'achat, reprise du cadrage de stage ;
- les dates, noms et informations de couverture, à compléter ;
- l'affirmation que l'exécution complète du pipeline reproduit les chiffres à l'identique : elle repose sur le manifeste du 20 août 2026 et sur les empreintes SHA-256 qu'il contient, non sur une réexécution effectuée pendant la rédaction de ce rapport.

---

## 6. Deuxième passe : lecture du code source et inspection visuelle des figures (21 août 2026)

La première passe s'appuyait sur la documentation du dépôt. Cette seconde passe a lu les modules d'implémentation (`boamp_pipeline/linkage.py`, `scripts/build_linkage_candidates.py`, `scripts/evaluate_linkage.py`, `scripts/build_survival_cohort.py`, `boamp_pipeline/technology_evidence.py`) et **ouvert les figures** plutôt que de les choisir sur leur nom de fichier.

### 6.1 Ce que le code confirme

Le code correspond à la documentation sur tous les points substantiels vérifiés : règle de blocage acheteur, fenêtre 90-2 920 jours, exclusion des SIREN validés conflictuels, règle de décision `M_B` (`text_component ≥ seuil`, top-1 par ancre), poids `M_C` (0,50 / 0,25 / 0,20 / 0,05), renormalisation sur les preuves disponibles, `M_A` sans seuil, filtre CPV « à au moins un code », départage du segment par division de plus petit numéro, censure au 31/12/2025.

### 6.2 Quatre précisions ajoutées au rapport après lecture du code

| Précision | Pourquoi elle compte |
|---|---|
| Le vectoriseur ajusté par bloc a une **raison** : les poids IDF deviennent locaux au vocabulaire de l'acheteur, ce qui empêche le formulaire administratif d'un acheteur de dominer la similarité. | Le confondant de détectabilité du § 5.7d n'est pas un défaut d'implémentation : c'est le **coût** d'un choix qui a un bénéfice. Le rapport le présente désormais ainsi. |
| Le plancher de 90 jours est calé sur des preuves chiffrées : 132 des 628 liens alors acceptés tombaient sous trois mois, médiane de 1,7 mois pour les attributions 2025 ; le plus court écart confirmé est de 139 jours. | Transforme « une pathologie observée » en une décision vérifiable. |
| La durée déclarée **entre bien** dans la composante temporelle du score pondéré de `M_C` — valeur 1 à l'échéance, décroissance linéaire jusqu'à zéro un an plus tard, `NaN` en l'absence de durée fiable — mais jamais dans la règle primaire ni comme filtre dur. | La première rédaction disait « durée écartée », ce qui était trop court. La formulation exacte est désormais au § 3.4. |
| La conception initiale substituait quatre ans lorsque l'échéance était inconnue ; cette substitution a été retirée explicitement du code. | Preuve documentaire directe de la progression « intention initiale → investigation → décision », que le § 3.4 cite désormais. |

### 6.3 Ce que l'inspection des figures a changé

| Constat | Décision |
|---|---|
| `reports/figures/survival_kaplan_meier.png` est une figure **à deux panneaux** qui contient déjà le Kaplan-Meier global *et* les courbes par segment. | Les figures 2 et 3 initialement prévues fusionnent en une seule. Le corps du rapport passe de sept à six figures. |
| Le panneau droit de cette figure a un **axe tronqué à 0,5**, ce qui amplifie visuellement l'écart entre segments. | Signalé dans la légende. |
| L'entonnoir de sélection est tracé en **échelle logarithmique**. | Signalé dans la légende : sans cette mention, un lecteur lit un facteur 4 là où le facteur réel est 38. |
| `technology_composition.png` **ne montre pas** la composition par division CPV, contrairement à ce que son nom suggérait lors du premier choix. Elle montre les effectifs prédits sur toute la cohorte, avec la part franchissant le seuil de confiance. | Légende entièrement réécrite. L'argument « une division CPV mélange plusieurs technologies » est désormais porté par un **tableau croisé** ajouté au § 6.5, construit sur `technology_cpv_crosswalk.csv`. |
| `benchmark_validation_m_b_threshold_tradeoff.png` a **trois panneaux** et devient dense à la taille du rapport. | Conservée, avec une note de lecture orientant vers le panneau gauche ; les deux autres peuvent être rognés sans perte d'argument. |
| La figure de sensibilité montre que les quatre bras se déplacent verticalement **en conservant la même forme**, épaule de 40-48 mois comprise. | Observation ajoutée au § 5.7a : le niveau dépend de la règle, la forme non. C'est un renforcement du résultat, découvert en regardant la figure. |
| **Défaut réel.** Le sous-titre de `technology_kaplan_meier.png` annonce « classes clearing the support gate only » alors que la figure trace les **cinq** classes franchissant *les deux* barrières ; l'ensemble « barrière statistique seule » en compte huit. | Le littéral a été corrigé dans `boamp_pipeline/technology_evidence.py` (ligne 1043). **L'image existante conserve l'ancien sous-titre jusqu'à la prochaine exécution du pipeline** : si elle est reprise telle quelle, la légende doit le rectifier. Le défaut est cosmétique mais il porte sur la barrière A, qui est l'argument d'intégrité méthodologique le plus fort du projet. |

### 6.4 Deux commentaires de code périmés (aucun résultat en dépend)

1. `time_plausibility_score` indique que l'hypothèse de quatre ans fabriquerait l'entrée temporelle la plus influente « pour 63,8 % de la cohorte ». La cohorte actuelle donne **74,9 %** de durées non fiables. Le chiffre du commentaire date d'une version antérieure de la cohorte.
2. La constante `MAX_GAP_DAYS` cite des liens confirmés couvrant « 163-2 644 jours » tandis que `MIN_GAP_DAYS` cite un écart minimal de « 139 jours ». Les deux se réfèrent vraisemblablement à des ensembles de référence différents, mais aucun des deux commentaires ne le dit.

Ni l'un ni l'autre n'affecte un résultat publié : ce sont des commentaires, pas des paramètres. Ils sont signalés pour que la correction se fasse à la source plutôt que dans le rapport.
