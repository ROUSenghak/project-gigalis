# Légendes des figures et tableaux / Figure and table captions

Bilingue : chaque entrée donne la légende française puis l'anglaise. Les chemins renvoient aux fichiers du dépôt.

---

## Figures du corps du rapport (6)

*Sept figures étaient prévues ; l'inspection visuelle a permis d'en fusionner deux (Kaplan-Meier global et par segment existent déjà en une seule figure à deux panneaux). Six figures portent donc le corps du rapport.*

### Figure 1 — Entonnoir de sélection de la cohorte
`data/processed/boamp/figures/cohort_and_data_quality/01_selection_funnel.png` — § 3.3

**FR.** *Entonnoir de sélection, des épisodes du Grand Ouest à la cohorte d'étude.* Chaque barre correspond à un critère appliqué dans l'ordre : périmètre géographique, présence d'au moins un code CPV numérique, existence d'un avis d'attribution, résolution de la date d'attribution. **L'axe est logarithmique** : la première barre représente 38 fois la dernière, et non quatre fois comme la longueur relative pourrait le suggérer. Les 26 épisodes perdus à la dernière étape n'ont pas de date exploitable et sont écartés plutôt que datés par défaut.

**EN.** *Selection funnel, from Grand Ouest episodes to the study cohort.* Each bar is one criterion applied in order: geographic scope, at least one digital CPV code, an award notice, a resolvable award date. **The axis is logarithmic**: the first bar represents 38 times the last, not four times as the relative lengths might suggest. The 26 episodes lost at the final step have no usable date and are dropped rather than dated by default.

---

### Figure 2 — Temps jusqu'au successeur observable, ensemble et par segment
`reports/figures/survival_kaplan_meier.png` — § 5.2

*Cette figure à deux panneaux remplace les deux figures séparées initialement prévues : elle porte le même argument en un seul objet, avec le rappel de la définition d'événement en sous-titre.*

**FR.** *Kaplan-Meier du temps jusqu'à un successeur observable, sous la règle figée `M_B @ 0,70`, censuré au 31/12/2025.* Panneau gauche, l'ensemble de la cohorte : la courbe ne descend jamais sous 0,5, donc la médiane n'est pas atteinte, et le décrochement se situe entre 40 et 48 mois. Panneau droit, par division CPV : CPV-35 décroche des trois autres. **Attention à l'axe du panneau droit, qui commence à 0,5 et non à 0** — l'écart entre segments y paraît plus grand qu'il ne l'est sur l'échelle complète. Les effectifs à risque, indispensables au-delà de 60 mois, figurent en annexe D.

**EN.** *Kaplan-Meier time to an observable successor under the frozen `M_B @ 0.70` rule, censored at 2025-12-31.* Left panel, the whole cohort: the curve never falls below 0.5, so the median is not reached, and the drop sits between 40 and 48 months. Right panel, by CPV division: CPV-35 separates from the other three. **Note that the right panel's axis starts at 0.5, not 0** — the separation looks larger there than on the full scale. At-risk counts, essential beyond 60 months, are in Annex D.

---

### Figure 3 — Probabilités conditionnelles à 12 et 24 mois
`data/processed/boamp/figures/survival_analysis/09_conditional_probability_intervals.png` — § 5.3

**FR.** *Probabilité qu'un successeur observable apparaisse dans les 12 ou 24 mois, selon l'âge atteint par le marché, avec intervalles bootstrap à 500 tirages.* Le profil n'est pas monotone : il culmine autour de 36 mois puis retombe. C'est le livrable opérationnel de l'étude — il classe des âges et des segments, il ne prédit pas un marché donné. La largeur des intervalles doit être lue en même temps que les points.

**EN.** *Probability that an observable successor appears within the next 12 or 24 months, by the age the contract has reached, with 500-draw bootstrap intervals.* The profile is not monotone: it peaks around 36 months and falls away. This is the study's operational output — it ranks ages and segments, it does not predict an individual contract. Interval width must be read alongside the point estimates.

---

### Figure 4 — Sensibilité à la définition de l'événement
`data/processed/boamp/figures/survival_analysis/05_linkage_sensitivity.png` — § 5.7

**FR.** *Courbes de survie sous les quatre règles d'appariement retenues.* Le nombre d'événements passe de 296 à 1 332 selon la règle, et les niveaux absolus se déplacent en conséquence. C'est la figure la plus importante du rapport pour l'interprétation : elle montre pourquoi aucune probabilité absolue ne peut être citée seule, alors même que l'ordre des segments reste stable.

**EN.** *Survival curves under the four retained linkage rules.* The event count moves from 296 to 1,332 depending on the rule, and absolute levels shift accordingly. This is the report's most important figure for interpretation: it shows why no absolute probability can be quoted alone, even though the ordering of segments remains stable.

---

### Figure 5 — Arbitrage précision-rappel au seuil d'acceptation
`reports/figures/benchmark_validation_m_b_threshold_tradeoff.png` — § 4.4

**FR.** *Précision, rappel, taux de faux positifs et nombre de liens acceptés en fonction du seuil de la méthode M_B, sur le sous-échantillon verrouillé.* Le seuil retenu, 0,70, a été fixé avant la lecture de cette figure ; c'est la seule raison pour laquelle le sous-échantillon peut être présenté comme tenu à l'écart. La figure montre qu'un seuil de 0,60 achèterait du rappel au prix de la précision ; elle n'a pas servi à choisir.

**EN.** *Precision, recall, false-positive rate and number of accepted links as a function of the M_B threshold, on the recorded locked stratum.* Project history shows that this evidence informed the retained 0.70 policy, so the figure is internal validation rather than an untouched holdout. It shows that 0.60 would buy recall at the cost of precision; 0.60 remains a sensitivity arm, and replacing the frozen post-development policy requires fresh independent evidence. **Reading note: the figure has three panels and is dense at report size — direct the reader to the left panel, where the three curves cross and the frozen threshold is marked; the middle and right panels are secondary and could be cropped for the main text without loss.**

---

### Figure 6 — Composition technologique prédite de la cohorte
`reports/figures/technology_composition.png` — § 6.8

*Correction apportée après inspection de la figure : elle ne montre pas la composition **par division CPV**, contrairement à ce qu'une lecture du seul nom de fichier laisserait croire. L'argument « une division CPV mélange plusieurs technologies » est porté par le tableau croisé du § 6.5, pas par cette figure.*

**FR.** *Effectifs prédits par classe technologique sur les 3 800 épisodes de la cohorte, avec en surimpression la part qui franchit le seuil de confiance de 0,70.* Deux lectures. La composition : `NETWORK_TELECOM` et `BUSINESS_SOFTWARE` dominent à effectifs quasi identiques, tandis que `AI` compte six épisodes en onze ans. Et la couverture : la portion foncée est faible partout, ce qui rappelle que le score de confiance est conservateur et que 6,2 % seulement des épisodes franchissent le seuil opérationnel. **Ces effectifs sont des prédictions portant le taux d'erreur du § 6.6, sur une cohorte définie par une règle CPV inclusive : ce ne sont pas des parts de marché.**

**EN.** *Predicted counts by technology class over the 3,800 cohort episodes, with the share clearing the 0.70 confidence cutoff overlaid.* Two readings. Composition: `NETWORK_TELECOM` and `BUSINESS_SOFTWARE` dominate at almost identical size, while `AI` holds six episodes across eleven years. And coverage: the dark portion is small everywhere, a reminder that the confidence score is conservative and that only 6.2 % of episodes clear the operational cutoff. **These counts are predictions carrying the error rate of § 6.6, over a cohort defined by an inclusive CPV rule: they are not market shares.**

---

## Tableaux du corps du rapport

| Nº | Titre (FR) | Title (EN) | § |
|---|---|---|---|
| T1 | Entonnoir de sélection de la cohorte | Cohort selection funnel | 3.3 |
| T2 | Complétude des champs et décisions de traitement | Field completeness and treatment decisions | 3.4 |
| T3 | Périmètre : Grand Ouest contre Pays de la Loire seuls | Scope: Grand Ouest versus Pays de la Loire alone | 3.5 |
| T4 | Récapitulatif des grains d'analyse | Summary of analytical grains | 3.6 |
| T5 | Les quatre méthodes d'appariement comparées | The four linkage methods compared | 4.3 |
| T6 | Performance sur le sous-échantillon verrouillé, comptabilité du successeur exact | Locked-split performance, exact-successor accounting | 4.6 |
| T7 | Kaplan-Meier : probabilité de successeur par horizon | Kaplan-Meier: successor probability by horizon | 5.2 |
| T8 | Probabilités conditionnelles à 12 et 24 mois par âge | Conditional 12- and 24-month probabilities by age | 5.3 |
| T9 | Modèle de Cox : rapports de risques et intervalles | Cox model: hazard ratios and intervals | 5.4 |
| T10 | Validation temporelle hors période | Out-of-time temporal validation | 5.5 |
| T11 | Comparaison des familles paramétriques (AIC/BIC) | Parametric family comparison (AIC/BIC) | 5.6 |
| T12 | Sensibilité : quatre définitions de l'événement | Sensitivity: four event definitions | 5.7a |
| T13 | Diagnostic de détectabilité et modèle de sensibilité | Detectability diagnostic and sensitivity model | 5.7d |
| T14 | Comparaison des spécifications de classification | Classification specification comparison | 6.4 |
| T15 | Texte contre CPV : différence appariée de macro-F1 | Text versus CPV: paired macro-F1 difference | 6.5 |
| T15b | Tableau croisé division CPV × classe technologique prédite | CPV division × predicted technology crosswalk | 6.5 |
| T16 | Performance par classe technologique avec support | Per-class technology performance with support | 6.6 |
| T17 | Survie par classe technologique après les deux barrières | Survival by technology class after the two gates | 7.2 |
| T18 | Matrice de signaux de tendance avec correction de multiplicité | Trend signal matrix with multiplicity correction | 8.3 |

**Règle de légende appliquée à tous les tableaux.** Chaque légende dit ce que compte une ligne, sous quelle définition d'événement le cas échéant, et ce que le tableau ne permet pas de conclure. Exemple pour T6 : « Performance de quatre méthodes d'appariement sur les 72 ancres exploitables du sous-échantillon verrouillé, en comptabilité du successeur exact : un successeur erroné compte à la fois comme faux positif et comme faux négatif. Les intervalles se recouvrent largement ; ce tableau ne sépare pas les méthodes de façon décisive et ne constitue pas une validation externe. »

**Caption rule applied to every table.** Each caption states what one row counts, under which event definition where relevant, and what the table does not support concluding. Example for T6: "Performance of four linkage methods on the 72 usable anchors of the locked split, under exact-successor accounting: a wrong successor counts as both a false positive and a false negative. The intervals overlap heavily; this table does not decisively separate the methods and is not external validation."

---

## Figures reléguées en annexe, et pourquoi

| Figure | Annexe | Raison |
|---|---|---|
| Matrices de confusion des quatre méthodes | C | Diagnostic ; le tableau T6 porte l'argument plus précisément |
| Courbe ROC au niveau couple | C | Secondaire : les positifs sont rares (16 sur 20 917), la courbe précision-rappel est plus informative |
| Courbe précision-rappel au niveau couple | C | Diagnostic de classement, pas de mesure d'exactitude ; couples non indépendants |
| Métriques des méthodes, pilote et verrouillé | C | Redondant avec T6 pour le corps du texte |
| Ajustement linéarisé et paramétrique contre empirique | D | Justifie un choix déjà énoncé en T11 ; utile à qui veut vérifier |
| Durée déclarée contre délai observé | D | Appuie une décision (§ 3.4) déjà chiffrée dans le texte |
| Biais d'appariement (liés contre censurés) | D | Le tableau de détectabilité T13 est plus lisible |
| Matrice de confusion de la classification | E | Onze classes : illisible à la taille du corps de texte |
| Courbe d'apprentissage | E | Diagnostic de variance, cité en une phrase au § 6.7 |
| Couverture par niveau de confiance | E | Diagnostic opérationnel, sans rôle dans l'argument principal |
| Complétude des durées par année | B | Preuve d'une décision de traitement, pas un résultat |
| Comptes trimestriels par segment | F | La matrice de signaux T18 dit l'essentiel ; la série brute est un support |
| Kaplan-Meier par technologie | E | Le tableau T17 est plus lisible que cinq courbes rapprochées. **Défaut à connaître : le sous-titre du PNG actuel indique « classes clearing the support gate only » alors que la figure trace les cinq classes franchissant *les deux* barrières. Le littéral a été corrigé dans `technology_evidence.py` le 2026-08-21 ; l'image ne portera le bon sous-titre qu'après la prochaine exécution du pipeline. Si la figure est reprise telle quelle, la légende doit rectifier le sous-titre.** L'axe commence à 0,5 |
