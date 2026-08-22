# Réponse au retour du tuteur — éléments de justification

Date : 21 août 2026
Objet : reprise point par point des remarques, avec les éléments du travail et les résultats qui les documentent
Sources : dépôt `project-gigalis` (rapports, tables matérialisées `data/processed/boamp/`, notebooks 13/14/15)

---

## Tableau de synthèse

| # | Remarque du tuteur | Statut | Preuve principale |
|---|---|---|---|
| 1 | Titre et sous-titre à préciser | À appliquer (forme) | — |
| 2 | Encadré « Mon interprétation » à professionnaliser | À appliquer (forme) | — |
| 3 | « Pas encore certain du bon niveau » → arbitrages méthodologiques | À appliquer (forme) | — |
| 4 | Documenter la logique PdL → Grand Ouest | **Chiffré ici**, à intégrer au rapport | recalculs § 4 |
| 5 | Documenter l'inclusion de 2025 et sa complétude | **Chiffré ici**, à intégrer au rapport | recalculs § 5 |
| 6 | Bonne reformulation : « successeur observable » | Déjà en place | `DATA_QUALITY_REPORT.md` § définition de l'événement |
| 7 | Architecture de linkage (large → conservateur, pas de filtre dur CPV/durée) | Déjà en place et justifiée empiriquement | `CANDIDATE_GENERATION_AUDIT.md`, notebook 13 § 9 |
| 8 | Vigilance sur la validation du linkage ; revue manuelle indépendante pré-spécifiée | **Protocole déjà écrit et échantillon prêt**, reviewer humain manquant | `INDEPENDENT_LINK_REVIEW_PROTOCOL.md`, `REVIEW_AUDIT_RESULTS.md` |
| 9 | Kaplan-Meier correctement interprété, probabilités conditionnelles utiles | Déjà en place | `SURVIVAL_ANALYSIS_REPORT.md` § probabilités opérationnelles |
| 10 | Cox : validation temporelle faible, pas de « top 20 » artificiel | Déjà en place — mais un fichier à renommer | § 8 ci-dessous |
| 11 | Modèles paramétriques : garder KM sur horizons observés | Déjà en place | `survival_parametric_comparison.csv` |
| 12 | Robustesse = point fort, à conserver dans le livrable | Déjà en place (4 bras + 2 contre-épreuves) | `survival_linkage_sensitivity.csv`, § template-risk |
| 13 | Tendances : ne pas empiler les modèles, CPV-48 = motif d'investigation | Déjà appliqué (correction de multiplicité) | `TREND_ANALYSIS_REPORT.md` |
| 14 | Volet NLP / taxonomie technologique peu couvert | **Livré depuis** : 11 classes, TF-IDF + régression logistique | `TECHNOLOGY_TAXONOMY_REPORT.md` |
| 15 | Prédiction individuelle → expérience séparée, pré-spécifiée | Accord ; protocole proposé § 12 | — |
| 16 | Hiérarchisation du message, dédoublonnage des limites | À faire (rédaction) | — |
| 17 | Synthèse exécutive d'une page en 5 questions | **Existe déjà**, à réordonner sur ses 5 questions | `EXECUTIVE_SUMMARY.md` |

Trois remarques appellent réellement du travail neuf : le cadrage écrit du périmètre (4, 5), la revue humaine indépendante (8) et la réécriture hiérarchisée (16, 17). Le reste est soit de la forme, soit déjà instrumenté dans le dépôt.

---

## 1-3. Remarques de forme

Aucune objection ; je reprends les formulations proposées.

- Titre retenu : **« Analyse et modélisation des marchés publics numériques à partir des données du BOAMP »**, sous-titre inchangé pour la note intermédiaire.
- Encadré renommé **« Statut du document et précautions d'interprétation »**.
- Reformulation de l'hésitation en arbitrage : la note listera explicitement les arbitrages restants — profondeur de la validation du linkage, granularité de la segmentation métier, périmètre de la partie tendances — plutôt qu'une incertitude sur « le bon niveau d'approfondissement ».

La distinction résultat / interprétation / incertitude est déjà structurelle dans le dépôt : chaque résultat des rapports techniques est écrit sous la forme *Observation → Confiance → Ce que l'on peut conclure → Ce que l'on ne peut pas conclure* (voir `TECHNOLOGY_TAXONOMY_REPORT.md`, Résultats 2 à 8). C'est cette grille que la remarque n° 16 demande de généraliser à toute la note.

---

## 4. Pourquoi le Grand Ouest, et non les Pays de la Loire seuls

La logique demandée (PdL → volume insuffisant → extension → conséquences) se chiffre exactement. Recalcul fait sur `survival_dataset.parquet` (cohorte figée) :

| Périmètre | Épisodes | Événements | Taux | P(successeur ≤ 12 m) | IC 95 % | Largeur IC |
|---|---:|---:|---:|---:|---|---:|
| Grand Ouest | 3 800 | 544 | 14,3 % | 4,62 % | [3,94 ; 5,30] | 1,37 pt |
| Pays de la Loire seuls | 1 452 | 219 | 15,1 % | 5,36 % | [4,17 ; 6,55] | 2,38 pt |

**Ce qui motive l'extension.** Ce n'est pas le volume global — 1 452 épisodes restent analysables — mais la **granularité** exigée par les questions posées :

- l'intervalle de confiance à 12 mois est **1,7 fois plus large** en PdL seuls (2,38 pt contre 1,37 pt), et 1,74 fois à 24 mois ;
- le segment qui porte le résultat comparatif le plus robuste de l'étude, **CPV-35, ne compte que 34 événements en PdL seuls contre 115 sur le Grand Ouest** ;
- le modèle de Cox compte 8 covariables : 68 événements par paramètre sur le Grand Ouest contre **27 en PdL seuls**, sous le seuil usuel de prudence pour des modèles de survie à covariables multiples ;
- le test du log-rank entre segments reste significatif en PdL seuls (statistique 14,68, p = 0,0021) mais avec une statistique 1,6 fois plus faible : le signal existe, il n'est simplement plus assez résolu pour supporter les déclinaisons par segment × âge qui constituent le livrable opérationnel.

**Conséquences sur l'interprétation, à écrire noir sur blanc.** L'extension n'est pas neutre : la région est une covariable du modèle et les trois régions ne se comportent pas identiquement. Taux d'événement observé : Bretagne 15,5 %, Pays de la Loire 15,1 %, Normandie 12,5 %. Dans le Cox, **Pays de la Loire vs Bretagne HR = 1,003 (p = 0,81)** — aucune différence mesurable — mais **Normandie HR = 0,800 (p = 0,046)**. Le regroupement est donc défendable pour les deux régions les plus proches et **ajusté explicitement** pour la Normandie ; les résultats agrégés doivent être lus comme du « Grand Ouest », pas comme un proxy des Pays de la Loire, et toute lecture Gigalis‑centrée se fait sur la ligne PdL du modèle, pas sur la moyenne régionale.

---

## 5. Pourquoi 2025, et avec quelle réserve

**Complétude en volume : oui.** Les attributions 2025 s'inscrivent dans la normale du corpus : 93 / 73 / 73 / 87 épisodes sur les quatre trimestres 2025, contre 72 / 57 / 72 / 85 en 2024. Il n'y a pas de trou de publication ; la coupure d'observation est fixée au 31/12/2025 et 2025 représente 8,6 % de la cohorte (326 épisodes).

**Complétude en suivi : non, et c'est là qu'il faut être explicite.** Un marché attribué en 2025 dispose d'un suivi médian de **5,9 mois** contre 59,9 mois pour 2015-2024. Il ne peut mécaniquement contribuer que des successeurs à écart court (écart maximal observé : 11,4 mois), et le 4e trimestre 2025 ne contient **aucun** événement par construction.

**Impact mesuré sur les analyses temporelles.** Recalcul en retirant les attributions 2025 :

| Cohorte | n | Événements | P(≤ 12 m) | P(≤ 24 m) |
|---|---:|---:|---:|---:|
| 2015-2025 (retenue) | 3 800 | 544 | 4,62 % | 6,73 % |
| 2015-2024 | 3 474 | 499 | 3,54 % | 5,68 % |

L'écart n'est pas nul : inclure 2025 relève l'estimation à 12 mois d'environ 1,1 point, parce que 45 événements à écart court entrent avec très peu de temps à risque. **Ce n'est donc pas un choix anodin et il sera présenté comme une sensibilité, pas comme un détail.** Deux protections sont déjà en place dans le travail :

1. la validation temporelle **principale** du Cox est la fenêtre alignée sur le cadrage initial, entraînement 2015-2021 → test **2022-2024** ; la variante 2022-2025 n'est portée qu'en lecture de sensibilité (C-index 0,479 vs 0,518) ;
2. la rupture de mesure de 2025 est déjà documentée : la complétude du champ durée passe de 30,4 % (2024) à **84,4 %** (2025). C'est un changement de schéma, pas un changement de comportement d'achat ; `TREND_ANALYSIS_REPORT.md` refuse pour cette raison toute détection de rupture sur les valeurs de durée.

**Justification retenue :** 2025 est conservé parce qu'il apporte 326 épisodes d'exposition censurée qui améliorent l'estimation des horizons courts et parce que la coupure au 31/12/2025 est propre ; il est signalé comme partiel en suivi, exclu de la fenêtre de validation principale, et accompagné du tableau de sensibilité ci-dessus.

---

## 6. Le « successeur observable » comme objet mesuré

C'est bien le pivot du travail, et il est verrouillé dans la définition de l'événement :

> `event = 1` lorsqu'un épisode postérieur du même acheteur est **accepté** par la règle de linkage figée ; `event = 0` lorsqu'aucun successeur n'est accepté avant le 31/12/2025, la ligne étant alors censurée à droite — ce qui **ne prouve pas un abandon**.

Le BOAMP n'encode pas le renouvellement juridique ; toute la chaîne de rédaction est cohérente avec cela (`INTERNSHIP_GUIDE_COMPLIANCE.md` §« Claims not yet allowed » interdit explicitement la phrase « les 544 liens sont des renouvellements juridiques confirmés »).

---

## 7. Architecture du linkage : génération large, acceptation conservatrice

**Génération large.** Blocage sur acheteur plausible + fenêtre 90–2 920 jours, sans présomption d'expiration.

**Pas de filtre CPV dur — argument empirique, pas de confort.** Parmi les 544 liens acceptés, 34,8 % traversent une division CPV. Or les successeurs de la référence régionale, labellisés **sans aucune connaissance des méthodes de linkage**, traversent une division dans 39,1 % des cas (9 sur 23). Imposer un blocage CPV strict détruirait donc 9 des 23 successeurs de référence et ferait tomber le plafond de rappel atteignable de **0,913 à 0,609**.

**Pas de durée déclarée en filtre dur — argument empirique également.** La durée fiable manque pour 74,9 % de la cohorte, et surtout : parmi les marchés disposant à la fois d'une durée fiable et d'un successeur accepté, **seuls 21,8 % sont remis en concurrence dans les six mois suivant la fin déclarée**, avec un écart absolu médian de **21,1 mois**. La durée déclarée est un mauvais prédicteur du calendrier réel ; elle est conservée comme diagnostic descriptif, pas comme contrainte.

**Précision plutôt que rappel.** L'argument que vous formulez (un faux lien crée à la fois un faux événement *et* un faux temps d'événement) est exactement celui retenu. Sur le split verrouillé, `M_B_text_ranking @ 0,70` affiche précision 0,875 et **taux de faux positifs 0,000**, contre 0,522 de précision pour `M_C` qui gagne pourtant du rappel (0,667 vs 0,389).

---

## 8. Validation du linkage : ce qui existe, ce qui manque

**Ce que je reconnais sans réserve** — et qui est déjà écrit dans le dépôt :

- la référence régionale (120 ancres, 112 résolues, split verrouillé de 72) a des **labels produits par une passe de recherche LLM, contrôlés par sondage** et non vérifiés ancre par ancre : ce n'est pas une vérité terrain humaine indépendante ;
- la précision de 0,875 repose sur **8 liens acceptés** seulement, IC 95 % [0,529 ; 0,978] — un intervalle qui interdit toute revendication de performance ;
- pire, et c'est documenté dans `REGIONAL_BENCHMARK_REFERENCE.md` : la règle qui a sélectionné les ~25 candidats exportés par ancre (sur des pools allant jusqu'à 3 258) **n'est pas enregistrée** et tous les successeurs retrouvables se classent en tête du score textuel de production. Le **rappel** et le plafond de 0,913 ne sont donc **pas indépendants du score qu'ils évaluent**. La précision, elle, n'est pas affectée : un faux positif reste un faux positif quelle que soit la façon dont la liste a été constituée.
- la revue de challenge sur 60 paires est **assistée par modèle**, pas humaine indépendante : sur les 20 liens acceptés, 14 confirmés, 5 infirmés, 1 incertain → précision conservatrice **0,700** (IC [0,457 ; 0,881]), **sous la cible de 0,80** annoncée. C'est écrit tel quel dans `REVIEW_AUDIT_RESULTS.md`, sans arrondi favorable.

**Votre recommandation est déjà instrumentée.** Le protocole que vous décrivez existe : `INDEPENDENT_LINK_REVIEW_PROTOCOL.md` définit un échantillon **aveugle** de 60 paires (20 liens acceptés en production, 20 négatifs structurels à forte similarité, 20 relations déclarées par l'acheteur), un fichier reviewer sans label ni score ni strate, une clé d'audit séparée, un schéma de décision en 4 champs, et une **règle d'acceptation pré-spécifiée** : geler `M_B @ 0,70` pendant la revue, calculer des intervalles binomiaux exacts, et interdiction d'ajuster le seuil sur ces mêmes lignes puis de les présenter comme validation.

**Ce qui manque est donc une personne, pas une méthode.** Je souscris entièrement à votre arbitrage : une revue humaine indépendante supplémentaire a plus de valeur qu'une cinquième méthode de linkage. Objectif retenu : **qualifier la confiance dans la règle actuelle, sans la recalibrer**. Deux réserves à porter au protocole avant relance : le tirage initial n'est plus reproductible (deux de ses trois strates venaient du benchmark national retiré le 15/08), et la strate exploitable comme preuve active est la seule strate `PRIMARY_ACCEPTED`. Un nouveau tirage devra donc être re-spécifié et documenté avant exécution.

---

## 9. Kaplan-Meier et probabilités conditionnelles

Lecture conforme à ce que vous validez :

- probabilité de successeur **observable** — jamais « probabilité de renouvellement » ;
- médiane KM **non atteinte** ; les 31,8 mois circulant ailleurs sont la médiane **parmi les événements liés uniquement**, et le rapport le signale explicitement pour éviter la confusion ;
- séparation entre segments bien alimentée : log-rank 23,45, p = 3,3 × 10⁻⁵ ; CPV-35 à 20,4 % d'événements contre 11,6 % pour CPV-48.

La traduction opérationnelle que vous évoquez est déjà produite sous forme de table conditionnelle, avec intervalles bootstrap à 500 tirages :

| Âge du marché | P(successeur ≤ 12 m) | IC 95 % | P(≤ 24 m) | IC 95 % |
|---:|---:|---|---:|---|
| 0 mois | 4,62 % | [3,91 ; 5,24] | 6,73 % | [5,91 ; 7,58] |
| 12 mois | 2,22 % | [1,70 ; 2,73] | 4,26 % | [3,55 ; 4,98] |
| 24 mois | 2,09 % | [1,58 ; 2,60] | 9,40 % | [8,33 ; 10,68] |
| 36 mois | 7,46 % | [6,52 ; 8,59] | 9,69 % | [8,58 ; 11,12] |
| 48 mois | 2,41 % | [1,77 ; 3,07] | 2,89 % | [2,15 ; 3,72] |

Le profil n'est pas monotone : il monte dans l'épaule de renouvellement à 36-48 mois puis retombe. C'est précisément une **logique de veille par cohorte** (segment × âge), pas un score individuel.

---

## 10. Cox et refus du « top 20 »

**Validation temporelle.** Modèle ajusté une fois sur 2015-2021, scoré hors temps sans réajustement : C-index 0,606 en apprentissage, **0,479** sur la fenêtre 2022-2024 (0,518 en incluant 2025). C'est indiscernable du hasard, et le rapport l'écrit comme un **résultat**, pas comme une invitation à retuner.

**Hypothèse PH.** Rejetée pour `award_year_centered`, `framework_flag` et `has_validated_siren`. Les coefficients sont donc présentés comme des associations descriptives moyennées dans le temps, jamais comme des effets.

**Positionnement à conserver dans la version finale**, tel que vous le formulez : Cox = associations descriptives à l'échelle de la population ; **pas** de moteur de scoring individuel.

**Un point d'honnêteté à signaler.** Il existe dans le dépôt un fichier `data/processed/boamp/renewal_watchlist_top20.csv`. Son **nom est trompeur** et je vais le renommer, car ce n'est pas un top 20 de marchés « les plus susceptibles d'être renouvelés » issu du Cox. C'est une liste **stratifiée par segment** (les 5 marchés les plus proches d'un horizon dans chacun des 4 segments CPV), construite à partir des courbes **Kaplan-Meier segmentées** et non du Cox, restreinte aux attributions ≥ 2021. Le notebook 13 documente explicitement pourquoi un top 20 non stratifié serait dégénéré : la probabilité n'étant fonction que du segment et de l'âge, un tri global renverrait simplement le segment à plus fort risque à l'âge le plus proche du pic — ce qui donnerait une fausse impression de granularité individuelle. La logique est donc bien celle que vous recommandez ; seul le nom du fichier ne l'est pas.

---

## 11. Modèles paramétriques

`GeneralizedGamma` obtient le meilleur AIC parmi exponentielle, Weibull, log-logistique, log-normale et gamma généralisée. Il **n'est pas** la source des chiffres opérationnels : toutes les familles lisses aplatissent l'épaule de renouvellement empirique, et tous les horizons publiés sont à l'intérieur de la fenêtre observée. Les probabilités à 12 et 24 mois sont donc lues sur Kaplan-Meier ; le paramétrique est conservé comme meilleure famille d'ajustement et comme instrument que toute extrapolation au‑delà du 31/12/2025 devrait utiliser. C'est exactement l'arbitrage que vous jugez plus raisonnable qu'une sélection mécanique par AIC/BIC.

---

## 12. Robustesse (que vous identifiez comme point fort)

Le résultat que vous demandez de préserver — niveaux absolus instables, associations relatives plus stables — est matérialisé sur trois épreuves indépendantes.

**a) Quatre bras de linkage.**

| Bras | Événements | Taux | P(≤ 12 m) | P(≤ 24 m) |
|---|---:|---:|---:|---:|
| strict (`M_B @ 0,80`) | 296 | 7,8 % | 2,37 % | 3,23 % |
| principal (`M_B @ 0,70`) | 544 | 14,3 % | 4,62 % | 6,73 % |
| plus large (`M_B @ 0,60`) | 853 | 22,4 % | 8,00 % | 11,47 % |
| contraste haut rappel (`M_C @ 0,70`) | 1 332 | 35,1 % | 12,21 % | 17,98 % |

Un facteur 4,5 sur le nombre d'événements : aucun niveau absolu ne peut être cité seul.

**b) Bande limite (± 0,05 autour du seuil).** Retirer les 280 épisodes limites (dont 133 événements) : KM 12 m 4,62 % → 3,72 % ; **CPV-35 HR 1,553 → 1,780** ; framework 1,751 → 1,616. Direction inchangée.

**c) Re-censure du risque « template ».** Les faux positifs identifiés par l'audit ne viennent pas du seuil mais du **boilerplate juridique des accords-cadres**, sur lequel les n-grammes de caractères notent haut entre objets non liés. 173 liens sur 544 (31,8 %) portent une des deux signatures observables ; ils sont **re-censurés** (et non supprimés) : KM 12 m tombe à 2,64 %, mais **CPV-35 reste à 1,541 et framework à 1,692**. C'est la contre-épreuve dont l'effet accord-cadre avait le plus besoin, puisque c'est ce texte-là qui alimente le mécanisme.

**d) Détectabilité — le point que je considère comme le plus important du travail.** Le plus grand déséquilibre entre épisodes liés et censurés n'est pas une propriété du marché : c'est la **taille du pool de candidats** (SMD +0,470 en log). Un acheteur qui publie beaucoup produit mécaniquement un maximum de score plus élevé. En sensibilité, ajouter log(1 + taille du pool) au Cox :

| Covariable | HR modèle principal | HR + log(pool) | p ajusté |
|---|---:|---:|---:|
| `framework_flag` | 1,751 | **1,617** | 2,6 × 10⁻⁶ |
| `digital_segment_CPV-35` | 1,553 | **1,512** | 8,5 × 10⁻⁴ |
| `log_candidate_pool_size` | — | 1,184 | 6 × 10⁻⁹ |

Lecture : **CPV-35 est le résultat comparatif le plus robuste de l'étude** (stable sur les 4 bras, la bande limite, la re-censure template et l'ajustement de détectabilité). **L'effet accord-cadre est partiellement de la détectabilité**, pas seulement du comportement — environ 14 % du log-HR s'évapore. Cette nuance est portée dans la synthèse exécutive, pas enfouie en annexe.

---

## 13. Tendances : prudence déjà appliquée

D'accord pour ne pas investir davantage. L'état actuel est volontairement sobre et **la correction de multiplicité est déjà en place** :

| Segment | Pente / trimestre | p brut | p Holm | p BH | Lecture |
|---|---:|---:|---:|---:|---|
| Global | −0,11 | 0,921 | 1,000 | 0,989 | aucun signal |
| CPV-32 | −0,01 | 0,989 | 1,000 | 0,989 | aucun signal |
| CPV-35 | +0,03 | 0,923 | 1,000 | 0,989 | aucun signal |
| **CPV-48** | **−0,84** | **0,032** | 0,159 | 0,159 | **signal nominal seulement** |
| CPV-72 | +0,70 | 0,285 | 1,000 | 0,714 | aucun signal |

CPV-48 est donc déjà traité comme **motif d'investigation** et non comme prévision : le rapport recommande explicitement de « l'observer encore quelques trimestres avant d'agir ». Aucune rupture PELT n'est attribuée à une cause (politique, COVID, technologie) sans preuve documentaire, et le modèle HMM est présenté comme une lecture de régime, pas comme une prévision. Les séries monétaires restent exclues faute d'une définition validée du montant attribué.

---

## 14. Volet NLP / taxonomie technologique — livré depuis la note

C'est le point où votre lecture est en retard sur le dépôt : **la taxonomie que vous décrivez a été construite**, et selon exactement l'approche pragmatique que vous recommandez.

**Taxonomie** : 8 classes substantielles — `CLOUD_HOSTING`, `CYBERSECURITY`, `NETWORK_TELECOM`, `IT_INFRASTRUCTURE`, `BUSINESS_SOFTWARE`, `DATA_BI`, `AI`, `IT_SERVICES` — plus 3 classes de repli assumées (`MIXED`, `OTHER_DIGITAL`, `OTHER`). Elle recouvre la liste de votre message. Elle a été **figée avant modélisation**.

**Corpus** : 500 notices annotées manuellement, 2015-2025, règles d'annotation écrites, champ d'entrée = `objet` (médiane 14 mots).

**Baseline exactement celle que vous proposez** : TF-IDF mots + régression logistique pondérée (`class_weight='balanced'`), variante SVM linéaire testée, hyperparamètres choisis par CV interne **dans chaque pli d'entraînement**.

**Anti-fuite** : le BOAMP republie et les acheteurs relancent des tenders quasi identiques. Chaque notice est rattachée à une **famille de marché** (même épisode reconstruit, ou cosinus caractère ≥ 0,80) et chaque famille tient dans un seul pli : 459 familles, 0 famille à cheval sur deux plis.

**Résultats** (validation croisée groupée en 3 plis, hors pli) :

| Modèle | Macro-F1 hors pli | IC 95 % (bootstrap familles) |
|---|---:|---|
| TF-IDF + régression logistique | **0,744** | [0,682 ; 0,790] |
| Benchmark CPV / descripteurs (mêmes plis, même budget) | 0,473 | [0,413 ; 0,526] |
| **Différence appariée** | **+0,271** | [0,201 ; 0,340] — exclut zéro |

C'est la réponse quantitative à votre remarque « la segmentation CPV reste très large » : **le texte porte l'information métier que le CPV ne porte pas**. La pureté moyenne d'un segment CPV vis-à-vis de la taxonomie est de 0,34 — chaque division CPV contient plusieurs technologies métier distinctes.

**Pas de complexité gratuite.** La règle d'usage d'un transformeur a été écrite *avant* de lire les résultats classiques : CamemBERT n'est testé que si le modèle classique est matériellement insuffisant (macro-F1 < 0,55) **et** si moins de la moitié des erreurs viennent d'ambiguïté d'étiquetage. Le modèle atteint 0,741 : la condition échoue, **CamemBERT n'a pas été lancé**. Il n'a pas été testé puis écarté ; le critère pour le lancer n'était pas rempli.

**Robustesse temporelle** : entraînement 2015-2022 → test 2023-2025. Macro-F1 sur les classes à support test ≥ 10 : **0,815**. Le vocabulaire des notices récentes n'a pas dérivé pour les classes à volume.

**Usage en aval derrière deux barrières explicites** : une classe n'alimente une courbe de survie que si (A) elle est substantielle, support ≥ 10 et F1 ≥ 0,65, et (B) elle a assez d'événements. 5 classes sur 11 passent. Résultat : différence de calendrier de re-procurement entre technologies, log-rank p = 0,036 sur 416 événements — `BUSINESS_SOFTWARE` à 8,6 % à 24 mois contre `IT_INFRASTRUCTURE` à 6,1 %.

**Limites que je porte moi-même** : corpus annoté en une seule passe, **aucun kappa de Cohen** disponible (le cadrage L2 demandait deux annotateurs) ; quotas d'annotation ⇒ les proportions ne sont pas des prévalences ; `AI` a 7 notices et est déclarée non évaluable ; la calibration de Platt a été **évaluée et rejetée** par la règle pré-spécifiée (gain ECE 0,140 accepté, mais coût 0,036 de macro-F1 > budget 0,02), donc le score publié est un **score de confiance brut, non calibré** — conservateur, utilisable pour le classement et le filtrage, pas comme probabilité.

**Meilleur investissement restant sur ce volet** : la courbe d'apprentissage est encore montante à n = 500. Une seconde passe d'annotation (kappa + volume) vaut mieux qu'un modèle plus sophistiqué — même arbitrage que celui que vous appliquez au linkage.

---

## 15. Prédiction individuelle : expérience séparée et pré-spécifiée

Accord complet sur le principe : **ne pas retoucher progressivement le Cox actuel jusqu'à obtenir un meilleur chiffre**. Le modèle actuel reste figé tel qu'il est, avec son C-index hors temps de 0,479.

Protocole proposé si cette piste est priorisée, à écrire **avant** de lancer quoi que ce soit :

1. **Variables disponibles au moment de l'attribution uniquement** : contenu textuel de l'objet (classe technologique prédite + score de confiance, déjà disponible pour les 3 800 épisodes), historique de l'acheteur (volume de publication antérieur, ancienneté, récurrence sur le segment), type de procédure, statut accord-cadre.
2. **Validation temporelle déclarée à l'avance** : entraînement 2015-2021, test 2022-2024, un seul passage, aucun réajustement.
3. **Piège à éviter explicitement** : la taille du pool de candidats (HR 1,184, p = 6 × 10⁻⁹) est un canal de **détectabilité**, pas un comportement d'achat. L'inclure ferait monter le C-index en apprenant à prédire *qui est détectable*, pas *qui renouvelle*. Elle doit être traitée en sensibilité, jamais dans la spécification principale — sinon l'expérience se conclut positivement pour la mauvaise raison.
4. **Critère de succès annoncé d'avance** (par ex. C-index hors temps ≥ 0,60), et publication du résultat même s'il est négatif.

Cela permet, comme vous l'écrivez, de conclure proprement sur l'apport réel de ces informations.

---

## 16-17. Hiérarchisation et synthèse exécutive

**La synthèse d'une page existe déjà** (`EXECUTIVE_SUMMARY.md`) mais elle est organisée en « ce que fait le projet / ce qui a été fait / ce qui marche / ce qui reste incertain / prochaines étapes ». Je la réordonne sur vos cinq questions, ce qui donne :

| Question | Contenu |
|---|---|
| **Ce que nous savons** | 1 620 712 notices standardisées → 3 800 épisodes numériques attribués Grand Ouest 2015-2025 ; pipeline reproductible de bout en bout ; le BOAMP n'encode pas le renouvellement juridique, on mesure un **successeur observable** |
| **Ce que les modèles indiquent** | 14,3 % des marchés reçoivent un successeur observable ; 4,6 % à 12 mois, 6,7 % à 24 mois ; épaule de renouvellement à 36-48 mois ; CPV-35 se remet en concurrence le plus vite (HR 1,553), résultat stable sur toutes les épreuves ; le texte porte une segmentation métier que le CPV ne porte pas (+0,271 de macro-F1) |
| **Ce qui reste incertain** | Les niveaux absolus dépendent de la règle de linkage (296 à 1 332 événements) ; la précision du linkage n'est pas validée humainement (0,70 en revue assistée, cible 0,80) ; le Cox ne discrimine pas hors temps (0,479) ; aucun kappa sur le corpus d'annotation ; l'effet accord-cadre est en partie de la détectabilité |
| **Ce que cela peut signifier pour Gigalis** | Un cadre de **mesure et de veille par cohorte** (segment × âge), pas un score par marché : prioriser la surveillance sur CPV-35 et sur la fenêtre 36-48 mois ; disposer d'une segmentation métier en 8 classes exploitable là où le CPV est trop large ; suivre CPV-48 comme signal à confirmer |
| **Ce que nous recommandons ensuite** | (1) revue humaine indépendante des 60 paires aveugles ; (2) seconde passe d'annotation pour obtenir un kappa ; (3) fournir les données d'adhésion Gigalis si la question causale est voulue ; (4) geler linkage/survie/tendances |

**Dédoublonnage.** Vous avez raison, quatre sections disent aujourd'hui des choses proches : « limites », « formulations à éviter », « points de discussion » et « mon interprétation ». Je fusionne en **deux** blocs : (a) *Statut du document et précautions d'interprétation* en tête, (b) *Limites et frontières de revendication* en fin, cette dernière reprenant la liste déjà formalisée dans `INTERNSHIP_GUIDE_COMPLIANCE.md` (« claims allowed » / « claims not yet allowed »), qui est exactement la grille « ce que l'on peut / ne peut pas conclure ».

**Grille à appliquer à chaque résultat** (votre liste en cinq points) : elle est déjà le format natif des rapports techniques (Observation → Confiance → Conclusion possible → Conclusion impossible) ; il manque la cinquième ligne, **« ce que Gigalis peut en faire »**, que j'ajoute systématiquement.

---

## Ce que je retiens comme plan de travail

| Priorité | Action | Justification |
|---|---|---|
| 1 | Réécriture hiérarchisée + synthèse en 5 questions + fusion des sections de limites | Coût faible, valeur immédiate sur la lisibilité |
| 2 | Cadrage écrit du périmètre (PdL → Grand Ouest, inclusion de 2025) avec les tableaux ci-dessus | Demande explicite, chiffres déjà calculés |
| 3 | Revue humaine indépendante d'un échantillon aveugle re-tiré et pré-spécifié | Seul verrou avant toute revendication de précision ; plus utile qu'une 5ᵉ méthode |
| 4 | Renommer `renewal_watchlist_top20.csv` en `veille_par_segment_km.csv` | Le nom suggère un top 20 individuel que le travail refuse justement de produire |
| 5 | Si temps : seconde passe d'annotation technologique (kappa) | Courbe d'apprentissage encore montante à n = 500 |
| 6 | Non prioritaire : tendances, prédiction individuelle | Séries courtes et bruitées ; prédiction individuelle à traiter comme expérience séparée pré-spécifiée |

---

### Fichiers cités

`EXECUTIVE_SUMMARY.md` · `SURVIVAL_ANALYSIS_REPORT.md` · `TREND_ANALYSIS_REPORT.md` · `TECHNOLOGY_TAXONOMY_REPORT.md` · `DATA_QUALITY_REPORT.md` · `REGIONAL_BENCHMARK_REFERENCE.md` · `CANDIDATE_GENERATION_AUDIT.md` · `REVIEW_AUDIT_RESULTS.md` · `INDEPENDENT_LINK_REVIEW_PROTOCOL.md` · `INTERNSHIP_GUIDE_COMPLIANCE.md` · `notebooks/13_survival_analysis.ipynb` · `notebooks/14_data_quality_and_trend_analysis.ipynb` · `notebooks/15_technology_taxonomy_classification.ipynb`

Les chiffres des sections 4 et 5 (Pays de la Loire seuls, effet de 2025) ont été recalculés le 21/08/2026 directement sur `data/processed/boamp/survival_dataset.parquet` ; tous les autres sont repris des tables et rapports figés du dépôt.
