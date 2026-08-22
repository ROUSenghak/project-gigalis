# Page de couverture

> **Brouillon archivé.** Le rapport final faisant autorité est le projet LaTeX
> `rapport/BOAMP_Report_EN_Overleaf/`, avec une note de synthèse française en
> fin de document. Ce brouillon Markdown est conservé pour la provenance et
> n'est pas synchronisé chiffre par chiffre.

> **À composer selon le modèle ENSAE imposé (annexe 1 des consignes).**
> Haut gauche : **{NOM Prénom}** — Haut droite : **ENSAE 2ᵉ année**, *Stage d'application*, *Année scolaire 2025-2026*
> Encadré central :
>
> ### Analyse et modélisation des marchés publics numériques à partir des données du BOAMP
> *Construction d'un successeur observable, analyse de survie et segmentation technologique par le texte*
>
> Bas gauche : **Gigalis** — {Ville}
> Bas droite : **Maître de stage : {Prénom NOM}** — {date de début} au {date de fin}

---

# Sommaire

1. Introduction
2. Contexte et reformulation du problème
3. Données, unités d'analyse et périmètre
4. Construction de la variable d'intérêt : l'appariement des successeurs
5. Analyse de survie
6. Signal textuel : une taxonomie technologique supervisée
7. Usage des prédictions en aval
8. Tendances et détection de ruptures
9. Discussion
10. Limites
11. Conclusion

Bibliographie
Annexes A à G
Note de synthèse (français)
Executive summary (anglais)

---

# 1. Introduction

Gigalis est le groupement d'intérêt public chargé du numérique en Pays de la Loire. Il est notamment centrale d'achat : il construit des accords-cadres mutualisés en cloud, cybersécurité, réseaux et intelligence artificielle, que ses membres utilisent sans y être obligés. La pertinence stratégique de ces contrats dépend donc de la capacité de Gigalis à anticiper les besoins d'achat de ses membres plutôt qu'à les constater. La question posée au stage était la suivante : à partir des données historiques de la commande publique numérique, peut-on estimer la probabilité qu'un marché ou un segment technologique génère un besoin d'achat identifiable dans les douze à vingt-quatre mois ?

Le stage s'est déroulé sur données ouvertes : les avis publiés au Bulletin officiel des annonces des marchés publics (BOAMP), soit 1 620 712 avis de 2015 à 2025, ramenés à une cohorte d'étude de 3 800 marchés numériques attribués dans le Grand Ouest. Les moyens sont ceux d'un poste de travail standard : Python, scikit-learn, lifelines, statsmodels, ruptures, avec un pipeline reproductible de bout en bout.

Le travail a rapidement fait apparaître que la difficulté principale n'était pas d'estimer une durée, mais de définir ce que l'on mesure. Le BOAMP ne comporte aucun champ indiquant qu'un marché renouvelle un marché antérieur. La variable à expliquer devait donc être construite, et cette construction commande tout ce qui suit. Le rapport présente cette construction, l'analyse de survie qu'elle permet, la segmentation technologique obtenue par apprentissage supervisé sur le texte des avis, l'analyse descriptive des tendances, et l'examen critique de ce que ces résultats autorisent — ainsi que de ce qu'ils n'autorisent pas.

---

# 2. Contexte et reformulation du problème

## 2.1 Le problème métier

Une centrale d'achat mutualise la commande. Son intérêt est double : faire baisser les prix par le volume, et éviter à chaque acheteur de refaire la procédure. Les deux supposent d'ouvrir le bon accord-cadre au bon moment. Ouvrir trop tôt, c'est immobiliser un contrat que personne n'utilise ; ouvrir trop tard, c'est laisser les membres passer leurs propres marchés. La question opérationnelle est donc temporelle : quand un besoin déjà satisfait va-t-il réapparaître ?

Le cadrage initial du stage décomposait cette question en trois sous-problèmes, chacun rattaché à une famille de méthodes : le problème de durée de vie (analyse de survie), le problème de tendance (détection de ruptures en séries temporelles), et le problème du signal textuel (traitement automatique du langage). Une quatrième dimension, causale, était mentionnée comme perspective de recherche si le temps le permettait.

## 2.2 Ce que le BOAMP contient, et ce qu'il ne contient pas

Le BOAMP publie des avis : avis d'appel public à la concurrence, rectificatifs, avis d'attribution. Il ne publie pas de contrats, et il ne publie surtout **aucun identifiant de renouvellement**. Rien n'indique qu'un marché notifié en 2023 remplace un marché notifié en 2019. Il n'existe donc pas de vérité terrain sur l'événement que le cadrage initial appelait « renouvellement ».

Cette absence n'est pas un détail de qualité de données : elle interdit de poser le problème sous la forme apparemment naturelle « estimer P(renouvellement dans 12 mois) », parce qu'aucune observation de renouvellement n'existe dans les données pour estimer quoi que ce soit. Deux voies s'ouvrent alors. La première consiste à supposer le renouvellement — par exemple en déclarant qu'un accord-cadre de quatre ans est renouvelé au bout de quatre ans. Elle a été écartée : elle produirait des dates d'échéance fabriquées pour les trois quarts de la cohorte, dont la durée déclarée est manquante (§ 3.4), et elle reviendrait à mesurer l'hypothèse plutôt que les données. La seconde consiste à construire un observable et à assumer l'écart entre cet observable et le concept juridique. C'est la voie retenue.

## 2.3 L'estimand retenu : le successeur observable

> **Définition.** Un *successeur observable* d'un marché numérique attribué est un épisode de marché ultérieur, publié au BOAMP par le même acheteur, qui poursuit ou remplace de manière plausible le même besoin, et qui est **accepté** comme tel par une règle d'appariement figée et documentée.

Trois conséquences suivent, et le rapport les porte de bout en bout.

D'abord, l'événement est *conditionnel à une règle*. Changer la règle change le nombre d'événements : de 296 à 1 332 selon le bras retenu (§ 5.6). Aucun niveau absolu ne peut donc être cité seul.

Ensuite, l'événement est *observable*, pas juridique. Un marché peut être renouvelé sans que cela se voie dans le BOAMP — par une centrale d'achat, sous les seuils de publication, ou par un successeur que la règle n'a pas su retrouver. L'absence de successeur accepté est une **censure**, jamais une preuve d'abandon.

Enfin, les erreurs d'appariement ne se compensent pas. Un lien manqué retire un événement et allonge artificiellement une exposition censurée ; un faux lien fabrique à la fois un événement et sa date. Les deux biaisent l'estimation en sens opposés, ce qui interdit de présenter le taux observé comme une borne inférieure du vrai taux de remise en concurrence — un point sur lequel la littérature d'appariement d'enregistrements est explicite (Doidge et Harron, 2019).

## 2.4 Les questions telles qu'elles sont traitées

Le rapport répond aux questions suivantes, dans cet ordre :

**Q1.** Peut-on construire, à partir du BOAMP seul, une variable de successeur observable dont la qualité soit mesurable ? (§ 4)
**Q2.** Combien de temps s'écoule entre l'attribution d'un marché numérique et l'apparition d'un successeur observable, et ce délai diffère-t-il selon le segment et le type de marché ? (§ 5)
**Q3.** Le texte des avis porte-t-il une segmentation technologique métier que la nomenclature administrative CPV ne porte pas ? (§ 6)
**Q4.** Cette segmentation apprise permet-elle des lectures que la segmentation CPV ne permet pas ? (§ 7)
**Q5.** Les volumes trimestriels par segment montrent-ils des tendances ou des ruptures exploitables ? (§ 8)
**Q6.** Le modèle permet-il une prédiction individuelle par marché ? (§ 5.5, et la réponse est non)

---

# 3. Données, unités d'analyse et périmètre

## 3.1 Source et acquisition

La source est l'API officielle du BOAMP, opérée par la DILA et publiée sur data.gouv.fr. L'ensemble des avis publiés du 1ᵉʳ janvier 2015 au 31 décembre 2025 a été téléchargé au format JSONL, puis standardisé par des analyseurs conscients du schéma : les valeurs brutes et la lignée de parsage sont conservées à côté des valeurs normalisées, de sorte qu'une décision de nettoyage reste traçable et réversible.

L'extraction produit **1 620 712 avis uniques**, sans identifiant dupliqué.

## 3.2 Du avis à l'épisode : le premier changement de grain

Un avis n'est pas un marché. Le BOAMP republie une même procédure plusieurs fois : appel à concurrence, rectificatif, attribution, parfois lot par lot. Analyser au niveau de l'avis compterait plusieurs fois la même procédure et gonflerait mécaniquement tout taux d'événement.

Les avis sont donc regroupés en **épisodes de marché** par une recherche de composantes connexes (union-find) sur trois types d'arêtes, par ordre de fiabilité décroissante :

1. identifiant de dossier `contractFolderID` partagé (45 885 arêtes acceptées, 737 rejetées pour conflit d'acheteur validé) ;
2. lien explicite déclaré entre avis (584 896 arêtes) ;
3. référence de procédure identique chez un même acheteur, dans une fenêtre de 730 jours (195 043 arêtes acceptées, 861 rejetées pour dépassement de fenêtre).

Résultat : **1 103 632 épisodes**, dont 60,1 % de singletons. La reconstruction est une *inférence*, non un identifiant fourni par le BOAMP ; c'est la première source d'incertitude introduite par le pipeline, et elle est mesurée : zéro épisode à conflit d'acheteur, zéro chronologie impossible, tous les avis affectés à exactement un épisode. 3 274 épisodes portent plus d'une référence de procédure distincte — attendu pour des marchés allotis ou republiés — et sont exportés pour inspection plutôt que corrigés silencieusement.

## 3.3 La cohorte d'étude

L'entonnoir de sélection est intégralement réconcilié :

| Étape | Épisodes |
|---|---:|
| Épisodes Grand Ouest | 144 269 |
| dont au moins un code CPV en divisions 32, 35, 48, 72 | 7 376 |
| dont porteurs d'un avis d'attribution | 3 826 |
| dont date d'attribution résolue | **3 800** |

Les 26 épisodes perdus à la dernière étape n'ont pas de date d'attribution exploitable et sont écartés plutôt que datés par défaut.

**Une précision qui compte.** Le filtre numérique est une règle *à au moins un code*, au niveau de l'épisode : formellement, l'épisode $e$ est retenu si $\exists\, c \in \mathrm{CPV}(e)$ tel que $\lfloor c/10^{6}\rfloor \in \{32,35,48,72\}$. Ce n'est pas une règle sur le CPV principal. En conséquence, **1 176 des 3 800 épisodes (30,9 %) ont un CPV principal hors de ces divisions** : ce sont des marchés allotis entrés par un seul lot numérique. La cohorte se décrit donc exactement comme « épisodes de marché attribués du Grand Ouest comportant au moins un lot numérique », et non comme « 3 800 marchés numériques ». La formulation retenue dans tout le rapport est la première.

La variable de stratification `digital_segment` est la division numérique de plus petit numéro présente, de sorte que chaque épisode alimente exactement une courbe. Ce départage est arbitraire et concerne les 412 épisodes (10,8 %) portant plusieurs divisions numériques. Son effet a été mesuré plutôt que supposé : parmi les 2 624 épisodes dont le CPV principal est lui-même numérique, le segment attribué coïncide avec la division principale dans 94,7 % des cas, et les taux d'événement par segment attribué suivent de près ceux par division principale (CPV-35 : 0,2039 contre 0,1863 ; CPV-32 : 0,1319 contre 0,1241). La règle est donc documentée et conservée.

Composition de la cohorte : 1 452 épisodes en Pays de la Loire, 1 282 en Normandie, 1 066 en Bretagne ; 1 294 en CPV-72, 1 152 en CPV-32, 790 en CPV-48, 564 en CPV-35 ; suivi médian de 2 015 jours.

## 3.4 Qualité des données et traitement des valeurs manquantes

| Champ | Taux manquant | Décision | Justification |
|---|---:|---|---|
| SIREN validé | 66,3 % | Conserver une clé acheteur par nom, auditer les liens risqués | Le SIREN identifie une unité légale ; un nom similaire ne prouve pas l'identité juridique |
| Durée fiable | 74,9 % | Aucune imputation | La complétude passe de 11,8 % en 2023 à 84,4 % en 2025 : la non-réponse n'est pas échangeable dans le temps |
| Conteneur de montants | 15,7 % | Exclu de toute conclusion monétaire | Le conteneur porte plusieurs montants candidats au niveau avis, sans valeur attribuée canonique validée |
| CPV principal, texte, date d'attribution | 0,0 % | Requis par la sélection | — |

Deux de ces décisions méritent d'être défendues explicitement, parce qu'elles ferment des voies que le cadrage initial envisageait.

**Ne pas imputer la durée.** Le cadrage proposait d'utiliser la durée déclarée pour borner la fenêtre de renouvellement (« un marché de quatre ans est renouvelé à ±6 mois de son échéance »). Cette voie a été testée puis abandonnée sur preuve, et non par commodité : parmi les marchés disposant à la fois d'une durée fiable et d'un successeur accepté, **21,8 % seulement sont remis en concurrence dans les six mois suivant la fin déclarée**, l'écart absolu médian étant de **21,1 mois**. La durée déclarée est donc un mauvais prédicteur du calendrier réel. Le droit européen lui-même traite les quatre ans comme une limite générale assortie d'exceptions justifiées (directive 2014/24/UE, art. 33), non comme la durée de tout marché.

Il faut être précis sur ce que « écarter la durée » signifie, car le mot est trompeur. La durée déclarée n'est
utilisée **ni comme filtre dur, ni dans la règle primaire** : `M_B` ne lit que le texte. Elle entre en revanche
comme *preuve graduée* dans la composante temporelle du score pondéré de `M_C`, le bras de contraste : lorsqu'une
durée fiable existe, la composante vaut 1 à l'échéance déclarée et décroît linéairement jusqu'à zéro un an plus
tard ; lorsqu'elle n'existe pas, elle vaut `NaN` et disparaît du numérateur **et** du dénominateur, plutôt que
d'être notée zéro. C'est la différence entre « je n'ai pas d'information » et « l'information est défavorable ».
La conception initiale du projet, elle, substituait une durée de quatre ans quand l'échéance était inconnue ;
cette substitution a été retirée explicitement, parce qu'elle fabriquait l'entrée temporelle la plus influente
pour la majorité de la cohorte.

**Ne pas agréger les montants.** Aucune analyse monétaire n'est produite. C'est une perte réelle — le cadrage prévoyait des séries de montants — mais un montant reconstruit à partir d'un conteneur non validé produirait des séries dont personne ne pourrait dire ce qu'elles mesurent.

## 3.5 Justification du périmètre

Deux écarts au cadrage initial doivent être explicités, car ils ne sont pas neutres.

**Pays de la Loire → Grand Ouest.** Le cadrage donnait la priorité aux Pays de la Loire, avec extension au Grand Ouest si le volume était insuffisant. Le volume seul ne l'imposait pas : les Pays de la Loire comptent 1 452 épisodes et 219 événements, au-dessus du minimum de 800 marchés fixé par le cadrage. C'est la **granularité** exigée par les questions posées qui l'impose :

| Périmètre | Épisodes | Événements | P(successeur ≤ 12 mois) | IC 95 % | Largeur |
|---|---:|---:|---:|---|---:|
| Grand Ouest | 3 800 | 544 | 4,62 % | [3,94 ; 5,30] | 1,37 pt |
| Pays de la Loire seuls | 1 452 | 219 | 5,36 % | [4,17 ; 6,55] | 2,38 pt |

L'intervalle à 12 mois est 1,7 fois plus large sur les seuls Pays de la Loire ; le segment CPV-35, qui porte le résultat comparatif le plus robuste de l'étude, y tombe à 34 événements contre 115 ; et le modèle de Cox, qui compte huit covariables, passerait de 68 à 27 événements par paramètre, sous le seuil de prudence usuel. L'extension n'est pas neutre pour autant : la région est une covariable du modèle, et les Pays de la Loire ne se distinguent pas de la Bretagne (RR 1,003, p = 0,81) tandis que la Normandie présente un risque plus faible (RR 0,800, p = 0,050). Les résultats agrégés se lisent donc « Grand Ouest » ; une lecture centrée sur les Pays de la Loire passe par la ligne régionale du modèle, non par la moyenne.

**2015-2024 → 2015-2025.** Le cadrage prévoyait dix années jusqu'en 2024. L'année 2025 a été conservée, pour deux raisons et sous une réserve. Elle est complète en volume : 93, 73, 73 et 87 épisodes sur les quatre trimestres, contre 72, 57, 72 et 85 en 2024 ; il n'y a pas de trou de publication avant la coupure du 31 décembre 2025. Elle apporte 326 épisodes d'exposition censurée qui améliorent l'estimation aux horizons courts. Mais elle est **incomplète en suivi** : le suivi médian d'un marché attribué en 2025 est de 5,9 mois contre 59,9 mois pour 2015-2024, et le quatrième trimestre 2025 ne peut contenir aucun événement par construction. L'effet est mesurable : en retirant les attributions 2025, la probabilité estimée à 12 mois passe de 4,62 % à 3,54 %. L'année 2025 est donc conservée, signalée comme partielle en suivi, et **exclue de la fenêtre principale de validation temporelle** (§ 5.5), où elle n'intervient qu'en sensibilité.

## 3.6 Récapitulatif des grains

| Couche | Une ligne représente | Effectif |
|---|---|---:|
| Avis standardisés | un avis officiel BOAMP | 1 620 712 |
| Épisodes | une procédure de marché reconstruite | 1 103 632 |
| Cohorte d'étude | un épisode numérique attribué du Grand Ouest | 3 800 |
| Table de candidats | un couple ancre-candidat | 763 417 |
| Table de survie | un épisode de cohorte avec un délai observé ou censuré | 3 800 |

Chaque flèche entre ces couches est un changement d'unité statistique, et chacune introduit une incertitude qui se propage jusqu'aux résultats finaux. C'est la raison pour laquelle le rapport les traite comme des objets méthodologiques, et non comme des étapes de préparation.

---

# 4. Construction de la variable d'intérêt : l'appariement des successeurs

## 4.1 Pourquoi cette étape existe

L'analyse de survie a besoin de deux nombres par marché : un délai et un indicateur d'événement. Le BOAMP ne fournit ni l'un ni l'autre. Cette section construit les deux, et mesure la qualité de cette construction — parce qu'une erreur ici ne se corrige nulle part en aval.

## 4.2 Génération des candidats

Pour chaque **ancre** $i$ (épisode attribué de la cohorte, d'origine $u_i$), un épisode ultérieur $j$ d'origine $v_j$ est exposé à la comparaison si, et seulement si :

- les identifiants d'acheteur sont compatibles — même clé acheteur ou même nom normalisé, deux SIREN validés différents étant rédhibitoires ;
- et la chronologie vérifie
$$u_i + 90\ \text{jours} \le v_j \le u_i + 2\,920\ \text{jours}.$$

La fenêtre de 90 jours à 8 ans est une **fenêtre de recherche opérationnelle**, non une hypothèse de durée contractuelle. Ses deux bornes sont fixées empiriquement, et le code en conserve la trace. Sans plancher, une pathologie apparaissait : `132` des `628` liens alors acceptés tombaient à moins de trois mois de l'attribution, et tous les liens des marchés attribués en 2025 se situaient sous douze mois, avec une médiane de 1,7 mois — ce qu'aucun cycle de renouvellement n'explique. Il s'agissait d'activité concurrente, typiquement un autre lot du même programme. Le plancher est ensuite calé sur les données de référence plutôt que réglé : parmi les 23 successeurs confirmés qui se rattachent à la cohorte, le plus court écart attribution-successeur est de **139 jours**, donc un plancher à 90 jours n'en écarte aucun tout en supprimant la bande de procédures concurrentes. Le plafond de huit ans, de son côté, n'exclut aucun successeur confirmé.

Résultat : **763 417 couples** pour **3 520 des 3 800 ancres**. Les 280 ancres sans aucun candidat restent dans l'analyse comme exposition censurée : leur statut est décidé par le blocage, non par le seuil d'acceptation.

**Pas de blocage CPV strict, et l'argument est empirique.** Imposer que le successeur partage la division CPV de l'ancre serait tentant. La référence l'interdit : parmi les 23 successeurs revus, **9 (39,1 %) changent de division**. Un blocage strict les détruirait et ramènerait le plafond de rappel atteignable de 0,913 à 0,609. La littérature va dans le même sens : l'attribution des codes CPV est documentée comme sujette à erreur, y compris pour des experts (Siciliani et al., 2023). Le CPV est donc utilisé comme indice de continuité, jamais comme contrainte dure.

## 4.3 Quatre méthodes comparées sur les mêmes couples

| Méthode | Rôle | Principe |
|---|---|---|
| `M_A_deterministic` | Comparateur conservateur | Exige acheteur, CPV et un plancher de similarité textuelle |
| `M_B_text_ranking` | **Méthode primaire** | Retient le candidat de plus forte similarité TF-IDF cosinus et l'accepte au-delà du seuil |
| `M_C_weighted_gated` | Comparateur à rappel élevé | Score pondéré acheteur 0,50 / texte 0,25 / CPV 0,20 / temps 0,05, renormalisé sur les preuves observées |
| `M_D_fellegi_sunter` | Comparateur probabiliste | Rapports de vraisemblance appariement / non-appariement (Fellegi et Sunter, 1969) |

La règle de décision primaire est :
$$\hat{j}_i = \arg\max_{j \in J_i} T_{ij}, \qquad Y_i = \mathbf{1}\!\left(T_{i\hat{j}_i} \ge 0{,}70\right),$$
où $T_{ij}$ est le maximum entre une similarité cosinus TF-IDF de mots et une similarité de $n$-grammes de caractères. **Au plus un successeur par ancre** ; sinon la méthode s'abstient.

Un détail d'implémentation a des conséquences analysées en § 5.7 : le vectoriseur est ajusté *par bloc d'ancre*. Ce choix a une raison — les poids IDF deviennent locaux au vocabulaire de l'acheteur, ce qui empêche le formulaire administratif commun à tous les avis d'un même acheteur de dominer la similarité — et il a un coût : le score devient relatif au bloc, un seuil fixe de 0,70 ne signifie pas exactement la même chose pour toutes les ancres, et le maximum sur un bloc plus grand est mécaniquement plus élevé. Le § 5.7d mesure ce coût plutôt que de le supposer négligeable.

## 4.4 Pourquoi le seuil a été gelé avant l'évaluation

Le seuil de 0,70 a été fixé *a priori*, sur un principe de précision d'abord : un faux lien fabrique à la fois un événement et sa date, tandis qu'un lien manqué ne fabrique rien. Il n'a pas été déplacé après consultation de la référence, et c'est **la seule raison pour laquelle le sous-échantillon verrouillé peut être présenté comme tenu à l'écart**. Le balayage de seuils (annexe C) montre que 0,60 offrirait un meilleur rappel ; il reste un bras de sensibilité et n'est pas promu, parce que choisir un seuil sur ces lignes transformerait le verrou en jeu de réglage.

Le cadrage initial anticipait un taux d'appariement de 40 à 60 %. Le taux réalisé est de **14,3 %**. Ce n'est pas un manque par rapport à un objectif : le chiffre du cadrage a été traité comme une hypothèse de planification, jamais comme une cible d'optimisation, et 14,3 % est la conséquence arithmétique d'une règle conçue pour la précision et d'un événement défini comme *observable dans le BOAMP*.

## 4.5 La référence régionale, et ce qu'elle vaut

L'évaluation repose sur une revue stratifiée de **120 ancres** du Grand Ouest, réalisée le 11 août 2026 contre les avis BOAMP réels et leurs URL officielles, **avant que les méthodes d'appariement n'existent**. 112 ancres se réattachent à exactement un épisode et 88 sont exploitables ; le partage pilote (16 ancres) / verrouillé (72 ancres) figure dans le fichier de référence lui-même et n'est pas postérieur.

Quatre limites en bornent la portée, et elles sont exposées ici plutôt qu'en note :

1. **Les étiquettes proviennent d'une passe de recherche unique par grand modèle de langue**, contrôlée par sondage par l'auteur du projet, et non vérifiée ancre par ancre ni jugée par un panel de spécialistes. Elles sont indépendantes des méthodes évaluées — aucune n'existait — mais elles ne constituent pas une vérité terrain humaine.
2. **La trace probante par ancre n'a pas été enregistrée** : les sources consultées pour une décision donnée ne peuvent pas être reconstituées.
3. **Les négatifs sont relatifs au corpus** : environ 25 candidats par ancre ont été considérés, pas le vivier complet. Un négatif signifie « aucun successeur identifié parmi ceux examinés ».
4. **L'indépendance de l'exposition des candidats ne tient pas.** La règle qui a sélectionné ces ~25 candidats (sur des viviers allant jusqu'à 3 258) n'est enregistrée nulle part, et les 16 successeurs retrouvables du verrou se classent tous dans les treize premiers selon le score textuel évalué. En conséquence, **le rappel et le plafond de 0,913 ne sont pas pleinement indépendants du score qu'ils bornent**. La précision, elle, ne l'est pas : un faux positif reste un faux positif quelle que soit la façon dont la liste a été construite.

## 4.6 Résultats sur le sous-échantillon verrouillé

Deux comptabilités coexistent et ne disent pas la même chose. La comptabilité *ancre* demande : le système a-t-il détecté qu'un successeur existe ? La comptabilité *successeur exact*, plus stricte, demande : a-t-il désigné le bon ? Un mauvais successeur sur une ancre positive compte alors comme un faux positif **et** un faux négatif. Les chiffres du projet sont les seconds.

| Méthode | Seuil | VP | FP | FN | VN | Précision | IC 95 % | Rappel | Taux FP |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| `M_A_deterministic` | n/a | 8 | 7 | 10 | 47 | 0,533 | 0,301–0,752 | 0,444 | 0,130 |
| **`M_B_text_ranking`** | **0,70** | **7** | **1** | **11** | **54** | **0,875** | **0,529–0,978** | **0,389** | **0,000** |
| `M_C_weighted_gated` | 0,70 | 12 | 11 | 6 | 44 | 0,522 | 0,330–0,708 | 0,667 | 0,185 |
| `M_D_fellegi_sunter` | 0,65 | 1 | 4 | 17 | 53 | 0,200 | 0,036–0,625 | 0,056 | 0,018 |

Trois lectures, dans l'ordre d'importance.

**Les intervalles se recouvrent largement.** Une précision de 0,875 assise sur **huit liens acceptés** a un intervalle qui va de « pile ou face » à « quasi parfait ». Ce chiffre ne doit jamais être cité sans son intervalle, et cette référence ne sépare les méthodes que grossièrement.

**Le rappel est plafonné avant tout score.** La génération de candidats n'expose que 21 des 23 successeurs revus, soit une complétude des couples de 0,913 (Christen et Goiser, 2007) ; les deux cas manquants sont attribués à une condition de blocage nommée — une ancre absente de la cohorte faute d'adresse structurée, un acheteur passé de CCAS à CIAS sans SIREN partagé. Aucun cas inexpliqué, ce qui distingue un plafond assumé d'un défaut d'implémentation.

**L'arbitrage précision/rappel est assumé.** `M_C` gagne du rappel (0,667) en perdant la moitié de sa précision (0,522). Pour une analyse de survie, ce n'est pas un échange neutre : les 11 faux positifs de `M_C` introduiraient onze événements inexistants avec onze dates inexistantes.

## 4.7 Revue de contrôle et audit d'appariement

Un échantillon aveugle de 60 couples a été préparé (20 liens acceptés en production, 20 négatifs structurels à forte similarité, 20 relations déclarées par l'acheteur), avec clé d'audit séparée et règle d'acceptation pré-spécifiée. La revue réalisée est **assistée par modèle**, non humaine indépendante — la provenance est enregistrée explicitement dans le dépôt. Sur les 20 liens acceptés : 14 confirmés, 5 infirmés, 1 incertain, soit une précision de **0,700** en lecture conservatrice (IC 95 % [0,457 ; 0,881]), **sous la cible de 0,80** que le projet s'était fixée.

Ce résultat est reporté tel quel. Il ne clôt pas la validation ; il la rend nécessaire. La revue humaine indépendante est la première recommandation du § 9.F.

## 4.8 Ce que l'appariement produit

**544 liens acceptés** sur 3 800 ancres, soit un taux d'événement de **14,32 %**. Continuité CPV : 351 des 538 liens dont les deux divisions sont observées restent dans la même division (65,2 %), un taux comparable à celui des successeurs de la référence (14 sur 23, soit 60,9 %). 461 successeurs distincts pour 544 liens : 44 épisodes sont acceptés par plusieurs ancres, le plus réutilisé l'étant par onze — signature de faux positif exploitée en § 5.6. Zéro lien accepté avec des SIREN validés conflictuels ; zéro mélange commune / intercommunalité.

---

# 5. Analyse de survie

## 5.1 Construction du temps et de la censure

Pour une ancre $i$ dont le successeur $\hat{j}_i$ est accepté :
$$\tau_i = v_{\hat{j}_i} - u_i, \qquad Y_i = 1.$$
Si aucun successeur n'est accepté avant le 31 décembre 2025 :
$$\tau_i = \text{2025-12-31} - u_i, \qquad Y_i = 0,$$
et la ligne est **censurée à droite administrativement**. L'origine des temps est la date d'attribution.

La cohorte comprend donc **3 800 lignes, 544 événements et 3 256 censures** (85,7 %). Ce taux de censure élevé est la raison même d'utiliser l'analyse de survie plutôt qu'une régression sur les seuls marchés appariés : ignorer la censure reviendrait à n'étudier que les marchés dont on a vu le successeur, c'est-à-dire à conditionner l'analyse sur la variable à expliquer.

## 5.2 Estimation non paramétrique

L'estimateur de Kaplan-Meier (Kaplan et Meier, 1958),
$$\hat S(t) = \prod_{t_k \le t}\left(1 - \frac{d_k}{n_k}\right),$$
où $d_k$ est le nombre d'événements en $t_k$ et $n_k$ l'effectif à risque, ne suppose aucune forme de distribution et traite la censure administrative sans biais tant que celle-ci est indépendante du risque.

| Horizon | P(successeur observable) | Survie |
|---|---:|---:|
| 12 mois | **4,62 %** | 0,9538 |
| 24 mois | **6,73 %** | 0,9327 |
| 36 mois | 8,68 % | 0,9132 |
| 48 mois | 15,50 % | 0,8450 |
| 60 mois | 17,54 % | 0,8246 |

**La médiane de Kaplan-Meier n'est pas atteinte** : la courbe ne descend jamais sous 0,5 sur la fenêtre d'observation. Les 31,8 mois qui circulent par ailleurs sont la médiane *parmi les seuls événements liés* ; les deux quantités portent le même mot et ne mesurent pas la même chose, la seconde étant conditionnée à l'occurrence de l'événement.

Le saut entre 36 et 48 mois — de 8,7 % à 15,5 % — est l'**épaule de renouvellement**, compatible avec la limite générale de quatre ans des accords-cadres. C'est la partie de la courbe la plus intéressante pour Gigalis et aussi la plus fragile : à 48 mois, l'ensemble à risque ne contient plus que des marchés attribués avant 2022, donc la composition de la cohorte change le long de la courbe. Elle ne doit pas être citée sans les effectifs à risque (annexe D).

**Comparaison entre segments.** Le test du log-rank multivarié entre les quatre segments CPV donne une statistique de **23,45** pour $p = 3{,}3\times10^{-5}$ : les segments ne se comportent pas de la même façon. Les taux d'événement bruts vont de 11,6 % (CPV-48) à 20,4 % (CPV-35).

## 5.3 Probabilités conditionnelles : le livrable opérationnel

Pour un marché ayant atteint l'âge $a$ sans successeur observé, la probabilité qu'un successeur apparaisse dans les $h$ mois suivants est
$$P(T \le a+h \mid T > a) = 1 - \frac{S(a+h)}{S(a)},$$
lue sur l'estimateur de Kaplan-Meier, avec intervalles obtenus par bootstrap d'épisodes à 500 tirages.

| Âge du marché | P(successeur ≤ 12 mois) | IC 95 % | P(≤ 24 mois) | IC 95 % |
|---:|---:|---|---:|---|
| 0 mois | 4,62 % | [3,91 ; 5,24] | 6,73 % | [5,91 ; 7,58] |
| 12 mois | 2,22 % | [1,70 ; 2,73] | 4,26 % | [3,55 ; 4,98] |
| 24 mois | 2,09 % | [1,58 ; 2,60] | 9,40 % | [8,33 ; 10,68] |
| **36 mois** | **7,46 %** | [6,52 ; 8,59] | **9,69 %** | [8,58 ; 11,12] |
| 48 mois | 2,41 % | [1,77 ; 3,07] | 2,89 % | [2,15 ; 3,72] |

Le profil n'est pas monotone en âge : il monte dans l'épaule de 36-48 mois puis retombe. C'est exactement ce que l'on attend d'un parc de contrats pluriannuels, et c'est ce qui rend la table exploitable comme **logique de veille par cohorte** : elle classe des âges et des segments, elle ne prédit pas un marché.

## 5.4 Modèle de Cox : associations comparatives

Le modèle à risques proportionnels (Cox, 1972),
$$h(t \mid X) = h_0(t)\exp(\beta_1X_1 + \cdots + \beta_pX_p),$$
estime l'effet des covariables sans spécifier la forme du risque de base. Cinq covariables sont retenues, conformément à la règle d'un paramètre pour dix événements : segment CPV, région, statut d'accord-cadre, disponibilité d'un SIREN validé, année d'attribution centrée — soit huit paramètres pour 544 événements.

| Covariable | RR | IC 95 % | p |
|---|---:|---|---:|
| Accord-cadre | 1,751 | [1,435 ; 2,136] | 3,4 × 10⁻⁸ |
| **CPV-35** (réf. CPV-32) | **1,553** | [1,218 ; 1,981] | **3,8 × 10⁻⁴** |
| CPV-48 | 0,828 | [0,638 ; 1,073] | 0,153 |
| CPV-72 | 1,056 | [0,850 ; 1,310] | 0,624 |
| Normandie (réf. Bretagne) | 0,800 | [0,640 ; 1,000] | 0,050 |
| Pays de la Loire | 1,003 | [0,815 ; 1,234] | 0,979 |
| SIREN validé | 1,082 | [0,885 ; 1,323] | 0,443 |
| Année d'attribution (centrée) | 1,107 | [1,066 ; 1,149] | 1,2 × 10⁻⁷ |

Concordance en échantillon : 0,626.

**Le diagnostic de proportionnalité rejette l'hypothèse pour trois covariables** : année d'attribution (statistique 70,7 ; $p = 4\times10^{-17}$), accord-cadre (7,04 ; $p = 0{,}008$) et SIREN validé (6,76 ; $p = 0{,}009$). La violation sur l'année d'attribution est massive, et attendue : l'année est structurellement confondue avec la longueur du suivi. En conséquence, **ces coefficients sont des associations descriptives moyennées dans le temps, et non des effets**. Une stratification sur des tranches d'année d'attribution serait plus propre ; elle a été jugée facultative parce que les coefficients d'intérêt bougent peu et que rien d'opérationnel ne repose sur ce modèle.

## 5.5 Validation temporelle : un résultat négatif

Le modèle est ajusté une seule fois sur les attributions 2015-2021 et appliqué hors période, sans réajustement ni réglage.

| Fenêtre | N appr. | Évén. appr. | N test | Évén. test | C appr. | **C test** |
|---|---:|---:|---:|---:|---:|---:|
| Principale, 2022-2024 | 2 470 | 392 | 1 004 | 107 | 0,606 | **0,479** |
| Sensibilité, 2022-2025 | 2 470 | 392 | 1 330 | 152 | 0,606 | **0,518** |

Un indice de concordance de 0,479 est **indiscernable du hasard**. Le modèle ne classe pas utilement les marchés individuels par délai jusqu'au successeur sur des années d'attribution non vues. Une partie de ce résultat est structurelle — les marchés attribués à partir de 2022 ne peuvent contribuer que des écarts courts, et l'indice de Harrell dépend de la distribution de censure (Uno et al., 2011), de sorte que deux fenêtres à suivi inégal ne sont pas strictement comparables.

Ce résultat est publié tel quel, et il commande une décision : **aucune prédiction individuelle n'est produite**. Le cadrage prévoyait un tableau des vingt marchés les plus susceptibles d'être renouvelés ; produire ce tableau donnerait une impression de précision que la validation ne soutient pas. Le livrable opérationnel est donc la table conditionnelle du § 5.3 et, en annexe, une liste de veille **stratifiée par segment** — cinq marchés par segment CPV parmi ceux attribués depuis 2021 — lue sur les courbes de Kaplan-Meier segmentées et non sur le modèle de Cox. La stratification est délibérée : la probabilité conditionnelle n'étant fonction que du segment et de l'âge, un classement global renverrait mécaniquement le segment à plus fort risque à l'âge le plus proche de l'épaule, en suggérant une granularité individuelle qui n'existe pas.

## 5.6 Modèles paramétriques

Cinq familles ont été ajustées et comparées par AIC et BIC :

| Modèle | Paramètres | Log-vraisemblance | AIC | BIC |
|---|---:|---:|---:|---:|
| **Gamma généralisée** | 3 | −3 757,5 | **7 521,0** | **7 539,7** |
| Log-normale | 2 | −3 784,6 | 7 573,2 | 7 585,7 |
| Log-logistique | 2 | −3 807,3 | 7 618,6 | 7 631,1 |
| Weibull | 2 | −3 814,7 | 7 633,4 | 7 645,8 |
| Exponentielle | 1 | −3 829,7 | 7 661,3 | 7 667,6 |

La gamma généralisée l'emporte sur les deux critères. Elle **n'est pas** la source des chiffres opérationnels, et ce choix mérite d'être défendu : toutes ces familles sont lisses et aplatissent l'épaule empirique de 36-48 mois, qui est précisément l'objet métier ; et tous les horizons publiés (12, 24 mois) sont à l'intérieur de la fenêtre observée, là où l'estimateur empirique n'a besoin d'aucune hypothèse de forme. Le modèle paramétrique est donc conservé comme meilleure famille d'ajustement et comme instrument que toute extrapolation au-delà du 31 décembre 2025 devrait utiliser. Retenir mécaniquement le meilleur AIC pour produire les chiffres opérationnels aurait échangé un ajustement global légèrement meilleur contre une moins bonne restitution de la seule zone qui intéresse l'utilisateur.

## 5.7 Robustesse : quatre épreuves indépendantes

C'est la partie la plus importante du rapport pour l'interprétation des résultats, parce qu'elle sépare ce qui bouge de ce qui tient.

**(a) Quatre définitions de l'événement.**

| Bras | Événements | Taux | P(≤ 12 m) | P(≤ 24 m) | Médiane des écarts observés |
|---|---:|---:|---:|---:|---:|
| `M_B @ 0,80` strict | 296 | 7,8 % | 2,37 % | 3,23 % | 35,7 mois |
| **`M_B @ 0,70` principal** | **544** | **14,3 %** | **4,62 %** | **6,73 %** | **31,8 mois** |
| `M_B @ 0,60` large | 853 | 22,4 % | 8,00 % | 11,47 % | 26,6 mois |
| `M_C @ 0,70` contraste | 1 332 | 35,1 % | 12,21 % | 17,98 % | 26,1 mois |

Un facteur 4,5 sur le nombre d'événements. **Aucune probabilité absolue ne peut être présentée seule.** Une observation complète toutefois ce constat, et elle va dans l'autre sens : superposées, les quatre courbes de survie se déplacent verticalement mais gardent la même **forme**, avec le décrochement entre 40 et 48 mois visible dans les quatre bras. Le niveau dépend de la règle ; l'épaule de renouvellement, elle, n'est pas un artefact de la règle d'acceptation.

**(b) Bande limite.** Les décisions les plus fragiles sont celles dont le meilleur candidat frôle le seuil. Retirer les 280 épisodes dont le score tombe dans $[0{,}65 ; 0{,}75]$ — bande symétrique fixée *a priori*, non explorée — supprime 133 événements et donne : P(≤ 12 m) 4,62 % → 3,72 %, RR CPV-35 1,553 → **1,780**, RR accord-cadre 1,751 → 1,616. La direction des deux rapports de risques est conservée.

**(c) Recensure du risque « modèle type ».** Les deux premières épreuves déplacent la barre d'acceptation. Or le mécanisme de faux positif identifié par l'audit produit des liens **au-dessus** de la barre : les avis d'attribution français comportent de longs passages standardisés d'accord-cadre, sur lesquels les $n$-grammes de caractères notent haut entre objets sans rapport ; et `M_B` classe indépendamment pour chaque ancre, donc un même épisode peut être accepté pour plusieurs ancres. Deux signatures observables délimitent le groupe à risque : similarité au niveau des mots inférieure à 0,50 (65 liens), ou successeur partagé avec une autre ancre (127 liens) — au total **173 des 544 liens (31,8 %)**. Ces ancres sont **recensurées à la coupure** plutôt que supprimées, parce que c'est le contrefactuel qu'implique un lien fallacieux : l'ancre n'avait pas de successeur observé et doit contribuer tout son suivi comme exposition censurée. Résultat : P(≤ 12 m) tombe à 2,64 %, mais RR CPV-35 = **1,541** et RR accord-cadre = **1,692**. C'est l'épreuve dont l'effet accord-cadre avait le plus besoin, puisque c'est ce texte-là qui alimente le mécanisme.

**(d) Détectabilité.** Le plus grand déséquilibre entre épisodes liés et censurés n'est pas une propriété du marché :

| Variable | Moyenne liés | Moyenne censurés | Différence standardisée |
|---|---:|---:|---:|
| **log(taille du vivier de candidats)** | 4,787 | 3,972 | **+0,470** |
| taille du vivier | 274,3 | 188,6 | +0,285 |
| longueur du texte | 1 611 | 1 087 | +0,262 |
| accord-cadre | 0,265 | 0,187 | +0,187 |

`M_B` retient le maximum du score textuel sur le bloc de candidats, et le maximum d'un plus grand nombre de tirages est plus grand : un acheteur qui publie beaucoup a mécaniquement plus de chances de produire un lien accepté. C'est un canal d'**observabilité**, pas une cause de remise en concurrence. Un modèle de sensibilité ajoute $\log(1 + \text{taille du vivier})$ :

| Covariable | RR modèle principal | RR + log(vivier) | p ajusté |
|---|---:|---:|---:|
| Accord-cadre | 1,751 | **1,617** | 2,6 × 10⁻⁶ |
| CPV-35 | 1,553 | **1,512** | 8,5 × 10⁻⁴ |
| log(taille du vivier) | — | 1,184 | 6,0 × 10⁻⁹ |

Deux lectures, et elles diffèrent. **CPV-35 y est insensible** : le rapport passe de 1,553 à 1,512. Avec sa stabilité à travers les quatre bras, la bande limite et la recensure, c'est le résultat comparatif le plus robuste de l'étude. **L'association accord-cadre est en partie de la détectabilité** : environ 14 % du logarithme du rapport de risques s'évapore. La direction survit à toutes les épreuves, mais l'association est plus faible que le modèle principal seul ne le laisse croire, et l'explication n'est pas seulement le texte standardisé — c'est aussi le volume de publication.

Le modèle principal reste inchangé : cette colonne est une sensibilité, pas une nouvelle spécification de référence.

## 5.8 Ce que l'analyse de survie établit

Sous la définition d'événement retenue : la probabilité qu'un marché numérique attribué du Grand Ouest montre un successeur observable est estimée à 4,6 % à douze mois et 6,7 % à vingt-quatre mois, avec une épaule marquée entre 36 et 48 mois ; les segments diffèrent nettement, CPV-35 étant le plus rapide ; les accords-cadres montrent un successeur plus tôt, pour partie parce qu'ils sont plus détectables ; les niveaux absolus dépendent de la règle d'appariement dans un rapport de un à quatre et demi, tandis que les comparaisons relatives résistent aux quatre épreuves. Le modèle ne discrimine pas hors période et n'est donc pas un outil de prédiction individuelle.

---

# 6. Signal textuel : une taxonomie technologique supervisée

## 6.1 Pourquoi le CPV ne suffit pas

La cohorte est définie par quatre divisions CPV. C'est reproductible, auditable — et grossier : le CPV dit qu'un marché est numérique, pas ce qui a été acheté. Or toutes les questions métier de Gigalis portent sur le second niveau : quels segments croissent, lesquels se remettent en concurrence le plus vite, où ouvrir un accord-cadre. Cette section apprend cette variable manquante à partir du texte de l'objet du marché :

$$\text{texte de l'objet du marché} \longrightarrow \text{classifieur supervisé} \longrightarrow \text{classe technologique métier}$$

Il s'agit d'une **couche d'enrichissement**. Elle ne remplace pas la segmentation CPV, qui reste la définition de la cohorte, la covariable du modèle de Cox et l'axe des séries temporelles.

## 6.2 Taxonomie et corpus

Huit classes substantielles — `CLOUD_HOSTING`, `CYBERSECURITY`, `NETWORK_TELECOM`, `IT_INFRASTRUCTURE`, `BUSINESS_SOFTWARE`, `DATA_BI`, `AI`, `IT_SERVICES` — et trois classes de repli qui sont des **décisions d'annotation** et non des valeurs manquantes : `MIXED` (aucune technologie dominante), `OTHER_DIGITAL` (achat numérique hors des huit), `OTHER` (CPV numérique sans achat technologique). La taxonomie a été figée avant toute modélisation.

Le corpus compte **500 avis annotés manuellement**, 2015-2025, tous porteurs d'un texte d'objet non vide (médiane 14 mots). Deux propriétés en contraignent la lecture. L'échantillon est **stratifié par quotas** : les proportions de classes sont une propriété du plan d'annotation, non une estimation de prévalence. Et la classe `AI` ne compte que **7 avis** en onze ans : aucun exemple synthétique n'a été créé, aucune ligne dupliquée, aucun suréchantillonnage appliqué ; la conséquence est publiée plutôt que corrigée.

## 6.3 Prévention des fuites

Le BOAMP republie une même procédure, et les acheteurs relancent des consultations quasi identiques à quelques années d'intervalle. Évaluer un modèle sur une quasi-copie d'un document vu à l'entraînement mesure la mémorisation. Chaque avis est donc rattaché à une **famille de marché**, définie comme l'union de deux règles : les avis déjà regroupés dans un même épisode reconstruit, et les avis dont les objets atteignent une similarité cosinus au niveau caractère de 0,80. Chaque famille appartient à un seul pli.

La seconde règle n'est pas superflue : la première seule donne 486 groupes ; l'ajout de la seconde fusionne 39 paires quasi dupliquées, dont **29 appartiennent à des épisodes différents** et auraient donc été réparties entre plis. Le résultat est de **459 familles**, dont aucune ne chevauche deux plis.

Un effet secondaire mérite d'être signalé : **3 familles contiennent des avis au texte quasi identique mais aux étiquettes différentes** — services de visioconférence classés `NETWORK_TELECOM` en 2017 et `OTHER_DIGITAL` en 2021, une infrastructure en tant que service classée `MIXED` en 2017 et `CLOUD_HOSTING` en 2021. Aucune étiquette n'a été modifiée : rien ne permet de trancher, et corriger des étiquettes après avoir vu les erreurs du modèle revient à ajuster le corpus au classifieur. Elles sont enregistrées comme plancher empirique de la performance atteignable.

## 6.4 Représentation et modèles

L'entrée est le seul champ `objet`. La normalisation est délibérément légère — réparation d'encodage, NFC, minuscules, espaces — et **les accents sont conservés**, sans racinisation : les classes se distinguent par des mots comme *cybersécurité*, *logiciel métier*, *intelligence artificielle*, et aplatir l'orthographe française détruirait la preuve. Les traits sont des $n$-grammes de mots TF-IDF ; unigrammes et unigrammes+bigrammes étaient tous deux dans l'espace de recherche, et **tous les plis ont retenu les unigrammes seuls**, le vocabulaire de bigrammes étant trop creux sur 500 documents courts.

Sont exclus des traits, par construction : identité de l'acheteur, géographie, dates, montants, type de procédure, statut d'accord-cadre, identifiants d'avis, toute variable d'appariement, et le CPV — qui sert de comparateur.

Six spécifications ont été comparées sur des plis identiques, avec sélection d'hyperparamètres par validation croisée interne *dans chaque pli d'entraînement* :

| Modèle | Traits | Macro-F1 hors pli |
|---|---|---:|
| Classe majoritaire | — | 0,027 |
| CPV seul | codes CPV | 0,441 |
| CPV + descripteurs BOAMP | administratif | 0,473 |
| TF-IDF + régression logistique | texte | 0,670 |
| **TF-IDF + régression logistique pondérée** | texte | **0,744** |
| TF-IDF + SVM linéaire | texte | 0,718 |
| TF-IDF + SVM linéaire pondéré | texte | 0,715 |

## 6.5 Le résultat principal

| Comparaison | Macro-F1 | IC 95 % (bootstrap de familles) |
|---|---:|---|
| TF-IDF + régression logistique pondérée | **0,7442** | [0,682 ; 0,791] |
| Meilleur comparateur CPV/descripteurs | 0,4731 | [0,413 ; 0,526] |
| **Différence appariée** | **+0,2711** | **[0,201 ; 0,340]** |

L'intervalle de la différence exclut zéro. Trois précautions rendent ce résultat solide plutôt que trivial : les deux côtés utilisent **les mêmes plis** et **le même budget de recherche d'hyperparamètres** ; la différence est **appariée** pli par pli ; et le bootstrap rééchantillonne des **familles de marchés**, c'est-à-dire l'unité à laquelle la dépendance existe, et non des avis.

Autrement dit : le texte des avis porte une information métier que la nomenclature administrative ne porte pas. Le tableau croisé le rend concret :

| Division CPV | Classe dominante | Effectif dominant | Pureté | Autres classes ≥ 100 épisodes |
|---|---|---:|---:|---|
| CPV-32 (télécoms) | NETWORK_TELECOM | 426 / 1 152 | 0,370 | OTHER_DIGITAL 202, BUSINESS_SOFTWARE 146, IT_INFRASTRUCTURE 128 |
| CPV-35 (sécurité) | NETWORK_TELECOM | 165 / 564 | 0,293 | OTHER 114, CYBERSECURITY 106 |
| CPV-48 (logiciels) | BUSINESS_SOFTWARE | 348 / 790 | 0,441 | — |
| CPV-72 (services IT) | IT_SERVICES | 325 / 1 294 | 0,251 | BUSINESS_SOFTWARE 310, NETWORK_TELECOM 204, OTHER_DIGITAL 165 |

Deux lectures. La pureté moyenne d'un segment CPV vis-à-vis de la taxonomie apprise est de **0,34** : aucune division ne correspond à une technologie. Et la division la plus large, CPV-72, est la moins pure des quatre (0,251) : ce que l'administration appelle « services informatiques » recouvre en réalité quatre familles métier d'effectifs comparables. Détail révélateur : CPV-32 et CPV-35, deux divisions distinctes de la nomenclature, ont la **même** classe dominante prédite.

## 6.6 Ce que le classifieur sait faire, et ce qu'il ne sait pas

| Classe | Précision | Rappel | F1 | Support |
|---|---:|---:|---:|---:|
| DATA_BI | 0,923 | 0,828 | **0,873** | 29 |
| OTHER | 0,923 | 0,800 | 0,857 | 15 |
| BUSINESS_SOFTWARE | 0,798 | 0,898 | 0,845 | 88 |
| NETWORK_TELECOM | 0,810 | 0,840 | 0,824 | 81 |
| CYBERSECURITY | 0,848 | 0,722 | 0,780 | 54 |
| IT_SERVICES | 0,778 | 0,757 | 0,767 | 74 |
| IT_INFRASTRUCTURE | 0,702 | 0,717 | 0,710 | 46 |
| OTHER_DIGITAL | 0,655 | 0,679 | 0,667 | 53 |
| AI | 0,800 | 0,571 | *0,667* | **7** |
| CLOUD_HOSTING | 0,731 | 0,594 | 0,655 | 32 |
| MIXED | 0,482 | 0,619 | 0,542 | 21 |

Macro-F1 0,741 (écart-type entre plis 0,034), F1 pondéré 0,765, exactitude 0,766. Deux agrégations coexistent et il faut les distinguer : 0,741 est la **moyenne des trois plis**, tandis que le chiffre de référence du § 6.5, 0,744, est calculé **hors pli sur les 500 prédictions mises en commun**. L'écart est celui, attendu, entre une moyenne de scores par pli et un score unique sur l'ensemble des prédictions.

Trois lectures. Les classes fiables sont `CYBERSECURITY`, `NETWORK_TELECOM`, `BUSINESS_SOFTWARE`, `DATA_BI`, `IT_SERVICES` et `OTHER`. La classe `MIXED` est faible, ce qui est cohérent avec sa définition — c'est le fourre-tout des marchés sans technologie dominante. Et **`AI` n'est pas interprétable** : son F1 de 0,667 repose sur trois prédictions correctes ; un chiffre élevé serait un artefact de petit échantillon et un chiffre bas serait tout aussi peu informatif.

L'analyse d'erreurs sur trente cas représentatifs donne : 16 erreurs de modèle, 7 ambiguïtés de frontière taxonomique, 4 marchés réellement multi-technologies, 2 incohérences d'annotation, 1 objet trop pauvre. Près de la moitié des erreurs relèvent donc de la définition des classes ou de l'information disponible, non de la capacité du modèle.

**Robustesse temporelle.** Entraînement 2015-2022 (n = 393), test 2023-2025 (n = 107), les 4 familles à cheval sur la frontière étant versées à l'entraînement — ce qui coûte des observations de test et ne peut pas flatter le résultat. Macro-F1 toutes classes 0,662, mais **0,815 sur les classes dont le support de test atteint 10**. Le vocabulaire des avis récents n'a pas dérivé pour les classes à volume ; rien ne peut être conclu pour les six classes à faible support récent.

## 6.7 Deux décisions négatives, prises sur critère écrit à l'avance

**Le transformeur n'a pas été lancé.** La règle avait été écrite avant de lire les résultats classiques : tester CamemBERT seulement si le modèle classique est matériellement insuffisant (macro-F1 < 0,55) **et** si moins de la moitié des erreurs proviennent d'ambiguïté d'étiquetage ou d'information absente, qu'aucun encodeur ne peut fournir. Le modèle atteint 0,741 : la première condition échoue nettement. CamemBERT **n'a pas été testé puis écarté ; il n'a pas été exécuté**, parce que le critère de son exécution n'était pas rempli. Le diagnostic de la courbe d'apprentissage confirme que c'était le bon arbitrage : le F1 d'entraînement est proche de 0,99 à toutes les tailles tandis que le F1 de validation monte de 0,434 à 0,747, c'est-à-dire un problème de **variance**, pour lequel ajouter de la capacité est le mauvais remède. Le cadrage prévoyait d'ailleurs de conserver la baseline si le gain était inférieur à cinq points de F1.

**La calibration a été évaluée et rejetée.** La confiance publiée est la probabilité brute de la régression logistique. Une calibration de Platt a été estimée à l'intérieur des mêmes plis groupés et jugée selon une règle fixée à l'avance : l'adopter seulement si elle réduit l'erreur de calibration attendue d'au moins 0,02 **et** coûte au plus 0,02 de macro-F1. Elle réduit l'erreur de 0,1405 — condition remplie — mais coûte 0,0364 de macro-F1, au-delà du budget. La règle n'a pas été assouplie pour l'admettre.

En conséquence, le score déployé est un **score de confiance non calibré**. Sa fiabilité hors pli est mesurée : l'exactitude observée croît de 0,52 dans la tranche [0 ; 0,3) à 1,00 dans la tranche [0,9 ; 1,0), mais l'écart entre confiance annoncée et exactitude observée est positif dans toutes les tranches au-dessus de 0,3 — le score **sous-estime** son propre taux de succès, l'erreur de calibration attendue étant de 0,350. Il est donc utilisable comme **ordre de classement et comme filtre** — au seuil opérationnel de 0,70, l'exactitude hors pli est de 0,956 sur les 9 % d'avis qui le franchissent, contre 0,750 en dessous — mais une valeur de confiance n'est pas la probabilité que l'étiquette soit correcte. Deux raisons : la miscalibration ci-dessus, et le fait que le corpus soit stratifié par quotas alors que la population de déploiement ne l'est pas, de sorte que la loi a priori encodée par le classifieur est un artefact du plan d'annotation.

## 6.8 Déploiement

Le modèle est réajusté sur les 500 étiquettes et appliqué aux **3 800 épisodes** de la cohorte, via le texte d'objet de l'avis d'origine. Ce réajustement n'a pas de score de validation et aucun n'est rapporté : la preuve du modèle est la validation croisée groupée et le contrôle temporel ci-dessus. Chaque épisode reçoit exactement une classe prédite et une valeur de confiance ; aucun n'est écarté. **235 épisodes (6,2 %)** franchissent le seuil de 0,70.

Composition prédite : `NETWORK_TELECOM` 859, `BUSINESS_SOFTWARE` 854, `IT_SERVICES` 492, `OTHER_DIGITAL` 462, `CYBERSECURITY` 316, `IT_INFRASTRUCTURE` 298, `OTHER` 173, `MIXED` 139, `CLOUD_HOSTING` 115, `DATA_BI` 86, `AI` 6. **Ces effectifs ne sont pas des parts de marché** : ce sont des prédictions portant le taux d'erreur du § 6.6, sur une cohorte définie par une règle CPV inclusive.

---

# 7. Usage des prédictions en aval

## 7.1 Deux barrières avant tout usage

Rien n'a été relancé mécaniquement pour onze classes. Une classe n'entre dans l'analyse en aval qu'en franchissant **deux** barrières fixées avant qu'aucune courbe ne soit ajustée.

**Barrière A — preuve du classifieur.** L'étiquette veut-elle dire quelque chose ? Une classe que le modèle ne sépare pas produit un groupe qui mélange plusieurs technologies, et une courbe ajustée sur ce groupe estime le mélange. La barrière exige une classe substantielle, un support annoté d'au moins 10 et un F1 hors pli d'au moins 0,65. Les trois classes de repli sont exclues d'office : `OTHER_DIGITAL` contient simultanément vidéosurveillance, RFID et maintenance de sites web ; la placer à côté de la cybersécurité dans une « comparaison entre technologies » inviterait à lire comme un effet technologique ce qui est en partie l'hétérogénéité du fourre-tout.

**Barrière B — support statistique.** Au moins 100 épisodes et 20 événements. Une classe parfaitement classée avec quatorze épisodes et un événement ne porte pas de courbe.

Cinq classes sur onze franchissent les deux : `CYBERSECURITY`, `NETWORK_TELECOM`, `IT_INFRASTRUCTURE`, `BUSINESS_SOFTWARE`, `IT_SERVICES`. `CLOUD_HOSTING` (115 épisodes, 11 événements) et `DATA_BI` (86, 15) échouent sur la barrière B ; `AI` sur les deux.

**Ce que la barrière A coûte.** Le même test du log-rank sur toutes les classes franchissant la seule barrière statistique — donc en réintégrant les résidus — donne $p = 0{,}000119$ contre $p = 0{,}0363$ avec la barrière. La barrière **affaiblit** le résultat, et elle a été conservée. C'est publié dans le rapport technique, et c'est l'élément le plus probant que l'analyse n'a pas été orientée vers un résultat.

## 7.2 Survie par technologie

| Classe | Épisodes | Événements | P(successeur ≤ 24 mois) | À risque à 24 mois |
|---|---:|---:|---:|---:|
| BUSINESS_SOFTWARE | 854 | 106 | 8,57 % | 673 |
| NETWORK_TELECOM | 859 | 140 | 6,49 % | 716 |
| CYBERSECURITY | 316 | 60 | 6,46 % | 246 |
| IT_SERVICES | 492 | 72 | 6,17 % | 393 |
| IT_INFRASTRUCTURE | 298 | 38 | 6,11 % | 239 |

Le test du log-rank multivarié sur ces cinq classes donne une statistique de 10,26 pour $p = 0{,}0363$ sur 416 événements : **le calendrier du successeur observable diffère entre les technologies analysées**. L'écart entre les extrêmes est de 2,5 points à 24 mois.

La confiance à accorder à ce résultat est modérée, et pour trois raisons cumulatives : l'événement est un successeur observable accepté par la règle figée, non un renouvellement ; les étiquettes de classe sont des **prédictions** portant le taux d'erreur du § 6.6, qui ne se propage pas dans les intervalles publiés ; et la comparaison n'est ajustée ni sur l'acheteur, ni sur la taille, ni sur la procédure.

## 7.3 Tendances par technologie

Cinq séries trimestrielles sont ajustées sur les classes retenues. **Aucune ne montre de tendance linéaire**, ni avant ni après correction de multiplicité : la plus petite valeur $p$ brute est 0,0563 (`NETWORK_TELECOM`, pente −0,17 épisode par trimestre), qui devient 0,28 après correction de Holm sur les cinq tests simultanés. La lecture est donc : sur la fenêtre observée et à ce niveau de bruit, **aucune tendance technologique n'est établie**.

## 7.4 Les trois incertitudes héritées

Toute figure au niveau technologique hérite, dans cet ordre : de l'**incertitude d'appariement** (l'événement lui-même dépend de la règle, § 5.7), de l'**incertitude du classifieur** (macro-F1 de 0,744, avec des erreurs concentrées sur les frontières de définition), et de l'**incertitude d'échantillonnage** (les intervalles publiés). Seule la troisième figure dans les intervalles de confiance. C'est acceptable pour une couche d'enrichissement explicitement présentée comme telle ; ce ne le serait pas pour une analyse de référence.

---

# 8. Tendances et détection de ruptures

## 8.1 Construction des séries

Pour un segment $s$ et un trimestre $q$, le compte est
$$N_{s,q} = \sum_i \mathbf{1}(S_i = s,\ Q_i = q).$$
Les séries couvrent **43 trimestres** (2015T2 à 2025T4) ; le premier trimestre 2015, partiel dans l'extraction, est exclu, et tous les trimestres suivants sont représentés, y compris les zéros. Les séries de montants sont absentes, faute de valeur attribuée canonique validée (§ 3.4).

L'objectif, conformément au cadrage, n'est pas de prévoir : c'est de **dater et qualifier des changements**.

## 8.2 Trois instruments, trois questions différentes

**PELT** (Killick, Fearnhead et Eckley, 2012) minimise
$$\sum_{r=0}^{m}\mathcal{C}\big(y_{\tau_r+1:\tau_{r+1}}\big) + \beta m,$$
où $\mathcal{C}$ est l'erreur quadratique intra-segment et $\beta = \lambda\log(n)$ après standardisation. Le premier terme récompense l'ajustement, le second facture chaque rupture — c'est ce qui empêche l'optimum de placer une rupture entre chaque paire de trimestres. Le résultat central utilise $\lambda = 1$, la sensibilité 0,5 et 2,0, et **une rupture n'est déclarée stable que si elle apparaît à un trimestre près sous les trois pénalités**. Cette exigence élimine la moitié des ruptures candidates.

**ADF et KPSS** testent des hypothèses nulles opposées — racine unitaire pour le premier, stationnarité en niveau pour le second — et sont donc lus ensemble. Ils ne sont pas forcés à s'accorder : un désaccord signifie que la série n'est pas classée proprement sur une fenêtre aussi courte, et c'est l'information utile.

**Le modèle de Markov caché** à trois états est ajusté sur la **variation trimestrielle** $\Delta N_t$, non sur le niveau. Ses états décrivent donc une direction typique de variation — recul, plateau, croissance — et non le niveau d'activité d'un segment. La probabilité rapportée est une probabilité a posteriori d'état courant, pas une prévision.

## 8.3 Matrice de signaux

| Segment | Direction récente | Pente (épisodes/trim.) | p brut | p Holm | p BH | Lecture | Dernière rupture stable |
|---|---|---:|---:|---:|---:|---|---|
| Ensemble | stable ou incertain | −0,11 | 0,921 | 1,000 | 0,989 | aucun signal | — |
| CPV-32 | stable ou incertain | −0,01 | 0,989 | 1,000 | 0,989 | aucun signal | 2020T2 |
| CPV-35 | stable ou incertain | +0,03 | 0,923 | 1,000 | 0,989 | aucun signal | — |
| **CPV-48** | **décroissante** | **−0,84** | **0,032** | 0,159 | 0,159 | **signal nominal seulement** | 2024T1 |
| CPV-72 | stable ou incertain | +0,70 | 0,285 | 1,000 | 0,714 | aucun signal | 2021T1 |

Cinq pentes sont ajustées et lues ensemble : les valeurs $p$ brutes sont donc accompagnées des ajustements de Holm (par famille) et de Benjamini-Hochberg (taux de fausses découvertes). Un segment dont le $p$ brut franchit le seuil exploratoire de 0,10 fixé à l'avance mais dont le $p$ de Holm ne le franchit pas est **un signal nominal à surveiller, pas un résultat**.

C'est le cas de CPV-48, et c'est la seule ligne du tableau qui demande une décision de rédaction. Elle est présentée telle quelle : le recul récent des marchés CPV-48 est le signal le plus net du panel, il ne survit pas à la correction pour les cinq segments testés, et il justifie une surveillance sur quelques trimestres supplémentaires — pas une conclusion, encore moins une prévision.

## 8.4 Ruptures et régimes

Trois ruptures seulement sont stables sous les trois pénalités : CPV-32 en 2020T2, CPV-48 en 2024T1 et CPV-72 en 2021T1. Ni la série d'ensemble ni CPV-35 n'en portent : leurs ruptures candidates disparaissent dès que la pénalité change. **Aucune n'est attribuée à une cause.** Une rupture PELT date un changement de niveau ; elle ne dit pas pourquoi. Attribuer la rupture de 2020T2 à la crise sanitaire serait plausible et non démontré, et le rapport s'en abstient : ces attributions demandent des preuves documentaires et une validation par les acteurs métier.

Le modèle de Markov caché place l'ensemble, CPV-32 et CPV-72 en régime de croissance au dernier trimestre (probabilités a posteriori 0,750, 0,992 et 0,594). Ce label peut sembler contredire les pentes à douze trimestres, qui sont nulles : il n'en est rien. La pente résume douze trimestres, le régime décrit la variation la plus récente. Les deux sont complémentaires, et le rapport ne les réconcilie pas artificiellement.

## 8.5 Ce que l'analyse de tendance établit

Sur la fenêtre observée, **aucun segment CPV ne montre de tendance linéaire récente survivant à la correction de multiplicité**, et aucune série technologique non plus. Trois ruptures de niveau sont datées de manière stable. La lecture opérationnelle est donc un dispositif de veille : maintenir la surveillance sur tous les segments, et suivre CPV-48 comme motif d'investigation. C'est un résultat plus faible que ce que le cadrage espérait, et il est cohérent avec la nature des données : quarante-trois trimestres à faibles effectifs ne portent pas d'inférence de tendance robuste.

---

# 9. Discussion

## A. Ce que le travail établit avec une confiance raisonnable

**Le texte des avis porte une segmentation métier que le CPV ne porte pas.** C'est le résultat le mieux établi du stage : différence appariée de +0,271 de macro-F1, intervalle [0,201 ; 0,340] estimé au bon niveau d'agrégation, sur des plis et un budget de recherche identiques. Il répond directement au problème du signal textuel posé par le cadrage, et il a une conséquence pratique immédiate : une lecture par technologie métier est possible là où la nomenclature administrative ne le permet pas.

**Les segments CPV diffèrent nettement par le délai avant successeur observable**, avec CPV-35 en tête. Le résultat survit à quatre perturbations indépendantes de la définition d'événement et à l'ajustement de détectabilité. Il peut être énoncé avec fermeté.

**Le pipeline est reproductible et structurellement cohérent.** Toutes les vérifications d'intégrité passent, chaque étape est rejouable par une commande unique, et un contrôle transversal vérifie que les artefacts publiés et les configurations disent la même chose.

## B. Ce que les modèles indiquent, sous condition

**Les niveaux de probabilité.** 4,6 % à douze mois et 6,7 % à vingt-quatre mois, avec une épaule à 36-48 mois, **sous la définition d'événement retenue**. Ces chiffres varient d'un facteur quatre et demi selon la règle d'appariement ; ils ne doivent jamais être cités seuls.

**L'effet accord-cadre.** Les accords-cadres montrent un successeur plus tôt (RR 1,751), mais l'association est **en partie de la détectabilité** : elle tombe à 1,617 lorsqu'on ajuste sur le volume de publication de l'acheteur. La direction résiste, l'ampleur non.

**Les technologies diffèrent par le calendrier** ($p = 0{,}036$ sur cinq classes et 416 événements), avec `BUSINESS_SOFTWARE` le plus rapide. Résultat non ajusté, portant des étiquettes prédites.

## C. Ce qui reste incertain

Par ordre d'importance : la **précision d'appariement** n'est pas validée de manière indépendante (0,875 sur huit liens, IC [0,529 ; 0,978] ; 0,700 en revue assistée, sous la cible de 0,80) ; les **niveaux absolus** dépendent de la règle ; la **discrimination hors période** du modèle de Cox est nulle ; le corpus d'annotation n'a **pas d'accord inter-annotateurs** ; et le **rappel** de la référence n'est pas indépendant du score qu'il évalue.

## D. Ce que Gigalis peut en faire aujourd'hui

Trois usages sont soutenus par les preuves disponibles.

**Une veille par cohorte, et non par contrat.** La table de probabilités conditionnelles par âge et segment (§ 5.3) permet de dire quels *groupes* de marchés entrent dans la fenêtre où un successeur devient probable. L'épaule de 36-48 mois est la période où l'attention porte le plus, et CPV-35 est le segment qui se remet en concurrence le plus vite. C'est une règle de priorisation de l'attention humaine, pas un score.

**Une segmentation métier exploitable.** Les huit classes technologiques offrent une lecture du portefeuille régional que les quatre divisions CPV ne donnent pas — avec une pureté moyenne de segment CPV de 0,34, chaque division mélange plusieurs métiers. Utilisée au niveau agrégé, avec le filtre de confiance pour les usages exigeants, elle peut servir à cartographier l'activité par technologie.

**Un dispositif de mesure reproductible.** Le pipeline peut être relancé chaque trimestre sur des données actualisées. C'est peut-être l'apport le plus durable : Gigalis dispose d'un instrument de mesure documenté, dont les limites sont connues, plutôt que d'un chiffre.

## E. Ce qu'il ne faut pas opérationnaliser

Un **score de renouvellement par marché**, sous quelque forme que ce soit : la validation hors période l'interdit. Un **classement individuel des marchés à surveiller** tiré du modèle de Cox, pour la même raison. La lecture d'une **classe technologique prédite comme un attribut observé** du marché, ou de ses effectifs comme des parts de marché. Une **prévision de tendance** par segment. Et toute **communication externe** citant la précision d'appariement comme une performance validée.

## F. Ce qui apporterait le plus, ensuite

1. **Une revue humaine indépendante** d'un échantillon aveugle de liens acceptés, ré-échantillonné et pré-spécifié. C'est le seul verrou avant toute revendication de précision, et cela apporte plus qu'une cinquième méthode d'appariement.
2. **Une seconde passe d'annotation technologique** par un annotateur distinct, permettant un $\kappa$ de Cohen. La courbe d'apprentissage est encore montante à n = 500 : plus d'annotations valent mieux qu'un modèle plus complexe.
3. **Les données d'adhésion Gigalis** (identité des membres, dates d'adhésion), sans lesquelles la question causale du cadrage — l'ouverture d'un accord-cadre mutualisé change-t-elle le comportement d'achat des membres ? — reste un schéma d'identification en différences de différences à adoption échelonnée (Callaway et Sant'Anna, 2021), décrit mais non estimé.
4. **Une expérience séparée sur la prédiction individuelle**, si cette piste est prioritaire : variables disponibles au moment de l'attribution (texte de l'objet, historique de l'acheteur), validation temporelle déclarée à l'avance, critère de succès annoncé, et publication du résultat même négatif. Le piège à éviter est nommé : la taille du vivier de candidats ferait monter la concordance en apprenant à prédire *qui est détectable*, pas *qui renouvelle*.

---

# 10. Limites

Elles sont classées par ce qu'elles peuvent faire aux résultats, non par catégorie.

**1. La validation de l'appariement n'est pas indépendante.** *Mécanisme* : les étiquettes de la référence proviennent d'une passe de modèle de langue contrôlée par sondage, et la revue de 60 couples est également assistée par modèle. La précision affichée pourrait donc être surestimée ou sous-estimée sans que rien ne le signale. *Effet* : tout chiffre de précision est provisoire ; la revue conservatrice donne d'ailleurs 0,700, sous la cible. *Atténuation en place* : freeze du seuil avant lecture, comptabilité stricte du successeur exact, intervalles publiés partout, protocole de revue indépendante écrit et échantillon préparé.

**2. Les erreurs d'appariement déforment à la fois le statut et la date.** *Mécanisme* : un faux lien crée un événement et une date d'événement ; un lien manqué crée une censure. *Effet* : les niveaux absolus ne sont pas une borne inférieure, contrairement à ce qu'une lecture rapide suggérerait. *Atténuation* : quatre bras d'événement, bande limite, recensure du risque « modèle type », et une règle de rédaction — ne jamais citer un niveau sans sa plage.

**3. La détectabilité n'est pas uniforme entre acheteurs.** *Mécanisme* : le score est un maximum sur un bloc, et les blocs sont de tailles très différentes. *Effet* : les acheteurs prolifiques produisent plus de liens, ce qui gonfle toute association avec une variable corrélée au volume de publication — au premier chef le statut d'accord-cadre. *Atténuation* : diagnostic publié (SMD +0,470) et modèle de sensibilité chiffrant l'atténuation.

**4. Les étiquettes technologiques sont des prédictions.** *Mécanisme* : macro-F1 de 0,744, avec des erreurs concentrées sur des frontières de définition. *Effet* : toute figure au niveau technologique porte une erreur de mesure qui n'entre pas dans ses intervalles. *Atténuation* : deux barrières avant tout usage en aval, une règle de rédaction interdisant de traiter une classe prédite comme un attribut observé, et l'exclusion des classes de repli.

**5. Le corpus d'annotation n'a pas de mesure de fiabilité.** *Mécanisme* : une seule passe, sans identifiant d'annotateur ni seconde lecture. *Effet* : aucun $\kappa$ de Cohen ; le plan L2 du cadrage, qui prévoyait deux annotateurs, n'est pas satisfait. *Atténuation* : trois incohérences internes documentées et laissées telles quelles comme plancher empirique, quotas d'annotation signalés, classe `AI` déclarée non évaluable.

**6. Le modèle de Cox viole la proportionnalité des risques.** *Mécanisme* : trois covariables, dont l'année d'attribution de façon massive, structurellement confondue avec la longueur du suivi. *Effet* : les coefficients sont des moyennes temporelles descriptives. *Atténuation* : diagnostic publié, interprétation restreinte, aucun usage prédictif.

**7. La cohorte est plus large que son nom.** *Mécanisme* : règle à au moins un code CPV au niveau épisode. *Effet* : 30,9 % des épisodes ont un CPV principal hors périmètre. *Atténuation* : formulation explicite, et mesure de l'effet du départage de segment (94,7 % de concordance).

**8. Les séries temporelles sont courtes et bruitées.** *Mécanisme* : 43 trimestres, faibles effectifs, ruptures de complétude documentaires en 2025. *Effet* : aucune tendance ne survit à la correction ; la détection de rupture propose des candidats sans les expliquer. *Atténuation* : stabilité des ruptures sous trois pénalités, correction de multiplicité dans les deux familles de tests, refus explicite de toute prévision.

**9. Portée géographique et temporelle.** Les résultats portent sur le Grand Ouest, 2015-2025, sur les marchés publiés au BOAMP. Ils ne s'étendent ni à la France entière, ni aux achats sous seuil, ni aux marchés passés via une centrale d'achat sans publication propre.

---

# 11. Conclusion

Le stage devait estimer la probabilité qu'un marché numérique génère un besoin d'achat identifiable dans les douze à vingt-quatre mois. Il a produit autre chose, et cette différence est le principal résultat.

**Ce qui a été construit.** Une chaîne reproductible qui va des 1 620 712 avis officiels du BOAMP à une cohorte de 3 800 marchés numériques attribués du Grand Ouest, avec une reconstruction d'épisodes qui empêche de compter plusieurs fois une même procédure, une variable de successeur observable dont la qualité est mesurée plutôt que supposée, et un contrôle transversal qui vérifie que les artefacts publiés et le code disent la même chose.

**Ce qui a été appris sur le calendrier.** La probabilité qu'un marché montre un successeur observable est estimée à 4,6 % à douze mois et 6,7 % à vingt-quatre mois, avec une accélération nette entre 36 et 48 mois compatible avec la durée usuelle des accords-cadres. Les segments diffèrent, CPV-35 étant le plus rapide, et les accords-cadres montrent un successeur plus tôt — pour partie parce qu'ils sont plus visibles.

**Ce qui a résisté aux épreuves de robustesse.** Les niveaux absolus varient d'un facteur quatre et demi selon la définition de l'événement ; les comparaisons relatives, elles, tiennent : CPV-35 conserve sa position sous quatre perturbations indépendantes. **Cette dissymétrie — niveaux fragiles, comparaisons stables — est le message scientifique central du travail**, et elle vaut mieux qu'un chiffre unique présenté comme certain.

**Ce que le volet textuel a apporté.** Un classifieur supervisé sur le seul objet du marché sépare onze classes technologiques métier avec une macro-F1 hors pli de 0,744, contre 0,473 pour le meilleur comparateur construit sur la nomenclature CPV, sur des plis et un budget identiques : une différence appariée de +0,271 dont l'intervalle exclut zéro. Le texte porte donc une information métier que l'administratif ne porte pas. Ses prédictions, filtrées par deux barrières explicites, font apparaître une différence de calendrier entre technologies ($p = 0{,}036$).

**Ce que l'analyse de tendance a montré.** Peu de choses, et c'est un résultat : aucune pente de segment ne survit à la correction pour tests multiples. Trois ruptures de niveau sont datées de façon stable, sans qu'aucune cause ne leur soit attribuée.

**Ce que la prédiction individuelle n'a pas atteint.** L'indice de concordance hors période du modèle de Cox est de 0,479 — indiscernable du hasard. Aucun score par marché n'est donc produit, et le tableau des vingt marchés prioritaires prévu par le cadrage a été remplacé par une logique de veille par cohorte, assumée comme telle. Un modèle qui ne discrimine pas hors période reste utile pour décrire une population ; il ne l'est pas pour classer un contrat.

**Ce qui n'a pas pu être entrepris.** La question causale — l'ouverture d'un accord-cadre mutualisé modifie-t-elle le comportement d'achat des membres de Gigalis ? — demande des données internes d'adhésion absentes du BOAMP ; le schéma d'identification est décrit, aucune estimation n'est produite. L'accord inter-annotateurs du corpus technologique demande un second annotateur. Et la validation externe de l'appariement demande une revue humaine indépendante, dont le protocole et l'échantillon sont prêts.

**La valeur pour Gigalis.** Elle n'est pas dans une prédiction. Elle est dans un cadre de mesure : une définition explicite de ce que l'on observe, un instrument reproductible pour la mesurer, une quantification honnête de ce qui est solide et de ce qui ne l'est pas, et une segmentation métier qui n'existait pas dans les données de départ. Passer d'une logique descriptive à une logique prédictive supposait une vérité terrain que la source ne fournit pas ; le travail montre précisément où elle manque, ce qu'il faudrait pour la produire, et ce que l'on peut déjà décider sans elle.

---

# Bibliographie

**Analyse de survie**

1. Kaplan, E. L. et Meier, P. (1958). *Nonparametric Estimation from Incomplete Observations*. Journal of the American Statistical Association, 53(282), 457-481.
2. Cox, D. R. (1972). *Regression Models and Life-Tables*. Journal of the Royal Statistical Society, Series B, 34(2), 187-220.
3. Grambsch, P. M. et Therneau, T. M. (1994). *Proportional Hazards Tests and Diagnostics Based on Weighted Residuals*. Biometrika, 81(3), 515-526.
4. Therneau, T. M. et Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*. Springer.
5. Kalbfleisch, J. D. et Prentice, R. L. (2002). *The Statistical Analysis of Failure Time Data*, 2ᵉ éd. Wiley.
6. Uno, H., Cai, T., Pencina, M. J., D'Agostino, R. B. et Wei, L. J. (2011). *On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data*. Statistics in Medicine, 30(10), 1105-1117.

**Appariement d'enregistrements**

7. Fellegi, I. P. et Sunter, A. B. (1969). *A Theory for Record Linkage*. Journal of the American Statistical Association, 64(328), 1183-1210.
8. Christen, P. et Goiser, K. (2007). *Quality and Complexity Measures for Data Linkage and Deduplication*. Dans : Quality Measures in Data Mining, Springer, 127-151.
9. Harron, K., Doidge, J., Knight, H., Gilbert, R., Goldstein, H., Cromwell, D. et van der Meulen, J. (2017). *A guide to evaluating linkage quality for the analysis of linked data*. International Journal of Epidemiology, 46(5), 1699-1710.
10. Doidge, J. C. et Harron, K. L. (2019). *Reflections on modern methods: linkage error bias*. International Journal of Epidemiology, 48(6), 2050-2060.

**Évaluation en données déséquilibrées**

11. Davis, J. et Goadrich, M. (2006). *The Relationship Between Precision-Recall and ROC Curves*. ICML 2006, 233-240.
12. Saito, T. et Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLoS ONE, 10(3), e0118432.

**Tests multiples**

13. Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics, 6(2), 65-70.
14. Benjamini, Y. et Hochberg, Y. (1995). *Controlling the False Discovery Rate*. Journal of the Royal Statistical Society, Series B, 57(1), 289-300.

**Séries temporelles et ruptures**

15. Killick, R., Fearnhead, P. et Eckley, I. A. (2012). *Optimal Detection of Changepoints With a Linear Computational Cost*. Journal of the American Statistical Association, 107(500), 1590-1598.
16. Truong, C., Oudre, L. et Vayatis, N. (2020). *Selective review of offline change point detection methods*. Signal Processing, 167, 107299.
17. Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*. Econometrica, 57(2), 357-384.

**Traitement du langage**

18. Blei, D. M., Ng, A. Y. et Jordan, M. I. (2003). *Latent Dirichlet Allocation*. Journal of Machine Learning Research, 3, 993-1022.
19. Devlin, J., Chang, M.-W., Lee, K. et Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019, 4171-4186.
20. Martin, L. et al. (2020). *CamemBERT: a Tasty French Language Model*. ACL 2020, 7203-7219.

**Commande publique et qualité des données**

21. Fazekas, M. et Kocsis, G. (2020). *Uncovering High-Level Corruption: Cross-National Objective Corruption Risk Indicators Using Public Procurement Data*. Political Research Quarterly, 73(1), 155-177.
22. Potin, L., Labatut, V., Morand, P.-H. et Largeron, C. (2023). *FOPPA: A database of French Open Public Procurement Award notices*. Scientific Data, 10, 303.
23. Siciliani, L., Tanzi, G., Basile, P. et Lops, P. (2023). *Automatic CPV Code Classification for Italian Public Tenders*. CLiC-it 2023.

**Cadre réglementaire et sources officielles**

24. API officielle du BOAMP, DILA, publiée sur data.gouv.fr.
25. Règlement (CE) nº 213/2008 de la Commission, relatif au vocabulaire commun pour les marchés publics (CPV).
26. Directive 2014/24/UE, article 33 (accords-cadres).
27. INSEE, définitions du SIREN et du SIRET.

**Inférence causale (perspective seulement)**

28. Angrist, J. D. et Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
29. Callaway, B. et Sant'Anna, P. H. C. (2021). *Difference-in-differences with multiple time periods*. Journal of Econometrics, 225(2), 200-230.

---

# Annexes

Les annexes sont accessoires : le rapport se lit sans elles. Chacune est appelée depuis le corps du texte.

**Annexe A — Notation et définitions formelles.** Unités et notation ($n$, $i$, $j$, $u_i$, $v_j$, $J_i$, $T_{ij}$, $Y_i$, $\tau_i$) ; règle de blocage ; règle de décision ; définition de la censure ; définition des familles de marchés. *(Appelée depuis § 4.2, § 4.3, § 5.1, § 6.3.)*

**Annexe B — Qualité des données.** Table complète de complétude par champ et par année ; onze contrôles d'intégrité avec leur valeur attendue ; comptes de méthodes de reconstruction d'épisodes et d'arêtes acceptées/rejetées ; figure de complétude des durées par année. *(Appelée depuis § 3.2, § 3.4.)*

**Annexe C — Évaluation de l'appariement.** Matrices de confusion au niveau ancre et au niveau successeur exact pour les quatre méthodes ; courbes ROC et précision-rappel au niveau couple ; balayage complet des seuils de 0,50 à 0,80 ; fiche descriptive de la référence régionale et ses quatre limites ; audit de blocage acheteur. *(Appelée depuis § 4.4, § 4.5, § 4.6.)*

**Annexe D — Détail de l'analyse de survie.** Courbe de Kaplan-Meier avec effectifs à risque ; courbes par segment ; table de Cox complète avec intervalles ; diagnostics de proportionnalité ; comparaison paramétrique et ajustement linéarisé ; quatre tables de sensibilité ; diagnostic de sélection complet ; liste de veille stratifiée par segment. *(Appelée depuis § 5.2, § 5.4, § 5.5, § 5.7.)*

**Annexe E — Détail du volet textuel.** Définitions des onze classes et règles d'annotation ; métriques par classe avec support ; matrice de confusion ; courbe d'apprentissage ; triage des trente erreurs ; registre des spécifications explorées ; table de fiabilité de la confiance et balayage des seuils. *(Appelée depuis § 6.2, § 6.6, § 6.7.)*

**Annexe F — Détail des tendances.** Matrice de signaux complète ; ruptures PELT sous les trois pénalités ; tests ADF et KPSS par segment ; paramètres du modèle de Markov caché ; séries trimestrielles. *(Appelée depuis § 8.3, § 8.4.)*

**Annexe G — Reproductibilité.** Commande unique de relance du pipeline ; liste ordonnée des étapes et de leurs sorties ; versions de l'environnement ; empreintes SHA-256 des entrées de référence et des sorties canoniques ; inventaire des tests automatisés ; emplacement des artefacts. *(Appelée depuis § 1, § 9.D.)*
