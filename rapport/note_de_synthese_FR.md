# Note de synthèse

> **Brouillon archivé.** La note de synthèse française faisant autorité est
> `rapport/BOAMP_Report_EN_Overleaf/sections/99_synthesis_fr.tex`. Cette version
> Markdown est conservée pour la provenance et n'est pas synchronisée chiffre
> par chiffre.

**{NOM Prénom} — ENSAE 2ᵉ année — Stage d'application 2025-2026**
**Gigalis — Analyse et modélisation des marchés publics numériques à partir des données du BOAMP**

*Cette note est indépendante du rapport et se lit seule.*

---

## Le contexte

Gigalis est le groupement d'intérêt public du numérique en Pays de la Loire. Il agit notamment comme centrale d'achat : il négocie des contrats mutualisés — cloud, cybersécurité, réseaux, intelligence artificielle — que les collectivités et établissements publics peuvent utiliser sans y être tenus. L'intérêt de ces contrats dépend du moment où ils sont ouverts. Trop tôt, ils restent inutilisés ; trop tard, les acheteurs ont déjà passé leurs propres marchés. D'où la question posée au stage : peut-on anticiper, à partir des données publiques, quand un besoin d'achat numérique va réapparaître ?

## Ce que nous savons

Les avis de marchés publics sont publiés au BOAMP, un bulletin officiel accessible librement. Le stage a traité **1,6 million d'avis** publiés entre 2015 et 2025, et en a tiré une population d'étude de **3 800 marchés numériques attribués** dans le Grand Ouest.

Un obstacle est apparu d'emblée, et il a structuré tout le travail : **le BOAMP n'indique jamais qu'un marché en renouvelle un autre**. Il n'existe donc aucune liste de renouvellements à partir de laquelle apprendre. Il a fallu construire l'information : pour chaque marché attribué, rechercher parmi les marchés publiés ensuite par le même acheteur celui qui poursuit le même besoin, et n'accepter ce rapprochement que lorsque la ressemblance est forte. Ce que l'étude mesure est donc un **« successeur observable »** — un marché ultérieur qui prend visiblement le relais — et non un renouvellement juridique. La distinction est maintenue partout, parce qu'elle change ce que l'on a le droit de conclure.

## Ce que les modèles indiquent

Sur cette base, environ **un marché sur sept** voit apparaître un successeur observable avant fin 2025. La probabilité qu'un successeur devienne visible est estimée à **4,6 % dans les douze mois** suivant l'attribution et **6,7 % dans les vingt-quatre mois**. Ces chiffres paraissent faibles parce que la plupart des marchés sont encore trop récents pour avoir été remis en concurrence, ce que la méthode statistique employée prend en compte explicitement.

Le résultat le plus utile n'est pas ce niveau moyen, mais son profil dans le temps : la probabilité **s'élève nettement entre la troisième et la quatrième année** du marché, ce qui correspond à la durée usuelle des contrats-cadres publics. C'est la fenêtre où la surveillance a le plus de valeur.

Deux différences sont solides. Les marchés relevant du **matériel de sécurité et de surveillance** se remettent en concurrence sensiblement plus vite que les autres. Et les **accords-cadres** montrent un successeur plus tôt — en partie parce que les acheteurs qui les utilisent publient davantage, donc sont plus faciles à suivre : une partie de l'écart tient à la visibilité, pas au comportement d'achat.

Le second apport est d'une autre nature. Les marchés publics sont classés par une nomenclature administrative européenne, trop large pour un usage métier : une même catégorie mélange téléphonie, logiciels et prestations informatiques. Un modèle entraîné sur **500 avis annotés à la main** apprend à retrouver, à partir du seul texte de l'objet du marché, une segmentation métier en huit familles — cloud, cybersécurité, réseaux, infrastructures, logiciels métier, données, intelligence artificielle, services informatiques. Il est **nettement meilleur que la nomenclature officielle** pour cet usage, et l'écart est mesuré avec sa marge d'incertitude. Gigalis dispose donc d'une lecture par technologie qui n'existait pas dans les données de départ.

## Ce qui reste incertain

Trois points doivent être dits clairement.

**La qualité des rapprochements n'est pas validée de façon indépendante.** Elle a été mesurée sur un échantillon de référence constitué pour le projet, et sur un très petit nombre de liens acceptés. L'ordre de grandeur est encourageant, l'intervalle d'incertitude est large. Une relecture par un spécialiste de la commande publique est le contrôle qui manque ; le protocole et l'échantillon sont prêts.

**Les niveaux absolus dépendent de la règle choisie.** Selon que l'on est plus ou moins exigeant sur le rapprochement, le nombre de successeurs identifiés varie de 296 à 1 332. Les pourcentages ne doivent donc jamais être cités seuls. En revanche — et c'est le résultat central du travail — **les comparaisons entre segments restent stables** quand la règle change. Ce qui est fragile, ce sont les niveaux ; ce qui tient, ce sont les écarts.

**La prédiction marché par marché n'est pas atteinte.** Un modèle a été entraîné sur les années anciennes puis testé sur les années récentes sans réajustement : sa capacité à classer correctement les marchés est **équivalente au hasard**. Aucun score individuel n'est donc produit, et la liste des « vingt marchés les plus susceptibles d'être renouvelés » prévue au départ n'a pas été livrée. Ce constat est un résultat, pas un échec de mise en œuvre : il indique que les informations disponibles au moment de l'attribution ne suffisent pas à prédire un calendrier individuel.

## Ce que cela signifie pour Gigalis

Le travail fournit **un instrument de mesure et de veille, pas un moteur de prédiction**.

Concrètement, il permet de dire quels *groupes* de marchés entrent dans la période où un successeur devient probable — par segment et par ancienneté — et de concentrer l'attention humaine sur la fenêtre de trois à quatre ans, en priorité sur les segments les plus rapides. Il fournit une segmentation technologique du portefeuille régional utilisable pour cartographier l'activité. Et il laisse un pipeline reproductible, documenté et testé, qui peut être relancé sur des données actualisées.

Il ne permet pas, en l'état, d'affirmer qu'un marché précis sera renouvelé, ni de prévoir l'évolution d'un segment : sur la période observée, aucune tendance de volume ne résiste aux tests statistiques appropriés.

## Ce que nous recommandons ensuite

1. Faire relire un échantillon de rapprochements par un spécialiste indépendant, avant toute communication chiffrée sur la précision de la méthode.
2. Faire annoter une partie du corpus technologique par une seconde personne, afin de mesurer la fiabilité des étiquettes.
3. Fournir les données internes d'adhésion à Gigalis pour traiter la question causale — l'ouverture d'un contrat mutualisé change-t-elle réellement le comportement d'achat des membres ? — que les données publiques seules ne permettent pas d'aborder.
4. Traiter la prédiction individuelle comme une expérience distincte, avec ses variables, son protocole de validation et son critère de réussite fixés à l'avance, plutôt que comme un ajustement du modèle actuel.
