# Catalogue des features candidates

Document de traçabilité de toutes les features envisagées pour le master dataset, **y compris celles rejetées** (avec justification).

Sert de complément au plan général [`06_features_engineering_plan.md`](06_features_engineering_plan.md) en descendant au niveau des features individuelles.

---

## Conventions

### Statuts

| Symbole | Signification |
|---|---|
| 💭 | **Proposée** -- listée comme candidate, aucune décision prise |
| 🟡 | **À tester** -- décision de la construire pour évaluation empirique (validée ensemble) |
| 🟢 | **Retenue** -- sera incluse dans le master dataset final (validée ensemble) |
| 🔴 | **Rejetée** -- non incluse, justification dans la fiche (validée ensemble) |
| ⏸ | **En attente** -- bloquée par une dépendance externe (info FFE, etc.) |

**Important** : aucun statut autre que 💭 ne doit apparaître sans **discussion explicite** entre Bryan et l'assistant. Le passage 💭 -> 🟡/🟢/🔴 est une décision méthodologique, pas une appréciation unilatérale.

### Format de fiche

Chaque feature est décrite avec les **10 champs** suivants :

- **Nom** : identifiant snake_case
- **Définition** : description en français
- **Formule** : pseudo-code / expression pandas
- **Source colonnes** : colonnes brutes utilisées
- **Couverture** : % de chevaux de la cohorte T1+N1≥10 avec valeur non-NaN
- **Motivation** : justification (littérature, intuition métier)
- **Risques** : leakage, redondance, sémantique floue
- **Statut** : 🟢 / 🟡 / 🔴 / ⏸
- **Décision** : aucune décision prise -- à discuter
- **Date décision** : YYYY-MM-DD

### Cohorte de référence

Toutes les features sont calculées sur la **cohorte T1 + filtre N1 ≥ 10 participations** :
- Né 2006-2013, hors poney (race + libellé), discipline SO
- ≥ 10 participations totales sur la carrière
- **47 683 chevaux** avec cible `hauteur_max_validée` calculable

### Fenêtre temporelle des features

Sauf mention contraire, les features sont calculées sur la **fenêtre 4-7 ans** (cohérent avec les cycles SHF jeunes chevaux et la nécessité d'éviter le leakage avec la cible).

---

## Famille 1 -- Activité / Volume

**Motivation** : famille universelle dans la littérature et l'industrie (EquiRatings, JPR, Hippomundo). Capte l'engagement du cheval, la régularité, et la précocité.

**Risque principal** : aucun majeur (pas de leakage, pas de choix sémantique délicat).

### Features

#### `nb_participations_4_7`

- **Définition** : nombre total de participations (lignes) sur la fenêtre 4-7 ans
- **Formule** : `df[in_fenetre_4_7].groupby('IDCHEVAL').size()`
- **Source colonnes** : aucune spécifique (compte de lignes)
- **Couverture** : 100% (par construction)
- **Motivation** : feature de volume universelle, présente dans toutes les approches
- **Risques** : aucun
- **Statut** : 🟡 À tester
- **Décision** : retenue pour construction et évaluation. Volume = signal incontournable, sans leakage ni piège.
- **Date décision** : 2026-05-02

#### `nb_evenements_4_7`

- **Définition** : nombre d'événements (concours/weekends) distincts sur 4-7 ans
- **Formule** : `df[in_fenetre_4_7].groupby('IDCHEVAL')['NUMERO_EVENEMENT2'].nunique()`
- **Source colonnes** : `NUMERO_EVENEMENT2`
- **Couverture** : 100%
- **Motivation** : capte la "fréquence de sorties" indépendamment du nombre d'épreuves par sortie. Distingue un cheval intensivement engagé sur peu de concours (style "rentabilisé") d'un cheval qui voyage beaucoup (style "régulier / aguerri par diversité").
- **Risques** : redondance partielle avec `nb_participations_4_7`, mais signal complémentaire
- **Statut** : 🟡 À tester
- **Décision** : retenue. Capture une dimension de style d'engagement non-redondante avec le volume brut.
- **Date décision** : 2026-05-02

#### `nb_participations_4ans` / `_5ans` / `_6ans` / `_7ans`

- **Définition** : nombre de participations à chaque âge (4 features distinctes)
- **Formule** : `df[in_fenetre_4_7 & (AGE==X)].groupby('IDCHEVAL').size()` pour X = 4,5,6,7
- **Source colonnes** : `AGE`
- **Couverture** : 100% (NaN remplacé par 0 = "n'a pas couru cette année-là")
- **Motivation** : encode la temporalité explicite. Permet au modèle de "voir" un cheval qui monte en charge progressivement, qui plafonne, ou qui décroche -- patterns invisibles dans un agrégat global.
- **Risques** : 4 features potentiellement corrélées entre elles
- **Statut** : 🟡 À tester
- **Décision** : retenues. Base de la temporalité fine, complémentaire des agrégats globaux.
- **Date décision** : 2026-05-02

#### `evolution_5_4` / `evolution_6_5` / `evolution_7_6`

- **Définition** : variations inter-annuelles du nombre de participations (3 features)
- **Formule** :
  ```python
  evolution_5_4 = nb_participations_5ans - nb_participations_4ans
  evolution_6_5 = nb_participations_6ans - nb_participations_5ans
  evolution_7_6 = nb_participations_7ans - nb_participations_6ans
  ```
- **Source colonnes** : `AGE` (via les 4 valeurs annuelles)
- **Couverture** : 100%
- **Motivation** : encode explicitement les transitions année par année. Utile pour les modèles linéaires qui ne calculent pas spontanément des différences entre features. Capture acceleration / deceleration / instabilité.
- **Risques** : redondance pour les modèles non-linéaires (tree-based) qui peuvent reconstruire les deltas à la volée. Mais coût marginal et utile pour interprétabilité du baseline linéaire.
- **Statut** : 🟡 À tester
- **Décision** : retenues pour interprétabilité du baseline linéaire. Coût marginal, bénéfice sur lisibilité.
- **Date décision** : 2026-05-02

#### `nb_annees_actives_4_7`

- **Définition** : nombre d'années (parmi 4, 5, 6, 7) avec ≥ 1 participation
- **Formule** : `df[in_fenetre_4_7].groupby('IDCHEVAL')['AGE'].nunique()`
- **Source colonnes** : `AGE`
- **Couverture** : 100% (valeurs 1 à 4)
- **Motivation** : régularité de l'engagement, complémentaire au volume brut. Un cheval qui court 30 fois étalé sur 4 ans n'est pas le même qu'un cheval qui court 30 fois concentré sur une année.
- **Risques** : aucun
- **Statut** : 🟡 À tester
- **Décision** : retenue. Régularité = signal complémentaire au volume.
- **Date décision** : 2026-05-02

#### `age_premiere_participation`

- **Définition** : âge de la première participation dans la fenêtre (4, 5, 6 ou 7)
- **Formule** : `df[in_fenetre_4_7].groupby('IDCHEVAL')['AGE'].min()`
- **Source colonnes** : `AGE`
- **Couverture** : 100% (par construction si le cheval a ≥1 participation 4-7)
- **Motivation** : capte la précocité du cheval. Souvent associée au pedigree et à l'environnement d'élevage (un cheval qui débute à 4 ans = suivi de près par un éleveur sérieux).
- **Risques** : aucun majeur (la précocité est observée tôt, pas de leakage avec la cible)
- **Statut** : 🟡 À tester
- **Décision** : retenue. Trait de précocité = signal métier reconnu.
- **Date décision** : 2026-05-02

#### `age_derniere_participation_4_7`

- **Définition** : âge de la dernière participation dans la fenêtre 4-7
- **Formule** : `df[in_fenetre_4_7].groupby('IDCHEVAL')['AGE'].max()`
- **Source colonnes** : `AGE`
- **Couverture** : 100%
- **Motivation** : capter un "abandon précoce" potentiel
- **Risques** : ⚠ leakage léger -- un cheval qui s'arrête à 5 ans est probablement de bas niveau (cible faible). De plus, signal ambigu (borne temporelle simple, mélange abandon, vente, blessure, etc.)
- **Statut** : 🔴 Rejetée
- **Décision** : remplacée par `a_saison_blanche_4_7` qui capture une information qualitative équivalente (interruption de carrière) avec moins de risque de leakage et un signal plus précis (détection de trou plutôt que simple borne temporelle).
- **Date décision** : 2026-05-02

#### `a_saison_blanche_4_7`

- **Définition** : indicateur booléen valant True si le cheval présente au moins une année blanche (sans aucune participation) entre sa première et sa dernière année active dans la fenêtre 4-7.
- **Formule** :
  ```python
  amplitude = (max_age_4_7 - min_age_4_7) + 1
  nb_annees_actives = nunique(AGE) in 4_7
  a_saison_blanche_4_7 = amplitude > nb_annees_actives
  ```
- **Source colonnes** : `AGE` (uniquement)
- **Couverture** : 100% (booléen toujours définissable sur les chevaux ayant ≥1 participation 4-7)
- **Motivation** : détecte mathématiquement un trou dans la carrière du jeune cheval. Signature potentielle d'une blessure (tendinite, suros, opération), d'un retard de croissance, ou d'une mise au pré prolongée. Red flag métier reconnu : un cheval qui a "cassé" sur petites épreuves est statistiquement moins susceptible d'atteindre un haut niveau.
- **Risques** :
  - Signal possiblement rare (la majorité des chevaux ont une activité continue) -- mais c'est précisément ce qui en fait un signal fort quand il s'allume
  - Pas de leakage tautologique : un gap au sein de 4-7 ne détermine pas la cible (calculée sur toute la carrière)
- **Statut** : 🟡 À tester
- **Décision** : retenue. Remplace avantageusement `age_derniere_participation_4_7`. Approche mathématiquement propre (amplitude vs actives), signal interprétable métier.
- **Date décision** : 2026-05-02

#### `delta_participations_4_7`

- **Définition** : `nb_participations_7ans - nb_participations_4ans`
- **Formule** : différence des features annuelles
- **Source colonnes** : `AGE`
- **Couverture** : 100%
- **Motivation** : trajectoire simple (montée / descente d'engagement)
- **Risques** : mathématiquement = somme des 3 deltas inter-annuels (`evolution_5_4 + evolution_6_5 + evolution_7_6`)
- **Statut** : 🔴 Rejetée
- **Décision** : redondante avec les 3 deltas fins inter-annuels (somme exacte). Aucune information unique apportée.
- **Date décision** : 2026-05-02

#### `nb_mois_actifs_4_7`

- **Définition** : nombre de mois calendaires distincts avec ≥ 1 participation sur 4-7 ans
- **Formule** : `df[in_fenetre_4_7].assign(mois=DATEEPREUVE.dt.to_period('M')).groupby('IDCHEVAL')['mois'].nunique()`
- **Source colonnes** : `DATEEPREUVE`
- **Couverture** : 100%
- **Motivation** : régularité intra-annuelle. Distingue saison étalée (mars-novembre, profil "cheval bien géré") de saison concentrée (3-4 mois de pic, profil "qualification à l'arrache" ou risque d'usure). Dimension complémentaire à `nb_annees_actives_4_7` (régularité inter-annuelle).
- **Risques** : colinéarité positive avec `nb_participations_4_7` -- gérable pour modèles d'arbre, à surveiller pour linéaire
- **Statut** : 🟡 À tester
- **Décision** : retenue. Capture une dimension de répartition temporelle non couverte par les autres features.
- **Date décision** : 2026-05-02

#### `intensite_moyenne_mensuelle`

- **Définition** : nombre moyen de participations par mois d'activité dans la fenêtre 4-7 ans
- **Formule** : `nb_participations_4_7 / nb_mois_actifs_4_7`
- **Source colonnes** : dérivée de `AGE`, `DATEEPREUVE`
- **Couverture** : 100% (dénominateur ≥ 1 par construction)
- **Motivation** : capte la concentration de l'effort sous forme de ratio. Permet à un modèle linéaire de capturer directement l'interaction "beaucoup de participations + peu de mois = enchaînement intensif" sans avoir à composer plusieurs features. Pour un modèle d'arbre, exprime cette interaction en un seul split.
- **Risques** :
  - Dérivée des deux autres features → redondance pour modèles d'arbre, mais coût marginal
  - L'interprétation métier "ratio élevé = risque sanitaire" est une hypothèse spéculative à valider empiriquement, pas une vérité acquise
- **Statut** : 🟡 À tester
- **Décision** : retenue. Utile pour interprétabilité du baseline linéaire et pour exprimer en 1 feature une interaction non-triviale.
- **Date décision** : 2026-05-02

#### `pente_participations`

- **Définition** : pente de régression linéaire de `nb_participations_an` sur `AGE` (4 points)
- **Formule** : `linregress(AGE, nb_participations).slope`
- **Source colonnes** : `AGE`
- **Couverture** : 100%
- **Motivation** : encoder la trajectoire (croissance/décroissance) en 1 scalaire
- **Risques** : double problème : (1) redondante avec les 4 valeurs annuelles + 3 deltas inter-annuels déjà retenus ; (2) une pente force une ligne droite, ce qui écrase la réalité non-linéaire d'une carrière (une carrière en V à cause d'une blessure devient invisible)
- **Statut** : 🔴 Rejetée
- **Décision** : redondante avec les features de temporalité déjà retenues, et trompeuse pour les carrières non-linéaires.
- **Date décision** : 2026-05-02

#### `annee_pic_activite`

- **Définition** : âge (4, 5, 6 ou 7) avec le plus de participations
- **Formule** : `argmax(nb_participations_par_age)`
- **Source colonnes** : `AGE`
- **Couverture** : 100%
- **Motivation** : forme de carrière (early peaker = cheval précoce/usé/commercialisé tôt vs late peaker = cheval préservé). Les modèles d'arbre captent mal les fonctions argmax (nécessitent plusieurs splits orthogonaux profonds), pré-calculer cette feature économise ce travail au modèle.
- **Risques** : redondance théorique avec les 4 valeurs annuelles, mais en pratique la pré-calculer simplifie la tâche du modèle
- **Statut** : 🟡 À tester
- **Décision** : retenue. Les arbres de décision peinent à reconstruire l'argmax, donc encoder cette feature catégorielle apporte une valeur réelle.
- **Date décision** : 2026-05-02

#### `nb_participations_par_evenement`

- **Définition** : `nb_participations_4_7 / nb_evenements_4_7`
- **Formule** : ratio
- **Source colonnes** : dérivée
- **Couverture** : 100%
- **Motivation** : miroir d'`intensite_moyenne_mensuelle` au niveau événement. Capture l'intensité par sortie : un cheval engagé dans plusieurs épreuves le même weekend (ratio élevé) vs un cheval engagé dans 1 seule épreuve par sortie (ratio = 1). Réalité équestre : aux concours jeunes chevaux, certains cavaliers engagent 2 jours sur 2, d'autres 1 jour pour préserver.
- **Risques** : redondance pour modèles d'arbre (dérivable des 2 sources), mais coût marginal et utile pour interprétabilité
- **Statut** : 🟡 À tester
- **Décision** : retenue. Capture une dimension d'intensité de sortie complémentaire à l'intensité mensuelle.
- **Date décision** : 2026-05-02

#### `duree_carriere_jeunesse_jours`

- **Définition** : `max(DATEEPREUVE) - min(DATEEPREUVE)` sur la fenêtre 4-7 ans, en jours
- **Formule** : `(date_max - date_min).days`
- **Source colonnes** : `DATEEPREUVE`
- **Couverture** : 100%
- **Motivation** : capture l'**amplitude** temporelle (différente de la **densité** mesurée par `nb_mois_actifs_4_7`). Exemple : 2 chevaux avec 2 mois actifs peuvent avoir des amplitudes très différentes (mars-avril vs mars-octobre).
- **Risques** : edge case "one-shot" (cheval avec 1 seule participation → 0 jour, à gérer)
- **Statut** : 🟡 À tester
- **Décision** : retenue. Capture l'amplitude temporelle, signal distinct de la densité (`nb_mois_actifs`).
- **Date décision** : 2026-05-02

#### `jours_moyens_entre_sorties`

- **Définition** : nombre moyen de jours entre deux participations consécutives sur la fenêtre 4-7 ans
- **Formule** : `duree_carriere_jeunesse_jours / (nb_participations_4_7 - 1)` (NaN si nb_participations == 1)
- **Source colonnes** : dérivée de `DATEEPREUVE`, `AGE`
- **Couverture** : ~99% (NaN pour les chevaux à 1 participation unique dans 4-7)
- **Motivation** : capture le **rythme global** sur la fenêtre, conceptuellement distinct d'`intensite_moyenne_mensuelle` qui mesure l'intensité dans les mois actifs uniquement. Un cheval qui sort 4 fois groupées en 1 mois puis pause 11 mois aura un rythme global espacé (~90 jours) malgré une intensité mensuelle élevée. Métrique de récupération biologique potentielle.
- **Risques** :
  - Edge case `nb_participations == 1` → division par zéro, à gérer (NaN ou flag)
  - L'interprétation "rythme industriel = mauvais, rythme d'or = bon" est une hypothèse à valider empiriquement, pas une vérité acquise
- **Statut** : 🟡 À tester
- **Décision** : retenue. Capture une dimension de rythme global complémentaire des intensités locales.
- **Date décision** : 2026-05-02

#### `max_participations_annee_4_7`

- **Définition** : maximum du nombre de participations sur une année (parmi 4, 5, 6, 7)
- **Formule** : `max(nb_participations_par_age)`
- **Source colonnes** : `AGE`
- **Couverture** : 100%
- **Motivation** : capturer le pic d'engagement
- **Risques** : redondante avec les 4 valeurs annuelles + `annee_pic_activite` (qui localise le pic, le seuil sur l'année concernée donnera l'amplitude)
- **Statut** : 🔴 Rejetée
- **Décision** : redondante. Les 4 valeurs annuelles + `annee_pic_activite` couvrent déjà l'information.
- **Date décision** : 2026-05-02

---

## Famille 2 -- Performance financière (gains)

**Motivation** : phénotype IFCE de référence (`log(GAINS+1)`), Horsetelex ISV, Hippomundo. Standard académique.

**Risque principal** : distribution très asymétrique (~70% des participations à 0€) -> à transformer.

### Convention de stockage

Les features de cette famille sont **stockées en valeurs brutes (€)** dans le master dataset. La transformation `log(GAINS + 1)` standardisée par année sera appliquée **au moment de la modélisation** : pas à la construction.

**Justification** :
- Sépare proprement le stockage des features (valeurs brutes interprétables) de leur transformation pour le modèle
- Permet de tester plusieurs transformations sans recalculer (log+1, racine, percentile, brut)
- Conserve la lisibilité pour analyses descriptives et graphiques

### Note sur la déformation log(x+1)

La transformation `log(x+1)` introduit une déformation non-triviale sur les **petites valeurs** (jusqu'à +11% pour les gains à 5€). Vérification empirique sur la cohorte 4-7 :
- 62% des participations à 0€ → la transformation est nécessaire (log(0) = -∞ inexploitable)
- 22,8% entre 1-100€, mais concentrés sur les valeurs hautes (mode à 19€, pics aux montants réglementaires)
- Sur les valeurs réellement présentes (12-25€), déformation moyenne 1-3%, acceptable
- Les valeurs vraiment problématiques (1-5€) ne représentent que ~0,4% des participations

Conclusion : `log(GAINS+1)` reste le standard de référence, déformation acceptée comme moindre mal vs log(0) indéfini.

### Note sur la normalisation par année (anti-inflation)

L'inflation française 2010-2025 (~25-30%) + l'évolution du barème FFE biaisent la comparaison absolue des gains entre cohortes (cheval né 2006 vs 2013). Solution retenue : **standardisation par année** (z-score intra-annuel après transformation log) au moment de la modélisation. Capture inflation + évolutions structurelles (sponsoring, professionnalisation) sans nécessiter de données externes.

### Features

#### `gains_total_4_7`

- **Définition** : somme totale des gains (€) sur la fenêtre 4-7 ans
- **Formule** : `df[in_fenetre_4_7].groupby('IDCHEVAL')['GAINS'].sum()`
- **Source colonnes** : `GAINS`
- **Couverture** : ~100% (NaN sur GAINS à 0,01% → remplacés par 0)
- **Motivation** : métrique financière de référence dans la littérature (Anne Ricard / IFCE utilise `log(GAINS+1)` standardisé comme phénotype BLUP). Mesure objective et synthétique de la performance économique en jeunesse.
- **Risques** : distribution très asymétrique (62% à 0€). Pas de leakage tautologique avec la cible. Corrélation positive attendue avec la cible.
- **Statut** : 🟡 À tester
- **Décision** : retenue. Standard académique, base de la famille gains.
- **Date décision** : 2026-05-03

#### `gains_4ans` / `gains_5ans` / `gains_6ans` / `gains_7ans`

- **Définition** : somme des gains à chaque âge précis (4 features distinctes)
- **Formule** : `df[in_fenetre_4_7 & (AGE==X)].groupby('IDCHEVAL')['GAINS'].sum()` pour X = 4,5,6,7
- **Source colonnes** : `GAINS`, `AGE`
- **Couverture** : 100% (0€ si pas de participation à cet âge)
- **Motivation** : encode la temporalité fine. Distingue un cheval qui passe de 0€ à 1500€ (progression) d'un cheval stable à 350€/an.
- **Risques** : 4 features potentiellement corrélées entre elles
- **Statut** : 🟡 À tester
- **Décision** : retenues, par cohérence avec l'approche temporelle de la Famille 1.
- **Date décision** : 2026-05-03

#### `evolution_gains_5_4` / `evolution_gains_6_5` / `evolution_gains_7_6`

- **Définition** : variations inter-annuelles des gains (3 features distinctes)
- **Formule** :
  ```python
  evolution_gains_5_4 = gains_5ans - gains_4ans
  evolution_gains_6_5 = gains_6ans - gains_5ans
  evolution_gains_7_6 = gains_7ans - gains_6ans
  ```
- **Source colonnes** : `GAINS`, `AGE` (via les 4 features annuelles)
- **Couverture** : 100%
- **Motivation** : encode explicitement les transitions année par année. Utile pour modèles linéaires. Capture acceleration / deceleration de la performance financière.
- **Risques** : redondance pour modèles d'arbre (dérivable des 4 valeurs annuelles), mais coût marginal
- **Statut** : 🟡 À tester
- **Décision** : retenues par cohérence avec les deltas inter-annuels validés en Famille 1 pour `nb_participations`.
- **Date décision** : 2026-05-03

#### `gains_par_participation_4_7`

- **Définition** : ratio gains totaux / nombre de participations sur la fenêtre 4-7 ans
- **Formule** : `gains_total_4_7 / nb_participations_4_7`
- **Source colonnes** : dérivée
- **Couverture** : 100% (dénominateur ≥ 1 par construction de la cohorte)
- **Motivation** : mesure de **rentabilité par round**. Différencie un cheval qui gagne 500€ en 2 sorties (efficient) d'un cheval qui gagne 500€ en 30 sorties (volume). Capture la qualité plutôt que la quantité.
- **Risques** : redondance pour modèles d'arbre, mais utile pour interprétabilité du baseline linéaire
- **Statut** : 🟡 À tester
- **Décision** : retenue. Capture une dimension d'efficacité non couverte par la somme brute.
- **Date décision** : 2026-05-03

#### `gains_par_evenement_4_7`

- **Définition** : ratio gains totaux / nombre d'événements (concours) sur la fenêtre 4-7 ans
- **Formule** : `gains_total_4_7 / nb_evenements_4_7`
- **Source colonnes** : dérivée
- **Couverture** : 100%
- **Motivation** : mesure de **rentabilité par sortie**. Miroir de `gains_par_participation_4_7` au niveau événement. Distingue un cheval qui rentabilise ses déplacements en concours d'un cheval qui sort souvent sans gain.
- **Risques** : redondance pour modèles d'arbre, mais utile pour interprétabilité
- **Statut** : 🟡 À tester
- **Décision** : retenue. Pendant événement du ratio par participation.
- **Date décision** : 2026-05-03

---

## Famille 3 -- Performance sportive (placement / résultats)

**Motivation** : standard universel (place, % top X, taux de sans-faute). EquiRatings "Clear Rounds", Hippomundo Rating.

**Risque principal** : `PLACE` n'est comparable qu'après normalisation par taille de peloton. La sémantique de "sans-faute" varie selon le barème de l'épreuve.

### Features

### Décision globale "sans-faute" (2026-05-03)

**La notion de "taux de sans-faute" est abandonnée pour cette famille** : malgré son statut de standard de la littérature équitation (EquiRatings "Clear Rounds", IFCE).

**Raisons** :
1. **Sémantique non-uniforme de POINTS** : un POINTS = 0 ne signifie pas la même chose en barème A, barème C, ou cycles SHF (Label, Formation, Cycle Libre). `POINTS != SO_POINTS_BAR + SO_TEMPS_BAR` dans 87% des cas.
2. **Couverture biaisée temporellement** de SO_POINTS_BAR : 1,4% à 4 ans → 11,4% à 5 ans → 39% à 6 ans → 84,3% à 7 ans. Le calcul "strict" sur 4-7 serait dominé par les performances tardives, biais structurel.
3. **Pas de doc FFE** disponible pour décoder précisément les barèmes par épreuve, et la FFE ne répondra probablement pas avant la fin du projet.
4. **Aucune fenêtre alternative** ne résout le problème : à 6 ans la couverture reste à 39% (61% NaN), insuffisant pour une feature stable.

**À mentionner dans le rapport final comme limite** : "Le taux de sans-faute, métrique standard de la littérature équitation, n'a pas pu être implémenté en raison de l'hétérogénéité sémantique de la variable POINTS et de l'absence de documentation FFE sur les barèmes."

### Features

#### `taux_sans_faute_strict_4_7`

- **Définition** : proportion de participations 4-7 ans où le cheval n'a pris aucune pénalité barre ni temps
- **Formule** : `((SO_POINTS_BAR == 0) & (SO_TEMPS_BAR == 0)).sum() / nb_lignes_avec_les_2_cols_remplies`
- **Source colonnes** : `SO_POINTS_BAR`, `SO_TEMPS_BAR`
- **Couverture** : 39% des participations (essentiellement Pro/Amateur barème A) -- mais biais temporel fort (1,4% à 4 ans → 84% à 7 ans). 10% des chevaux de la cohorte sans aucune ligne classique → feature NaN.
- **Motivation** : équivalent direct du "Clear Rounds" EquiRatings, sémantique propre et reconnue
- **Risques** : NaN élevé au niveau cheval, biais temporel (calcul dominé par les perfs 6-7 ans), pas de fenêtre alternative qui règle le problème
- **Statut** : 🔴 Rejetée
- **Décision** : abandonnée. Voir décision globale sans-faute ci-dessus.
- **Date décision** : 2026-05-03

#### `taux_zero_points_4_7`

- **Définition** : proportion de participations 4-7 ans avec `POINTS == 0`
- **Formule** : `(POINTS == 0).sum() / nb_participations_4_7`
- **Source colonnes** : `POINTS`
- **Couverture** : 99% des participations (POINTS quasi-toujours rempli)
- **Motivation** : approximation pragmatique de "pas de pénalité" couvrant tous les barèmes
- **Risques** : ⚠ sémantique mixte. POINTS == 0 ne signifie pas la même chose en barème A, barème C, ou cycles SHF Label/Formation
- **Statut** : 🔴 Rejetée
- **Décision** : abandonnée. Sa raison d'être était d'élargir la couverture là où le strict était limité ; sans le strict retenu, garder un proxy de sémantique mixte juste pour avoir "quelque chose" est artificiel. Voir décision globale ci-dessus.
- **Date décision** : 2026-05-03

#### `nb_participations_avec_so_pointsbar_4_7`

- **Définition** : nombre de participations 4-7 ans dans des épreuves "classiques" (SO_POINTS_BAR rempli)
- **Formule** : `df[in_fenetre_4_7 & SO_POINTS_BAR.notna()].groupby('IDCHEVAL').size()`
- **Source colonnes** : `SO_POINTS_BAR`
- **Couverture** : 100% (zéro autorisé)
- **Motivation** : méta-feature qui devait aider à pondérer la confiance dans `taux_sans_faute_strict_4_7`
- **Risques** : aucun mais perd son utilité sans une feature de sans-faute à pondérer
- **Statut** : 🔴 Rejetée
- **Décision** : abandonnée par cohérence avec le rejet des deux autres features de sans-faute. Sans cible à pondérer, n'a plus de sens.
- **Date décision** : 2026-05-03

### Décision globale codes PLACE (2026-05-03)

**Audit empirique** : 5 codes administratifs identifiés dans PLACE (899, 900, 902, 992, 993), hors continuité des vraies places (1-351). Recherche documentaire infructueuse pour leur sémantique exacte (codes internes FFE Compet, non documentés publiquement).

**Décision** : traitement "en bloc" sans tenter de mapping spéculatif. Conséquences pour la famille placement :
- Les features de **performance** (médiane, percentile, top X) sont calculées **uniquement sur les vraies places (1-351)**
- Les codes (≥800) sont comptés à part dans des features dédiées de **non-classement**
- Pour le calcul du **percentile dans le peloton** : la taille du peloton inclut tous les participants (vrais classés + non-classés), pour ne pas biaiser

### Features de non-classement

#### `taux_non_classement_4_7`

- **Définition** : proportion de participations 4-7 ans avec PLACE >= 800 (= code de non-classement)
- **Formule** : `(PLACE >= 800).sum() / nb_participations_4_7`
- **Source colonnes** : `PLACE`
- **Couverture** : 100%
- **Motivation** : capture la propension du cheval aux non-classements (élimination, abandon, hors concours, non-classé administratif des cycles SHF). Signal mixte assumé (sémantique ambiguë des codes).
- **Risques** : signal mixte (ne distingue pas les types de non-classement), partiellement corrélé à la transition cycles SHF → barème classique au fil de l'âge
- **Statut** : 🟡 À tester
- **Décision** : retenue. Base de la sous-famille non-classement.
- **Date décision** : 2026-05-03

#### `taux_nc_4ans` / `taux_nc_5ans` / `taux_nc_6ans` / `taux_nc_7ans`

- **Définition** : taux de non-classement pour chaque âge précis (4 features distinctes)
- **Formule** : `(PLACE >= 800 & AGE == X).sum() / (AGE == X).sum()` pour X = 4,5,6,7
- **Source colonnes** : `PLACE`, `AGE`
- **Couverture** : 100% si le cheval a participé à cet âge, NaN sinon (à gérer par imputation 0 ou flag)
- **Motivation** : encode la temporalité fine du non-classement, par cohérence avec l'approche temporelle des autres familles. Permet de capturer une dynamique (un cheval qui plafonne dans le non-classement vs un cheval qui en sort progressivement).
- **Risques** : ambiguïté redoublée (mixité des codes + transition cycles SHF / Amateur Pro entre 4 et 7 ans -- 1,4% participations en barème classique à 4 ans vs 84,3% à 7 ans)
- **Statut** : 🟡 À tester
- **Décision** : retenues par cohérence avec l'approche temporelle des autres familles. Ambiguïté assumée.
- **Date décision** : 2026-05-03

#### `evolution_nc_5_4` / `evolution_nc_6_5` / `evolution_nc_7_6`

- **Définition** : variations inter-annuelles du taux de non-classement (3 features)
- **Formule** :
  ```python
  evolution_nc_5_4 = taux_nc_5ans - taux_nc_4ans
  evolution_nc_6_5 = taux_nc_6ans - taux_nc_5ans
  evolution_nc_7_6 = taux_nc_7ans - taux_nc_6ans
  ```
- **Source colonnes** : dérivées
- **Couverture** : 100% (sauf si manque de participations à un âge)
- **Motivation** : encode explicitement les transitions année par année du non-classement. Cohérent avec les deltas inter-annuels des autres familles.
- **Risques** : redondance pour modèles d'arbre, mais utile pour interprétabilité
- **Statut** : 🟡 À tester
- **Décision** : retenues par cohérence avec les deltas inter-annuels des autres familles.
- **Date décision** : 2026-05-03

### Sous-famille : victoires (PLACE == 1)

Sémantique : "être 1er" est un signal qualitatif unique (un cheval ne peut être qu'à une seule 1re place dans une épreuve). Distinct du percentile qui dilue ce signal binaire. Pas de feature `nb_podiums` car le top 3 est arbitraire et redondant avec le percentile.

#### `nb_victoires_4_7`

- **Définition** : nombre de fois où le cheval a été 1er sur la fenêtre 4-7 ans
- **Formule** : `(PLACE == 1).sum()` sur les participations 4-7 ans
- **Source colonnes** : `PLACE`
- **Couverture** : 100% (compte ≥ 0)
- **Motivation** : signal absolu de performance au sommet. Distinct du percentile.
- **Risques** : biaisé par le volume (un cheval qui court beaucoup a plus de chances de gagner au moins une fois)
- **Statut** : 🟡 À tester
- **Décision** : retenue. Base brute du signal "victoire".
- **Date décision** : 2026-05-03

#### `taux_victoires_4_7`

- **Définition** : ratio nombre de victoires / nombre de participations sur la fenêtre 4-7 ans
- **Formule** : `nb_victoires_4_7 / nb_participations_4_7`
- **Source colonnes** : dérivée
- **Couverture** : 100%
- **Motivation** : élimine le biais volume (cheval qui gagne 3 fois sur 5 plus fort que celui qui gagne 3 fois sur 50)
- **Risques** : ne tient pas compte de la taille des pelotons
- **Statut** : 🟡 À tester
- **Décision** : retenue. Version normalisée du compte brut.
- **Date décision** : 2026-05-03

#### `a_au_moins_une_victoire_4_7`

- **Définition** : booléen indiquant si le cheval a gagné au moins une fois dans la fenêtre 4-7 ans
- **Formule** : `nb_victoires_4_7 >= 1`
- **Source colonnes** : dérivée
- **Couverture** : 100%
- **Motivation** : capture l'effet seuil "n'a jamais gagné" vs "a déjà goûté à la victoire". Signal binaire qualitatif.
- **Risques** : binaire, perd les nuances entre 1 et N victoires
- **Statut** : 🟡 À tester
- **Décision** : retenue. Capture une dimension qualitative non-redondante avec compte/taux.
- **Date décision** : 2026-05-03

#### `nb_victoires_4ans` / `nb_victoires_5ans` / `nb_victoires_6ans` / `nb_victoires_7ans`

- **Définition** : nombre de victoires à chaque âge précis (4 features)
- **Formule** : `(PLACE == 1 & AGE == X).sum()` pour X = 4,5,6,7
- **Source colonnes** : `PLACE`, `AGE`
- **Couverture** : 100% (0 si pas de victoire à cet âge)
- **Motivation** : encode la temporalité fine. Une victoire à 4-5 ans est un signal très fort (rare et précoce).
- **Risques** : sparsité élevée à 4-5 ans (peu de victoires à ces âges)
- **Statut** : 🟡 À tester
- **Décision** : retenues par cohérence avec l'approche temporelle des autres familles.
- **Date décision** : 2026-05-03

#### `evolution_victoires_5_4` / `evolution_victoires_6_5` / `evolution_victoires_7_6`

- **Définition** : variations inter-annuelles du nombre de victoires (3 features)
- **Formule** :
  ```python
  evolution_victoires_5_4 = nb_victoires_5ans - nb_victoires_4ans
  evolution_victoires_6_5 = nb_victoires_6ans - nb_victoires_5ans
  evolution_victoires_7_6 = nb_victoires_7ans - nb_victoires_6ans
  ```
- **Source colonnes** : dérivées
- **Couverture** : 100%
- **Motivation** : encode explicitement les transitions année par année. Cohérent avec les deltas inter-annuels des autres familles.
- **Risques** : redondance pour modèles d'arbre, mais utile pour interprétabilité
- **Statut** : 🟡 À tester
- **Décision** : retenues par cohérence.
- **Date décision** : 2026-05-03

#### `taux_victoires_4ans` / `taux_victoires_5ans` / `taux_victoires_6ans` / `taux_victoires_7ans`

- **Définition** : ratio victoires/participations à chaque âge précis (4 features)
- **Formule** : `nb_victoires_Xans / nb_participations_Xans` (NaN si pas de participation à cet âge)
- **Source colonnes** : dérivées
- **Couverture** : 100% si participation à cet âge, NaN sinon
- **Motivation** : version normalisée par âge. Capture la qualité (et pas seulement le volume) à chaque tranche d'âge.
- **Risques** : instable quand peu de participations (ex : 1 victoire / 1 participation = 100% mais peu significatif)
- **Statut** : 🟡 À tester
- **Décision** : retenues. Version normalisée par âge en complément du compte brut.
- **Date décision** : 2026-05-03

#### `evolution_taux_victoires_5_4` / `evolution_taux_victoires_6_5` / `evolution_taux_victoires_7_6`

- **Définition** : variations inter-annuelles du taux de victoires (3 features)
- **Formule** : différence du taux entre années consécutives
- **Source colonnes** : dérivées
- **Couverture** : ~100% (NaN si une des deux années sans participation)
- **Motivation** : encode la trajectoire de la qualité, pas seulement du compte
- **Risques** : redondance, sensibilité aux NaN
- **Statut** : 🟡 À tester
- **Décision** : retenues par cohérence.
- **Date décision** : 2026-05-03

### Sous-famille : performance relative (percentile dans peloton)

**Convention de calcul du percentile** (décision 2026-05-03) :
- Pour chaque ligne avec PLACE valide (1-351) : `percentile = PLACE / dénominateur`
- Pour les lignes avec PLACE >= 800 (codes) : percentile = NaN, exclu du calcul d'agrégation
- Au niveau cheval, agrégation (médiane, min, max, etc.) sur les valeurs non-NaN

**Choix méthodologique sur le dénominateur** : on garde **2 versions en parallèle** car la sémantique des codes est ambiguë :
- **Version "partants"** : dénominateur = nb total de lignes de l'épreuve (incluant codes/non-classés). Lecture : "devant X% des chevaux engagés". Biais : peut surestimer la performance relative si beaucoup de non-classés.
- **Version "finishers"** : dénominateur = nb de lignes avec PLACE valide (1-351). Lecture : "devant X% des chevaux qui ont fini classés". Biais : ne pénalise pas la qualité du sous-ensemble des classés.

Coût marginal de garder les 2 (+1 feature par métrique), bénéfice : laisser le modèle combiner les deux signaux.

**Convention de lecture** : plus le percentile est bas, meilleur est le cheval (0 = 1er, 1 = dernier). Inverse de l'intuition naturelle, à expliciter dans la doc/code.

#### `percentile_partants_median_4_7`

- **Définition** : médiane du percentile (PLACE / nb_partants_total) sur la fenêtre 4-7 ans
- **Formule** :
  ```python
  # Pour chaque ligne avec PLACE valide :
  nb_partants_total = nb total de lignes de l'épreuve (vrais classés + codes)
  percentile = PLACE / nb_partants_total
  # Au niveau cheval :
  percentile_partants_median_4_7 = median(percentile)
  ```
- **Source colonnes** : `PLACE`, `NUMERO_EVENEMENT2`, `NUMEROSEQUENCE`
- **Couverture** : ~100% (sauf chevaux avec uniquement des codes -- rare)
- **Motivation** : performance relative dans le peloton initial. Robuste aux outliers (médiane > moyenne).
- **Risques** : peut surestimer la performance si beaucoup de non-classés dans l'épreuve
- **Statut** : 🟡 À tester
- **Décision** : retenue. Version "inclusive" du percentile pour capturer le rang dans le peloton initial.
- **Date décision** : 2026-05-03

#### `percentile_finishers_median_4_7`

- **Définition** : médiane du percentile (PLACE / nb_finishers) sur la fenêtre 4-7 ans
- **Formule** :
  ```python
  # Pour chaque ligne avec PLACE valide :
  nb_finishers = nb de lignes avec PLACE valide (1-351) dans l'épreuve
  percentile = PLACE / nb_finishers
  # Au niveau cheval :
  percentile_finishers_median_4_7 = median(percentile)
  ```
- **Source colonnes** : `PLACE`, `NUMERO_EVENEMENT2`, `NUMEROSEQUENCE`
- **Couverture** : ~100%
- **Motivation** : performance relative parmi ceux qui ont fini classés. Évite le biais de l'option "partants" lié aux non-classés.
- **Risques** : ne pénalise pas la qualité absolue des "vrais classés"
- **Statut** : 🟡 À tester
- **Décision** : retenue. Version "stricte" du percentile.
- **Date décision** : 2026-05-03

#### Déclinaison complète des features percentile (format compact)

**Décision 2026-05-03** : par cohérence avec l'approche "inclusif maintenant, épurer plus tard", on décline pour les 4 statistiques (médiane, min, max, std) × 2 versions (partants, finishers) × 8 granularités temporelles (global + 4 valeurs annuelles + 3 deltas).

**Total : 64 features percentile** (dont 2 médianes globales déjà détaillées ci-dessus).

| Statistique | Version | Global | Par âge (4 features) | Deltas (3 features) | Sous-total |
|---|---|---|---|---|---|
| **Médiane** | partants | `percentile_partants_median_4_7` | `percentile_partants_median_4ans/5ans/6ans/7ans` | `evolution_percentile_partants_median_5_4/6_5/7_6` | 8 |
| **Médiane** | finishers | `percentile_finishers_median_4_7` | `percentile_finishers_median_4ans/5ans/6ans/7ans` | `evolution_percentile_finishers_median_5_4/6_5/7_6` | 8 |
| **Min** | partants | `percentile_partants_min_4_7` | `percentile_partants_min_4ans/5ans/6ans/7ans` | `evolution_percentile_partants_min_5_4/6_5/7_6` | 8 |
| **Min** | finishers | `percentile_finishers_min_4_7` | `percentile_finishers_min_4ans/5ans/6ans/7ans` | `evolution_percentile_finishers_min_5_4/6_5/7_6` | 8 |
| **Max** | partants | `percentile_partants_max_4_7` | `percentile_partants_max_4ans/5ans/6ans/7ans` | `evolution_percentile_partants_max_5_4/6_5/7_6` | 8 |
| **Max** | finishers | `percentile_finishers_max_4_7` | `percentile_finishers_max_4ans/5ans/6ans/7ans` | `evolution_percentile_finishers_max_5_4/6_5/7_6` | 8 |
| **Std** | partants | `percentile_partants_std_4_7` | `percentile_partants_std_4ans/5ans/6ans/7ans` | `evolution_percentile_partants_std_5_4/6_5/7_6` | 8 |
| **Std** | finishers | `percentile_finishers_std_4_7` | `percentile_finishers_std_4ans/5ans/6ans/7ans` | `evolution_percentile_finishers_std_5_4/6_5/7_6` | 8 |
| | | | | **TOTAL** | **64** |

**Sémantique des statistiques** :
- **Médiane** : performance typique du cheval
- **Min** : meilleure performance atteinte (= valeur la plus basse car convention "bas = bon")
- **Max** : pire performance atteinte parmi les classés
- **Std** : variabilité / régularité de la performance

**Couverture** :
- Globales : ~100%
- Par âge : 100% si participation à cet âge avec au moins une vraie place, NaN sinon
- Deltas : NaN si une des deux années est sans participation valide

**Statut commun** : 🟡 À tester
**Décision commune** : retenues par cohérence (déclinaison exhaustive). Épuration à faire en phase modélisation (importance features, redondance).

### Sous-famille : taux dans les top X% du peloton

**Décision 2026-05-03** : 4 seuils standards retenus (5%, 10%, 25%, 50%). Couvre l'élite, le top, la limite "prize money" (cf. validation empirique sur Amateur/Pro), et la moyenne supérieure. Plus de seuils créerait de la redondance.

**Convention** : un cheval est "dans le top X%" pour une participation si `percentile <= X/100`. Comme pour les percentiles, on garde les **2 versions partants/finishers** car la sémantique des codes est ambiguë.

**Calcul** :
```python
# Pour chaque ligne avec PLACE valide :
percentile_partants = PLACE / nb_partants_total
percentile_finishers = PLACE / nb_finishers
top_X_partants = (percentile_partants <= X/100)  # booléen
top_X_finishers = (percentile_finishers <= X/100)
# Au niveau cheval :
taux_top_X = mean(top_X) sur les vraies places de la fenêtre
```

**Déclinaison complète : 4 seuils × 2 versions × 8 granularités temporelles = 64 features**.

| Seuil | Version | Global | Par âge (4 features) | Deltas (3 features) | Sous-total |
|---|---|---|---|---|---|
| **Top 5%** | partants | `taux_top_5pct_partants_4_7` | `taux_top_5pct_partants_4ans/5ans/6ans/7ans` | `evolution_top_5pct_partants_5_4/6_5/7_6` | 8 |
| **Top 5%** | finishers | `taux_top_5pct_finishers_4_7` | `taux_top_5pct_finishers_4ans/5ans/6ans/7ans` | `evolution_top_5pct_finishers_5_4/6_5/7_6` | 8 |
| **Top 10%** | partants | `taux_top_10pct_partants_4_7` | `taux_top_10pct_partants_4ans/5ans/6ans/7ans` | `evolution_top_10pct_partants_5_4/6_5/7_6` | 8 |
| **Top 10%** | finishers | `taux_top_10pct_finishers_4_7` | `taux_top_10pct_finishers_4ans/5ans/6ans/7ans` | `evolution_top_10pct_finishers_5_4/6_5/7_6` | 8 |
| **Top 25%** | partants | `taux_top_25pct_partants_4_7` | `taux_top_25pct_partants_4ans/5ans/6ans/7ans` | `evolution_top_25pct_partants_5_4/6_5/7_6` | 8 |
| **Top 25%** | finishers | `taux_top_25pct_finishers_4_7` | `taux_top_25pct_finishers_4ans/5ans/6ans/7ans` | `evolution_top_25pct_finishers_5_4/6_5/7_6` | 8 |
| **Top 50%** | partants | `taux_top_50pct_partants_4_7` | `taux_top_50pct_partants_4ans/5ans/6ans/7ans` | `evolution_top_50pct_partants_5_4/6_5/7_6` | 8 |
| **Top 50%** | finishers | `taux_top_50pct_finishers_4_7` | `taux_top_50pct_finishers_4ans/5ans/6ans/7ans` | `evolution_top_50pct_finishers_5_4/6_5/7_6` | 8 |
| | | | | **TOTAL** | **64** |

**Sémantique** :
- **Top 5%** : élite, signal "futur haut niveau" potentiel
- **Top 10%** : top du peloton, standard EquiRatings/Hippomundo
- **Top 25%** : limite empirique du prize money en SO Amateur/Pro
- **Top 50%** : moitié supérieure, indicateur "performance > médiocre"

**Couverture** :
- Globales : ~100%
- Par âge : 100% si participation à cet âge avec au moins une vraie place, NaN sinon
- Deltas : NaN si une des deux années sans participation valide

**Statut commun** : 🟡 À tester
**Décision commune** : retenues par cohérence (déclinaison exhaustive). Forte corrélation attendue entre seuils consécutifs ; épuration empirique à prévoir en phase modélisation.

---

## Famille 4 -- Hauteurs explorées

**Motivation initiale** : Chapard 2023 ("hauteur d'obstacle ajustée"), dimension naturelle de la performance en SO.

**Statut famille (décision 2026-05-03)** : 🔴 **FAMILLE REJETÉE EN BLOC**.

### Justification du rejet

Quatre raisons cumulatives :

**1. Cohérence avec le sens prédictif du sujet**

La cible est `hauteur_max_validée`. Utiliser des features dérivées de la hauteur (médiane, min, distribution, etc.) revient à prédire la cible avec des dérivés de la cible elle-même. Le modèle risque d'être une "boîte noire qui reproduit la cible" plutôt qu'un vrai prédicteur. Renoncer à la hauteur force le modèle à trouver d'autres signaux prédictifs (gains, places, divisions, race, pedigree, cavalier).

**2. Cohérence avec le rejet déjà fait de `hauteur_max_4_7`**

Cette feature avait été identifiée dès le début comme tautologique (60,9% des chevaux atteignent leur cible avant 8 ans, jusqu'à 85% pour les niveaux modestes). Étendre ce raisonnement aux autres statistiques de hauteur est cohérent -- toutes souffrent à des degrés divers du même problème de proximité avec la cible.

**3. Biais de couverture structurel**

47% des participations de la cohorte 4-7 ont HAUTEUR=NaN (concentré sur les cycles SHF). Au niveau cheval :
- 10,5% n'ont aucune hauteur connue (chevaux uniquement SHF) → toutes les features hauteur seraient NaN
- 89,5% ont au moins une hauteur, mais avec une couverture très variable (de 1 à 189 participations exploitables)
- Les statistiques calculées (médiane, min, etc.) ne reflètent que la **partie visible** (= Amateur/Pro), pas l'activité totale

**4. Biais de trajectoire**

Pour les chevaux mixtes (70% de la cohorte), la hauteur n'est observée que sur la partie non-SHF. Exemple : un cheval qui fait Cycle SHF à 4-5 ans (HAUTEUR NaN, hauteurs réelles ~0,95-1,00m) puis Amateur à 6-7 ans à 1,20-1,30m apparaîtrait comme "commençant à 1,20m" pour le modèle, alors qu'il a en réalité progressé depuis 0,95m. Sa transition cycles SHF → Amateur est invisible. Toute feature de progression hauteur serait biaisée pour la majorité des chevaux.

### Conséquences pratiques

- Aucune feature de la famille 4 n'est construite
- Le signal "hauteur" sera partiellement capté par d'autres familles : gains (Famille 2), placement (Famille 3), niveau/type d'épreuve (Famille 5), cavalier (Famille 7), pedigree (Famille 8)
- À mentionner dans le rapport final comme **choix méthodologique fort assumé** : "Les features dérivées de la hauteur d'obstacle ont été délibérément écartées pour éviter une auto-corrélation avec la cible et pour des raisons de qualité de données (47% de NaN structurels concentrés sur les cycles SHF, biais de trajectoire pour les parcours mixtes)"
- Limitation à mentionner : impossibilité de comparer directement avec les approches utilisant la "hauteur d'obstacle ajustée" (Chapard 2023) -- mais ces approches étaient sur cohortes plus restreintes et avec accès à la grille hauteur officielle

### Features rejetées (toutes en bloc)

Liste indicative des features qui auraient pu être construites :
- Statistiques tendance centrale : `hauteur_mediane_4_7`, `hauteur_min_4_7`, `hauteur_q25_4_7`, `hauteur_q75_4_7`, `hauteur_moyenne_4_7`
- Diversité : `nb_hauteurs_distinctes_4_7`, `range_hauteur_4_7`, `hauteur_std_4_7`
- Couverture : `nb_participations_avec_hauteur_4_7`, `taux_participations_avec_hauteur_4_7`
- Temporalité : versions par âge + deltas
- Distribution : `taux_participations_haute_hauteur_4_7`, `taux_participations_basse_hauteur_4_7`

→ Toutes 🔴 Rejetées par décision globale de famille.

---

## Famille 5 -- Niveau / type d'épreuves

**Motivation** : capter l'environnement compétitif du cheval (où il a couru, dans quelle division). Signal indirect du niveau atteint, complémentaire des features de performance pure.

**Risque principal** : mixité fréquente des divisions au sein d'une même année (73% des chevaux explorent au moins 2 divisions sur 4-7) -- exclut les catégorisations binaires "le cheval est en X à l'âge Y", impose les ratios.

### Décision globale (2026-05-03)

**Approche minimaliste retenue** : seuls les **ratios par division** (3 divisions disponibles : Amateur, Pro, Élevage). Pas de features extraites du libellé d'épreuve (Grand Prix, championnat, finale...) car :
- "Grand Prix" couvre une plage trop large de niveaux (1m → 1,40m+) -- compte sans distinction = signal mixte
- Si on segmentait par niveau, on revient à utiliser la hauteur (rejetée en Famille 4)
- Biais d'opportunité : nb_grands_prix corrélé avec nb_participations (déjà capturé)
- Fragilité d'extraction texte sur libellés variants

→ **3 ratios × 8 granularités temporelles = 24 features**.

### Features

| Division | Global | Par âge (4 features) | Deltas (3 features) | Sous-total |
|---|---|---|---|---|
| **Amateur** | `taux_part_amateur_4_7` | `taux_part_amateur_4ans/5ans/6ans/7ans` | `evolution_part_amateur_5_4/6_5/7_6` | 8 |
| **Pro** | `taux_part_pro_4_7` | `taux_part_pro_4ans/5ans/6ans/7ans` | `evolution_part_pro_5_4/6_5/7_6` | 8 |
| **Élevage** | `taux_part_elevage_4_7` | `taux_part_elevage_4ans/5ans/6ans/7ans` | `evolution_part_elevage_5_4/6_5/7_6` | 8 |
| | | | **TOTAL** | **24** |

**Formule générale** :
```python
# Global
taux_part_DIV_4_7 = (DIVISION_LIB == DIV).sum() / nb_participations_4_7

# Par âge
taux_part_DIV_Xans = (DIVISION_LIB == DIV & AGE == X).sum() / (AGE == X).sum()

# Deltas
evolution_part_DIV_X_Y = taux_part_DIV_Xans - taux_part_DIV_Yans
```

**Source colonnes** : `DIVISION_LIB`, `AGE`

**Couverture** :
- Globales : 100% (DIVISION_LIB rempli à 100%)
- Par âge : 100% si participation à cet âge, NaN sinon
- Deltas : NaN si une des deux années sans participation

**Risques** :
- Redondance linéaire structurelle : les 3 ratios somment à 1 par construction (par cheval × granularité), donc le 3e est dérivable des 2 premiers. Acceptable pour modèles d'arbre, à surveiller pour modèles linéaires (à éventuellement réduire à 2 ratios + 1 référence)
- Pas de leakage avec la cible
- Capture l'environnement compétitif, pas la performance directe

**Statut commun** : 🟡 À tester
**Décision commune** : retenues. Approche minimaliste mais cohérente avec la déclinaison des autres familles. Épuration possible en phase modélisation (notamment résolution de la dépendance linéaire des 3 ratios).

---

## Famille 6 -- Progression temporelle

**Statut famille (décision 2026-05-10)** : 🔴 **FAMILLE REJETÉE EN BLOC**.

### Justification du rejet

La progression temporelle est **déjà massivement capturée** par les deltas inter-annuels disséminés dans toutes les autres familles :

- **Famille 1** : `evolution_5_4`, `_6_5`, `_7_6` sur `nb_participations`
- **Famille 2** : 3 deltas sur `gains`
- **Famille 3** : 3 deltas sur place, percentile, top X% (≈ 60 deltas)
- **Famille 5** : 3 deltas sur chaque ratio division
- **Famille 7** : 3 deltas sur chaque métrique cavalier

→ ~70-80 features de progression déjà présentes dans le master dataset.

### Ce que Famille 6 aurait pu ajouter

- Indicateurs composites (synthèse multi-dimensionnelle) -- intéressant mais ad hoc
- Accélérations / second-order deltas -- signal réel mais marginal pour un projet M1
- Forme de trajectoire -- complexe à encoder, peu robuste

→ L'apport unique de cette famille au-delà des deltas déjà présents serait **marginal**. Cohérent avec la rigueur méthodologique appliquée ailleurs (rejet Famille 4 Hauteurs, Famille 11 Géographique).

### Conséquence

Aucune feature de la famille 6 n'est construite. La progression temporelle reste pleinement capturée par les deltas inter-annuels des autres familles. À mentionner dans le rapport : "La dimension progression temporelle n'a pas fait l'objet d'une famille de features dédiée, étant déjà nativement capturée par les variations inter-annuelles dans toutes les autres familles."

---

## Famille 7 -- Cavalier

**Motivation** : standard académique (Chapard, Sanchez-Guerrero, Ricard). Le niveau du cavalier est un facteur majeur de la performance d'un cheval.

**Risque principal** : ⚠ leakage si mal construit (cf. erreur PERE_ELITE de la binôme).

### Décisions de cadrage (2026-05-07)

| Décision | Choix |
|---|---|
| **Anti-leakage** | Solution A : leave-one-out (calcul du niveau cavalier sur ses autres chevaux, exclut le cheval analysé) |
| **Fenêtre temporelle** | Fenêtre **passée 3 ans glissante** (n-2, n-1, n) -- évite leakage temporel "futur du cavalier" |
| **Cas LOO impossibles** (~8% des participations) | NaN |
| **Métriques principales** | Place médiane, percentile médian (pas de division -- ambigu) |
| **Granularité temporelle** | Global 4-7 + par âge + 3 deltas inter-annuels (cohérent avec les autres familles) |
| **Agrégation multi-cavaliers** | Moyenne pondérée des scores cavalier sur les participations du cheval, + feature diversité (`nb_cavaliers_distincts_4_7`) |
| **Filtre minimum nb chevaux dans LOO** | À fixer empiriquement après évaluation (NaN si insuffisant) |
| **Effet transmission** (formateur vs preneur) | Mélange accepté -- limite à documenter dans le rapport |
| **Évolution cavalier dans le temps** | Naturellement capturé par la fenêtre passée 3 ans glissante (pas besoin d'ajustement supplémentaire) |

### Validation empirique de la fenêtre temporelle

| Stratégie | LOO impossible | < 5 chevaux | ≥ 5 fiable |
|---|---|---|---|
| Fenêtre 1 an | 16,7% | 27,7% | 55,7% |
| Centrée 3 ans (n-1, n, n+1) | 8,9% | 23,3% | 67,8% |
| **Passée 3 ans (n-2, n-1, n)** retenue | **7,9%** | 24,4% | **67,7%** |
| Passée 5 ans | 5,9% | 22,2% | 71,9% |
| Passée totale | 5,4% | 20,5% | 74,1% |

→ La fenêtre passée 3 ans donne une stabilité comparable à la centrée tout en évitant le leakage temporel.

### Features

### Sous-famille : niveau cavalier basé sur le percentile (place normalisée)

**Décision 2026-05-08** : la PLACE brute n'est pas comparable entre épreuves de tailles différentes (5e sur 80 ≠ 5e sur 8). On utilise donc le **percentile** (PLACE / nb_partants), comme pour Famille 3. Et par cohérence avec Famille 3, on garde **2 versions** (partants et finishers) car la sémantique des codes est ambiguë.

**Calcul commun** :
```python
# Pour chaque participation (cheval analysé, cavalier, année X) :
autres_participations = participations du cavalier sur [X-2, X-1, X],
                         excluant le cheval analysé
# Pour chaque autre participation valide (PLACE 1-351) :
percentile_partants = PLACE / nb_partants_total_epreuve
percentile_finishers = PLACE / nb_finishers_epreuve
# Score du cavalier pour cette année X :
cavalier_score = median(percentile sur autres_participations)
# Au niveau cheval :
cavalier_*_4_7 = mean(cavalier_score sur participations 4-7 du cheval)
```

**Source colonnes** : `PLACE`, `LICENCE`, `ANNEE`, `IDCHEVAL`, `NUMERO_EVENEMENT2`, `NUMEROSEQUENCE`

**Couverture** : ~92% (NaN pour les ~8% de cas LOO impossibles, à gérer en NaN)

**Motivation** : niveau de performance moyen du cavalier dans son contexte récent (3 dernières années), sans contamination par le cheval analysé, normalisé par taille de peloton. Plus la valeur est basse, mieux le cavalier place ses chevaux. Standard académique (Chapard, Sanchez-Guerrero, Ricard).

**Risques** :
- Pas de leakage avec la cible (LOO + fenêtre passée + percentile)
- Effet de sélection : un bon cavalier choisit/reçoit de bons chevaux -- biais résiduel inhérent
- Multi-cavaliers : agrégation par moyenne pondérée sur les participations du cheval

#### Déclinaison complète : 16 features

| Version | Global | Par âge (4 features) | Deltas (3 features) | Sous-total |
|---|---|---|---|---|
| **partants** | `cavalier_percentile_partants_median_passe3_4_7` | `cavalier_percentile_partants_median_passe3_4ans/5ans/6ans/7ans` | `evolution_cavalier_percentile_partants_median_5_4/6_5/7_6` | 8 |
| **finishers** | `cavalier_percentile_finishers_median_passe3_4_7` | `cavalier_percentile_finishers_median_passe3_4ans/5ans/6ans/7ans` | `evolution_cavalier_percentile_finishers_median_5_4/6_5/7_6` | 8 |
| | | | **TOTAL** | **16** |

**Statut commun** : 🟡 À tester
**Décision commune** : retenues. Première sous-famille de Famille 7 -- niveau cavalier mesuré par la performance relative de ses chevaux (hors cheval analysé) sur fenêtre passée 3 ans.
**Date décision** : 2026-05-08

### Sous-famille : gains du cavalier (approche Hurdle / Two-Part Model)

**Décision 2026-05-11** : approche **Hurdle** retenue après consultation de 3 avis externes. La question initiale était "log avant ou après agrégation ?" -- aucune des deux options n'était satisfaisante :
- **Log avant agrégation** : standard Ricard mais piège Zero-Inflation (69% de gains à 0€ → `log(0+1)=0` aplatit le signal)
- **Brut puis log à la modélisation** : viole l'inégalité de Jensen (la moyenne brute est dominée par les outliers ; Cavalier régulier 50€×101 ≡ Cavalier chanceux 5050€×1 + 0€×100 → mêmes moyennes brutes alors que profils opposés)

**La solution Hurdle** sépare le problème en deux variables distinctes :

#### Concept

- **Feature A, Fréquence** : taux de participations où le cavalier a gagné (GAINS > 0)
  → Évite le Zero-Inflation (variable de fréquence pure)
- **Feature B, Magnitude pure** : `mean(log(GAINS))` calculé **uniquement sur les participations avec GAINS > 0**
  → Évite l'aplatissement par les zéros (on filtre AVANT)
  → Garde l'effet log avant agrégation (gère la skewness, Jensen)

Le modèle combine ces 2 dimensions :
- Modèle linéaire : 2 coefficients distincts et interprétables ("fréquence de victoire" vs "niveau économique quand victoire")
- Modèle d'arbre (XGBoost) : splits croisés possibles (ex : `taux_gain > 0.3 AND magnitude_log > 5.0`)

#### Référence académique

L'approche **Hurdle / Two-Part Model** :
- **Cragg (1971)** : "Some Statistical Models for Limited Dependent Variables with Application to the Demand for Durable Goods" -- papier fondateur en économétrie
- **Mullahy (1998)** : application massive en économie de la santé (frais médicaux : beaucoup de patients à 0€, certains avec factures astronomiques -- structure très similaire à nos gains équins)

#### Mécanique commune aux features Hurdle

- **Leave-one-out** : excluant le cheval analysé (cohérent avec le reste de Famille 7)
- **Fenêtre passée 3 ans glissante** : n-2, n-1, n
- **Smoothing Bayésien** : pour cavaliers à peu de chevaux
- **Cas LOO impossibles** : NaN

### Features Hurdle

#### `cavalier_taux_gains_positifs_passe3_4_7` (+ déclinaisons)

- **Définition** : proportion des participations du cavalier (sur les autres chevaux, fenêtre passée 3 ans) où GAINS > 0. Agrégée par moyenne sur les participations 4-7 du cheval analysé.
- **Formule** :
  ```python
  # Pour chaque participation (cheval, cavalier, année X) :
  participations = autres_chevaux du cavalier sur [X-2, X-1, X], excluant le cheval
  taux = sum(GAINS > 0) / nb_participations
  # Agrégation au niveau cheval
  ```
- **Source colonnes** : `GAINS`, `LICENCE`, `ANNEE`, `IDCHEVAL`
- **Couverture** : ~92% (NaN pour cas LOO impossibles)
- **Motivation** : capture la **fréquence** à laquelle le cavalier fait gagner ses chevaux. Variable de comptage pur, immune au Zero-Inflation.
- **Risques** : aucun majeur (variable bornée entre 0 et 1)
- **Statut** : 🟡 À tester
- **Décision** : retenue (Hurdle Feature A).
- **Date décision** : 2026-05-11

**Déclinaison complète** : 8 features (1 global + 4 par âge + 3 deltas)

#### `cavalier_mean_log_gains_pos_passe3_4_7` (+ déclinaisons)

- **Définition** : moyenne de `log(GAINS)` des participations du cavalier (sur les autres chevaux, fenêtre passée 3 ans) **où GAINS > 0 strictement**. Agrégée au niveau cheval.
- **Formule** :
  ```python
  # Pour chaque participation (cheval, cavalier, année X) :
  participations = autres_chevaux du cavalier sur [X-2, X-1, X], excluant le cheval
  participations_pos = filtre(GAINS > 0)
  magnitude_log = mean(log(GAINS) sur participations_pos)  # NaN si aucune
  # Agrégation au niveau cheval
  ```
- **Source colonnes** : `GAINS`, `LICENCE`, `ANNEE`, `IDCHEVAL`
- **Couverture** : ~85-90% (NaN si cavalier n'a aucune participation positive dans la fenêtre passée 3 ans, ou cas LOO impossibles)
- **Motivation** : capture le **niveau économique** auquel le cavalier classe ses chevaux quand il gagne. Variable de magnitude propre, sans pollution par les 69% de zéros.
- **Risques** : NaN si cavalier sans aucune victoire dans la fenêtre -- à gérer (smoothing ou imputation)
- **Statut** : 🟡 À tester
- **Décision** : retenue (Hurdle Feature B).
- **Date décision** : 2026-05-11

**Déclinaison complète** : 8 features (1 global + 4 par âge + 3 deltas)

### Récap sous-famille gains du cavalier

| Feature | Granularités | Sous-total |
|---|---|---|
| Taux gains positifs (Hurdle A) | global + 4 par âge + 3 deltas | 8 |
| Mean log gains positifs (Hurdle B) | idem | 8 |
| **TOTAL** | | **16 features** |

### Sous-famille : diversité et volume

**Décision 2026-05-10** : 3 métriques de diversité / volume, toutes déclinées en 8 granularités temporelles (global + 4 par âge + 3 deltas). **Total : 24 features**.

#### Métriques

| # | Feature racine | Sémantique | Fenêtre | LOO ? |
|---|---|---|---|---|
| 1 | `nb_chevaux_distincts_cavalier_passe3` | Expérience / diversité du cavalier | Passée 3 ans cavalier | Oui |
| 2 | `nb_cavaliers_distincts_du_cheval` | Stabilité du cheval (peu = stable, beaucoup = ballottement) | 4-7 ans cheval | Non (métrique du cheval) |
| 3 | `nb_participations_cavalier_passe3` | Volume d'activité du cavalier | Passée 3 ans cavalier | Oui |

**Calculs** :
```python
# Métrique 1 (diversité cavalier, agrégée niveau cheval)
nb_chevaux_distincts_cavalier_passe3 = mean sur participations 4-7 du cheval de :
    nb chevaux distincts montés par cavalier sur [année-2, année-1, année], exclu cheval analysé

# Métrique 2 (diversité cheval-cavaliers, directement niveau cheval)
nb_cavaliers_distincts_du_cheval_4_7 = nb LICENCE distinctes sur les participations 4-7 du cheval

# Métrique 3 (volume cavalier, agrégée niveau cheval)
nb_participations_cavalier_passe3 = mean sur participations 4-7 du cheval de :
    nb participations du cavalier sur [année-2, année-1, année], exclu participations sur cheval analysé
```

#### Déclinaison complète : 24 features

| Métrique | Global | Par âge (4 features) | Deltas (3 features) | Sous-total |
|---|---|---|---|---|
| **Diversité cavalier** | `nb_chevaux_distincts_cavalier_passe3_4_7` | `nb_chevaux_distincts_cavalier_passe3_4ans/5ans/6ans/7ans` | `evolution_nb_chevaux_cavalier_5_4/6_5/7_6` | 8 |
| **Diversité cheval-cavaliers** | `nb_cavaliers_distincts_du_cheval_4_7` | `nb_cavaliers_distincts_du_cheval_4ans/5ans/6ans/7ans` | `evolution_nb_cavaliers_du_cheval_5_4/6_5/7_6` | 8 |
| **Volume cavalier** | `nb_participations_cavalier_passe3_4_7` | `nb_participations_cavalier_passe3_4ans/5ans/6ans/7ans` | `evolution_nb_participations_cavalier_5_4/6_5/7_6` | 8 |
| | | | **TOTAL** | **24** |

**Source colonnes** : `LICENCE`, `ANNEE`, `IDCHEVAL`, `AGE`

**Couverture** : ~92% pour les métriques 1 et 3 (NaN pour cas LOO impossibles), ~100% pour la métrique 2

**Statut commun** : 🟡 À tester
**Décision commune** : retenues. Métriques de diversité et volume cavalier/cheval-cavaliers.
**Date décision** : 2026-05-10

---

## Famille 8 -- Pedigree / génétique

**Motivation** : standard BLUP universel en évaluation génétique équine (Ricard / IFCE, Henderson). Le pedigree capture l'effet "lignée" / héritabilité du potentiel sportif.

**Risque principal** : ⚠ leakage massif si mal construit (cf. erreur PERE_ELITE de la binôme : calculer le score du père en incluant le cheval analysé parmi ses descendants → tautologie directe).

### Données disponibles

| Variable | NaN | Sémantique (vérifiée empiriquement) |
|---|---|---|
| `NUMSIREPERE` | 0,11% | Père |
| `NUMSIREMERE` | 0,11% | Mère |
| `NUMSIREPEREMERE` | 1,17% | Grand-père maternel (vérification 100% match sur 31 627 chevaux où la mère est dans la base) |

Pas de pedigree profond. Pas de date de naissance des ancêtres.

### Décisions de cadrage (2026-05-11)

| Décision | Choix | Justification |
|---|---|---|
| **Méthode d'encodage** | Target encoding LOO + smoothing Bayésien | Cohérent avec Famille 10 (race) et standard BLUP/Ricard |
| **Ancêtres retenus** | Père + Grand-père maternel | Mère écartée : médiane 1-2 descendants par mère (biologique : 10-15 poulains max par jument vs 100+ par étalon). LOO instable. |
| **Périmètre du calcul** | Cohorte de modélisation T1+N1≥10 + cible calculable (Option A) | Évite biais de censure (Option B exclue) et era drift / fuite temporelle (Option C exclue). Couverture : 86,7% fiable pour père, 82,4% pour GP maternel. |
| **Smoothing Bayésien** | Idem Famille 10 | Gère les ancêtres à peu de descendants |
| **Fit train uniquement** | Idem Famille 10 | Évite leakage train→test |
| **Time series split** | À gérer en phase modélisation | Évite fuite temporelle (entraîner sur cohortes anciennes, tester sur récentes) |

### Avis externes (2026-05-11)

Deux questions méthodologiques soumises à une IA externe :
1. **Cohérence du target encoding LOO avec le rejet de Famille 4** → Position A validée (référence Micci-Barreca 2001, BLUP de Henderson)
2. **Choix du périmètre A/B/C** → Option A recommandée (refus catégorique de B = biais de censure ; refus de C = era drift + fuite temporelle ; A = rigueur méthodologique)

**Caveat** : les 2 avis externes peuvent être biaisés/complaisants. Leurs arguments techniques tiennent néanmoins (concepts vérifiables, références réelles).

### Vérification empirique de la couverture (par ancêtre)

| Ancêtre | LOO fiable (≥5 descendants) | 50+ descendants |
|---|---|---|
| **Père** | 86,7% (41 283 chevaux) | 25 785 chevaux (54%) |
| Mère | 2,7% | 0 |
| **Grand-père maternel** | 82,4% (39 235 chevaux) | 21 781 chevaux (46%) |

→ Pères et GP maternels = étalons avec beaucoup de descendants → métriques solides.

### Features

#### `pere_target_encoded_LOO`

- **Définition** : moyenne LOO de `hauteur_max_validée` des autres descendants du même père dans la cohorte T1+N1≥10, avec smoothing Bayésien
- **Formule** :
  ```python
  # Sur le train set uniquement :
  for chaque cheval C dans la cohorte :
      P = NUMSIREPERE(C)
      autres_descendants = chevaux avec NUMSIREPERE == P, exclu C
      n = len(autres_descendants)
      moyenne_pere_LOO = mean(cible des autres_descendants)
      λ = n / (n + k)  # k = smoothing factor (à calibrer, ex : k=30)
      pere_target_encoded_LOO[C] = λ × moyenne_pere_LOO + (1-λ) × moyenne_globale_train
  ```
- **Source colonnes** : `NUMSIREPERE`, cible
- **Couverture** : 86,7% fiable, 95% calculable (smoothing pour le reste)
- **Motivation** : effet père / lignée paternelle sur le potentiel hauteur. Standard BLUP.
- **Risques** : surapprentissage si smoothing mal calibré, fuite temporelle si pas de time series split
- **Statut** : 🟡 À tester
- **Décision** : retenue. Père = principal vecteur de transmission génétique mesurable, couverture excellente.
- **Date décision** : 2026-05-11

#### `gp_maternel_target_encoded_LOO`

- **Définition** : idem `pere_target_encoded_LOO` mais sur le grand-père maternel
- **Formule** : identique mais avec `NUMSIREPEREMERE` au lieu de `NUMSIREPERE`
- **Couverture** : 82,4% fiable, 93,6% calculable
- **Motivation** : effet "lignée maternelle profonde" via le grand-père maternel (étalon). Complète l'effet père.
- **Risques** : idem
- **Statut** : 🟡 À tester
- **Décision** : retenue. Le GP maternel est le 2e étalon dans le pedigree disponible, avec couverture comparable au père.
- **Date décision** : 2026-05-11

#### `pere_mean_gains_LOO` + `gp_maternel_mean_gains_LOO`

- **Définition** : target encoding LOO sur la variable `gains totaux carrière` des autres descendants (au lieu de la cible)
- **Source colonnes** : `NUMSIREPERE` / `NUMSIREPEREMERE`, `GAINS`
- **Couverture** : ~95% calculable
- **Motivation** : capter une dimension financière du potentiel transmis par l'ancêtre
- **Risques** : corrélation forte avec target encoding cible, mêmes précautions
- **Statut** : 🟡 À tester
- **Décision** : retenues par cohérence avec Famille 10 (target encoding sur plusieurs variables).
- **Date décision** : 2026-05-11

#### `pere_mean_percentile_partants_LOO` + `gp_maternel_mean_percentile_partants_LOO`

- **Définition** : target encoding LOO sur le percentile médian carrière (version "partants")
- **Source colonnes** : `NUMSIREPERE` / `NUMSIREPEREMERE`, `PLACE`, `NUMERO_EVENEMENT2`, `NUMEROSEQUENCE`
- **Motivation** : capter la dimension "performance relative" du potentiel transmis
- **Statut** : 🟡 À tester
- **Décision** : retenues. Cohérence avec Famille 10.
- **Date décision** : 2026-05-11

#### `pere_mean_percentile_finishers_LOO` + `gp_maternel_mean_percentile_finishers_LOO`

- **Définition** : idem avec percentile "finishers" (dénominateur = vrais classés uniquement)
- **Source colonnes** : idem
- **Statut** : 🟡 À tester
- **Décision** : retenues.
- **Date décision** : 2026-05-11

### Mère écartée (documenté pour traçabilité)

`mere_target_encoded_LOO` envisagée mais **rejetée** : seuls 2,7% des chevaux ont une mère avec ≥5 autres descendants dans la cohorte. Médiane = 1-2 descendants par mère. Le LOO est statistiquement instable même avec smoothing. Raison biologique (juments = ~10-15 poulains max sur leur vie vs étalons = centaines/milliers via insémination). À documenter comme **limitation méthodologique** dans le rapport.

### Récap Famille 8

| Variable encodée | Père | GP maternel | Total |
|---|---|---|---|
| Cible (hauteur_max_validée) | 1 | 1 | 2 |
| Gains totaux carrière | 1 | 1 | 2 |
| Percentile partants médian carrière | 1 | 1 | 2 |
| Percentile finishers médian carrière | 1 | 1 | 2 |
| **TOTAL** | **4** | **4** | **8 features** |

---

## Famille 10 -- Race

**Motivation** : standard académique. La race est un attribut structurel important du cheval, lié à son potentiel sportif intrinsèque (lignées d'élevage, sélection génétique). Référence : modèles BLUP français (Anne Ricard / IFCE) utilisent les effets de race comme effets fixes ou aléatoires.

**Variable source** : `RACECHEVAL` (catégorielle, ~139 races dans la base, **78 races** dans la cohorte T1+N1≥10).

**Sexe** : non disponible -- limitation à mentionner dans le rapport (cf. `13_limites_methodologiques.md` §6).

### Décisions de cadrage (2026-05-11)

| Décision | Choix |
|---|---|
| **Approche d'encodage** | **Target encoding LOO** avec smoothing Bayésien |
| **Justification** | Approche statistiquement la plus rigoureuse, standard ML (Micci-Barreca 2001) et cohérente avec l'approche BLUP en génétique équine (Henderson) |
| **Cohérence vs rejet Famille 4 (hauteurs)** | Justifiée : Famille 4 utilisait des stats individuelles du cheval (auto-corrélation directe), target encoding LOO utilise la moyenne raciale (attribut structurel) -- distinction validée par avis externe |
| **Précaution 1 -- Smoothing Bayésien** | Pour gérer les races à faible effectif (48 races sur 78 ont <10 chevaux). Formule : `TE_lissé = λ × moyenne(race) + (1-λ) × moyenne(globale)`. λ proche de 1 si race nombreuse, proche de 0 si rare. |
| **Précaution 2 -- Fit train uniquement** | Le target encoding est calculé exclusivement sur le train set, puis appliqué (transform) au test. Pas de leakage train→test. |
| **Précaution 3 -- Fuite temporelle** | À gérer lors de la phase modélisation via time series split (train sur cohortes anciennes, test sur cohortes récentes) -- car le target encoding "voyage dans le temps" sinon. |

### Avis externe (2026-05-11)

Question méthodologique soumise à une IA externe : target encoding LOO est-il cohérent avec le rejet de Famille 4 ?

**Réponse externe** : Position A (accepter target encoding LOO) validée. Distinction "comportement individuel rejeté" (Famille 4) vs "attribut structurel utilisé" (race) est solide. Références citées : Micci-Barreca 2001 (papier fondateur du target encoding avec smoothing), Henderson (BLUP).

**Caveat** : l'avis externe peut être biaisé / complaisant. Les références ont été vérifiées comme réelles. Le choix Position A reste un choix méthodologique assumé, défendable dans le rapport.

### Solution de repli si nécessaire

Si la solution target encoding LOO + 3 précautions s'avère trop complexe ou contestable lors de la modélisation, repli sur :
- **Option hybride** : regroupement métier en 4 méta-catégories (Selle Français / Warmbloods étrangers / Anglo-Arabe+Pur-Sang / Autres) + one-hot encoding
- Évite d'utiliser la cible mais perd en finesse (Holsteiner et KWPN traités pareil)

### Features

#### `race_target_encoded_LOO`

- **Définition** : pour chaque cheval, valeur de target encoding LOO de sa race, avec smoothing Bayésien
- **Formule** :
  ```python
  # Sur le train set uniquement :
  for chaque cheval i :
      autres_meme_race = chevaux de la même race que i, excluant i (LOO)
      moyenne_race_LOO = mean(cible des autres_meme_race)
      n = nb_autres_meme_race
      λ = n / (n + k)  # k = paramètre de lissage (à calibrer, ex : k = 30)
      TE_lissé_i = λ × moyenne_race_LOO + (1 - λ) × moyenne_globale_train
  ```
- **Source colonnes** : `RACECHEVAL` + cible
- **Couverture** : 100% (par construction)
- **Motivation** : capture l'effet "race" sur la cible en une seule feature numérique. Approche standard ML / BLUP.
- **Risques** :
  - Surapprentissage si smoothing mal calibré (λ trop élevé pour races rares)
  - Fuite temporelle si pas de time series split
  - Leakage train→test si pas de fit/transform séparé
- **Statut** : 🟡 À tester
- **Décision** : retenue. Approche statistiquement rigoureuse, cohérente avec littérature, avec 3 précautions documentées.
- **Date décision** : 2026-05-11

### Note sur le paramètre k du smoothing

Le paramètre `k` (souvent appelé `prior` ou `smoothing factor`) détermine à quelle vitesse on fait confiance à la moyenne de la race vs la moyenne globale :
- `k = 10` : on commence à faire confiance à la race à partir de ~10 chevaux
- `k = 30` : à partir de ~30 chevaux
- `k = 100` : seules les races très peuplées sont traitées avec leur propre moyenne

Valeur à **calibrer par cross-validation** en phase modélisation. Pas de valeur "magique" universelle.

### Encodages complémentaires (autres dimensions de l'effet race)

**Décision 2026-05-11** : ajouter 3 encodages "race" complémentaires en target encoding LOO + smoothing, sur des variables différentes que la cible directe. Permet de capter plusieurs dimensions de l'effet race (potentiel hauteur vs capacité de gains vs classement relatif).

#### `race_mean_gains_LOO`

- **Définition** : moyenne LOO des gains totaux carrière des autres chevaux de la même race, avec smoothing Bayésien
- **Variable encodée** : gains totaux de carrière (en €, brut)
- **Formule** : identique à `race_target_encoded_LOO` mais sur gains au lieu de cible
- **Source colonnes** : `RACECHEVAL`, `GAINS`
- **Couverture** : 100%
- **Motivation** : capter la capacité moyenne de la race à générer des gains. Dimension financière complémentaire à la hauteur.
- **Risques** :
  - Forte corrélation attendue avec `race_target_encoded_LOO` (race performante en hauteur = race qui rapporte)
  - Mêmes précautions : smoothing, fit train uniquement, fuite temporelle
- **Statut** : 🟡 À tester
- **Décision** : retenue par cohérence avec l'approche "inclusif maintenant".
- **Date décision** : 2026-05-11

#### `race_mean_percentile_partants_LOO`

- **Définition** : moyenne LOO du percentile médian (version "partants") des autres chevaux de la même race, calculé sur leur carrière, avec smoothing Bayésien
- **Variable encodée** : percentile médian (PLACE / nb_partants_total) sur la carrière entière
- **Source colonnes** : `RACECHEVAL`, `PLACE`, `NUMERO_EVENEMENT2`, `NUMEROSEQUENCE`
- **Couverture** : 100%
- **Motivation** : capter la capacité moyenne de la race à se classer relativement au peloton initial. Dimension performance relative complémentaire.
- **Risques** : corrélation avec target encoding sur cible, mêmes précautions
- **Statut** : 🟡 À tester
- **Décision** : retenue. Cohérence avec la double version partants/finishers de Famille 3.
- **Date décision** : 2026-05-11

#### `race_mean_percentile_finishers_LOO`

- **Définition** : idem mais sur le percentile "finishers" (dénominateur = vrais classés uniquement)
- **Source colonnes** : idem
- **Statut** : 🟡 À tester
- **Décision** : retenue. Pendant "finishers" du percentile partants.
- **Date décision** : 2026-05-11

### Note sur la place brute

La feature `race_mean_place_LOO` (sur place brute) a été **écartée** : la PLACE brute n'est pas comparable entre épreuves de tailles différentes (5e sur 80 ≠ 5e sur 8). Cohérent avec le rejet de la place brute en Famille 3 et Famille 7 au profit du percentile.

---

## Famille 11 -- Géographique

**Statut famille (décision 2026-05-10)** : 🔴 **FAMILLE REJETÉE EN BLOC**.

### Justification du rejet

Quatre raisons :

1. **Aucune donnée géographique structurée** dans la base : pas de coordonnées GPS, pas de département/région explicite, pas d'adresse postale (cheval, cavalier, organisateur). La seule info géographique est embarquée dans le texte libre de `DESIGNATION`.

2. **Diversité approximative** : compter les `DESIGNATION` distinctes mélange des événements qui ont lieu au même endroit (ex : "FONTAINEBLEAU GRANDE SEMAINE" et "FONTAINEBLEAU SPECIAL CSI" = 2 valeurs distinctes pour le même lieu).

3. **Marqueurs de prestige arbitraires** : la notion de "grand lieu" (Fontainebleau, Saumur, Lamotte...) repose sur un savoir métier subjectif. Liste arbitraire impossible à défendre rigoureusement.

4. **Redondance avec Famille 1** : un cheval qui visite beaucoup de lieux distincts est aussi un cheval qui sort dans beaucoup d'événements (`nb_evenements_4_7` capté en Famille 1). Apport marginal d'une feature géographique.

### Conséquence

Aucune feature de la famille 11 n'est construite. À mentionner dans le rapport comme limitation : "L'analyse géographique (mobilité, lieu, prestige du circuit) n'a pas été conduite faute de données géographiques structurées dans la base FFE fournie."

### Étude bonus si temps

Pourrait éventuellement être tentée avec géocodage des `DESIGNATION` via une API externe (Nominatim, IGN). Effort important pour un apport probablement marginal. Parquée.

---

## Récapitulatif global

| Famille | Proposées (💭) | À tester (🟡) | Retenues (🟢) | Rejetées (🔴) | En attente (⏸) |
|---|---|---|---|---|---|
| 1 -- Activité / Volume (22 features) | 0 | 18 | 0 | 4 | 0 |
| 2 -- Gains (10 features) | 0 | 10 | 0 | 0 | 0 |
| 3 -- Performance sportive (156 features statuées en cours) | 0 | 153 | 0 | 3 | 0 |
| 4 -- Hauteurs (rejetée en bloc) | 0 | 0 | 0 | -- | 0 |
| 5 -- Niveau / type d'épreuves (24 features) | 0 | 24 | 0 | 0 | 0 |
| 6 -- Progression (rejetée en bloc) | 0 | 0 | 0 | -- | 0 |
| 7 -- Cavalier (56 features, Hurdle inclus) | 0 | 56 | 0 | 0 | 0 |
| 8 -- Pedigree (père + GP maternel × 4 variables) | 0 | 8 | 0 | 0 | 0 |
| 10 -- Race (4 target encodings LOO + smoothing) | 0 | 4 | 0 | 0 | 0 |
| 11 -- Géographique (rejetée en bloc) | 0 | 0 | 0 | -- | 0 |

---

**Document créé le 2026-05-02 -- à mettre à jour à chaque ajout/décision sur une feature**
