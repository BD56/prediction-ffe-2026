# Limites méthodologiques du projet

Document centralisant les **limites, biais, et choix méthodologiques contestables** du projet, à mentionner clairement dans le rapport final pour assurer la transparence et la rigueur scientifique.

L'objectif n'est pas d'invalider le projet mais d'**assumer ouvertement** ses limites pour que le lecteur puisse interpréter correctement les résultats.

---

## 1. Restriction du périmètre (cohorte de modélisation)

### Filtres appliqués

La cohorte de modélisation finale (**T1 + N1≥10**) résulte de filtres successifs :

| Filtre | Justification | Impact |
|---|---|---|
| Né entre 2006 et 2013 | Permet d'observer la carrière complète des chevaux dans la fenêtre 2010-2025. Borne sup 2013 validée empiriquement (seuls 5% des chevaux dépassent leur cible après 12 ans). | -21 races |
| Hors poney | Le sujet concerne les chevaux de sport, pas les poneys (grille de hauteurs différente). | -24 races |
| Discipline SO uniquement | Le sujet est centré sur le saut d'obstacles ; le CE (4,2% de la base) a des métriques fondamentalement différentes. | -3 races |
| ≥ 10 participations totales | Exclut les chevaux "passants" non représentatifs (essais ponctuels). | -13 races |

### Conséquences

- **Sur 198 659 chevaux totaux** dans la base → **52 959 chevaux** dans la cohorte (27%)
- **Sur 139 races distinctes** → **78 races** dans la cohorte (-61 races)
- 47 683 chevaux avec cible `hauteur_max_validée` calculable

### Domaine d'application du modèle

> "Ce modèle est entraîné et applicable **aux chevaux nés entre 2006 et 2013, hors poney, ayant au moins 10 participations en SO sur leur carrière**. Les chevaux hors cohorte (poneys, races exotiques marginales, chevaux peu actifs) ne sont pas couverts par ce modèle. Pour ces sous-populations, des modèles dédiés seraient nécessaires."

### Pourquoi c'est défendable

- Cohérent avec le sujet ("jeunes chevaux de sport en SO")
- Aucune race "importante en SO" n'est éliminée (SF, Warmbloods étrangers, Anglo-Arabe restent tous présents)
- Les races éliminées sont marginales (chevaux de trait, races exotiques rares)
- Pratique standard ML : on définit une cohorte d'entraînement, on rapporte la performance sur cette cohorte, on indique le domaine d'application

---

## 2. Biais de sélection et de représentation

### Biais "survivants" (N1 ≥ 10)

Le filtre "au moins 10 participations totales" privilégie les chevaux **qui ont continué à courir** sur leur carrière. On rate :
- Les chevaux **talentueux mais arrêtés tôt** (blessure, vente à l'étranger, problème comportemental)
- Les chevaux **passés en élevage** (juments, étalons écartés du sport)

**Conséquence** : le modèle est calibré sur des "survivants compétitifs", pas sur tous les chevaux potentiellement talentueux.

### Biais de représentation par race

La cohorte est dominée par **Selle Français toutes sections confondues à 69%**. Conséquences :
- Le signal sera plus fiable pour les Selle Français
- Les races minoritaires (KWPN, BWP, Holsteiner...) auront moins de données → variance plus forte sur leurs prédictions
- Risque de **biais d'inférence** : le modèle pourrait apprendre des patterns spécifiques au SF qui ne se généraliseraient pas aux autres races

### Mitigations possibles

- Stratification par race lors des analyses post-modèle
- Évaluation séparée du modèle sur chaque grand groupe (SF / Warmbloods / Anglo-Arabe / Autres)
- Mention explicite dans le rapport

---

## 3. Familles de features rejetées (et pourquoi)

### Famille 4 -- Hauteurs explorées (rejetée en bloc 2026-05-08)

**Raisons** :
1. La cible étant `hauteur_max_validée`, utiliser des dérivés de la hauteur (médiane, min, distribution...) reviendrait à prédire la cible avec elle-même → risque d'auto-corrélation
2. 47% de NaN HAUTEUR sur la fenêtre 4-7 (concentré sur les cycles SHF Élevage)
3. Biais de trajectoire pour les parcours mixtes (cheval qui passe de cycle SHF en Élevage à 4-5 ans, vers Amateur/Pro à 6-7 ans → le modèle voit "commence à 1,20m" alors qu'il avait commencé à ~0,95m en cycle SHF)
4. Pas d'accès à la grille SHF des hauteurs réglementaires (demande FFE non aboutie)

**Conséquence rapport** : "Les features dérivées de la hauteur d'obstacle ont été délibérément écartées pour éviter une auto-corrélation avec la cible et pour des raisons de qualité de données."

### Famille 6 -- Progression temporelle (rejetée en bloc 2026-05-10)

**Raisons** : la dimension progression est déjà nativement capturée par les ~70-80 deltas inter-annuels dispersés dans toutes les autres familles (Famille 1, 2, 3, 5, 7). Une famille dédiée n'apporterait que de la redondance.

### Famille 11 -- Géographique (rejetée en bloc 2026-05-10)

**Raisons** :
1. Aucune donnée géographique structurée dans la base (pas de GPS, pas de département)
2. La seule info est embarquée dans le texte libre de `DESIGNATION`
3. Diversité approximative ("FONTAINEBLEAU GRANDE SEMAINE" et "FONTAINEBLEAU SPECIAL CSI" = 2 valeurs distinctes pour le même lieu)
4. Marqueurs de prestige subjectifs
5. Redondance avec `nb_evenements_4_7` (Famille 1)

### Famille 3 sous-famille "sans-faute" (rejetée 2026-05-03)

**Raisons** :
1. Sémantique non-uniforme de la variable POINTS (différents barèmes : A, C, cycles SHF Label/Formation)
2. Couverture de SO_POINTS_BAR biaisée temporellement (1,4% à 4 ans → 84% à 7 ans)
3. Aucune fenêtre alternative ne résout le problème
4. Pas de documentation FFE sur les barèmes (demande non aboutie)

**Conséquence rapport** : "Le taux de sans-faute, métrique standard de la littérature équitation (EquiRatings, IFCE), n'a pas pu être implémenté en raison de l'hétérogénéité sémantique de la variable POINTS et de l'absence de documentation FFE sur les barèmes."

---

## 4. Codes PLACE administratifs (sémantique inconnue)

### Constat

5 codes administratifs identifiés dans la variable PLACE : **899, 900, 902, 992, 993** (rupture nette à partir de 351). Représentent **~7,3% des participations SO**.

### Limite

La **sémantique exacte** de ces codes n'a pas pu être déterminée :
- Aucune documentation publique (FFE Compet portail `toutsavoir.ffecompet.com` inaccessible)
- Recherche web infructueuse
- Hypothèses possibles (élimination, abandon, hors concours, non-classé administratif, disqualifié) mais non vérifiables

### Décision méthodologique

Traitement "en bloc" comme catégorie unique de **non-classement**. Conséquences :
- Features de performance (médiane, percentile, top X) calculées **uniquement sur les vraies places (1-351)**
- Feature séparée `taux_non_classement_4_7` qui compte tous les codes ensemble
- Perte d'information sur la distinction "élimination" vs "non-classé administratif" -- limite assumée

---

## 5. Fenêtre temporelle des features (4-7 ans)

### Choix

Les features sont calculées sur les participations du cheval entre 4 et 7 ans inclus (cohérent avec les cycles SHF jeunes chevaux français).

### Risque de leakage partiel

**36,7% des chevaux de la cohorte** valident leur cible `hauteur_max_validée` **avant 8 ans** (= dans la fenêtre). Pour ces chevaux, certaines features liées à la hauteur seraient fortement corrélées à la cible.

### Mitigation

- **Toute la Famille 4 (Hauteurs) a été rejetée** précisément pour éviter ce piège
- Analyse fine du leakage par niveau atteint :
  - Bas niveau (≤1,10m) : 85% atteignent leur cible dans 4-7 → leakage massif pour ces chevaux
  - Top niveau (≥1,45m) : seulement 1,5% atteignent leur cible dans 4-7 → fenêtre pertinente
- La fenêtre 4-7 reste défendable car le leakage est concentré sur les chevaux faciles à prédire de toute façon

---

## 6. Variable "sexe" non disponible

### Constat

La variable **sexe** (mâle entier, hongre, femelle) n'est **pas présente** dans les données fournies par la FFE. Inférable partiellement à partir du libellé de certaines épreuves (épreuves "5 ans Hongre", "Femelle 4 ans") mais ne couvre que **8,4% des chevaux** (16 687 / 198 659).

### Conséquence

- Impossible d'utiliser le sexe comme feature, alors que c'est une variable importante en évaluation génétique équine (les juments et les hongres ont des trajectoires de performance différentes)
- Limite **forte** à mentionner dans le rapport
- Question complémentaire posée à la FFE : peuvent-ils enrichir le dataset avec le sexe depuis le SIRE ? Non aboutie.

---

## 7. FFE non répondue

### Questions FFE laissées en suspens

Le mail à Emmanuel HUDE (V5, prêt depuis 2026-04-28) n'a pas reçu de réponse :

| Question | Conséquence du non-retour |
|---|---|
| Définition métier de "jeune cheval" | Adopté définition basée sur cycles SHF (4-7 ans) -- défendable empiriquement |
| Définition métier de "haut niveau" | Définition à trancher empiriquement ou en phase modélisation |
| Usage final du modèle | Adopté positionnement neutre (régression continue) |
| Grille hauteur SHF | Famille 4 rejetée en bloc (pas de reconstruction possible) |
| Sémantique barèmes FFE | Sans-faute non implémenté |
| Sexe via SIRE | Variable manquante |

### Conséquence

Plusieurs choix méthodologiques pris **sans validation métier**. À mentionner comme limite : "Plusieurs choix de cadrage ont été pris sans validation métier FFE, faute de retour de notre correspondant. Ces choix sont défendus empiriquement et sont reproductibles."

---

## 8. Choix méthodologiques arbitraires (assumés)

### Convention de transformation des gains

- Décision : stocker `GAINS` en **valeurs brutes (€)**, transformation `log(GAINS + 1)` standardisée par année appliquée **à la modélisation**
- Limite : `log(x+1)` introduit une légère déformation pour les petits gains (~11% pour 5€, ~1-3% pour 12-25€)
- Cohérence : standard Anne Ricard / IFCE

### Convention "leave-one-out" pour Famille 7 (Cavalier)

- Décision : fenêtre passée 3 ans glissante (n-2, n-1, n) du cavalier, excluant le cheval analysé
- Limite : 7,9% des participations ont un LOO impossible (cavalier qui n'a monté aucun autre cheval) → NaN
- Limite : un cavalier qui a peu de chevaux dans la fenêtre donne une métrique instable

### ✅ Target encoding LOO -- LEAKAGE CORRIGÉ (mai 2026)

**Limitation initiale identifiée** : le target encoding LOO + smoothing Bayésien des Familles 8 (Pedigree) et 10 (Race) était calculé **sur l'ensemble de la cohorte de modélisation** (52 959 chevaux), ce qui créait un leakage train → test sur ces 12 features.

**Correction appliquée** (script `20_correct_target_encoding.py`, mai 2026) :
1. Split time series par génération appliqué (train 2006-2010 / valid 2011-2012 / test 2013)
2. Target encoding recalculé **uniquement sur le train**
3. Appliqué (transform) sur valid et test sans recalcul
4. Sortie : `master_dataset_clean.parquet` (190 features f*)

**Impact quantifié a posteriori** (script `57_hurdle_compare_datasets.py`) :
- **Avant correction (dataset brut)** : MAE Hurdle = 6,14 cm, R² = 0,5977, AUC classifier = 0,929
- **Après correction** : MAE Hurdle = 6,89 cm, R² = 0,5070, AUC classifier = 0,899
- **Gonflement artificiel** : 0,75 cm sur la MAE, +0,09 sur R², +3 points d'AUC
- Sans cette correction, les performances rapportées auraient été surévaluées de ~12%.

**Valorisation pour le rapport** : *"Nous avons identifié et corrigé un leakage dans les target encodings au début de la phase modélisation. Cette correction a fait reculer la MAE apparente de 6,14 cm à 6,89 cm, révélant la véritable performance du modèle."*

### Hypothèses sémantiques spéculatives (Famille 1 -- Activité)

Plusieurs features sont retenues avec des **interprétations métier non validées empiriquement** :
- `intensite_moyenne_mensuelle` : interprétée comme *"rythme industriel = risque sanitaire"* — spéculatif
- `jours_moyens_entre_sorties` : interprétée comme *"rythme d'or = bon"* — spéculatif
- `ratio_cycle_total` : interprétée comme indicateur de surcharge — spéculatif

Ces interprétations sont des **hypothèses à tester**, pas des vérités acquises. Le rapport doit présenter ces features comme **signaux candidats** sans imposer leur sens métier.

### Famille 7 percentile : moyenne vs médiane

- Le score par participation est calculé comme moyenne LOO sur fenêtre 3 ans (limitation des sums vectorisés pour gérer ~1,5M participations cohorte 4-7)
- L'agrégation au niveau cheval est en médiane (cohérent avec naming "...median..." du catalogue)
- Pour une version "médiane à tous niveaux" (plus rigoureuse), il faudrait stocker toutes les valeurs individuelles avant agrégation -- coût mémoire et calcul beaucoup plus élevé. Limitation pragmatique acceptée.

### Définition de "haut niveau" (question parquée)

- Plusieurs définitions possibles (seuil 1,40m, zone 1,40-1,50m, trois zones)
- Aucune tranchée à ce stade -- à décider en phase modélisation avec les distributions empiriques
- Limite : choix arbitraire à défendre dans le rapport

---

## 9. Vérification de la littérature

Certaines références citées (notamment Chapard 2023, Chapard 2024) ont été **vérifiées rétroactivement** au cours du projet (2026-05-10) :

- ✅ **Chapard 2023** ("Adjusted fence height") : confirmé, auteur **Léa Chapard** (Université Paris-Saclay, INRAE, GABI)
- ✅ Concept "traits de jeunesse → performance future" : confirmé, **mais chiffres précédemment cités inexacts** (corrigés en 0,40-0,65 au lieu de 0,30-0,77)
- ⚠ Quelques autres références (Ricard & Blouin 2011) restent présumées et nécessitent une vérification individuelle avant citation finale

### Mitigation

Document `12_sources_litterature.md` créé pour tracer toutes les sources et leur statut de vérification.

---

## 10. Limites structurelles complémentaires (issues d'autres documents)

### Démixtion cheval/cavalier impossible pour 37% des chevaux

Source : [01_relation_cheval_cavalier.md](01_relation_cheval_cavalier.md) §6.

- **37% des chevaux n'ont eu qu'un seul cavalier** dans toute leur carrière
- Pour ces chevaux, il est **impossible de séparer** statistiquement "effet cheval" et "effet cavalier"
- La performance d'un cheval monté toute sa carrière par un excellent cavalier sera **artificiellement gonflée**, sans qu'on puisse distinguer la part du cheval vs celle du cavalier

**Mitigation appliquée** : Famille 7 (Cavalier) inclut des features décrivant le niveau du cavalier (percentiles, gains positifs) pour que le modèle puisse partiellement compenser. Mais c'est imparfait pour les chevaux à cavalier unique.

**Approche académique standard non implémentée** : modèle mixte avec effet aléatoire cavalier (cf. Chapard 2023, Sanchez-Guerrero 2024, Équidata "EquiDT HI®"). Notre approche RF/Hurdle ne permet pas cette décomposition.

### Effet de seuil sur la cible et fenêtre temporelle 4-7

Source : [06_features_engineering_plan.md](06_features_engineering_plan.md) §"Analyse data leakage".

**Phénomène observé** : l'âge auquel un cheval **valide** sa hauteur maximale est fortement corrélé à son niveau final.

| Âge de validation | % à ≤1.15m | % à 1.20-1.30m | % à 1.35-1.40m | % à ≥1.45m |
|---|---|---|---|---|
| 4-5 ans | **98%** | 2% | 0% | 0% |
| 6 ans | 74% | 26% | 0% | 0% |
| 7 ans | 38% | 46% | 16% | 0% |
| 8+ ans | 34% | 39% | 22% | 6% |

**Conséquence** : **89,5% des futurs tops (≥1,40m) ne valident leur hauteur qu'après 7 ans**. Notre fenêtre features (4-7 ans) capture donc leur **trajectoire avant la révélation de leur niveau réel**.

- Pour les **chevaux faibles** (≤1,15m) : la cible est connue dès 4-5 ans → leakage partiel possible si on utilise des features dérivées de la hauteur
- Pour les **chevaux tops** : la cible n'est révélée qu'après 7 ans → notre prédiction repose sur **des signaux précoces faibles**, ce qui explique le **plateau de performance** observé (MAE plancher ~6 cm, R² plafond ~0,57)

**Conséquence rapport** : *"La fenêtre temporelle features 4-7 ans capture pour l'essentiel le **potentiel** des futurs tops avant que celui-ci ne se révèle. Cette asymétrie (cible connue tôt pour les bas niveaux, tard pour les hauts) impose une limite intrinsèque à la précision prédictive sur les tops."*

### Anomalies de données brutes documentées

Source : [04_analyse_variables_raw.md](04_analyse_variables_raw.md) "Points d'attention pour la suite".

Valeurs aberrantes administratives identifiées et **traitées comme NaN** dans le pipeline :
- **Codes placeholders** : `9999` (POINTS, SO_POINTS_BAR), `1000` (SO_TEMPS), `999` / `999.99` (CE_*)
- **Valeurs négatives** dans POINTS, SO_POINTS_BAR, CE_POINTSDRESSAGE (pénalités ou corrections administratives — sémantique non documentée)
- **5 NaN sur DATENAISSANCE** : chevaux sans année de naissance → exclus
- **12 libellés avec 2 codes d'épreuve différents** : sources potentielles d'incohérence (résolu par jointure majoritaire)

Ces anomalies ont été traitées **avant la construction du master dataset**, mais leur existence souligne la qualité hétérogène des données administratives FFE.

---

## 11. Limites détectées en phase modélisation (mai 2026)

### Plateau de performance — limite irréductible

**Observation empirique** : malgré 10+ modèles testés (RF, ElasticNet, XGBoost, CatBoost, Hurdle, Stacking + variantes, Poly40, Multi-Hurdle) et toutes les optimisations méthodologiques (TE corrigé, épuration features, calibration, sample weights), la MAE plafonne autour de **6,28-6,89 cm** et le R² autour de **0,55-0,57**.

**Interprétation** : cette limite est **structurelle au dataset**, pas méthodologique. Elle reflète :
- L'**incertitude irréductible** dans la prédiction (santé, opportunités sportives, qualité cavalier adulte non observables)
- L'**asymétrie temporelle** décrite en §10 (les tops se révèlent après la fenêtre de features)

Aucun ajustement de modèle ne franchira ce plafond sans nouvelles variables explicatives.

### Régression vers la moyenne forte sur les tops

Tous les modèles "uniformes" (RF, ElasticNet, etc.) sous-estiment systématiquement les vrais tops :
- Pour les chevaux ≥1,45m (vrai = 1,469m), RF default prédit en moyenne **1,330m** → biais −13,9 cm
- Hurdle réduit ce biais à −9,5 cm via son régresseur conditionnel, mais ne l'élimine pas

**Cause** : la fonction de perte (MAE/RMSE) symétrique pénalise pareillement sur- et sous-estimations. Comme les tops ne représentent que 10% du dataset, le modèle "consent" à les sous-estimer pour mieux prédire la majorité.

**Mitigation appliquée** : Hurdle (2 classes) — meilleur modèle pour les tops.

### Défaut Hurdle sur les chevaux peu actifs

Source : [00_journal_decisions.md](00_journal_decisions.md) §"ANALYSE DES RÉSIDUS EN 4 ANGLES" (2026-05-11).

Hurdle a un biais systématique de **−5,13 cm** sur les chevaux du Q1 d'activité (peu de participations sur 7 ans). Le classifier balanced donne `P(top) > 0` même à des chevaux qui n'ont quasi pas couru → activation injustifiée du `pred_tops`.

**Voie d'amélioration possible** (non testée) : seuil garde-fou sur `nb_participations_7ans` pour désactiver le mécanisme Hurdle sur les chevaux sous-observés.

### Outliers à parcours interrompu (~5%)

Source : [00_journal_decisions.md](00_journal_decisions.md) §"ANALYSE DES RÉSIDUS" + analyse des fiches de prédiction.

Environ **5% des chevaux** ont des résidus > 3σ. Pattern dominant identifié : **chevaux avec très peu de participations sur 7 ans** mais profil race/pedigree "haut". Hypothèse : chevaux à **parcours interrompu** (trauma précoce, problèmes vétérinaires, vente/changement de propriétaire).

**Limite irréductible** : aucun modèle ne capte ces cas sans variable explicite **santé / continuité de carrière**, qui n'est pas dans le dataset.

### Non-normalité des résidus

Tests D'Agostino-Pearson p < 1e-13 sur les 3 modèles principaux → **distribution non-normale** avec queues lourdes (kurtosis 0,44-0,75).

**Implications** :
- Les métriques rapportées (MAE, RMSE, R², Spearman) restent valides (non-paramétriques)
- Les **intervalles de confiance individuels** ne peuvent **pas** être calculés par formule paramétrique (ŷ ± 1,96σ interdit)
- Solution adoptée : **Conformal Prediction** (couverture garantie sans hypothèse de loi)

### Instabilité du Stacking sur les tops

Source : [00_journal_decisions.md](00_journal_decisions.md) §"VALIDATION CROISÉE DE LA ROBUSTESSE".

Lors de la validation sur 10 splits différents (4 expanding window + 1 random + 5 TimeSeriesSplit), le Stacking + Calib a un **écart-type de 4,46 cm sur ≥1,45m**. Sa MAE varie entre 6,43 et 20,98 cm selon le split. La calibration isotonic est très sensible aux particularités du valid set.

Hurdle est en comparaison **plus stable** (écart-type 0,65-0,81 cm). C'est un argument supplémentaire pour le préférer en production.

### Confiance σ(x) ≠ vérité métier

Source : [00_journal_decisions.md](00_journal_decisions.md) §"FICHES DE PRÉDICTION FFE".

L'indicateur "Confiance modèle" (1-5 étoiles, basé sur les quintiles de σ(x) = écart-type entre les 500 arbres) mesure la **cohérence interne** du modèle, **pas la justesse**.

**Illustration** : un cheval outlier (parcours interrompu) peut recevoir 5/5 étoiles ET avoir une prédiction totalement fausse. Tous les arbres voient les mêmes features "race haute + pedigree fort", votent unanimement "Pro", mais ignorent l'information manquante (le cheval s'est arrêté à 5 ans).

**Limite à mentionner explicitement dans le livrable** pour ne pas induire l'utilisateur en erreur.

### Échec pédagogique du bootstrap naïf pour IC

Source : [00_journal_decisions.md](00_journal_decisions.md) §"ÉCHEC PÉDAGOGIQUE".

Premier essai d'intervalles de confiance par bootstrap K=50 → couverture réelle **8-10%** pour un IC nominal à 95% (catastrophique).

**Cause identifiée** : le bootstrap naïf mesure la variance **épistémique** (variance des prédictions due à la variabilité du training set), pas la variance **aléatoire** (bruit irréductible). Avec RF 500 arbres sur 31k chevaux, la variance bootstrap est très faible (≈1-2 cm) alors que l'incertitude réelle est de 10-20 cm.

**Leçon méthodologique valorisable** : on a mesuré la couverture empirique avant de publier les IC → preuve de rigueur. Correction adoptée : **Conformal Prediction** (split puis adaptatif).

---

## 12. Voies d'amélioration potentielles (pour un M2 ou un projet de suite)

### Acquérir de nouvelles variables explicatives (priorité haute)

Le plateau de performance ne sera pas franchi sans nouvelles features. Variables qu'on aurait aimé avoir :

1. **Santé / continuité de carrière** : interruptions vétérinaires, blessures déclarées, périodes d'inactivité justifiées
2. **Environnement compétition** : qualité du circuit fréquenté, niveau régional vs national
3. **Performance des poulinées** (mère) : on n'a pas pu l'exploiter car seulement 2,7% des juments ont ≥5 descendants dans la cohorte
4. **Sexe via SIRE** : variable non fournie par le FFE
5. **Caractéristiques physiques** : taille au garrot, conformation (si disponibles via SIRE)
6. **Indicateurs vétérinaires** : suivi médical, ostéo, etc. (probablement non disponibles)

### Approches modélisation non testées

1. **Modèles mixtes avec effet aléatoire cavalier** (BLUP-style, approche Chapard/Sanchez-Guerrero) — permettrait de démixer cheval/cavalier proprement et corrigerait le biais des 37% à cavalier unique
2. **Hurdle avec garde-fou activité** — désactiver le mécanisme tops sur `nb_participations_7ans < 5` pour corriger le biais Q1
3. **Mixture of Experts** avec gating network softmax — généralisation continue de Hurdle (transition douce au lieu de mélange binaire)
4. **Poids temporels** (piste IA externe non testée) — pondérer le train par la récence de génération
5. **Locally weighted conformal V3** — IC adaptatif basé sur KNN novelty detection (mesure de "à quel point ce cheval ressemble aux chevaux du train")

### Étude complémentaire CE (Concours Complet)

Mentionnée en §1 et dans le journal : sous-dataset CE (~340k lignes) écarté de l'étude principale, à traiter comme étude bonus si temps. Métriques différentes (CE_POINTSDRESSAGE/FOND/SO), cible à redéfinir.

### Améliorations ergonomiques du livrable

1. **Détection automatique des "chevaux atypiques"** : flag si σ(x) > Q5 ET features signal "rare", pour signaler à l'utilisateur "ce cheval ne ressemble à rien de connu"
2. **Visualisation de la distribution prédite** : au lieu d'un IC fixe, montrer un histogramme des K prédictions (style fan chart)
3. **API ou Excel template** pour saisir un nouveau cheval et obtenir sa fiche automatiquement

### Validation prospective

À ce stade, le modèle a été validé uniquement **rétrospectivement** sur les générations 2006-2013. Une **validation prospective** sur les chevaux nés en 2018-2020 (dont la cible se révèle actuellement) serait la véritable preuve de robustesse en conditions opérationnelles.

---

## 13. Synthèse pour le rapport

Section "Limites" à inclure dans le rapport final, articulée autour de :

### Limites de cadrage et de données

1. **Périmètre cohorte** -- définition explicite + domaine d'application (§1)
2. **Biais de sélection et de représentation** (survivants, dominance SF 69%) (§2)
3. **Familles écartées** (Hauteur, Progression, Géographique) -- justifications (§3)
4. **Codes PLACE** non interprétables (~7% non-classement) (§4)
5. **Fenêtre 4-7** -- leakage partiel concentré sur bas niveau ET effet de seuil sur les tops (§5 + §10)
6. **Sexe non disponible** (§6)
7. **Non-retour FFE** -- choix méthodologiques sans validation métier (§7)
8. **Décisions arbitraires** assumées (transformation gains, LOO cavalier, hypothèses sémantiques spéculatives, seuil "haut niveau") (§8)
9. **Démixtion cheval/cavalier** impossible pour 37% des chevaux (§10)
10. **Anomalies de données brutes** documentées et traitées (§10)

### Limites détectées en modélisation (§11)

11. **Plateau de performance** : MAE ~6,3 cm, R² ~0,57 — limite irréductible des données
12. **Régression vers la moyenne** : sous-estimation systématique des tops (jusqu'à −13,9 cm)
13. **Défaut Hurdle sur les chevaux peu actifs** (biais −5,13 cm sur Q1 d'activité)
14. **Outliers à parcours interrompu** (~5%, non-prédictibles sans variable santé)
15. **Non-normalité des résidus** (queues lourdes, kurtosis +0,68)
16. **Instabilité du Stacking** sur les tops (std 4,46 cm selon le split)
17. **Confiance σ(x) ≠ vérité métier** (cohérence interne mesurée, pas justesse)
18. **Échec bootstrap naïf** pour IC (leçon méthodologique : Conformal adopté)

### Voies d'amélioration (§12)

19. **Variables manquantes** : santé, environnement, sexe, conformation
20. **Approches non testées** : modèles mixtes (BLUP), Mixture of Experts, garde-fou activité
21. **Étude CE** en bonus
22. **Validation prospective** à conduire sur générations 2018-2020

### Vérification littérature

23. **Doc [12_sources_litterature.md](12_sources_litterature.md)** : sources tracées avec statut de vérification (Chapard ✅, Ricard ⚠ à confirmer)

---

**Document créé le 2026-05-10, étendu le 2026-05-12 avec les retours d'expérience de la phase modélisation et le scan exhaustif des limites mentionnées dans les autres documents du dossier.**
