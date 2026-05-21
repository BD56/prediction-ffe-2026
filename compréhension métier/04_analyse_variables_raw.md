# Analyse variable par variable du dataset RAW (8 075 470 lignes × 27 colonnes)

Dataset : fusion des 16 fichiers CSV de 2010 à 2025. Aucun filtre appliqué.

Analyse menée en collaboration, à finaliser au terme de la revue complète.

---

## Vue d'ensemble

- **Lignes** : 8 075 470 (granularité : 1 ligne = 1 participation d'un cheval à une épreuve)
- **Colonnes** : 27
- **Chevaux uniques** : 198 659
- **Cavaliers uniques** : 91 406
- **Événements uniques** : 30 237
- **Période** : 2010-01-17 → 2025-12-28

---

## 1. Identification cheval

### `N° SIRE` -- identifiant cheval

| Propriété | Valeur |
|---|---|
| Type | text |
| NaN | 0 |
| Unique | 198 659 |
| Longueur | 6 à 9 caractères |
| Lignes par cheval | min 1, médiane 22, max 792 |

Identifiant du registre national géré par l'IFCE. **Stable** tout au long de la carrière du cheval (vérifié : 0 incohérence de métadonnées entre divisions). Clé primaire fiable.

### `NOMCHEVAL` -- nom du cheval

| Propriété | Valeur |
|---|---|
| Type | text |
| NaN | 0 |
| Unique | 196 127 |
| Longueur | 1 à 25 caractères |

Environ 196 127 noms pour 198 659 chevaux -> **1 787 noms sont homonymes**, concernant 4 319 chevaux (2,2% du total). Ne pas utiliser le nom comme clé. Uniquement valeur d'affichage.

### `DATENAISSANCE` -- année de naissance

| Propriété | Valeur |
|---|---|
| Type | float64 (contient année seulement, pas de mois/jour) |
| NaN | 5 lignes (anomalie à investiguer) |
| Range | 1977 → 2021 |
| Q1/médiane/Q3 | 2005 / 2008 / 2012 |
| Cohérence | 1 cheval = 1 seule année de naissance (vérifié) |

Attention : la variable s'appelle "DATENAISSANCE" mais ne contient que l'**année**. C'est courant dans le sport hippique où les chevaux sont classés par année. Les 5 NaN sont à flaguer -- sans année de naissance, impossible de calculer l'âge.

### `RACECHEVAL` -- race du cheval

| Propriété | Valeur |
|---|---|
| Type | text |
| NaN | 0 |
| Unique | 139 races |

Top races (% du total) :

| Race | % |
|---|---|
| SELLE FRANCAIS | 34,2% |
| SELLE FRANCAIS SECTION A | 32,3% |
| SELLE FRANCAIS SECTION B | 4,6% |
| ZANGERSHEIDE | 3,5% |
| BELGIAN WARMBLOOD | 3,2% |
| ORIGINE CONSTATEE | 3,0% |
| KON. WARM PAARD NEDERLAND | 2,9% |
| PONEY FRANCAIS DE SELLE | 2,2% |
| ANGLO-ARABE | 1,9% |
| CHEVAL DE SPORT BELGE | 1,7% |

**~70% sont des Selle Français** (SF + SF Section A + SF Section B). Forte dominance française. La queue longue (139 races) inclut des poneys et races étrangères.

---

## 2. Généalogie (3 variables)

### `NUMSIREPERE`, `NUMSIREMERE`, `NUMSIREPEREMERE`

| Variable | NaN lignes | NaN % | % chevaux avec ≥1 valeur | Nb SIRE uniques |
|---|---|---|---|---|
| `NUMSIREPERE` | 8 642 | 0,11% | 99,01% | 16 711 |
| `NUMSIREMERE` | 8 614 | 0,11% | 99,01% | 100 403 |
| `NUMSIREPEREMERE` | 94 576 | 1,17% | 96,67% | 15 723 |

**Cohérence intra-cheval : parfaite** (0 incohérence sur 198k chevaux).

Points notables :
- **Beaucoup moins de pères uniques que de mères** (16k vs 100k) : les étalons sont sur-utilisés par la sélection (chaque étalon a en moyenne ~12 descendants dans les données, chaque jument ~2)
- NUMSIREPEREMERE (grand-père maternel) est légèrement moins renseigné (1,2% manquant vs 0,1%)

Pour un modèle de pedigree, **généalogie disponible pour 96,7% des chevaux sur 3 niveaux** (père, mère, grand-père maternel).

---

## 3. Identification cavalier

### `LICENCE` -- identifiant cavalier

| Propriété | Valeur |
|---|---|
| Type | text |
| NaN | 0 |
| Unique | 91 406 |
| Longueur | 8 caractères (fixe) |
| Lignes par cavalier | min 1, médiane 24, max **8 217** |

Format alphanumérique stable. Le max de 8 217 lignes pour un seul cavalier correspond à un cavalier Pro très actif sur 16 ans. Anonyme mais cohérent dans le temps.

---

## 4. Identification événement

### `NUMERO_EVENEMENT2` -- identifiant événement

| Propriété | Valeur |
|---|---|
| Type | int64 |
| NaN | 0 |
| Unique | 30 237 événements |
| Range | 201001013 → 202598050 |

**Structure du code** : les 4 premiers chiffres encodent l'année (préfixe = année sauf 25 lignes en anomalie, négligeable).

Lignes par événement :
- Min : 1
- Médiane : 186
- Max : 6 666 (un gros événement concentre jusqu'à 6 666 participations)

Un événement = un concours, potentiellement sur plusieurs jours et plusieurs épreuves.

### `DESIGNATION` -- nom du concours

| Propriété | Valeur |
|---|---|
| NaN | 0 |
| Unique | 6 641 concours |

Relation avec NUMERO_EVENEMENT2 : 1 événement = 1 designation unique (vérifié, 0 cas multiple). Donc `DESIGNATION` est **redondante** avec `NUMERO_EVENEMENT2` au niveau métier.

Top concours par volume :

| Concours | Participations totales |
|---|---|
| BARBASTE | 113 071 |
| AUVERS | 104 499 |
| LE PIN AU HARAS | 74 597 |
| NOTRE DAME D'ESTREES | 69 542 |
| LIVERDY EN BRIE | 69 030 |
| DEAUVILLE | 65 687 |

Note : 30 237 événements vs 6 641 désignations = certains lieux (ex: Barbaste, Fontainebleau) accueillent plusieurs concours par an sur 16 ans.

### `NUMEROSEQUENCE` -- séquence de l'épreuve dans l'événement

| Propriété | Valeur |
|---|---|
| Type | int64 |
| NaN | 0 |
| Range | 1 → 952 |
| Percentiles 5/50/95/99 | 2 / 9 / 25 / 50 |
| Valeurs uniques par événement | min 1, médiane 8, max 110 |

Numéro d'ordre de l'épreuve dans l'événement. Le max de 952 est probablement un identifiant numérique (code administratif), pas un rang réel. La médiane de 8 épreuves uniques par événement est cohérente avec un concours typique (2-3 jours × 3-4 épreuves/jour).

### `DATEEPREUVE` -- date de l'épreuve

| Propriété | Valeur |
|---|---|
| NaN | 0 |
| Range | 2010-01-17 → 2025-12-28 |
| Format | datetime (string "YYYY-MM-DD HH:MM:SS") |

**Saisonnalité marquée** (% des participations par mois) :

| Mois | % | Période |
|---|---|---|
| Janvier | 0,16% | Creux hivernal |
| Février | 1,35% | |
| Mars | 8,96% | Début de saison |
| Avril | 15,05% | **Haute saison** |
| Mai | 15,58% | **Haute saison** |
| Juin | 14,76% | **Haute saison** |
| Juillet | 12,81% | |
| Août | 9,90% | Creux estival |
| Septembre | 9,06% | |
| Octobre | 7,87% | |
| Novembre | 3,83% | Fin de saison |
| Décembre | 0,68% | Creux hivernal |

**75% des participations** se concentrent d'avril à septembre. Implications modélisation : effet saisonnier à considérer, les concours indoor d'hiver sont minoritaires mais différents (CSI indoor).

---

## 5. Classification de l'épreuve

### `CLASSEEPREUVE_CODE` et `CLASSEEPREUVE_LIB`

| Propriété | CODE | LIB |
|---|---|---|
| NaN | 0 | 0 |
| Unique | 1 552 | 1 540 |

**Relation code ↔ libellé** :
- 1 code a 1 libellé unique (0 exception)
- 12 libellés ont 2 codes différents (0,8% des libellés)

Top 10 codes par volume :

| Code | Libellé | % total |
|---|---|---|
| SOA2G | Amateur 2 Grand Prix (1,10 m) | 9,23% |
| SOP2G | Pro 2 Grand Prix (1,30 m) | 7,97% |
| SOA1G0 | Amateur 1 Grand Prix (1,15 m) | 7,71% |
| SOA2G0 | Amateur 2 Grand Prix (1,05 m) | 6,15% |
| SOA1G | Amateur 1 Grand Prix (1,20 m) | 5,71% |
| SOP2G5 | Pro 2 Grand Prix (1,35 m) | 4,03% |
| SOA2V0 | Amateur 2 Vitesse (1,05 m) | 3,89% |
| SOAEG0 | Amateur Elite Grand Prix (1,25 m) | 2,95% |
| SOA2V | Amateur 2 Vitesse (1,10 m) | 2,90% |
| SOA3G | Amateur 3 Grand Prix (1,00 m) | 2,51% |

**Constat** : les 10 codes les plus courus représentent ~53% du volume. La queue longue des 1 552 codes couvre les épreuves rares (championnats, finales, formats spéciaux).

**Note** : l'extraction d'un "niveau standardisé" depuis ces variables fait l'objet du document [02_probleme_niveau_epreuve.md](02_probleme_niveau_epreuve.md).

### `DISCIPLINE_CODE`

| Valeur | Nb lignes | % |
|---|---|---|
| SO (Saut d'obstacles) | 7 735 863 | 95,80% |
| CE (Concours complet) | 339 607 | 4,20% |

**Uniquement 2 disciplines** dans les données, alors que la FFE recense plus de 30 disciplines. L'extraction a été volontairement limitée. Voir discussion dans [04 bis] (notes de chat).

### `DIVISION_LIB`

| Valeur | Nb lignes | % |
|---|---|---|
| Amateur | 4 531 711 | 56,1% |
| Pro | 1 801 159 | 22,3% |
| Elevage (jeunes chevaux) | 1 742 600 | 21,6% |

Répartition stable sur 16 ans. Voir [01_relation_cheval_cavalier.md](01_relation_cheval_cavalier.md) pour la dynamique de transition.

---

## 6. Résultats généraux

### `PLACE`

| Propriété | Valeur |
|---|---|
| Type | float64 (mais entiers uniquement, 0 décimal) |
| NaN | 2 286 (0,03%) |
| Range | 1 → 993 |
| Q1/médiane/Q3 | 6 / 18 / 37 |
| Moyenne / écart-type | 89,5 / 237 |

Classement du cheval dans l'épreuve. **Moyenne (89) très supérieure à la médiane (18)** -> distribution très asymétrique, avec une grosse queue aux places non qualifiantes (dernières, éliminés, abandons codés >100 ?).

Max = 993 : valeurs probablement administratives pour "non placé" ou "éliminé". À vérifier avant usage.

### `GAINS`

| Propriété | Valeur |
|---|---|
| Type | float64 (euros) |
| NaN | 641 (0,01%) |
| Range | 0 → 15 000 € |
| Q1/médiane/Q3 | 0 / 0 / 19 € |
| Moyenne | 31 € |
| **% = 0** | **69,2%** |

Prize money. **70% des participations rapportent 0 €** (classique : seules les premières places sont dotées). La dotation dépend fortement du niveau et du format de l'épreuve.

### `POINTS`

| Propriété | Valeur |
|---|---|
| Type | float64 |
| NaN | 82 595 (1,02%) |
| Range | **-376,5** → 9 999 |
| Q1/médiane/Q3 | 0 / 4 / 8 |
| Moyenne | 7,82 |
| % = 0 | 47,1% |
| % négatif | 0,02% |

**Valeurs négatives à investiguer** : probablement des pénalités pour chevaux non classés avec retrait de points. 9 999 est vraisemblablement un code administratif ("non pertinent" ou "invalide"), pas un vrai score.

---

## 7. Métriques Saut d'Obstacles

(Analyse sur les 7 735 863 lignes SO uniquement)

### `SO_POINTS_BAR` -- points de barrage

| Propriété | Valeur |
|---|---|
| NaN (sur SO) | 2 544 647 (32,9%) |
| Range | -28 → 9 999 |
| Q1/médiane/Q3 | 0 / 0 / 0 |
| **% = 0** | 75,6% |

### `SO_TEMPS` -- temps du parcours (secondes)

| Propriété | Valeur |
|---|---|
| NaN (sur SO) | 1 730 322 (22,4%) |
| Range | 0 → 1 000 |
| Q1/médiane/Q3 | 0 / 42,62 / 73,45 |
| % = 0 | 44,2% |

### `SO_TEMPS_BAR` -- temps de barrage (secondes)

| Propriété | Valeur |
|---|---|
| NaN (sur SO) | 2 561 588 (33,1%) |
| Range | 0 → 1 000 |
| Q1/médiane/Q3 | 0 / 0 / 38,05 |
| % = 0 | 52,1% |

**Interprétation des "manquants" massifs** :

Le manquant est **structurel** :
- `SO_POINTS_BAR` et `SO_TEMPS_BAR` : présents uniquement si l'épreuve a un barrage (jump-off pour départager les sans-fautes)
- `SO_TEMPS` : présent si épreuve chronométrée

Vérification empirique : sur les épreuves Grand Prix Amateur/Pro (codes `SOA*G*`, `SOP*G*`), `SO_POINTS_BAR` est rempli à **99,8-99,9%**. Donc le NaN = "pas de barrage prévu par le format", pas un problème de qualité.

Codes administratifs à surveiller : `1 000` dans `SO_TEMPS` / `SO_TEMPS_BAR` et `9 999` dans `SO_POINTS_BAR` semblent être des placeholders pour "non applicable", à traiter avant toute modélisation.

---

## 8. Métriques Concours Complet

(Analyse sur les 339 607 lignes CE uniquement)

### `CE_POINTSDRESSAGE`

| Propriété | Valeur |
|---|---|
| NaN (sur CE) | 26 (0,01%) |
| Range | -50 → 999 |
| Q1/médiane/Q3 | 32,8 / 43,4 / 52,5 |
| Moyenne | 40,04 |

Score de la reprise de dressage. Plage typique 0-100, les valeurs négatives et 999 sont à investiguer (probablement codes administratifs ou pénalités massives).

### `CE_POINTSFOND`

| Propriété | Valeur |
|---|---|
| NaN (sur CE) | 26 |
| Range | 0 → 156,4 |
| % = 0 | 83,3% |

Points sur le cross (pénalités de saut). 83% à zéro = la grande majorité des chevaux ne collecte aucune pénalité.

### `CE_POINTSSO`

| Propriété | Valeur |
|---|---|
| NaN (sur CE) | 26 |
| Range | 0 → 100 |
| % = 0 | 53,4% |

Points au saut en complet. 53% à zéro.

### `CE_POINTSFONDTEMPS`

| Propriété | Valeur |
|---|---|
| NaN (sur CE) | 26 |
| Range | -1,2 → 365 |
| % = 0 | 56% |

Temps au cross (pénalités de temps). Valeurs négatives étonnantes -- à investiguer.

### `CE_POINTSSOTEMPS`

| Propriété | Valeur |
|---|---|
| NaN (sur CE) | 26 |
| Range | 0 → 999,99 |
| % = 0 | 89,1% |

Temps au saut en complet. Très souvent à zéro. Le 999,99 est probablement un code "non applicable".

---

## Points d'attention pour la suite

### Anomalies à investiguer

1. **5 NaN sur DATENAISSANCE** : chevaux sans année de naissance -> à exclure ou imputer
2. **Valeurs 9 999 dans POINTS et SO_POINTS_BAR** : codes administratifs à remplacer par NaN avant modélisation
3. **Valeur 1 000 dans SO_TEMPS / SO_TEMPS_BAR** : idem, probablement un placeholder
4. **Valeur 999 dans CE_POINTSDRESSAGE** : idem
5. **Valeur 999,99 dans CE_POINTSSOTEMPS** : idem
6. **Valeurs négatives dans POINTS, SO_POINTS_BAR, CE_POINTSDRESSAGE, CE_POINTSFONDTEMPS** : à comprendre (pénalités, corrections administratives ?)
7. **12 libellés avec 2 codes différents** : sources d'incohérence potentielles

### Variables redondantes ou à simplifier

- `NUMERO_EVENEMENT2` et `DESIGNATION` : 1:1 au niveau métier (chaque événement a une seule désignation)
- `CLASSEEPREUVE_CODE` et `CLASSEEPREUVE_LIB` : quasi 1:1 (à 0,8% près)
- `DISCIPLINE_CODE` : informative mais seulement 2 valeurs (SO et CE)

### Qualité globale

- **Identification et pedigree : excellente** (>96% au niveau cheval, 0 incohérence intra-cheval)
- **Résultats généraux : très bonne** (0-1% manquant sur PLACE/GAINS/POINTS)
- **Métriques SO/CE : manquants structurels** (30-96%), correspondent aux lignes où la métrique ne s'applique pas par nature de l'épreuve

### Limites conceptuelles

- **2 disciplines uniquement** : pas de dressage pur, pas d'attelage, pas d'endurance
- **Hauteur d'obstacle absente dans 23,4% des libellés** (cycles jeunes chevaux SHF et CCE) -- voir [03_probleme_hauteur_reconstruite.md](03_probleme_hauteur_reconstruite.md). Couverture mise à jour le 2026-04-28 après correction de la regex d'extraction (passage de 71,8% à 76,6% de complétude). La regex initiale ne capturait pas les hauteurs au format "1 m" (sans décimale, ex: Amateur 3 Grand Prix). 312 codes d'épreuves supplémentaires ont leur hauteur extractible avec la regex corrigée.
- **Niveau d'épreuve non standardisé** dans une échelle ordinale -- voir [02_probleme_niveau_epreuve.md](02_probleme_niveau_epreuve.md)
