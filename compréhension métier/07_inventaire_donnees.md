# Inventaire des données du projet

État des fichiers de données disponibles dans le projet.

---

## Sources originales (FFE)

**Localisation** : `ExtraxtionEtudiants2010-2025/`

16 fichiers CSV (un par année de 2010 à 2025), fournis par la FFE :

| Fichier | Année | Lignes (approx) |
|---|---|---|
| `ExtraxtionEtudiants2010.csv` | 2010 | 602 528 |
| `ExtraxtionEtudiants2011.csv` | 2011 | 515 024 |
| ... | ... | ... |
| `ExtraxtionEtudiants2024.csv` | 2024 | 494 185 |
| `ExtraxtionEtudiants2025.csv` | 2025 | 494 165 |

**Total cumulé** : 8 075 470 lignes × 27 colonnes

**Granularité** : 1 ligne = 1 participation d'un cheval à une épreuve.

**Disciplines** : SO (Saut d'Obstacles, 95,8%) + CE (Concours Complet, 4,2%).

---

## Fichiers fusionnés (générés)

**Localisation** : `data/`

### Fichier brut (concaténation simple)

| Fichier | Format | Taille | Description |
|---|---|---|---|
| `ffe_2010-2025_raw.csv` | CSV | 1 725 MB | Concaténation des 16 années + colonne `ANNEE` |
| `ffe_2010-2025_raw.parquet` | Parquet | 201 MB | Idem, format binaire compressé |

**Structure** : 8 075 470 lignes × **28 colonnes** (27 originales + `ANNEE`)

### Fichier enrichi (raw + 6 colonnes dérivées)

| Fichier | Format | Taille | Description |
|---|---|---|---|
| `ffe_2010-2025_enriched.csv` | CSV | 1 965 MB | Brut + 6 colonnes dérivées |
| `ffe_2010-2025_enriched.parquet` | Parquet | 207 MB | Idem, format binaire compressé |

**Structure** : 8 075 470 lignes × **34 colonnes** (28 du raw + 6 dérivées)

**Colonnes dérivées ajoutées** :

| Colonne | Type | Calcul | Couverture |
|---|---|---|---|
| `AGE` | int | `DATEEPREUVE.year - DATENAISSANCE` | 99,99% (5 NaN -- chevaux sans naissance) |
| `HAUTEUR` | float | Extraction regex V2 `(\d(?:[.,]\d{1,2})?)\s*m\b` + filtre [0,85 ; 1,60] | 76,6% |
| `est_poney_race` | bool | `RACECHEVAL` matche `PONEY\|PONY\|CONNEMARA\|...` | 3,4% True |
| `est_libelle_poney` | bool | `CLASSEEPREUVE_LIB` contient "poney" ou "pony" | 1,7% True |
| `in_cohorte_T1` | bool | né 2006-2013, hors poney, discipline SO | 45,9% True |
| `in_fenetre_4_7` | bool | AGE entre 4 et 7 ans (inclus) | 40,4% True |

---

## Fichiers de support

**Localisation** : `compréhension métier/data_quality/`

Outputs auto-générés et CSV de stats :

| Fichier | Contenu |
|---|---|
| `rapport_ydata_profiling.html` | Rapport ydata-profiling sur le dataset complet |
| `missingno_matrix.png` | Visualisation des manquants (échantillon) |
| `missingno_bar.png` | Bar chart des manquants par variable |
| `missingno_heatmap.png` | Corrélation des manquants entre variables |
| `missingno_dendrogram.png` | Clustering des variables manquantes |
| `missing_by_tier.csv` | Missingness par tier (T0/T1/T2/T3/T4) |
| `missing_by_year.csv` | Évolution annuelle de la missingness |
| `missing_by_discipline_division.csv` | Missingness par SO/CE × Amateur/Pro/Élevage |
| `missing_at_horse_level.csv` | Missingness au niveau cheval |
| `codes_sans_hauteur.csv` | Liste codes sans hauteur (toutes disciplines) |
| `codes_sans_hauteur_SO_horspony.csv` | Liste codes sans hauteur (SO uniquement, hors poney) |
| `borne_validation_results.csv` | Validation empirique de la borne 2013 |
| `test3_N1_N2_combinations.csv` | Test combinaisons N1 × N2 (V1) |
| `test3_N1_N2_combinations_v2.csv` | Idem (V2 corrigée) |

---

## Documents méthodologiques

**Localisation** : `compréhension métier/`

| Doc | Contenu |
|---|---|
| `00_journal_decisions.md` | Journal des décisions, alternatives explorées, parking lot |
| `01_relation_cheval_cavalier.md` | Analyse des interactions cheval/cavalier |
| `02_probleme_niveau_epreuve.md` | Cadrage du problème "niveau d'épreuve" (parqué) |
| `03_probleme_hauteur_reconstruite.md` | Cadrage du problème "hauteur manquante" (parqué) |
| `04_analyse_variables_raw.md` | Analyse exhaustive des 27 variables |
| `05_codes_sans_hauteur_SO.md` | Liste codes SO sans hauteur (filtres pony appliqués) |
| `06_features_engineering_plan.md` | Plan d'ingénierie des features |
| `07_inventaire_donnees.md` | (Ce document) |
| `08_mail_FFE_questions.md` | Mail aux questions à Emmanuel HUDE (FFE) |

---

## Comment charger les données

### En Python (Parquet, recommandé)

```python
import pandas as pd

# Dataset enrichi complet
df = pd.read_parquet("data/ffe_2010-2025_enriched.parquet")

# Filtrer la cohorte de modélisation
mask = df['in_cohorte_T1'] & ~df['est_libelle_poney']
df_t1 = df[mask]

# Filtrer la fenêtre features 4-7 ans
df_features = df_t1[df_t1['in_fenetre_4_7']]
```

### En Python (CSV, plus lent)

```python
df = pd.read_csv("data/ffe_2010-2025_enriched.csv", parse_dates=['DATEEPREUVE'])
```

### En R

```r
library(arrow)  # pour parquet
df <- read_parquet("data/ffe_2010-2025_enriched.parquet")

# OU
df <- read.csv("data/ffe_2010-2025_enriched.csv")
```

### Excel

Possible mais limite : le fichier CSV fait 2 GB. Excel charge mais peut être lent.
Mieux : ouvrir un sous-échantillon via Python/R d'abord.

---

## Évolution prévue

Au fur et à mesure du projet, ce dossier `data/` évoluera :

- **À court terme** : ajout du master dataset agrégé au niveau cheval (`ffe_master_dataset.parquet`) une fois le feature engineering terminé
- **À moyen terme** : ajout d'une version "panel longitudinal" (1 ligne par cheval-année) pour modèles temporels
- **Si reçu de la FFE** : ajout d'un fichier complémentaire avec la grille SHF des hauteurs réglementaires

---

**Document créé le 2026-05-02 -- à mettre à jour à chaque évolution majeure du dossier `data/`**
