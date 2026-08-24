# Prédiction du plafond sportif des chevaux de Saut d'Obstacles

Projet tuteuré, Master 1 Data Science et Modélisation Statistique, Université Bretagne Sud (Vannes), promotion 2025-2026, en partenariat avec la **Fédération Française d'Équitation**.

Ce dépôt contient le code source du pipeline de prédiction et la documentation méthodologique du projet. Les **données FFE elles-mêmes ne sont pas publiées** pour des raisons de confidentialité. Un dataset synthétique de 20 chevaux fictifs est fourni pour valider l'exécutabilité du code.

---

## Objectif

Prédire la **hauteur maximale validée** (plafond sportif, en mètres) qu'un cheval atteindra au cours de sa carrière en Saut d'Obstacles, à partir des seules données de ses années 4 à 7 ans. La cible `hauteur_max_validée` est définie comme la hauteur maximale atteinte au moins trois fois sur la carrière complète.

Cible métier : aider l'éleveur, l'acheteur et le cavalier propriétaire à prendre des décisions d'orientation (cycle classique vs amateur, calibrage des prix, choix d'engagement) sur un signal statistique solide.

---

## Architecture du modèle

Le modèle final est un **Hurdle Random Forest** (Cragg 1971, Mullahy 1998) couplé à une couche **Locally Adaptive Conformal Prediction** (Vovk et al. 2005 ; Angelopoulos & Bates 2023) qui produit un intervalle de confiance à 95 % calibré.

```
ŷ_Hurdle(x) = P(top | x) · ŷ_tops(x) + (1 − P(top | x)) · ŷ_default(x)

IC_95%(x)   = ŷ_Hurdle(x) ± q_norm · σ(x)
```

- **Classifieur** : Random Forest balanced (500 arbres, max_depth=15), prédit la probabilité que le cheval dépasse 1,40 m.
- **Régresseur tops** : Random Forest entraîné sur les chevaux ≥ 1,40 m exclusivement (n ≈ 3 600).
- **Régresseur default** : Random Forest entraîné sur l'intégralité de la cohorte (n ≈ 47 600).
- **σ(x)** : écart-type des prédictions entre les 500 arbres, sert à la fois de mesure d'incertitude locale et de score pour l'IC adaptatif.

**Performance globale** (test set n = 5 045) :
- MAE globale (Hurdle) : **6,89 cm**
- MAE par tranche : 5,27 cm sur 1,35-1,40 m, **9,46 cm sur ≥ 1,45 m** (le Hurdle est le meilleur modèle testé sur les tops, gain de 3-4 cm face aux baselines)
- Couverture empirique de l'IC à 95 % : **94,7 %**
- Largeur de l'IC : moyenne 36 cm, adaptative entre 12 cm (cas faciles) et 58 cm (cas incertains)

---

## Structure du dépôt

```
.
├── scripts/                     Pipeline complet (54 scripts numérotés)
│   ├── 01_build_famille1.py     Construction des familles de features
│   ├── ...
│   ├── 27_hurdle_model.py       Modèle Hurdle de référence
│   ├── 50_conformal_prediction.py    Couche conformelle
│   ├── 51_locally_adaptive_conformal.py
│   ├── fiche.py                 CLI, génère une fiche pour un cheval donné
│   ├── fiche_app.py             Application GUI (tkinter)
│   └── generate_synthetic_data.py    Génère des chevaux fictifs
├── data/master/
│   ├── master_dataset_synthetic.parquet    20 chevaux fictifs (publiés)
│   └── figures_rapport/         Figures et fiches du rapport (5 fiches d'exemple)
├── compréhension métier/        Documentation méthodologique
│   ├── 06_features_engineering_plan.md
│   ├── 09_catalogue_features.md
│   ├── 12_sources_litterature.md
│   ├── 13_limites_methodologiques.md
│   ├── 14_validation_modele.md
│   └── 17 à 20 annexe_*.md      Annexes du rapport final
├── Fiche FFE.command            Lanceur macOS (dialogue natif)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone <url-du-depot>
cd "Projet tuteuré"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Utilisation

### Mode 1 : génération d'une fiche pour un cheval

Le script `fiche.py` prend un numéro SIRE en argument, entraîne le modèle au premier appel (~ 15 s, cache sur disque ensuite), et produit une fiche PNG dans `data/master/figures_rapport/fiche_<SIRE>.png`.

```bash
python3 scripts/fiche.py 47237708P
```

Sur un dépôt fraîchement cloné, les seuls SIRE disponibles sont ceux du **dataset synthétique** (20 chevaux fictifs générés par `generate_synthetic_data.py`). Pour les lister :

```bash
python3 scripts/fiche.py --list-test 20
```

Les 5 SIRE des fiches d'exemple publiées (47237708P, 13417805D, 13414483P, 60049124M, 47261754C) ne sont **pas exploitables** sur le dépôt cloné : leurs features n'existent pas dans le dataset synthétique. Les fiches PNG sont conservées comme illustrations du rapport.

### Mode 2 : application graphique (macOS)

Double-cliquer sur `Fiche FFE.command` ouvre un dialogue système qui demande un numéro SIRE, génère la fiche et l'ouvre dans Aperçu. Boucle jusqu'à annulation.

### Mode 3 : reproductibilité partielle

Le pipeline est découpé en deux moitiés.

**Scripts qui fonctionnent sur le dataset synthétique** (entraînement, prédiction et validation à partir du master déjà construit) :

```bash
python3 scripts/27_hurdle_model.py             # entraînement Hurdle
python3 scripts/47_validation_splits.py        # validation multi-splits
python3 scripts/51_locally_adaptive_conformal.py    # calibration IC
python3 scripts/fiche.py 3496834K              # fiche d'un cheval (synthétique)
```

Les résultats numériques **différeront** de ceux du rapport (20 chevaux fictifs ne reproduisent pas la statistique de 47 617 chevaux réels), mais l'exécution doit aboutir. `fiche.py` bascule automatiquement sur le dataset synthétique si `master_dataset_epure_v2.parquet` est absent.

**Scripts qui exigent les données FFE originales** (construction du master à partir des CSV bruts par discipline et année) :

```
scripts/00_compute_cible.py          # nécessite ffe_2010-2025_enriched.parquet
scripts/01_build_famille1.py à 10    # construction des 7 familles de features
scripts/20_correct_target_encoding.py
scripts/99_merge_master.py
scripts/prepare_enriched.py          # construction du enriched depuis CSV bruts
```

Ces scripts ne peuvent pas être exécutés sans accord FFE. Leur logique reste lisible (commentée) et constitue la documentation principale de la chaîne d'engineering.

---

## Données

### Périmètre métier (rapport)

- Cohorte T1 + N1 ≥ 10 : chevaux nés 2006-2013, discipline Saut d'Obstacles, hors poneys, ayant au moins 10 participations totales.
- Effectif final : **47 617 chevaux**.
- 156 features (sur 273 candidates) après épuration par corrélation puis par importance.

### Schéma du master dataset

Le fichier `data/master/master_dataset_synthetic.parquet` contient un échantillon respectant exactement le schéma du dataset réel : 162 colonnes (1 IDCHEVAL, 1 SPLIT, 1 hauteur_max_validee, 158 features candidates, dont 156 utilisées par le modèle), 20 lignes fictives. Pour l'utiliser avec un dataset réel, il suffit de remplacer ce fichier par `master_dataset_epure_v2.parquet` au même emplacement.

### Confidentialité

Les données originelles FFE sont **soumises à un accord de confidentialité** et ne sont pas redistribuées. L'accès au dataset réel pour rejouer les résultats publiés du rapport doit être obtenu directement auprès de la FFE.

---

## Documentation méthodologique

| Fichier | Contenu |
|---|---|
| `06_features_engineering_plan.md` | Plan général d'engineering des familles de features |
| `09_catalogue_features.md` | Catalogue exhaustif des 273 features candidates avec statuts |
| `12_sources_litterature.md` | Bibliographie complète |
| `13_limites_methodologiques.md` | Discussion transparente des limites du modèle |
| `14_validation_modele.md` | Protocole de validation multi-splits (A, B, C) |
| `17_annexe_fiches_rapport.md` | Anatomie d'une fiche de prédiction (annexe B du rapport) |
| `18_annexe_applications_rapport.md` | Cadre d'usage opérationnel (annexe C) |
| `19_annexe_glossaire_rapport.md` | Glossaire métier (annexe D) |
| `20_annexe_features_rapport.md` | Récapitulatif des features (annexe E) |

---

## Auteurs et encadrement

**Bryan Desjardins** et **Julianne Festoc**, M1 Data Science et Modélisation Statistique, UBS Vannes (promo 2025-2026).
Projet réalisé en partenariat avec la Fédération Française d'Équitation.

---

## Licence

Code sous licence **MIT** (voir le fichier [`LICENSE`](LICENSE)) : libre réutilisation académique et commerciale avec attribution.

Les données FFE et les résultats numériques publiés dans le rapport ne sont **pas redistribuables** sans accord FFE explicite : voir [`DATA_LICENSE.md`](DATA_LICENSE.md).
