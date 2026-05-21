# Plan d'ingénierie des features (master dataset)

Document de cadrage pour l'étape de feature engineering. Liste les familles de features à creuser, les analyses de leakage à respecter, et le plan d'attaque.

---

## Contexte

**Cible** : `hauteur_max_validée` (continue, [0,95m - 1,55m]) -- max hauteur où le cheval a participé ≥ 3 fois sur sa carrière entière.

**Cohorte d'entraînement** : T1 = né 2006-2013, hors poney, SO uniquement, ≥ 10 participations totales. **47 683 chevaux** avec cible calculable.

**Fenêtre temporelle des features** : 4 à 7 ans (cohérent avec les cycles SHF jeunes chevaux).

**Objectif** : prédire la cible à partir de variables observables pendant la fenêtre 4-7 ans.

---

## Les 11 familles candidates -- décisions

### Familles retenues (9)

| # | Famille | Justification | Risque principal |
|---|---|---|---|
| 1 | **Activité / Volume** | Universel dans la littérature et l'industrie | Aucun majeur |
| 2 | **Performance financière (gains)** | Phénotype IFCE de référence (`log(GAINS+1)`), Horsetelex ISV, Hippomundo | Distribution très asymétrique (70% à 0€) -> à transformer |
| 3 | **Performance sportive (placement)** | Universel (place, % top 3/10, % sans-faute) | PLACE comparable seulement après normalisation par taille de peloton |
| 4 | **Hauteurs explorées** | Chapard 2023 ("hauteur d'obstacle ajustée") | **Leakage si on utilise `hauteur_max`** (tautologique avec la cible) |
| 5 | **Niveau / type d'épreuves** | Catégoriel utile (division, code, % par cycle) | Redondance possible avec hauteurs |
| 6 | **Progression temporelle** | Cohérent avec le sujet "parcours de jeunesse" | La progression de hauteur peut leakager |
| 7 | **Cavalier** | Standard académique (Chapard, Sanchez-Guerrero, Ricard) | **Leakage si mal construit** (cf. erreur PERE_ELITE de la binôme) |
| 8 | **Pedigree / génétique** | Standard BLUP universel | **Leakage massif si mal construit** (calculer en excluant le cheval) |
| 10 | **Race** | Standard, race uniquement (sexe non disponible) | Aucun majeur |

### Famille exclue (1)

| # | Famille | Raison de l'exclusion |
|---|---|---|
| 9 | **Saisonnalité** | Marginal pour notre problème prédictif. Effet capté par d'autres variables si pertinent. |

### Famille à voir (1)

| # | Famille | Statut |
|---|---|---|
| 11 | **Géographique** | À inclure si temps. Nécessite extraction depuis DESIGNATION ou codes événement. Non bloquant. |

---

## Analyse du data leakage par famille

Cette section synthétise les risques identifiés et les contre-mesures.

### Sur la cohorte de modélisation (47 683 chevaux T1+N1≥10)

**Âge de validation de la cible** :
- Médiane = 8 ans
- 36,7% valident dans la fenêtre 4-7 ans
- 63,3% valident après 7 ans

**Insight crucial** : les chevaux qui valident TÔT (4-7 ans) sont essentiellement de **niveau modeste** (≤1.15m).

| Âge validation | ≤1.15m | 1.20-1.30m | 1.35-1.40m | ≥1.45m |
|---|---|---|---|---|
| 4-5 ans | 98% | 2% | 0% | 0% |
| 6 ans | 74% | 26% | 0% | 0% |
| 7 ans | 38% | 46% | 16% | 0% |
| 8 ans | 34% | 40% | 23% | 3% |
| 9-10 ans | 34% | 38% | 20% | 8% |

**Pour les chevaux "tops" (hauteur_max_validée ≥ 1.40m, n=4 888)** :
- 0% valident à 4-5 ans
- 0% à 6 ans (avec hauteur ≥ 1.40m)
- 10,5% valident en 4-7 ans (essentiellement à 7 ans, 1.35-1.40m)
- 89,5% valident après 7 ans

### Conséquences pratiques

**Features à éviter (tautologie totale)** :
- ❌ `hauteur_max_observée_4_7_ans` -- pour 36,7% des chevaux, égale à la cible

**Features avec leakage partiel acceptable** :
- ⚠️ `hauteur_médiane_4_7_ans` -- corrélée mais pas tautologique
- ⚠️ `progression_hauteur_4_7` -- corrélée mais informative

**Features sans leakage (à utiliser librement)** :
- ✅ Volume : nb participations, nb événements
- ✅ Gains : log(gains+1), distribution
- ✅ Placement : place médiane, % top X
- ✅ Niveau d'épreuve : division, code (catégorielles)
- ✅ Cavalier (avec construction rigoureuse)
- ✅ Pedigree (avec construction rigoureuse)
- ✅ Race

---

## Pièges spécifiques à éviter (leçons de la binôme)

### Erreur PERE_ELITE de la binôme

La binôme a calculé : "le père est élite si le taux de ses descendants en Pro est supérieur à la moyenne globale". Le piège : **le cheval lui-même fait partie des descendants utilisés pour calculer le taux du père**, et son statut Pro alimente la cible.

Conséquence : data leakage massif, le coefficient PERE_ELITE était artificiellement élevé.

### Solutions méthodologiques

Pour les features pedigree :
- **Solution A** : calculer le score du père sur ses descendants nés AVANT le cheval analysé
- **Solution B** : pour chaque cheval, exclure ce cheval du calcul du score de son père
- **Solution C** : calculer le score sur le train set uniquement, puis appliquer au test set

Pour les features cavalier :
- Calculer le niveau du cavalier sur les autres chevaux qu'il a montés
- Ou utiliser un effet aléatoire dans un modèle mixte (approche académique standard)

---

## Plan d'attaque proposé

Approche progressive, famille par famille, du plus simple au plus subtil.

### Phase 1 -- Familles simples (sans leakage)

1. **Activité / Volume** -- compter, c'est tout
2. **Performance financière** -- agréger les gains
3. **Performance sportive** -- agréger les places
4. **Niveau d'épreuves** -- proportions par division, par cycle
5. **Race** -- variable catégorielle

À ce stade : **5 familles, ~15-20 features, modèle baseline possible**.

### Phase 2 -- Familles avec précautions

6. **Hauteurs explorées** -- éviter `hauteur_max`, utiliser `hauteur_médiane`, distribution, etc.
7. **Progression temporelle** -- delta gains, delta hauteur (avec précaution)
8. **Cavalier** -- avec calcul rigoureux du niveau cavalier
9. **Pedigree** -- avec correction du leakage potentiel

À ce stade : **9 familles, ~25-35 features, modèle plus riche**.

### Phase 3 -- Si temps

10. **Géographique** -- extraction depuis DESIGNATION
11. (Saisonnalité écartée)

---

## Décisions ouvertes pour la suite

🔴 **À discuter dans la prochaine étape** :
- Quel transformation pour les gains (log+1, racine, percentiles) ?
- Quel encodage pour les variables catégorielles (one-hot, target encoding, frequency encoding) ?
- Comment gérer les valeurs manquantes (imputation, indicateurs missing, modèles tolérants comme XGBoost) ?
- Faut-il une étape de réduction dimensionnelle (PCA) ou laisser le modèle gérer ?

🔴 **À surveiller dans le modèle** :
- Performance séparée sur les early peakers (37%) vs late peakers (63%)
- Importance des features liées à la hauteur (faut-il les retirer pour test ?)
- Comparer avec un modèle baseline qui n'utilise pas du tout la hauteur

---

## Annexe : sources d'inspiration

| Acteur | Famille principale utilisée |
|---|---|
| EquiRatings | Activité, placement (clear rounds), Elo factor |
| JPR (USA) | Volume, placement, gains, consistance |
| Hippomundo Rating | Niveau d'épreuve, plateau, fréquence sans-faute |
| Horsetelex ISV | Gains / nb départs |
| IFCE / Anne Ricard | log(GAINS+1) standardisé, effet cavalier random |
| Chapard 2023 | Hauteur d'obstacle ajustée (combine niveau + classement) |
| Chapard 2024 | Traits de jeunesse → performance future (rg=0.30-0.77) |
| Ricard & Blouin 2011 | Survival analysis sur longévité avec features démographiques |

---

**Document créé le 2026-05-02 -- à mettre à jour à chaque itération de feature engineering**
