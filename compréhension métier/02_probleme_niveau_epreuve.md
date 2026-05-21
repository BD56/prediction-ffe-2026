# Problème : extraction du NIVEAU d'une épreuve

## Objectif

Pour chaque ligne du dataset (= une participation à une épreuve), on voudrait associer un **niveau standardisé** qui permette de :

- Comparer la difficulté entre épreuves
- Construire une hiérarchie sportive utilisable comme feature ou cible
- Identifier les "vrais" hauts niveaux (Pro Elite, Grand Prix 1,50m...)

## Pourquoi c'est compliqué : les dimensions du "niveau"

Le concept de "niveau" est en réalité **multi-dimensionnel** dans la FFE, et les variables sources ne le donnent pas directement :

### 1. Division de licence (partiellement dans les données)

- Elevage / Amateur / Pro (variable `DIVISION_LIB`)
- **C'est lié au cavalier, pas au cheval** (cf. document cheval-cavalier)
- Une division haute n'implique pas forcément une difficulté d'épreuve supérieure (ex: Amateur Elite 1,25m = Pro 3 1,25m)

### 2. Niveau sportif dans la division (pas dans les données, à extraire)

Par exemple en SO Pro :
- Pro Elite (1,50m)
- Pro 1 (1,40m)
- Pro 2 (1,30m)
- Pro 3 (1,25m)
- Pro 4 (rare)

En SO Amateur :
- Amateur Elite (1,25m)
- Amateur 1 (1,15m)
- Amateur 2 (1,10m)
- Amateur 3 (0,95m)

### 3. Format de l'épreuve (pas dans les données explicitement)

- Grand Prix (épreuve principale)
- Spéciale au chrono / ss chrono
- Vitesse
- Préparatoire
- Style
- Maniabilité

Deux épreuves de même niveau peuvent avoir des formats différents et donc des "dotations" en points FFE différents.

### 4. Statut compétitif (pas dans les données)

- Épreuve ordinaire
- Championnat (départemental, régional, national, France)
- Finale (cycles jeunes chevaux, championnats)
- Critérium / Circuit des As / Trophée
- Coupe

Ces statuts valorisent l'épreuve au-delà de sa hauteur nominale.

### 5. Cycles jeunes chevaux (hors grille Pro/Amateur)

Les épreuves SHF pour chevaux de 4-7 ans ne rentrent pas dans la nomenclature Pro/Amateur :
- Cycle Classique 4/5/6/7 ans (hauteurs progressives 1,00 → 1,20m)
- Cycle Libre 1ère/2ème/3ème année (hauteurs 0,95 → 1,15m)
- Label SHF SF+AA (qualification jeunes chevaux)
- Finales SHF nationales

Ils ont leur propre hiérarchie, non directement comparable à Pro/Amateur.

### 6. Concours Complet (CE) sans hauteur dans le libellé

- Pas de hauteur d'obstacle dans les libellés CE
- Niveau défini par la qualification du cavalier + la classe de l'épreuve
- Grille parallèle à celle du SO

## Ambiguïtés recensées

Lors d'une tentative d'extraction par regex sur code + libellé (2025-04-20), les cas suivants ont posé problème :

1. **~1,3 million de lignes classées "Autre"** : majoritairement des codes de cycles jeunes chevaux (`SO5QS`, `SOCCF1`, `SOCL2`...) qui ne matchent pas la grille Pro/Amateur
2. **12 libellés ont plusieurs codes** différents (0,8%) : potentielles évolutions de nomenclature
3. **Championnats/Finales trans-divisions** : un "Championnat Cycle Classique 4 ans" est-il un niveau Elevage ou un niveau "Championnat" ?
4. **Variantes de format** : "Grand Prix 1,30m" vs "Spéciale au chrono 1,30m" : même hauteur, pas le même niveau de challenge
5. **Évolution dans le temps** : les règlements FFE/SHF évoluent tous les 3-5 ans, des codes peuvent changer de sens

## Ce qu'il faudrait pour résoudre proprement

### Ressources externes nécessaires

1. **Grille officielle des épreuves FFE/SHF** par année
2. **Table de correspondance code d'épreuve → (division, niveau, format, hauteur officielle)**
3. **Historique des changements de nomenclature** sur 2010-2025
4. **Règlements des championnats** (qui participe, selon quel critère)

### Validation nécessaire

1. Vérifier sur un échantillon que notre extraction correspond bien au niveau réel des épreuves
2. Quantifier le taux d'ambiguïté (lignes où la classification est incertaine)
3. Flager les cas ambigus plutôt que de forcer un classement

### Questions de conception à trancher

1. **Une seule variable "niveau" ou plusieurs** ?
   - Option A : un rang ordinal unique (ex: 1 à 20)
   - Option B : plusieurs variables (`division`, `niveau_dans_division`, `format`, `statut`)
2. **Comment placer les cycles jeunes chevaux** dans la hiérarchie ?
   - Option A : catégorie à part
   - Option B : équivalence par hauteur (cycle 6 ans 1,10m ≈ Amateur 2)
3. **Comment pondérer le statut compétitif** (championnat, finale) ?
4. **Que faire des 12 libellés à multi-codes ?** Unifier ou garder séparés ?

## Décision pour le master dataset

En attendant la résolution de ces questions :
- **NIVEAU n'est pas inclus comme colonne persistante** dans le master dataset
- Les variables sources `DIVISION_LIB`, `CLASSEEPREUVE_LIB`, `CLASSEEPREUVE_CODE` restent la référence
- Une classification approximative peut être générée à la volée pour exploration, avec un flag explicite "classification non validée"

## Prochaines étapes (à planifier)

1. Rechercher la grille officielle FFE/SHF (règlement sport)
2. Construire une table de correspondance validée pour les codes les plus fréquents (top 50 couvrent probablement 95% des lignes)
3. Documenter les cas ambigus dans un fichier séparé
4. Décider de la représentation finale (ordinale, catégorielle ou multi-variable)
