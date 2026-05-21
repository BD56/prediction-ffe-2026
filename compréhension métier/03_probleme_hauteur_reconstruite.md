# Problème : reconstruction de la HAUTEUR pour les épreuves sans hauteur explicite

## Objectif

Pour chaque ligne du dataset, avoir une **hauteur d'obstacles fiable**, qu'elle soit :
- Extraite directement du libellé (cas facile)
- Reconstruite à partir du règlement officiel (cas des cycles jeunes chevaux et CE)

## Situation actuelle

L'extraction par regex sur `CLASSEEPREUVE_LIB` avec le motif `r'(\d,\d{2})\s*m'` donne :

- **72% des lignes** : hauteur extractible avec fiabilité 100%
- **28% des lignes** : hauteur absente du libellé → actuellement NaN

Répartition du 28% non extractible (source : analyse missingness 2026-04-20) :

| Discipline × Division | % hauteur manquante |
|---|---|
| SO Pro | 1,0% (négligeable) |
| SO Amateur | 10,6% |
| **SO Elevage** (cycles jeunes) | **87,9%** |
| CE (toutes divisions) | 100% |

L'essentiel du trou provient donc :
1. Des **cycles SHF jeunes chevaux** (4 à 7 ans)
2. Du **concours complet** (CE) qui n'utilise pas "hauteur" comme métrique principale

## Pourquoi la hauteur n'est pas dans le libellé

### Cas 1 : Cycles jeunes chevaux SHF

La hauteur n'est pas écrite dans le libellé parce qu'elle est **standardisée par le règlement SHF** selon l'âge du cheval et l'étape du cycle. Le libellé contient le code du cycle (ex: "Cycle Libre 1ère année 4 ans") et la hauteur est implicite.

Hauteurs approximatives connues (à confirmer par règlement officiel) :
- Cycle Libre 1ère année (4 ans) : ~0,90-1,00m
- Cycle Libre 2ème année (5 ans) : ~1,00-1,10m
- Cycle Libre 3ème année (6 ans) : ~1,10-1,15m
- Cycle Classique 4 ans : ~1,00-1,05m
- Cycle Classique 5 ans : ~1,05-1,15m
- Cycle Classique 6 ans : ~1,15-1,25m
- Cycle Classique 7 ans : ~1,20-1,30m

**Complication : les étapes de qualification vs finales**
- Étape 1 / Étape 2 / Qualification / Finale n'ont pas toujours la même hauteur
- Les épreuves "label SF+AA" peuvent avoir des critères différents des autres

### Cas 2 : Concours Complet (CE)

Le CCE utilise des métriques à base de **points** (dressage, cross, SO) plutôt que de hauteur. Dans les libellés :
- "Amateur 2" n'a pas de hauteur mentionnée
- "CE Pro Elite" n'a pas de hauteur mentionnée
- Les hauteurs existent mais sont définies par la catégorie officielle (CCI*, CCI**, etc.)

Une reconstruction par équivalence est possible mais demande le règlement CCE.

## Défis techniques

### 1. Multiplicité des régimes réglementaires

Les règles peuvent varier selon :
- L'année (le règlement SHF évolue)
- La région / ligue
- Le type de concours (national vs régional)
- Le label attribué au cheval (SF, AA, SF+AA, autres...)

### 2. Qualité de la reconstruction

Si on reconstruit une hauteur "standardisée" pour les cycles jeunes chevaux, on introduit :
- Une **valeur théorique** (du règlement)
- Qui peut différer de la hauteur **réellement sautée** (un organisateur peut moduler)
- Cette valeur est uniforme pour toutes les occurrences de l'épreuve → **zéro variance** sur ces lignes

Risque : créer une **fausse précision**. Le modèle apprendrait des patterns sur des hauteurs uniformes artificielles.

### 3. Cohérence avec les hauteurs extraites directement

Les hauteurs du libellé sont les hauteurs **officielles annoncées**. Mais :
- Elles ne varient pas sur 16 ans pour un même code ?
- Quid si un organisateur a monté les obstacles plus haut que le nominal (concours d'élite) ?
- Quid des barrages (souvent à +0,05 ou +0,10m) ?

Pour l'instant, on assume que la hauteur extraite = hauteur réelle. À valider.

## Ce qu'il faudrait pour résoudre proprement

### Ressources externes nécessaires

1. **Règlement SHF officiel** année par année (2010-2025)
2. **Grille des épreuves** avec hauteurs réglementaires par code
3. **Règlement CCE** pour équivalences hauteur en complet
4. **Historique des évolutions** de nomenclature

### Démarche proposée (à planifier)

1. **Identifier les codes d'épreuves à hauteur manquante** (~450 codes concernés d'après cartographie)
2. **Rechercher dans les règlements** la hauteur officielle de chaque code
3. **Construire une table de correspondance** `(code, année) → hauteur réglementaire`
4. **Fusionner** cette table avec le dataset pour remplir les NaN
5. **Flaguer** la source : `hauteur_source ∈ {'libelle', 'reglement', 'inconnue'}`

### Questions à trancher

1. **Quel compromis** entre complétude (reconstruire partout) et fidélité (ne reconstruire que là où on a une source fiable) ?
2. **Comment gérer les cas où la hauteur officielle varie** dans l'année (qualifs vs finales) ? Moyenne ? Valeur par étape ?
3. **Que faire du CCE ?** Peut-on établir une équivalence hauteur par catégorie CE → SO, ou vaut-il mieux exclure la hauteur pour le CCE ?
4. **Si la reconstruction est partielle**, faut-il maintenir `HAUTEUR` brute et ajouter `HAUTEUR_ENRICHIE` en parallèle, ou fusionner les deux dans une seule colonne ?

## Décision pour le master dataset

En attendant la résolution de ces questions :
- La colonne `HAUTEUR` reste l'extraction par regex depuis le libellé (fiable à 100% où présente, NaN sinon)
- Pas de reconstruction depuis le règlement pour l'instant
- Les 28% de NaN sont à traiter ultérieurement

## Prochaines étapes (à planifier)

1. Vérifier si la grille SHF officielle est disponible publiquement (site FFE, archives SHF)
2. Lister les codes d'épreuves à hauteur manquante par fréquence décroissante
3. Commencer la construction de la table sur les codes les plus fréquents (top 20 couvrent probablement 80% des lignes à reconstruire)
4. Documenter les hypothèses faites à chaque étape
5. Décider si `HAUTEUR_RECONSTRUITE` sera une colonne séparée ou si on enrichit `HAUTEUR` avec un flag de source
