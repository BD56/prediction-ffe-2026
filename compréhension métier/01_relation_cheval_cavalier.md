# Relation cheval / cavalier dans les données FFE

Analyse menée sur l'ensemble des données 2010-2025 (8 075 470 lignes).

---

## La variable LICENCE

- **Format** : identifiant alphanumérique à 8 caractères (ex: `0932858P`, `2064654P`)
- **Anonyme** mais **stable** dans le temps : un même cavalier garde la même licence d'une année à l'autre
- **100% renseignée** (pas de valeurs manquantes)
- ~29 000 cavaliers uniques par année

---

## Question 1 : Combien de cavaliers pour un cheval dans l'année ?

### Vue globale

| Situation | % des cheval-années |
|---|---|
| 1 cavalier | **74,7%** |
| 2 cavaliers | 20,5% |
| 3 cavaliers | 3,9% |
| 4+ cavaliers | 1,0% |

Médiane = 1, moyenne = 1,31, max = 13

### Ventilation par division (division max atteinte par le cheval dans l'année)

| Division | 1 cavalier | 2 cavaliers | 3+ cavaliers | Moyenne |
|---|---|---|---|---|
| Élevage (jeunes, 137 210 cheval-années) | 80,6% | 16,7% | 2,7% | 1,23 |
| Amateur (449 191 cheval-années) | 77,7% | 18,4% | 3,9% | 1,27 |
| **Pro** (180 892 cheval-années) | **62,5%** | **28,8%** | **8,6%** | **1,49** |

**Constat** : les chevaux Pro changent significativement plus de cavalier que les chevaux Amateur ou Élevage. 37,5% des cheval-années Pro ont 2+ cavaliers, contre ~22% chez les autres divisions.

---

## Question 2 : Combien de chevaux pour un cavalier dans l'année ?

### Vue globale

| Situation | % des cavalier-années |
|---|---|
| 1 cheval | **61,9%** |
| 2 chevaux | 18,4% |
| 3-5 chevaux | 12,8% |
| 6-10 chevaux | 4,3% |
| 11+ chevaux | 2,6% |

Médiane = 1, moyenne = 2,23, max = 77

### Ventilation par division max du cavalier

| Division max du cavalier | 1 cheval | 2 chevaux | 3-5 | 6-10 | 11+ | Médiane |
|---|---|---|---|---|---|---|
| Élevage (19 019 cav-années) | 74,5% | 13,5% | 8,9% | 2,4% | 0,6% | 1 |
| Amateur (362 616 cav-années) | 69,6% | 18,8% | 9,9% | 1,5% | 0,3% | 1 |
| **Pro** (71 348 cav-années) | **19,6%** | 17,5% | **29,0%** | **18,7%** | **15,2%** | **4** |

**Constat** : différence spectaculaire entre Pro et Amateur.
- Cavalier Amateur : médiane = 1 cheval/an
- **Cavalier Pro : médiane = 4 chevaux/an** (moyenne 5,75, max 77)

Les Pros sont des "professionnels du pilotage" qui montent majoritairement les chevaux des autres. Les Amateurs montent généralement leur propre cheval.

---

## Question 3 : Carrière complète d'un cheval (2010-2025)

Sur la durée de vie totale d'un cheval dans les données (198 659 chevaux uniques).

### Vue globale

| Nb de cavaliers différents | % des chevaux |
|---|---|
| 1 cavalier | 36,7% |
| 2 cavaliers | 23,6% |
| 3-5 cavaliers | 30,4% |
| 6+ cavaliers | 9,3% |

- Médiane de cavaliers = 2, moyenne = 2,66, max = 39
- Durée de carrière médiane = 3 ans actifs (moyenne 3,9, max 16)

### Ventilation par division max atteinte dans la carrière

| Chevaux (div. max) | 1 cav | 2 cav | 3-5 cav | 6-10 cav | 11+ cav | Médiane cav | Carrière médiane |
|---|---|---|---|---|---|---|---|
| Élevage (31 001) | **67,5%** | 23,3% | 9,0% | 0,2% | 0% | 1 | **1 an** |
| Amateur (104 146) | 40,3% | 26,1% | 28,4% | 4,9% | 0,3% | 2 | 3 ans |
| **Pro** (63 512) | 15,9% | 19,7% | **43,9%** | **18,8%** | 1,7% | **3** | **5 ans** |

**Constats** :
- Un cheval qui atteint le Pro a traversé la main de **3 cavaliers en médiane** (jusqu'à 29 au maximum)
- Un cheval qui s'arrête en Élevage n'a connu qu'un seul cavalier dans 67,5% des cas et ~1 an d'activité
- Les chevaux Pro ont des carrières **5× plus longues** que ceux qui restent en Élevage

---

## Question 4 : Carrière complète d'un cavalier (2010-2025)

Sur la durée de vie totale d'un cavalier dans les données (91 406 cavaliers uniques).

### Vue globale

| Nb chevaux en carrière | % cavaliers |
|---|---|
| 1 cheval | **38,6%** |
| 2 chevaux | 19,0% |
| 3-5 chevaux | 22,6% |
| 6-10 chevaux | 10,0% |
| 11-25 chevaux | 6,0% |
| 26+ chevaux | 3,9% |

- Médiane = 2 chevaux, moyenne = 5,79, max = 539
- Durée de carrière médiane = 4 ans (moyenne 5,0, max 16)

Distribution très asymétrique : beaucoup de cavaliers "occasionnels" (1 cheval, carrière courte), quelques cavaliers "ultra-actifs" tirent la moyenne.

### Ventilation par division max atteinte par le cavalier

| Cavaliers (div. max) | 1 cheval | 2 chevaux | 3-5 | 6-10 | 11-25 | 26+ | Médiane chx | Carrière médiane |
|---|---|---|---|---|---|---|---|---|
| Élevage (2 225) | 72,4% | 16,0% | 8,1% | 2,4% | 0,9% | 0% | 1 | 1 an |
| Amateur (73 777) | 44,5% | 21,5% | 23,3% | 7,9% | 2,5% | 0,3% | 2 | 3 ans |
| **Pro** (15 404) | **5,5%** | 7,8% | 21,3% | **20,8%** | **23,2%** | **21,4%** | **9** | **10 ans** |

**C'est le tableau le plus parlant de l'analyse** :
- Cavalier Amateur : médiane 2 chevaux en carrière, 3 ans d'activité
- **Cavalier Pro : médiane 9 chevaux, 10 ans d'activité, certains jusqu'à 539 chevaux montés**
- 21,4% des Pros ont monté **26 chevaux ou plus** dans leur carrière

Les cavaliers Pro sont des "experts multi-chevaux sur le long terme" -- source majeure d'information pour démixer les effets cheval/cavalier (énormément de variation intra-cavalier observable).

---

## Bonus : mobilité entre divisions

### Cavaliers qui traversent plusieurs divisions

**34,7% des cavaliers** (31 696 sur 91 406) évoluent entre divisions :

| Transition | Nb cavaliers | % des mobiles |
|---|---|---|
| Élevage -> Amateur | 16 682 | 52,6% |
| Élevage -> Pro | 10 055 | 31,7% |
| **Amateur -> Pro** | **4 959** | **15,6%** |

### Chevaux qui traversent plusieurs divisions

**50,4% des chevaux** (100 073 sur 198 659) évoluent entre divisions :

| Transition | Nb chevaux | % des mobiles |
|---|---|---|
| Élevage -> Amateur | 41 680 | 41,6% |
| Élevage -> Pro | 39 093 | 39,1% |
| **Amateur -> Pro** | **19 300** | **19,3%** |

### Nuance méthodologique

La transition "Élevage -> Amateur/Pro" n'est pas une vraie promotion : c'est la **trajectoire normale** d'un jeune cheval qui mûrit (Cycles Classique/Libre 4-6 ans -> compétition adulte).

En revanche, **Amateur -> Pro** (19% des chevaux mobiles, 16% des cavaliers mobiles) est une **vraie ascension sportive**, particulièrement intéressante comme variable cible binaire.

---

## Enseignements pour la modélisation

### 1. L'effet cavalier est massif et dépend fortement du niveau

- **Chevaux Amateur** : effet cavalier simple à modéliser (1-2 cavaliers, peu de variation intra-cheval)
- **Chevaux Pro** : modélisation complexe mais très informative (beaucoup de variation intra-cheval pour isoler la part "cheval")

### 2. Les cavaliers Pro = experts multi-chevaux sur le long terme

- 9 chevaux en médiane sur la carrière, 10 ans d'activité, jusqu'à 539 chevaux montés
- Score cavalier très fiable à construire à partir de leurs résultats
- Leur présence sur un cheval = signal fort (cheval commercial ou talent identifié)

### 3. Fort déséquilibre de volume de données par niveau

- 67,5% des chevaux Élevage sont vus 1 an avec 1 cavalier -> peu d'information disponible
- Les chevaux Pro ont une trajectoire riche (3-5 cavaliers, 5 ans) -> signal fort mais post-hoc
- Design du modèle doit composer avec cette asymétrie

### 4. L'historique cavalier = feature prédictive à construire

- Cheval monté jeune par un cavalier Pro -> probable prospect élite
- Changements fréquents -> potentiel instable ou cheval de marchand
- Transition Amateur -> Pro (19% des chevaux mobiles) = trajectoire d'ascension pure

### 5. Variable cible candidate : la transition Amateur -> Pro

- 19 300 chevaux ont effectué cette transition sur 2010-2025
- Phénotype binaire clair, aligné avec la notion de "haut niveau"
- Moins bruité que les gains ou les classements bruts

### 6. Limite naturelle pour la démixtion cheval/cavalier

- 37% des chevaux n'ont qu'un cavalier dans leur vie : séparation "effet cheval" vs "effet cavalier" impossible pour eux
- 63% restants (2+ cavaliers) : démixtion possible, plus fiable chez les Pros

### 7. Pièges à éviter

- Ignorer l'effet cavalier -> perte d'une information essentielle
- Ne pas le modéliser -> biais systématique (un bon cavalier gonfle artificiellement la valeur perçue d'un cheval moyen)

---

## Conclusion

La démixtion cheval/cavalier est un pilier incontournable du modèle prédictif.

C'est l'angle central d'Équidata Sport (indice "EquiDT HI®" -- Horse-Rider Differential Index) et un standard dans la littérature académique récente (Chapard 2023, Sanchez-Guerrero 2024) où l'effet cavalier est traité comme effet aléatoire permanent dans les modèles mixtes.
