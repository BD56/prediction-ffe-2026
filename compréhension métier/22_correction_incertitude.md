# Correction de l'estimation d'incertitude

**Branche `correction-incertitude`, août 2026.** Travail postérieur au rendu, mené par
Bryan Desjardins. Rien de ce qui a été remis n'est modifié : `main` est intacte, l'ancien
calibrage reste accessible via `fiche.py --ancien`.

---

## Le problème

Le score de non-conformité du modèle est `|y − ŷ| / σ`, où σ est l'écart type des
prédictions entre les 500 arbres. Ce score suppose que **l'erreur est proportionnelle à σ**.

Mesuré sur le jeu de validation et sur le jeu de test :

- la relation est en réalité **affine**, avec une ordonnée à l'origine de 3 à 4 cm
  (intervalles de confiance à 95 % excluant zéro sur les deux jeux) ;
- **σ n'explique que 1,4 % de la variance de l'erreur absolue** ;
- le quantile conforme calculé **par quintile de σ** varie du simple au double (6,4 contre
  3,2), alors qu'un score correctement normalisé le rendrait constant.

Conséquence : la couverture est correcte **en moyenne** (94,7 %) et fortement déséquilibrée
**par sous-groupe**.

| quintile de σ | affichage | couverture réelle |
|---|---|---|
| le plus bas | **5 étoiles** | **84,7 %** |
| 2ᵉ | 4 étoiles | 95,0 % |
| 3ᵉ | 3 étoiles | 98,2 % |
| 4ᵉ | 2 étoiles | 98,4 % |
| le plus haut | 1 étoile | 97,0 % |

**Le groupe affiché comme le plus fiable est celui qui l'est le moins**, avec trois fois le
taux d'erreur annoncé. L'indicateur en étoiles dit aujourd'hui le contraire de ce qu'il promet.

Un test global le confirme sans dépendre d'un découpage choisi à l'avance : un classifieur
entraîné à prédire, à partir des 156 variables, si un cheval sera couvert atteint une **AUC de
0,855**. La garantie « 95 % » est donc vraie en moyenne et pratiquement vide pour un cheval
donné — on peut savoir à l'avance, à partir de son profil, s'il fait partie des bien couverts.

## La correction

Le dénominateur devient une **forêt prédisant `|y − ŷ|`** à partir des variables des familles
**f1** (activité : volume de sorties, régularité, rythme, durée de carrière) et **f5** (type de
circuit : élevage, amateur, professionnel), plus **σ** — soit 28 variables.

Trois raisons à ce choix :

1. **Aucune variable encodée sur la cible.** Les variables de pedigree (`f8_*`, `f10_*`) sont
   écartées : elles sont les plus mauvaises pour prédire l'erreur (AUC 0,788 en solo) et leur
   inclusion ferait peser un risque de fuite.
2. **28 variables font aussi bien que 157**, à 2 % de score d'intervalle près.
3. Ce sont **des variables interprétables par un éleveur**, ce qui permettra d'expliquer
   l'incertitude et non seulement de l'afficher.

σ est conservé : seul il ne vaut rien, mais il complète les autres (f1 seule donne une AUC de
0,586 ; f1 + σ donne 0,522).

Le jeu de validation est coupé en deux : une moitié pour ajuster la forêt d'erreur, l'autre
pour calibrer le quantile **et** les quintiles d'étoiles. Ces derniers étaient auparavant
calculés sur le **jeu de test** — fuite mineure du calibrage d'origine, corrigée au passage.

## Résultats

Validation croisée à 5 plis sur les **47 617 chevaux**, avec réentraînement complet du Hurdle
à chaque pli (sans quoi des chevaux du test auraient servi à l'entraînement) :

| | σ (actuel) | corrigé |
|---|---|---|
| couverture globale | 94,89 % | 94,97 % |
| largeur médiane | 35,1 cm | **28,1 cm** (−20 %) |
| score d'intervalle | 45,42 | **36,72** |
| AUC du détecteur | 0,855 | **0,593** |
| couverture « 5 étoiles » | **82,4 %** | **93,4 %** |
| amplitude entre étoiles | **16,4 points** | **2,7 points** |

**Les prédictions ne changent pas.** La MAE reste 6,89 cm : le modèle Hurdle, la prédiction
conforme et l'architecture sont inchangés. Seule la largeur des intervalles et le classement
par fiabilité sont corrigés.

### Sur les cinq fiches du rapport

| cheval | IC ancien | IC corrigé | étoiles | couvert |
|---|---|---|---|---|
| 47237708P | 22,0 cm | 26,6 cm | 5/5 → 4/5 | oui → oui |
| 13417805D | 24,5 cm | 23,0 cm | 5/5 → 5/5 | oui → oui |
| 13414483P | 43,6 cm | **36,2 cm** | 1/5 → **2/5** | oui → oui |
| **60049124M** | **17,3 cm** | **50,0 cm** | **5/5 → 1/5** | **NON → NON** |
| 47261754C | 15,3 cm | 42,4 cm | 5/5 → 1/5 | oui → oui |

Le cas `60049124M` résume le problème : l'ancien calibrage lui donnait **cinq étoiles et le
plus étroit des cinq intervalles**, et il n'était pas couvert. Il reste non couvert après
correction — celle-ci ne rend pas le modèle omniscient — mais il est désormais signalé comme
le cas le plus incertain.

Le cas `13414483P` montre que la correction **redistribue** au lieu d'élargir : son intervalle
rétrécit de 7,4 cm et il gagne une étoile. C'est pourquoi la largeur médiane globale diminue
de 20 % malgré les élargissements individuels.

Images côte à côte : `python3 scripts/comparaison_correction.py` écrit dans
`data/master/comparaison_correction/` (dossier non publié).

## Ce qui change dans le code

- `scripts/fiche.py`
  - `CACHE_V2_PATH` : nouveau cache, écrit **à côté** de l'ancien, jamais à sa place.
  - `fit_and_cache()` : ajoute l'entraînement de la forêt d'erreur, la recalibration du
    quantile et le calcul des quintiles sur la validation.
  - `load_cache(ancien=False)` : charge la version corrigée si elle existe.
  - `predict_one()` : choisit le dénominateur selon le cache chargé.
  - `--ancien` : force l'ancien calibrage, pour comparer.
- `scripts/comparaison_correction.py` : génère les fiches en vis-à-vis.

Reproduire : `python3 scripts/fiche.py --retrain <IDCHEVAL>`.
⚠️ **`scikit-learn==1.6.1`** — vérifié : cette version reproduit exactement le calibrage
d'origine (`q_norm = 3,923`, quintiles σ `[3,72 ; 4,41 ; 4,90 ; 5,41]`).

## Limites

- **Le défaut de couverture conditionnelle est réduit, pas supprimé.** L'AUC passe de 0,855 à
  0,593, ce qui reste au-dessus du seuil de détection (≈ 0,545 par permutation). Un déséquilibre
  résiduel subsiste.
- **Biais de sélection non levé.** Les familles f1 et f5 ont été retenues après comparaison
  d'une vingtaine de variantes sur ces mêmes chevaux. Seule une cohorte nouvelle — des chevaux
  nés après 2013 — le lèverait ; c'est la validation prospective que le projet identifie déjà
  comme manquante.
- **Le dénominateur est une forêt**, donc moins directement lisible que σ. L'explication
  destinée aux éleveurs devra s'appuyer sur les relations marginales des variables (plus de
  sorties → moins d'erreur, plus d'épreuves d'élevage → plus d'erreur), et non sur le modèle.
- **Les fiches déjà remises à la FFE ne sont pas régénérées**, et le rapport n'est pas modifié.

## Ce qui n'est pas fait

Aucune régénération des fiches publiées, aucune modification du rapport, des annexes ou du
README. Cette branche propose un changement ; elle ne l'impose pas.
