# Annexe B : Fiches de prédiction

Chaque cheval prédit donne lieu à une **fiche individuelle** agrégeant statut probable, hauteur ponctuelle prédite, intervalle de confiance à 95 % et indicateur d'incertitude locale. L'anatomie complète d'une fiche est détaillée en section 5 du rapport. Cette annexe présente cinq profils représentatifs et décrit l'application qui génère une fiche à partir d'un numéro SIRE.

## B.1 : Cinq profils représentatifs

### B.1.1 : Crack confirmé, modèle correct

![](../data/master/figures_rapport/fiche_47237708P.png){ width=45% }

**Cheval 47237708P** : P_top = 0,94, prédit 1,43 m, IC étroit (22 cm), σ = 2,80 cm (5/5 étoiles). Réel 1,35 m, dans l'IC. Cas nominal : signal fort, prédiction précise.

### B.1.2 : Plafond amateur, modèle correct

![](../data/master/figures_rapport/fiche_13417805D.png){ width=45% }

**Cheval 13417805D** : P_top = 0,00, prédit 1,11 m, IC de 24,5 cm. Réel 1,10 m, erreur 1,0 cm. Cas le plus fréquent (~70 % de la cohorte), valide l'utilité métier d'écarter les fausses promesses pro.

### B.1.3 : Cas incertain, modèle honnête

![](../data/master/figures_rapport/fiche_13414483P.png){ width=45% }

**Cheval 13414483P** : P_top = 0,40, prédit 1,34 m, IC très large (43,6 cm), σ = 5,56 cm (1/5 étoile). Réel 1,35 m au cœur de l'IC. Illustre la propriété clé de la chaîne LAC : **quand le modèle ne sait pas, il le dit** en élargissant l'IC.

### B.1.4 : Anomalie statistique, modèle confiant et faux

![](../data/master/figures_rapport/fiche_60049124M.png){ width=45% }

**Cheval 60049124M** : P_top = 0,73, prédit 1,39 m, IC étroit (17,3 cm), σ = 2,20 cm (5/5 étoiles). Réel 1,05 m, **hors IC** (erreur 33,6 cm). Conséquence assumée d'un IC à 95 % : ~252 chevaux sur 5 045 doivent par construction tomber dehors. La confiance traduit la précision interne, pas une garantie individuelle.

### B.1.5 : Vrai TOP ≥ 1,45 m, prédit correctement

![](../data/master/figures_rapport/fiche_47261754C.png){ width=45% }

**Cheval 47261754C** : P_top = 0,68 (statut Incertain, juste sous le seuil 0,70), prédit 1,38 m, IC étroit (15,3 cm). Réel 1,45 m en haut de l'IC (erreur 7,1 cm). Le modèle détecte le crack mais reste prudent sur la hauteur exacte.

## B.2 : Application interactive

Trois interfaces, par ordre croissant d'autonomie :

**CLI Python** : `python3 scripts/fiche.py 47237708P`. Premier appel ~13 s (entraînement + cache), suivants ~2 s.

**Application graphique macOS** : double-clic sur `Fiche FFE.command`. Dialogue système qui boucle jusqu'à annulation, ouvre chaque fiche dans Aperçu.

**Mode production** : pour un cheval hors test set (sans hauteur réelle connue), l'application bascule sur un rendu sans bloc validation. Cible : cheval de 4 ans dont on attend la consécration à 7 ans.

## B.3 : Limites de l'interface

1. **La confiance en étoiles n'est pas une probabilité d'exactitude.** Elle traduit la position de σ(x) dans la distribution d'incertitude interne du modèle (cas B.1.4).
2. **L'IC à 95 % est une garantie marginale, pas conditionnelle.** Le taux empirique peut s'écarter de la cible sur un sous-segment particulier.
3. **La fiche ne se substitue pas à une expertise vétérinaire ou cavalière.** Le modèle ignore la morphologie, l'historique de blessures, l'environnement de travail.
