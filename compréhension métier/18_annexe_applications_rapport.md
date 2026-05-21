# Annexe C : Cadre d'usage opérationnel

La fiche de prédiction (annexe B) n'a de valeur que si elle déclenche une action différente de celle que l'utilisateur aurait prise sans le modèle. Cette annexe propose un **cadre de lecture** des deux principales grandeurs affichées (P_top et largeur d'IC) en regard de décisions métier typiques. Les seuils proposés sont des **points de fonctionnement suggérés** : calibrés sur les performances observées (MAE = 6,9 cm sur les tops, couverture empirique 95,3 %), et restent à **valider en condition réelle** par une étude d'usage avec des acteurs de la filière.

---

## C.1 : Grille de lecture

| Profil de fiche | P_top | Largeur IC | Cadre de décision suggéré |
|---|---|---|---|
| Top confirmé | ≥ 0,70 | ≤ 25 cm | Orientation haut niveau assumée (cycle classique, achat ferme envisageable) |
| Plafond bas confirmé | < 0,30 | ≤ 25 cm | Orientation amateur / loisir, programme allégé |
| Zone incertaine | [0,30 ; 0,70] | tout | **Reporter la décision** : attendre une saison supplémentaire avant engagement coûteux |
| Confiance dégradée | tout | > 30 cm | Le modèle signale qu'il ne sait pas, décision à la main de l'expert humain |

Trois principes encadrent la lecture de cette grille :

1. **La fiche est une aide à la décision, pas une décision.** Elle identifie les profils qui méritent un examen approfondi, pas un automatisme.

2. **Les seuils ne sont pas universels.** Un acteur plus prudent (par exemple un assureur) peut translater les bornes vers le haut ; un acteur plus tolérant au risque (par exemple un marchand spécialisé en jeunes chevaux) peut les abaisser. La calibration définitive relève d'une étude utilisateur dédiée.

3. **Les grandeurs interagissent.** Une probabilité haute avec un IC large ne porte pas la même information qu'une probabilité haute avec un IC étroit. La grille croise systématiquement deux dimensions pour éviter les fausses certitudes.

---

## C.2 : Limites du périmètre opérationnel

Quatre restrictions doivent être explicitées pour éviter toute sur-interprétation :

1. **Le modèle prédit un plafond, pas une carrière.** Un cheval peut atteindre 1,45 m une fois puis se blesser et finir à 1,30 m de moyenne. La dimension de longévité n'est pas capturée par la cible *hauteur_max_validée*.

2. **Les chevaux importés ou changeant de structure majeure en cours de carrière** sont mal modélisés : les features s'appuient sur l'historique français complet, qui n'existe pas pour les chevaux acquis tardivement à l'étranger.

3. **Les chevaux à très faible activité (T1 < 10 participations)** sont exclus de la cohorte par construction. Le modèle ne peut rien dire d'un poulain qui n'a pas encore amorcé sa carrière.

4. **Le seuil 1,40 m est métier, pas naturel.** Les chevaux dont le plafond se situe à la frontière (1,38 m ou 1,42 m) sont mécaniquement plus exposés à une erreur de classification top / non-top que les profils francs.
