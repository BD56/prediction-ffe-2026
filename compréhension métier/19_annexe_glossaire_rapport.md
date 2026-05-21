# Annexe D : Glossaire métier

Ce glossaire définit les termes spécifiques à la filière équestre française et à la base FFE qui apparaissent dans le rapport. Il s'adresse en priorité aux lecteurs non équestres pour faciliter la compréhension des choix méthodologiques. Les termes sont regroupés par thématique plutôt qu'alphabétiquement, pour rendre la lecture en continu plus naturelle.

---

## D.1 : Discipline et niveaux d'épreuves

**Saut d'Obstacles (SO ou CSO)** : Discipline équestre où le couple cheval-cavalier doit franchir un parcours d'obstacles sans en faire tomber les barres. Le chronométrage et le décompte des pénalités déterminent le classement. C'est la discipline modélisée dans ce projet ; le jeu de données inclut également l'attelage, le dressage, le complet, mais ils sont écartés en amont.

**Hauteur** : Hauteur du plus gros obstacle d'une épreuve, exprimée en mètres (typiquement 0,90 m à 1,60 m). C'est la mesure de difficulté de référence du SO.

**Niveau d'épreuve** : Hiérarchie FFE des compétitions :
- **Club** : épreuves de loisir / formation, généralement ≤ 1,15 m.
- **Amateur 1 / 2 / 3** : niveau intermédiaire, 1,05 m à 1,30 m.
- **Pro 2 / Pro 1 / Pro Élite** : niveau professionnel, 1,30 m à 1,60 m+. Le seuil **1,40 m** marque conventionnellement l'entrée dans le Pro confirmé et constitue la frontière retenue pour la classification binaire du modèle Hurdle.

**Grand Prix (GP)** : Épreuve la plus haute d'un concours, souvent à 1,40 m – 1,60 m. Indicateur fort dans les features de carrière.

---

## D.2 : Identification et suivi des chevaux

**Numéro SIRE** : Identifiant unique du cheval dans la base nationale française (IFCE), composé de 8 caractères alphanumériques (ex. *47237708P*). Sert de clé primaire dans tout le pipeline, sous le nom de colonne `IDCHEVAL`.

**Stud-book** : Registre généalogique d'une race. Les principaux pour le SO français sont **SBF** (Selle Français), **AA** (Anglo-Arabe), **KWPN** (Néerlandais), **BWP** (Belge). Le stud-book conditionne souvent l'éligibilité à certaines épreuves d'élevage.

**WBFSH** : *World Breeding Federation for Sport Horses*. Fédération qui publie un classement mondial des stud-books de chevaux de sport, utilisé en feature comme variable d'environnement.

---

## D.3 : Mesures de performance et indices

**Hauteur max validée** (`hauteur_max_validee`), **Cible du modèle**. Définie comme la hauteur maximale à laquelle le cheval a participé au moins **trois fois** au cours de sa carrière, sur toute sa cohorte. Le seuil de trois participations évite de retenir une participation exceptionnelle (épreuve « tentée une fois » par dépassement de niveau).

**Indices Ricard** : Indices statistiques calculés par l'IFCE pour mesurer la valeur génétique d'un cheval en SO. Souvent fortement corrélés à la performance future ; non utilisés en feature dans ce projet pour éviter la fuite (calculés *a posteriori*).

**Percentile partants** (`f3_percentile_partants_*`), Pour chaque épreuve courue par un cheval, on calcule son percentile dans le classement des partants. Indicateur de progression : un cheval qui progresse améliore son percentile au fil des saisons. C'est l'une des familles de features les plus prédictives du modèle (rang 12 et 24 d'importance globale).

**Coef départemental** : Coefficient FFE multiplicateur appliqué aux gains pour normaliser le niveau de concurrence selon le département de l'épreuve. Présent en feature dans la famille « gains ».

---

## D.4 : Cycle de carrière et formation

**Cycle classique** : Filière FFE de formation des jeunes chevaux de 4 à 7 ans, structurée en épreuves dédiées (cycle classique 4 ans, 5 ans, 6 ans, 7 ans). Engagement coûteux (~ 1 500 € de droits annuels) qui suppose une orientation haut niveau assumée.

**Cycle libre** : Engagement en épreuves Club / Amateur classiques, sans inscription au cycle classique. Voie d'orientation pour les jeunes chevaux destinés à un usage amateur.

**Cohorte T1 + N1 ≥ 10** : Périmètre de modélisation retenu dans ce projet :
- **T1** : nés entre 2006 et 2013, discipline SO, hors poneys (race et libellé). Filtre temporel pour garantir une carrière complète observée.
- **N1 ≥ 10** : au moins 10 participations totales en SO. Filtre d'activité pour éviter les chevaux à carrière trop courte pour porter un signal exploitable.

Effectif final : **47 617 chevaux**.

---

## D.5 : Acteurs de la filière

**Éleveur** : Propriétaire qui fait naître et élève le cheval. Prend les premières décisions d'orientation (cycle classique vs cycle libre, choix du cavalier de débourrage).

**Cavalier propriétaire** : Cavalier qui possède son cheval. Décide du niveau d'engagement en concours.

**Marchand de chevaux** : Acteur professionnel de la transaction. Premier utilisateur cible du modèle pour le calibrage des prix d'achat.

**Centre équestre** : Structure d'enseignement et de location. Utilisateur potentiel pour le tri de la cavalerie école (chevaux destinés à l'enseignement plutôt qu'au sport).

**Fédération Française d'Équitation (FFE)** : Autorité de tutelle sportive, gestionnaire de la base de données utilisée dans ce projet. Définit les niveaux d'épreuves, les indices et les programmes de détection des talents.

**IFCE (Institut français du cheval et de l'équitation)** : Établissement public chargé du SIRE et des indices Ricard. Référent généalogique et statistique de la filière.
