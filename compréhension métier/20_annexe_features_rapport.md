# Annexe E — Récapitulatif des features

Le master dataset final contient **156 features numériques** (deux booléens initialement comptés sont écartés par filtrage de type au moment du fit), construites à partir de la cohorte T1 + N1 ≥ 10 (47 617 chevaux nés 2006-2013). Les features sont organisées en **sept familles** thématiques, chacune codée par un préfixe `f<N>_` permettant un suivi de bout en bout dans le pipeline.

L'objectif de cette annexe est triple : donner une vue d'ensemble du périmètre couvert, expliciter la logique métier de chaque famille, et identifier les features les plus prédictives selon les deux régresseurs du modèle Hurdle.

---

## E.1 — Vue synthétique par famille

| Famille | Préfixe | Thème | Nb features | Logique métier |
|---|---|---|---|---|
| F1 | `f1_` | Activité / Volume | 16 | Régularité d'engagement, intensité de carrière, saisons blanches |
| F2 | `f2_` | Performance financière (gains) | 8 | Gains absolus et normalisés, évolution inter-saisons |
| F3 | `f3_` | Performance sportive (placement) | 75 | Percentiles vs partants, ratios sans-faute, évolution |
| F5 | `f5_` | Niveau / type d'épreuves | 12 | Hauteur moyenne courue, mix Pro/Amateur/Club |
| F7 | `f7_` | Cavalier(s) | 38 | Niveau et expérience du / des cavaliers associés |
| F8 | `f8_` | Pedigree (père + grand-père maternel) | 6 | Performance moyenne des frères et descendants |
| F10 | `f10_` | Race / stud-book | 3 | Target encoding par stud-book |
| **Total** | | | **158** (dont **156 utilisées**) | |

**Familles écartées en cours de projet.** Les numéros F4, F6, F9 manquants ne sont pas un oubli : ce sont des familles qui ont été explorées puis abandonnées (cf. journal de décisions). F4 (cavalier-cheval croisé) et F6 (météo / saisonnalité) ont été jugées trop bruitées au regard du coût de construction. F9 (couleur / robe) ne portait pas de signal exploitable.

---

## E.2 — Description détaillée par famille

### F1 — Activité et volume (16 features)

Mesure l'**intensité d'engagement** du cheval sur la fenêtre 4-7 ans : nombre total de participations, nombre de saisons actives, présence d'une saison blanche, écart-type du volume entre saisons. Variables typiques : `f1_nb_participations_4_7`, `f1_nb_saisons_actives`, `f1_a_saison_blanche_4_7`.

**Hypothèse métier** : un cheval orienté haut niveau cumule un volume d'épreuves régulier ; les profils amateurs ont souvent une trajectoire plus irrégulière.

### F2 — Gains (8 features)

Capture la **performance financière** : gains totaux, gains par épreuve, log-gains, évolution inter-saisons. Variables typiques : `f2_gains_7ans`, `f2_gains_par_evenement_4_7`, `f2_evolution_gains_7_6`.

**Hypothèse métier** : les gains sont une mesure indirecte du niveau (allocations indexées sur la hauteur de l'épreuve). C'est l'une des deux familles les plus prédictives globalement (cf. E.3).

### F3 — Placement sportif (75 features — la plus grosse)

Décrit la **performance relative** du cheval dans ses épreuves : percentile des partants, ratios sans-faute, parcours classés, médiane / écart-type de classement, évolution annuelle de ces grandeurs. Variables typiques : `f3_percentile_partants_median_7ans`, `f3_percentile_partants_std_7ans`, `f3_ratio_sans_faute_7ans`.

**Hypothèse métier** : un cheval *progressif* (qui améliore son percentile au fil des saisons) signale un potentiel d'évolution non encore exploité. La forte cardinalité de la famille s'explique par la déclinaison systématique en quatre dimensions (statistique × tranche d'âge × catégorie × type d'épreuve).

### F5 — Niveau / type d'épreuves (12 features)

Synthétise la **gamme de hauteurs courues** : hauteur moyenne, hauteur maximale tentée, proportion d'épreuves Pro / Amateur / Club. Variables typiques : `f5_hauteur_moyenne_7ans`, `f5_part_pro_4_7`, `f5_hauteur_max_tentee_6ans`.

**Risque de fuite contrôlé** : la *hauteur max tentée* n'est pas la cible (`hauteur_max_validée`), car la cible exige trois participations à la hauteur. Une « tentative ponctuelle » est donc une information licite, sans risque de leak.

### F7 — Cavalier (38 features)

Décrit le **niveau du ou des cavaliers** ayant monté le cheval : nombre de cavaliers distincts, log-gains moyens du cavalier principal, expérience cumulée du cavalier, percentile du cavalier dans la population. Variables typiques : `f7_nb_participations_cavalier_passe3_7ans`, `f7_cavalier_mean_log_gains_pos_passe3_7ans`.

**Hypothèse métier** : un bon cavalier valorise un cheval à potentiel. Cette famille porte le plus gros risque de **target encoding** (le cavalier ayant monté ce cheval est aussi évalué sur ce cheval), traité par leave-one-out + smoothing bayésien (annexe A.2 / GitHub).

### F8 — Pedigree (6 features)

Caractérise l'**ascendance** du cheval : performance moyenne des descendants du père, performance moyenne des descendants du grand-père maternel. Variables typiques : `f8_pere_target_encoded_LOO`, `f8_pere_mean_gains_LOO`, `f8_gp_maternel_mean_percentile_partants_LOO`.

**Choix méthodologique** : seuls père et grand-père maternel sont retenus (la mère seule est trop rarement renseignée pour être exploitable, et la couverture chute fortement au-delà de la 2ème génération).

### F10 — Race / stud-book (3 features)

Target encoding par stud-book : `f10_race_target_encoded_LOO`, `f10_race_mean_gains_LOO`, `f10_race_mean_percentile_partants_LOO`. Famille très compacte mais **rang 1 d'importance pour le régresseur des tops** (cf. E.3).

---

## E.3 — Top 15 features par importance

Le tableau ci-dessous reporte les 15 features les plus importantes au sens de la **mean decrease in impurity** (Random Forest), comparées entre les deux régresseurs du Hurdle : celui qui prédit la masse de la population (« default ») et celui qui prédit les tops (≥ 1,40 m). La colonne **Δ rang** = rang_default − rang_tops.

| Rang tops | Feature | Famille | Rang default | Δ rang |
|---|---|---|---|---|
| 1 | `f10_race_target_encoded_LOO` | F10 | 5 | +4 |
| 2 | `f10_race_mean_gains_LOO` | F10 | 12 | +10 |
| 3 | `f10_race_mean_percentile_partants_LOO` | F10 | 20 | +17 |
| 4 | `f8_pere_target_encoded_LOO` | F8 | 13 | +9 |
| 5 | `f2_gains_7ans` | F2 | 4 | −1 |
| 6 | `f2_gains_par_evenement_4_7` | F2 | 11 | +5 |
| 7 | `f7_nb_participations_cavalier_passe3_7ans` | F7 | 8 | +1 |
| 8 | `f8_pere_mean_gains_LOO` | F8 | 28 | +20 |
| 9 | `f7_cavalier_mean_log_gains_pos_passe3_7ans` | F7 | 6 | −3 |
| 10 | `f7_nb_chevaux_distincts_cavalier_passe3_7ans` | F7 | 9 | −1 |
| 11 | `f2_gains_total_4_7` | F2 | 7 | −4 |
| 12 | `f3_percentile_partants_median_7ans` | F3 | 49 | +37 |
| 13 | `f2_evolution_gains_7_6` | F2 | 17 | +4 |
| 14 | `f7_cavalier_mean_log_gains_pos_passe3_6ans` | F7 | 25 | +11 |
| 15 | `f8_gp_maternel_mean_percentile_partants_LOO` | F8 | 51 | +36 |

**Lectures clés du tableau :**

1. **F10 (Race) et F8 (Pedigree) sont disproportionnellement importants pour les tops.** La race et l'ascendance sont presque inopérantes sur le grand public (rangs 5-51 sur le régresseur default), mais structurent la prédiction du haut niveau (rangs 1-15 sur le régresseur tops). C'est la justification statistique du choix Hurdle : la frontière du top obéit à une logique métier différente du reste de la distribution.

2. **Les features de progression (Δ rang positif élevé)** émergent spécifiquement chez les tops. `f3_percentile_partants_median_7ans` passe du rang 49 au rang 12 (+37) : la médiane du percentile à 7 ans n'est utile que pour distinguer les futurs cracks entre eux, pas pour positionner un cheval amateur.

3. **Les features de gains (F2)** sont importantes dans les deux régimes mais légèrement *plus* importantes sur le default (Δ rang négatif). Les gains restent l'ancrage de la prédiction quantitative ; ce sont les *familles complémentaires* (pedigree, race, progression) qui font la différence sur l'élite.

---

## E.4 — Note méthodologique

Le décompte de 156 features actives est issu d'une **épuration en deux temps** (cf. section 3.4 du rapport principal) :

1. **Cohorte initiale** : ~ 273 features candidates listées dans le catalogue (`09_catalogue_features.md`), incluant toutes les variantes envisagées (médianes, moyennes, ratios, écarts-types par tranche d'âge et par catégorie d'épreuve).

2. **Première épuration (corrélation)** : élimination des features dont la corrélation absolue avec une autre est ≥ 0,95, en conservant la plus interprétable. Réduction à 190 features.

3. **Seconde épuration (importance)** : élimination des features dont l'importance Random Forest est inférieure à 0,002 sur les deux régresseurs simultanément. Réduction finale à **156 features**.

Cette double épuration améliore le MAE global de 0,3 cm et réduit le temps d'entraînement de ~40 %, sans perte de couverture sur l'IC.
