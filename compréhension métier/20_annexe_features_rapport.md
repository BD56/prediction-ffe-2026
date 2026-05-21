# Annexe E : Récapitulatif des features

Le master dataset contient **156 features numériques** (deux booléens écartés au fit) construites sur la cohorte T1 + N1 ≥ 10 (47 617 chevaux). Sept familles thématiques codées par préfixe `f<N>_`. Les numéros F4, F6, F9 manquants correspondent à des familles explorées puis abandonnées (cavalier-cheval croisé, météo / saisonnalité, robe), jugées trop bruitées ou sans signal.

## E.1 : Synthèse des sept familles

| Famille | Préfixe | Thème | Nb | Exemples typiques | Hypothèse métier |
|---|---|---|---|---|---|
| F1 | `f1_` | Activité / volume | 16 | `nb_participations_4_7`, `nb_saisons_actives`, `a_saison_blanche_4_7` | Un cheval haut niveau cumule un volume régulier d'épreuves |
| F2 | `f2_` | Gains | 8 | `gains_7ans`, `gains_par_evenement_4_7`, `evolution_gains_7_6` | Les allocations sont indexées sur la hauteur, donc proxy direct du niveau |
| F3 | `f3_` | Placement | 75 | `percentile_partants_median_7ans`, `ratio_sans_faute_7ans` | Un cheval progressif signale un potentiel non encore exploité |
| F5 | `f5_` | Niveau d'épreuves | 12 | `hauteur_moyenne_7ans`, `part_pro_4_7`, `hauteur_max_tentee_6ans` | La gamme de hauteurs courues révèle la trajectoire d'orientation |
| F7 | `f7_` | Cavalier | 38 | `nb_participations_cavalier_passe3_7ans`, `cavalier_mean_log_gains_pos` | Un bon cavalier valorise un cheval à potentiel (TE LOO + smoothing) |
| F8 | `f8_` | Pedigree | 6 | `pere_target_encoded_LOO`, `gp_maternel_mean_percentile_partants_LOO` | Père et grand-père maternel uniquement (couverture mère insuffisante) |
| F10 | `f10_` | Race / stud-book | 3 | `race_target_encoded_LOO`, `race_mean_gains_LOO` | TE par stud-book, rang 1 d'importance sur le régresseur tops |

## E.2 : Top 15 features par importance

Mean decrease in impurity (RF), comparée entre le régresseur des tops (≥ 1,40 m) et le régresseur default. La colonne **Δ rang** = rang_default − rang_tops.

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

**Trois lectures clés :**

1. **F10 (Race) et F8 (Pedigree) sont disproportionnellement importants pour les tops** (rangs 1-15 tops, rangs 5-51 default). Justification statistique du choix Hurdle : la frontière du top obéit à une logique différente.
2. **Les features de progression émergent spécifiquement chez les tops.** `f3_percentile_partants_median_7ans` passe du rang 49 au rang 12.
3. **Les gains (F2) sont importants partout** mais légèrement plus sur default. Ils restent l'ancrage de la prédiction quantitative ; pedigree, race et progression font la différence sur l'élite.
