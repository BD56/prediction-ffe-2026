"""
01 - Famille 1 : Activité / Volume.

Génère 17 features de volume et régularité, calculées sur la fenêtre 4-7 ans
du cheval, pour les chevaux de la cohorte T1+N1>=10.

Features :
  Volume / événements (3) :
    - f1_nb_participations_4_7
    - f1_nb_evenements_4_7
    - f1_nb_participations_par_evenement
  Régularité inter-annuelle (3) :
    - f1_nb_annees_actives_4_7
    - f1_age_premiere_participation
    - f1_a_saison_blanche_4_7
  Valeurs annuelles (4) :
    - f1_nb_participations_4ans, _5ans, _6ans, _7ans
  Deltas inter-annuels (3) :
    - f1_evolution_5_4, _6_5, _7_6
  Régularité intra-annuelle / rythme (4) :
    - f1_nb_mois_actifs_4_7
    - f1_intensite_moyenne_mensuelle
    - f1_annee_pic_activite
    - f1_duree_carriere_jeunesse_jours
    - f1_jours_moyens_entre_sorties

Entrée : data/ffe_2010-2025_enriched.parquet
Sortie : data/master/intermediates/famille1_activite.parquet
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (load_enriched, get_cohorte_t1_n1_10,
                   save_intermediate, sanity_check)


def main():
    print("=== 01 - Build Famille 1 (Activité / Volume) ===\n")

    # Cohorte de modélisation
    cohorte_ids = get_cohorte_t1_n1_10(min_participations=10)
    print(f"Cohorte T1+N1>=10 : {len(cohorte_ids):,} chevaux")

    # Chargement des colonnes nécessaires
    df = load_enriched(columns=[
        "N° SIRE", "AGE", "in_cohorte_T1", "est_libelle_poney",
        "in_fenetre_4_7", "NUMERO_EVENEMENT2", "DATEEPREUVE"
    ])
    df = df.rename(columns={"N° SIRE": "IDCHEVAL"}) if "N° SIRE" in df.columns else df
    # Note: load_enriched renomme déjà 'N° SIRE' en 'IDCHEVAL'.

    # Filtre cohorte + fenêtre 4-7
    mask = (df["in_cohorte_T1"]
            & ~df["est_libelle_poney"]
            & df["IDCHEVAL"].isin(cohorte_ids)
            & df["in_fenetre_4_7"])
    d = df[mask].copy()
    print(f"Lignes cohorte fenêtre 4-7 : {len(d):,}")

    # Conversion date pour calculs de mois et durée
    d["DATEEPREUVE_DT"] = pd.to_datetime(d["DATEEPREUVE"])
    d["MOIS"] = d["DATEEPREUVE_DT"].dt.to_period("M").astype(str)
    d["AGE"] = d["AGE"].astype(int)

    print("\nCalcul des features...")

    # ==========================================
    # Volume / événements
    # ==========================================
    nb_part_47 = d.groupby("IDCHEVAL").size().rename("f1_nb_participations_4_7")
    nb_evt_47 = (d.groupby("IDCHEVAL")["NUMERO_EVENEMENT2"].nunique()
                  .rename("f1_nb_evenements_4_7"))

    # ==========================================
    # Régularité inter-annuelle
    # ==========================================
    nb_annees_actives = (d.groupby("IDCHEVAL")["AGE"].nunique()
                         .rename("f1_nb_annees_actives_4_7"))
    age_premiere = (d.groupby("IDCHEVAL")["AGE"].min()
                    .rename("f1_age_premiere_participation"))
    age_derniere = d.groupby("IDCHEVAL")["AGE"].max()  # utile pour saison blanche

    # Saison blanche : amplitude > nb années actives ?
    amplitude = (age_derniere - age_premiere + 1)
    saison_blanche = (amplitude > nb_annees_actives).rename("f1_a_saison_blanche_4_7")

    # ==========================================
    # Valeurs annuelles (nb_participations à 4, 5, 6, 7 ans)
    # ==========================================
    by_age = d.groupby(["IDCHEVAL", "AGE"]).size().unstack(fill_value=0)
    # Garantir les 4 colonnes 4, 5, 6, 7
    for age in [4, 5, 6, 7]:
        if age not in by_age.columns:
            by_age[age] = 0
    by_age = by_age[[4, 5, 6, 7]]
    by_age.columns = [f"f1_nb_participations_{age}ans" for age in [4, 5, 6, 7]]

    # ==========================================
    # Deltas inter-annuels
    # ==========================================
    delta_5_4 = (by_age["f1_nb_participations_5ans"]
                 - by_age["f1_nb_participations_4ans"]).rename("f1_evolution_5_4")
    delta_6_5 = (by_age["f1_nb_participations_6ans"]
                 - by_age["f1_nb_participations_5ans"]).rename("f1_evolution_6_5")
    delta_7_6 = (by_age["f1_nb_participations_7ans"]
                 - by_age["f1_nb_participations_6ans"]).rename("f1_evolution_7_6")

    # ==========================================
    # Régularité intra-annuelle / rythme
    # ==========================================
    nb_mois_actifs = (d.groupby("IDCHEVAL")["MOIS"].nunique()
                      .rename("f1_nb_mois_actifs_4_7"))
    intensite_moyenne_mensuelle = (nb_part_47 / nb_mois_actifs).rename(
        "f1_intensite_moyenne_mensuelle")
    annee_pic_activite = (by_age.idxmax(axis=1)
                            .str.replace("f1_nb_participations_", "")
                            .str.replace("ans", "")
                            .astype(int)
                            .rename("f1_annee_pic_activite"))

    # Ratio participations / événements
    nb_part_par_evt = (nb_part_47 / nb_evt_47).rename(
        "f1_nb_participations_par_evenement")

    # Durée carrière en jours (max - min de DATEEPREUVE)
    duree_jours = ((d.groupby("IDCHEVAL")["DATEEPREUVE_DT"].max()
                    - d.groupby("IDCHEVAL")["DATEEPREUVE_DT"].min())
                   .dt.days.rename("f1_duree_carriere_jeunesse_jours"))

    # Jours moyens entre sorties : duree / (nb_part - 1). NaN si nb_part == 1.
    jours_entre = (duree_jours / (nb_part_47 - 1)).rename("f1_jours_moyens_entre_sorties")
    # Pour les chevaux à 1 seule participation : NaN (division par zéro)

    # ==========================================
    # Concaténation finale
    # ==========================================
    features = pd.concat([
        nb_part_47,
        nb_evt_47,
        nb_part_par_evt,
        nb_annees_actives,
        age_premiere,
        saison_blanche,
        by_age,
        delta_5_4,
        delta_6_5,
        delta_7_6,
        nb_mois_actifs,
        intensite_moyenne_mensuelle,
        annee_pic_activite,
        duree_jours,
        jours_entre,
    ], axis=1).reset_index()

    print(f"Nb chevaux avec features : {len(features):,}")
    print(f"Nb colonnes : {len(features.columns)}")

    # Sanity check
    sanity_check(features, family_id=1, expected_n_features=17)

    # Sauvegarde
    save_intermediate(features, family_id=1, family_name="activite")


if __name__ == "__main__":
    main()
