"""
02 - Famille 2 : Performance financière (Gains du cheval).

Génère 10 features sur les gains du cheval lui-même, sur la fenêtre 4-7 ans.

Convention : stockage en valeurs brutes (€). Transformation log(GAINS+1)
standardisée par année appliquée à la modélisation (cf. journal des décisions).

Features :
  Volume global (1) :
    - f2_gains_total_4_7
  Valeurs annuelles (4) :
    - f2_gains_4ans, _5ans, _6ans, _7ans
  Deltas inter-annuels (3) :
    - f2_evolution_gains_5_4, _6_5, _7_6
  Ratios d'efficacité (2) :
    - f2_gains_par_participation_4_7
    - f2_gains_par_evenement_4_7

Entrée : data/ffe_2010-2025_enriched.parquet
Sortie : data/master/intermediates/famille2_gains.parquet
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (load_enriched, get_cohorte_t1_n1_10,
                   save_intermediate, sanity_check)


def main():
    print("=== 02 - Build Famille 2 (Gains du cheval) ===\n")

    cohorte_ids = get_cohorte_t1_n1_10(min_participations=10)
    print(f"Cohorte T1+N1>=10 : {len(cohorte_ids):,} chevaux")

    df = load_enriched(columns=[
        "N° SIRE", "AGE", "in_cohorte_T1", "est_libelle_poney",
        "in_fenetre_4_7", "GAINS", "NUMERO_EVENEMENT2"
    ])

    mask = (df["in_cohorte_T1"]
            & ~df["est_libelle_poney"]
            & df["IDCHEVAL"].isin(cohorte_ids)
            & df["in_fenetre_4_7"])
    d = df[mask].copy()
    print(f"Lignes cohorte fenêtre 4-7 : {len(d):,}")

    # Remplacer GAINS NaN par 0 (rare : 0,01%)
    d["GAINS"] = d["GAINS"].fillna(0)
    d["AGE"] = d["AGE"].astype(int)

    print("\nCalcul des features...")

    # ==========================================
    # Volume global
    # ==========================================
    gains_total = (d.groupby("IDCHEVAL")["GAINS"].sum()
                   .rename("f2_gains_total_4_7"))

    # ==========================================
    # Valeurs annuelles : somme des gains par âge
    # ==========================================
    by_age = (d.groupby(["IDCHEVAL", "AGE"])["GAINS"].sum()
              .unstack(fill_value=0))
    for age in [4, 5, 6, 7]:
        if age not in by_age.columns:
            by_age[age] = 0
    by_age = by_age[[4, 5, 6, 7]]
    by_age.columns = [f"f2_gains_{age}ans" for age in [4, 5, 6, 7]]

    # ==========================================
    # Deltas inter-annuels
    # ==========================================
    delta_5_4 = (by_age["f2_gains_5ans"] - by_age["f2_gains_4ans"]).rename(
        "f2_evolution_gains_5_4")
    delta_6_5 = (by_age["f2_gains_6ans"] - by_age["f2_gains_5ans"]).rename(
        "f2_evolution_gains_6_5")
    delta_7_6 = (by_age["f2_gains_7ans"] - by_age["f2_gains_6ans"]).rename(
        "f2_evolution_gains_7_6")

    # ==========================================
    # Ratios d'efficacité (besoin de nb_participations / nb_evenements)
    # ==========================================
    nb_part = d.groupby("IDCHEVAL").size()
    nb_evt = d.groupby("IDCHEVAL")["NUMERO_EVENEMENT2"].nunique()
    gains_par_part = (gains_total / nb_part).rename("f2_gains_par_participation_4_7")
    gains_par_evt = (gains_total / nb_evt).rename("f2_gains_par_evenement_4_7")

    # ==========================================
    # Concaténation
    # ==========================================
    features = pd.concat([
        gains_total,
        by_age,
        delta_5_4, delta_6_5, delta_7_6,
        gains_par_part, gains_par_evt,
    ], axis=1).reset_index()

    print(f"Nb chevaux avec features : {len(features):,}")
    print(f"Nb colonnes : {len(features.columns)}")

    sanity_check(features, family_id=2, expected_n_features=10)
    save_intermediate(features, family_id=2, family_name="gains")


if __name__ == "__main__":
    main()
