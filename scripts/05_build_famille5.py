"""
05 - Famille 5 : Niveau / type d'épreuves.

Génère 24 features : 3 ratios par division (Amateur / Pro / Elevage)
× 8 granularités temporelles (global + 4 par âge + 3 deltas).

Entrée : data/ffe_2010-2025_enriched.parquet
Sortie : data/master/intermediates/famille5_division.parquet
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (load_enriched, get_cohorte_t1_n1_10,
                   save_intermediate, sanity_check)


def main():
    print("=== 05 - Build Famille 5 (Niveau / type d'épreuves) ===\n")

    cohorte_ids = get_cohorte_t1_n1_10(min_participations=10)
    print(f"Cohorte T1+N1>=10 : {len(cohorte_ids):,} chevaux")

    df = load_enriched(columns=[
        "N° SIRE", "AGE", "in_cohorte_T1", "est_libelle_poney",
        "in_fenetre_4_7", "DIVISION_LIB"
    ])

    mask = (df["in_cohorte_T1"]
            & ~df["est_libelle_poney"]
            & df["IDCHEVAL"].isin(cohorte_ids)
            & df["in_fenetre_4_7"])
    d = df[mask].copy()
    d["AGE"] = d["AGE"].astype(int)
    print(f"Lignes cohorte fenêtre 4-7 : {len(d):,}")

    DIVISIONS = ["Amateur", "Pro", "Elevage"]
    # Mapping nom propre (sans accent) pour les noms de features
    DIV_SLUG = {"Amateur": "amateur", "Pro": "pro", "Elevage": "elevage"}

    print("\nCalcul des features...")

    # ==========================================
    # Global 4-7 : taux par division
    # ==========================================
    nb_part = d.groupby("IDCHEVAL").size()
    feats_global = {}
    for div in DIVISIONS:
        nb_div = (d[d["DIVISION_LIB"] == div]
                  .groupby("IDCHEVAL").size()
                  .reindex(nb_part.index, fill_value=0))
        feats_global[f"f5_taux_part_{DIV_SLUG[div]}_4_7"] = (nb_div / nb_part)

    # ==========================================
    # Par âge : taux par division × âge
    # ==========================================
    feats_age = {}
    for div in DIVISIONS:
        for age in [4, 5, 6, 7]:
            d_div_age = d[(d["DIVISION_LIB"] == div) & (d["AGE"] == age)]
            nb_div_age = d_div_age.groupby("IDCHEVAL").size()
            nb_total_age = d[d["AGE"] == age].groupby("IDCHEVAL").size()
            # Ratio : NaN si pas de participation à cet âge
            ratio = (nb_div_age.reindex(nb_total_age.index, fill_value=0)
                     / nb_total_age)
            ratio = ratio.reindex(nb_part.index)  # complet sur tous chevaux
            feats_age[f"f5_taux_part_{DIV_SLUG[div]}_{age}ans"] = ratio

    # ==========================================
    # Deltas inter-annuels
    # ==========================================
    feats_delta = {}
    for div in DIVISIONS:
        for age_from, age_to in [(4, 5), (5, 6), (6, 7)]:
            from_col = feats_age[f"f5_taux_part_{DIV_SLUG[div]}_{age_from}ans"]
            to_col = feats_age[f"f5_taux_part_{DIV_SLUG[div]}_{age_to}ans"]
            feats_delta[f"f5_evolution_part_{DIV_SLUG[div]}_{age_to}_{age_from}"] = (
                to_col - from_col
            )

    # ==========================================
    # Concaténation
    # ==========================================
    features = pd.DataFrame({**feats_global, **feats_age, **feats_delta})
    features = features.reset_index().rename(columns={"index": "IDCHEVAL"})

    print(f"Nb chevaux : {len(features):,}")
    print(f"Nb colonnes features : {len(features.columns) - 1}")

    sanity_check(features, family_id=5, expected_n_features=24)
    save_intermediate(features, family_id=5, family_name="division")


if __name__ == "__main__":
    main()
