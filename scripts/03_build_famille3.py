"""
03 - Famille 3 : Performance sportive (placement / résultats).

Génère 153 features :
  Non-classement (8) : taux des codes 800+ × 8 granularités
  Victoires (17) : compte et taux × 8 granularités + booléen au moins 1 victoire
  Percentile (64) : médiane/min/max/std × 2 versions partants/finishers × 8 granularités
  Top X% (64) : 4 seuils (5/10/25/50%) × 2 versions × 8 granularités

Codes PLACE administratifs identifiés empiriquement (rupture nette à partir
de 351) : 899, 900, 902, 992, 993 → traités en bloc comme "non-classement".

Convention percentile :
  partants = PLACE / nb_total_lignes_epreuve (incluant codes)
  finishers = PLACE / nb_lignes_avec_PLACE_valide

Entrée : data/ffe_2010-2025_enriched.parquet
Sortie : data/master/intermediates/famille3_performance.parquet
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (load_enriched, get_cohorte_t1_n1_10,
                   save_intermediate, sanity_check)

CODES_NON_CLASSEMENT = {899, 900, 902, 992, 993}
PLACE_MAX_VRAIE = 800  # Au-dessus = code administratif
TOP_X_SEUILS = [0.05, 0.10, 0.25, 0.50]


def main():
    print("=== 03 - Build Famille 3 (Performance sportive) ===\n")

    cohorte_ids = get_cohorte_t1_n1_10(min_participations=10)
    print(f"Cohorte T1+N1>=10 : {len(cohorte_ids):,} chevaux")

    # On charge TOUTES les lignes SO (pas seulement la cohorte) pour calculer
    # nb_partants et nb_finishers par épreuve correctement.
    df = load_enriched(columns=[
        "N° SIRE", "AGE", "PLACE", "in_cohorte_T1", "est_libelle_poney",
        "in_fenetre_4_7", "NUMERO_EVENEMENT2", "NUMEROSEQUENCE",
        "DISCIPLINE_CODE"
    ])
    df = df[df["DISCIPLINE_CODE"] == "SO"].copy()
    df["AGE"] = df["AGE"].astype("Int64")
    print(f"Lignes SO totales : {len(df):,}")

    # Identifier les codes et les vraies places
    df["is_code"] = df["PLACE"] >= PLACE_MAX_VRAIE  # codes >= 800
    df["place_valide"] = ~df["is_code"] & df["PLACE"].notna()
    df["epreuve_id"] = (df["NUMERO_EVENEMENT2"].astype(str)
                        + "_" + df["NUMEROSEQUENCE"].astype(str))

    # Calcul nb_partants_total et nb_finishers par épreuve
    print("\nCalcul nb_partants et nb_finishers par épreuve...")
    nb_partants = df.groupby("epreuve_id").size().rename("nb_partants_total")
    nb_finishers = (df[df["place_valide"]]
                    .groupby("epreuve_id").size()
                    .rename("nb_finishers"))
    df = df.merge(nb_partants, on="epreuve_id", how="left")
    df = df.merge(nb_finishers, on="epreuve_id", how="left")

    # Calcul percentiles par ligne (uniquement sur les vraies places)
    df["percentile_partants"] = np.where(
        df["place_valide"],
        df["PLACE"] / df["nb_partants_total"],
        np.nan
    )
    df["percentile_finishers"] = np.where(
        df["place_valide"] & (df["nb_finishers"] > 0),
        df["PLACE"] / df["nb_finishers"],
        np.nan
    )

    # Restriction cohorte + fenêtre 4-7
    mask = (df["in_cohorte_T1"]
            & ~df["est_libelle_poney"]
            & df["IDCHEVAL"].isin(cohorte_ids)
            & df["in_fenetre_4_7"])
    d = df[mask].copy()
    print(f"Lignes cohorte fenêtre 4-7 : {len(d):,}")

    # Préparer toutes les colonnes booléennes AVANT de construire granularites
    d["is_victoire"] = (d["PLACE"] == 1)
    # is_code est déjà créée plus haut, héritée de df

    print("\nCalcul des features...")

    # On va construire feat_dict = {feature_name: pd.Series indexée par IDCHEVAL}
    feat_dict = {}

    # Pour chaque granularité temporelle, on calcule un sous-DataFrame
    # 'global', 'age_4', 'age_5', 'age_6', 'age_7'
    # Puis on calcule les deltas (5_4, 6_5, 7_6)
    granularites = {
        "4_7": d,
        "4ans": d[d["AGE"] == 4],
        "5ans": d[d["AGE"] == 5],
        "6ans": d[d["AGE"] == 6],
        "7ans": d[d["AGE"] == 7],
    }

    # ==========================================
    # Sous-famille NON-CLASSEMENT (8 features)
    # ==========================================
    print("  Non-classement...")
    for gran_name, sub in granularites.items():
        total = sub.groupby("IDCHEVAL").size()
        nc = sub[sub["is_code"]].groupby("IDCHEVAL").size()
        nc = nc.reindex(total.index, fill_value=0)
        feat_dict[f"f3_taux_non_classement_{gran_name}"] = (nc / total)
    # Deltas
    for from_age, to_age in [(4, 5), (5, 6), (6, 7)]:
        feat_dict[f"f3_evolution_nc_{to_age}_{from_age}"] = (
            feat_dict[f"f3_taux_non_classement_{to_age}ans"]
            - feat_dict[f"f3_taux_non_classement_{from_age}ans"]
        )

    # ==========================================
    # Sous-famille VICTOIRES (17 features)
    # ==========================================
    print("  Victoires...")
    # is_victoire est déjà créée avant le dict granularites
    # Global et par âge : compte + taux
    for gran_name, sub in granularites.items():
        nb_vic = sub[sub["is_victoire"]].groupby("IDCHEVAL").size()
        total = sub.groupby("IDCHEVAL").size()
        nb_vic = nb_vic.reindex(total.index, fill_value=0)
        feat_dict[f"f3_nb_victoires_{gran_name}"] = nb_vic
        feat_dict[f"f3_taux_victoires_{gran_name}"] = nb_vic / total
    # Booléen : au moins 1 victoire 4-7
    feat_dict["f3_a_au_moins_une_victoire_4_7"] = (
        feat_dict["f3_nb_victoires_4_7"] >= 1
    )
    # Deltas nb_victoires
    for from_age, to_age in [(4, 5), (5, 6), (6, 7)]:
        feat_dict[f"f3_evolution_victoires_{to_age}_{from_age}"] = (
            feat_dict[f"f3_nb_victoires_{to_age}ans"]
            - feat_dict[f"f3_nb_victoires_{from_age}ans"]
        )
    # Deltas taux_victoires (3 features oubliées initialement)
    for from_age, to_age in [(4, 5), (5, 6), (6, 7)]:
        feat_dict[f"f3_evolution_taux_victoires_{to_age}_{from_age}"] = (
            feat_dict[f"f3_taux_victoires_{to_age}ans"]
            - feat_dict[f"f3_taux_victoires_{from_age}ans"]
        )

    # ==========================================
    # Sous-famille PERCENTILE (64 features)
    # 4 stats (mediane, min, max, std) × 2 versions × 8 granularités
    # ==========================================
    print("  Percentile (médiane/min/max/std × partants/finishers)...")
    STATS = {"median": "median", "min": "min", "max": "max", "std": "std"}
    for version in ["partants", "finishers"]:
        col_pct = f"percentile_{version}"
        for gran_name, sub in granularites.items():
            g = sub.dropna(subset=[col_pct]).groupby("IDCHEVAL")[col_pct]
            for stat_name in STATS:
                feat_dict[f"f3_percentile_{version}_{stat_name}_{gran_name}"] = (
                    getattr(g, stat_name)()
                )
        # Deltas (pour chaque stat)
        for stat_name in STATS:
            for from_age, to_age in [(4, 5), (5, 6), (6, 7)]:
                key_from = f"f3_percentile_{version}_{stat_name}_{from_age}ans"
                key_to = f"f3_percentile_{version}_{stat_name}_{to_age}ans"
                feat_dict[f"f3_evolution_percentile_{version}_{stat_name}_{to_age}_{from_age}"] = (
                    feat_dict[key_to] - feat_dict[key_from]
                )

    # ==========================================
    # Sous-famille TOP X% (64 features)
    # 4 seuils × 2 versions × 8 granularités
    # ==========================================
    print("  Top X% (5/10/25/50 × partants/finishers)...")
    for version in ["partants", "finishers"]:
        col_pct = f"percentile_{version}"
        # Pour chaque ligne, flag top X% (sur lignes avec percentile valide)
        for seuil in TOP_X_SEUILS:
            seuil_label = f"{int(seuil * 100)}pct"
            flag_col = f"_is_top_{version}_{seuil_label}"
            d[flag_col] = d[col_pct] <= seuil
            for gran_name, sub in granularites.items():
                # Reconstruire sub avec la colonne flag
                sub_with_flag = sub.copy()
                sub_with_flag[flag_col] = sub[col_pct] <= seuil
                # Taux = nb top X% / nb lignes avec percentile valide
                valid = sub_with_flag.dropna(subset=[col_pct])
                count_valid = valid.groupby("IDCHEVAL").size()
                count_top = valid[valid[flag_col]].groupby("IDCHEVAL").size()
                count_top = count_top.reindex(count_valid.index, fill_value=0)
                feat_dict[f"f3_taux_top_{seuil_label}_{version}_{gran_name}"] = (
                    count_top / count_valid
                )
            # Deltas
            for from_age, to_age in [(4, 5), (5, 6), (6, 7)]:
                key_from = f"f3_taux_top_{seuil_label}_{version}_{from_age}ans"
                key_to = f"f3_taux_top_{seuil_label}_{version}_{to_age}ans"
                feat_dict[f"f3_evolution_top_{seuil_label}_{version}_{to_age}_{from_age}"] = (
                    feat_dict[key_to] - feat_dict[key_from]
                )

    # ==========================================
    # Concaténation
    # ==========================================
    print("\nConcaténation...")
    features = pd.DataFrame(feat_dict).reset_index()
    features = features.rename(columns={"index": "IDCHEVAL"})

    print(f"Nb chevaux : {len(features):,}")
    print(f"Nb colonnes features : {len(features.columns) - 1}")

    sanity_check(features, family_id=3, expected_n_features=153)
    save_intermediate(features, family_id=3, family_name="performance")


if __name__ == "__main__":
    main()
