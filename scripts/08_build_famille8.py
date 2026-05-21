"""
08 - Famille 8 : Pedigree (Père + Grand-père maternel).

Génère 8 features de target encoding LOO + smoothing Bayésien :
  - 4 sur le PÈRE (cible, gains, percentile partants, percentile finishers)
  - 4 sur le GRAND-PÈRE MATERNEL (idem)

Mère écartée (couverture LOO trop faible : 2,7% fiables).

NOTE IMPORTANTE : version naïve sur TOUTE la cohorte. À recalculer
proprement avec split train/test à la phase modélisation.

Entrée : data/ffe_2010-2025_enriched.parquet + cible.parquet
Sortie : data/master/intermediates/famille8_pedigree.parquet
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (load_enriched, get_cohorte_t1_n1_10,
                   save_intermediate, sanity_check,
                   INTERMEDIATES_DIR)

K_SMOOTHING = 30


def target_encoding_loo(df_cohorte, key_col, target_col, k=K_SMOOTHING):
    """Target encoding LOO avec smoothing Bayésien."""
    valid = df_cohorte.dropna(subset=[key_col, target_col]).copy()
    grp = valid.groupby(key_col)[target_col]
    sum_grp = grp.sum()
    count_grp = grp.count()
    moyenne_globale = valid[target_col].mean()
    valid["sum_grp"] = valid[key_col].map(sum_grp)
    valid["count_grp"] = valid[key_col].map(count_grp)
    valid["n_autres"] = valid["count_grp"] - 1
    valid["sum_autres"] = valid["sum_grp"] - valid[target_col]
    valid["mean_loo"] = np.where(
        valid["n_autres"] > 0,
        valid["sum_autres"] / valid["n_autres"], np.nan)
    valid["te_lisse"] = np.where(
        valid["n_autres"] > 0,
        (valid["n_autres"] * valid["mean_loo"] + k * moyenne_globale)
        / (valid["n_autres"] + k),
        moyenne_globale)
    result = pd.Series(np.nan, index=df_cohorte.index)
    result.loc[valid.index] = valid["te_lisse"].values
    return result


def main():
    print("=== 08 - Build Famille 8 (Pedigree) ===\n")
    print("⚠ Version naïve : TE calculé sur TOUTE la cohorte (sans split).\n")

    cohorte_ids = get_cohorte_t1_n1_10(min_participations=10)
    cible = pd.read_parquet(INTERMEDIATES_DIR / "cible.parquet")
    cible = cible.set_index("IDCHEVAL")
    cible.columns = ["cible"]

    df = load_enriched(columns=[
        "N° SIRE", "NUMSIREPERE", "NUMSIREPEREMERE",
        "GAINS", "PLACE",
        "in_cohorte_T1", "est_libelle_poney",
        "NUMERO_EVENEMENT2", "NUMEROSEQUENCE",
        "DISCIPLINE_CODE"
    ])
    df = df[df["DISCIPLINE_CODE"] == "SO"].copy()

    # Percentiles par ligne
    PLACE_MAX_VRAIE = 800
    df["place_valide"] = (df["PLACE"] < PLACE_MAX_VRAIE) & df["PLACE"].notna()
    df["epreuve_id"] = (df["NUMERO_EVENEMENT2"].astype(str) + "_"
                        + df["NUMEROSEQUENCE"].astype(str))
    nb_p = df.groupby("epreuve_id").size().rename("nb_partants_total")
    nb_f = (df[df["place_valide"]].groupby("epreuve_id").size()
            .rename("nb_finishers"))
    df = df.merge(nb_p, on="epreuve_id", how="left")
    df = df.merge(nb_f, on="epreuve_id", how="left")
    df["pct_p"] = np.where(df["place_valide"],
                            df["PLACE"] / df["nb_partants_total"], np.nan)
    df["pct_f"] = np.where(df["place_valide"] & (df["nb_finishers"] > 0),
                            df["PLACE"] / df["nb_finishers"], np.nan)

    # Agrégations niveau cheval (toute carrière)
    print("Agrégation niveau cheval (toute carrière)...")
    gains_car = df.groupby("IDCHEVAL")["GAINS"].sum()
    pct_p_med = (df.dropna(subset=["pct_p"]).groupby("IDCHEVAL")["pct_p"]
                 .median())
    pct_f_med = (df.dropna(subset=["pct_f"]).groupby("IDCHEVAL")["pct_f"]
                 .median())

    # Pedigree par cheval
    pedigree = df.drop_duplicates("IDCHEVAL").set_index("IDCHEVAL")[
        ["NUMSIREPERE", "NUMSIREPEREMERE"]]

    chev = pd.DataFrame(index=list(cohorte_ids))
    chev.index.name = "IDCHEVAL"
    chev["NUMSIREPERE"] = pedigree["NUMSIREPERE"]
    chev["NUMSIREPEREMERE"] = pedigree["NUMSIREPEREMERE"]
    chev["cible"] = cible["cible"]
    chev["gains_carriere"] = gains_car
    chev["pct_p_med"] = pct_p_med
    chev["pct_f_med"] = pct_f_med

    print(f"Nb chevaux cohorte : {len(chev):,}")
    print(f"  Avec cible : {chev['cible'].notna().sum():,}")
    print(f"  Avec père : {chev['NUMSIREPERE'].notna().sum():,}")
    print(f"  Avec GP maternel : {chev['NUMSIREPEREMERE'].notna().sum():,}")

    print("\nCalcul target encodings...")
    feat = {}
    # Père
    feat["f8_pere_target_encoded_LOO"] = target_encoding_loo(
        chev, "NUMSIREPERE", "cible")
    feat["f8_pere_mean_gains_LOO"] = target_encoding_loo(
        chev, "NUMSIREPERE", "gains_carriere")
    feat["f8_pere_mean_percentile_partants_LOO"] = target_encoding_loo(
        chev, "NUMSIREPERE", "pct_p_med")
    feat["f8_pere_mean_percentile_finishers_LOO"] = target_encoding_loo(
        chev, "NUMSIREPERE", "pct_f_med")
    # Grand-père maternel
    feat["f8_gp_maternel_target_encoded_LOO"] = target_encoding_loo(
        chev, "NUMSIREPEREMERE", "cible")
    feat["f8_gp_maternel_mean_gains_LOO"] = target_encoding_loo(
        chev, "NUMSIREPEREMERE", "gains_carriere")
    feat["f8_gp_maternel_mean_percentile_partants_LOO"] = target_encoding_loo(
        chev, "NUMSIREPEREMERE", "pct_p_med")
    feat["f8_gp_maternel_mean_percentile_finishers_LOO"] = target_encoding_loo(
        chev, "NUMSIREPEREMERE", "pct_f_med")

    features = pd.DataFrame(feat).reset_index()
    sanity_check(features, family_id=8, expected_n_features=8)
    save_intermediate(features, family_id=8, family_name="pedigree")


if __name__ == "__main__":
    main()
