"""
10 - Famille 10 : Race.

Génère 4 features de target encoding LOO + smoothing Bayésien :
  - f10_race_target_encoded_LOO (sur la cible hauteur_max_validee)
  - f10_race_mean_gains_LOO (sur gains totaux carrière)
  - f10_race_mean_percentile_partants_LOO (sur percentile médian carrière)
  - f10_race_mean_percentile_finishers_LOO (sur percentile finishers médian carrière)

NOTE IMPORTANTE : pour cette version "naïve", le TE est calculé sur TOUTE
la cohorte (pas seulement le train). À recalculer proprement avec un split
train/valid/test (time series par génération) à la phase modélisation.

Formule smoothing Bayésien :
  TE_lissé(race, cheval) = (n_autres * mean_race_LOO + k * mean_globale)
                         / (n_autres + k)
  avec k = paramètre de lissage (à calibrer, défaut k=30)

Entrée : data/ffe_2010-2025_enriched.parquet + data/master/intermediates/cible.parquet
Sortie : data/master/intermediates/famille10_race.parquet
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
    """Target encoding leave-one-out avec smoothing Bayésien.

    df_cohorte : pd.DataFrame avec key_col et target_col, 1 ligne par cheval
    key_col : nom de la colonne catégorielle à encoder (ex RACECHEVAL)
    target_col : nom de la variable cible à moyenner (ex hauteur_max_validee)
    k : paramètre de lissage

    Returns pd.Series indexée par index de df_cohorte, avec la valeur TE lissée
    """
    valid = df_cohorte.dropna(subset=[key_col, target_col]).copy()

    # Stats par groupe
    grp = valid.groupby(key_col)[target_col]
    sum_grp = grp.sum()
    count_grp = grp.count()

    moyenne_globale = valid[target_col].mean()

    # Pour chaque ligne, LOO = (sum_grp - target) / (count_grp - 1)
    valid["sum_grp"] = valid[key_col].map(sum_grp)
    valid["count_grp"] = valid[key_col].map(count_grp)
    valid["n_autres"] = valid["count_grp"] - 1
    valid["sum_autres"] = valid["sum_grp"] - valid[target_col]

    # Moyenne LOO (NaN si n_autres == 0)
    valid["mean_loo"] = np.where(
        valid["n_autres"] > 0,
        valid["sum_autres"] / valid["n_autres"],
        np.nan
    )
    # Smoothing Bayésien : TE = (n * mean_loo + k * mean_globale) / (n + k)
    valid["te_lisse"] = np.where(
        valid["n_autres"] > 0,
        (valid["n_autres"] * valid["mean_loo"] + k * moyenne_globale)
        / (valid["n_autres"] + k),
        moyenne_globale  # fallback si race unique : moyenne globale
    )

    # Retour : Series indexée comme df_cohorte
    result = pd.Series(np.nan, index=df_cohorte.index)
    result.loc[valid.index] = valid["te_lisse"].values
    return result


def main():
    print("=== 10 - Build Famille 10 (Race) ===\n")
    print("⚠ Version naïve : TE calculé sur TOUTE la cohorte (sans split).")
    print("   À recalculer proprement avec split train/test en phase modélisation.\n")

    cohorte_ids = get_cohorte_t1_n1_10(min_participations=10)

    # Cible
    cible = pd.read_parquet(INTERMEDIATES_DIR / "cible.parquet")
    cible = cible.rename(columns={"IDCHEVAL": "IDCHEVAL",
                                  cible.columns[-1]: "cible"})

    # Charger 1 ligne par cheval avec race + variables à encoder
    df = load_enriched(columns=[
        "N° SIRE", "RACECHEVAL", "GAINS", "PLACE",
        "in_cohorte_T1", "est_libelle_poney",
        "NUMERO_EVENEMENT2", "NUMEROSEQUENCE",
        "DISCIPLINE_CODE"
    ])
    df = df[df["DISCIPLINE_CODE"] == "SO"].copy()
    print(f"Lignes SO totales : {len(df):,}")

    # Calculer percentiles par ligne pour pouvoir prendre la médiane carrière
    PLACE_MAX_VRAIE = 800
    df["place_valide"] = (df["PLACE"] < PLACE_MAX_VRAIE) & df["PLACE"].notna()
    df["epreuve_id"] = (df["NUMERO_EVENEMENT2"].astype(str)
                        + "_" + df["NUMEROSEQUENCE"].astype(str))
    nb_partants = df.groupby("epreuve_id").size().rename("nb_partants_total")
    nb_finishers = (df[df["place_valide"]]
                    .groupby("epreuve_id").size()
                    .rename("nb_finishers"))
    df = df.merge(nb_partants, on="epreuve_id", how="left")
    df = df.merge(nb_finishers, on="epreuve_id", how="left")
    df["percentile_partants"] = np.where(
        df["place_valide"], df["PLACE"] / df["nb_partants_total"], np.nan)
    df["percentile_finishers"] = np.where(
        df["place_valide"] & (df["nb_finishers"] > 0),
        df["PLACE"] / df["nb_finishers"], np.nan)

    # Agréger au niveau cheval (sur toute carrière)
    print("\nAgrégation niveau cheval (toute carrière)...")
    gains_carriere = df.groupby("IDCHEVAL")["GAINS"].sum().rename("gains_total_carriere")
    pct_partants_med = (df.dropna(subset=["percentile_partants"])
                        .groupby("IDCHEVAL")["percentile_partants"].median()
                        .rename("percentile_partants_median_carriere"))
    pct_finishers_med = (df.dropna(subset=["percentile_finishers"])
                         .groupby("IDCHEVAL")["percentile_finishers"].median()
                         .rename("percentile_finishers_median_carriere"))

    # Race par cheval
    race_par_cheval = df.drop_duplicates("IDCHEVAL").set_index("IDCHEVAL")["RACECHEVAL"]

    # Construire le dataframe niveau cheval pour la cohorte
    chev_cohorte = pd.DataFrame(index=list(cohorte_ids))
    chev_cohorte.index.name = "IDCHEVAL"
    chev_cohorte["RACECHEVAL"] = race_par_cheval
    chev_cohorte["cible"] = cible.set_index("IDCHEVAL")["cible"]
    chev_cohorte["gains_total_carriere"] = gains_carriere
    chev_cohorte["percentile_partants_median_carriere"] = pct_partants_med
    chev_cohorte["percentile_finishers_median_carriere"] = pct_finishers_med

    print(f"Nb chevaux cohorte : {len(chev_cohorte):,}")
    print(f"  Avec cible : {chev_cohorte['cible'].notna().sum():,}")
    print(f"  Avec race : {chev_cohorte['RACECHEVAL'].notna().sum():,}")

    print("\nCalcul target encodings LOO + smoothing (k=30)...")

    feat_dict = {}
    feat_dict["f10_race_target_encoded_LOO"] = target_encoding_loo(
        chev_cohorte, "RACECHEVAL", "cible")
    feat_dict["f10_race_mean_gains_LOO"] = target_encoding_loo(
        chev_cohorte, "RACECHEVAL", "gains_total_carriere")
    feat_dict["f10_race_mean_percentile_partants_LOO"] = target_encoding_loo(
        chev_cohorte, "RACECHEVAL", "percentile_partants_median_carriere")
    feat_dict["f10_race_mean_percentile_finishers_LOO"] = target_encoding_loo(
        chev_cohorte, "RACECHEVAL", "percentile_finishers_median_carriere")

    features = pd.DataFrame(feat_dict).reset_index()

    sanity_check(features, family_id=10, expected_n_features=4)
    save_intermediate(features, family_id=10, family_name="race")


if __name__ == "__main__":
    main()
