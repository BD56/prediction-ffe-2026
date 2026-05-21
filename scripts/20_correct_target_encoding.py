"""
20 - Correction du target encoding (Familles 8 et 10) avec respect du split.

Recalcule les 12 features TE en ne fittant QUE sur le train.
Élimine le leakage train→test sur ces features dérivées de la cible.

Familles concernées :
  - F10 (Race) : 4 features
  - F8 (Pedigree) : 8 features

Entrée :
  - data/master/master_dataset_epure.parquet (master épuré)
  - data/ffe_2010-2025_enriched.parquet (pour recalculer)

Sortie :
  - data/master/master_dataset_clean.parquet (master avec TE corrigé)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_enriched, MASTER_DIR

K_SMOOTHING = 30
PLACE_MAX_VRAIE = 800


def fit_target_encoding(df_train, key_col, target_col, k=K_SMOOTHING):
    """Fit du target encoding LOO+smoothing sur le train.

    Returns dict {key_value: (sum, count)} pour transform ultérieur.
    """
    valid = df_train.dropna(subset=[key_col, target_col])
    grp = valid.groupby(key_col)[target_col]
    sum_grp = grp.sum().to_dict()
    count_grp = grp.count().to_dict()
    moyenne_globale = valid[target_col].mean()
    return {
        "sum": sum_grp,
        "count": count_grp,
        "moyenne_globale": moyenne_globale,
        "k": k,
    }


def transform_target_encoding(df, key_col, target_col, fit_params, is_train=True):
    """Transform target encoding.

    Pour le train : LOO (soustrait la contribution de chaque cheval).
    Pour valid/test : moyenne lissée sans LOO (le cheval n'est pas dans le fit).
    """
    sum_grp = fit_params["sum"]
    count_grp = fit_params["count"]
    mg = fit_params["moyenne_globale"]
    k = fit_params["k"]

    keys = df[key_col]
    targets = df[target_col] if target_col in df.columns else None

    result = pd.Series(np.nan, index=df.index)

    for idx, key in keys.items():
        if pd.isna(key) or key not in count_grp:
            result.loc[idx] = mg
            continue
        n = count_grp[key]
        s = sum_grp[key]
        if is_train and targets is not None and not pd.isna(targets.loc[idx]):
            # LOO : soustraire la contribution de ce cheval
            n_others = n - 1
            s_others = s - targets.loc[idx]
            if n_others > 0:
                mean_others = s_others / n_others
                result.loc[idx] = (n_others * mean_others + k * mg) / (n_others + k)
            else:
                result.loc[idx] = mg
        else:
            # Pas LOO pour valid/test : moyenne lissée standard
            mean_grp = s / n if n > 0 else mg
            result.loc[idx] = (n * mean_grp + k * mg) / (n + k)
    return result


def main():
    print("=== 20 - Correction Target Encoding (Familles 8 et 10) ===\n")

    # Chargement du master épuré
    master = pd.read_parquet(MASTER_DIR / "master_dataset_epure.parquet")
    master = master.set_index("IDCHEVAL")
    print(f"Master épuré : {len(master):,} chevaux × {len(master.columns)} colonnes")
    print(f"  Train : {(master['SPLIT'] == 'train').sum():,}")
    print(f"  Valid : {(master['SPLIT'] == 'valid').sum():,}")
    print(f"  Test  : {(master['SPLIT'] == 'test').sum():,}")

    # Identifier les features TE à corriger
    te_cols = [c for c in master.columns if "_LOO" in c or "target_encoded" in c]
    print(f"\nFeatures TE à corriger : {len(te_cols)}")
    for c in te_cols:
        print(f"  - {c}")

    # Charger données brutes pour recalculer
    df = load_enriched(columns=[
        "N° SIRE", "RACECHEVAL", "NUMSIREPERE", "NUMSIREPEREMERE",
        "GAINS", "PLACE", "in_cohorte_T1", "est_libelle_poney",
        "NUMERO_EVENEMENT2", "NUMEROSEQUENCE", "DISCIPLINE_CODE"
    ])
    df = df[df["DISCIPLINE_CODE"] == "SO"].copy()

    # Percentiles par ligne
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

    # Agrégations niveau cheval (toute carrière)
    gains_car = df.groupby("IDCHEVAL")["GAINS"].sum().rename("gains_total_carriere")
    pct_p_med = (df.dropna(subset=["pct_p"]).groupby("IDCHEVAL")["pct_p"].median()
                 .rename("pct_p_med"))

    # Récupérer race et pedigree par cheval
    info = df.drop_duplicates("IDCHEVAL").set_index("IDCHEVAL")[
        ["RACECHEVAL", "NUMSIREPERE", "NUMSIREPEREMERE"]]

    # Construire le DF niveau cheval avec cible + variables encodables
    chev = master[["hauteur_max_validee", "SPLIT"]].copy()
    chev["RACECHEVAL"] = info["RACECHEVAL"]
    chev["NUMSIREPERE"] = info["NUMSIREPERE"]
    chev["NUMSIREPEREMERE"] = info["NUMSIREPEREMERE"]
    chev["gains_total_carriere"] = gains_car
    chev["pct_p_med"] = pct_p_med

    train_mask = chev["SPLIT"] == "train"
    print(f"\nFit sur train ({train_mask.sum():,} chevaux)")

    # ============================================================
    # Famille 10 : Race
    # ============================================================
    print("\n--- Famille 10 (Race) ---")
    for var_name, target in [
        ("f10_race_target_encoded_LOO", "hauteur_max_validee"),
        ("f10_race_mean_gains_LOO", "gains_total_carriere"),
        ("f10_race_mean_percentile_partants_LOO", "pct_p_med"),
    ]:
        # Fit sur train
        fit = fit_target_encoding(chev[train_mask], "RACECHEVAL", target)
        # Transform : LOO sur train, normal sur valid/test
        chev[var_name + "_NEW"] = np.nan
        chev.loc[train_mask, var_name + "_NEW"] = transform_target_encoding(
            chev[train_mask], "RACECHEVAL", target, fit, is_train=True)
        chev.loc[~train_mask, var_name + "_NEW"] = transform_target_encoding(
            chev[~train_mask], "RACECHEVAL", target, fit, is_train=False)
        # Comparer avec l'ancien
        diff = (master[var_name] - chev[var_name + "_NEW"]).abs()
        print(f"  {var_name}: diff médiane = {diff.median():.6f}, max = {diff.max():.6f}")

    # Famille 10 : finishers (n'existe peut-être plus après épuration ; on vérifie)
    f_fin = "f10_race_mean_percentile_finishers_LOO"
    if f_fin in master.columns:
        # finishers fait pareil que partants si on l'a gardé
        df["pct_f"] = np.where(
            df["place_valide"] & (df["nb_finishers"] > 0),
            df["PLACE"] / df["nb_finishers"], np.nan)
        pct_f_med = (df.dropna(subset=["pct_f"]).groupby("IDCHEVAL")["pct_f"].median())
        chev["pct_f_med"] = pct_f_med
        fit = fit_target_encoding(chev[train_mask], "RACECHEVAL", "pct_f_med")
        chev[f_fin + "_NEW"] = np.nan
        chev.loc[train_mask, f_fin + "_NEW"] = transform_target_encoding(
            chev[train_mask], "RACECHEVAL", "pct_f_med", fit, is_train=True)
        chev.loc[~train_mask, f_fin + "_NEW"] = transform_target_encoding(
            chev[~train_mask], "RACECHEVAL", "pct_f_med", fit, is_train=False)
        print(f"  {f_fin}: corrigé")

    # ============================================================
    # Famille 8 : Pedigree (Père + GP maternel)
    # ============================================================
    print("\n--- Famille 8 (Pedigree) ---")
    for ancestor_col, prefix in [("NUMSIREPERE", "pere"),
                                  ("NUMSIREPEREMERE", "gp_maternel")]:
        for var_name, target in [
            (f"f8_{prefix}_target_encoded_LOO", "hauteur_max_validee"),
            (f"f8_{prefix}_mean_gains_LOO", "gains_total_carriere"),
            (f"f8_{prefix}_mean_percentile_partants_LOO", "pct_p_med"),
        ]:
            if var_name not in master.columns:
                continue  # peut avoir été supprimé par épuration
            fit = fit_target_encoding(chev[train_mask], ancestor_col, target)
            chev[var_name + "_NEW"] = np.nan
            chev.loc[train_mask, var_name + "_NEW"] = transform_target_encoding(
                chev[train_mask], ancestor_col, target, fit, is_train=True)
            chev.loc[~train_mask, var_name + "_NEW"] = transform_target_encoding(
                chev[~train_mask], ancestor_col, target, fit, is_train=False)
            diff = (master[var_name] - chev[var_name + "_NEW"]).abs()
            print(f"  {var_name}: diff médiane = {diff.median():.6f}, max = {diff.max():.6f}")

    # ============================================================
    # Construire le master clean
    # ============================================================
    master_clean = master.copy()
    for c in master.columns:
        if c + "_NEW" in chev.columns:
            master_clean[c] = chev[c + "_NEW"]

    out_path = MASTER_DIR / "master_dataset_clean.parquet"
    master_clean.reset_index().to_parquet(out_path)
    print(f"\n→ Master clean : {out_path}")
    print(f"   {len(master_clean):,} chevaux × {len(master_clean.columns)} colonnes")


if __name__ == "__main__":
    main()
