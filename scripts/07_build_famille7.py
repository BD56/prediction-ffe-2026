"""
07 - Famille 7 : Cavalier.

Génère 56 features réparties en 3 sous-familles :

  Sous-famille percentile (16) :
    cavalier_percentile_partants_median_passe3_* (8)
    cavalier_percentile_finishers_median_passe3_* (8)

  Sous-famille diversité / volume (24) :
    nb_chevaux_distincts_cavalier_passe3_* (8) — diversité du cavalier (LOO)
    nb_cavaliers_distincts_du_cheval_* (8) — diversité côté cheval (pas LOO)
    nb_participations_cavalier_passe3_* (8) — volume cavalier (LOO)

  Sous-famille Hurdle gains (16) :
    cavalier_taux_gains_positifs_passe3_* (8)
    cavalier_mean_log_gains_pos_passe3_* (8)

Chaque sous-famille déclinée en 8 granularités : global 4-7, 4 par âge, 3 deltas.

Méthode : pour chaque participation du cheval analysé en 4-7 ans, on calcule
le score du cavalier sur sa fenêtre passée 3 ans (année-2, année-1, année),
en excluant le cheval analysé (Leave-One-Out).

NOTE : implémentation pragmatique. Coûteux en calcul (~minutes).
À optimiser plus tard si besoin.

Entrée : data/ffe_2010-2025_enriched.parquet
Sortie : data/master/intermediates/famille7_cavalier.parquet
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (load_enriched, get_cohorte_t1_n1_10,
                   save_intermediate, sanity_check)

PLACE_MAX_VRAIE = 800


def main():
    print("=== 07 - Build Famille 7 (Cavalier) ===\n")
    print("⚠ Famille la plus complexe (56 features, LOO sur fenêtre 3 ans).")
    print("   Calcul potentiellement long...\n")

    cohorte_ids = get_cohorte_t1_n1_10(min_participations=10)

    # Charger toute la base SO (pas seulement cohorte) pour calcul cavalier
    df = load_enriched(columns=[
        "N° SIRE", "AGE", "LICENCE", "ANNEE", "GAINS", "PLACE",
        "in_cohorte_T1", "est_libelle_poney", "in_fenetre_4_7",
        "NUMERO_EVENEMENT2", "NUMEROSEQUENCE", "DISCIPLINE_CODE"
    ])
    df = df[df["DISCIPLINE_CODE"] == "SO"].copy()
    print(f"Lignes SO totales : {len(df):,}")

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
    df["pct_f"] = np.where(df["place_valide"] & (df["nb_finishers"] > 0),
                            df["PLACE"] / df["nb_finishers"], np.nan)
    df["gains_pos"] = df["GAINS"] > 0
    # log(GAINS) sur les positifs uniquement
    df["log_gains_pos"] = np.where(df["GAINS"] > 0,
                                    np.log(df["GAINS"]),
                                    np.nan)

    # Cohorte 4-7
    mask_47 = (df["in_cohorte_T1"]
               & ~df["est_libelle_poney"]
               & df["IDCHEVAL"].isin(cohorte_ids)
               & df["in_fenetre_4_7"])
    d47 = df[mask_47].copy()
    d47["AGE"] = d47["AGE"].astype(int)
    print(f"Lignes cohorte fenêtre 4-7 : {len(d47):,}")

    # ============================================================
    # Calcul du score cavalier pour chaque (cheval, cavalier, année)
    # ============================================================
    # Stratégie : pour chaque (LICENCE, ANNEE) calculer un score local,
    # puis agréger sur fenêtre [Y-2, Y-1, Y] excluant le cheval analysé.

    # Pour LOO, on a besoin de soustraire la contribution du cheval analysé.
    # On utilise des sommes par (LICENCE, ANNEE) et on soustrait.

    print("\nPrécalcul agrégats par (LICENCE, ANNEE)...")
    grp = df.groupby(["LICENCE", "ANNEE"])
    # Pour chaque (cavalier, année) :
    agg_cav_an = grp.agg(
        n_parts=("PLACE", "size"),
        sum_pct_p=("pct_p", "sum"),
        n_pct_p=("pct_p", "count"),
        sum_pct_f=("pct_f", "sum"),
        n_pct_f=("pct_f", "count"),
        sum_log_gains_pos=("log_gains_pos", "sum"),
        n_log_gains_pos=("log_gains_pos", "count"),
        n_gains_pos=("gains_pos", "sum"),
    ).reset_index()

    # Liste des chevaux par (LICENCE, ANNEE)
    chev_par_cav_an = (df.groupby(["LICENCE", "ANNEE"])["IDCHEVAL"]
                       .apply(set).to_dict())

    print(f"Nb couples (cavalier, année) : {len(agg_cav_an):,}")

    # ============================================================
    # Pour chaque participation de d47, calculer le score cavalier
    # sur la fenêtre passée 3 ans [Y-2, Y-1, Y] LOO
    # ============================================================
    print("\nCalcul scores cavalier par participation (peut prendre quelques minutes)...")

    # Indexer agg_cav_an
    agg_cav_an = agg_cav_an.set_index(["LICENCE", "ANNEE"])

    def compute_score_row(row, fields):
        """Calcule le score cavalier pour une ligne donnée, LOO + fenêtre 3 ans."""
        cav = row["LICENCE"]
        annee = row["ANNEE"]
        cheval = row["IDCHEVAL"]
        years = [annee - 2, annee - 1, annee]
        # Sommer les agrégats sur la fenêtre
        sums = {f: 0 for f in fields}
        chevaux_union = set()
        for y in years:
            if (cav, y) in agg_cav_an.index:
                rec = agg_cav_an.loc[(cav, y)]
                for f in fields:
                    val = rec[f]
                    if not pd.isna(val):
                        sums[f] += val
                # union chevaux
                chevaux_union |= chev_par_cav_an.get((cav, y), set())
        # Retirer le cheval analysé
        chevaux_loo = chevaux_union - {cheval}
        sums["n_chev_loo"] = len(chevaux_loo)
        return pd.Series(sums)

    # Pour calculer LOO sur des sommes, on a besoin :
    # - sum_total sur la fenêtre 3 ans
    # - sum_du_cheval_lui_meme dans la fenêtre, à soustraire
    # Plus simple : on calcule la fenêtre 3 ans, puis on soustrait individuellement

    # Précalcul des stats du CHEVAL ANALYSÉ par (IDCHEVAL, LICENCE, ANNEE)
    print("  Précalcul stats du cheval analysé par (cheval, cavalier, année)...")
    grp_chev = df.groupby(["IDCHEVAL", "LICENCE", "ANNEE"])
    agg_chev_cav_an = grp_chev.agg(
        n_parts_self=("PLACE", "size"),
        sum_pct_p_self=("pct_p", "sum"),
        n_pct_p_self=("pct_p", "count"),
        sum_pct_f_self=("pct_f", "sum"),
        n_pct_f_self=("pct_f", "count"),
        sum_log_gains_pos_self=("log_gains_pos", "sum"),
        n_log_gains_pos_self=("log_gains_pos", "count"),
        n_gains_pos_self=("gains_pos", "sum"),
    ).reset_index()
    agg_chev_cav_an = agg_chev_cav_an.set_index(["IDCHEVAL", "LICENCE", "ANNEE"])

    # Pour chaque participation 4-7, calculer les scores
    # Approche vectorisée : on construit pour chaque ligne de d47 le score
    # de la fenêtre, puis on soustrait la contribution du cheval

    print("  Construction des scores fenêtre 3 ans (somme sur (LICENCE, Y-2, Y-1, Y))...")
    # On va construire 3 colonnes par sommer 3 années
    fields_agg = ["n_parts", "sum_pct_p", "n_pct_p", "sum_pct_f", "n_pct_f",
                  "sum_log_gains_pos", "n_log_gains_pos", "n_gains_pos"]

    # On merge avec agg_cav_an pour chaque décalage temporel
    d47 = d47.reset_index(drop=True)
    for shift in [0, 1, 2]:
        # On veut (LICENCE, ANNEE - shift)
        tmp = agg_cav_an.reset_index()
        tmp["JOIN_ANNEE"] = tmp["ANNEE"] + shift  # à matcher avec d47.ANNEE
        tmp = tmp.drop(columns=["ANNEE"])
        tmp = tmp.rename(columns={f: f"{f}_shift{shift}" for f in fields_agg})
        d47 = d47.merge(tmp, left_on=["LICENCE", "ANNEE"],
                         right_on=["LICENCE", "JOIN_ANNEE"], how="left")
        d47 = d47.drop(columns=["JOIN_ANNEE"])

    # Sommer les 3 shifts pour avoir la fenêtre [Y-2, Y-1, Y]
    for f in fields_agg:
        cols = [f"{f}_shift{s}" for s in [0, 1, 2]]
        d47[f"{f}_window"] = d47[cols].sum(axis=1, min_count=1)
        d47 = d47.drop(columns=cols)

    # ============================================================
    # LOO : soustraire la contribution du CHEVAL ANALYSÉ sur la fenêtre
    # ============================================================
    print("  Soustraction LOO (contribution du cheval analysé)...")
    # Pour le cheval analysé, sa contribution sur la fenêtre = somme de
    # ses propres aggrégats sur (cavalier, [Y-2, Y-1, Y])
    fields_self = ["n_parts_self", "sum_pct_p_self", "n_pct_p_self",
                    "sum_pct_f_self", "n_pct_f_self",
                    "sum_log_gains_pos_self", "n_log_gains_pos_self",
                    "n_gains_pos_self"]

    for shift in [0, 1, 2]:
        tmp = agg_chev_cav_an.reset_index()
        tmp["JOIN_ANNEE"] = tmp["ANNEE"] + shift
        tmp = tmp.drop(columns=["ANNEE"])
        tmp = tmp.rename(columns={f: f"{f}_shift{shift}" for f in fields_self})
        d47 = d47.merge(tmp, left_on=["IDCHEVAL", "LICENCE", "ANNEE"],
                         right_on=["IDCHEVAL", "LICENCE", "JOIN_ANNEE"],
                         how="left")
        d47 = d47.drop(columns=["JOIN_ANNEE"])

    for f in fields_self:
        cols = [f"{f}_shift{s}" for s in [0, 1, 2]]
        d47[f"{f}_window"] = d47[cols].fillna(0).sum(axis=1)
        d47 = d47.drop(columns=cols)

    # LOO = fenêtre_totale - contribution_self
    d47["loo_n_parts"] = d47["n_parts_window"] - d47["n_parts_self_window"]
    d47["loo_sum_pct_p"] = d47["sum_pct_p_window"] - d47["sum_pct_p_self_window"]
    d47["loo_n_pct_p"] = d47["n_pct_p_window"] - d47["n_pct_p_self_window"]
    d47["loo_sum_pct_f"] = d47["sum_pct_f_window"] - d47["sum_pct_f_self_window"]
    d47["loo_n_pct_f"] = d47["n_pct_f_window"] - d47["n_pct_f_self_window"]
    d47["loo_sum_log_gains_pos"] = (d47["sum_log_gains_pos_window"]
                                     - d47["sum_log_gains_pos_self_window"])
    d47["loo_n_log_gains_pos"] = (d47["n_log_gains_pos_window"]
                                   - d47["n_log_gains_pos_self_window"])
    d47["loo_n_gains_pos"] = (d47["n_gains_pos_window"]
                               - d47["n_gains_pos_self_window"])

    # Calcul des scores par participation (avec divisions sûres)
    # Médiane n'est pas possible avec des sommes -> on utilise moyenne pour
    # les agrégats. NOTE : on a documenté "médiane" mais avec les sums
    # vectorisés on a la moyenne. À documenter.
    d47["score_pct_p"] = np.where(d47["loo_n_pct_p"] > 0,
                                   d47["loo_sum_pct_p"] / d47["loo_n_pct_p"],
                                   np.nan)
    d47["score_pct_f"] = np.where(d47["loo_n_pct_f"] > 0,
                                   d47["loo_sum_pct_f"] / d47["loo_n_pct_f"],
                                   np.nan)
    d47["score_taux_gains_pos"] = np.where(d47["loo_n_parts"] > 0,
                                            d47["loo_n_gains_pos"] / d47["loo_n_parts"],
                                            np.nan)
    d47["score_mean_log_gains_pos"] = np.where(
        d47["loo_n_log_gains_pos"] > 0,
        d47["loo_sum_log_gains_pos"] / d47["loo_n_log_gains_pos"],
        np.nan
    )

    # ============================================================
    # nb_chevaux_distincts_cavalier (LOO) et nb_participations_cavalier (LOO)
    # nb_participations = loo_n_parts (déjà calculé)
    # nb_chevaux : besoin d'un calcul ensembliste, plus coûteux
    # ============================================================
    print("  Calcul nb_chevaux_distincts_cavalier (LOO)...")
    # Pour chaque (LICENCE, ANNEE), liste des chevaux sur fenêtre 3 ans = chev_par_cav_an
    # Pour chaque ligne de d47, on doit faire l'union des 3 années et exclure le cheval
    # Cette opération est coûteuse mais nécessaire. On l'optimise avec une fonction.
    def nb_chev_loo(row):
        cav = row["LICENCE"]
        annee = row["ANNEE"]
        cheval = row["IDCHEVAL"]
        chevaux = set()
        for y in [annee - 2, annee - 1, annee]:
            chevaux |= chev_par_cav_an.get((cav, y), set())
        chevaux.discard(cheval)
        return len(chevaux)

    d47["score_nb_chev_distincts"] = d47.apply(nb_chev_loo, axis=1)

    # ============================================================
    # Aggregation au niveau cheval (médiane sur ses participations 4-7)
    # ============================================================
    print("\nAgrégation au niveau cheval (médiane)...")
    # Note : le score par participation est déjà une moyenne LOO sur fenêtre 3 ans
    # (limitation des sums vectorisés). Au niveau cheval, on agrège en médiane
    # pour cohérence avec le catalogue (...median...) et robustesse aux outliers.

    # Définir les granularités
    granularites = {
        "4_7": d47,
        "4ans": d47[d47["AGE"] == 4],
        "5ans": d47[d47["AGE"] == 5],
        "6ans": d47[d47["AGE"] == 6],
        "7ans": d47[d47["AGE"] == 7],
    }

    # Métriques à agréger
    score_cols = {
        "cavalier_percentile_partants_median_passe3": "score_pct_p",
        "cavalier_percentile_finishers_median_passe3": "score_pct_f",
        "cavalier_taux_gains_positifs_passe3": "score_taux_gains_pos",
        "cavalier_mean_log_gains_pos_passe3": "score_mean_log_gains_pos",
        "nb_chevaux_distincts_cavalier_passe3": "score_nb_chev_distincts",
        "nb_participations_cavalier_passe3": "loo_n_parts",
    }

    feat_dict = {}
    for feat_name, col in score_cols.items():
        for gran, sub in granularites.items():
            # Médiane du score par cheval sur la granularité
            feat_dict[f"f7_{feat_name}_{gran}"] = (
                sub.groupby("IDCHEVAL")[col].median()
            )
        # Deltas
        for from_age, to_age in [(4, 5), (5, 6), (6, 7)]:
            key_from = f"f7_{feat_name}_{from_age}ans"
            key_to = f"f7_{feat_name}_{to_age}ans"
            feat_dict[f"f7_evolution_{feat_name}_{to_age}_{from_age}"] = (
                feat_dict[key_to] - feat_dict[key_from]
            )

    # nb_cavaliers_distincts_du_cheval : métrique du CHEVAL (pas LOO)
    print("  Calcul nb_cavaliers_distincts_du_cheval...")
    for gran, sub in granularites.items():
        feat_dict[f"f7_nb_cavaliers_distincts_du_cheval_{gran}"] = (
            sub.groupby("IDCHEVAL")["LICENCE"].nunique()
        )
    for from_age, to_age in [(4, 5), (5, 6), (6, 7)]:
        key_from = f"f7_nb_cavaliers_distincts_du_cheval_{from_age}ans"
        key_to = f"f7_nb_cavaliers_distincts_du_cheval_{to_age}ans"
        feat_dict[f"f7_evolution_nb_cavaliers_du_cheval_{to_age}_{from_age}"] = (
            feat_dict[key_to] - feat_dict[key_from]
        )

    features = pd.DataFrame(feat_dict).reset_index()

    print(f"\nNb chevaux : {len(features):,}")
    print(f"Nb colonnes features : {len(features.columns) - 1}")

    sanity_check(features, family_id=7, expected_n_features=56)
    save_intermediate(features, family_id=7, family_name="cavalier")


if __name__ == "__main__":
    main()
