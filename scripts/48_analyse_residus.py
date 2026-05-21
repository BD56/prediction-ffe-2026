"""
48 - Analyse des résidus en 4 angles.

  Angle 1 : Diagnostic statistique (homoscédasticité, distribution, normalité)
  Angle 2 : Sous-groupes mal prédits (race_TE, année, activité)
  Angle 3 : Comparaison des erreurs entre modèles
  Angle 4 : Outliers et cas extrêmes

Modèles évalués : RF default, Hurdle (mélange), Stacking + Calib.

Sortie : data/master/residus_analyse_*.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostRegressor
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def main():
    print("=== 48 - Analyse des résidus en 4 angles ===\n")

    # ---------- Charger les données ----------
    v2 = pd.read_parquet(MASTER_DIR / "master_dataset_epure_v2.parquet")
    v2 = v2.set_index("IDCHEVAL")
    feat_cols = [c for c in v2.columns if c.startswith("f")]
    for c in feat_cols:
        if v2[c].dtype == bool:
            v2[c] = v2[c].astype(int)
    X = v2[feat_cols].select_dtypes(include=[np.number])
    y = v2["hauteur_max_validee"]
    split = v2["SPLIT"]
    annee = v2["DATENAISSANCE"].astype(int)

    X_train, y_train = X[split == "train"], y[split == "train"]
    X_valid, y_valid = X[split == "valid"], y[split == "valid"]
    X_test, y_test = X[split == "test"], y[split == "test"]
    annee_test = annee[split == "test"]

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xva = imp.transform(X_valid)
    Xte = imp.transform(X_test)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xva_sc = sc.transform(Xva)
    Xte_sc = sc.transform(Xte)

    # ---------- Fit des 3 modèles ----------
    print("Fit des modèles...")
    t0 = time.time()

    rf = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                 min_samples_split=10, max_features="sqrt",
                                 random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train)
    pred_rf = rf.predict(Xte)
    pred_rf_valid = rf.predict(Xva)

    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                      n_jobs=-1, random_state=42)
    en.fit(Xtr_sc, y_train)
    pred_en_valid = en.predict(Xva_sc)
    pred_en_test = en.predict(Xte_sc)

    cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3, random_seed=42, loss_function="MAE",
                            eval_metric="MAE", early_stopping_rounds=30, verbose=0)
    cb.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    pred_cb_valid = cb.predict(X_valid)
    pred_cb_test = cb.predict(X_test)

    # Hurdle
    y_train_bin = (y_train >= 1.40).astype(int)
    clf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                   min_samples_split=10, max_features="sqrt",
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(Xtr, y_train_bin)
    p_test = clf.predict_proba(Xte)[:, 1]
    mask = (y_train >= 1.40).values
    rf_tops = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                      min_samples_split=5, max_features="sqrt",
                                      random_state=42, n_jobs=-1)
    rf_tops.fit(Xtr[mask], y_train[mask])
    pred_tops = rf_tops.predict(Xte)
    pred_hurdle = p_test * pred_tops + (1 - p_test) * pred_rf

    # Stacking + Calib
    X_meta_v = np.column_stack([pred_rf_valid, pred_en_valid, pred_cb_valid])
    X_meta_t = np.column_stack([pred_rf, pred_en_test, pred_cb_test])
    meta = LinearRegression()
    meta.fit(X_meta_v, y_valid)
    pred_stack = meta.predict(X_meta_t)
    pred_stack_v = meta.predict(X_meta_v)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(pred_stack_v, y_valid)
    pred_stack_cal = cal.predict(pred_stack)

    print(f"  ✓ {time.time()-t0:.1f}s\n")

    # ---------- DataFrame des résidus ----------
    df = pd.DataFrame({
        "y_true": y_test.values,
        "pred_RF": pred_rf,
        "pred_Hurdle": pred_hurdle,
        "pred_Stack": pred_stack_cal,
        "annee": annee_test.values,
        "race_TE": X_test["f10_race_target_encoded_LOO"].values,
        "nb_part_7ans": X_test["f1_nb_participations_7ans"].fillna(0).values,
    }, index=y_test.index)
    df["res_RF"] = df["y_true"] - df["pred_RF"]
    df["res_Hurdle"] = df["y_true"] - df["pred_Hurdle"]
    df["res_Stack"] = df["y_true"] - df["pred_Stack"]
    df["res_RF_cm"] = df["res_RF"] * 100
    df["res_Hurdle_cm"] = df["res_Hurdle"] * 100
    df["res_Stack_cm"] = df["res_Stack"] * 100

    # ============================================================
    # ANGLE 1 — Diagnostic statistique
    # ============================================================
    print("=" * 70)
    print("ANGLE 1 — DIAGNOSTIC STATISTIQUE DES RÉSIDUS")
    print("=" * 70)

    print(f"\n{'Modèle':<15s} | {'moyenne':>9s} | {'médiane':>9s} | {'std':>6s} | "
          f"{'skew':>6s} | {'kurtosis':>9s} | {'biais':>7s}")
    print("-" * 80)
    for label, col in [("RF default", "res_RF_cm"),
                        ("Hurdle", "res_Hurdle_cm"),
                        ("Stacking+Calib", "res_Stack_cm")]:
        r = df[col]
        # Biais = % de prédictions sous-estimées (résidu > 0 = pred < vrai)
        biais = (r > 0).mean() * 100
        print(f"{label:<15s} | {r.mean():>+8.2f} | {r.median():>+8.2f} | {r.std():>5.2f} | "
              f"{r.skew():>+5.2f} | {r.kurt():>+8.2f} | {biais:>6.1f}%")

    # Test d'hétéroscédasticité (Breusch-Pagan léger : corrélation |résidu| vs prédiction)
    print(f"\n--- Test d'hétéroscédasticité (corr Pearson |résidu| vs prédiction) ---")
    print(f"{'Modèle':<15s} | {'corr':>7s} | {'interprétation'}")
    print("-" * 75)
    for label, res_col, pred_col in [("RF default", "res_RF_cm", "pred_RF"),
                                        ("Hurdle", "res_Hurdle_cm", "pred_Hurdle"),
                                        ("Stacking+Calib", "res_Stack_cm", "pred_Stack")]:
        c = stats.pearsonr(df[res_col].abs(), df[pred_col])
        interp = "hétéroscédastique" if abs(c.statistic) > 0.1 else "homoscédastique OK"
        print(f"{label:<15s} | {c.statistic:>+6.3f} | {interp} (p={c.pvalue:.2e})")

    # Distribution des résidus : tests de normalité (Shapiro pour échantillon < 5000, sinon D'Agostino)
    print(f"\n--- Tests de normalité des résidus ---")
    for label, col in [("RF default", "res_RF_cm"),
                        ("Hurdle", "res_Hurdle_cm"),
                        ("Stacking+Calib", "res_Stack_cm")]:
        r = df[col].dropna().values
        # D'Agostino-Pearson (test combinant skewness + kurtosis)
        stat, p = stats.normaltest(r)
        print(f"  {label:<15s} D'Agostino-Pearson p={p:.2e} "
              f"→ {'rejet normalité' if p < 0.01 else 'pas de rejet'}")

    # ============================================================
    # ANGLE 2 — Sous-groupes mal prédits
    # ============================================================
    print("\n" + "=" * 70)
    print("ANGLE 2 — SOUS-GROUPES MAL PRÉDITS")
    print("=" * 70)

    # 2.1 par tranche de race_TE (race plus ou moins prestigieuse)
    print("\n--- Par quintile de race_TE (race target-encoded) ---")
    df["race_TE_q"] = pd.qcut(df["race_TE"].rank(method="first"), q=5,
                                labels=["Q1 (basse)", "Q2", "Q3", "Q4", "Q5 (haute)"])
    print(f"{'Quintile':<15s} | {'n':>5s} | "
          f"{'MAE RF':>7s} | {'MAE Hurdle':>11s} | {'MAE Stack':>10s} | "
          f"{'biais RF':>9s} | {'biais Hurdle':>13s}")
    print("-" * 95)
    by_race = []
    for q in df["race_TE_q"].cat.categories:
        sub = df[df["race_TE_q"] == q]
        mae_rf = sub["res_RF_cm"].abs().mean()
        mae_hu = sub["res_Hurdle_cm"].abs().mean()
        mae_st = sub["res_Stack_cm"].abs().mean()
        bi_rf = sub["res_RF_cm"].mean()
        bi_hu = sub["res_Hurdle_cm"].mean()
        print(f"{str(q):<15s} | {len(sub):>5d} | "
              f"{mae_rf:>6.2f}c | {mae_hu:>10.2f}c | {mae_st:>9.2f}c | "
              f"{bi_rf:>+8.2f}c | {bi_hu:>+12.2f}c")
        by_race.append({"quintile": str(q), "n": len(sub),
                         "MAE_RF": mae_rf, "MAE_Hurdle": mae_hu, "MAE_Stack": mae_st,
                         "biais_RF": bi_rf, "biais_Hurdle": bi_hu})
    pd.DataFrame(by_race).to_csv(MASTER_DIR / "residus_par_race.csv", index=False)

    # 2.2 par année de naissance (drift résiduel)
    # En test on a seulement 2013, donc on regarde la sous-population valid + test ensemble pour avoir plus d'années
    print("\n--- Par année de naissance (test set = 2013 uniquement, donc sans variance ici) ---")
    print("  Note : tout le test est sur 2013, drift par année peu informatif sur ce split")
    print(f"  Médiane résidus 2013 : RF {df['res_RF_cm'].median():+.2f}c, "
          f"Hurdle {df['res_Hurdle_cm'].median():+.2f}c, Stack {df['res_Stack_cm'].median():+.2f}c")

    # 2.3 par quintile d'activité (nb_part_7ans)
    print("\n--- Par quintile de nb_participations_7ans (niveau d'activité) ---")
    df["activ_q"] = pd.qcut(df["nb_part_7ans"].rank(method="first"), q=5,
                              labels=["Q1 (faible)", "Q2", "Q3", "Q4", "Q5 (forte)"])
    print(f"{'Quintile':<15s} | {'n':>5s} | "
          f"{'MAE RF':>7s} | {'MAE Hurdle':>11s} | {'MAE Stack':>10s} | "
          f"{'biais RF':>9s} | {'biais Hurdle':>13s}")
    print("-" * 95)
    by_act = []
    for q in df["activ_q"].cat.categories:
        sub = df[df["activ_q"] == q]
        mae_rf = sub["res_RF_cm"].abs().mean()
        mae_hu = sub["res_Hurdle_cm"].abs().mean()
        mae_st = sub["res_Stack_cm"].abs().mean()
        bi_rf = sub["res_RF_cm"].mean()
        bi_hu = sub["res_Hurdle_cm"].mean()
        print(f"{str(q):<15s} | {len(sub):>5d} | "
              f"{mae_rf:>6.2f}c | {mae_hu:>10.2f}c | {mae_st:>9.2f}c | "
              f"{bi_rf:>+8.2f}c | {bi_hu:>+12.2f}c")
        by_act.append({"quintile": str(q), "n": len(sub),
                        "MAE_RF": mae_rf, "MAE_Hurdle": mae_hu, "MAE_Stack": mae_st,
                        "biais_RF": bi_rf, "biais_Hurdle": bi_hu})
    pd.DataFrame(by_act).to_csv(MASTER_DIR / "residus_par_activite.csv", index=False)

    # ============================================================
    # ANGLE 3 — Comparaison des erreurs entre modèles
    # ============================================================
    print("\n" + "=" * 70)
    print("ANGLE 3 — COMPARAISON DES ERREURS ENTRE MODÈLES")
    print("=" * 70)

    # Corrélation des résidus entre modèles
    print("\n--- Corrélation des résidus entre modèles ---")
    corrs = df[["res_RF_cm", "res_Hurdle_cm", "res_Stack_cm"]].corr().round(3)
    print(corrs.to_string())
    print("\n  Interprétation : forte corrélation = les modèles font les mêmes erreurs")
    print("                   faible corrélation = les modèles sont complémentaires")

    # Comptage par modèle : qui est le meilleur sur chaque cheval ?
    print("\n--- Pour chaque cheval, quel modèle est le meilleur ? ---")
    df["best_model"] = df[["res_RF_cm", "res_Hurdle_cm", "res_Stack_cm"]].abs().idxmin(axis=1)
    df["best_model"] = df["best_model"].str.replace("res_", "").str.replace("_cm", "")
    print(df["best_model"].value_counts().to_string())
    print(f"\n  → Stacking est meilleur sur {(df['best_model'] == 'Stack').mean()*100:.1f}% des chevaux")
    print(f"  → Hurdle    est meilleur sur {(df['best_model'] == 'Hurdle').mean()*100:.1f}% des chevaux")
    print(f"  → RF        est meilleur sur {(df['best_model'] == 'RF').mean()*100:.1f}% des chevaux")

    # Comptage par tranche : qui est meilleur sur les tops vs bas ?
    print("\n--- Modèle le meilleur par tranche réelle ---")
    df["tranche"] = pd.cut(df["y_true"], bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m", "1.35-1.40m", "≥1.45m"])
    ctab = pd.crosstab(df["tranche"], df["best_model"], normalize="index").round(3) * 100
    print(ctab.to_string())

    # Zones de désaccord : où Hurdle gagne vs perd contre les autres
    print("\n--- Cas où Hurdle gagne vs perd contre Stacking ---")
    df["hurdle_vs_stack"] = df["res_Hurdle_cm"].abs() - df["res_Stack_cm"].abs()
    # hurdle_vs_stack négatif = Hurdle meilleur ; positif = Stacking meilleur
    cas_hurdle_gagne = df[df["hurdle_vs_stack"] < -2]  # Hurdle meilleur d'au moins 2cm
    cas_stack_gagne = df[df["hurdle_vs_stack"] > 2]
    print(f"  Hurdle bien meilleur ({len(cas_hurdle_gagne):,} chevaux) : y_true moyen = {cas_hurdle_gagne['y_true'].mean():.3f}m")
    print(f"  Stacking bien meilleur ({len(cas_stack_gagne):,} chevaux) : y_true moyen = {cas_stack_gagne['y_true'].mean():.3f}m")

    # ============================================================
    # ANGLE 4 — Outliers et cas extrêmes
    # ============================================================
    print("\n" + "=" * 70)
    print("ANGLE 4 — OUTLIERS ET CAS EXTRÊMES")
    print("=" * 70)

    print("\n--- Top 10 SOUS-ESTIMATIONS Hurdle (pred << vrai) ---")
    worst_under = df.nlargest(10, "res_Hurdle_cm")[
        ["y_true", "pred_Hurdle", "pred_RF", "pred_Stack",
         "res_Hurdle_cm", "race_TE", "nb_part_7ans"]
    ]
    print(worst_under.round(3).to_string())

    print("\n--- Top 10 SUR-ESTIMATIONS Hurdle (pred >> vrai) ---")
    worst_over = df.nsmallest(10, "res_Hurdle_cm")[
        ["y_true", "pred_Hurdle", "pred_RF", "pred_Stack",
         "res_Hurdle_cm", "race_TE", "nb_part_7ans"]
    ]
    print(worst_over.round(3).to_string())

    # Stats sur les outliers (>3 std de résidus)
    seuil = 3 * df["res_Hurdle_cm"].std()
    n_outliers = (df["res_Hurdle_cm"].abs() > seuil).sum()
    print(f"\n--- Nombre d'outliers ({'résidu > 3σ = '}{seuil:.1f} cm) ---")
    print(f"  Hurdle : {n_outliers} chevaux ({n_outliers/len(df)*100:.1f}%)")
    print(f"  RF default : {(df['res_RF_cm'].abs() > 3*df['res_RF_cm'].std()).sum()} chevaux")
    print(f"  Stacking : {(df['res_Stack_cm'].abs() > 3*df['res_Stack_cm'].std()).sum()} chevaux")

    # Sauvegarde
    df_save = df.drop(columns=["res_RF", "res_Hurdle", "res_Stack"])
    df_save.to_csv(MASTER_DIR / "residus_complet.csv")
    print(f"\n→ residus_complet.csv ({len(df):,} chevaux), "
          f"residus_par_race.csv, residus_par_activite.csv")


if __name__ == "__main__":
    main()
