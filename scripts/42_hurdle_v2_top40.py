"""
42 - Hurdle v2 : même architecture que Hurdle d'origine, mais sur top 40 features.

Architecture :
  - Classifier RF (≥1,40m) avec class_weight="balanced"
  - Régresseur RF conditionnel (sur y >= 1,40m uniquement)
  - RF default pour les non-tops
  - Mélange : p * pred_tops + (1-p) * pred_rf

Comparaisons :
  - Hurdle original (156 features) → référence depuis recap
  - Hurdle v2 (top 40 features)
  - RF default sur top 40
  - Poly40 (rappel)

Sortie : data/master/hurdle_v2_top40_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def evaluate_all(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
    r2 = r2_score(y_true, y_pred)
    df = pd.DataFrame({"y": y_true.values, "p": y_pred})
    df["tranche"] = pd.cut(df["y"], bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])
    df["err_abs"] = (df["y"] - df["p"]).abs() * 100
    df["err_sq"] = (df["y"] - df["p"]) ** 2 * 10000
    mae_tr = df.groupby("tranche", observed=True)["err_abs"].mean().to_dict()
    rmse_tr = df.groupby("tranche", observed=True)["err_sq"].mean().apply(np.sqrt).to_dict()
    return mae, rmse, r2, mae_tr, rmse_tr


def main():
    print("=== 42 - Hurdle v2 sur top 40 features ===\n")

    # ---------- Données : top 40 features ----------
    imp = pd.read_csv(MASTER_DIR / "top_flop_v2_avec_valeurs.csv")
    top40 = imp.sort_values("rank_moyen").head(40)["feature"].tolist()
    v2 = pd.read_parquet(MASTER_DIR / "master_dataset_epure_v2.parquet")
    v2 = v2.set_index("IDCHEVAL")
    for c in top40:
        if v2[c].dtype == bool:
            v2[c] = v2[c].astype(int)
    X = v2[top40]
    y = v2["hauteur_max_validee"]
    split = v2["SPLIT"]

    X_train, y_train = X[split == "train"], y[split == "train"]
    X_test, y_test = X[split == "test"], y[split == "test"]
    print(f"Train : {len(X_train):,} | Test : {len(X_test):,} | Features : {X.shape[1]}\n")

    imp_med = SimpleImputer(strategy="median")
    X_train_i = imp_med.fit_transform(X_train)
    X_test_i = imp_med.transform(X_test)

    # ---------- RF default top 40 ----------
    print("[1/3] RF default sur top 40...")
    t0 = time.time()
    rf_def = RandomForestRegressor(n_estimators=500, max_depth=15,
                                     min_samples_leaf=10, min_samples_split=10,
                                     max_features="sqrt", random_state=42, n_jobs=-1)
    rf_def.fit(X_train_i, y_train)
    pred_rf = rf_def.predict(X_test_i)
    mae, rmse, r2, mae_tr_rf, rmse_tr_rf = evaluate_all(y_test, pred_rf)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f}cm | R²={r2:.4f}")

    # ---------- Classifier ----------
    print("\n[2/3] Classifier ≥1,40m (RF + balanced)...")
    t0 = time.time()
    y_train_bin = (y_train >= 1.40).astype(int)
    y_test_bin = (y_test >= 1.40).astype(int)
    clf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                   min_samples_leaf=10, min_samples_split=10,
                                   max_features="sqrt", class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    clf.fit(X_train_i, y_train_bin)
    p_test = clf.predict_proba(X_test_i)[:, 1]
    p_bin = (p_test >= 0.5).astype(int)
    print(f"  ✓ {time.time()-t0:.1f}s")
    print(f"  AUC ROC   : {roc_auc_score(y_test_bin, p_test):.4f}")
    print(f"  Précision : {precision_score(y_test_bin, p_bin):.4f}")
    print(f"  Rappel    : {recall_score(y_test_bin, p_bin):.4f}")

    # ---------- Régresseur conditionnel ----------
    print("\n[3/3] Régresseur conditionnel sur tops...")
    t0 = time.time()
    mask = (y_train >= 1.40).values
    rf_tops = RandomForestRegressor(n_estimators=500, max_depth=15,
                                      min_samples_leaf=5, min_samples_split=5,
                                      max_features="sqrt", random_state=42, n_jobs=-1)
    rf_tops.fit(X_train_i[mask], y_train[mask])
    pred_tops = rf_tops.predict(X_test_i)
    print(f"  ✓ {time.time()-t0:.1f}s | Train tops : {mask.sum():,}")

    # ---------- Hurdle v2 ----------
    pred_hurdle_v2 = p_test * pred_tops + (1 - p_test) * pred_rf
    mae_h, rmse_h, r2_h, mae_tr_h, rmse_tr_h = evaluate_all(y_test, pred_hurdle_v2)

    # ---------- Comparaisons ----------
    print("\n" + "=" * 80)
    print("RÉSULTATS GLOBAUX")
    print("=" * 80)

    # Récup Hurdle v1 + Poly40 + Stacking + Calib depuis le récap
    recap = pd.read_csv(MASTER_DIR / "recap_avec_poly40_global.csv")
    mae_tr_all = pd.read_csv(MASTER_DIR / "recap_avec_poly40_mae.csv")
    rmse_tr_all = pd.read_csv(MASTER_DIR / "recap_avec_poly40_rmse.csv")

    refs = {}
    for m in ["Hurdle (mélange)", "Stacking + Calib", "Poly40 (deg2 + interact.)"]:
        g = recap[recap["modele"] == m].iloc[0]
        mt = mae_tr_all[mae_tr_all["modele"] == m].iloc[0]
        rt = rmse_tr_all[rmse_tr_all["modele"] == m].iloc[0]
        refs[m] = {"MAE": g["MAE_cm"], "RMSE": g["RMSE_cm"], "R2": g["R2"],
                    "mae_tr": {k: mt[k] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]},
                    "rmse_tr": {k: rt[k] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]}}

    rows_g = [
        ("RF default (top 40)", mae, rmse, r2, mae_tr_rf, rmse_tr_rf),
        ("Hurdle v2 (top 40)", mae_h, rmse_h, r2_h, mae_tr_h, rmse_tr_h),
        ("Hurdle v1 (156 feat.) — réf.", refs["Hurdle (mélange)"]["MAE"],
         refs["Hurdle (mélange)"]["RMSE"], refs["Hurdle (mélange)"]["R2"],
         refs["Hurdle (mélange)"]["mae_tr"], refs["Hurdle (mélange)"]["rmse_tr"]),
        ("Poly40 — réf.", refs["Poly40 (deg2 + interact.)"]["MAE"],
         refs["Poly40 (deg2 + interact.)"]["RMSE"], refs["Poly40 (deg2 + interact.)"]["R2"],
         refs["Poly40 (deg2 + interact.)"]["mae_tr"], refs["Poly40 (deg2 + interact.)"]["rmse_tr"]),
        ("Stacking + Calib — réf.", refs["Stacking + Calib"]["MAE"],
         refs["Stacking + Calib"]["RMSE"], refs["Stacking + Calib"]["R2"],
         refs["Stacking + Calib"]["mae_tr"], refs["Stacking + Calib"]["rmse_tr"]),
    ]

    print(f"{'Modèle':<32s} | {'MAE':>7s} | {'RMSE':>7s} | {'R²':>7s}")
    print("-" * 70)
    for label, mae_, rmse_, r2_, _, _ in rows_g:
        print(f"{label:<32s} | {mae_:>6.2f}c | {rmse_:>6.2f}c | {r2_:>7.4f}")

    print("\n" + "=" * 105)
    print("MAE PAR TRANCHE (cm)")
    print("=" * 105)
    print(f"{'Modèle':<32s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 100)
    for label, _, _, _, mt, _ in rows_g:
        print(f"{label:<32s} | {mt['≤1.10m']:>6.2f}c | {mt['1.15-1.20m']:>9.2f}c | "
              f"{mt['1.25-1.30m']:>9.2f}c | {mt['1.35-1.40m']:>9.2f}c | {mt['≥1.45m']:>6.2f}c")

    print("\n" + "=" * 105)
    print("RMSE PAR TRANCHE (cm)")
    print("=" * 105)
    print(f"{'Modèle':<32s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 100)
    for label, _, _, _, _, rt in rows_g:
        print(f"{label:<32s} | {rt['≤1.10m']:>6.2f}c | {rt['1.15-1.20m']:>9.2f}c | "
              f"{rt['1.25-1.30m']:>9.2f}c | {rt['1.35-1.40m']:>9.2f}c | {rt['≥1.45m']:>6.2f}c")

    out = pd.DataFrame([{
        "modele": label, "MAE_cm": mae_, "RMSE_cm": rmse_, "R2": r2_,
        **mt, **{f"RMSE_{k}": v for k, v in rt.items()}
    } for label, mae_, rmse_, r2_, mt, rt in rows_g])
    out.to_csv(MASTER_DIR / "hurdle_v2_top40_results.csv", index=False)
    print(f"\n→ hurdle_v2_top40_results.csv")


if __name__ == "__main__":
    main()
