"""
34 - Épuration v2 : supprimer les features rang_moyen >= 100 et refit.

Étape 1 : construire master_dataset_epure_v2.parquet (188 → 156 features)
Étape 2 : refit RF, XGBoost, CatBoost, ElasticNet sur v2
Étape 3 : comparer vs v1 (master_dataset_clean.parquet)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def mae_par_tranche(y_true, y_pred):
    df = pd.DataFrame({"y": y_true.values, "p": y_pred})
    df["tranche"] = pd.cut(df["y"], bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])
    df["err"] = (df["y"] - df["p"]).abs() * 100
    return df.groupby("tranche", observed=True)["err"].mean().to_dict()


def main():
    print("=== 34 - Épuration v2 + refit des 4 modèles ===\n")

    # ---------- Étape 1 : construire le dataset v2 ----------
    print("[1/3] Construction master_dataset_epure_v2.parquet...")
    master = pd.read_parquet(MASTER_DIR / "master_dataset_clean.parquet")
    imp = pd.read_csv(MASTER_DIR / "importance_comparative.csv")

    to_drop = imp[imp["rank_moyen"] >= 100]["feature"].tolist()
    keep = [c for c in master.columns if c not in to_drop]
    v2 = master[keep].copy()
    v2.to_parquet(MASTER_DIR / "master_dataset_epure_v2.parquet", index=False)
    print(f"  Features supprimées : {len(to_drop)}")
    print(f"  Colonnes v1 : {master.shape[1]} → v2 : {v2.shape[1]}")

    # ---------- Étape 2 : préparer X, y, splits ----------
    print("\n[2/3] Refit des 4 modèles sur v2...")
    v2 = v2.set_index("IDCHEVAL")
    feat_cols = [c for c in v2.columns if c.startswith("f")]
    for c in feat_cols:
        if v2[c].dtype == bool:
            v2[c] = v2[c].astype(int)
    X = v2[feat_cols].select_dtypes(include=[np.number])
    y = v2["hauteur_max_validee"]
    split = v2["SPLIT"]

    X_train, y_train = X[split == "train"], y[split == "train"]
    X_valid, y_valid = X[split == "valid"], y[split == "valid"]
    X_test, y_test = X[split == "test"], y[split == "test"]
    print(f"  Train/Valid/Test : {len(X_train):,} / {len(X_valid):,} / {len(X_test):,}")
    print(f"  Nb features      : {X.shape[1]}")

    # Imputation pour RF / EN
    imp_med = SimpleImputer(strategy="median")
    X_train_med = imp_med.fit_transform(X_train)
    X_valid_med = imp_med.transform(X_valid)
    X_test_med = imp_med.transform(X_test)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train_med)
    X_test_sc = sc.transform(X_test_med)

    results = {}

    # --- ElasticNet ---
    print("  - ElasticNet...", end="", flush=True)
    t0 = time.time()
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                      n_jobs=-1, random_state=42)
    en.fit(X_train_sc, y_train)
    p_en = en.predict(X_test_sc)
    results["ElasticNet"] = (*evaluate(y_test, p_en), mae_par_tranche(y_test, p_en))
    print(f" {time.time()-t0:.0f}s")

    # --- RF ---
    print("  - RF default...", end="", flush=True)
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500, max_depth=15,
                                 min_samples_leaf=10, min_samples_split=10,
                                 max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(X_train_med, y_train)
    p_rf = rf.predict(X_test_med)
    results["RF default"] = (*evaluate(y_test, p_rf), mae_par_tranche(y_test, p_rf))
    print(f" {time.time()-t0:.0f}s")

    # --- XGBoost ---
    print("  - XGBoost default...", end="", flush=True)
    t0 = time.time()
    xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
                        random_state=42, n_jobs=-1, eval_metric="mae",
                        early_stopping_rounds=30)
    xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    p_xgb = xgb.predict(X_test)
    results["XGBoost default"] = (*evaluate(y_test, p_xgb), mae_par_tranche(y_test, p_xgb))
    print(f" {time.time()-t0:.0f}s")

    # --- CatBoost ---
    print("  - CatBoost...", end="", flush=True)
    t0 = time.time()
    cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3, random_seed=42, loss_function="MAE",
                            eval_metric="MAE", early_stopping_rounds=30, verbose=0)
    cb.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    p_cb = cb.predict(X_test)
    results["CatBoost"] = (*evaluate(y_test, p_cb), mae_par_tranche(y_test, p_cb))
    print(f" {time.time()-t0:.0f}s")

    # ---------- Étape 3 : comparaison v1 vs v2 ----------
    print("\n[3/3] Comparaison v1 (188 feat) vs v2 (156 feat)...")

    # Charger v1 (depuis recap_global.csv)
    v1 = pd.read_csv(MASTER_DIR / "recap_global.csv").set_index("modele")

    print("\n" + "=" * 85)
    print("COMPARAISON GLOBALE (test set)")
    print("=" * 85)
    print(f"{'Modèle':<22s} | {'MAE v1':>7s} | {'MAE v2':>7s} | {'Δ MAE':>7s} | "
          f"{'RMSE v1':>8s} | {'RMSE v2':>8s} | {'R² v1':>7s} | {'R² v2':>7s}")
    print("-" * 95)
    rows = []
    for m in ["RF default", "ElasticNet", "CatBoost", "XGBoost default"]:
        mae_v2, rmse_v2, r2_v2, _ = results[m]
        mae_v1 = v1.loc[m, "MAE_cm"]
        rmse_v1 = v1.loc[m, "RMSE_cm"]
        r2_v1 = v1.loc[m, "R2"]
        d_mae = mae_v2 - mae_v1
        sign = "↓" if d_mae < 0 else ("↑" if d_mae > 0 else "=")
        print(f"{m:<22s} | {mae_v1:>6.2f}c | {mae_v2:>6.2f}c | {d_mae:>+5.2f} {sign} | "
              f"{rmse_v1:>7.2f}c | {rmse_v2:>7.2f}c | {r2_v1:>7.4f} | {r2_v2:>7.4f}")
        rows.append({"modele": m, "MAE_v1": mae_v1, "MAE_v2": mae_v2, "delta_MAE": d_mae,
                     "RMSE_v1": rmse_v1, "RMSE_v2": rmse_v2,
                     "R2_v1": r2_v1, "R2_v2": r2_v2})

    pd.DataFrame(rows).to_csv(MASTER_DIR / "epuration_v2_comparison.csv", index=False)
    print(f"\n→ master_dataset_epure_v2.parquet, epuration_v2_comparison.csv")

    # MAE par tranche en bonus
    print("\n" + "=" * 90)
    print("MAE PAR TRANCHE — v2 (cm)")
    print("=" * 90)
    print(f"{'Modèle':<22s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 85)
    for m, (_, _, _, tr) in results.items():
        print(f"{m:<22s} | {tr.get('≤1.10m', 0):>6.2f}c | {tr.get('1.15-1.20m', 0):>9.2f}c | "
              f"{tr.get('1.25-1.30m', 0):>9.2f}c | {tr.get('1.35-1.40m', 0):>9.2f}c | "
              f"{tr.get('≥1.45m', 0):>6.2f}c")


if __name__ == "__main__":
    main()
