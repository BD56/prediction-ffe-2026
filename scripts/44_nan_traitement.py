"""
44 - Piste 3 : traitement des NaN (NaN à -999 + HistGradientBoosting).

L'idée : le NaN porte un signal métier ("ce cheval n'a pas couru à 4 ans"),
que l'imputation médiane efface.

Tests :
  - Baseline   : RF default + SimpleImputer(median) — référence existante
  - Variante A : RF default avec NaN remplacés par -999
  - Variante B : RF default avec NaN remplacés par -9 (échelle plus modeste, après standardisation z)
  - Variante C : HistGradientBoosting (gère nativement les NaN)
  - Variante D : HistGradientBoosting + loss="absolute_error" (MAE)

Sortie : data/master/nan_traitement_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    print("=== 44 - Piste 3 : traitement des NaN ===\n")

    v2 = pd.read_parquet(MASTER_DIR / "master_dataset_epure_v2.parquet")
    v2 = v2.set_index("IDCHEVAL")
    feat_cols = [c for c in v2.columns if c.startswith("f")]
    for c in feat_cols:
        if v2[c].dtype == bool:
            v2[c] = v2[c].astype(int)
    X = v2[feat_cols].select_dtypes(include=[np.number])
    y = v2["hauteur_max_validee"]
    split = v2["SPLIT"]

    X_train, y_train = X[split == "train"], y[split == "train"]
    X_test, y_test = X[split == "test"], y[split == "test"]
    print(f"Train : {len(X_train):,} | Test : {len(X_test):,} | Features : {X.shape[1]}")
    print(f"% de NaN globaux (train) : {X_train.isna().sum().sum() / X_train.size * 100:.2f}%")
    print(f"Nb features avec >0% NaN : {(X_train.isna().mean() > 0).sum()} / {X.shape[1]}")
    print(f"Nb features avec >25% NaN : {(X_train.isna().mean() > 0.25).sum()}")
    print(f"Nb features avec >50% NaN : {(X_train.isna().mean() > 0.50).sum()}")

    rows = []

    # ---------- Baseline : RF + impute median ----------
    print("\n[Baseline] RF default + impute médiane...")
    t0 = time.time()
    imp_med = SimpleImputer(strategy="median")
    X_train_med = imp_med.fit_transform(X_train)
    X_test_med = imp_med.transform(X_test)
    rf = RandomForestRegressor(n_estimators=500, max_depth=15,
                                 min_samples_leaf=10, min_samples_split=10,
                                 max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(X_train_med, y_train)
    pred = rf.predict(X_test_med)
    mae, rmse, r2, mtr, rtr = evaluate_all(y_test, pred)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    rows.append(("Baseline (RF + impute médiane)", mae, rmse, r2, mtr, rtr))

    # ---------- Variante A : RF + NaN à -999 ----------
    print("\n[A] RF default + NaN remplacés par -999...")
    t0 = time.time()
    X_train_a = X_train.fillna(-999).values
    X_test_a = X_test.fillna(-999).values
    rf_a = RandomForestRegressor(n_estimators=500, max_depth=15,
                                   min_samples_leaf=10, min_samples_split=10,
                                   max_features="sqrt", random_state=42, n_jobs=-1)
    rf_a.fit(X_train_a, y_train)
    pred_a = rf_a.predict(X_test_a)
    mae, rmse, r2, mtr, rtr = evaluate_all(y_test, pred_a)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    rows.append(("A. RF + NaN=-999", mae, rmse, r2, mtr, rtr))

    # ---------- Variante B : RF + NaN à -9 ----------
    print("\n[B] RF default + NaN remplacés par -9 (sentinel modéré)...")
    t0 = time.time()
    X_train_b = X_train.fillna(-9).values
    X_test_b = X_test.fillna(-9).values
    rf_b = RandomForestRegressor(n_estimators=500, max_depth=15,
                                   min_samples_leaf=10, min_samples_split=10,
                                   max_features="sqrt", random_state=42, n_jobs=-1)
    rf_b.fit(X_train_b, y_train)
    pred_b = rf_b.predict(X_test_b)
    mae, rmse, r2, mtr, rtr = evaluate_all(y_test, pred_b)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    rows.append(("B. RF + NaN=-9", mae, rmse, r2, mtr, rtr))

    # ---------- Variante C : HistGradientBoosting (NaN natif) ----------
    print("\n[C] HistGradientBoosting (NaN natif, loss=squared_error)...")
    t0 = time.time()
    hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                          max_depth=6, min_samples_leaf=10,
                                          l2_regularization=1.0,
                                          random_state=42)
    hgb.fit(X_train, y_train)  # accepte NaN natif
    pred_c = hgb.predict(X_test)
    mae, rmse, r2, mtr, rtr = evaluate_all(y_test, pred_c)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    rows.append(("C. HistGradientBoosting (squared)", mae, rmse, r2, mtr, rtr))

    # ---------- Variante D : HistGradientBoosting + MAE ----------
    print("\n[D] HistGradientBoosting + loss=absolute_error...")
    t0 = time.time()
    hgb_mae = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                              max_depth=6, min_samples_leaf=10,
                                              l2_regularization=1.0,
                                              loss="absolute_error",
                                              random_state=42)
    hgb_mae.fit(X_train, y_train)
    pred_d = hgb_mae.predict(X_test)
    mae, rmse, r2, mtr, rtr = evaluate_all(y_test, pred_d)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    rows.append(("D. HistGradientBoosting (MAE)", mae, rmse, r2, mtr, rtr))

    # ---------- Récap ----------
    print("\n" + "=" * 80)
    print("RÉSULTATS GLOBAUX")
    print("=" * 80)
    print(f"{'Modèle':<38s} | {'MAE':>7s} | {'RMSE':>7s} | {'R²':>7s}")
    print("-" * 75)
    for label, mae_, rmse_, r2_, _, _ in rows:
        print(f"{label:<38s} | {mae_:>6.2f}c | {rmse_:>6.2f}c | {r2_:>7.4f}")

    print("\n" + "=" * 110)
    print("MAE PAR TRANCHE (cm)")
    print("=" * 110)
    print(f"{'Modèle':<38s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 100)
    for label, _, _, _, mt, _ in rows:
        print(f"{label:<38s} | {mt['≤1.10m']:>6.2f}c | {mt['1.15-1.20m']:>9.2f}c | "
              f"{mt['1.25-1.30m']:>9.2f}c | {mt['1.35-1.40m']:>9.2f}c | {mt['≥1.45m']:>6.2f}c")

    print("\n" + "=" * 110)
    print("RMSE PAR TRANCHE (cm)")
    print("=" * 110)
    print(f"{'Modèle':<38s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 100)
    for label, _, _, _, _, rt in rows:
        print(f"{label:<38s} | {rt['≤1.10m']:>6.2f}c | {rt['1.15-1.20m']:>9.2f}c | "
              f"{rt['1.25-1.30m']:>9.2f}c | {rt['1.35-1.40m']:>9.2f}c | {rt['≥1.45m']:>6.2f}c")

    out = pd.DataFrame([{
        "modele": label, "MAE_cm": mae_, "RMSE_cm": rmse_, "R2": r2_,
        **mt, **{f"RMSE_{k}": v for k, v in rt.items()}
    } for label, mae_, rmse_, r2_, mt, rt in rows])
    out.to_csv(MASTER_DIR / "nan_traitement_results.csv", index=False)
    print(f"\n→ nan_traitement_results.csv")


if __name__ == "__main__":
    main()
