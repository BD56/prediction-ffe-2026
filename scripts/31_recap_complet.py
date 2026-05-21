"""
31 - Récap complet de tous les modèles : MAE, RMSE, R² + MAE par tranche.

Réentraîne tous les modèles principaux et produit un tableau récapitulatif
complet pour le rapport.

Note : utilise SimpleImputer(median) pour tous (cohérence, rapidité).
ElasticNet original utilisait KNNImputer(k=5) — résultats légèrement
différents mais ordres de grandeur identiques.

Sortie : data/master/recap_complet.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def metrics_par_tranche(y_true, y_pred):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["tranche"] = pd.cut(df["y_true"],
                            bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])
    df["err_abs"] = (df["y_true"] - df["y_pred"]).abs() * 100
    return df.groupby("tranche", observed=True)["err_abs"].mean().to_dict()


def main():
    print("=== 31 - Récap complet (MAE + RMSE + R²) ===\n")

    master = pd.read_parquet(MASTER_DIR / "master_dataset_clean.parquet")
    master = master.set_index("IDCHEVAL")
    feat_cols = [c for c in master.columns if c.startswith("f")]
    for c in feat_cols:
        if master[c].dtype == bool:
            master[c] = master[c].astype(int)
    X = master[feat_cols].select_dtypes(include=[np.number])
    y = master["hauteur_max_validee"]
    split = master["SPLIT"]

    X_train, y_train = X[split == "train"], y[split == "train"]
    X_valid, y_valid = X[split == "valid"], y[split == "valid"]
    X_test, y_test = X[split == "test"], y[split == "test"]

    # Imputation médiane + scaling
    imp = SimpleImputer(strategy="median")
    X_train_med = imp.fit_transform(X_train)
    X_valid_med = imp.transform(X_valid)
    X_test_med = imp.transform(X_test)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train_med)
    X_valid_sc = sc.transform(X_valid_med)
    X_test_sc = sc.transform(X_test_med)

    results = []

    # ============================================================
    # 1. ElasticNet
    # ============================================================
    print("[1/9] ElasticNet (avec imputation médiane)...")
    t0 = time.time()
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                     n_jobs=-1, random_state=42)
    en.fit(X_train_sc, y_train)
    pred_en = en.predict(X_test_sc)
    pred_en_valid = en.predict(X_valid_sc)
    mae, rmse, r2 = metrics(y_test, pred_en)
    tranches = metrics_par_tranche(y_test, pred_en)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f}cm RMSE={rmse:.2f}cm R²={r2:.4f}")
    results.append({"modele": "ElasticNet", "MAE_cm": mae, "RMSE_cm": rmse,
                    "R2": r2, **tranches})

    # ============================================================
    # 2. RF default
    # ============================================================
    print("\n[2/9] RF default...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500, max_depth=15,
                                 min_samples_leaf=10, min_samples_split=10,
                                 max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(X_train_med, y_train)
    pred_rf = rf.predict(X_test_med)
    pred_rf_valid = rf.predict(X_valid_med)
    mae, rmse, r2 = metrics(y_test, pred_rf)
    tranches = metrics_par_tranche(y_test, pred_rf)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f}cm RMSE={rmse:.2f}cm R²={r2:.4f}")
    results.append({"modele": "RF default", "MAE_cm": mae, "RMSE_cm": rmse,
                    "R2": r2, **tranches})

    # ============================================================
    # 3. XGBoost default
    # ============================================================
    print("\n[3/9] XGBoost default...")
    t0 = time.time()
    xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
                        random_state=42, n_jobs=-1, eval_metric="mae",
                        early_stopping_rounds=30)
    xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    pred_xgb = xgb.predict(X_test)
    mae, rmse, r2 = metrics(y_test, pred_xgb)
    tranches = metrics_par_tranche(y_test, pred_xgb)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f}cm RMSE={rmse:.2f}cm R²={r2:.4f}")
    results.append({"modele": "XGBoost default", "MAE_cm": mae, "RMSE_cm": rmse,
                    "R2": r2, **tranches})

    # ============================================================
    # 4. CatBoost
    # ============================================================
    print("\n[4/9] CatBoost...")
    t0 = time.time()
    cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3, random_seed=42, loss_function="MAE",
                            eval_metric="MAE", early_stopping_rounds=30, verbose=0)
    cb.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    pred_cb = cb.predict(X_test)
    pred_cb_valid = cb.predict(X_valid)
    mae, rmse, r2 = metrics(y_test, pred_cb)
    tranches = metrics_par_tranche(y_test, pred_cb)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f}cm RMSE={rmse:.2f}cm R²={r2:.4f}")
    results.append({"modele": "CatBoost", "MAE_cm": mae, "RMSE_cm": rmse,
                    "R2": r2, **tranches})

    # ============================================================
    # 5. Hurdle (mélange pondéré)
    # ============================================================
    print("\n[5/9] Hurdle...")
    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                   min_samples_leaf=10, min_samples_split=10,
                                   max_features="sqrt", class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    y_train_bin = (y_train >= 1.40).astype(int)
    clf.fit(X_train_med, y_train_bin)
    p_test = clf.predict_proba(X_test_med)[:, 1]
    mask_tops = (y_train >= 1.40).values
    rf_tops = RandomForestRegressor(n_estimators=500, max_depth=15,
                                      min_samples_leaf=5, min_samples_split=5,
                                      max_features="sqrt", random_state=42, n_jobs=-1)
    rf_tops.fit(X_train_med[mask_tops], y_train[mask_tops])
    pred_tops = rf_tops.predict(X_test_med)
    pred_hurdle = p_test * pred_tops + (1 - p_test) * pred_rf
    mae, rmse, r2 = metrics(y_test, pred_hurdle)
    tranches = metrics_par_tranche(y_test, pred_hurdle)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f}cm RMSE={rmse:.2f}cm R²={r2:.4f}")
    results.append({"modele": "Hurdle (mélange)", "MAE_cm": mae, "RMSE_cm": rmse,
                    "R2": r2, **tranches})

    # ============================================================
    # 6. RF + Sample weights ×3
    # ============================================================
    print("\n[6/9] RF + sample weights ×3...")
    t0 = time.time()
    w = np.where(y_train >= 1.40, 3.0, 1.0)
    rf_w3 = RandomForestRegressor(n_estimators=500, max_depth=15,
                                    min_samples_leaf=10, min_samples_split=10,
                                    max_features="sqrt", random_state=42, n_jobs=-1)
    rf_w3.fit(X_train_med, y_train, sample_weight=w)
    pred_w3 = rf_w3.predict(X_test_med)
    mae, rmse, r2 = metrics(y_test, pred_w3)
    tranches = metrics_par_tranche(y_test, pred_w3)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f}cm RMSE={rmse:.2f}cm R²={r2:.4f}")
    results.append({"modele": "RF + sample_w ×3", "MAE_cm": mae, "RMSE_cm": rmse,
                    "R2": r2, **tranches})

    # ============================================================
    # 7. RF + Sample weights ×5
    # ============================================================
    print("\n[7/9] RF + sample weights ×5...")
    t0 = time.time()
    w = np.where(y_train >= 1.40, 5.0, 1.0)
    rf_w5 = RandomForestRegressor(n_estimators=500, max_depth=15,
                                    min_samples_leaf=10, min_samples_split=10,
                                    max_features="sqrt", random_state=42, n_jobs=-1)
    rf_w5.fit(X_train_med, y_train, sample_weight=w)
    pred_w5 = rf_w5.predict(X_test_med)
    mae, rmse, r2 = metrics(y_test, pred_w5)
    tranches = metrics_par_tranche(y_test, pred_w5)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f}cm RMSE={rmse:.2f}cm R²={r2:.4f}")
    results.append({"modele": "RF + sample_w ×5", "MAE_cm": mae, "RMSE_cm": rmse,
                    "R2": r2, **tranches})

    # ============================================================
    # 8. RF + Calibration isotonic
    # ============================================================
    print("\n[8/9] RF + Calibration isotonic...")
    t0 = time.time()
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(pred_rf_valid, y_valid)
    pred_cal = cal.predict(pred_rf)
    mae, rmse, r2 = metrics(y_test, pred_cal)
    tranches = metrics_par_tranche(y_test, pred_cal)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f}cm RMSE={rmse:.2f}cm R²={r2:.4f}")
    results.append({"modele": "RF + calib_isotonic", "MAE_cm": mae, "RMSE_cm": rmse,
                    "R2": r2, **tranches})

    # ============================================================
    # 9. Stacking + Calibration
    # ============================================================
    print("\n[9/9] Stacking + Calibration isotonic...")
    t0 = time.time()
    X_meta_valid = np.column_stack([pred_rf_valid, pred_en_valid, pred_cb_valid])
    X_meta_test = np.column_stack([pred_rf, pred_en, pred_cb])
    meta = LinearRegression()
    meta.fit(X_meta_valid, y_valid)
    pred_stack = meta.predict(X_meta_test)
    pred_stack_valid = meta.predict(X_meta_valid)
    cal2 = IsotonicRegression(out_of_bounds="clip")
    cal2.fit(pred_stack_valid, y_valid)
    pred_stack_cal = cal2.predict(pred_stack)
    mae, rmse, r2 = metrics(y_test, pred_stack_cal)
    tranches = metrics_par_tranche(y_test, pred_stack_cal)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f}cm RMSE={rmse:.2f}cm R²={r2:.4f}")
    results.append({"modele": "Stacking + Calib", "MAE_cm": mae, "RMSE_cm": rmse,
                    "R2": r2, **tranches})

    # ============================================================
    # Récap complet
    # ============================================================
    df_res = pd.DataFrame(results)
    df_res["RMSE/MAE"] = df_res["RMSE_cm"] / df_res["MAE_cm"]
    df_res = df_res[["modele", "MAE_cm", "RMSE_cm", "R2", "RMSE/MAE",
                      "≤1.10m", "1.15-1.20m", "1.25-1.30m", "1.35-1.40m", "≥1.45m"]]
    df_res = df_res.sort_values("MAE_cm")

    print("\n" + "=" * 105)
    print("RÉCAP COMPLET (test set, 5045 chevaux)")
    print("=" * 105)
    print(df_res.round(2).to_string(index=False))

    df_res.to_csv(MASTER_DIR / "recap_complet.csv", index=False)
    print(f"\n→ Sauvegardé : {MASTER_DIR / 'recap_complet.csv'}")


if __name__ == "__main__":
    main()
