"""
32 - Récap complet avec RMSE par tranche en plus.

Produit 3 tableaux :
  - Globaux : MAE, RMSE, R²
  - MAE par tranche
  - RMSE par tranche

Sortie : data/master/recap_complet_rmse.csv
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
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def evaluate_all(y_true, y_pred):
    """Retourne MAE global, RMSE global, R² global + MAE et RMSE par tranche."""
    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
    r2 = r2_score(y_true, y_pred)

    df = pd.DataFrame({"y_true": y_true.values if hasattr(y_true, "values") else y_true,
                        "y_pred": y_pred})
    df["tranche"] = pd.cut(df["y_true"],
                            bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])
    df["err"] = df["y_true"] - df["y_pred"]
    df["err_abs"] = df["err"].abs() * 100
    df["err_sq"] = (df["err"] ** 2) * 10000  # cm²

    mae_par_tranche = df.groupby("tranche", observed=True)["err_abs"].mean().to_dict()
    rmse_par_tranche = df.groupby("tranche", observed=True)["err_sq"].mean().apply(np.sqrt).to_dict()

    return {
        "MAE_global": mae,
        "RMSE_global": rmse,
        "R2_global": r2,
        "mae_tr": mae_par_tranche,
        "rmse_tr": rmse_par_tranche,
    }


def main():
    print("=== 32 - Récap complet avec RMSE par tranche ===\n")

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

    imp = SimpleImputer(strategy="median")
    X_train_med = imp.fit_transform(X_train)
    X_valid_med = imp.transform(X_valid)
    X_test_med = imp.transform(X_test)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train_med)
    X_valid_sc = sc.transform(X_valid_med)
    X_test_sc = sc.transform(X_test_med)

    all_preds = {}

    # 1. ElasticNet
    print("[1/9] ElasticNet...")
    t0 = time.time()
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                     n_jobs=-1, random_state=42)
    en.fit(X_train_sc, y_train)
    all_preds["ElasticNet"] = en.predict(X_test_sc)
    all_preds["__valid_en"] = en.predict(X_valid_sc)
    print(f"  ✓ {time.time()-t0:.1f}s")

    # 2. RF default
    print("[2/9] RF default...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500, max_depth=15,
                                 min_samples_leaf=10, min_samples_split=10,
                                 max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(X_train_med, y_train)
    all_preds["RF default"] = rf.predict(X_test_med)
    all_preds["__valid_rf"] = rf.predict(X_valid_med)
    print(f"  ✓ {time.time()-t0:.1f}s")

    # 3. XGBoost
    print("[3/9] XGBoost default...")
    t0 = time.time()
    xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
                        random_state=42, n_jobs=-1, eval_metric="mae",
                        early_stopping_rounds=30)
    xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    all_preds["XGBoost default"] = xgb.predict(X_test)
    print(f"  ✓ {time.time()-t0:.1f}s")

    # 4. CatBoost
    print("[4/9] CatBoost...")
    t0 = time.time()
    cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3, random_seed=42, loss_function="MAE",
                            eval_metric="MAE", early_stopping_rounds=30, verbose=0)
    cb.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    all_preds["CatBoost"] = cb.predict(X_test)
    all_preds["__valid_cb"] = cb.predict(X_valid)
    print(f"  ✓ {time.time()-t0:.1f}s")

    # 5. Hurdle
    print("[5/9] Hurdle...")
    t0 = time.time()
    y_train_bin = (y_train >= 1.40).astype(int)
    clf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                   min_samples_leaf=10, min_samples_split=10,
                                   max_features="sqrt", class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    clf.fit(X_train_med, y_train_bin)
    p_test = clf.predict_proba(X_test_med)[:, 1]
    mask_tops = (y_train >= 1.40).values
    rf_tops = RandomForestRegressor(n_estimators=500, max_depth=15,
                                      min_samples_leaf=5, min_samples_split=5,
                                      max_features="sqrt", random_state=42, n_jobs=-1)
    rf_tops.fit(X_train_med[mask_tops], y_train[mask_tops])
    pred_tops = rf_tops.predict(X_test_med)
    all_preds["Hurdle (mélange)"] = p_test * pred_tops + (1 - p_test) * all_preds["RF default"]
    print(f"  ✓ {time.time()-t0:.1f}s")

    # 6. RF + sample_w ×3
    print("[6/9] RF + sample_w ×3...")
    t0 = time.time()
    w = np.where(y_train >= 1.40, 3.0, 1.0)
    rfw3 = RandomForestRegressor(n_estimators=500, max_depth=15,
                                    min_samples_leaf=10, min_samples_split=10,
                                    max_features="sqrt", random_state=42, n_jobs=-1)
    rfw3.fit(X_train_med, y_train, sample_weight=w)
    all_preds["RF + sample_w ×3"] = rfw3.predict(X_test_med)
    print(f"  ✓ {time.time()-t0:.1f}s")

    # 7. RF + sample_w ×5
    print("[7/9] RF + sample_w ×5...")
    t0 = time.time()
    w = np.where(y_train >= 1.40, 5.0, 1.0)
    rfw5 = RandomForestRegressor(n_estimators=500, max_depth=15,
                                    min_samples_leaf=10, min_samples_split=10,
                                    max_features="sqrt", random_state=42, n_jobs=-1)
    rfw5.fit(X_train_med, y_train, sample_weight=w)
    all_preds["RF + sample_w ×5"] = rfw5.predict(X_test_med)
    print(f"  ✓ {time.time()-t0:.1f}s")

    # 8. RF + Calib isotonic
    print("[8/9] RF + Calib isotonic...")
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(all_preds["__valid_rf"], y_valid)
    all_preds["RF + calib isotonic"] = cal.predict(all_preds["RF default"])
    print(f"  ✓")

    # 9. Stacking + Calib
    print("[9/9] Stacking + Calib...")
    X_meta_v = np.column_stack([all_preds["__valid_rf"], all_preds["__valid_en"],
                                  all_preds["__valid_cb"]])
    X_meta_t = np.column_stack([all_preds["RF default"], all_preds["ElasticNet"],
                                  all_preds["CatBoost"]])
    meta = LinearRegression()
    meta.fit(X_meta_v, y_valid)
    pred_stack = meta.predict(X_meta_t)
    pred_stack_v = meta.predict(X_meta_v)
    cal2 = IsotonicRegression(out_of_bounds="clip")
    cal2.fit(pred_stack_v, y_valid)
    all_preds["Stacking + Calib"] = cal2.predict(pred_stack)
    print(f"  ✓")

    # ============================================================
    # Évaluation complète
    # ============================================================
    modeles_ordre = ["Stacking + Calib", "RF + calib isotonic", "RF + sample_w ×3",
                      "RF default", "RF + sample_w ×5", "ElasticNet",
                      "Hurdle (mélange)", "CatBoost", "XGBoost default"]

    tranches = ["≤1.10m", "1.15-1.20m", "1.25-1.30m", "1.35-1.40m", "≥1.45m"]

    print("\n" + "=" * 90)
    print("TABLEAU 1 : MÉTRIQUES GLOBALES")
    print("=" * 90)
    print(f"{'Modèle':<25s} | {'MAE (cm)':>10s} | {'RMSE (cm)':>10s} | {'R²':>8s} | {'RMSE/MAE':>9s}")
    print("-" * 75)
    rows_glob = []
    rows_mae = []
    rows_rmse = []
    for m in modeles_ordre:
        ev = evaluate_all(y_test, all_preds[m])
        print(f"{m:<25s} | {ev['MAE_global']:>9.2f} | {ev['RMSE_global']:>9.2f} | "
              f"{ev['R2_global']:>7.4f} | {ev['RMSE_global']/ev['MAE_global']:>8.3f}")
        rows_glob.append({"modele": m, "MAE_cm": ev["MAE_global"],
                          "RMSE_cm": ev["RMSE_global"], "R2": ev["R2_global"]})
        rows_mae.append({"modele": m, **ev["mae_tr"]})
        rows_rmse.append({"modele": m, **ev["rmse_tr"]})

    print("\n" + "=" * 100)
    print("TABLEAU 2 : MAE PAR TRANCHE (cm)")
    print("=" * 100)
    print(pd.DataFrame(rows_mae).set_index("modele").round(2).to_string())

    print("\n" + "=" * 100)
    print("TABLEAU 3 : RMSE PAR TRANCHE (cm)")
    print("=" * 100)
    print(pd.DataFrame(rows_rmse).set_index("modele").round(2).to_string())

    # Sauvegarde
    df_glob = pd.DataFrame(rows_glob)
    df_mae = pd.DataFrame(rows_mae)
    df_rmse = pd.DataFrame(rows_rmse)
    df_glob.to_csv(MASTER_DIR / "recap_global.csv", index=False)
    df_mae.to_csv(MASTER_DIR / "recap_mae_par_tranche.csv", index=False)
    df_rmse.to_csv(MASTER_DIR / "recap_rmse_par_tranche.csv", index=False)
    print(f"\n→ recap_global.csv, recap_mae_par_tranche.csv, recap_rmse_par_tranche.csv")


if __name__ == "__main__":
    main()
