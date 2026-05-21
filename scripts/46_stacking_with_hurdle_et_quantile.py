"""
46 - Tester deux pistes pour battre Hurdle sur les tops :

Option C : Stacking enrichi avec Hurdle comme 4e base
   - Bases : RF, ElasticNet, CatBoost, Hurdle
   - Méta : régression linéaire sur valid
   - + Calibration isotonic optionnelle

Option D : Régression quantile (XGBoost) avec différents τ
   - τ = 0.5 (médiane, baseline)
   - τ = 0.6, 0.7, 0.8 (quantiles supérieurs)

Sortie : data/master/stacking_hurdle_et_quantile_results.csv
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
from xgboost import XGBRegressor
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
    print("=== 46 - Stacking+Hurdle (C) et Quantile (D) ===\n")

    # ---------- Données ----------
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
    print(f"Train/Valid/Test : {len(X_train):,} / {len(X_valid):,} / {len(X_test):,}\n")

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xva = imp.transform(X_valid)
    Xte = imp.transform(X_test)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xva_sc = sc.transform(Xva)
    Xte_sc = sc.transform(Xte)

    # ============================================================
    # OPTION C — Stacking avec 4 bases (RF, EN, CB, Hurdle)
    # ============================================================
    print("=" * 60)
    print("OPTION C : Stacking enrichi avec Hurdle")
    print("=" * 60)

    # --- Base 1 : RF ---
    print("\n[Base 1/4] Random Forest...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500, max_depth=15,
                                 min_samples_leaf=10, min_samples_split=10,
                                 max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train)
    valid_rf = rf.predict(Xva)
    test_rf = rf.predict(Xte)
    print(f"  ✓ {time.time()-t0:.1f}s | Test MAE = {mean_absolute_error(y_test, test_rf)*100:.2f}cm")

    # --- Base 2 : ElasticNet ---
    print("[Base 2/4] ElasticNet...")
    t0 = time.time()
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                      n_jobs=-1, random_state=42)
    en.fit(Xtr_sc, y_train)
    valid_en = en.predict(Xva_sc)
    test_en = en.predict(Xte_sc)
    print(f"  ✓ {time.time()-t0:.1f}s | Test MAE = {mean_absolute_error(y_test, test_en)*100:.2f}cm")

    # --- Base 3 : CatBoost ---
    print("[Base 3/4] CatBoost...")
    t0 = time.time()
    cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3, random_seed=42, loss_function="MAE",
                            eval_metric="MAE", early_stopping_rounds=30, verbose=0)
    cb.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    valid_cb = cb.predict(X_valid)
    test_cb = cb.predict(X_test)
    print(f"  ✓ {time.time()-t0:.1f}s | Test MAE = {mean_absolute_error(y_test, test_cb)*100:.2f}cm")

    # --- Base 4 : Hurdle ---
    print("[Base 4/4] Hurdle (mélange)...")
    t0 = time.time()
    y_train_bin = (y_train >= 1.40).astype(int)
    clf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                   min_samples_leaf=10, min_samples_split=10,
                                   max_features="sqrt", class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    clf.fit(Xtr, y_train_bin)
    p_valid = clf.predict_proba(Xva)[:, 1]
    p_test = clf.predict_proba(Xte)[:, 1]
    mask = (y_train >= 1.40).values
    rf_tops = RandomForestRegressor(n_estimators=500, max_depth=15,
                                      min_samples_leaf=5, min_samples_split=5,
                                      max_features="sqrt", random_state=42, n_jobs=-1)
    rf_tops.fit(Xtr[mask], y_train[mask])
    valid_tops = rf_tops.predict(Xva)
    test_tops = rf_tops.predict(Xte)
    valid_hurdle = p_valid * valid_tops + (1 - p_valid) * valid_rf
    test_hurdle = p_test * test_tops + (1 - p_test) * test_rf
    print(f"  ✓ {time.time()-t0:.1f}s | Test MAE = {mean_absolute_error(y_test, test_hurdle)*100:.2f}cm")

    # --- Méta-modèle 4 bases ---
    print("\n[Méta] Régression linéaire sur 4 bases...")
    X_meta_v = np.column_stack([valid_rf, valid_en, valid_cb, valid_hurdle])
    X_meta_t = np.column_stack([test_rf, test_en, test_cb, test_hurdle])
    meta = LinearRegression()
    meta.fit(X_meta_v, y_valid)
    print(f"  Poids méta : RF={meta.coef_[0]:+.4f}, EN={meta.coef_[1]:+.4f}, "
          f"CB={meta.coef_[2]:+.4f}, Hurdle={meta.coef_[3]:+.4f}")
    print(f"  Intercept  : {meta.intercept_:+.4f}")

    pred_stack4 = meta.predict(X_meta_t)
    # + Calibration
    pred_stack4_v = meta.predict(X_meta_v)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(pred_stack4_v, y_valid)
    pred_stack4_cal = cal.predict(pred_stack4)

    rows = []
    for label, p in [("C1. Stacking 4 bases (avec Hurdle)", pred_stack4),
                      ("C2. Stacking 4 bases + Calib", pred_stack4_cal)]:
        mae, rmse, r2, mtr, rtr = evaluate_all(y_test, p)
        rows.append((label, mae, rmse, r2, mtr, rtr))
        print(f"\n  {label}: MAE={mae:.2f}cm | RMSE={rmse:.2f}cm | R²={r2:.4f}")

    # ============================================================
    # OPTION D — Régression quantile (XGBoost)
    # ============================================================
    print("\n" + "=" * 60)
    print("OPTION D : Régression quantile XGBoost")
    print("=" * 60)

    for tau in [0.5, 0.6, 0.7, 0.8]:
        print(f"\n[τ={tau}] XGBoost quantile...")
        t0 = time.time()
        xgb_q = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
                              random_state=42, n_jobs=-1,
                              objective="reg:quantileerror",
                              quantile_alpha=tau,
                              eval_metric="mae",
                              early_stopping_rounds=30)
        xgb_q.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        pred_q = xgb_q.predict(X_test)
        mae, rmse, r2, mtr, rtr = evaluate_all(y_test, pred_q)
        print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f}cm | RMSE={rmse:.2f}cm | R²={r2:.4f}")
        rows.append((f"D. XGB quantile τ={tau}", mae, rmse, r2, mtr, rtr))

    # ============================================================
    # Référence : Stacking + Calib actuel (3 bases) + Hurdle pur
    # ============================================================
    recap = pd.read_csv(MASTER_DIR / "recap_avec_poly40_global.csv")
    mt_all = pd.read_csv(MASTER_DIR / "recap_avec_poly40_mae.csv")
    rt_all = pd.read_csv(MASTER_DIR / "recap_avec_poly40_rmse.csv")
    for m in ["Stacking + Calib", "Hurdle (mélange)"]:
        g = recap[recap["modele"] == m].iloc[0]
        mt = mt_all[mt_all["modele"] == m].iloc[0]
        rt = rt_all[rt_all["modele"] == m].iloc[0]
        rows.append((f"Réf. {m}", g["MAE_cm"], g["RMSE_cm"], g["R2"],
                      {k: mt[k] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]},
                      {k: rt[k] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]}))

    # ============================================================
    # Récap
    # ============================================================
    print("\n" + "=" * 80)
    print("RÉSULTATS GLOBAUX")
    print("=" * 80)
    print(f"{'Modèle':<40s} | {'MAE':>7s} | {'RMSE':>7s} | {'R²':>7s}")
    print("-" * 75)
    for label, mae_, rmse_, r2_, _, _ in rows:
        print(f"{label:<40s} | {mae_:>6.2f}c | {rmse_:>6.2f}c | {r2_:>7.4f}")

    print("\n" + "=" * 110)
    print("MAE PAR TRANCHE (cm)")
    print("=" * 110)
    print(f"{'Modèle':<40s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 105)
    for label, _, _, _, mt, _ in rows:
        print(f"{label:<40s} | {mt['≤1.10m']:>6.2f}c | {mt['1.15-1.20m']:>9.2f}c | "
              f"{mt['1.25-1.30m']:>9.2f}c | {mt['1.35-1.40m']:>9.2f}c | {mt['≥1.45m']:>6.2f}c")

    print("\n" + "=" * 110)
    print("RMSE PAR TRANCHE (cm)")
    print("=" * 110)
    print(f"{'Modèle':<40s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 105)
    for label, _, _, _, _, rt in rows:
        print(f"{label:<40s} | {rt['≤1.10m']:>6.2f}c | {rt['1.15-1.20m']:>9.2f}c | "
              f"{rt['1.25-1.30m']:>9.2f}c | {rt['1.35-1.40m']:>9.2f}c | {rt['≥1.45m']:>6.2f}c")

    out = pd.DataFrame([{
        "modele": label, "MAE_cm": mae_, "RMSE_cm": rmse_, "R2": r2_,
        **mt, **{f"RMSE_{k}": v for k, v in rt.items()}
    } for label, mae_, rmse_, r2_, mt, rt in rows])
    out.to_csv(MASTER_DIR / "stacking_hurdle_et_quantile_results.csv", index=False)
    print("\n→ stacking_hurdle_et_quantile_results.csv")


if __name__ == "__main__":
    main()
