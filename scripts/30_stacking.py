"""
30 - Stacking (RF + ElasticNet + CatBoost).

Approche :
1. Modèles base : RF, ElasticNet, CatBoost (diversité)
2. Chaque modèle prédit sur valid (set d'apprentissage du méta-modèle)
   et sur test
3. Méta-modèle (régression linéaire) apprend à combiner les 3 prédictions
   sur valid → y_valid
4. Évaluation finale sur test

Variante testée en bonus : Stacking + calibration post-hoc.

Entrée : data/master/master_dataset_clean.parquet
Sortie : data/master/stacking_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def evaluate(y_true, y_pred, label="test"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"label": label, "MAE_cm": mae*100, "RMSE_cm": rmse*100, "R2": r2}


def mae_par_tranche(y_test, y_pred):
    df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    df["tranche"] = pd.cut(df["y_true"],
                            bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])
    df["err_abs"] = (df["y_true"] - df["y_pred"]).abs() * 100
    result = {}
    for tranche in df["tranche"].cat.categories:
        sub = df[df["tranche"] == tranche]
        if len(sub) > 0:
            result[str(tranche)] = sub["err_abs"].mean()
    return result


def main():
    print("=== 30 - Stacking (RF + ElasticNet + CatBoost) ===\n")

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
    print(f"Train : {len(X_train):,} | Valid : {len(X_valid):,} | Test : {len(X_test):,}\n")

    # ============================================================
    # Préparation des données pour les différents modèles
    # ============================================================
    # Pour RF et ElasticNet : imputation médiane (rapide)
    imp_med = SimpleImputer(strategy="median")
    X_train_med = imp_med.fit_transform(X_train)
    X_valid_med = imp_med.transform(X_valid)
    X_test_med = imp_med.transform(X_test)

    # Pour ElasticNet : standardiser après imputation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_med)
    X_valid_scaled = scaler.transform(X_valid_med)
    X_test_scaled = scaler.transform(X_test_med)

    # ============================================================
    # Modèle 1 : Random Forest
    # ============================================================
    print("[1/3] Random Forest...")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=15,
        min_samples_leaf=10, min_samples_split=10,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_med, y_train)
    pred_valid_rf = rf.predict(X_valid_med)
    pred_test_rf = rf.predict(X_test_med)
    print(f"  ✓ {time.time()-t0:.1f}s | Test MAE = {mean_absolute_error(y_test, pred_test_rf)*100:.2f}cm")

    # ============================================================
    # Modèle 2 : ElasticNet
    # ============================================================
    print("\n[2/3] ElasticNet...")
    t0 = time.time()
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                      n_jobs=-1, random_state=42)
    en.fit(X_train_scaled, y_train)
    pred_valid_en = en.predict(X_valid_scaled)
    pred_test_en = en.predict(X_test_scaled)
    print(f"  ✓ {time.time()-t0:.1f}s | Test MAE = {mean_absolute_error(y_test, pred_test_en)*100:.2f}cm")

    # ============================================================
    # Modèle 3 : CatBoost (gère NaN nativement)
    # ============================================================
    print("\n[3/3] CatBoost...")
    t0 = time.time()
    cb = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        l2_leaf_reg=3, random_seed=42, loss_function="MAE",
        eval_metric="MAE", early_stopping_rounds=30, verbose=0
    )
    cb.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    pred_valid_cb = cb.predict(X_valid)
    pred_test_cb = cb.predict(X_test)
    print(f"  ✓ {time.time()-t0:.1f}s | Test MAE = {mean_absolute_error(y_test, pred_test_cb)*100:.2f}cm")

    # ============================================================
    # Méta-modèle : combinaison des 3 prédictions
    # ============================================================
    print("\n" + "=" * 60)
    print("MÉTA-MODÈLE : régression linéaire sur les 3 prédictions")
    print("=" * 60)
    X_meta_valid = np.column_stack([pred_valid_rf, pred_valid_en, pred_valid_cb])
    X_meta_test = np.column_stack([pred_test_rf, pred_test_en, pred_test_cb])

    meta = LinearRegression()
    meta.fit(X_meta_valid, y_valid)
    print(f"\nPoids du méta-modèle :")
    print(f"  RF       : {meta.coef_[0]:>7.4f}")
    print(f"  ElasticNet : {meta.coef_[1]:>7.4f}")
    print(f"  CatBoost   : {meta.coef_[2]:>7.4f}")
    print(f"  Intercept  : {meta.intercept_:>7.4f}")
    print(f"  Somme coef : {sum(meta.coef_):>7.4f}")

    pred_test_stack = meta.predict(X_meta_test)
    res_stack = evaluate(y_test, pred_test_stack, "stacking")
    tr_stack = mae_par_tranche(y_test, pred_test_stack)
    print(f"\nStacking (test) : MAE={res_stack['MAE_cm']:.2f}cm, R²={res_stack['R2']:.4f}")

    # ============================================================
    # Bonus : Stacking + Calibration (isotonic sur valid)
    # ============================================================
    print("\n" + "=" * 60)
    print("BONUS : Stacking + Calibration isotonic")
    print("=" * 60)
    pred_valid_stack = meta.predict(X_meta_valid)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(pred_valid_stack, y_valid)
    pred_test_stack_cal = cal.predict(pred_test_stack)
    res_stack_cal = evaluate(y_test, pred_test_stack_cal, "stacking+calib")
    tr_stack_cal = mae_par_tranche(y_test, pred_test_stack_cal)
    print(f"\nStacking + Calib (test) : MAE={res_stack_cal['MAE_cm']:.2f}cm, R²={res_stack_cal['R2']:.4f}")

    # ============================================================
    # Récap final TOUT compris
    # ============================================================
    print("\n" + "=" * 95)
    print("RÉCAP COMPARATIF FINAL")
    print("=" * 95)
    print(f"{'Modèle':<28s} | {'Global':>7s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | {'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 110)

    individuals = [
        ("RF default", pred_test_rf),
        ("ElasticNet", pred_test_en),
        ("CatBoost", pred_test_cb),
        ("Stacking", pred_test_stack),
        ("Stacking + Calib", pred_test_stack_cal),
    ]
    rows = []
    for label, pred in individuals:
        res = evaluate(y_test, pred, label)
        tr = mae_par_tranche(y_test, pred)
        print(f"{label:<28s} | {res['MAE_cm']:>6.2f}cm | "
              f"{tr.get('≤1.10m', 0):>6.2f}cm | "
              f"{tr.get('1.15-1.20m', 0):>9.2f}cm | "
              f"{tr.get('1.25-1.30m', 0):>9.2f}cm | "
              f"{tr.get('1.35-1.40m', 0):>9.2f}cm | "
              f"{tr.get('≥1.45m', 0):>6.2f}cm")
        rows.append(res)

    pd.DataFrame(rows).to_csv(MASTER_DIR / "stacking_results.csv", index=False)


if __name__ == "__main__":
    main()
