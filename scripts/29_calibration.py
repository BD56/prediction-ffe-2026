"""
29 - Calibration post-hoc des prédictions RF.

Principe : le modèle "tire vers la moyenne" (sous-estime les tops,
surestime les bas). On apprend une fonction de correction sur les
prédictions du valid set, puis on l'applique au test.

Trois calibrations testées :
  A. Linéaire (y = a*x + b)
  B. Isotonic regression (monotone, flexible)
  C. Polynomiale degré 2 (légèrement non-linéaire)

Entrée : data/master/master_dataset_clean.parquet
Sortie : data/master/calibration_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
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
    print("=== 29 - Calibration post-hoc ===\n")

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

    # Imputation
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_test_imp = imputer.transform(X_test)

    # ============================================================
    # ÉTAPE 1 : RF default (modèle de base)
    # ============================================================
    print("[1/4] Entraînement RF default...")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=15,
        min_samples_leaf=10, min_samples_split=10,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_imp, y_train)
    print(f"  ✓ {time.time()-t0:.1f}s")

    # Prédictions brutes
    pred_valid_raw = rf.predict(X_valid_imp)
    pred_test_raw = rf.predict(X_test_imp)

    res_raw = evaluate(y_test, pred_test_raw, "RF default (brute)")
    tr_raw = mae_par_tranche(y_test, pred_test_raw)
    print(f"\nRF default brute (test) : MAE={res_raw['MAE_cm']:.2f}cm, R²={res_raw['R2']:.4f}")

    # ============================================================
    # ÉTAPE 2 : Calibration sur valid set
    # ============================================================
    print(f"\n[2/4] Apprentissage des calibrations sur valid (n={len(y_valid):,})...")

    # A. Linéaire
    calib_A = LinearRegression()
    calib_A.fit(pred_valid_raw.reshape(-1, 1), y_valid)
    print(f"  A. Linéaire : y_corrigé = {calib_A.coef_[0]:.4f} × y_pred + {calib_A.intercept_:.4f}")

    # B. Isotonic regression
    calib_B = IsotonicRegression(out_of_bounds="clip")
    calib_B.fit(pred_valid_raw, y_valid)
    print(f"  B. Isotonic : OK")

    # C. Polynomiale degré 2
    calib_C = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    calib_C.fit(pred_valid_raw.reshape(-1, 1), y_valid)
    print(f"  C. Polynomiale degré 2 : OK")

    # ============================================================
    # ÉTAPE 3 : Appliquer les calibrations sur test
    # ============================================================
    print("\n[3/4] Application au test...")
    pred_test_A = calib_A.predict(pred_test_raw.reshape(-1, 1))
    pred_test_B = calib_B.predict(pred_test_raw)
    pred_test_C = calib_C.predict(pred_test_raw.reshape(-1, 1))

    res_A = evaluate(y_test, pred_test_A, "calib_linéaire")
    res_B = evaluate(y_test, pred_test_B, "calib_isotonic")
    res_C = evaluate(y_test, pred_test_C, "calib_poly2")
    tr_A = mae_par_tranche(y_test, pred_test_A)
    tr_B = mae_par_tranche(y_test, pred_test_B)
    tr_C = mae_par_tranche(y_test, pred_test_C)

    # ============================================================
    # Récap
    # ============================================================
    print("\n" + "=" * 90)
    print("RÉCAP COMPARATIF")
    print("=" * 90)
    print(f"{'Stratégie':<22s} | {'Global':>7s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | {'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 100)
    for label, res, tr in [
        ("RF default (brute)", res_raw, tr_raw),
        ("calib linéaire (A)", res_A, tr_A),
        ("calib isotonic (B)", res_B, tr_B),
        ("calib poly2 (C)", res_C, tr_C),
    ]:
        print(f"{label:<22s} | {res['MAE_cm']:>6.2f}cm | "
              f"{tr.get('≤1.10m', 0):>6.2f}cm | "
              f"{tr.get('1.15-1.20m', 0):>9.2f}cm | "
              f"{tr.get('1.25-1.30m', 0):>9.2f}cm | "
              f"{tr.get('1.35-1.40m', 0):>9.2f}cm | "
              f"{tr.get('≥1.45m', 0):>6.2f}cm")

    pd.DataFrame([res_raw, res_A, res_B, res_C]).to_csv(
        MASTER_DIR / "calibration_results.csv", index=False)


if __name__ == "__main__":
    main()
