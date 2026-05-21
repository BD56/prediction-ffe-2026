"""
21 - Baseline ElasticNet (régression continue).

Pipeline : StandardScaler → KNNImputer(k=5, weights="distance") → ElasticNetCV.
Cible : hauteur_max_validee (régression continue).
Split : time series par génération (train/valid/test).

Entrée : data/master/master_dataset_clean.parquet (TE corrigé)
Sortie : data/master/baseline_elasticnet_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR

K_KNN = 5


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:>6s} : MAE={mae*100:.2f}cm, RMSE={rmse*100:.2f}cm, R²={r2:.4f}")
    return {"label": label, "MAE_cm": mae*100, "RMSE_cm": rmse*100, "R2": r2}


def main():
    print("=== 21 - Baseline ElasticNet (régression continue) ===\n")

    # Chargement
    master = pd.read_parquet(MASTER_DIR / "master_dataset_clean.parquet")
    master = master.set_index("IDCHEVAL")
    print(f"Master clean : {len(master):,} × {len(master.columns)}")

    # Features = colonnes f*
    feat_cols = [c for c in master.columns if c.startswith("f")]
    # Convertir booléens en int si présents
    for c in feat_cols:
        if master[c].dtype == bool:
            master[c] = master[c].astype(int)

    X = master[feat_cols]
    y = master["hauteur_max_validee"]
    split = master["SPLIT"]

    # Garder uniquement les colonnes numériques (sécurité)
    X = X.select_dtypes(include=[np.number])
    print(f"Features numériques : {len(X.columns)}")

    # Split
    X_train = X[split == "train"]
    y_train = y[split == "train"]
    X_valid = X[split == "valid"]
    y_valid = y[split == "valid"]
    X_test = X[split == "test"]
    y_test = y[split == "test"]
    print(f"Train : {len(X_train):,} | Valid : {len(X_valid):,} | Test : {len(X_test):,}")

    # NaN par split
    print(f"\nNaN dans X_train : {X_train.isna().sum().sum():,} (sur {X_train.size:,})")

    # ============================================================
    # Pipeline : StandardScaler → KNNImputer → ElasticNetCV
    # ============================================================
    print("\n[1/4] StandardScaler (fit train)...")
    t0 = time.time()
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_valid_sc = scaler.transform(X_valid)
    X_test_sc = scaler.transform(X_test)
    print(f"  ✓ {time.time()-t0:.1f}s")

    print(f"\n[2/4] KNNImputer (K={K_KNN}, weights=distance, fit train)...")
    print("  Peut prendre quelques minutes...")
    t0 = time.time()
    imputer = KNNImputer(n_neighbors=K_KNN, weights="distance")
    X_train_imp = imputer.fit_transform(X_train_sc)
    X_valid_imp = imputer.transform(X_valid_sc)
    X_test_imp = imputer.transform(X_test_sc)
    print(f"  ✓ {time.time()-t0:.1f}s")

    print(f"\n[3/4] ElasticNetCV (cross-validation 5-fold sur train)...")
    t0 = time.time()
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        cv=5,
        max_iter=10000,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_imp, y_train)
    print(f"  ✓ {time.time()-t0:.1f}s")
    print(f"  Meilleurs hyperparamètres : alpha={model.alpha_:.4f}, l1_ratio={model.l1_ratio_}")
    n_nonzero = (model.coef_ != 0).sum()
    print(f"  Nb features avec coef non-nul : {n_nonzero}/{len(X.columns)}")

    print(f"\n[4/4] Évaluation...")
    y_train_pred = model.predict(X_train_imp)
    y_valid_pred = model.predict(X_valid_imp)
    y_test_pred = model.predict(X_test_imp)
    results = []
    results.append(evaluate(y_train, y_train_pred, "train"))
    results.append(evaluate(y_valid, y_valid_pred, "valid"))
    results.append(evaluate(y_test, y_test_pred, "test"))

    # ============================================================
    # MAE par tranche de cible (sur test)
    # ============================================================
    print("\n=== MAE par tranche de cible (sur test) ===")
    test_df = pd.DataFrame({"y_true": y_test, "y_pred": y_test_pred})
    test_df["tranche"] = pd.cut(test_df["y_true"],
                                 bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                                 labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                         "1.35-1.40m", "≥1.45m"])
    test_df["err_abs"] = (test_df["y_true"] - test_df["y_pred"]).abs() * 100  # cm
    for tranche in test_df["tranche"].cat.categories:
        sub = test_df[test_df["tranche"] == tranche]
        if len(sub) > 0:
            print(f"  {str(tranche):>12s} (n={len(sub):>5,}) : MAE = {sub['err_abs'].mean():>5.2f}cm")

    # ============================================================
    # Top features par importance (coefficients absolus)
    # ============================================================
    print("\n=== Top 20 features par |coefficient| ===")
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coef": model.coef_,
        "abs_coef": np.abs(model.coef_)
    }).sort_values("abs_coef", ascending=False)
    print(coef_df.head(20).to_string(index=False))

    # Sauvegarde
    results_df = pd.DataFrame(results)
    results_df.to_csv(MASTER_DIR / "baseline_elasticnet_results.csv", index=False)
    coef_df.to_csv(MASTER_DIR / "baseline_elasticnet_coefficients.csv", index=False)
    print(f"\n→ Résultats : {MASTER_DIR / 'baseline_elasticnet_results.csv'}")
    print(f"→ Coefficients : {MASTER_DIR / 'baseline_elasticnet_coefficients.csv'}")


if __name__ == "__main__":
    main()
