"""
25 - Random Forest tuned avec TimeSeriesSplit.

Au lieu d'un CV classique (qui mélange les générations), on utilise
TimeSeriesSplit qui respecte l'ordre temporel :
- Fold 1 : train sur 2006-2007, test sur 2008
- Fold 2 : train sur 2006-2008, test sur 2009
- Fold 3 : train sur 2006-2009, test sur 2010
(approximation, dépend du nb de folds)

Objectif : voir si ce CV temporel donne un score CV plus représentatif
de la performance attendue sur test (cohorte 2013).

Entrée : data/master/master_dataset_clean.parquet
Sortie : data/master/rf_timeseries_results.csv + best_params.json
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR

N_ITER = 20
N_SPLITS_TS = 4  # nb de folds TimeSeriesSplit


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:>6s} : MAE={mae*100:.2f}cm, RMSE={rmse*100:.2f}cm, R²={r2:.4f}")
    return {"label": label, "MAE_cm": mae*100, "RMSE_cm": rmse*100, "R2": r2}


def mae_par_tranche(y_test, y_pred):
    df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    df["tranche"] = pd.cut(df["y_true"],
                            bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])
    df["err_abs"] = (df["y_true"] - df["y_pred"]).abs() * 100
    for tranche in df["tranche"].cat.categories:
        sub = df[df["tranche"] == tranche]
        if len(sub) > 0:
            print(f"  {str(tranche):>12s} (n={len(sub):>5,}) : MAE = {sub['err_abs'].mean():>5.2f}cm")


def main():
    print(f"=== 25 - Random Forest tuned avec TimeSeriesSplit ===\n")

    master = pd.read_parquet(MASTER_DIR / "master_dataset_clean.parquet")
    master = master.set_index("IDCHEVAL")
    feat_cols = [c for c in master.columns if c.startswith("f")]
    for c in feat_cols:
        if master[c].dtype == bool:
            master[c] = master[c].astype(int)

    # Trier par DATENAISSANCE pour que TimeSeriesSplit respecte la chronologie
    master = master.sort_values("DATENAISSANCE")

    X = master[feat_cols].select_dtypes(include=[np.number])
    y = master["hauteur_max_validee"]
    split = master["SPLIT"]

    # Train, valid, test
    X_train, y_train = X[split == "train"], y[split == "train"]
    X_valid, y_valid = X[split == "valid"], y[split == "valid"]
    X_test, y_test = X[split == "test"], y[split == "test"]
    print(f"Train : {len(X_train):,} (2006-2010, triés par DATENAISSANCE)")
    print(f"Valid : {len(X_valid):,} | Test : {len(X_test):,}")

    # Imputation médiane
    print("\nImputation médiane (fit train)...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_test_imp = imputer.transform(X_test)

    # ============================================================
    # TimeSeriesSplit
    # ============================================================
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_TS)
    print(f"\nTimeSeriesSplit avec {N_SPLITS_TS} folds :")
    for i, (tr_idx, val_idx) in enumerate(tscv.split(X_train_imp)):
        print(f"  Fold {i+1} : train [{tr_idx[0]:>5}..{tr_idx[-1]:>5}] "
              f"(n={len(tr_idx):>5}), val [{val_idx[0]:>5}..{val_idx[-1]:>5}] "
              f"(n={len(val_idx):>5})")

    # ============================================================
    # Grille raisonnable (régularisée pour limiter overfitting)
    # ============================================================
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(5, 15),  # plus contraint que la dernière fois
        "min_samples_leaf": randint(10, 50),  # plus de feuilles pleines
        "min_samples_split": randint(10, 50),
        "max_features": [0.3, 0.5, 0.7, "sqrt"],
    }
    base = RandomForestRegressor(random_state=42, n_jobs=-1)

    print(f"\nLancement RandomizedSearchCV avec TimeSeriesSplit (n_iter={N_ITER})...")
    t0 = time.time()
    search = RandomizedSearchCV(
        base, param_distributions=param_dist,
        n_iter=N_ITER, cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train_imp, y_train)
    print(f"✓ Tuning : {time.time()-t0:.1f}s")
    print(f"\nBest params (TimeSeriesSplit) :")
    for k, v in search.best_params_.items():
        print(f"  {k} = {v}")
    print(f"Best CV MAE : {-search.best_score_*100:.2f}cm")

    # ============================================================
    # Évaluation
    # ============================================================
    best_rf = search.best_estimator_
    print("\n=== Évaluation ===")
    y_train_pred = best_rf.predict(X_train_imp)
    y_valid_pred = best_rf.predict(X_valid_imp)
    y_test_pred = best_rf.predict(X_test_imp)
    results = []
    results.append(evaluate(y_train, y_train_pred, "train"))
    results.append(evaluate(y_valid, y_valid_pred, "valid"))
    results.append(evaluate(y_test, y_test_pred, "test"))

    print("\n=== MAE par tranche (test) ===")
    mae_par_tranche(y_test, y_test_pred)

    # Sauvegardes
    pd.DataFrame(results).to_csv(MASTER_DIR / "rf_timeseries_results.csv", index=False)
    bp_serial = {k: (int(v) if isinstance(v, np.integer) else
                     float(v) if isinstance(v, np.floating) else v)
                 for k, v in search.best_params_.items()}
    with open(MASTER_DIR / "rf_timeseries_best_params.json", "w") as f:
        json.dump(bp_serial, f, indent=2)
    print(f"\n→ Résultats : {MASTER_DIR / 'rf_timeseries_results.csv'}")

    # ============================================================
    # Comparaison finale
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARAISON FINALE")
    print("=" * 70)
    print(f"{'Modèle':<25s} | {'Train MAE':>10s} | {'Valid MAE':>10s} | {'Test MAE':>10s} | {'Test R²':>8s}")
    print("-" * 80)
    print(f"{'ElasticNet':<25s} | {'6.05cm':>10s} | {'6.23cm':>10s} | {'6.67cm':>10s} | {'0.5280':>8s}")
    print(f"{'XGBoost default':<25s} | {'5.01cm':>10s} | {'7.54cm':>10s} | {'7.91cm':>10s} | {'0.3933':>8s}")
    print(f"{'XGBoost tuned (KFold)':<25s} | {'1.20cm':>10s} | {'9.43cm':>10s} | {'9.77cm':>10s} | {'0.1103':>8s}")
    print(f"{'RF default':<25s} | {'4.57cm':>10s} | {'6.13cm':>10s} | {'6.51cm':>10s} | {'0.5541':>8s}")
    print(f"{'RF tuned (KFold)':<25s} | {'1.71cm':>10s} | {'7.67cm':>10s} | {'8.13cm':>10s} | {'0.3617':>8s}")
    print(f"{'RF tuned (TimeSeriesSplit)':<25s} | {results[0]['MAE_cm']:>9.2f}cm | {results[1]['MAE_cm']:>9.2f}cm | {results[2]['MAE_cm']:>9.2f}cm | {results[2]['R2']:>8.4f}")


if __name__ == "__main__":
    main()
