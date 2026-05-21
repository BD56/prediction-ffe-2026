"""
24 - Random Forest (régression continue).

Random Forest gère mieux le drift temporel que XGBoost (arbres indépendants).
Imputation médiane (plus rapide que KNN pour ce baseline).

Compare 2 versions :
  - RF default (paramètres raisonnables)
  - RF tuned (RandomizedSearchCV)

Entrée : data/master/master_dataset_clean.parquet
Sortie : data/master/rf_results.csv + rf_tuned_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


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
    print("=== 24 - Random Forest (régression continue) ===\n")

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
    print(f"Train : {len(X_train):,} | Valid : {len(X_valid):,} | Test : {len(X_test):,}")
    print(f"Features : {len(X.columns)}\n")

    # Imputation médiane (rapide vs KNN)
    print("Imputation par médiane (fit train)...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_test_imp = imputer.transform(X_test)
    print("  ✓\n")

    # ============================================================
    # PARTIE A : RF avec paramètres raisonnables
    # ============================================================
    print("=" * 60)
    print("PARTIE A : RandomForest default (paramètres raisonnables)")
    print("=" * 60)
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=10,
        min_samples_split=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_imp, y_train)
    print(f"✓ Entraînement : {time.time()-t0:.1f}s\n")

    y_train_pred = rf.predict(X_train_imp)
    y_valid_pred = rf.predict(X_valid_imp)
    y_test_pred = rf.predict(X_test_imp)
    results_default = []
    results_default.append(evaluate(y_train, y_train_pred, "train"))
    results_default.append(evaluate(y_valid, y_valid_pred, "valid"))
    results_default.append(evaluate(y_test, y_test_pred, "test"))

    print("\n=== MAE par tranche (test) ===")
    mae_par_tranche(y_test, y_test_pred)

    # Importance
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\n=== Top 15 features RF default ===")
    print(imp.head(15).to_string(index=False))

    pd.DataFrame(results_default).to_csv(MASTER_DIR / "rf_results.csv", index=False)
    imp.to_csv(MASTER_DIR / "rf_importance.csv", index=False)
    print(f"\n→ Résultats default : {MASTER_DIR / 'rf_results.csv'}")

    # ============================================================
    # PARTIE B : RF tuned par RandomizedSearchCV
    # ============================================================
    print("\n" + "=" * 60)
    print("PARTIE B : RandomForest tuned (RandomizedSearchCV)")
    print("=" * 60)
    param_dist = {
        "n_estimators": randint(100, 600),
        "max_depth": randint(5, 25),
        "min_samples_leaf": randint(5, 50),
        "min_samples_split": randint(5, 50),
        "max_features": [0.3, 0.5, 0.7, "sqrt", "log2"],
    }
    base = RandomForestRegressor(random_state=42, n_jobs=-1)
    print("\nLancement RandomizedSearchCV (n_iter=20, cv=3)...")
    t0 = time.time()
    search = RandomizedSearchCV(
        base, param_distributions=param_dist,
        n_iter=20, cv=3, scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train_imp, y_train)
    print(f"✓ Tuning : {time.time()-t0:.1f}s")
    print(f"\nBest params :")
    for k, v in search.best_params_.items():
        print(f"  {k} = {v}")
    print(f"Best CV MAE : {-search.best_score_*100:.2f}cm")

    best_rf = search.best_estimator_
    print("\n=== Évaluation RF tuned ===")
    y_train_pred = best_rf.predict(X_train_imp)
    y_valid_pred = best_rf.predict(X_valid_imp)
    y_test_pred = best_rf.predict(X_test_imp)
    results_tuned = []
    results_tuned.append(evaluate(y_train, y_train_pred, "train"))
    results_tuned.append(evaluate(y_valid, y_valid_pred, "valid"))
    results_tuned.append(evaluate(y_test, y_test_pred, "test"))

    print("\n=== MAE par tranche (test) ===")
    mae_par_tranche(y_test, y_test_pred)

    pd.DataFrame(results_tuned).to_csv(MASTER_DIR / "rf_tuned_results.csv", index=False)
    best_params_serializable = {k: (int(v) if isinstance(v, np.integer) else
                                    float(v) if isinstance(v, np.floating) else v)
                                for k, v in search.best_params_.items()}
    with open(MASTER_DIR / "rf_best_params.json", "w") as f:
        json.dump(best_params_serializable, f, indent=2)
    print(f"\n→ Résultats tuned : {MASTER_DIR / 'rf_tuned_results.csv'}")
    print(f"→ Best params : {MASTER_DIR / 'rf_best_params.json'}")

    # ============================================================
    # Récap comparatif
    # ============================================================
    print("\n" + "=" * 60)
    print("RÉCAP COMPARATIF")
    print("=" * 60)
    print(f"{'Modèle':<20s} | {'Train MAE':>10s} | {'Valid MAE':>10s} | {'Test MAE':>10s} | {'Test R²':>8s}")
    print("-" * 75)
    for name, results in [("RF default", results_default), ("RF tuned", results_tuned)]:
        print(f"{name:<20s} | {results[0]['MAE_cm']:>9.2f}cm | {results[1]['MAE_cm']:>9.2f}cm | {results[2]['MAE_cm']:>9.2f}cm | {results[2]['R2']:>8.4f}")


if __name__ == "__main__":
    main()
