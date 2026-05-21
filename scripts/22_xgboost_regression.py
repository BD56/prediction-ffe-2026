"""
22 - XGBoost (régression continue).

Avantages vs ElasticNet :
- Gère nativement les NaN (pas besoin de KNN imputation -> gain ~12 min)
- Pas de standardisation requise
- Modèle non-linéaire, capable de capter les interactions

Hyperparamètres : valeurs raisonnables par défaut.
Tuning fin par RandomizedSearchCV possible plus tard si besoin.

Entrée : data/master/master_dataset_clean.parquet
Sortie : data/master/xgboost_results.csv + feature_importance.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:>6s} : MAE={mae*100:.2f}cm, RMSE={rmse*100:.2f}cm, R²={r2:.4f}")
    return {"label": label, "MAE_cm": mae*100, "RMSE_cm": rmse*100, "R2": r2}


def main():
    print("=== 22 - XGBoost (régression continue) ===\n")

    # Chargement
    master = pd.read_parquet(MASTER_DIR / "master_dataset_clean.parquet")
    master = master.set_index("IDCHEVAL")
    feat_cols = [c for c in master.columns if c.startswith("f")]

    # Convertir booléens en int
    for c in feat_cols:
        if master[c].dtype == bool:
            master[c] = master[c].astype(int)

    X = master[feat_cols].select_dtypes(include=[np.number])
    y = master["hauteur_max_validee"]
    split = master["SPLIT"]

    X_train = X[split == "train"]
    y_train = y[split == "train"]
    X_valid = X[split == "valid"]
    y_valid = y[split == "valid"]
    X_test = X[split == "test"]
    y_test = y[split == "test"]
    print(f"Features : {len(X.columns)}")
    print(f"Train : {len(X_train):,} | Valid : {len(X_valid):,} | Test : {len(X_test):,}")
    print(f"NaN dans X_train : {X_train.isna().sum().sum():,} (XGBoost les gère nativement)")

    # ============================================================
    # XGBoost avec hyperparamètres raisonnables
    # ============================================================
    print("\n[1/3] Entraînement XGBoost...")
    t0 = time.time()
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=10,
        random_state=42,
        n_jobs=-1,
        eval_metric="mae",
        early_stopping_rounds=30,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=50
    )
    print(f"  ✓ {time.time()-t0:.1f}s")
    print(f"  Best iteration : {model.best_iteration}")

    # ============================================================
    # Évaluation
    # ============================================================
    print(f"\n[2/3] Évaluation...")
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)
    results = []
    results.append(evaluate(y_train, y_train_pred, "train"))
    results.append(evaluate(y_valid, y_valid_pred, "valid"))
    results.append(evaluate(y_test, y_test_pred, "test"))

    # MAE par tranche
    print("\n=== MAE par tranche de cible (sur test) ===")
    test_df = pd.DataFrame({"y_true": y_test, "y_pred": y_test_pred})
    test_df["tranche"] = pd.cut(test_df["y_true"],
                                 bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                                 labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                         "1.35-1.40m", "≥1.45m"])
    test_df["err_abs"] = (test_df["y_true"] - test_df["y_pred"]).abs() * 100
    for tranche in test_df["tranche"].cat.categories:
        sub = test_df[test_df["tranche"] == tranche]
        if len(sub) > 0:
            print(f"  {str(tranche):>12s} (n={len(sub):>5,}) : MAE = {sub['err_abs'].mean():>5.2f}cm")

    # ============================================================
    # Feature importance (gain)
    # ============================================================
    print(f"\n[3/3] Feature importance (gain)...")
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\n=== Top 20 features par importance (gain) ===")
    print(imp.head(20).to_string(index=False))

    # Sauvegardes
    pd.DataFrame(results).to_csv(MASTER_DIR / "xgboost_results.csv", index=False)
    imp.to_csv(MASTER_DIR / "xgboost_feature_importance.csv", index=False)
    print(f"\n→ Résultats : {MASTER_DIR / 'xgboost_results.csv'}")
    print(f"→ Importance : {MASTER_DIR / 'xgboost_feature_importance.csv'}")


if __name__ == "__main__":
    main()
