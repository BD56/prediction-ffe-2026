"""
26 - CatBoost (régression continue).

Avantages :
- Gère nativement les NaN (sans imputation -> évite le bruit)
- Plus robuste au surapprentissage que XGBoost en général
- Tuning souvent peu nécessaire (paramètres par défaut bien calibrés)

Entrée : data/master/master_dataset_clean.parquet
Sortie : data/master/catboost_results.csv + importance
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    print("=== 26 - CatBoost (régression continue) ===\n")

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
    print(f"Features : {len(X.columns)}")
    print(f"NaN dans X_train : {X_train.isna().sum().sum():,} (CatBoost les gère nativement)\n")

    # ============================================================
    # CatBoost default (paramètres raisonnables)
    # ============================================================
    print("[1/2] Entraînement CatBoost...")
    t0 = time.time()
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        loss_function="MAE",  # cohérent avec notre métrique de communication
        eval_metric="MAE",
        early_stopping_rounds=50,
        verbose=100,
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    print(f"  ✓ {time.time()-t0:.1f}s")
    print(f"  Best iteration : {model.get_best_iteration()}")

    # ============================================================
    # Évaluation
    # ============================================================
    print(f"\n[2/2] Évaluation...")
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)
    results = []
    results.append(evaluate(y_train, y_train_pred, "train"))
    results.append(evaluate(y_valid, y_valid_pred, "valid"))
    results.append(evaluate(y_test, y_test_pred, "test"))

    print("\n=== MAE par tranche (test) ===")
    mae_par_tranche(y_test, y_test_pred)

    # Feature importance
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\n=== Top 15 features ===")
    print(imp.head(15).to_string(index=False))

    # Sauvegardes
    pd.DataFrame(results).to_csv(MASTER_DIR / "catboost_results.csv", index=False)
    imp.to_csv(MASTER_DIR / "catboost_importance.csv", index=False)

    # ============================================================
    # Récap comparatif
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARAISON AVEC AUTRES MODÈLES")
    print("=" * 70)
    print(f"{'Modèle':<25s} | {'Test MAE':>10s} | {'Test RMSE':>10s} | {'Test R²':>8s}")
    print("-" * 65)
    print(f"{'ElasticNet':<25s} | {'6.67cm':>10s} | {'8.59cm':>10s} | {'0.5280':>8s}")
    print(f"{'XGBoost default':<25s} | {'7.91cm':>10s} | {'9.74cm':>10s} | {'0.3933':>8s}")
    print(f"{'RF default':<25s} | {'6.51cm':>10s} | {'8.35cm':>10s} | {'0.5541':>8s}")
    print(f"{'CatBoost (loss=MAE)':<25s} | {results[2]['MAE_cm']:>9.2f}cm | {results[2]['RMSE_cm']:>9.2f}cm | {results[2]['R2']:>8.4f}")


if __name__ == "__main__":
    main()
