"""
23 - XGBoost tuning par RandomizedSearchCV.

Recherche aléatoire d'hyperparamètres sur une grille raisonnable.
Cross-validation 3-fold pour limiter le temps de calcul.

Hyperparamètres tunés :
  - max_depth, min_child_weight (complexité des arbres)
  - subsample, colsample_bytree (régularisation par sampling)
  - reg_alpha, reg_lambda (régularisation L1/L2)
  - learning_rate, n_estimators (apprentissage)
  - gamma (gain minimum de split)

Entrée : data/master/master_dataset_clean.parquet
Sortie : data/master/xgboost_tuned_results.csv + best_params.json
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import time

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import uniform, randint

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR

N_ITER = 30  # nb de combinaisons d'hyperparamètres à tester
CV_FOLDS = 3


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:>6s} : MAE={mae*100:.2f}cm, RMSE={rmse*100:.2f}cm, R²={r2:.4f}")
    return {"label": label, "MAE_cm": mae*100, "RMSE_cm": rmse*100, "R2": r2}


def main():
    print(f"=== 23 - XGBoost tuning (RandomizedSearchCV, n_iter={N_ITER}, cv={CV_FOLDS}) ===\n")

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

    # ============================================================
    # Grille d'hyperparamètres
    # ============================================================
    param_dist = {
        "n_estimators": randint(100, 800),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3),  # 0.01 à 0.31
        "subsample": uniform(0.6, 0.4),  # 0.6 à 1.0
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 30),
        "reg_alpha": uniform(0, 2),
        "reg_lambda": uniform(0, 5),
        "gamma": uniform(0, 1),
    }

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        eval_metric="mae",
    )

    print(f"Lancement RandomizedSearchCV (peut prendre 20-60 min)...")
    t0 = time.time()
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
        verbose=2,
    )
    search.fit(X_train, y_train)
    print(f"\n✓ Tuning terminé en {time.time()-t0:.1f}s")
    print(f"\nMeilleurs hyperparamètres :")
    for k, v in search.best_params_.items():
        print(f"  {k} = {v}")
    print(f"\nMeilleur score CV (MAE) : {-search.best_score_*100:.2f}cm")

    # ============================================================
    # Évaluation du meilleur modèle
    # ============================================================
    best_model = search.best_estimator_
    print("\n=== Évaluation ===")
    y_train_pred = best_model.predict(X_train)
    y_valid_pred = best_model.predict(X_valid)
    y_test_pred = best_model.predict(X_test)
    results = []
    results.append(evaluate(y_train, y_train_pred, "train"))
    results.append(evaluate(y_valid, y_valid_pred, "valid"))
    results.append(evaluate(y_test, y_test_pred, "test"))

    # MAE par tranche
    print("\n=== MAE par tranche (test) ===")
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

    # Feature importance
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\n=== Top 20 features ===")
    print(imp.head(20).to_string(index=False))

    # Sauvegardes
    pd.DataFrame(results).to_csv(MASTER_DIR / "xgboost_tuned_results.csv", index=False)
    imp.to_csv(MASTER_DIR / "xgboost_tuned_importance.csv", index=False)
    # Best params
    best_params_serializable = {k: (int(v) if isinstance(v, np.integer) else
                                    float(v) if isinstance(v, np.floating) else v)
                                for k, v in search.best_params_.items()}
    with open(MASTER_DIR / "xgboost_best_params.json", "w") as f:
        json.dump(best_params_serializable, f, indent=2)
    print(f"\n→ Résultats : {MASTER_DIR / 'xgboost_tuned_results.csv'}")
    print(f"→ Best params : {MASTER_DIR / 'xgboost_best_params.json'}")


if __name__ == "__main__":
    main()
