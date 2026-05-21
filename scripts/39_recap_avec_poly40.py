"""
39 - Récap complet avec Poly40 ajouté.

Reprend les 9 modèles testés + ajoute Poly40 (degré 2 + interactions sur top 40).
Tableaux : global, MAE par tranche, RMSE par tranche.

Sortie : data/master/recap_avec_poly40.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

from sklearn.linear_model import ElasticNetCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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
    print("=== 39 - Récap complet avec Poly40 ===\n")

    # Refit Poly40 pour récupérer ses résultats par tranche
    print("Refit Poly40...")
    imp = pd.read_csv(MASTER_DIR / "top_flop_v2_avec_valeurs.csv")
    top40 = imp.sort_values("rank_moyen").head(40)["feature"].tolist()
    v2 = pd.read_parquet(MASTER_DIR / "master_dataset_epure_v2.parquet")
    v2 = v2.set_index("IDCHEVAL")
    for c in top40:
        if v2[c].dtype == bool:
            v2[c] = v2[c].astype(int)
    X = v2[top40]
    y = v2["hauteur_max_validee"]
    split = v2["SPLIT"]
    X_train, y_train = X[split == "train"], y[split == "train"]
    X_test, y_test = X[split == "test"], y[split == "test"]

    imp_med = SimpleImputer(strategy="median")
    X_train_med = imp_med.fit_transform(X_train)
    X_test_med = imp_med.transform(X_test)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train_med)
    X_test_sc = sc.transform(X_test_med)
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_sc)
    X_test_poly = poly.transform(X_test_sc)

    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=20000,
                      n_jobs=-1, random_state=42)
    en.fit(X_train_poly, y_train)
    pred_poly = en.predict(X_test_poly)
    mae, rmse, r2, mae_tr, rmse_tr = evaluate_all(y_test, pred_poly)
    print(f"  Poly40 : MAE={mae:.2f}cm, RMSE={rmse:.2f}cm, R²={r2:.4f}")

    # Charger les recaps existants (9 modèles v1)
    recap_glob = pd.read_csv(MASTER_DIR / "recap_global.csv")
    recap_mae = pd.read_csv(MASTER_DIR / "recap_mae_par_tranche.csv")
    recap_rmse = pd.read_csv(MASTER_DIR / "recap_rmse_par_tranche.csv")

    # Ajouter Poly40
    new_glob = pd.DataFrame([{"modele": "Poly40 (deg2 + interact.)",
                               "MAE_cm": mae, "RMSE_cm": rmse, "R2": r2}])
    new_mae = pd.DataFrame([{"modele": "Poly40 (deg2 + interact.)", **mae_tr}])
    new_rmse = pd.DataFrame([{"modele": "Poly40 (deg2 + interact.)", **rmse_tr}])

    recap_glob = pd.concat([recap_glob, new_glob], ignore_index=True).sort_values("MAE_cm")
    recap_mae = pd.concat([recap_mae, new_mae], ignore_index=True)
    recap_rmse = pd.concat([recap_rmse, new_rmse], ignore_index=True)

    # Reordonner selon MAE
    order = recap_glob["modele"].tolist()
    recap_mae["__o"] = recap_mae["modele"].map({m: i for i, m in enumerate(order)})
    recap_rmse["__o"] = recap_rmse["modele"].map({m: i for i, m in enumerate(order)})
    recap_mae = recap_mae.sort_values("__o").drop(columns="__o")
    recap_rmse = recap_rmse.sort_values("__o").drop(columns="__o")

    # ---------- Affichage ----------
    print("\n" + "=" * 80)
    print("TABLEAU 1 : MÉTRIQUES GLOBALES")
    print("=" * 80)
    print(f"{'Modèle':<28s} | {'MAE (cm)':>10s} | {'RMSE (cm)':>10s} | {'R²':>8s}")
    print("-" * 70)
    for _, r in recap_glob.iterrows():
        print(f"{r['modele']:<28s} | {r['MAE_cm']:>9.2f} | {r['RMSE_cm']:>9.2f} | {r['R2']:>7.4f}")

    print("\n" + "=" * 100)
    print("TABLEAU 2 : MAE PAR TRANCHE (cm)")
    print("=" * 100)
    print(recap_mae.set_index("modele").round(2).to_string())

    print("\n" + "=" * 100)
    print("TABLEAU 3 : RMSE PAR TRANCHE (cm)")
    print("=" * 100)
    print(recap_rmse.set_index("modele").round(2).to_string())

    # Sauvegarde
    recap_glob.to_csv(MASTER_DIR / "recap_avec_poly40_global.csv", index=False)
    recap_mae.to_csv(MASTER_DIR / "recap_avec_poly40_mae.csv", index=False)
    recap_rmse.to_csv(MASTER_DIR / "recap_avec_poly40_rmse.csv", index=False)
    print("\n→ recap_avec_poly40_{global,mae,rmse}.csv")


if __name__ == "__main__":
    main()
