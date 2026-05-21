"""
45 - Piste 1 : transformation de la cible (y², y³).

Idée : entraîner le modèle à prédire y² (ou y³), puis inverser par sqrt (ou cbrt).
Justification : étirer mathématiquement les hauts y pour que le modèle "ose"
prédire plus haut sur les tops.

Tests sur deux baselines :
  - RF default v2
  - Poly40 (degré 2 + interactions)

Pour chaque, on compare :
  - cible = y (baseline)
  - cible = y² → prédiction inversée par sqrt
  - cible = y³ → prédiction inversée par cbrt

Sortie : data/master/target_transformation_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor
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


def fit_predict_rf(X_train, X_test, y_train_t, inverse_fn):
    rf = RandomForestRegressor(n_estimators=500, max_depth=15,
                                 min_samples_leaf=10, min_samples_split=10,
                                 max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_t)
    pred_t = rf.predict(X_test)
    return inverse_fn(pred_t)


def fit_predict_poly(X_train_sc, X_test_sc, y_train_t, inverse_fn):
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_sc)
    X_test_poly = poly.transform(X_test_sc)
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=20000,
                      n_jobs=-1, random_state=42)
    en.fit(X_train_poly, y_train_t)
    pred_t = en.predict(X_test_poly)
    return inverse_fn(pred_t)


def main():
    print("=== 45 - Piste 1 : transformation de la cible ===\n")

    # ---------- Données ----------
    v2 = pd.read_parquet(MASTER_DIR / "master_dataset_epure_v2.parquet")
    v2 = v2.set_index("IDCHEVAL")
    feat_cols = [c for c in v2.columns if c.startswith("f")]
    for c in feat_cols:
        if v2[c].dtype == bool:
            v2[c] = v2[c].astype(int)
    X_all = v2[feat_cols].select_dtypes(include=[np.number])
    y = v2["hauteur_max_validee"]
    split = v2["SPLIT"]

    # RF : sur 156 features de v2
    X_train, y_train = X_all[split == "train"], y[split == "train"]
    X_test, y_test = X_all[split == "test"], y[split == "test"]
    imp = SimpleImputer(strategy="median")
    Xtr_med = imp.fit_transform(X_train)
    Xte_med = imp.transform(X_test)

    # Poly40 : sur top 40 + standardisation
    imp_t = pd.read_csv(MASTER_DIR / "top_flop_v2_avec_valeurs.csv")
    top40 = imp_t.sort_values("rank_moyen").head(40)["feature"].tolist()
    X_train_t = X_train[top40]
    X_test_t = X_test[top40]
    imp40 = SimpleImputer(strategy="median")
    Xtr40 = imp40.fit_transform(X_train_t)
    Xte40 = imp40.transform(X_test_t)
    sc = StandardScaler()
    Xtr40_sc = sc.fit_transform(Xtr40)
    Xte40_sc = sc.transform(Xte40)

    rows = []

    # =============================
    # RF default sur 156 features
    # =============================
    print("=== RF default (156 features v2) ===\n")
    for label, transform, inverse in [
        ("baseline (y)", lambda v: v, lambda v: v),
        ("y²",  lambda v: v**2, lambda v: np.sqrt(np.maximum(v, 0))),
        ("y³",  lambda v: v**3, lambda v: np.cbrt(v)),
    ]:
        print(f"  Cible = {label}...", end=" ", flush=True)
        t0 = time.time()
        y_t = transform(y_train.values)
        pred = fit_predict_rf(Xtr_med, Xte_med, y_t, inverse)
        mae, rmse, r2, mtr, rtr = evaluate_all(y_test, pred)
        print(f"{time.time()-t0:.1f}s | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
        rows.append((f"RF + {label}", mae, rmse, r2, mtr, rtr))

    # =============================
    # Poly40
    # =============================
    print("\n=== Poly40 (top 40 + interactions) ===\n")
    for label, transform, inverse in [
        ("baseline (y)", lambda v: v, lambda v: v),
        ("y²",  lambda v: v**2, lambda v: np.sqrt(np.maximum(v, 0))),
        ("y³",  lambda v: v**3, lambda v: np.cbrt(v)),
    ]:
        print(f"  Cible = {label}...", end=" ", flush=True)
        t0 = time.time()
        y_t = transform(y_train.values)
        pred = fit_predict_poly(Xtr40_sc, Xte40_sc, y_t, inverse)
        mae, rmse, r2, mtr, rtr = evaluate_all(y_test, pred)
        print(f"{time.time()-t0:.1f}s | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
        rows.append((f"Poly40 + {label}", mae, rmse, r2, mtr, rtr))

    # ---------- Récap ----------
    print("\n" + "=" * 80)
    print("RÉSULTATS GLOBAUX")
    print("=" * 80)
    print(f"{'Modèle':<28s} | {'MAE':>7s} | {'RMSE':>7s} | {'R²':>7s}")
    print("-" * 65)
    for label, mae_, rmse_, r2_, _, _ in rows:
        print(f"{label:<28s} | {mae_:>6.2f}c | {rmse_:>6.2f}c | {r2_:>7.4f}")

    print("\n" + "=" * 105)
    print("MAE PAR TRANCHE (cm)")
    print("=" * 105)
    print(f"{'Modèle':<28s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 95)
    for label, _, _, _, mt, _ in rows:
        print(f"{label:<28s} | {mt['≤1.10m']:>6.2f}c | {mt['1.15-1.20m']:>9.2f}c | "
              f"{mt['1.25-1.30m']:>9.2f}c | {mt['1.35-1.40m']:>9.2f}c | {mt['≥1.45m']:>6.2f}c")

    print("\n" + "=" * 105)
    print("RMSE PAR TRANCHE (cm)")
    print("=" * 105)
    print(f"{'Modèle':<28s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 95)
    for label, _, _, _, _, rt in rows:
        print(f"{label:<28s} | {rt['≤1.10m']:>6.2f}c | {rt['1.15-1.20m']:>9.2f}c | "
              f"{rt['1.25-1.30m']:>9.2f}c | {rt['1.35-1.40m']:>9.2f}c | {rt['≥1.45m']:>6.2f}c")

    out = pd.DataFrame([{
        "modele": label, "MAE_cm": mae_, "RMSE_cm": rmse_, "R2": r2_,
        **mt, **{f"RMSE_{k}": v for k, v in rt.items()}
    } for label, mae_, rmse_, r2_, mt, rt in rows])
    out.to_csv(MASTER_DIR / "target_transformation_results.csv", index=False)
    print(f"\n→ target_transformation_results.csv")


if __name__ == "__main__":
    main()
