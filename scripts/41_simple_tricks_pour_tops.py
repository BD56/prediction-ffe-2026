"""
41 - Trois trucs simples pour qu'un modèle linéaire devienne meilleur sur les tops.

Baseline : ElasticNet linéaire sur top 40 features de v2.

Option A : ElasticNet + feature P(top) (proba issue d'une logistic sur race/pedigree)
Option B : ElasticNet avec sample_weight = y (poids croissant avec la hauteur)
Option C : ElasticNet sur log(y) au lieu de y

Comparaison vs baseline et vs Hurdle.

Sortie : data/master/simple_tricks_pour_tops.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
    mae_tr = df.groupby("tranche", observed=True)["err_abs"].mean().to_dict()
    return mae, rmse, r2, mae_tr


def main():
    print("=== 41 - Trucs simples pour les tops ===\n")

    # ---------- Données ----------
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
    X_train_i = imp_med.fit_transform(X_train)
    X_test_i = imp_med.transform(X_test)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train_i)
    X_test_sc = sc.transform(X_test_i)

    results = []

    # ============================================================
    # BASELINE : ElasticNet pur sur top 40
    # ============================================================
    print("[Baseline] ElasticNet linéaire top 40...")
    t0 = time.time()
    en_base = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                            n_jobs=-1, random_state=42)
    en_base.fit(X_train_sc, y_train)
    pred = en_base.predict(X_test_sc)
    mae, rmse, r2, mae_tr = evaluate_all(y_test, pred)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    results.append({"modele": "Baseline (EN top 40)", "MAE_cm": mae, "RMSE_cm": rmse,
                     "R2": r2, **mae_tr})

    # ============================================================
    # OPTION A : EN + feature P(top)
    # ============================================================
    print("\n[Option A] EN + feature P(top) issue d'une logistic...")
    # On entraîne une logistic sur les 40 features pour prédire P(y≥1,40)
    t0 = time.time()
    y_train_bin = (y_train >= 1.40).astype(int)
    logit = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=3, max_iter=5000,
                                   class_weight="balanced", n_jobs=-1, random_state=42)
    logit.fit(X_train_sc, y_train_bin)
    p_train = logit.predict_proba(X_train_sc)[:, 1]
    p_test = logit.predict_proba(X_test_sc)[:, 1]
    # Ajouter P(top) comme 41e feature
    X_train_A = np.column_stack([X_train_sc, p_train])
    X_test_A = np.column_stack([X_test_sc, p_test])
    en_A = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                         n_jobs=-1, random_state=42)
    en_A.fit(X_train_A, y_train)
    pred_A = en_A.predict(X_test_A)
    mae, rmse, r2, mae_tr = evaluate_all(y_test, pred_A)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    print(f"  Coef sur P(top) : {en_A.coef_[-1]:+.4f}")
    results.append({"modele": "A. EN + P(top)", "MAE_cm": mae, "RMSE_cm": rmse,
                     "R2": r2, **mae_tr})

    # ============================================================
    # OPTION B : EN avec sample_weight = y
    # ============================================================
    print("\n[Option B] EN avec sample_weight = y (poids ∝ hauteur)...")
    t0 = time.time()
    en_B = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                         n_jobs=-1, random_state=42)
    en_B.fit(X_train_sc, y_train, sample_weight=y_train.values)
    pred_B = en_B.predict(X_test_sc)
    mae, rmse, r2, mae_tr = evaluate_all(y_test, pred_B)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    results.append({"modele": "B. EN + sample_w=y", "MAE_cm": mae, "RMSE_cm": rmse,
                     "R2": r2, **mae_tr})

    # Variante B' : sample_weight = y² (plus agressif)
    print("\n[Option B'] EN avec sample_weight = y² (plus agressif)...")
    t0 = time.time()
    en_B2 = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                          n_jobs=-1, random_state=42)
    en_B2.fit(X_train_sc, y_train, sample_weight=(y_train.values ** 2))
    pred_B2 = en_B2.predict(X_test_sc)
    mae, rmse, r2, mae_tr = evaluate_all(y_test, pred_B2)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    results.append({"modele": "B'. EN + sample_w=y²", "MAE_cm": mae, "RMSE_cm": rmse,
                     "R2": r2, **mae_tr})

    # ============================================================
    # OPTION C : EN sur log(y)
    # ============================================================
    print("\n[Option C] EN sur log(y)...")
    t0 = time.time()
    en_C = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                         n_jobs=-1, random_state=42)
    en_C.fit(X_train_sc, np.log(y_train))
    pred_C = np.exp(en_C.predict(X_test_sc))
    mae, rmse, r2, mae_tr = evaluate_all(y_test, pred_C)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    results.append({"modele": "C. EN sur log(y)", "MAE_cm": mae, "RMSE_cm": rmse,
                     "R2": r2, **mae_tr})

    # ============================================================
    # Référence Hurdle (depuis recap_mae_par_tranche)
    # ============================================================
    recap_glob = pd.read_csv(MASTER_DIR / "recap_global.csv")
    recap_mae = pd.read_csv(MASTER_DIR / "recap_mae_par_tranche.csv")
    hurd_g = recap_glob[recap_glob["modele"] == "Hurdle (mélange)"].iloc[0]
    hurd_t = recap_mae[recap_mae["modele"] == "Hurdle (mélange)"].iloc[0]
    results.append({"modele": "Hurdle (référence)",
                     "MAE_cm": hurd_g["MAE_cm"], "RMSE_cm": hurd_g["RMSE_cm"],
                     "R2": hurd_g["R2"],
                     "≤1.10m": hurd_t["≤1.10m"], "1.15-1.20m": hurd_t["1.15-1.20m"],
                     "1.25-1.30m": hurd_t["1.25-1.30m"], "1.35-1.40m": hurd_t["1.35-1.40m"],
                     "≥1.45m": hurd_t["≥1.45m"]})

    # ============================================================
    # Récap
    # ============================================================
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("RÉSULTATS GLOBAUX")
    print("=" * 80)
    print(f"{'Modèle':<28s} | {'MAE':>7s} | {'RMSE':>7s} | {'R²':>7s}")
    print("-" * 65)
    for _, r in df.iterrows():
        print(f"{r['modele']:<28s} | {r['MAE_cm']:>6.2f}c | {r['RMSE_cm']:>6.2f}c | {r['R2']:>7.4f}")

    print("\n" + "=" * 100)
    print("MAE PAR TRANCHE (cm) — focus sur les tops")
    print("=" * 100)
    print(f"{'Modèle':<28s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 95)
    for _, r in df.iterrows():
        print(f"{r['modele']:<28s} | {r['≤1.10m']:>6.2f}c | {r['1.15-1.20m']:>9.2f}c | "
              f"{r['1.25-1.30m']:>9.2f}c | {r['1.35-1.40m']:>9.2f}c | {r['≥1.45m']:>6.2f}c")

    df.to_csv(MASTER_DIR / "simple_tricks_pour_tops.csv", index=False)
    print(f"\n→ simple_tricks_pour_tops.csv")


if __name__ == "__main__":
    main()
