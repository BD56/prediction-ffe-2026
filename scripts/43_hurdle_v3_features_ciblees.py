"""
43 - Hurdle v3 : tester avec top 40 features ciblées tops (issues du régresseur conditionnel).

v3a — Tout (classifier + reg conditionnel + RF default) sur top 40 tops
v3b — Classifier + reg conditionnel sur top 40 tops ; RF default sur 156 features

Comparaison vs :
  - Hurdle v1 (tout sur 156 features) — référence d'origine
  - Hurdle v2 (tout sur top 40 consensus) — version qui dégrade

Sortie : data/master/hurdle_v3_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              precision_score, recall_score, roc_auc_score)

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
    df["err_sq"] = (df["y"] - df["p"]) ** 2 * 10000
    rmse_tr = df.groupby("tranche", observed=True)["err_sq"].mean().apply(np.sqrt).to_dict()
    return mae, rmse, r2, mae_tr, rmse_tr


def train_hurdle(X_clf, X_reg_def, X_test_clf, X_test_reg_def,
                  y_train, label=""):
    """Entraîne un Hurdle complet et renvoie la prédiction test + métriques classifier."""
    y_train_bin = (y_train >= 1.40).astype(int)

    # Classifier
    clf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                   min_samples_leaf=10, min_samples_split=10,
                                   max_features="sqrt", class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    clf.fit(X_clf, y_train_bin)
    p_test = clf.predict_proba(X_test_clf)[:, 1]

    # Régresseur conditionnel (sur les mêmes features que le classifier)
    mask = (y_train >= 1.40).values
    rf_tops = RandomForestRegressor(n_estimators=500, max_depth=15,
                                      min_samples_leaf=5, min_samples_split=5,
                                      max_features="sqrt", random_state=42, n_jobs=-1)
    rf_tops.fit(X_clf[mask], y_train[mask])
    pred_tops = rf_tops.predict(X_test_clf)

    # RF default (sur ses propres features)
    rf_def = RandomForestRegressor(n_estimators=500, max_depth=15,
                                     min_samples_leaf=10, min_samples_split=10,
                                     max_features="sqrt", random_state=42, n_jobs=-1)
    rf_def.fit(X_reg_def, y_train)
    pred_def = rf_def.predict(X_test_reg_def)

    pred_hurdle = p_test * pred_tops + (1 - p_test) * pred_def
    return pred_hurdle, p_test


def main():
    print("=== 43 - Hurdle v3 (features ciblées tops) ===\n")

    # ---------- Chargement données ----------
    v2 = pd.read_parquet(MASTER_DIR / "master_dataset_epure_v2.parquet")
    v2 = v2.set_index("IDCHEVAL")
    feat_cols = [c for c in v2.columns if c.startswith("f")]
    for c in feat_cols:
        if v2[c].dtype == bool:
            v2[c] = v2[c].astype(int)
    X_all = v2[feat_cols].select_dtypes(include=[np.number])
    y = v2["hauteur_max_validee"]
    split = v2["SPLIT"]

    # Features sets
    imp_tops = pd.read_csv(MASTER_DIR / "diagnostic_hurdle_importances.csv")
    top40_tops = imp_tops.sort_values("rang_tops").head(40)["feature"].tolist()
    top40_tops = [f for f in top40_tops if f in X_all.columns]
    print(f"Top 40 ciblées tops : {len(top40_tops)} features dispo dans v2")

    X_train, y_train = X_all[split == "train"], y[split == "train"]
    X_test, y_test = X_all[split == "test"], y[split == "test"]
    X_train_top = X_train[top40_tops]
    X_test_top = X_test[top40_tops]

    # Imputation
    imp_all = SimpleImputer(strategy="median")
    Xtr_all = imp_all.fit_transform(X_train)
    Xte_all = imp_all.transform(X_test)
    imp_top = SimpleImputer(strategy="median")
    Xtr_top = imp_top.fit_transform(X_train_top)
    Xte_top = imp_top.transform(X_test_top)

    # ---------- Diagnostic du classifier sur top 40 tops ----------
    print("\n--- Diagnostic du classifier sur top 40 tops ---")
    y_train_bin = (y_train >= 1.40).astype(int)
    y_test_bin = (y_test >= 1.40).astype(int)
    clf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                   min_samples_leaf=10, min_samples_split=10,
                                   max_features="sqrt", class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    clf.fit(Xtr_top, y_train_bin)
    p_test = clf.predict_proba(Xte_top)[:, 1]
    p_bin = (p_test >= 0.5).astype(int)
    print(f"  AUC ROC   : {roc_auc_score(y_test_bin, p_test):.4f}")
    print(f"  Précision : {precision_score(y_test_bin, p_bin):.4f}")
    print(f"  Rappel    : {recall_score(y_test_bin, p_bin):.4f}")

    # ---------- v3a : tout sur top 40 tops ----------
    print("\n[v3a] Tout sur top 40 ciblées tops...")
    t0 = time.time()
    pred_v3a, _ = train_hurdle(Xtr_top, Xtr_top, Xte_top, Xte_top, y_train)
    mae_a, rmse_a, r2_a, mtr_a, rtr_a = evaluate_all(y_test, pred_v3a)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae_a:.2f}cm | RMSE={rmse_a:.2f}cm | R²={r2_a:.4f}")

    # ---------- v3b : classifier + cond sur top 40 tops, RF default sur 156 ----------
    print("\n[v3b] Classifier + cond sur top 40 tops, RF default sur 156 features...")
    t0 = time.time()
    pred_v3b, _ = train_hurdle(Xtr_top, Xtr_all, Xte_top, Xte_all, y_train)
    mae_b, rmse_b, r2_b, mtr_b, rtr_b = evaluate_all(y_test, pred_v3b)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae_b:.2f}cm | RMSE={rmse_b:.2f}cm | R²={r2_b:.4f}")

    # ---------- Comparaison vs Hurdle v1 et v2 ----------
    recap = pd.read_csv(MASTER_DIR / "recap_avec_poly40_global.csv")
    mae_tr_all = pd.read_csv(MASTER_DIR / "recap_avec_poly40_mae.csv")
    rmse_tr_all = pd.read_csv(MASTER_DIR / "recap_avec_poly40_rmse.csv")
    h_v1 = recap[recap["modele"] == "Hurdle (mélange)"].iloc[0]
    mt_v1 = mae_tr_all[mae_tr_all["modele"] == "Hurdle (mélange)"].iloc[0]
    rt_v1 = rmse_tr_all[rmse_tr_all["modele"] == "Hurdle (mélange)"].iloc[0]

    h_v2 = pd.read_csv(MASTER_DIR / "hurdle_v2_top40_results.csv")
    h_v2_row = h_v2[h_v2["modele"] == "Hurdle v2 (top 40)"].iloc[0]

    rows = [
        ("Hurdle v1 (tout sur 156)", h_v1["MAE_cm"], h_v1["RMSE_cm"], h_v1["R2"],
         {k: mt_v1[k] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]},
         {k: rt_v1[k] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]}),
        ("Hurdle v2 (tout sur 40 consensus)",
         h_v2_row["MAE_cm"], h_v2_row["RMSE_cm"], h_v2_row["R2"],
         {k: h_v2_row[k] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]},
         {k: h_v2_row[f"RMSE_{k}"] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]}),
        ("Hurdle v3a (tout sur 40 tops)", mae_a, rmse_a, r2_a, mtr_a, rtr_a),
        ("Hurdle v3b (40 tops + 156 def)", mae_b, rmse_b, r2_b, mtr_b, rtr_b),
    ]

    print("\n" + "=" * 80)
    print("RÉSULTATS GLOBAUX")
    print("=" * 80)
    print(f"{'Modèle':<36s} | {'MAE':>7s} | {'RMSE':>7s} | {'R²':>7s}")
    print("-" * 72)
    for label, mae_, rmse_, r2_, _, _ in rows:
        print(f"{label:<36s} | {mae_:>6.2f}c | {rmse_:>6.2f}c | {r2_:>7.4f}")

    print("\n" + "=" * 110)
    print("MAE PAR TRANCHE (cm)")
    print("=" * 110)
    print(f"{'Modèle':<36s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 100)
    for label, _, _, _, mt, _ in rows:
        print(f"{label:<36s} | {mt['≤1.10m']:>6.2f}c | {mt['1.15-1.20m']:>9.2f}c | "
              f"{mt['1.25-1.30m']:>9.2f}c | {mt['1.35-1.40m']:>9.2f}c | {mt['≥1.45m']:>6.2f}c")

    print("\n" + "=" * 110)
    print("RMSE PAR TRANCHE (cm)")
    print("=" * 110)
    print(f"{'Modèle':<36s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 100)
    for label, _, _, _, _, rt in rows:
        print(f"{label:<36s} | {rt['≤1.10m']:>6.2f}c | {rt['1.15-1.20m']:>9.2f}c | "
              f"{rt['1.25-1.30m']:>9.2f}c | {rt['1.35-1.40m']:>9.2f}c | {rt['≥1.45m']:>6.2f}c")

    out = pd.DataFrame([{
        "modele": label, "MAE_cm": mae_, "RMSE_cm": rmse_, "R2": r2_,
        **mt, **{f"RMSE_{k}": v for k, v in rt.items()}
    } for label, mae_, rmse_, r2_, mt, rt in rows])
    out.to_csv(MASTER_DIR / "hurdle_v3_results.csv", index=False)
    print("\n→ hurdle_v3_results.csv")


if __name__ == "__main__":
    main()
