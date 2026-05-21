"""
55 - Multi-Hurdle hiérarchique top-first (variante A, seuils 1,10/1,40).

Architecture (Stratégie 2) :
  Étape 1 : clf_top entraîné sur TOUT le train (binaire balanced : top vs reste)
            → P(top)
  Étape 2 : clf_flop_cond entraîné UNIQUEMENT sur les chevaux non-top
            → P(flop | non-top)

Probabilités finales (cohérentes par construction, somme = 1) :
  P(top) = clf_top(X)
  P(flop) = (1 - P(top)) × clf_flop_cond(X)
  P(middle) = (1 - P(top)) × (1 - clf_flop_cond(X))

Trois régresseurs spécialisés (RF) :
  rf_flop sur train[y ≤ 1,10]
  rf_mid sur train[1,10 < y < 1,40]
  rf_top sur train[y ≥ 1,40]

Prédiction finale : ŷ = P(top)×pred_top + P(flop)×pred_flop + P(mid)×pred_mid

Sortie : data/master/multi_hurdle_hierarchique_results.csv
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
    df = pd.DataFrame({"y": y_true.values if hasattr(y_true, 'values') else y_true, "p": y_pred})
    df["tranche"] = pd.cut(df["y"], bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])
    df["err_abs"] = (df["y"] - df["p"]).abs() * 100
    df["err_sq"] = (df["y"] - df["p"]) ** 2 * 10000
    mae_tr = df.groupby("tranche", observed=True)["err_abs"].mean().to_dict()
    rmse_tr = df.groupby("tranche", observed=True)["err_sq"].mean().apply(np.sqrt).to_dict()
    return mae, rmse, r2, mae_tr, rmse_tr


def main():
    print("=== 55 - Multi-Hurdle hiérarchique top-first ===\n")

    # ---------- Chargement ----------
    v2 = pd.read_parquet(MASTER_DIR / "master_dataset_epure_v2.parquet")
    v2 = v2.set_index("IDCHEVAL")
    feat_cols = [c for c in v2.columns if c.startswith("f")]
    for c in feat_cols:
        if v2[c].dtype == bool:
            v2[c] = v2[c].astype(int)
    X = v2[feat_cols].select_dtypes(include=[np.number])
    y = v2["hauteur_max_validee"]
    split = v2["SPLIT"]

    X_train, y_train = X[split == "train"], y[split == "train"]
    X_test, y_test = X[split == "test"], y[split == "test"]

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xte = imp.transform(X_test)
    yt_arr = y_train.values
    LOW, HIGH = 1.10, 1.40

    mask_top = yt_arr >= HIGH
    mask_flop = yt_arr <= LOW
    mask_mid = (yt_arr > LOW) & (yt_arr < HIGH)
    mask_non_top = ~mask_top

    print(f"Distribution train :")
    print(f"  flop   (y ≤ {LOW}m) : {mask_flop.sum():>5d} ({mask_flop.mean()*100:.1f}%)")
    print(f"  middle ({LOW} < y < {HIGH}m) : {mask_mid.sum():>5d} ({mask_mid.mean()*100:.1f}%)")
    print(f"  top    (y ≥ {HIGH}m) : {mask_top.sum():>5d} ({mask_top.mean()*100:.1f}%)")

    # ============================================================
    # Étape 1 — clf_top sur TOUT le train
    # ============================================================
    print("\n[1/5] Classifier 1 : top vs reste (entraîné sur tout le train)...")
    t0 = time.time()
    y_train_bin_top = (mask_top).astype(int)
    y_test_bin_top = (y_test >= HIGH).astype(int)
    clf_top = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                       min_samples_split=10, max_features="sqrt",
                                       class_weight="balanced", random_state=42, n_jobs=-1)
    clf_top.fit(Xtr, y_train_bin_top)
    p_top_test = clf_top.predict_proba(Xte)[:, 1]
    print(f"  ✓ {time.time()-t0:.1f}s")
    print(f"  AUC top : {roc_auc_score(y_test_bin_top, p_top_test):.4f}")
    print(f"  Précision/Rappel top (seuil 0.5) : "
          f"{precision_score(y_test_bin_top, p_top_test >= 0.5):.3f} / "
          f"{recall_score(y_test_bin_top, p_top_test >= 0.5):.3f}")

    # ============================================================
    # Étape 2 — clf_flop_cond sur les chevaux NON-TOP du train
    # ============================================================
    print("\n[2/5] Classifier 2 : flop | non-top (entraîné sur les non-tops uniquement)...")
    t0 = time.time()
    Xtr_nt = Xtr[mask_non_top]
    ytr_nt = yt_arr[mask_non_top]
    y_nt_flop = (ytr_nt <= LOW).astype(int)
    clf_flop_cond = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                              min_samples_split=10, max_features="sqrt",
                                              class_weight="balanced", random_state=42, n_jobs=-1)
    clf_flop_cond.fit(Xtr_nt, y_nt_flop)
    p_flop_given_nt = clf_flop_cond.predict_proba(Xte)[:, 1]
    print(f"  ✓ {time.time()-t0:.1f}s | Train : {len(Xtr_nt):,} non-tops ({y_nt_flop.mean()*100:.1f}% flops)")
    # AUC sur le sous-ensemble test non-top
    y_test_nt_mask = (y_test < HIGH).values
    y_test_flop_in_nt = (y_test[y_test_nt_mask] <= LOW).astype(int)
    p_flop_in_nt = p_flop_given_nt[y_test_nt_mask]
    print(f"  AUC flop | non-top : {roc_auc_score(y_test_flop_in_nt, p_flop_in_nt):.4f}")

    # ============================================================
    # Étape 3 — Calcul des probabilités cohérentes
    # ============================================================
    print("\n[3/5] Calcul probabilités cohérentes...")
    P_top = p_top_test
    P_flop = (1 - P_top) * p_flop_given_nt
    P_mid = (1 - P_top) * (1 - p_flop_given_nt)
    print(f"  Vérification : somme P(top)+P(flop)+P(mid) min={np.min(P_top+P_flop+P_mid):.6f} "
          f"max={np.max(P_top+P_flop+P_mid):.6f}")
    print(f"  P moyennes : P(top)={P_top.mean():.3f}, P(flop)={P_flop.mean():.3f}, "
          f"P(mid)={P_mid.mean():.3f}")

    # ============================================================
    # Étape 4 — 3 régresseurs spécialisés
    # ============================================================
    print("\n[4/5] 3 régresseurs RF spécialisés...")
    t0 = time.time()
    rf_flop = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                      min_samples_split=5, max_features="sqrt",
                                      random_state=42, n_jobs=-1)
    rf_flop.fit(Xtr[mask_flop], yt_arr[mask_flop])
    pred_flop_test = rf_flop.predict(Xte)
    print(f"  RF flop fit : {time.time()-t0:.1f}s | pred moy test = {pred_flop_test.mean():.3f}m")

    t0 = time.time()
    rf_mid = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                     min_samples_split=10, max_features="sqrt",
                                     random_state=42, n_jobs=-1)
    rf_mid.fit(Xtr[mask_mid], yt_arr[mask_mid])
    pred_mid_test = rf_mid.predict(Xte)
    print(f"  RF middle fit : {time.time()-t0:.1f}s | pred moy test = {pred_mid_test.mean():.3f}m")

    t0 = time.time()
    rf_top = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                     min_samples_split=5, max_features="sqrt",
                                     random_state=42, n_jobs=-1)
    rf_top.fit(Xtr[mask_top], yt_arr[mask_top])
    pred_top_test = rf_top.predict(Xte)
    print(f"  RF top fit : {time.time()-t0:.1f}s | pred moy test = {pred_top_test.mean():.3f}m")

    # ============================================================
    # Étape 5 — Combinaison + références
    # ============================================================
    pred_hier = P_top * pred_top_test + P_flop * pred_flop_test + P_mid * pred_mid_test
    mae_h, rmse_h, r2_h, mtr_h, rtr_h = evaluate_all(y_test, pred_hier)

    print("\n[5/5] Références : RF default + Hurdle 2 classes + Multi-Hurdle A...")
    # RF default
    rf_def = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                     min_samples_split=10, max_features="sqrt",
                                     random_state=42, n_jobs=-1)
    rf_def.fit(Xtr, y_train)
    pred_rf_test = rf_def.predict(Xte)
    mae_r, rmse_r, r2_r, mtr_r, rtr_r = evaluate_all(y_test, pred_rf_test)

    # Hurdle 2 classes
    pred_hu_2 = p_top_test * pred_top_test + (1 - p_top_test) * pred_rf_test
    mae_h2, rmse_h2, r2_h2, mtr_h2, rtr_h2 = evaluate_all(y_test, pred_hu_2)

    # Multi-Hurdle A (depuis CSV)
    mh_a = pd.read_csv(MASTER_DIR / "multi_hurdle_A_results.csv")
    mh_a = mh_a[mh_a["modele"] == "Multi-Hurdle A (1,10/1,40)"].iloc[0]
    mh_a_mt = {k: mh_a[k] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]}
    mh_a_rt = {k: mh_a[f"RMSE_{k}"] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]}

    rows = [
        ("RF default", mae_r, rmse_r, r2_r, mtr_r, rtr_r),
        ("Hurdle 2 classes", mae_h2, rmse_h2, r2_h2, mtr_h2, rtr_h2),
        ("Multi-Hurdle A (multi-classe)", mh_a["MAE_cm"], mh_a["RMSE_cm"], mh_a["R2"], mh_a_mt, mh_a_rt),
        ("Multi-Hurdle Hiérarchique", mae_h, rmse_h, r2_h, mtr_h, rtr_h),
    ]

    print("\n" + "=" * 80)
    print("RÉSULTATS GLOBAUX")
    print("=" * 80)
    print(f"{'Modèle':<34s} | {'MAE':>7s} | {'RMSE':>7s} | {'R²':>7s}")
    print("-" * 65)
    for label, mae_, rmse_, r2_, _, _ in rows:
        print(f"{label:<34s} | {mae_:>6.2f}c | {rmse_:>6.2f}c | {r2_:>7.4f}")

    print("\n" + "=" * 105)
    print("MAE PAR TRANCHE (cm)")
    print("=" * 105)
    print(f"{'Modèle':<34s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 100)
    for label, _, _, _, mt, _ in rows:
        print(f"{label:<34s} | {mt['≤1.10m']:>6.2f}c | {mt['1.15-1.20m']:>9.2f}c | "
              f"{mt['1.25-1.30m']:>9.2f}c | {mt['1.35-1.40m']:>9.2f}c | {mt['≥1.45m']:>6.2f}c")

    print("\n" + "=" * 105)
    print("RMSE PAR TRANCHE (cm)")
    print("=" * 105)
    print(f"{'Modèle':<34s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 100)
    for label, _, _, _, _, rt in rows:
        print(f"{label:<34s} | {rt['≤1.10m']:>6.2f}c | {rt['1.15-1.20m']:>9.2f}c | "
              f"{rt['1.25-1.30m']:>9.2f}c | {rt['1.35-1.40m']:>9.2f}c | {rt['≥1.45m']:>6.2f}c")

    # Sauvegarde
    out = pd.DataFrame([{
        "modele": label, "MAE_cm": mae_, "RMSE_cm": rmse_, "R2": r2_,
        **mt, **{f"RMSE_{k}": v for k, v in rt.items()}
    } for label, mae_, rmse_, r2_, mt, rt in rows])
    out.to_csv(MASTER_DIR / "multi_hurdle_hierarchique_results.csv", index=False)
    print(f"\n→ multi_hurdle_hierarchique_results.csv")


if __name__ == "__main__":
    main()
