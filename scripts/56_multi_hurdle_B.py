"""
56 - Multi-Hurdle variante B (seuils 1,20m / 1,40m).

Deux versions testées en parallèle :
  - B-multi : classifier multi-classe (comme script 54)
  - B-hier  : hiérarchique top-first (comme script 55)

Comparaison aux 4 modèles déjà testés.

Sortie : data/master/multi_hurdle_B_results.csv
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


def assign_class(y, low, high):
    return np.where(y <= low, 0, np.where(y >= high, 2, 1))


def main():
    print("=== 56 - Multi-Hurdle variante B (seuils 1,20m / 1,40m) ===\n")

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

    LOW, HIGH = 1.20, 1.40

    # Distribution
    mask_top = yt_arr >= HIGH
    mask_flop = yt_arr <= LOW
    mask_mid = (yt_arr > LOW) & (yt_arr < HIGH)
    print(f"Distribution train (seuils {LOW}m / {HIGH}m) :")
    print(f"  flop   (y ≤ {LOW}m) : {mask_flop.sum():>5d} ({mask_flop.mean()*100:.1f}%)")
    print(f"  middle ({LOW} < y < {HIGH}m) : {mask_mid.sum():>5d} ({mask_mid.mean()*100:.1f}%)")
    print(f"  top    (y ≥ {HIGH}m) : {mask_top.sum():>5d} ({mask_top.mean()*100:.1f}%)")

    # ============================================================
    # 3 régresseurs spécialisés (communs aux deux versions)
    # ============================================================
    print("\n[1/4] 3 régresseurs RF spécialisés...")
    t0 = time.time()
    rf_flop = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                      min_samples_split=10, max_features="sqrt",
                                      random_state=42, n_jobs=-1)
    rf_flop.fit(Xtr[mask_flop], yt_arr[mask_flop])
    pred_flop_test = rf_flop.predict(Xte)
    print(f"  RF flop fit : pred moy = {pred_flop_test.mean():.3f}m")

    rf_mid = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                     min_samples_split=10, max_features="sqrt",
                                     random_state=42, n_jobs=-1)
    rf_mid.fit(Xtr[mask_mid], yt_arr[mask_mid])
    pred_mid_test = rf_mid.predict(Xte)
    print(f"  RF mid fit : pred moy = {pred_mid_test.mean():.3f}m")

    rf_top = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                     min_samples_split=5, max_features="sqrt",
                                     random_state=42, n_jobs=-1)
    rf_top.fit(Xtr[mask_top], yt_arr[mask_top])
    pred_top_test = rf_top.predict(Xte)
    print(f"  RF top fit : pred moy = {pred_top_test.mean():.3f}m | total {time.time()-t0:.1f}s")

    # ============================================================
    # Version B-multi : classifier multi-classe
    # ============================================================
    print("\n[2/4] B-multi : classifier multi-classe...")
    t0 = time.time()
    cls_train = assign_class(yt_arr, LOW, HIGH)
    mclf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                    min_samples_split=10, max_features="sqrt",
                                    class_weight="balanced", random_state=42, n_jobs=-1)
    mclf.fit(Xtr, cls_train)
    proba_multi = mclf.predict_proba(Xte)
    pred_B_multi = (proba_multi[:, 0] * pred_flop_test +
                     proba_multi[:, 1] * pred_mid_test +
                     proba_multi[:, 2] * pred_top_test)
    mae_bm, rmse_bm, r2_bm, mtr_bm, rtr_bm = evaluate_all(y_test, pred_B_multi)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae_bm:.2f}c | R²={r2_bm:.4f}")
    # Rapport classification
    cls_test = assign_class(y_test.values, LOW, HIGH)
    pred_cls = mclf.predict(Xte)
    for c, label in [(0, "flop"), (1, "middle"), (2, "top")]:
        prec = precision_score(cls_test == c, pred_cls == c, zero_division=0)
        rec = recall_score(cls_test == c, pred_cls == c, zero_division=0)
        auc = roc_auc_score(cls_test == c, proba_multi[:, c])
        print(f"    {label:<7s} Précision {prec:.3f} | Rappel {rec:.3f} | AUC {auc:.4f}")

    # ============================================================
    # Version B-hier : hiérarchique top-first
    # ============================================================
    print("\n[3/4] B-hier : hiérarchique top-first...")
    t0 = time.time()
    y_train_bin_top = mask_top.astype(int)
    clf_top = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                       min_samples_split=10, max_features="sqrt",
                                       class_weight="balanced", random_state=42, n_jobs=-1)
    clf_top.fit(Xtr, y_train_bin_top)
    p_top_test = clf_top.predict_proba(Xte)[:, 1]
    print(f"  AUC clf_top : {roc_auc_score((y_test >= HIGH).astype(int), p_top_test):.4f}")

    # Clf_flop_cond sur non-tops
    mask_non_top = ~mask_top
    Xtr_nt = Xtr[mask_non_top]
    y_nt = yt_arr[mask_non_top]
    y_nt_flop = (y_nt <= LOW).astype(int)
    print(f"  Train non-tops : {len(Xtr_nt):,} chevaux, {y_nt_flop.mean()*100:.1f}% flops")
    clf_flop_cond = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                              min_samples_split=10, max_features="sqrt",
                                              class_weight="balanced", random_state=42, n_jobs=-1)
    clf_flop_cond.fit(Xtr_nt, y_nt_flop)
    p_flop_given_nt = clf_flop_cond.predict_proba(Xte)[:, 1]

    P_top = p_top_test
    P_flop = (1 - P_top) * p_flop_given_nt
    P_mid = (1 - P_top) * (1 - p_flop_given_nt)
    pred_B_hier = P_top * pred_top_test + P_flop * pred_flop_test + P_mid * pred_mid_test
    mae_bh, rmse_bh, r2_bh, mtr_bh, rtr_bh = evaluate_all(y_test, pred_B_hier)
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={mae_bh:.2f}c | R²={r2_bh:.4f}")

    # ============================================================
    # Récupérer les références
    # ============================================================
    print("\n[4/4] Récupération des références déjà calculées...")
    mh_h = pd.read_csv(MASTER_DIR / "multi_hurdle_hierarchique_results.csv")
    refs = {}
    for label in ["RF default", "Hurdle 2 classes",
                   "Multi-Hurdle A (multi-classe)", "Multi-Hurdle Hiérarchique"]:
        r = mh_h[mh_h["modele"] == label].iloc[0]
        refs[label] = {
            "MAE": r["MAE_cm"], "RMSE": r["RMSE_cm"], "R2": r["R2"],
            "mae_tr": {k: r[k] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]},
            "rmse_tr": {k: r[f"RMSE_{k}"] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]}
        }

    rows = [
        ("RF default", refs["RF default"]["MAE"], refs["RF default"]["RMSE"], refs["RF default"]["R2"],
         refs["RF default"]["mae_tr"], refs["RF default"]["rmse_tr"]),
        ("Hurdle 2 classes", refs["Hurdle 2 classes"]["MAE"], refs["Hurdle 2 classes"]["RMSE"],
         refs["Hurdle 2 classes"]["R2"], refs["Hurdle 2 classes"]["mae_tr"], refs["Hurdle 2 classes"]["rmse_tr"]),
        ("Multi-Hurdle A (1,10/1,40) multi-classe",
         refs["Multi-Hurdle A (multi-classe)"]["MAE"], refs["Multi-Hurdle A (multi-classe)"]["RMSE"],
         refs["Multi-Hurdle A (multi-classe)"]["R2"], refs["Multi-Hurdle A (multi-classe)"]["mae_tr"],
         refs["Multi-Hurdle A (multi-classe)"]["rmse_tr"]),
        ("Multi-Hurdle A hiérarchique",
         refs["Multi-Hurdle Hiérarchique"]["MAE"], refs["Multi-Hurdle Hiérarchique"]["RMSE"],
         refs["Multi-Hurdle Hiérarchique"]["R2"], refs["Multi-Hurdle Hiérarchique"]["mae_tr"],
         refs["Multi-Hurdle Hiérarchique"]["rmse_tr"]),
        ("Multi-Hurdle B (1,20/1,40) multi-classe", mae_bm, rmse_bm, r2_bm, mtr_bm, rtr_bm),
        ("Multi-Hurdle B hiérarchique", mae_bh, rmse_bh, r2_bh, mtr_bh, rtr_bh),
    ]

    # Affichage
    print("\n" + "=" * 80)
    print("RÉSULTATS GLOBAUX")
    print("=" * 80)
    print(f"{'Modèle':<42s} | {'MAE':>7s} | {'RMSE':>7s} | {'R²':>7s}")
    print("-" * 75)
    for label, mae_, rmse_, r2_, _, _ in rows:
        print(f"{label:<42s} | {mae_:>6.2f}c | {rmse_:>6.2f}c | {r2_:>7.4f}")

    print("\n" + "=" * 110)
    print("MAE PAR TRANCHE (cm)")
    print("=" * 110)
    print(f"{'Modèle':<42s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 105)
    for label, _, _, _, mt, _ in rows:
        print(f"{label:<42s} | {mt['≤1.10m']:>6.2f}c | {mt['1.15-1.20m']:>9.2f}c | "
              f"{mt['1.25-1.30m']:>9.2f}c | {mt['1.35-1.40m']:>9.2f}c | {mt['≥1.45m']:>6.2f}c")

    print("\n" + "=" * 110)
    print("RMSE PAR TRANCHE (cm)")
    print("=" * 110)
    print(f"{'Modèle':<42s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 105)
    for label, _, _, _, _, rt in rows:
        print(f"{label:<42s} | {rt['≤1.10m']:>6.2f}c | {rt['1.15-1.20m']:>9.2f}c | "
              f"{rt['1.25-1.30m']:>9.2f}c | {rt['1.35-1.40m']:>9.2f}c | {rt['≥1.45m']:>6.2f}c")

    out = pd.DataFrame([{
        "modele": label, "MAE_cm": mae_, "RMSE_cm": rmse_, "R2": r2_,
        **mt, **{f"RMSE_{k}": v for k, v in rt.items()}
    } for label, mae_, rmse_, r2_, mt, rt in rows])
    out.to_csv(MASTER_DIR / "multi_hurdle_B_results.csv", index=False)
    print(f"\n→ multi_hurdle_B_results.csv")


if __name__ == "__main__":
    main()
