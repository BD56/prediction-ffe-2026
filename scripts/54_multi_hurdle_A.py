"""
54 - Multi-Hurdle (3 catégories) variante A : seuils 1,10m et 1,40m.

Architecture :
  - Classifier multi-classe : prédit P(flop), P(middle), P(top)
  - 3 régresseurs RF spécialisés sur chaque catégorie
  - Prédiction finale : ŷ = P(flop)×pred_flop + P(mid)×pred_mid + P(top)×pred_top

Comparaisons :
  - RF default (référence baseline)
  - Hurdle 2 classes (référence Hurdle d'origine)
  - Stacking + Calib (référence performance globale)
  - Multi-Hurdle A (nouveau)

Sortie : data/master/multi_hurdle_A_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              precision_score, recall_score, roc_auc_score,
                              classification_report)

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


def assign_class(y, low=1.10, high=1.40):
    """0 = flop (≤low), 1 = middle (]low, high[), 2 = top (≥high)."""
    cls = np.where(y <= low, 0, np.where(y >= high, 2, 1))
    return cls


def main():
    print("=== 54 - Multi-Hurdle variante A (seuils 1,10m / 1,40m) ===\n")

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
    X_valid, y_valid = X[split == "valid"], y[split == "valid"]
    X_test, y_test = X[split == "test"], y[split == "test"]

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xva = imp.transform(X_valid)
    Xte = imp.transform(X_test)

    # Distribution des classes
    LOW, HIGH = 1.10, 1.40
    cls_train = assign_class(y_train.values, LOW, HIGH)
    cls_test = assign_class(y_test.values, LOW, HIGH)
    print(f"Distribution classes train :")
    print(f"  flop   (≤{LOW}m) : {(cls_train==0).sum():>5d} ({(cls_train==0).mean()*100:.1f}%)")
    print(f"  middle ({LOW}-{HIGH}m) : {(cls_train==1).sum():>5d} ({(cls_train==1).mean()*100:.1f}%)")
    print(f"  top    (≥{HIGH}m) : {(cls_train==2).sum():>5d} ({(cls_train==2).mean()*100:.1f}%)")

    # ============================================================
    # MULTI-HURDLE A
    # ============================================================
    print("\n[1/3] Classifier multi-classe (RF balanced)...")
    t0 = time.time()
    mclf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                    min_samples_split=10, max_features="sqrt",
                                    class_weight="balanced", random_state=42, n_jobs=-1)
    mclf.fit(Xtr, cls_train)
    proba_test = mclf.predict_proba(Xte)  # shape (n_test, 3)
    print(f"  ✓ {time.time()-t0:.1f}s")
    print(f"  P moyennes test : P(flop)={proba_test[:,0].mean():.3f}, "
          f"P(mid)={proba_test[:,1].mean():.3f}, P(top)={proba_test[:,2].mean():.3f}")
    pred_class = mclf.predict(Xte)
    print(f"\n  Rapport classification (seuil argmax) :")
    for c, label in [(0, "flop"), (1, "middle"), (2, "top")]:
        prec = precision_score(cls_test == c, pred_class == c, zero_division=0)
        rec = recall_score(cls_test == c, pred_class == c, zero_division=0)
        print(f"    {label:<7s} | Précision {prec:.3f} | Rappel {rec:.3f} | "
              f"AUC {roc_auc_score(cls_test == c, proba_test[:, c]):.4f}")

    # ---------- 3 régresseurs spécialisés ----------
    print("\n[2/3] 3 régresseurs RF spécialisés...")
    mask_flop = (cls_train == 0)
    mask_mid = (cls_train == 1)
    mask_top = (cls_train == 2)
    print(f"  Train flop   : {mask_flop.sum():,} (moyenne y = {y_train.values[mask_flop].mean():.3f}m)")
    print(f"  Train middle : {mask_mid.sum():,} (moyenne y = {y_train.values[mask_mid].mean():.3f}m)")
    print(f"  Train top    : {mask_top.sum():,} (moyenne y = {y_train.values[mask_top].mean():.3f}m)")

    t0 = time.time()
    rf_flop = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                      min_samples_split=5, max_features="sqrt",
                                      random_state=42, n_jobs=-1)
    rf_flop.fit(Xtr[mask_flop], y_train.values[mask_flop])
    pred_flop = rf_flop.predict(Xte)
    print(f"  RF flop fit : {time.time()-t0:.1f}s | pred test moy = {pred_flop.mean():.3f}m")

    t0 = time.time()
    rf_mid = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                     min_samples_split=10, max_features="sqrt",
                                     random_state=42, n_jobs=-1)
    rf_mid.fit(Xtr[mask_mid], y_train.values[mask_mid])
    pred_mid = rf_mid.predict(Xte)
    print(f"  RF middle fit : {time.time()-t0:.1f}s | pred test moy = {pred_mid.mean():.3f}m")

    t0 = time.time()
    rf_top = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                     min_samples_split=5, max_features="sqrt",
                                     random_state=42, n_jobs=-1)
    rf_top.fit(Xtr[mask_top], y_train.values[mask_top])
    pred_top = rf_top.predict(Xte)
    print(f"  RF top fit : {time.time()-t0:.1f}s | pred test moy = {pred_top.mean():.3f}m")

    # ---------- Combinaison Multi-Hurdle ----------
    pred_multi = (proba_test[:, 0] * pred_flop +
                   proba_test[:, 1] * pred_mid +
                   proba_test[:, 2] * pred_top)
    mae_m, rmse_m, r2_m, mtr_m, rtr_m = evaluate_all(y_test, pred_multi)

    # ============================================================
    # Références : Hurdle 2 classes + RF default
    # ============================================================
    print("\n[3/3] Références : Hurdle 2 classes + RF default + Stacking...")
    # RF default
    rf_def = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                     min_samples_split=10, max_features="sqrt",
                                     random_state=42, n_jobs=-1)
    rf_def.fit(Xtr, y_train)
    pred_rf = rf_def.predict(Xte)
    mae_r, rmse_r, r2_r, mtr_r, rtr_r = evaluate_all(y_test, pred_rf)

    # Hurdle 2 classes
    y_train_bin = (y_train >= 1.40).astype(int)
    clf2 = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                    min_samples_split=10, max_features="sqrt",
                                    class_weight="balanced", random_state=42, n_jobs=-1)
    clf2.fit(Xtr, y_train_bin)
    p_test2 = clf2.predict_proba(Xte)[:, 1]
    mask_top2 = (y_train >= 1.40).values
    rf_tops2 = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                       min_samples_split=5, max_features="sqrt",
                                       random_state=42, n_jobs=-1)
    rf_tops2.fit(Xtr[mask_top2], y_train.values[mask_top2])
    pred_tops2 = rf_tops2.predict(Xte)
    pred_hurdle2 = p_test2 * pred_tops2 + (1 - p_test2) * pred_rf
    mae_h, rmse_h, r2_h, mtr_h, rtr_h = evaluate_all(y_test, pred_hurdle2)

    # Stacking + Calib (depuis recap)
    recap = pd.read_csv(MASTER_DIR / "recap_avec_poly40_global.csv")
    mt_all = pd.read_csv(MASTER_DIR / "recap_avec_poly40_mae.csv")
    rt_all = pd.read_csv(MASTER_DIR / "recap_avec_poly40_rmse.csv")
    stk = recap[recap["modele"] == "Stacking + Calib"].iloc[0]
    mt_stk = mt_all[mt_all["modele"] == "Stacking + Calib"].iloc[0]
    rt_stk = rt_all[rt_all["modele"] == "Stacking + Calib"].iloc[0]

    # ============================================================
    # Synthèse
    # ============================================================
    rows = [
        ("RF default", mae_r, rmse_r, r2_r, mtr_r, rtr_r),
        ("Hurdle 2 classes (référence)", mae_h, rmse_h, r2_h, mtr_h, rtr_h),
        ("Multi-Hurdle A (1,10/1,40)", mae_m, rmse_m, r2_m, mtr_m, rtr_m),
        ("Stacking + Calib (réf perf.)", stk["MAE_cm"], stk["RMSE_cm"], stk["R2"],
         {k: mt_stk[k] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]},
         {k: rt_stk[k] for k in ["≤1.10m","1.15-1.20m","1.25-1.30m","1.35-1.40m","≥1.45m"]}),
    ]

    print("\n" + "=" * 80)
    print("RÉSULTATS GLOBAUX")
    print("=" * 80)
    print(f"{'Modèle':<32s} | {'MAE':>7s} | {'RMSE':>7s} | {'R²':>7s}")
    print("-" * 65)
    for label, mae_, rmse_, r2_, _, _ in rows:
        print(f"{label:<32s} | {mae_:>6.2f}c | {rmse_:>6.2f}c | {r2_:>7.4f}")

    print("\n" + "=" * 105)
    print("MAE PAR TRANCHE (cm)")
    print("=" * 105)
    print(f"{'Modèle':<32s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 100)
    for label, _, _, _, mt, _ in rows:
        print(f"{label:<32s} | {mt['≤1.10m']:>6.2f}c | {mt['1.15-1.20m']:>9.2f}c | "
              f"{mt['1.25-1.30m']:>9.2f}c | {mt['1.35-1.40m']:>9.2f}c | {mt['≥1.45m']:>6.2f}c")

    print("\n" + "=" * 105)
    print("RMSE PAR TRANCHE (cm)")
    print("=" * 105)
    print(f"{'Modèle':<32s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 100)
    for label, _, _, _, _, rt in rows:
        print(f"{label:<32s} | {rt['≤1.10m']:>6.2f}c | {rt['1.15-1.20m']:>9.2f}c | "
              f"{rt['1.25-1.30m']:>9.2f}c | {rt['1.35-1.40m']:>9.2f}c | {rt['≥1.45m']:>6.2f}c")

    # Sauvegarde
    out = pd.DataFrame([{
        "modele": label, "MAE_cm": mae_, "RMSE_cm": rmse_, "R2": r2_,
        **mt, **{f"RMSE_{k}": v for k, v in rt.items()}
    } for label, mae_, rmse_, r2_, mt, rt in rows])
    out.to_csv(MASTER_DIR / "multi_hurdle_A_results.csv", index=False)
    print(f"\n→ multi_hurdle_A_results.csv")


if __name__ == "__main__":
    main()
