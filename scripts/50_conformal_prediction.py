"""
50 - Conformal Prediction (split conformal) pour intervalles de prédiction.

Principe :
  1. Entraîner le modèle sur train
  2. Prédire sur valid → calculer résidus absolus |y_valid - ŷ_valid|
  3. Calculer q = quantile(résidus_valid, 1-α) avec ajustement (n+1)/n
  4. Pour un cheval test : IC = [ŷ - q, ŷ + q]

Garantie mathématique (Vovk 2005, Angelopoulos & Bates 2023) :
  P(y_test ∈ IC) ≥ 1 - α sous échangeabilité

Variante : Locally Adaptive Conformal — IC modulé par une estimation
de l'incertitude locale.

Modèles : RF default, Hurdle, Stacking + Calib.

Sortie : data/master/conformal_prediction_results.csv
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

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def conformal_quantile(residuals, alpha):
    """Calcule le quantile conformal avec ajustement (n+1)/n."""
    n = len(residuals)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return np.quantile(residuals, level)


def main():
    print("=== 50 - Conformal Prediction (split conformal) ===\n")

    # ---------- Charger les données ----------
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
    print(f"Train : {len(X_train):,} | Valid (calibration) : {len(X_valid):,} | Test : {len(X_test):,}\n")

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xva = imp.transform(X_valid)
    Xte = imp.transform(X_test)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xva_sc = sc.transform(Xva)
    Xte_sc = sc.transform(Xte)

    # ---------- Fit des modèles ----------
    print("Fit des modèles...")
    t0 = time.time()

    rf = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                 min_samples_split=10, max_features="sqrt",
                                 random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train)
    pred_rf_v = rf.predict(Xva)
    pred_rf_t = rf.predict(Xte)

    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                      n_jobs=-1, random_state=42)
    en.fit(Xtr_sc, y_train)
    pred_en_v = en.predict(Xva_sc)
    pred_en_t = en.predict(Xte_sc)

    cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3, random_seed=42, loss_function="MAE",
                            eval_metric="MAE", early_stopping_rounds=30, verbose=0)
    cb.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    pred_cb_v = cb.predict(X_valid)
    pred_cb_t = cb.predict(X_test)

    # Hurdle
    y_train_bin = (y_train >= 1.40).astype(int)
    clf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                   min_samples_split=10, max_features="sqrt",
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(Xtr, y_train_bin)
    p_v = clf.predict_proba(Xva)[:, 1]
    p_t = clf.predict_proba(Xte)[:, 1]
    mask = (y_train >= 1.40).values
    rf_tops = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                      min_samples_split=5, max_features="sqrt",
                                      random_state=42, n_jobs=-1)
    rf_tops.fit(Xtr[mask], y_train[mask])
    pred_tops_v = rf_tops.predict(Xva)
    pred_tops_t = rf_tops.predict(Xte)
    pred_hurdle_v = p_v * pred_tops_v + (1 - p_v) * pred_rf_v
    pred_hurdle_t = p_t * pred_tops_t + (1 - p_t) * pred_rf_t

    # Stacking + Calib
    X_meta_v = np.column_stack([pred_rf_v, pred_en_v, pred_cb_v])
    X_meta_t = np.column_stack([pred_rf_t, pred_en_t, pred_cb_t])
    meta = LinearRegression()
    meta.fit(X_meta_v, y_valid)
    pred_stack_v_raw = meta.predict(X_meta_v)
    pred_stack_t_raw = meta.predict(X_meta_t)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(pred_stack_v_raw, y_valid)
    pred_stack_v = cal.predict(pred_stack_v_raw)
    pred_stack_t = cal.predict(pred_stack_t_raw)

    print(f"  ✓ {time.time()-t0:.1f}s\n")

    # ---------- Conformal Prediction sur chaque modèle ----------
    yv = y_valid.values
    yt = y_test.values
    results = []

    for label, pred_v, pred_t in [
        ("RF default", pred_rf_v, pred_rf_t),
        ("Hurdle", pred_hurdle_v, pred_hurdle_t),
        ("Stacking + Calib", pred_stack_v, pred_stack_t),
    ]:
        # Résidus absolus sur valid (set de calibration)
        residuals_v = np.abs(yv - pred_v)

        # Quantiles conformaux
        q_95 = conformal_quantile(residuals_v, alpha=0.05)
        q_80 = conformal_quantile(residuals_v, alpha=0.20)

        # IC sur test
        lo_95 = pred_t - q_95
        hi_95 = pred_t + q_95
        lo_80 = pred_t - q_80
        hi_80 = pred_t + q_80

        # Couverture empirique
        cov_95 = ((yt >= lo_95) & (yt <= hi_95)).mean() * 100
        cov_80 = ((yt >= lo_80) & (yt <= hi_80)).mean() * 100

        # Largeur moyenne (cm)
        w_95 = (hi_95 - lo_95).mean() * 100
        w_80 = (hi_80 - lo_80).mean() * 100

        results.append({"modele": label, "q_95_cm": q_95*100, "q_80_cm": q_80*100,
                         "cov_95": cov_95, "width_95_cm": w_95,
                         "cov_80": cov_80, "width_80_cm": w_80,
                         "pred_t": pred_t, "lo_95": lo_95, "hi_95": hi_95,
                         "lo_80": lo_80, "hi_80": hi_80})

    # ---------- Affichage global ----------
    print("=" * 90)
    print("COUVERTURE EMPIRIQUE GLOBALE (test 2013)")
    print("=" * 90)
    print(f"{'Modèle':<22s} | {'IC nominal':>11s} | {'Couverture':>11s} | {'Largeur':>10s} | {'q conformal':>13s}")
    print("-" * 90)
    for r in results:
        print(f"{r['modele']:<22s} | {'95%':>11s} | {r['cov_95']:>9.1f}%  | "
              f"{r['width_95_cm']:>8.2f}c | ±{r['q_95_cm']:>10.2f}c")
        print(f"{'':<22s} | {'80%':>11s} | {r['cov_80']:>9.1f}%  | "
              f"{r['width_80_cm']:>8.2f}c | ±{r['q_80_cm']:>10.2f}c")
        print("-" * 90)

    # ---------- Couverture par tranche ----------
    print("\n" + "=" * 100)
    print("COUVERTURE PAR TRANCHE (test 2013, IC 95%)")
    print("=" * 100)
    df = pd.DataFrame({"y_true": yt})
    df["tranche"] = pd.cut(df["y_true"], bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])
    for r in results:
        df[f"{r['modele']}_cov95"] = (yt >= r["lo_95"]) & (yt <= r["hi_95"])
        df[f"{r['modele']}_width95"] = (r["hi_95"] - r["lo_95"]) * 100

    print(f"{'Tranche':<12s} | {'n':>5s} | "
          f"{'cov RF':>7s} | {'cov Hurdle':>11s} | {'cov Stack':>10s} | "
          f"{'wid RF':>7s} | {'wid Hurdle':>11s} | {'wid Stack':>10s}")
    print("-" * 100)
    by_tr = []
    for tr in df["tranche"].cat.categories:
        sub = df[df["tranche"] == tr]
        if len(sub) == 0:
            continue
        c_rf = sub["RF default_cov95"].mean() * 100
        c_hu = sub["Hurdle_cov95"].mean() * 100
        c_st = sub["Stacking + Calib_cov95"].mean() * 100
        w_rf = sub["RF default_width95"].mean()
        w_hu = sub["Hurdle_width95"].mean()
        w_st = sub["Stacking + Calib_width95"].mean()
        print(f"{str(tr):<12s} | {len(sub):>5d} | "
              f"{c_rf:>6.1f}% | {c_hu:>10.1f}% | {c_st:>9.1f}% | "
              f"{w_rf:>6.2f}c | {w_hu:>10.2f}c | {w_st:>9.2f}c")
        by_tr.append({"tranche": str(tr), "n": len(sub),
                       "cov95_RF": c_rf, "cov95_Hurdle": c_hu, "cov95_Stack": c_st,
                       "width95_RF": w_rf, "width95_Hurdle": w_hu, "width95_Stack": w_st})

    # ---------- Exemples concrets ----------
    print("\n" + "=" * 100)
    print("EXEMPLES — 5 CHEVAUX (IC 95% selon Hurdle)")
    print("=" * 100)
    rng = np.random.default_rng(seed=42)
    sample_idx = rng.choice(len(df), 5, replace=False)
    hu = results[1]  # Hurdle
    for i in sample_idx:
        v = yt[i]
        tr = df.iloc[i]["tranche"]
        ic = (hu["lo_95"][i], hu["hi_95"][i])
        cov = "✓" if (ic[0] <= v <= ic[1]) else "✗"
        print(f"  Cheval idx {i} (vrai = {v:.2f}m, tranche {tr}) :")
        print(f"    Hurdle médiane = {hu['pred_t'][i]:.2f}m | IC95 = [{ic[0]:.2f} ; {ic[1]:.2f}] | couvert {cov}")

    # ---------- Sauvegarde ----------
    df.to_csv(MASTER_DIR / "conformal_prediction_complet.csv")
    pd.DataFrame(by_tr).to_csv(MASTER_DIR / "conformal_prediction_par_tranche.csv", index=False)
    pd.DataFrame([{k: r[k] for k in ["modele", "q_95_cm", "q_80_cm",
                                       "cov_95", "width_95_cm",
                                       "cov_80", "width_80_cm"]}
                   for r in results]).to_csv(
        MASTER_DIR / "conformal_prediction_results.csv", index=False)
    print(f"\n→ conformal_prediction_results.csv, conformal_prediction_par_tranche.csv, "
          f"conformal_prediction_complet.csv")


if __name__ == "__main__":
    main()
