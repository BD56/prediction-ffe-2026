"""
51 - Locally Adaptive Conformal Prediction.

Principe (Lei et al. 2018, Romano et al. 2019) :
  1. Entraîner modèle de prédiction f(x) sur train
  2. Entraîner modèle d'incertitude σ(x) sur train, qui prédit |résidu|
  3. Sur le calibration set (valid) : calculer résidus normalisés r_i = |y - ŷ| / σ(x)
  4. q_norm = quantile(r, 1-α)
  5. Pour test : PI(x) = [ŷ - q_norm * σ(x), ŷ + q_norm * σ(x)]

L'IC s'adapte au cheval : étroit pour les cas "faciles", large pour les cas "ambigus".

Deux variantes testées pour σ(x) :
  (V1) σ = std des prédictions des 500 arbres du RF (gratuit, exploite la structure du RF)
  (V2) σ = prédiction d'un RF entraîné sur (X, |résidus_train|)

Comparaison : conformal standard vs adaptatif.

Sortie : data/master/locally_adaptive_conformal_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def conformal_quantile(values, alpha):
    n = len(values)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return np.quantile(values, level)


def main():
    print("=== 51 - Locally Adaptive Conformal Prediction ===\n")

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
    print(f"Train : {len(X_train):,} | Valid (calibration) : {len(X_valid):,} | Test : {len(X_test):,}\n")

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xva = imp.transform(X_valid)
    Xte = imp.transform(X_test)
    yv = y_valid.values
    yt = y_test.values

    # ============================================================
    # ÉTAPE 1 — Fit du modèle principal (RF default puis Hurdle)
    # ============================================================
    print("[1/4] Fit RF default + Hurdle...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                 min_samples_split=10, max_features="sqrt",
                                 random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train)
    pred_rf_v = rf.predict(Xva)
    pred_rf_t = rf.predict(Xte)
    # Pour RF on récupère aussi les prédictions individuelles des arbres
    preds_trees_v = np.stack([est.predict(Xva) for est in rf.estimators_])
    preds_trees_t = np.stack([est.predict(Xte) for est in rf.estimators_])
    sigma_rf_v = preds_trees_v.std(axis=0)  # std entre les 500 arbres
    sigma_rf_t = preds_trees_t.std(axis=0)

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
    pred_hu_v = p_v * pred_tops_v + (1 - p_v) * pred_rf_v
    pred_hu_t = p_t * pred_tops_t + (1 - p_t) * pred_rf_t
    print(f"  ✓ {time.time()-t0:.1f}s")

    # ============================================================
    # ÉTAPE 2 — Estimer σ(x)
    # ============================================================
    print("\n[2/4] Estimation de σ(x) — deux variantes...")
    # V1 : std entre arbres du RF (gratuit, déjà calculé pour RF)
    # Pour Hurdle, on prend la std combinée des deux RF (default + tops)
    preds_tops_trees_v = np.stack([est.predict(Xva) for est in rf_tops.estimators_])
    preds_tops_trees_t = np.stack([est.predict(Xte) for est in rf_tops.estimators_])
    sigma_tops_v = preds_tops_trees_v.std(axis=0)
    sigma_tops_t = preds_tops_trees_t.std(axis=0)
    sigma_hu_v_v1 = p_v * sigma_tops_v + (1 - p_v) * sigma_rf_v
    sigma_hu_t_v1 = p_t * sigma_tops_t + (1 - p_t) * sigma_rf_t

    # V2 : modèle séparé entraîné sur (X_train, |résidus_train|)
    pred_rf_tr = rf.predict(Xtr)
    res_train_rf = np.abs(y_train.values - pred_rf_tr)
    rf_sigma = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=20,
                                       max_features="sqrt", random_state=42, n_jobs=-1)
    rf_sigma.fit(Xtr, res_train_rf)
    sigma_rf_v_v2 = rf_sigma.predict(Xva) + 1e-3
    sigma_rf_t_v2 = rf_sigma.predict(Xte) + 1e-3
    print(f"  V1 (std arbres) : sigma RF v ∈ [{sigma_rf_v.min():.4f}, {sigma_rf_v.max():.4f}], "
          f"moy {sigma_rf_v.mean():.4f}")
    print(f"  V2 (modèle séparé) : sigma RF v ∈ [{sigma_rf_v_v2.min():.4f}, {sigma_rf_v_v2.max():.4f}], "
          f"moy {sigma_rf_v_v2.mean():.4f}")

    # ============================================================
    # ÉTAPE 3 — Conformal Adaptatif
    # ============================================================
    print("\n[3/4] Calcul des quantiles conformaux adaptatifs...")

    def adaptive_ci(pred_v, sigma_v, yv, pred_t, sigma_t, alpha):
        """Calcule IC adaptatif locally weighted."""
        # Résidus normalisés sur valid
        residuals_norm = np.abs(yv - pred_v) / sigma_v
        q_norm = conformal_quantile(residuals_norm, alpha)
        # IC sur test
        lo = pred_t - q_norm * sigma_t
        hi = pred_t + q_norm * sigma_t
        return lo, hi, q_norm

    def standard_ci(pred_v, yv, pred_t, alpha):
        """Conformal standard (référence)."""
        residuals = np.abs(yv - pred_v)
        q = conformal_quantile(residuals, alpha)
        return pred_t - q, pred_t + q, q

    results = []

    # === RF default ===
    print("\n--- RF default ---")
    lo_std, hi_std, q_std = standard_ci(pred_rf_v, yv, pred_rf_t, 0.05)
    lo_v1, hi_v1, qn_v1 = adaptive_ci(pred_rf_v, sigma_rf_v, yv,
                                         pred_rf_t, sigma_rf_t, 0.05)
    lo_v2, hi_v2, qn_v2 = adaptive_ci(pred_rf_v, sigma_rf_v_v2, yv,
                                         pred_rf_t, sigma_rf_t_v2, 0.05)
    for label, lo, hi in [("RF Standard", lo_std, hi_std),
                           ("RF Adaptif V1 (std arbres)", lo_v1, hi_v1),
                           ("RF Adaptif V2 (modèle séparé)", lo_v2, hi_v2)]:
        cov = ((yt >= lo) & (yt <= hi)).mean() * 100
        widths_cm = (hi - lo) * 100
        print(f"  {label:<32s} couverture {cov:>5.1f}% | "
              f"largeur moy {widths_cm.mean():>5.1f}cm | "
              f"écart-type largeurs {widths_cm.std():>5.1f}cm | "
              f"min-max [{widths_cm.min():>4.1f} ; {widths_cm.max():>5.1f}]")
        results.append({"modele": label, "couverture_95": cov,
                         "largeur_moy": widths_cm.mean(),
                         "largeur_std": widths_cm.std(),
                         "largeur_min": widths_cm.min(),
                         "largeur_max": widths_cm.max(),
                         "lo": lo, "hi": hi})

    # === Hurdle ===
    print("\n--- Hurdle ---")
    lo_std, hi_std, q_std = standard_ci(pred_hu_v, yv, pred_hu_t, 0.05)
    lo_v1, hi_v1, qn_v1 = adaptive_ci(pred_hu_v, sigma_hu_v_v1, yv,
                                         pred_hu_t, sigma_hu_t_v1, 0.05)
    for label, lo, hi in [("Hurdle Standard", lo_std, hi_std),
                           ("Hurdle Adaptif V1", lo_v1, hi_v1)]:
        cov = ((yt >= lo) & (yt <= hi)).mean() * 100
        widths_cm = (hi - lo) * 100
        print(f"  {label:<32s} couverture {cov:>5.1f}% | "
              f"largeur moy {widths_cm.mean():>5.1f}cm | "
              f"écart-type largeurs {widths_cm.std():>5.1f}cm | "
              f"min-max [{widths_cm.min():>4.1f} ; {widths_cm.max():>5.1f}]")
        results.append({"modele": label, "couverture_95": cov,
                         "largeur_moy": widths_cm.mean(),
                         "largeur_std": widths_cm.std(),
                         "largeur_min": widths_cm.min(),
                         "largeur_max": widths_cm.max(),
                         "lo": lo, "hi": hi})

    # ============================================================
    # ÉTAPE 4 — Couverture par tranche pour Hurdle adaptatif
    # ============================================================
    print("\n[4/4] Couverture par tranche — focus sur Hurdle Adaptatif V1 vs Standard...")
    df = pd.DataFrame({"y_true": yt})
    df["tranche"] = pd.cut(df["y_true"], bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])
    hu_std = next(r for r in results if r["modele"] == "Hurdle Standard")
    hu_ad = next(r for r in results if r["modele"] == "Hurdle Adaptif V1")
    df["std_cov"] = (yt >= hu_std["lo"]) & (yt <= hu_std["hi"])
    df["std_w"] = (hu_std["hi"] - hu_std["lo"]) * 100
    df["ad_cov"] = (yt >= hu_ad["lo"]) & (yt <= hu_ad["hi"])
    df["ad_w"] = (hu_ad["hi"] - hu_ad["lo"]) * 100

    print("\n" + "=" * 95)
    print("HURDLE — COUVERTURE ET LARGEUR PAR TRANCHE")
    print("=" * 95)
    print(f"{'Tranche':<12s} | {'n':>5s} | "
          f"{'cov std':>8s} | {'cov adapt.':>11s} | "
          f"{'wid std':>8s} | {'wid adapt.':>12s} | {'gain wid':>9s}")
    print("-" * 95)
    for tr in df["tranche"].cat.categories:
        sub = df[df["tranche"] == tr]
        if len(sub) == 0:
            continue
        c_std = sub["std_cov"].mean() * 100
        c_ad = sub["ad_cov"].mean() * 100
        w_std = sub["std_w"].mean()
        w_ad = sub["ad_w"].mean()
        gain = w_std - w_ad
        sign = "↓" if gain > 0 else "↑"
        print(f"{str(tr):<12s} | {len(sub):>5d} | "
              f"{c_std:>7.1f}% | {c_ad:>10.1f}% | "
              f"{w_std:>7.2f}c | {w_ad:>11.2f}c | {gain:>+6.2f} {sign}")

    # ---------- Exemples concrets adaptatif ----------
    print("\n" + "=" * 100)
    print("EXEMPLES — 8 CHEVAUX (IC95% Hurdle adaptatif vs standard)")
    print("=" * 100)
    rng = np.random.default_rng(seed=42)
    sample_idx = rng.choice(len(df), 8, replace=False)
    for i in sample_idx:
        v = yt[i]
        tr = df.iloc[i]["tranche"]
        s_lo, s_hi = hu_std["lo"][i], hu_std["hi"][i]
        a_lo, a_hi = hu_ad["lo"][i], hu_ad["hi"][i]
        s_cov = "✓" if (s_lo <= v <= s_hi) else "✗"
        a_cov = "✓" if (a_lo <= v <= a_hi) else "✗"
        print(f"  Cheval idx {i} (vrai = {v:.2f}m, tranche {tr})")
        print(f"    Standard  : [{s_lo:.2f} ; {s_hi:.2f}] largeur {(s_hi-s_lo)*100:.1f}cm  {s_cov}")
        print(f"    Adaptatif : [{a_lo:.2f} ; {a_hi:.2f}] largeur {(a_hi-a_lo)*100:.1f}cm  {a_cov}")

    # Sauvegarde
    out = pd.DataFrame([{k: r[k] for k in ["modele", "couverture_95", "largeur_moy",
                                              "largeur_std", "largeur_min", "largeur_max"]}
                          for r in results])
    out.to_csv(MASTER_DIR / "locally_adaptive_conformal_results.csv", index=False)
    print(f"\n→ locally_adaptive_conformal_results.csv")


if __name__ == "__main__":
    main()
