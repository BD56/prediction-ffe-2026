"""
49 - Intervalles de confiance par bootstrap.

Méthode :
  - K = 50 itérations bootstrap (échantillon train avec remise)
  - Pour chaque itération : refit complet du modèle, prédiction sur test
  - Pour chaque cheval test : 50 prédictions → IC 95% = [P2.5, P97.5], IC 80% = [P10, P90]

Modèles : RF default et Hurdle (mélange).

Métriques :
  - Largeur moyenne d'IC par tranche
  - Couverture empirique (% de fois où y_true ∈ IC)

Sortie : data/master/bootstrap_ci_results.csv
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


def fit_rf_default(Xtr, y_train, Xte, seed):
    rf = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                 min_samples_split=10, max_features="sqrt",
                                 random_state=seed, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict(Xte)


def fit_hurdle(Xtr, y_train, Xte, seed):
    y_train_bin = (y_train >= 1.40).astype(int)
    # Si pas assez de tops dans le bootstrap, fallback sur RF default
    if y_train_bin.sum() < 50:
        return fit_rf_default(Xtr, y_train, Xte, seed)
    clf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                   min_samples_split=10, max_features="sqrt",
                                   class_weight="balanced", random_state=seed, n_jobs=-1)
    clf.fit(Xtr, y_train_bin)
    p_test = clf.predict_proba(Xte)[:, 1]
    mask = (y_train >= 1.40).values
    rf_tops = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                      min_samples_split=5, max_features="sqrt",
                                      random_state=seed, n_jobs=-1)
    rf_tops.fit(Xtr[mask], y_train[mask])
    pred_tops = rf_tops.predict(Xte)
    rf_def = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                     min_samples_split=10, max_features="sqrt",
                                     random_state=seed, n_jobs=-1)
    rf_def.fit(Xtr, y_train)
    pred_def = rf_def.predict(Xte)
    return p_test * pred_tops + (1 - p_test) * pred_def


def main():
    print("=== 49 - Intervalles de confiance par bootstrap (K=50) ===\n")

    K = 50

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

    X_train_full, y_train_full = X[split == "train"], y[split == "train"]
    X_test, y_test = X[split == "test"], y[split == "test"]
    n_train = len(X_train_full)
    print(f"Train : {n_train:,} | Test : {len(X_test):,}\n")

    # Imputation médiane sur le train original (fixe), appliquée test
    imp_med = SimpleImputer(strategy="median")
    Xtr_full_med = imp_med.fit_transform(X_train_full)
    Xte_med = imp_med.transform(X_test)
    y_train_full_arr = y_train_full.values

    # ---------- Bootstrap RF default ----------
    print(f"[1/2] Bootstrap RF default ({K} itérations)...")
    preds_rf = np.zeros((K, len(X_test)))
    t0 = time.time()
    for k in range(K):
        rng = np.random.default_rng(seed=k)
        idx = rng.integers(0, n_train, n_train)  # sample avec remise
        Xtr_k = Xtr_full_med[idx]
        ytr_k = y_train_full_arr[idx]
        preds_rf[k] = fit_rf_default(Xtr_k, ytr_k, Xte_med, seed=k)
        if (k+1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k+1) * (K - k - 1)
            print(f"  Itération {k+1}/{K} | {elapsed:.0f}s écoulé | ETA {eta:.0f}s")

    # ---------- Bootstrap Hurdle ----------
    print(f"\n[2/2] Bootstrap Hurdle ({K} itérations)...")
    preds_hu = np.zeros((K, len(X_test)))
    t0 = time.time()
    for k in range(K):
        rng = np.random.default_rng(seed=k)
        idx = rng.integers(0, n_train, n_train)
        Xtr_k = Xtr_full_med[idx]
        ytr_k = pd.Series(y_train_full_arr[idx])  # Hurdle a besoin de .values
        preds_hu[k] = fit_hurdle(Xtr_k, ytr_k, Xte_med, seed=k)
        if (k+1) % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k+1) * (K - k - 1)
            print(f"  Itération {k+1}/{K} | {elapsed:.0f}s écoulé | ETA {eta:.0f}s")

    # ---------- Calcul des IC ----------
    print("\n[Calcul] Intervalles de confiance...")
    def percentiles(arr, qs):
        return np.percentile(arr, qs, axis=0)

    rf_q = percentiles(preds_rf, [2.5, 10, 50, 90, 97.5])
    hu_q = percentiles(preds_hu, [2.5, 10, 50, 90, 97.5])

    df = pd.DataFrame({
        "y_true": y_test.values,
        "rf_p2.5": rf_q[0], "rf_p10": rf_q[1], "rf_median": rf_q[2],
        "rf_p90": rf_q[3], "rf_p97.5": rf_q[4],
        "hu_p2.5": hu_q[0], "hu_p10": hu_q[1], "hu_median": hu_q[2],
        "hu_p90": hu_q[3], "hu_p97.5": hu_q[4],
    }, index=y_test.index)

    df["rf_IC95_width_cm"] = (df["rf_p97.5"] - df["rf_p2.5"]) * 100
    df["rf_IC80_width_cm"] = (df["rf_p90"] - df["rf_p10"]) * 100
    df["hu_IC95_width_cm"] = (df["hu_p97.5"] - df["hu_p2.5"]) * 100
    df["hu_IC80_width_cm"] = (df["hu_p90"] - df["hu_p10"]) * 100

    df["rf_covered_95"] = (df["y_true"] >= df["rf_p2.5"]) & (df["y_true"] <= df["rf_p97.5"])
    df["rf_covered_80"] = (df["y_true"] >= df["rf_p10"]) & (df["y_true"] <= df["rf_p90"])
    df["hu_covered_95"] = (df["y_true"] >= df["hu_p2.5"]) & (df["y_true"] <= df["hu_p97.5"])
    df["hu_covered_80"] = (df["y_true"] >= df["hu_p10"]) & (df["y_true"] <= df["hu_p90"])

    df["tranche"] = pd.cut(df["y_true"], bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])

    # ---------- Synthèse ----------
    print("\n" + "=" * 90)
    print("COUVERTURE EMPIRIQUE GLOBALE")
    print("=" * 90)
    print(f"{'Modèle':<15s} | {'IC nominal':>11s} | {'Couverture':>11s} | {'Largeur moy (cm)':>16s}")
    print("-" * 60)
    print(f"{'RF default':<15s} | {'95%':>11s} | {df['rf_covered_95'].mean()*100:>9.1f}%  | {df['rf_IC95_width_cm'].mean():>15.2f}")
    print(f"{'RF default':<15s} | {'80%':>11s} | {df['rf_covered_80'].mean()*100:>9.1f}%  | {df['rf_IC80_width_cm'].mean():>15.2f}")
    print(f"{'Hurdle':<15s} | {'95%':>11s} | {df['hu_covered_95'].mean()*100:>9.1f}%  | {df['hu_IC95_width_cm'].mean():>15.2f}")
    print(f"{'Hurdle':<15s} | {'80%':>11s} | {df['hu_covered_80'].mean()*100:>9.1f}%  | {df['hu_IC80_width_cm'].mean():>15.2f}")

    print("\n" + "=" * 100)
    print("LARGEUR MOYENNE D'IC ET COUVERTURE PAR TRANCHE")
    print("=" * 100)
    print(f"{'Tranche':<12s} | {'n':>5s} | "
          f"{'IC95 RF':>10s} | {'cov RF':>7s} | {'IC95 Hu':>10s} | {'cov Hu':>7s}")
    print("-" * 70)
    rows_by_t = []
    for tr in df["tranche"].cat.categories:
        sub = df[df["tranche"] == tr]
        if len(sub) == 0:
            continue
        rf_w = sub["rf_IC95_width_cm"].mean()
        rf_c = sub["rf_covered_95"].mean() * 100
        hu_w = sub["hu_IC95_width_cm"].mean()
        hu_c = sub["hu_covered_95"].mean() * 100
        print(f"{str(tr):<12s} | {len(sub):>5d} | "
              f"{rf_w:>9.2f}c | {rf_c:>6.1f}% | {hu_w:>9.2f}c | {hu_c:>6.1f}%")
        rows_by_t.append({"tranche": str(tr), "n": len(sub),
                           "IC95_RF_cm": rf_w, "cov95_RF": rf_c,
                           "IC95_Hurdle_cm": hu_w, "cov95_Hurdle": hu_c})

    # ---------- Exemples concrets ----------
    print("\n" + "=" * 100)
    print("EXEMPLES — 5 CHEVAUX TIRÉS AU HASARD")
    print("=" * 100)
    rng = np.random.default_rng(seed=42)
    sample_idx = rng.choice(len(df), 5, replace=False)
    for i in sample_idx:
        r = df.iloc[i]
        print(f"\n  Cheval {df.index[i]} (vrai = {r['y_true']:.2f}m, tranche {r['tranche']}) :")
        print(f"    RF default IC95 : [{r['rf_p2.5']:.2f} ; {r['rf_p97.5']:.2f}]  "
              f"(largeur {r['rf_IC95_width_cm']:.1f}cm), médiane = {r['rf_median']:.2f}")
        print(f"    Hurdle      IC95 : [{r['hu_p2.5']:.2f} ; {r['hu_p97.5']:.2f}]  "
              f"(largeur {r['hu_IC95_width_cm']:.1f}cm), médiane = {r['hu_median']:.2f}")
        covered_rf = "✓" if r["rf_covered_95"] else "✗"
        covered_hu = "✓" if r["hu_covered_95"] else "✗"
        print(f"    Couverture : RF {covered_rf}  |  Hurdle {covered_hu}")

    # ---------- Sauvegarde ----------
    df.to_csv(MASTER_DIR / "bootstrap_ci_complet.csv")
    pd.DataFrame(rows_by_t).to_csv(MASTER_DIR / "bootstrap_ci_par_tranche.csv", index=False)
    print(f"\n→ bootstrap_ci_complet.csv ({len(df)} chevaux), bootstrap_ci_par_tranche.csv")


if __name__ == "__main__":
    main()
