"""
57 - Comparaison de Hurdle sur les 3 datasets (brut / épuré corr / épuré v2).

Datasets comparés :
  D1. master_dataset_final.parquet     — 273 features brutes, TE NON corrigé (leakage train→test)
  D2. master_dataset_clean.parquet     — 190 features, TE corrigé + épuration corrélation 0.95
  D3. master_dataset_epure_v2.parquet  — 156 features, + épuration empirique consensus

Métriques étendues sur le test set :
  Précision : MAE, RMSE, R², MAPE
  Corrélations : Pearson, Spearman (rang-based, robuste outliers)
  Biais : moyenne et médiane des résidus signés
  Distribution : skewness, kurtosis des résidus
  Classifier : AUC, précision, rappel, F1
  Par tranche : MAE et RMSE

Sortie : data/master/hurdle_compare_datasets_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              mean_absolute_percentage_error,
                              precision_score, recall_score, f1_score, roc_auc_score)
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def fit_hurdle(X_train, y_train, X_test, y_test, seed=42):
    """Entraîne Hurdle complet et retourne (pred, p_top, n_features, classifier metrics)."""
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xte = imp.transform(X_test)

    rf_def = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                     min_samples_split=10, max_features="sqrt",
                                     random_state=seed, n_jobs=-1)
    rf_def.fit(Xtr, y_train)
    pred_rf = rf_def.predict(Xte)

    y_train_bin = (y_train >= 1.40).astype(int)
    clf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                   min_samples_split=10, max_features="sqrt",
                                   class_weight="balanced", random_state=seed, n_jobs=-1)
    clf.fit(Xtr, y_train_bin)
    p_test = clf.predict_proba(Xte)[:, 1]

    mask = (y_train >= 1.40).values
    rf_tops = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                      min_samples_split=5, max_features="sqrt",
                                      random_state=seed, n_jobs=-1)
    rf_tops.fit(Xtr[mask], y_train.values[mask])
    pred_tops = rf_tops.predict(Xte)

    pred_hurdle = p_test * pred_tops + (1 - p_test) * pred_rf

    # Métriques classifier
    y_test_bin = (y_test >= 1.40).astype(int)
    p_bin = (p_test >= 0.5).astype(int)
    clf_metrics = {
        "clf_AUC": roc_auc_score(y_test_bin, p_test),
        "clf_Precision": precision_score(y_test_bin, p_bin, zero_division=0),
        "clf_Rappel": recall_score(y_test_bin, p_bin, zero_division=0),
        "clf_F1": f1_score(y_test_bin, p_bin, zero_division=0),
    }

    return pred_hurdle, p_test, X_train.shape[1], clf_metrics


def compute_metrics(y_true, y_pred):
    """Calcule toutes les métriques étendues."""
    yv = y_true.values if hasattr(y_true, 'values') else y_true
    res = yv - y_pred  # résidus signés

    mae = mean_absolute_error(yv, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(yv, y_pred)) * 100
    r2 = r2_score(yv, y_pred)
    mape = mean_absolute_percentage_error(yv, y_pred) * 100  # en %

    pearson_r, _ = stats.pearsonr(yv, y_pred)
    spearman_r, _ = stats.spearmanr(yv, y_pred)

    biais_moy = res.mean() * 100
    biais_med = np.median(res) * 100
    res_skew = pd.Series(res).skew()
    res_kurt = pd.Series(res).kurt()

    # Par tranche
    df = pd.DataFrame({"y": yv, "p": y_pred})
    df["tranche"] = pd.cut(df["y"], bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])
    df["err_abs"] = (df["y"] - df["p"]).abs() * 100
    df["err_sq"] = (df["y"] - df["p"]) ** 2 * 10000
    mae_tr = df.groupby("tranche", observed=True)["err_abs"].mean().to_dict()
    rmse_tr = df.groupby("tranche", observed=True)["err_sq"].mean().apply(np.sqrt).to_dict()

    return {
        "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape,
        "Pearson_r": pearson_r, "Spearman_r": spearman_r,
        "biais_moy_cm": biais_moy, "biais_med_cm": biais_med,
        "skew": res_skew, "kurtosis": res_kurt,
        "mae_tr": mae_tr, "rmse_tr": rmse_tr,
    }


def main():
    print("=== 57 - Comparaison Hurdle sur 3 datasets ===\n")

    datasets = [
        ("D1. Brut (273 feat., TE non corrigé)", "master_dataset_final.parquet"),
        ("D2. Clean (190 feat., TE corr. + corr.)", "master_dataset_clean.parquet"),
        ("D3. Epure v2 (156 feat., + empirique)", "master_dataset_epure_v2.parquet"),
    ]

    results = []

    for label, fname in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET : {label}")
        print(f"{'='*70}")
        t0 = time.time()
        df = pd.read_parquet(MASTER_DIR / fname)
        df = df.set_index("IDCHEVAL")
        feat_cols = [c for c in df.columns if c.startswith("f")]
        for c in feat_cols:
            if df[c].dtype == bool:
                df[c] = df[c].astype(int)
        X = df[feat_cols].select_dtypes(include=[np.number])
        y = df["hauteur_max_validee"]
        split = df["SPLIT"]

        X_train, y_train = X[split == "train"], y[split == "train"]
        X_test, y_test = X[split == "test"], y[split == "test"]
        print(f"  Train : {len(X_train):,} | Test : {len(X_test):,} | Features : {X.shape[1]}")

        pred, p_test, n_feat, clf_m = fit_hurdle(X_train, y_train, X_test, y_test)
        metrics = compute_metrics(y_test, pred)
        metrics.update(clf_m)
        metrics["dataset"] = label
        metrics["n_features"] = n_feat
        metrics["fit_time_s"] = time.time() - t0
        results.append(metrics)
        print(f"  ✓ {metrics['fit_time_s']:.1f}s | MAE {metrics['MAE']:.2f} | "
              f"RMSE {metrics['RMSE']:.2f} | R² {metrics['R2']:.4f}")

    # ============================================================
    # Affichage comparatif
    # ============================================================
    print("\n" + "="*100)
    print("TABLEAU COMPARATIF — MÉTRIQUES GLOBALES")
    print("="*100)
    print(f"{'Dataset':<40s} | {'MAE':>6s} | {'RMSE':>6s} | {'R²':>7s} | {'MAPE':>5s} | "
          f"{'Pears.':>6s} | {'Spear.':>6s} | {'biais':>6s}")
    print("-"*100)
    for r in results:
        print(f"{r['dataset']:<40s} | {r['MAE']:>5.2f}c | {r['RMSE']:>5.2f}c | {r['R2']:>7.4f} | "
              f"{r['MAPE_%']:>4.2f}% | {r['Pearson_r']:>6.4f} | {r['Spearman_r']:>6.4f} | "
              f"{r['biais_moy_cm']:>+5.2f}c")

    print("\n" + "="*100)
    print("DISTRIBUTION DES RÉSIDUS")
    print("="*100)
    print(f"{'Dataset':<40s} | {'biais moy':>9s} | {'biais méd':>9s} | {'skew':>6s} | {'kurtosis':>8s}")
    print("-"*90)
    for r in results:
        print(f"{r['dataset']:<40s} | {r['biais_moy_cm']:>+8.2f}c | {r['biais_med_cm']:>+8.2f}c | "
              f"{r['skew']:>+5.2f} | {r['kurtosis']:>+7.2f}")

    print("\n" + "="*100)
    print("CLASSIFIER (≥1,40m)")
    print("="*100)
    print(f"{'Dataset':<40s} | {'AUC':>6s} | {'Précision':>10s} | {'Rappel':>7s} | {'F1':>5s}")
    print("-"*80)
    for r in results:
        print(f"{r['dataset']:<40s} | {r['clf_AUC']:>6.4f} | {r['clf_Precision']:>10.4f} | "
              f"{r['clf_Rappel']:>7.4f} | {r['clf_F1']:>5.4f}")

    print("\n" + "="*110)
    print("MAE PAR TRANCHE (cm)")
    print("="*110)
    print(f"{'Dataset':<40s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-"*105)
    for r in results:
        mt = r["mae_tr"]
        print(f"{r['dataset']:<40s} | {mt['≤1.10m']:>6.2f}c | {mt['1.15-1.20m']:>9.2f}c | "
              f"{mt['1.25-1.30m']:>9.2f}c | {mt['1.35-1.40m']:>9.2f}c | {mt['≥1.45m']:>6.2f}c")

    print("\n" + "="*110)
    print("RMSE PAR TRANCHE (cm)")
    print("="*110)
    print(f"{'Dataset':<40s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-"*105)
    for r in results:
        rt = r["rmse_tr"]
        print(f"{r['dataset']:<40s} | {rt['≤1.10m']:>6.2f}c | {rt['1.15-1.20m']:>9.2f}c | "
              f"{rt['1.25-1.30m']:>9.2f}c | {rt['1.35-1.40m']:>9.2f}c | {rt['≥1.45m']:>6.2f}c")

    # Sauvegarde
    flat_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ["mae_tr", "rmse_tr"]}
        row.update({f"MAE_{k}": v for k, v in r["mae_tr"].items()})
        row.update({f"RMSE_{k}": v for k, v in r["rmse_tr"].items()})
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(MASTER_DIR / "hurdle_compare_datasets_results.csv", index=False)
    print(f"\n→ hurdle_compare_datasets_results.csv")


if __name__ == "__main__":
    main()
