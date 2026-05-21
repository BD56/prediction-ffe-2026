"""
47 - Validation croisée du résultat : robustesse de Hurdle vs RF default vs Stacking+Calib
     sur différents découpages des données.

Trois protocoles :
  A. 4 expanding window splits (1-year test each)
  B. 1 split aléatoire 70/15/15 (contrôle drift)
  C. 5-fold TimeSeriesSplit sklearn (CV temporelle formelle)

Modèles évalués : RF default, Hurdle (mélange), Stacking + Calib.
Métriques : MAE globale, MAE ≥1,40m, MAE ≥1,45m, R² global.

Sortie : data/master/validation_splits_results.csv
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
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def safe_mae(y_true, y_pred, mask):
    """MAE sur un sous-ensemble, ou NaN si vide."""
    if mask.sum() == 0:
        return np.nan
    return mean_absolute_error(y_true[mask], y_pred[mask]) * 100


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
    r2 = r2_score(y_true, y_pred)
    yv = y_true.values if hasattr(y_true, "values") else y_true
    mae_140 = safe_mae(yv, y_pred, (yv >= 1.40) & (yv < 1.45))
    mae_145 = safe_mae(yv, y_pred, yv >= 1.45)
    return {"MAE_global": mae, "RMSE_global": rmse, "R2": r2,
            "MAE_1.40-1.45": mae_140, "MAE_>=1.45": mae_145}


def fit_eval_models(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """Fit RF default, Hurdle, Stacking+Calib. Retourne dict des métriques par modèle."""
    # Imputation
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xva = imp.transform(X_valid)
    Xte = imp.transform(X_test)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xva_sc = sc.transform(Xva)
    Xte_sc = sc.transform(Xte)

    # --- RF default ---
    rf = RandomForestRegressor(n_estimators=500, max_depth=15,
                                 min_samples_leaf=10, min_samples_split=10,
                                 max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train)
    pred_rf_test = rf.predict(Xte)
    pred_rf_valid = rf.predict(Xva)

    # --- ElasticNet (pour Stacking) ---
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                      n_jobs=-1, random_state=42)
    en.fit(Xtr_sc, y_train)
    pred_en_valid = en.predict(Xva_sc)
    pred_en_test = en.predict(Xte_sc)

    # --- CatBoost (pour Stacking) ---
    cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3, random_seed=42, loss_function="MAE",
                            eval_metric="MAE", early_stopping_rounds=30, verbose=0)
    cb.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    pred_cb_valid = cb.predict(X_valid)
    pred_cb_test = cb.predict(X_test)

    # --- Hurdle ---
    y_train_bin = (y_train >= 1.40).astype(int)
    clf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                   min_samples_leaf=10, min_samples_split=10,
                                   max_features="sqrt", class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    clf.fit(Xtr, y_train_bin)
    p_test = clf.predict_proba(Xte)[:, 1]
    mask = (y_train >= 1.40).values
    if mask.sum() < 50:
        return None  # trop peu de tops, fold non utilisable
    rf_tops = RandomForestRegressor(n_estimators=500, max_depth=15,
                                      min_samples_leaf=5, min_samples_split=5,
                                      max_features="sqrt", random_state=42, n_jobs=-1)
    rf_tops.fit(Xtr[mask], y_train[mask])
    pred_tops_test = rf_tops.predict(Xte)
    pred_hurdle_test = p_test * pred_tops_test + (1 - p_test) * pred_rf_test

    # --- Stacking + Calib ---
    X_meta_v = np.column_stack([pred_rf_valid, pred_en_valid, pred_cb_valid])
    X_meta_t = np.column_stack([pred_rf_test, pred_en_test, pred_cb_test])
    meta = LinearRegression()
    meta.fit(X_meta_v, y_valid)
    pred_stack_v = meta.predict(X_meta_v)
    pred_stack_t = meta.predict(X_meta_t)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(pred_stack_v, y_valid)
    pred_stack_cal = cal.predict(pred_stack_t)

    return {
        "RF default": evaluate(y_test, pred_rf_test),
        "Hurdle (mélange)": evaluate(y_test, pred_hurdle_test),
        "Stacking + Calib": evaluate(y_test, pred_stack_cal),
    }


def main():
    print("=== 47 - Validation croisée : robustesse de Hurdle ===\n")

    # ---------- Chargement ----------
    v2 = pd.read_parquet(MASTER_DIR / "master_dataset_epure_v2.parquet")
    v2["annee"] = v2["DATENAISSANCE"].astype(int)
    feat_cols = [c for c in v2.columns if c.startswith("f")]
    for c in feat_cols:
        if v2[c].dtype == bool:
            v2[c] = v2[c].astype(int)

    rows = []

    # ============================================================
    # OPTION A : 4 expanding window splits
    # ============================================================
    print("=" * 70)
    print("OPTION A : 4 expanding window splits (1-year test each)")
    print("=" * 70)

    configs_A = [
        {"name": "A1", "train_max": 2008, "valid": 2009, "test": 2010},
        {"name": "A2", "train_max": 2009, "valid": 2010, "test": 2011},
        {"name": "A3", "train_max": 2010, "valid": 2011, "test": 2012},
        {"name": "A4", "train_max": 2011, "valid": 2012, "test": 2013},
    ]
    for cfg in configs_A:
        print(f"\n[{cfg['name']}] Train 2006-{cfg['train_max']} | Valid {cfg['valid']} | Test {cfg['test']}...")
        tr_mask = v2["annee"] <= cfg["train_max"]
        va_mask = v2["annee"] == cfg["valid"]
        te_mask = v2["annee"] == cfg["test"]
        sub_tr = v2[tr_mask].set_index("IDCHEVAL")
        sub_va = v2[va_mask].set_index("IDCHEVAL")
        sub_te = v2[te_mask].set_index("IDCHEVAL")
        X_train = sub_tr[feat_cols].select_dtypes(include=[np.number])
        X_valid = sub_va[feat_cols].select_dtypes(include=[np.number])
        X_test = sub_te[feat_cols].select_dtypes(include=[np.number])
        y_train = sub_tr["hauteur_max_validee"]
        y_valid = sub_va["hauteur_max_validee"]
        y_test = sub_te["hauteur_max_validee"]
        print(f"  Train={len(X_train):,} | Valid={len(X_valid):,} | Test={len(X_test):,}")
        t0 = time.time()
        results = fit_eval_models(X_train, y_train, X_valid, y_valid, X_test, y_test)
        print(f"  ✓ {time.time()-t0:.0f}s")
        for model_name, mets in results.items():
            print(f"    {model_name:<22s} MAE={mets['MAE_global']:>5.2f} | "
                  f"≥1.40m={mets['MAE_1.40-1.45']:>5.2f} | ≥1.45m={mets['MAE_>=1.45']:>5.2f}")
            rows.append({"protocole": "A", "config": cfg["name"], "modele": model_name,
                          **mets, "n_test": len(X_test)})

    # ============================================================
    # OPTION B : Random 70/15/15 split (contrôle drift)
    # ============================================================
    print("\n" + "=" * 70)
    print("OPTION B : Random 70/15/15 (contrôle drift, pas de respect temporel)")
    print("=" * 70)

    np.random.seed(42)
    n = len(v2)
    idx = np.random.permutation(n)
    n_train = int(0.70 * n)
    n_valid = int(0.15 * n)
    idx_tr = idx[:n_train]
    idx_va = idx[n_train:n_train+n_valid]
    idx_te = idx[n_train+n_valid:]
    sub_tr = v2.iloc[idx_tr].set_index("IDCHEVAL")
    sub_va = v2.iloc[idx_va].set_index("IDCHEVAL")
    sub_te = v2.iloc[idx_te].set_index("IDCHEVAL")
    X_train = sub_tr[feat_cols].select_dtypes(include=[np.number])
    X_valid = sub_va[feat_cols].select_dtypes(include=[np.number])
    X_test = sub_te[feat_cols].select_dtypes(include=[np.number])
    y_train = sub_tr["hauteur_max_validee"]
    y_valid = sub_va["hauteur_max_validee"]
    y_test = sub_te["hauteur_max_validee"]
    print(f"  Train={len(X_train):,} | Valid={len(X_valid):,} | Test={len(X_test):,}")
    t0 = time.time()
    results = fit_eval_models(X_train, y_train, X_valid, y_valid, X_test, y_test)
    print(f"  ✓ {time.time()-t0:.0f}s")
    for model_name, mets in results.items():
        print(f"    {model_name:<22s} MAE={mets['MAE_global']:>5.2f} | "
              f"≥1.40m={mets['MAE_1.40-1.45']:>5.2f} | ≥1.45m={mets['MAE_>=1.45']:>5.2f}")
        rows.append({"protocole": "B", "config": "random_70_15_15", "modele": model_name,
                      **mets, "n_test": len(X_test)})

    # ============================================================
    # OPTION C : TimeSeriesSplit 5-fold (CV temporelle formelle)
    # ============================================================
    print("\n" + "=" * 70)
    print("OPTION C : TimeSeriesSplit 5-fold (CV temporelle expanding window)")
    print("=" * 70)

    # Ordonner par année puis ID pour avoir un ordre temporel
    v2_sorted = v2.sort_values(["annee", "IDCHEVAL"]).reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=5)
    for i, (tr_idx, te_idx) in enumerate(tscv.split(v2_sorted), 1):
        sub_tr_all = v2_sorted.iloc[tr_idx]
        sub_te = v2_sorted.iloc[te_idx]
        # Découper le train en train+valid (90/10) pour stacking
        n_tr = int(0.9 * len(sub_tr_all))
        sub_tr = sub_tr_all.iloc[:n_tr].set_index("IDCHEVAL")
        sub_va = sub_tr_all.iloc[n_tr:].set_index("IDCHEVAL")
        sub_te = sub_te.set_index("IDCHEVAL")
        X_train = sub_tr[feat_cols].select_dtypes(include=[np.number])
        X_valid = sub_va[feat_cols].select_dtypes(include=[np.number])
        X_test = sub_te[feat_cols].select_dtypes(include=[np.number])
        y_train = sub_tr["hauteur_max_validee"]
        y_valid = sub_va["hauteur_max_validee"]
        y_test = sub_te["hauteur_max_validee"]
        # Plage d'années pour info
        annees_tr = (sub_tr_all["annee"].min(), sub_tr_all["annee"].max())
        annees_te = (v2_sorted.iloc[te_idx]["annee"].min(),
                      v2_sorted.iloc[te_idx]["annee"].max())
        print(f"\n[C-fold{i}] Train années {annees_tr[0]}-{annees_tr[1]} ({len(X_train):,} "
              f"+ valid {len(X_valid):,}) | Test années {annees_te[0]}-{annees_te[1]} ({len(X_test):,})")
        t0 = time.time()
        results = fit_eval_models(X_train, y_train, X_valid, y_valid, X_test, y_test)
        if results is None:
            print(f"  ⚠ Fold ignoré (trop peu de tops dans train)")
            continue
        print(f"  ✓ {time.time()-t0:.0f}s")
        for model_name, mets in results.items():
            print(f"    {model_name:<22s} MAE={mets['MAE_global']:>5.2f} | "
                  f"≥1.40m={mets['MAE_1.40-1.45']:>5.2f} | ≥1.45m={mets['MAE_>=1.45']:>5.2f}")
            rows.append({"protocole": "C", "config": f"fold_{i}", "modele": model_name,
                          **mets, "n_test": len(X_test)})

    # ============================================================
    # Synthèse
    # ============================================================
    df = pd.DataFrame(rows)
    df.to_csv(MASTER_DIR / "validation_splits_results.csv", index=False)

    print("\n" + "=" * 70)
    print("SYNTHÈSE FINALE")
    print("=" * 70)
    for proto in ["A", "B", "C"]:
        sub = df[df["protocole"] == proto]
        if sub.empty:
            continue
        print(f"\n--- Protocole {proto} ---")
        summary = sub.groupby("modele")[["MAE_global", "MAE_1.40-1.45", "MAE_>=1.45", "R2"]].agg(
            ["mean", "std"]).round(2)
        print(summary.to_string())

    print(f"\n→ validation_splits_results.csv ({len(df)} lignes)")


if __name__ == "__main__":
    main()
