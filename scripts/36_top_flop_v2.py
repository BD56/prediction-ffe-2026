"""
36 - Top/Flop des variables sur master_dataset_epure_v2.

Refit RF / XGB / CB / EN sur v2, extrait les importances, et combine
avec corrélation cible + stats descriptives.

Sortie : data/master/top_flop_v2_avec_valeurs.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def main():
    print("=== 36 - Top/Flop sur v2 ===\n")

    # ---------- Charger v2 ----------
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
    print(f"Features : {X.shape[1]} | Train/Valid : {len(X_train):,} / {len(X_valid):,}\n")

    # Imputation pour RF / EN
    imp = SimpleImputer(strategy="median")
    X_train_med = imp.fit_transform(X_train)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train_med)

    # ---------- Refit + importances ----------
    print("[1/4] RF...", end="", flush=True)
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                 min_samples_split=10, max_features="sqrt",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train_med, y_train)
    imp_rf = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
    print(f" {time.time()-t0:.0f}s")

    print("[2/4] XGBoost...", end="", flush=True)
    t0 = time.time()
    xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                        reg_lambda=1.0, min_child_weight=10, random_state=42,
                        n_jobs=-1, eval_metric="mae", early_stopping_rounds=30)
    xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    imp_xgb = pd.DataFrame({"feature": X.columns, "importance": xgb.feature_importances_})
    print(f" {time.time()-t0:.0f}s")

    print("[3/4] CatBoost...", end="", flush=True)
    t0 = time.time()
    cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                            random_seed=42, loss_function="MAE", eval_metric="MAE",
                            early_stopping_rounds=30, verbose=0)
    cb.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    imp_cb = pd.DataFrame({"feature": X.columns,
                            "importance": cb.get_feature_importance()})
    print(f" {time.time()-t0:.0f}s")

    print("[4/4] ElasticNet...", end="", flush=True)
    t0 = time.time()
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                      n_jobs=-1, random_state=42)
    en.fit(X_train_sc, y_train)
    imp_en = pd.DataFrame({"feature": X.columns,
                            "coef": en.coef_,
                            "abs_coef": np.abs(en.coef_)})
    print(f" {time.time()-t0:.0f}s")

    # ---------- Normaliser + rangs ----------
    def add(df, val_col, name):
        out = df[["feature", val_col]].copy()
        out[f"imp_{name}_%"] = 100 * out[val_col] / out[val_col].sum()
        out[f"rank_{name}"] = out[val_col].rank(ascending=False, method="min").astype(int)
        return out[["feature", f"imp_{name}_%", f"rank_{name}"]]

    a = add(imp_rf, "importance", "RF")
    b = add(imp_xgb, "importance", "XGB")
    c = add(imp_cb, "importance", "CB")
    d = add(imp_en, "abs_coef", "EN")
    merged = a.merge(b, on="feature").merge(c, on="feature").merge(d, on="feature")
    rank_cols = [k for k in merged.columns if k.startswith("rank_")]
    merged["rank_moyen"] = merged[rank_cols].mean(axis=1)

    # ---------- Stats descriptives + corr ----------
    stats = []
    for f in X_train.columns:
        col = X_train[f]
        stats.append({
            "feature": f, "mean": col.mean(), "std": col.std(),
            "nan_pct": 100 * col.isna().mean(), "corr_y": col.corr(y_train),
        })
    stats = pd.DataFrame(stats)
    full = merged.merge(stats, on="feature").sort_values("rank_moyen")

    # ---------- Affichage ----------
    def show(df, title, n=20):
        print("\n" + "=" * 130)
        print(title)
        print("=" * 130)
        print(f"{'#':>3} | {'Feature':<42s} | {'%RF':>5s} | {'%XGB':>5s} | {'%CB':>5s} | {'%EN':>5s} | "
              f"{'corr_y':>7s} | {'NaN%':>5s} | {'mean':>9s} | {'std':>9s}")
        print("-" * 120)
        for i, (_, r) in enumerate(df.head(n).iterrows(), 1):
            def f(v, fmt="5.2f"):
                return f"{v:{fmt}}" if pd.notna(v) else "   --"
            print(f"{i:>3} | {r['feature']:<42s} | {f(r['imp_RF_%'])} | {f(r['imp_XGB_%'])} | "
                  f"{f(r['imp_CB_%'])} | {f(r['imp_EN_%'])} | {f(r['corr_y'],'7.3f')} | "
                  f"{f(r['nan_pct'])} | {f(r['mean'],'9.3f')} | {f(r['std'],'9.3f')}")

    show(full, "TOP 20 — v2 (156 features)")
    show(full.sort_values("rank_moyen", ascending=False), "FLOP 20 — v2 (156 features)")

    full.to_csv(MASTER_DIR / "top_flop_v2_avec_valeurs.csv", index=False)
    print(f"\n→ top_flop_v2_avec_valeurs.csv ({len(full)} features)")


if __name__ == "__main__":
    main()
