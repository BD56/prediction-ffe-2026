"""
33 - Comparaison des importances de variables entre les 4 modèles.

Sources :
  - RF        : rf_importance.csv (déjà calculé)
  - CatBoost  : catboost_importance.csv (déjà calculé)
  - ElasticNet: baseline_elasticnet_coefficients.csv (déjà calculé)
  - XGBoost   : calculé ici (gain)

Sortie : data/master/importance_comparative.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def main():
    print("=== 33 - Importances comparatives ===\n")

    # ---------- 1. XGBoost importance ----------
    print("[1/2] Entraînement XGBoost pour récupérer son importance...")
    master = pd.read_parquet(MASTER_DIR / "master_dataset_clean.parquet")
    master = master.set_index("IDCHEVAL")
    feat_cols = [c for c in master.columns if c.startswith("f")]
    for c in feat_cols:
        if master[c].dtype == bool:
            master[c] = master[c].astype(int)
    X = master[feat_cols].select_dtypes(include=[np.number])
    y = master["hauteur_max_validee"]
    split = master["SPLIT"]

    X_train, y_train = X[split == "train"], y[split == "train"]
    X_valid, y_valid = X[split == "valid"], y[split == "valid"]

    xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
                       random_state=42, n_jobs=-1, eval_metric="mae",
                       early_stopping_rounds=30)
    xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    imp_xgb = pd.DataFrame({"feature": X.columns,
                            "importance": xgb.feature_importances_})
    imp_xgb.to_csv(MASTER_DIR / "xgb_importance.csv", index=False)
    print(f"  ✓ {len(imp_xgb)} features")

    # ---------- 2. Charger les autres + consolider ----------
    print("\n[2/2] Consolidation des importances...")
    rf = pd.read_csv(MASTER_DIR / "rf_importance.csv")
    cb = pd.read_csv(MASTER_DIR / "catboost_importance.csv")
    en = pd.read_csv(MASTER_DIR / "baseline_elasticnet_coefficients.csv")

    # Normaliser chaque importance en rang (1 = plus important)
    def add_rank(df, value_col, name):
        df = df[["feature", value_col]].copy()
        df.columns = ["feature", f"imp_{name}"]
        df[f"rank_{name}"] = df[f"imp_{name}"].rank(ascending=False, method="min").astype(int)
        return df

    rf_r = add_rank(rf, "importance", "RF")
    cb_r = add_rank(cb, "importance", "CB")
    xgb_r = add_rank(imp_xgb, "importance", "XGB")
    en_r = add_rank(en, "abs_coef", "EN")

    merged = rf_r.merge(cb_r, on="feature", how="outer") \
                 .merge(xgb_r, on="feature", how="outer") \
                 .merge(en_r, on="feature", how="outer")

    # Rang moyen (consensus)
    rank_cols = [c for c in merged.columns if c.startswith("rank_")]
    merged["rank_moyen"] = merged[rank_cols].mean(axis=1)
    merged = merged.sort_values("rank_moyen")

    # ---------- Affichage ----------
    print("\n" + "=" * 110)
    print("TOP 20 — RANG MOYEN (consensus des 4 modèles)")
    print("=" * 110)
    print(f"{'#':>3} | {'Feature':<40s} | {'RF':>4} | {'XGB':>4} | {'CB':>4} | {'EN':>4} | {'Moyen':>6}")
    print("-" * 90)
    for i, r in merged.head(20).iterrows():
        print(f"{int(r['rank_RF']) if pd.notna(r['rank_RF']) else '--':>3} | "
              f"{r['feature']:<40s} | "
              f"{int(r['rank_RF']) if pd.notna(r['rank_RF']) else '-':>4} | "
              f"{int(r['rank_XGB']) if pd.notna(r['rank_XGB']) else '-':>4} | "
              f"{int(r['rank_CB']) if pd.notna(r['rank_CB']) else '-':>4} | "
              f"{int(r['rank_EN']) if pd.notna(r['rank_EN']) else '-':>4} | "
              f"{r['rank_moyen']:>6.1f}")

    # Top 10 individuels
    print("\n" + "=" * 90)
    print("TOP 10 PAR MODÈLE")
    print("=" * 90)
    for name, df, col in [("RF", rf, "importance"),
                          ("XGBoost", imp_xgb, "importance"),
                          ("CatBoost", cb, "importance"),
                          ("ElasticNet", en, "abs_coef")]:
        print(f"\n— {name} —")
        top = df.sort_values(col, ascending=False).head(10)
        for i, (_, row) in enumerate(top.iterrows(), 1):
            print(f"  {i:>2}. {row['feature']:<45s} ({row[col]:.4f})")

    merged.to_csv(MASTER_DIR / "importance_comparative.csv", index=False)
    print(f"\n→ importance_comparative.csv ({len(merged)} features)")


if __name__ == "__main__":
    main()
