"""
35 - Top / Flop des variables avec valeurs détaillées.

Pour chaque feature, affiche :
  - Importance normalisée (% du total) par modèle : RF, XGB, CB, EN
  - Corrélation Pearson avec hauteur_max_validee (sur train)
  - Moyenne, écart-type, % de NaN

Sortie : data/master/top_flop_avec_valeurs.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def main():
    print("=== 35 - Top/Flop avec valeurs ===\n")

    # ---------- Charger importances ----------
    rf = pd.read_csv(MASTER_DIR / "rf_importance.csv")
    cb = pd.read_csv(MASTER_DIR / "catboost_importance.csv")
    xgb = pd.read_csv(MASTER_DIR / "xgb_importance.csv")
    en = pd.read_csv(MASTER_DIR / "baseline_elasticnet_coefficients.csv")
    cmp = pd.read_csv(MASTER_DIR / "importance_comparative.csv")

    # Normalisation en % du total par modèle
    def normalize(df, col, name):
        out = df[["feature", col]].copy()
        out[f"imp_{name}_%"] = 100 * out[col] / out[col].sum()
        return out[["feature", f"imp_{name}_%"]]

    rf_n = normalize(rf, "importance", "RF")
    cb_n = normalize(cb, "importance", "CB")
    xgb_n = normalize(xgb, "importance", "XGB")
    en_n = normalize(en, "abs_coef", "EN")

    imp_all = rf_n.merge(cb_n, on="feature", how="outer") \
                  .merge(xgb_n, on="feature", how="outer") \
                  .merge(en_n, on="feature", how="outer") \
                  .merge(cmp[["feature", "rank_moyen"]], on="feature", how="left")

    # ---------- Stats descriptives + corrélation ----------
    master = pd.read_parquet(MASTER_DIR / "master_dataset_clean.parquet")
    master = master.set_index("IDCHEVAL")
    feat_cols = [c for c in master.columns if c.startswith("f")]
    for c in feat_cols:
        if master[c].dtype == bool:
            master[c] = master[c].astype(int)
    X = master[feat_cols].select_dtypes(include=[np.number])
    y = master["hauteur_max_validee"]
    split = master["SPLIT"]
    X_train = X[split == "train"]
    y_train = y[split == "train"]

    stats_rows = []
    for f in X_train.columns:
        col = X_train[f]
        corr = col.corr(y_train)
        stats_rows.append({
            "feature": f,
            "mean": col.mean(),
            "std": col.std(),
            "nan_pct": 100 * col.isna().mean(),
            "corr_y": corr,
        })
    stats = pd.DataFrame(stats_rows)

    full = imp_all.merge(stats, on="feature", how="left").sort_values("rank_moyen")

    # ---------- Affichage TOP 20 ----------
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

    show(full, "TOP 20 — Importances en % du total + corrélation avec la cible")
    show(full.sort_values("rank_moyen", ascending=False),
         "FLOP 20 — Importances en % du total + corrélation avec la cible")

    full.to_csv(MASTER_DIR / "top_flop_avec_valeurs.csv", index=False)
    print(f"\n→ top_flop_avec_valeurs.csv ({len(full)} features)")


if __name__ == "__main__":
    main()
