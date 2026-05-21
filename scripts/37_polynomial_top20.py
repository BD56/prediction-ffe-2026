"""
37 - Régression polynomiale degré 2 (avec interactions) sur top 20 features.

Démarche :
1. Sélection des 20 features avec le meilleur rang_moyen (consensus v2)
2. PolynomialFeatures degré 2 avec interactions → 20 + 20*19/2 + 20 = 230 termes
3. ElasticNetCV pour régulariser (sinon overfit)
4. Comparaison vs baseline ElasticNet sur les 156 features

Sortie : data/master/polynomial_top20_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.linear_model import ElasticNetCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def evaluate(y_true, y_pred):
    return {
        "MAE_cm": mean_absolute_error(y_true, y_pred) * 100,
        "RMSE_cm": np.sqrt(mean_squared_error(y_true, y_pred)) * 100,
        "R2": r2_score(y_true, y_pred),
    }


def mae_par_tranche(y_true, y_pred):
    df = pd.DataFrame({"y": y_true.values, "p": y_pred})
    df["tranche"] = pd.cut(df["y"], bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])
    df["err"] = (df["y"] - df["p"]).abs() * 100
    return df.groupby("tranche", observed=True)["err"].mean().to_dict()


def main():
    print("=== 37 - Polynomial degré 2 (avec interactions) sur top 20 ===\n")

    # ---------- 1. Top 20 features ----------
    imp = pd.read_csv(MASTER_DIR / "top_flop_v2_avec_valeurs.csv")
    top20 = imp.sort_values("rank_moyen").head(20)["feature"].tolist()
    print("Top 20 features sélectionnées (rang moyen consensus v2) :")
    for i, f in enumerate(top20, 1):
        print(f"  {i:>2}. {f}")

    # ---------- 2. Données ----------
    v2 = pd.read_parquet(MASTER_DIR / "master_dataset_epure_v2.parquet")
    v2 = v2.set_index("IDCHEVAL")
    for c in top20:
        if v2[c].dtype == bool:
            v2[c] = v2[c].astype(int)
    X = v2[top20]
    y = v2["hauteur_max_validee"]
    split = v2["SPLIT"]

    X_train, y_train = X[split == "train"], y[split == "train"]
    X_test, y_test = X[split == "test"], y[split == "test"]

    # Imputation médiane + standardisation
    imp_med = SimpleImputer(strategy="median")
    X_train_imp = imp_med.fit_transform(X_train)
    X_test_imp = imp_med.transform(X_test)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train_imp)
    X_test_sc = sc.transform(X_test_imp)

    # ---------- 3. PolynomialFeatures degré 2 avec interactions ----------
    print(f"\n[1/3] Construction features polynomiales degré 2 (avec interactions)...")
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_sc)
    X_test_poly = poly.transform(X_test_sc)
    print(f"  Features originales : {X_train_sc.shape[1]}")
    print(f"  Features polynomiales : {X_train_poly.shape[1]}")
    print(f"  Ratio samples / param : {len(y_train) / X_train_poly.shape[1]:.1f}")

    # ---------- 4. ElasticNet ----------
    print(f"\n[2/3] Fit ElasticNetCV (régularisé)...")
    t0 = time.time()
    en_poly = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=20000,
                            n_jobs=-1, random_state=42)
    en_poly.fit(X_train_poly, y_train)
    pred_poly = en_poly.predict(X_test_poly)
    print(f"  ✓ {time.time()-t0:.1f}s | l1_ratio choisi : {en_poly.l1_ratio_} | "
          f"alpha : {en_poly.alpha_:.5f}")
    print(f"  Coefficients non-nuls : {(en_poly.coef_ != 0).sum()} / {len(en_poly.coef_)}")

    res_poly = evaluate(y_test, pred_poly)
    tr_poly = mae_par_tranche(y_test, pred_poly)

    # ---------- 5. Baseline EN linéaire pour comparaison ----------
    print(f"\n[3/3] Baseline ElasticNet linéaire (sur les mêmes 20 features)...")
    t0 = time.time()
    en_lin = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000,
                          n_jobs=-1, random_state=42)
    en_lin.fit(X_train_sc, y_train)
    pred_lin = en_lin.predict(X_test_sc)
    print(f"  ✓ {time.time()-t0:.1f}s")
    res_lin = evaluate(y_test, pred_lin)
    tr_lin = mae_par_tranche(y_test, pred_lin)

    # ---------- 6. Récap ----------
    # Récupérer baseline EN sur les 156 features de v2 (depuis epuration_v2_comparison)
    ev2 = pd.read_csv(MASTER_DIR / "epuration_v2_comparison.csv")
    en_v2 = ev2[ev2["modele"] == "ElasticNet"].iloc[0]
    res_v2 = {"MAE_cm": en_v2["MAE_v2"], "RMSE_cm": en_v2["RMSE_v2"], "R2": en_v2["R2_v2"]}

    print("\n" + "=" * 80)
    print("COMPARAISON")
    print("=" * 80)
    print(f"{'Modèle':<40s} | {'MAE':>7s} | {'RMSE':>7s} | {'R²':>7s}")
    print("-" * 75)
    rows = []
    for label, res in [
        ("ElasticNet v2 (156 feat. linéaires)", res_v2),
        ("ElasticNet top 20 (linéaire)", res_lin),
        ("ElasticNet top 20 + Poly deg2 + interact.", res_poly),
    ]:
        print(f"{label:<40s} | {res['MAE_cm']:>6.2f}c | {res['RMSE_cm']:>6.2f}c | {res['R2']:>7.4f}")
        rows.append({"modele": label, **res})

    print("\n" + "=" * 100)
    print("MAE PAR TRANCHE (cm)")
    print("=" * 100)
    print(f"{'Modèle':<40s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | "
          f"{'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 105)
    for label, tr in [
        ("ElasticNet top 20 (linéaire)", tr_lin),
        ("ElasticNet top 20 + Poly deg2 + interact.", tr_poly),
    ]:
        print(f"{label:<40s} | {tr.get('≤1.10m', 0):>6.2f}c | "
              f"{tr.get('1.15-1.20m', 0):>9.2f}c | {tr.get('1.25-1.30m', 0):>9.2f}c | "
              f"{tr.get('1.35-1.40m', 0):>9.2f}c | {tr.get('≥1.45m', 0):>6.2f}c")

    pd.DataFrame(rows).to_csv(MASTER_DIR / "polynomial_top20_results.csv", index=False)

    # ---------- 7. Quelles interactions sont gardées ? ----------
    feat_names = poly.get_feature_names_out([f"x{i+1}" for i in range(20)])
    coefs = pd.DataFrame({"term": feat_names, "coef": en_poly.coef_})
    coefs["abs_coef"] = coefs["coef"].abs()
    coefs["type"] = coefs["term"].apply(
        lambda t: "interaction" if " " in t else
                  ("carré" if "^2" in t else "linéaire")
    )
    coefs = coefs[coefs["abs_coef"] > 0].sort_values("abs_coef", ascending=False)

    # Remplacer x1, x2, ... par les vrais noms (regex avec frontière de mot
    # pour éviter que x1 matche x10, x11, etc.)
    import re
    name_map = {f"x{i+1}": top20[i] for i in range(20)}
    def replace_names(t):
        return re.sub(r"x(\d+)", lambda m: name_map.get(m.group(0), m.group(0)), t)
    coefs["term"] = coefs["term"].apply(replace_names)

    print(f"\n{'=' * 80}")
    print(f"TOP 15 TERMES POLYNOMIAUX (par |coef|)")
    print(f"{'=' * 80}")
    print(f"{'Type':<12s} | {'Coef':>8s} | Terme")
    print("-" * 80)
    for _, r in coefs.head(15).iterrows():
        print(f"{r['type']:<12s} | {r['coef']:>+8.4f} | {r['term']}")

    print(f"\nRépartition des termes non-nuls :")
    print(coefs["type"].value_counts().to_string())

    coefs.to_csv(MASTER_DIR / "polynomial_top20_coefs.csv", index=False)


if __name__ == "__main__":
    main()
