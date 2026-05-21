"""
28 - Random Forest avec sample weights (pondération des tops).

L'objectif : forcer le modèle à mieux apprendre les chevaux de haut niveau
en leur donnant plus de poids pendant l'entraînement.

Plusieurs stratégies de poids testées :
  A. Poids linéaire : weight = cible
  B. Poids quadratique : weight = cible² (accentue plus les hauts)
  C. Poids inverse fréquence : pondère selon la rareté de la tranche
  D. Poids binaire : tops (>= 1.40m) ont poids X, autres poids 1

Entrée : data/master/master_dataset_clean.parquet
Sortie : data/master/sample_weights_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def evaluate(y_true, y_pred, label="test"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"label": label, "MAE_cm": mae*100, "RMSE_cm": rmse*100, "R2": r2}


def mae_par_tranche(y_test, y_pred):
    df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    df["tranche"] = pd.cut(df["y_true"],
                            bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])
    df["err_abs"] = (df["y_true"] - df["y_pred"]).abs() * 100
    result = {}
    for tranche in df["tranche"].cat.categories:
        sub = df[df["tranche"] == tranche]
        if len(sub) > 0:
            result[str(tranche)] = sub["err_abs"].mean()
    return result


def train_eval(X_train, y_train, sample_weight, X_test, y_test, label):
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=15,
        min_samples_leaf=10, min_samples_split=10,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = rf.predict(X_test)
    res = evaluate(y_test, y_pred, label)
    tranches = mae_par_tranche(y_test, y_pred)
    return res, tranches, y_pred


def main():
    print("=== 28 - RF avec sample weights ===\n")

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
    X_test, y_test = X[split == "test"], y[split == "test"]

    # Imputation
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # ============================================================
    # Stratégie 0 : référence (sans poids)
    # ============================================================
    print("Stratégie 0 : RF default (sans poids, référence)")
    t0 = time.time()
    res0, tr0, _ = train_eval(X_train_imp, y_train, None,
                                X_test_imp, y_test, "default")
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={res0['MAE_cm']:.2f}cm, R²={res0['R2']:.4f}")

    # ============================================================
    # Stratégie A : poids linéaire (weight = cible)
    # ============================================================
    print("\nStratégie A : poids linéaire (weight = cible)")
    w_A = y_train.values
    t0 = time.time()
    resA, trA, _ = train_eval(X_train_imp, y_train, w_A,
                                X_test_imp, y_test, "linéaire")
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={resA['MAE_cm']:.2f}cm, R²={resA['R2']:.4f}")

    # ============================================================
    # Stratégie B : poids quadratique (weight = cible²)
    # ============================================================
    print("\nStratégie B : poids quadratique (weight = cible²)")
    w_B = y_train.values ** 2
    t0 = time.time()
    resB, trB, _ = train_eval(X_train_imp, y_train, w_B,
                                X_test_imp, y_test, "quadratique")
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={resB['MAE_cm']:.2f}cm, R²={resB['R2']:.4f}")

    # ============================================================
    # Stratégie C : poids inverse fréquence (par tranche)
    # ============================================================
    print("\nStratégie C : poids inverse fréquence (par tranche)")
    tranches_train = pd.cut(y_train,
                             bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                             labels=[0, 1, 2, 3, 4]).astype(int)
    freq = tranches_train.value_counts(normalize=True).to_dict()
    w_C = tranches_train.map(lambda t: 1.0 / freq[t]).values
    # Normaliser pour avoir poids moyen = 1
    w_C = w_C / w_C.mean()
    t0 = time.time()
    resC, trC, _ = train_eval(X_train_imp, y_train, w_C,
                                X_test_imp, y_test, "inv_freq")
    print(f"  ✓ {time.time()-t0:.1f}s | MAE={resC['MAE_cm']:.2f}cm, R²={resC['R2']:.4f}")

    # ============================================================
    # Stratégie D : poids binaire (tops vs autres)
    # ============================================================
    for poids_top in [3, 5, 10]:
        print(f"\nStratégie D : poids binaire (tops >= 1.40m -> poids {poids_top})")
        w_D = np.where(y_train >= 1.40, poids_top, 1.0)
        t0 = time.time()
        resD, trD, _ = train_eval(X_train_imp, y_train, w_D,
                                    X_test_imp, y_test, f"binaire_x{poids_top}")
        print(f"  ✓ {time.time()-t0:.1f}s | MAE={resD['MAE_cm']:.2f}cm, R²={resD['R2']:.4f}")
        if poids_top == 5:
            res_D5, tr_D5 = resD, trD  # garder pour comparer

    # ============================================================
    # Récap comparatif
    # ============================================================
    print("\n" + "=" * 80)
    print("RÉCAP COMPARATIF (MAE par tranche, test)")
    print("=" * 80)
    print(f"{'Stratégie':<22s} | {'Global':>7s} | {'≤1.10m':>7s} | {'1.15-1.20m':>10s} | {'1.25-1.30m':>10s} | {'1.35-1.40m':>10s} | {'≥1.45m':>7s}")
    print("-" * 100)
    for label, res, tr in [
        ("default", res0, tr0),
        ("linéaire (A)", resA, trA),
        ("quadratique (B)", resB, trB),
        ("inv_freq (C)", resC, trC),
        ("binaire ×5 (D)", res_D5, tr_D5),
    ]:
        print(f"{label:<22s} | {res['MAE_cm']:>6.2f}cm | "
              f"{tr.get('≤1.10m', 0):>6.2f}cm | "
              f"{tr.get('1.15-1.20m', 0):>9.2f}cm | "
              f"{tr.get('1.25-1.30m', 0):>9.2f}cm | "
              f"{tr.get('1.35-1.40m', 0):>9.2f}cm | "
              f"{tr.get('≥1.45m', 0):>6.2f}cm")

    # Sauvegarde
    pd.DataFrame([res0, resA, resB, resC, res_D5]).to_csv(
        MASTER_DIR / "sample_weights_results.csv", index=False)


if __name__ == "__main__":
    main()
