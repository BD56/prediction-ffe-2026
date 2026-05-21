"""
27 - Modèle Hurdle (2 étapes pour mieux capturer les tops).

Étape 1 : Classifier RF (cible >= 1.40m ou non)
Étape 2 : Régresseur global RF (notre champion actuel)
Étape 3 : Régresseur "tops" RF (entraîné sur chevaux cible >= 1.40m uniquement)
Prédiction finale : mélange pondéré par la probabilité du classifier.

Évaluation :
- Performance globale (MAE, RMSE, R²)
- Performance sur les tops (≥1.40m)
- Évaluation aussi du classifier seul (accuracy, precision, recall, AUC)

Entrée : data/master/master_dataset_clean.parquet
Sortie : data/master/hurdle_results.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR

SEUIL_HAUT_NIVEAU = 1.40


def evaluate_reg(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:>6s} : MAE={mae*100:.2f}cm, RMSE={rmse*100:.2f}cm, R²={r2:.4f}")
    return {"label": label, "MAE_cm": mae*100, "RMSE_cm": rmse*100, "R2": r2}


def mae_par_tranche(y_test, y_pred):
    df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    df["tranche"] = pd.cut(df["y_true"],
                            bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])
    df["err_abs"] = (df["y_true"] - df["y_pred"]).abs() * 100
    for tranche in df["tranche"].cat.categories:
        sub = df[df["tranche"] == tranche]
        if len(sub) > 0:
            print(f"  {str(tranche):>12s} (n={len(sub):>5,}) : MAE = {sub['err_abs'].mean():>5.2f}cm")


def main():
    print(f"=== 27 - Modèle Hurdle (seuil top = {SEUIL_HAUT_NIVEAU}m) ===\n")

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
    X_test, y_test = X[split == "test"], y[split == "test"]

    # Cible binaire
    y_train_bin = (y_train >= SEUIL_HAUT_NIVEAU).astype(int)
    y_valid_bin = (y_valid >= SEUIL_HAUT_NIVEAU).astype(int)
    y_test_bin = (y_test >= SEUIL_HAUT_NIVEAU).astype(int)
    print(f"Train : {len(X_train):,} ({y_train_bin.sum():,} tops = {100*y_train_bin.mean():.1f}%)")
    print(f"Valid : {len(X_valid):,} ({y_valid_bin.sum():,} tops = {100*y_valid_bin.mean():.1f}%)")
    print(f"Test  : {len(X_test):,} ({y_test_bin.sum():,} tops = {100*y_test_bin.mean():.1f}%)")

    # Imputation médiane (RF ne gère pas NaN nativement)
    print("\nImputation médiane (fit train)...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_test_imp = imputer.transform(X_test)

    # ============================================================
    # ÉTAPE 1 : Classifier (cible >= 1.40m)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"ÉTAPE 1 : Classifier RF -- 'cible >= {SEUIL_HAUT_NIVEAU}m ?'")
    print("=" * 60)
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=500, max_depth=15,
        min_samples_leaf=10, min_samples_split=10,
        max_features="sqrt", class_weight="balanced",  # déséquilibre 90/10
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train_imp, y_train_bin)
    print(f"✓ {time.time()-t0:.1f}s")

    # Évaluation classifier sur test
    p_test = clf.predict_proba(X_test_imp)[:, 1]
    y_test_bin_pred = clf.predict(X_test_imp)

    print(f"\n--- Performance classifier sur test ---")
    print(f"  Accuracy : {accuracy_score(y_test_bin, y_test_bin_pred):.4f}")
    print(f"  Precision : {precision_score(y_test_bin, y_test_bin_pred):.4f}")
    print(f"  Recall : {recall_score(y_test_bin, y_test_bin_pred):.4f}")
    print(f"  F1 : {f1_score(y_test_bin, y_test_bin_pred):.4f}")
    print(f"  ROC-AUC : {roc_auc_score(y_test_bin, p_test):.4f}")
    cm = confusion_matrix(y_test_bin, y_test_bin_pred)
    print(f"  Matrice confusion :")
    print(f"    Vrais Négatifs : {cm[0,0]:>5,}  | Faux Positifs : {cm[0,1]:>5,}")
    print(f"    Faux Négatifs  : {cm[1,0]:>5,}  | Vrais Positifs : {cm[1,1]:>5,}")

    # ============================================================
    # ÉTAPE 2 : Régresseur global (notre RF default)
    # ============================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Régresseur global RF (default)")
    print("=" * 60)
    t0 = time.time()
    reg_global = RandomForestRegressor(
        n_estimators=500, max_depth=15,
        min_samples_leaf=10, min_samples_split=10,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    reg_global.fit(X_train_imp, y_train)
    y_test_pred_global = reg_global.predict(X_test_imp)
    print(f"✓ {time.time()-t0:.1f}s")

    # ============================================================
    # ÉTAPE 3 : Régresseur tops uniquement
    # ============================================================
    print("\n" + "=" * 60)
    print(f"ÉTAPE 3 : Régresseur 'tops' (cible >= {SEUIL_HAUT_NIVEAU}m)")
    print("=" * 60)
    mask_tops = y_train >= SEUIL_HAUT_NIVEAU
    X_train_tops = X_train_imp[mask_tops.values]
    y_train_tops = y_train[mask_tops]
    print(f"Train tops : {len(X_train_tops):,} chevaux (>=  {SEUIL_HAUT_NIVEAU}m)")
    t0 = time.time()
    reg_tops = RandomForestRegressor(
        n_estimators=500, max_depth=15,
        min_samples_leaf=5, min_samples_split=5,  # plus permissif vu petit dataset
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    reg_tops.fit(X_train_tops, y_train_tops)
    y_test_pred_tops = reg_tops.predict(X_test_imp)
    print(f"✓ {time.time()-t0:.1f}s")

    # ============================================================
    # COMBINAISON Hurdle : mélange pondéré
    # ============================================================
    print("\n" + "=" * 60)
    print("COMBINAISON FINALE : p_top * pred_tops + (1-p_top) * pred_global")
    print("=" * 60)
    y_test_pred_hurdle = p_test * y_test_pred_tops + (1 - p_test) * y_test_pred_global

    print("\n--- Performance hurdle sur test ---")
    print(f"  Global (référence) :")
    evaluate_reg(y_test, y_test_pred_global, "test")
    print(f"  Hurdle :")
    results_hurdle = evaluate_reg(y_test, y_test_pred_hurdle, "test")

    print("\n=== MAE par tranche (test) ===")
    print("--- Global (RF default) ---")
    mae_par_tranche(y_test, y_test_pred_global)
    print("\n--- Hurdle ---")
    mae_par_tranche(y_test, y_test_pred_hurdle)

    # ============================================================
    # Variante : argmax strict (au lieu de mélange)
    # ============================================================
    print("\n" + "=" * 60)
    print("VARIANTE : Argmax strict (si p_top > 0.5 -> regr tops, sinon global)")
    print("=" * 60)
    y_test_pred_argmax = np.where(p_test > 0.5, y_test_pred_tops, y_test_pred_global)
    results_argmax = evaluate_reg(y_test, y_test_pred_argmax, "test")
    print("\n=== MAE par tranche (argmax) ===")
    mae_par_tranche(y_test, y_test_pred_argmax)

    # Sauvegarde
    pd.DataFrame([
        {"strategie": "global (RF default)", **evaluate_reg(y_test, y_test_pred_global, "test")},
        {"strategie": "hurdle (mélange pondéré)", **results_hurdle},
        {"strategie": "hurdle (argmax strict)", **results_argmax},
    ]).to_csv(MASTER_DIR / "hurdle_results.csv", index=False)

    # ============================================================
    # Récap
    # ============================================================
    print("\n" + "=" * 70)
    print("RÉCAP FINAL")
    print("=" * 70)
    mae_gl = mean_absolute_error(y_test, y_test_pred_global) * 100
    mae_hu = mean_absolute_error(y_test, y_test_pred_hurdle) * 100
    mae_ar = mean_absolute_error(y_test, y_test_pred_argmax) * 100
    print(f"{'Stratégie':<30s} | {'Test MAE':>10s}")
    print("-" * 45)
    print(f"{'RF default (global)':<30s} | {mae_gl:>9.2f}cm")
    print(f"{'Hurdle (mélange pondéré)':<30s} | {mae_hu:>9.2f}cm")
    print(f"{'Hurdle (argmax strict)':<30s} | {mae_ar:>9.2f}cm")


if __name__ == "__main__":
    main()
