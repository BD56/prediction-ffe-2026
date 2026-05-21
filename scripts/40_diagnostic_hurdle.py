"""
40 - Diagnostic Hurdle : pourquoi est-il si bon sur les tops ?

Quatre analyses :
1. Distribution des prédictions RF default vs Hurdle (par tranche)
2. Métriques du classifier interne (≥1,40m vs <1,40m)
3. Décomposition : pour les tops, part venant du régresseur conditionnel
4. Importance du régresseur conditionnel vs RF default

Sortie : data/master/diagnostic_hurdle_*.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix)

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def main():
    print("=== 40 - Diagnostic Hurdle ===\n")

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

    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp = imp.transform(X_test)

    feat_names = X.columns.tolist()

    # ============================================================
    # ÉTAPE 1 : Refit RF default + Hurdle (classifier + reg conditionnel)
    # ============================================================
    print("[1/5] RF default (référence)...")
    t0 = time.time()
    rf_def = RandomForestRegressor(n_estimators=500, max_depth=15,
                                     min_samples_leaf=10, min_samples_split=10,
                                     max_features="sqrt", random_state=42, n_jobs=-1)
    rf_def.fit(X_train_imp, y_train)
    pred_rf = rf_def.predict(X_test_imp)
    imp_rf_def = pd.DataFrame({"feature": feat_names,
                                "importance": rf_def.feature_importances_})
    print(f"  ✓ {time.time()-t0:.1f}s")

    print("[2/5] Classifier ≥1,40m...")
    t0 = time.time()
    y_train_bin = (y_train >= 1.40).astype(int)
    y_test_bin = (y_test >= 1.40).astype(int)
    clf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                   min_samples_leaf=10, min_samples_split=10,
                                   max_features="sqrt", class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    clf.fit(X_train_imp, y_train_bin)
    p_test = clf.predict_proba(X_test_imp)[:, 1]
    p_pred_bin = (p_test >= 0.5).astype(int)
    print(f"  ✓ {time.time()-t0:.1f}s")
    print(f"  Taux de tops réels (train) : {y_train_bin.mean()*100:.1f}%")
    print(f"  Taux de tops réels (test)  : {y_test_bin.mean()*100:.1f}%")

    auc = roc_auc_score(y_test_bin, p_test)
    prec = precision_score(y_test_bin, p_pred_bin)
    rec = recall_score(y_test_bin, p_pred_bin)
    f1 = f1_score(y_test_bin, p_pred_bin)
    cm = confusion_matrix(y_test_bin, p_pred_bin)
    print(f"\n  CLASSIFIER (seuil 0.5) :")
    print(f"    AUC ROC   : {auc:.4f}")
    print(f"    Précision : {prec:.4f}")
    print(f"    Rappel    : {rec:.4f}")
    print(f"    F1        : {f1:.4f}")
    print(f"    Matrice de confusion :")
    print(f"      VN={cm[0,0]:>5d}  FP={cm[0,1]:>4d}")
    print(f"      FN={cm[1,0]:>5d}  VP={cm[1,1]:>4d}")

    print("\n[3/5] Régresseur conditionnel (entraîné sur y >= 1,40m)...")
    t0 = time.time()
    mask = (y_train >= 1.40).values
    print(f"  Taille train tops : {mask.sum():,} chevaux (sur {len(y_train):,})")
    print(f"  Moyenne y_train tops : {y_train[mask].mean():.3f}m (vs {y_train.mean():.3f}m global)")
    rf_tops = RandomForestRegressor(n_estimators=500, max_depth=15,
                                      min_samples_leaf=5, min_samples_split=5,
                                      max_features="sqrt", random_state=42, n_jobs=-1)
    rf_tops.fit(X_train_imp[mask], y_train[mask])
    pred_tops = rf_tops.predict(X_test_imp)
    imp_rf_tops = pd.DataFrame({"feature": feat_names,
                                 "importance": rf_tops.feature_importances_})
    print(f"  ✓ {time.time()-t0:.1f}s")

    pred_hurdle = p_test * pred_tops + (1 - p_test) * pred_rf

    # ============================================================
    # ÉTAPE 2 : Distributions des prédictions
    # ============================================================
    print("\n[4/5] Distributions des prédictions par tranche réelle...\n")
    df = pd.DataFrame({"y_true": y_test.values, "rf": pred_rf,
                        "tops": pred_tops, "hurdle": pred_hurdle, "p": p_test})
    df["tranche"] = pd.cut(df["y_true"],
                            bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                    "1.35-1.40m", "≥1.45m"])

    print("=" * 100)
    print("STATS DES PRÉDICTIONS PAR TRANCHE RÉELLE")
    print("=" * 100)
    print(f"{'Tranche':<12s} | {'n':>5s} | {'y_moy':>6s} | "
          f"{'pred RF moy':>11s} | {'pred Hurdle moy':>15s} | "
          f"{'pred Tops moy':>13s} | {'p_moy':>5s}")
    print("-" * 90)
    rows = []
    for tr in df["tranche"].cat.categories:
        sub = df[df["tranche"] == tr]
        if len(sub) > 0:
            r = {"tranche": str(tr), "n": len(sub),
                 "y_moy": sub["y_true"].mean(),
                 "rf_moy": sub["rf"].mean(),
                 "hurdle_moy": sub["hurdle"].mean(),
                 "tops_moy": sub["tops"].mean(),
                 "p_moy": sub["p"].mean()}
            rows.append(r)
            print(f"{str(tr):<12s} | {r['n']:>5d} | {r['y_moy']:>6.3f} | "
                  f"{r['rf_moy']:>11.3f} | {r['hurdle_moy']:>15.3f} | "
                  f"{r['tops_moy']:>13.3f} | {r['p_moy']:>5.2f}")
    pd.DataFrame(rows).to_csv(MASTER_DIR / "diagnostic_hurdle_distributions.csv", index=False)

    # ============================================================
    # ÉTAPE 3 : Décomposition pour les tops réels
    # ============================================================
    print("\n" + "=" * 90)
    print("DÉCOMPOSITION POUR LES VRAIS TOPS (y_true >= 1,40m)")
    print("=" * 90)
    tops_mask = df["y_true"] >= 1.40
    sub = df[tops_mask]
    print(f"N tops réels dans test : {len(sub):,}")
    print(f"y_true moyen : {sub['y_true'].mean():.3f}m")
    print(f"")
    print(f"  Prédiction Hurdle moyenne : {sub['hurdle'].mean():.3f}m")
    print(f"    - dont contribution 'p × pred_tops'   : {(sub['p']*sub['tops']).mean():.3f}m "
          f"(poids moy p={sub['p'].mean():.2f})")
    print(f"    - dont contribution '(1-p) × pred_rf' : {((1-sub['p'])*sub['rf']).mean():.3f}m "
          f"(poids moy 1-p={1-sub['p'].mean():.2f})")
    print(f"")
    print(f"  Erreur signée moyenne :")
    print(f"    RF default : {(sub['rf']-sub['y_true']).mean()*100:>+6.2f} cm (biais négatif = sous-estimation)")
    print(f"    Hurdle     : {(sub['hurdle']-sub['y_true']).mean()*100:>+6.2f} cm")

    # ≥1,45m subset
    elite_mask = df["y_true"] >= 1.45
    sub_e = df[elite_mask]
    if len(sub_e) > 0:
        print(f"\n  Sous-ensemble ≥1,45m (n={len(sub_e):,}) :")
        print(f"    y_true moyen : {sub_e['y_true'].mean():.3f}m")
        print(f"    RF default moy : {sub_e['rf'].mean():.3f}m  (biais : {(sub_e['rf']-sub_e['y_true']).mean()*100:+.2f} cm)")
        print(f"    Hurdle moy     : {sub_e['hurdle'].mean():.3f}m  (biais : {(sub_e['hurdle']-sub_e['y_true']).mean()*100:+.2f} cm)")
        print(f"    p moyen        : {sub_e['p'].mean():.3f}")

    # ============================================================
    # ÉTAPE 4 : Comparaison des importances RF default vs RF tops
    # ============================================================
    print("\n[5/5] Top 15 features — RF default vs RF tops")
    print("=" * 110)
    imp_rf_def["rang_def"] = imp_rf_def["importance"].rank(ascending=False, method="min").astype(int)
    imp_rf_tops["rang_tops"] = imp_rf_tops["importance"].rank(ascending=False, method="min").astype(int)
    mg = imp_rf_def[["feature", "rang_def", "importance"]].merge(
        imp_rf_tops[["feature", "rang_tops", "importance"]],
        on="feature", suffixes=("_def", "_tops"))
    mg["delta_rang"] = mg["rang_def"] - mg["rang_tops"]  # >0 : monte chez tops
    mg = mg.sort_values("rang_tops")

    print(f"\n— TOP 15 selon RF tops (importance dans le régresseur conditionnel) —")
    print(f"{'#':>3} | {'Feature':<45s} | {'rang def':>8s} | {'rang tops':>9s} | {'Δ':>5s}")
    print("-" * 85)
    for i, (_, r) in enumerate(mg.head(15).iterrows(), 1):
        print(f"{i:>3} | {r['feature']:<45s} | {r['rang_def']:>8d} | "
              f"{r['rang_tops']:>9d} | {r['delta_rang']:>+5d}")

    print(f"\n— Features qui MONTENT le plus chez les tops (devenues importantes pour l'élite) —")
    top_climb = mg.sort_values("delta_rang", ascending=False).head(10)
    print(f"{'Feature':<45s} | {'rang def':>8s} | {'rang tops':>9s} | {'Δ':>5s}")
    print("-" * 85)
    for _, r in top_climb.iterrows():
        print(f"{r['feature']:<45s} | {r['rang_def']:>8d} | {r['rang_tops']:>9d} | {r['delta_rang']:>+5d}")

    mg.to_csv(MASTER_DIR / "diagnostic_hurdle_importances.csv", index=False)
    print("\n→ diagnostic_hurdle_distributions.csv, diagnostic_hurdle_importances.csv")


if __name__ == "__main__":
    main()
