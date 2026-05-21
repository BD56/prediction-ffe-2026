"""
53 - Fiches de prédiction v2 : confiance modèle basée sur quintiles de σ(x).

Refonte de la confiance par rapport à la v1 (script 52) :
  v1 : mapping linéaire arbitraire largeur d'IC → étoiles (bornes 12-60 cm)
  v2 : quintile empirique de σ(x) sur la population test → 5 niveaux
       de "rang d'incertitude" calibrés sur la distribution réelle

σ(x) = écart-type des prédictions entre les 500 arbres du RF (Hurdle).
C'est la grandeur statistique sous-jacente qui module l'IC adaptatif.

Score :
  Q1 σ (top 20% les plus confiants) → 5/5 ●●●●●
  Q2                                → 4/5 ●●●●○
  Q3                                → 3/5 ●●●○○
  Q4                                → 2/5 ●●○○○
  Q5 σ (bottom 20% les moins confiants) → 1/5 ●○○○○

Sortie : data/master/fiches_prediction_v2.txt
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


def conformal_quantile(values, alpha):
    n = len(values)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return np.quantile(values, level)


def statut(p):
    if p >= 0.70:
        return "Pro probable"
    if p >= 0.30:
        return "Incertain"
    return "Pro improbable"


def main():
    print("=== 53 - Fiches de prédiction v2 (confiance via quintiles σ) ===\n")

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
    X_test, y_test = X[split == "test"], y[split == "test"]

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xva = imp.transform(X_valid)
    Xte = imp.transform(X_test)

    # ---------- Fit Hurdle + σ(x) ----------
    print("[1/3] Fit Hurdle + estimation σ(x)...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                 min_samples_split=10, max_features="sqrt",
                                 random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train)
    pred_rf_v = rf.predict(Xva)
    pred_rf_t = rf.predict(Xte)
    sigma_rf_v = np.stack([est.predict(Xva) for est in rf.estimators_]).std(axis=0)
    sigma_rf_t = np.stack([est.predict(Xte) for est in rf.estimators_]).std(axis=0)

    y_train_bin = (y_train >= 1.40).astype(int)
    clf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=10,
                                   min_samples_split=10, max_features="sqrt",
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(Xtr, y_train_bin)
    p_v = clf.predict_proba(Xva)[:, 1]
    p_t = clf.predict_proba(Xte)[:, 1]

    mask = (y_train >= 1.40).values
    rf_tops = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                      min_samples_split=5, max_features="sqrt",
                                      random_state=42, n_jobs=-1)
    rf_tops.fit(Xtr[mask], y_train[mask])
    pred_tops_v = rf_tops.predict(Xva)
    pred_tops_t = rf_tops.predict(Xte)
    sigma_tops_v = np.stack([est.predict(Xva) for est in rf_tops.estimators_]).std(axis=0)
    sigma_tops_t = np.stack([est.predict(Xte) for est in rf_tops.estimators_]).std(axis=0)

    pred_hu_v = p_v * pred_tops_v + (1 - p_v) * pred_rf_v
    pred_hu_t = p_t * pred_tops_t + (1 - p_t) * pred_rf_t
    sigma_hu_v = p_v * sigma_tops_v + (1 - p_v) * sigma_rf_v
    sigma_hu_t = p_t * sigma_tops_t + (1 - p_t) * sigma_rf_t
    print(f"  ✓ {time.time()-t0:.1f}s")

    # ---------- Conformal Adaptatif ----------
    print("\n[2/3] Calcul IC adaptatif + quintiles σ...")
    residuals_norm = np.abs(y_valid.values - pred_hu_v) / sigma_hu_v
    q_norm = conformal_quantile(residuals_norm, 0.05)
    lo_t = pred_hu_t - q_norm * sigma_hu_t
    hi_t = pred_hu_t + q_norm * sigma_hu_t
    width_t = (hi_t - lo_t) * 100

    # Quintiles de σ_hu_t sur le test set
    quintile_bounds = np.quantile(sigma_hu_t, [0.20, 0.40, 0.60, 0.80])
    print(f"  Bornes des quintiles σ (en m) : {quintile_bounds.round(4)}")
    print(f"  Soit en σ × 100 (≈ écart-type cm) : {(quintile_bounds*100).round(2)}")

    def stars_from_sigma(sigma_val):
        """5 étoiles si σ dans le Q1 (plus confiant), 1 si dans Q5."""
        if sigma_val <= quintile_bounds[0]:
            return 5  # Q1 → top 20% confiance
        if sigma_val <= quintile_bounds[1]:
            return 4
        if sigma_val <= quintile_bounds[2]:
            return 3
        if sigma_val <= quintile_bounds[3]:
            return 2
        return 1  # Q5 → bottom 20% confiance

    def render_stars(n):
        return "●" * n + "○" * (5 - n)

    # ---------- Sélection des profils ----------
    print("\n[3/3] Sélection de 6 chevaux représentatifs...")
    yt = y_test.values
    df = pd.DataFrame({
        "y_true": yt,
        "pred": pred_hu_t,
        "p_top": p_t,
        "lo": lo_t,
        "hi": hi_t,
        "width_cm": width_t,
        "sigma": sigma_hu_t,
        "couvert": (yt >= lo_t) & (yt <= hi_t),
    }, index=y_test.index)
    df["confiance"] = df["sigma"].apply(stars_from_sigma)

    profils = {}
    # Si on peut, prendre les mêmes chevaux que script 52 pour comparer
    ids = ["47237708P", "13417805D", "13414483P", "60049124M", "47261754C"]
    nom_profils = [
        "1. Pro probable, modèle confiant, CORRECT",
        "2. Pro improbable, modèle confiant, CORRECT",
        "3. Cas INCERTAIN, modèle peu confiant",
        "4. Pro probable, ÉCHEC modèle (outlier)",
        "5. Vrai TOP ≥1,45m, prédit correctement",
    ]
    for id_, nom in zip(ids, nom_profils):
        if id_ in df.index:
            profils[nom] = df.loc[id_]

    # ---------- Génération des fiches ----------
    output = []
    output.append("RAPPEL — bornes des quintiles σ(x) sur le test set :")
    output.append(f"  Q1 (top 20% confiance)  : σ < {quintile_bounds[0]*100:.2f}cm")
    output.append(f"  Q2                       : σ < {quintile_bounds[1]*100:.2f}cm")
    output.append(f"  Q3                       : σ < {quintile_bounds[2]*100:.2f}cm")
    output.append(f"  Q4                       : σ < {quintile_bounds[3]*100:.2f}cm")
    output.append(f"  Q5 (bottom 20% confiance): σ ≥ {quintile_bounds[3]*100:.2f}cm")
    output.append("")
    for nom, r in profils.items():
        s = "═" * 60
        output.append(s)
        output.append(f"║  Cheval ID {r.name} — Profil : {nom}".ljust(60) + "║")
        output.append(s)
        output.append(f"  STATUT PROBABLE   : {statut(r['p_top']):<20s}  (P_top = {r['p_top']:.2f})")
        output.append(f"  HAUTEUR PRÉDITE   : {r['pred']:.2f} m")
        output.append(f"  INTERVALLE 95%    : [{r['lo']:.2f} m ; {r['hi']:.2f} m]  (largeur {r['width_cm']:.1f} cm)")
        output.append(f"  σ LOCAL           : {r['sigma']*100:.2f} cm")
        output.append(f"  CONFIANCE MODÈLE  : {render_stars(int(r['confiance']))} ({int(r['confiance'])}/5)")
        output.append("")
        output.append(f"  ─── Validation (vraie hauteur réelle 2013) ───")
        output.append(f"  HAUTEUR RÉELLE    : {r['y_true']:.2f} m")
        couvert = "✓ couvert par l'IC" if r["couvert"] else "✗ HORS IC (outlier)"
        output.append(f"  COUVERTURE        : {couvert}")
        err_cm = abs(r["y_true"] - r["pred"]) * 100
        output.append(f"  ERREUR POINTUELLE : {err_cm:.1f} cm")
        output.append(s)
        output.append("")

    rapport = "\n".join(output)
    print("\n" + rapport)

    with open(MASTER_DIR / "fiches_prediction_v2.txt", "w") as f:
        f.write(rapport)
    print(f"\n→ fiches_prediction_v2.txt")


if __name__ == "__main__":
    main()
