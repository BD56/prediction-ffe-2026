"""
52 - Génération de fiches de prédiction pour des chevaux représentatifs.

Pour chaque cheval, on produit :
  - STATUT PROBABLE  : Pro probable / Incertain / Pro improbable (selon P(top))
  - HAUTEUR PRÉDITE  : médiane Hurdle (en mètres)
  - INTERVALLE 95%   : IC adaptatif (Locally Adaptive Conformal)
  - CONFIANCE MODÈLE : 1 à 5 étoiles selon la largeur d'IC
  - VRAIE VALEUR (test 2013, pour validation)

On sélectionne 6 profils représentatifs pour illustrer le rapport :
  1. Pro probable, IC étroit, correct
  2. Pro improbable, IC étroit, correct
  3. Incertain, IC large
  4. Pro probable, IC étroit, ÉCHEC modèle (outlier)
  5. Top niveau (≥1.45m), IC serré
  6. Cas atypique avec peu d'historique

Sortie : data/master/fiches_prediction.txt (rendu lisible)
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


def stars(width_cm, max_w=60.0, min_w=12.0):
    """Convertir largeur d'IC en 5-étoiles (5 = très confiant, 1 = peu confiant)."""
    # Plus l'IC est étroit, plus on a d'étoiles
    norm = 1 - min(max(width_cm - min_w, 0) / (max_w - min_w), 1)
    n_stars = max(1, round(norm * 5))
    return "●" * n_stars + "○" * (5 - n_stars)


def statut(p):
    """Classe selon P(top)."""
    if p >= 0.70:
        return "Pro probable"
    if p >= 0.30:
        return "Incertain"
    return "Pro improbable"


def main():
    print("=== 52 - Fiches de prédiction FFE ===\n")

    # ---------- Charger ----------
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

    # ---------- Fit Hurdle ----------
    print("[1/3] Fit Hurdle...")
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
    print("\n[2/3] Calcul IC adaptatif...")
    residuals_norm = np.abs(y_valid.values - pred_hu_v) / sigma_hu_v
    q_norm = conformal_quantile(residuals_norm, 0.05)
    lo_t = pred_hu_t - q_norm * sigma_hu_t
    hi_t = pred_hu_t + q_norm * sigma_hu_t
    width_t = (hi_t - lo_t) * 100

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
        "couvert": (yt >= lo_t) & (yt <= hi_t),
    }, index=y_test.index)

    profils = {}
    # 1. Pro probable, IC étroit, prédiction correcte
    cand = df[(df["p_top"] > 0.85) & (df["width_cm"] < 30) & (df["couvert"])]
    if len(cand) > 0:
        profils["1. Pro probable, IC étroit, CORRECT"] = cand.sort_values("p_top", ascending=False).iloc[0]

    # 2. Pro improbable (P<0.15), IC étroit, correct (cheval clairement non-Pro)
    cand = df[(df["p_top"] < 0.10) & (df["width_cm"] < 25) & (df["y_true"] < 1.20) & (df["couvert"])]
    if len(cand) > 0:
        profils["2. Pro improbable, IC étroit, CORRECT"] = cand.sort_values("p_top").iloc[0]

    # 3. Cas incertain, IC large
    cand = df[(df["p_top"] > 0.35) & (df["p_top"] < 0.65) & (df["width_cm"] > 35)]
    if len(cand) > 0:
        profils["3. Cas INCERTAIN, IC large"] = cand.sort_values("width_cm", ascending=False).iloc[0]

    # 4. Pro probable mais ÉCHEC (outlier)
    cand = df[(df["p_top"] > 0.50) & (~df["couvert"]) & (df["y_true"] < 1.20)]
    if len(cand) > 0:
        profils["4. Pro probable, ÉCHEC modèle (outlier)"] = cand.sort_values("p_top", ascending=False).iloc[0]

    # 5. Vrai top (≥1,45m), IC serré
    cand = df[(df["y_true"] >= 1.45) & (df["couvert"])]
    if len(cand) > 0:
        profils["5. Vrai TOP ≥1,45m, prédit correctement"] = cand.sort_values("width_cm").iloc[0]

    # 6. Très haut niveau attendu mais cheval modeste
    cand = df[(df["pred"] > 1.30) & (df["y_true"] < 1.10)]
    if len(cand) > 0:
        profils["6. Modèle SUR-ESTIME (faux espoir)"] = cand.sort_values("pred", ascending=False).iloc[0]

    # ---------- Génération des fiches ----------
    output = []
    for nom, r in profils.items():
        s = "═" * 60
        h = f"║  Cheval ID {r.name} — Profil : {nom}".ljust(60) + "║"
        output.append(s)
        output.append(h)
        output.append(s)
        output.append(f"  STATUT PROBABLE   : {statut(r['p_top']):<20s}  (P_top = {r['p_top']:.2f})")
        output.append(f"  HAUTEUR PRÉDITE   : {r['pred']:.2f} m")
        output.append(f"  INTERVALLE 95%    : [{r['lo']:.2f} m ; {r['hi']:.2f} m]  (largeur {r['width_cm']:.1f} cm)")
        output.append(f"  CONFIANCE MODÈLE  : {stars(r['width_cm'])} ({stars(r['width_cm']).count('●')}/5)")
        output.append(f"")
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

    with open(MASTER_DIR / "fiches_prediction.txt", "w") as f:
        f.write(rapport)
    print(f"\n→ fiches_prediction.txt")


if __name__ == "__main__":
    main()
