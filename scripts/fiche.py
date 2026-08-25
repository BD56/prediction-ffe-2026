"""
Fiche de prédiction à la demande pour un cheval donné.

Usage :
    python scripts/fiche.py <IDCHEVAL>
    python scripts/fiche.py <IDCHEVAL> --output <chemin.png>
    python scripts/fiche.py --retrain       # force le ré-entraînement du Hurdle
    python scripts/fiche.py --list-test 10  # liste 10 IDCHEVAL du test set

Au premier appel : entraîne le Hurdle + calibre LAC + cache sur disque (~1min).
Aux appels suivants : recharge le cache, prédit la fiche en < 1s.
"""

import argparse
import pickle
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent))
from utils import INTERMEDIATES_DIR, MASTER_DIR  # noqa: E402

from produce_fiches_visuelles import COULEUR_FOND_STATUT, COULEUR_STATUT, dessine_fiche  # noqa: E402

CACHE_PATH = INTERMEDIATES_DIR / "hurdle_lac_cache.pkl"
# Branche correction-incertitude : cache enrichi, ecrit A COTE de l'ancien,
# qui n'est jamais ecrase. Voir docs/correction-incertitude.md
CACHE_V2_PATH = INTERMEDIATES_DIR / "hurdle_incertitude_v2_cache.pkl"
OUT_DIR_DEFAULT = MASTER_DIR / "figures_rapport"


def _dataset_path():
    """Renvoie le master dataset disponible. Bascule sur le synthétique
    si le dataset réel (sous accord FFE) n'est pas présent — typique d'un
    clone depuis GitHub où seules les données fictives sont publiées."""
    real = MASTER_DIR / "master_dataset_epure_v2.parquet"
    synth = MASTER_DIR / "master_dataset_synthetic.parquet"
    if real.exists():
        return real
    if synth.exists():
        print(f"  ⚠ master_dataset_epure_v2.parquet absent → bascule sur "
              f"master_dataset_synthetic.parquet (20 chevaux fictifs).")
        return synth
    raise FileNotFoundError(
        "Aucun master dataset trouvé. Lance d'abord scripts/generate_synthetic_data.py "
        "ou place master_dataset_epure_v2.parquet dans data/master/."
    )


def conformal_quantile(values, alpha=0.05):
    n = len(values)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(values, level))


def statut(p_top):
    if p_top >= 0.70:
        return "Pro probable"
    if p_top >= 0.30:
        return "Incertain"
    return "Pro improbable"


def profil_court(p_top):
    if p_top >= 0.70:
        return "Profil orienté pro (≥1,40m)"
    if p_top >= 0.30:
        return "Profil ambigu — à confirmer"
    return "Profil amateur (<1,40m)"


def fit_and_cache():
    """Entraîne Hurdle + calibre LAC + sauvegarde tout dans CACHE_PATH."""
    print("→ Pas de cache — entraînement du Hurdle (~1 min)...")
    t0 = time.time()

    v2 = pd.read_parquet(_dataset_path())
    v2 = v2.set_index("IDCHEVAL")
    feat_cols = [c for c in v2.columns if c.startswith("f")]
    for c in feat_cols:
        if v2[c].dtype == bool:
            v2[c] = v2[c].astype(int)
    X = v2[feat_cols].select_dtypes(include=[np.number])
    feat_cols = list(X.columns)
    y = v2["hauteur_max_validee"]
    split = v2["SPLIT"]

    X_train, y_train = X[split == "train"], y[split == "train"]
    X_valid, y_valid = X[split == "valid"], y[split == "valid"]
    X_test = X[split == "test"]

    imp = SimpleImputer(strategy="median").fit(X_train)
    Xtr = imp.transform(X_train)
    Xva = imp.transform(X_valid)
    Xte = imp.transform(X_test)

    rf = RandomForestRegressor(
        n_estimators=500, max_depth=15, min_samples_leaf=10,
        min_samples_split=10, max_features="sqrt",
        random_state=42, n_jobs=-1,
    ).fit(Xtr, y_train)

    y_bin = (y_train >= 1.40).astype(int)
    clf = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_leaf=10,
        min_samples_split=10, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=-1,
    ).fit(Xtr, y_bin)

    mask_tops = (y_train >= 1.40).values
    rf_tops = RandomForestRegressor(
        n_estimators=500, max_depth=15, min_samples_leaf=5,
        min_samples_split=5, max_features="sqrt",
        random_state=42, n_jobs=-1,
    ).fit(Xtr[mask_tops], y_train[mask_tops])

    # Calibration LAC sur valid
    p_v = clf.predict_proba(Xva)[:, 1]
    pred_v = rf.predict(Xva)
    pred_tops_v = rf_tops.predict(Xva)
    sigma_v = np.stack([e.predict(Xva) for e in rf.estimators_]).std(axis=0)
    sigma_tops_v = np.stack([e.predict(Xva) for e in rf_tops.estimators_]).std(axis=0)
    pred_hu_v = p_v * pred_tops_v + (1 - p_v) * pred_v
    sigma_hu_v = p_v * sigma_tops_v + (1 - p_v) * sigma_v
    residuals_norm = np.abs(y_valid.values - pred_hu_v) / sigma_hu_v
    q_norm = conformal_quantile(residuals_norm, alpha=0.05)

    # Quintiles σ sur le test set (pour la confiance en étoiles)
    p_t = clf.predict_proba(Xte)[:, 1]
    pred_t = rf.predict(Xte)
    pred_tops_t = rf_tops.predict(Xte)
    sigma_t = np.stack([e.predict(Xte) for e in rf.estimators_]).std(axis=0)
    sigma_tops_t = np.stack([e.predict(Xte) for e in rf_tops.estimators_]).std(axis=0)
    sigma_hu_t = p_t * sigma_tops_t + (1 - p_t) * sigma_t
    quintile_bounds = np.quantile(sigma_hu_t, [0.20, 0.40, 0.60, 0.80]).tolist()

    # ------------------------------------------------------------------
    # Correction de l'incertitude (branche correction-incertitude)
    #
    # Le score |y - y_hat| / sigma suppose une erreur PROPORTIONNELLE a sigma.
    # Mesure : la relation est affine avec une ordonnee a l'origine de 3 a 4 cm,
    # sigma n'explique que 1,4 % de la variance de l'erreur, et la couverture
    # decroche a 84,7 % sur le quintile de sigma le plus bas -- celui affiche
    # "5 etoiles". Remplacement du denominateur par une foret predisant
    # |y - y_hat| a partir des familles f1 (activite) et f5 (type de circuit)
    # et de sigma. Aucune variable encodee sur la cible, donc aucune fuite.
    #
    # Le jeu de validation est coupe en deux : moitie A pour ajuster la foret
    # d'erreur, moitie B pour calibrer le quantile ET les quintiles d'etoiles.
    # Les quintiles ne sont donc plus calcules sur le jeu de test, ce qui
    # corrige au passage une fuite du calibrage d'origine.
    # ------------------------------------------------------------------
    err_v = np.abs(y_valid.values - pred_hu_v)
    cols_f1f5 = [i for i, c in enumerate(feat_cols) if c.startswith(("f1_", "f5_"))]
    W_v = np.c_[Xva[:, cols_f1f5], sigma_hu_v]
    perm = np.random.default_rng(42).permutation(len(err_v))
    demi_A, demi_B = perm[: len(perm) // 2], perm[len(perm) // 2 :]
    foret_erreur = RandomForestRegressor(
        n_estimators=250, max_depth=12, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    ).fit(W_v[demi_A], err_v[demi_A])
    den_v = np.clip(foret_erreur.predict(W_v), 1e-4, None)
    q_v2 = conformal_quantile(err_v[demi_B] / den_v[demi_B], alpha=0.05)
    quintile_bounds_v2 = np.quantile(den_v[demi_B], [0.20, 0.40, 0.60, 0.80]).tolist()

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_V2_PATH, "wb") as f:
        pickle.dump({
            "rf": rf, "clf": clf, "rf_tops": rf_tops,
            "imputer": imp, "feat_cols": feat_cols,
            "q_norm": q_norm, "quintile_bounds": quintile_bounds,
            "foret_erreur": foret_erreur, "cols_f1f5": cols_f1f5,
            "q_v2": q_v2, "quintile_bounds_v2": quintile_bounds_v2,
        }, f)
    print(f"  \u2713 cache corrige ecrit : {CACHE_V2_PATH.name}")
    print(f"  q corrige = {q_v2:.3f} | quintiles denominateur (cm) = "
          f"{[round(b_ * 100, 2) for b_ in quintile_bounds_v2]}")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            "rf": rf, "clf": clf, "rf_tops": rf_tops,
            "imputer": imp, "feat_cols": feat_cols,
            "q_norm": q_norm, "quintile_bounds": quintile_bounds,
        }, f)
    print(f"  ✓ cache écrit : {CACHE_PATH.relative_to(CACHE_PATH.parent.parent.parent)}")
    print(f"  q_norm = {q_norm:.3f} | quintiles σ (cm) = "
          f"{[round(b*100, 2) for b in quintile_bounds]}")
    print(f"  durée : {time.time()-t0:.1f}s")


def load_cache(ancien=False):
    """Charge le cache corrige s'il existe, sinon l'ancien.

    ancien=True force l'ancien calibrage (denominateur = sigma), pour comparer.
    """
    chemin = CACHE_PATH if ancien or not CACHE_V2_PATH.exists() else CACHE_V2_PATH
    with open(chemin, "rb") as f:
        return pickle.load(f)


def predict_one(idcheval, cache):
    """Renvoie le dict prêt pour dessine_fiche, pour le cheval demandé."""
    v2 = pd.read_parquet(_dataset_path())
    v2 = v2.set_index("IDCHEVAL")
    if idcheval not in v2.index:
        raise KeyError(f"IDCHEVAL '{idcheval}' absent du master dataset")

    row = v2.loc[[idcheval]]
    feat_cols = cache["feat_cols"]
    for c in feat_cols:
        if c in row.columns and row[c].dtype == bool:
            row[c] = row[c].astype(int)
    x = cache["imputer"].transform(row[feat_cols].select_dtypes(include=[np.number]))

    rf, clf, rf_tops = cache["rf"], cache["clf"], cache["rf_tops"]
    p_top = float(clf.predict_proba(x)[:, 1][0])
    pred_default = float(rf.predict(x)[0])
    pred_tops = float(rf_tops.predict(x)[0])
    pred_hu = p_top * pred_tops + (1 - p_top) * pred_default
    sigma_default = float(np.stack([e.predict(x) for e in rf.estimators_]).std(axis=0)[0])
    sigma_tops = float(np.stack([e.predict(x) for e in rf_tops.estimators_]).std(axis=0)[0])
    sigma_hu = p_top * sigma_tops + (1 - p_top) * sigma_default

    # Denominateur de l'intervalle, selon le cache charge :
    #   cache corrige -> foret d'erreur sur f1+f5+sigma
    #   ancien cache  -> sigma seul (conserve pour comparaison)
    if "foret_erreur" in cache:
        w = np.c_[x[:, cache["cols_f1f5"]], [[sigma_hu]]]
        denominateur = float(np.clip(cache["foret_erreur"].predict(w)[0], 1e-4, None))
        q, qb = cache["q_v2"], cache["quintile_bounds_v2"]
    else:
        denominateur = sigma_hu
        q, qb = cache["q_norm"], cache["quintile_bounds"]

    ic_low = pred_hu - q * denominateur
    ic_high = pred_hu + q * denominateur
    ic_width_cm = (ic_high - ic_low) * 100

    # Confiance en etoiles : quintile du denominateur
    if denominateur <= qb[0]:
        conf = 5
    elif denominateur <= qb[1]:
        conf = 4
    elif denominateur <= qb[2]:
        conf = 3
    elif denominateur <= qb[3]:
        conf = 2
    else:
        conf = 1

    split_val = v2.loc[idcheval, "SPLIT"] if "SPLIT" in v2.columns else None
    y_real = v2.loc[idcheval, "hauteur_max_validee"]
    y_real = None if pd.isna(y_real) else float(y_real)

    fiche = {
        "id": idcheval,
        "profil": profil_court(p_top),
        "statut": statut(p_top),
        "p_top": round(p_top, 2),
        "pred": round(pred_hu, 2),
        "ic_low": round(ic_low, 2),
        "ic_high": round(ic_high, 2),
        "ic_width_cm": round(ic_width_cm, 1),
        "sigma": round(sigma_hu * 100, 2),
        "denominateur_cm": round(denominateur * 100, 2),
        "confiance": conf,
        "split": split_val,
        "reel": y_real,
    }
    if y_real is not None:
        fiche["couvert"] = ic_low <= y_real <= ic_high
        fiche["erreur"] = round(abs(y_real - pred_hu) * 100, 1)
    return fiche


def dessine_fiche_sans_validation(fiche, ax):
    """Variante de dessine_fiche quand la vraie hauteur n'est pas dispo
    (cas production : on prédit pour un cheval sans cible connue)."""
    # On supprime le bloc validation et on remonte le pied de page
    from matplotlib.patches import FancyBboxPatch, Rectangle

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    couleur = COULEUR_STATUT[fiche["statut"]]
    couleur_fond = COULEUR_FOND_STATUT[fiche["statut"]]

    ax.add_patch(FancyBboxPatch((0.15, 0.15), 9.7, 13.7,
                                boxstyle="round,pad=0.05,rounding_size=0.25",
                                linewidth=3.5, edgecolor=couleur, facecolor="white"))
    ax.add_patch(FancyBboxPatch((0.3, 11.9), 9.4, 1.8,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                linewidth=0, facecolor=couleur))
    ax.text(5.0, 13.15, f"Cheval {fiche['id']}",
            ha="center", va="center", fontsize=18, fontweight="bold", color="white")
    ax.text(5.0, 12.35, fiche["profil"],
            ha="center", va="center", fontsize=11, fontstyle="italic", color="white")

    ax.add_patch(FancyBboxPatch((0.6, 9.7), 8.8, 1.85,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=2, edgecolor=couleur, facecolor=couleur_fond))
    ax.text(5.0, 10.95, fiche["statut"].upper(),
            ha="center", va="center", fontsize=22, fontweight="bold", color=couleur)
    ax.text(5.0, 10.05, f"Probabilité TOP (≥1,40m) : {fiche['p_top']:.2f}",
            ha="center", va="center", fontsize=11, color="#333")

    ax.text(2.6, 8.7, "Hauteur prédite", ha="center", va="center", fontsize=10, color="#555")
    ax.text(2.6, 7.85, f"{fiche['pred']:.2f} m",
            ha="center", va="center", fontsize=28, fontweight="bold", color="#222")
    ax.text(7.4, 8.7, "σ local (LAC)", ha="center", va="center", fontsize=10, color="#555")
    ax.text(7.4, 7.85, f"{fiche['sigma']:.2f} cm",
            ha="center", va="center", fontsize=20, fontweight="bold", color="#222")

    ax.text(5.0, 7.15, "Intervalle de confiance 95 % (Locally Adaptive Conformal)",
            ha="center", va="center", fontsize=10, color="#555")
    x_min, x_max = 0.95, 1.60
    bar_left, bar_right = 1.0, 9.0

    def x_to_bar(x):
        return bar_left + (x - x_min) / (x_max - x_min) * (bar_right - bar_left)

    ax.add_patch(Rectangle((bar_left, 6.05), bar_right - bar_left, 0.12,
                           facecolor="#e5e5e5", edgecolor="none"))
    ic_l, ic_r = x_to_bar(fiche["ic_low"]), x_to_bar(fiche["ic_high"])
    ax.add_patch(Rectangle((ic_l, 5.95), ic_r - ic_l, 0.32,
                           facecolor=couleur, alpha=0.45, edgecolor=couleur, linewidth=1.5))
    px = x_to_bar(fiche["pred"])
    ax.plot([px], [6.11], marker="o", markersize=14,
            markerfacecolor=couleur, markeredgecolor="white", markeredgewidth=2, zorder=5)
    ax.text(ic_l, 5.55, f"{fiche['ic_low']:.2f}", ha="center", va="top", fontsize=9, color="#444")
    ax.text(ic_r, 5.55, f"{fiche['ic_high']:.2f}", ha="center", va="top", fontsize=9, color="#444")
    ax.text(5.0, 5.15, f"Largeur IC : {fiche['ic_width_cm']:.1f} cm",
            ha="center", va="center", fontsize=9, fontstyle="italic", color="#666")

    for v in [1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60]:
        bx = x_to_bar(v)
        ax.plot([bx, bx], [6.05, 6.17], color="#999", linewidth=0.8)
        ax.text(bx, 6.62, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#777")
    seuil_x = x_to_bar(1.40)
    ax.plot([seuil_x, seuil_x], [6.17, 6.55], color="#c97800", linewidth=1.2, linestyle="--", alpha=0.85)
    ax.text(seuil_x, 6.78, "seuil TOP", ha="center", va="bottom",
            fontsize=7, color="#c97800", fontstyle="italic")

    ax.text(5.0, 4.45, "Confiance du modèle", ha="center", va="center", fontsize=10, color="#555")
    n = fiche["confiance"]
    for i in range(5):
        ax.text(3.5 + i * 0.75, 3.85, "★", ha="center", va="center",
                fontsize=22, color="#d4af37" if i < n else "#dcdcdc")
    ax.text(5.0, 3.15, f"{n}/5  (quintile σ local)",
            ha="center", va="center", fontsize=9, color="#666")

    ax.plot([0.6, 9.4], [2.5, 2.5], color="#ccc", linewidth=1)
    msg = "Hauteur réelle non disponible (cheval hors test set)"
    if fiche.get("split") is not None:
        msg = f"Cheval issu du split « {fiche['split']} » — hauteur réelle non rapportée"
    ax.text(5.0, 1.7, msg, ha="center", va="center", fontsize=10,
            fontstyle="italic", color="#888")
    ax.text(5.0, 0.45, "Modèle Hurdle RF + Locally Adaptive Conformal",
            ha="center", va="center", fontsize=7, color="#999", fontstyle="italic")


def render_png(fiche, out_path):
    fig, ax = plt.subplots(figsize=(7, 9.8), dpi=200)
    if fiche.get("reel") is not None and fiche.get("split") == "test":
        dessine_fiche(fiche, ax)
    else:
        dessine_fiche_sans_validation(fiche, ax)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fiche de prédiction pour un cheval")
    parser.add_argument("idcheval", nargs="?", help="IDCHEVAL (numéro SIRE)")
    parser.add_argument("--output", "-o", help="Chemin de sortie PNG")
    parser.add_argument("--retrain", action="store_true",
                        help="Force le ré-entraînement du Hurdle")
    parser.add_argument("--list-test", type=int, metavar="N",
                        help="Affiche N IDCHEVAL du test set et quitte")
    parser.add_argument("--no-open", action="store_true",
                        help="Ne pas ouvrir le PNG dans l'aperçu après génération")
    parser.add_argument("--ancien", action="store_true",
                        help="Utiliser l'ancien calibrage (dénominateur = sigma), "
                             "pour comparer avec la version corrigée")
    args = parser.parse_args()

    if args.retrain or not CACHE_PATH.exists() or not CACHE_V2_PATH.exists():
        fit_and_cache()
    cache = load_cache(ancien=args.ancien)

    if args.list_test:
        v2 = pd.read_parquet(_dataset_path())
        ids = v2.loc[v2["SPLIT"] == "test", "IDCHEVAL"].head(args.list_test).tolist()
        for i in ids:
            print(i)
        return

    idcheval = args.idcheval
    if not idcheval:
        # Mode interactif (utile sous Spyder : F5 sans configurer d'argument)
        try:
            idcheval = input("IDCHEVAL (numéro SIRE) > ").strip()
        except EOFError:
            idcheval = ""
        if not idcheval:
            parser.error("IDCHEVAL requis (ou utiliser --list-test / --retrain)")

    fiche = predict_one(idcheval, cache)

    out_path = Path(args.output) if args.output else OUT_DIR_DEFAULT / f"fiche_{fiche['id']}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_png(fiche, out_path)

    print(f"\nCheval {fiche['id']} — split={fiche['split']}")
    print(f"  Statut         : {fiche['statut']}  (P_top = {fiche['p_top']:.2f})")
    print(f"  Prédite        : {fiche['pred']:.2f} m")
    print(f"  IC 95 %        : [{fiche['ic_low']:.2f} ; {fiche['ic_high']:.2f}]  "
          f"(largeur {fiche['ic_width_cm']:.1f} cm)")
    print(f"  σ local        : {fiche['sigma']:.2f} cm")
    print(f"  Dénominateur   : {fiche['denominateur_cm']:.2f} cm"
          f"{'  (= σ, ancien calibrage)' if args.ancien else '  (forêt f1+f5+σ)'}")
    print(f"  Confiance      : {fiche['confiance']}/5")
    if fiche.get("reel") is not None:
        print(f"  Réelle observée: {fiche['reel']:.2f} m  "
              f"({'couvert' if fiche['couvert'] else 'HORS IC'}, "
              f"erreur {fiche['erreur']:.1f} cm)")
    print(f"\n→ PNG écrit : {out_path}")

    if not args.no_open:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(out_path)], check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", str(out_path)], check=False)
            elif sys.platform == "win32":
                subprocess.run(["start", "", str(out_path)], shell=True, check=False)
        except Exception as e:
            print(f"  (impossible d'ouvrir automatiquement : {e})")


if __name__ == "__main__":
    main()
