"""
produce_figures_rapport.py — Génère toutes les figures pour le rapport final.

Produit 8 figures dans `data/master/figures_rapport/` à partir des résultats déjà
calculés par les scripts du pipeline. Style sobre, fond clair, prêt pour rapport
académique en français.

Figures produites (10) — numérotation alignée sur la trame 15_trame_rapport.md :
  Fig 3.  distribution_cible.png         — histogramme hauteur_max_validee avec tranches métier (§2.3)
  Fig 4.  architecture_hurdle.png        — schéma du modèle Hurdle (§3.5)
  Fig 5.  mae_par_tranche.png            — MAE par tranche × 4 modèles (§4.1)
  Fig 6.  boxplot_robustesse.png         — variabilité MAE ≥1,45m sur 9 splits (§4.3)
  Fig 7.  couverture_conformal.png       — couverture observée vs nominale par tranche (§4.4)
  Fig 8.  modele_gagnant_par_tranche.png — % chevaux gagnés par modèle dans chaque tranche (§4.5)
  Fig 9.  importance_features_top15.png  — top 15 features consensus (§5.1)
  Fig 10. split_temporel.png             — visualisation du time series split (§3.2)
  Fig 11. residus_histogramme.png        — distribution des résidus Hurdle + courbe normale (§4 / §6)
  Fig 12. residus_qqplot.png             — Q-Q plot des résidus vs loi normale (§4 / §6)

Usage :
    python3 scripts/produce_figures_rapport.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR

# ============================================================
# Configuration globale
# ============================================================

FIG_DIR = MASTER_DIR / "figures_rapport"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

# Palette cohérente pour les modèles
COLORS = {
    "RF default": "#3b82f6",          # bleu
    "Hurdle": "#ef4444",                # rouge — modèle recommandé
    "Hurdle (mélange)": "#ef4444",
    "Stacking + Calib": "#10b981",      # vert
    "Poly40 (deg2 + interact.)": "#a855f7",  # violet
    "Poly40": "#a855f7",
}

TRANCHES = ["≤1.10m", "1.15-1.20m", "1.25-1.30m", "1.35-1.40m", "≥1.45m"]


# ============================================================
# Figure 1 : distribution de la cible
# ============================================================

def fig_distribution_cible():
    print("[1/8] Distribution de la cible...")
    df = pd.read_parquet(MASTER_DIR / "master_dataset_epure_v2.parquet")
    y = df["hauteur_max_validee"]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Les hauteurs FFE sont discrètes par paliers de 5cm.
    # On utilise des bins alignés sur ces paliers (centrés sur 0,95 / 1,00 / 1,05...)
    # pour éviter les "trous" entre les barres.
    bins = np.arange(0.925, 1.575 + 0.05, 0.05)
    ax.hist(y, bins=bins, color="#64748b", edgecolor="white", linewidth=0.8)

    # Lignes de coupure des tranches (décalées légèrement à droite pour ne pas
    # tomber sur le bord de barre)
    seuils = [1.125, 1.225, 1.325, 1.425]
    seuils_aff = [1.10, 1.20, 1.30, 1.40]  # valeurs affichées dans les labels
    labels_tranches = ["≤1,10m\n(amateur faible)", "1,15-1,20m", "1,25-1,30m",
                        "1,35-1,40m\n(Pro 2)", "≥1,45m\n(Pro 1)"]
    for s in seuils:
        ax.axvline(s, linestyle="--", color="#dc2626", alpha=0.5, linewidth=1)

    # Annotations des tranches au-dessus
    centers = [1.025, 1.175, 1.275, 1.375, 1.50]
    for c, lab in zip(centers, labels_tranches):
        ax.text(c, ax.get_ylim()[1] * 0.95, lab, ha="center", va="top",
                fontsize=8, color="#475569", style="italic")

    ax.set_xlabel("Hauteur maximale validée (m)")
    ax.set_ylabel("Nombre de chevaux")
    ax.set_title(f"Distribution de la cible — {len(y):,} chevaux de la cohorte de modélisation\n"
                 f"(médiane = {y.median():.2f}m, Q1 = {y.quantile(0.25):.2f}m, Q3 = {y.quantile(0.75):.2f}m)")
    ax.set_xlim(0.90, 1.60)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "distribution_cible.png")
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / 'distribution_cible.png'}")


# ============================================================
# Figure 2 : MAE par tranche × 4 modèles
# ============================================================

def fig_mae_par_tranche():
    print("[2/8] MAE par tranche × 4 modèles...")
    df = pd.read_csv(MASTER_DIR / "recap_avec_poly40_mae.csv")
    modeles = ["RF default", "Hurdle (mélange)", "Stacking + Calib", "Poly40 (deg2 + interact.)"]
    labels = ["RF default", "Hurdle", "Stacking + Calib", "Poly40"]
    sub = df[df["modele"].isin(modeles)].set_index("modele").loc[modeles]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(TRANCHES))
    width = 0.20

    for i, (m, lab) in enumerate(zip(modeles, labels)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, sub.loc[m, TRANCHES].values, width,
               label=lab, color=COLORS.get(m, f"C{i}"))

    ax.set_xticks(x)
    ax.set_xticklabels(TRANCHES)
    ax.set_xlabel("Tranche réelle de hauteur")
    ax.set_ylabel("MAE (cm)")
    ax.set_title("MAE par tranche — comparaison des 4 modèles principaux\n"
                 "Hurdle est meilleur sur les hauts niveaux, Stacking est meilleur sur les bas")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mae_par_tranche.png")
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / 'mae_par_tranche.png'}")


# ============================================================
# Figure 3 : couverture conformal par tranche
# ============================================================

def fig_couverture_conformal():
    print("[3/8] Couverture conformal par tranche...")
    df = pd.read_csv(MASTER_DIR / "conformal_prediction_par_tranche.csv")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    width = 0.27

    ax.bar(x - width, df["cov95_RF"], width, label="RF default", color=COLORS["RF default"])
    ax.bar(x,         df["cov95_Hurdle"], width, label="Hurdle", color=COLORS["Hurdle"])
    ax.bar(x + width, df["cov95_Stack"], width, label="Stacking + Calib", color=COLORS["Stacking + Calib"])

    ax.axhline(95, linestyle="--", color="#64748b", alpha=0.6, label="Niveau nominal 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(df["tranche"])
    ax.set_xlabel("Tranche réelle de hauteur")
    ax.set_ylabel("Couverture IC 95% observée (%)")
    ax.set_title("Couverture empirique des intervalles de confiance (Conformal Prediction)\n"
                 "Hurdle conserve la meilleure couverture sur les chevaux ≥1,45m")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "couverture_conformal.png")
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / 'couverture_conformal.png'}")


# ============================================================
# Figure 4 : importance des features top 15 consensus
# ============================================================

def fig_importance_features():
    print("[4/8] Top 15 features (consensus)...")
    df = pd.read_csv(MASTER_DIR / "top_flop_v2_avec_valeurs.csv")
    top = df.sort_values("rank_moyen").head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(top))

    # Importance moyenne (moyenne des 4 modèles)
    imp_mean = top[["imp_RF_%", "imp_XGB_%", "imp_CB_%", "imp_EN_%"]].mean(axis=1)

    # Couleurs par famille
    families = top["feature"].str.extract(r"^(f\d+)")[0]
    fam_colors = {
        "f1": "#3b82f6", "f2": "#10b981", "f3": "#f59e0b",
        "f5": "#a855f7", "f7": "#ec4899", "f8": "#06b6d4", "f10": "#ef4444"
    }
    colors = [fam_colors.get(f, "#64748b") for f in families]

    ax.barh(y_pos, imp_mean, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"].str.replace("_", " ").str.slice(0, 50))
    ax.invert_yaxis()
    ax.set_xlabel("Importance moyenne (%) sur 4 modèles (RF, XGB, CB, EN)")
    ax.set_title("Top 15 features par consensus — colorées par famille")

    # Légende des familles
    fam_names = {
        "f1": "F1 Activité", "f2": "F2 Gains", "f3": "F3 Performance",
        "f5": "F5 Niveau", "f7": "F7 Cavalier", "f8": "F8 Pedigree", "f10": "F10 Race"
    }
    patches = [mpatches.Patch(color=fam_colors[f], label=fam_names[f])
                for f in fam_colors if f in families.unique()]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "importance_features_top15.png")
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / 'importance_features_top15.png'}")


# ============================================================
# Figure 5 : split temporel par génération
# ============================================================

def fig_split_temporel():
    print("[5/8] Split temporel...")
    df = pd.read_parquet(MASTER_DIR / "master_dataset_epure_v2.parquet")
    df["annee"] = df["DATENAISSANCE"].astype(int)
    counts = df.groupby(["annee", "SPLIT"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    annees = counts.index.values

    train_color = "#3b82f6"
    valid_color = "#f59e0b"
    test_color = "#ef4444"

    ax.bar(annees, counts.get("train", 0), color=train_color, label="Train (2006-2010)")
    ax.bar(annees, counts.get("valid", 0), bottom=counts.get("train", 0),
           color=valid_color, label="Valid (2011-2012)")
    bottom = counts.get("train", 0) + counts.get("valid", 0)
    ax.bar(annees, counts.get("test", 0), bottom=bottom,
           color=test_color, label="Test (2013)")

    ax.set_xlabel("Année de naissance du cheval")
    ax.set_ylabel("Nombre de chevaux")
    ax.set_title("Split temporel par génération — simulation du contexte d'usage réel")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(annees)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "split_temporel.png")
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / 'split_temporel.png'}")


# ============================================================
# Figure 6 : robustesse — variabilité MAE ≥1,45m sur 10 splits
# ============================================================

def fig_boxplot_robustesse():
    print("[6/8] Boxplot robustesse...")
    df = pd.read_csv(MASTER_DIR / "validation_splits_results.csv")
    # Filtrer protocoles A et C (plusieurs runs), exclure B (1 seul run)
    df_multi = df[df["protocole"].isin(["A", "C"])].copy()

    modeles_plot = ["RF default", "Hurdle (mélange)", "Stacking + Calib"]
    fig, ax = plt.subplots(figsize=(10, 5))

    positions = np.arange(len(modeles_plot))
    data_to_plot = [df_multi[df_multi["modele"] == m]["MAE_>=1.45"].dropna().values
                     for m in modeles_plot]
    box = ax.boxplot(data_to_plot, positions=positions, widths=0.5,
                      patch_artist=True, showfliers=True)
    colors_box = [COLORS["RF default"], COLORS["Hurdle"], COLORS["Stacking + Calib"]]
    for patch, c in zip(box["boxes"], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in box["medians"]:
        median.set_color("black")

    # Points individuels (chaque split = 1 point)
    for i, m in enumerate(modeles_plot):
        ys = df_multi[df_multi["modele"] == m]["MAE_>=1.45"].dropna().values
        xs = np.random.normal(i, 0.04, size=len(ys))
        ax.scatter(xs, ys, color="black", s=15, alpha=0.5, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(["RF default", "Hurdle", "Stacking + Calib"])
    ax.set_ylabel("MAE sur ≥1,45m (cm)")
    ax.set_title("Variabilité de la MAE sur les Pro 1 — 9 splits indépendants\n"
                 "(Hurdle plus stable que Stacking malgré sa structure plus complexe)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boxplot_robustesse.png")
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / 'boxplot_robustesse.png'}")


# ============================================================
# Figure 7 : modèle gagnant par tranche
# ============================================================

def fig_modele_gagnant():
    print("[7/8] Modèle gagnant par tranche...")
    # Données déjà calculées dans script 48 (analyse résidus, angle 3)
    # Reproduit à partir de residus_complet.csv
    df = pd.read_csv(MASTER_DIR / "residus_complet.csv")
    df["tranche"] = pd.cut(df["y_true"], bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                            labels=TRANCHES)
    df["best"] = df[["res_RF_cm", "res_Hurdle_cm", "res_Stack_cm"]].abs().idxmin(axis=1)
    df["best"] = df["best"].str.replace("res_", "").str.replace("_cm", "")
    df["best"] = df["best"].map({"RF": "RF default", "Hurdle": "Hurdle",
                                    "Stack": "Stacking + Calib"})

    pct = (pd.crosstab(df["tranche"], df["best"], normalize="index") * 100)
    pct = pct[["Stacking + Calib", "RF default", "Hurdle"]]  # ordre métier

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(pct))
    for col, color in zip(pct.columns, [COLORS["Stacking + Calib"],
                                          COLORS["RF default"], COLORS["Hurdle"]]):
        ax.bar(pct.index.astype(str), pct[col], bottom=bottom, label=col, color=color)
        # Étiquettes pourcentages
        for i, (h, b) in enumerate(zip(pct[col].values, bottom)):
            if h > 8:
                ax.text(i, b + h / 2, f"{h:.0f}%", ha="center", va="center",
                         color="white", fontweight="bold", fontsize=9)
        bottom += pct[col].values

    ax.set_ylabel("% de chevaux où ce modèle est le meilleur")
    ax.set_xlabel("Tranche réelle de hauteur")
    ax.set_title("Modèle gagnant par tranche — division métier nette\n"
                 "Stacking domine les chevaux faibles, Hurdle domine les Pro 1 (98%)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.5))
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "modele_gagnant_par_tranche.png")
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / 'modele_gagnant_par_tranche.png'}")


# ============================================================
# Figure 8 bis : schéma de l'architecture Hurdle
# ============================================================

def fig_architecture_hurdle():
    print("[9/9] Schéma architecture Hurdle...")
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Style des boîtes
    def box(x, y, w, h, label, sub=None, color="#e0e7ff", border="#3b82f6"):
        bb = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                              linewidth=2, edgecolor=border, facecolor=color)
        ax.add_patch(bb)
        ax.text(x + w/2, y + h/2 + (0.2 if sub else 0), label,
                ha="center", va="center", fontsize=10.5, fontweight="bold")
        if sub:
            ax.text(x + w/2, y + h/2 - 0.35, sub,
                    ha="center", va="center", fontsize=8.5, style="italic",
                    color="#475569")

    def arrow(x1, y1, x2, y2, color="#64748b", label=None):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle="-|>", mutation_scale=15,
                              linewidth=1.5, color=color)
        ax.add_patch(a)
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.25, label,
                    ha="center", fontsize=9, color=color, style="italic")

    # Entrée
    box(0.3, 7.5, 2.2, 1.4, "Features X",
        sub="(156 variables)", color="#fef3c7", border="#f59e0b")
    ax.text(1.4, 9.2, "Cheval à prédire", ha="center", fontsize=10,
            color="#475569", style="italic")

    # 3 sous-modèles
    box(3.5, 7.7, 2.7, 1.0, "Classifier RF",
        sub="balanced — P(top)", color="#ddd6fe", border="#a855f7")
    box(3.5, 5.5, 2.7, 1.0, "Régresseur RF",
        sub="conditionnel (tops only)", color="#fecaca", border="#ef4444")
    box(3.5, 3.3, 2.7, 1.0, "Régresseur RF",
        sub="default (tous)", color="#bfdbfe", border="#3b82f6")

    # Flèches X vers sous-modèles
    arrow(2.5, 8.4, 3.5, 8.2)
    arrow(2.5, 8.1, 3.5, 6.0)
    arrow(2.5, 7.8, 3.5, 3.8)

    # Sorties intermédiaires (carrés simples)
    box(7.4, 7.7, 2.0, 1.0, "p", sub="∈ [0, 1]",
        color="#f5f3ff", border="#a855f7")
    box(7.4, 5.5, 2.0, 1.0, "pred_tops", sub="hauteur si top",
        color="#fef2f2", border="#ef4444")
    box(7.4, 3.3, 2.0, 1.0, "pred_default", sub="hauteur si non-top",
        color="#eff6ff", border="#3b82f6")

    # Flèches sous-modèles → sorties
    arrow(6.2, 8.2, 7.4, 8.2)
    arrow(6.2, 6.0, 7.4, 6.0)
    arrow(6.2, 3.8, 7.4, 3.8)

    # Combinaison
    combine_x, combine_y = 10.7, 5.5
    box(combine_x, combine_y, 2.8, 1.7,
        "Combinaison",
        sub="ŷ = p · pred_tops\n+ (1−p) · pred_default",
        color="#dcfce7", border="#10b981")

    # Flèches vers combinaison
    arrow(9.4, 8.2, 10.7, 6.8, color="#a855f7", label="p")
    arrow(9.4, 6.0, 10.7, 6.0, color="#ef4444", label="× p")
    arrow(9.4, 3.8, 10.7, 5.9, color="#3b82f6", label="× (1−p)")

    # Sortie finale
    box(11.0, 1.5, 2.3, 1.3, "ŷ = hauteur prédite",
        sub="+ IC adaptatif 95%", color="#fef3c7", border="#f59e0b")
    arrow(12.1, 5.5, 12.1, 2.8, color="#64748b")

    # Titre
    ax.text(7, 9.7,
            "Architecture du modèle Hurdle — combinaison de 3 sous-modèles spécialisés",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(7, 0.7,
            "Inspiré du Two-Part Model (Cragg 1971, Mullahy 1998), adapté à la régression bornée avec valeurs rares hautes",
            ha="center", fontsize=8.5, color="#64748b", style="italic")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "architecture_hurdle.png")
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / 'architecture_hurdle.png'}")


# ============================================================
# Figure 8a : histogramme des résidus Hurdle (avec courbe normale)
# ============================================================

def fig_residus_histogramme():
    print("[8/10] Histogramme des résidus Hurdle...")
    from scipy import stats

    df = pd.read_csv(MASTER_DIR / "residus_complet.csv")
    res = df["res_Hurdle_cm"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(-50, 60, 2.5)
    ax.hist(res, bins=bins, density=True, color="#64748b",
            edgecolor="white", linewidth=0.5, alpha=0.85,
            label="Résidus observés")
    # Normale théorique (même moyenne et std)
    x = np.linspace(res.min(), res.max(), 200)
    ax.plot(x, stats.norm.pdf(x, res.mean(), res.std()),
            color="#dc2626", linewidth=2, label="Normale théorique")
    ax.axvline(0, color="black", linestyle="--", alpha=0.5,
                label="Résidu nul (modèle parfait)")
    ax.set_xlabel("Résidu signé (cm) — Hurdle")
    ax.set_ylabel("Densité")
    ax.set_title(f"Distribution des résidus du modèle Hurdle\n"
                 f"(moyenne = {res.mean():.2f}cm, écart-type = {res.std():.2f}cm, "
                 f"kurtosis = {res.kurt():.2f})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residus_histogramme.png")
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / 'residus_histogramme.png'}")


# ============================================================
# Figure 8b : Q-Q plot des résidus Hurdle vs loi normale
# ============================================================

def fig_residus_qqplot():
    print("[9/10] Q-Q plot des résidus Hurdle...")
    from scipy import stats

    df = pd.read_csv(MASTER_DIR / "residus_complet.csv")
    res = df["res_Hurdle_cm"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    stats.probplot(res, dist="norm", plot=ax)
    # Personnalisation des lignes du probplot
    ax.get_lines()[0].set_markerfacecolor("#64748b")
    ax.get_lines()[0].set_markeredgecolor("#475569")
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[1].set_color("#dc2626")
    ax.get_lines()[1].set_linewidth(2)
    ax.set_title("Q-Q plot des résidus Hurdle vs loi normale\n"
                 "Test D'Agostino-Pearson : p < 1e-13 → rejet de la normalité")
    ax.set_xlabel("Quantiles théoriques (loi normale)")
    ax.set_ylabel("Quantiles observés (résidus, cm)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residus_qqplot.png")
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / 'residus_qqplot.png'}")


# ============================================================
# Pipeline principal
# ============================================================

def main():
    print("=" * 70)
    print(f"GÉNÉRATION DES FIGURES POUR LE RAPPORT")
    print(f"Sortie : {FIG_DIR}")
    print("=" * 70 + "\n")

    fig_distribution_cible()
    fig_mae_par_tranche()
    fig_couverture_conformal()
    fig_importance_features()
    fig_split_temporel()
    fig_boxplot_robustesse()
    fig_modele_gagnant()
    fig_residus_histogramme()
    fig_residus_qqplot()
    fig_architecture_hurdle()

    print("\n" + "=" * 70)
    print(f"✓ 10 figures générées dans {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
