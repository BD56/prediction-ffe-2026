"""
Génère 5 fiches de prédiction visuelles (PNG) pour les chevaux représentatifs
du rapport, à partir des valeurs déjà calculées dans fiches_prediction_v2.txt.

Sortie : data/master/figures_rapport/fiche_<ID>.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "master" / "figures_rapport"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Données des 5 chevaux (extraites de fiches_prediction_v2.txt)
FICHES = [
    {
        "id": "47237708P",
        "profil": "Crack confirmé — modèle correct",
        "statut": "Pro probable",
        "p_top": 0.94,
        "pred": 1.43,
        "ic_low": 1.33,
        "ic_high": 1.54,
        "ic_width_cm": 22.0,
        "sigma": 2.80,
        "confiance": 5,
        "reel": 1.35,
        "couvert": True,
        "erreur": 8.5,
    },
    {
        "id": "13417805D",
        "profil": "Plafond amateur — modèle correct",
        "statut": "Pro improbable",
        "p_top": 0.00,
        "pred": 1.11,
        "ic_low": 0.99,
        "ic_high": 1.23,
        "ic_width_cm": 24.5,
        "sigma": 3.12,
        "confiance": 5,
        "reel": 1.10,
        "couvert": True,
        "erreur": 1.0,
    },
    {
        "id": "13414483P",
        "profil": "Cas incertain — modèle peu confiant",
        "statut": "Incertain",
        "p_top": 0.40,
        "pred": 1.34,
        "ic_low": 1.12,
        "ic_high": 1.55,
        "ic_width_cm": 43.6,
        "sigma": 5.56,
        "confiance": 1,
        "reel": 1.35,
        "couvert": True,
        "erreur": 1.4,
    },
    {
        "id": "60049124M",
        "profil": "Anomalie statistique — IC dépassé",
        "statut": "Pro probable",
        "p_top": 0.73,
        "pred": 1.39,
        "ic_low": 1.30,
        "ic_high": 1.47,
        "ic_width_cm": 17.3,
        "sigma": 2.20,
        "confiance": 5,
        "reel": 1.05,
        "couvert": False,
        "erreur": 33.6,
    },
    {
        "id": "47261754C",
        "profil": "Vrai TOP ≥1,45m — prédit correctement",
        "statut": "Incertain",
        "p_top": 0.68,
        "pred": 1.38,
        "ic_low": 1.30,
        "ic_high": 1.46,
        "ic_width_cm": 15.3,
        "sigma": 1.95,
        "confiance": 5,
        "reel": 1.45,
        "couvert": True,
        "erreur": 7.1,
    },
]

# Palette par statut
COULEUR_STATUT = {
    "Pro probable": "#1f7a3a",      # vert foncé
    "Pro improbable": "#a8324a",    # rouge bordeaux
    "Incertain": "#c97800",         # orange foncé
}
COULEUR_FOND_STATUT = {
    "Pro probable": "#e6f4ec",
    "Pro improbable": "#fbeaee",
    "Incertain": "#fdf2e1",
}


def dessine_fiche(fiche, ax):
    """Rend une fiche stylisée sur l'axe matplotlib donné."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    couleur = COULEUR_STATUT[fiche["statut"]]
    couleur_fond = COULEUR_FOND_STATUT[fiche["statut"]]

    # Cadre extérieur (couleur statut)
    cadre = FancyBboxPatch(
        (0.15, 0.15), 9.7, 13.7,
        boxstyle="round,pad=0.05,rounding_size=0.25",
        linewidth=3.5, edgecolor=couleur, facecolor="white",
    )
    ax.add_patch(cadre)

    # Bandeau supérieur
    bandeau = FancyBboxPatch(
        (0.3, 11.9), 9.4, 1.8,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=0, facecolor=couleur,
    )
    ax.add_patch(bandeau)

    ax.text(5.0, 13.15, f"Cheval {fiche['id']}",
            ha="center", va="center", fontsize=18, fontweight="bold", color="white")
    ax.text(5.0, 12.35, fiche["profil"],
            ha="center", va="center", fontsize=11, fontstyle="italic", color="white")

    # Bloc statut (gros)
    statut_box = FancyBboxPatch(
        (0.6, 9.7), 8.8, 1.85,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2, edgecolor=couleur, facecolor=couleur_fond,
    )
    ax.add_patch(statut_box)
    ax.text(5.0, 10.95, fiche["statut"].upper(),
            ha="center", va="center", fontsize=22, fontweight="bold", color=couleur)
    ax.text(5.0, 10.05, f"Probabilité TOP (≥1,40m) : {fiche['p_top']:.2f}",
            ha="center", va="center", fontsize=11, color="#333")

    # Hauteur prédite (chiffre central)
    ax.text(2.6, 8.7, "Hauteur prédite", ha="center", va="center",
            fontsize=10, color="#555")
    ax.text(2.6, 7.85, f"{fiche['pred']:.2f} m",
            ha="center", va="center", fontsize=28, fontweight="bold", color="#222")

    # σ local
    ax.text(7.4, 8.7, "σ local (LAC)", ha="center", va="center",
            fontsize=10, color="#555")
    ax.text(7.4, 7.85, f"{fiche['sigma']:.2f} cm",
            ha="center", va="center", fontsize=20, fontweight="bold", color="#222")

    # IC visualisé en barre horizontale
    ax.text(5.0, 7.15, "Intervalle de confiance 95 % (Locally Adaptive Conformal)",
            ha="center", va="center", fontsize=10, color="#555")

    # Échelle 0.95 → 1.60 m
    x_min, x_max = 0.95, 1.60
    bar_left, bar_right = 1.0, 9.0
    def x_to_bar(x):
        return bar_left + (x - x_min) / (x_max - x_min) * (bar_right - bar_left)

    # Rail
    ax.add_patch(Rectangle((bar_left, 6.05), bar_right - bar_left, 0.12,
                           facecolor="#e5e5e5", edgecolor="none"))
    # IC
    ic_l = x_to_bar(fiche["ic_low"])
    ic_r = x_to_bar(fiche["ic_high"])
    ax.add_patch(Rectangle((ic_l, 5.95), ic_r - ic_l, 0.32,
                           facecolor=couleur, alpha=0.45, edgecolor=couleur, linewidth=1.5))
    # Point prédiction
    px = x_to_bar(fiche["pred"])
    ax.plot([px], [6.11], marker="o", markersize=14,
            markerfacecolor=couleur, markeredgecolor="white", markeredgewidth=2, zorder=5)

    # Labels bornes
    ax.text(ic_l, 5.55, f"{fiche['ic_low']:.2f}", ha="center", va="top",
            fontsize=9, color="#444")
    ax.text(ic_r, 5.55, f"{fiche['ic_high']:.2f}", ha="center", va="top",
            fontsize=9, color="#444")
    ax.text(5.0, 5.15, f"Largeur IC : {fiche['ic_width_cm']:.1f} cm",
            ha="center", va="center", fontsize=9, fontstyle="italic", color="#666")

    # Graduations
    for v in [1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60]:
        bx = x_to_bar(v)
        ax.plot([bx, bx], [6.05, 6.17], color="#999", linewidth=0.8)
        ax.text(bx, 6.62, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#777")
    # Marqueur seuil TOP 1.40m (trait au-dessus de la barre, label au-dessus)
    seuil_x = x_to_bar(1.40)
    ax.plot([seuil_x, seuil_x], [6.17, 6.55], color="#c97800",
            linewidth=1.2, linestyle="--", alpha=0.85)
    ax.text(seuil_x, 6.78, "seuil TOP", ha="center", va="bottom",
            fontsize=7, color="#c97800", fontstyle="italic")

    # Étoiles confiance
    ax.text(5.0, 4.45, "Confiance du modèle",
            ha="center", va="center", fontsize=10, color="#555")
    n = fiche["confiance"]
    for i in range(5):
        x_star = 3.5 + i * 0.75
        color = "#d4af37" if i < n else "#dcdcdc"
        ax.text(x_star, 3.85, "★", ha="center", va="center",
                fontsize=22, color=color)
    ax.text(5.0, 3.15, f"{n}/5  (quintile σ local)",
            ha="center", va="center", fontsize=9, color="#666")

    # Séparateur
    ax.plot([0.6, 9.4], [2.7, 2.7], color="#ccc", linewidth=1)

    # Bloc validation
    ax.text(5.0, 2.35, "Validation (hauteur réelle observée)",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#444")

    ax.text(2.3, 1.55, "Réelle",
            ha="center", va="center", fontsize=9, color="#666")
    ax.text(2.3, 1.0, f"{fiche['reel']:.2f} m",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#222")

    couv_color = "#1f7a3a" if fiche["couvert"] else "#a8324a"
    couv_text = "Couvert" if fiche["couvert"] else "HORS IC"
    couv_sym = "✓" if fiche["couvert"] else "✗"
    ax.text(5.0, 1.55, "Couverture IC",
            ha="center", va="center", fontsize=9, color="#666")
    ax.text(5.0, 1.0, f"{couv_sym} {couv_text}",
            ha="center", va="center", fontsize=15, fontweight="bold", color=couv_color)

    ax.text(7.7, 1.55, "Erreur",
            ha="center", va="center", fontsize=9, color="#666")
    ax.text(7.7, 1.0, f"{fiche['erreur']:.1f} cm",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#222")

    # Pied de page
    ax.text(5.0, 0.45, "Modèle Hurdle RF + Locally Adaptive Conformal — cohorte test 2013",
            ha="center", va="center", fontsize=7, color="#999", fontstyle="italic")


def main():
    for fiche in FICHES:
        fig, ax = plt.subplots(figsize=(7, 9.8), dpi=200)
        dessine_fiche(fiche, ax)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        out = OUT_DIR / f"fiche_{fiche['id']}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"écrit : {out}")

    # Bonus : planche A4 contenant les 5 fiches côte à côte
    fig, axes = plt.subplots(2, 3, figsize=(20, 18), dpi=160)
    axes = axes.flatten()
    for i, fiche in enumerate(FICHES):
        dessine_fiche(fiche, axes[i])
    axes[5].axis("off")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                        wspace=0.05, hspace=0.08)
    out = OUT_DIR / "fiches_planche.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"écrit : {out}")


if __name__ == "__main__":
    main()
