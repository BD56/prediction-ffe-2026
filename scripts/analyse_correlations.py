"""
Analyse exploratoire des corrélations entre features du master dataset.

Génère :
- Heatmap globale (toutes features, vue d'ensemble)
- Heatmaps par famille (lisibles)
- Liste des paires fortement corrélées (> 0.9 et > 0.95)

Outputs sauvegardés dans data/master/correlations/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR


def main():
    print("=== Analyse des corrélations entre features ===\n")

    # Chargement
    df = pd.read_parquet(MASTER_DIR / "master_dataset_final.parquet")
    df = df.set_index("IDCHEVAL")

    # Colonnes features uniquement (préfixées f1_, f2_, ...)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    print(f"Nb features : {len(feat_cols)}")

    # Garder uniquement les features numériques (pas les booléens en string etc.)
    X = df[feat_cols].select_dtypes(include=[np.number])
    print(f"Nb features numériques : {len(X.columns)}")

    # Sortie
    out_dir = MASTER_DIR / "correlations"
    out_dir.mkdir(exist_ok=True)

    # ========== 1. Matrice de corrélation globale ==========
    print("\nCalcul matrice de corrélation (peut prendre ~1 min)...")
    corr = X.corr().abs()  # valeur absolue pour repérer les forts liens

    # Heatmap globale (sans labels, juste pattern visuel)
    print("Heatmap globale...")
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(corr, cmap="RdBu_r", center=0.5, vmin=0, vmax=1,
                xticklabels=False, yticklabels=False, ax=ax,
                cbar_kws={"label": "|corrélation|"})
    ax.set_title(f"Corrélations entre toutes les features ({len(X.columns)})", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_globale.png", dpi=100)
    plt.close()
    print(f"  → {out_dir / 'heatmap_globale.png'}")

    # ========== 2. Heatmap par famille ==========
    print("\nHeatmaps par famille...")
    familles = {
        "f1": "Activité",
        "f2": "Gains",
        "f3": "Performance",
        "f5": "Division",
        "f7": "Cavalier",
        "f8": "Pedigree",
        "f10": "Race",
    }
    for prefix, name in familles.items():
        # Tenir compte du double caractère pour f10 vs f1
        cols_fam = [c for c in X.columns if c.startswith(prefix + "_")]
        if not cols_fam:
            continue
        corr_fam = X[cols_fam].corr().abs()
        # Adapter taille selon nb features
        n = len(cols_fam)
        size = max(8, min(24, n * 0.3))
        fig, ax = plt.subplots(figsize=(size, size * 0.9))
        # Afficher labels si peu de features
        labels = cols_fam if n <= 30 else False
        sns.heatmap(corr_fam, cmap="RdBu_r", center=0.5, vmin=0, vmax=1,
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    cbar_kws={"label": "|corrélation|"})
        ax.set_title(f"Famille {prefix} - {name} ({n} features)", fontsize=12)
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / f"heatmap_{prefix}_{name.lower()}.png", dpi=100)
        plt.close()
        print(f"  → famille {prefix} ({n} features)")

    # ========== 3. Liste des paires fortement corrélées ==========
    print("\nPaires fortement corrélées :")
    # Prendre la moitié supérieure de la matrice (sans la diagonale)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    paires = upper.stack().reset_index()
    paires.columns = ["feature_1", "feature_2", "abs_corr"]
    paires = paires.sort_values("abs_corr", ascending=False)

    for seuil in [0.99, 0.95, 0.90, 0.80]:
        n = (paires["abs_corr"] > seuil).sum()
        print(f"  Paires avec |corr| > {seuil} : {n:,}")

    # Sauvegarde top paires
    top_paires = paires[paires["abs_corr"] > 0.80].head(200)
    top_paires.to_csv(out_dir / "top_paires_correlees.csv", index=False)
    print(f"\n  → Top paires sauvegardées : {out_dir / 'top_paires_correlees.csv'}")

    # Affichage top 30
    print("\nTop 30 paires fortement corrélées :")
    print(top_paires.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
