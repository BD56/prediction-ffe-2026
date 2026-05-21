"""
Épuration des features par corrélation (seuil 0.95).

Algorithme :
1. Calcule matrice de corrélation sur le train set uniquement (pas de leakage)
2. Pour chaque paire |corr| > 0.95, choisit laquelle supprimer selon règles :
   - Règle métier : si paire "partants/finishers", garder "partants"
   - Sinon : supprimer celle avec moins de variance (= moins informative)
   - En cas d'égalité : ordre alphabétique
3. Sauvegarde master_dataset_epure.parquet + log des suppressions

Entrée : data/master/master_dataset_final.parquet
Sortie : data/master/master_dataset_epure.parquet + suppressions.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR

SEUIL = 0.95


def main():
    print("=== Épuration features par corrélation (seuil 0.95) ===\n")

    df = pd.read_parquet(MASTER_DIR / "master_dataset_final.parquet")
    df = df.set_index("IDCHEVAL")

    # Travailler sur le TRAIN UNIQUEMENT (pas de leakage)
    train_mask = df["SPLIT"] == "train"
    print(f"Train set : {train_mask.sum():,} chevaux")

    feat_cols = [c for c in df.columns if c.startswith("f")]
    print(f"Features de départ : {len(feat_cols)}")

    # Restreindre aux features numériques
    X_train = df.loc[train_mask, feat_cols].select_dtypes(include=[np.number])
    print(f"Features numériques : {len(X_train.columns)}")

    # Matrice de corrélation absolue
    print("\nCalcul de la matrice de corrélation sur le train...")
    corr = X_train.corr().abs()

    # Calcul de la variance pour départager
    variances = X_train.var()

    # ==========================================
    # Identifier les paires > 0.95
    # ==========================================
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    paires = upper.stack().reset_index()
    paires.columns = ["feature_1", "feature_2", "abs_corr"]
    paires = paires[paires["abs_corr"] > SEUIL].sort_values("abs_corr", ascending=False)
    print(f"\nNb paires |corr| > {SEUIL} : {len(paires):,}")

    # ==========================================
    # Choisir quelle feature supprimer pour chaque paire
    # ==========================================
    to_drop = set()
    log_suppressions = []

    for _, row in paires.iterrows():
        f1, f2 = row["feature_1"], row["feature_2"]
        corr_val = row["abs_corr"]

        # Si l'une est déjà marquée pour suppression, skip
        if f1 in to_drop or f2 in to_drop:
            continue

        # Règle métier 1 : partants/finishers → garder partants
        if "_partants_" in f1 and "_finishers_" in f2:
            drop, keep, reason = f2, f1, "règle partants/finishers"
        elif "_finishers_" in f1 and "_partants_" in f2:
            drop, keep, reason = f1, f2, "règle partants/finishers"
        else:
            # Règle 2 : supprimer celle avec moins de variance
            v1 = variances.get(f1, 0)
            v2 = variances.get(f2, 0)
            if v1 > v2:
                drop, keep, reason = f2, f1, "moins de variance"
            elif v2 > v1:
                drop, keep, reason = f1, f2, "moins de variance"
            else:
                # En cas d'égalité : alphabétique
                if f1 < f2:
                    drop, keep, reason = f2, f1, "alphabétique (variances égales)"
                else:
                    drop, keep, reason = f1, f2, "alphabétique (variances égales)"

        to_drop.add(drop)
        log_suppressions.append({
            "supprimee": drop,
            "gardee": keep,
            "abs_corr": corr_val,
            "raison": reason
        })

    print(f"Features à supprimer : {len(to_drop)}")
    print(f"Features restantes : {len(X_train.columns) - len(to_drop)}")

    # ==========================================
    # Statistiques par règle
    # ==========================================
    log_df = pd.DataFrame(log_suppressions)
    print("\nRépartition des règles :")
    print(log_df["raison"].value_counts().to_string())

    # ==========================================
    # Sauvegarde
    # ==========================================
    # Master épuré
    cols_to_keep = [c for c in df.columns if c not in to_drop]
    df_epure = df[cols_to_keep].reset_index()
    out_path = MASTER_DIR / "master_dataset_epure.parquet"
    df_epure.to_parquet(out_path)
    print(f"\n→ Master épuré : {out_path}")
    print(f"   {len(df_epure):,} chevaux × {len(df_epure.columns)} colonnes")

    # Log des suppressions
    log_path = MASTER_DIR / "correlations" / "suppressions_seuil95.csv"
    log_df.to_csv(log_path, index=False)
    print(f"   Log suppressions : {log_path}")

    # Stats par famille
    print("\n=== Suppression par famille ===")
    for prefix in ["f1", "f2", "f3", "f5", "f7", "f8", "f10"]:
        n_avant = sum(c.startswith(prefix + "_") for c in feat_cols)
        n_apres = sum(c.startswith(prefix + "_") for c in cols_to_keep
                      if c.startswith("f"))
        n_supp = n_avant - n_apres
        if n_avant > 0:
            print(f"  Famille {prefix} : {n_avant} → {n_apres} (-{n_supp})")


if __name__ == "__main__":
    main()
