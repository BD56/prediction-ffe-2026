"""
99 - Merge final du master dataset + split train/valid/test par génération.

Concatène toutes les familles + la cible.

Split time series :
  - Train : chevaux nés 2006-2010
  - Validation : nés 2011-2012
  - Test : nés 2013

Entrée  : data/master/intermediates/*.parquet
Sortie  : data/master/master_dataset_final.parquet (avec colonne SPLIT)
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (load_enriched, INTERMEDIATES_DIR, MASTER_DIR, SPLIT_DIR)


def main():
    print("=== 99 - Merge master dataset + split train/valid/test ===\n")

    # 1. Charger la cible
    cible = pd.read_parquet(INTERMEDIATES_DIR / "cible.parquet")
    cible = cible.set_index("IDCHEVAL")
    print(f"Cible : {len(cible):,} chevaux avec cible")

    # 2. Charger toutes les familles intermédiaires
    family_files = {
        1: "famille1_activite.parquet",
        2: "famille2_gains.parquet",
        3: "famille3_performance.parquet",
        5: "famille5_division.parquet",
        7: "famille7_cavalier.parquet",
        8: "famille8_pedigree.parquet",
        10: "famille10_race.parquet",
    }
    master = None
    for fam_id, fname in family_files.items():
        path = INTERMEDIATES_DIR / fname
        if not path.exists():
            print(f"  ⚠ {fname} introuvable, skip.")
            continue
        f = pd.read_parquet(path).set_index("IDCHEVAL")
        print(f"  Famille {fam_id} : {len(f):,} chevaux × {len(f.columns)} features")
        if master is None:
            master = f
        else:
            master = master.join(f, how="outer")

    # 3. Ajouter la cible
    master = master.join(cible, how="left")
    print(f"\nMaster dataset avant filtre : {len(master):,} chevaux × {len(master.columns)} colonnes")

    # 4. Garder uniquement les chevaux avec cible (= cohorte de modélisation)
    master_with_target = master.dropna(subset=["hauteur_max_validee"])
    print(f"Master avec cible (cohorte modélisation) : {len(master_with_target):,} chevaux")

    # 5. Récupérer DATENAISSANCE pour split
    df = load_enriched(columns=["N° SIRE", "DATENAISSANCE"])
    naissance = df.drop_duplicates("IDCHEVAL").set_index("IDCHEVAL")["DATENAISSANCE"]
    master_with_target["DATENAISSANCE"] = naissance

    # 6. Définir le SPLIT time series
    def assign_split(annee):
        if annee <= 2010:
            return "train"
        elif annee <= 2012:
            return "valid"
        else:  # 2013
            return "test"

    master_with_target["SPLIT"] = master_with_target["DATENAISSANCE"].astype(int).apply(assign_split)

    # 7. Stats du split
    print("\n=== Stats du split time series ===")
    split_stats = master_with_target.groupby("SPLIT").size()
    for split_name, n in split_stats.items():
        pct = 100 * n / len(master_with_target)
        print(f"  {split_name:>7s} : {n:>7,} chevaux ({pct:5.2f}%)")

    print("\n=== Stats cible par split ===")
    print(master_with_target.groupby("SPLIT")["hauteur_max_validee"].describe().round(3).to_string())

    # 8. Sauvegarde
    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MASTER_DIR / "master_dataset_final.parquet"
    master_with_target.reset_index().to_parquet(out_path)
    print(f"\n→ Master dataset final : {out_path}")
    print(f"   {len(master_with_target):,} chevaux × {len(master_with_target.columns)} colonnes")

    # 9. Sauvegarder aussi les IDs par split séparément
    for split_name in ["train", "valid", "test"]:
        ids = master_with_target[master_with_target["SPLIT"] == split_name].index.to_series()
        ids.to_frame("IDCHEVAL").to_parquet(SPLIT_DIR / f"{split_name}_ids.parquet")
    print(f"   IDs split sauvegardés dans : {SPLIT_DIR}")


if __name__ == "__main__":
    main()
