"""
Test de reproductibilité du script prepare_enriched.py.

Vérifie que les transformations définies dans prepare_enriched.py reproduisent
EXACTEMENT les colonnes du parquet enrichi existant, sans avoir à relancer
tout le pipeline (qui prendrait plusieurs minutes sur 8M lignes).

On charge l'enrichi existant, on recalcule les colonnes avec les fonctions
du script de préparation, et on compare.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare_enriched import (
    extract_hauteur, is_poney_race, is_libelle_poney,
    ENRICHED_PARQUET
)


def main():
    print("=" * 70)
    print("TEST DE REPRODUCTIBILITÉ de prepare_enriched.py")
    print("=" * 70)

    print("\nChargement du parquet enrichi existant...")
    df = pd.read_parquet(ENRICHED_PARQUET)
    print(f"  {len(df):,} lignes × {df.shape[1]} colonnes")

    # ---- 1. Test AGE ----
    print("\n[1/6] Test AGE = ANNEE - DATENAISSANCE...")
    age_recalc = df["ANNEE"] - df["DATENAISSANCE"].astype(float)
    match = ((age_recalc.isna() & df["AGE"].isna())
              | (age_recalc == df["AGE"])).all()
    print(f"  Match : {'✓' if match else '✗'}")

    # ---- 2. Test HAUTEUR ----
    print("\n[2/6] Test HAUTEUR (regex V2 sur CLASSEEPREUVE_LIB)...")
    haut_recalc = df["CLASSEEPREUVE_LIB"].map(extract_hauteur).astype(float)
    match = ((haut_recalc.isna() & df["HAUTEUR"].isna())
              | (haut_recalc == df["HAUTEUR"])).all()
    print(f"  Match : {'✓' if match else '✗'}")
    print(f"  Recalc non-NaN : {haut_recalc.notna().sum():,} | "
          f"Existant non-NaN : {df['HAUTEUR'].notna().sum():,}")

    # ---- 3. Test est_poney_race ----
    print("\n[3/6] Test est_poney_race (regex sur RACECHEVAL)...")
    poney_recalc = df["RACECHEVAL"].map(is_poney_race).astype(bool)
    match = (poney_recalc == df["est_poney_race"]).all()
    print(f"  Match : {'✓' if match else '✗'}")
    print(f"  Recalc True : {poney_recalc.sum():,} | "
          f"Existant True : {df['est_poney_race'].sum():,}")

    # ---- 4. Test est_libelle_poney ----
    print("\n[4/6] Test est_libelle_poney (regex 'Poney' sur libellé)...")
    libpon_recalc = df["CLASSEEPREUVE_LIB"].map(is_libelle_poney).astype(bool)
    match = (libpon_recalc == df["est_libelle_poney"]).all()
    print(f"  Match : {'✓' if match else '✗'}")

    # ---- 5. Test in_cohorte_T1 ----
    print("\n[5/6] Test in_cohorte_T1...")
    coh_recalc = (
        df["DATENAISSANCE"].between(2006, 2013)
        & (df["DISCIPLINE_CODE"] == "SO")
        & ~poney_recalc
    )
    match = (coh_recalc == df["in_cohorte_T1"]).all()
    print(f"  Match : {'✓' if match else '✗'}")
    print(f"  Recalc True : {coh_recalc.sum():,} | "
          f"Existant True : {df['in_cohorte_T1'].sum():,}")

    # ---- 6. Test in_fenetre_4_7 ----
    print("\n[6/6] Test in_fenetre_4_7...")
    fen_recalc = age_recalc.between(4, 7)
    match = (fen_recalc == df["in_fenetre_4_7"]).all()
    print(f"  Match : {'✓' if match else '✗'}")

    print("\n" + "=" * 70)
    print("Test terminé. Si tous les ✓ sont verts, le script prepare_enriched.py")
    print("reproduit fidèlement le parquet enrichi existant.")
    print("=" * 70)


if __name__ == "__main__":
    main()
