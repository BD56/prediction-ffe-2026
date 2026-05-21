"""
prepare_enriched.py — Pipeline de préparation du master dataset enrichi.

Ce script fait la transformation `raw → enriched` (étape upstream du pipeline
de modélisation). Il est à exécuter une fois en amont, puis tous les scripts
00-99 du dossier scripts/ travaillent depuis `data/ffe_2010-2025_enriched.parquet`.

ÉTAPES :
  1. Concatène les 16 CSV bruts (2010-2025) en un DataFrame unique
  2. Sauvegarde `data/ffe_2010-2025_raw.parquet` (concaténation pure)
  3. Calcule les colonnes enrichies :
     - AGE = ANNEE - DATENAISSANCE
     - HAUTEUR : extraite par regex V2 du libellé d'épreuve
     - est_poney_race : flag (regex sur RACECHEVAL)
     - est_libelle_poney : flag (regex sur CLASSEEPREUVE_LIB)
     - in_cohorte_T1 : cohorte de modélisation (né 2006-2013, hors poney race, SO uniquement)
     - in_fenetre_4_7 : fenêtre temporelle features (AGE entre 4 et 7)
  4. Sauvegarde `data/ffe_2010-2025_enriched.parquet`

Référence pour la regex hauteur V2 (correction du 2026-04-28) :
  Pattern `\\b(\\d(?:[.,]\\d{1,2})?)\\s*m\\b`
  Couvre les formats "1 m", "1,35m", "1.35 m", etc.
  Couverture passée de 71,8% à 76,6% après correction (vs regex V1
  qui ne capturait pas les hauteurs au format "1 m" sans décimale).

INPUT : 16 fichiers CSV dans `BD/` (nommés 2010.csv à 2016.csv puis
        ExtraxtionEtudiants2017.csv à 2025.csv)
OUTPUT :
  - data/ffe_2010-2025_raw.parquet     (concaténation, 8,1M lignes × 28 colonnes)
  - data/ffe_2010-2025_enriched.parquet (avec 6 colonnes ajoutées)

Usage :
    python3 scripts/prepare_enriched.py
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional
import time

PROJECT_ROOT = Path(__file__).parent.parent
BD_DIR = PROJECT_ROOT / "BD"
DATA_DIR = PROJECT_ROOT / "data"
RAW_PARQUET = DATA_DIR / "ffe_2010-2025_raw.parquet"
ENRICHED_PARQUET = DATA_DIR / "ffe_2010-2025_enriched.parquet"

# ============================================================
# Regex utilisées
# ============================================================

# Extraction de la hauteur d'obstacle depuis le libellé d'épreuve.
# Capture les formats : "1m", "1 m", "1,35m", "1.35 m", "1 m" entre parenthèses, etc.
HAUTEUR_REGEX = re.compile(r"\b(\d(?:[.,]\d{1,2})?)\s*m\b")

# Identification des races de type poney
# (regex sur RACECHEVAL, insensible à la casse)
RACE_PONEY_REGEX = re.compile(
    r"PONEY|PONY|CONNEMARA|WELSH|HAFLINGER|DARTMOOR|SHETLAND|"
    r"FJORD|NEW FOREST|FELL|LANDAIS",
    re.IGNORECASE,
)

# Identification des libellés d'épreuves poneys (cycles SHF jeunes poneys)
LIBELLE_PONEY_REGEX = re.compile(r"Poney")


# ============================================================
# Fonctions utilitaires
# ============================================================

def extract_hauteur(libelle: Optional[str]) -> Optional[float]:
    """Extrait la hauteur d'obstacle depuis le libellé d'épreuve.

    Retourne float ∈ [0,90 ; 1,55] ou None si pas de match.
    """
    if not isinstance(libelle, str):
        return None
    match = HAUTEUR_REGEX.search(libelle)
    if match:
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:
            return None
    return None


def is_poney_race(race: Optional[str]) -> bool:
    """Détecte si la race appartient aux types poney."""
    if not isinstance(race, str):
        return False
    return bool(RACE_PONEY_REGEX.search(race))


def is_libelle_poney(libelle: Optional[str]) -> bool:
    """Détecte si le libellé concerne une épreuve poney."""
    if not isinstance(libelle, str):
        return False
    return bool(LIBELLE_PONEY_REGEX.search(libelle))


# ============================================================
# Étape 1 — Concaténation des CSV bruts
# ============================================================

def load_and_concat_csv() -> pd.DataFrame:
    """Charge les 16 fichiers CSV de BD/ et les concatène en un DataFrame."""
    # Recherche des fichiers (noms 2010.csv à 2016.csv puis ExtraxtionEtudiantsXXXX.csv)
    csv_files = []
    for year in range(2010, 2026):
        candidates = [
            BD_DIR / f"{year}.csv",
            BD_DIR / f"ExtraxtionEtudiants{year}.csv",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"Aucun fichier trouvé pour {year}. "
                f"Cherché : {[str(c) for c in candidates]}"
            )
        csv_files.append((year, found))

    print(f"  Chargement de {len(csv_files)} fichiers CSV...")
    dfs = []
    for year, path in csv_files:
        t0 = time.time()
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        df["ANNEE"] = year
        dfs.append(df)
        print(f"    {path.name:<40s} {len(df):>9,} lignes ({time.time()-t0:.1f}s)")

    print("  Concaténation...")
    raw = pd.concat(dfs, ignore_index=True)
    print(f"  Total : {len(raw):,} lignes × {raw.shape[1]} colonnes")
    return raw


# ============================================================
# Étape 2 — Enrichissement
# ============================================================

def enrich(raw: pd.DataFrame) -> pd.DataFrame:
    """Calcule les 6 colonnes enrichies."""
    df = raw.copy()

    print("\n  Calcul AGE...")
    df["AGE"] = df["ANNEE"] - df["DATENAISSANCE"].astype(float)

    print("  Extraction HAUTEUR (regex V2)...")
    df["HAUTEUR"] = df["CLASSEEPREUVE_LIB"].map(extract_hauteur).astype(float)
    n_extracted = df["HAUTEUR"].notna().sum()
    print(f"    HAUTEUR extraite sur {n_extracted:,} lignes "
          f"({n_extracted/len(df)*100:.1f}%)")

    print("  Flag est_poney_race...")
    df["est_poney_race"] = df["RACECHEVAL"].map(is_poney_race).astype(bool)
    print(f"    {df['est_poney_race'].sum():,} lignes flag poney race")

    print("  Flag est_libelle_poney...")
    df["est_libelle_poney"] = df["CLASSEEPREUVE_LIB"].map(is_libelle_poney).astype(bool)
    print(f"    {df['est_libelle_poney'].sum():,} lignes flag libellé poney")

    print("  Flag in_cohorte_T1 (né 2006-2013, hors poney, SO)...")
    df["in_cohorte_T1"] = (
        df["DATENAISSANCE"].between(2006, 2013)
        & (df["DISCIPLINE_CODE"] == "SO")
        & ~df["est_poney_race"]
    )
    print(f"    {df['in_cohorte_T1'].sum():,} lignes dans la cohorte T1")

    print("  Flag in_fenetre_4_7 (AGE 4-7)...")
    df["in_fenetre_4_7"] = df["AGE"].between(4, 7)
    print(f"    {df['in_fenetre_4_7'].sum():,} lignes dans la fenêtre 4-7")

    return df


# ============================================================
# Pipeline principal
# ============================================================

def main():
    print("=" * 70)
    print("PREPARE ENRICHED — pipeline raw → enriched")
    print("=" * 70)

    DATA_DIR.mkdir(exist_ok=True)
    t_start = time.time()

    # Étape 1 : concaténation des CSV
    print("\n[Étape 1/3] Concaténation des CSV bruts...")
    raw = load_and_concat_csv()
    print(f"  → Sauvegarde {RAW_PARQUET.name}...")
    raw.to_parquet(RAW_PARQUET, index=False)
    print(f"  ✓ Fichier sauvé ({RAW_PARQUET.stat().st_size / 1024**2:.1f} Mo)")

    # Étape 2 : enrichissement
    print("\n[Étape 2/3] Enrichissement...")
    enriched = enrich(raw)
    del raw  # Libère la mémoire

    # Étape 3 : sauvegarde
    print(f"\n[Étape 3/3] Sauvegarde {ENRICHED_PARQUET.name}...")
    enriched.to_parquet(ENRICHED_PARQUET, index=False)
    print(f"  ✓ Fichier sauvé ({ENRICHED_PARQUET.stat().st_size / 1024**2:.1f} Mo)")

    # Récapitulatif
    print("\n" + "=" * 70)
    print(f"TERMINÉ en {time.time() - t_start:.1f}s")
    print("=" * 70)
    print(f"  raw      : {RAW_PARQUET}")
    print(f"             {len(enriched):,} lignes × 28 colonnes")
    print(f"  enriched : {ENRICHED_PARQUET}")
    print(f"             {len(enriched):,} lignes × {enriched.shape[1]} colonnes")
    print()
    print("  Colonnes ajoutées :")
    for col in ["AGE", "HAUTEUR", "est_poney_race", "est_libelle_poney",
                 "in_cohorte_T1", "in_fenetre_4_7"]:
        if col in enriched.columns:
            if enriched[col].dtype == bool:
                print(f"    {col:<22s} : {enriched[col].sum():>9,} True / "
                      f"{(~enriched[col]).sum():>9,} False")
            else:
                non_na = enriched[col].notna().sum()
                print(f"    {col:<22s} : {non_na:>9,} non-NaN "
                      f"({non_na/len(enriched)*100:.1f}%)")


if __name__ == "__main__":
    main()
