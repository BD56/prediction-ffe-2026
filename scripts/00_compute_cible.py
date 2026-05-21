"""
00 - Calcule la cible `hauteur_max_validee` pour la cohorte de modélisation.

Cible = max(HAUTEUR) où le cheval a participé >= 3 fois sur sa carrière entière
        (toutes années 2010-2025 confondues).

Restreint à la cohorte T1+N1>=10 (52 959 chevaux), avec cible calculable pour
~47 617 chevaux (les autres sont essentiellement des chevaux qui ne courent
qu'en cycles SHF jeunes chevaux où la hauteur n'est pas extractible).

Entrée  : data/ffe_2010-2025_enriched.parquet
Sortie  : data/master/intermediates/cible.parquet (1 ligne par cheval avec cible)
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (compute_cible, get_cohorte_t1_n1_10,
                   INTERMEDIATES_DIR, PROJECT_ROOT)


def main():
    print("=== 00 - Calcul de la cible hauteur_max_validee ===\n")

    # Cohorte de modélisation (52 959 chevaux)
    cohorte = get_cohorte_t1_n1_10(min_participations=10)
    print(f"Cohorte T1+N1>=10 : {len(cohorte):,} chevaux")

    # Cible : hauteur_max_validee (N2=3)
    cible = compute_cible(min_participations_hauteur=3)
    print(f"Chevaux avec cible calculable (toute la base T1) : {len(cible):,}")

    # Restriction à la cohorte
    cible_cohorte = cible[cible.index.isin(cohorte)]
    print(f"Chevaux cohorte ET cible calculable : {len(cible_cohorte):,}")

    # Distribution
    print(f"\nDistribution cible :")
    print(cible_cohorte.describe().to_string())

    # Distribution par tranches
    print(f"\nRépartition par niveau :")
    bins = [0, 1.10, 1.20, 1.30, 1.40, 2.0]
    labels = ["<=1.10m", "1.15-1.20m", "1.25-1.30m", "1.35-1.40m", ">=1.45m"]
    counts = pd.cut(cible_cohorte, bins=bins, labels=labels, right=True).value_counts().sort_index()
    for label, count in counts.items():
        print(f"  {str(label):<12s} : {count:>6,} ({100*count/len(cible_cohorte):.2f}%)")

    # Sauvegarde
    INTERMEDIATES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERMEDIATES_DIR / "cible.parquet"
    cible_cohorte.reset_index().to_parquet(out_path)
    print(f"\n→ Sauvegardé : {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
