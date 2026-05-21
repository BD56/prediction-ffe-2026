"""
Génère un mini-dataset synthétique respectant le schéma de master_dataset_epure_v2,
mais composé exclusivement de chevaux fictifs (IDCHEVAL aléatoires, valeurs simulées).

Permet à n'importe qui de cloner le dépôt GitHub et de faire tourner le pipeline
end-to-end sans accès aux vraies données FFE.

Sortie : data/master/master_dataset_synthetic.parquet (20 chevaux fictifs).
"""

import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR  # noqa: E402

N_CHEVAUX = 20
RANDOM_STATE = 42


def fake_sire(rng):
    """Génère un identifiant SIRE-like (8 caractères, chiffres + 1 lettre finale)."""
    digits = "".join(rng.choice(list(string.digits), size=7))
    suffix = rng.choice(list(string.ascii_uppercase))
    return digits + suffix


def main():
    schema_path = MASTER_DIR / "master_dataset_epure_v2.parquet"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Le dataset de référence est introuvable : {schema_path}\n"
            "Ce script ne sert qu'à générer des données fictives respectant un schéma\n"
            "existant. Si tu publies ce script seul, remplace les statistiques par défaut\n"
            "ci-dessous par des valeurs hardcodées."
        )

    print(f"→ Lecture du schéma depuis {schema_path.name}…")
    ref = pd.read_parquet(schema_path)
    rng = np.random.default_rng(RANDOM_STATE)

    rows = []
    for i in range(N_CHEVAUX):
        row = {}
        # IDCHEVAL fictif et split
        row["IDCHEVAL"] = fake_sire(rng)
        row["SPLIT"] = rng.choice(["train", "valid", "test"], p=[0.66, 0.23, 0.11])

        # Cible : on tire dans la vraie distribution (1.00–1.55 m)
        row["hauteur_max_validee"] = float(np.round(
            rng.normal(loc=1.20, scale=0.12), 2
        ).clip(0.95, 1.55))

        # Features : pour chaque colonne du schéma, tirer une valeur dans le range observé
        for col in ref.columns:
            if col in row:
                continue
            ser = ref[col].dropna()
            if len(ser) == 0:
                row[col] = np.nan
                continue
            if ser.dtype == object or ser.dtype == bool:
                # Tirage uniforme dans les valeurs observées
                row[col] = rng.choice(ser.unique())
            elif np.issubdtype(ser.dtype, np.integer):
                row[col] = int(rng.integers(int(ser.min()), int(ser.max()) + 1))
            else:
                # Numérique : tirage gaussien centré sur la moyenne, clipé sur le range
                mean, std = float(ser.mean()), float(ser.std())
                value = rng.normal(mean, std if std > 0 else 1.0)
                value = float(np.clip(value, float(ser.min()), float(ser.max())))
                row[col] = value

        rows.append(row)

    df = pd.DataFrame(rows, columns=ref.columns.tolist())

    # Types cohérents avec le référentiel
    for col in df.columns:
        try:
            df[col] = df[col].astype(ref[col].dtype)
        except (ValueError, TypeError):
            pass

    out = MASTER_DIR / "master_dataset_synthetic.parquet"
    df.to_parquet(out)
    print(f"écrit : {out} ({len(df)} chevaux fictifs, {len(df.columns)} colonnes)")
    print(f"\nRappel : ces données sont entièrement aléatoires et ne reflètent")
    print(f"aucun cheval réel. Elles servent uniquement à valider l'exécutabilité")
    print(f"du pipeline pour un utilisateur n'ayant pas accès aux données FFE.")


if __name__ == "__main__":
    main()
