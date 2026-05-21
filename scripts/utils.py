"""
Fonctions communes pour la construction du master dataset.

Utilisé par tous les scripts build_familleN.py.
"""

from pathlib import Path
import pandas as pd

# Chemins du projet
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MASTER_DIR = DATA_DIR / "master"
INTERMEDIATES_DIR = MASTER_DIR / "intermediates"
SPLIT_DIR = MASTER_DIR / "split"

ENRICHED_PARQUET = DATA_DIR / "ffe_2010-2025_enriched.parquet"


def load_enriched(columns=None):
    """Charge le dataset enrichi (8M lignes, 34 colonnes par défaut).

    Args:
        columns: liste de colonnes à charger (None = toutes).
            Recommandé pour ne pas saturer la mémoire.

    Returns:
        pd.DataFrame avec IDCHEVAL renommé depuis 'N° SIRE'.
    """
    df = pd.read_parquet(ENRICHED_PARQUET, columns=columns)
    if "N° SIRE" in df.columns:
        df = df.rename(columns={"N° SIRE": "IDCHEVAL"})
    return df


def get_cohorte_t1_n1_10(min_participations=10):
    """Renvoie la liste des IDCHEVAL de la cohorte T1+N1>=10.

    Cohorte T1 = né 2006-2013, hors poney (race et libellé), discipline SO.
    Filtre N1 = au moins `min_participations` participations totales.

    Returns:
        set d'IDCHEVAL (string) de la cohorte de modélisation.
    """
    df = load_enriched(columns=["N° SIRE", "in_cohorte_T1", "est_libelle_poney"])
    df = df.rename(columns={"N° SIRE": "IDCHEVAL"})
    mask = df["in_cohorte_T1"] & ~df["est_libelle_poney"]
    cohorte = df[mask]
    n_part = cohorte.groupby("IDCHEVAL").size()
    return set(n_part[n_part >= min_participations].index)


def compute_cible(min_participations_hauteur=3):
    """Calcule la cible `hauteur_max_validee` pour tous les chevaux.

    Définition : max(HAUTEUR) où le cheval a participé
    >= `min_participations_hauteur` fois, sur toute sa carrière.

    Returns:
        pd.Series indexée par IDCHEVAL avec la cible (float, en mètres).
        Les chevaux sans cible calculable sont absents de la série.
    """
    df = load_enriched(columns=["N° SIRE", "HAUTEUR", "in_cohorte_T1",
                                "est_libelle_poney"])
    df = df.rename(columns={"N° SIRE": "IDCHEVAL"})
    # Restriction T1 (la cible n'a de sens que sur la cohorte)
    mask = df["in_cohorte_T1"] & ~df["est_libelle_poney"]
    df = df[mask]
    # Compte des participations par (cheval, hauteur)
    hpc = (df.dropna(subset=["HAUTEUR"])
             .groupby(["IDCHEVAL", "HAUTEUR"])
             .size()
             .reset_index(name="n"))
    validees = hpc[hpc["n"] >= min_participations_hauteur]
    cible = validees.groupby("IDCHEVAL")["HAUTEUR"].max()
    cible.name = "hauteur_max_validee"
    return cible


def save_intermediate(df, family_id, family_name):
    """Sauvegarde un dataset intermédiaire de famille.

    Args:
        df: pd.DataFrame avec 1 ligne par cheval, IDCHEVAL comme colonne ou index.
        family_id: identifiant numérique (1, 2, 3, ...).
        family_name: nom court ('activite', 'gains', ...).
    """
    INTERMEDIATES_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERMEDIATES_DIR / f"famille{family_id}_{family_name}.parquet"
    df.to_parquet(out)
    print(f"  → Sauvegardé : {out.relative_to(PROJECT_ROOT)} ({len(df):,} lignes, {len(df.columns)} colonnes)")
    return out


def sanity_check(df, family_id, expected_n_features=None):
    """Imprime des stats descriptives basiques pour vérifier le résultat."""
    print(f"\n=== Sanity check Famille {family_id} ===")
    print(f"Nb chevaux : {len(df):,}")
    print(f"Nb colonnes features : {len(df.columns) - (1 if 'IDCHEVAL' in df.columns else 0)}")
    if expected_n_features is not None:
        actual = len(df.columns) - (1 if 'IDCHEVAL' in df.columns else 0)
        if actual != expected_n_features:
            print(f"  ⚠ Attendu : {expected_n_features} features, obtenu : {actual}")
    # NaN par colonne
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"NaN par colonne :")
        for col in df.columns:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                print(f"  {col} : {n_nan:,} ({100*n_nan/len(df):.2f}%)")
    else:
        print("Pas de NaN")
