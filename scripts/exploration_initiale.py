"""
exploration_initiale.py — Analyses exploratoires initiales (équivalent du code R `.Rhistory`).

Reproduit en Python les analyses qui avaient été faites en R lors de la phase
d'exploration des données brutes :
  1. Distribution de la variable PLACE (identification des codes administratifs)
  2. Détail des valeurs hautes (≥ 100) — utile pour identifier les codes 900, 999, etc.
  3. Statistiques de qualité (NaN, anomalies)
  4. Distribution de la cible et profil métier

INPUT : data/ffe_2010-2025_enriched.parquet
OUTPUT :
  - distribution_place_hautes.csv (valeurs PLACE ≥ 100)
  - distribution_place_par_tranche.csv (tableau d'effectifs)

Usage :
    python3 scripts/exploration_initiale.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_enriched, DATA_DIR


def fmt(n):
    """Formatage des grands nombres avec espace."""
    return f"{n:,}".replace(",", " ")


def main():
    print("=" * 70)
    print("EXPLORATION INITIALE — analyses descriptives")
    print("=" * 70)

    # ---------- 1. Distribution PLACE ----------
    print("\n[1/4] Distribution de la variable PLACE (SO uniquement)...")
    df = load_enriched(columns=["PLACE", "DISCIPLINE_CODE"])
    d_so = df[df["DISCIPLINE_CODE"] == "SO"].copy()
    print(f"  Lignes SO totales : {fmt(len(d_so))}")

    # Découpage par tranches
    breaks = [0, 10, 50, 100, 500, 900, 990, 1000]
    labels = ["1-10", "11-50", "51-100", "101-500", "501-900", "901-989", "990-999"]
    d_so["tranche"] = pd.cut(d_so["PLACE"], bins=breaks, labels=labels, right=True)

    tbl = d_so.dropna(subset=["PLACE"]).groupby("tranche", observed=True).size().reset_index()
    tbl.columns = ["tranche", "N"]
    tbl["pct"] = (100 * tbl["N"] / tbl["N"].sum()).round(2)

    print("\n  === Distribution PLACE (par tranches) ===")
    print(tbl.to_string(index=False))
    tbl.to_csv("distribution_place_par_tranche.csv", index=False)

    # ---------- 2. Valeurs hautes (≥ 100) ----------
    print("\n[2/4] Détail des valeurs PLACE ≥ 100 (codes administratifs suspects)...")
    hautes = (
        d_so[d_so["PLACE"] >= 100]
        .groupby("PLACE")
        .size()
        .reset_index(name="N")
        .sort_values("PLACE")
    )
    print(f"  Total valeurs distinctes ≥ 100 : {len(hautes)}")
    print(f"  Total lignes ≥ 100 : {fmt(hautes['N'].sum())}")
    hautes.to_csv("distribution_place_hautes.csv", index=False)

    # ---------- 3. Codes hauts les plus fréquents (≥ 500) ----------
    print("\n[3/4] Top valeurs PLACE ≥ 500 (placeholders administratifs typiques)...")
    top_500 = (
        d_so[d_so["PLACE"] >= 500]
        .groupby("PLACE")
        .size()
        .reset_index(name="N")
        .sort_values("N", ascending=False)
        .head(15)
    )
    print(top_500.to_string(index=False))

    # Nb valeurs distinctes par tranche
    print("\n  Nb de valeurs distinctes par tranche :")
    for low, high in [(101, 500), (501, 900), (900, 999)]:
        sub = d_so[(d_so["PLACE"] >= low) & (d_so["PLACE"] <= high)]
        print(f"    {low}-{high} : {sub['PLACE'].nunique()} valeurs distinctes "
              f"({fmt(len(sub))} lignes)")

    # ---------- 4. Qualité globale des données ----------
    print("\n[4/4] Qualité globale — % de NaN par colonne (top 15)...")
    full = load_enriched()
    nan_pct = (full.isna().mean() * 100).sort_values(ascending=False).head(15)
    print(nan_pct.round(2).to_string())

    # ---------- Distribution de la cible (préview) ----------
    from utils import compute_cible
    print("\n[Bonus] Distribution de la cible hauteur_max_validee (≥ 3 participations)...")
    cible = compute_cible()
    print(f"  Nb chevaux avec cible calculable : {fmt(len(cible))}")
    print(f"  Min/Médiane/Max : {cible.min():.2f} / {cible.median():.2f} / {cible.max():.2f}")
    print(f"  Q1/Q3 : {cible.quantile(0.25):.2f} / {cible.quantile(0.75):.2f}")
    print("\n  Distribution par tranches :")
    tranches = pd.cut(cible, bins=[0, 1.10, 1.20, 1.30, 1.40, 2.0],
                       labels=["≤1.10m", "1.15-1.20m", "1.25-1.30m",
                                "1.35-1.40m", "≥1.45m"])
    dist = tranches.value_counts().sort_index()
    for tr, n in dist.items():
        print(f"    {tr:<12s} : {fmt(n):>8s} chevaux ({n/len(cible)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("Fichiers de sortie générés :")
    print("  - distribution_place_par_tranche.csv")
    print("  - distribution_place_hautes.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
