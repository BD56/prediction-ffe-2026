"""
Génère deux fichiers de référence pour piocher des numéros SIRE :
  - data/master/sires_chevaux.csv  : 47 617 chevaux du dataset
  - data/master/sires_demo.txt     : sélection commentée pour démo soutenance
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR  # noqa: E402


def main():
    df = pd.read_parquet(MASTER_DIR / "master_dataset_epure_v2.parquet")
    df = df[["IDCHEVAL", "SPLIT", "hauteur_max_validee"]].copy()
    df = df.sort_values(["hauteur_max_validee", "IDCHEVAL"], ascending=[False, True])
    df["hauteur_max_validee"] = df["hauteur_max_validee"].round(2)
    df.columns = ["IDCHEVAL", "SPLIT", "hauteur_reelle_m"]

    out_csv = MASTER_DIR / "sires_chevaux.csv"
    df.to_csv(out_csv, index=False)
    print(f"écrit : {out_csv}  ({len(df):,} chevaux)")

    # ---------- Fichier démo ----------
    lines = [
        "# Numéros SIRE recommandés pour la démo de l'application Fiche FFE",
        "# Copie-colle le SIRE dans la boîte de dialogue.",
        "",
        "## Les 5 profils du rapport (test set)",
        "",
        "47237708P    # Crack confirmé : P_top=0.94, prédit 1.43m, réel 1.35m",
        "13417805D    # Plafond amateur : P_top=0.00, prédit 1.11m, réel 1.10m",
        "13414483P    # Cas incertain : P_top=0.40, IC très large (43 cm)",
        "60049124M    # Anomalie statistique : prédit 1.39m mais réel 1.05m (HORS IC)",
        "47261754C    # Vrai TOP ≥1.45m : prédit 1.38m, réel 1.45m",
        "",
        "## Exemples supplémentaires par tranche de hauteur (test set)",
        "",
    ]

    test = df[df["SPLIT"] == "test"]
    tranches = [
        ("Très hauts (≥ 1.45 m)", test["hauteur_reelle_m"] >= 1.45, 5),
        ("Hauts (1.35 - 1.45 m)",
         (test["hauteur_reelle_m"] >= 1.35) & (test["hauteur_reelle_m"] < 1.45), 5),
        ("Moyens (1.20 - 1.35 m)",
         (test["hauteur_reelle_m"] >= 1.20) & (test["hauteur_reelle_m"] < 1.35), 5),
        ("Bas (< 1.20 m)", test["hauteur_reelle_m"] < 1.20, 5),
    ]
    for titre, mask, n in tranches:
        lines.append(f"### {titre}")
        lines.append("")
        echantillon = test[mask].sample(n=min(n, mask.sum()), random_state=42)
        for _, row in echantillon.iterrows():
            lines.append(f"{row['IDCHEVAL']}    # réelle {row['hauteur_reelle_m']:.2f} m")
        lines.append("")

    lines.extend([
        "## Quelques chevaux hors test set (pour montrer le mode 'production')",
        "# (aucune hauteur réelle ne sera affichée, juste la prédiction + IC)",
        "",
    ])
    for split in ("train", "valid"):
        ech = df[df["SPLIT"] == split].sample(n=3, random_state=42)
        for _, row in ech.iterrows():
            lines.append(f"{row['IDCHEVAL']}    # split={split}")
    lines.append("")

    out_txt = MASTER_DIR / "sires_demo.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"écrit : {out_txt}")


if __name__ == "__main__":
    main()
