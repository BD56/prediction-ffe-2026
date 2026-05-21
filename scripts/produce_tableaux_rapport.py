"""
produce_tableaux_rapport.py — Génère tous les tableaux du rapport final.

Produit 6 tableaux dans `data/master/tableaux_rapport/` + un fichier markdown
consolidé prêt à coller dans le rapport.

Tableaux produits :
  T1. Inventaire des variables principales (§2.1)
  T2. Récapitulatif des 9 familles de features (§3.1)
  T3. 10 modèles × métriques globales (MAE, RMSE, R²) (§3.4)
  T4. Top 6 modèles × MAE par tranche (§4.1)
  T5. Top 15 features — RF default vs RF conditionnel tops (§4.2)
  T6. Cas d'usage → modèle recommandé (§5.2)

Sortie :
  - data/master/tableaux_rapport/T1_inventaire_variables.csv ... T6_*.csv
  - data/master/tableaux_rapport/tous_tableaux.md (markdown consolidé)

Usage :
    python3 scripts/produce_tableaux_rapport.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import MASTER_DIR, load_enriched

OUT_DIR = MASTER_DIR / "tableaux_rapport"
OUT_DIR.mkdir(exist_ok=True)

MARKDOWN_LINES = []


def add_md(*lines):
    """Ajoute des lignes au markdown consolidé."""
    for line in lines:
        MARKDOWN_LINES.append(line)


def df_to_md(df, align=None):
    """Convertit un DataFrame en markdown propre (sans dépendance tabulate)."""
    headers = list(df.columns)
    rows = df.astype(str).values.tolist()
    # Largeur de chaque colonne
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows) if rows else 0)
              for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    body = "\n".join(
        "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
        for r in rows
    )
    return f"{head}\n{sep}\n{body}"


# ============================================================
# T1 — Inventaire des variables principales
# ============================================================

def t1_inventaire_variables():
    print("[T1] Inventaire des variables...")
    df = load_enriched()

    # Classifier les variables par type/usage
    var_info = [
        ("IDCHEVAL", "ID", "Identifiant unique du cheval (N° SIRE renommé)"),
        ("NOMCHEVAL", "Texte", "Nom du cheval"),
        ("DATENAISSANCE", "Année", "Année de naissance du cheval"),
        ("RACECHEVAL", "Catégoriel", "Race officielle du cheval"),
        ("NUMSIREPERE", "ID", "Identifiant SIRE du père"),
        ("NUMSIREMERE", "ID", "Identifiant SIRE de la mère"),
        ("NUMSIREPEREMERE", "ID", "Identifiant SIRE du grand-père maternel"),
        ("LICENCE", "ID", "Numéro de licence du cavalier"),
        ("NUMERO_EVENEMENT2", "ID", "Identifiant unique de l'événement"),
        ("DESIGNATION", "Texte", "Nom de l'événement"),
        ("CLASSEEPREUVE_LIB", "Texte", "Libellé complet de l'épreuve"),
        ("CLASSEEPREUVE_CODE", "Catégoriel", "Code court de l'épreuve"),
        ("DATEEPREUVE", "Date", "Date de l'épreuve"),
        ("DISCIPLINE_CODE", "Catégoriel", "Discipline (SO, CE)"),
        ("DIVISION_LIB", "Catégoriel", "Division (Pro 1/2/3, Amateur Elite/1/2/3/4, Club...)"),
        ("PLACE", "Numérique", "Classement final dans l'épreuve"),
        ("GAINS", "Numérique", "Gains en euros pour cette participation"),
        ("POINTS", "Numérique", "Points classement (selon barème)"),
        ("SO_POINTS_BAR", "Numérique", "Points pénalité au barrage SO"),
        ("SO_TEMPS", "Numérique", "Temps de parcours SO (secondes)"),
        ("SO_TEMPS_BAR", "Numérique", "Temps de parcours au barrage"),
        ("CE_POINTSDRESSAGE", "Numérique", "Points dressage en Concours Complet"),
        ("CE_POINTSFOND", "Numérique", "Points cross en Concours Complet"),
        ("CE_POINTSSO", "Numérique", "Points SO en Concours Complet"),
        ("ANNEE", "Année", "Année d'observation"),
        ("HAUTEUR", "Numérique", "Hauteur d'obstacle extraite du libellé (regex V2)"),
        ("AGE", "Numérique", "Âge du cheval à la participation"),
    ]
    rows = []
    for name, typ, desc in var_info:
        if name not in df.columns:
            continue
        nan_pct = df[name].isna().mean() * 100
        rows.append({"Variable": name, "Type": typ,
                      "% NaN": f"{nan_pct:.1f}%",
                      "Description": desc})
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT_DIR / "T1_inventaire_variables.csv", index=False)

    add_md("## Tableau 1 — Inventaire des variables principales (§2.1)",
           "",
           df_to_md(inv),
           "",
           f"*Source : `data/ffe_2010-2025_enriched.parquet` ({len(df):,} lignes).*",
           "")
    print(f"  ✓ {OUT_DIR / 'T1_inventaire_variables.csv'}")
    return inv


# ============================================================
# T2 — Récapitulatif des 9 familles de features
# ============================================================

def t2_familles():
    print("[T2] Récap des familles...")
    rows = [
        ("F1", "Activité / Volume",
         "18", "nb participations, nb événements, durée carrière, intensité",
         "Aucun majeur (signal incontournable)", "✅ Retenue"),
        ("F2", "Gains du cheval",
         "10", "gains totaux, gains par âge, ratios efficacité",
         "Aucun majeur (transformations log standardisées)", "✅ Retenue"),
        ("F3", "Performance sportive",
         "153", "victoires, placement (percentile partants/finishers), taux top X%",
         "Codes PLACE administratifs (~7%) traités en non-classement", "✅ Retenue"),
        ("F4", "Hauteurs explorées",
         "0", "(distribution de la hauteur d'obstacle, médiane, max…)",
         "Auto-corrélation forte avec la cible + 47% NaN structurel",
         "❌ Rejetée"),
        ("F5", "Niveau / division",
         "24", "taux participation Amateur/Pro/Élevage par âge",
         "Aucun majeur", "✅ Retenue"),
        ("F6", "Progression temporelle",
         "0", "(dérivées et tendances)",
         "Redondance avec deltas inter-annuels des autres familles",
         "❌ Rejetée"),
        ("F7", "Cavalier",
         "56", "expérience cavalier (percentile, gains), Hurdle gains 16 features",
         "Leakage si mal construit (erreur PERE_ELITE) → LOO appliqué",
         "✅ Retenue"),
        ("F8", "Pedigree (Père + GP maternel)",
         "8", "target encoding LOO sur 4 stats × 2 ancêtres",
         "Mère écartée (2,7% couverture LOO insuffisante)",
         "✅ Retenue"),
        ("F9", "Saisonnalité",
         "0", "(mois, saison de l'épreuve)",
         "Marginal pour la cible prédictive (effet capté ailleurs)",
         "❌ Rejetée"),
        ("F10", "Race",
         "4", "target encoding LOO sur 4 stats par race",
         "Aucun majeur (LOO+smoothing Bayésien)",
         "✅ Retenue"),
        ("F11", "Géographique",
         "0", "(département, région)",
         "Pas de données GPS/géographiques structurées dans le dataset",
         "❌ Rejetée"),
    ]
    fam = pd.DataFrame(rows, columns=["Code", "Famille", "Nb features",
                                        "Type de variables",
                                        "Risque / décision",
                                        "Statut"])
    fam.to_csv(OUT_DIR / "T2_familles.csv", index=False)
    add_md("## Tableau 2 — Récapitulatif des 9 familles de features (§3.1)",
           "",
           df_to_md(fam),
           "",
           "*Total : 273 features candidates construites sur les 6 familles retenues, puis 156 après épuration.*",
           "")
    print(f"  ✓ {OUT_DIR / 'T2_familles.csv'}")
    return fam


# ============================================================
# T3 — 10 modèles × métriques globales
# ============================================================

def t3_modeles_globaux():
    print("[T3] 10 modèles × métriques globales...")
    df = pd.read_csv(MASTER_DIR / "recap_avec_poly40_global.csv")
    df = df.sort_values("MAE_cm").reset_index(drop=True)
    df["Rang"] = range(1, len(df) + 1)
    out = df[["Rang", "modele", "MAE_cm", "RMSE_cm", "R2"]].copy()
    out.columns = ["Rang", "Modèle", "MAE (cm)", "RMSE (cm)", "R²"]
    out["MAE (cm)"] = out["MAE (cm)"].round(2)
    out["RMSE (cm)"] = out["RMSE (cm)"].round(2)
    out["R²"] = out["R²"].round(4)
    out.to_csv(OUT_DIR / "T3_modeles_globaux.csv", index=False)
    add_md("## Tableau 3 — Performance globale des 10 modèles testés (§3.4)",
           "",
           df_to_md(out),
           "",
           "*Métriques calculées sur le test 2013 (5 045 chevaux), pipeline reproductible (script `32_recap_avec_rmse.py` puis `39_recap_avec_poly40.py`).*",
           "")
    print(f"  ✓ {OUT_DIR / 'T3_modeles_globaux.csv'}")
    return out


# ============================================================
# T4 — Top 6 modèles × MAE par tranche
# ============================================================

def t4_top6_par_tranche():
    print("[T4] Top 6 modèles + Hurdle × MAE par tranche...")
    mae = pd.read_csv(MASTER_DIR / "recap_avec_poly40_mae.csv")
    glob = pd.read_csv(MASTER_DIR / "recap_avec_poly40_global.csv")
    glob = glob.sort_values("MAE_cm")
    # Top 5 par MAE globale + Hurdle (toujours inclus car central au rapport)
    top5 = glob["modele"].head(5).tolist()
    selection = top5 + (["Hurdle (mélange)"] if "Hurdle (mélange)" not in top5 else [])

    mae = mae[mae["modele"].isin(selection)].copy()
    # Ordre : MAE globale croissante puis Hurdle à la fin pour le mettre en évidence
    ordre = {m: i for i, m in enumerate(top5)}
    ordre["Hurdle (mélange)"] = 999
    mae["__o"] = mae["modele"].map(ordre)
    mae = mae.sort_values("__o").drop(columns="__o")
    mae = mae.rename(columns={"modele": "Modèle"})
    cols_tranches = ["≤1.10m", "1.15-1.20m", "1.25-1.30m", "1.35-1.40m", "≥1.45m"]
    for c in cols_tranches:
        mae[c] = mae[c].round(2)

    mae.to_csv(OUT_DIR / "T4_top6_mae_par_tranche.csv", index=False)
    add_md("## Tableau 4 — Top 5 modèles globaux + Hurdle, MAE par tranche (cm) (§4.1)",
           "",
           df_to_md(mae),
           "",
           "*Lecture : Hurdle (placé en dernière ligne pour comparaison) est moins bon globalement mais **meilleur sur les hauts niveaux** : 5,27 cm sur 1,35-1,40m et 9,52 cm sur ≥1,45m, contre 7,13 cm et 12,71 cm pour le champion global Stacking + Calib. Cette inversion légitime la recommandation « deux modèles selon le cas d'usage ».*",
           "")
    print(f"  ✓ {OUT_DIR / 'T4_top6_mae_par_tranche.csv'}")
    return mae


# ============================================================
# T5 — Top 15 features RF default vs RF conditionnel tops
# ============================================================

def t5_importances_comparees():
    print("[T5] Importances RF default vs RF conditionnel...")
    df = pd.read_csv(MASTER_DIR / "diagnostic_hurdle_importances.csv")
    top = df.sort_values("rang_tops").head(15)
    out = top[["feature", "rang_def", "rang_tops", "delta_rang"]].copy()
    out.columns = ["Feature", "Rang RF default", "Rang RF conditionnel (tops)",
                    "Δ rang (default → tops)"]
    out["Δ rang (default → tops)"] = out["Δ rang (default → tops)"].apply(
        lambda x: f"+{x}" if x > 0 else f"{x}"
    )
    out.to_csv(OUT_DIR / "T5_importances_comparees.csv", index=False)
    add_md("## Tableau 5 — Top 15 features chez le régresseur conditionnel des tops vs RF default (§4.2)",
           "",
           df_to_md(out),
           "",
           "*Lecture : un Δ positif signifie que la feature monte en importance chez le régresseur conditionnel (= devient plus utile pour distinguer les chevaux de haut niveau entre eux). La génétique (race, pedigree) et le cavalier précoce gagnent jusqu'à +51 rangs.*",
           "")
    print(f"  ✓ {OUT_DIR / 'T5_importances_comparees.csv'}")
    return out


# ============================================================
# T6 — Cas d'usage → modèle recommandé
# ============================================================

def t6_cas_usage_modele():
    print("[T6] Cas d'usage → modèle recommandé...")
    rows = [
        ("Détecter les futurs Pro 1 (priorité FFE)",
         "Hurdle (mélange)", "9,52", "6,89",
         "Gagne sur 98% des chevaux ≥1,45m, MAE 3 cm meilleure que les autres modèles sur les tops"),
        ("Suivi général de tous les chevaux",
         "Stacking + Calib isotonic", "12,71", "6,28",
         "Meilleure MAE globale, mais moins bon que Hurdle sur la zone ≥1,40m"),
        ("Communication aux éleveurs (modèle interprétable)",
         "Poly40 (degré 2 + interactions)", "13,19", "6,44",
         "Formule polynomiale explicite avec 40 variables ; performance comparable à RF avec 4× moins de variables"),
        ("Détection précoce d'un cheval modeste (orienter amateur)",
         "Stacking + Calib", "6,86 (cm sur ≤1,10m)", "—",
         "Précision la plus fine sur les bas niveaux (≤1,10m), évite la sur-estimation systématique de Hurdle"),
    ]
    df = pd.DataFrame(rows, columns=[
        "Cas d'usage métier", "Modèle recommandé",
        "MAE ≥1,45m (cm)", "MAE globale (cm)",
        "Justification"
    ])
    df.to_csv(OUT_DIR / "T6_cas_usage_modele.csv", index=False)
    add_md("## Tableau 6 — Cas d'usage métier → modèle recommandé (§5.2)",
           "",
           df_to_md(df),
           "",
           "*Aligné sur le sujet FFE « identifier des modèles prédictifs de performance à haut niveau » : Hurdle est le modèle principal recommandé pour le cas d'usage prioritaire (détection des futurs Pro 1).*",
           "")
    print(f"  ✓ {OUT_DIR / 'T6_cas_usage_modele.csv'}")
    return df


# ============================================================
# Pipeline principal
# ============================================================

def main():
    print("=" * 70)
    print(f"GÉNÉRATION DES TABLEAUX POUR LE RAPPORT")
    print(f"Sortie : {OUT_DIR}")
    print("=" * 70 + "\n")

    add_md("# Tableaux pour le rapport final — projet FFE",
           "",
           "Document généré automatiquement par `scripts/produce_tableaux_rapport.py`.",
           "Tableaux prêts à coller dans le rapport (format markdown / convertible Word ou LaTeX).",
           "",
           "---", "")

    t1_inventaire_variables()
    t2_familles()
    t3_modeles_globaux()
    t4_top6_par_tranche()
    t5_importances_comparees()
    t6_cas_usage_modele()

    # Sauvegarder le markdown consolidé
    md_path = OUT_DIR / "tous_tableaux.md"
    md_path.write_text("\n".join(MARKDOWN_LINES))

    print("\n" + "=" * 70)
    print(f"✓ 6 tableaux CSV générés dans {OUT_DIR}")
    print(f"✓ Markdown consolidé : {md_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
