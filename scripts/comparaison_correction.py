"""Genere les fiches ANCIENNE et CORRIGEE cote a cote, pour comparaison visuelle.

Branche correction-incertitude. Ne touche a aucune fiche publiee : tout est ecrit
dans data/master/comparaison_correction/.

Usage :
    python3 scripts/comparaison_correction.py                # les 5 fiches du rapport
    python3 scripts/comparaison_correction.py IDCHEVAL ...   # chevaux au choix
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from fiche import MASTER_DIR, load_cache, predict_one, render_png  # noqa: E402

# Les 5 chevaux du rapport : crack confirme, top detecte, plafond amateur,
# cas incertain, hors intervalle.
FICHES_RAPPORT = ["47237708P", "13417805D", "13414483P", "60049124M", "47261754C"]
SORTIE = MASTER_DIR / "comparaison_correction"


def cote_a_cote(gauche: Path, droite: Path, sortie: Path, titre: str):
    g, d = Image.open(gauche), Image.open(droite)
    marge, bandeau = 20, 46
    largeur = g.width + d.width + 3 * marge
    hauteur = max(g.height, d.height) + bandeau + 2 * marge
    img = Image.new("RGB", (largeur, hauteur), "white")
    img.paste(g, (marge, bandeau + marge))
    img.paste(d, (2 * marge + g.width, bandeau + marge))
    dessin = ImageDraw.Draw(img)
    dessin.text((marge, 14), f"{titre}   —   GAUCHE : calibrage actuel (sigma)   |   "
                             f"DROITE : corrige (foret f1+f5+sigma)", fill="black")
    img.save(sortie)
    return sortie


def main():
    ids = sys.argv[1:] or FICHES_RAPPORT
    SORTIE.mkdir(parents=True, exist_ok=True)
    cache_ancien, cache_corrige = load_cache(ancien=True), load_cache()
    if "foret_erreur" not in cache_corrige:
        sys.exit("Cache corrige absent. Lancer d'abord : python3 scripts/fiche.py --retrain <ID>")

    print(f"{'cheval':<12} {'IC ancien':>11} {'IC corrige':>12} {'ecart':>8} "
          f"{'etoiles':>9} {'couvert':>18}")
    for idc in ids:
        f_a, f_c = predict_one(idc, cache_ancien), predict_one(idc, cache_corrige)
        pa, pc = SORTIE / f"{idc}_ancien.png", SORTIE / f"{idc}_corrige.png"
        render_png(f_a, pa)
        render_png(f_c, pc)
        cote_a_cote(pa, pc, SORTIE / f"{idc}_comparaison.png", f"Cheval {idc}")
        ecart = f_c["ic_width_cm"] - f_a["ic_width_cm"]
        couv = ""
        if "couvert" in f_a:
            couv = f"{'oui' if f_a['couvert'] else 'NON'} -> {'oui' if f_c['couvert'] else 'NON'}"
        print(f"{idc:<12} {f_a['ic_width_cm']:9.1f}cm {f_c['ic_width_cm']:10.1f}cm "
              f"{ecart:+7.1f}cm {f_a['confiance']}/5 -> {f_c['confiance']}/5 {couv:>18}")

    print(f"\nImages ecrites dans {SORTIE}")


if __name__ == "__main__":
    main()
