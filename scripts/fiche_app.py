"""
Application autonome (sans Spyder) pour générer une fiche de prédiction
à partir d'un numéro SIRE.

Usage :
    python3 scripts/fiche_app.py

Au lancement, l'app entraîne ou recharge le modèle Hurdle puis affiche
une fenêtre avec un champ SIRE et un bouton "Générer". La fiche prédite
s'affiche directement dans la fenêtre.
"""

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).parent))
from fiche import (  # noqa: E402
    CACHE_PATH,
    OUT_DIR_DEFAULT,
    fit_and_cache,
    load_cache,
    predict_one,
    render_png,
)


class FicheApp:
    def __init__(self, root):
        self.root = root
        self.cache = None
        self.current_path = None
        self.current_photo = None  # garde une ref pour éviter le GC

        root.title("Fiche de prédiction FFE")
        root.geometry("780x1100")

        # ---------- Bandeau du haut ----------
        top = ttk.Frame(root, padding=12)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Numéro SIRE :", font=("Helvetica", 13)).pack(side=tk.LEFT)
        self.entry = ttk.Entry(top, width=18, font=("Helvetica", 13))
        self.entry.pack(side=tk.LEFT, padx=8)
        self.entry.bind("<Return>", lambda e: self.on_generer())

        self.btn_generer = ttk.Button(top, text="Générer", command=self.on_generer)
        self.btn_generer.pack(side=tk.LEFT, padx=4)
        self.btn_save = ttk.Button(top, text="Enregistrer sous…",
                                    command=self.on_save, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=4)
        self.btn_retrain = ttk.Button(top, text="Ré-entraîner",
                                       command=self.on_retrain)
        self.btn_retrain.pack(side=tk.RIGHT, padx=4)

        # ---------- Zone d'affichage de la fiche ----------
        self.canvas_frame = ttk.Frame(root, padding=8)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.image_label = ttk.Label(self.canvas_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # ---------- Bandeau de statut ----------
        self.status = tk.StringVar(value="Initialisation…")
        ttk.Label(root, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W,
                  padding=(8, 4)).pack(side=tk.BOTTOM, fill=tk.X)

        # Désactive l'UI tant que le modèle n'est pas chargé
        self._set_busy(True)
        threading.Thread(target=self._load_or_train, daemon=True).start()

    # ---------- Chargement modèle ----------
    def _load_or_train(self):
        try:
            if not CACHE_PATH.exists():
                self._set_status("Premier lancement : entraînement du modèle (~15s)…")
                fit_and_cache()
            else:
                self._set_status("Chargement du modèle en cache…")
            self.cache = load_cache()
            self._set_status("Prêt. Saisis un numéro SIRE et clique sur Générer.")
        except Exception as e:
            self._set_status(f"Erreur de chargement : {e}")
            self.root.after(0, lambda: messagebox.showerror("Erreur modèle", str(e)))
        finally:
            self.root.after(0, lambda: self._set_busy(False))

    # ---------- Génération ----------
    def on_generer(self):
        if self.cache is None:
            messagebox.showinfo("Patientez", "Le modèle est encore en cours de chargement.")
            return
        sire = self.entry.get().strip()
        if not sire:
            messagebox.showwarning("Saisie manquante", "Entre un numéro SIRE.")
            return
        self._set_busy(True)
        self._set_status(f"Prédiction pour {sire}…")
        threading.Thread(target=self._predict_worker, args=(sire,), daemon=True).start()

    def _predict_worker(self, sire):
        try:
            fiche = predict_one(sire, self.cache)
            out_path = OUT_DIR_DEFAULT / f"fiche_{fiche['id']}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            render_png(fiche, out_path)
            self.current_path = out_path

            ligne_reel = ""
            if fiche.get("reel") is not None:
                couv = "couvert" if fiche["couvert"] else "HORS IC"
                ligne_reel = (f"  •  réelle {fiche['reel']:.2f} m ({couv}, "
                              f"err {fiche['erreur']:.1f} cm)")
            resume = (
                f"{sire} ({fiche['split']}) — {fiche['statut']} "
                f"P_top={fiche['p_top']:.2f}  pred={fiche['pred']:.2f} m  "
                f"IC [{fiche['ic_low']:.2f} ; {fiche['ic_high']:.2f}]  "
                f"conf {fiche['confiance']}/5{ligne_reel}"
            )
            self.root.after(0, lambda: self._afficher_image(out_path))
            self._set_status(resume)
        except KeyError as e:
            self._set_status("Cheval introuvable.")
            self.root.after(0, lambda: messagebox.showerror(
                "IDCHEVAL inconnu", str(e)))
        except Exception as e:
            self._set_status(f"Erreur : {e}")
            self.root.after(0, lambda: messagebox.showerror("Erreur prédiction", str(e)))
        finally:
            self.root.after(0, lambda: self._set_busy(False))

    def _afficher_image(self, path):
        img = Image.open(path)
        # Adapte la taille à la zone d'affichage en gardant le ratio
        max_w = max(self.canvas_frame.winfo_width() - 16, 400)
        max_h = max(self.canvas_frame.winfo_height() - 16, 600)
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.current_photo)
        self.btn_save.configure(state=tk.NORMAL)

    # ---------- Save as ----------
    def on_save(self):
        if not self.current_path:
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=self.current_path.name,
            filetypes=[("PNG", "*.png")],
        )
        if dest:
            Path(dest).write_bytes(Path(self.current_path).read_bytes())
            self._set_status(f"Copié dans {dest}")

    # ---------- Ré-entraînement manuel ----------
    def on_retrain(self):
        if not messagebox.askyesno(
            "Ré-entraîner ?",
            "Cela va supprimer le cache et ré-entraîner le Hurdle (~15s).\nContinuer ?",
        ):
            return
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
        self.cache = None
        self._set_busy(True)
        self._set_status("Ré-entraînement en cours…")
        threading.Thread(target=self._load_or_train, daemon=True).start()

    # ---------- Helpers ----------
    def _set_busy(self, busy):
        state = tk.DISABLED if busy else tk.NORMAL
        for w in (self.entry, self.btn_generer, self.btn_retrain):
            w.configure(state=state)
        if not busy and self.current_path is None:
            self.btn_save.configure(state=tk.DISABLED)

    def _set_status(self, msg):
        self.root.after(0, lambda: self.status.set(msg))


def main():
    root = tk.Tk()
    FicheApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
