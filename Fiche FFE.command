#!/bin/bash
# Lanceur natif macOS : dialogue système pour saisir un SIRE, puis génère
# la fiche via le CLI Python et l'ouvre dans Aperçu. Boucle jusqu'à annulation.

cd "$(dirname "$0")"

# Détection automatique de l'interpréteur Python disponible.
# On préfère un environnement virtuel local s'il existe, sinon python3 du PATH.
if [ -x ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif [ -x "/Library/Developer/CommandLineTools/usr/bin/python3" ]; then
    PYTHON="/Library/Developer/CommandLineTools/usr/bin/python3"
else
    osascript -e 'display alert "Python introuvable" message "Installer Python 3 puis relancer." as critical'
    exit 1
fi
SCRIPT="scripts/fiche.py"

# Premier appel : prépare le cache si besoin (avec dialogue de progression)
if [ ! -f "data/master/intermediates/hurdle_lac_cache.pkl" ]; then
    osascript -e 'display notification "Entraînement du modèle Hurdle (~15s)…" with title "Fiche FFE"'
    "$PYTHON" "$SCRIPT" --list-test 1 > /dev/null
fi

while true; do
    SIRE=$(osascript <<'EOF'
        try
            set resultat to display dialog "Entre un numéro SIRE :" ¬
                default answer "" ¬
                with title "Fiche de prédiction FFE" ¬
                buttons {"Annuler", "Générer"} ¬
                default button "Générer" ¬
                with icon note
            if button returned of resultat is "Générer" then
                return text returned of resultat
            else
                return "__CANCEL__"
            end if
        on error number -128
            return "__CANCEL__"
        end try
EOF
    )

    # Annulation ou Échap → on quitte
    if [ "$SIRE" = "__CANCEL__" ] || [ -z "$SIRE" ]; then
        exit 0
    fi

    # Génération + ouverture (le CLI ouvre déjà l'image dans Aperçu)
    OUTPUT=$("$PYTHON" "$SCRIPT" "$SIRE" 2>&1)
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        # Erreur (ex: SIRE inconnu) → dialogue d'alerte
        MESSAGE=$(echo "$OUTPUT" | tail -3 | head -1 | sed 's/"/\\"/g')
        osascript -e "display alert \"Erreur\" message \"$MESSAGE\" as critical"
    fi
done
