#!/bin/bash
# ╔══════════════════════════════════════════════════════╗
# ║  ALADIN — Importă date NQ/ES în mario_trading.db    ║
# ║  Rulează acest script în Terminal pe Mac             ║
# ║  Usage: bash RUN_IMPORT_NQ.sh                       ║
# ╚══════════════════════════════════════════════════════╝

ALADIN_DIR="$HOME/Desktop/Aladin"
DB_PATH="$ALADIN_DIR/mario_trading.db"

echo "======================================================"
echo "  ALADIN NT8 IMPORTER — NQ/ES → mario_trading.db"
echo "======================================================"

# Ștergem journal-ul incomplet (dacă există)
if [ -f "${DB_PATH}-journal" ]; then
    echo "⚠️  Găsit journal incomplet — șterg..."
    rm -f "${DB_PATH}-journal"
    rm -f "${DB_PATH}-wal"
    rm -f "${DB_PATH}-shm"
    echo "   Journal șters."
fi

# Verificăm că există fișierele NQ/ES
NQ_FILES=""
ES_FILES=""
DINGRS_DIR="$ALADIN_DIR/dingrs"

# Caută NQ în Aladin + dingrs + Desktop
for dir in "$ALADIN_DIR" "$DINGRS_DIR" "$HOME/Desktop"; do
    for f in "$dir"/NQ*.txt "$dir"/NQ*.Last.txt "$dir"/"NQ "*.txt; do
        [ -f "$f" ] && NQ_FILES="${NQ_FILES},${f}"
    done
done
NQ_FILES="${NQ_FILES#,}"  # eliminăm virgula de la început

# Caută ES în Aladin + dingrs + Desktop
for dir in "$ALADIN_DIR" "$DINGRS_DIR" "$HOME/Desktop"; do
    for f in "$dir"/ES*.txt "$dir"/ES*.Last.txt "$dir"/"ES "*.txt; do
        [ -f "$f" ] && ES_FILES="${ES_FILES},${f}"
    done
done
ES_FILES="${ES_FILES#,}"

if [ -z "$NQ_FILES" ]; then
    echo "❌ Nu am găsit fișiere NQ în $ALADIN_DIR"
    echo "   Copiază fișierele NQ din NT8 Historical Data Manager în folder-ul Aladin"
    exit 1
fi

echo ""
echo "📁 Fișiere NQ găsite:"
echo "$NQ_FILES" | tr ',' '\n' | while read f; do echo "  - $(basename "$f")"; done

if [ -n "$ES_FILES" ]; then
    echo "📁 Fișiere ES găsite:"
    echo "$ES_FILES" | tr ',' '\n' | while read f; do echo "  - $(basename "$f")"; done
fi

echo ""
echo "🚀 Pornire import..."
echo ""

python3 "$ALADIN_DIR/import_nt8_nq.py" \
    --nq "$NQ_FILES" \
    --es "$ES_FILES" \
    --db "$DB_PATH" \
    --mode replace

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo ""
    echo "✅ Import reușit! mario_trading.db actualizat cu date NQ."
    echo ""
    echo "Pasul următor: rulează antrenarea modelului"
    echo "  python3 $ALADIN_DIR/train_mario_ai.py"
else
    echo ""
    echo "❌ Import eșuat (exit code $STATUS). Verifică erorile de mai sus."
fi
