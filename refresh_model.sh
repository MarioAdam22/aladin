#!/bin/bash
# ╔══════════════════════════════════════════════════════╗
# ║  ALADIN — Refresh complet model zilnic              ║
# ║  1. Import date NQ din CSV → mario_trading.db       ║
# ║  2. Antrenare model AI                              ║
# ║  Usage: bash ~/Desktop/Aladin/refresh_model.sh      ║
# ╚══════════════════════════════════════════════════════╝

ALADIN_DIR="$HOME/Desktop/Aladin"
DB_PATH="$ALADIN_DIR/mario_trading.db"
LOG_FILE="$ALADIN_DIR/refresh_log.txt"
# Cauta cel mai recent NQ_06-26 din Aladin/ sau AladinExport/
_F1="$ALADIN_DIR/NQ_06-26.Last.txt"
_F2="$ALADIN_DIR/AladinExport/NQ_06-26.Last.txt"
if [ -f "$_F1" ] && [ -f "$_F2" ]; then
    NQ_FILE=$([ "$_F2" -nt "$_F1" ] && echo "$_F2" || echo "$_F1")
elif [ -f "$_F2" ]; then
    NQ_FILE="$_F2"
else
    NQ_FILE="$_F1"
fi

echo "======================================================"
echo "  ALADIN — REFRESH MODEL $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
echo "" | tee -a "$LOG_FILE"
echo "=== REFRESH $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_FILE"

# ── Verificam ca NT8 a exportat datele (fisierul sa fie din ziua asta) ─────────
if [ -f "$NQ_FILE" ]; then
    FILE_DATE=$(date -r "$NQ_FILE" '+%Y-%m-%d')
    TODAY=$(date '+%Y-%m-%d')
    if [ "$FILE_DATE" != "$TODAY" ]; then
        echo "⚠️  NQ_06-26.Last.txt nu e de azi (e din $FILE_DATE) — continuam oricum cu append." | tee -a "$LOG_FILE"
    else
        echo "✅ NQ_06-26.Last.txt e de azi ($TODAY) — OK" | tee -a "$LOG_FILE"
    fi
else
    echo "❌ Fisierul NQ_06-26.Last.txt lipseste. Asigura-te ca NT8 ruleaza si AutoExportNQ a exportat." | tee -a "$LOG_FILE"
    exit 1
fi

# ── Stergem journal incomplet ─────────────────────────────────────────────────
if [ -f "${DB_PATH}-journal" ]; then
    rm -f "${DB_PATH}-journal" "${DB_PATH}-wal" "${DB_PATH}-shm"
    echo "⚠️  Journal incomplet sters." | tee -a "$LOG_FILE"
fi

# ── Import date NQ/ES → mario_trading.db ─────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📥 IMPORT date NQ → mario_trading.db..." | tee -a "$LOG_FILE"

DINGRS_DIR="$ALADIN_DIR/dingrs"
NQ_FILES=""
ES_FILES=""

# Append mode: doar fisierul recent (NQ_06-26) — rapid, nu rescana tot istoricul
NQ_FILES="$NQ_FILE"
ES_FILES=""
# ES recent daca exista
_ES_RECENT="$ALADIN_DIR/AladinExport/ES_06-26.Last.txt"
[ ! -f "$_ES_RECENT" ] && _ES_RECENT="$ALADIN_DIR/ES_06-26.Last.txt"
[ -f "$_ES_RECENT" ] && ES_FILES="$_ES_RECENT"

# Opreste bridge-ul ca sa elibereze DB-ul
echo "⏸️  Opresc bridge-ul..." | tee -a "$LOG_FILE"
pkill -f "bridge_api.py" 2>/dev/null
sleep 2

python3 "$ALADIN_DIR/import_nt8_nq.py" \
    --nq "$NQ_FILES" \
    --es "$ES_FILES" \
    --db "$DB_PATH" \
    --mode append 2>&1 | tee -a "$LOG_FILE"

IMPORT_STATUS=${PIPESTATUS[0]}

# Reporneste bridge-ul
echo "▶️  Repornesc bridge-ul..." | tee -a "$LOG_FILE"
bash "$ALADIN_DIR/start.sh" > /dev/null 2>&1 &
sleep 2

if [ $IMPORT_STATUS -ne 0 ]; then
    echo "❌ Import esuat! Verifica log-ul." | tee -a "$LOG_FILE"
    exit 1
fi

echo "✅ Import reusit." | tee -a "$LOG_FILE"

# ── Antrenare model AI (doar dacă e cerut explicit) ──────────────────────────
# Import zilnic = suficient pentru inferență corectă
# Training săptămânal = rulat separat de scheduled task duminică
if [ "$1" == "--train" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "🧠 ANTRENARE model AI..." | tee -a "$LOG_FILE"
    python3 "$ALADIN_DIR/train_mario_ai.py" 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "❌ Training esuat! Verifica log-ul." | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "✅ Training complet." | tee -a "$LOG_FILE"
else
    echo "" | tee -a "$LOG_FILE"
    echo "ℹ️  Training sarit (doar import zilnic). Pentru training complet:" | tee -a "$LOG_FILE"
    echo "   bash ~/Desktop/Aladin/refresh_model.sh --train" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "✅ REFRESH COMPLET — DB actualizat!" | tee -a "$LOG_FILE"
echo "   Log salvat in: $LOG_FILE"
echo "======================================================"
