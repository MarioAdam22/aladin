#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ALADIN — SETUP DATE ISTORICE v2                                            ║
# ║  Rulează pe Mac: bash ~/Desktop/Aladin/setup_data.sh                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

ALADIN_DIR="$HOME/Desktop/Aladin"
DATA_DIR="$ALADIN_DIR/data"
mkdir -p "$DATA_DIR"
cd "$ALADIN_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ALADIN — Setup Date Istorice                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════
# 1. ȘTIRI ECONOMICE — 3 surse verificate, în ordine
# ══════════════════════════════════════════════════════════
echo "📰 Pasul 1: Descărcare calendar economic (2007–prezent)..."
echo ""

FF_OUT="$DATA_DIR/historical_news.csv"
FF_OUT_JSON="$DATA_DIR/historical_news.json"
DOWNLOADED=false

# Helper: verifică dacă un CSV descărcat e valid (are >100 rânduri și nu e HTML)
_is_valid_csv() {
    local f="$1"
    [ -f "$f" ] || return 1
    local lines
    lines=$(wc -l < "$f" 2>/dev/null)
    [ "$lines" -gt 100 ] || return 1
    # verifică că prima linie nu e HTML
    head -1 "$f" | grep -qi "<!doctype\|<html\|<head\|error\|not found" && return 1
    return 0
}

# ── Sursa 1: Python datasets library (Hugging Face API — CEA MAI FIABILĂ) ────
echo "   ⬇️  Sursa 1: Hugging Face Python datasets API (Ehsanrs2/Forex_Factory_Calendar)..."
python3 - <<'PYEOF'
import sys, os
out = os.path.expanduser("~/Desktop/Aladin/data/historical_news.csv")
try:
    from datasets import load_dataset
    print("      📥 Conectare la HuggingFace Hub...")
    ds = load_dataset("Ehsanrs2/Forex_Factory_Calendar", split="train")
    import pandas as pd
    df = ds.to_pandas()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"      ✅ Salvat: {out} ({len(df):,} rânduri)")
    sys.exit(0)
except ImportError:
    print("      ⚠️  Pachetul 'datasets' lipsă — se instalează...")
    os.system("pip3 install datasets -q")
    try:
        from datasets import load_dataset
        ds = load_dataset("Ehsanrs2/Forex_Factory_Calendar", split="train")
        import pandas as pd
        df = ds.to_pandas()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)
        print(f"      ✅ Salvat: {out} ({len(df):,} rânduri)")
        sys.exit(0)
    except Exception as e2:
        print(f"      ❌ Eroare după instalare: {e2}")
        sys.exit(1)
except Exception as e:
    print(f"      ❌ Eroare HuggingFace datasets: {e}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ] && _is_valid_csv "$FF_OUT"; then
    LINES=$(wc -l < "$FF_OUT")
    SIZE=$(du -sh "$FF_OUT" | cut -f1)
    echo "   ✅ Descărcat: $FF_OUT ($LINES rânduri, $SIZE)"
    DOWNLOADED=true
else
    echo "   ⚠️  Sursa 1 eșuată sau date invalide. Încerc sursa 2..."
fi

# ── Sursa 2: mdeverna/economic_calendar (CSV, GitHub) ─────────────────────────
if [ "$DOWNLOADED" = false ]; then
    MDEV_URL="https://raw.githubusercontent.com/mdeverna/economic_calendar/master/data/calendar_raw.csv"
    echo "   ⬇️  Sursa 2: mdeverna/economic_calendar (GitHub)..."
    TMP_OUT="$DATA_DIR/_tmp_download.csv"
    if curl -L --progress-bar --max-time 60 -A "Mozilla/5.0" "$MDEV_URL" -o "$TMP_OUT" 2>/dev/null && _is_valid_csv "$TMP_OUT"; then
        mv "$TMP_OUT" "$FF_OUT"
        LINES=$(wc -l < "$FF_OUT")
        SIZE=$(du -sh "$FF_OUT" | cut -f1)
        echo "   ✅ Descărcat: $FF_OUT ($LINES rânduri, $SIZE)"
        DOWNLOADED=true
    else
        rm -f "$TMP_OUT"
        echo "   ⚠️  Sursa 2 eșuată sau date invalide. Încerc sursa 3..."
    fi
fi

# ── Sursa 3: HuggingFace direct URL cu token header ───────────────────────────
if [ "$DOWNLOADED" = false ]; then
    HF_URL="https://huggingface.co/datasets/Ehsanrs2/Forex_Factory_Calendar/resolve/main/ForexFactory.csv"
    echo "   ⬇️  Sursa 3: HuggingFace direct download..."
    TMP_OUT="$DATA_DIR/_tmp_download.csv"
    if curl -L --progress-bar --max-time 120 \
        -H "Accept: text/csv,application/csv,*/*" \
        -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
        "$HF_URL" -o "$TMP_OUT" 2>/dev/null && _is_valid_csv "$TMP_OUT"; then
        mv "$TMP_OUT" "$FF_OUT"
        LINES=$(wc -l < "$FF_OUT")
        SIZE=$(du -sh "$FF_OUT" | cut -f1)
        echo "   ✅ Descărcat: $FF_OUT ($LINES rânduri, $SIZE)"
        DOWNLOADED=true
    else
        rm -f "$TMP_OUT"
        echo "   ⚠️  Sursa 3 eșuată."
    fi
fi

# ── Rezultat final ─────────────────────────────────────────────────────────────
if [ "$DOWNLOADED" = true ] && [ -f "$FF_OUT" ]; then
    echo ""
    echo "   ✅ Știri economice descărcate cu succes!"
    # Arată primele rânduri pentru verificare
    echo "   📋 Preview (primele 3 rânduri):"
    head -4 "$FF_OUT" | sed 's/^/      /'
else
    echo ""
    echo "   ❌ Nicio sursă nu a funcționat automat."
    echo ""
    echo "   ─── DESCĂRCARE MANUALĂ ─────────────────────────────────────"
    echo "   Varianta 1 — Hugging Face (browser):"
    echo "   1. Deschide: https://huggingface.co/datasets/Ehsanrs2/Forex_Factory_Calendar"
    echo "   2. Files → ForexFactory.csv → Download"
    echo "   3. Mută fișierul în: ~/Desktop/Aladin/data/historical_news.csv"
    echo ""
    echo "   Varianta 2 — pip install datasets:"
    echo "   pip3 install datasets"
    echo "   python3 -c \""
    echo "   from datasets import load_dataset; import pandas as pd"
    echo "   ds = load_dataset('Ehsanrs2/Forex_Factory_Calendar', split='train')"
    echo "   ds.to_pandas().to_csv('$FF_OUT', index=False)"
    echo "   \""
    echo "   ─────────────────────────────────────────────────────────────"
fi

# ══════════════════════════════════════════════════════════
# 2. DATE ISTORICE NQ/ES — din NinjaTrader (Parallels VM)
# ══════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📈 Pasul 2: Date Istorice NQ / ES / BTC / XAU din NinjaTrader"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  ⚠️  NT8 rulează în Parallels (Windows VM) — nu poate fi automatizat."
echo "  Urmează pașii de mai jos MANUAL:"
echo ""
echo "  ─── PASUL A: Verifică/descarcă date istorice în NT8 ────────────"
echo "  1. Deschide NT8 în Parallels"
echo "  2. Tools → Historical Data Manager"
echo "  3. Pentru fiecare instrument (NQ, ES, XAUUSD, BTCUSD):"
echo "     → Dacă nu există: Add → caută instrumentul → Download"
echo "     → Settings: Type=1 Minute, From=01/01/2015, To=today"
echo "     → Apasă OK și așteaptă (poate dura 10-30 min per instrument)"
echo ""
echo "  ─── PASUL B: Export CSV din NT8 ────────────────────────────────"
echo "  1. Tools → Export → Data"
echo "  2. Setări:"
echo "     • Instrument: @NQ# (Continuous Contract — NU NQ 06-25!)"
echo "     • Data Series: 1 Minute"
echo "     • From: 01/01/2015  To: azi"
echo "     • Format: CSV"
echo "     • Output folder: C:\\Users\\Public\\AladinData\\"
echo "  3. Click Export și așteaptă"
echo "  4. Repetă pentru: @ES#, XAUUSD, BTCUSD"
echo ""
echo "  ⚠️  Folosește @NQ# nu NQ 06-25 pentru date continue fără gap-uri!"
echo ""
echo "  ─── PASUL C: Transfer Parallels → Mac ──────────────────────────"
echo ""
echo "  Metoda SIMPLĂ (Shared Folder):"
echo "  • În Windows Explorer: stânga → 'Questo Mac' (Network Drive)"
echo "  • Navighează la: \\\\Mac\\Home\\Desktop\\Aladin\\data\\"
echo "  • Copiază toate CSV-urile acolo"
echo ""
echo "  Sau din PowerShell (Windows):"
echo "  copy C:\\Users\\Public\\AladinData\\*.csv \"\\\\Mac\\Home\\Desktop\\Aladin\\data\\\""
echo ""

# ══════════════════════════════════════════════════════════
# 3. INSTALARE DEPENDENȚE PYTHON
# ══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════════"
echo "🐍 Pasul 3: Verificare/instalare dependențe Python..."
echo "═══════════════════════════════════════════════════════════════"
echo ""

PACKAGES="pandas numpy fastapi uvicorn httpx pydantic pyarrow datasets"
for pkg in $PACKAGES; do
    MODULE="${pkg//-/_}"
    if python3 -c "import $MODULE" 2>/dev/null; then
        echo "  ✅ $pkg"
    else
        echo "  ⬇️  Instalez $pkg..."
        if pip3 install "$pkg" -q 2>/dev/null; then
            echo "  ✅ $pkg instalat"
        else
            echo "  ⚠️  $pkg — eșuat (încearcă: pip3 install $pkg)"
        fi
    fi
done

# ══════════════════════════════════════════════════════════
# 4. VERIFICARE FINALĂ
# ══════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📁 Pasul 4: Status final fișiere"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Știri
if [ -f "$FF_OUT" ]; then
    LINES=$(wc -l < "$FF_OUT")
    echo "  ✅ Știri economice: $FF_OUT ($LINES rânduri)"
elif [ -f "$FF_OUT_JSON" ]; then
    echo "  ✅ Știri economice (JSON): $FF_OUT_JSON"
else
    echo "  ❌ Știri economice: LIPSĂ — fă descărcarea manuală din Pasul 1"
fi

# Date de piață
CSV_COUNT=$(ls "$DATA_DIR"/*.csv 2>/dev/null | grep -v historical_news | wc -l | tr -d ' ')
if [ "$CSV_COUNT" -gt 0 ]; then
    echo "  ✅ Date piață: $CSV_COUNT fișiere CSV găsite"
    ls "$DATA_DIR"/*.csv 2>/dev/null | grep -v historical_news | while read f; do
        LINES=$(wc -l < "$f" 2>/dev/null || echo "?")
        echo "     → $(basename $f): $LINES bare"
    done
else
    echo "  ⚠️  Date piață: LIPSĂ — exportă din NT8 (Pasul 2)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  DONE!                                               ║"
echo "║                                                      ║"
echo "║  După ce ai datele, pornește sistemul:              ║"
echo "║   python3 news_clustering.py   (test știri)         ║"
echo "║   python3 bridge_api.py        (server Mac)         ║"
echo "║   → apoi pornește AladinBridge.cs în NT8            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
