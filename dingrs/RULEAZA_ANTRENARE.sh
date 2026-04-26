#!/bin/bash
# ============================================================
# ALADIN v6.4 — Update #2: Reantrenare model XGBoost
# Ruleaza acest script din Terminal cu:
#   bash ~/Desktop/RULEAZA_ANTRENARE.sh
# ============================================================

echo ""
echo "============================================================"
echo "  ALADIN v6.4 — Reantrenare XGBoost (41 features)"
echo "============================================================"
echo ""

# Cauta python cu xgboost instalat
PYTHON=""
for PY in python3 python3.11 python3.10 python3.9 python; do
    if command -v $PY &>/dev/null; then
        if $PY -c "import xgboost" 2>/dev/null; then
            PYTHON=$PY
            echo "✅ Python gasit: $($PY --version) cu XGBoost $($PY -c 'import xgboost; print(xgboost.__version__)')"
            break
        fi
    fi
done

# Incearca si conda
if [ -z "$PYTHON" ]; then
    for CONDA_PY in ~/miniconda3/bin/python ~/anaconda3/bin/python ~/opt/anaconda3/bin/python; do
        if [ -f "$CONDA_PY" ]; then
            if $CONDA_PY -c "import xgboost" 2>/dev/null; then
                PYTHON=$CONDA_PY
                echo "✅ Conda Python gasit cu XGBoost"
                break
            fi
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    echo "❌ Nu s-a gasit Python cu XGBoost instalat."
    echo "   Instaleaza cu: pip3 install xgboost scikit-learn pandas numpy"
    exit 1
fi

echo ""
echo "🚀 Pornesc antrenarea... (poate dura 3-8 minute)"
echo "   Date: ~/Desktop/mario_trading.db (1.19M bare, 2019-2026)"
echo "   Model output: ~/Desktop/mario_bot.json"
echo "   Features output: ~/Desktop/mario_features.json"
echo ""

cd ~/Desktop
$PYTHON train_mario_ai.py

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================================"
    echo "✅ ANTRENARE COMPLETA!"
    echo "   mario_bot.json actualizat cu 41 features"
    echo "   mario_features.json actualizat"
    echo "   Update #2 bifat ✅"
    echo "============================================================"
else
    echo "❌ Antrenarea a esuat cu cod: $EXIT_CODE"
    echo "   Verifica erorile de mai sus"
fi
echo ""
