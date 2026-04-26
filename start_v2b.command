#!/bin/bash
# start_v2b.command — Re-run over-regularized models with GAP_PENALTY=2.5
# =========================================================================
# Modele: ts_lon/EXPANSION, ts_ny/PRE_EXP+EXP+RET, ny_v3/RETRACEMENT
# Naming: {prefix}_{regime}_v2_calibrated.pkl (corect pt quality_gate_live.py)

cd "$(dirname "$0")"
echo "=== Aladin retrain_v2b Launcher ==="
echo "Working dir: $(pwd)"
echo "Started at: $(date)"
echo "GAP_PENALTY = 2.5"
echo ""

mkdir -p qa

echo "[1/1] retrain_v2b.py (ts_lon + ts_ny + ny_v3, ~25-35 min) ..."
nohup python3 retrain_v2b.py > /tmp/retrain_v2b.log 2>&1 &
PID_B=$!
echo "  → PID: $PID_B | Log: /tmp/retrain_v2b.log"
echo ""

echo "$PID_B" > /tmp/retrain_v2b.pid

echo "Verificare după 20 secunde..."
sleep 20

if kill -0 $PID_B 2>/dev/null; then
    echo "PID $PID_B → ✅ RUNNING"
else
    echo "PID $PID_B → ❌ STOPPED (eroare la start?)"
fi

echo ""
echo "--- Prime linii log ---"
head -20 /tmp/retrain_v2b.log 2>/dev/null || echo "(gol)"

echo ""
echo "=== Monitor: tail -f /tmp/retrain_v2b.log ==="
