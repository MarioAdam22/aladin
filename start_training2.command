#!/bin/bash
# start_training2.command — Backfill ADX + Retrain v6 RETRACEMENT _v2 + ny_v3 EXPANSION _v2
# Task 1: Fix overfit cu GAP_PENALTY=4.0

cd "$(dirname "$0")"
echo "=== Aladin Training2 Launcher ==="
echo "Working dir: $(pwd)"
echo "Started at: $(date)"
echo ""

# 1) Backfill ADX pentru Apr 9-22 (necesar pentru precompute_regimes)
echo "[1/4] Backfill ADX Apr 2026 ..."
nohup python3 backfill_adx_apr2026.py > /tmp/backfill_adx.log 2>&1
echo "  → Backfill terminat. Log: /tmp/backfill_adx.log"
echo ""
tail -5 /tmp/backfill_adx.log
echo ""

# 2) Re-run precompute_regimes (acum cu date până la Apr 22)
echo "[2/4] Precompute regimes (update la Apr 22) ..."
nohup python3 precompute_regimes.py > /tmp/regime_precompute2.log 2>&1 &
PID_REG=$!
echo "  → PID: $PID_REG | Log: /tmp/regime_precompute2.log"

# 3) Retrain v6 RETRACEMENT _v2 (GAP_PENALTY=4.0)
echo "[3/4] Retrain v6 RETRACEMENT v2 ..."
nohup python3 retrain_v6_retracement_v2.py > /tmp/v6_retracement_v2.log 2>&1 &
PID_V6=$!
echo "  → PID: $PID_V6 | Log: /tmp/v6_retracement_v2.log"

# 4) Retrain ny_v3 EXPANSION _v2 (GAP_PENALTY=4.0)
echo "[4/4] Retrain ny_v3 EXPANSION v2 ..."
nohup python3 retrain_ny_v3_expansion_v2.py > /tmp/ny_v3_expansion_v2.log 2>&1 &
PID_NY=$!
echo "  → PID: $PID_NY | Log: /tmp/ny_v3_expansion_v2.log"

# 5) Regime classifier retrain (independent — can run in parallel with everything)
echo "[5/6] Regime classifier retrain (OF lagged, extins la Apr 22) ..."
nohup python3 train_regime.py > /tmp/regime_retrain.log 2>&1 &
PID_REGIME=$!
echo "  → PID: $PID_REGIME | Log: /tmp/regime_retrain.log"

# 6) Stacking retrain cu LOM v4 + NOM v4 (independent)
echo "[6/6] Stacking retrain (LOM v4 + NOM v4) ..."
nohup python3 train_stacking_advanced.py > /tmp/stacking_retrain.log 2>&1 &
PID_STACK=$!
echo "  → PID: $PID_STACK | Log: /tmp/stacking_retrain.log"

echo ""
echo "=== Toate procesele pornite ==="
echo "REGIME_PRECOMPUTE PID=$PID_REG"
echo "V6_RETRACEMENT_V2 PID=$PID_V6"
echo "NY_V3_EXPANSION_V2 PID=$PID_NY"
echo "REGIME_RETRAIN    PID=$PID_REGIME"
echo "STACKING_RETRAIN  PID=$PID_STACK"
echo ""

cat > /tmp/training2_pids.txt << PIDS
REGIME_PRECOMPUTE=$PID_REG
V6_RET_V2=$PID_V6
NY_V3_EXP_V2=$PID_NY
REGIME_RETRAIN=$PID_REGIME
STACKING=$PID_STACK
STARTED=$(date)
PIDS

echo "Verificare după 15 secunde..."
sleep 15
echo ""
echo "--- Status procese ---"
for pid in $PID_REG $PID_V6 $PID_NY $PID_REGIME $PID_STACK; do
    if kill -0 $pid 2>/dev/null; then
        echo "PID $pid → ✅ RUNNING"
    else
        echo "PID $pid → ❌ STOPPED (sau terminat rapid)"
    fi
done

echo ""
echo "--- Prime linii logs ---"
echo ">> v6_retracement_v2.log:"
head -10 /tmp/v6_retracement_v2.log 2>/dev/null || echo "(gol)"
echo ""
echo ">> ny_v3_expansion_v2.log:"
head -10 /tmp/ny_v3_expansion_v2.log 2>/dev/null || echo "(gol)"
echo ""
echo ">> regime_precompute2.log:"
head -5 /tmp/regime_precompute2.log 2>/dev/null || echo "(gol)"

echo ""
echo "=== Monitoring: Ctrl+C nu opreste procesele (nohup). ==="
echo "Verifică progresul cu: tail -f /tmp/v6_retracement_v2.log"
