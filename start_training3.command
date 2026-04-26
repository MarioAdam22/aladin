#!/bin/bash
# start_training3.command — Task B + Task A + Task C
# ===================================================
# Task B: GAP_PENALTY retrain pentru restul modelelor MARGINAL (_v2)
#         v6 EXPANSION, ts_lon EXPANSION+RETRACEMENT,
#         ts_ny PRE_EXPANSION+EXPANSION+RETRACEMENT, ny_v3 PRE_EXPANSION+RETRACEMENT
#
# Task A: Regime classifier v2 cu 14 lagged OF features
#         (retrain independent — nu depinde de precompute dacă regime_labels e ok)
#
# Task C: Sweep ensemble QA — replay 2025 signals, compare old vs ensemble
#
# Toate 3 pornesc în paralel (B și A durează ~40 min, C ~2 min)

cd "$(dirname "$0")"
echo "=== Aladin Training3 Launcher ==="
echo "Working dir: $(pwd)"
echo "Started at: $(date)"
echo ""

# ── Task C: Sweep QA (rapid, ~2 min) ─────────────────────────────────────────
echo "[1/3] Task C — Sweep Ensemble QA ..."
mkdir -p qa
nohup python3 qa_sweep_ensemble.py > /tmp/sweep_ensemble_qa.log 2>&1 &
PID_QA=$!
echo "  → PID: $PID_QA | Log: /tmp/sweep_ensemble_qa.log"
echo ""

# ── Task B: retrain_all_marginal_v2.py (Optuna ~40 min) ──────────────────────
echo "[2/3] Task B — Retrain all marginal models _v2 ..."
nohup python3 retrain_all_marginal_v2.py > /tmp/retrain_marginal_v2.log 2>&1 &
PID_B=$!
echo "  → PID: $PID_B | Log: /tmp/retrain_marginal_v2.log"
echo ""

# ── Task A: train_regime_v2.py (XGBoost ~5 min) ──────────────────────────────
echo "[3/3] Task A — Regime classifier v2 (14 lagged OF features) ..."
nohup python3 train_regime_v2.py > /tmp/regime_v2_train.log 2>&1 &
PID_A=$!
echo "  → PID: $PID_A | Log: /tmp/regime_v2_train.log"
echo ""

# ── Save PIDs ─────────────────────────────────────────────────────────────────
cat > /tmp/training3_pids.txt << PIDS
SWEEP_QA=$PID_QA
RETRAIN_MARGINAL_V2=$PID_B
REGIME_V2=$PID_A
STARTED=$(date)
PIDS

echo "=== Toate procesele pornite ==="
echo "SWEEP_QA          PID=$PID_QA"
echo "RETRAIN_MARGINAL  PID=$PID_B"
echo "REGIME_V2         PID=$PID_A"
echo ""

# ── Wait 20s then check ───────────────────────────────────────────────────────
echo "Verificare după 20 secunde..."
sleep 20
echo ""
echo "--- Status procese ---"
for pid in $PID_QA $PID_B $PID_A; do
    if kill -0 $pid 2>/dev/null; then
        echo "PID $pid → ✅ RUNNING"
    else
        echo "PID $pid → ❌ STOPPED (sau terminat rapid)"
    fi
done

echo ""
echo "--- Prime linii logs ---"
echo ">> sweep_ensemble_qa.log:"
head -15 /tmp/sweep_ensemble_qa.log 2>/dev/null || echo "(gol)"
echo ""
echo ">> retrain_marginal_v2.log:"
head -10 /tmp/retrain_marginal_v2.log 2>/dev/null || echo "(gol)"
echo ""
echo ">> regime_v2_train.log:"
head -10 /tmp/regime_v2_train.log 2>/dev/null || echo "(gol)"

echo ""
echo "=== Monitoring (Ctrl+C nu opreste procesele): ==="
echo "  tail -f /tmp/retrain_marginal_v2.log"
echo "  tail -f /tmp/regime_v2_train.log"
echo "  cat qa/sweep_ensemble_qa.md"
