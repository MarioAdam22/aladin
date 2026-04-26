#!/bin/bash
# start_training.command — LOM v4 + NOM v4 + precompute_regimes
# Sweep ensemble deja terminat ✅

cd "$(dirname "$0")"
echo "=== Aladin Training Launcher (LOM + NOM + Regimes) ==="
echo "Working dir: $(pwd)"
echo "Started at: $(date)"
echo ""

# 1) LOM v4 (3 modele, 80 trials)
echo "[1/3] Starting train_lom_v4.py 3 80 ..."
nohup python3 train_lom_v4.py 3 80 > /tmp/lom_v4.log 2>&1 &
PID_LOM=$!
echo "  → PID: $PID_LOM | Log: /tmp/lom_v4.log"

# 2) NOM v4 (3 modele, 80 trials)
echo "[2/3] Starting train_nom_v4.py 3 80 ..."
nohup python3 train_nom_v4.py 3 80 > /tmp/nom_v4.log 2>&1 &
PID_NOM=$!
echo "  → PID: $PID_NOM | Log: /tmp/nom_v4.log"

# 3) Precompute regimes (actualizare la Apr 25, 2026)
echo "[3/3] Starting precompute_regimes.py ..."
nohup python3 precompute_regimes.py > /tmp/regime_precompute.log 2>&1 &
PID_REG=$!
echo "  → PID: $PID_REG | Log: /tmp/regime_precompute.log"

echo ""
echo "=== Toate procesele pornite ==="
echo "LOM    PID=$PID_LOM"
echo "NOM    PID=$PID_NOM"
echo "REGIME PID=$PID_REG"
echo ""

# Salvez PID-urile
cat > /tmp/training_pids.txt << PIDS
LOM_V4=$PID_LOM
NOM_V4=$PID_NOM
REGIME=$PID_REG
STARTED=$(date)
PIDS

echo "Verificare după 8 secunde..."
sleep 8
echo ""
echo "--- Status procese ---"
for pid in $PID_LOM $PID_NOM $PID_REG; do
    if kill -0 $pid 2>/dev/null; then
        echo "PID $pid → ✅ RUNNING"
    else
        echo "PID $pid → ❌ STOPPED"
    fi
done

echo ""
echo "--- Prime linii logs ---"
echo ">> lom_v4.log:"
head -8 /tmp/lom_v4.log 2>/dev/null || echo "(gol)"
echo ">> nom_v4.log:"
head -8 /tmp/nom_v4.log 2>/dev/null || echo "(gol)"
echo ">> regime_precompute.log:"
head -8 /tmp/regime_precompute.log 2>/dev/null || echo "(gol)"

echo ""
echo "=== Monitoring activ. Ctrl+C nu opreste procesele (nohup). ==="
