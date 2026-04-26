#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ALADIN — Training secvențial COMPLET cu Synthetic Order Flow features  ║
# ║  Rulează din Terminal: bash RUN_TRAINING_ORDERFLOW.sh                   ║
# ║                                                                          ║
# ║  Scripturi incluse (în ordine):                                          ║
# ║  1. train_quality_v6.py        (LON quality gate v6)                    ║
# ║  2. train_quality_ts_lon_v1.py (LON quality gate ts)                    ║
# ║  3. train_quality_ny_v3.py     (NY quality gate v3)                     ║
# ║  4. train_quality_ts_ny_v1.py  (NY quality gate ts)                     ║
# ║  5. train_lom_v2.py            (London Open Manipulation)               ║
# ║  6. train_nom_v2.py            (NY Open Manipulation)                   ║
# ║  7. train_dsm.py               (Double Sweep Model)                     ║
# ║  8. train_sweep_unified.py     (Sweep Unified per Regime)               ║
# ║  9. train_scorer_v4_1.py       (ICT Setup Scorer v4.1)                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# IMPORTANT: NU rula în paralel — fiecare script folosește 2-4GB RAM

ALADIN_DIR="$HOME/Desktop/Aladin"
export PYTHONPATH="$ALADIN_DIR:${PYTHONPATH:-}"
LOG_DIR="$ALADIN_DIR/logs/training_orderflow"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="$LOG_DIR/summary_${TIMESTAMP}.log"

echo "======================================================" | tee "$SUMMARY_LOG"
echo "  ALADIN — Training Secvențial COMPLET cu Order Flow" | tee -a "$SUMMARY_LOG"
echo "  $(date)" | tee -a "$SUMMARY_LOG"
echo "======================================================" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# ─── Verificare prerequisite ──────────────────────────────────────────────
if [ ! -f "$ALADIN_DIR/data/orderflow_features.parquet" ]; then
    echo "❌ orderflow_features.parquet lipsă!" | tee -a "$SUMMARY_LOG"
    echo "   Rulează mai întâi: python3 $ALADIN_DIR/generate_synthetic_orderflow.py"
    exit 1
fi

if [ ! -f "$ALADIN_DIR/data/NQ_continuous.parquet" ]; then
    echo "❌ NQ_continuous.parquet lipsă!" | tee -a "$SUMMARY_LOG"
    echo "   Rulează mai întâi: python3 $ALADIN_DIR/stitch_continuous_nq.py"
    exit 1
fi

echo "✅ Prerequisite OK" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# ─── Contoare ─────────────────────────────────────────────────────────────
TOTAL=0
PASSED=0
FAILED=0

# ─── Funcție training ─────────────────────────────────────────────────────
run_training() {
    local script="$1"
    local name="$2"
    local logfile="$LOG_DIR/${name}_${TIMESTAMP}.log"
    TOTAL=$((TOTAL + 1))

    echo "──────────────────────────────────────────────────" | tee -a "$SUMMARY_LOG"
    echo "[$TOTAL/9] $name" | tee -a "$SUMMARY_LOG"
    echo "  Script: $script" | tee -a "$SUMMARY_LOG"
    echo "  Start:  $(date)" | tee -a "$SUMMARY_LOG"
    echo ""

    python3 "$ALADIN_DIR/$script" 2>&1 | tee "$logfile"
    STATUS=${PIPESTATUS[0]}

    echo ""
    if [ $STATUS -eq 0 ]; then
        PASSED=$((PASSED + 1))
        echo "  ✅ DONE ($(date))" | tee -a "$SUMMARY_LOG"
        # Extrage AUC + OF info din log
        grep -E "AUC|OOS|IS =|gap|Order flow|order flow|features|EXPANSION|RETRACEMENT|Salvat|pkl" \
            "$logfile" | tail -15 | sed 's/^/     /' | tee -a "$SUMMARY_LOG"
    else
        FAILED=$((FAILED + 1))
        echo "  ❌ EROARE (exit $STATUS)" | tee -a "$SUMMARY_LOG"
        grep -E "Error|Traceback|error" "$logfile" | tail -5 | sed 's/^/     /' | tee -a "$SUMMARY_LOG"
    fi
    echo "" | tee -a "$SUMMARY_LOG"
}

# ─── Training secvențial — TOATE 9 scripturi ─────────────────────────────
echo "Starting training secvențial (9 modele, ~40-80 minute total)..." | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# Quality Gates (LON)
run_training "train/train_quality_v6.py"        "v6_LON"
run_training "train/train_quality_ts_lon_v1.py" "ts_lon_LON"

# Quality Gates (NY)
run_training "train/train_quality_ny_v3.py"     "ny_v3_NY"
run_training "train/train_quality_ts_ny_v1.py"  "ts_ny_NY"

# Manipulation Models
run_training "train_lom_v2.py"                  "LOM_v2"
run_training "train_nom_v2.py"                  "NOM_v2"
run_training "train_dsm.py"                     "DSM"

# Sweep + Scorer
run_training "train_sweep_unified.py"           "sweep_unified"
run_training "train/train_scorer_v4_1.py"       "scorer_v4_1"

# ─── Sumar training ───────────────────────────────────────────────────────
echo "======================================================" | tee -a "$SUMMARY_LOG"
echo "  SUMAR TRAINING" | tee -a "$SUMMARY_LOG"
echo "  Total: $TOTAL | ✅ $PASSED | ❌ $FAILED" | tee -a "$SUMMARY_LOG"
echo "======================================================" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# ─── QA Automat ──────────────────────────────────────────────────────────
echo "Rulez QA automat..." | tee -a "$SUMMARY_LOG"
python3 "$ALADIN_DIR/qa_orderflow_models.py" 2>&1 | tee -a "$SUMMARY_LOG"

echo "" | tee -a "$SUMMARY_LOG"
echo "Log complet: $SUMMARY_LOG" | tee -a "$SUMMARY_LOG"
echo "FINALIZAT: $(date)" | tee -a "$SUMMARY_LOG"
