#!/bin/bash
# pipeline_chain.sh — Runs sequentially: backfill done? → precompute → retrain _v2
# Called AFTER backfill_adx has been launched as PID $1

BACKFILL_PID=$1
ALADIN_DIR="$(dirname "$0")"
cd "$ALADIN_DIR"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a /tmp/pipeline_chain.log; }

log "=== Pipeline chain started (waiting for backfill PID=$BACKFILL_PID) ==="

# ── Wait for backfill to finish ─────────────────────────────────────────────
if [ -n "$BACKFILL_PID" ]; then
    log "Waiting for backfill PID=$BACKFILL_PID ..."
    while kill -0 $BACKFILL_PID 2>/dev/null; do
        sleep 10
    done
    log "✅ Backfill done."
    tail -5 /tmp/backfill_adx.log >> /tmp/pipeline_chain.log
fi

# ── Precompute regimes (now that ADX is filled) ──────────────────────────────
log "[STEP 2] precompute_regimes.py ..."
python3 precompute_regimes.py > /tmp/regime_precompute3.log 2>&1
PRECOMPUTE_EXIT=$?
log "Precompute exit=$PRECOMPUTE_EXIT"
tail -5 /tmp/regime_precompute3.log >> /tmp/pipeline_chain.log

if [ $PRECOMPUTE_EXIT -ne 0 ]; then
    log "❌ Precompute failed — aborting retrain chain"
    exit 1
fi

# ── Verify regime_labels extended past Apr 8 ──────────────────────────────
LAST_DATE=$(python3 -c "
import pandas as pd
df = pd.read_parquet('data/regime_labels.parquet')
print(df['date'].max())
" 2>/dev/null)
log "regime_labels max date after precompute: $LAST_DATE"

# ── Retrain v6 RETRACEMENT _v2 + ny_v3 EXPANSION _v2 (parallel) ─────────────
log "[STEP 3a] retrain_v6_retracement_v2.py ..."
nohup python3 retrain_v6_retracement_v2.py > /tmp/v6_retracement_v2.log 2>&1 &
PID_V6=$!
log "  v6 RETRACEMENT _v2 PID=$PID_V6"

log "[STEP 3b] retrain_ny_v3_expansion_v2.py ..."
nohup python3 retrain_ny_v3_expansion_v2.py > /tmp/ny_v3_expansion_v2.log 2>&1 &
PID_NY=$!
log "  ny_v3 EXPANSION _v2 PID=$PID_NY"

echo "V6_RETRACEMENT_V2=$PID_V6" >> /tmp/training2_pids.txt
echo "NY_V3_EXPANSION_V2=$PID_NY" >> /tmp/training2_pids.txt

# Wait for both retrains
log "Waiting for v6+ny_v3 retrains ..."
wait $PID_V6 $PID_NY
log "✅ Both _v2 retrains done."

# Final check — did PKLs appear?
log "=== PKL check ==="
for f in mario_quality_v6_RETRACEMENT_v2_calibrated.pkl mario_quality_ny_v3_EXPANSION_v2_calibrated.pkl; do
    if [ -f "$f" ]; then
        log "  ✅ $f ($(du -h "$f" | cut -f1))"
    else
        log "  ❌ $f MISSING"
    fi
done

log "=== Pipeline chain COMPLETE ==="
