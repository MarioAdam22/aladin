#!/bin/bash
# ╔══════════════════════════════════════════════════════╗
# ║  ALADIN — Start Bridge                              ║
# ║  Usage: bash ~/Desktop/Aladin/start.sh              ║
# ╚══════════════════════════════════════════════════════╝

ALADIN_DIR="$HOME/Desktop/Aladin"
LOG_FILE="$ALADIN_DIR/bridge.log"

echo "======================================================"
echo "  ALADIN BRIDGE — START $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

cd "$ALADIN_DIR"

# Oprește orice instanță anterioară
pkill -f "bridge_api.py" 2>/dev/null
sleep 1

# stdout (print) + stderr (logging) → ambele în bridge.log
nohup python3 -u "$ALADIN_DIR/bridge_api.py" >> "$LOG_FILE" 2>&1 &
BRIDGE_PID=$!

sleep 2
echo "✅ Bridge pornit (PID=$BRIDGE_PID)"
echo "📄 Log: tail -f $LOG_FILE"
echo "======================================================"
