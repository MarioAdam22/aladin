"""
╔══════════════════════════════════════════════════════════════════════════╗
║  ALADIN WEEKLY RETRAINER v9.0                                          ║
║  Walk-forward retraining — rulează săptămânal (cron/manual)            ║
║                                                                         ║
║  Usage:                                                                 ║
║    python3 retrain_weekly.py              # retrain dacă e nevoie       ║
║    python3 retrain_weekly.py --force      # forțează retrain            ║
║    crontab: 0 6 * * 0 cd ~/Desktop/Aladin && python3 retrain_weekly.py ║
╚══════════════════════════════════════════════════════════════════════════╝

Walk-Forward Logic:
  1. Verifică dacă au trecut >= 7 zile de la ultimul retrain
  2. Verifică dacă sunt >= 500 bare noi în DB de la ultimul retrain
  3. Verifică performance-ul modelului curent (dacă accuracy a scăzut)
  4. Dacă oricare condiție e True → retrain cu train_mario_ai.py
  5. Salvează backup al modelului vechi înainte de overwrite
  6. Loghează rezultatele în retrain_history.json
"""

import os
import sys
import json
import shutil
import subprocess
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

ALADIN_DIR = Path.home() / "Desktop" / "Aladin"
MODEL_PATH = ALADIN_DIR / "mario_bot.json"
FEATURES_PATH = ALADIN_DIR / "mario_features.json"
DB_PATH = ALADIN_DIR / "nq_data.db"
RETRAIN_HISTORY = ALADIN_DIR / "retrain_history.json"
BACKUP_DIR = ALADIN_DIR / "model_backups"

# ── Configurare ──────────────────────────────────────────────────────────
MIN_DAYS_BETWEEN_RETRAIN = 7       # minim 7 zile între retrain-uri
MIN_NEW_BARS = 500                  # minim 500 bare noi (~ 8h de trading)
ACCURACY_DROP_THRESHOLD = 0.03     # retrain dacă accuracy a scăzut cu >3%
MAX_BACKUPS = 10                    # păstrăm max 10 backup-uri


def load_retrain_history() -> list:
    """Încarcă istoricul de retrain-uri."""
    if RETRAIN_HISTORY.exists():
        try:
            return json.loads(RETRAIN_HISTORY.read_text())
        except Exception:
            return []
    return []


def save_retrain_history(history: list):
    """Salvează istoricul de retrain-uri."""
    RETRAIN_HISTORY.write_text(json.dumps(history[-50:], indent=2))  # max 50 entries


def get_last_retrain_info(history: list) -> dict:
    """Returnează info despre ultimul retrain."""
    if history:
        return history[-1]
    return {"timestamp": "1970-01-01T00:00:00", "db_rows": 0, "accuracy": 0.0}


def count_db_rows() -> int:
    """Numără rândurile din DB."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.execute("SELECT COUNT(*) FROM market_data")
        count = cur.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"   ⚠️ DB read error: {e}")
        return 0


def get_current_model_accuracy() -> float:
    """Citește accuracy-ul modelului curent din features metadata."""
    try:
        if FEATURES_PATH.exists():
            meta = json.loads(FEATURES_PATH.read_text())
            return float(meta.get("accuracy", 0.0))
    except Exception:
        pass
    return 0.0


def backup_current_model():
    """Salvează backup al modelului curent."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for src in ALADIN_DIR.glob("mario_bot*"):
        dst = BACKUP_DIR / f"{ts}_{src.name}"
        shutil.copy2(src, dst)
        print(f"   📦 Backup: {src.name} → {dst.name}")

    # Cleanup old backups (keep MAX_BACKUPS newest)
    all_backups = sorted(BACKUP_DIR.glob("*mario_bot*"))
    if len(all_backups) > MAX_BACKUPS * 4:  # ~4 files per model
        for old in all_backups[:len(all_backups) - MAX_BACKUPS * 4]:
            old.unlink()
            print(f"   🗑️  Old backup removed: {old.name}")


def should_retrain(force: bool = False) -> tuple:
    """Decide dacă trebuie retrain. Returnează (should, reason)."""
    if force:
        return True, "FORCED by --force flag"

    history = load_retrain_history()
    last = get_last_retrain_info(history)

    # Check 1: Time since last retrain
    last_ts = datetime.fromisoformat(last["timestamp"])
    days_since = (datetime.now() - last_ts).days
    if days_since >= MIN_DAYS_BETWEEN_RETRAIN:
        return True, f"TIMER: {days_since} zile de la ultimul retrain (min={MIN_DAYS_BETWEEN_RETRAIN})"

    # Check 2: New data available
    current_rows = count_db_rows()
    last_rows = last.get("db_rows", 0)
    new_rows = current_rows - last_rows
    if new_rows >= MIN_NEW_BARS:
        return True, f"NEW DATA: {new_rows} bare noi (min={MIN_NEW_BARS})"

    # Check 3: Accuracy degradation (compared to last retrain)
    current_acc = get_current_model_accuracy()
    last_acc = last.get("accuracy", 0.0)
    if last_acc > 0 and current_acc > 0 and (last_acc - current_acc) > ACCURACY_DROP_THRESHOLD:
        return True, f"ACCURACY DROP: {last_acc:.4f} → {current_acc:.4f} (drop={last_acc - current_acc:.4f})"

    return False, f"NO RETRAIN NEEDED: {days_since}d ago, {new_rows} new bars, acc={current_acc:.4f}"


def run_retrain() -> dict:
    """Execută retrain-ul efectiv."""
    print("\n🚀 Starting retrain...")
    start_time = datetime.now()

    # Backup model vechi
    if MODEL_PATH.exists():
        backup_current_model()

    # Run train_mario_ai.py
    train_script = ALADIN_DIR / "train_mario_ai.py"
    if not train_script.exists():
        return {"success": False, "error": "train_mario_ai.py not found"}

    try:
        result = subprocess.run(
            [sys.executable, str(train_script)],
            cwd=str(ALADIN_DIR),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )

        duration = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            # Citim accuracy-ul nou
            new_acc = get_current_model_accuracy()
            print(f"\n   ✅ Retrain complet în {duration:.0f}s | Accuracy: {new_acc:.4f}")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return {
                "success": True,
                "accuracy": new_acc,
                "duration_s": round(duration, 1),
            }
        else:
            print(f"\n   ❌ Retrain FAILED (exit code {result.returncode})")
            print(f"   STDERR: {result.stderr[-300:]}")
            return {
                "success": False,
                "error": result.stderr[-200:],
                "duration_s": round(duration, 1),
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout (>10min)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    print("=" * 60)
    print("  ALADIN WEEKLY RETRAINER v9.0")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    force = "--force" in sys.argv

    should, reason = should_retrain(force)
    print(f"\n   📋 Decision: {reason}")

    if not should:
        print("   ✅ Modelul e up-to-date. Nicio acțiune necesară.")
        return

    # Run retrain
    result = run_retrain()

    # Log results
    history = load_retrain_history()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "db_rows": count_db_rows(),
        "accuracy": result.get("accuracy", 0.0),
        "success": result.get("success", False),
        "duration_s": result.get("duration_s", 0),
        "error": result.get("error", ""),
    }
    history.append(entry)
    save_retrain_history(history)

    if result["success"]:
        print(f"\n   🎉 Retrain SUCCESS — model actualizat!")
    else:
        print(f"\n   ❌ Retrain FAILED — modelul vechi păstrat (backup disponibil)")
        print(f"   Error: {result.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
