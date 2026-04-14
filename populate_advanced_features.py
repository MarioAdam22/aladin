"""
╔══════════════════════════════════════════════════════════════════════════╗
║  ALADIN — Populate Advanced Features on Existing DB                     ║
║  populate_advanced_features.py  (v3 — FAST vectorized)                 ║
║                                                                          ║
║  Loads entire market_data into RAM, computes features vectorized,       ║
║  writes back in one shot. ~10-15 min instead of 3+ hours.              ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
  cd ~/Desktop/Aladin
  python3 populate_advanced_features.py
"""

import sqlite3
import pandas as pd
import numpy as np
import time
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import advanced_features as af

DB_PATH = os.path.join(_script_dir, "mario_trading.db")

ADVANCED_COLS = [
    'hurst', 'garch_vol', 'kalman_smooth', 'kalman_noise', 'adx_14',
    'vwap', 'dist_vwap', 'sample_entropy', 'fisher_transform',
    'fft_cycle', 'acf_lag1', 'acf_lag5',
]


def main():
    print("=" * 70)
    print("  ALADIN — Populate Advanced Features (v3 FAST)")
    print("=" * 70)

    if not os.path.exists(DB_PATH):
        print(f"❌ DB not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH, timeout=60)
    cur = conn.cursor()

    # Ensure columns exist
    existing_cols = [r[1] for r in cur.execute("PRAGMA table_info(market_data)").fetchall()]
    for col in ADVANCED_COLS:
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE market_data ADD COLUMN {col} REAL DEFAULT 0")
    conn.commit()

    # ── Step 1: Load entire table into RAM ──────────────────────────────────
    total = cur.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
    print(f"\n📊 Total rows: {total:,}")
    print(f"📥 Loading entire market_data into RAM...")

    t0 = time.time()
    df = pd.read_sql_query(
        "SELECT rowid, timestamp, date, open, high, low, close, volume "
        "FROM market_data ORDER BY rowid",
        conn,
    )
    t_load = time.time() - t0
    print(f"   ✅ Loaded {len(df):,} rows in {t_load:.1f}s ({len(df)*8*8/1e6:.0f} MB est.)")

    # Convert types
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

    # ── Step 2: Compute ALL features vectorized (one pass) ──────────────────
    print(f"\n🧮 Computing 12 advanced features on {len(df):,} rows (vectorized)...")
    t1 = time.time()
    df = af.compute_all_advanced(df)
    t_compute = time.time() - t1
    print(f"   ✅ Features computed in {t_compute:.1f}s ({t_compute/60:.1f} min)")

    # ── Step 3: Write back to DB via temp table (fastest method) ────────────
    print(f"\n💾 Writing features back to DB...")
    t2 = time.time()

    # Create temp table with just rowid + advanced cols
    update_df = df[['rowid'] + ADVANCED_COLS].copy()
    for col in ADVANCED_COLS:
        update_df[col] = update_df[col].fillna(0).astype(float)

    # Drop temp table if exists from previous failed run
    cur.execute("DROP TABLE IF EXISTS _tmp_advanced")

    # Write to temp table
    update_df.to_sql('_tmp_advanced', conn, if_exists='replace', index=False)
    print(f"   ✅ Temp table written ({time.time()-t2:.1f}s)")

    # Update market_data from temp table in one SQL statement
    print(f"   🔄 Merging into market_data...")
    t3 = time.time()
    set_parts = ", ".join(f"{c} = _tmp_advanced.{c}" for c in ADVANCED_COLS)
    cur.execute(f"""
        UPDATE market_data
        SET {set_parts}
        FROM _tmp_advanced
        WHERE market_data.rowid = _tmp_advanced.rowid
    """)
    conn.commit()
    t_merge = time.time() - t3
    print(f"   ✅ Merged in {t_merge:.1f}s")

    # Cleanup
    cur.execute("DROP TABLE IF EXISTS _tmp_advanced")
    conn.commit()

    total_time = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"✅ Done! {total:,} rows updated in {total_time/60:.1f} min")
    print(f"   Load: {t_load:.0f}s | Compute: {t_compute:.0f}s | Write: {t_merge:.0f}s")
    print(f"{'=' * 70}")

    # Verification
    sample = pd.read_sql_query(
        "SELECT hurst, garch_vol, adx_14, dist_vwap, sample_entropy, acf_lag1 "
        "FROM market_data WHERE hurst != 0 ORDER BY rowid DESC LIMIT 5",
        conn,
    )
    if not sample.empty:
        print("\n🔍 Sample (last 5 rows):")
        print(sample.to_string(index=False))

    still_zero = cur.execute("SELECT COUNT(*) FROM market_data WHERE hurst = 0").fetchone()[0]
    print(f"\n📊 Rows still at 0: {still_zero:,} / {total:,}")

    conn.close()


if __name__ == "__main__":
    main()
