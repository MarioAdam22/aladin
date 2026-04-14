#!/usr/bin/env python3
"""
BACKFILL FEATURES — populează FVG, SMT, displacement, advanced features în mario_trading.db
Rulează o singură dată (sau la nevoie) pe toți 4M bare.

Calculează din OHLCV (nu necesită tick data):
  1. FVG Bullish/Bearish (Fair Value Gap — gap între high/low pe 3 bare consecutive)
  2. SMT Bearish/Bullish (fake breakout: new high/low dar close revine)
  3. has_displacement (range > 1.5×ATR = candelă instituțională)
  4. Advanced features pe BRIDGE_LIVE rows: hurst, vwap, adx_14, garch_vol, etc.

Usage:
  python3 backfill_features.py              # full backfill
  python3 backfill_features.py --from 2026  # doar din 2026
"""

import sqlite3
import numpy as np
import sys
import time

DB_PATH = "mario_trading.db"

def main():
    from_year = None
    if "--from" in sys.argv:
        idx = sys.argv.index("--from")
        if idx + 1 < len(sys.argv):
            from_year = sys.argv[idx + 1]

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cur = conn.cursor()

    where = f"WHERE timestamp >= '{from_year}-01-01'" if from_year else ""

    # ══════════════════════════════════════════════════════════════════════
    # 0. ADD OF COLUMNS (if missing) — ALTER TABLE safe (ignore if exists)
    # ══════════════════════════════════════════════════════════════════════
    print("═══ 0. ENSURE OF COLUMNS EXIST ═══")
    of_columns = [
        ("bar_delta",       "REAL DEFAULT 0"),
        ("cum_delta",       "REAL DEFAULT 0"),
        ("bar_buy_vol",     "REAL DEFAULT 0"),
        ("bar_sell_vol",    "REAL DEFAULT 0"),
        ("delta_at_high",   "REAL DEFAULT 0"),
        ("delta_at_low",    "REAL DEFAULT 0"),
        ("big_buy_count",   "INTEGER DEFAULT 0"),
        ("big_sell_count",  "INTEGER DEFAULT 0"),
        ("imbalance_pct",   "REAL DEFAULT 0"),
        ("tape_speed",      "REAL DEFAULT 0"),
        ("dom_bid_total",   "INTEGER DEFAULT 0"),
        ("dom_ask_total",   "INTEGER DEFAULT 0"),
        ("dom_ratio",       "REAL DEFAULT 1.0"),
        ("vwap_live",       "REAL DEFAULT 0"),
        ("rvol",            "REAL DEFAULT 1.0"),
        ("profile_shape_enc", "INTEGER DEFAULT 0"),
        ("delta_exhaust_enc", "INTEGER DEFAULT 0"),
        ("dist_prev_poc",   "REAL DEFAULT 0"),
        ("absorption_score","REAL DEFAULT 0"),
        ("absorption_side", "TEXT DEFAULT ''"),
        ("stacked_bull",    "INTEGER DEFAULT 0"),
        ("stacked_bear",    "INTEGER DEFAULT 0"),
        ("of_doi",          "REAL DEFAULT 0"),  # delta oscillation index
        ("of_bilateral_abs","INTEGER DEFAULT 0"),
        ("of_big_balance",  "REAL DEFAULT 0.5"),
        ("of_d_shape_count","INTEGER DEFAULT 0"),
    ]
    existing = {r[1] for r in cur.execute("PRAGMA table_info(market_data)").fetchall()}
    added = 0
    for col_name, col_def in of_columns:
        if col_name not in existing:
            try:
                cur.execute(f"ALTER TABLE market_data ADD COLUMN {col_name} {col_def}")
                added += 1
                print(f"  + {col_name} ({col_def})")
            except Exception as e:
                print(f"  skip {col_name}: {e}")
    conn.commit()
    print(f"  ✅ {added} new columns added ({len(of_columns) - added} already existed)")

    # ── Count total rows ──
    cur.execute(f"SELECT COUNT(*) FROM market_data {where}")
    total = cur.fetchone()[0]
    print(f"\nTotal rows to process: {total:,}")

    # ══════════════════════════════════════════════════════════════════════
    # 1. FVG BACKFILL
    # ══════════════════════════════════════════════════════════════════════
    print("\n═══ 1. FVG BACKFILL ═══")
    t0 = time.time()

    # Load all OHLC data ordered by timestamp
    cur.execute(f"""
        SELECT rowid, high, low, atr_14 FROM market_data
        {where}
        ORDER BY timestamp ASC
    """)
    rows = cur.fetchall()
    print(f"  Loaded {len(rows):,} rows in {time.time()-t0:.1f}s")

    fvg_up_updates = []
    fvg_down_updates = []
    disp_updates = []
    batch_size = 50000

    for i in range(2, len(rows)):
        rowid = rows[i][0]
        hi_0 = rows[i-2][1]  # bar[i-2] high
        lo_0 = rows[i-2][2]  # bar[i-2] low
        hi_2 = rows[i][1]    # bar[i] high (current)
        lo_2 = rows[i][2]    # bar[i] low (current)
        atr  = rows[i][3] or 0

        # FVG Bullish: low[current] > high[2 bars ago] — gap up
        fvg_up = 1 if (lo_2 > hi_0 and lo_2 - hi_0 > 0.25) else 0
        # FVG Bearish: high[current] < low[2 bars ago] — gap down
        fvg_down = 1 if (hi_2 < lo_0 and lo_0 - hi_2 > 0.25) else 0

        if fvg_up:
            fvg_up_updates.append((fvg_up, rowid))
        if fvg_down:
            fvg_down_updates.append((fvg_down, rowid))

        # Displacement: bar range > 1.5 × ATR (candelă instituțională)
        bar_range = hi_2 - lo_2
        if atr > 0 and bar_range > 1.5 * atr:
            disp_updates.append((1, rowid))

    # Batch update FVG
    print(f"  FVG Bullish: {len(fvg_up_updates):,} gaps found")
    print(f"  FVG Bearish: {len(fvg_down_updates):,} gaps found")
    print(f"  Displacement: {len(disp_updates):,} bars found")

    cur.executemany("UPDATE market_data SET fvg_up = ? WHERE rowid = ?", fvg_up_updates)
    cur.executemany("UPDATE market_data SET fvg_down = ? WHERE rowid = ?", fvg_down_updates)
    cur.executemany("UPDATE market_data SET has_displacement = ? WHERE rowid = ?", disp_updates)
    conn.commit()
    print(f"  ✅ FVG + displacement backfill done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # 2. SMT BACKFILL (internal divergence — no ES needed)
    # ══════════════════════════════════════════════════════════════════════
    print("\n═══ 2. SMT BACKFILL ═══")
    t0 = time.time()

    # SMT lookback: 25 bars
    SMT_LOOKBACK = 25
    smt_bear_updates = []
    smt_bull_updates = []

    for i in range(SMT_LOOKBACK, len(rows)):
        rowid    = rows[i][0]
        cur_high = rows[i][1]
        cur_low  = rows[i][2]
        # Close not in our query — need it. We'll do a second pass.
        pass

    # Need close too — reload with close
    cur.execute(f"""
        SELECT rowid, high, low, close FROM market_data
        {where}
        ORDER BY timestamp ASC
    """)
    rows_smt = cur.fetchall()
    print(f"  Loaded {len(rows_smt):,} rows for SMT")

    for i in range(SMT_LOOKBACK, len(rows_smt)):
        rowid     = rows_smt[i][0]
        cur_high  = rows_smt[i][1]
        cur_low   = rows_smt[i][2]
        cur_close = rows_smt[i][3]
        ref_high  = rows_smt[i - SMT_LOOKBACK][1]
        ref_low   = rows_smt[i - SMT_LOOKBACK][2]

        # SMT Bearish: new high above ref BUT close below ref_high → fake breakout
        if cur_high > ref_high and cur_close < ref_high:
            smt_bear_updates.append((1, rowid))
        # SMT Bullish: new low below ref BUT close above ref_low → fake breakdown
        elif cur_low < ref_low and cur_close > ref_low:
            smt_bull_updates.append((1, rowid))

    print(f"  SMT Bearish: {len(smt_bear_updates):,} signals found")
    print(f"  SMT Bullish: {len(smt_bull_updates):,} signals found")

    cur.executemany("UPDATE market_data SET is_smt_bearish = ? WHERE rowid = ?", smt_bear_updates)
    cur.executemany("UPDATE market_data SET is_smt_bullish = ? WHERE rowid = ?", smt_bull_updates)
    conn.commit()
    print(f"  ✅ SMT backfill done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # 3. ADVANCED FEATURES pe BRIDGE_LIVE rows
    # ══════════════════════════════════════════════════════════════════════
    print("\n═══ 3. ADVANCED FEATURES (BRIDGE_LIVE) ═══")
    t0 = time.time()

    # Check how many BRIDGE_LIVE rows need filling
    cur.execute("SELECT COUNT(*) FROM market_data WHERE source = 'BRIDGE_LIVE' AND vwap = 0")
    bridge_empty = cur.fetchone()[0]
    print(f"  BRIDGE_LIVE rows needing features: {bridge_empty:,}")

    if bridge_empty > 0:
        # Load surrounding context for each BRIDGE_LIVE row
        # We need 100 bars before each BRIDGE_LIVE row for rolling calculations
        cur.execute("""
            SELECT rowid, timestamp, open, high, low, close, volume
            FROM market_data
            WHERE source = 'BRIDGE_LIVE' AND vwap = 0
            ORDER BY timestamp ASC
        """)
        bridge_rows = cur.fetchall()

        # For each BRIDGE_LIVE row, get 100 preceding bars for context
        updates = []
        for br in bridge_rows:
            rid, ts = br[0], br[1]

            cur.execute("""
                SELECT open, high, low, close, volume
                FROM market_data
                WHERE timestamp <= ?
                ORDER BY timestamp DESC LIMIT 100
            """, (ts,))
            context = cur.fetchall()

            if len(context) < 20:
                continue

            # Arrays (most recent first, reverse for chronological)
            closes  = np.array([r[3] for r in reversed(context)], dtype=float)
            highs   = np.array([r[1] for r in reversed(context)], dtype=float)
            lows    = np.array([r[2] for r in reversed(context)], dtype=float)
            volumes = np.array([r[4] for r in reversed(context)], dtype=float)

            # VWAP (session — use all available context)
            typical = (highs + lows + closes) / 3.0
            cum_tp_vol = np.cumsum(typical * volumes)
            cum_vol    = np.cumsum(volumes)
            vwap_val   = float(cum_tp_vol[-1] / cum_vol[-1]) if cum_vol[-1] > 0 else 0.0
            dist_vwap  = float(closes[-1] - vwap_val) if vwap_val > 0 else 0.0

            # ADX(14)
            adx_val = _compute_adx(highs, lows, closes, 14)

            # Hurst exponent (R/S method, last 50 bars)
            hurst_val = _compute_hurst(closes[-50:]) if len(closes) >= 50 else 0.5

            # Simple entropy (price returns)
            if len(closes) >= 20:
                rets = np.diff(np.log(closes[-20:]))
                rets = rets[np.isfinite(rets)]
                if len(rets) > 5:
                    hist, _ = np.histogram(rets, bins=10, density=True)
                    hist = hist[hist > 0]
                    entropy_val = float(-np.sum(hist * np.log(hist + 1e-10)))
                else:
                    entropy_val = 0.0
            else:
                entropy_val = 0.0

            updates.append((
                round(vwap_val, 2),
                round(dist_vwap, 2),
                round(adx_val, 2),
                round(hurst_val, 4),
                round(entropy_val, 4),
                rid
            ))

        if updates:
            cur.executemany("""
                UPDATE market_data SET
                    vwap = ?, dist_vwap = ?, adx_14 = ?,
                    hurst = ?, sample_entropy = ?
                WHERE rowid = ?
            """, updates)
            conn.commit()
            print(f"  ✅ Updated {len(updates)} BRIDGE_LIVE rows in {time.time()-t0:.1f}s")
    else:
        print("  ✅ No BRIDGE_LIVE rows need filling")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n═══ SUMMARY ═══")
    for col in ['fvg_up', 'fvg_down', 'is_smt_bearish', 'is_smt_bullish', 'has_displacement']:
        cur.execute(f"SELECT COUNT(*) FROM market_data WHERE {col} != 0 AND {col} IS NOT NULL")
        cnt = cur.fetchone()[0]
        print(f"  {col:20s}: {cnt:>8,} non-zero")

    cur.execute("SELECT COUNT(*) FROM market_data WHERE source = 'BRIDGE_LIVE' AND vwap != 0")
    print(f"  {'BRIDGE_LIVE w/vwap':20s}: {cur.fetchone()[0]:>8,}")

    conn.close()
    print("\n✅ BACKFILL COMPLETE")


def _compute_adx(highs, lows, closes, period=14):
    """ADX(14) simplified computation."""
    n = len(closes)
    if n < period + 1:
        return 0.0
    try:
        tr = np.maximum(highs[1:] - lows[1:],
                        np.maximum(np.abs(highs[1:] - closes[:-1]),
                                   np.abs(lows[1:] - closes[:-1])))
        up_move   = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Wilder smoothing
        atr_arr  = np.zeros(len(tr))
        pdi_arr  = np.zeros(len(tr))
        mdi_arr  = np.zeros(len(tr))

        atr_arr[period-1]  = np.mean(tr[:period])
        pdi_arr[period-1]  = np.mean(plus_dm[:period])
        mdi_arr[period-1]  = np.mean(minus_dm[:period])

        for i in range(period, len(tr)):
            atr_arr[i] = (atr_arr[i-1] * (period-1) + tr[i]) / period
            pdi_arr[i] = (pdi_arr[i-1] * (period-1) + plus_dm[i]) / period
            mdi_arr[i] = (mdi_arr[i-1] * (period-1) + minus_dm[i]) / period

        atr_nz = np.where(atr_arr > 0, atr_arr, 1.0)
        plus_di  = 100 * pdi_arr / atr_nz
        minus_di = 100 * mdi_arr / atr_nz

        di_sum = plus_di + minus_di
        di_sum = np.where(di_sum > 0, di_sum, 1.0)
        dx = 100 * np.abs(plus_di - minus_di) / di_sum

        # ADX = smoothed DX
        adx = np.zeros(len(dx))
        start = 2 * period - 1
        if start < len(dx):
            adx[start] = np.mean(dx[period:start+1])
            for i in range(start+1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period

        return float(adx[-1]) if len(adx) > 0 else 0.0
    except:
        return 0.0


def _compute_hurst(prices):
    """Hurst exponent via R/S method."""
    try:
        n = len(prices)
        if n < 20:
            return 0.5
        returns = np.diff(np.log(prices))
        returns = returns[np.isfinite(returns)]
        if len(returns) < 10:
            return 0.5

        max_k = min(len(returns) // 2, 30)
        if max_k < 4:
            return 0.5

        rs_list = []
        ns_list = []
        for k in range(4, max_k + 1):
            rs_vals = []
            for start in range(0, len(returns) - k + 1, k):
                chunk = returns[start:start+k]
                mean_c = np.mean(chunk)
                dev = np.cumsum(chunk - mean_c)
                r = np.max(dev) - np.min(dev)
                s = np.std(chunk, ddof=1)
                if s > 0:
                    rs_vals.append(r / s)
            if rs_vals:
                rs_list.append(np.log(np.mean(rs_vals)))
                ns_list.append(np.log(k))

        if len(rs_list) >= 3:
            coeffs = np.polyfit(ns_list, rs_list, 1)
            return float(np.clip(coeffs[0], 0.0, 1.0))
        return 0.5
    except:
        return 0.5


if __name__ == "__main__":
    main()
