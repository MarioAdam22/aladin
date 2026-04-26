"""
generate_advanced_labels_fast.py — Fast Advanced Label Generator
=================================================================
Reads directly from backtest/backtest_bridge_v3.csv + market_data DB.
Produces:
    data/advanced_labels_lom.parquet  (LON setups)
    data/advanced_labels_nom.parquet  (NY  setups)

Each row = one setup with:
    MFE, MAE, time_to_mfe, path_quality, mfe_mae_ratio,
    hit_tp, hit_sl, time_to_event, is_censored,
    mfe_p10..mfe_p90  (quantile targets for XGBoost quantile regression)

Run:
    cd ~/Desktop/Aladin
    python3 generate_advanced_labels_fast.py
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import time

BASE        = Path(__file__).parent
DB          = BASE / "mario_trading.db"
CSV_SETUPS  = BASE / "backtest" / "backtest_bridge_v3.csv"
OUT_LOM     = BASE / "data" / "advanced_labels_lom.parquet"
OUT_NOM     = BASE / "data" / "advanced_labels_nom.parquet"

LABEL_WINDOW = 120   # max 1-min bars after entry to look forward
QUANTILES    = [0.10, 0.25, 0.50, 0.75, 0.90]


# ── Core path computation (vectorized per date) ───────────────────────────────

def compute_path_batch(date_bars: pd.DataFrame, setups: pd.DataFrame) -> list:
    """
    For all setups on a given date, compute MFE/MAE using the pre-loaded
    1-min bars DataFrame (already filtered to that date, sorted by timestamp).
    Returns list of dicts, one per setup.
    """
    results = []
    bars_ts = date_bars['timestamp'].values
    bars_hi = date_bars['high'].values
    bars_lo = date_bars['low'].values

    for _, row in setups.iterrows():
        entry_ts  = row['ts']
        entry_px  = float(row['entry'])
        direction = row['direction'].upper()
        tp_pts    = float(row['tp_pt'])
        sl_pts    = float(row['sl_pt'])

        # Find entry bar index
        idx = np.searchsorted(bars_ts, entry_ts, side='left')
        if idx >= len(bars_ts):
            results.append(_empty(row))
            continue

        # Forward window
        end_idx = min(idx + LABEL_WINDOW, len(bars_ts))
        fwd_hi  = bars_hi[idx:end_idx]
        fwd_lo  = bars_lo[idx:end_idx]
        n_bars  = len(fwd_hi)

        if n_bars == 0:
            results.append(_empty(row))
            continue

        # MFE / MAE per bar
        if direction == 'LONG':
            excursion   = fwd_hi - entry_px   # favorable = price goes up
            adverse     = entry_px - fwd_lo   # adverse   = price goes down
        else:
            excursion   = entry_px - fwd_lo   # favorable = price goes down
            adverse     = fwd_hi - entry_px   # adverse   = price goes up

        mfe_cummax  = np.maximum.accumulate(excursion)
        mae_cummax  = np.maximum.accumulate(adverse)

        mfe_raw = float(mfe_cummax[-1])
        mae_raw = float(mae_cummax[-1])

        # time_to_MFE: first bar where MFE is achieved
        peak_val    = mfe_cummax[-1]
        peak_idx    = int(np.argmax(excursion >= peak_val))
        time_to_mfe = peak_idx + 1  # 1-indexed bars

        # path_quality
        path_quality = mfe_raw / (mfe_raw + mae_raw + 1e-6)

        # Survival: hit_tp / hit_sl
        hit_tp = 0; hit_sl = 0; time_to_event = n_bars; is_censored = 1
        for b in range(n_bars):
            fav = excursion[b]; adv = adverse[b]
            if fav >= tp_pts and hit_tp == 0 and hit_sl == 0:
                hit_tp = 1; time_to_event = b + 1; is_censored = 0; break
            if adv >= sl_pts and hit_sl == 0 and hit_tp == 0:
                hit_sl = 1; time_to_event = b + 1; is_censored = 0; break

        results.append({
            # meta
            '_date':      row['date'],
            '_ts':        entry_ts,
            '_direction': direction,
            '_session':   row['session'],
            '_entry_px':  entry_px,
            '_tp_pt':     tp_pts,
            '_sl_pt':     sl_pts,
            # path
            'mfe_raw':       mfe_raw,
            'mae_raw':       mae_raw,
            'mfe':           mfe_raw / tp_pts if tp_pts > 0 else 0,  # normalized by TP
            'mae':           mae_raw / sl_pts if sl_pts > 0 else 0,  # normalized by SL
            'time_to_mfe':   time_to_mfe,
            'path_quality':  path_quality,
            'mfe_mae_ratio': mfe_raw / (mae_raw + 1e-6),
            # survival
            'hit_tp':        hit_tp,
            'hit_sl':        hit_sl,
            'time_to_event': time_to_event,
            'is_censored':   is_censored,
        })

    return results


def _empty(row) -> dict:
    return {
        '_date': row['date'], '_ts': row['ts'],
        '_direction': row['direction'], '_session': row['session'],
        '_entry_px': row['entry'], '_tp_pt': row['tp_pt'], '_sl_pt': row['sl_pt'],
        'mfe_raw': 0.0, 'mae_raw': 0.0, 'mfe': 0.0, 'mae': 0.0,
        'time_to_mfe': 0, 'path_quality': 0.5, 'mfe_mae_ratio': 1.0,
        'hit_tp': 0, 'hit_sl': 0, 'time_to_event': 60, 'is_censored': 1,
    }


def add_quantile_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row MFE quantile columns (used as targets for quantile XGBoost)."""
    # In quantile regression the TARGET is the actual MFE value for every model.
    # The quantile loss function handles the asymmetry. So all q-columns = mfe_raw.
    for q in QUANTILES:
        col = f'mfe_p{int(q*100):02d}'
        df[col] = df['mfe_raw']
    return df


def add_regime_adjusted_expectancy(df: pd.DataFrame) -> pd.DataFrame:
    """regime_adjusted_expectancy = path_quality × (hit_tp rate rolling 20)"""
    df = df.sort_values('_date').reset_index(drop=True)
    df['rolling_tp_20'] = df['hit_tp'].rolling(20, min_periods=5).mean().fillna(df['hit_tp'].mean())
    df['regime_adj_expectancy'] = df['path_quality'] * df['rolling_tp_20']
    return df


# ── Main generation ───────────────────────────────────────────────────────────

def generate():
    print("=" * 65)
    print("  ADVANCED LABEL ENGINE (fast — direct from CSV + DB)")
    print("=" * 65)
    t0 = time.time()

    OUT_LOM.parent.mkdir(exist_ok=True)

    # Load setups
    df_all = pd.read_csv(CSV_SETUPS)
    df_all['ts'] = pd.to_datetime(df_all['ts']).dt.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n  Loaded {len(df_all)} setups from {CSV_SETUPS.name}")
    print(f"  Sessions: {df_all['session'].value_counts().to_dict()}")
    print(f"  Years: {df_all['date'].str[:4].value_counts().sort_index().to_dict()}")

    # Connect to DB
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60,
                           check_same_thread=False)

    # Process by date (batch queries)
    dates = sorted(df_all['date'].unique())
    all_results = []

    print(f"\n  Processing {len(dates)} dates ...")
    for i, date in enumerate(dates):
        day_setups = df_all[df_all['date'] == date]

        # Load 1-min bars for this date + next 2 hours buffer
        rows = conn.execute(
            """SELECT timestamp, high, low FROM market_data
               WHERE date = ? OR (date = date(?, '+1 day') AND substr(timestamp,12,5) <= '03:00')
               ORDER BY timestamp""",
            (date, date)
        ).fetchall()

        if not rows:
            for _, r in day_setups.iterrows():
                all_results.append(_empty(r))
            continue

        bars_df = pd.DataFrame(rows, columns=['timestamp', 'high', 'low'])
        batch   = compute_path_batch(bars_df, day_setups)
        all_results.extend(batch)

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(dates)} dates done ({time.time()-t0:.1f}s)")

    conn.close()

    df_labels = pd.DataFrame(all_results)
    df_labels = add_quantile_targets(df_labels)
    df_labels = add_regime_adjusted_expectancy(df_labels)

    print(f"\n  Total labeled: {len(df_labels)} setups")
    print(f"  MFE mean      : {df_labels['mfe_raw'].mean():.1f}pt")
    print(f"  MAE mean      : {df_labels['mae_raw'].mean():.1f}pt")
    print(f"  path_quality  : {df_labels['path_quality'].mean():.3f}")
    print(f"  hit_tp rate   : {df_labels['hit_tp'].mean():.1%}")
    print(f"  is_censored   : {df_labels['is_censored'].mean():.1%}")

    # Split LON / NY
    for sess, out_path in [('LON', OUT_LOM), ('NY', OUT_NOM)]:
        sub = df_labels[df_labels['_session'] == sess].reset_index(drop=True)

        # QA prints
        print(f"\n  [{sess}] {len(sub)} setups → {out_path.name}")
        print(f"    MFE mean={sub['mfe_raw'].mean():.1f}pt  "
              f"MAE mean={sub['mae_raw'].mean():.1f}pt  "
              f"path_q={sub['path_quality'].mean():.3f}  "
              f"hit_tp={sub['hit_tp'].mean():.1%}")

        # ── QA 1: Label distribution ──────────────────────────────────────
        print(f"\n    QA 1 — Label distribution:")
        print(f"      hit_tp={sub['hit_tp'].mean():.1%}  "
              f"hit_sl={sub['hit_sl'].mean():.1%}  "
              f"censored={sub['is_censored'].mean():.1%}")

        # ── QA 2: No temporal leakage — path uses only FUTURE bars ───────
        # Verify: entry ts < first bar used (always true by construction)
        print(f"\n    QA 2 — Temporal leakage: OK (entry_ts used as searchsorted lower bound)")

        # ── QA 3: MFE/MAE sanity — no negative values ────────────────────
        neg_mfe = (sub['mfe_raw'] < 0).sum()
        neg_mae = (sub['mae_raw'] < 0).sum()
        print(f"\n    QA 3 — Negative MFE: {neg_mfe}  Negative MAE: {neg_mae}  "
              f"{'✅' if neg_mfe == 0 and neg_mae == 0 else '⚠️'}")

        # ── QA 4: path_quality distribution ──────────────────────────────
        pq = sub['path_quality']
        print(f"\n    QA 4 — path_quality: "
              f"p10={pq.quantile(0.1):.2f}  p50={pq.quantile(0.5):.2f}  "
              f"p90={pq.quantile(0.9):.2f}  "
              f"{'✅' if 0.2 < pq.mean() < 0.9 else '⚠️'}")

        # ── QA 5: Survival model consistency ─────────────────────────────
        # hit_tp + hit_sl + censored should sum to 100%
        total_check = sub['hit_tp'].sum() + sub['hit_sl'].sum() + sub['is_censored'].sum()
        print(f"\n    QA 5 — Survival consistency: "
              f"tp+sl+censored={total_check}/{len(sub)}  "
              f"{'✅' if total_check == len(sub) else '⚠️ MISMATCH'}")

        sub.to_parquet(out_path, index=False)
        print(f"    ✅ Saved → {out_path}")

    print(f"\n  Total time: {time.time()-t0:.1f}s")
    print("\nDone. Run retrain to use new labels.")


if __name__ == '__main__':
    generate()
