"""
compute_advanced_labels.py — Advanced Label Engine
====================================================
Computes path-based labels for each trade setup (LOM/NOM style):

  1. MFE  (Maximum Favorable Excursion, points)  — best outcome before exit
  2. MAE  (Maximum Adverse  Excursion, points)   — worst drawdown before exit
  3. time_to_MFE  (bars until MFE is reached)
  4. path_quality (MFE / (MFE + MAE), 0‒1; higher = cleaner path)
  5. Quantile targets P10/P25/P50/P75/P90 of MFE (for quantile regression)
  6. Survival labels:
       hit_tp   — 1 if TP reached before SL (same as binary label)
       hit_sl   — 1 if SL reached before TP
       time_to_event — bars until first of (TP or SL) is hit  (censored at LABEL_WINDOW)
       is_censored   — 1 if neither TP nor SL hit within window (censored event)

Usage (called from train_lom_v3.py / train_nom_v3.py):
    from compute_advanced_labels import compute_path_labels
    df_setups_with_labels = compute_path_labels(df_setups, conn, tp_pts, sl_pts, label_window, direction_col)

Standalone: generates advanced_labels_lom.parquet / advanced_labels_nom.parquet
  python compute_advanced_labels.py
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB   = Path(__file__).parent / "mario_trading.db"
OUT_LOM = Path(__file__).parent / "data" / "advanced_labels_lom.parquet"
OUT_NOM = Path(__file__).parent / "data" / "advanced_labels_nom.parquet"

LABEL_WINDOW   = 60   # max bars after entry (1-min bars)
TP_PT_LOM      = 18.0
TP_PT_NOM      = None   # NOM uses ATR×2; computed per trade from _entry_px / ATR
SL_MULT        = 0.6    # approx SL = TP × 0.6 (used if sl_pts not provided)


# ══════════════════════════════════════════════════════════════════════════════
# Core path computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_path_for_trade(
    conn,
    date_str: str,
    entry_ts: str,       # 'YYYY-MM-DD HH:MM:SS'
    entry_price: float,
    direction: str,      # 'LONG' or 'SHORT'
    tp_pts: float,       # TP distance in points
    sl_pts: float,       # SL distance in points
    label_window: int = LABEL_WINDOW,
) -> dict:
    """
    Fetch bars after entry and compute full path metrics.
    Returns dict with MFE, MAE, time_to_MFE, path_quality,
    hit_tp, hit_sl, time_to_event, is_censored.
    """
    # Fetch next label_window 1-min bars after entry
    q = f"""
        SELECT timestamp, high, low, close
        FROM market_data
        WHERE date = '{date_str}'
          AND timestamp > '{entry_ts}'
        ORDER BY timestamp
        LIMIT {label_window}
    """
    bars = pd.read_sql(q, conn)
    if len(bars) < 2:
        return _empty_path()

    highs  = bars['high'].values.astype(float)
    lows   = bars['low'].values.astype(float)

    if direction == 'LONG':
        fav  = highs - entry_price   # favorable = going up
        adv  = entry_price - lows    # adverse   = going down
        tp_lvl = entry_price + tp_pts
        sl_lvl = entry_price - sl_pts
    else:  # SHORT
        fav  = entry_price - lows    # favorable = going down
        adv  = highs - entry_price   # adverse   = going up
        tp_lvl = entry_price - tp_pts
        sl_lvl = entry_price + sl_pts

    fav = np.maximum(fav, 0.0)
    adv = np.maximum(adv, 0.0)

    mfe = float(np.max(fav))
    mae = float(np.max(adv))

    # time_to_MFE: bar index where MFE first reached
    mfe_idx_arr = np.where(fav >= mfe)[0]
    time_to_mfe = int(mfe_idx_arr[0]) + 1 if len(mfe_idx_arr) > 0 else label_window

    path_quality = mfe / (mfe + mae + 1e-6)   # 0=adversarial, 1=clean run

    # Survival: scan bar by bar for TP/SL hit
    hit_tp = 0; hit_sl = 0; time_to_event = label_window; is_censored = 1
    for i, (h, l) in enumerate(zip(highs, lows)):
        if direction == 'LONG':
            tp_hit = h >= tp_lvl
            sl_hit = l <= sl_lvl
        else:
            tp_hit = l <= tp_lvl
            sl_hit = h >= sl_lvl

        if tp_hit or sl_hit:
            time_to_event = i + 1
            is_censored = 0
            # If both hit same bar, conservative: attribute to SL first
            if sl_hit:
                hit_sl = 1
            else:
                hit_tp = 1
            break

    return {
        'mfe':            mfe,
        'mae':            mae,
        'time_to_mfe':    time_to_mfe,
        'path_quality':   path_quality,
        'mfe_mae_ratio':  mfe / (mae + 1e-6),
        'hit_tp':         hit_tp,
        'hit_sl':         hit_sl,
        'time_to_event':  time_to_event,
        'is_censored':    is_censored,
        # For quantile regression (MFE in ATR units computed later)
        'mfe_raw':        mfe,
        'mae_raw':        mae,
    }


def _empty_path() -> dict:
    return {
        'mfe': 0.0, 'mae': 0.0, 'time_to_mfe': 60,
        'path_quality': 0.5, 'mfe_mae_ratio': 1.0,
        'hit_tp': 0, 'hit_sl': 0, 'time_to_event': 60, 'is_censored': 1,
        'mfe_raw': 0.0, 'mae_raw': 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Batch computation — vectorized via per-date grouping
# ══════════════════════════════════════════════════════════════════════════════

def compute_path_labels(
    df_setups: pd.DataFrame,
    conn,
    tp_pts_col: str  = None,   # column name for TP pts; if None uses tp_pts_default
    sl_pts_col: str  = None,   # column name for SL pts; if None uses tp_pts × SL_MULT
    tp_pts_default: float = TP_PT_LOM,
    direction_col: str = '_direction',
    date_col: str = '_date',
    entry_ts_col: str = '_entry_ts',
    entry_px_col: str = '_entry_px',
    label_window: int = LABEL_WINDOW,
) -> pd.DataFrame:
    """
    Adds path-based label columns to df_setups.
    Processes in batches by date for efficiency.
    """
    import warnings; warnings.filterwarnings('ignore')

    path_rows = []
    n = len(df_setups)
    print(f"  compute_path_labels: {n} setups ...")

    for i, (_, row) in enumerate(df_setups.iterrows()):
        if i % 500 == 0:
            print(f"    [{i}/{n}] ...")

        date_str    = str(row[date_col])
        entry_ts    = str(row[entry_ts_col]) if entry_ts_col in row.index else \
                      f"{date_str} {int(row.get('_entry_hhmm', 900))//100:02d}:{int(row.get('_entry_hhmm', 900))%100:02d}:00"
        entry_price = float(row[entry_px_col])
        direction   = str(row[direction_col])

        tp_pts = float(row[tp_pts_col]) if tp_pts_col and tp_pts_col in row.index \
                 else tp_pts_default
        sl_pts = float(row[sl_pts_col]) if sl_pts_col and sl_pts_col in row.index \
                 else tp_pts * SL_MULT

        path = compute_path_for_trade(
            conn, date_str, entry_ts, entry_price, direction,
            tp_pts, sl_pts, label_window
        )
        path_rows.append(path)

    path_df = pd.DataFrame(path_rows, index=df_setups.index)

    # Compute quantile targets from empirical distribution of MFE in training
    # (quantiles are computed over the whole dataset passed in)
    mfe_vals = path_df['mfe_raw'].values
    for pct, qname in [(10, 'mfe_p10'), (25, 'mfe_p25'), (50, 'mfe_p50'),
                       (75, 'mfe_p75'), (90, 'mfe_p90')]:
        path_df[qname] = float(np.percentile(mfe_vals, pct))

    # MFE normalized by ATR (for training — needs atr column from setups)
    if 'atr_5d' in df_setups.columns:
        atr = df_setups['atr_5d'].values.clip(min=1.0)
        path_df['mfe_atr'] = path_df['mfe_raw'].values / atr
        path_df['mae_atr'] = path_df['mae_raw'].values / atr

    result = pd.concat([df_setups, path_df], axis=1)
    print(f"  Done. MFE stats: mean={mfe_vals.mean():.1f}pt, "
          f"p50={np.percentile(mfe_vals,50):.1f}pt, "
          f"p90={np.percentile(mfe_vals,90):.1f}pt")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Quantile regression target builder
# ══════════════════════════════════════════════════════════════════════════════

def add_quantile_targets(df: pd.DataFrame, mfe_col: str = 'mfe_raw') -> pd.DataFrame:
    """
    Adds per-trade MFE targets for quantile regression.
    The targets ARE the actual MFE values — quantile XGBoost learns to predict
    each quantile of the distribution directly.
    (distinct from the dataset-level percentiles added in compute_path_labels)
    """
    df = df.copy()
    # The actual MFE is the target for every quantile model
    # (XGBoost with reg:quantileerror + quantile_alpha=0.1..0.9)
    df['target_mfe'] = df[mfe_col].clip(lower=0.0)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Survival label builder (for Cox proportional hazards / Weibull)
# ══════════════════════════════════════════════════════════════════════════════

def add_survival_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares survival analysis targets:
      - duration:   time_to_event (bars until TP/SL hit, censored at LABEL_WINDOW)
      - event:      1 = TP hit (event of interest), 0 = SL hit or censored
      - event_any:  1 = any exit (TP or SL), 0 = censored
    """
    df = df.copy()
    # Survival: event = TP hit (what we want to happen)
    df['surv_duration'] = df['time_to_event'].clip(lower=1, upper=LABEL_WINDOW)
    df['surv_event']    = df['hit_tp'].astype(int)     # 1 = TP hit = "success"
    df['surv_any_exit'] = (df['hit_tp'] | df['hit_sl']).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Standalone: generate parquet files from existing LOM/NOM datasets
# ══════════════════════════════════════════════════════════════════════════════

def _rebuild_setups_lom(conn, years):
    """Reconstruct LOM setups for the given years (abridged, just entry info)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_lom_v2 import build_dataset
    df = build_dataset(years)
    return df


def _rebuild_setups_nom(conn, years):
    """Reconstruct NOM setups for the given years."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_nom_v2 import build_dataset
    df = build_dataset(years)
    return df


def generate_advanced_labels():
    """Standalone: rebuild datasets and compute advanced labels, save as parquet."""
    print("=" * 65)
    print("  ADVANCED LABEL ENGINE")
    print("=" * 65)

    OUT_LOM.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)

    for name, rebuild_fn, out_path, tp_pts_default in [
        ('LOM', _rebuild_setups_lom, OUT_LOM, TP_PT_LOM),
        ('NOM', _rebuild_setups_nom, OUT_NOM, 24.0),   # NOM ATR×2 ≈ 24pt
    ]:
        print(f"\n[{name}] Rebuilding setups ...")
        conn.close()
        df_all = rebuild_fn(None, [2023, 2024, 2025, 2026])
        conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)

        if df_all is None or len(df_all) == 0:
            print(f"  ⚠️  No setups for {name}, skipping")
            continue

        print(f"  {len(df_all)} setups → computing path labels ...")
        df_labeled = compute_path_labels(
            df_all, conn,
            tp_pts_default=tp_pts_default,
        )
        df_labeled = add_quantile_targets(df_labeled)
        df_labeled = add_survival_targets(df_labeled)

        # Keep only meta + path label columns
        path_cols = ['mfe', 'mae', 'time_to_mfe', 'path_quality', 'mfe_mae_ratio',
                     'hit_tp', 'hit_sl', 'time_to_event', 'is_censored',
                     'mfe_raw', 'mae_raw', 'mfe_p10', 'mfe_p25', 'mfe_p50',
                     'mfe_p75', 'mfe_p90', 'target_mfe',
                     'surv_duration', 'surv_event', 'surv_any_exit']
        meta_cols = [c for c in df_labeled.columns if c.startswith('_')]
        save_cols = meta_cols + [c for c in path_cols if c in df_labeled.columns]
        df_save = df_labeled[[c for c in save_cols if c in df_labeled.columns]]

        df_save.to_parquet(out_path, index=False)
        print(f"  ✅ Saved {len(df_save)} rows → {out_path}")
        print(f"     MFE mean={df_save['mfe'].mean():.1f}pt  "
              f"path_quality mean={df_save['path_quality'].mean():.2f}  "
              f"hit_tp rate={df_save['hit_tp'].mean():.1%}")

    conn.close()
    print("\nDone.")


if __name__ == '__main__':
    generate_advanced_labels()
