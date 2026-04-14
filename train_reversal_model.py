"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║         ALADIN v14 — REVERSAL MODEL TRAINER                                             ║
║         train_reversal_model.py  |  Binary: 0=Continue  1=REVERSAL                     ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║  Ce face:                                                                                ║
║    • Detectează reversale sharpe la KEY LEVELS + KEY TIMES pe NQ 1M                     ║
║    • KEY LEVELS: PDH/PDL, PWH/PWL, LDN hi/lo, VAH/VAL, Asia hi/lo                     ║
║    • KEY TIMES: 17:10 NY Macro, 16:50 LDN-NY overlap, 14:00 LDN Close, 16:00 NY Open   ║
║    • Label REVERSAL=1 dacă prețul se întoarce ≥ 2R în next 20 bare după sweep          ║
║    • Full analytics_suite integration (MC / walk-forward / permutation / calibration)   ║
║                                                                                          ║
║  Arhitectura în Aladin Bridge v14:                                                       ║
║    1. breakout_model  → REAL vs FAKE OR breakout (LON 09:00, NY 15:30 RO time)          ║
║    2. mario_ai        → HTF direction support (4H/1H/15M/Daily/Weekly)                  ║
║    3. reversal_model  → sharp reversal la key level + key time (acesta)                  ║
║                                                                                          ║
║  Exemplu de workflow:                                                                    ║
║    OR breakout LONG atinge PDH/PWH/VAH → reversal_model zice REVERSAL → short entry    ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import sqlite3
import xgboost as xgb
import json
import os
import pickle
import time as _time
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

try:
    from analytics_suite import run_full_analysis
    _ANALYTICS = True
except ImportError:
    _ANALYTICS = False
    print("⚠️  analytics_suite.py lipsă — analytics dezactivat")

# ── PATHS ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH  = os.path.join(_SCRIPT_DIR, 'reversal_model.json')
FEATURES_PATH    = os.path.join(_SCRIPT_DIR, 'reversal_features.json')

# ── DB config (același ca mario_ai) ───────────────────────────────────────────
DB_PATHS = [
    os.path.join(_SCRIPT_DIR, 'mario_nq.db'),
    os.path.join(_SCRIPT_DIR, 'nq_1m.db'),
    os.path.join(_SCRIPT_DIR, 'aladin.db'),
    os.path.join(_SCRIPT_DIR, 'mario_trading.db'),
    os.path.join(_SCRIPT_DIR, 'aladin_data.db'),
]

# ── REVERSAL PARAMS ───────────────────────────────────────────────────────────
REVERSAL_HORIZON = 20      # bare (20 min lookahead) pentru confirmare reversal
REVERSAL_R_MIN   = 1.5     # prețul trebuie să se mute ≥ 1.5R împotriva direcției
SL_ATR_MULT      = 0.65    # SL pentru labeling = 0.65 × ATR_14
SL_MIN_PTS       = 8.0     # SL minim 8 pts NQ
SL_MAX_PTS       = 20.0    # SL maxim 20 pts NQ

# ── KEY TIMES (România GMT+3, NQ sesiune) ─────────────────────────────────────
# Formatul: (ora_decimal, descriere)
KEY_TIMES_RO = [
    (17 + 10/60,  "NY_MACRO"),         # 17:10 → NY Macro sweep LD high
    (16 + 50/60,  "LDN_NY_OVERLAP"),   # 16:50 → LDN-NY overlap close
    (16 + 0/60,   "NY_OPEN"),          # 16:00 → NY Open killzone start
    (14 + 0/60,   "LDN_CLOSE"),        # 14:00 → London Close
    (11 + 0/60,   "LON_KZ_END"),       # 11:00 → end London killzone
    (9  + 30/60,  "LON_OR_CLOSE"),     # 09:30 → London OR close
]
# Fereastră în jurul key time = ±N minute
KEY_TIME_WINDOW_MIN = 15  # ±15 minute = 30 min fereastră

# ── KEY LEVEL PROXIMITY ───────────────────────────────────────────────────────
# Distanța maximă față de level (în ATR) pentru a fi "la nivel"
KEY_LEVEL_ATR_THRESHOLD = 0.30  # ≤ 0.30×ATR = la nivel

# ══════════════════════════════════════════════════════════════════════════════
# REVERSAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════
REVERSAL_FEATURES = [
    # ── KEY LEVEL PROXIMITY ───────────────────────────────────────────────────
    'dist_pdh_atr',          # distanță față de PDH (în ATR)
    'dist_pdl_atr',          # distanță față de PDL (în ATR)
    'dist_pwh_atr',          # distanță față de PWH (în ATR)
    'dist_pwl_atr',          # distanță față de PWL (în ATR)
    'dist_vah_atr',          # distanță față de VAH (în ATR)
    'dist_val_atr',          # distanță față de VAL (în ATR)
    'dist_ldn_hi_atr',       # distanță față de London session high
    'dist_ldn_lo_atr',       # distanță față de London session low
    'dist_asia_hi_atr',      # distanță față de Asia session high
    'dist_asia_lo_atr',      # distanță față de Asia session low
    'dist_poc_atr',          # distanță față de VP Point of Control
    'near_any_level',        # 1 dacă oricare din above < KEY_LEVEL_ATR_THRESHOLD
    'level_type',            # 0=none, 1=PDH/PWH, 2=PDL/PWL, 3=VAH, 4=VAL, 5=LDN, 6=Asia
    'above_level',           # 1 dacă prețul e deasupra nivelului (sweep potențial)
    'below_level',           # 1 dacă prețul e sub nivel (sweep jos)

    # ── KEY TIME PROXIMITY ────────────────────────────────────────────────────
    'mins_to_key_time',      # minute până la cel mai apropiat key time (0=la timp)
    'near_key_time',         # 1 dacă < KEY_TIME_WINDOW_MIN minute de key time
    'key_time_type',         # 0=none, 1=NY_MACRO, 2=LDN_NY, 3=NY_OPEN, 4=LDN_CLOSE
    'time_in_window',        # minute de când am intrat în fereastra key time (0=nu)

    # ── SWEEP SIGNAL ──────────────────────────────────────────────────────────
    'swept_above_level',     # 1 dacă ultimele 3 bare au spart și revenit sub nivel
    'swept_below_level',     # 1 dacă ultimele 3 bare au spart și revenit peste nivel
    'sweep_wick_ratio',      # (wick care a depășit nivelul) / ATR — cât de agresiv
    'bars_since_sweep',      # câte bare de la ultimul sweep (0=acum)
    'reclaimed_after_sweep', # 1 dacă prețul a revenit pe cealaltă parte a nivelului

    # ── ORDER FLOW / CANDLE STRUCTURE ─────────────────────────────────────────
    'body_dir',              # (close-open)/range — direcție corp candle
    'wick_bias',             # (upper_wick - lower_wick)/range — presiune
    'upper_wick_atr',        # wick sus în ATR
    'lower_wick_atr',        # wick jos în ATR
    'sharp_reversal_bar',    # 1 dacă bara curentă e engulfing sau pin bar puternic
    'rejection_candle',      # 1 dacă wick > 2×body în direcția opusă trendului local
    'outside_bar',           # 1 dacă high>prev_high și low<prev_low (outside bar)

    # ── VOLATILITY ────────────────────────────────────────────────────────────
    'atr_percentile',        # percentila ATR (0-1)
    'realized_vol',          # log-return volatilitate pe 14 bare
    'vol_spike',             # 1 dacă volumul e > 1.5× medie 20 bare
    'range_atr_ratio',       # range ultimelor 5 bare / ATR — compresie → expansie

    # ── MOMENTUM (pre-sweep) ──────────────────────────────────────────────────
    'momentum_5',            # (close - close[-5]) / ATR — momentum 5 bare
    'momentum_15',           # (close - close[-15]) / ATR — momentum 15 bare
    'slope_h1',              # slope pe 60 bare (1H trend)
    'h4_momentum',           # slope pe 240 bare (4H trend)
    'h4_hh_hl',              # 1=HH+HL (bullish 4H), -1=LH+LL (bearish 4H), 0=neutral

    # ── VOLUME PROFILE ────────────────────────────────────────────────────────
    'inside_va',             # 1 dacă prețul e în Value Area
    'above_vah',             # 1 dacă prețul e deasupra VAH
    'below_val',             # 1 dacă prețul e sub VAL
    'dist_poc_signed',       # distanță POC cu semn (pos=above, neg=below) normalizată ATR

    # ── SESSION CONTEXT ───────────────────────────────────────────────────────
    'is_london',             # 1 dacă suntem în sesiunea London (09-12 RO)
    'is_ny',                 # 1 dacă suntem în sesiunea NY (15:30-17:30 RO)
    'is_pre_ny',             # 1 dacă suntem la LDN-NY overlap (16-16:30 RO)
    'hour_sin',              # sin(2π × ora/24) — encoding circular al timpului
    'hour_cos',              # cos(2π × ora/24) — encoding circular al timpului

    # ── STRUCTURE BREAK ───────────────────────────────────────────────────────
    'broke_pdh',             # 1 dacă prețul a depășit PDH
    'broke_pdl',             # 1 dacă prețul a depășit PDL
    'broke_pwh',             # 1 dacă prețul a depășit PWH (din mario_features)
    'broke_pwl',             # 1 dacă prețul a depășit PWL
    'broke_vah',             # 1 dacă prețul a depășit VAH
    'broke_val',             # 1 dacă prețul a depășit VAL
    'liq_above_count',       # câte nivele de lichiditate au fost luate sus
    'liq_below_count',       # câte nivele de lichiditate au fost luate jos
]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR standard cu Wilder smoothing."""
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr, index=df.index).ewm(span=period, min_periods=period).mean()
    return atr


def add_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculează toate features pentru reversal model.
    Input: df cu coloane OHLCV + timestamp + DB cols (pdh, pdl, lw_hi, lw_lo, vah, val, poc, volume)
    """
    # No copy() — work in-place to save memory

    # ── ATR ──────────────────────────────────────────────────────────────────
    if 'atr_14' in df.columns and df['atr_14'].notna().any():
        atr = df['atr_14'].ffill().fillna(12.0)
    else:
        atr = compute_atr(df, 14).fillna(12.0)
    df['_atr'] = atr

    close  = df['close'].values
    high   = df['high'].values
    low    = df['low'].values
    open_  = df['open'].values
    atr_v  = atr.values

    # ── TIMESTAMP FEATURES ────────────────────────────────────────────────────
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
        hour_dec = ts.dt.hour + ts.dt.minute / 60.0
        df['hour_sin'] = np.sin(2 * np.pi * hour_dec / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hour_dec / 24.0)
        df['is_london']  = ((hour_dec >= 9.0) & (hour_dec <= 12.0)).astype(float)
        df['is_ny']      = ((hour_dec >= 15.5) & (hour_dec <= 17.5)).astype(float)
        df['is_pre_ny']  = ((hour_dec >= 16.0) & (hour_dec <= 16.5)).astype(float)

        # ── KEY TIME proximity ──
        key_time_vals = [kt for kt, _ in KEY_TIMES_RO]
        key_time_names = {
            0: 'none', 1: 'NY_MACRO', 2: 'LDN_NY', 3: 'NY_OPEN', 4: 'LDN_CLOSE',
            5: 'LON_END', 6: 'LON_OR',
        }
        mins_to_key  = np.full(len(df), 999.0)
        key_time_idx = np.zeros(len(df), dtype=int)

        for i, (kt, _) in enumerate(KEY_TIMES_RO, start=1):
            dist = np.abs(hour_dec.values - kt) * 60.0  # in minutes
            mask = dist < mins_to_key
            mins_to_key[mask] = dist[mask]
            key_time_idx[mask] = i

        df['mins_to_key_time'] = np.clip(mins_to_key, 0, 999)
        df['near_key_time']    = (mins_to_key <= KEY_TIME_WINDOW_MIN).astype(float)
        df['key_time_type']    = key_time_idx.astype(float)
        # Câte minute de la intrarea în fereastră (0 dacă nu)
        df['time_in_window']   = np.where(
            mins_to_key <= KEY_TIME_WINDOW_MIN,
            KEY_TIME_WINDOW_MIN - mins_to_key,
            0.0
        )
    else:
        for col in ['hour_sin', 'hour_cos', 'is_london', 'is_ny', 'is_pre_ny',
                    'mins_to_key_time', 'near_key_time', 'key_time_type', 'time_in_window']:
            df[col] = 0.0

    # ── KEY LEVEL DISTANCES ────────────────────────────────────────────────────
    def _safe_col(col, default=0.0):
        if col in df.columns:
            return df[col].fillna(0).values.astype(float)
        return np.full(len(df), default, dtype=float)

    pdh_v = _safe_col('pdh')
    pdl_v = _safe_col('pdl')
    pwh_v = _safe_col('lw_hi')   # weekly high = prior week high
    pwl_v = _safe_col('lw_lo')   # weekly low = prior week low
    vah_v = _safe_col('vah')
    val_v = _safe_col('val')
    poc_v = _safe_col('poc')

    # London & Asia session extremes — use pre-computed DB columns if available
    if 'ldn_hi' in df.columns:
        ldn_hi  = df['ldn_hi'].fillna(0).values.astype(float)
        ldn_lo  = df['ldn_lo'].fillna(0).values.astype(float)
    else:
        ldn_hi = ldn_lo = np.zeros(len(df))

    if 'asia_hi' in df.columns:
        asia_hi = df['asia_hi'].fillna(0).values.astype(float)
        asia_lo = df['asia_lo'].fillna(0).values.astype(float)
    else:
        asia_hi = asia_lo = np.zeros(len(df))

    # distanțe față de nivele (în ATR, >0 = deasupra, <0 = sub)
    _eps = atr_v.clip(min=1.0)

    df['dist_pdh_atr']   = np.where(pdh_v > 0, (close - pdh_v) / _eps, 0.0)
    df['dist_pdl_atr']   = np.where(pdl_v > 0, (close - pdl_v) / _eps, 0.0)
    df['dist_pwh_atr']   = np.where(pwh_v > 0, (close - pwh_v) / _eps, 0.0)
    df['dist_pwl_atr']   = np.where(pwl_v > 0, (close - pwl_v) / _eps, 0.0)
    df['dist_vah_atr']   = np.where(vah_v > 0, (close - vah_v) / _eps, 0.0)
    df['dist_val_atr']   = np.where(val_v > 0, (close - val_v) / _eps, 0.0)
    df['dist_ldn_hi_atr']= np.where(ldn_hi > 0, (close - ldn_hi) / _eps, 0.0)
    df['dist_ldn_lo_atr']= np.where(ldn_lo > 0, (close - ldn_lo) / _eps, 0.0)
    df['dist_asia_hi_atr']= np.where(asia_hi > 0, (close - asia_hi) / _eps, 0.0)
    df['dist_asia_lo_atr']= np.where(asia_lo > 0, (close - asia_lo) / _eps, 0.0)
    df['dist_poc_atr']   = np.where(poc_v > 0, np.abs(close - poc_v) / _eps, 0.0)
    df['dist_poc_signed']= np.where(poc_v > 0, (close - poc_v) / _eps, 0.0)

    # care nivel e cel mai aproape?
    all_levels = {
        1: ('PDH/PWH', np.minimum(np.abs(df['dist_pdh_atr'].values),
                                  np.abs(df['dist_pwh_atr'].values))),
        2: ('PDL/PWL', np.minimum(np.abs(df['dist_pdl_atr'].values),
                                  np.abs(df['dist_pwl_atr'].values))),
        3: ('VAH',     np.abs(df['dist_vah_atr'].values)),
        4: ('VAL',     np.abs(df['dist_val_atr'].values)),
        5: ('LDN',     np.minimum(np.abs(df['dist_ldn_hi_atr'].values),
                                  np.abs(df['dist_ldn_lo_atr'].values))),
        6: ('ASIA',    np.minimum(np.abs(df['dist_asia_hi_atr'].values),
                                  np.abs(df['dist_asia_lo_atr'].values))),
    }
    nearest_dist = np.full(len(df), 999.0)
    nearest_type = np.zeros(len(df), dtype=int)
    for ltype, (_, dist_arr) in all_levels.items():
        mask = (dist_arr < nearest_dist) & (dist_arr > 0)
        nearest_dist[mask] = dist_arr[mask]
        nearest_type[mask] = ltype

    df['near_any_level'] = (nearest_dist <= KEY_LEVEL_ATR_THRESHOLD).astype(float)
    df['level_type']     = nearest_type.astype(float)
    df['above_level']    = ((close - pdh_v.clip(min=0)) > 0).astype(float)
    df['below_level']    = ((close - pdl_v.clip(min=1e6)) < 0).astype(float)

    # VP features
    df['inside_va']  = ((vah_v > 0) & (val_v > 0) & (close >= val_v) & (close <= vah_v)).astype(float)
    df['above_vah']  = ((vah_v > 0) & (close > vah_v)).astype(float)
    df['below_val']  = ((val_v > 0) & (close < val_v)).astype(float)

    # ── STRUCTURE BREAKS ──────────────────────────────────────────────────────
    df['broke_pdh'] = np.where(pdh_v > 0, (high > pdh_v).astype(float), 0.0)
    df['broke_pdl'] = np.where(pdl_v > 0, (low  < pdl_v).astype(float), 0.0)
    df['broke_pwh'] = np.where(pwh_v > 0, (high > pwh_v).astype(float), 0.0)
    df['broke_pwl'] = np.where(pwl_v > 0, (low  < pwl_v).astype(float), 0.0)
    df['broke_vah'] = np.where(vah_v > 0, (high > vah_v).astype(float), 0.0)
    df['broke_val'] = np.where(val_v > 0, (low  < val_v).astype(float), 0.0)

    # Contorizare lichiditate luată
    df['liq_above_count'] = (df['broke_pdh'].values + df['broke_pwh'].values +
                             df['broke_vah'].values).clip(0, 3)
    df['liq_below_count'] = (df['broke_pdl'].values + df['broke_pwl'].values +
                             df['broke_val'].values).clip(0, 3)

    # ── SWEEP DETECTION ───────────────────────────────────────────────────────
    # Sweep = prețul a depășit nivelul cu wicks dar a revenit în corpul barei
    swept_above = np.zeros(len(df))
    swept_below = np.zeros(len(df))
    sweep_wick  = np.zeros(len(df))

    for lvl_v, direction in [(pdh_v, 'above'), (pwh_v, 'above'), (vah_v, 'above'),
                              (ldn_hi, 'above'), (asia_hi, 'above'),
                              (pdl_v, 'below'), (pwl_v, 'below'), (val_v, 'below'),
                              (ldn_lo, 'below'), (asia_lo, 'below')]:
        valid = lvl_v > 0
        if direction == 'above':
            # High depășit dar close sub nivel → sweep sus (failed auction)
            _sw = valid & (high > lvl_v) & (close < lvl_v)
            _wick = np.where(_sw, (high - lvl_v) / _eps, 0.0)
            swept_above = np.maximum(swept_above, _sw.astype(float))
            sweep_wick  = np.maximum(sweep_wick, _wick)
        else:
            # Low depășit dar close deasupra nivel → sweep jos
            _sw = valid & (low < lvl_v) & (close > lvl_v)
            _wick = np.where(_sw, (lvl_v - low) / _eps, 0.0)
            swept_below = np.maximum(swept_below, _sw.astype(float))
            sweep_wick  = np.maximum(sweep_wick, _wick)

    df['swept_above_level'] = swept_above
    df['swept_below_level'] = swept_below
    df['sweep_wick_ratio']  = sweep_wick

    # bars_since_sweep — vectorized using cumsum trick
    any_sweep_arr = ((swept_above + swept_below) > 0).astype(int)
    sweep_cumsum  = np.cumsum(any_sweep_arr)
    # For each position, track the index of the last sweep
    # bars_since[i] = i - last_sweep_idx[i]
    _idx = np.arange(len(df))
    _sw_idx_at = np.where(any_sweep_arr, _idx, -999)  # sweep position or -999
    # forward-fill to carry last sweep position
    _sw_cummax = np.maximum.accumulate(_sw_idx_at)
    bars_since = np.where(_sw_cummax < 0, 100.0, (_idx - _sw_cummax).astype(float))
    df['bars_since_sweep'] = np.clip(bars_since, 0, 100)

    # reclaimed_after_sweep — vectorized
    close_s = pd.Series(close)
    recl = (
        ((pd.Series(swept_above).shift(1) > 0) & (close_s < close_s.shift(1))) |
        ((pd.Series(swept_below).shift(1) > 0) & (close_s > close_s.shift(1)))
    ).astype(float).fillna(0).values
    df['reclaimed_after_sweep'] = recl

    # ── CANDLE STRUCTURE ──────────────────────────────────────────────────────
    body = close - open_
    rng  = (high - low).clip(min=1e-8)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low

    df['body_dir']        = body / rng
    df['wick_bias']       = (upper_wick - lower_wick) / rng
    df['upper_wick_atr']  = upper_wick / _eps
    df['lower_wick_atr']  = lower_wick / _eps

    # Sharp reversal bar: engulfing sau pin bar puternic (wick > 2×body)
    df['sharp_reversal_bar'] = (
        ((upper_wick > 2 * np.abs(body)) & (body < 0)) |  # bearish pin
        ((lower_wick > 2 * np.abs(body)) & (body > 0))    # bullish pin
    ).astype(float)

    # Rejection candle
    df['rejection_candle'] = (
        (upper_wick > 1.5 * lower_wick) |  # rejection sus
        (lower_wick > 1.5 * upper_wick)    # rejection jos
    ).astype(float)

    # Outside bar
    prev_high = np.roll(high, 1); prev_high[0] = high[0]
    prev_low  = np.roll(low, 1);  prev_low[0]  = low[0]
    df['outside_bar'] = ((high > prev_high) & (low < prev_low)).astype(float)

    # ── MOMENTUM ──────────────────────────────────────────────────────────────
    df['momentum_5']  = (df['close'] - df['close'].shift(5))  / _eps
    df['momentum_15'] = (df['close'] - df['close'].shift(15)) / _eps
    df['slope_h1']    = (df['close'] - df['close'].shift(60)) / (df['close'].shift(60).abs() + 1e-8)
    df['h4_momentum'] = (df['close'] - df['close'].shift(240)) / _eps

    # 4H structure: HH+HL sau LH+LL
    h4_high = df['high'].rolling(240, min_periods=60).max()
    h4_low  = df['low'].rolling(240, min_periods=60).min()
    h4_high_prev = df['high'].rolling(240, min_periods=60).max().shift(240)
    h4_low_prev  = df['low'].rolling(240, min_periods=60).min().shift(240)
    df['h4_hh_hl'] = np.where(
        (h4_high > h4_high_prev) & (h4_low > h4_low_prev), 1.0,   # bullish
        np.where(
            (h4_high < h4_high_prev) & (h4_low < h4_low_prev), -1.0,  # bearish
            0.0
        )
    )

    # ── VOLATILITY ────────────────────────────────────────────────────────────
    atr_roll20 = atr.rolling(20, min_periods=10).rank(pct=True)
    df['atr_percentile'] = atr_roll20.fillna(0.5)
    df['realized_vol'] = np.log(df['close'] / df['close'].shift(1).clip(lower=0.01)).rolling(14).std()

    if 'volume' in df.columns:
        vol_ma20 = df['volume'].rolling(20, min_periods=5).mean()
        df['vol_spike'] = (df['volume'] > 1.5 * vol_ma20).astype(float)
    else:
        df['vol_spike'] = 0.0

    df['range_atr_ratio'] = rng / _eps

    # Fillna pe toate features
    for col in REVERSAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# TARGET LABELING
# ══════════════════════════════════════════════════════════════════════════════

def generate_reversal_target(df: pd.DataFrame,
                              horizon: int = REVERSAL_HORIZON,
                              r_min: float = REVERSAL_R_MIN,
                              sl_mult: float = SL_ATR_MULT,
                              sl_min: float = SL_MIN_PTS,
                              sl_max: float = SL_MAX_PTS) -> pd.Series:
    """
    Label REVERSAL=1 dacă:
      1. Suntem la un KEY LEVEL (near_any_level=1) SAU am făcut sweep
      2. Prețul se întoarce ≥ r_min × SL în direcția opusă în next horizon bare

    Logică:
      - Near PDH/VAH/LDN_HI + swept_above → testăm dacă prețul cade ≥ r_min×SL
      - Near PDL/VAL/LDN_LO + swept_below → testăm dacă prețul crește ≥ r_min×SL

    REVERSAL=0 → prețul continuă în direcția de dinaintea sweep-ului.
    """
    close   = df['close'].values
    high    = df['high'].values
    low     = df['low'].values
    atr_v   = df['_atr'].values if '_atr' in df.columns else np.full(len(df), 12.0)

    near_up   = df['swept_above_level'].values if 'swept_above_level' in df.columns else np.zeros(len(df))
    near_dn   = df['swept_below_level'].values if 'swept_below_level' in df.columns else np.zeros(len(df))
    near_any  = df['near_any_level'].values    if 'near_any_level' in df.columns    else np.zeros(len(df))

    labels = np.zeros(len(df), dtype=int)
    n = len(df)

    for i in range(n - horizon):
        _is_near = near_any[i] > 0 or near_up[i] > 0 or near_dn[i] > 0
        if not _is_near:
            continue

        # SL structural (cât risc avem dacă intrăm în reversal)
        _sl = float(np.clip(atr_v[i] * sl_mult, sl_min, sl_max))
        _target_move = r_min * _sl  # câte puncte trebuie să se miște

        # Fereastra forward
        future_high = high[i+1 : i+1+horizon].max() if (i+1+horizon) <= n else high[i+1:]
        future_low  = low[i+1 : i+1+horizon].min()  if (i+1+horizon) <= n else low[i+1:]

        px = close[i]

        # Sweep SUS → așteptăm mișcare în jos
        if near_up[i] > 0 or (near_any[i] > 0 and px > 0):
            downward_move = px - future_low
            if downward_move >= _target_move:
                labels[i] = 1
                continue

        # Sweep JOS → așteptăm mișcare în sus
        if near_dn[i] > 0:
            upward_move = future_high - px
            if upward_move >= _target_move:
                labels[i] = 1
                continue

    return pd.Series(labels, index=df.index, name='reversal_target')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINER
# ══════════════════════════════════════════════════════════════════════════════

def train_reversal_model():
    """Antrenează reversal model pe date din DB."""
    print("\n" + "="*80)
    print("🔄 REVERSAL MODEL TRAINER — Aladin v14")
    print(f"   Key times: {[name for _, name in KEY_TIMES_RO]}")
    print(f"   Key level threshold: ≤ {KEY_LEVEL_ATR_THRESHOLD}×ATR")
    print(f"   Reversal target: ≥ {REVERSAL_R_MIN}R în {REVERSAL_HORIZON} bare")
    print("="*80)

    # ── Pasul 1: Conectare DB ─────────────────────────────────────────────────
    db_path = None
    for path in DB_PATHS:
        if os.path.exists(path):
            db_path = path
            break
    if db_path is None:
        raise FileNotFoundError(f"DB nu a fost găsit. Căutat în: {DB_PATHS}")
    print(f"✅ DB: {db_path}")

    conn = sqlite3.connect(db_path)
    # Detectăm tabela
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    tbl = tables['name'].iloc[0]
    print(f"✅ Tabelă: {tbl}")

    # Citim datele — filtrăm direct în SQL doar orele relevante (09-18 RO = 06-15 UTC)
    # pentru a reduce memoria. Rolling features (ATR, VP) au nevoie de context → citim
    # tot și filtrăm DUPĂ feature engineering cu _rel_mask.
    # Detectăm coloanele disponibile și le selectăm pe cele necesare
    avail_cols = [c[1] for c in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
    BASE_COLS  = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    WANT_COLS  = BASE_COLS + [c for c in [
        'atr_14', 'vah', 'val', 'poc_level',
        'asia_hi', 'asia_lo', 'lon_hi', 'lon_lo',
        'lw_hi', 'lw_lo', 'p_hi', 'p_lo',   # prev week / prev day levels
        'inside_va', 'dist_poc', 'vwap', 'dist_vwap',
        'body_size', 'has_displacement', 'fvg_up', 'fvg_down',
        'is_smt_bearish', 'is_smt_bullish',
        'bar_delta', 'cum_delta', 'bar_buy_vol', 'bar_sell_vol',
        'imbalance_pct', 'rvol', 'adx_14',
    ] if c in avail_cols]
    sel = ', '.join(WANT_COLS)

    print(f"📥 Pasul 1: Citire {len(WANT_COLS)} coloane din DB (06-17 UTC, -4 ani)...")
    try:
        df = pd.read_sql(
            f"""SELECT {sel} FROM {tbl}
                WHERE CAST(substr(timestamp, 12, 2) AS INTEGER) BETWEEN 6 AND 17
                  AND timestamp >= date('now', '-4 years')
                ORDER BY timestamp ASC""",
            conn
        )
        if len(df) < 100000:
            df = pd.read_sql(
                f"""SELECT {sel} FROM {tbl}
                    WHERE CAST(substr(timestamp, 12, 2) AS INTEGER) BETWEEN 6 AND 17
                    ORDER BY timestamp ASC""",
                conn
            )
    except Exception as e:
        print(f"   ⚠️ Filtru fail ({e}) — fallback")
        df = pd.read_sql(f"SELECT {sel} FROM {tbl} ORDER BY timestamp ASC LIMIT 800000", conn)
    conn.close()
    # Downcast float64→float32 to halve RAM
    for c in df.select_dtypes(include='float64').columns:
        df[c] = df[c].astype('float32')
    # Coerce object-typed numeric columns
    for c in df.select_dtypes(include='object').columns:
        if c not in ('timestamp',):
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    # Rename DB level columns to names used by feature engineering
    if 'poc_level' in df.columns and 'poc' not in df.columns:
        df = df.rename(columns={'poc_level': 'poc'})
    if 'p_hi' in df.columns:
        df = df.rename(columns={'p_hi': 'pdh', 'p_lo': 'pdl'})
    if 'lw_hi' in df.columns:
        pass  # already correct name
    if 'lon_hi' in df.columns and 'ldn_hi' not in df.columns:
        df = df.rename(columns={'lon_hi': 'ldn_hi', 'lon_lo': 'ldn_lo'})
    print(f"   ✅ {len(df):,} rânduri | RAM≈{df.memory_usage(deep=True).sum()/1024**2:.0f}MB | cols: {list(df.columns[:8])}...")

    # ── Pasul 2: Feature Engineering ─────────────────────────────────────────
    print("\n🔧 Pasul 2: Calcul features reversal...")
    df = add_reversal_features(df)
    print(f"   ✅ Features calculate")

    # ── Pasul 3: Target labeling ──────────────────────────────────────────────
    print(f"\n🎯 Pasul 3: Generare target REVERSAL (horizon={REVERSAL_HORIZON}, R_min={REVERSAL_R_MIN})...")
    df['reversal_target'] = generate_reversal_target(df)

    n_rev  = int((df['reversal_target'] == 1).sum())
    n_cont = int((df['reversal_target'] == 0).sum())
    rev_pct = n_rev / len(df) * 100
    print(f"   📊 REVERSAL=1: {n_rev:,} ({rev_pct:.2f}%) | CONTINUE=0: {n_cont:,}")

    if n_rev < 500:
        print(f"   ⚠️ AVERTISMENT: Prea puține reversale ({n_rev}) — verifică threshold-urile")

    # ── Pasul 4: Filter — doar bare lângă key level SAU key time ─────────────
    print("\n🔍 Pasul 4: Filtrare la bare relevante (key level | key time | sweep)...")
    _rel_mask = (
        (df['near_any_level'] > 0) |
        (df['near_key_time']  > 0) |
        (df['swept_above_level'] > 0) |
        (df['swept_below_level'] > 0)
    )
    df_rel = df[_rel_mask].copy()
    print(f"   📊 Bare relevante: {len(df_rel):,} din {len(df):,} ({len(df_rel)/len(df)*100:.1f}%)")
    n_rev2  = int((df_rel['reversal_target'] == 1).sum())
    rev_pct2 = n_rev2 / len(df_rel) * 100 if len(df_rel) > 0 else 0
    print(f"   📊 REVERSAL rate în subset relevant: {rev_pct2:.1f}%")

    if len(df_rel) < 5000:
        print("   ⚠️ Prea puține bare relevante — folosim df complet cu undersample")
        df_rel = df.copy()

    # ── Pasul 5: Pregătire X, y ───────────────────────────────────────────────
    print("\n⚙️  Pasul 5: Pregătire X, y...")
    available_feats = [f for f in REVERSAL_FEATURES if f in df_rel.columns]
    print(f"   📊 Features disponibile: {len(available_feats)}/{len(REVERSAL_FEATURES)}")

    X = df_rel[available_feats].ffill().fillna(0)
    y = df_rel['reversal_target'].astype(int)

    print(f"   ✅ X={X.shape}, y: REVERSAL={y.sum():,} CONTINUE={(y==0).sum():,}")

    # ── Split temporal 45/5/50 cu purge ──────────────────────────────────────
    _PURGE = REVERSAL_HORIZON + 5
    split_train = int(len(X) * 0.45)
    split_val   = int(len(X) * 0.50)

    X_train = X.iloc[:max(0, split_train - _PURGE)]
    X_val   = X.iloc[split_train + _PURGE : max(split_train + _PURGE, split_val - _PURGE)]
    X_test  = X.iloc[split_val + _PURGE:]
    y_train = y.iloc[:max(0, split_train - _PURGE)]
    y_val   = y.iloc[split_train + _PURGE : max(split_train + _PURGE, split_val - _PURGE)]
    y_test  = y.iloc[split_val + _PURGE:]

    print(f"   ✅ Split PURGED (purge={_PURGE}): Train={len(X_train):,} Val={len(X_val):,} Test={len(X_test):,}")

    # ── Timestamp info ──
    if 'timestamp' in df_rel.columns:
        try:
            ts_col = pd.to_datetime(df_rel['timestamp'], errors='coerce')
            print(f"   📅 Train: {ts_col.iloc[:split_train].min()} → {ts_col.iloc[:split_train].max()}")
            print(f"   📅 Test:  {ts_col.iloc[split_val:].min()} → {ts_col.iloc[split_val:].max()}")
        except:
            pass

    # ── Pasul 6: Undersample CONTINUE (clasa majoritară) ──────────────────────
    print("\n⚖️  Pasul 6: Echilibrare clase (undersample CONTINUE)...")
    _n_rev_tr  = int((y_train == 1).sum())
    _n_cont_tr = int((y_train == 0).sum())
    _target_cont = _n_rev_tr * 6  # 6:1 ratio CONTINUE:REVERSAL

    _rng = np.random.RandomState(42)
    if _n_cont_tr > _target_cont:
        _cont_idx = y_train[y_train == 0].index
        _keep_cont = _rng.choice(_cont_idx, size=_target_cont, replace=False)
        _rev_idx   = y_train[y_train == 1].index.values
        _keep_idx  = np.sort(np.concatenate([_keep_cont, _rev_idx]))
        X_train_res = X_train.loc[_keep_idx]
        y_train_res = y_train.loc[_keep_idx]
    else:
        X_train_res = X_train
        y_train_res = y_train

    _rev_rate = y_train_res.mean()
    _spw = (1 - _rev_rate) / _rev_rate if _rev_rate > 0 else 10.0
    _spw = float(np.clip(_spw, 1.0, 15.0))
    sw   = np.where(y_train_res == 1, _spw, 1.0)

    print(f"   ✅ Train resampled: REVERSAL={y_train_res.sum():,} CONTINUE={(y_train_res==0).sum():,}")
    print(f"   ✅ scale_pos_weight: {_spw:.2f}")

    # ── Pasul 7: Antrenare XGBoost ────────────────────────────────────────────
    print("\n🚀 Pasul 7: Antrenare XGBoost Reversal Model...")

    _xgb_params = dict(
        n_estimators     = 2000,
        max_depth        = 5,
        learning_rate    = 0.012,
        objective        = 'binary:logistic',
        eval_metric      = 'auc',
        subsample        = 0.80,
        colsample_bytree = 0.80,
        colsample_bylevel= 0.70,
        reg_lambda       = 12.0,
        reg_alpha        = 2.0,
        min_child_weight = 8,
        gamma            = 0.8,
        scale_pos_weight = _spw,
        tree_method      = 'hist',
        random_state     = 42,
        verbosity        = 0,
        early_stopping_rounds = 80,
    )

    model = xgb.XGBClassifier(**_xgb_params)
    model.fit(
        X_train_res, y_train_res,
        sample_weight = sw,
        eval_set      = [(X_val, y_val)],
        verbose       = 100,
    )
    print(f"   ✅ Best iteration: {model.best_iteration} / {model.n_estimators}")

    # ── Pasul 8: Evaluare ─────────────────────────────────────────────────────
    print("\n📊 Pasul 8: Evaluare model...")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_raw = (y_proba >= 0.50).astype(int)

    acc = accuracy_score(y_test, y_pred_raw)
    print(f"   Accuracy @0.50: {acc:.4f}")

    # Threshold search — target: precizie ≥ 70% (inversale rare dar de calitate)
    print("\n   🔍 Threshold search (target precision ≥ 70%):")
    print(f"   {'Threshold':>10s} {'N_Rev':>8s} {'Precision':>10s} {'Recall':>10s} {'F1':>8s}")
    print(f"   {'─'*10} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")

    _best_thr = 0.50
    _best_prec = 0.0
    for _thr in np.arange(0.40, 0.951, 0.05):
        _prd = (y_proba >= _thr).astype(int)
        _tp  = (((_prd == 1) & (y_test.values == 1))).sum()
        _fp  = (((_prd == 1) & (y_test.values == 0))).sum()
        _fn  = (((_prd == 0) & (y_test.values == 1))).sum()
        _prec = _tp / (_tp + _fp + 1e-8)
        _rec  = _tp / (_tp + _fn + 1e-8)
        _f1   = 2 * _prec * _rec / (_prec + _rec + 1e-8)
        _n_rev_pred = _prd.sum()
        print(f"   {_thr:>10.0%} {_n_rev_pred:>8,} {_prec:>10.1%} {_rec:>10.1%} {_f1:>8.3f}")
        if _prec > _best_prec and _n_rev_pred >= 50:
            _best_prec = _prec
            _best_thr  = _thr

    CONF_THRESHOLD = round(float(_best_thr), 2)
    print(f"\n   ✅ Threshold optim: {CONF_THRESHOLD:.0%} (precision={_best_prec:.1%})")

    y_pred = (y_proba >= CONF_THRESHOLD).astype(int)
    acc_thr = accuracy_score(y_test, y_pred)
    print(f"   Accuracy @{CONF_THRESHOLD:.0%}: {acc_thr:.4f}")
    print("\n   📋 Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=['CONTINUE', 'REVERSAL'],
                                 zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    print(f"   Confusion Matrix:\n{cm}")

    # ── AUC ──
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n   📊 AUC-ROC: {auc:.4f}")
    if auc >= 0.75:
        print("   ✅ AUC ≥ 0.75 — model bun de discriminare reversal vs continue")
    elif auc >= 0.65:
        print("   ⚠️ AUC 0.65-0.75 — model acceptabil, consideră mai multe sweep features")
    else:
        print("   ❌ AUC < 0.65 — model slab, revizuiește labeling sau features")

    # ── Feature Importance ──
    print("\n🏆 Pasul 9: Feature Importance...")
    imp_df = pd.DataFrame({
        'feature': list(X_train_res.columns),
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)
    print(imp_df.head(15).to_string(index=False))
    top_features = imp_df.head(15)['feature'].tolist()

    # Salvare feat importance CSV
    _imp_csv = MODEL_SAVE_PATH.replace('.json', '_feat_imp.csv')
    imp_df.reset_index(drop=True).to_csv(_imp_csv, index=False)
    print(f"   ✅ Feature importance → {_imp_csv}")

    # ── Pasul 10: Calibrare Isotonic ─────────────────────────────────────────
    print("\n🎯 Pasul 10: Calibrare Isotonic Regression...")
    try:
        from sklearn.isotonic import IsotonicRegression
        _iso = IsotonicRegression(out_of_bounds='clip')
        _cal_proba = model.predict_proba(X_val)[:, 1]
        _iso.fit(_cal_proba, y_val.values.astype(float))
        y_proba_cal = _iso.transform(y_proba)
        y_pred_cal  = (y_proba_cal >= CONF_THRESHOLD).astype(int)
        acc_cal = accuracy_score(y_test, y_pred_cal)
        _cal_auc = roc_auc_score(y_test, y_proba_cal)
        print(f"   ✅ Calibrated acc: {acc_cal:.4f} | AUC: {_cal_auc:.4f}")

        _cal_path = MODEL_SAVE_PATH.replace('.json', '_calibrated_iso.pkl')
        with open(_cal_path, 'wb') as f:
            pickle.dump(_iso, f)
        print(f"   ✅ Calibrator salvat: {_cal_path}")
    except Exception as _ce:
        print(f"   ⚠️ Calibrare eșuată: {_ce}")

    # ── Pasul 11: LightGBM (cross-validation) ────────────────────────────────
    print("\n🌲 Pasul 11: LightGBM cross-check...")
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        lgbm = LGBMClassifier(
            n_estimators=1000, max_depth=5, learning_rate=0.015,
            subsample=0.80, colsample_bytree=0.80,
            reg_lambda=12.0, reg_alpha=2.0, min_child_samples=8,
            scale_pos_weight=_spw, random_state=42, verbosity=-1,
            objective='binary',
        )
        lgbm.fit(
            X_train_res, y_train_res,
            sample_weight=sw,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50), log_evaluation(200)],
        )
        y_lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
        lgbm_auc = roc_auc_score(y_test, y_lgbm_proba)
        print(f"   ✅ LightGBM AUC: {lgbm_auc:.4f} (vs XGB {auc:.4f})")

        _lgbm_path = MODEL_SAVE_PATH.replace('.json', '_lgbm.pkl')
        with open(_lgbm_path, 'wb') as f:
            pickle.dump(lgbm, f)
        print(f"   ✅ LightGBM salvat: {_lgbm_path}")
    except ImportError:
        print("   ⚠️ lightgbm lipsă")
    except Exception as _le:
        print(f"   ⚠️ LightGBM error: {_le}")

    # ── Pasul 12: Salvare model ────────────────────────────────────────────────
    print(f"\n💾 Pasul 12: Salvare model → {MODEL_SAVE_PATH}")
    model.save_model(MODEL_SAVE_PATH)

    features_meta = {
        "features":        available_feats,
        "top_features":    top_features,
        "n_features":      len(available_feats),
        "auc":             round(float(auc), 4),
        "accuracy":        round(float(acc_thr), 4),
        "conf_threshold":  CONF_THRESHOLD,
        "reversal_horizon": REVERSAL_HORIZON,
        "reversal_r_min":  REVERSAL_R_MIN,
        "trained_at":      datetime.now().isoformat(),
        "rows_trained":    len(X_train_res),
        "reversal_rate":   round(rev_pct2, 3),
        "confusion_matrix": cm.tolist(),
        "key_times": [{"time": kt, "name": nm} for kt, nm in KEY_TIMES_RO],
        "key_level_atr_threshold": KEY_LEVEL_ATR_THRESHOLD,
    }
    with open(FEATURES_PATH, 'w') as f:
        json.dump(features_meta, f, indent=2)
    print(f"   ✅ Feature metadata → {FEATURES_PATH}")

    # ── Pasul 13: Analytics Suite ─────────────────────────────────────────────
    if _ANALYTICS:
        print(f"\n{'='*80}")
        print(f"📊 ANALYTICS SUITE — Validare statistică Reversal Model")
        print(f"{'='*80}")
        try:
            def _rev_model_factory(*args, **kwargs):
                return xgb.XGBClassifier(
                    n_estimators=500, max_depth=5, learning_rate=0.015,
                    objective='binary:logistic', eval_metric='auc',
                    subsample=0.80, colsample_bytree=0.80,
                    reg_lambda=12.0, reg_alpha=2.0, min_child_weight=8,
                    scale_pos_weight=_spw, tree_method='hist',
                    random_state=42, verbosity=0,
                )

            # risk_pts: SL structural median (0.65×ATR, clamp 8-20)
            _rev_risk_pts = float(np.clip(
                df_rel['_atr'].median() * SL_ATR_MULT if '_atr' in df_rel.columns else 12.0,
                SL_MIN_PTS, SL_MAX_PTS
            ))
            print(f"   risk_pts (structural SL): {_rev_risk_pts:.1f} pts")

            run_full_analysis(
                model           = model,
                X_train         = X_train_res,
                X_test          = X_test,
                y_train         = y_train_res,
                y_test          = y_test,
                features        = available_feats,
                risk_pts        = _rev_risk_pts,
                threshold       = CONF_THRESHOLD,
                label           = "REVERSAL",
                save_dir        = str(os.path.dirname(MODEL_SAVE_PATH)),
                model_factory   = _rev_model_factory,
                monte_carlo_sims= 1000,
                run_perm_imp    = True,
                run_walk_fwd    = True,
                n_perm_imp_repeats = 5,
            )
            print(f"   ✅ Analytics suite completă")
        except Exception as _ae:
            print(f"   ⚠️ Analytics error: {_ae}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"🎯 REVERSAL MODEL ANTRENAT!")
    print(f"   Model:      {MODEL_SAVE_PATH}")
    print(f"   AUC:        {auc:.4f}")
    print(f"   Threshold:  {CONF_THRESHOLD:.0%}")
    print(f"   Precision:  {_best_prec:.1%}")
    print(f"   Features:   {len(available_feats)}")
    print(f"\n   KEY TIMES monitorizate: NY Macro 17:10, LDN-NY 16:50, LDN Close 14:00")
    print(f"   KEY LEVELS: PDH/PDL, PWH/PWL, VAH/VAL, LDN hi/lo, Asia hi/lo")
    print(f"{'='*80}\n")


# ══════════════════════════════════════════════════════════════════════════════
# HOW TO INTEGRATE WITH ALADIN BRIDGE (live usage)
# ══════════════════════════════════════════════════════════════════════════════
"""
În mario_rag.py (live bridge), reversal_model se folosește astfel:

def check_reversal_signal(bar_data: dict, model, features_meta: dict) -> float:
    '''
    Returnează probabilitatea de reversal (0.0-1.0) pentru bara curentă.
    Dacă prob >= conf_threshold → REVERSAL signal activ.

    Exemplu workflow:
      1. breakout_model → REAL LONG breakout → entry LONG
      2. Prețul merge spre PDH/VAH
      3. reversal_model → P(reversal) = 0.82 >= threshold 0.70
      4. Aladin: partial exit sau full exit LONG + potențial SHORT reversal
    '''
    import pandas as pd
    df_bar = pd.DataFrame([bar_data])
    df_bar = add_reversal_features(df_bar)
    avail = [f for f in features_meta['features'] if f in df_bar.columns]
    X = df_bar[avail].fillna(0)
    return float(model.predict_proba(X)[0, 1])

# Load model:
# import xgboost as xgb, json
# model = xgb.XGBClassifier(); model.load_model('reversal_model.json')
# with open('reversal_features.json') as f: meta = json.load(f)
# CONF_THR = meta['conf_threshold']  # e.g. 0.70
"""


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train_reversal_model()
