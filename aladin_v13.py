"""
Aladin v13 — REGIME-AWARE ICT Trading Engine
═════════════════════════════════════════════
Modul centralizat pentru logica v13:
  • FEATURES_WEEKLY   — profile săptămânale (day-of-week ICT)
  • FEATURES_DAYTYPE  — clasificare zi (trend/range/reversal/outside/inside/gap)
  • FEATURES_SWEEP    — detectare sweep multi-level + OF confirmation
  • add_weekly_features(df)
  • add_daytype_features(df)
  • add_sweep_features(df)
  • generate_regime_target(df)  — 9-class multiclass target
  • dynamic_sl(entry, dir, recent_bars, atr, or_hi, or_lo, regime) — SL adaptiv
  • compute_exit_signal(...) — rule-based dynamic exit (safety net peste model)

Import & use pattern:
    from aladin_v13 import (
        FEATURES_WEEKLY, FEATURES_DAYTYPE, FEATURES_SWEEP,
        add_weekly_features, add_daytype_features, add_sweep_features,
        generate_regime_target, dynamic_sl, REGIME_NAMES,
    )
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# ════════════════════════════════════════════════════════════════════════════
# REGIME CLASSES
# ════════════════════════════════════════════════════════════════════════════
# v14: 5-class scheme (collapsed din 9 — clasele 3,4,7,8 nu aveau suficiente sample-uri
# și au produs model biased). Directional clean: WAIT / SHORT_BREAK / LONG_BREAK / SHORT_REV / LONG_REV.
REGIME_WAIT        = 0
REGIME_SHORT_BREAK = 1  # any directional SHORT (breakout sau post-sweep)
REGIME_LONG_BREAK  = 2  # any directional LONG
REGIME_SHORT_REV   = 3  # fade from high (retracement)
REGIME_LONG_REV    = 4  # fade from low

# Backward-compat aliases (cod vechi din dynamic_sl, compute_exit_signal continuă să meargă)
REGIME_SHORT_EXPANSION     = REGIME_SHORT_BREAK
REGIME_LONG_EXPANSION      = REGIME_LONG_BREAK
REGIME_SHORT_AFTER_SWEEP   = REGIME_SHORT_BREAK
REGIME_LONG_AFTER_SWEEP    = REGIME_LONG_BREAK
REGIME_SHORT_REVERSAL_HIGH = REGIME_SHORT_REV
REGIME_LONG_REVERSAL_LOW   = REGIME_LONG_REV
REGIME_MEAN_REV_SHORT_VAH  = REGIME_SHORT_REV  # MR clasele nu aveau semnal → colapsate în REV
REGIME_MEAN_REV_LONG_VAL   = REGIME_LONG_REV

REGIME_NAMES: Dict[int, str] = {
    0: "WAIT",
    1: "SHORT_BREAK",
    2: "LONG_BREAK",
    3: "SHORT_REV",
    4: "LONG_REV",
}

REGIME_DIRECTION: Dict[int, str] = {
    0: "WAIT",
    1: "SHORT", 2: "LONG", 3: "SHORT", 4: "LONG",
}

# ════════════════════════════════════════════════════════════════════════════
# FEATURE LISTS
# ════════════════════════════════════════════════════════════════════════════
FEATURES_WEEKLY = [
    'dow_monday', 'dow_tuesday', 'dow_wednesday', 'dow_thursday', 'dow_friday',
    'week_phase',                  # 0=Mon, 1=Tue-Wed, 2=Thu, 3=Fri
    'weekly_bias_up',              # 1 dacă close > previous weekly close
    'weekly_bias_down',
    'dist_weekly_high_atr',        # (lw_hi - close) / ATR
    'dist_weekly_low_atr',         # (close - lw_lo) / ATR
    'dist_weekly_open_atr',        # (close - weekly_open) / ATR  (folosește true_open dacă există)
    'weekly_range_atr',            # (lw_hi - lw_lo) / ATR
    'session_at_weekly_high',      # 1 dacă close > lw_hi - 0.3×ATR (aproape de max săpt)
    'session_at_weekly_low',       # 1 dacă close < lw_lo + 0.3×ATR
    'prev_week_bullish',           # 1 dacă săptămâna trecută a închis bullish
]

FEATURES_DAYTYPE = [
    'trend_day_score',             # |close - day_open| / ATR (clamped 0-5)
    'range_day_score',              # 1 / max((day_high - day_low) / ATR, 1.0) — invers: range mic = score mare
    'reversal_day_score',          # max(|high - open|, |open - low|) - |close - open| / ATR
    'outside_day',                 # high > pdh AND low < pdl
    'inside_day',                  # high < pdh AND low > pdl
    'gap_up',                      # day_open > prev_close + 0.3×ATR
    'gap_down',                    # day_open < prev_close - 0.3×ATR
    'gap_filled',                  # low ≤ prev_close (după gap up) sau high ≥ prev_close (după gap down)
    'day_vs_open_atr',             # (close - day_open) / ATR
    'session_sweep_asia_hi',       # spart asia_hi today
    'session_sweep_asia_lo',
    'session_sweep_pdh',
    'session_sweep_pdl',
    'session_sweep_lon_hi',
    'session_sweep_lon_lo',
    'liq_above_count',             # câte nivele (asia_hi, pdh, lw_hi, h4_hi) sunt peste close
    'liq_below_count',
    'early_session_high',          # high-ul a fost format în prima oră (vs late day)
    'early_session_low',
]

FEATURES_SWEEP = [
    'broke_asia_hi',
    'broke_asia_lo',
    'broke_pdh',
    'broke_pdl',
    'broke_lon_hi',
    'broke_lon_lo',
    'broke_h4_hi',
    'broke_h4_lo',
    'broke_h1_hi',
    'broke_h1_lo',
    'reclaimed_after_sweep',       # spart nivel + revenit în ≤10 bare
    'sweep_depth_atr',             # profunzime sweep în ATR
    'delta_at_sweep_extreme',      # delta la vârf/fund sweep-ului
    'delta_reversal_at_sweep',     # 1 dacă delta a flip-uit semnul la extrem
    'big_trades_at_sweep',         # max(big_sell_count, big_buy_count) la extrem
    'absorption_at_sweep',         # absorption_score la extrem
    'displacement_after_sweep',    # FVG formed în 5 bare post-sweep
    'bars_since_sweep',            # câte bare de la ultimul sweep
    'sweep_direction',             # +1 = sweep up (short bias), -1 = sweep down (long bias), 0 = none
]


# ════════════════════════════════════════════════════════════════════════════
# HELPER — safe numeric ops
# ════════════════════════════════════════════════════════════════════════════
def _safe_div(num, den, default=0.0):
    """Division with safe handling of zeros/NaN."""
    num_a = np.asarray(num, dtype=np.float64)
    den_a = np.asarray(den, dtype=np.float64)
    out = np.full_like(num_a, default, dtype=np.float64)
    mask = (den_a != 0) & np.isfinite(den_a) & np.isfinite(num_a)
    out[mask] = num_a[mask] / den_a[mask]
    return out


def _ensure_atr(df: pd.DataFrame, default: float = 9.0) -> pd.Series:
    if 'atr_14' in df.columns:
        atr = df['atr_14'].fillna(default)
        return atr.where(atr > 0, default)
    return pd.Series(default, index=df.index)


# ════════════════════════════════════════════════════════════════════════════
# WEEKLY PROFILE FEATURES
# ════════════════════════════════════════════════════════════════════════════
def add_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adaugă features de profil săptămânal (day-of-week ICT)."""
    if 'timestamp' not in df.columns:
        for col in FEATURES_WEEKLY:
            df[col] = 0.0
        return df

    ts = pd.to_datetime(df['timestamp'], errors='coerce')
    dow = ts.dt.dayofweek  # 0=Mon .. 6=Sun

    df['dow_monday']    = (dow == 0).astype(int)
    df['dow_tuesday']   = (dow == 1).astype(int)
    df['dow_wednesday'] = (dow == 2).astype(int)
    df['dow_thursday']  = (dow == 3).astype(int)
    df['dow_friday']    = (dow == 4).astype(int)

    # week_phase: 0=Mon (accumulation), 1=Tue-Wed (build), 2=Thu (manipulation), 3=Fri (close)
    phase = pd.Series(0, index=df.index)
    phase[(dow == 1) | (dow == 2)] = 1
    phase[dow == 3] = 2
    phase[dow == 4] = 3
    df['week_phase'] = phase.astype(int)

    atr = _ensure_atr(df)

    # Weekly levels (lw_hi/lo = last week high/low)
    # Guard: lw_hi/lo = 0 sau NaN → folosim close ca fallback (dist = 0)
    lw_hi_raw = df.get('lw_hi', pd.Series(np.nan, index=df.index))
    lw_lo_raw = df.get('lw_lo', pd.Series(np.nan, index=df.index))
    close = df.get('close', pd.Series(np.nan, index=df.index))
    # Replace 0 or NaN cu close (safe default)
    lw_hi = lw_hi_raw.where((lw_hi_raw > 0) & lw_hi_raw.notna(), close)
    lw_lo = lw_lo_raw.where((lw_lo_raw > 0) & lw_lo_raw.notna(), close)

    # Clip la max 100×ATR (ar fi ~900 pts pe NQ — orice peste e outlier/bug)
    df['dist_weekly_high_atr'] = np.clip(_safe_div(lw_hi - close, atr, 0.0), -100, 100)
    df['dist_weekly_low_atr']  = np.clip(_safe_div(close - lw_lo, atr, 0.0), -100, 100)
    df['weekly_range_atr']     = np.clip(_safe_div(lw_hi - lw_lo, atr, 0.0), 0, 200)

    # weekly_open = true_open (dacă există), altfel folosim first bar of week
    if 'true_open' in df.columns:
        w_open_raw = df['true_open']
        w_open = w_open_raw.where((w_open_raw > 0) & w_open_raw.notna(), close)
    else:
        w_open = close
    df['dist_weekly_open_atr'] = np.clip(_safe_div(close - w_open, atr, 0.0), -100, 100)

    # weekly_bias: comparăm close curent cu lw_hi/lw_lo midpoint ca proxy
    lw_mid = (lw_hi + lw_lo) / 2.0
    df['weekly_bias_up']   = (close > lw_mid).astype(int)
    df['weekly_bias_down'] = (close < lw_mid).astype(int)

    # at weekly high/low
    threshold = 0.3 * atr
    df['session_at_weekly_high'] = (close > (lw_hi - threshold)).astype(int)
    df['session_at_weekly_low']  = (close < (lw_lo + threshold)).astype(int)

    # prev_week_bullish: comparăm close curent cu lw_mid (proxy: bullish dacă peste mid)
    df['prev_week_bullish'] = (close > lw_mid).astype(int)

    # cleanup: fill any NaN from shifts
    for col in FEATURES_WEEKLY:
        if col in df.columns:
            df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

    return df


# ════════════════════════════════════════════════════════════════════════════
# DAY-TYPE CLASSIFICATION FEATURES
# ════════════════════════════════════════════════════════════════════════════
def add_daytype_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de tip zi (trend/range/reversal/outside/inside/gap)."""
    if 'timestamp' not in df.columns:
        for col in FEATURES_DAYTYPE:
            df[col] = 0.0
        return df

    ts = pd.to_datetime(df['timestamp'], errors='coerce')
    df['_date_v13'] = ts.dt.date

    atr = _ensure_atr(df)

    # Per-day accumulative high/low/open
    day_grp = df.groupby('_date_v13')
    df['_day_high']  = day_grp['high'].cummax()
    df['_day_low']   = day_grp['low'].cummin()
    df['_day_open']  = day_grp['open'].transform('first')

    # Prev day close (from pdh/pdl shifted — use close of prev day if available)
    # Folosim p_hi/p_lo (prev day) ca proxy pentru nivelele anterioare
    pdh = df.get('p_hi', df.get('dist_pdh', pd.Series(np.nan, index=df.index)))
    pdl = df.get('p_lo', df.get('dist_pdl', pd.Series(np.nan, index=df.index)))
    prev_close_approx = (pdh + pdl) / 2.0  # aproximat midpoint anterior
    close = df['close']

    # trend day: |close - day_open| / ATR
    df['trend_day_score'] = np.clip(_safe_div(np.abs(close - df['_day_open']), atr, 0.0), 0, 5)

    # range day: invers — range mic = score mare
    day_range = df['_day_high'] - df['_day_low']
    range_atr = _safe_div(day_range, atr, 1.0)
    df['range_day_score'] = 1.0 / np.maximum(range_atr, 1.0)

    # reversal day: high-low mare dar close aproape de open (V/Λ)
    wick_top    = np.maximum(df['_day_high'] - df['_day_open'], df['_day_high'] - close)
    wick_bottom = np.maximum(df['_day_open'] - df['_day_low'],  close - df['_day_low'])
    body        = np.abs(close - df['_day_open'])
    df['reversal_day_score'] = _safe_div(
        np.maximum(wick_top, wick_bottom) - body, atr, 0.0
    ).clip(0, 5)

    # outside/inside day (vs prev day high/low)
    df['outside_day'] = ((df['_day_high'] > pdh) & (df['_day_low'] < pdl)).astype(int)
    df['inside_day']  = ((df['_day_high'] < pdh) & (df['_day_low'] > pdl)).astype(int)

    # gap up/down (day_open vs prev_close_approx)
    gap_threshold = 0.3 * atr
    df['gap_up']   = (df['_day_open'] > prev_close_approx + gap_threshold).astype(int)
    df['gap_down'] = (df['_day_open'] < prev_close_approx - gap_threshold).astype(int)
    df['gap_filled'] = (
        ((df['gap_up'] == 1) & (df['_day_low'] <= prev_close_approx)) |
        ((df['gap_down'] == 1) & (df['_day_high'] >= prev_close_approx))
    ).astype(int)

    # day_vs_open_atr
    df['day_vs_open_atr'] = _safe_div(close - df['_day_open'], atr, 0.0)

    # session sweeps (level-uri spart intraday)
    asia_hi = df.get('asia_hi', pd.Series(np.nan, index=df.index))
    asia_lo = df.get('asia_lo', pd.Series(np.nan, index=df.index))
    lon_hi  = df.get('lon_hi',  pd.Series(np.nan, index=df.index))
    lon_lo  = df.get('lon_lo',  pd.Series(np.nan, index=df.index))

    df['session_sweep_asia_hi'] = (df['_day_high'] > asia_hi).fillna(False).astype(int)
    df['session_sweep_asia_lo'] = (df['_day_low']  < asia_lo).fillna(False).astype(int)
    df['session_sweep_pdh']     = (df['_day_high'] > pdh).fillna(False).astype(int)
    df['session_sweep_pdl']     = (df['_day_low']  < pdl).fillna(False).astype(int)
    df['session_sweep_lon_hi']  = (df['_day_high'] > lon_hi).fillna(False).astype(int)
    df['session_sweep_lon_lo']  = (df['_day_low']  < lon_lo).fillna(False).astype(int)

    # liq counts (nivele peste/sub close)
    h4_hi = df.get('h4_hi', pd.Series(np.nan, index=df.index))
    h4_lo = df.get('h4_lo', pd.Series(np.nan, index=df.index))
    lw_hi = df.get('lw_hi', pd.Series(np.nan, index=df.index))
    lw_lo = df.get('lw_lo', pd.Series(np.nan, index=df.index))

    above = pd.DataFrame({
        'a': (asia_hi > close).fillna(False),
        'b': (pdh     > close).fillna(False),
        'c': (h4_hi   > close).fillna(False),
        'd': (lw_hi   > close).fillna(False),
    })
    below = pd.DataFrame({
        'a': (asia_lo < close).fillna(False),
        'b': (pdl     < close).fillna(False),
        'c': (h4_lo   < close).fillna(False),
        'd': (lw_lo   < close).fillna(False),
    })
    df['liq_above_count'] = above.sum(axis=1).astype(int)
    df['liq_below_count'] = below.sum(axis=1).astype(int)

    # early session high/low — high/low-ul zilei s-a format în prima oră (primele 60 bare per zi)
    # Calculăm pe fiecare zi: bar idx when cummax/cummin was reached
    def _early_flag(grp: pd.DataFrame, col: str, extreme: str) -> pd.Series:
        vals = grp[col].values
        if extreme == 'max':
            idx = np.argmax(vals)
        else:
            idx = np.argmin(vals)
        flag = np.zeros(len(grp), dtype=int)
        # flag rămâne 1 doar pentru bare după momentul stabilirii extremei dacă extrema < 60
        if idx < 60:
            flag[idx:] = 1
        return pd.Series(flag, index=grp.index)

    try:
        df['early_session_high'] = day_grp.apply(
            lambda g: _early_flag(g, 'high', 'max')
        ).reset_index(level=0, drop=True)
        df['early_session_low'] = day_grp.apply(
            lambda g: _early_flag(g, 'low', 'min')
        ).reset_index(level=0, drop=True)
    except Exception:
        df['early_session_high'] = 0
        df['early_session_low']  = 0

    # cleanup
    df = df.drop(columns=['_date_v13', '_day_high', '_day_low', '_day_open'], errors='ignore')
    for col in FEATURES_DAYTYPE:
        if col in df.columns:
            df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

    return df


# ════════════════════════════════════════════════════════════════════════════
# SWEEP DETECTION FEATURES
# ════════════════════════════════════════════════════════════════════════════
def add_sweep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Detectare sweep multi-level + confirmare OF."""
    atr = _ensure_atr(df)
    close = df['close']
    high  = df['high']
    low   = df['low']

    levels_hi = {
        'broke_asia_hi': df.get('asia_hi', pd.Series(np.nan, index=df.index)),
        'broke_pdh':     df.get('p_hi',    pd.Series(np.nan, index=df.index)),
        'broke_lon_hi':  df.get('lon_hi',  pd.Series(np.nan, index=df.index)),
        'broke_h4_hi':   df.get('h4_hi',   pd.Series(np.nan, index=df.index)),
        'broke_h1_hi':   df.get('h1_hi',   pd.Series(np.nan, index=df.index)),
    }
    levels_lo = {
        'broke_asia_lo': df.get('asia_lo', pd.Series(np.nan, index=df.index)),
        'broke_pdl':     df.get('p_lo',    pd.Series(np.nan, index=df.index)),
        'broke_lon_lo':  df.get('lon_lo',  pd.Series(np.nan, index=df.index)),
        'broke_h4_lo':   df.get('h4_lo',   pd.Series(np.nan, index=df.index)),
        'broke_h1_lo':   df.get('h1_lo',   pd.Series(np.nan, index=df.index)),
    }

    for name, lvl in levels_hi.items():
        df[name] = (high > lvl).fillna(False).astype(int)
    for name, lvl in levels_lo.items():
        df[name] = (low < lvl).fillna(False).astype(int)

    # sweep direction: ultimul sweep (up/down) în ultimele 10 bare
    swept_up   = df[[k for k in levels_hi.keys()]].sum(axis=1)
    swept_down = df[[k for k in levels_lo.keys()]].sum(axis=1)
    df['sweep_direction'] = np.where(
        swept_up > swept_down, 1,
        np.where(swept_down > swept_up, -1, 0)
    ).astype(int)

    # sweep_depth_atr: cât de adânc peste nivel (max din levels spart)
    depth_up = pd.Series(0.0, index=df.index)
    for name, lvl in levels_hi.items():
        d = (high - lvl).fillna(0).clip(lower=0)
        depth_up = np.maximum(depth_up, d)
    depth_dn = pd.Series(0.0, index=df.index)
    for name, lvl in levels_lo.items():
        d = (lvl - low).fillna(0).clip(lower=0)
        depth_dn = np.maximum(depth_dn, d)
    df['sweep_depth_atr'] = _safe_div(np.maximum(depth_up, depth_dn), atr, 0.0)

    # reclaimed_after_sweep: spart în ultimele 10 bare dar close actual sub/peste nivel
    sweep_up_10  = swept_up.rolling(10, min_periods=1).max()
    sweep_dn_10  = swept_down.rolling(10, min_periods=1).max()

    # close curent înapoi în range (pentru reclaim)
    reclaim_up = pd.Series(0, index=df.index)
    for name, lvl in levels_hi.items():
        had_sweep = (df[name].rolling(10, min_periods=1).max() > 0)
        back_below = (close < lvl)
        reclaim_up = reclaim_up | (had_sweep & back_below.fillna(False)).astype(int)
    reclaim_dn = pd.Series(0, index=df.index)
    for name, lvl in levels_lo.items():
        had_sweep = (df[name].rolling(10, min_periods=1).max() > 0)
        back_above = (close > lvl)
        reclaim_dn = reclaim_dn | (had_sweep & back_above.fillna(False)).astype(int)
    df['reclaimed_after_sweep'] = (reclaim_up | reclaim_dn).astype(int)

    # bars_since_sweep: bare de la ultimul sweep any direction
    any_sweep = ((swept_up + swept_down) > 0).astype(int)
    bars_since = []
    counter = 999
    for v in any_sweep.values:
        if v == 1:
            counter = 0
        else:
            counter = min(counter + 1, 999)
        bars_since.append(counter)
    df['bars_since_sweep'] = bars_since

    # OF confirmation at sweep
    delta = df.get('bar_delta', df.get('cum_delta', pd.Series(0.0, index=df.index)))
    # delta_at_sweep_extreme: delta bar current când e sweep
    df['delta_at_sweep_extreme'] = np.where(any_sweep == 1, delta, 0.0)

    # delta_reversal_at_sweep: 1 dacă semnul delta s-a inversat în bara sweep vs previous
    delta_prev = delta.shift(1).fillna(0)
    reversal_mask = ((np.sign(delta) != np.sign(delta_prev)) & (any_sweep == 1))
    df['delta_reversal_at_sweep'] = reversal_mask.astype(int)

    # big trades at sweep
    big_buy  = df.get('big_buy_count',  pd.Series(0, index=df.index))
    big_sell = df.get('big_sell_count', pd.Series(0, index=df.index))
    df['big_trades_at_sweep'] = np.where(
        any_sweep == 1, np.maximum(big_buy, big_sell), 0
    ).astype(int)

    # absorption at sweep
    absorption = df.get('absorption_score', pd.Series(0.0, index=df.index))
    df['absorption_at_sweep'] = np.where(any_sweep == 1, absorption, 0.0)

    # displacement after sweep (FVG format în 5 bare post-sweep)
    fvg_up   = df.get('fvg_up',   pd.Series(0, index=df.index))
    fvg_down = df.get('fvg_down', pd.Series(0, index=df.index))
    fvg_any  = ((fvg_up + fvg_down) > 0).astype(int)
    post_sweep_fvg = fvg_any.rolling(5, min_periods=1).max().shift(1).fillna(0)
    df['displacement_after_sweep'] = np.where(
        any_sweep.shift(1).fillna(0) == 1, post_sweep_fvg, 0
    ).astype(int)

    # cleanup
    for col in FEATURES_SWEEP:
        if col in df.columns:
            df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

    return df


# ════════════════════════════════════════════════════════════════════════════
# REGIME-AWARE TARGET GENERATION (9 CLASES)
# ════════════════════════════════════════════════════════════════════════════
def generate_regime_target(
    df: pd.DataFrame,
    horizon: int = 60,
    atr_expansion: float = 1.5,        # v14: relaxat (era 2.0)
    atr_reversal_push: float = 1.8,    # v14: relaxat
    atr_reversal_retrace: float = 1.2, # v14: relaxat
    va_tolerance_pts: float = 5.0,
    or_window_min: int = 30,
    tp_atr: float = 2.0,               # v14: ținta pentru resolve (stop lookahead)
    invalid_atr: float = 1.2,          # v14: mișcare opusă → invalid
) -> pd.Series:
    """
    Generează target multiclass v13 (9 clase).

    Pentru fiecare bar din killzone post-OR:
      - Privim înainte `horizon` bare pentru a clasifica cum s-a comportat prețul
      - Atribuim una din cele 9 clase (0-8)

    Args:
        horizon: lookahead în bare (default 60 min)
        atr_expansion: prag minim expansion (în ATR)
        atr_reversal_push: cât de sus/jos să meargă înainte de reversal
        atr_reversal_retrace: cât să se întoarcă pentru a califica ca reversal
        va_tolerance_pts: toleranță absorption în afara VA (5 pts)
        or_window_min: dimensiune OR period (30 min)
    """
    n = len(df)
    target = np.zeros(n, dtype=np.int8)

    atr = _ensure_atr(df).values
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values

    vah = df.get('vah', pd.Series(np.nan, index=df.index)).values
    val = df.get('val', pd.Series(np.nan, index=df.index)).values
    poc = df.get('poc_level', pd.Series(np.nan, index=df.index)).values

    asia_hi = df.get('asia_hi', pd.Series(np.nan, index=df.index)).values
    asia_lo = df.get('asia_lo', pd.Series(np.nan, index=df.index)).values
    pdh     = df.get('p_hi',    pd.Series(np.nan, index=df.index)).values
    pdl     = df.get('p_lo',    pd.Series(np.nan, index=df.index)).values

    # Killzone detection (folosește timestamp)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
        hour_dec = (ts.dt.hour + ts.dt.minute / 60.0).values
    else:
        hour_dec = np.zeros(n)

    # Determinăm la fiecare bar dacă suntem în killzone post-OR
    in_kz_lon = (hour_dec >= 9.0) & (hour_dec <= 11.0)
    in_kz_ny  = (hour_dec >= 15.5) & (hour_dec <= 17.5)
    in_kz     = in_kz_lon | in_kz_ny
    # Post-OR: bar time > killzone_start + 30 min
    post_or_lon = (hour_dec >= 9.5)  & (hour_dec <= 11.0)
    post_or_ny  = (hour_dec >= 16.0) & (hour_dec <= 17.5)
    post_or     = (post_or_lon & in_kz_lon) | (post_or_ny & in_kz_ny)

    # Calculăm OR_high/OR_low per zi × killzone (bar-wise cummax peste OR period)
    dates = pd.to_datetime(df['timestamp'], errors='coerce').dt.date.values
    or_hi_arr = np.full(n, np.nan)
    or_lo_arr = np.full(n, np.nan)

    # Group iterativ: pentru fiecare (date, kz), OR = first 30 min
    cur_date = None
    cur_kz   = None
    cur_or_hi = -np.inf
    cur_or_lo = np.inf
    cur_or_done = False
    for i in range(n):
        if not in_kz[i]:
            cur_date = None
            cur_kz   = None
            continue
        this_kz = 'LON' if in_kz_lon[i] else 'NY'
        if dates[i] != cur_date or this_kz != cur_kz:
            cur_date = dates[i]
            cur_kz   = this_kz
            cur_or_hi = high[i]
            cur_or_lo = low[i]
            cur_or_done = False
        # Suntem în OR period?
        in_or = (not post_or[i])
        if in_or and not cur_or_done:
            cur_or_hi = max(cur_or_hi, high[i])
            cur_or_lo = min(cur_or_lo, low[i])
        else:
            cur_or_done = True
        if cur_or_done:
            or_hi_arr[i] = cur_or_hi
            or_lo_arr[i] = cur_or_lo

    # Extra features pentru regime rules
    delta = df.get('bar_delta', pd.Series(0.0, index=df.index)).values
    fvg_up = df.get('fvg_up', pd.Series(0, index=df.index)).values
    fvg_down = df.get('fvg_down', pd.Series(0, index=df.index)).values
    absorption = df.get('absorption_score', pd.Series(0.0, index=df.index)).values
    inside_va_arr = df.get('inside_va', pd.Series(0, index=df.index)).values

    # Iterăm doar pe barele post-OR
    qualifying = np.where(post_or)[0]

    for idx in qualifying:
        if idx + horizon >= n:
            continue

        entry_px = close[idx]
        a = atr[idx] if atr[idx] > 0 else 9.0

        # ── v14: VARIABLE HORIZON RESOLUTION ──
        # În loc să privim 60 bare fix (autocorelație!), iterăm bară cu bară
        # până setup-ul se rezolvă: +tp_atr (win) sau -invalid_atr (loss).
        # Rezultatul: label-ul reprezintă real outcome-ul, nu "cum se mișcă în 60 min".
        tp_up   = tp_atr * a
        tp_dn   = tp_atr * a
        inv_up  = invalid_atr * a  # pentru SHORT: cât poate urca înainte să devină invalid
        inv_dn  = invalid_atr * a  # pentru LONG: cât poate coborî

        resolved_dir = 0   # 0=timeout, +1=up hit, -1=down hit
        resolved_bar = 0
        max_up_pre = 0.0
        max_dn_pre = 0.0

        for h in range(1, horizon + 1):
            fi = idx + h
            if fi >= n:
                break
            up_move = high[fi] - entry_px
            dn_move = entry_px - low[fi]
            max_up_pre = max(max_up_pre, up_move)
            max_dn_pre = max(max_dn_pre, dn_move)

            # Rezoluție: care atinge prima tp_atr?
            if up_move >= tp_up and dn_move >= tp_dn:
                # Ambele în același bar — folosim close pentru tiebreaker
                resolved_dir = +1 if close[fi] > entry_px else -1
                resolved_bar = h
                break
            if up_move >= tp_up:
                resolved_dir = +1
                resolved_bar = h
                break
            if dn_move >= tp_dn:
                resolved_dir = -1
                resolved_bar = h
                break

        if resolved_dir == 0:
            # Timeout fără resolve → skip (target rămâne 0=WAIT)
            continue

        # ── Clasificare pe baza resolve + pre-resolve drawdown ──
        if resolved_dir == +1:
            # Mișcare LONG rezolvată. Diferențiere: REV (a avut sweep jos înainte) vs BREAK.
            # Sweep jos = max_dn_pre > invalid_atr*0.5 sau broke_pdl/asia_lo ÎNAINTE de resolve
            fut_window_lo = low[idx+1: idx+resolved_bar+1]
            swept_before = False
            if not np.isnan(asia_lo[idx]):
                swept_before = swept_before or (fut_window_lo.min() < asia_lo[idx] - 0.2*a)
            if not np.isnan(pdl[idx]):
                swept_before = swept_before or (fut_window_lo.min() < pdl[idx] - 0.2*a)
            # Alternativ: pre-resolve DD >= 1×ATR = clear reversal
            deep_dd = max_dn_pre >= 1.0 * a

            if swept_before or deep_dd:
                target[idx] = REGIME_LONG_REV
            else:
                # Drum curat sus → BREAKOUT
                if max_up_pre >= atr_expansion * a:
                    target[idx] = REGIME_LONG_BREAK
                # altfel prea puțin convingător — rămâne 0
        else:  # resolved_dir == -1
            fut_window_hi = high[idx+1: idx+resolved_bar+1]
            swept_before = False
            if not np.isnan(asia_hi[idx]):
                swept_before = swept_before or (fut_window_hi.max() > asia_hi[idx] + 0.2*a)
            if not np.isnan(pdh[idx]):
                swept_before = swept_before or (fut_window_hi.max() > pdh[idx] + 0.2*a)
            deep_du = max_up_pre >= 1.0 * a

            if swept_before or deep_du:
                target[idx] = REGIME_SHORT_REV
            else:
                if max_dn_pre >= atr_expansion * a:
                    target[idx] = REGIME_SHORT_BREAK

    return pd.Series(target, index=df.index, name='target_regime')


# ════════════════════════════════════════════════════════════════════════════
# DYNAMIC SL (structural, OR-width, regime-adjusted)
# ════════════════════════════════════════════════════════════════════════════
def dynamic_sl(
    entry_px: float,
    direction: str,
    recent_bars: pd.DataFrame,  # ultimele ~10 bare ale candelabrului (open/high/low/close)
    atr: float,
    or_high: Optional[float] = None,
    or_low: Optional[float] = None,
    regime: int = REGIME_WAIT,
    sl_floor: float = 12.0,
    sl_ceil: float = 40.0,
) -> float:
    """
    Returnează SL point-distance adaptiv.
      - Structural: swing low/high din ultimele bare
      - OR-based: 50% OR width
      - Regime-adjusted: expansion = larg, sweep/mean-rev = strâns, reversal = moderat
    """
    # Structural SL
    if direction == "LONG":
        swing_low = recent_bars['low'].min() if len(recent_bars) > 0 else entry_px - atr
        struct_sl_pts = entry_px - swing_low
    else:
        swing_high = recent_bars['high'].max() if len(recent_bars) > 0 else entry_px + atr
        struct_sl_pts = swing_high - entry_px

    # OR-based fallback
    if or_high is not None and or_low is not None:
        or_sl_pts = (or_high - or_low) * 0.5
    else:
        or_sl_pts = atr * 1.0

    # Regime-based base
    if regime in (REGIME_LONG_AFTER_SWEEP, REGIME_SHORT_AFTER_SWEEP):
        # Sweep: SL imediat sub/peste swept level (tight)
        base = max(struct_sl_pts, 0.6 * atr)
    elif regime in (REGIME_MEAN_REV_SHORT_VAH, REGIME_MEAN_REV_LONG_VAL):
        # Mean-rev: SL la celălalt capăt VA / OR extrem (tight)
        base = max(struct_sl_pts, 0.5 * atr)
    elif regime in (REGIME_LONG_EXPANSION, REGIME_SHORT_EXPANSION):
        # Expansion: SL larg pentru pullback
        base = max(struct_sl_pts, or_sl_pts, 1.2 * atr)
    elif regime in (REGIME_SHORT_REVERSAL_HIGH, REGIME_LONG_REVERSAL_LOW):
        # Reversal: SL la vârf/fund swing + buffer
        base = max(struct_sl_pts + 0.2 * atr, 1.0 * atr)
    else:
        base = max(struct_sl_pts, 1.0 * atr)

    return max(sl_floor, min(base, sl_ceil))


# ════════════════════════════════════════════════════════════════════════════
# DYNAMIC EXIT SIGNAL (rule-based — runs each bar post-entry)
# ════════════════════════════════════════════════════════════════════════════
def compute_exit_signal(
    direction: str,
    entry_px: float,
    current_bar: Dict[str, Any],
    recent_bars: pd.DataFrame,
    regime: int,
    atr: float,
    mfe_so_far: float,
    mae_so_far: float,
    bars_since_entry: int,
    key_levels: Dict[str, float],  # {'poc', 'vah', 'val', 'pdh', 'pdl', 'asia_hi', 'asia_lo'}
) -> Tuple[bool, str]:
    """
    Returnează (should_exit, reason).

    Exit signals (prioritizate):
      1. Key level reached (TP hit la target regime-specific)
      2. OF exhaustion (absorption opposite, displacement against, delta flip)
      3. Time limit (regime-specific)

    Atenție: NU înlocuiește trailing SL — e suplimentar, rule-based safety.
    """
    cur_price = current_bar.get('close', entry_px)
    cur_delta = current_bar.get('bar_delta', 0.0)
    cur_absorption = current_bar.get('absorption_score', 0.0)
    cur_absorption_side = current_bar.get('absorption_side', '')

    pnl_pts = (cur_price - entry_px) if direction == "LONG" else (entry_px - cur_price)

    # ── 1. Key level targets (regime-specific) ──
    # LONG_EXPANSION → target VAH sau next HVN
    if regime == REGIME_LONG_EXPANSION and direction == "LONG":
        target_px = key_levels.get('pdh') or (entry_px + 2.5 * atr)
        if cur_price >= target_px:
            return True, "KL_PDH_HIT"
    if regime == REGIME_SHORT_EXPANSION and direction == "SHORT":
        target_px = key_levels.get('pdl') or (entry_px - 2.5 * atr)
        if cur_price <= target_px:
            return True, "KL_PDL_HIT"

    # Sweep trades → target = prior range midpoint / pdh-pdl mid
    if regime == REGIME_LONG_AFTER_SWEEP and direction == "LONG":
        target_px = key_levels.get('pdh') or key_levels.get('vah') or (entry_px + 2.0 * atr)
        if cur_price >= target_px:
            return True, "KL_SWEEP_TGT"
    if regime == REGIME_SHORT_AFTER_SWEEP and direction == "SHORT":
        target_px = key_levels.get('pdl') or key_levels.get('val') or (entry_px - 2.0 * atr)
        if cur_price <= target_px:
            return True, "KL_SWEEP_TGT"

    # Reversal → target = VWAP / POC
    if regime == REGIME_SHORT_REVERSAL_HIGH and direction == "SHORT":
        target_px = key_levels.get('poc') or key_levels.get('val') or (entry_px - 1.5 * atr)
        if cur_price <= target_px:
            return True, "KL_POC"
    if regime == REGIME_LONG_REVERSAL_LOW and direction == "LONG":
        target_px = key_levels.get('poc') or key_levels.get('vah') or (entry_px + 1.5 * atr)
        if cur_price >= target_px:
            return True, "KL_POC"

    # Mean-rev → target = POC
    if regime == REGIME_MEAN_REV_SHORT_VAH and direction == "SHORT":
        target_px = key_levels.get('poc') or key_levels.get('val')
        if target_px and cur_price <= target_px:
            return True, "KL_POC_MR"
    if regime == REGIME_MEAN_REV_LONG_VAL and direction == "LONG":
        target_px = key_levels.get('poc') or key_levels.get('vah')
        if target_px and cur_price >= target_px:
            return True, "KL_POC_MR"

    # ── 2. OF exhaustion against ──
    if pnl_pts > 0.5 * atr:  # doar dacă suntem în profit, nu tăiem loss early
        if direction == "LONG":
            # selling absorption la vârf
            if cur_absorption > 0.7 and cur_absorption_side in ("SELL", "ASK"):
                return True, "OF_ABSORPTION"
            if cur_delta < -100 and pnl_pts > 1.0 * atr:
                return True, "DELTA_FLIP"
        else:
            if cur_absorption > 0.7 and cur_absorption_side in ("BUY", "BID"):
                return True, "OF_ABSORPTION"
            if cur_delta > 100 and pnl_pts > 1.0 * atr:
                return True, "DELTA_FLIP"

    # ── 3. Regime time limits (overtime exit) ──
    time_limits = {
        REGIME_LONG_EXPANSION: 90, REGIME_SHORT_EXPANSION: 90,
        REGIME_LONG_AFTER_SWEEP: 60, REGIME_SHORT_AFTER_SWEEP: 60,
        REGIME_SHORT_REVERSAL_HIGH: 45, REGIME_LONG_REVERSAL_LOW: 45,
        REGIME_MEAN_REV_SHORT_VAH: 30, REGIME_MEAN_REV_LONG_VAL: 30,
    }
    limit = time_limits.get(regime, 45)
    if bars_since_entry >= limit:
        return True, f"TIME_{limit}M"

    return False, ""


# ════════════════════════════════════════════════════════════════════════════
# QA / SELF-TEST
# ════════════════════════════════════════════════════════════════════════════
def self_test():
    """Smoke test: generează un DataFrame sintetic și verifică că toate funcțiile rulează."""
    print("🔬 Aladin v13 self-test...")
    n = 500
    rng = np.random.default_rng(42)

    ts = pd.date_range('2025-01-01 08:00', periods=n, freq='1min')
    prices = 21000 + np.cumsum(rng.standard_normal(n))
    df = pd.DataFrame({
        'timestamp': ts,
        'open':  prices + rng.standard_normal(n) * 0.5,
        'high':  prices + np.abs(rng.standard_normal(n)) * 2.0,
        'low':   prices - np.abs(rng.standard_normal(n)) * 2.0,
        'close': prices,
        'volume': rng.integers(100, 5000, n),
        'atr_14': np.full(n, 9.0),
        'asia_hi': prices + 15, 'asia_lo': prices - 15,
        'p_hi':    prices + 25, 'p_lo':    prices - 25,
        'lon_hi':  prices + 10, 'lon_lo':  prices - 10,
        'h4_hi':   prices + 40, 'h4_lo':   prices - 40,
        'h1_hi':   prices + 20, 'h1_lo':   prices - 20,
        'lw_hi':   prices + 80, 'lw_lo':   prices - 80,
        'true_open': prices[0] * np.ones(n),
        'vah':     prices + 8, 'val':     prices - 8, 'poc_level': prices,
        'inside_va': rng.integers(0, 2, n),
        'bar_delta': rng.standard_normal(n) * 100,
        'cum_delta': np.cumsum(rng.standard_normal(n) * 100),
        'fvg_up':  rng.integers(0, 2, n), 'fvg_down': rng.integers(0, 2, n),
        'absorption_score': rng.random(n),
        'absorption_side': rng.choice(['BUY','SELL','NONE'], n),
        'big_buy_count':  rng.integers(0, 10, n),
        'big_sell_count': rng.integers(0, 10, n),
        'dist_pdh': prices + 25, 'dist_pdl': prices - 25,
    })

    print(f"   📊 DataFrame: {len(df)} bars")

    # Test weekly
    df1 = add_weekly_features(df.copy())
    missing = [c for c in FEATURES_WEEKLY if c not in df1.columns]
    assert not missing, f"Missing weekly features: {missing}"
    print(f"   ✓ add_weekly_features: {len(FEATURES_WEEKLY)} cols ok")

    # Test daytype
    df2 = add_daytype_features(df1.copy())
    missing = [c for c in FEATURES_DAYTYPE if c not in df2.columns]
    assert not missing, f"Missing daytype features: {missing}"
    print(f"   ✓ add_daytype_features: {len(FEATURES_DAYTYPE)} cols ok")

    # Test sweep
    df3 = add_sweep_features(df2.copy())
    missing = [c for c in FEATURES_SWEEP if c not in df3.columns]
    assert not missing, f"Missing sweep features: {missing}"
    print(f"   ✓ add_sweep_features: {len(FEATURES_SWEEP)} cols ok")

    # Test regime target
    target = generate_regime_target(df3, horizon=60)
    uniq = sorted(np.unique(target))
    counts = {int(k): int((target == k).sum()) for k in uniq}
    print(f"   ✓ generate_regime_target: classes={counts}")

    # Test dynamic_sl
    recent = df3.iloc[100:110]
    for reg in range(9):
        sl = dynamic_sl(
            entry_px=21000, direction="LONG",
            recent_bars=recent, atr=9.0,
            or_high=21010, or_low=20990, regime=reg,
        )
        assert 12.0 <= sl <= 40.0, f"SL out of clamp: regime={reg} sl={sl}"
    print(f"   ✓ dynamic_sl: clamp 12-40 ok pentru toate regimele")

    # Test exit signal
    current_bar = {'close': 21020, 'bar_delta': -150, 'absorption_score': 0.8, 'absorption_side': 'SELL'}
    should_exit, reason = compute_exit_signal(
        direction="LONG", entry_px=21000, current_bar=current_bar,
        recent_bars=recent, regime=REGIME_LONG_EXPANSION, atr=9.0,
        mfe_so_far=20, mae_so_far=3, bars_since_entry=15,
        key_levels={'pdh': 21025, 'vah': 21008, 'poc': 21000},
    )
    print(f"   ✓ compute_exit_signal: should_exit={should_exit} reason={reason}")

    # Data integrity: no NaN/Inf
    all_new = FEATURES_WEEKLY + FEATURES_DAYTYPE + FEATURES_SWEEP
    nan_cols = [c for c in all_new if df3[c].isna().any()]
    inf_cols = [c for c in all_new if np.isinf(df3[c]).any()]
    assert not nan_cols, f"NaN in: {nan_cols}"
    assert not inf_cols, f"Inf in: {inf_cols}"
    print(f"   ✓ No NaN/Inf in {len(all_new)} new features")

    print("✅ Aladin v13 self-test PASSED")
    return True


if __name__ == "__main__":
    self_test()
