"""
train_regime.py — Regime Classifier v1.1
=========================================
Antrenează un model XGBoost multi-clasă pentru detectarea regimului de piață.

Regimuri:
  0 = CONSOLIDATION   — ADX scăzut, inside VA, fără direcție
  1 = PRE_EXPANSION   — sweep lichiditate la session open, manipulare activă
  2 = EXPANSION       — mișcare HTF în desfășurare
  3 = RETRACEMENT     — pullback în expansiune spre VWAP/POC
  4 = DISTRIBUTION    — exhaustion la extreme, smart money descarcă

v1.1 changes:
  ✅ Session-aware PRE_EXPANSION thresholds: LON MIN_MOVE_ATR=0.8, FWD_BARS=20
     (LON moves are smaller than NY; 1.2×ATR was killing LON PRE_EXPANSION)
  ✅ Lagged OF features: previous session cvd_final, absorption_score_mean,
     opening_drive_dir joined as leading indicators
     (prev session OF predicts next session opening regime)

Features: combinate din v2/v5/v6 + NOM/LOM + DSM + lagged OF (toate din DB)
Labeling: rule-based pe date istorice + forward validation

Output: regime_classifier_v1.pkl
"""

import sqlite3, warnings, joblib, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH    = Path(__file__).parent / "mario_trading.db"
OUT_MODEL  = Path(__file__).parent / "regime_classifier_v1.pkl"
OUT_META   = Path(__file__).parent / "regime_classifier_v1_meta.json"

TRAIN_START = "2022-01-01"
TRAIN_END   = "2024-12-31"
OOS_START   = "2025-01-01"
OOS_END     = "2026-04-22"

# Sampling — evităm OOM pe 1.8M bare
SAMPLE_IS   = 300_000   # bare IS pentru antrenare
SAMPLE_OOS  =  80_000   # bare OOS pentru evaluare

# Ferestre sesiuni în ET (hour_min e stocat ET în DB)
LON_OPEN_ET   = (400,  700)   # 04:00–07:00 ET = 08:00–11:00 UTC
NY_OPEN_ET    = (900, 1130)   # 09:00–11:30 ET = 13:00–15:30 UTC

# Praguri labeling (calibrate pe dist_vwap_atr: median=4.35, p75=8, p90=13)
ADX_LOW       = 18    # sub asta → consolidare
ADX_HIGH      = 25    # peste asta → expansion
HURST_TREND   = 0.52  # > asta → trending
VWAP_CLOSE    = 3.0   # dist_vwap / atr sub asta → aproape VWAP (sub median)
VWAP_EXTREME  = 10.0  # dist_vwap / atr peste asta → extrem (p80+)
LOOKBACK      = 30    # bare lookback pentru context

# PRE_EXPANSION — session-aware thresholds
# LON: mișcări mai mici, manipulare mai comprimată → prag mai permisiv
# NY:  mișcări mai mari, sample mai mare → prag standard
FWD_BARS_LON     = 20   # LON: 20 bare look-ahead (era 30 → tăia prea mult)
FWD_BARS_NY      = 30   # NY:  30 bare look-ahead (neschimbat)
MIN_MOVE_ATR_LON = 0.8  # LON: 0.8×ATR (era 1.2 → cause PRE_EXP=1/881)
MIN_MOVE_ATR_NY  = 1.2  # NY:  1.2×ATR (neschimbat)

# Backward compat aliases (folosite în afara label_regimes, e.g. unit tests)
FWD_BARS     = FWD_BARS_NY
MIN_MOVE_ATR = MIN_MOVE_ATR_NY

# Path la orderflow features (pentru lagged OF features)
OF_PATH = Path(__file__).parent / "data" / "orderflow_features.parquet"


REGIME_NAMES = {
    0: 'CONSOLIDATION',
    1: 'PRE_EXPANSION',
    2: 'EXPANSION',
    3: 'RETRACEMENT',
    4: 'DISTRIBUTION',
}

FEATURES = [
    # Structură / trend (din v5/v6)
    'adx_14', 'hurst', 'garch_vol', 'kalman_smooth',
    'acf_lag1', 'acf_lag5', 'fisher_transform', 'sample_entropy',
    # Value Area / VWAP
    'inside_va', 'dist_vwap_atr', 'dist_poc_atr', 'dist_pdh_atr', 'dist_pdl_atr',
    # Displacement / setup (din NOM/LOM/DSM)
    'has_displacement', 'body_size_atr', 'rvol',
    # Orderflow (disponibil în DB)
    'bar_delta_atr', 'cum_delta_20_atr',
    'delta_at_high_atr', 'delta_at_low_atr',
    'big_buy_count', 'big_sell_count',
    # Imbalance
    'imbalance_pct', 'dom_ratio',
    # Session / poziție
    'hhmm_enc', 'is_session_open', 'is_lon_session', 'is_ny_session',
    'dist_sess_hi_atr', 'dist_sess_lo_atr',
    'h4_bias_atr', 'h1_bias_atr', 'above_true_open_atr',
    # Calendar
    'day_of_week', 'month',
    # FVG
    'fvg_up', 'fvg_down',
    # Sweep context (derivate din NOM/LOM/DSM)
    'pre_range_atr', 'sweep_dn_atr', 'sweep_up_atr',
    # Lagged OF features — previous session predicts current session opening regime
    # (e.g. LON cvd predicts NY opening regime; previous LON predicts current LON)
    'of_cvd_lag1', 'of_absorption_lag1', 'of_opening_drive_lag1',
    'of_cvd_zscore_lag1', 'of_stacked_imbalance_lag1', 'of_opening_range_lag1',
]


# ── Lagged OF features ────────────────────────────────────────────────────────
def build_lagged_of_lookup():
    """
    Returns a dict: date_str → {'of_cvd_lag1': ..., 'of_absorption_lag1': ...}
    Logic:
      - For LON session on day D: lag1 = NY session from day D-1
        (prev NY OF → LON opening regime)
      - For NY session on day D: lag1 = LON session from same day D
        (LON OF → NY opening regime, same day)
    We build two lookups (by date) and merge them during feature engineering.
    """
    if not OF_PATH.exists():
        print(f"  ⚠️  OF parquet not found at {OF_PATH}, skipping lagged OF features")
        return {}

    of = pd.read_parquet(OF_PATH)
    of['date'] = of['date'].astype(str)

    # Extract key columns (use 0 if missing)
    def safe_col(df, col):
        return df[col] if col in df.columns else pd.Series(0.0, index=df.index)

    lon = of[of['session_type'] == 'LON'][['date']].copy()
    lon['cvd_f']   = safe_col(of[of['session_type'] == 'LON'], 'cvd_final').values
    lon['abs_f']   = safe_col(of[of['session_type'] == 'LON'], 'absorption_score_mean').values
    lon['od_f']    = safe_col(of[of['session_type'] == 'LON'], 'opening_drive_dir').values
    lon['cvdz_f']  = safe_col(of[of['session_type'] == 'LON'], 'cvd_zscore_20d').values
    lon['si_f']    = safe_col(of[of['session_type'] == 'LON'], 'stacked_imbalance_count').values
    lon['or_f']    = safe_col(of[of['session_type'] == 'LON'], 'opening_range').values

    ny = of[of['session_type'] == 'NY'][['date']].copy()
    ny['cvd_f']   = safe_col(of[of['session_type'] == 'NY'], 'cvd_final').values
    ny['abs_f']   = safe_col(of[of['session_type'] == 'NY'], 'absorption_score_mean').values
    ny['od_f']    = safe_col(of[of['session_type'] == 'NY'], 'opening_drive_dir').values
    ny['cvdz_f']  = safe_col(of[of['session_type'] == 'NY'], 'cvd_zscore_20d').values
    ny['si_f']    = safe_col(of[of['session_type'] == 'NY'], 'stacked_imbalance_count').values
    ny['or_f']    = safe_col(of[of['session_type'] == 'NY'], 'opening_range').values

    lon = lon.sort_values('date').reset_index(drop=True)
    ny  = ny.sort_values('date').reset_index(drop=True)

    # LON on day D uses NY from D-1 (previous trading day)
    # Build lookup: for each date, what was the NY OF the previous trading day?
    lon_lag_lookup = {}
    ny_dict        = {r['date']: r for _, r in ny.iterrows()}
    lon_dict_check = set(lon['date'].tolist())
    ny_dict_check  = set(ny['date'].tolist())
    all_dates = sorted(lon_dict_check | ny_dict_check)
    prev_ny_row = None
    for d in all_dates:
        # Check LON FIRST, using prev_ny_row (from previous trading day's NY)
        if d in lon_dict_check:
            if prev_ny_row is not None:
                lon_lag_lookup[d] = {
                    'of_cvd_lag1':               float(prev_ny_row['cvd_f']),
                    'of_absorption_lag1':         float(prev_ny_row['abs_f']),
                    'of_opening_drive_lag1':      float(prev_ny_row['od_f']),
                    'of_cvd_zscore_lag1':         float(prev_ny_row['cvdz_f']),
                    'of_stacked_imbalance_lag1':  float(prev_ny_row['si_f']),
                    'of_opening_range_lag1':      float(prev_ny_row['or_f']),
                }
        # THEN update prev_ny_row for the next day's LON
        if d in ny_dict_check:
            prev_ny_row = ny_dict.get(d)

    # NY on day D uses LON from same day D (LON precedes NY)
    lon_dict = {r['date']: r for _, r in lon.iterrows()}
    ny_lag_lookup = {}
    for _, r in ny.iterrows():
        d = r['date']
        if d in lon_dict:
            lr = lon_dict[d]
            ny_lag_lookup[d] = {
                'of_cvd_lag1':             float(lr['cvd_f']),
                'of_absorption_lag1':      float(lr['abs_f']),
                'of_opening_drive_lag1':   float(lr['od_f']),
                'of_cvd_zscore_lag1':      float(lr['cvdz_f']),
                'of_stacked_imbalance_lag1': float(lr['si_f']),
                'of_opening_range_lag1':   float(lr['or_f']),
            }

    print(f"  Lagged OF lookup: {len(lon_lag_lookup)} LON dates, {len(ny_lag_lookup)} NY dates")
    return {'LON': lon_lag_lookup, 'NY': ny_lag_lookup}


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data(start, end, max_rows=None):
    print(f"  Loading {start} → {end} ...")
    conn = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True, timeout=30,
                           check_same_thread=False)

    COLS = """timestamp, date, hour_min, open, high, low, close, volume,
               adx_14, hurst, garch_vol, kalman_smooth,
               acf_lag1, acf_lag5, fisher_transform, sample_entropy,
               inside_va, dist_vwap, dist_poc, dist_pdh, dist_pdl,
               has_displacement, body_size, rvol,
               bar_delta, cum_delta, absorption_score,
               of_doi, of_big_balance, stacked_bull, stacked_bear,
               delta_at_high, delta_at_low, big_buy_count, big_sell_count,
               imbalance_pct, dom_ratio, fvg_up, fvg_down,
               atr_14, true_open, h4_hi, h4_lo, h1_hi, h1_lo,
               lon_hi, lon_lo, p_hi, p_lo, lw_hi, lw_lo,
               vah, val, poc_level, vwap,
               day_of_week, month"""

    base_where = f"date BETWEEN '{start}' AND '{end}' AND adx_14>0 AND atr_14>0 AND garch_vol>0"

    # Număr total disponibil
    total = pd.read_sql(f"SELECT COUNT(*) as n FROM market_data WHERE {base_where}", conn).iloc[0, 0]
    print(f"  → disponibil: {total:,} bare")

    if max_rows and max_rows < total:
        every_n = max(1, int(total / max_rows))

        # ── Stratified: always include session-open bars (first 10 bars of LON/NY) ──
        # PRE_EXPANSION is transient (1-2 bars at session open) and gets skipped by
        # uniform sampling.  Pull session-open bars separately and union them in.
        session_open_q = f"""
            SELECT {COLS} FROM market_data
            WHERE {base_where}
              AND (
                CAST(REPLACE(hour_min,':','') AS INT) BETWEEN 400 AND 410
                OR
                CAST(REPLACE(hour_min,':','') AS INT) BETWEEN 900 AND 910
              )
            ORDER BY timestamp
        """
        df_open = pd.read_sql(session_open_q, conn)
        print(f"  → session-open bars always included: {len(df_open):,}")

        # Remaining sample (uniform, minus what we already have)
        sample_q = f"""
            SELECT {COLS} FROM market_data
            WHERE {base_where}
              AND NOT (
                CAST(REPLACE(hour_min,':','') AS INT) BETWEEN 400 AND 410
                OR
                CAST(REPLACE(hour_min,':','') AS INT) BETWEEN 900 AND 910
              )
              AND (ROWID % {every_n} = 0)
            ORDER BY timestamp
        """
        df_rest = pd.read_sql(sample_q, conn)
        df = pd.concat([df_open, df_rest], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    else:
        q = f"SELECT {COLS} FROM market_data WHERE {base_where} ORDER BY timestamp"
        df = pd.read_sql(q, conn)

    conn.close()
    print(f"  → {len(df):,} bare încărcate (stratified sample)")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df, lagged_of_lookup=None):
    print("  Building features ...")
    df = df.copy()
    atr = df['atr_14'].clip(lower=0.01)

    # Normalizare per ATR
    df['dist_vwap_atr']      = df['dist_vwap'].abs() / atr
    df['dist_poc_atr']       = df['dist_poc'].abs() / atr
    df['dist_pdh_atr']       = df['dist_pdh'].abs() / atr
    df['dist_pdl_atr']       = df['dist_pdl'].abs() / atr
    df['body_size_atr']      = df['body_size'] / atr
    df['bar_delta_atr']      = df['bar_delta'].abs() / (df['volume'].clip(lower=1))
    df['delta_at_high_atr']  = df['delta_at_high'].abs() / atr
    df['delta_at_low_atr']   = df['delta_at_low'].abs() / atr

    # Cum delta rolling 20
    df['cum_delta_20_atr'] = df['cum_delta'].rolling(20, min_periods=1).sum() / atr

    # Session position
    df['hhmm_enc'] = df['hour_min'].str.replace(':', '').astype(int)

    # Session flags — LON vs NY (folosite pentru session-aware thresholds)
    h = df['hhmm_enc']
    df['is_lon_session'] = ((h >= LON_OPEN_ET[0]) & (h <= LON_OPEN_ET[1])).astype(int)
    df['is_ny_session']  = ((h >= NY_OPEN_ET[0])  & (h <= NY_OPEN_ET[1])).astype(int)
    df['is_session_open'] = ((df['is_lon_session'] == 1) | (df['is_ny_session'] == 1)).astype(int)

    # Distance to session range
    df['sess_hi'] = df[['lon_hi', 'p_hi']].max(axis=1)
    df['sess_lo'] = df[['lon_lo', 'p_lo']].min(axis=1)
    df['dist_sess_hi_atr'] = (df['sess_hi'] - df['close']).abs() / atr
    df['dist_sess_lo_atr'] = (df['close'] - df['sess_lo']).abs() / atr

    # HTF bias
    df['h4_mid'] = (df['h4_hi'] + df['h4_lo']) / 2
    df['h1_mid'] = (df['h1_hi'] + df['h1_lo']) / 2
    df['h4_bias_atr']          = (df['close'] - df['h4_mid']) / atr
    df['h1_bias_atr']          = (df['close'] - df['h1_mid']) / atr
    df['above_true_open_atr']  = (df['close'] - df['true_open']) / atr

    # Pre-session range (p_hi/p_lo = previous day range ca proxy)
    df['pre_range'] = (df['p_hi'] - df['p_lo']).clip(lower=0.01)
    df['pre_range_atr'] = df['pre_range'] / atr

    # Sweep magnitude vs previous-day range AND Asia session range
    # NOTE: lon_hi/lon_lo are the final full-session H/L (forward-looking), so a
    # current bar can never exceed them.  Use p_hi/p_lo (prev-day range) and
    # asia_hi/asia_lo as the "levels to sweep" instead.
    asia_hi = df['asia_hi'].values if 'asia_hi' in df.columns else df['p_hi'].values
    asia_lo = df['asia_lo'].values if 'asia_lo' in df.columns else df['p_lo'].values
    # Sweep = bar breaks BELOW p_lo or asia_lo (down sweep) / ABOVE p_hi or asia_hi (up sweep)
    ref_hi = pd.DataFrame({'p': df['p_hi'].values, 'a': asia_hi}).max(axis=1).values
    ref_lo = pd.DataFrame({'p': df['p_lo'].values, 'a': asia_lo}).min(axis=1).values
    df['sweep_dn_atr'] = ((ref_lo - df['low']).clip(lower=0) / atr)
    df['sweep_up_atr'] = ((df['high'] - ref_hi).clip(lower=0) / atr)

    # ── Lagged OF features ────────────────────────────────────────────────────
    lag_cols = ['of_cvd_lag1', 'of_absorption_lag1', 'of_opening_drive_lag1',
                'of_cvd_zscore_lag1', 'of_stacked_imbalance_lag1', 'of_opening_range_lag1']
    for c in lag_cols:
        df[c] = 0.0

    if lagged_of_lookup:
        lon_lk = lagged_of_lookup.get('LON', {})
        ny_lk  = lagged_of_lookup.get('NY',  {})

        # Build DataFrame from lookups and merge vectorized
        def _lk_to_df(lk, session_flag_col):
            if not lk:
                return None
            rows = [{'date': d, **v} for d, v in lk.items()]
            return pd.DataFrame(rows).rename(columns={'date': '_lag_date'})

        lon_lag_df = _lk_to_df(lon_lk, 'is_lon_session')
        ny_lag_df  = _lk_to_df(ny_lk,  'is_ny_session')

        df_idx = df.index
        df = df.reset_index(drop=True)
        df['_lag_date'] = df['date'].astype(str)

        # LON bars: merge with LON lag lookup
        if lon_lag_df is not None:
            lon_mask = df['is_lon_session'] == 1
            df_lon = df[lon_mask][['_lag_date']].merge(lon_lag_df, on='_lag_date', how='left')
            df_lon.index = df.index[lon_mask]
            for c in lag_cols:
                if c in df_lon.columns:
                    df.loc[lon_mask, c] = df_lon[c].values

        # NY bars: merge with NY lag lookup
        if ny_lag_df is not None:
            ny_mask = df['is_ny_session'] == 1
            df_ny = df[ny_mask][['_lag_date']].merge(ny_lag_df, on='_lag_date', how='left')
            df_ny.index = df.index[ny_mask]
            for c in lag_cols:
                if c in df_ny.columns:
                    df.loc[ny_mask, c] = df_ny[c].values

        df = df.drop(columns=['_lag_date'])
        df.index = df_idx if len(df_idx) == len(df) else range(len(df))
        filled_lon = int(df['is_lon_session'].sum())
        filled_ny  = int(df['is_ny_session'].sum())
        print(f"  Lagged OF: LON session bars={filled_lon:,}, NY session bars={filled_ny:,}")
    else:
        print("  Lagged OF features: skipped (no lookup provided)")

    return df


# ── Rule-based labeling ───────────────────────────────────────────────────────
def label_regimes(df):
    """
    Labeling vectorizat pe baze de reguli ICT + forward validation.
    """
    print("  Labeling regimes ...")
    n = len(df)
    labels = np.full(n, -1, dtype=int)

    atr     = df['atr_14'].values
    adx     = df['adx_14'].values
    hurst   = df['hurst'].values
    iv      = df['inside_va'].values
    dv_atr  = df['dist_vwap_atr'].values
    has_d   = df['has_displacement'].values
    # stacked_bull/bear și absorption_score sunt zero în DB → excluse din labeling
    hhmm    = df['hhmm_enc'].values
    is_open = df['is_session_open'].values
    close   = df['close'].values
    sw_dn   = df['sweep_dn_atr'].values
    sw_up   = df['sweep_up_atr'].values
    h4_b    = df['h4_bias_atr'].values

    # Rolling max ADX pe 30 bare (pentru RETRACEMENT)
    adx_roll_max = pd.Series(adx).rolling(30, min_periods=1).max().values

    print("    → CONSOLIDATION ...")
    # 0: CONSOLIDATION — ADX scăzut, inside VA, aproape VWAP
    mask_cons = (adx < ADX_LOW) & (iv == 1) & (dv_atr < VWAP_CLOSE)
    labels[mask_cons] = 0

    print("    → EXPANSION ...")
    # 2: EXPANSION — ADX ridicat, trending (fără stacked_bull/bear — all zeros în DB)
    mask_exp = (adx > ADX_HIGH) & (hurst > HURST_TREND)
    labels[mask_exp] = 2

    print("    → DISTRIBUTION ...")
    # 4: DISTRIBUTION — la extreme de VWAP, momentum slăbind (TS territory)
    # dist_vwap_atr median=4.35, p75=8, p90=13 → EXTREME > 10
    mask_dist = (
        (dv_atr > VWAP_EXTREME) &
        (adx < adx_roll_max * 0.75) &   # ADX scade din maxim
        (hurst < HURST_TREND)            # momentum fading
    )
    labels[mask_dist] = 4

    print("    → RETRACEMENT ...")
    # 3: RETRACEMENT — a fost expansion (ADX > 25), acum trage spre VWAP
    mask_ret = (
        (adx_roll_max > ADX_HIGH) &
        (adx < ADX_HIGH) &
        (dv_atr < VWAP_EXTREME) &
        (labels != 2) &
        (hurst < HURST_TREND)
    )
    labels[mask_ret] = 3

    print("    → PRE_EXPANSION (session-aware forward validation) ...")
    # 1: PRE_EXPANSION — session open + sweep + displacement + mișcare forward
    # Session-aware thresholds:
    #   LON: MIN_MOVE_ATR=0.8, FWD_BARS=20 (mișcări LON mai compacte)
    #   NY:  MIN_MOVE_ATR=1.2, FWD_BARS=30 (standard)
    is_lon = df['is_lon_session'].values if 'is_lon_session' in df.columns else np.zeros(n)
    is_ny  = df['is_ny_session'].values  if 'is_ny_session'  in df.columns else np.zeros(n)

    candidate_mask = (
        (is_open == 1) &
        (has_d == 1) &
        (adx < ADX_HIGH) &
        ((sw_dn > 0.3) | (sw_up > 0.3))  # minim sweep de 0.3×ATR
    )
    candidate_idx = np.where(candidate_mask)[0]

    count_pre_lon = 0; count_pre_ny = 0; count_pre_other = 0
    for i in candidate_idx:
        # Choose session-aware params
        if is_lon[i]:
            fwd_bars   = FWD_BARS_LON
            min_move   = MIN_MOVE_ATR_LON
        elif is_ny[i]:
            fwd_bars   = FWD_BARS_NY
            min_move   = MIN_MOVE_ATR_NY
        else:
            fwd_bars   = FWD_BARS_NY
            min_move   = MIN_MOVE_ATR_NY

        fwd_end = min(i + fwd_bars + 1, n)
        fwd_close = close[i+1:fwd_end]
        if len(fwd_close) < 5:
            continue
        fwd_range = np.max(fwd_close) - np.min(fwd_close)
        if fwd_range >= min_move * atr[i]:
            labels[i] = 1
            if is_lon[i]: count_pre_lon += 1
            elif is_ny[i]: count_pre_ny += 1
            else: count_pre_other += 1

    count_pre = count_pre_lon + count_pre_ny + count_pre_other
    print(f"    → PRE_EXPANSION: {count_pre:,} bare validate "
          f"(LON={count_pre_lon:,} [thresh={MIN_MOVE_ATR_LON}×ATR/{FWD_BARS_LON}bars], "
          f"NY={count_pre_ny:,} [thresh={MIN_MOVE_ATR_NY}×ATR/{FWD_BARS_NY}bars])")

    # Default: orice nelabelat → CONSOLIDATION
    labels[labels == -1] = 0

    # Distribuție
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    print("\n  Distribuție regimuri:")
    for u, c in zip(unique, counts):
        print(f"    {REGIME_NAMES[u]:15s}: {c:7,} bare ({100*c/total:.1f}%)")

    return labels


# ── Train model ───────────────────────────────────────────────────────────────
def train(X_tr, y_tr, X_oos, y_oos, features_used):
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import classification_report, balanced_accuracy_score
    from sklearn.preprocessing import LabelEncoder

    # LabelEncoder — handle clase lipsă din date
    le = LabelEncoder()
    le.fit(np.concatenate([y_tr, y_oos]))
    y_tr_enc  = le.transform(y_tr)
    y_oos_enc = le.transform(y_oos)
    n_classes = len(le.classes_)
    class_names = [REGIME_NAMES[c] for c in le.classes_]
    print(f"  Clase prezente: {class_names} ({n_classes} clase)")

    print(f"\n  Antrenare XGBoost multi-class ({n_classes} clase) ...")
    xgb = XGBClassifier(
        n_estimators      = 400,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.7,
        min_child_weight  = 10,
        reg_alpha         = 0.5,
        reg_lambda        = 2.0,
        objective         = 'multi:softprob',
        num_class         = n_classes,
        eval_metric       = 'mlogloss',
        n_jobs            = -1,
        random_state      = 42,
        verbosity         = 0,
    )

    xgb.fit(X_tr, y_tr_enc,
            eval_set=[(X_oos, y_oos_enc)],
            verbose=False)

    print("  Calibrare probabilități ...")
    cal = CalibratedClassifierCV(xgb, cv='prefit', method='isotonic')
    cal_size = min(50_000, len(X_oos))
    idx_cal = np.random.choice(len(X_oos), cal_size, replace=False)
    cal.fit(X_oos.iloc[idx_cal], y_oos_enc[idx_cal])

    # Metrici
    for name, Xv, yv_enc in [("IS", X_tr, y_tr_enc), ("OOS", X_oos, y_oos_enc)]:
        preds = cal.predict(Xv)
        bac   = balanced_accuracy_score(yv_enc, preds)
        print(f"\n  {name} Balanced Accuracy: {bac:.3f}")
        print(classification_report(yv_enc, preds,
                                    target_names=class_names,
                                    zero_division=0))

    # Feature importance
    fi = pd.Series(xgb.feature_importances_, index=features_used).sort_values(ascending=False)
    print("\n  Top 10 features:")
    print(fi.head(10).to_string())

    return cal, le, fi


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("  REGIME CLASSIFIER v1 — Training")
    print("=" * 65)

    # 1. Load
    print("\n[1] Loading IS data ...")
    df_tr  = load_data(TRAIN_START, TRAIN_END, max_rows=SAMPLE_IS)
    print("[2] Loading OOS data ...")
    df_oos = load_data(OOS_START,   OOS_END,   max_rows=SAMPLE_OOS)

    # 2. Lagged OF lookup
    print("\n[2b] Building lagged OF lookup ...")
    lagged_of = build_lagged_of_lookup()

    # 3. Features
    print("\n[3] Feature engineering ...")
    df_tr  = build_features(df_tr,  lagged_of_lookup=lagged_of)
    df_oos = build_features(df_oos, lagged_of_lookup=lagged_of)

    # 3. Labels
    print("\n[4] Labeling IS ...")
    y_tr  = label_regimes(df_tr)
    print("\n[5] Labeling OOS ...")
    y_oos = label_regimes(df_oos)

    # 4. Pregătire arrays
    missing = [f for f in FEATURES if f not in df_tr.columns]
    if missing:
        print(f"\n⚠️  Features lipsă: {missing}")
        FEATURES_USED = [f for f in FEATURES if f in df_tr.columns]
    else:
        FEATURES_USED = FEATURES

    X_tr  = df_tr[FEATURES_USED].fillna(0).astype(float)
    X_oos = df_oos[FEATURES_USED].fillna(0).astype(float)

    print(f"\n[6] Features: {len(FEATURES_USED)} | IS: {len(X_tr):,} | OOS: {len(X_oos):,}")

    # 5. Train
    model, label_encoder, feature_importance = train(
        X_tr, y_tr, X_oos, y_oos, FEATURES_USED)

    # 6. Save
    pkg = {
        'model':          model,
        'label_encoder':  label_encoder,
        'features':       FEATURES_USED,
        'regimes':        REGIME_NAMES,
        'classes':        label_encoder.classes_.tolist(),
        'trained':        datetime.now().isoformat(),
        'version':        'v1.1_session_aware_lagged_of',
        'lon_min_move_atr': MIN_MOVE_ATR_LON,
        'ny_min_move_atr':  MIN_MOVE_ATR_NY,
        'lon_fwd_bars':   FWD_BARS_LON,
        'ny_fwd_bars':    FWD_BARS_NY,
    }
    joblib.dump(pkg, OUT_MODEL)
    print(f"\n✅  Model salvat → {OUT_MODEL}")

    meta = {
        'features':     FEATURES_USED,
        'regimes':      REGIME_NAMES,
        'classes':      label_encoder.classes_.tolist(),
        'n_is':         int(len(X_tr)),
        'n_oos':        int(len(X_oos)),
        'trained':      datetime.now().isoformat(),
        'top_features': feature_importance.head(15).to_dict(),
    }
    with open(OUT_META, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"✅  Meta salvat → {OUT_META}")
    print("\nDone.")
