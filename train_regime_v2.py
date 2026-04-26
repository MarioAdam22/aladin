"""
train_regime_v2.py — Regime Classifier v2
==========================================
Extinde v1.1 cu 8 OF lagged features suplimentare din orderflow_features.parquet:
  absorption_zscore_20d, opening_drive_strength, opening_delta_ratio,
  stacked_zscore, cvd_momentum, cvd_momentum_z, vwap_zscore, amihud_mean_zscore

Total lagged OF features: 14 (vs 6 în v1.1)

v2 changes:
  ✅ +8 lagged OF features din prev session (absorption_z, drive_strength,
     delta_ratio, stacked_z, cvd_momentum, cvd_momentum_z, vwap_z, amihud_z)
  ✅ Session-aware PRE_EXPANSION thresholds (moștenite din v1.1)
  ✅ Output: regime_classifier_v2.pkl

Output: regime_classifier_v2.pkl, regime_classifier_v2_meta.json
"""

import sqlite3, warnings, joblib, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH    = Path(__file__).parent / "mario_trading.db"
OUT_MODEL  = Path(__file__).parent / "regime_classifier_v2.pkl"
OUT_META   = Path(__file__).parent / "regime_classifier_v2_meta.json"

TRAIN_START = "2022-01-01"
TRAIN_END   = "2024-12-31"
OOS_START   = "2025-01-01"
OOS_END     = "2026-04-22"

SAMPLE_IS   = 300_000
SAMPLE_OOS  =  80_000

LON_OPEN_ET   = (400,  700)
NY_OPEN_ET    = (900, 1130)

ADX_LOW       = 18
ADX_HIGH      = 25
HURST_TREND   = 0.52
VWAP_CLOSE    = 3.0
VWAP_EXTREME  = 10.0
LOOKBACK      = 30

FWD_BARS_LON     = 20
FWD_BARS_NY      = 30
MIN_MOVE_ATR_LON = 0.8
MIN_MOVE_ATR_NY  = 1.2

FWD_BARS     = FWD_BARS_NY
MIN_MOVE_ATR = MIN_MOVE_ATR_NY

OF_PATH = Path(__file__).parent / "data" / "orderflow_features.parquet"

REGIME_NAMES = {
    0: 'CONSOLIDATION',
    1: 'PRE_EXPANSION',
    2: 'EXPANSION',
    3: 'RETRACEMENT',
    4: 'DISTRIBUTION',
}

FEATURES = [
    # Structură / trend
    'adx_14', 'hurst', 'garch_vol', 'kalman_smooth',
    'acf_lag1', 'acf_lag5', 'fisher_transform', 'sample_entropy',
    # Value Area / VWAP
    'inside_va', 'dist_vwap_atr', 'dist_poc_atr', 'dist_pdh_atr', 'dist_pdl_atr',
    # Displacement / setup
    'has_displacement', 'body_size_atr', 'rvol',
    # Orderflow
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
    # Sweep context
    'pre_range_atr', 'sweep_dn_atr', 'sweep_up_atr',
    # ── Lagged OF features v1 (6) ────────────────────────────────────────────
    'of_cvd_lag1', 'of_absorption_lag1', 'of_opening_drive_lag1',
    'of_cvd_zscore_lag1', 'of_stacked_imbalance_lag1', 'of_opening_range_lag1',
    # ── Lagged OF features v2 NEW (8) ────────────────────────────────────────
    'of_absorption_zscore_lag1',       # absorption_zscore_20d — absorpție normalizată pe 20d
    'of_opening_drive_strength_lag1',  # opening_drive_strength — magnitudinea drivului de deschidere
    'of_delta_ratio_lag1',             # opening_delta_ratio — raport delta la deschidere
    'of_stacked_zscore_lag1',          # stacked_zscore — imbalance stacked normalizat
    'of_cvd_momentum_lag1',            # cvd_momentum — momentum CVD raw
    'of_cvd_momentum_z_lag1',          # cvd_momentum_z — momentum CVD z-score
    'of_vwap_zscore_lag1',             # vwap_zscore — distanță VWAP normalizată
    'of_amihud_zscore_lag1',           # amihud_mean_zscore — lichiditate Amihud normalizată
]


# ── Lagged OF features ────────────────────────────────────────────────────────
def build_lagged_of_lookup():
    """
    Returns {'LON': {date: {feature: val}}, 'NY': {date: {feature: val}}}.

    LON on day D uses NY from D-1.
    NY on day D uses LON from same day D.

    v2 additions: absorption_zscore_20d, opening_drive_strength, opening_delta_ratio,
                  stacked_zscore, cvd_momentum, cvd_momentum_z, vwap_zscore, amihud_mean_zscore
    """
    if not OF_PATH.exists():
        print(f"  ⚠️  OF parquet not found at {OF_PATH}, skipping lagged OF features")
        return {}

    of = pd.read_parquet(OF_PATH)
    of['date'] = of['date'].astype(str)

    def safe_col(df, col):
        return df[col].values if col in df.columns else np.zeros(len(df))

    def extract_session(df, sess):
        s = df[df['session_type'] == sess].copy()
        if s.empty:
            return s
        # v1 features
        s['cvd_f']    = safe_col(s, 'cvd_final')
        s['abs_f']    = safe_col(s, 'absorption_score_mean')
        s['od_f']     = safe_col(s, 'opening_drive_dir')
        s['cvdz_f']   = safe_col(s, 'cvd_zscore_20d')
        s['si_f']     = safe_col(s, 'stacked_imbalance_count')
        s['or_f']     = safe_col(s, 'opening_range')
        # v2 new features
        s['absz_f']   = safe_col(s, 'absorption_zscore_20d')
        s['ods_f']    = safe_col(s, 'opening_drive_strength')
        s['dr_f']     = safe_col(s, 'opening_delta_ratio')
        s['sz_f']     = safe_col(s, 'stacked_zscore')
        s['cm_f']     = safe_col(s, 'cvd_momentum')
        s['cmz_f']    = safe_col(s, 'cvd_momentum_z')
        s['vz_f']     = safe_col(s, 'vwap_zscore')
        s['ah_f']     = safe_col(s, 'amihud_mean_zscore')
        return s.sort_values('date').reset_index(drop=True)

    lon = extract_session(of, 'LON')
    ny  = extract_session(of, 'NY')

    def row_to_feats(r):
        return {
            # v1
            'of_cvd_lag1':                float(r['cvd_f']),
            'of_absorption_lag1':          float(r['abs_f']),
            'of_opening_drive_lag1':       float(r['od_f']),
            'of_cvd_zscore_lag1':          float(r['cvdz_f']),
            'of_stacked_imbalance_lag1':   float(r['si_f']),
            'of_opening_range_lag1':       float(r['or_f']),
            # v2
            'of_absorption_zscore_lag1':       float(r['absz_f']),
            'of_opening_drive_strength_lag1':  float(r['ods_f']),
            'of_delta_ratio_lag1':             float(r['dr_f']),
            'of_stacked_zscore_lag1':          float(r['sz_f']),
            'of_cvd_momentum_lag1':            float(r['cm_f']),
            'of_cvd_momentum_z_lag1':          float(r['cmz_f']),
            'of_vwap_zscore_lag1':             float(r['vz_f']),
            'of_amihud_zscore_lag1':           float(r['ah_f']),
        }

    # LON on day D → uses NY from previous trading day
    lon_set = set(lon['date'].tolist())
    ny_set  = set(ny['date'].tolist())
    ny_dict = {r['date']: r for _, r in ny.iterrows()}
    all_dates = sorted(lon_set | ny_set)

    lon_lag_lookup = {}
    prev_ny_row = None
    for d in all_dates:
        if d in lon_set:
            if prev_ny_row is not None:
                lon_lag_lookup[d] = row_to_feats(prev_ny_row)
        if d in ny_set:
            prev_ny_row = ny_dict.get(d)

    # NY on day D → uses LON from same day D
    lon_dict = {r['date']: r for _, r in lon.iterrows()}
    ny_lag_lookup = {}
    for _, r in ny.iterrows():
        d = r['date']
        if d in lon_dict:
            ny_lag_lookup[d] = row_to_feats(lon_dict[d])

    print(f"  Lagged OF lookup v2: {len(lon_lag_lookup)} LON dates, {len(ny_lag_lookup)} NY dates")
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

    total = pd.read_sql(f"SELECT COUNT(*) as n FROM market_data WHERE {base_where}", conn).iloc[0, 0]
    print(f"  → disponibil: {total:,} bare")

    if max_rows and max_rows < total:
        every_n = max(1, int(total / max_rows))

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

    df['dist_vwap_atr']      = df['dist_vwap'].abs() / atr
    df['dist_poc_atr']       = df['dist_poc'].abs() / atr
    df['dist_pdh_atr']       = df['dist_pdh'].abs() / atr
    df['dist_pdl_atr']       = df['dist_pdl'].abs() / atr
    df['body_size_atr']      = df['body_size'] / atr
    df['bar_delta_atr']      = df['bar_delta'].abs() / (df['volume'].clip(lower=1))
    df['delta_at_high_atr']  = df['delta_at_high'].abs() / atr
    df['delta_at_low_atr']   = df['delta_at_low'].abs() / atr
    df['cum_delta_20_atr']   = df['cum_delta'].rolling(20, min_periods=1).sum() / atr

    df['hhmm_enc'] = df['hour_min'].str.replace(':', '').astype(int)

    h = df['hhmm_enc']
    df['is_lon_session'] = ((h >= LON_OPEN_ET[0]) & (h <= LON_OPEN_ET[1])).astype(int)
    df['is_ny_session']  = ((h >= NY_OPEN_ET[0])  & (h <= NY_OPEN_ET[1])).astype(int)
    df['is_session_open'] = ((df['is_lon_session'] == 1) | (df['is_ny_session'] == 1)).astype(int)

    df['sess_hi'] = df[['lon_hi', 'p_hi']].max(axis=1)
    df['sess_lo'] = df[['lon_lo', 'p_lo']].min(axis=1)
    df['dist_sess_hi_atr'] = (df['sess_hi'] - df['close']).abs() / atr
    df['dist_sess_lo_atr'] = (df['close'] - df['sess_lo']).abs() / atr

    df['h4_mid'] = (df['h4_hi'] + df['h4_lo']) / 2
    df['h1_mid'] = (df['h1_hi'] + df['h1_lo']) / 2
    df['h4_bias_atr']          = (df['close'] - df['h4_mid']) / atr
    df['h1_bias_atr']          = (df['close'] - df['h1_mid']) / atr
    df['above_true_open_atr']  = (df['close'] - df['true_open']) / atr

    df['pre_range'] = (df['p_hi'] - df['p_lo']).clip(lower=0.01)
    df['pre_range_atr'] = df['pre_range'] / atr

    asia_hi = df['asia_hi'].values if 'asia_hi' in df.columns else df['p_hi'].values
    asia_lo = df['asia_lo'].values if 'asia_lo' in df.columns else df['p_lo'].values
    ref_hi = pd.DataFrame({'p': df['p_hi'].values, 'a': asia_hi}).max(axis=1).values
    ref_lo = pd.DataFrame({'p': df['p_lo'].values, 'a': asia_lo}).min(axis=1).values
    df['sweep_dn_atr'] = ((ref_lo - df['low']).clip(lower=0) / atr)
    df['sweep_up_atr'] = ((df['high'] - ref_hi).clip(lower=0) / atr)

    # ── Lagged OF features (v1 + v2 new) ─────────────────────────────────────
    lag_cols_v1 = ['of_cvd_lag1', 'of_absorption_lag1', 'of_opening_drive_lag1',
                   'of_cvd_zscore_lag1', 'of_stacked_imbalance_lag1', 'of_opening_range_lag1']
    lag_cols_v2 = ['of_absorption_zscore_lag1', 'of_opening_drive_strength_lag1',
                   'of_delta_ratio_lag1', 'of_stacked_zscore_lag1',
                   'of_cvd_momentum_lag1', 'of_cvd_momentum_z_lag1',
                   'of_vwap_zscore_lag1', 'of_amihud_zscore_lag1']
    lag_cols = lag_cols_v1 + lag_cols_v2

    for c in lag_cols:
        df[c] = 0.0

    if lagged_of_lookup:
        lon_lk = lagged_of_lookup.get('LON', {})
        ny_lk  = lagged_of_lookup.get('NY',  {})

        def _lk_to_df(lk):
            if not lk:
                return None
            rows = [{'date': d, **v} for d, v in lk.items()]
            return pd.DataFrame(rows).rename(columns={'date': '_lag_date'})

        lon_lag_df = _lk_to_df(lon_lk)
        ny_lag_df  = _lk_to_df(ny_lk)

        df_idx = df.index
        df = df.reset_index(drop=True)
        df['_lag_date'] = df['date'].astype(str)

        if lon_lag_df is not None:
            lon_mask = df['is_lon_session'] == 1
            df_lon = df[lon_mask][['_lag_date']].merge(lon_lag_df, on='_lag_date', how='left')
            df_lon.index = df.index[lon_mask]
            for c in lag_cols:
                if c in df_lon.columns:
                    df.loc[lon_mask, c] = df_lon[c].values

        if ny_lag_df is not None:
            ny_mask = df['is_ny_session'] == 1
            df_ny = df[ny_mask][['_lag_date']].merge(ny_lag_df, on='_lag_date', how='left')
            df_ny.index = df.index[ny_mask]
            for c in lag_cols:
                if c in df_ny.columns:
                    df.loc[ny_mask, c] = df_ny[c].values

        df = df.drop(columns=['_lag_date'])
        df.index = df_idx if len(df_idx) == len(df) else range(len(df))
        print(f"  Lagged OF v2: {int(df['is_lon_session'].sum()):,} LON bars, "
              f"{int(df['is_ny_session'].sum()):,} NY bars filled")
    else:
        print("  Lagged OF features: skipped (no lookup provided)")

    return df


# ── Rule-based labeling ───────────────────────────────────────────────────────
def label_regimes(df):
    print("  Labeling regimes ...")
    n = len(df)
    labels = np.full(n, -1, dtype=int)

    atr     = df['atr_14'].values
    adx     = df['adx_14'].values
    hurst   = df['hurst'].values
    iv      = df['inside_va'].values
    dv_atr  = df['dist_vwap_atr'].values
    has_d   = df['has_displacement'].values
    hhmm    = df['hhmm_enc'].values
    is_open = df['is_session_open'].values
    close   = df['close'].values
    sw_dn   = df['sweep_dn_atr'].values
    sw_up   = df['sweep_up_atr'].values
    h4_b    = df['h4_bias_atr'].values

    adx_roll_max = pd.Series(adx).rolling(30, min_periods=1).max().values

    print("    → CONSOLIDATION ...")
    mask_cons = (adx < ADX_LOW) & (iv == 1) & (dv_atr < VWAP_CLOSE)
    labels[mask_cons] = 0

    print("    → EXPANSION ...")
    mask_exp = (adx > ADX_HIGH) & (hurst > HURST_TREND)
    labels[mask_exp] = 2

    print("    → DISTRIBUTION ...")
    mask_dist = (
        (dv_atr > VWAP_EXTREME) &
        (adx < adx_roll_max * 0.75) &
        (hurst < HURST_TREND)
    )
    labels[mask_dist] = 4

    print("    → RETRACEMENT ...")
    mask_ret = (
        (adx_roll_max > ADX_HIGH) &
        (adx < ADX_HIGH) &
        (dv_atr < VWAP_EXTREME) &
        (labels != 2) &
        (hurst < HURST_TREND)
    )
    labels[mask_ret] = 3

    print("    → PRE_EXPANSION (session-aware forward validation) ...")
    is_lon = df['is_lon_session'].values if 'is_lon_session' in df.columns else np.zeros(n)
    is_ny  = df['is_ny_session'].values  if 'is_ny_session'  in df.columns else np.zeros(n)

    candidate_mask = (
        (is_open == 1) &
        (has_d == 1) &
        (adx < ADX_HIGH) &
        ((sw_dn > 0.3) | (sw_up > 0.3))
    )
    candidate_idx = np.where(candidate_mask)[0]

    count_pre_lon = 0; count_pre_ny = 0; count_pre_other = 0
    for i in candidate_idx:
        if is_lon[i]:
            fwd_bars = FWD_BARS_LON
            min_move = MIN_MOVE_ATR_LON
        elif is_ny[i]:
            fwd_bars = FWD_BARS_NY
            min_move = MIN_MOVE_ATR_NY
        else:
            fwd_bars = FWD_BARS_NY
            min_move = MIN_MOVE_ATR_NY

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

    labels[labels == -1] = 0

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

    for name, Xv, yv_enc in [("IS", X_tr, y_tr_enc), ("OOS", X_oos, y_oos_enc)]:
        preds = cal.predict(Xv)
        bac   = balanced_accuracy_score(yv_enc, preds)
        print(f"\n  {name} Balanced Accuracy: {bac:.3f}")
        print(classification_report(yv_enc, preds,
                                    target_names=class_names,
                                    zero_division=0))

    fi = pd.Series(xgb.feature_importances_, index=features_used).sort_values(ascending=False)
    print("\n  Top 15 features:")
    print(fi.head(15).to_string())
    print("\n  Top v2 lagged features:")
    v2_feats = [f for f in features_used if f in [
        'of_absorption_zscore_lag1', 'of_opening_drive_strength_lag1',
        'of_delta_ratio_lag1', 'of_stacked_zscore_lag1',
        'of_cvd_momentum_lag1', 'of_cvd_momentum_z_lag1',
        'of_vwap_zscore_lag1', 'of_amihud_zscore_lag1'
    ]]
    if v2_feats:
        print(fi[v2_feats].sort_values(ascending=False).to_string())

    return cal, le, fi


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("  REGIME CLASSIFIER v2 — Training (14 lagged OF features)")
    print("=" * 65)

    print("\n[1] Loading IS data ...")
    df_tr  = load_data(TRAIN_START, TRAIN_END, max_rows=SAMPLE_IS)
    print("[2] Loading OOS data ...")
    df_oos = load_data(OOS_START,   OOS_END,   max_rows=SAMPLE_OOS)

    print("\n[2b] Building lagged OF lookup (v2: 14 features) ...")
    lagged_of = build_lagged_of_lookup()

    print("\n[3] Feature engineering ...")
    df_tr  = build_features(df_tr,  lagged_of_lookup=lagged_of)
    df_oos = build_features(df_oos, lagged_of_lookup=lagged_of)

    print("\n[4] Labeling IS ...")
    y_tr  = label_regimes(df_tr)
    print("\n[5] Labeling OOS ...")
    y_oos = label_regimes(df_oos)

    missing = [f for f in FEATURES if f not in df_tr.columns]
    if missing:
        print(f"\n⚠️  Features lipsă: {missing}")
        FEATURES_USED = [f for f in FEATURES if f in df_tr.columns]
    else:
        FEATURES_USED = list(FEATURES)

    X_tr  = df_tr[FEATURES_USED].fillna(0).astype(float)
    X_oos = df_oos[FEATURES_USED].fillna(0).astype(float)

    print(f"\n[6] Features: {len(FEATURES_USED)} | IS: {len(X_tr):,} | OOS: {len(X_oos):,}")

    model, label_encoder, feature_importance = train(
        X_tr, y_tr, X_oos, y_oos, FEATURES_USED)

    pkg = {
        'model':          model,
        'label_encoder':  label_encoder,
        'features':       FEATURES_USED,
        'regimes':        REGIME_NAMES,
        'classes':        label_encoder.classes_.tolist(),
        'trained':        datetime.now().isoformat(),
        'version':        'v2_lagged_of_14features',
        'lon_min_move_atr': MIN_MOVE_ATR_LON,
        'ny_min_move_atr':  MIN_MOVE_ATR_NY,
        'lon_fwd_bars':   FWD_BARS_LON,
        'ny_fwd_bars':    FWD_BARS_NY,
        'n_lagged_features': 14,
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
        'version':      'v2_lagged_of_14features',
        'top_features': feature_importance.head(15).to_dict(),
    }
    with open(OUT_META, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"✅  Meta salvat → {OUT_META}")
    print("\nDone.")
