"""
train_lom_v3.py — London Open Manipulation v3
=============================================
vs v2 (OOS=0.6339–0.8034):
  ✅ regime_enc embedded as numeric feature (NOT split criterion)
     → Separate model for LON, regime context as a feature
  ✅ Lagged OF session features: prev session CVD/absorption/opening_drive
  ✅ Quantile regression sub-models: P10/P25/P50/P75/P90 of MFE
     → XGBoost with objective='reg:quantileerror'
  ✅ Survival model: XGBoost regressor for time_to_tp (Weibull-like)
  ✅ Removed use_label_encoder=False (deprecated since XGBoost 1.7)
  ✅ regime_multiscale features if available

PKL output: lom_model_v3.pkl containing:
  main_model:       binary classifier (win/loss)
  quantile_models:  {0.1: model, 0.25: model, ..., 0.9: model}
  survival_model:   time-to-TP regression
  features:         feature names for main model
  quantile_features feature names for quantile models
  is_auc, oos_auc, version, metadata
"""

import sqlite3, pickle, logging, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import BorderlineSMOTE
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("LOM_V3")

# ── Config ────────────────────────────────────────────────────────────────────
DB            = Path(__file__).parent / "mario_trading.db"
OUT           = Path(__file__).parent / "lom_model_v3.pkl"
OPTUNA_TRIALS = 80

MIN_SPIKE_PT       = 5.0
MIN_DISP_PT        = 4.0
TP_PT              = 18.0
TP_MULT            = 1.5
LABEL_WINDOW       = 60
PARTIAL_THRESH_PCT = 0.30

TRAIN_YEARS    = [2019, 2020, 2021, 2022, 2023, 2024]
TEST_YEARS     = [2025, 2026]
YEAR_WEIGHTS   = {2019: 0.50, 2020: 0.60, 2021: 0.70, 2022: 0.80, 2023: 0.90, 2024: 1.00}
TOP_N_FEATURES = 75     # +5 for regime_enc + lagged OF
N_WF_FOLDS     = 0      # LON: 348 samples/4 folds = ~87/fold insuficient

QUANTILE_ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]   # MFE quantile targets

LON_SESS_START_ET = 400
LON_SESS_END_ET   = 700
PRE_LON_END_ET    = 359
ASIA_START_ET     = 0
ASIA_END_ET       = 359

# ── Economic Calendar ─────────────────────────────────────────────────────────
_CAL_PATH = Path(__file__).parent / "data" / "economic_calendar.json"
try:
    _cal = json.loads(_CAL_PATH.read_text())
    FOMC_DATES = set(_cal.get('fomc', []))
    NFP_DATES  = set(_cal.get('nfp',  []))
    CPI_DATES  = set(_cal.get('cpi',  []))
    PPI_DATES  = set(_cal.get('ppi',  []))
    NEWS_DAYS  = FOMC_DATES | NFP_DATES | CPI_DATES | PPI_DATES
    log.info(f"Calendar: NFP={len(NFP_DATES)}, FOMC={len(FOMC_DATES)}, CPI={len(CPI_DATES)}")
except Exception as _e:
    log.warning(f"Calendar: {_e}")
    FOMC_DATES = NFP_DATES = CPI_DATES = PPI_DATES = NEWS_DAYS = set()

def _fomc_prox(date_str):
    try:
        d = pd.Timestamp(date_str).date()
        diffs = [abs((d - pd.Timestamp(x).date()).days) for x in FOMC_DATES]
        return float(min(diffs)) if diffs else 30.0
    except: return 30.0

def sv(v, d=0.0):
    try: x = float(v); return x if np.isfinite(x) else d
    except: return d


# ── Regime encoder helper ─────────────────────────────────────────────────────
def load_regime_classifier():
    """Load regime_classifier_v1.pkl for regime_enc feature."""
    rc_path = Path(__file__).parent / "regime_classifier_v1.pkl"
    if not rc_path.exists():
        log.warning("  regime_classifier_v1.pkl not found — regime_enc will be 0")
        return None
    try:
        import joblib
        pkg = joblib.load(rc_path)
        log.info("  Loaded regime_classifier_v1.pkl")
        return pkg
    except Exception as e:
        log.warning(f"  Failed to load regime_classifier: {e}")
        return None


def predict_regime_enc(regime_pkg, bar_features_dict):
    """Return regime_enc (int 0-4) for a single bar."""
    if regime_pkg is None:
        return 2  # default: EXPANSION
    try:
        model   = regime_pkg['model']
        feats   = regime_pkg['features']
        le      = regime_pkg['label_encoder']
        x = pd.DataFrame([{f: bar_features_dict.get(f, 0.0) for f in feats}]).fillna(0)
        enc = model.predict(x)[0]
        return int(le.inverse_transform([enc])[0])
    except Exception:
        return 2


# ── OF lag features ───────────────────────────────────────────────────────────
def load_of_lag_features():
    """
    Returns a dict: date_str → {of_cvd_lag1, of_abs_lag1, of_od_lag1, ...}
    For LON session on day D: uses NY session OF from day D-1.
    """
    of_path = Path(__file__).parent / "data" / "orderflow_features.parquet"
    if not of_path.exists():
        return {}
    of = pd.read_parquet(of_path)
    of['date'] = of['date'].astype(str)
    ny = of[of['session_type'] == 'NY'].sort_values('date').reset_index(drop=True)

    def scol(df, c):
        return df[c].values if c in df.columns else np.zeros(len(df))

    lag_lookup = {}
    for i in range(1, len(ny)):  # lag-1: today's LON sees yesterday's NY
        today_d = ny.iloc[i]['date']  # this is wrong — we need next LON date
        prev_d  = ny.iloc[i-1]['date']
        # Map: what LON date will use this NY OF?
        # LON on date = prev_d + 1 trading day ≈ today_d (since they're consecutive)
        # Simplification: LON on today_d uses NY OF from prev_d
        lag_lookup[today_d] = {
            'of_cvd_lag1':    float(scol(ny.iloc[i-1:i], 'cvd_final')[0]),
            'of_abs_lag1':    float(scol(ny.iloc[i-1:i], 'absorption_score_mean')[0]),
            'of_od_lag1':     float(scol(ny.iloc[i-1:i], 'opening_drive_dir')[0]),
            'of_cvdz_lag1':   float(scol(ny.iloc[i-1:i], 'cvd_zscore_20d')[0]),
            'of_si_lag1':     float(scol(ny.iloc[i-1:i], 'stacked_imbalance_count')[0]),
            'of_or_lag1':     float(scol(ny.iloc[i-1:i], 'opening_range')[0]),
        }
    log.info(f"  OF lag lookup: {len(lag_lookup)} dates")
    return lag_lookup


# ══════════════════════════════════════════════════════════════════════════════
# MTF ICT (identical to v2)
# ══════════════════════════════════════════════════════════════════════════════
def compute_ict_on_tf(df_tf, lookback=20):
    H = df_tf['high'].values.astype(float); L = df_tf['low'].values.astype(float)
    C = df_tf['close'].values.astype(float); O = df_tf['open'].values.astype(float)
    A = np.maximum(df_tf['atr'].values.astype(float), 1.0)
    n = len(H)
    bull_top = np.zeros(n); bull_bot = np.zeros(n)
    bear_top = np.zeros(n); bear_bot = np.zeros(n)
    for i in range(2, n):
        if H[i-2] < L[i] and (L[i] - H[i-2]) > 0.5:
            bull_top[i] = L[i]; bull_bot[i] = H[i-2]
        if L[i-2] > H[i] and (L[i-2] - H[i]) > 0.5:
            bear_top[i] = L[i-2]; bear_bot[i] = H[i]
    in_bull = np.zeros(n); in_bear = np.zeros(n)
    dist_bull = np.full(n, 9.9); dist_bear = np.full(n, 9.9)
    in_ifvg_b = np.zeros(n); in_ifvg_s = np.zeros(n)
    breaker_b = np.zeros(n); breaker_s = np.zeros(n); rejection = np.zeros(n)
    active_bull = []; active_bear = []; inv_bull = []; inv_bear = []; bull_obs = []; bear_obs = []
    for i in range(n):
        c = C[i]; l = L[i]; h = H[i]; a = A[i]
        new_ab = []
        for top, bot, j in active_bull:
            if i - j > lookback: continue
            if l < bot: inv_bull.append((top, bot, i))
            else: new_ab.append((top, bot, j))
        active_bull = new_ab
        new_ab2 = []
        for top, bot, j in active_bear:
            if i - j > lookback: continue
            if h > top: inv_bear.append((top, bot, i))
            else: new_ab2.append((top, bot, j))
        active_bear = new_ab2
        if bull_top[i] > 0: active_bull.append((bull_top[i], bull_bot[i], i))
        if bear_top[i] > 0: active_bear.append((bear_top[i], bear_bot[i], i))
        if i >= 2:
            pb = C[i-1]-O[i-1]; pr = max(H[i-1]-L[i-1], 0.01)
            if pb > 0.55*pr and pb > 1.0: bull_obs.append((C[i-1], O[i-1], i-1))
            if pb < -0.55*pr and abs(pb) > 1.0: bear_obs.append((O[i-1], C[i-1], i-1))
        for top, bot, j in active_bull:
            if bot <= c <= top: in_bull[i] = 1.0
            d = min(abs(c-top), abs(c-bot)) / a; dist_bull[i] = min(dist_bull[i], d)
        for top, bot, j in active_bear:
            if bot <= c <= top: in_bear[i] = 1.0
            d = min(abs(c-top), abs(c-bot)) / a; dist_bear[i] = min(dist_bear[i], d)
        for top, bot, k in inv_bull[-15:]:
            if i - k <= lookback * 2 and bot <= c <= top: in_ifvg_b[i] = 1.0
        for top, bot, k in inv_bear[-15:]:
            if i - k <= lookback * 2 and bot <= c <= top: in_ifvg_s[i] = 1.0
        for top, bot, j in bull_obs[-20:]:
            if i - j <= lookback and c < min(bot, O[j]) - a*0.05:
                if abs(c-top)/a < 0.8 or abs(c-bot)/a < 0.8: breaker_s[i] = 1.0
        for top, bot, j in bear_obs[-20:]:
            if i - j <= lookback and c > max(top, O[j]) + a*0.05:
                if abs(c-top)/a < 0.8 or abs(c-bot)/a < 0.8: breaker_b[i] = 1.0
        if i >= 2:
            wu = H[i-1]-max(C[i-1],O[i-1]); wd = min(C[i-1],O[i-1])-L[i-1]
            bz = abs(C[i-1]-O[i-1])
            if wu > 2.5*max(bz,0.5) and wu > a*0.3:
                if abs(c-H[i-1])/a < 0.6 or abs(c-max(C[i-1],O[i-1]))/a < 0.6: rejection[i] = 1.0
            if wd > 2.5*max(bz,0.5) and wd > a*0.3:
                if abs(c-min(C[i-1],O[i-1]))/a < 0.6 or abs(c-L[i-1])/a < 0.6: rejection[i] = 1.0
    return pd.DataFrame({
        'in_bull': in_bull, 'in_bear': in_bear,
        'dist_bull': np.clip(dist_bull, 0, 9.9), 'dist_bear': np.clip(dist_bear, 0, 9.9),
        'in_ifvg_b': in_ifvg_b, 'in_ifvg_s': in_ifvg_s,
        'breaker_b': breaker_b, 'breaker_s': breaker_s, 'rejection': rejection,
    }, index=df_tf.index)


def compute_mtf_features(conn, setup_dates):
    min_d = min(setup_dates); max_d = max(setup_dates)
    warmup = (pd.Timestamp(min_d) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    df1m = pd.read_sql(f"""
        SELECT timestamp, open, high, low, close, atr_14 FROM market_data
        WHERE timestamp >= '{warmup} 00:00:00' AND timestamp <= '{max_d} 23:59:59'
        ORDER BY timestamp
    """, conn)
    df1m['ts'] = pd.to_datetime(df1m['timestamp'])
    df1m = df1m.set_index('ts').rename(columns={'atr_14': 'atr'})
    df1m['atr'] = df1m['atr'].ffill().fillna(9.0)
    all_features = pd.DataFrame(index=df1m.index)
    for tf_label, tf_rule, lookback in [('5m','5min',25),('15m','15min',20),('1h','1h',20),('4h','4h',15)]:
        df_tf = df1m.resample(tf_rule, label='left', closed='left').agg(
            open=('open','first'), high=('high','max'), low=('low','min'),
            close=('close','last'), atr=('atr','last')).dropna(subset=['open'])
        df_tf['atr'] = df_tf['atr'].ffill().fillna(9.0)
        ict = compute_ict_on_tf(df_tf, lookback=lookback)
        ict_ff = ict.reindex(df1m.index, method='ffill')
        for col in ict.columns:
            all_features[f'{col}_{tf_label}'] = ict_ff[col]
    all_features = all_features.fillna(0.0)
    all_features['ts_str'] = all_features.index.strftime('%Y-%m-%d %H:%M:%S')
    return all_features


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (identical to v2)
# ══════════════════════════════════════════════════════════════════════════════
def load_day(conn, date_str):
    df = pd.read_sql(f"""
        SELECT timestamp, open, high, low, close, volume, atr_14, bar_delta, cum_delta,
               fvg_up, fvg_down, has_displacement, body_size, adx_14, hurst, dist_poc,
               inside_va, dist_vwap, delta_at_high, delta_at_low, big_buy_count, big_sell_count,
               absorption_score, stacked_bull, stacked_bear, of_doi, of_big_balance,
               bar_buy_vol, bar_sell_vol, garch_vol, kalman_smooth, fisher_transform,
               acf_lag1, acf_lag5, rvol, vah, val, poc_level, p_hi, p_lo, lw_hi, lw_lo,
               h4_hi, h4_lo, h1_hi, h1_lo, true_open, asia_hi, asia_lo,
               is_smt_bullish, is_smt_bearish, day_of_week, month
        FROM market_data WHERE date = '{date_str}' ORDER BY timestamp
    """, conn)
    if len(df) < 15: return None
    df['ts']   = pd.to_datetime(df['timestamp'])
    df['hhmm'] = df['ts'].dt.hour * 100 + df['ts'].dt.minute
    return df


def _days_since_win(win_list, current_date_str):
    cd = pd.Timestamp(current_date_str)
    wins = [pd.Timestamp(d) for d, lbl in reversed(win_list) if lbl == 1]
    return float((cd - wins[0]).days) if wins else 30.0


# ══════════════════════════════════════════════════════════════════════════════
# Daily rolling context
# ══════════════════════════════════════════════════════════════════════════════
def build_daily_context(conn, dates):
    min_d = min(dates); max_d = max(dates)
    warmup = (pd.Timestamp(min_d) - pd.Timedelta(days=40)).strftime('%Y-%m-%d')
    dr = pd.read_sql(f"""
        SELECT date(timestamp) as date, (MAX(high)-MIN(low)) as daily_range,
               AVG(atr_14) as avg_atr, AVG(adx_14) as avg_adx, AVG(hurst) as avg_hurst
        FROM market_data
        WHERE date(timestamp) >= '{warmup}' AND date(timestamp) <= '{max_d}'
        GROUP BY date(timestamp) ORDER BY date
    """, conn)
    dr['date'] = dr['date'].astype(str)
    dr['avg_atr'] = dr['avg_atr'].ffill().fillna(9.0)
    dr['daily_range'] = dr['daily_range'].fillna(dr['avg_atr'] * 2)
    dr['range_atr_ratio'] = dr['daily_range'] / dr['avg_atr'].clip(lower=1)
    dr['vix_proxy_5d']  = dr['range_atr_ratio'].rolling(5,  min_periods=2).mean().shift(1)
    dr['vix_proxy_20d'] = dr['range_atr_ratio'].rolling(20, min_periods=5).mean().shift(1)
    dr['vol_regime']    = (dr['vix_proxy_5d'] / dr['vix_proxy_20d'].clip(lower=0.5)).clip(upper=3)
    dr['adx_10d_mean']  = dr['avg_adx'].rolling(10, min_periods=3).mean().shift(1)
    dr['hurst_20d_mean']= dr['avg_hurst'].rolling(20, min_periods=5).mean().shift(1)
    dr['atr_5d']        = dr['avg_atr'].rolling(5, min_periods=2).mean().shift(1)
    dr['atr_10d']       = dr['avg_atr'].rolling(10, min_periods=3).mean().shift(1)
    dr['atr_trend']     = (dr['atr_5d'] / dr['atr_10d'].clip(lower=1)).clip(upper=3)
    dr = dr.ffill().fillna(1.0)
    return {r['date']: r.to_dict() for _, r in dr.iterrows()}


# ══════════════════════════════════════════════════════════════════════════════
# Setup extraction — v3: adds regime_enc + lagged OF + MFE labels
# ══════════════════════════════════════════════════════════════════════════════
def extract_setups(df, date_str, daily_ctx, cross_ctx=None,
                   regime_pkg=None, of_lag_lookup=None):
    setups = []
    pre_lon  = df[df['hhmm'] <= PRE_LON_END_ET]
    lon_sess = df[df['hhmm'].between(LON_SESS_START_ET, LON_SESS_END_ET)]
    if len(pre_lon) < 5 or len(lon_sess) < 3: return setups

    pre_hi = float(pre_lon['high'].max()); pre_lo = float(pre_lon['low'].min())
    pre_rng = pre_hi - pre_lo
    if pre_rng < 3: return setups

    atr = float(df['atr_14'].replace(0, np.nan).dropna().iloc[-1]) if len(df) > 0 else 10.0
    if atr <= 0: atr = 10.0
    tp_pt = TP_PT  # fixed 18 pts (keeps label consistency; drift handled via regime_enc)

    asia_df    = df[df['hhmm'].between(ASIA_START_ET, ASIA_END_ET)]
    asia_open  = float(asia_df['open'].iloc[0])  if len(asia_df) > 0 else pre_hi
    asia_close = float(asia_df['close'].iloc[-1]) if len(asia_df) > 0 else pre_lo
    lon_open   = float(lon_sess['open'].iloc[0])  if len(lon_sess) > 0 else asia_close
    asia_hi_s  = float(asia_df['high'].max())     if len(asia_df) > 0 else pre_hi
    asia_lo_s  = float(asia_df['low'].min())      if len(asia_df) > 0 else pre_lo
    asia_rng   = asia_hi_s - asia_lo_s
    asia_mid   = (asia_hi_s + asia_lo_s) / 2 if asia_hi_s > asia_lo_s else pre_hi
    asia_dir   = 1 if asia_close > asia_mid else -1
    asia_tight = 1 if asia_rng < atr * 0.6 else 0
    asia_wide  = 1 if asia_rng > atr * 1.4 else 0
    eq_tol = atr * 0.3
    pre_highs = pre_lon['high'].values; pre_lows = pre_lon['low'].values
    eq_hi = max(0, sum(1 for h in pre_highs if abs(h - pre_hi) <= eq_tol) - 1)
    eq_lo = max(0, sum(1 for l in pre_lows  if abs(l - pre_lo) <= eq_tol) - 1)

    partial_thresh = pre_rng * 0.50
    lon_reset = lon_sess.reset_index(drop=False)
    last_setup_hhmm = {'LONG': -999, 'SHORT': -999}

    # Lagged OF features for this date
    of_lag = (of_lag_lookup or {}).get(date_str, {})

    # Regime features for this bar (use representative bar = LON open bar)
    dctx = daily_ctx.get(date_str, {})
    regime_bar_features = {
        'adx_14':          float(lon_sess['adx_14'].iloc[0]) if len(lon_sess) > 0 else 20.0,
        'hurst':           float(lon_sess['hurst'].iloc[0])  if len(lon_sess) > 0 else 0.5,
        'garch_vol':       float(lon_sess['garch_vol'].iloc[0]) if len(lon_sess) > 0 else 1.0,
        'inside_va':       float(lon_sess['inside_va'].iloc[0]) if len(lon_sess) > 0 else 0.0,
        'dist_vwap':       float(lon_sess['dist_vwap'].iloc[0]) if len(lon_sess) > 0 else 5.0,
        'has_displacement':0,  # at session open, no displacement yet
        'rvol':            float(lon_sess['rvol'].iloc[0]) if len(lon_sess) > 0 else 1.0,
        'bar_delta':       float(lon_sess['bar_delta'].iloc[0]) if len(lon_sess) > 0 else 0.0,
        'cum_delta':       float(lon_sess['cum_delta'].iloc[0]) if len(lon_sess) > 0 else 0.0,
        'imbalance_pct':   0.0,
        'dom_ratio':       1.0,
        'hhmm_enc':        LON_SESS_START_ET,
        'is_session_open': 1,
        'is_lon_session':  1,
        'is_ny_session':   0,
        'sweep_dn_atr':    0.0,
        'sweep_up_atr':    0.0,
        'dist_poc':        float(lon_sess['dist_poc'].iloc[0]) if len(lon_sess) > 0 else 5.0,
        'dist_pdh':        0.0,
        'dist_pdl':        0.0,
        'body_size':       0.0,
        'fvg_up':          0,
        'fvg_down':        0,
        'acf_lag1':        float(lon_sess['acf_lag1'].iloc[0]) if len(lon_sess) > 0 else 0.0,
        'acf_lag5':        float(lon_sess['acf_lag5'].iloc[0]) if len(lon_sess) > 0 else 0.0,
        'fisher_transform':float(lon_sess['fisher_transform'].iloc[0]) if len(lon_sess) > 0 else 0.0,
        'sample_entropy':  0.5,
        'day_of_week':     float(df['day_of_week'].iloc[0]) if len(df) > 0 else 0.0,
        'month':           float(df['month'].iloc[0]) if len(df) > 0 else 1.0,
        'of_cvd_lag1':     of_lag.get('of_cvd_lag1', 0.0),
        'of_absorption_lag1': of_lag.get('of_abs_lag1', 0.0),
        'of_opening_drive_lag1': of_lag.get('of_od_lag1', 0.0),
        'of_cvd_zscore_lag1': of_lag.get('of_cvdz_lag1', 0.0),
        'of_stacked_imbalance_lag1': of_lag.get('of_si_lag1', 0.0),
        'of_opening_range_lag1': of_lag.get('of_or_lag1', 0.0),
    }
    regime_enc = predict_regime_enc(regime_pkg, regime_bar_features)

    for i in range(1, len(lon_reset) - 2):
        bar = lon_reset.iloc[i]
        bar_hi = sv(bar['high']); bar_lo = sv(bar['low']); bar_hhmm = int(bar['hhmm'])
        spike_up = bar_hi - pre_hi; spike_dn = pre_lo - bar_lo

        for direction, spike_mag_raw, is_valid in [
            ('SHORT', max(spike_up, 0),
             spike_up >= MIN_SPIKE_PT or (spike_up > 0 and spike_up >= partial_thresh)),
            ('LONG',  max(spike_dn, 0),
             spike_dn >= MIN_SPIKE_PT or (spike_dn > 0 and spike_dn >= partial_thresh)),
        ]:
            if not is_valid or (bar_hhmm - last_setup_hhmm[direction]) < 30:
                continue

            spike_mag = spike_mag_raw
            spike_hi_val = bar_hi; spike_lo_val = bar_lo

            after_spike = lon_reset[lon_reset['hhmm'].between(bar_hhmm + 1, bar_hhmm + 45)]
            disp_bar = None
            for _, ab in after_spike.iterrows():
                ab_body = abs(sv(ab['close']) - sv(ab['open']))
                if direction == 'SHORT' and sv(ab['close']) < sv(ab['open']) and ab_body >= MIN_DISP_PT:
                    disp_bar = ab; break
                elif direction == 'LONG' and sv(ab['close']) > sv(ab['open']) and ab_body >= MIN_DISP_PT:
                    disp_bar = ab; break

            if disp_bar is None: continue

            entry_price = sv(disp_bar['close'])
            entry_hhmm  = int(disp_bar['hhmm'])
            entry_ts    = str(disp_bar['timestamp'])
            dir_num     = 1 if direction == 'LONG' else -1

            future = df[df['hhmm'] > entry_hhmm].head(LABEL_WINDOW)
            if len(future) < 3: continue

            if direction == 'LONG':
                reached_tp = float(future['high'].max()) >= entry_price + tp_pt
                max_fwd    = float(future['high'].max() - entry_price)
                mae_raw    = float(entry_price - future['low'].min())
            else:
                reached_tp = float(future['low'].min()) <= entry_price - tp_pt
                max_fwd    = float(entry_price - future['low'].min())
                mae_raw    = float(future['high'].max() - entry_price)

            label = 1 if reached_tp else 0
            mae_raw = max(mae_raw, 0.0)
            max_fwd = max(max_fwd, 0.0)

            # Survival: time to TP or SL
            tp_lvl = entry_price + tp_pt if direction == 'LONG' else entry_price - tp_pt
            sl_pts = tp_pt * 0.6
            sl_lvl = entry_price - sl_pts if direction == 'LONG' else entry_price + sl_pts
            hit_tp = 0; hit_sl = 0; time_to_event = LABEL_WINDOW; is_censored = 1
            for _, fb in future.iterrows():
                fh = sv(fb['high']); fl = sv(fb['low'])
                if direction == 'LONG':
                    tp_h = fh >= tp_lvl; sl_h = fl <= sl_lvl
                else:
                    tp_h = fl <= tp_lvl; sl_h = fh >= sl_lvl
                if tp_h or sl_h:
                    time_to_event = min(int(fb['hhmm']) - entry_hhmm, LABEL_WINDOW)
                    is_censored = 0
                    hit_sl = 1 if sl_h else 0
                    hit_tp = 1 if tp_h and not sl_h else 0
                    break

            r0 = df.iloc[-1]
            spike_bar_range = max(sv(bar['high'] - bar['low']), 0.01)
            after_early = lon_reset[lon_reset['hhmm'].between(bar_hhmm + 1, bar_hhmm + 45)]

            if direction == 'SHORT':
                ts_close_inside  = 1 if sv(bar['close']) <= pre_hi else 0
                wick             = (sv(bar['high']) - max(sv(bar['close']), sv(bar['open']))) / atr
                ts_rejection_str = (spike_hi_val - sv(bar['close'])) / spike_mag if spike_mag > 0 else 0
                ts_wick_pct      = (spike_hi_val - sv(bar['close'])) / spike_bar_range
                ts_body_pct      = abs(sv(bar['open']) - sv(bar['close'])) / spike_bar_range
                ts_close_quality = max(0, (pre_hi - sv(bar['close'])) / pre_rng) if pre_rng > 0 else 0
            else:
                ts_close_inside  = 1 if sv(bar['close']) >= pre_lo else 0
                wick             = (min(sv(bar['close']), sv(bar['open'])) - sv(bar['low'])) / atr
                ts_rejection_str = (sv(bar['close']) - spike_lo_val) / spike_mag if spike_mag > 0 else 0
                ts_wick_pct      = (sv(bar['close']) - spike_lo_val) / spike_bar_range
                ts_body_pct      = abs(sv(bar['open']) - sv(bar['close'])) / spike_bar_range
                ts_close_quality = max(0, (sv(bar['close']) - pre_lo) / pre_rng) if pre_rng > 0 else 0

            wick_pct = wick * atr / spike_bar_range
            sweep_wick_clean = 1 if wick_pct > 0.5 else 0
            sweep_depth_atr  = spike_mag / atr
            deep_sweep = 1 if sweep_depth_atr > 1.5 else 0
            sweep_quality = ts_close_inside*0.4 + sweep_wick_clean*0.3 + deep_sweep*0.2 + 0.1

            sb_rng = max(sv(bar['high'] - bar['low']), 0.01)
            sweep_close_pct = (sv(bar['close']) - sv(bar['low'])) / sb_rng
            sweep_close_rejection = (1 - sweep_close_pct) if direction == 'SHORT' else sweep_close_pct

            disp_body = abs(sv(disp_bar['close']) - sv(disp_bar['open']))
            h4_hi = sv(r0['h4_hi']); h4_lo = sv(r0['h4_lo'])
            h4_mid = (h4_hi + h4_lo) / 2 if h4_hi > 0 and h4_lo > 0 else 0
            h1_hi = sv(r0['h1_hi']); h1_lo = sv(r0['h1_lo'])
            h1_mid = (h1_hi + h1_lo) / 2 if h1_hi > 0 and h1_lo > 0 else 0
            h4_bias = 1 if entry_price < h4_mid else (-1 if h4_mid > 0 else 0)
            h1_bias = 1 if entry_price < h1_mid else (-1 if h1_mid > 0 else 0)
            lw_hi = sv(r0['lw_hi']); lw_lo = sv(r0['lw_lo']); lw_rng = lw_hi - lw_lo
            weekly_prem = (entry_price - lw_lo) / lw_rng if lw_rng > 0 else 0.5
            sweep_lvl = spike_hi_val if direction == 'SHORT' else spike_lo_val
            level_at_weekly = 1 if lw_hi > 0 and (abs(sweep_lvl - lw_hi) < atr*0.8 or abs(sweep_lvl - lw_lo) < atr*0.8) else 0
            asia_hi_v = sv(r0.get('asia_hi', 0)); asia_lo_v = sv(r0.get('asia_lo', 0))
            lon15 = lon_sess[lon_sess['hhmm'].between(LON_SESS_START_ET, LON_SESS_START_ET + 15)]
            lon15_rng = float(lon15['high'].max() - lon15['low'].min()) if len(lon15) > 0 else 0
            lon15_close = float(lon15['close'].iloc[-1]) if len(lon15) > 0 else entry_price
            lon15_mid   = (float(lon15['high'].max()) + float(lon15['low'].min())) / 2 if len(lon15) > 0 else entry_price
            lon15_bias  = 1 if lon15_close > lon15_mid else -1

            asia_half = len(asia_df) // 2 if len(asia_df) >= 4 else 0
            if asia_half > 1:
                a1 = float(asia_df['close'].iloc[:asia_half].mean())
                a2 = float(asia_df['close'].iloc[asia_half:].mean())
                asia_trending = 1 if abs(a2 - a1) > atr * 0.2 else 0
                asia_trend_dir = 1 if a2 > a1 else -1
            else:
                asia_trending = 0; asia_trend_dir = 0
            asia_close_pct = (asia_close - asia_lo_s) / max(asia_rng, 0.01)
            db_hi = sv(disp_bar['high']); db_lo = sv(disp_bar['low']); db_rng = max(db_hi-db_lo, 0.01)
            db_close = sv(disp_bar['close'])
            disp_close_pct = (db_close - db_lo) / db_rng
            disp_close_conviction = disp_close_pct if direction == 'LONG' else (1 - disp_close_pct)
            disp_bars_after_spike = max(0, int(disp_bar['hhmm']) - bar_hhmm)
            disp_fast = 1 if disp_bars_after_spike <= 15 else 0
            lon_avg_vol = float(lon_sess['volume'].mean()) if len(lon_sess) > 1 else 1.0
            disp_vol = sv(disp_bar.get('volume', lon_avg_vol)); disp_vol_ratio = disp_vol / max(lon_avg_vol, 1)
            pre_vol = float(pre_lon['volume'].sum()) if len(pre_lon) > 0 else 1.0
            lon_vol = float(lon_sess['volume'].sum()) if len(lon_sess) > 0 else 1.0
            vol_ratio = lon_vol / pre_vol if pre_vol > 0 else 1.0
            spike_delta = sv(lon_sess['bar_delta'].sum()) if len(lon_sess) > 0 else 0
            fvg_up_v   = int(lon_sess['fvg_up'].any())
            fvg_down_v = int(lon_sess['fvg_down'].any())
            adx_v   = sv(r0['adx_14']); hurst_v = sv(r0['hurst'], 0.5)
            atr5d   = dctx.get('atr_5d', atr)
            vix5    = dctx.get('vix_proxy_5d', 2.0); vix20 = dctx.get('vix_proxy_20d', 2.0)
            vol_rg  = dctx.get('vol_regime', 1.0); atr_tr = dctx.get('atr_trend', 1.0)
            adx10   = dctx.get('adx_10d_mean', 20.0); hst20 = dctx.get('hurst_20d_mean', 0.5)
            roll_wr = dctx.get('rolling_wr', 0.5)
            cctx    = cross_ctx or {}
            dsw_dir = cctx.get('dsw_L' if direction == 'LONG' else 'dsw_S', 30.0)
            dsw_any = min(cctx.get('dsw_L', 30.0), cctx.get('dsw_S', 30.0))
            wk_cnt  = float(cctx.get('week_cnt', 0))
            td_cnt  = float(cctx.get('td_L' if direction == 'LONG' else 'td_S', 0))

            feat = {
                # ── Spike ─────────────────────────────────────────────────────
                'spike_mag': spike_mag, 'spike_mag_atr': spike_mag/atr,
                'spike_vs_range': spike_mag/pre_rng if pre_rng > 0 else 0,
                'pre_rng_atr': pre_rng/atr,
                # ── TS anti-fakeout ───────────────────────────────────────────
                'ts_close_inside': ts_close_inside, 'ts_rejection_str': ts_rejection_str,
                'ts_wick_pct': ts_wick_pct, 'ts_body_pct': ts_body_pct,
                'ts_close_quality': ts_close_quality,
                'ts_wick_dom': 1 if ts_wick_pct > 0.6 else 0,
                'ts_htf_anti': 1 if h4_bias == dir_num else 0,
                'ts_combo_score': ts_close_inside * ts_rejection_str,
                'ts_sweep_depth_pts': spike_mag, 'ts_sweep_depth_atr': sweep_depth_atr,
                # ── Sweep quality ─────────────────────────────────────────────
                'sweep_wick_atr': wick, 'sweep_wick_pct': wick_pct,
                'sweep_wick_clean': sweep_wick_clean, 'sweep_depth_atr': sweep_depth_atr,
                'deep_sweep': deep_sweep, 'shallow_sweep': 1 if sweep_depth_atr < 0.5 else 0,
                'sweep_with_disp': 1, 'sweep_quality_score': sweep_quality,
                'equal_level_score': (eq_hi if direction=='SHORT' else eq_lo)/max(len(pre_lon),1),
                'equal_hi_count': float(eq_hi), 'equal_lo_count': float(eq_lo),
                'sweep_aligned_eq': (eq_hi if direction=='SHORT' else eq_lo)/max(1, eq_hi+eq_lo+1),
                # ── Displacement ──────────────────────────────────────────────
                'disp_body': disp_body, 'disp_body_atr': disp_body/atr,
                'disp_range': sv(disp_bar['high']-disp_bar['low']),
                'has_disp': 1, 'body_pct': disp_body/max(sv(disp_bar['high']-disp_bar['low']), 0.01),
                'body_bear': 1 if direction=='SHORT' else 0,
                'disp_close_pct': disp_close_pct, 'disp_close_conviction': disp_close_conviction,
                'disp_fast': float(disp_fast), 'disp_bars_mins': float(min(disp_bars_after_spike,60)),
                'disp_vol_spike': float(np.clip(disp_vol_ratio, 0, 5)),
                'disp_range_atr': sv(disp_bar['high']-disp_bar['low'])/atr,
                # ── HTF bias ──────────────────────────────────────────────────
                'h4_bias': h4_bias, 'h1_bias': h1_bias,
                'h4_h1_aligned': 1 if h4_bias==h1_bias and h4_bias!=0 else 0,
                'h4_bias_aligned': 1 if h4_bias==dir_num else 0,
                # ── Weekly context ────────────────────────────────────────────
                'weekly_premium_pct': weekly_prem,
                'in_weekly_premium': 1 if weekly_prem > 0.5 else 0,
                'in_weekly_discount': 1 if weekly_prem < 0.5 else 0,
                'weekly_prem_aligned': 1 if (direction=='SHORT' and weekly_prem>0.5) or (direction=='LONG' and weekly_prem<0.5) else 0,
                'lw_range_atr': lw_rng/atr if atr > 0 else 0,
                'level_at_weekly': float(level_at_weekly),
                # ── Asia ─────────────────────────────────────────────────────
                'dist_asia_hi_atr': abs(entry_price-asia_hi_v)/atr if asia_hi_v > 0 else 0,
                'dist_asia_lo_atr': abs(entry_price-asia_lo_v)/atr if asia_lo_v > 0 else 0,
                'asia_range_atr': asia_rng/atr if asia_rng > 0 else 0,
                'asia_dir_explicit': float(asia_dir),
                'asia_dir_aligned': 1 if asia_dir==dir_num else 0,
                'asia_close_vs_mid': (asia_close-asia_mid)/atr if atr > 0 else 0,
                'asia_tight': float(asia_tight), 'asia_wide': float(asia_wide),
                'asia_range_vs_atr5d': float(np.clip(asia_rng/max(atr5d,1.0), 0, 10)),
                'asia_trending': float(asia_trending), 'asia_trend_dir': float(asia_trend_dir),
                'asia_close_pct': asia_close_pct,
                'asia_close_aligned': 1 if (asia_close_pct<0.3 and direction=='SHORT') or (asia_close_pct>0.7 and direction=='LONG') else 0,
                'asia_trend_aligned': 1 if asia_trend_dir==dir_num else 0,
                # ── London 15min ──────────────────────────────────────────────
                'lon15_range_atr': lon15_rng/atr, 'in_first_15': 1 if bar_hhmm<=LON_SESS_START_ET+15 else 0,
                'lon15_bias': float(lon15_bias), 'lon15_aligned': 1 if lon15_bias==dir_num else 0,
                'sweep_time_early': 1 if bar_hhmm<=LON_SESS_START_ET+30 else 0,
                'sweep_time_mid': 1 if LON_SESS_START_ET+30<bar_hhmm<=LON_SESS_START_ET+90 else 0,
                'sweep_time_late': 1 if bar_hhmm>LON_SESS_START_ET+90 else 0,
                # ── PDH/PDL/True open ─────────────────────────────────────────
                'above_true_open': 1 if entry_price>sv(r0['true_open']) else 0,
                'dist_pdh_atr': abs(entry_price-sv(r0['p_hi']))/atr,
                'dist_pdl_atr': abs(entry_price-sv(r0['p_lo']))/atr,
                # ── VA / POC ──────────────────────────────────────────────────
                'inside_va': sv(r0['inside_va']), 'dist_poc_atr': sv(r0['dist_poc'])/atr,
                'dist_vwap_atr': sv(r0['dist_vwap'])/atr,
                # ── Volume / delta ────────────────────────────────────────────
                'vol_ratio': vol_ratio, 'spike_delta': spike_delta,
                'disp_delta': sv(after_early['bar_delta'].sum()) if len(after_early) > 0 else 0,
                'delta_at_high': sv(lon_sess['delta_at_high'].sum()),
                'delta_at_low':  sv(lon_sess['delta_at_low'].sum()),
                'absorption': sv(lon_sess['absorption_score'].mean()),
                'buy_sell_ratio': sv(lon_sess['bar_buy_vol'].sum())/max(sv(lon_sess['bar_sell_vol'].sum()),1),
                # ── FVG ───────────────────────────────────────────────────────
                'fvg_up': fvg_up_v, 'fvg_down': fvg_down_v,
                # ── Technical ────────────────────────────────────────────────
                'adx': adx_v, 'adx_strong': 1 if adx_v>25 else 0,
                'hurst': hurst_v, 'fisher_transform': sv(r0['fisher_transform']),
                'acf_lag1': sv(r0['acf_lag1']), 'acf_lag5': sv(r0['acf_lag5']),
                'rvol': sv(r0['rvol'], 1.0),
                # ── Rolling regime ────────────────────────────────────────────
                'vix_proxy_5d': float(vix5), 'vix_proxy_20d': float(vix20),
                'vol_regime': float(vol_rg), 'vol_high': 1 if vol_rg>1.2 else 0,
                'atr_trend': float(atr_tr), 'atr_5d': float(atr5d),
                'adx_10d_mean': float(adx10), 'hurst_20d_mean': float(hst20),
                'rolling_5sess_wr': float(roll_wr),
                # ── Calendar ─────────────────────────────────────────────────
                'is_nfp_day': 1 if date_str in NFP_DATES else 0,
                'is_fomc_day': 1 if date_str in FOMC_DATES else 0,
                'is_news_day': 1 if date_str in NEWS_DAYS else 0,
                'fomc_proximity': float(np.clip(_fomc_prox(date_str)/14.0, 0, 1)),
                # ── Time ─────────────────────────────────────────────────────
                'day_of_week': sv(r0['day_of_week']),
                'is_monday': 1 if int(sv(r0['day_of_week']))==0 else 0,
                'month': sv(r0['month']),
                # ── Cross-setup context ───────────────────────────────────────
                'days_since_win_dir': float(np.clip(dsw_dir, 0, 30)),
                'days_since_win_any': float(np.clip(dsw_any, 0, 30)),
                'week_setup_count': float(np.clip(wk_cnt, 0, 10)),
                'hot_streak': 1 if dsw_dir<=2 else 0,
                'cold_streak': 1 if dsw_dir>=7 else 0,
                # ── NUEVO: Regime enc (embedded, not split criterion) ─────────
                'regime_enc':     float(regime_enc),         # 0=CONSOL, 1=PRE_EXP, 2=EXP, 3=RET, 4=DIST
                'regime_is_pre':  1 if regime_enc == 1 else 0,
                'regime_is_exp':  1 if regime_enc == 2 else 0,
                'regime_is_ret':  1 if regime_enc == 3 else 0,
                'regime_aligned': 1 if (regime_enc in [1,2] and h4_bias==dir_num) else 0,
                # ── NUEVO: Lagged OF features ─────────────────────────────────
                'of_cvd_lag1':      of_lag.get('of_cvd_lag1', 0.0),
                'of_abs_lag1':      of_lag.get('of_abs_lag1', 0.0),
                'of_od_lag1':       of_lag.get('of_od_lag1', 0.0),
                'of_cvdz_lag1':     of_lag.get('of_cvdz_lag1', 0.0),
                'of_si_lag1':       of_lag.get('of_si_lag1', 0.0),
                'of_or_lag1':       of_lag.get('of_or_lag1', 0.0),
                'of_cvd_regime':    of_lag.get('of_cvd_lag1', 0.0) * float(regime_enc),
                # ── Cross-model features (LOM v1 + NOM v1 + DSM + sweep) ─────
                # From LOM v1 (high importance, were missing from v3 feat dict)
                'garch_vol':        sv(r0.get('garch_vol', 1.0)),
                'atr_vs_5d':        float(np.clip(atr / max(atr5d, 1.0), 0, 3)),
                'dir_x_adx':        float(dir_num) * adx_v,
                'dir_x_hurst':      float(dir_num) * hurst_v,
                'vol_x_sweep':      float(vol_rg) * sweep_quality,
                'asia_dir_x_h4':    float(asia_dir) * float(h4_bias),
                'disp_wick_ratio':  (sv(disp_bar['high']-disp_bar['low']) - disp_body) / max(disp_body, 0.01),
                'spike_vs_asia_hi': (spike_hi_val - asia_hi_s) / atr if asia_hi_s > 0 else 0.0,
                'spike_vs_asia_lo': (asia_lo_s - spike_lo_val) / atr if asia_lo_s > 0 else 0.0,
                'dist_lw_hi':       abs(entry_price - lw_hi) / atr if lw_hi > 0 else 0.0,
                'dist_lw_lo':       abs(entry_price - lw_lo) / atr if lw_lo > 0 else 0.0,
                'h4_x_weekly':      float(1 if h4_bias==dir_num else 0) * float(1 if (direction=='SHORT' and weekly_prem>0.5) or (direction=='LONG' and weekly_prem<0.5) else 0),
                'is_thursday':      1 if int(sv(r0['day_of_week'])) == 3 else 0,
                'is_friday':        1 if int(sv(r0['day_of_week'])) == 4 else 0,
                # From NOM v3 (high importance in NOM, potentially useful in LON too)
                'pre_rng_vs_lw':    pre_rng / max(lw_rng, 0.01),
                'sweep_vs_lw_rng':  spike_mag / max(lw_rng, 0.01),
                # Direction ─────────────────────────────────────────────────
                'direction_enc':  1 if direction=='SHORT' else 0,
                # ── Sweep bar structure ───────────────────────────────────────
                'sweep_close_pct': sweep_close_pct, 'sweep_close_rejection': sweep_close_rejection,
                # ── SMT ──────────────────────────────────────────────────────
                'is_smt_bullish': sv(r0.get('is_smt_bullish',0)),
                'is_smt_bearish': sv(r0.get('is_smt_bearish',0)),
                # ── Meta ─────────────────────────────────────────────────────
                '_label':      label,
                '_direction':  direction,
                '_date':       str(date_str),
                '_entry_px':   entry_price,
                '_max_fwd':    max_fwd,
                '_mae_raw':    mae_raw,
                '_entry_hhmm': entry_hhmm,
                '_entry_ts':   entry_ts,
                '_hit_tp':     hit_tp,
                '_hit_sl':     hit_sl,
                '_time_to_event': time_to_event,
                '_is_censored': is_censored,
            }
            setups.append(feat)
            last_setup_hhmm[direction] = bar_hhmm
            break

    return setups


def build_dataset(years, regime_pkg=None, of_lag_lookup=None):
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    days = pd.read_sql(f"""
        SELECT DISTINCT date FROM market_data
        WHERE year IN ({','.join(map(str, years))})
          AND day_of_week BETWEEN 1 AND 5
        ORDER BY date
    """, conn)['date'].tolist()

    daily_ctx = build_daily_context(conn, days)
    all_setups = []
    wr_window = []; prev_lon_dir = 0
    win_hist = {'LONG': [], 'SHORT': []}; week_counts = {}

    for date_str in days:
        df = load_day(conn, date_str)
        if df is None: continue
        roll_wr = float(np.mean(wr_window[-5:])) if wr_window else 0.5
        if date_str in daily_ctx:
            daily_ctx[date_str]['rolling_wr'] = roll_wr
            daily_ctx[date_str]['prev_lon_dir'] = prev_lon_dir
        wk = pd.Timestamp(date_str).isocalendar()
        week_str = f"{wk.year}_{wk.week}"
        cross_ctx = {
            'dsw_L': _days_since_win(win_hist['LONG'], date_str),
            'dsw_S': _days_since_win(win_hist['SHORT'], date_str),
            'week_cnt': float(week_counts.get(week_str, 0)),
            'td_L': 0.0, 'td_S': 0.0,
        }
        setups = extract_setups(df, date_str, daily_ctx, cross_ctx,
                                regime_pkg=regime_pkg, of_lag_lookup=of_lag_lookup)
        all_setups.extend(setups)
        lon_bars = df[df['hhmm'].between(LON_SESS_START_ET, LON_SESS_END_ET)]
        if len(lon_bars) >= 3:
            lcl = float(lon_bars['close'].iloc[-1]); lmid = (lon_bars['high'].max()+lon_bars['low'].min())/2
            prev_lon_dir = 1 if lcl > lmid else -1
        for s in setups:
            d = s['_direction']
            win_hist[d].append((date_str, s['_label']))
            week_counts[week_str] = week_counts.get(week_str, 0) + 1
            wr_window.append(s['_label'])

    conn.close()
    log.info(f"  {years}: {len(days)} days → {len(all_setups)} setups")
    if not all_setups: return pd.DataFrame()
    df_out = pd.DataFrame(all_setups)

    # MTF ICT
    log.info("   Joining MTF ICT features ...")
    conn2 = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    mtf = compute_mtf_features(conn2, sorted(df_out['_date'].unique()))
    conn2.close()
    df_out = df_out.merge(
        mtf.drop_duplicates('ts_str')[['ts_str'] + [c for c in mtf.columns if c != 'ts_str']],
        left_on='_entry_ts', right_on='ts_str', how='left'
    )
    for c in [c for c in mtf.columns if c != 'ts_str']:
        df_out[c] = df_out[c].fillna(0.0)
    for tf in ['5m', '15m', '1h', '4h']:
        dir_n = np.where(df_out['direction_enc'].values == 0, 1.0, -1.0)
        df_out[f'fvg_aligned_{tf}']     = np.where(dir_n==1, df_out[f'in_bull_{tf}'], df_out[f'in_bear_{tf}'])
        df_out[f'ifvg_aligned_{tf}']    = np.where(dir_n==1, df_out[f'in_ifvg_s_{tf}'], df_out[f'in_ifvg_b_{tf}'])
        df_out[f'breaker_aligned_{tf}'] = np.where(dir_n==1, df_out[f'breaker_b_{tf}'], df_out[f'breaker_s_{tf}'])
    df_out['fvg_tf_confluence'] = sum(df_out.get(f'fvg_aligned_{tf}', pd.Series(0, index=df_out.index)).values for tf in ['5m','15m','1h','4h'])
    df_out['htf_fvg_aligned_mtf'] = np.maximum(
        df_out.get('fvg_aligned_1h', pd.Series(0, index=df_out.index)).values,
        df_out.get('fvg_aligned_4h', pd.Series(0, index=df_out.index)).values)

    # Synthetic OF features
    _OF = Path(__file__).parent / "data" / "orderflow_features.parquet"
    if _OF.exists():
        _of = pd.read_parquet(_OF)
        _of = _of[_of['session_type'] == 'LON'].copy()
        _of['date'] = _of['date'].astype(str)
        _OF_COLS = [c for c in _of.columns if c not in ['session_id','date','session_type',
                    'session_open','session_close','session_high','session_low','total_vol']]
        df_out = df_out.merge(_of[['date'] + _OF_COLS].rename(columns={'date': '_date'}), on='_date', how='left')
        for _c in _OF_COLS:
            df_out[_c] = df_out[_c].fillna(0.0)
        log.info(f"   OF features: {len(_OF_COLS)} merged (LON)")

    # ── Correct regime_enc from precomputed regime_labels.parquet ─────────────
    # predict_regime_enc() was called with sweep_dn/up=0 (hardcoded) → always
    # returned 2 (EXPANSION). Fix: join precomputed labels by (date, LON session).
    _RL = Path(__file__).parent / "data" / "regime_labels.parquet"
    if _RL.exists():
        try:
            rl = pd.read_parquet(_RL)
            rl_lon = rl[rl['session'] == 'LON'][['date', 'regime_enc', 'regime']].copy()
            rl_lon['date'] = pd.to_datetime(rl_lon['date']).dt.strftime('%Y-%m-%d')
            df_out = df_out.merge(rl_lon.rename(columns={
                'date': '_date', 'regime_enc': '_rl_enc', 'regime': '_rl_regime'
            }), on='_date', how='left')
            # Override regime_enc with precomputed value (fill missing with 2=EXPANSION)
            df_out['regime_enc']    = df_out['_rl_enc'].fillna(2).astype(int)
            # Recompute derived regime features
            re = df_out['regime_enc'].values
            df_out['regime_is_pre']  = (re == 1).astype(float)
            df_out['regime_is_exp']  = (re == 2).astype(float)
            df_out['regime_is_ret']  = (re == 3).astype(float)
            df_out['regime_aligned'] = np.where(
                np.isin(re, [1, 2]),
                (df_out.get('h4_bias_aligned', pd.Series(0, index=df_out.index)).values == 1).astype(float),
                0.0)
            # Recompute interaction feature with OF
            if 'of_cvd_lag1' in df_out.columns:
                df_out['of_cvd_regime'] = df_out['of_cvd_lag1'].fillna(0) * re.astype(float)
            df_out.drop(columns=['_rl_enc', '_rl_regime'], errors='ignore', inplace=True)
            dist = dict(pd.Series(re).value_counts().sort_index())
            log.info(f"   regime_enc from precomputed labels (LON): {dist}")
        except Exception as e:
            log.warning(f"   regime_labels join failed: {e} — keeping predict_regime_enc values")
    else:
        log.warning("   regime_labels.parquet not found — regime_enc may be inaccurate")

    log.info(f"   Total columns: {df_out.shape[1]}")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
def train_main_model(X_tr, y_tr, X_te, y_te, sw_, feature_cols):
    """Binary classifier with Optuna + BorderlineSMOTE."""
    val_cut = int(len(X_tr) * 0.80)
    wf_folds = [(np.array([True]*val_cut + [False]*(len(X_tr)-val_cut)),
                 np.array([False]*val_cut + [True]*(len(X_tr)-val_cut)))]

    def objective(trial):
        p = {
            'n_estimators':     trial.suggest_int('n_estimators', 150, 800),
            'max_depth':        trial.suggest_int('max_depth', 2, 3),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.35, 0.75),
            'min_child_weight': trial.suggest_int('min_child_weight', 20, 80),
            'gamma':            trial.suggest_float('gamma', 1.0, 8.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1.0, 8.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 3.0, 10.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 3.0, 12.0),
        }
        smote_r = trial.suggest_float('smote', 0.10, 0.40)
        aucs = []
        for tm, vm in wf_folds:
            Xf = X_tr[tm]; yf = y_tr.values[tm]; swf = sw_[tm]
            Xv = X_tr[vm]; yv = y_tr.values[vm]
            try:
                sm = BorderlineSMOTE(sampling_strategy=smote_r, random_state=42, k_neighbors=5)
                Xs, ys = sm.fit_resample(Xf, yf)
                sws = np.concatenate([swf, np.ones(len(Xs)-len(Xf))])
            except Exception:
                Xs, ys, sws = Xf, yf, swf
            m = xgb.XGBClassifier(**p, eval_metric='logloss', random_state=42, n_jobs=-1,
                                   tree_method='hist', early_stopping_rounds=30)
            m.fit(Xs, ys, sample_weight=sws, eval_set=[(Xv, yv)], verbose=False)
            if yv.sum() > 0 and yv.sum() < len(yv):
                aucs.append(roc_auc_score(yv, m.predict_proba(Xv)[:,1]))
        return float(np.mean(aucs)) if aucs else 0.5

    log.info(f"▶  Optuna ({OPTUNA_TRIALS} trials) ...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False, n_jobs=1)
    bp = study.best_params; smote_r = bp.pop('smote')
    log.info(f"   Best val AUC: {study.best_value:.4f}")

    try:
        sm = BorderlineSMOTE(sampling_strategy=smote_r, random_state=42, k_neighbors=5)
        X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
        sw_sm = np.concatenate([sw_, np.ones(len(X_sm)-len(X_tr))])
    except Exception:
        X_sm, y_sm = X_tr, y_tr; sw_sm = sw_

    model = xgb.XGBClassifier(**bp, eval_metric='logloss', random_state=42, n_jobs=-1, tree_method='hist')
    model.fit(X_sm, y_sm, sample_weight=sw_sm, verbose=False)

    is_auc = roc_auc_score(y_tr, model.predict_proba(X_tr)[:,1])
    oos_auc = 0.0
    if len(X_te) > 20:
        oos_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1])
    log.info(f"   IS={is_auc:.4f}  OOS={oos_auc:.4f}")
    return model, is_auc, oos_auc


def train_quantile_models(X_tr, X_te, df_tr, df_te, feature_cols):
    """
    Train one XGBoost regressor per quantile α for predicting MFE.
    Target: actual MFE value (in points) per trade.
    """
    q_models = {}
    target = '_max_fwd'
    if target not in df_tr.columns:
        log.warning("  _max_fwd not found — skipping quantile models")
        return {}

    y_tr_mfe = df_tr[target].values.astype(float)
    y_te_mfe = df_te[target].values.astype(float) if len(df_te) > 0 else None

    log.info(f"▶  Quantile regression models (MFE): "
             f"mean={y_tr_mfe.mean():.1f}pt, p50={np.median(y_tr_mfe):.1f}pt")

    for alpha in QUANTILE_ALPHAS:
        log.info(f"   Training Q{int(alpha*100)} model ...")
        qm = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=alpha,
            n_estimators=300, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.6, min_child_weight=15,
            reg_alpha=1.0, reg_lambda=3.0,
            random_state=42, n_jobs=-1, tree_method='hist',
        )
        qm.fit(X_tr, y_tr_mfe, verbose=False)
        if y_te_mfe is not None and len(X_te) > 0:
            preds = qm.predict(X_te)
            # Pinball loss
            diff = y_te_mfe - preds
            pinball = np.mean(np.where(diff >= 0, alpha * diff, (alpha-1) * diff))
            log.info(f"   Q{int(alpha*100)} OOS pinball={pinball:.3f}")
        q_models[alpha] = qm

    return q_models


def train_survival_model(X_tr, X_te, df_tr, df_te, feature_cols):
    """
    Train XGBoost regressor for time_to_event (survival analysis).
    Target: log(time_to_event) to approximate Weibull distribution.
    """
    if '_time_to_event' not in df_tr.columns:
        log.warning("  _time_to_event not found — skipping survival model")
        return None

    y_tr_surv = np.log1p(df_tr['_time_to_event'].values.astype(float))
    log.info(f"▶  Survival model (time-to-event): mean={np.exp(y_tr_surv.mean())-1:.1f} bars")

    surv_m = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300, max_depth=3,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.6, min_child_weight=15,
        reg_alpha=1.0, reg_lambda=3.0,
        random_state=42, n_jobs=-1, tree_method='hist',
    )
    surv_m.fit(X_tr, y_tr_surv, verbose=False)
    if len(df_te) > 0:
        y_te_surv = np.log1p(df_te['_time_to_event'].values.astype(float))
        preds = surv_m.predict(X_te)
        mae = np.mean(np.abs(np.expm1(preds) - np.expm1(y_te_surv)))
        log.info(f"   Survival OOS MAE={mae:.1f} bars")
    return surv_m


def train_and_save():
    log.info("═" * 60)
    log.info("LOM TRAIN v3 — regime_enc + quantile + survival")
    log.info("═" * 60)

    # Load helpers
    regime_pkg    = load_regime_classifier()
    of_lag_lookup = load_of_lag_features()

    log.info(f"Extrag IS ({TRAIN_YEARS}) ...")
    df_tr = build_dataset(TRAIN_YEARS, regime_pkg=regime_pkg, of_lag_lookup=of_lag_lookup)
    log.info(f"Extrag OOS ({TEST_YEARS}) ...")
    df_te = build_dataset(TEST_YEARS, regime_pkg=regime_pkg, of_lag_lookup=of_lag_lookup)

    meta_cols    = [c for c in df_tr.columns if c.startswith('_') or c == 'ts_str']
    feature_cols = [c for c in df_tr.columns if c not in meta_cols]

    log.info(f"\nIS:  {len(df_tr)} setups | features: {len(feature_cols)}")
    log.info(f"OOS: {len(df_te)} setups")
    log.info(f"Label IS: {df_tr['_label'].value_counts().to_dict()}")
    log.info(f"Regime distribution IS: {df_tr['regime_enc'].value_counts().to_dict()}")

    if len(df_tr) < 50:
        log.error("Prea puțin data IS"); return

    X_tr = df_tr[feature_cols].fillna(0)
    y_tr = df_tr['_label']
    yr_  = df_tr['_date'].apply(lambda d: int(d[:4]))
    sw_  = np.array([YEAR_WEIGHTS.get(yr, 1.0) for yr in yr_])
    X_te = df_te[feature_cols].fillna(0).reindex(columns=feature_cols, fill_value=0) if len(df_te) > 0 else pd.DataFrame(columns=feature_cols)
    y_te = df_te['_label'] if len(df_te) > 0 else pd.Series(dtype=int)

    # Feature selection
    log.info(f"\n▶  Feature selection (top {TOP_N_FEATURES}) ...")
    neg, pos = (y_tr==0).sum(), (y_tr==1).sum()
    _spw = neg / max(pos, 1)
    _pre = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                              subsample=0.7, colsample_bytree=0.6, min_child_weight=25,
                              gamma=1.5, reg_alpha=2.0, reg_lambda=4.0,
                              scale_pos_weight=_spw, random_state=42, n_jobs=-1,
                              eval_metric='logloss', verbosity=0)
    _pre.fit(X_tr, y_tr, sample_weight=sw_, verbose=False)
    _imp = pd.Series(_pre.feature_importances_, index=feature_cols).sort_values(ascending=False)
    # Always keep regime_enc + lagged OF even if not in top N
    must_keep = [c for c in [
        # Regime + OF
        'regime_enc','regime_is_pre','regime_is_exp','regime_is_ret','regime_aligned',
        'of_cvd_lag1','of_abs_lag1','of_od_lag1','of_cvd_regime',
        # Cross-model (LOM v1 + NOM v1 high-importance features)
        'garch_vol','atr_vs_5d','pre_rng_atr','asia_dir_explicit','asia_range_vs_atr5d',
        'dir_x_hurst','vol_x_sweep','asia_dir_x_h4','h4_x_weekly',
        'spike_vs_asia_hi','spike_vs_asia_lo','dist_lw_hi','dist_lw_lo',
        'value_area_width','dist_asia_hi_atr','dist_asia_lo_atr',
    ] if c in feature_cols]
    selected = list(dict.fromkeys(must_keep + _imp.head(TOP_N_FEATURES).index.tolist()))
    log.info(f"   Selectate {len(selected)} | top5: {_imp.head(5).index.tolist()}")
    X_tr = X_tr[selected]; X_te = X_te.reindex(columns=selected, fill_value=0)
    feature_cols = selected

    # Binary classifier
    log.info("\n── Binary classifier ──")
    main_model, is_auc, oos_auc = train_main_model(X_tr, y_tr, X_te, y_te, sw_, feature_cols)

    # Quantile models
    log.info("\n── Quantile regression ──")
    q_models = train_quantile_models(X_tr, X_te, df_tr, df_te, feature_cols)

    # Survival model
    log.info("\n── Survival model ──")
    surv_model = train_survival_model(X_tr, X_te, df_tr, df_te, feature_cols)

    # Save
    old_auc = 0.0
    if OUT.exists():
        try:
            old_pkg = pickle.load(open(OUT, 'rb'))
            old_auc = old_pkg.get('oos_auc', 0.0)
        except Exception: pass

    if oos_auc >= old_auc - 0.005:
        pkg = {
            'model':            main_model,
            'quantile_models':  q_models,
            'survival_model':   surv_model,
            'features':         feature_cols,
            'is_auc':           round(is_auc, 4),
            'oos_auc':          round(oos_auc, 4),
            'n_features':       len(feature_cols),
            'train_years':      TRAIN_YEARS,
            'test_years':       TEST_YEARS,
            'version':          'v3_regime_enc_quantile_survival',
            'tp_pt':            TP_PT,
            'label_window':     LABEL_WINDOW,
            'quantile_alphas':  QUANTILE_ALPHAS,
            'has_regime_enc':   True,
            'has_lagged_of':    True,
        }
        with open(OUT, 'wb') as f:
            pickle.dump(pkg, f)
        log.info(f"\n💾 Salvat: {OUT}")
        log.info(f"   IS={is_auc:.4f}  OOS={oos_auc:.4f}  (prev={old_auc:.4f})")
        log.info(f"   Quantile models: {list(q_models.keys())}")
        log.info(f"   Survival model: {'✅' if surv_model else '❌'}")
    else:
        log.warning(f"\n⚠️  OOS regression ({oos_auc:.4f} < {old_auc:.4f} - 0.005) — old model kept")

    # Top features
    imp = pd.Series(main_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    log.info(f"\nTop 15 features:\n{imp.head(15).to_string()}")


if __name__ == '__main__':
    train_and_save()
