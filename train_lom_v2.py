"""
train_lom_v2.py — London Open Manipulation v1.2 Enhanced
=========================================================
vs v1.1 (OOS=0.621, IS=0.817, 104 features):
  ✅ Optuna 80 trials (fixed params → tuned)
  ✅ Year weights {2023:0.85, 2024:1.00} + BorderlineSMOTE
  ✅ Rolling regime: vix_proxy_5d/20d, vol_regime, atr_trend, adx_10d_mean, hurst_20d_mean
  ✅ MTF ICT features: FVG/IFVG/Breaker/Rejection pe 5m/15m/1h/4h
  ✅ Calendar context: fomc_proximity, is_nfp_day, is_fomc_day (after session, anticipation)
  ✅ Asia quality: asia_dir_explicit (+1/-1), asia_range_vs_atr5d, equal highs/lows enhanced
  ✅ LOM-specific: asia_close_vs_mid, prev_lon_dir, lon15_bias, sweep_time_et

Output: lom_model_v1.pkl (suprascrie modelul vechi dacă OOS e mai bun)
"""

import sqlite3, pickle, logging, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import BorderlineSMOTE
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("LOM_V2")

DB            = Path(__file__).parent / "mario_trading.db"
OUT           = Path(__file__).parent / "lom_model_v1.pkl"
OPTUNA_TRIALS = 80
# YEAR_WEIGHTS moved below TRAIN_YEARS

MIN_SPIKE_PT       = 5.0
MIN_DISP_PT        = 4.0
TP_PT              = 18.0   # LOM: TP fix (ATR-relativ hurt OOS cu 348 samples)
TP_MULT            = 1.5    # backup (neutilizat direct)
LABEL_WINDOW       = 60
PARTIAL_THRESH_PCT = 0.30

TRAIN_YEARS    = [2023, 2024]
TEST_YEARS     = [2025, 2026]
YEAR_WEIGHTS   = {2023: 0.85, 2024: 1.00}
TOP_N_FEATURES = 70    # 60→70: mai mult semnal cu features noi
# WF CV dezactivat pentru LOM (348 samples/4 folds = ~87/fold, insuficient)
N_WF_FOLDS     = 0

LON_SESS_START_ET = 400
LON_SESS_END_ET   = 700
PRE_LON_END_ET    = 359
ASIA_START_ET     = 0
ASIA_END_ET       = 359

# ── Economic Calendar ─────────────────────────────────────────────────────────
_CAL_PATH = Path(__file__).parent / "data" / "economic_calendar.json"
try:
    _cal = json.loads(_CAL_PATH.read_text())
    FOMC_DATES   = set(_cal.get('fomc',   []))
    NFP_DATES    = set(_cal.get('nfp',    []))
    CPI_DATES    = set(_cal.get('cpi',    []))
    PPI_DATES    = set(_cal.get('ppi',    []))
    ANY_HIGH     = set(_cal.get('any_high', []))
    NEWS_DAYS    = FOMC_DATES | NFP_DATES | CPI_DATES | PPI_DATES
    log.info(f"Calendar: NFP={len(NFP_DATES)}, FOMC={len(FOMC_DATES)}, CPI={len(CPI_DATES)}")
except Exception as _e:
    log.warning(f"Calendar load error: {_e}")
    FOMC_DATES = NFP_DATES = CPI_DATES = PPI_DATES = ANY_HIGH = NEWS_DAYS = set()

def _fomc_prox(date_str):
    try:
        d = pd.Timestamp(date_str).date()
        diffs = [abs((d - pd.Timestamp(x).date()).days) for x in FOMC_DATES]
        return float(min(diffs)) if diffs else 30.0
    except Exception:
        return 30.0


def sv(v, d=0.0):
    try: x = float(v); return x if np.isfinite(x) else d
    except: return d


# ════════════════════════════════════════════════════════════════════════════
# MTF ICT features (identic cu ts_lon)
# ════════════════════════════════════════════════════════════════════════════
def compute_ict_on_tf(df_tf: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    H = df_tf['high'].values.astype(float)
    L = df_tf['low'].values.astype(float)
    C = df_tf['close'].values.astype(float)
    O = df_tf['open'].values.astype(float)
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
    breaker_b = np.zeros(n); breaker_s = np.zeros(n)
    rejection = np.zeros(n)
    active_bull = []; active_bear = []
    inv_bull_zones = []; inv_bear_zones = []
    bull_obs = []; bear_obs = []

    for i in range(n):
        c = C[i]; l = L[i]; h = H[i]; a = A[i]
        new_ab = []
        for top, bot, j in active_bull:
            if i - j > lookback: continue
            if l < bot: inv_bull_zones.append((top, bot, i))
            else: new_ab.append((top, bot, j))
        active_bull = new_ab
        new_ab2 = []
        for top, bot, j in active_bear:
            if i - j > lookback: continue
            if h > top: inv_bear_zones.append((top, bot, i))
            else: new_ab2.append((top, bot, j))
        active_bear = new_ab2
        if bull_top[i] > 0: active_bull.append((bull_top[i], bull_bot[i], i))
        if bear_top[i] > 0: active_bear.append((bear_top[i], bear_bot[i], i))
        if i >= 2:
            pb = C[i-1] - O[i-1]; pr = max(H[i-1] - L[i-1], 0.01)
            if pb > 0.55 * pr and pb > 1.0: bull_obs.append((C[i-1], O[i-1], i-1))
            if pb < -0.55 * pr and abs(pb) > 1.0: bear_obs.append((O[i-1], C[i-1], i-1))
        for top, bot, j in active_bull:
            if bot <= c <= top: in_bull[i] = 1.0
            d = min(abs(c-top), abs(c-bot)) / a
            dist_bull[i] = min(dist_bull[i], d)
        for top, bot, j in active_bear:
            if bot <= c <= top: in_bear[i] = 1.0
            d = min(abs(c-top), abs(c-bot)) / a
            dist_bear[i] = min(dist_bear[i], d)
        for top, bot, k in inv_bull_zones[-15:]:
            if i - k <= lookback * 2 and bot <= c <= top: in_ifvg_b[i] = 1.0
        for top, bot, k in inv_bear_zones[-15:]:
            if i - k <= lookback * 2 and bot <= c <= top: in_ifvg_s[i] = 1.0
        for top, bot, j in bull_obs[-20:]:
            if i - j <= lookback and c < min(bot, O[j]) - a * 0.05:
                if abs(c-top)/a < 0.8 or abs(c-bot)/a < 0.8: breaker_s[i] = 1.0
        for top, bot, j in bear_obs[-20:]:
            if i - j <= lookback and c > max(top, O[j]) + a * 0.05:
                if abs(c-top)/a < 0.8 or abs(c-bot)/a < 0.8: breaker_b[i] = 1.0
        if i >= 2:
            wu = H[i-1]-max(C[i-1],O[i-1]); wd = min(C[i-1],O[i-1])-L[i-1]
            bz = abs(C[i-1]-O[i-1])
            if wu > 2.5*max(bz,0.5) and wu > a*0.3:
                rt = H[i-1]; rb = max(C[i-1],O[i-1])
                if abs(c-rt)/a < 0.6 or abs(c-rb)/a < 0.6: rejection[i] = 1.0
            if wd > 2.5*max(bz,0.5) and wd > a*0.3:
                rt = min(C[i-1],O[i-1]); rb = L[i-1]
                if abs(c-rt)/a < 0.6 or abs(c-rb)/a < 0.6: rejection[i] = 1.0

    return pd.DataFrame({
        'in_bull': in_bull, 'in_bear': in_bear,
        'dist_bull': np.clip(dist_bull, 0, 9.9),
        'dist_bear': np.clip(dist_bear, 0, 9.9),
        'in_ifvg_b': in_ifvg_b, 'in_ifvg_s': in_ifvg_s,
        'breaker_b': breaker_b, 'breaker_s': breaker_s,
        'rejection': rejection,
    }, index=df_tf.index)


def compute_mtf_features(conn, setup_dates: list) -> pd.DataFrame:
    """Identic cu ts_lon — returnează DataFrame indexat pe timestamp cu ICT features MTF."""
    min_d = min(setup_dates); max_d = max(setup_dates)
    warmup_start = (pd.Timestamp(min_d) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    log.info(f"   MTF: loading 1-min data {warmup_start} → {max_d} ...")
    df1m = pd.read_sql(f"""
        SELECT timestamp, open, high, low, close, atr_14
        FROM market_data
        WHERE timestamp >= '{warmup_start} 00:00:00'
          AND timestamp <= '{max_d} 23:59:59'
        ORDER BY timestamp
    """, conn)
    df1m['ts'] = pd.to_datetime(df1m['timestamp'])
    df1m = df1m.set_index('ts').rename(columns={'atr_14': 'atr'})
    df1m['atr'] = df1m['atr'].ffill().fillna(9.0)
    log.info(f"   1-min bars: {len(df1m):,}")

    all_features = pd.DataFrame(index=df1m.index)
    for tf_label, tf_rule, lookback in [
        ('5m', '5min', 25), ('15m', '15min', 20),
        ('1h', '1h', 20), ('4h', '4h', 15),
    ]:
        df_tf = df1m.resample(tf_rule, label='left', closed='left').agg(
            open=('open','first'), high=('high','max'),
            low=('low','min'), close=('close','last'), atr=('atr','last')
        ).dropna(subset=['open'])
        df_tf['atr'] = df_tf['atr'].ffill().fillna(9.0)
        ict = compute_ict_on_tf(df_tf, lookback=lookback)
        ict_ff = ict.reindex(df1m.index, method='ffill')
        for col in ict.columns:
            all_features[f'{col}_{tf_label}'] = ict_ff[col]
    all_features = all_features.fillna(0.0)
    all_features['ts_str'] = all_features.index.strftime('%Y-%m-%d %H:%M:%S')
    log.info(f"   MTF computed: {all_features.shape[1]-1} cols × {len(all_features):,} rows")
    return all_features


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════
def load_day(conn, date_str):
    df = pd.read_sql(f"""
        SELECT timestamp, open, high, low, close, volume,
               atr_14, bar_delta, cum_delta, fvg_up, fvg_down, has_displacement,
               body_size, adx_14, hurst, dist_poc, inside_va, dist_vwap,
               delta_at_high, delta_at_low, big_buy_count, big_sell_count,
               absorption_score, stacked_bull, stacked_bear, of_doi, of_big_balance,
               bar_buy_vol, bar_sell_vol, garch_vol, kalman_smooth,
               fisher_transform, acf_lag1, acf_lag5, rvol,
               vah, val, poc_level, p_hi, p_lo, lw_hi, lw_lo,
               h4_hi, h4_lo, h1_hi, h1_lo, true_open, asia_hi, asia_lo,
               is_smt_bullish, is_smt_bearish,
               day_of_week, month
        FROM market_data
        WHERE date = '{date_str}'
        ORDER BY timestamp
    """, conn)
    if len(df) < 15:
        return None
    df['ts']   = pd.to_datetime(df['timestamp'])
    df['hhmm'] = df['ts'].dt.hour * 100 + df['ts'].dt.minute
    return df


# ════════════════════════════════════════════════════════════════════════════
# Setup extraction (event-driven, identic cu v1.1 + entry_ts pentru MTF join)
# ════════════════════════════════════════════════════════════════════════════
def _days_since_win(win_list, current_date_str):
    """Câte zile de la ultimul win pentru această direcție."""
    cd = pd.Timestamp(current_date_str)
    wins = [pd.Timestamp(d) for d, lbl in reversed(win_list) if lbl == 1]
    return float((cd - wins[0]).days) if wins else 30.0


def extract_setups(df, date_str, daily_ctx: dict, cross_ctx: dict = None):
    setups = []
    pre_lon  = df[df['hhmm'] <= PRE_LON_END_ET]
    lon_sess = df[df['hhmm'].between(LON_SESS_START_ET, LON_SESS_END_ET)]
    if len(pre_lon) < 5 or len(lon_sess) < 3:
        return setups

    pre_hi  = float(pre_lon['high'].max())
    pre_lo  = float(pre_lon['low'].min())
    pre_rng = pre_hi - pre_lo
    if pre_rng < 3:
        return setups

    atr = float(df['atr_14'].replace(0, np.nan).dropna().iloc[-1]) if len(df) > 0 else 10.0
    if atr <= 0: atr = 10.0

    # Asia session quality (LOM sweeps Asia levels → context crucial)
    asia_df    = df[df['hhmm'].between(ASIA_START_ET, ASIA_END_ET)]
    asia_open  = float(asia_df['open'].iloc[0])  if len(asia_df) > 0 else pre_hi
    asia_close = float(asia_df['close'].iloc[-1]) if len(asia_df) > 0 else pre_lo
    lon_open   = float(lon_sess['open'].iloc[0])  if len(lon_sess) > 0 else asia_close
    asia_hi_s  = float(asia_df['high'].max())     if len(asia_df) > 0 else pre_hi
    asia_lo_s  = float(asia_df['low'].min())      if len(asia_df) > 0 else pre_lo
    asia_rng   = asia_hi_s - asia_lo_s
    asia_mid   = (asia_hi_s + asia_lo_s) / 2 if asia_hi_s > asia_lo_s else pre_hi
    # Asia direction: closed above or below its own midpoint
    asia_dir   = 1 if asia_close > asia_mid else -1
    asia_tight = 1 if asia_rng < atr * 0.6 else 0
    asia_wide  = 1 if asia_rng > atr * 1.4 else 0
    # Equal lows/highs in Asia (liquidity pools)
    eq_tol    = atr * 0.3
    pre_highs = pre_lon['high'].values; pre_lows = pre_lon['low'].values
    eq_hi = max(0, sum(1 for h in pre_highs if abs(h - pre_hi) <= eq_tol) - 1)
    eq_lo = max(0, sum(1 for l in pre_lows  if abs(l - pre_lo) <= eq_tol) - 1)

    partial_thresh = pre_rng * 0.50
    lon_reset = lon_sess.reset_index(drop=False)
    last_setup_hhmm = {'LONG': -999, 'SHORT': -999}

    for i in range(1, len(lon_reset) - 2):
        bar      = lon_reset.iloc[i]
        bar_hi   = sv(bar['high'])
        bar_lo   = sv(bar['low'])
        bar_hhmm = int(bar['hhmm'])

        spike_up = bar_hi - pre_hi
        spike_dn = pre_lo - bar_lo

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

            if disp_bar is None:
                continue

            entry_price = sv(disp_bar['close'])
            entry_hhmm  = int(disp_bar['hhmm'])
            entry_ts    = str(disp_bar['timestamp'])   # pentru MTF join
            dir_num     = 1 if direction == 'LONG' else -1

            future = df[df['hhmm'] > entry_hhmm].head(LABEL_WINDOW)
            if len(future) < 3:
                continue
            if direction == 'LONG':
                reached_tp = float(future['high'].max()) >= entry_price + TP_PT
                max_fwd    = float(future['high'].max() - entry_price)
            else:
                reached_tp = float(future['low'].min()) <= entry_price - TP_PT
                max_fwd    = float(entry_price - future['low'].min())
            label = 1 if reached_tp else 0

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

            wick_pct         = wick * atr / spike_bar_range
            sweep_wick_clean = 1 if wick_pct > 0.5 else 0
            sweep_depth_atr  = spike_mag / atr
            deep_sweep       = 1 if sweep_depth_atr > 1.5 else 0
            sweep_quality    = ts_close_inside*0.4 + sweep_wick_clean*0.3 + deep_sweep*0.2 + 0.1

            # Sweep bar close position (unde se închide bara de spike în propriul range)
            sb_hi  = sv(bar['high']); sb_lo = sv(bar['low'])
            sb_rng = max(sb_hi - sb_lo, 0.01)
            sweep_close_pct = (sv(bar['close']) - sb_lo) / sb_rng  # 0=close jos, 1=close sus
            # Pentru SHORT bun: close jos (0) = wick sus = rejection curat
            sweep_close_rejection = (1 - sweep_close_pct) if direction=='SHORT' else sweep_close_pct

            disp_body = abs(sv(disp_bar['close']) - sv(disp_bar['open']))

            h4_hi = sv(r0['h4_hi']); h4_lo = sv(r0['h4_lo'])
            h1_hi = sv(r0['h1_hi']); h1_lo = sv(r0['h1_lo'])
            h4_mid = (h4_hi + h4_lo) / 2 if h4_hi > 0 and h4_lo > 0 else 0
            h1_mid = (h1_hi + h1_lo) / 2 if h1_hi > 0 and h1_lo > 0 else 0
            h4_bias = 1 if entry_price < h4_mid else (-1 if h4_mid > 0 else 0)
            h1_bias = 1 if entry_price < h1_mid else (-1 if h1_mid > 0 else 0)

            lw_hi = sv(r0['lw_hi']); lw_lo = sv(r0['lw_lo']); lw_rng = lw_hi - lw_lo
            weekly_prem = (entry_price - lw_lo) / lw_rng if lw_rng > 0 else 0.5
            # Confluența nivelului sweepat cu prev_week levels
            sweep_lvl_for_conf = spike_hi_val if direction=='SHORT' else spike_lo_val
            level_at_wk_hi = 1 if lw_hi > 0 and abs(sweep_lvl_for_conf - lw_hi) < atr * 0.8 else 0
            level_at_wk_lo = 1 if lw_lo > 0 and abs(sweep_lvl_for_conf - lw_lo) < atr * 0.8 else 0
            level_at_weekly = max(level_at_wk_hi, level_at_wk_lo)

            asia_hi_v = sv(r0.get('asia_hi', 0)); asia_lo_v = sv(r0.get('asia_lo', 0))

            lon15 = lon_sess[lon_sess['hhmm'].between(LON_SESS_START_ET, LON_SESS_START_ET + 15)]
            lon15_rng = float(lon15['high'].max() - lon15['low'].min()) if len(lon15) > 0 else 0
            lon15_close = float(lon15['close'].iloc[-1]) if len(lon15) > 0 else entry_price
            lon15_mid   = (float(lon15['high'].max()) + float(lon15['low'].min())) / 2 if len(lon15) > 0 else entry_price
            lon15_bias  = 1 if lon15_close > lon15_mid else -1

            # Asia internal structure (trending vs ranging)
            asia_half = len(asia_df) // 2 if len(asia_df) >= 4 else 0
            if asia_half > 1:
                asia_first_close  = float(asia_df['close'].iloc[:asia_half].mean())
                asia_second_close = float(asia_df['close'].iloc[asia_half:].mean())
                asia_trending = 1 if abs(asia_second_close - asia_first_close) > atr * 0.2 else 0
                asia_trend_dir = 1 if asia_second_close > asia_first_close else -1  # +1=bullish trend
            else:
                asia_trending = 0; asia_trend_dir = 0
            asia_close_pct = (asia_close - asia_lo_s) / max(asia_rng, 0.01)  # 0=closes at lows, 1=closes at highs
            # Displacement quality
            db_hi = sv(disp_bar['high']); db_lo = sv(disp_bar['low'])
            db_rng = max(db_hi - db_lo, 0.01)
            db_close = sv(disp_bar['close']); db_open = sv(disp_bar['open'])
            disp_close_pct = (db_close - db_lo) / db_rng  # 0=closes at lows, 1=closes at highs
            disp_close_conviction = disp_close_pct if direction=='LONG' else (1 - disp_close_pct)
            # How many bars between spike and displacement
            disp_bars_after_spike = max(0, int(disp_bar['hhmm']) - bar_hhmm) // 1  # rough minute diff
            disp_fast = 1 if disp_bars_after_spike <= 15 else 0
            # Displacement volume vs LON average
            lon_avg_vol = float(lon_sess['volume'].mean()) if len(lon_sess) > 1 else 1.0
            disp_vol = sv(disp_bar.get('volume', lon_avg_vol)); disp_vol_ratio = disp_vol / max(lon_avg_vol, 1)

            pre_vol  = float(pre_lon['volume'].sum()) if len(pre_lon) > 0 else 1.0
            lon_vol  = float(lon_sess['volume'].sum()) if len(lon_sess) > 0 else 1.0
            vol_ratio = lon_vol / pre_vol if pre_vol > 0 else 1.0

            spike_delta = sv(lon_sess['bar_delta'].sum()) if len(lon_sess) > 0 else 0
            fvg_up_v   = int(lon_sess['fvg_up'].any())   if 'fvg_up'   in lon_sess.columns else 0
            fvg_down_v = int(lon_sess['fvg_down'].any()) if 'fvg_down' in lon_sess.columns else 0

            adx_v   = sv(r0['adx_14'])
            hurst_v = sv(r0['hurst'], 0.5)

            # Rolling regime (pre-computed per date)
            dctx = daily_ctx.get(date_str, {})
            vix5   = dctx.get('vix_proxy_5d',   2.0)
            vix20  = dctx.get('vix_proxy_20d',  2.0)
            vol_rg = dctx.get('vol_regime',     1.0)
            atr_tr = dctx.get('atr_trend',      1.0)
            adx10  = dctx.get('adx_10d_mean',   20.0)
            hst20  = dctx.get('hurst_20d_mean', 0.5)
            atr5d  = dctx.get('atr_5d',         atr)
            roll_wr= dctx.get('rolling_wr',     0.5)
            # cross-setup context (cold/hot streak, weekly frequency)
            cctx   = cross_ctx or {}
            dsw_dir= cctx.get('dsw_L' if direction=='LONG' else 'dsw_S', 30.0)
            dsw_any= min(cctx.get('dsw_L', 30.0), cctx.get('dsw_S', 30.0))
            wk_cnt = float(cctx.get('week_cnt', 0))
            td_cnt = float(cctx.get('td_L' if direction=='LONG' else 'td_S', 0))

            feat = {
                # ── Spike ────────────────────────────────────────────────────
                'spike_mag':            spike_mag,
                'spike_mag_atr':        spike_mag / atr,
                'spike_vs_range':       spike_mag / pre_rng if pre_rng > 0 else 0,
                'pre_rng_atr':          pre_rng / atr,
                # ── TS anti-fakeout ──────────────────────────────────────────
                'ts_close_inside':      ts_close_inside,
                'ts_rejection_str':     ts_rejection_str,
                'ts_wick_pct':          ts_wick_pct,
                'ts_body_pct':          ts_body_pct,
                'ts_close_quality':     ts_close_quality,
                'ts_wick_dom':          1 if ts_wick_pct > 0.6 else 0,
                'ts_htf_anti':          1 if h4_bias == dir_num else 0,
                'ts_combo_score':       ts_close_inside * ts_rejection_str,
                'ts_sweep_depth_pts':   spike_mag,
                'ts_sweep_depth_atr':   sweep_depth_atr,
                # ── Sweep quality ────────────────────────────────────────────
                'sweep_wick_atr':       wick,
                'sweep_wick_pct':       wick_pct,
                'sweep_wick_clean':     sweep_wick_clean,
                'sweep_depth_atr':      sweep_depth_atr,
                'deep_sweep':           deep_sweep,
                'shallow_sweep':        1 if sweep_depth_atr < 0.5 else 0,
                'sweep_with_disp':      1,
                'sweep_quality_score':  sweep_quality,
                'equal_level_score':    (eq_hi if direction == 'SHORT' else eq_lo) / max(len(pre_lon), 1),
                'equal_hi_count':       float(eq_hi),
                'equal_lo_count':       float(eq_lo),
                'sweep_aligned_eq':     (eq_hi if direction == 'SHORT' else eq_lo) / max(1, eq_hi + eq_lo + 1),
                # ── Displacement ─────────────────────────────────────────────
                'disp_body':            disp_body,
                'disp_body_atr':        disp_body / atr,
                'disp_range':           sv(disp_bar['high'] - disp_bar['low']),
                'disp_wick_ratio':      (sv(disp_bar['high'] - disp_bar['low']) - disp_body) / max(disp_body, 0.01),
                'has_disp':             1,
                'body_pct':             disp_body / max(sv(disp_bar['high'] - disp_bar['low']), 0.01),
                'body_bear':            1 if direction == 'SHORT' else 0,
                # ── HTF bias ─────────────────────────────────────────────────
                'h4_bias':              h4_bias,
                'h1_bias':              h1_bias,
                'h4_h1_aligned':        1 if h4_bias == h1_bias and h4_bias != 0 else 0,
                'h4_bias_aligned':      1 if h4_bias == dir_num else 0,
                # ── Weekly context ───────────────────────────────────────────
                'weekly_premium_pct':   weekly_prem,
                'in_weekly_premium':    1 if weekly_prem > 0.5 else 0,
                'in_weekly_discount':   1 if weekly_prem < 0.5 else 0,
                'weekly_prem_aligned':  1 if (direction == 'SHORT' and weekly_prem > 0.5) or (direction == 'LONG' and weekly_prem < 0.5) else 0,
                'h4_x_weekly':          (1 if h4_bias == dir_num else 0) * (1 if (direction=='SHORT' and weekly_prem>0.5) or (direction=='LONG' and weekly_prem<0.5) else 0),
                'lw_range_atr':         lw_rng / atr if atr > 0 else 0,
                'week_range_so_far':    (df['high'].max() - df['low'].min()) / atr if atr > 0 else 0,
                'dist_prev_wk_lo':      abs(entry_price - lw_lo) / atr,
                'dist_lw_hi':           abs(entry_price - lw_hi) / atr,
                'dist_lw_lo':           abs(entry_price - lw_lo) / atr,
                # ── Asia context (LOM: pre-LON = Asia) ───────────────────────
                'dist_asia_hi_atr':     abs(entry_price - asia_hi_v) / atr if asia_hi_v > 0 else 0,
                'dist_asia_lo_atr':     abs(entry_price - asia_lo_v) / atr if asia_lo_v > 0 else 0,
                'asia_range_atr':       asia_rng / atr if asia_rng > 0 else 0,
                'spike_vs_asia_hi':     (spike_hi_val - asia_hi_s) / atr if asia_hi_s > 0 else 0,
                'spike_vs_asia_lo':     (asia_lo_s - spike_lo_val) / atr if asia_lo_s > 0 else 0,
                # NUOVO: Asia quality features
                'asia_dir_explicit':    float(asia_dir),          # +1=bull Asia, -1=bear Asia
                'asia_dir_aligned':     1 if asia_dir == dir_num else 0,
                'asia_dir_opposite':    1 if asia_dir != dir_num else 0,
                'asia_close_vs_mid':    (asia_close - asia_mid) / atr if atr > 0 else 0,
                'asia_tight':           float(asia_tight),
                'asia_wide':            float(asia_wide),
                'asia_range_vs_atr5d':  float(np.clip(asia_rng / max(atr5d, 1.0), 0, 10)),
                'sweep_vs_asia_pct':    spike_mag / max(asia_rng, 0.1),
                # ── London first 15min ────────────────────────────────────────
                'lon15_range_atr':      lon15_rng / atr,
                'in_first_15':          1 if bar_hhmm <= LON_SESS_START_ET + 15 else 0,
                'lon15_bias':           float(lon15_bias),
                'lon15_aligned':        1 if lon15_bias == dir_num else 0,
                'sweep_time_early':     1 if bar_hhmm <= LON_SESS_START_ET + 30 else 0,
                'sweep_time_mid':       1 if LON_SESS_START_ET + 30 < bar_hhmm <= LON_SESS_START_ET + 90 else 0,
                'sweep_time_late':      1 if bar_hhmm > LON_SESS_START_ET + 90 else 0,
                # ── PDH/PDL/True open ────────────────────────────────────────
                'above_true_open':      1 if entry_price > sv(r0['true_open']) else 0,
                'dist_true_open':       abs(entry_price - sv(r0['true_open'])) / atr,
                'dist_pdh_atr':         abs(entry_price - sv(r0['p_hi'])) / atr,
                'dist_pdl_atr':         abs(entry_price - sv(r0['p_lo'])) / atr,
                # ── VA / POC ─────────────────────────────────────────────────
                'inside_va':            sv(r0['inside_va']),
                'dist_poc_entry':       sv(r0['dist_poc']) / atr,
                'entry_in_pre_range':   int(pre_lo <= entry_price <= pre_hi),
                'dist_poc_atr':         sv(r0['dist_poc']) / atr,
                'dist_vwap_atr':        sv(r0['dist_vwap']) / atr,
                # ── Volume / delta ───────────────────────────────────────────
                'vol_ratio':            vol_ratio,
                'spike_delta':          spike_delta,
                'disp_delta':           sv(after_early['bar_delta'].sum()) if len(after_early) > 0 else 0,
                'delta_at_high':        sv(lon_sess['delta_at_high'].sum()) if 'delta_at_high' in lon_sess.columns else 0,
                'delta_at_low':         sv(lon_sess['delta_at_low'].sum())  if 'delta_at_low'  in lon_sess.columns else 0,
                'big_buy':              1 if vol_ratio > 2 and direction == 'LONG' else 0,
                'big_sell':             1 if vol_ratio > 2 and direction == 'SHORT' else 0,
                'big_imbalance':        1 if vol_ratio > 2 else 0,
                'absorption':           sv(lon_sess['absorption_score'].mean()) if 'absorption_score' in lon_sess.columns else 0,
                'bar_delta_norm':       spike_delta / atr,
                'cum_delta_norm':       sv(disp_bar.get('cum_delta', 0)) / atr,
                'buy_sell_ratio':       sv(lon_sess['bar_buy_vol'].sum()) / max(sv(lon_sess['bar_sell_vol'].sum()), 1),
                'of_doi':               sv(lon_sess['of_doi'].mean()) if 'of_doi' in lon_sess.columns else 0,
                'stacked_bull':         int(lon_sess['stacked_bull'].any()) if 'stacked_bull' in lon_sess.columns else 0,
                'stacked_bear':         int(lon_sess['stacked_bear'].any()) if 'stacked_bear' in lon_sess.columns else 0,
                # ── FVG (1-min) ───────────────────────────────────────────────
                'fvg_up':               fvg_up_v,
                'fvg_down':             fvg_down_v,
                'htf_fvg_aligned':      1 if (direction == 'SHORT' and fvg_down_v) or (direction == 'LONG' and fvg_up_v) else 0,
                'ob_proxy_bull':        int(lon_sess['stacked_bull'].any()) if 'stacked_bull' in lon_sess.columns else 0,
                'ob_proxy_bear':        int(lon_sess['stacked_bear'].any()) if 'stacked_bear' in lon_sess.columns else 0,
                'ob_aligned':           1 if (direction == 'SHORT' and int(lon_sess['stacked_bear'].any() if 'stacked_bear' in lon_sess.columns else 0)) or (direction == 'LONG' and int(lon_sess['stacked_bull'].any() if 'stacked_bull' in lon_sess.columns else 0)) else 0,
                'vol_x_fvg':            vol_ratio * (1 if (direction == 'LONG' and fvg_up_v) or (direction == 'SHORT' and fvg_down_v) else 0),
                # ── Technical ────────────────────────────────────────────────
                'adx':                  adx_v,
                'adx_strong':           1 if adx_v > 25 else 0,
                'hurst':                hurst_v,
                'fisher_transform':     sv(r0['fisher_transform']),
                'fisher_extreme':       1 if abs(sv(r0['fisher_transform'])) > 2 else 0,
                'acf_lag1':             sv(r0['acf_lag1']),
                'acf_lag5':             sv(r0['acf_lag5']),
                'kalman_smooth':        sv(r0['kalman_smooth']),
                'garch_vol':            sv(r0['garch_vol']),
                'rvol':                 sv(r0['rvol'], 1.0),
                # ── Rolling regime (NEW) ──────────────────────────────────────
                'vix_proxy_5d':         float(vix5),
                'vix_proxy_20d':        float(vix20),
                'vol_regime':           float(vol_rg),
                'vol_high':             1 if vol_rg > 1.2 else 0,
                'vol_low':              1 if vol_rg < 0.8 else 0,
                'atr_trend':            float(atr_tr),
                'atr_expanding':        1 if atr_tr > 1.15 else 0,
                'atr_contracting':      1 if atr_tr < 0.85 else 0,
                'adx_10d_mean':         float(adx10),
                'hurst_20d_mean':       float(hst20),
                'atr_5d':               float(atr5d),
                'atr_vs_5d':            float(np.clip(atr / max(atr5d, 1.0), 0, 3)),
                'regime_trending':      1 if adx10 > 22 and hst20 > 0.52 else 0,
                'rolling_5sess_wr':     float(roll_wr),
                'recent_wr_high':       1 if roll_wr > 0.35 else 0,
                'recent_wr_low':        1 if roll_wr < 0.12 else 0,
                # ── Calendar (after-session context — anticipation effect) ────
                'is_nfp_day':           1 if date_str in NFP_DATES    else 0,
                'is_fomc_day':          1 if date_str in FOMC_DATES   else 0,
                'is_cpi_day':           1 if date_str in CPI_DATES    else 0,
                'is_news_day':          1 if date_str in NEWS_DAYS    else 0,
                'fomc_proximity':       float(np.clip(_fomc_prox(date_str) / 14.0, 0, 1)),
                # ── Time ─────────────────────────────────────────────────────
                'day_of_week':          sv(r0['day_of_week']),
                'is_monday':            1 if int(sv(r0['day_of_week'])) == 0 else 0,
                'is_tuesday':           1 if int(sv(r0['day_of_week'])) == 1 else 0,
                'is_wednesday':         1 if int(sv(r0['day_of_week'])) == 2 else 0,
                'is_thursday':          1 if int(sv(r0['day_of_week'])) == 3 else 0,
                'is_friday':            1 if int(sv(r0['day_of_week'])) == 4 else 0,
                'month':                sv(r0['month']),
                # ── Interactions ─────────────────────────────────────────────
                'dir_x_adx':            dir_num * adx_v,
                'dir_x_hurst':          dir_num * hurst_v,
                'sweep_x_h4':           sweep_quality * (1 if h4_bias == dir_num else 0),
                'ts_close_x_h4':        ts_close_inside * (1 if h4_bias == dir_num else 0),
                'vol_x_sweep':          vol_rg * sweep_quality,
                'vol_x_ts_close':       vol_rg * ts_close_inside,
                'asia_dir_x_h4':        float(asia_dir) * float(h4_bias),
                'sweep_x_eq_level':     sweep_quality * ((eq_hi if direction=='SHORT' else eq_lo) / max(1, len(pre_lon))),
                # ── SMT (Smart Money Trap) ───────────────────────────────────
                'is_smt_bullish':       sv(r0.get('is_smt_bullish', 0)),
                'is_smt_bearish':       sv(r0.get('is_smt_bearish', 0)),
                'smt_aligned':          sv(r0.get('is_smt_bullish', 0)) if dir_num == 1 else sv(r0.get('is_smt_bearish', 0)),
                # ── Gap LON open vs Asia close ───────────────────────────────
                'gap_vs_asia_close_atr': (lon_open - asia_close) / atr,
                'gap_up_lon_open':       1 if lon_open > asia_close + atr * 0.2 else 0,
                'gap_down_lon_open':     1 if lon_open < asia_close - atr * 0.2 else 0,
                'gap_lon_aligned':       1 if ((dir_num == 1 and lon_open < asia_close - atr*0.2) or
                                               (dir_num == -1 and lon_open > asia_close + atr*0.2)) else 0,
                # ── Previous LON direction ───────────────────────────────────
                'prev_lon_dir':         float(dctx.get('prev_lon_dir', 0)),
                'prev_lon_aligned':     1 if dctx.get('prev_lon_dir', 0) == dir_num else 0,
                'prev_lon_opposite':    1 if dctx.get('prev_lon_dir', 0) == -dir_num else 0,
                # ── Features noi: Displacement quality ───────────────────────
                'disp_close_pct':       disp_close_pct,           # unde se închide bara disp (0=jos, 1=sus)
                'disp_close_conviction':disp_close_conviction,    # aliniat cu direcția (1=bun)
                'disp_fast':            float(disp_fast),          # displacement rapid (<15min) = mai bun
                'disp_bars_mins':       float(min(disp_bars_after_spike, 60)),
                'disp_vol_spike':       float(np.clip(disp_vol_ratio, 0, 5)),  # volum disp vs LON avg
                # ── Features noi: Sweep bar structure ────────────────────────
                'sweep_close_pct':      sweep_close_pct,          # unde se închide bara de spike
                'sweep_close_rejection':sweep_close_rejection,    # cât de curat e rejection (aliniat cu dir)
                # ── Features noi: Asia structure internă ─────────────────────
                'asia_trending':        float(asia_trending),     # Asia a trendat intern
                'asia_trend_dir':       float(asia_trend_dir),    # +1=bull intern, -1=bear intern
                'asia_close_pct':       asia_close_pct,           # poziția close în range Asia (0=bottom, 1=top)
                'asia_close_aligned':   1 if (asia_close_pct < 0.3 and direction=='SHORT') or
                                            (asia_close_pct > 0.7 and direction=='LONG') else 0,
                'asia_trend_aligned':   1 if asia_trend_dir == dir_num else 0,
                # ── Features noi: Level confluență ───────────────────────────
                'level_at_weekly':      float(level_at_weekly),   # sweep la nivel de confluență weekly
                'level_wk_hi_conf':     float(level_at_wk_hi),
                'level_wk_lo_conf':     float(level_at_wk_lo),
                # ── Normalizare completă (features în puncte absolute → ATR) ──
                'disp_range_atr':       sv(disp_bar['high'] - disp_bar['low']) / atr,
                # ── Cross-setup context (cold/hot streak) ────────────────────
                'days_since_win_dir':   float(np.clip(dsw_dir, 0, 30)),
                'days_since_win_any':   float(np.clip(dsw_any, 0, 30)),
                'week_setup_count':     float(np.clip(wk_cnt, 0, 10)),
                'today_same_dir_cnt':   float(np.clip(td_cnt, 0, 5)),
                'hot_streak':           1 if dsw_dir <= 2 else 0,
                'cold_streak':          1 if dsw_dir >= 7 else 0,
                'first_today':          1 if td_cnt == 0 else 0,
                # ── Direction ────────────────────────────────────────────────
                'direction_enc':        1 if direction == 'SHORT' else 0,
                # ── Meta ─────────────────────────────────────────────────────
                '_label':      label,
                '_direction':  direction,
                '_date':       str(date_str),
                '_entry_px':   entry_price,
                '_max_fwd':    max_fwd,
                '_entry_hhmm': entry_hhmm,
                '_entry_ts':   entry_ts,      # pentru MTF join
            }
            setups.append(feat)
            last_setup_hhmm[direction] = bar_hhmm
            break
    return setups


# ════════════════════════════════════════════════════════════════════════════
# Daily rolling stats (pre-computate pentru toți anii de training)
# ════════════════════════════════════════════════════════════════════════════
def build_daily_context(conn, dates: list) -> dict:
    log.info("   Pre-computing rolling daily regime ...")
    min_d = min(dates); max_d = max(dates)
    warmup = (pd.Timestamp(min_d) - pd.Timedelta(days=40)).strftime('%Y-%m-%d')
    dr = pd.read_sql(f"""
        SELECT date(timestamp) as date,
               (MAX(high)-MIN(low)) as daily_range,
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


def build_dataset(years):
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    days = pd.read_sql(f"""
        SELECT DISTINCT date FROM market_data
        WHERE year IN ({','.join(map(str, years))})
          AND day_of_week BETWEEN 1 AND 5
        ORDER BY date
    """, conn)['date'].tolist()

    daily_ctx = build_daily_context(conn, days)

    all_setups   = []
    wr_window    = []
    prev_lon_dir = 0
    win_hist     = {'LONG': [], 'SHORT': []}  # (date_str, label)
    week_counts  = {}                          # week_str → total setups în săptămână
    for date_str in days:
        df = load_day(conn, date_str)
        if df is None:
            continue
        roll_wr = float(np.mean(wr_window[-5:])) if wr_window else 0.5
        if date_str in daily_ctx:
            daily_ctx[date_str]['rolling_wr']  = roll_wr
            daily_ctx[date_str]['prev_lon_dir'] = prev_lon_dir
        # cross-setup context pentru această zi
        wk = pd.Timestamp(date_str).isocalendar()
        week_str = f"{wk.year}_{wk.week}"
        cross_ctx = {
            'dsw_L':   _days_since_win(win_hist['LONG'],  date_str),
            'dsw_S':   _days_since_win(win_hist['SHORT'], date_str),
            'week_cnt': float(week_counts.get(week_str, 0)),
            'td_L': 0.0, 'td_S': 0.0,  # reset la fiecare zi
        }
        setups = extract_setups(df, date_str, daily_ctx, cross_ctx)
        all_setups.extend(setups)

        # Actualizează prev_lon_dir pentru ziua următoare
        lon_bars = df[df['hhmm'].between(LON_SESS_START_ET, LON_SESS_END_ET)]
        if len(lon_bars) >= 3:
            lhi = float(lon_bars['high'].max()); llo = float(lon_bars['low'].min())
            lcl = float(lon_bars['close'].iloc[-1]); lmid = (lhi + llo) / 2
            prev_lon_dir = 1 if lcl > lmid else -1

        # Update cross-setup tracking
        for s in setups:
            d = s['_direction']
            win_hist[d].append((date_str, s['_label']))
            week_counts[week_str] = week_counts.get(week_str, 0) + 1
            wr_window.append(s['_label'])

    conn.close()
    log.info(f"  {years}: {len(days)} zile → {len(all_setups)} setups")

    if not all_setups:
        return pd.DataFrame()

    df_out = pd.DataFrame(all_setups)

    # ── Join MTF ICT features ────────────────────────────────────────────
    log.info("   Joining MTF ICT features ...")
    conn2 = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    setup_dates = sorted(df_out['_date'].unique())
    mtf = compute_mtf_features(conn2, setup_dates)
    conn2.close()

    df_out = df_out.merge(
        mtf.drop_duplicates('ts_str')[['ts_str'] + [c for c in mtf.columns if c != 'ts_str']],
        left_on='_entry_ts', right_on='ts_str', how='left'
    )
    mtf_cols = [c for c in mtf.columns if c != 'ts_str']
    for c in mtf_cols:
        df_out[c] = df_out[c].fillna(0.0)

    # Add derived MTF features
    for tf_label in ['5m', '15m', '1h', '4h']:
        in_bull = df_out[f'in_bull_{tf_label}'].values
        in_bear = df_out[f'in_bear_{tf_label}'].values
        in_ifvg_b = df_out[f'in_ifvg_b_{tf_label}'].values
        in_ifvg_s = df_out[f'in_ifvg_s_{tf_label}'].values
        brk_b = df_out[f'breaker_b_{tf_label}'].values
        brk_s = df_out[f'breaker_s_{tf_label}'].values
        dir_enc = df_out['direction_enc'].values
        dir_n = np.where(dir_enc == 0, 1.0, -1.0)
        df_out[f'fvg_aligned_{tf_label}']     = np.where(dir_n == 1, in_bull, in_bear)
        df_out[f'ifvg_aligned_{tf_label}']    = np.where(dir_n == 1, in_ifvg_s, in_ifvg_b)
        df_out[f'breaker_aligned_{tf_label}'] = np.where(dir_n == 1, brk_b, brk_s)

    df_out['fvg_tf_confluence'] = (
        df_out.get('fvg_aligned_5m', pd.Series(0, index=df_out.index)).values +
        df_out.get('fvg_aligned_15m', pd.Series(0, index=df_out.index)).values +
        df_out.get('fvg_aligned_1h', pd.Series(0, index=df_out.index)).values +
        df_out.get('fvg_aligned_4h', pd.Series(0, index=df_out.index)).values
    )
    df_out['htf_fvg_aligned_mtf'] = np.maximum(
        df_out.get('fvg_aligned_1h', pd.Series(0, index=df_out.index)).values,
        df_out.get('fvg_aligned_4h', pd.Series(0, index=df_out.index)).values
    )
    df_out['vol_x_htf_fvg'] = df_out['vol_regime'].values * df_out['htf_fvg_aligned_mtf'].values
    df_out['sweep_x_htf_fvg'] = df_out['sweep_quality_score'].values * df_out['htf_fvg_aligned_mtf'].values


    # ── Synthetic Order Flow features ─────────────────────────────────────
    _OF_PATH = Path(__file__).parent / "data" / "orderflow_features.parquet"
    if _OF_PATH.exists():
        import pandas as _pd2
        _of = _pd2.read_parquet(_OF_PATH)
        _of = _of[_of['session_type'] == 'LON'].copy()
        _of['date'] = _of['date'].astype(str)
        _OF_COLS = [c for c in _of.columns if c not in ['session_id','date','session_type',
                    'session_open','session_close','session_high','session_low','total_vol']]
        _of_m = _of[['date'] + _OF_COLS].rename(columns={'date': '_date'})
        df_out = df_out.merge(_of_m, on='_date', how='left')
        for _c in _OF_COLS:
            df_out[_c] = df_out[_c].fillna(0.0)
        log.info(f"   Order flow: {len(_OF_COLS)} features merged (LON)")
    log.info(f"   MTF joined: {df_out.shape[1]} total columns")
    return df_out


# ════════════════════════════════════════════════════════════════════════════
# Training with Optuna
# ════════════════════════════════════════════════════════════════════════════
def train_and_save():
    log.info("═" * 60)
    log.info("LOM TRAIN v1.2 — Enhanced (Optuna + MTF + Rolling Regime)")
    log.info("═" * 60)

    log.info(f"Extrag IS ({TRAIN_YEARS})...")
    df_tr = build_dataset(TRAIN_YEARS)
    log.info(f"Extrag OOS ({TEST_YEARS})...")
    df_te = build_dataset(TEST_YEARS)

    meta_cols    = [c for c in df_tr.columns if c.startswith('_') or c == 'ts_str']
    feature_cols = [c for c in df_tr.columns if c not in meta_cols]

    log.info(f"\nIS:  {len(df_tr)} setups | features: {len(feature_cols)}")
    log.info(f"OOS: {len(df_te)} setups")
    log.info(f"Label IS: {df_tr['_label'].value_counts().to_dict()}")

    if len(df_tr) < 50:
        log.error("Prea puțin data IS")
        return

    X_tr = df_tr[feature_cols].fillna(0)
    y_tr = df_tr['_label']
    yr_  = df_tr['_date'].apply(lambda d: int(d[:4]))
    sw_  = np.array([YEAR_WEIGHTS.get(yr, 1.0) for yr in yr_])

    X_te = df_te[feature_cols].fillna(0).reindex(columns=feature_cols, fill_value=0)
    y_te = df_te['_label']

    # ── Feature selection ─────────────────────────────────────────────────────
    log.info(f"\n▶  Feature selection (top {TOP_N_FEATURES} din {len(feature_cols)}) ...")
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    _spw = neg / max(pos, 1)
    _pre = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                              subsample=0.7, colsample_bytree=0.6, min_child_weight=25,
                              gamma=1.5, reg_alpha=2.0, reg_lambda=4.0,
                              scale_pos_weight=_spw, random_state=42, n_jobs=-1,
                              use_label_encoder=False, eval_metric='logloss', verbosity=0)
    _pre.fit(X_tr, y_tr, sample_weight=sw_, verbose=False)
    _imp = pd.Series(_pre.feature_importances_, index=feature_cols).sort_values(ascending=False)
    selected_features = _imp.head(TOP_N_FEATURES).index.tolist()
    log.info(f"   Selectate {len(selected_features)} | top5: {selected_features[:5]}")
    X_tr = X_tr[selected_features]
    X_te = X_te.reindex(columns=selected_features, fill_value=0)
    feature_cols = selected_features

    # ── Walk-forward CV folds ──────────────────────────────────────────────────
    ts_tr = pd.DatetimeIndex(pd.to_datetime(df_tr['_date']))
    y_tr_arr = y_tr.values

    def make_wf_folds(dates, n_folds, min_train_m=8, val_m=4):
        min_d = dates.min()
        folds = []
        for i in range(n_folds):
            tr_end = min_d + pd.DateOffset(months=min_train_m + i*val_m)
            vl_end = tr_end + pd.DateOffset(months=val_m)
            tm = np.array(dates < tr_end)
            vm = np.array((dates >= tr_end) & (dates < vl_end))
            if tm.sum() >= 30 and vm.sum() >= 15:
                folds.append((tm, vm))
        log.info(f"   Walk-forward: {len(folds)} folds")
        return folds

    wf_folds = make_wf_folds(ts_tr, N_WF_FOLDS)
    if not wf_folds:  # fallback la split simplu dacă date puține
        val_cut = int(len(X_tr) * 0.80)
        wf_folds = [(np.array([True]*val_cut + [False]*(len(X_tr)-val_cut)),
                     np.array([False]*val_cut + [True]*(len(X_tr)-val_cut)))]
        log.warning("   Fallback la split 80/20 (date insuficiente pentru WF)")

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 150, 800),
            'max_depth':        trial.suggest_int('max_depth', 2, 3),       # max 3 → anti-overfit dur
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.35, 0.75),
            'min_child_weight': trial.suggest_int('min_child_weight', 20, 80),  # ridicat → anti-overfit
            'gamma':            trial.suggest_float('gamma', 1.0, 8.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1.0, 8.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 3.0, 10.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 3.0, 12.0),
        }
        smote_ratio = trial.suggest_float('smote_ratio', 0.10, 0.40)
        fold_aucs = []
        for tm, vm in wf_folds:
            Xf = X_tr[tm]; yf = y_tr_arr[tm]; swf = sw_[tm]
            Xv = X_tr[vm]; yv = y_tr_arr[vm]
            try:
                sm = BorderlineSMOTE(sampling_strategy=smote_ratio, random_state=42, k_neighbors=5)
                Xs, ys = sm.fit_resample(Xf, yf)
                sws = np.concatenate([swf, np.ones(len(Xs)-len(Xf))])
            except Exception:
                Xs, ys, sws = Xf, yf, swf
            m = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss',
                                   random_state=42, n_jobs=-1, tree_method='hist',
                                   early_stopping_rounds=30)
            m.fit(Xs, ys, sample_weight=sws, eval_set=[(Xv, yv)], verbose=False)
            if yv.sum() > 0 and yv.sum() < len(yv):
                fold_aucs.append(roc_auc_score(yv, m.predict_proba(Xv)[:,1]))
        return float(np.mean(fold_aucs)) if fold_aucs else 0.5

    log.info(f"\n▶  Optuna tuning ({OPTUNA_TRIALS} trials) ...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False, n_jobs=1)
    bp = study.best_params
    smote_ratio_best = bp.pop('smote_ratio')
    log.info(f"   Best val AUC: {study.best_value:.4f} | params: {bp}")

    # Final training on full IS
    log.info("\n▶  Final training (full IS) ...")
    try:
        sm = BorderlineSMOTE(sampling_strategy=smote_ratio_best, random_state=42, k_neighbors=5)
        X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
        n_synth = len(X_sm) - len(X_tr)
        sw_sm = np.concatenate([sw_, np.ones(n_synth)])
    except Exception:
        X_sm, y_sm = X_tr, y_tr; sw_sm = sw_
    model = xgb.XGBClassifier(**bp, use_label_encoder=False, eval_metric='logloss',
                               random_state=42, n_jobs=-1, tree_method='hist')
    model.fit(X_sm, y_sm, sample_weight=sw_sm, verbose=False)

    is_auc = roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1])
    log.info(f"   IS AUC = {is_auc:.4f}")

    te_auc = 0.0
    if len(df_te) > 20:
        te_proba = model.predict_proba(X_te)[:, 1]
        te_auc   = roc_auc_score(y_te, te_proba)
        log.info(f"   OOS AUC = {te_auc:.4f}")
        for thr in [0.55, 0.60, 0.65, 0.70]:
            mask = te_proba >= thr
            if mask.sum() > 5:
                wr = float(y_te[mask].mean())
                log.info(f"   threshold={thr}: {int(mask.sum())} setups, WR={wr:.1%}")

    try:
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        log.info(f"\nTop 15 features:\n{imp.head(15).to_string()}")
    except Exception:
        pass

    # Salvare — dacă OOS nou e mai bun
    old_auc = 0.6212
    old_model = None
    if OUT.exists():
        try:
            old_pkg = pickle.load(open(OUT, 'rb'))
            old_auc = old_pkg.get('oos_auc', 0.0)
            old_model = old_pkg
        except Exception:
            pass

    if te_auc >= old_auc - 0.005:  # salvăm dacă nu e mai rău cu >0.5%
        pkg = {
            'model':            model,
            'features':         feature_cols,
            'is_auc':           round(is_auc, 4),
            'oos_auc':          round(te_auc, 4),
            'n_features':       len(feature_cols),
            'train_years':      TRAIN_YEARS,
            'test_years':       TEST_YEARS,
            'version':          'v1.2_enhanced_mtf_optuna',
            'label_tp_mult':     TP_MULT,
            'label_window_min': LABEL_WINDOW,
            'optuna_trials':    OPTUNA_TRIALS,
        }
        with open(OUT, 'wb') as f:
            pickle.dump(pkg, f)
        log.info(f"\n💾 Salvat: {OUT}")
        log.info(f"   IS AUC={is_auc:.4f} | OOS AUC={te_auc:.4f} (was {old_auc:.4f})")
    else:
        log.warning(f"\n⚠️  OOS regresie ({te_auc:.4f} < {old_auc:.4f} - 0.005) — model vechi păstrat")

    df_tr.drop(columns=meta_cols, errors='ignore').to_pickle(Path(__file__).parent / "lom_dataset_train.pkl")
    df_te.drop(columns=meta_cols, errors='ignore').to_pickle(Path(__file__).parent / "lom_dataset_test.pkl")
    log.info("   Datasets salvate.")


if __name__ == "__main__":
    train_and_save()
