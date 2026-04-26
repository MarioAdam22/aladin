"""
train_nom.py — NY Open Manipulation (NOM) — Training Script v1.1 EVENT-DRIVEN
==============================================================================
Detectează sweep-uri față de pre-NY range ORICÂND în sesiunea NY (09:00-13:00 ET).
Logica de detecție este IDENTICĂ cu nom_checker_v1.py v1.1.

Label: după sweep + displacement, prețul face TP_PT puncte în 60 min → 1, altfel → 0

RULARE:
  cd ~/Desktop/Aladin && python train_nom.py

Output:
  nom_model_v1.pkl  — înlocuiește modelul vechi (cu AUC valid event-driven)
  nom_dataset_train.pkl / nom_dataset_test.pkl — pentru debugging
"""

import sqlite3, pickle, logging, json
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from sklearn.metrics import roc_auc_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("NOM_TRAIN")

DB   = Path(__file__).parent / "mario_trading.db"
OUT  = Path(__file__).parent / "nom_model_v1.pkl"

# ── Economic Calendar ─────────────────────────────────────────────────────────
_CAL_PATH = Path(__file__).parent / "data" / "economic_calendar.json"
try:
    _cal = json.loads(_CAL_PATH.read_text())
    FOMC_DATES   = set(_cal.get('fomc',   []))
    NFP_DATES    = set(_cal.get('nfp',    []))
    CPI_DATES    = set(_cal.get('cpi',    []))
    PPI_DATES    = set(_cal.get('ppi',    []))
    RETAIL_DATES = set(_cal.get('retail', []))
    ISM_DATES    = set(_cal.get('ism',    []))
    ANY_HIGH     = set(_cal.get('any_high', []))
    NEWS_DAYS    = FOMC_DATES | NFP_DATES | CPI_DATES | PPI_DATES
    log.info(f"Calendar: NFP={len(NFP_DATES)}, FOMC={len(FOMC_DATES)}, CPI={len(CPI_DATES)}")
except Exception as _e:
    log.warning(f"Calendar load error: {_e} — calendar features vor fi 0")
    FOMC_DATES = NFP_DATES = CPI_DATES = PPI_DATES = RETAIL_DATES = ISM_DATES = ANY_HIGH = NEWS_DAYS = set()

# ── Parametri (identici cu nom_checker_v1.py) ─────────────────────────────────
MIN_SPIKE_PT  = 5.0
MIN_DISP_PT   = 4.0
TP_PT         = 24.0   # target: 24 puncte (NOM are mai mult spațiu = 2R pe 12pt SL)
LABEL_WINDOW  = 60     # minute forward pentru label
PARTIAL_THRESH_PCT = 0.30  # partial sweep: 30% din pre_range

TRAIN_YEARS   = [2023, 2024]   # IS: 2023-2024
TEST_YEARS    = [2025, 2026]   # OOS: 2025-2026

# ── Sesiune NY (ET — timestamps DB sunt ET) ───────────────────────────────────
NY_SESS_START_ET = 900    # 09:00 ET
NY_SESS_END_ET   = 1300   # 13:00 ET
PRE_NY_END_ET    = 859    # tot înainte de 09:00 ET
LON_START_ET     = 400
LON_END_ET       = 630


def sv(v, d=0.0):
    try: x = float(v); return x if np.isfinite(x) else d
    except: return d


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
               lon_hi, lon_lo, day_of_week, month
        FROM market_data
        WHERE date = '{date_str}'
        ORDER BY timestamp
    """, conn)
    if len(df) < 30:
        return None
    df['ts']   = pd.to_datetime(df['timestamp'])
    df['hhmm'] = df['ts'].dt.hour * 100 + df['ts'].dt.minute
    return df


def extract_setups(df, date_str=''):
    """
    Detectare event-driven identică cu nom_checker_v1.py v1.1.
    Scanează toată sesiunea NY (09:00-13:00 ET) pentru sweep față de pre-NY range.
    """
    setups = []

    pre_ny   = df[df['hhmm'] <= PRE_NY_END_ET]
    ny_sess  = df[df['hhmm'].between(NY_SESS_START_ET, NY_SESS_END_ET)]
    london   = df[df['hhmm'].between(LON_START_ET, LON_END_ET)]

    if len(pre_ny) < 20 or len(ny_sess) < 5:
        return setups

    pre_hi  = float(pre_ny['high'].max())
    pre_lo  = float(pre_ny['low'].min())
    pre_rng = pre_hi - pre_lo
    if pre_rng < 5:
        return setups

    atr = float(df['atr_14'].replace(0, np.nan).dropna().iloc[-1]) if len(df) > 0 else 10.0
    if atr <= 0: atr = 10.0

    # London context
    if len(london) > 0:
        lon_hi   = float(london['high'].max())
        lon_lo   = float(london['low'].min())
        lon_rng  = lon_hi - lon_lo
        lon_mid  = (lon_hi + lon_lo) / 2
        lon_close = float(london['close'].iloc[-1])
    else:
        lon_hi = pre_hi; lon_lo = pre_lo
        lon_rng = pre_rng; lon_mid = (pre_hi + pre_lo) / 2; lon_close = lon_mid

    # Partial sweep mai strict decât checker-ul (50% din pre_range, nu 30%)
    partial_thresh = pre_rng * 0.50
    ny_reset = ny_sess.reset_index(drop=False)
    last_setup_hhmm = {'LONG': -999, 'SHORT': -999}  # cooldown 30min între setups

    # Scanăm bara cu bara pentru sweep
    for i in range(1, len(ny_reset) - 2):
        bar      = ny_reset.iloc[i]
        bar_hi   = sv(bar['high'])
        bar_lo   = sv(bar['low'])
        bar_hhmm = int(bar['hhmm'])

        sweep_up = bar_hi - pre_hi
        sweep_dn = pre_lo - bar_lo

        for direction, spike_mag_raw, is_valid in [
            ('SHORT', max(sweep_up, 0),
             sweep_up >= MIN_SPIKE_PT or (sweep_up > 0 and sweep_up >= partial_thresh)),
            ('LONG',  max(sweep_dn, 0),
             sweep_dn >= MIN_SPIKE_PT or (sweep_dn > 0 and sweep_dn >= partial_thresh)),
        ]:
            if not is_valid or (bar_hhmm - last_setup_hhmm[direction]) < 30:
                continue

            spike_mag = spike_mag_raw
            spike_hi  = bar_hi; spike_lo = bar_lo

            # Displacement: 45min după spike (identic cu checker)
            after_spike = ny_reset[ny_reset['hhmm'].between(bar_hhmm + 1, bar_hhmm + 45)]
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
            dir_num     = 1 if direction == 'LONG' else -1

            # ── Label (forward 60min) ─────────────────────────────────────────
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

            # ── Feature engineering (identic cu nom_checker_v1.py) ───────────
            r0  = df.iloc[-1]
            after_early = ny_reset[ny_reset['hhmm'].between(bar_hhmm + 1, bar_hhmm + 45)]
            spike_bar_range = max(sv(bar['high'] - bar['low']), 0.01)

            if direction == 'SHORT':
                ts_close_inside  = 1 if sv(bar['close']) <= pre_hi else 0
                wick             = (sv(bar['high']) - max(sv(bar['close']), sv(bar['open']))) / atr
                ts_rejection_str = (spike_hi - sv(bar['close'])) / spike_mag if spike_mag > 0 else 0
                ts_wick_pct      = (spike_hi - sv(bar['close'])) / spike_bar_range
                ts_body_pct      = abs(sv(bar['open']) - sv(bar['close'])) / spike_bar_range
                ts_close_quality = max(0, (pre_hi - sv(bar['close'])) / pre_rng) if pre_rng > 0 else 0
            else:
                ts_close_inside  = 1 if sv(bar['close']) >= pre_lo else 0
                wick             = (min(sv(bar['close']), sv(bar['open'])) - sv(bar['low'])) / atr
                ts_rejection_str = (sv(bar['close']) - spike_lo) / spike_mag if spike_mag > 0 else 0
                ts_wick_pct      = (sv(bar['close']) - spike_lo) / spike_bar_range
                ts_body_pct      = abs(sv(bar['open']) - sv(bar['close'])) / spike_bar_range
                ts_close_quality = max(0, (sv(bar['close']) - pre_lo) / pre_rng) if pre_rng > 0 else 0

            wick_pct         = wick * atr / spike_bar_range
            sweep_wick_clean = 1 if wick_pct > 0.5 else 0
            sweep_depth_atr  = spike_mag / atr
            deep_sweep       = 1 if sweep_depth_atr > 1.5 else 0
            sweep_quality    = ts_close_inside*0.4 + sweep_wick_clean*0.3 + deep_sweep*0.2 + 0.1

            disp_body = abs(sv(disp_bar['close']) - sv(disp_bar['open']))

            # HTF
            h4_hi = sv(r0['h4_hi']); h4_lo = sv(r0['h4_lo'])
            h1_hi = sv(r0['h1_hi']); h1_lo = sv(r0['h1_lo'])
            h4_mid = (h4_hi + h4_lo) / 2 if h4_hi > 0 and h4_lo > 0 else 0
            h1_mid = (h1_hi + h1_lo) / 2 if h1_hi > 0 and h1_lo > 0 else 0
            h4_bias = 1 if entry_price < h4_mid else (-1 if h4_mid > 0 else 0)
            h1_bias = 1 if entry_price < h1_mid else (-1 if h1_mid > 0 else 0)

            lw_hi = sv(r0['lw_hi']); lw_lo = sv(r0['lw_lo']); lw_rng = lw_hi - lw_lo
            weekly_prem = (entry_price - lw_lo) / lw_rng if lw_rng > 0 else 0.5

            asia_hi_v = sv(r0.get('asia_hi', 0)); asia_lo_v = sv(r0.get('asia_lo', 0))
            asia_rng  = asia_hi_v - asia_lo_v

            eq_tol = atr * 0.3
            pre_highs = pre_ny['high'].values; pre_lows = pre_ny['low'].values
            eq_hi = max(0, sum(1 for h in pre_highs if abs(h - pre_hi) <= eq_tol) - 1)
            eq_lo = max(0, sum(1 for l in pre_lows  if abs(l - pre_lo) <= eq_tol) - 1)

            ny15 = ny_sess[ny_sess['hhmm'].between(NY_SESS_START_ET, NY_SESS_START_ET + 15)]
            ny15_rng = float(ny15['high'].max() - ny15['low'].min()) if len(ny15) > 0 else 0
            ny_open_price = float(ny_sess['open'].iloc[0]) if len(ny_sess) > 0 else entry_price

            pre_vol   = float(pre_ny['volume'].mean()) if len(pre_ny) > 0 else 1.0
            spike_vol = float(ny_sess['volume'].mean()) if len(ny_sess) > 0 else 1.0
            vol_ratio = spike_vol / pre_vol if pre_vol > 0 else 1.0

            spike_delta = sv(ny_sess['bar_delta'].sum()) if len(ny_sess) > 0 else 0
            fvg_up_v   = int(ny_sess['fvg_up'].any())   if 'fvg_up'   in ny_sess.columns else 0
            fvg_down_v = int(ny_sess['fvg_down'].any()) if 'fvg_down' in ny_sess.columns else 0

            adx_v   = sv(r0['adx_14'])
            hurst_v = sv(r0['hurst'], 0.5)

            feat = {
                # Spike
                'spike_mag':            spike_mag,
                'spike_mag_atr':        spike_mag / atr,
                'spike_vs_range':       spike_mag / pre_rng if pre_rng > 0 else 0,
                'pre_rng_atr':          pre_rng / atr,
                # TS anti-fakeout
                'ts_close_inside':      ts_close_inside,
                'ts_rejection_str':     ts_rejection_str,
                'ts_wick_pct':          ts_wick_pct,
                'ts_body_pct':          ts_body_pct,
                'ts_close_quality':     ts_close_quality,
                'ts_wick_dom':          1 if ts_wick_pct > 0.6 else 0,
                'ts_htf_anti':          0,
                'ts_combo_score':       ts_close_inside * ts_rejection_str,
                'ts_sweep_depth_pts':   spike_mag,
                'ts_sweep_depth_atr':   sweep_depth_atr,
                'ts_sweep_pct_lon':     spike_mag / lon_rng if lon_rng > 0 else 0,
                'ts_lon_mid_dist':      (entry_price - lon_mid) / atr,
                'ts_entry_prox':        abs(entry_price - lon_mid) / lon_rng if lon_rng > 0 else 0,
                # Sweep quality
                'sweep_wick_atr':       wick,
                'sweep_wick_pct':       wick_pct,
                'sweep_wick_clean':     sweep_wick_clean,
                'sweep_depth_atr':      sweep_depth_atr,
                'deep_sweep':           deep_sweep,
                'shallow_sweep':        1 if sweep_depth_atr < 0.5 else 0,
                'sweep_with_disp':      1,
                'sweep_quality_score':  sweep_quality,
                'equal_level_score':    (eq_hi if direction == 'SHORT' else eq_lo) / max(len(pre_ny), 1),
                # Displacement
                'disp_body':            disp_body,
                'disp_body_atr':        disp_body / atr,
                'disp_range':           sv(disp_bar['high'] - disp_bar['low']),
                'disp_wick_ratio':      (sv(disp_bar['high'] - disp_bar['low']) - disp_body) / max(disp_body, 0.01),
                'has_disp':             1,
                'body_pct':             disp_body / max(sv(disp_bar['high'] - disp_bar['low']), 0.01),
                'body_bear':            1 if direction == 'SHORT' else 0,
                # HTF bias
                'h4_bias':              h4_bias,
                'h1_bias':              h1_bias,
                'h4_h1_aligned':        1 if h4_bias == h1_bias and h4_bias != 0 else 0,
                'h4_bias_aligned':      1 if h4_bias == dir_num else 0,
                # Weekly context
                'weekly_premium_pct':   weekly_prem,
                'in_weekly_premium':    1 if weekly_prem > 0.5 else 0,
                'in_weekly_discount':   1 if weekly_prem < 0.5 else 0,
                'weekly_prem_aligned':  1 if (direction == 'SHORT' and weekly_prem > 0.5) or (direction == 'LONG' and weekly_prem < 0.5) else 0,
                'h4_x_weekly':          (1 if h4_bias == dir_num else 0) * (1 if (direction == 'SHORT' and weekly_prem > 0.5) or (direction == 'LONG' and weekly_prem < 0.5) else 0),
                'lw_range_atr':         lw_rng / atr if atr > 0 else 0,
                'week_range_so_far':    (df['high'].max() - df['low'].min()) / atr if atr > 0 else 0,
                'dist_prev_wk_lo':      abs(entry_price - lw_lo) / atr,
                'dist_lw_hi':           abs(entry_price - lw_hi) / atr,
                'dist_lw_lo':           abs(entry_price - lw_lo) / atr,
                # London context (NOM-specific)
                'lon_range_atr':        lon_rng / atr,
                'dist_lon_hi_atr':      abs(entry_price - lon_hi) / atr,
                'dist_lon_lo_atr':      abs(entry_price - lon_lo) / atr,
                'lon_close_vs_mid':     (lon_close - lon_mid) / lon_rng if lon_rng > 0 else 0,
                'ny15_range_atr':       ny15_rng / atr,
                'gap_vs_lon_close_atr': (ny_open_price - lon_close) / atr,
                # Asia context
                'dist_asia_hi_atr':     abs(entry_price - asia_hi_v) / atr if asia_hi_v > 0 else 0,
                'dist_asia_lo_atr':     abs(entry_price - asia_lo_v) / atr if asia_lo_v > 0 else 0,
                'asia_range_atr':       asia_rng / atr if asia_rng > 0 else 0,
                'spike_vs_asia_hi':     (spike_hi - asia_hi_v) / atr if asia_hi_v > 0 else 0,
                'spike_vs_asia_lo':     (asia_lo_v - spike_lo) / atr if asia_lo_v > 0 else 0,
                # True open / PDH / PDL
                'above_true_open':      1 if entry_price > sv(r0['true_open']) else 0,
                'dist_true_open':       abs(entry_price - sv(r0['true_open'])) / atr,
                'dist_pdh_atr':         abs(entry_price - sv(r0['p_hi'])) / atr,
                'dist_pdl_atr':         abs(entry_price - sv(r0['p_lo'])) / atr,
                # VA / POC
                'inside_va':            sv(r0['inside_va']),
                'dist_poc_entry':       sv(r0['dist_poc']) / atr,
                'entry_in_pre_range':   int(pre_lo <= entry_price <= pre_hi),
                'dist_poc_atr':         sv(r0['dist_poc']) / atr,
                'dist_vwap_atr':        sv(r0['dist_vwap']) / atr,
                # Volume / delta
                'vol_ratio':            vol_ratio,
                'spike_delta':          spike_delta,
                'disp_delta':           sv(after_early['bar_delta'].sum()) if len(after_early) > 0 else 0,
                'delta_at_high':        sv(ny_sess['delta_at_high'].sum()) if 'delta_at_high' in ny_sess.columns else 0,
                'delta_at_low':         sv(ny_sess['delta_at_low'].sum()) if 'delta_at_low' in ny_sess.columns else 0,
                'big_buy':              1 if vol_ratio > 2 and direction == 'LONG' else 0,
                'big_sell':             1 if vol_ratio > 2 and direction == 'SHORT' else 0,
                'big_imbalance':        1 if vol_ratio > 2 else 0,
                'absorption':           sv(ny_sess['absorption_score'].mean()) if 'absorption_score' in ny_sess.columns else 0,
                'bar_delta_norm':       spike_delta / atr,
                'cum_delta_norm':       sv(disp_bar.get('cum_delta', 0)) / atr,
                'buy_sell_ratio':       sv(ny_sess['bar_buy_vol'].sum()) / max(sv(ny_sess['bar_sell_vol'].sum()), 1),
                'of_doi':               sv(ny_sess['of_doi'].mean()) if 'of_doi' in ny_sess.columns else 0,
                'stacked_bull':         int(ny_sess['stacked_bull'].any()) if 'stacked_bull' in ny_sess.columns else 0,
                'stacked_bear':         int(ny_sess['stacked_bear'].any()) if 'stacked_bear' in ny_sess.columns else 0,
                # FVG
                'htf_fvg_aligned':      1 if (direction == 'SHORT' and fvg_down_v) or (direction == 'LONG' and fvg_up_v) else 0,
                'ob_proxy_bull':        int(ny_sess['stacked_bull'].any()) if 'stacked_bull' in ny_sess.columns else 0,
                'ob_proxy_bear':        int(ny_sess['stacked_bear'].any()) if 'stacked_bear' in ny_sess.columns else 0,
                'ob_aligned':           0,
                'vol_x_fvg_1h':         vol_ratio * (1 if (direction == 'LONG' and fvg_up_v) or (direction == 'SHORT' and fvg_down_v) else 0),
                # Technical
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
                'vix_proxy_20d':        float(df['close'].pct_change().rolling(20).std().iloc[-1]) if len(df) > 20 else 0,
                'atr_vs_10d':           atr / max(sv(r0['lw_hi']) - sv(r0['lw_lo']), 0.01) * 10,
                # Time
                'day_of_week':          sv(r0['day_of_week']),
                'month':                sv(r0['month']),
                # Direction
                'direction_enc':        1 if direction == 'SHORT' else 0,
                # Economic Calendar (la fel ca în ny_v3/ts_ny)
                'is_nfp_day':           1 if date_str in NFP_DATES    else 0,
                'is_fomc_day':          1 if date_str in FOMC_DATES   else 0,
                'is_cpi_day':           1 if date_str in CPI_DATES    else 0,
                'is_ppi_day':           1 if date_str in PPI_DATES    else 0,
                'is_retail_day':        1 if date_str in RETAIL_DATES else 0,
                'is_ism_day':           1 if date_str in ISM_DATES    else 0,
                'is_any_high_day':      1 if date_str in ANY_HIGH     else 0,
                'is_news_day':          1 if date_str in NEWS_DAYS    else 0,
                # Pre/post release timing (ET timestamps — NFP/CPI/PPI la 8:30 ET = hhmm 830)
                # NOM sesiune 09:00-13:00 ET → FOMC la 14:00 ET e DUPĂ toată sesiunea
                'is_pre_nfp':           1 if (date_str in NFP_DATES   and entry_hhmm < 830) else 0,
                'is_post_nfp':          1 if (date_str in NFP_DATES   and entry_hhmm >= 830) else 0,
                'is_pre_cpi':           1 if (date_str in CPI_DATES   and entry_hhmm < 830) else 0,
                'is_post_cpi':          1 if (date_str in CPI_DATES   and entry_hhmm >= 830) else 0,
                'is_pre_ppi':           1 if (date_str in PPI_DATES   and entry_hhmm < 830) else 0,
                'is_post_ppi':          1 if (date_str in PPI_DATES   and entry_hhmm >= 830) else 0,
                'is_fomc_wait':         1 if date_str in FOMC_DATES   else 0,   # FOMC la 14:00 ET, după sesiunea NOM
                # LON direction explicit (era lipsă, acum adăugat — același concept ca în v6)
                'lon_dir_explicit':     1 if lon_close > lon_mid else -1,
                'lon_dir_aligned':      1 if (dir_num == 1 and lon_close > lon_mid) or (dir_num == -1 and lon_close <= lon_mid) else 0,
                'lon_dir_opposite':     1 if (dir_num == 1 and lon_close <= lon_mid) or (dir_num == -1 and lon_close > lon_mid) else 0,
                'lon_big_day':          1 if lon_rng > atr * 1.5 else 0,
                'lon_small_day':        1 if lon_rng < atr * 0.7 else 0,
                'ny_open_dist_lon_mid': (ny_open_price - lon_mid) / atr if atr > 0 else 0,
                # Interactions calendar × LON (identic cu ny_v3)
                'nfp_post_x_open':      (1 if (date_str in NFP_DATES and entry_hhmm >= 830) else 0),
                'lon_dir_x_vol':        (1 if lon_close > lon_mid else -1) * vol_ratio / 3.0,
                # Meta
                '_label':      label,
                '_direction':  direction,
                '_date':       str(date_str),
                '_entry_px':   entry_price,
                '_max_fwd':    max_fwd,
                '_entry_hhmm': entry_hhmm,
            }
            setups.append(feat)
            last_setup_hhmm[direction] = bar_hhmm
            break  # trece la bara următoare
    return setups


def build_dataset(years):
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    days = pd.read_sql(f"""
        SELECT DISTINCT date FROM market_data
        WHERE year IN ({','.join(map(str, years))})
          AND day_of_week BETWEEN 1 AND 5
        ORDER BY date
    """, conn)['date'].tolist()
    all_setups = []
    for date_str in days:
        df = load_day(conn, date_str)
        if df is None:
            continue
        setups = extract_setups(df, date_str)
        all_setups.extend(setups)
    conn.close()
    log.info(f"  {years}: {len(days)} zile → {len(all_setups)} setups")
    return pd.DataFrame(all_setups)


def train_and_save():
    log.info("═" * 60)
    log.info("NOM TRAIN v1.1 — Event-Driven")
    log.info("═" * 60)

    log.info(f"Extrag IS ({TRAIN_YEARS})...")
    df_tr = build_dataset(TRAIN_YEARS)
    log.info(f"Extrag OOS ({TEST_YEARS})...")
    df_te = build_dataset(TEST_YEARS)

    meta_cols = [c for c in df_tr.columns if c.startswith('_')]
    features  = [c for c in df_tr.columns if not c.startswith('_')]

    log.info(f"\nIS: {len(df_tr)} setups | label dist: {df_tr['_label'].value_counts().to_dict()}")
    log.info(f"OOS: {len(df_te)} setups | label dist: {df_te['_label'].value_counts().to_dict()}")
    log.info(f"Direcție IS: {df_tr['_direction'].value_counts().to_dict()}")

    if len(df_tr) < 50:
        log.error("Prea puțin data IS — verifică DB path și ani")
        return

    X_tr = df_tr[features].fillna(0)
    y_tr = df_tr['_label']
    X_te = df_te[features].fillna(0).reindex(columns=features, fill_value=0)
    y_te = df_te['_label']

    # Validare time-based: ultimele 20% din IS (cronologic) → early stopping
    val_cut  = int(len(X_tr) * 0.80)
    X_val_es = X_tr.iloc[val_cut:]
    y_val_es = y_tr.iloc[val_cut:]
    X_tr_es  = X_tr.iloc[:val_cut]
    y_tr_es  = y_tr.iloc[:val_cut]

    neg, pos = (y_tr_es == 0).sum(), (y_tr_es == 1).sum()
    scale_pos = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=20,
        gamma=1.0,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos,
        eval_metric='auc',
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_tr_es, y_tr_es,
        eval_set=[(X_val_es, y_val_es)],
        verbose=False,
    )
    log.info(f"Best iteration: {model.best_iteration} | val AUC={model.best_score:.4f}")

    is_proba = model.predict_proba(X_tr)[:, 1]
    is_auc   = roc_auc_score(y_tr, is_proba)
    log.info(f"✅ IS AUC  = {is_auc:.4f}")

    te_auc = 0.0
    if len(df_te) > 20:
        te_proba = model.predict_proba(X_te)[:, 1]
        te_auc   = roc_auc_score(y_te, te_proba)
        log.info(f"✅ OOS AUC = {te_auc:.4f}")
        for thr in [0.55, 0.60, 0.65, 0.70]:
            mask = te_proba >= thr
            if mask.sum() > 0:
                wr = float(y_te[mask].mean())
                log.info(f"   threshold={thr}: {int(mask.sum())} setups, WR={wr:.1%}")
    else:
        log.warning("OOS dataset prea mic pentru evaluare")

    try:
        imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        log.info(f"\nTop 10 features:\n{imp.head(10).to_string()}")
    except Exception:
        pass

    pkg = {
        'model':            model,
        'features':         features,
        'is_auc':           round(is_auc, 4),
        'oos_auc':          round(te_auc, 4),
        'n_features':       len(features),
        'train_years':      TRAIN_YEARS,
        'test_years':       TEST_YEARS,
        'version':          'v1.1_event_driven',
        'label_tp_pt':      TP_PT,
        'label_window_min': LABEL_WINDOW,
    }
    with open(OUT, 'wb') as f:
        pickle.dump(pkg, f)
    log.info(f"\n💾 Salvat: {OUT}")
    log.info(f"   IS AUC={is_auc:.4f} | OOS AUC={te_auc:.4f}")
    log.info(f"   → Actualizează docstring nom_checker_v1.py cu noile AUC-uri")

    # Salvare dataset pentru debugging
    df_tr.to_pickle(Path(__file__).parent / "nom_dataset_train.pkl")
    df_te.to_pickle(Path(__file__).parent / "nom_dataset_test.pkl")
    log.info("   Datasets salvate: nom_dataset_train.pkl / nom_dataset_test.pkl")


if __name__ == "__main__":
    train_and_save()
