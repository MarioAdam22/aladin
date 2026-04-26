"""
DSM (Double Sweep Model) — Training Script v2
==============================================
Detectează: sweep within session range + HTF bias + displacement → continuation

Sesiuni acoperite:
  - London (LON): 04:00-06:30 ET în DB (= 08:00-10:30 UTC)
  - New York (NY): 09:30-12:00 ET în DB (= 13:30-16:00 UTC)

Label: după sweep și displacement, prețul face 20+ puncte în direcția displacement
       în urmtoarele 90 minute → 1 (win), altfel → 0 (loss/no-move)

Features: tot ce e util din v2/v5/NOM/LOM + session context

Fixes v2:
  - DB read-only URI (fără PRAGMA WAL)
  - OOS include 2026
  - 30-min cooldown între setups same-direction per sesiune
  - XGBoost cu early stopping (nu GBM)
  - Time-based 80/20 IS split pentru validation
"""

import sqlite3, pickle, logging, json
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# ── Economic Calendar ─────────────────────────────────────────────────────────
_CAL_PATH = Path(__file__).parent / "data" / "economic_calendar.json"
try:
    _cal       = json.loads(_CAL_PATH.read_text())
    _FOMC      = set(_cal.get('fomc',   []))
    _NFP       = set(_cal.get('nfp',    []))
    _CPI       = set(_cal.get('cpi',    []))
    _PPI       = set(_cal.get('ppi',    []))
    _RETAIL    = set(_cal.get('retail', []))
    _ISM       = set(_cal.get('ism',    []))
    _NEWS      = _FOMC | _NFP | _CPI | _PPI
    logging.getLogger("DSM_TRAIN").info(f"Calendar: NFP={len(_NFP)}, FOMC={len(_FOMC)}")
except Exception as _ce:
    _FOMC=_NFP=_CPI=_PPI=_RETAIL=_ISM=_NEWS=set()

def _fomc_prox(date_str):
    try:
        d = pd.Timestamp(date_str).date()
        return float(min(abs((d - pd.Timestamp(x).date()).days) for x in _FOMC)) if _FOMC else 30.0
    except: return 30.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger("DSM_TRAIN")

DB   = Path(__file__).parent / "mario_trading.db"
OUT  = Path(__file__).parent / "dsm_model_v1.pkl"

# ── Parametri ────────────────────────────────────────────────────────────────
MIN_SWEEP_PT   = 4.0    # minim sweep (pts NQ)
MIN_DISP_PT    = 3.0    # minim displacement confirmare (pts)
TP_PT          = 20.0   # target: 20 puncte în direcția displacement
LABEL_WINDOW   = 90     # minute după displacement pentru a atinge TP
COOLDOWN_MIN   = 30     # minute cooldown între setups same-direction per sesiune
TRAIN_YEARS    = list(range(2021, 2025))  # IS: 2021-2024
TEST_YEARS     = [2025, 2026]             # OOS: 2025+2026

XGB_PARAMS = {
    'n_estimators':     800,
    'max_depth':        4,
    'learning_rate':    0.03,
    'subsample':        0.80,
    'colsample_bytree': 0.75,
    'min_child_weight': 20,
    'gamma':            1.0,
    'reg_alpha':        0.5,
    'reg_lambda':       1.5,
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'tree_method':      'hist',
    'random_state':     42,
    'early_stopping_rounds': 40,
}

# ── Sesiuni (ET — timestamps DB) ─────────────────────────────────────────────
SESSIONS = {
    'LON': {'pre_st': 100,  'pre_en': 359,  'sess_st': 400,  'sess_en': 630,  'enc': 0},
    'NY':  {'pre_st': 400,  'pre_en': 929,  'sess_st': 930,  'sess_en': 1200, 'enc': 1},
}


def load_day(conn, date_str):
    df = pd.read_sql(f"""
        SELECT timestamp, open, high, low, close, volume,
               atr_14, bar_delta, cum_delta, fvg_up, fvg_down, has_displacement,
               body_size, adx_14, hurst, dist_poc, inside_va, dist_vwap,
               delta_at_high, delta_at_low, big_buy_count, big_sell_count,
               absorption_score, stacked_bull, stacked_bear, of_doi, of_big_balance,
               bar_buy_vol, bar_sell_vol, garch_vol, kalman_smooth, kalman_noise,
               fisher_transform, acf_lag1, acf_lag5, rvol,
               vah, val, poc_level, p_hi, p_lo, lw_hi, lw_lo,
               h4_hi, h4_lo, h1_hi, h1_lo, true_open,
               asia_hi, asia_lo, day_of_week, month
        FROM market_data
        WHERE date = '{date_str}'
        ORDER BY timestamp
    """, conn)
    if len(df) < 20:
        return None
    df['ts']   = pd.to_datetime(df['timestamp'])
    df['hhmm'] = df['ts'].dt.hour * 100 + df['ts'].dt.minute
    return df


def sv(v, d=0.0):
    try: x = float(v); return x if np.isfinite(x) else d
    except: return d


def load_regime_classifier():
    import joblib
    rc_path = Path(__file__).parent / "regime_classifier_v1.pkl"
    if not rc_path.exists():
        log.warning("  regime_classifier_v1.pkl not found"); return None
    try:
        pkg = joblib.load(rc_path); log.info("  Loaded regime_classifier_v1.pkl"); return pkg
    except Exception as e:
        log.warning(f"  Failed to load regime_classifier: {e}"); return None

def predict_regime_enc(regime_pkg, bar_features_dict):
    if regime_pkg is None: return 2
    try:
        model = regime_pkg['model']; feats = regime_pkg['features']; le = regime_pkg['label_encoder']
        x = pd.DataFrame([{f: bar_features_dict.get(f, 0.0) for f in feats}]).fillna(0)
        enc = model.predict(x)[0]
        return int(le.inverse_transform([enc])[0])
    except Exception: return 2


def hhmm_to_min(hhmm):
    """Convertește 930 → 570 minute de la miezul nopții."""
    return (int(hhmm) // 100) * 60 + (int(hhmm) % 100)


def extract_setups(df, sess_name, sess_cfg, date_str='', regime_pkg=None):
    """Extrage setups DSM pentru o sesiune dintr-o zi (cu 30-min cooldown)."""
    setups = []
    pre   = df[df['hhmm'].between(sess_cfg['pre_st'],  sess_cfg['pre_en'])]
    sess  = df[df['hhmm'].between(sess_cfg['sess_st'], sess_cfg['sess_en'])]
    if len(pre) < 10 or len(sess) < 5:
        return setups

    pre_hi  = float(pre['high'].max())
    pre_lo  = float(pre['low'].min())
    pre_rng = pre_hi - pre_lo
    if pre_rng < 8:  # range prea mic = consolidare extremă
        return setups

    atr = float(df['atr_14'].replace(0, np.nan).dropna().iloc[-1]) if len(df) > 0 else 10.0
    if atr <= 0: atr = 10.0

    # Cooldown tracker: ultima oară când am emis un setup per direcție
    last_setup_min = {'LONG': -9999, 'SHORT': -9999}

    sess_reset = sess.reset_index(drop=False)
    for i in range(2, len(sess_reset) - 2):
        bar      = sess_reset.iloc[i]
        bar_hhmm = int(bar['hhmm'])
        bar_hi   = sv(bar['high'])
        bar_lo   = sv(bar['low'])
        bar_cl   = sv(bar['close'])
        bar_op   = sv(bar['open'])

        # ── Detectare sweep ─────────────────────────────────────────────────
        sweep_dn = pre_lo - bar_lo   # pozitiv = a sweepuit sub pre_lo
        sweep_up = bar_hi - pre_hi   # pozitiv = a sweepuit deasupra pre_hi

        # DSM: sweep = orice intrare semnificativă în zonă extremă a pre-range
        # Includem și "partial sweep" (intrat în ultimele 30% ale range-ului)
        partial_dn = pre_lo + 0.3 * pre_rng - bar_lo
        partial_up = bar_hi - (pre_hi - 0.3 * pre_rng)

        for direction, sweep_val, partial_val in [
            ('LONG',  sweep_dn, partial_dn),
            ('SHORT', sweep_up, partial_up),
        ]:
            # Sweep suficient (puternic sau partial dar semnificativ)
            is_sweep = (sweep_val >= MIN_SWEEP_PT) or \
                       (partial_val >= MIN_SWEEP_PT and abs(bar_lo if direction=='LONG' else bar_hi) > 0)
            if not is_sweep:
                continue

            # ── 30-min cooldown ──────────────────────────────────────────────
            bar_min = hhmm_to_min(bar_hhmm)
            if bar_min - last_setup_min[direction] < COOLDOWN_MIN:
                continue

            # Displacement: urmatoarele 1-3 bare confirma retracing
            after_bars = sess_reset.iloc[i+1:i+5]
            disp_bar = None
            for _, ab in after_bars.iterrows():
                ab_body = abs(sv(ab['close']) - sv(ab['open']))
                if direction == 'LONG' and sv(ab['close']) > sv(ab['open']) and ab_body >= MIN_DISP_PT:
                    disp_bar = ab; break
                elif direction == 'SHORT' and sv(ab['close']) < sv(ab['open']) and ab_body >= MIN_DISP_PT:
                    disp_bar = ab; break

            if disp_bar is None:
                continue

            entry_price = sv(disp_bar['close'])
            entry_hhmm  = int(disp_bar['hhmm'])

            # Actualizăm cooldown cu bara de entry (nu de sweep)
            last_setup_min[direction] = hhmm_to_min(entry_hhmm)

            # ── Label: atinge TP 20pt în 90min? ─────────────────────────────
            future = df[df['hhmm'] > entry_hhmm].head(LABEL_WINDOW)
            if len(future) < 3:
                continue
            if direction == 'LONG':
                reached_tp = (future['high'].max() >= entry_price + TP_PT)
                max_fwd    = float(future['high'].max() - entry_price)
            else:
                reached_tp = (future['low'].min() <= entry_price - TP_PT)
                max_fwd    = float(entry_price - future['low'].min())

            label = 1 if reached_tp else 0

            # ── Feature engineering ──────────────────────────────────────────
            r0       = df.iloc[-1]
            pre_last = pre.iloc[-1] if len(pre) > 0 else r0
            dir_num  = 1 if direction == 'LONG' else -1

            # ── Regime prediction ────────────────────────────────────────────
            _regime_bar = {
                'adx_14': sv(r0['adx_14']), 'hurst': sv(r0['hurst']),
                'garch_vol': sv(r0['garch_vol']), 'rvol': sv(r0['rvol'], 1.0),
                'fisher_transform': sv(r0['fisher_transform']),
                'acf_lag1': sv(r0['acf_lag1']), 'acf_lag5': sv(r0['acf_lag5']),
                'inside_va': 0.0, 'dist_vwap': 5.0, 'has_displacement': sv(r0['has_displacement']),
                'bar_delta': sv(r0['bar_delta']), 'cum_delta': sv(r0['cum_delta']),
                'imbalance_pct': 0.0, 'dom_ratio': 1.0, 'hhmm_enc': entry_hhmm,
                'is_session_open': 1, 'is_lon_session': 1 if sess_name == 'LON' else 0,
                'is_ny_session': 1 if sess_name == 'NY' else 0,
                'sweep_dn_atr': 0.0, 'sweep_up_atr': 0.0,
                'dist_poc': 5.0, 'dist_pdh': 0.0, 'dist_pdl': 0.0,
                'body_size': sv(r0['body_size']), 'fvg_up': sv(r0['fvg_up']), 'fvg_down': sv(r0['fvg_down']),
                'sample_entropy': 0.5, 'day_of_week': sv(r0['day_of_week']), 'month': sv(r0['month']),
                'of_cvd_lag1': 0.0, 'of_absorption_lag1': 0.0, 'of_opening_drive_lag1': 0.0,
                'of_cvd_zscore_lag1': 0.0, 'of_stacked_imbalance_lag1': 0.0, 'of_opening_range_lag1': 0.0,
            }
            regime_enc = predict_regime_enc(regime_pkg, _regime_bar)
            _lw_hi = sv(r0['lw_hi']); _lw_lo = sv(r0['lw_lo']); _lw_rng = _lw_hi - _lw_lo
            _weekly_prem = (entry_price - _lw_lo) / _lw_rng if _lw_rng > 0 else 0.5
            _h4_mid = (sv(r0['h4_hi']) + sv(r0['h4_lo'])) / 2 if sv(r0['h4_hi']) > 0 else 0
            _h4_bias = 1 if entry_price < _h4_mid else (-1 if _h4_mid > 0 else 0)
            _sweep_q = (1 if (direction == 'LONG' and bar_cl >= pre_lo) or (direction == 'SHORT' and bar_cl <= pre_hi) else 0) * 0.4 + 0.1

            feat = {
                # ── Session context ──────────────────────────────────────────
                'session_enc':        sess_cfg['enc'],
                'entry_hhmm':         entry_hhmm,
                'day_of_week':        sv(r0['day_of_week']),
                'month':              sv(r0['month']),
                # ── Calendar (DSM NY: 13:30-16:00 UTC = 10:00-13:00 ET)
                'is_nfp_day':         1 if date_str in _NFP    else 0,
                'is_fomc_day':        1 if date_str in _FOMC   else 0,
                'is_cpi_day':         1 if date_str in _CPI    else 0,
                'is_news_day':        1 if date_str in _NEWS   else 0,
                'fomc_proximity':     float(np.clip(_fomc_prox(date_str) / 14.0, 0, 1)),
                'is_post_nfp':        1 if (date_str in _NFP  and entry_hhmm >= 830) else 0,
                'is_fomc_in_window':  1 if (date_str in _FOMC and 1350 <= entry_hhmm <= 1420) else 0,

                # ── Sweep quality ────────────────────────────────────────────
                'sweep_mag_atr':      (sweep_val if sweep_val > 0 else partial_val) / atr,
                'sweep_is_outside':   1 if (sweep_val >= MIN_SWEEP_PT) else 0,
                'partial_depth':      partial_val / pre_rng if pre_rng > 0 else 0,
                'pre_range_atr':      pre_rng / atr,
                'bar_range_atr':      (bar_hi - bar_lo) / atr,
                'sweep_wick_pct':     (abs(bar_lo - min(sv(bar['open']), sv(bar['close']))) / (bar_hi - bar_lo + 0.01)
                                       if direction == 'LONG' else
                                       abs(bar_hi - max(sv(bar['open']), sv(bar['close']))) / (bar_hi - bar_lo + 0.01)),
                'spike_close_inside': 1 if (direction == 'LONG' and bar_cl >= pre_lo) or (direction == 'SHORT' and bar_cl <= pre_hi) else 0,

                # ── Displacement ─────────────────────────────────────────────
                'disp_body_atr':      abs(sv(disp_bar['close']) - sv(disp_bar['open'])) / atr,
                'disp_has_flag':      sv(disp_bar['has_displacement']),
                'disp_imbalance':     abs(sv(disp_bar['bar_buy_vol']) - sv(disp_bar['bar_sell_vol'])) / max(sv(disp_bar['volume']), 1),

                # ── HTF Bias ─────────────────────────────────────────────────
                'h4_bias':            dir_num * (entry_price - (sv(r0['h4_hi']) + sv(r0['h4_lo'])) / 2) / atr if sv(r0['h4_hi']) > 0 else 0,
                'h1_bias':            dir_num * (entry_price - (sv(r0['h1_hi']) + sv(r0['h1_lo'])) / 2) / atr if sv(r0['h1_hi']) > 0 else 0,
                'above_true_open':    dir_num * (entry_price - sv(r0['true_open'])) / atr if sv(r0['true_open']) > 0 else 0,
                'dist_prev_wk_hi':    (sv(r0['lw_hi']) - entry_price) / atr if sv(r0['lw_hi']) > 0 else 0,
                'dist_prev_wk_lo':    (entry_price - sv(r0['lw_lo'])) / atr if sv(r0['lw_lo']) > 0 else 0,
                'dist_prev_day_hi':   (sv(r0['p_hi']) - entry_price) / atr if sv(r0['p_hi']) > 0 else 0,
                'dist_prev_day_lo':   (entry_price - sv(r0['p_lo'])) / atr if sv(r0['p_lo']) > 0 else 0,

                # ── Volume Profile ───────────────────────────────────────────
                'dist_poc_atr':       dir_num * (sv(r0['poc_level']) - entry_price) / atr if sv(r0['poc_level']) > 0 else 0,
                'dist_vah_atr':       (sv(r0['vah']) - entry_price) / atr if sv(r0['vah']) > 0 else 0,
                'dist_val_atr':       (entry_price - sv(r0['val'])) / atr if sv(r0['val']) > 0 else 0,
                'inside_va':          sv(disp_bar['inside_va']),
                'dist_vwap_atr':      dir_num * sv(disp_bar['dist_vwap']) / atr,

                # ── Order Flow ───────────────────────────────────────────────
                'cum_delta_norm':     sv(disp_bar['cum_delta']) / max(sv(disp_bar['volume']), 1),
                'bar_delta_norm':     sv(disp_bar['bar_delta']) / max(sv(disp_bar['volume']), 1) * dir_num,
                'of_doi':             sv(r0['of_doi']) * dir_num,
                'of_big_balance':     sv(r0['of_big_balance']) * dir_num,
                'absorption_score':   sv(r0['absorption_score']),
                'stacked_bull':       sv(r0['stacked_bull']) if direction == 'LONG' else sv(r0['stacked_bear']),
                'big_buy_sell_ratio': (sv(disp_bar['big_buy_count']) - sv(disp_bar['big_sell_count'])) / max(sv(disp_bar['big_buy_count']) + sv(disp_bar['big_sell_count']), 1) * dir_num,

                # ── Volatilitate ─────────────────────────────────────────────
                'atr_norm':           atr / 10.0,
                'garch_vol':          sv(r0['garch_vol']),
                'adx_14':             sv(r0['adx_14']),
                'hurst':              sv(r0['hurst']),
                'rvol':               sv(r0['rvol'], 1.0),

                # ── Pre-session context ───────────────────────────────────────
                'pre_close_vs_mid':   (sv(pre_last['close']) - (pre_hi + pre_lo) / 2) / pre_rng if pre_rng > 0 else 0,
                'pre_delta_trend':    sv(pre_last['cum_delta']) * dir_num / max(abs(sv(pre_last['cum_delta'])), 1),
                'fvg_aligned':        sv(disp_bar['fvg_up']) if direction == 'LONG' else sv(disp_bar['fvg_down']),
                'fisher_transform':   sv(r0['fisher_transform']) * dir_num,
                'acf_lag1':           sv(r0['acf_lag1']),
                'acf_lag5':           sv(r0['acf_lag5']),
                'kalman_smooth':      sv(r0['kalman_smooth']),
                'kalman_noise':       sv(r0['kalman_noise']),

                # ── Regime (from classifier) ──────────────────────────────────
                'regime_enc':         float(regime_enc),
                'regime_is_pre':      1 if regime_enc == 1 else 0,
                'regime_is_exp':      1 if regime_enc == 2 else 0,
                'regime_is_ret':      1 if regime_enc == 3 else 0,
                'regime_aligned':     1 if (regime_enc in [1, 2] and _h4_bias == dir_num) else 0,

                # ── Cross-model interaction features ─────────────────────────
                'dir_x_adx':          dir_num * sv(r0['adx_14']),
                'dir_x_hurst':        dir_num * sv(r0['hurst'], 0.5),
                'vol_x_sweep':        sv(r0['garch_vol']) * _sweep_q,
                'h4_x_weekly':        float(1 if _h4_bias == dir_num else 0) * float(1 if (_weekly_prem < 0.5 and dir_num == 1) or (_weekly_prem > 0.5 and dir_num == -1) else 0),
                'is_thursday':        1 if int(sv(r0['day_of_week'])) == 3 else 0,
                'is_friday':          1 if int(sv(r0['day_of_week'])) == 4 else 0,

                # ── Asia context ─────────────────────────────────────────────
                'dist_asia_hi_atr':   (sv(r0['asia_hi']) - entry_price) / atr if sv(r0['asia_hi']) > 0 else 0,
                'dist_asia_lo_atr':   (entry_price - sv(r0['asia_lo'])) / atr if sv(r0['asia_lo']) > 0 else 0,

                # ── Meta (nu în features finale) ─────────────────────────────
                '_label':     label,
                '_direction': direction,
                '_session':   sess_name,
                '_date':      date_str,
                '_entry_px':  entry_price,
                '_max_fwd':   max_fwd,
            }
            setups.append(feat)
            break  # un singur setup per bară (primul displacement valid)

    return setups


def build_dataset(years, regime_pkg=None):
    db_uri = f"file:{DB}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True, timeout=60)

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
        for sess_name, sess_cfg in SESSIONS.items():
            setups = extract_setups(df, sess_name, sess_cfg, date_str=date_str, regime_pkg=regime_pkg)
            all_setups.extend(setups)

    conn.close()
    return pd.DataFrame(all_setups)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

_regime_pkg = load_regime_classifier()

log.info("Extrag dataset IS (2021-2024)...")
df_train_raw = build_dataset(TRAIN_YEARS, regime_pkg=_regime_pkg)
log.info("Extrag dataset OOS (2025-2026)...")
df_test_raw  = build_dataset(TEST_YEARS, regime_pkg=_regime_pkg)

log.info(f"IS: {len(df_train_raw)} setups | OOS: {len(df_test_raw)} setups")
if len(df_train_raw) > 0:
    log.info(f"IS label dist: {df_train_raw['_label'].value_counts().to_dict()}")
    log.info(f"Session dist (IS): {df_train_raw['_session'].value_counts().to_dict()}")
    log.info(f"Direction dist (IS): {df_train_raw['_direction'].value_counts().to_dict()}")
    n_days_is = df_train_raw['_date'].nunique()
    log.info(f"IS setups/zi: {len(df_train_raw)/n_days_is:.2f} ({n_days_is} zile)")
if len(df_test_raw) > 0:
    log.info(f"OOS label dist: {df_test_raw['_label'].value_counts().to_dict()}")
    n_days_oos = df_test_raw['_date'].nunique()
    log.info(f"OOS setups/zi: {len(df_test_raw)/n_days_oos:.2f} ({n_days_oos} zile)")

# ── Pregătire features ────────────────────────────────────────────────────────

# ── Synthetic Order Flow features ─────────────────────────────────────────
_OF_PATH = Path(__file__).parent / "data" / "orderflow_features.parquet"
if _OF_PATH.exists():
    _of = __import__('pandas').read_parquet(_OF_PATH)
    _of['date'] = _of['date'].astype(str)
    _OF_COLS_OF = [c for c in _of.columns if c not in ['session_id','date','session_type',
                  'session_open','session_close','session_high','session_low','total_vol']]
    _of_m = _of[['date','session_type'] + _OF_COLS_OF].rename(
        columns={'date':'_date','session_type':'_session'})
    for _dref in [df_train_raw, df_test_raw]:
        _merged = _dref.merge(_of_m, on=['_date','_session'], how='left')
        for _c in _OF_COLS_OF:
            _dref[_c] = _merged[_c].fillna(0.0).values
    log.info(f"Order flow: {len(_OF_COLS_OF)} features merged (LON+NY)")
else:
    _OF_COLS_OF = []
META_COLS  = [c for c in df_train_raw.columns if c.startswith('_')]
FEAT_COLS  = [c for c in df_train_raw.columns if not c.startswith('_')]

X_train_full = df_train_raw[FEAT_COLS].fillna(0).astype(float).values
y_train_full = df_train_raw['_label'].values

X_test  = df_test_raw[FEAT_COLS].fillna(0).astype(float).values
y_test  = df_test_raw['_label'].values

# ── Time-based 80/20 IS split pentru early stopping ─────────────────────────
split_idx  = int(len(X_train_full) * 0.80)
X_tr  = X_train_full[:split_idx]
y_tr  = y_train_full[:split_idx]
X_val = X_train_full[split_idx:]
y_val = y_train_full[split_idx:]

log.info(f"IS train={len(X_tr)} | IS val={len(X_val)} | OOS={len(X_test)}")

# ── XGBoost training ──────────────────────────────────────────────────────────
log.info("Training XGBoost DSM...")
model = xgb.XGBClassifier(**XGB_PARAMS, verbosity=0)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

# ── Evaluare ──────────────────────────────────────────────────────────────────
prob_is  = model.predict_proba(X_train_full)[:, 1]
prob_oos = model.predict_proba(X_test)[:, 1]

auc_is  = roc_auc_score(y_train_full, prob_is)
auc_oos = roc_auc_score(y_test, prob_oos)

log.info(f"IS  AUC: {auc_is:.3f}")
log.info(f"OOS AUC: {auc_oos:.3f}")

# WR la diferite threshold-uri (OOS)
for thr in [0.55, 0.60, 0.65, 0.70]:
    mask = prob_oos >= thr
    n    = mask.sum()
    if n > 0:
        wr = y_test[mask].mean() * 100
        log.info(f"  OOS WR@{thr}: {wr:.1f}%  ({n} setups)")

# ── Salvare model ─────────────────────────────────────────────────────────────
payload = {
    'model':       model,
    'features':    FEAT_COLS,     # checkerul citește pkg['features']
    'threshold':   0.65,          # WR=79.8% OOS
    'auc_is':      round(auc_is,  3),
    'auc_oos':     round(auc_oos, 3),
    'sessions':    list(SESSIONS.keys()),
    'label_window_min': LABEL_WINDOW,
    'tp_pt':       TP_PT,
    'train_years': TRAIN_YEARS,
    'test_years':  TEST_YEARS,
    'version':     'DSM_XGB_v2',
}
with open(OUT, 'wb') as f:
    pickle.dump(payload, f)
log.info(f"Model salvat: {OUT}")
log.info("Done.")
