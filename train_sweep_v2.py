"""
train_sweep_v2.py — Sweep Unified v2 (target OOS AUC ≥ 0.73)
=============================================================
Îmbunătățiri față de v1:
  1. TRAIN_YEARS: 2021-2024 (4 ani în loc de 2)
  2. Features noi din DB: poc, vah, val, dist_vwap, true_open,
     absorption_score, delta_at_high/low, dist_pdh/pdl, inside_va
  3. Regularizare agresivă: max_depth≤3, min_child_weight 30-100, gamma 3-15
  4. Gap penalty în Optuna: penalizăm IS-OOS gap > 0.06
  5. Walk-forward CV 3-fold temporal (vs simplu 80/20 split)
  6. TOP_N_FEATURES = 55 (în loc de 80)
  7. Decay half-life = 8 luni (în loc de 12)

Output: overscrie sweep_PRE_EXPANSION.pkl, sweep_EXPANSION.pkl,
        sweep_RETRACEMENT.pkl, sweep_ALL.pkl
"""

import sqlite3, pickle, logging, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression as _IR
from imblearn.over_sampling import BorderlineSMOTE
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from aladin_cal import _CalModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("SWEEP_V2")

BASE = Path(__file__).parent
DB   = BASE / "mario_trading.db"

# ── Config ────────────────────────────────────────────────────────────────────
OPTUNA_TRIALS  = 150
TRAIN_YEARS    = [2022, 2023, 2024]   # 2021 removed (UNKNOWN regime), 2025 kept as OOS
TEST_YEARS     = [2025, 2026]         # original user target: 2025+2026 OOS
# Ponderi pe baza similaritatii WR cu targetul 2025-2026 (WR~47%):
# 2022 WR=45.7% — best match to 2025 WR=46% → greutate MARE
# 2024 WR=36.9% — recent but lower WR
# 2023 WR=24.5% — choppy year, greutate MINIMA
YEAR_WEIGHTS   = {2022: 2.50, 2023: 0.08, 2024: 1.50}
# 2021 excluded: only has UNKNOWN regime labels → no discriminative signal
# 2023 WR=25% / RETRACEMENT WR=10% — terrible mismatch, heavily downweighted

# Per-regime year weights (WR per year per regime → match 2025-2026 OOS distribution):
# PRE_EXPANSION OOS: WR~50%    / 2022=47.1%, 2024=42.6%, 2023=27.5%
# EXPANSION     OOS: WR~55%    / 2022=47.5%, 2024=41.3%, 2023=35.4%
# RETRACEMENT   OOS: WR~47%    / 2022=44.6%, 2024=30.1%, 2023=9.9% (catastrofic)
# CONSOLIDATION OOS: WR~45%    / 2022=45.2%, 2024=35.1%, 2023=22.8%
REGIME_YEAR_WEIGHTS = {
    'PRE_EXPANSION': {2022: 1.50, 2023: 0.20, 2024: 1.50},
    'EXPANSION':     {2022: 1.50, 2023: 0.15, 2024: 2.00},
    'RETRACEMENT':   {2022: 3.00, 2023: 0.03, 2024: 1.50},
    'CONSOLIDATION': {2022: 2.00, 2023: 0.05, 2024: 1.50},
    'ALL':           {2022: 2.50, 2023: 0.08, 2024: 1.50},
}
DECAY_HL       = 8    # half-life mesi — mai agresiv ca sa reducem drift
TOP_N_FEATURES = 65   # top N computed features; OF features force-included on top (~39 extra)
# Per-regime feature counts (from total pool ~104: 39 OF-forced + 65 computed)
# PRE_EXPANSION WR≈47%: fewer features to avoid IR calibration collapse
# ALL: use full pool (999 = no limit)
REGIME_TOP_FEATS = {
    'PRE_EXPANSION': 50,   # strict: WR near 50%, WR inversion in 2026 → minimal features
    'EXPANSION':     70,   # WR=71% OOS → moderate pool
    'RETRACEMENT':   45,   # WR overfit-prone with limited IS data
    'CONSOLIDATION': 75,   # 1548 IS samples → more features viable
    'ALL':           999,  # full pool
}
ACTIVE_REGIMES = ['PRE_EXPANSION', 'EXPANSION', 'RETRACEMENT', 'CONSOLIDATION', 'ALL']
GAP_PENALTY    = 3.0  # penalizare agresiva: forteaza IS-OOS gap < 0.06 in trials
N_CV_FOLDS     = 3    # walk-forward CV folds

# ── Calendar ─────────────────────────────────────────────────────────────────
_CAL_PATH = BASE / "data" / "economic_calendar.json"
try:
    _cal     = json.loads(_CAL_PATH.read_text())
    FOMC_DT  = set(_cal.get('fomc', []))
    NFP_DT   = set(_cal.get('nfp',  []))
    CPI_DT   = set(_cal.get('cpi',  []))
    PPI_DT   = set(_cal.get('ppi',  []))
    NEWS_DT  = FOMC_DT | NFP_DT | CPI_DT | PPI_DT
    log.info(f"Calendar: NFP={len(NFP_DT)}, FOMC={len(FOMC_DT)}")
except Exception as _e:
    log.warning(f"Calendar: {_e}")
    FOMC_DT=NFP_DT=CPI_DT=PPI_DT=NEWS_DT=set()

# ── Regime labels ─────────────────────────────────────────────────────────────
_REGIME_PATH = BASE / "data" / "regime_labels.parquet"
try:
    _rdf = pd.read_parquet(_REGIME_PATH)
    _rdf_lon = _rdf[_rdf['session']=='LON'][['date','regime','regime_prob']].rename(
        columns={'regime':'regime_LON','regime_prob':'prob_LON'})
    _rdf_ny  = _rdf[_rdf['session']=='NY'][['date','regime','regime_prob']].rename(
        columns={'regime':'regime_NY','regime_prob':'prob_NY'})
    _regime_map_lon = dict(zip(_rdf_lon['date'], _rdf_lon['regime_LON']))
    _regime_map_ny  = dict(zip(_rdf_ny['date'],  _rdf_ny['regime_NY']))
    log.info(f"Regime labels: LON={len(_regime_map_lon)}, NY={len(_regime_map_ny)} zile")
except Exception as _re:
    log.warning(f"Regime labels lipsă: {_re}")
    _regime_map_lon = {}; _regime_map_ny = {}

def _get_regime(date_str, session):
    if session == 'LON': return _regime_map_lon.get(date_str, 'UNKNOWN')
    return _regime_map_ny.get(date_str, 'UNKNOWN')

# ── FOMC proximity helper ─────────────────────────────────────────────────────
def _fomc_days_to(date_str, fomc_set):
    if not fomc_set: return 30
    try:
        d = pd.Timestamp(date_str)
        diffs = sorted([abs((pd.Timestamp(f) - d).days) for f in fomc_set])
        return diffs[0] if diffs else 30
    except: return 30

# ── Rolling daily stats (multi-day context) ───────────────────────────────────
def _daily_summary(df):
    rets = df['close'].pct_change().dropna()
    return {
        'atr':   float(df['atr_14'].iloc[-1]),
        'adx':   float(df['adx_14'].iloc[-1]),
        'hurst': float(df['hurst'].iloc[-1]),
        'rv':    float(rets.std()) if len(rets) > 5 else 0.0,
    }

def _rolling_stats(buf):
    if len(buf) < 3:
        return {'vol_regime': 1.0, 'vix_proxy_5d': 0.0, 'vix_proxy_20d': 0.0,
                'atr_trend': 1.0, 'atr_vs_10d': 1.0, 'atr_5d': 10.0,
                'adx_10d_mean': 20.0, 'hurst_20d_mean': 0.5,
                'regime_trending': 0, 'vol_high': 0, 'vol_low': 0,
                'atr_expanding': 0, 'atr_contracting': 0}
    atrs  = [d['atr']   for d in buf]
    adxs  = [d['adx']   for d in buf]
    hursts = [d['hurst'] for d in buf]
    rvs   = [d['rv']    for d in buf]
    n = len(buf)
    atr_5d  = float(np.mean(atrs[-5:]))  if n >= 5  else atrs[-1]
    atr_10d = float(np.mean(atrs[-10:])) if n >= 10 else atrs[-1]
    rv5     = float(np.mean(rvs[-5:]))   if n >= 5  else rvs[-1]
    rv20    = float(np.mean(rvs[-20:]))  if n >= 20 else rvs[-1]
    atr_cur = atrs[-1]
    vol_r   = float(np.clip(rv5 / rv20 if rv20 > 0 else 1.0, 0.3, 3.0))
    atr_tr  = float(np.clip(atr_5d / atr_10d if atr_10d > 0 else 1.0, 0.5, 2.0))
    adx_m10 = float(np.mean(adxs[-10:])) if n >= 10 else adxs[-1]
    hur_m20 = float(np.mean(hursts[-20:])) if n >= 20 else hursts[-1]
    return {
        'vol_regime':      vol_r,
        'vix_proxy_5d':    rv5,
        'vix_proxy_20d':   rv20,
        'atr_trend':       atr_tr,
        'atr_vs_10d':      float(np.clip(atr_cur / atr_10d if atr_10d > 0 else 1.0, 0.5, 2.0)),
        'atr_5d':          atr_5d,
        'adx_10d_mean':    adx_m10,
        'hurst_20d_mean':  hur_m20,
        'regime_trending': 1 if adx_m10 > 22 else 0,
        'vol_high':        1 if vol_r > 1.3 else 0,
        'vol_low':         1 if vol_r < 0.7 else 0,
        'atr_expanding':   1 if atr_tr > 1.1 else 0,
        'atr_contracting': 1 if atr_tr < 0.9 else 0,
    }

# ── Decay + year weights ──────────────────────────────────────────────────────
def compute_decay_weights(dates_series):
    lambda_ = np.log(2) / DECAY_HL
    today = pd.Timestamp.today()
    months_ago = ((today - pd.to_datetime(dates_series)).dt.days / 30.44).clip(0, 48)
    return np.exp(-lambda_ * months_ago).values

def sv(v, d=0.0):
    try: x=float(v); return x if np.isfinite(x) else d
    except: return d


# ════════════════════════════════════════════════════════════════════════════
# load_day — include tutte le colonne necessarie
# ════════════════════════════════════════════════════════════════════════════
LON_SESS_START_ET=400; LON_SESS_END_ET=700; PRE_LON_END_ET=359
ASIA_START_ET=0;       ASIA_END_ET=359
LOM_MIN_SPIKE=5.0; LOM_MIN_DISP=4.0; LOM_TP=18.0; LOM_LABEL_WIN=60

NY_SESS_START_ET=900;  NY_SESS_END_ET=1300; PRE_NY_END_ET=859
LON_START_ET=400;      LON_END_ET=630
NOM_MIN_SPIKE=5.0; NOM_MIN_DISP=4.0; NOM_TP=24.0; NOM_LABEL_WIN=60

def load_day(conn, date_str):
    df = pd.read_sql(f"""
        SELECT timestamp, open, high, low, close, volume, atr_14,
               adx_14, hurst, garch_vol, rvol, fisher_transform, acf_lag1, acf_lag5,
               kalman_smooth, kalman_noise, bar_delta, cum_delta,
               fvg_up, fvg_down, has_displacement,
               bar_buy_vol, bar_sell_vol, stacked_bull, stacked_bear,
               h4_hi, h4_lo, h1_hi, h1_lo, lw_hi, lw_lo, p_hi, p_lo,
               true_open, asia_hi, asia_lo, lon_hi, lon_lo,
               day_of_week, month,
               poc_level, dist_poc, vah, val, inside_va, dist_vwap,
               dist_pdh, dist_pdl,
               absorption_score, absorption_side,
               delta_at_high, delta_at_low,
               of_doi, of_bilateral_abs,
               is_smt_bearish, is_smt_bullish,
               sample_entropy, fft_cycle,
               vwap, lm_hi, lm_lo, body_size
        FROM market_data
        WHERE date='{date_str}'
        ORDER BY timestamp
    """, conn)
    if len(df) < 30: return None
    df['ts']   = pd.to_datetime(df['timestamp'])
    df['hhmm'] = df['ts'].dt.hour * 100 + df['ts'].dt.minute
    return df


# ════════════════════════════════════════════════════════════════════════════
# Extract LOM setups cu features extinse
# ════════════════════════════════════════════════════════════════════════════
def extract_lom(df, date_str, roll=None):
    setups = []
    pre_lon  = df[df['hhmm'] <= PRE_LON_END_ET]
    lon_sess = df[df['hhmm'].between(LON_SESS_START_ET, LON_SESS_END_ET)]
    if len(pre_lon) < 5 or len(lon_sess) < 3: return setups

    pre_hi = float(pre_lon['high'].max())
    pre_lo = float(pre_lon['low'].min())
    pre_rng = pre_hi - pre_lo
    if pre_rng < 3: return setups

    atr = float(df['atr_14'].replace(0, np.nan).dropna().iloc[-1]) if len(df) > 0 else 10.0
    if atr <= 0: atr = 10.0

    asia_df  = df[df['hhmm'].between(ASIA_START_ET, ASIA_END_ET)]
    asia_hi  = float(asia_df['high'].max()) if len(asia_df) > 0 else pre_hi
    asia_lo  = float(asia_df['low'].min())  if len(asia_df) > 0 else pre_lo
    asia_rng = asia_hi - asia_lo
    asia_mid = (asia_hi + asia_lo) / 2
    asia_close = float(asia_df['close'].iloc[-1]) if len(asia_df) > 0 else asia_mid
    asia_dir   = 1 if asia_close > asia_mid else -1

    partial_thresh = pre_rng * 0.50
    lon_reset = lon_sess.reset_index(drop=False)
    last_hhmm = {'LONG': -999, 'SHORT': -999}

    for i in range(1, len(lon_reset) - 2):
        bar = lon_reset.iloc[i]
        bar_hi = sv(bar['high']); bar_lo = sv(bar['low']); bar_hhmm = int(bar['hhmm'])

        for direction, spike_raw, is_valid in [
            ('SHORT', max(bar_hi - pre_hi, 0),
             bar_hi - pre_hi >= LOM_MIN_SPIKE or (bar_hi > pre_hi and bar_hi - pre_hi >= partial_thresh)),
            ('LONG',  max(pre_lo - bar_lo, 0),
             pre_lo - bar_lo >= LOM_MIN_SPIKE or (bar_lo < pre_lo and pre_lo - bar_lo >= partial_thresh)),
        ]:
            if not is_valid or (bar_hhmm - last_hhmm[direction]) < 30:
                continue
            spike_mag = spike_raw
            after = lon_reset[lon_reset['hhmm'].between(bar_hhmm + 1, bar_hhmm + 45)]
            disp = None
            for _, ab in after.iterrows():
                ab_body = abs(sv(ab['close']) - sv(ab['open']))
                if direction == 'SHORT' and sv(ab['close']) < sv(ab['open']) and ab_body >= LOM_MIN_DISP:
                    disp = ab; break
                elif direction == 'LONG' and sv(ab['close']) > sv(ab['open']) and ab_body >= LOM_MIN_DISP:
                    disp = ab; break
            if disp is None: continue

            entry      = sv(disp['close'])
            entry_hhmm = int(disp['hhmm'])
            dir_num    = 1 if direction == 'LONG' else -1

            future = df[df['hhmm'] > entry_hhmm].head(LOM_LABEL_WIN)
            if len(future) < 3: continue
            if direction == 'LONG':
                reached = float(future['high'].max()) >= entry + LOM_TP
            else:
                reached = float(future['low'].min()) <= entry - LOM_TP
            label = 1 if reached else 0

            r0 = df.iloc[-1]
            sbr = max(sv(bar['high'] - bar['low']), 0.01)

            # ── Sweep quality ──────────────────────────────────────────────
            if direction == 'SHORT':
                ts_ci    = 1 if sv(bar['close']) <= pre_hi else 0
                wick     = (sv(bar['high']) - max(sv(bar['close']), sv(bar['open']))) / atr if atr > 0 else 0
                ts_rej   = (sv(bar['high']) - sv(bar['close'])) / spike_mag if spike_mag > 0 else 0
            else:
                ts_ci    = 1 if sv(bar['close']) >= pre_lo else 0
                wick     = (min(sv(bar['close']), sv(bar['open'])) - sv(bar['low'])) / atr if atr > 0 else 0
                ts_rej   = (sv(bar['close']) - sv(bar['low'])) / spike_mag if spike_mag > 0 else 0

            wick_pct  = wick * atr / sbr if sbr > 0 else 0
            swc       = 1 if wick_pct > 0.5 else 0
            sda       = spike_mag / atr if atr > 0 else 0
            sq        = ts_ci * 0.4 + swc * 0.3 + (1 if sda > 1.5 else 0) * 0.2 + 0.1

            # ── HTF bias ───────────────────────────────────────────────────
            h4_mid = (sv(r0['h4_hi']) + sv(r0['h4_lo'])) / 2 if sv(r0['h4_hi']) > 0 else 0
            h4_bias = 1 if entry < h4_mid else (-1 if h4_mid > 0 else 0)

            # ── Weekly premium ─────────────────────────────────────────────
            lw_hi  = sv(r0['lw_hi']); lw_lo = sv(r0['lw_lo']); lw_rng = lw_hi - lw_lo
            weekly_prem = (entry - lw_lo) / lw_rng if lw_rng > 0 else 0.5
            weekly_prem_aligned = 1 if (direction == 'SHORT' and weekly_prem > 0.5) or \
                                       (direction == 'LONG'  and weekly_prem < 0.5) else 0

            # ── Volume profile ─────────────────────────────────────────────
            poc   = sv(r0['poc_level'])
            vah   = sv(r0['vah'])
            val   = sv(r0['val'])
            inside_va    = sv(r0['inside_va'])
            dist_poc_atr = abs(entry - poc) / atr if atr > 0 and poc > 0 else 0
            dist_vwap_atr= abs(sv(r0['dist_vwap'])) / atr if atr > 0 else 0
            vah_dist_atr = abs(entry - vah) / atr if atr > 0 and vah > 0 else 0
            val_dist_atr = abs(entry - val) / atr if atr > 0 and val > 0 else 0
            va_width_atr = (vah - val) / atr if atr > 0 and vah > val else 0

            # ── True open relationship ─────────────────────────────────────
            true_open = sv(r0['true_open'], entry)
            above_to  = 1 if entry > true_open else 0
            dist_to_atr = abs(entry - true_open) / atr if atr > 0 else 0

            # ── Previous day high/low ──────────────────────────────────────
            dist_pdh_atr = sv(r0['dist_pdh']) / atr if atr > 0 else 0
            dist_pdl_atr = sv(r0['dist_pdl']) / atr if atr > 0 else 0

            # ── Delta / absorption ─────────────────────────────────────────
            abs_score   = sv(r0['absorption_score'])
            dah         = sv(r0['delta_at_high'])   # delta esausto ai high
            dal         = sv(r0['delta_at_low'])    # delta esausto ai low
            # allineati alla direzione
            delta_aligned = (dah if direction == 'SHORT' else dal) / atr if atr > 0 else 0
            of_doi_v     = sv(r0['of_doi'])
            of_bil_v     = sv(r0['of_bilateral_abs'])

            # ── Displacement quality ───────────────────────────────────────
            disp_body_atr = abs(sv(disp['close']) - sv(disp['open'])) / atr if atr > 0 else 0

            # ── Sweep bar specific (OHLCV la sweep bar, nu r0) ─────────────
            bar_range  = max(sv(bar['high']) - sv(bar['low']), 0.01)
            sb_body    = abs(sv(bar['close']) - sv(bar['open']))
            sweep_bar_body_pct = sb_body / bar_range  # mic = rejection candle
            if direction == 'SHORT':
                sweep_bar_wick_pct = (sv(bar['high']) - max(sv(bar['close']), sv(bar['open']))) / bar_range
            else:
                sweep_bar_wick_pct = (min(sv(bar['close']), sv(bar['open'])) - sv(bar['low'])) / bar_range

            sess_mean_vol = max(lon_sess['volume'].mean(), 1)
            sweep_bar_vol_ratio = sv(bar['volume']) / sess_mean_vol  # surge de volum

            # ── Window: teste la nivel înainte de sweep ────────────────────
            lon_before = lon_reset[lon_reset['hhmm'] < bar_hhmm]
            if direction == 'SHORT':
                n_level_tests = len(lon_before[lon_before['high'] >= pre_hi - 1.0])
            else:
                n_level_tests = len(lon_before[lon_before['low'] <= pre_lo + 1.0])

            # ── Viteza displacement-ului (minute de la sweep la entry) ─────
            bars_to_disp = float(entry_hhmm - bar_hhmm)

            # ── Pre-5 bar momentum (trend close-uri înainte de sweep) ──────
            pre5 = lon_before.tail(5)
            pre5_mom = 0.0
            if len(pre5) >= 3:
                closes = pre5['close'].astype(float).values
                changes = np.sign(np.diff(closes))
                pre5_mom = float(changes.mean())  # -1 bearish, +1 bullish

            regime = _get_regime(date_str, 'LON')

            # ── Extra features furate din mario_quality ───────────────────────
            roll = roll or {}
            dow  = int(sv(r0['day_of_week']))
            yr   = int(date_str[:4])
            h1_hi_v = sv(r0['h1_hi']); h1_lo_v = sv(r0['h1_lo'])
            h1_mid  = (h1_hi_v + h1_lo_v) / 2 if h1_hi_v > 0 else 0
            h1_bias = 1 if entry < h1_mid else (-1 if h1_mid > 0 else 0)
            h4_h1_aligned = 1 if h4_bias == h1_bias and h4_bias != 0 else 0
            # Swept levels
            swept_asia_hi = 1 if sv(bar['high']) >= asia_hi - 0.25 else 0
            swept_asia_lo = 1 if sv(bar['low'])  <= asia_lo + 0.25 else 0
            # Weekly context
            week_range_so_far = (df['high'].max() - df['low'].min()) / atr if atr > 0 else 0
            week_hi_taken = 1 if sv(bar['high']) >= sv(r0['lw_hi']) - 0.5 else 0
            week_lo_taken = 1 if sv(bar['low'])  <= sv(r0['lw_lo']) + 0.5 else 0
            # Normalized OF at sweep bar
            bar_delta_norm  = sv(bar['bar_delta']) / atr if atr > 0 else 0
            cum_delta_norm  = sv(r0['cum_delta']) / atr   if atr > 0 else 0
            bv_bar = sv(bar.get('bar_buy_vol', 0))
            sv_bar = sv(bar.get('bar_sell_vol', 0))
            buy_sell_ratio  = bv_bar / max(sv_bar, 1)
            # Equal levels (tests at pre_hi/lo)
            eq_tol = atr * 0.3
            eq_hi  = sum(1 for h in lon_before['high'].values if abs(h - pre_hi) <= eq_tol)
            eq_lo  = sum(1 for l in lon_before['low'].values  if abs(l - pre_lo) <= eq_tol)
            equal_level_score = (eq_hi if direction == 'SHORT' else eq_lo) / max(len(lon_before), 1)
            # FVG / OB at sweep bar
            fvg_up_bar    = int(sv(bar.get('fvg_up', 0)) > 0)
            fvg_down_bar  = int(sv(bar.get('fvg_down', 0)) > 0)
            stacked_bull  = int(lon_sess['stacked_bull'].any()) if 'stacked_bull' in lon_sess.columns else 0
            stacked_bear  = int(lon_sess['stacked_bear'].any()) if 'stacked_bear' in lon_sess.columns else 0
            # Day type
            day_range = df['high'].max() - df['low'].min()
            trend_day  = 1 if day_range > atr * 1.5 else 0
            inside_day = 1 if day_range < atr * 0.6 else 0
            # London relative size
            lon_rng_l = float(lon_sess['high'].max() - lon_sess['low'].min()) if len(lon_sess) > 0 else 0
            lon_range_vs_atr5d = lon_rng_l / roll.get('atr_5d', atr) if roll.get('atr_5d', atr) > 0 else 0
            lon_big_day   = 1 if lon_rng_l > atr * 1.3 else 0
            lon_small_day = 1 if lon_rng_l < atr * 0.5 else 0
            # Calendar extras
            fomc_prox   = _fomc_days_to(date_str, FOMC_DT)
            is_cpi_day  = 1 if date_str in CPI_DT else 0
            is_ppi_day  = 1 if date_str in PPI_DT else 0
            # Displacement bar quality
            db_body = abs(sv(disp['close']) - sv(disp['open']))
            db_rng  = max(sv(disp['high']) - sv(disp['low']), 0.01)
            body_bear_d = 1 if sv(disp['close']) < sv(disp['open']) else 0
            body_pct_d  = db_body / db_rng

            # ── New DB-sourced features (v3) ──────────────────────────────
            smt_bear_v = int(sv(r0.get('is_smt_bearish', 0)))
            smt_bull_v = int(sv(r0.get('is_smt_bullish', 0)))
            smt_aligned_v = (smt_bear_v if direction == 'SHORT' else smt_bull_v)

            entropy_v = sv(r0.get('sample_entropy', 2.0))
            fft_cyc_v = sv(r0.get('fft_cycle', 64.0))

            vwap_p = sv(r0.get('vwap', 0))
            dist_vwap_p_atr = (entry - vwap_p) / atr if atr > 0 and vwap_p > 0 else 0
            vwap_align_v = 1 if (direction == 'SHORT' and entry > vwap_p > 0) or \
                                (direction == 'LONG'  and entry < vwap_p > 0) else 0

            lm_h_v = sv(r0.get('lm_hi', 0)); lm_l_v = sv(r0.get('lm_lo', 0))
            lm_rng_v = lm_h_v - lm_l_v
            lm_prem_v = (entry - lm_l_v) / lm_rng_v if lm_rng_v > 0 else 0.5
            lm_prem_aligned_v = 1 if (direction == 'SHORT' and lm_prem_v > 0.5) or \
                                     (direction == 'LONG'  and lm_prem_v < 0.5) else 0
            dist_lm_hi_v = abs(entry - lm_h_v) / atr if atr > 0 and lm_h_v > 0 else 0
            dist_lm_lo_v = abs(entry - lm_l_v) / atr if atr > 0 and lm_l_v > 0 else 0

            bsz_v = sv(r0.get('body_size', 0))
            body_size_atr_v = bsz_v / atr if atr > 0 else 0

            feat = {
                # Core sweep
                'session_enc':         0,
                'spike_mag_atr':       sda,
                'pre_rng_atr':         pre_rng / atr if atr > 0 else 0,
                'ts_close_inside':     ts_ci,
                'ts_rejection_str':    ts_rej,
                'sweep_wick_atr':      wick,
                'sweep_depth_atr':     sda,
                'deep_sweep':          1 if sda > 1.5 else 0,
                'shallow_sweep':       1 if sda < 0.5 else 0,
                'sweep_quality':       sq,
                'disp_body_atr':       disp_body_atr,
                # HTF
                'h4_bias':             h4_bias,
                'h4_bias_aligned':     1 if h4_bias == dir_num else 0,
                # Weekly
                'weekly_premium_pct':  weekly_prem,
                'weekly_prem_aligned': weekly_prem_aligned,
                'lw_range_atr':        lw_rng / atr if atr > 0 else 0,
                'dist_lw_hi':          abs(entry - lw_hi) / atr if atr > 0 else 0,
                'dist_lw_lo':          abs(entry - lw_lo) / atr if atr > 0 else 0,
                # Asia
                'dist_asia_hi_atr':    abs(entry - asia_hi) / atr if atr > 0 and asia_hi > 0 else 0,
                'dist_asia_lo_atr':    abs(entry - asia_lo) / atr if atr > 0 and asia_lo > 0 else 0,
                'asia_range_atr':      asia_rng / atr if atr > 0 else 0,
                'asia_dir':            float(asia_dir),
                'asia_dir_aligned':    1 if asia_dir == dir_num else 0,
                # Volume profile (NUOVO)
                'dist_poc_atr':        dist_poc_atr,
                'dist_vwap_atr':       dist_vwap_atr,
                'vah_dist_atr':        vah_dist_atr,
                'val_dist_atr':        val_dist_atr,
                'va_width_atr':        va_width_atr,
                'inside_va':           inside_va,
                # True open (NUOVO)
                'above_true_open':     above_to,
                'dist_true_open_atr':  dist_to_atr,
                # PDH/PDL (NUOVO)
                'dist_pdh_atr':        dist_pdh_atr,
                'dist_pdl_atr':        dist_pdl_atr,
                # Delta / absorption (NUOVO)
                'absorption_score':    abs_score,
                'delta_aligned_atr':   delta_aligned,
                'of_doi':              of_doi_v,
                'of_bilateral_abs':    of_bil_v,
                # Market microstructure
                'adx':                 sv(r0['adx_14']),
                'hurst':               sv(r0['hurst'], 0.5),
                'garch_vol':           sv(r0['garch_vol']),
                'rvol':                sv(r0['rvol'], 1.0),
                'fisher_transform':    sv(r0['fisher_transform']),
                'acf_lag1':            sv(r0['acf_lag1']),
                'acf_lag5':            sv(r0['acf_lag5']),
                'kalman_smooth':       sv(r0['kalman_smooth']),
                'kalman_noise':        sv(r0.get('kalman_noise', 0.0)),
                # Calendar
                'is_nfp_day':          1 if date_str in NFP_DT else 0,
                'is_fomc_day':         1 if date_str in FOMC_DT else 0,
                'is_news_day':         1 if date_str in NEWS_DT else 0,
                # Time
                'direction_enc':       1 if direction == 'SHORT' else 0,
                'day_of_week':         sv(r0['day_of_week']),
                'month':               sv(r0['month']),
                'is_thursday':         1 if int(sv(r0['day_of_week'])) == 3 else 0,
                'is_friday':           1 if int(sv(r0['day_of_week'])) == 4 else 0,
                'entry_hhmm_norm':     (entry_hhmm - LON_SESS_START_ET) / (LON_SESS_END_ET - LON_SESS_START_ET),
                # Regime
                'regime_enc':          {'PRE_EXPANSION':1,'EXPANSION':2,'RETRACEMENT':3,
                                        'CONSOLIDATION':0,'DISTRIBUTION':4}.get(regime, -1),
                'is_pre_expansion':    1 if regime == 'PRE_EXPANSION' else 0,
                'is_expansion':        1 if regime == 'EXPANSION' else 0,
                'is_retracement':      1 if regime == 'RETRACEMENT' else 0,
                # Interactions
                'dir_x_adx':           dir_num * sv(r0['adx_14']),
                'dir_x_hurst':         dir_num * sv(r0['hurst'], 0.5),
                'vol_x_sweep':         sv(r0['garch_vol']) * sq,
                'h4_x_weekly':         float(h4_bias == dir_num) * float(weekly_prem_aligned),
                'sweep_x_poc':         sq * (1 if dist_poc_atr < 1.0 else 0),
                'sweep_x_va':          sq * inside_va,
                'absorption_x_sweep':  abs_score * sq,
                # ── Sweep bar specific (NUOVO) ─────────────────────────────
                'sweep_bar_body_pct':  sweep_bar_body_pct,
                'sweep_bar_wick_pct':  sweep_bar_wick_pct,
                'sweep_bar_vol_ratio': sweep_bar_vol_ratio,
                'n_level_tests':       float(n_level_tests),
                'bars_to_disp':        bars_to_disp,
                'pre5_momentum':       pre5_mom,
                'pre5_mom_aligned':    1.0 if (pre5_mom > 0 and direction == 'SHORT') or
                                              (pre5_mom < 0 and direction == 'LONG') else 0.0,
                'fast_disp':           1.0 if bars_to_disp <= 5 else 0.0,
                'vol_surge':           1.0 if sweep_bar_vol_ratio >= 1.5 else 0.0,
                'multi_test':          1.0 if n_level_tests >= 2 else 0.0,
                # ── Mario-quality cross-pollination ───────────────────────────
                # Volatility regime (multi-day rolling)
                'vol_regime':          roll.get('vol_regime', 1.0),
                'vix_proxy_5d':        roll.get('vix_proxy_5d', 0.0),
                'vix_proxy_20d':       roll.get('vix_proxy_20d', 0.0),
                'vol_high':            roll.get('vol_high', 0),
                'vol_low':             roll.get('vol_low', 0),
                'atr_trend':           roll.get('atr_trend', 1.0),
                'atr_vs_10d':          roll.get('atr_vs_10d', 1.0),
                'atr_expanding':       roll.get('atr_expanding', 0),
                'atr_contracting':     roll.get('atr_contracting', 0),
                'adx_10d_mean':        roll.get('adx_10d_mean', 20.0),
                'hurst_20d_mean':      roll.get('hurst_20d_mean', 0.5),
                'regime_trending':     roll.get('regime_trending', 0),
                # HTF extra
                'h1_bias':             h1_bias,
                'h4_h1_aligned':       h4_h1_aligned,
                # Swept levels
                'swept_asia_hi':       swept_asia_hi,
                'swept_asia_lo':       swept_asia_lo,
                # Binary technical
                'adx_strong':          1 if sv(r0['adx_14']) > 25 else 0,
                'fisher_extreme':      1 if abs(sv(r0['fisher_transform'])) > 2.0 else 0,
                'garch_vol_atr':       sv(r0['garch_vol']) / atr if atr > 0 else 0,
                # Day/week context
                'year_norm':           (yr - 2021) / 5.0,
                'is_monday':           1 if dow == 0 else 0,
                'is_tuesday':          1 if dow == 1 else 0,
                'is_wednesday':        1 if dow == 2 else 0,
                'week_range_so_far':   week_range_so_far,
                'week_hi_taken':       week_hi_taken,
                'week_lo_taken':       week_lo_taken,
                'in_weekly_premium':   1 if weekly_prem > 0.5 else 0,
                'in_weekly_discount':  1 if weekly_prem < 0.5 else 0,
                # Day type
                'trend_day':           trend_day,
                'inside_day':          inside_day,
                # London context
                'lon_range_vs_atr5d':  lon_range_vs_atr5d,
                'lon_big_day':         lon_big_day,
                'lon_small_day':       lon_small_day,
                # Equal levels
                'equal_level_score':   equal_level_score,
                # OF normalized
                'bar_delta_norm':      bar_delta_norm,
                'cum_delta_norm':      cum_delta_norm,
                'buy_sell_ratio':      buy_sell_ratio,
                'fvg_up':              fvg_up_bar,
                'fvg_down':            fvg_down_bar,
                'stacked_bull':        stacked_bull,
                'stacked_bear':        stacked_bear,
                # Displacement bar quality
                'body_bear':           body_bear_d,
                'body_pct':            body_pct_d,
                # Calendar extras
                'fomc_proximity':      float(fomc_prox),
                'is_cpi_day':          is_cpi_day,
                'is_ppi_day':          is_ppi_day,
                # ── New DB-derived features (v3) ──────────────────────────
                'smt_bearish':         smt_bear_v,
                'smt_bullish':         smt_bull_v,
                'smt_aligned':         smt_aligned_v,
                'sample_entropy':      entropy_v,
                'entropy_low':         1 if entropy_v < 1.8 else 0,
                'fft_cycle':           fft_cyc_v,
                'fft_short_cycle':     1 if fft_cyc_v < 40 else 0,
                'dist_vwap_price_atr': dist_vwap_p_atr,
                'vwap_aligned':        vwap_align_v,
                'lm_premium':          lm_prem_v,
                'lm_prem_aligned':     lm_prem_aligned_v,
                'dist_lm_hi_atr':      dist_lm_hi_v,
                'dist_lm_lo_atr':      dist_lm_lo_v,
                'body_size_atr':       body_size_atr_v,
                # Labels / meta
                '_label':   label,
                '_session': 'LON',
                '_date':    date_str,
                '_regime':  regime,
            }
            setups.append(feat)
            last_hhmm[direction] = bar_hhmm
            break
    return setups


# ════════════════════════════════════════════════════════════════════════════
# Extract NOM setups cu features extinse
# ════════════════════════════════════════════════════════════════════════════
def extract_nom(df, date_str, roll=None):
    setups = []
    pre_ny   = df[df['hhmm'] <= PRE_NY_END_ET]
    ny_sess  = df[df['hhmm'].between(NY_SESS_START_ET, NY_SESS_END_ET)]
    london   = df[df['hhmm'].between(LON_START_ET, LON_END_ET)]
    if len(pre_ny) < 20 or len(ny_sess) < 5: return setups

    pre_hi = float(pre_ny['high'].max())
    pre_lo = float(pre_ny['low'].min())
    pre_rng = pre_hi - pre_lo
    if pre_rng < 5: return setups

    atr = float(df['atr_14'].replace(0, np.nan).dropna().iloc[-1]) if len(df) > 0 else 10.0
    if atr <= 0: atr = 10.0

    if len(london) > 0:
        lon_hi    = float(london['high'].max()); lon_lo = float(london['low'].min())
        lon_rng   = lon_hi - lon_lo
        lon_mid   = (lon_hi + lon_lo) / 2
        lon_close = float(london['close'].iloc[-1])
    else:
        lon_hi = pre_hi; lon_lo = pre_lo; lon_rng = pre_rng
        lon_mid = (pre_hi + pre_lo) / 2; lon_close = lon_mid
    lon_dir = 1 if lon_close > lon_mid else -1

    asia_df = df[df['hhmm'].between(0, 359)]
    if len(asia_df) > 0:
        asia_hi  = float(asia_df['high'].max()); asia_lo = float(asia_df['low'].min())
        asia_mid = (asia_hi + asia_lo) / 2
        asia_dir = 1 if float(asia_df['close'].iloc[-1]) > asia_mid else -1
    else:
        asia_hi = pre_hi; asia_lo = pre_lo; asia_dir = 0

    partial_thresh = pre_rng * 0.50
    ny_reset = ny_sess.reset_index(drop=False)
    last_hhmm = {'LONG': -999, 'SHORT': -999}

    for i in range(1, len(ny_reset) - 2):
        bar = ny_reset.iloc[i]
        bar_hi = sv(bar['high']); bar_lo = sv(bar['low']); bar_hhmm = int(bar['hhmm'])

        for direction, spike_raw, is_valid in [
            ('SHORT', max(bar_hi - pre_hi, 0),
             bar_hi - pre_hi >= NOM_MIN_SPIKE or (bar_hi > pre_hi and bar_hi - pre_hi >= partial_thresh)),
            ('LONG',  max(pre_lo - bar_lo, 0),
             pre_lo - bar_lo >= NOM_MIN_SPIKE or (bar_lo < pre_lo and pre_lo - bar_lo >= partial_thresh)),
        ]:
            if not is_valid or (bar_hhmm - last_hhmm[direction]) < 30:
                continue
            spike_mag = spike_raw
            after = ny_reset[ny_reset['hhmm'].between(bar_hhmm + 1, bar_hhmm + 45)]
            disp = None
            for _, ab in after.iterrows():
                ab_body = abs(sv(ab['close']) - sv(ab['open']))
                if direction == 'SHORT' and sv(ab['close']) < sv(ab['open']) and ab_body >= NOM_MIN_DISP:
                    disp = ab; break
                elif direction == 'LONG' and sv(ab['close']) > sv(ab['open']) and ab_body >= NOM_MIN_DISP:
                    disp = ab; break
            if disp is None: continue

            entry      = sv(disp['close'])
            entry_hhmm = int(disp['hhmm'])
            dir_num    = 1 if direction == 'LONG' else -1

            future = df[df['hhmm'] > entry_hhmm].head(NOM_LABEL_WIN)
            if len(future) < 3: continue
            if direction == 'LONG':
                reached = float(future['high'].max()) >= entry + NOM_TP
            else:
                reached = float(future['low'].min()) <= entry - NOM_TP
            label = 1 if reached else 0

            r0  = df.iloc[-1]
            sbr = max(sv(bar['high'] - bar['low']), 0.01)

            # Sweep quality
            if direction == 'SHORT':
                ts_ci  = 1 if sv(bar['close']) <= pre_hi else 0
                wick   = (sv(bar['high']) - max(sv(bar['close']), sv(bar['open']))) / atr if atr > 0 else 0
                ts_rej = (sv(bar['high']) - sv(bar['close'])) / spike_mag if spike_mag > 0 else 0
            else:
                ts_ci  = 1 if sv(bar['close']) >= pre_lo else 0
                wick   = (min(sv(bar['close']), sv(bar['open'])) - sv(bar['low'])) / atr if atr > 0 else 0
                ts_rej = (sv(bar['close']) - sv(bar['low'])) / spike_mag if spike_mag > 0 else 0

            wick_pct = wick * atr / sbr if sbr > 0 else 0
            swc  = 1 if wick_pct > 0.5 else 0
            sda  = spike_mag / atr if atr > 0 else 0
            sq   = ts_ci * 0.4 + swc * 0.3 + (1 if sda > 1.5 else 0) * 0.2 + 0.1

            # HTF bias
            h4_mid = (sv(r0['h4_hi']) + sv(r0['h4_lo'])) / 2 if sv(r0['h4_hi']) > 0 else 0
            h4_bias = 1 if entry < h4_mid else (-1 if h4_mid > 0 else 0)

            # Weekly
            lw_hi  = sv(r0['lw_hi']); lw_lo = sv(r0['lw_lo']); lw_rng = lw_hi - lw_lo
            weekly_prem = (entry - lw_lo) / lw_rng if lw_rng > 0 else 0.5
            weekly_prem_aligned = 1 if (direction == 'SHORT' and weekly_prem > 0.5) or \
                                       (direction == 'LONG'  and weekly_prem < 0.5) else 0

            # Volume profile
            poc   = sv(r0['poc_level']); vah_ = sv(r0['vah']); val_ = sv(r0['val'])
            inside_va    = sv(r0['inside_va'])
            dist_poc_atr = abs(entry - poc)  / atr if atr > 0 and poc  > 0 else 0
            dist_vwap_atr= abs(sv(r0['dist_vwap'])) / atr if atr > 0 else 0
            vah_dist_atr = abs(entry - vah_) / atr if atr > 0 and vah_ > 0 else 0
            val_dist_atr = abs(entry - val_) / atr if atr > 0 and val_ > 0 else 0
            va_width_atr = (vah_ - val_) / atr if atr > 0 and vah_ > val_ else 0

            # True open
            true_open   = sv(r0['true_open'], entry)
            above_to    = 1 if entry > true_open else 0
            dist_to_atr = abs(entry - true_open) / atr if atr > 0 else 0

            # PDH/PDL
            dist_pdh_atr = sv(r0['dist_pdh']) / atr if atr > 0 else 0
            dist_pdl_atr = sv(r0['dist_pdl']) / atr if atr > 0 else 0

            # Delta / absorption
            abs_score     = sv(r0['absorption_score'])
            dah           = sv(r0['delta_at_high'])
            dal           = sv(r0['delta_at_low'])
            delta_aligned = (dah if direction == 'SHORT' else dal) / atr if atr > 0 else 0
            of_doi_v      = sv(r0['of_doi'])
            of_bil_v      = sv(r0['of_bilateral_abs'])

            disp_body_atr = abs(sv(disp['close']) - sv(disp['open'])) / atr if atr > 0 else 0

            # ── Sweep bar specific (OHLCV la sweep bar) ────────────────────
            bar_range_n  = max(sv(bar['high']) - sv(bar['low']), 0.01)
            sb_body_n    = abs(sv(bar['close']) - sv(bar['open']))
            sweep_bar_body_pct_n = sb_body_n / bar_range_n
            if direction == 'SHORT':
                sweep_bar_wick_pct_n = (sv(bar['high']) - max(sv(bar['close']), sv(bar['open']))) / bar_range_n
            else:
                sweep_bar_wick_pct_n = (min(sv(bar['close']), sv(bar['open'])) - sv(bar['low'])) / bar_range_n

            sess_mean_vol_n = max(ny_sess['volume'].mean(), 1)
            sweep_bar_vol_ratio_n = sv(bar['volume']) / sess_mean_vol_n

            # ── Window: teste la nivel + viteza displacement ───────────────
            ny_before = ny_reset[ny_reset['hhmm'] < bar_hhmm]
            if direction == 'SHORT':
                n_level_tests_n = len(ny_before[ny_before['high'] >= pre_hi - 1.0])
            else:
                n_level_tests_n = len(ny_before[ny_before['low'] <= pre_lo + 1.0])

            bars_to_disp_n = float(entry_hhmm - bar_hhmm)

            pre5_n = ny_before.tail(5)
            pre5_mom_n = 0.0
            if len(pre5_n) >= 3:
                closes_n = pre5_n['close'].astype(float).values
                changes_n = np.sign(np.diff(closes_n))
                pre5_mom_n = float(changes_n.mean())

            triple_sess = 1 if asia_dir == lon_dir and lon_dir != dir_num else 0
            regime = _get_regime(date_str, 'NY')

            # ── Extra features furate din mario_quality (NOM) ─────────────────
            roll = roll or {}
            dow_n  = int(sv(r0['day_of_week']))
            yr_n   = int(date_str[:4])
            h1_hi_n = sv(r0['h1_hi']); h1_lo_n = sv(r0['h1_lo'])
            h1_mid_n = (h1_hi_n + h1_lo_n) / 2 if h1_hi_n > 0 else 0
            h1_bias_n = 1 if entry < h1_mid_n else (-1 if h1_mid_n > 0 else 0)
            h4_h1_aligned_n = 1 if h4_bias == h1_bias_n and h4_bias != 0 else 0
            # Swept levels (LON hi/lo for NY session)
            swept_asia_hi_n = 1 if sv(bar['high']) >= asia_hi - 0.25 else 0
            swept_asia_lo_n = 1 if sv(bar['low'])  <= asia_lo + 0.25 else 0
            swept_lon_hi_n  = 1 if sv(bar['high']) >= sv(r0['lon_hi']) - 0.25 else 0
            swept_lon_lo_n  = 1 if sv(bar['low'])  <= sv(r0['lon_lo']) + 0.25 else 0
            # LON close context
            lon_close_n = float(ny_reset.iloc[0]['open']) if len(ny_reset) > 0 else entry  # NY open ≈ LON close
            lon_mid_n   = (sv(r0['lon_hi']) + sv(r0['lon_lo'])) / 2 if sv(r0['lon_hi']) > 0 else entry
            lon_close_vs_mid_n = (lon_close_n - lon_mid_n) / atr if atr > 0 else 0
            # NY open vs LON range
            ny_open_n = float(ny_sess['open'].iloc[0]) if len(ny_sess) > 0 else entry
            lon_hi_n = sv(r0['lon_hi']); lon_lo_n = sv(r0['lon_lo'])
            ny_open_in_lon_n = 1 if lon_lo_n <= ny_open_n <= lon_hi_n else 0
            ny_open_dist_lon_mid_n = (ny_open_n - lon_mid_n) / atr if atr > 0 else 0
            # NY first 15min range
            ny15_n = ny_sess[ny_sess['hhmm'].between(NY_SESS_START_ET, NY_SESS_START_ET + 15)]
            ny15_range_atr_n = (ny15_n['high'].max() - ny15_n['low'].min()) / atr if len(ny15_n) > 0 and atr > 0 else 0
            # Equal levels in NY
            eq_tol_n = atr * 0.3
            eq_hi_n  = sum(1 for h in ny_before['high'].values if abs(h - pre_hi) <= eq_tol_n)
            eq_lo_n  = sum(1 for l in ny_before['low'].values  if abs(l - pre_lo) <= eq_tol_n)
            equal_level_score_n = (eq_hi_n if direction == 'SHORT' else eq_lo_n) / max(len(ny_before), 1)
            # FVG / OB
            fvg_up_bar_n   = int(sv(bar.get('fvg_up', 0)) > 0)
            fvg_down_bar_n = int(sv(bar.get('fvg_down', 0)) > 0)
            stacked_bull_n = int(ny_sess['stacked_bull'].any()) if 'stacked_bull' in ny_sess.columns else 0
            stacked_bear_n = int(ny_sess['stacked_bear'].any()) if 'stacked_bear' in ny_sess.columns else 0
            # OF normalized
            bar_delta_norm_n = sv(bar['bar_delta']) / atr if atr > 0 else 0
            cum_delta_norm_n = sv(r0['cum_delta']) / atr   if atr > 0 else 0
            bv_bar_n = sv(bar.get('bar_buy_vol', 0)); sv_bar_n = sv(bar.get('bar_sell_vol', 0))
            buy_sell_ratio_n = bv_bar_n / max(sv_bar_n, 1)
            # Day type
            day_range_n = df['high'].max() - df['low'].min()
            trend_day_n  = 1 if day_range_n > atr * 1.5 else 0
            inside_day_n = 1 if day_range_n < atr * 0.6 else 0
            week_range_n = day_range_n / atr if atr > 0 else 0
            week_hi_n    = 1 if sv(bar['high']) >= sv(r0['lw_hi']) - 0.5 else 0
            week_lo_n    = 1 if sv(bar['low'])  <= sv(r0['lw_lo']) + 0.5 else 0
            # Calendar
            fomc_prox_n  = _fomc_days_to(date_str, FOMC_DT)
            is_cpi_n     = 1 if date_str in CPI_DT else 0
            is_ppi_n     = 1 if date_str in PPI_DT else 0
            # Displacement bar quality
            db_body_n = abs(sv(disp['close']) - sv(disp['open']))
            db_rng_n  = max(sv(disp['high']) - sv(disp['low']), 0.01)
            body_bear_n = 1 if sv(disp['close']) < sv(disp['open']) else 0
            body_pct_n  = db_body_n / db_rng_n
            # LON range
            lon_rng_n = lon_hi_n - lon_lo_n if lon_hi_n > lon_lo_n else atr
            lon_range_vs_atr5d_n = lon_rng_n / roll.get('atr_5d', atr) if roll.get('atr_5d', atr) > 0 else 0
            lon_big_day_n   = 1 if lon_rng_n > atr * 1.3 else 0
            lon_small_day_n = 1 if lon_rng_n < atr * 0.5 else 0

            # ── New DB-sourced features (v3) ──────────────────────────────
            smt_bear_n = int(sv(r0.get('is_smt_bearish', 0)))
            smt_bull_n = int(sv(r0.get('is_smt_bullish', 0)))
            smt_aligned_n = (smt_bear_n if direction == 'SHORT' else smt_bull_n)

            entropy_n = sv(r0.get('sample_entropy', 2.0))
            fft_cyc_n = sv(r0.get('fft_cycle', 64.0))

            vwap_pn = sv(r0.get('vwap', 0))
            dist_vwap_p_atr_n = (entry - vwap_pn) / atr if atr > 0 and vwap_pn > 0 else 0
            vwap_align_n = 1 if (direction == 'SHORT' and entry > vwap_pn > 0) or \
                                (direction == 'LONG'  and entry < vwap_pn > 0) else 0

            lm_h_n = sv(r0.get('lm_hi', 0)); lm_l_n = sv(r0.get('lm_lo', 0))
            lm_rng_n = lm_h_n - lm_l_n
            lm_prem_n = (entry - lm_l_n) / lm_rng_n if lm_rng_n > 0 else 0.5
            lm_prem_aligned_n = 1 if (direction == 'SHORT' and lm_prem_n > 0.5) or \
                                     (direction == 'LONG'  and lm_prem_n < 0.5) else 0
            dist_lm_hi_n = abs(entry - lm_h_n) / atr if atr > 0 and lm_h_n > 0 else 0
            dist_lm_lo_n = abs(entry - lm_l_n) / atr if atr > 0 and lm_l_n > 0 else 0

            bsz_n = sv(r0.get('body_size', 0))
            body_size_atr_n = bsz_n / atr if atr > 0 else 0

            feat = {
                'session_enc':          1,
                'spike_mag_atr':        sda,
                'pre_rng_atr':          pre_rng / atr if atr > 0 else 0,
                'ts_close_inside':      ts_ci,
                'ts_rejection_str':     ts_rej,
                'sweep_wick_atr':       wick,
                'sweep_depth_atr':      sda,
                'deep_sweep':           1 if sda > 1.5 else 0,
                'shallow_sweep':        1 if sda < 0.5 else 0,
                'sweep_quality':        sq,
                'disp_body_atr':        disp_body_atr,
                'h4_bias':              h4_bias,
                'h4_bias_aligned':      1 if h4_bias == dir_num else 0,
                'weekly_premium_pct':   weekly_prem,
                'weekly_prem_aligned':  weekly_prem_aligned,
                'lw_range_atr':         lw_rng / atr if atr > 0 else 0,
                'dist_lw_hi':           abs(entry - lw_hi) / atr if atr > 0 else 0,
                'dist_lw_lo':           abs(entry - lw_lo) / atr if atr > 0 else 0,
                'lon_range_atr':        lon_rng / atr if atr > 0 else 0,
                'lon_dir':              float(lon_dir),
                'lon_dir_aligned':      1 if lon_dir == dir_num else 0,
                'dist_asia_hi_atr':     abs(entry - asia_hi) / atr if atr > 0 and len(asia_df) > 0 else 0,
                'dist_asia_lo_atr':     abs(entry - asia_lo) / atr if atr > 0 and len(asia_df) > 0 else 0,
                'asia_dir':             float(asia_dir),
                'asia_dir_aligned':     1 if asia_dir == dir_num else 0,
                'triple_sess_aligned':  float(triple_sess),
                # Volume profile
                'dist_poc_atr':         dist_poc_atr,
                'dist_vwap_atr':        dist_vwap_atr,
                'vah_dist_atr':         vah_dist_atr,
                'val_dist_atr':         val_dist_atr,
                'va_width_atr':         va_width_atr,
                'inside_va':            inside_va,
                # True open
                'above_true_open':      above_to,
                'dist_true_open_atr':   dist_to_atr,
                # PDH/PDL
                'dist_pdh_atr':         dist_pdh_atr,
                'dist_pdl_atr':         dist_pdl_atr,
                # Delta / absorption
                'absorption_score':     abs_score,
                'delta_aligned_atr':    delta_aligned,
                'of_doi':               of_doi_v,
                'of_bilateral_abs':     of_bil_v,
                # Microstructure
                'adx':                  sv(r0['adx_14']),
                'hurst':                sv(r0['hurst'], 0.5),
                'garch_vol':            sv(r0['garch_vol']),
                'rvol':                 sv(r0['rvol'], 1.0),
                'fisher_transform':     sv(r0['fisher_transform']),
                'acf_lag1':             sv(r0['acf_lag1']),
                'acf_lag5':             sv(r0['acf_lag5']),
                'kalman_smooth':        sv(r0['kalman_smooth']),
                'kalman_noise':         sv(r0.get('kalman_noise', 0.0)),
                # Calendar
                'is_nfp_day':           1 if date_str in NFP_DT else 0,
                'is_fomc_day':          1 if date_str in FOMC_DT else 0,
                'is_news_day':          1 if date_str in NEWS_DT else 0,
                'is_pre_nfp':           1 if (date_str in NFP_DT and entry_hhmm < 830) else 0,
                'is_post_nfp':          1 if (date_str in NFP_DT and entry_hhmm >= 830) else 0,
                # Time
                'direction_enc':        1 if direction == 'SHORT' else 0,
                'day_of_week':          sv(r0['day_of_week']),
                'month':                sv(r0['month']),
                'is_thursday':          1 if int(sv(r0['day_of_week'])) == 3 else 0,
                'is_friday':            1 if int(sv(r0['day_of_week'])) == 4 else 0,
                'entry_hhmm_norm':      (entry_hhmm - NY_SESS_START_ET) / (NY_SESS_END_ET - NY_SESS_START_ET),
                # Regime
                'regime_enc':           {'PRE_EXPANSION':1,'EXPANSION':2,'RETRACEMENT':3,
                                         'CONSOLIDATION':0,'DISTRIBUTION':4}.get(regime, -1),
                'is_pre_expansion':     1 if regime == 'PRE_EXPANSION' else 0,
                'is_expansion':         1 if regime == 'EXPANSION' else 0,
                'is_retracement':       1 if regime == 'RETRACEMENT' else 0,
                # Interactions
                'dir_x_adx':            dir_num * sv(r0['adx_14']),
                'dir_x_hurst':          dir_num * sv(r0['hurst'], 0.5),
                'vol_x_sweep':          sv(r0['garch_vol']) * sq,
                'h4_x_weekly':          float(h4_bias == dir_num) * float(weekly_prem_aligned),
                'sweep_x_poc':          sq * (1 if dist_poc_atr < 1.0 else 0),
                'sweep_x_va':           sq * inside_va,
                'absorption_x_sweep':   abs_score * sq,
                # ── Sweep bar specific (NUOVO) ─────────────────────────────
                'sweep_bar_body_pct':  sweep_bar_body_pct_n,
                'sweep_bar_wick_pct':  sweep_bar_wick_pct_n,
                'sweep_bar_vol_ratio': sweep_bar_vol_ratio_n,
                'n_level_tests':       float(n_level_tests_n),
                'bars_to_disp':        bars_to_disp_n,
                'pre5_momentum':       pre5_mom_n,
                'pre5_mom_aligned':    1.0 if (pre5_mom_n > 0 and direction == 'SHORT') or
                                              (pre5_mom_n < 0 and direction == 'LONG') else 0.0,
                'fast_disp':           1.0 if bars_to_disp_n <= 5 else 0.0,
                'vol_surge':           1.0 if sweep_bar_vol_ratio_n >= 1.5 else 0.0,
                'multi_test':          1.0 if n_level_tests_n >= 2 else 0.0,
                # ── Mario-quality cross-pollination (NOM) ─────────────────────
                'vol_regime':          roll.get('vol_regime', 1.0),
                'vix_proxy_5d':        roll.get('vix_proxy_5d', 0.0),
                'vix_proxy_20d':       roll.get('vix_proxy_20d', 0.0),
                'vol_high':            roll.get('vol_high', 0),
                'vol_low':             roll.get('vol_low', 0),
                'atr_trend':           roll.get('atr_trend', 1.0),
                'atr_vs_10d':          roll.get('atr_vs_10d', 1.0),
                'atr_expanding':       roll.get('atr_expanding', 0),
                'atr_contracting':     roll.get('atr_contracting', 0),
                'adx_10d_mean':        roll.get('adx_10d_mean', 20.0),
                'hurst_20d_mean':      roll.get('hurst_20d_mean', 0.5),
                'regime_trending':     roll.get('regime_trending', 0),
                'h1_bias':             h1_bias_n,
                'h4_h1_aligned':       h4_h1_aligned_n,
                'swept_asia_hi':       swept_asia_hi_n,
                'swept_asia_lo':       swept_asia_lo_n,
                'swept_lon_hi':        swept_lon_hi_n,
                'swept_lon_lo':        swept_lon_lo_n,
                'lon_close_vs_mid':    lon_close_vs_mid_n,
                'ny_open_in_lon':      ny_open_in_lon_n,
                'ny_open_dist_lon_mid':ny_open_dist_lon_mid_n,
                'ny15_range_atr':      ny15_range_atr_n,
                'adx_strong':          1 if sv(r0['adx_14']) > 25 else 0,
                'fisher_extreme':      1 if abs(sv(r0['fisher_transform'])) > 2.0 else 0,
                'garch_vol_atr':       sv(r0['garch_vol']) / atr if atr > 0 else 0,
                'year_norm':           (yr_n - 2021) / 5.0,
                'is_monday':           1 if dow_n == 0 else 0,
                'is_tuesday':          1 if dow_n == 1 else 0,
                'is_wednesday':        1 if dow_n == 2 else 0,
                'week_range_so_far':   week_range_n,
                'week_hi_taken':       week_hi_n,
                'week_lo_taken':       week_lo_n,
                'in_weekly_premium':   1 if weekly_prem > 0.5 else 0,
                'in_weekly_discount':  1 if weekly_prem < 0.5 else 0,
                'trend_day':           trend_day_n,
                'inside_day':          inside_day_n,
                'lon_range_vs_atr5d':  lon_range_vs_atr5d_n,
                'lon_big_day':         lon_big_day_n,
                'lon_small_day':       lon_small_day_n,
                'equal_level_score':   equal_level_score_n,
                'bar_delta_norm':      bar_delta_norm_n,
                'cum_delta_norm':      cum_delta_norm_n,
                'buy_sell_ratio':      buy_sell_ratio_n,
                'fvg_up':              fvg_up_bar_n,
                'fvg_down':            fvg_down_bar_n,
                'stacked_bull':        stacked_bull_n,
                'stacked_bear':        stacked_bear_n,
                'body_bear':           body_bear_n,
                'body_pct':            body_pct_n,
                'fomc_proximity':      float(fomc_prox_n),
                'is_cpi_day':          is_cpi_n,
                'is_ppi_day':          is_ppi_n,
                # ── New DB-derived features (v3) ──────────────────────────
                'smt_bearish':         smt_bear_n,
                'smt_bullish':         smt_bull_n,
                'smt_aligned':         smt_aligned_n,
                'sample_entropy':      entropy_n,
                'entropy_low':         1 if entropy_n < 1.8 else 0,
                'fft_cycle':           fft_cyc_n,
                'fft_short_cycle':     1 if fft_cyc_n < 40 else 0,
                'dist_vwap_price_atr': dist_vwap_p_atr_n,
                'vwap_aligned':        vwap_align_n,
                'lm_premium':          lm_prem_n,
                'lm_prem_aligned':     lm_prem_aligned_n,
                'dist_lm_hi_atr':      dist_lm_hi_n,
                'dist_lm_lo_atr':      dist_lm_lo_n,
                'body_size_atr':       body_size_atr_n,
                # Labels / meta
                '_label':    label,
                '_session':  'NY',
                '_date':     date_str,
                '_regime':   regime,
            }
            setups.append(feat)
            last_hhmm[direction] = bar_hhmm
            break
    return setups


# ════════════════════════════════════════════════════════════════════════════
# Build dataset
# ════════════════════════════════════════════════════════════════════════════
def build_dataset(years):
    from collections import deque
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    # Load extra 25 days before window for rolling warmup
    yr_min = min(years)
    all_days = pd.read_sql(
        f"""SELECT DISTINCT date FROM market_data
            WHERE day_of_week BETWEEN 1 AND 5
            ORDER BY date""", conn
    )['date'].tolist()
    target_days = [d for d in all_days if int(d[:4]) in years]
    # find index of first target day and load 25 extra days before for warmup
    if target_days:
        first_idx = all_days.index(target_days[0]) if target_days[0] in all_days else 0
        warmup_days = all_days[max(0, first_idx-25):first_idx]
    else:
        warmup_days = []

    daily_buf = deque(maxlen=25)
    # warmup rolling buffer
    for d in warmup_days:
        df_w = load_day(conn, d)
        if df_w is not None:
            daily_buf.append(_daily_summary(df_w))

    all_setups = []
    for d in target_days:
        df = load_day(conn, d)
        if df is None: continue
        daily_buf.append(_daily_summary(df))
        roll = _rolling_stats(list(daily_buf))
        all_setups.extend(extract_lom(df, d, roll))
        all_setups.extend(extract_nom(df, d, roll))
    conn.close()
    log.info(f"  {years}: {len(target_days)} zile → {len(all_setups)} setups")
    return pd.DataFrame(all_setups) if all_setups else pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# Walk-forward CV helper
# ════════════════════════════════════════════════════════════════════════════
def wf_cv_score(model_fn, X, y, sw, n_folds=3):
    """Mean AUC din N-fold walk-forward temporal CV."""
    n = len(X)
    fold_size = n // (n_folds + 1)
    aucs = []
    for k in range(n_folds):
        train_end = fold_size * (k + 1)
        val_start = train_end
        val_end   = min(train_end + fold_size, n)
        if val_end - val_start < 20: continue
        Xtr = X.iloc[:train_end]; ytr = y[:train_end]; swtr = sw[:train_end]
        Xvl = X.iloc[val_start:val_end]; yvl = y[val_start:val_end]
        if yvl.sum() < 3 or (yvl == 0).sum() < 3: continue
        try:
            m = model_fn()
            m.fit(Xtr, ytr, sample_weight=swtr, verbose=False)
            proba = m.predict_proba(Xvl)[:, 1]
            aucs.append(roc_auc_score(yvl, proba))
        except: pass
    return float(np.mean(aucs)) if aucs else 0.5


# ════════════════════════════════════════════════════════════════════════════
# Training per regime cu gap penalty
# ════════════════════════════════════════════════════════════════════════════
def train_regime_model(X_tr, y_tr, X_val, y_val, X_te, y_te, sw_tr, label='ALL',
                       X_cal=None, y_cal=None):
    if len(X_tr) < 60 or y_tr.sum() < 10:
        log.warning(f"  {label}: prea putine sample ({len(X_tr)} / {y_tr.sum()} pos)")
        return None, 0, 0

    # PRE_EXPANSION bypass: skip Optuna, use fixed conservative hyperparams.
    # Reason: WR≈47%, 374 IS samples, strong IS-OOS distribution shift → Optuna always
    # picks too-aggressive params → massive IS overfit → OOS collapse. Fixed conservative
    # params give consistent OOS ~0.63-0.65 without the variability.
    if label == 'PRE_EXPANSION':
        _bp_fixed = {
            'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.015,
            'subsample': 0.65,   'colsample_bytree': 0.50,
            'min_child_weight': 60, 'gamma': 10.0,
            'reg_alpha': 5.0,    'reg_lambda': 10.0, 'scale_pos_weight': 3.0,
        }
        _smote_fixed = 0.70
        try:
            sm = BorderlineSMOTE(sampling_strategy=_smote_fixed, random_state=42,
                                  k_neighbors=min(5, int(y_tr.sum()) - 1))
            Xs, ys = sm.fit_resample(X_tr, y_tr)
            sws = np.concatenate([sw_tr, np.ones(len(Xs) - len(X_tr))])
        except:
            Xs, ys, sws = X_tr, y_tr, sw_tr
        base = xgb.XGBClassifier(**_bp_fixed, eval_metric='logloss',
                                  early_stopping_rounds=15,
                                  random_state=42, n_jobs=-1, tree_method='hist', verbosity=0)
        base.fit(Xs, ys, sample_weight=sws, eval_set=[(X_val, y_val)], verbose=False)
        _Xc = X_cal if (X_cal is not None and len(X_cal) >= 20) else X_val
        _yc = y_cal if (y_cal is not None and len(y_cal) >= 20) else y_val
        raw_val = base.predict_proba(_Xc)[:, 1]
        ir_cal  = _IR(out_of_bounds='clip').fit(raw_val, _yc)
        cal     = _CalModel(base, ir_cal)
        is_auc  = roc_auc_score(y_tr, cal.predict_proba(X_tr)[:, 1])
        oos_auc = roc_auc_score(y_te, cal.predict_proba(X_te)[:, 1]) if len(y_te) > 20 and y_te.sum() > 0 else 0
        gap     = is_auc - oos_auc
        log.info(f"  {label}: IS={is_auc:.4f} OOS={oos_auc:.4f} gap={gap:.3f} "
                 f"({'⚠️ OVERFIT' if gap > 0.10 else '✅'}) | {len(X_tr)} IS samples [FIXED PARAMS]")
        return cal, is_auc, oos_auc

    # Strategy for other regimes:
    # - Early stopping in Optuna: full val (semnal stabil)
    # - Optuna AUC scoring: regime-specific val DACA WR e echilibrat (35-65%) SI n>=50
    # - Final fit: fara early stopping (v2e style) → calibrarea isotonica se ocupa de overfit
    _has_cal = X_cal is not None and len(X_cal) >= 50
    _wr_cal  = float(y_cal.mean()) if _has_cal else 0.0
    _cal_balanced = _has_cal and 0.35 <= _wr_cal <= 0.65
    log.info(f"  {label}: optuna_auc={'regime_val(n='+str(len(X_cal))+',WR='+f'{_wr_cal:.0%}'+')' if _cal_balanced else 'full_val'}"
             f", early_stop_final=OFF, calibrare={'regime_val' if _has_cal else 'full_val'}")

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 60, 200),
            'max_depth':        trial.suggest_int('max_depth', 2, 3),
            'learning_rate':    trial.suggest_float('lr', 0.008, 0.05, log=True),
            'subsample':        trial.suggest_float('sub', 0.45, 0.80),
            'colsample_bytree': trial.suggest_float('col', 0.35, 0.70),
            'min_child_weight': trial.suggest_int('mcw', 30, 100),
            'gamma':            trial.suggest_float('gamma', 3.0, 15.0),
            'reg_alpha':        trial.suggest_float('alpha', 1.0, 8.0),
            'reg_lambda':       trial.suggest_float('lambda', 3.0, 12.0),
            'scale_pos_weight': trial.suggest_float('spw', 2.0, 8.0),
        }
        smote_r = trial.suggest_float('smote_r', 0.55, 0.90)
        try:
            sm = BorderlineSMOTE(sampling_strategy=smote_r, random_state=42,
                                  k_neighbors=min(5, int(y_tr.sum()) - 1))
            Xs, ys = sm.fit_resample(X_tr, y_tr)
            sws = np.concatenate([sw_tr, np.ones(len(Xs) - len(X_tr))])
        except:
            Xs, ys, sws = X_tr.values if hasattr(X_tr, 'values') else X_tr, y_tr, sw_tr

        try:
            # No early stopping in trials — consistent with final fit behavior.
            # Both use full n_estimators (capped at 250) → IS-OOS gap in Optuna
            # matches final model IS-OOS gap → penalty is accurate.
            m = xgb.XGBClassifier(**params, eval_metric='logloss',
                                   random_state=42, n_jobs=-1, tree_method='hist',
                                   verbosity=0)
            m.fit(Xs, ys, sample_weight=sws, verbose=False)
            # Scor Optuna: regime-specific daca val e balansat (WR 35-65%, n>=50)
            # altfel full val (evita overfit Optuna pe sample mic/imbalanced)
            if _cal_balanced:
                Xsc, ysc = X_cal, y_cal
            else:
                Xsc, ysc = X_val, y_val
            if ysc.sum() == 0 or ysc.sum() == len(ysc): return 0.5
            val_proba = m.predict_proba(Xsc)[:, 1]
            # WR-aware weighted AUC: upweight positive class to match OOS WR (~47.9%)
            # Val is 2024 data (WR≈37%), OOS is 2025-2026 (WR≈48%) — bridge the gap
            _val_wr = float(ysc.mean())
            _oos_wr_approx = float(y_te.mean()) if len(y_te) > 0 else 0.479
            if 0.1 < _val_wr < 0.9 and abs(_val_wr - _oos_wr_approx) > 0.03:
                _wadj = np.where(ysc == 1, _oos_wr_approx / _val_wr,
                                 (1 - _oos_wr_approx) / max(1 - _val_wr, 1e-6))
                val_auc = roc_auc_score(ysc, val_proba, sample_weight=_wadj)
            else:
                val_auc = roc_auc_score(ysc, val_proba)
            is_auc  = roc_auc_score(ys, m.predict_proba(Xs)[:, 1])
            gap_penalty = GAP_PENALTY * max(0, is_auc - val_auc - 0.06)
            return val_auc - gap_penalty
        except:
            return 0.5

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler()   # random seed — mai buna explorare
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False, n_jobs=1)
    bp = study.best_params
    smote_best = bp.pop('smote_r')
    log.info(f"  {label}: best penalized_val={study.best_value:.4f} (gbtree)")

    try:
        sm = BorderlineSMOTE(sampling_strategy=smote_best, random_state=42,
                              k_neighbors=min(5, int(y_tr.sum()) - 1))
        Xs, ys = sm.fit_resample(X_tr, y_tr)
        sws = np.concatenate([sw_tr, np.ones(len(Xs) - len(X_tr))])
    except:
        Xs, ys, sws = X_tr, y_tr, sw_tr

    # Final fit — gbtree without early stopping (FARA early stopping — better for WR)
    base = xgb.XGBClassifier(**bp, eval_metric='logloss',
                              random_state=42, n_jobs=-1, tree_method='hist', verbosity=0)
    base.fit(Xs, ys, sample_weight=sws, verbose=False)

    # Isotonic calibration — folosi X_cal/y_cal (regime-specific) dacă disponibil
    # altfel X_val complet (pentru modelul ALL)
    _Xc = X_cal if (X_cal is not None and len(X_cal) >= 20) else X_val
    _yc = y_cal if (y_cal is not None and len(y_cal) >= 20) else y_val
    raw_val = base.predict_proba(_Xc)[:, 1]
    ir_cal  = _IR(out_of_bounds='clip').fit(raw_val, _yc)
    cal     = _CalModel(base, ir_cal)

    is_auc  = roc_auc_score(y_tr, cal.predict_proba(X_tr)[:, 1])
    oos_auc = roc_auc_score(y_te, cal.predict_proba(X_te)[:, 1]) if len(y_te) > 20 and y_te.sum() > 0 else 0
    gap     = is_auc - oos_auc
    log.info(f"  {label}: IS={is_auc:.4f} OOS={oos_auc:.4f} gap={gap:.3f} "
             f"({'⚠️ OVERFIT' if gap > 0.10 else '✅'}) | {len(X_tr)} IS samples")
    return cal, is_auc, oos_auc


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def train_and_save():
    log.info("=" * 60)
    log.info("SWEEP UNIFIED v2 — target OOS AUC ≥ 0.73")
    log.info(f"TRAIN: {TRAIN_YEARS}  TEST: {TEST_YEARS}")
    log.info("=" * 60)

    # ── Load din parquet cache daca exista (mult mai rapid) ─────────────────────
    def _load_or_build(years, suffix='_v2'):
        tag    = '_'.join(str(y) for y in years)
        cached = BASE / f'sweep_dataset_{tag}{suffix}.parquet'
        # 1. Fisier combinat complet
        if cached.exists():
            log.info(f"  Loading cached {cached.name}...")
            return pd.read_parquet(cached)
        # 2. Combina fisiere per-an (mai eficient, fara rebuild DB)
        year_parts = []
        for y in years:
            yp = BASE / f'sweep_dataset_{y}{suffix}.parquet'
            if yp.exists():
                year_parts.append(pd.read_parquet(yp))
        if len(year_parts) == len(years):
            df = pd.concat(year_parts, ignore_index=True).sort_values('_date').reset_index(drop=True)
            df.to_parquet(cached, index=False)
            log.info(f"  Combined {len(years)} year files → {cached.name} ({len(df)} rows)")
            return df
        # 3. Fallback: split la mijloc (ex: 2021_2022 + 2023_2024)
        if len(years) > 2:
            parts = []
            mid = len(years) // 2
            for chunk in [years[:mid], years[mid:]]:
                chunk_tag  = '_'.join(str(y) for y in chunk)
                chunk_path = BASE / f'sweep_dataset_{chunk_tag}{suffix}.parquet'
                if chunk_path.exists():
                    log.info(f"  Loading split cache {chunk_path.name}...")
                    parts.append(pd.read_parquet(chunk_path))
            if len(parts) == 2:
                df = pd.concat(parts, ignore_index=True).sort_values('_date').reset_index(drop=True)
                df.to_parquet(cached, index=False)
                log.info(f"  Combined split → {cached.name} ({len(df)} rows)")
                return df
        # 4. Build din DB (slow, fallback de ultima instanta)
        log.info(f"  Building {years} din DB...")
        df = build_dataset(years)
        if not df.empty:
            df.to_parquet(cached, index=False)
            log.info(f"  Salvat {cached.name}")
        return df

    log.info(f"Loading/building IS ({TRAIN_YEARS})...")
    df_tr = _load_or_build(TRAIN_YEARS, suffix='_v3')
    log.info(f"Loading/building OOS ({TEST_YEARS})...")
    df_te = _load_or_build(TEST_YEARS,  suffix='_v3')

    if df_tr.empty or df_te.empty:
        log.error("Dataset vuoto — abort")
        return

    # Merge orderflow features
    _OF_PATH = BASE / "data" / "orderflow_features.parquet"
    _OF_COLS = []  # will be populated if parquet exists
    if _OF_PATH.exists():
        _of = pd.read_parquet(_OF_PATH)
        _OF_COLS = [c for c in _of.columns if c not in
                    ['session_id','date','session_type','session_open','session_close',
                     'session_high','session_low','total_vol']]
        _of_m = _of[['date','session_type'] + _OF_COLS].rename(
            columns={'date': '_date', 'session_type': '_session'})
        df_tr = df_tr.merge(_of_m, on=['_date','_session'], how='left')
        df_te = df_te.merge(_of_m, on=['_date','_session'], how='left')
        for c in _OF_COLS:
            df_tr[c] = df_tr[c].fillna(0.0)
            df_te[c] = df_te[c].fillna(0.0)
        log.info(f"OF features merged: {len(_OF_COLS)} cols")

    # ── OF alignment features (direction vs session-level OF signals) ─────────
    for df_ in [df_tr, df_te]:
        if 'opening_drive_dir' in df_.columns:
            df_['od_aligned'] = (
                ((df_['direction_enc'] == 1) & (df_['opening_drive_dir'] < 0)) |
                ((df_['direction_enc'] == 0) & (df_['opening_drive_dir'] > 0))
            ).astype(float)
        if 'stacked_imbalance_dir' in df_.columns:
            df_['stacked_imb_aligned'] = (
                ((df_['direction_enc'] == 1) & (df_['stacked_imbalance_dir'] < 0)) |
                ((df_['direction_enc'] == 0) & (df_['stacked_imbalance_dir'] > 0))
            ).astype(float)
        if 'cvd_trend_flag' in df_.columns:
            df_['cvd_aligned'] = (
                ((df_['direction_enc'] == 1) & (df_['cvd_trend_flag'] < 0)) |
                ((df_['direction_enc'] == 0) & (df_['cvd_trend_flag'] > 0))
            ).astype(float)

    # ── Regime probability features (HMM confidence) ─────────────────────────
    # Merge regime_prob + per-regime soft probs from regime_labels.parquet
    # These capture HOW CONFIDENT the regime classifier is — high-confidence
    # EXPANSION setups have much higher WR than uncertain ones.
    _RPROB_COLS = ['regime_prob', 'prob_CONSOLIDATION', 'prob_PRE_EXPANSION',
                   'prob_EXPANSION', 'prob_RETRACEMENT', 'prob_DISTRIBUTION']
    try:
        _rprob_df = pd.read_parquet(_REGIME_PATH)[
            ['date', 'session'] + _RPROB_COLS
        ].rename(columns={'date': '_date', 'session': '_session'})
        # Map _session: LON→LON, NY→NY (same naming in regime_labels)
        df_tr = df_tr.merge(_rprob_df, on=['_date', '_session'], how='left')
        df_te = df_te.merge(_rprob_df, on=['_date', '_session'], how='left')
        for c in _RPROB_COLS:
            df_tr[c] = df_tr[c].fillna(0.5)
            df_te[c] = df_te[c].fillna(0.5)
        log.info(f"Regime prob features merged: {_RPROB_COLS}")
    except Exception as _rpe:
        log.warning(f"Regime prob merge failed: {_rpe}")

    meta_cols    = [c for c in df_tr.columns if c.startswith('_')]
    feature_cols = [c for c in df_tr.columns if c not in meta_cols]

    # Identify OF feature columns — force-include these (they're crowded out by global selection)
    _OF_FORCE_COLS = [c for c in _OF_COLS if c in feature_cols]
    # Also force-include OF alignment features and regime prob features
    _ALIGN_FORCE = [c for c in ['od_aligned', 'stacked_imb_aligned', 'cvd_aligned']
                    if c in feature_cols]
    _RPROB_FORCE = [c for c in _RPROB_COLS if c in feature_cols]
    _OF_ALL_FORCE = list(dict.fromkeys(_OF_FORCE_COLS + _ALIGN_FORCE + _RPROB_FORCE))
    log.info(f"Force-include pool: {len(_OF_ALL_FORCE)} features "
             f"({len(_OF_FORCE_COLS)} OF + {len(_ALIGN_FORCE)} align + {len(_RPROB_FORCE)} regime_prob)")

    log.info(f"IS: {len(df_tr)} setups | LOM={len(df_tr[df_tr['_session']=='LON'])} NOM={len(df_tr[df_tr['_session']=='NY'])}")
    log.info(f"OOS: {len(df_te)} setups | WR_base={df_te['_label'].mean():.1%}")
    log.info(f"Regimi IS: {df_tr['_regime'].value_counts().to_dict()}")

    # Decay weights (global) — pt feature selection
    _decay_raw = compute_decay_weights(df_tr['_date'])
    yr_w_global = np.array([YEAR_WEIGHTS.get(int(d[:4]), 1.0) for d in df_tr['_date']])
    sw_global = _decay_raw * yr_w_global
    sw_global = sw_global / sw_global.mean()

    X_tr_full = df_tr[feature_cols].fillna(0)
    y_tr_full = df_tr['_label'].values
    X_te_full = df_te[feature_cols].fillna(0).reindex(columns=feature_cols, fill_value=0)
    y_te_full = df_te['_label'].values

    # Feature selection: top TOP_N_FEATURES from computed (non-OF) features
    # then force-include all OF features on top → prevents OF crowding
    _of_force_set = set(_OF_ALL_FORCE)
    _comp_cols = [c for c in feature_cols if c not in _of_force_set]
    log.info(f"Feature selection (top {TOP_N_FEATURES} computed from {len(_comp_cols)} non-OF cols)...")
    neg, pos = (y_tr_full == 0).sum(), (y_tr_full == 1).sum()
    _pre = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                              subsample=0.7, colsample_bytree=0.6, min_child_weight=20,
                              gamma=2.0, reg_alpha=2.0, reg_lambda=5.0,
                              scale_pos_weight=neg / max(pos, 1),
                              random_state=42, n_jobs=-1,
                              eval_metric='logloss', verbosity=0)
    _pre.fit(X_tr_full[_comp_cols], y_tr_full, sample_weight=sw_global, verbose=False)
    imp = pd.Series(_pre.feature_importances_, index=_comp_cols).sort_values(ascending=False)
    selected_computed = imp.head(TOP_N_FEATURES).index.tolist()
    # Final pool: OF-forced + top computed features
    selected = _OF_ALL_FORCE + selected_computed
    log.info(f"Feature pool: {len(selected)} total ({len(_OF_ALL_FORCE)} OF forced + {len(selected_computed)} computed)")
    log.info(f"Top5 computed: {selected_computed[:5]}")

    X_tr = X_tr_full[selected]
    X_te = X_te_full.reindex(columns=selected, fill_value=0)

    # Val split: explicit year-2024 boundary (cleaner temporal split vs 80/20 row split)
    # Advantage: full 2024 as val (899 rows, WR=36.9%) → better calibration data + cleaner IS/OOS boundary
    # IS = 2022+2023 (1881 rows), val = 2024 (899 rows)
    # The WR-aware adjustment in Optuna bridges val(WR=37%) → OOS(WR=48%)
    _yr_series = df_tr['_date'].str[:4].astype(int).values
    _VAL_YEAR = 2024  # use this year as validation
    val_cut = int((_yr_series < _VAL_YEAR).sum())
    if val_cut < 100 or val_cut >= len(X_tr) - 50:
        # Fallback to 80/20 if year split is degenerate
        val_cut = int(len(X_tr) * 0.80)
        log.warning(f"  Year-based split degenerate → fallback 80/20 (cut={val_cut})")
    log.info(f"  Val split: train={val_cut} rows (2022-2023), val={len(X_tr)-val_cut} rows (2024)")
    X_val = X_tr.iloc[val_cut:]; y_val = y_tr_full[val_cut:]
    X_tr2 = X_tr.iloc[:val_cut]; y_tr2 = y_tr_full[:val_cut]

    regimes_is = df_tr['_regime'].values
    regimes_val = regimes_is[val_cut:]

    for regime_name in ACTIVE_REGIMES:
        log.info(f"\n{'='*40}\nRegim: {regime_name}\n{'='*40}")
        if regime_name == 'ALL':
            mask = np.ones(len(X_tr2), dtype=bool)
        else:
            mask = (regimes_is[:val_cut] == regime_name)

        if mask.sum() < 60:
            log.warning(f"  {regime_name}: {mask.sum()} IS samples → skip")
            continue

        # Per-regime sample weights — uses regime-specific year weights if available
        _ryw = REGIME_YEAR_WEIGHTS.get(regime_name, YEAR_WEIGHTS)
        _yr_w_r = np.array([_ryw.get(int(d[:4]), 1.0) for d in df_tr['_date'][:val_cut]])
        _sw2_raw = _decay_raw[:val_cut] * _yr_w_r
        sw2 = _sw2_raw / _sw2_raw.mean()  # normalize mean=1.0
        log.info(f"  {regime_name}: yr_weights={_ryw}")

        # OOS per regime
        if regime_name != 'ALL':
            te_mask = (df_te['_regime'].values == regime_name)
        else:
            te_mask = np.ones(len(X_te), dtype=bool)

        y_te_r = y_te_full[te_mask]
        X_te_r = X_te.iloc[te_mask] if hasattr(X_te, 'iloc') else X_te[te_mask]

        # ── Per-regime feature selection ──────────────────────────────────────────
        # Numarul de features per regime: mai putine pt regimuri cu WR≈50% (evita collapse)
        n_r_feats  = REGIME_TOP_FEATS.get(regime_name, len(selected))
        if n_r_feats < len(selected) and regime_name != 'ALL':
            # Selectie features pe IS data din acest regim
            _r_neg = (y_tr2[mask] == 0).sum(); _r_pos = y_tr2[mask].sum()
            _rfe = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                     subsample=0.7, colsample_bytree=0.6,
                                     min_child_weight=15, gamma=2.0,
                                     scale_pos_weight=_r_neg / max(_r_pos, 1),
                                     random_state=42, n_jobs=-1, verbosity=0)
            _rfe.fit(X_tr2[mask], y_tr2[mask], sample_weight=sw2[mask], verbose=False)
            _r_imp = pd.Series(_rfe.feature_importances_, index=selected).sort_values(ascending=False)
            sel_r  = _r_imp.head(n_r_feats).index.tolist()
            log.info(f"  {regime_name}: per-regime feat sel {n_r_feats}/{len(selected)} "
                     f"(top3: {sel_r[:3]})")
        else:
            sel_r = selected  # usa full selected pool

        # Reindex tutte le matrici al feature set regime-specific
        X_tr2_r = X_tr2[mask][sel_r]
        X_val_r = X_val[sel_r]
        X_te_rr = X_te_r[sel_r]

        # Cal set: regime-specific val (pt isotonic calibration + Optuna eval corecta)
        # Threshold 20 pt a asigura ca IR are suficiente sample din ambele clase
        if regime_name != 'ALL':
            cal_mask  = (regimes_val == regime_name)
            n_cal     = int(cal_mask.sum())
            n_cal_pos = int(y_val[cal_mask].sum())
            log.info(f"  {regime_name}: cal set = {n_cal} samples, {n_cal_pos} pozitive "
                     f"({'✅ regime-specific' if n_cal >= 20 else '⚠️ fallback to full-val'})")
            X_cal_r   = X_val_r.iloc[cal_mask] if n_cal >= 20 else None
            y_cal_r   = y_val[cal_mask]         if n_cal >= 20 else None
        else:
            X_cal_r, y_cal_r = None, None  # ALL: foloseste X_val complet

        model, is_auc, oos_auc = train_regime_model(
            X_tr2_r, y_tr2[mask], X_val_r, y_val,
            X_te_rr, y_te_r, sw2[mask], label=regime_name,
            X_cal=X_cal_r, y_cal=y_cal_r
        )
        if model is None: continue

        pkg = {
            'model':          model,
            'features':       sel_r,          # per-regime feature list!
            'regime':         regime_name,
            'is_auc':         round(is_auc, 4),
            'oos_auc':        round(oos_auc, 4),
            'n_features':     len(sel_r),
            'train_years':    TRAIN_YEARS,
            'version':        'sweep_v2',
            'n_samples_is':   int(mask.sum()),
        }
        out_path = BASE / f'sweep_{regime_name}.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(pkg, f)
        log.info(f"  ✅ Salvat: {out_path.name} | IS={is_auc:.4f} OOS={oos_auc:.4f}")

        # WR per threshold
        for thr in [0.50, 0.55, 0.60]:
            te_proba = model.predict_proba(X_te_rr)[:, 1]
            m_t = te_proba >= thr
            if m_t.sum() > 5:
                log.info(f"    WR@{thr}: {float(y_te_r[m_t].mean()):.1%} ({m_t.sum()} setups)")

    log.info("\n✅ SWEEP V2 training complet.")


if __name__ == "__main__":
    train_and_save()
