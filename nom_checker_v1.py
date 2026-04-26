"""
New York Open Manipulation (NOM) Checker v1.1 — EVENT-DRIVEN
=============================================================
v1.1: Sweep detectat dinamic pe toată sesiunea NY (09:00-13:00 ET),
      nu doar în fereastra fixă 09:30-10:00 ET.
      Regimul (PRE_EXPANSION) înlocuiește fereastra ca filtru principal.

Pattern: sweep față de pre-NY range (London session high/low)
         + displacement în direcție opusă
         + regime = PRE_EXPANSION (sau UNKNOWN ca fallback)

OOS (2025-2026, event-driven, full session NY 09:00-13:00 ET):
  AUC=0.695 | WR=79.5%@0.65 (794 setups)
  IS AUC=0.887 | 2129 IS setups (765 zile) | best_iter=270
"""

import pickle, sqlite3, logging, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger("ALADIN.NOM")

# ── v1.1 EVENT-DRIVEN: ferestre ET (DB stochează ET, UTC-4) ──────────────────
_ET_H = 4

# ── Daily cache for regime features (computed once per day) ──────────────────
_NOM_DAILY = {
    'date': None,
    'regime': {},
    'mtf': None,
}

# Sesiunea NY în ET: 09:00-13:00 ET = 13:00-17:00 UTC
NY_SESS_START_ET = 900    # 09:00 ET — gate activ din 09:00 ET
NY_SESS_END_ET   = 1300   # 13:00 ET — gate se închide la 13:00 ET
# Pre-NY = tot ce e înainte de 09:00 ET (London + Asia)
PRE_NY_END_ET    = 859    # 08:59 ET
LON_START_ET     = 400    # 04:00 ET
LON_END_ET       = 630    # 06:30 ET
ASIA_END_ET      = 359    # 03:59 ET

# UTC gate: 13:00-17:00 UTC
NOM_UTC_START    = 1300
NOM_UTC_END      = 1700

MIN_SPIKE_PT     = 5.0
MIN_DISP_PT      = 4.0
MAX_SL_PT        = 12.0
TP_MULT          = 2.0    # 2R pentru NOM (NY mai mult spatiu)
MODEL_THRESHOLD  = 0.65   # WR=70.0% la 0.65 (OOS 2025-2026, 223 setups)
import sys as _sys
_NOM_DIR = str(Path(__file__).parent)
if _NOM_DIR not in _sys.path:
    _sys.path.insert(0, _NOM_DIR)

# Prioritate: v4 ensemble → v1 single (fallback)
_MODEL_CANDIDATES = [
    Path(__file__).parent / "nom_model_v4.pkl",
    Path(__file__).parent / "nom_model_v1.pkl",
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if p.exists()), _MODEL_CANDIDATES[-1])

_NOM_MODEL = None


def _nom_predict(pkg, row_df: "pd.DataFrame") -> float:
    """Dispatch prediction: ensemble (N modele) sau single model."""
    import numpy as np
    if pkg.get('type') == 'ensemble':
        preds = [float(m.predict_proba(row_df)[0, 1]) for m in pkg['models']]
        return float(np.mean(preds))
    return float(pkg['model'].predict_proba(row_df)[0, 1])


def _load_nom_model():
    global _NOM_MODEL
    if _NOM_MODEL is not None:
        return _NOM_MODEL
    if not MODEL_PATH.exists():
        logger.warning(f"NOM model lipsă: {MODEL_PATH}")
        return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            _NOM_MODEL = pickle.load(f)
        mtype = _NOM_MODEL.get('type', 'single')
        nmod  = _NOM_MODEL.get('n_models', 1) if mtype == 'ensemble' else 1
        logger.info(f"✅ NOM loaded [{MODEL_PATH.name}]: type={mtype} n={nmod} "
                    f"OOS={_NOM_MODEL.get('oos_auc',0):.3f} | {_NOM_MODEL.get('n_features',0)} feats")
        return _NOM_MODEL
    except Exception as e:
        logger.error(f"NOM load error: {e}")
        return None


def compute_ict_on_tf(df_tf, lookback=20):
    """Multi-timeframe ICT: FVG, IFVG, breaker, rejection detection."""
    H = df_tf['high'].values.astype(float); L = df_tf['low'].values.astype(float)
    C = df_tf['close'].values.astype(float); O = df_tf['open'].values.astype(float)
    A = np.maximum(df_tf['atr'].values.astype(float), 1.0) if 'atr' in df_tf.columns else np.ones(len(H))
    n = len(H)
    bull_top = np.zeros(n); bull_bot = np.zeros(n)
    bear_top = np.zeros(n); bear_bot = np.zeros(n)
    for i in range(2, n):
        if H[i-2] < L[i] and (L[i]-H[i-2]) > 0.5: bull_top[i]=L[i]; bull_bot[i]=H[i-2]
        if L[i-2] > H[i] and (L[i-2]-H[i]) > 0.5: bear_top[i]=L[i-2]; bear_bot[i]=H[i]
    in_bull=np.zeros(n); in_bear=np.zeros(n); dist_bull=np.full(n,9.9); dist_bear=np.full(n,9.9)
    in_ifvg_b=np.zeros(n); in_ifvg_s=np.zeros(n); breaker_b=np.zeros(n); breaker_s=np.zeros(n)
    active_bull=[]; active_bear=[]; inv_bull=[]; inv_bear=[]; bull_obs=[]; bear_obs=[]
    for i in range(n):
        c=C[i]; l=L[i]; h=H[i]; a=A[i]
        new_ab=[]
        for top,bot,j in active_bull:
            if i-j>lookback: continue
            if l<bot: inv_bull.append((top,bot,i))
            else: new_ab.append((top,bot,j))
        active_bull=new_ab
        new_ab2=[]
        for top,bot,j in active_bear:
            if i-j>lookback: continue
            if h>top: inv_bear.append((top,bot,i))
            else: new_ab2.append((top,bot,j))
        active_bear=new_ab2
        if bull_top[i]>0: active_bull.append((bull_top[i],bull_bot[i],i))
        if bear_top[i]>0: active_bear.append((bear_top[i],bear_bot[i],i))
        if i>=2:
            pb=C[i-1]-O[i-1]; pr=max(H[i-1]-L[i-1],0.01)
            if pb>0.55*pr and pb>1.0: bull_obs.append((C[i-1],O[i-1],i-1))
            if pb<-0.55*pr and abs(pb)>1.0: bear_obs.append((O[i-1],C[i-1],i-1))
        for top,bot,j in active_bull:
            if bot<=c<=top: in_bull[i]=1.0
            dist_bull[i]=min(dist_bull[i],min(abs(c-top),abs(c-bot))/a)
        for top,bot,j in active_bear:
            if bot<=c<=top: in_bear[i]=1.0
            dist_bear[i]=min(dist_bear[i],min(abs(c-top),abs(c-bot))/a)
        for top,bot,k in inv_bull[-15:]:
            if i-k<=lookback*2 and bot<=c<=top: in_ifvg_b[i]=1.0
        for top,bot,k in inv_bear[-15:]:
            if i-k<=lookback*2 and bot<=c<=top: in_ifvg_s[i]=1.0
        for top,bot,j in bull_obs[-20:]:
            if i-j<=lookback and c<min(bot,O[j])-a*0.05:
                if abs(c-top)/a<0.8 or abs(c-bot)/a<0.8: breaker_s[i]=1.0
        for top,bot,j in bear_obs[-20:]:
            if i-j<=lookback and c>max(top,O[j])+a*0.05:
                if abs(c-top)/a<0.8 or abs(c-bot)/a<0.8: breaker_b[i]=1.0
    return pd.DataFrame({'in_bull':in_bull,'in_bear':in_bear,
        'dist_bull':np.clip(dist_bull,0,9.9),'dist_bear':np.clip(dist_bear,0,9.9),
        'in_ifvg_b':in_ifvg_b,'in_ifvg_s':in_ifvg_s,
        'breaker_b':breaker_b,'breaker_s':breaker_s}, index=df_tf.index)


def _load_nom_state():
    """Load NOM persistent state (last wins, weekly setup count, prev closes)."""
    state_path = Path(__file__).parent / "nom_checker_state.json"
    default = {
        'last_win_LONG': None,
        'last_win_SHORT': None,
        'week_YYYY_WW': None,
        'prev_dir_close': None,
        'prev_dir_mid': None,
    }
    try:
        if state_path.exists():
            return json.loads(state_path.read_text())
    except:
        pass
    return default


def _load_calendar():
    """Load economic calendar for FOMC proximity."""
    cal_path = Path(__file__).parent / "data" / "economic_calendar.json"
    try:
        if cal_path.exists():
            return json.loads(cal_path.read_text())
    except:
        pass
    return {'fomc': []}


def _fomc_proximity(date_str, fomc_dates):
    """Days to nearest FOMC, normalized to [-1, 1]."""
    try:
        d = pd.Timestamp(date_str).date()
        if not fomc_dates:
            return 0.0
        diffs = [abs((d - pd.Timestamp(x).date()).days) for x in fomc_dates]
        min_diff = float(min(diffs)) if diffs else 30.0
        return min(1.0, max(-1.0, 1.0 - min_diff / 30.0))
    except:
        return 0.0


def check_nom_setup(db_path: str, now_utc: datetime = None,
                    regime: str = None, regime_prob: float = 0.0,
                    regime_enc: int = 2) -> dict | None:
    """
    Checker NOM event-driven v1.1.

    Detectează sweep față de pre-NY range ORICÂND în sesiunea NY (09:00-13:00 ET).
    Regimul înlocuiește fereastra fixă ca filtru principal.

    Args:
        regime:      regimul curent din classify_regime() — ex. 'PRE_EXPANSION'
        regime_prob: probabilitatea regimului (0-1)
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    hhmm_utc = now_utc.hour * 100 + now_utc.minute

    # Gate UTC: sesiunea NY activă 13:00-17:00 UTC
    if not (NOM_UTC_START <= hhmm_utc <= NOM_UTC_END):
        return None

    # Regime gate: preferăm PRE_EXPANSION, acceptăm UNKNOWN ca fallback
    if regime is not None and regime_prob >= 0.65:
        if regime == 'CONSOLIDATION':
            logger.debug("NOM: CONSOLIDATION detectat → skip")
            return None
        if regime in ('EXPANSION', 'DISTRIBUTION') and regime_prob >= 0.75:
            logger.debug(f"NOM: {regime} → nu e momentul pentru sweep manipulation")
            return None

    now_et = now_utc - timedelta(hours=_ET_H)
    today  = now_et.date()

    # ── Fetch date ────────────────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True,
                               timeout=30, check_same_thread=False)
        day_df = pd.read_sql(f"""
            SELECT timestamp, open, high, low, close, volume,
                   asia_hi, asia_lo, p_hi, p_lo, vah, val, poc_level,
                   lw_hi, lw_lo, true_open, h4_hi, h4_lo, h1_hi, h1_lo,
                   atr_14, bar_delta, cum_delta, fvg_up, fvg_down, has_displacement,
                   body_size, adx_14, hurst, dist_poc, inside_va, dist_vwap,
                   delta_at_high, delta_at_low, big_buy_count, big_sell_count,
                   absorption_score, stacked_bull, stacked_bear, of_doi, of_big_balance,
                   bar_buy_vol, bar_sell_vol, garch_vol, kalman_smooth,
                   fisher_transform, acf_lag1, acf_lag5, rvol,
                   day_of_week, month
            FROM market_data
            WHERE date = '{today}'
              AND hour_min BETWEEN '00:00' AND '{now_et.strftime("%H:%M")}'
            ORDER BY timestamp
        """, conn)
        conn.close()
    except Exception as e:
        logger.error(f"NOM DB error: {e}")
        return None

    if len(day_df) < 30:
        return None

    day_df['ts']   = pd.to_datetime(day_df['timestamp'])
    day_df['hhmm'] = day_df['ts'].dt.hour * 100 + day_df['ts'].dt.minute

    # ── v1.1: Sesiuni dinamice ────────────────────────────────────────────────
    london    = day_df[day_df['hhmm'].between(LON_START_ET,  LON_END_ET)]
    pre_ny    = day_df[day_df['hhmm'] <= PRE_NY_END_ET]        # tot ce e înainte de NY open
    ny_sess   = day_df[day_df['hhmm'].between(NY_SESS_START_ET, NY_SESS_END_ET)]
    asia_bars = day_df[day_df['hhmm'].between(0, ASIA_END_ET)]

    if len(pre_ny) < 20 or len(ny_sess) < 3:
        logger.debug(f"NOM: date insuficiente pre_ny={len(pre_ny)} ny_sess={len(ny_sess)}")
        return None

    # ── Pre-NY range (London session = 08:00-13:29 UTC) ──────────────────────
    pre_hi  = float(pre_ny['high'].max())
    pre_lo  = float(pre_ny['low'].min())
    pre_rng = pre_hi - pre_lo
    if pre_rng < 5:
        return None

    # London session high/low (08:00-10:30)
    if len(london) > 0:
        lon_hi = float(london['high'].max())
        lon_lo = float(london['low'].min())
        lon_rng = lon_hi - lon_lo
        lon_mid = (lon_hi + lon_lo) / 2
        lon_close = float(london['close'].iloc[-1])
    else:
        lon_hi = pre_hi; lon_lo = pre_lo
        lon_rng = pre_rng; lon_mid = (pre_hi + pre_lo) / 2
        lon_close = lon_mid

    # ATR
    atr_vals = day_df['atr_14'].replace(0, np.nan).dropna()
    atr = float(atr_vals.iloc[-1]) if len(atr_vals) > 0 else 10.0

    # ── v1.1: Detectare sweep dinamic în toată sesiunea NY ───────────────────
    spike_hi  = float(ny_sess['high'].max())
    spike_lo  = float(ny_sess['low'].min())
    spike_up  = spike_hi - pre_hi
    spike_dn  = pre_lo - spike_lo

    # Acceptăm și partial sweep (30% din pre_range)
    partial_thresh = max(MIN_SPIKE_PT, pre_rng * 0.30)
    sweep_dn_valid = spike_dn >= MIN_SPIKE_PT or (spike_dn >= -partial_thresh and spike_dn > -pre_rng * 0.5)
    sweep_up_valid = spike_up >= MIN_SPIKE_PT or (spike_up >= -partial_thresh and spike_up > -pre_rng * 0.5)

    direction     = None
    spike_mag     = 0.0
    spike_bar_idx = None

    if spike_up >= spike_dn and sweep_up_valid and spike_up >= MIN_SPIKE_PT * 0.5:
        direction     = 'SHORT'
        spike_mag     = max(spike_up, 0)
        spike_bar_idx = ny_sess['high'].idxmax()
    elif spike_dn > spike_up and sweep_dn_valid and spike_dn >= MIN_SPIKE_PT * 0.5:
        direction     = 'LONG'
        spike_mag     = max(spike_dn, 0)
        spike_bar_idx = ny_sess['low'].idxmin()

    if direction is None:
        logger.debug(f"NOM: sweep insuficient (up={spike_up:.1f} dn={spike_dn:.1f} min={MIN_SPIKE_PT})")
        return None

    spike_bar   = day_df.loc[spike_bar_idx]
    after_spike = day_df[day_df.index > spike_bar_idx]
    # 45min după spike (mai generos decât 30min fix)
    spike_hhmm  = int(day_df.loc[spike_bar_idx, 'hhmm'])
    after_early = after_spike[after_spike['hhmm'] <= spike_hhmm + 45]

    if len(after_early) < 2:
        return None

    # ── Detectare displacement ────────────────────────────────────────────────
    disp_bar = None
    for _, bar in after_early.iterrows():
        body = abs(bar['close'] - bar['open'])
        if direction == 'SHORT' and bar['close'] < bar['open'] and body >= MIN_DISP_PT:
            disp_bar = bar; break
        elif direction == 'LONG' and bar['close'] > bar['open'] and body >= MIN_DISP_PT:
            disp_bar = bar; break

    if disp_bar is None:
        logger.debug(f"NOM: {direction} spike detectat dar fără displacement")
        return None

    entry        = float(disp_bar['close'])
    _sweep_feats = {}   # compact features for sweep_scorer — populated inside try block

    # ── Feature engineering ───────────────────────────────────────────────────
    pkg = _load_nom_model()
    nom_score = 0.5

    if pkg is not None:
        try:
            def sv(v, d=0.0):
                try: x = float(v); return x if np.isfinite(x) else d
                except: return d

            r0      = day_df.iloc[-1]
            dir_num = -1 if direction == 'SHORT' else 1
            disp_body = abs(disp_bar['close'] - disp_bar['open'])
            spike_bar_range = sv(spike_bar['high'] - spike_bar['low'], 1.0)

            # ── ts_close_inside (spike bar inchide inapoi in range) ───────────
            if direction == 'SHORT':
                ts_close_inside  = 1 if sv(spike_bar['close']) <= pre_hi else 0
                wick             = (sv(spike_bar['high']) - max(sv(spike_bar['close']), sv(spike_bar['open']))) / atr if atr > 0 else 0
                ts_rejection_str = (spike_hi - sv(spike_bar['close'])) / spike_mag if spike_mag > 0 else 0
                ts_wick_pct      = (spike_hi - sv(spike_bar['close'])) / spike_bar_range if spike_bar_range > 0 else 0
                ts_body_pct      = abs(sv(spike_bar['open']) - sv(spike_bar['close'])) / spike_bar_range if spike_bar_range > 0 else 0
                ts_close_quality = max(0, (pre_hi - sv(spike_bar['close'])) / pre_rng) if pre_rng > 0 else 0
            else:
                ts_close_inside  = 1 if sv(spike_bar['close']) >= pre_lo else 0
                wick             = (min(sv(spike_bar['close']), sv(spike_bar['open'])) - sv(spike_bar['low'])) / atr if atr > 0 else 0
                ts_rejection_str = (sv(spike_bar['close']) - spike_lo) / spike_mag if spike_mag > 0 else 0
                ts_wick_pct      = (sv(spike_bar['close']) - spike_lo) / spike_bar_range if spike_bar_range > 0 else 0
                ts_body_pct      = abs(sv(spike_bar['open']) - sv(spike_bar['close'])) / spike_bar_range if spike_bar_range > 0 else 0
                ts_close_quality = max(0, (sv(spike_bar['close']) - pre_lo) / pre_rng) if pre_rng > 0 else 0

            wick_pct         = wick * atr / spike_bar_range if spike_bar_range > 0 else 0
            ts_wick_dom      = 1 if ts_wick_pct > 0.6 else 0
            ts_htf_anti      = 0
            ts_combo_score   = ts_close_inside * ts_rejection_str

            # ── Sweep quality ─────────────────────────────────────────────────
            sweep_wick_clean = 1 if wick_pct > 0.5 else 0
            sweep_depth_atr  = spike_mag / atr if atr > 0 else 0
            deep_sweep       = 1 if sweep_depth_atr > 1.5 else 0
            shallow_sweep    = 1 if sweep_depth_atr < 0.5 else 0
            sweep_quality    = ts_close_inside*0.4 + sweep_wick_clean*0.3 + deep_sweep*0.2 + 0.1

            # ── NOM-specific: London context ──────────────────────────────────
            swept_lon_hi      = 1 if spike_hi > lon_hi else 0
            swept_lon_lo      = 1 if spike_lo < lon_lo else 0
            dist_lon_hi_atr   = abs(entry - lon_hi) / atr if atr > 0 else 0
            dist_lon_lo_atr   = abs(entry - lon_lo) / atr if atr > 0 else 0
            lon_range_atr     = lon_rng / atr if atr > 0 else 0
            lon_close_vs_mid  = (lon_close - lon_mid) / lon_rng if lon_rng > 0 else 0
            ts_sweep_pct_lon  = spike_mag / lon_rng if lon_rng > 0 else 0
            ts_lon_mid_dist   = (entry - lon_mid) / atr if atr > 0 else 0
            ts_entry_prox     = abs(entry - lon_mid) / lon_rng if lon_rng > 0 else 0
            ts_sweep_depth_pts = spike_mag
            ts_sweep_depth_atr = sweep_depth_atr

            # ── NY first 15min drive ──────────────────────────────────────────
            ny15 = ny_sess[ny_sess['hhmm'].between(NY_SESS_START_ET, NY_SESS_START_ET + 15)]
            ny15_rng = (ny15['high'].max() - ny15['low'].min()) if len(ny15) > 0 else 0
            ny15_range_atr = ny15_rng / atr if atr > 0 else 0

            # ── Gap NY open vs London close ───────────────────────────────────
            ny_open_price      = float(ny_sess['open'].iloc[0]) if len(ny_sess) > 0 else entry
            gap_vs_lon_close   = ny_open_price - lon_close
            gap_vs_lon_close_atr = gap_vs_lon_close / atr if atr > 0 else 0

            # ── ATR regime ────────────────────────────────────────────────────
            # atr_vs_10d: approximat cu ATR curent vs medie din lw_hi/lw_lo
            lw_hi = sv(r0['lw_hi']); lw_lo = sv(r0['lw_lo'])
            lw_rng = lw_hi - lw_lo
            atr_vs_10d = atr / (lw_rng / 10.0) if lw_rng > 0 else 1.0

            # ── HTF bias ──────────────────────────────────────────────────────
            h4_hi = sv(r0['h4_hi']); h4_lo = sv(r0['h4_lo'])
            h1_hi = sv(r0['h1_hi']); h1_lo = sv(r0['h1_lo'])
            h4_mid = (h4_hi + h4_lo) / 2 if h4_hi > 0 and h4_lo > 0 else 0
            h1_mid = (h1_hi + h1_lo) / 2 if h1_hi > 0 and h1_lo > 0 else 0
            h4_bias = 1 if entry < h4_mid else (-1 if h4_mid > 0 else 0)
            h1_bias = 1 if entry < h1_mid else (-1 if h1_mid > 0 else 0)
            h4_h1_aligned  = 1 if h4_bias == h1_bias and h4_bias != 0 else 0
            h4_bias_aligned = 1 if h4_bias == dir_num else 0

            # ── Weekly premium/discount ───────────────────────────────────────
            weekly_prem = (entry - lw_lo) / lw_rng if lw_rng > 0 else 0.5
            in_weekly_premium  = 1 if weekly_prem > 0.5 else 0
            in_weekly_discount = 1 if weekly_prem < 0.5 else 0
            weekly_prem_aligned = 1 if (direction == 'SHORT' and in_weekly_premium) or \
                                        (direction == 'LONG'  and in_weekly_discount) else 0
            h4_x_weekly = (1 if h4_bias == dir_num else 0) * weekly_prem_aligned
            lw_range_atr = lw_rng / atr if atr > 0 else 0
            week_range_so_far = (day_df['high'].max() - day_df['low'].min()) / atr if atr > 0 else 0
            dist_prev_wk_lo = abs(entry - lw_lo) / atr if atr > 0 else 0

            # ── Asia context ──────────────────────────────────────────────────
            asia_hi_v = sv(r0['asia_hi']); asia_lo_v = sv(r0['asia_lo'])
            asia_rng  = asia_hi_v - asia_lo_v
            dist_asia_hi_atr = abs(entry - asia_hi_v) / atr if atr > 0 and asia_hi_v > 0 else 0
            dist_asia_lo_atr = abs(entry - asia_lo_v) / atr if atr > 0 and asia_lo_v > 0 else 0
            asia_range_atr   = asia_rng / atr if atr > 0 and asia_rng > 0 else 0
            spike_vs_asia_hi = (spike_hi - asia_hi_v) / atr if atr > 0 and asia_hi_v > 0 else 0
            spike_vs_asia_lo = (asia_lo_v - spike_lo) / atr if atr > 0 and asia_lo_v > 0 else 0

            # ── True open & PDH/PDL ───────────────────────────────────────────
            above_true_open = 1 if entry > sv(r0['true_open']) else 0
            dist_true_open  = abs(entry - sv(r0['true_open'])) / atr if atr > 0 else 0
            dist_pdh_atr    = abs(entry - sv(r0['p_hi'])) / atr if atr > 0 else 0
            dist_pdl_atr    = abs(entry - sv(r0['p_lo'])) / atr if atr > 0 else 0
            dist_lw_hi      = abs(entry - lw_hi) / atr if atr > 0 else 0
            dist_lw_lo      = abs(entry - lw_lo) / atr if atr > 0 else 0

            # ── Equal levels ──────────────────────────────────────────────────
            eq_tol = atr * 0.3
            pre_highs = pre_ny['high'].values; pre_lows = pre_ny['low'].values
            eq_hi = sum(1 for h in pre_highs if abs(h - pre_hi) <= eq_tol) - 1
            eq_lo = sum(1 for l in pre_lows  if abs(l - pre_lo) <= eq_tol) - 1
            equal_level_score = (eq_hi if direction == 'SHORT' else eq_lo) / max(len(pre_ny), 1)

            # ── Volume / delta ────────────────────────────────────────────────
            pre_vol = float(pre_ny['volume'].mean()) if len(pre_ny) > 0 else 1.0
            spike_vol = float(ny_sess['volume'].mean()) if len(ny_sess) > 0 else 1.0
            vol_ratio = spike_vol / pre_vol if pre_vol > 0 else 1.0

            bv  = sv(ny_sess['bar_buy_vol'].sum())
            sv2 = sv(ny_sess['bar_sell_vol'].sum())

            spike_delta = sv(ny_sess['bar_delta'].sum())
            disp_delta  = sv(after_early['bar_delta'].sum()) if len(after_early) > 0 else 0
            delta_at_high = sv(ny_sess['delta_at_high'].sum()) if 'delta_at_high' in ny_sess.columns else 0
            delta_at_low  = sv(ny_sess['delta_at_low'].sum())  if 'delta_at_low'  in ny_sess.columns else 0
            big_buy     = 1 if vol_ratio > 2.0 and direction == 'LONG'  else 0
            big_sell    = 1 if vol_ratio > 2.0 and direction == 'SHORT' else 0
            big_imbalance = 1 if vol_ratio > 2.0 else 0
            absorption  = sv(ny_sess['absorption_score'].mean()) if 'absorption_score' in ny_sess.columns else 0

            # ── Technical indicators ──────────────────────────────────────────
            adx_v   = sv(r0['adx_14'])
            hurst_v = sv(r0['hurst'], 0.5)

            # ── FVG alignment ─────────────────────────────────────────────────
            fvg_up_v   = int(ny_sess['fvg_up'].any())   if 'fvg_up'   in ny_sess.columns else 0
            fvg_down_v = int(ny_sess['fvg_down'].any()) if 'fvg_down' in ny_sess.columns else 0
            htf_fvg_aligned = 1 if (direction == 'SHORT' and fvg_down_v) or \
                                    (direction == 'LONG'  and fvg_up_v)  else 0
            # Multi-TF FVG proxy
            def has_fvg(bars, d):
                if len(bars) < 3: return 0
                h = bars['high'].values; l = bars['low'].values
                if d == 'LONG':  return int(any(l[i] > h[i-2] for i in range(2, len(h))))
                else:            return int(any(h[i] < l[i-2] for i in range(2, len(l))))
            fvg_1h = has_fvg(day_df[day_df['hhmm'].between(800, 1330)].tail(60), direction)
            vol_x_fvg_1h = vol_ratio * fvg_1h

            # ── OB proxies ────────────────────────────────────────────────────
            ob_proxy_bull = int(ny_sess['stacked_bull'].any()) if 'stacked_bull' in ny_sess.columns else 0
            ob_proxy_bear = int(ny_sess['stacked_bear'].any()) if 'stacked_bear' in ny_sess.columns else 0
            ob_aligned    = 0

            # ── VIX proxy ─────────────────────────────────────────────────────
            rets = day_df['close'].pct_change().dropna()
            vix_proxy_20d = float(rets.rolling(20).std().iloc[-1]) if len(rets) > 20 else 0

            # ── Rolling regime features (cached daily) ────────────────────────
            global _NOM_DAILY
            today_str = today.isoformat()
            if _NOM_DAILY['date'] != today_str:
                _NOM_DAILY['date'] = today_str
                _NOM_DAILY['regime'] = {}
                try:
                    vix_5d = float(rets.rolling(5).std().iloc[-1]) if len(rets) > 5 else 0
                    vix_20d = vix_proxy_20d
                    vix_proxy_5d = vix_5d
                    vol_regime = (vix_proxy_5d / vix_20d) if vix_20d > 0 else 1.0

                    lw_range_1d = day_df[day_df['hhmm'] >= 1400]['close'].iloc[-1] - day_df[day_df['hhmm'] < 100]['open'].iloc[0] if len(day_df) > 5 else 0
                    atr_5d = float(day_df['atr_14'].rolling(5).mean().iloc[-1]) if len(day_df) > 5 else atr
                    atr_10d = float(day_df['atr_14'].rolling(10).mean().iloc[-1]) if len(day_df) > 10 else atr
                    atr_trend = (atr_5d / atr_10d) if atr_10d > 0 else 1.0

                    adx_10d_mean = float(day_df['adx_14'].rolling(10).mean().iloc[-1]) if len(day_df) > 10 else sv(r0['adx_14'])

                    _NOM_DAILY['regime'] = {
                        'vix_proxy_5d': vix_proxy_5d,
                        'vol_regime': vol_regime,
                        'atr_trend': atr_trend,
                        'adx_10d_mean': adx_10d_mean,
                        'atr_5d': atr_5d,
                    }
                except Exception as e:
                    logger.debug(f"Regime computation error: {e}")
                    _NOM_DAILY['regime'] = {
                        'vix_proxy_5d': 0.0,
                        'vol_regime': 1.0,
                        'atr_trend': 1.0,
                        'adx_10d_mean': 20.0,
                        'atr_5d': atr,
                    }

            vix_proxy_5d = _NOM_DAILY['regime'].get('vix_proxy_5d', 0.0)
            vol_regime = _NOM_DAILY['regime'].get('vol_regime', 1.0)
            atr_trend = _NOM_DAILY['regime'].get('atr_trend', 1.0)
            adx_10d_mean = _NOM_DAILY['regime'].get('adx_10d_mean', 20.0)
            atr_5d = _NOM_DAILY['regime'].get('atr_5d', atr)

            # ── MTF ICT features (cached daily, computed once) ────────────────
            try:
                if _NOM_DAILY['mtf'] is None:
                    mtf_data = {}
                    spike_hhmm_int = int(spike_hhmm)
                    spike_idx = None
                    for tf_name, tf_min in [('5m', 5), ('15m', 15), ('1h', 60), ('4h', 240)]:
                        mtf_data[tf_name] = {}
                    _NOM_DAILY['mtf'] = mtf_data

                mtf_data = _NOM_DAILY['mtf']
            except:
                mtf_data = {}

            dist_bear_5m = mtf_data.get('5m', {}).get('dist_bear', 5.0)
            dist_bull_5m = mtf_data.get('5m', {}).get('dist_bull', 5.0)
            dist_bear_15m = mtf_data.get('15m', {}).get('dist_bear', 5.0)
            dist_bull_15m = mtf_data.get('15m', {}).get('dist_bull', 5.0)
            dist_bear_1h = mtf_data.get('1h', {}).get('dist_bear', 5.0)
            dist_bull_1h = mtf_data.get('1h', {}).get('dist_bull', 5.0)
            in_ifvg_b_1h = mtf_data.get('1h', {}).get('in_ifvg_b', 0)
            htf_fvg_aligned_mtf = max(mtf_data.get('1h', {}).get('fvg_aligned', 0),
                                      mtf_data.get('4h', {}).get('fvg_aligned', 0))

            # ── Persistent state (wins, weekly setup count, prev closes) ──────
            nom_state = _load_nom_state()
            rolling_5sess_wr = 0.0
            days_since_win_dir = 999
            days_since_win_any = 999
            week_setup_count = 0
            prev_ny_dir = 0
            prev_ny_aligned = 0

            try:
                if nom_state.get('last_win_LONG') or nom_state.get('last_win_SHORT'):
                    last_long = pd.Timestamp(nom_state['last_win_LONG']) if nom_state.get('last_win_LONG') else None
                    last_short = pd.Timestamp(nom_state['last_win_SHORT']) if nom_state.get('last_win_SHORT') else None
                    today_ts = pd.Timestamp(today)
                    if last_long:
                        days_since_win_dir = min(days_since_win_dir, (today_ts - last_long).days) if direction == 'LONG' else days_since_win_dir
                        days_since_win_any = min(days_since_win_any, (today_ts - last_long).days)
                    if last_short:
                        days_since_win_dir = min(days_since_win_dir, (today_ts - last_short).days) if direction == 'SHORT' else days_since_win_dir
                        days_since_win_any = min(days_since_win_any, (today_ts - last_short).days)
                    rolling_5sess_wr = 0.6

                if nom_state.get('week_YYYY_WW'):
                    week_setup_count = nom_state['week_YYYY_WW']

                prev_close = nom_state.get('prev_dir_close', entry)
                prev_mid = nom_state.get('prev_dir_mid', entry)
                if prev_close and prev_mid:
                    prev_ny_dir = 1 if prev_close > prev_mid else -1
                    prev_ny_aligned = 1 if prev_ny_dir == dir_num else 0
            except:
                pass

            if days_since_win_dir >= 999: days_since_win_dir = 30
            if days_since_win_any >= 999: days_since_win_any = 30

            # ── Calendar proximity ───────────────────────────────────────────
            cal = _load_calendar()
            fomc_prox = _fomc_proximity(today_str, cal.get('fomc', []))

            # ── Asia & London direction context ───────────────────────────────
            asia_hi_v = sv(r0['asia_hi']); asia_lo_v = sv(r0['asia_lo'])
            asia_rng = asia_hi_v - asia_lo_v
            asia_dir = 1 if asia_bars['close'].iloc[-1] > (asia_hi_v + asia_lo_v) / 2 else -1 if len(asia_bars) > 0 else 0

            lon_dir = 1 if lon_close > lon_mid else -1
            lon_asia_aligned = 1 if lon_dir == asia_dir else 0
            asia_dir_x_lon_dir = float(asia_dir) * float(lon_dir)
            asia_range_vs_atr5d = asia_rng / atr_5d if atr_5d > 0 else 0

            # ── NOM-specific inline features ─────────────────────────────────
            atr_tp_norm = np.clip(atr / 20.0, 0.5, 3.0)
            disp_range_atr = (sv(disp_bar['high']) - sv(disp_bar['low'])) / atr if atr > 0 else 0

            is_pre_nfp = 1 if (today_str in cal.get('nfp', []) and int(now_et.strftime("%H%M")) < 830) else 0

            drive_aligned_dir = h4_bias if h4_bias == dir_num else 0
            drive_x_sweep = drive_aligned_dir * sweep_quality

            triple_sess = 1 if asia_dir == lon_dir and lon_dir != dir_num else 0
            triple_x_h4 = triple_sess * (1 if h4_bias == dir_num else 0)

            smt_aligned = 1 if (direction == 'SHORT' and fvg_down_v) or (direction == 'LONG' and fvg_up_v) else 0
            smt_x_sweep = smt_aligned * sweep_quality

            nfp_post_x_sweep = (1 if (today_str in cal.get('nfp', []) and int(now_et.strftime("%H%M")) >= 830) else 0) * sweep_quality
            vol_x_sweep    = vol_regime * sweep_quality
            vol_x_ts_close = vol_regime * ts_close_inside
            # Features simple lipsă
            sweep_x_h4     = sweep_quality * (1 if h4_bias == dir_num else 0)
            dir_x_hurst    = dir_num * hurst_v
            dir_x_adx      = dir_num * adx_v
            entry_hhmm     = int(disp_bar['hhmm']) if 'hhmm' in disp_bar.index else spike_hhmm
            sweep_time_late= 1 if entry_hhmm > NY_SESS_START_ET + 120 else 0
            ny_open_in_lon = 1 if lon_lo <= ny_open_price <= lon_hi else 0
            prev_ny_opp    = 1 if prev_ny_dir == -dir_num else 0

            # ── Sweep bar specific features (NUOVO) ──────────────────────────
            _sb_range_n = max(sv(spike_bar['high']) - sv(spike_bar['low']), 0.01)
            _sb_body_n  = abs(sv(spike_bar['close']) - sv(spike_bar['open']))
            sweep_bar_body_pct_n = _sb_body_n / _sb_range_n
            if direction == 'SHORT':
                sweep_bar_wick_pct_n = (sv(spike_bar['high']) - max(sv(spike_bar['close']), sv(spike_bar['open']))) / _sb_range_n
            else:
                sweep_bar_wick_pct_n = (min(sv(spike_bar['close']), sv(spike_bar['open'])) - sv(spike_bar['low'])) / _sb_range_n
            _sess_mean_vol_n = max(ny_sess['volume'].mean(), 1)
            sweep_bar_vol_ratio_n = sv(spike_bar['volume']) / _sess_mean_vol_n
            _ny_before = ny_sess[ny_sess['hhmm'] < spike_hhmm]
            if direction == 'SHORT':
                n_level_tests_n = len(_ny_before[_ny_before['high'] >= pre_hi - 1.0])
            else:
                n_level_tests_n = len(_ny_before[_ny_before['low'] <= pre_lo + 1.0])
            bars_to_disp_n = float(entry_hhmm - spike_hhmm)
            _pre5_n = _ny_before.tail(5)
            pre5_mom_n = 0.0
            if len(_pre5_n) >= 3:
                _closes_n = _pre5_n['close'].astype(float).values
                _changes_n = np.sign(np.diff(_closes_n))
                pre5_mom_n = float(_changes_n.mean())

            # ── Compact sweep features (for sweep_scorer) ─────────────────────
            _sweep_feats = {
                'session_enc':          1,  # NY
                'spike_mag_atr':        spike_mag / atr if atr > 0 else 0,
                'pre_rng_atr':          pre_rng / atr if atr > 0 else 0,
                'ts_close_inside':      ts_close_inside,
                'ts_rejection_str':     ts_rejection_str,
                'sweep_wick_atr':       wick,
                'sweep_depth_atr':      sweep_depth_atr,
                'deep_sweep':           int(deep_sweep),
                'sweep_quality':        sweep_quality,
                'disp_body_atr':        abs(sv(disp_bar['close']) - sv(disp_bar['open'])) / atr if atr > 0 else 0,
                'h4_bias':              h4_bias,
                'h4_bias_aligned':      h4_bias_aligned,
                'weekly_premium_pct':   weekly_prem,
                'weekly_prem_aligned':  weekly_prem_aligned,
                'lw_range_atr':         lw_range_atr,
                'dist_lw_hi':           dist_lw_hi,
                'dist_lw_lo':           dist_lw_lo,
                'lon_range_atr':        lon_rng / atr if atr > 0 else 0,
                'lon_dir':              float(lon_dir),
                'lon_dir_aligned':      1 if lon_dir == dir_num else 0,
                'dist_asia_hi_atr':     dist_asia_hi_atr,
                'dist_asia_lo_atr':     dist_asia_lo_atr,
                'asia_dir':             float(asia_dir),
                'asia_dir_aligned':     1 if asia_dir == dir_num else 0,
                'triple_sess_aligned':  float(triple_sess),
                'adx':                  adx_v,
                'hurst':                hurst_v,
                'garch_vol':            sv(r0['garch_vol']),
                'rvol':                 sv(r0['rvol'], 1.0),
                'fisher_transform':     sv(r0['fisher_transform']),
                'acf_lag1':             sv(r0['acf_lag1']),
                'acf_lag5':             sv(r0['acf_lag5']),
                'kalman_smooth':        sv(r0['kalman_smooth']),
                'kalman_noise':         sv(r0.get('kalman_noise', 0.0)),
                'is_nfp_day':           1 if today_str in cal.get('nfp', []) else 0,
                'is_fomc_day':          1 if fomc_prox <= 1 else 0,
                'is_news_day':          1 if (fomc_prox <= 1 or today_str in cal.get('nfp', [])) else 0,
                'is_pre_nfp':           1 if (today_str in cal.get('nfp', []) and entry_hhmm < 830) else 0,
                'is_post_nfp':          1 if (today_str in cal.get('nfp', []) and entry_hhmm >= 830) else 0,
                'direction_enc':        1 if direction == 'SHORT' else 0,
                'day_of_week':          int(r0['day_of_week']),
                'is_thursday':          1 if int(r0['day_of_week']) == 3 else 0,
                'is_friday':            1 if int(r0['day_of_week']) == 4 else 0,
                'month':                int(r0['month']),
                'regime_enc':           regime_enc,
                'is_pre_expansion':     1 if (regime or '') == 'PRE_EXPANSION' else 0,
                'is_expansion':         1 if (regime or '') == 'EXPANSION' else 0,
                'is_retracement':       1 if (regime or '') == 'RETRACEMENT' else 0,
                'dir_x_adx':            dir_num * adx_v,
                'dir_x_hurst':          dir_num * hurst_v,
                'vol_x_sweep':          vol_x_sweep,
                'h4_x_weekly':          h4_x_weekly,
                # ── Sweep bar specific (NUOVO) ─────────────────────────────
                'sweep_bar_body_pct':   sweep_bar_body_pct_n,
                'sweep_bar_wick_pct':   sweep_bar_wick_pct_n,
                'sweep_bar_vol_ratio':  sweep_bar_vol_ratio_n,
                'n_level_tests':        float(n_level_tests_n),
                'bars_to_disp':         bars_to_disp_n,
                'pre5_momentum':        pre5_mom_n,
                'pre5_mom_aligned':     1.0 if (pre5_mom_n > 0 and direction == 'SHORT') or
                                               (pre5_mom_n < 0 and direction == 'LONG') else 0.0,
                'fast_disp':            1.0 if bars_to_disp_n <= 5 else 0.0,
                'vol_surge':            1.0 if sweep_bar_vol_ratio_n >= 1.5 else 0.0,
                'multi_test':           1.0 if n_level_tests_n >= 2 else 0.0,
            }

            # ── Build feature row ─────────────────────────────────────────────
            row = pd.DataFrame([{
                # Spike
                'spike_mag':           spike_mag,
                'spike_mag_atr':       spike_mag / atr if atr > 0 else 0,
                'spike_vs_range':      spike_mag / pre_rng if pre_rng > 0 else 0,
                'pre_rng_atr':         pre_rng / atr if atr > 0 else 0,
                # TS / anti-fakeout
                'ts_close_inside':     ts_close_inside,
                'ts_rejection_str':    ts_rejection_str,
                'ts_wick_pct':         ts_wick_pct,
                'ts_body_pct':         ts_body_pct,
                'ts_close_quality':    ts_close_quality,
                'ts_wick_dom':         ts_wick_dom,
                'ts_htf_anti':         ts_htf_anti,
                'ts_combo_score':      ts_combo_score,
                'ts_sweep_depth_pts':  ts_sweep_depth_pts,
                'ts_sweep_depth_atr':  ts_sweep_depth_atr,
                'ts_sweep_pct_lon':    ts_sweep_pct_lon,
                'ts_lon_mid_dist':     ts_lon_mid_dist,
                'ts_entry_prox':       ts_entry_prox,
                # Sweep quality
                'sweep_wick_atr':      wick,
                'sweep_wick_pct':      wick_pct,
                'sweep_wick_clean':    sweep_wick_clean,
                'sweep_depth_atr':     sweep_depth_atr,
                'deep_sweep':          deep_sweep,
                'shallow_sweep':       shallow_sweep,
                'sweep_with_disp':     1,
                'sweep_quality_score': sweep_quality,
                'equal_level_score':   equal_level_score,
                # Displacement
                'disp_body':           disp_body,
                'disp_body_atr':       disp_body / atr if atr > 0 else 0,
                'disp_range':          sv(disp_bar['high'] - disp_bar['low']),
                'disp_wick_ratio':     (sv(disp_bar['high'] - disp_bar['low']) - disp_body) / disp_body if disp_body > 0 else 0,
                'has_disp':            1,
                # Bar structure
                'body_pct':            disp_body / sv(disp_bar['high'] - disp_bar['low'], 1),
                'body_bear':           1 if direction == 'SHORT' else 0,
                # HTF bias
                'h4_bias':             h4_bias,
                'h1_bias':             h1_bias,
                'h4_h1_aligned':       h4_h1_aligned,
                'h4_bias_aligned':     h4_bias_aligned,
                # Weekly context
                'weekly_premium_pct':  weekly_prem,
                'in_weekly_premium':   in_weekly_premium,
                'in_weekly_discount':  in_weekly_discount,
                'weekly_prem_aligned': weekly_prem_aligned,
                'h4_x_weekly':         h4_x_weekly,
                'lw_range_atr':        lw_range_atr,
                'week_range_so_far':   week_range_so_far,
                'dist_prev_wk_lo':     dist_prev_wk_lo,
                'dist_lw_hi':          dist_lw_hi,
                'dist_lw_lo':          dist_lw_lo,
                # London context (NOM-specific)
                'lon_range_atr':       lon_range_atr,
                'dist_lon_hi_atr':     dist_lon_hi_atr,
                'dist_lon_lo_atr':     dist_lon_lo_atr,
                'lon_close_vs_mid':    lon_close_vs_mid,
                'ny15_range_atr':      ny15_range_atr,
                'gap_vs_lon_close_atr': gap_vs_lon_close_atr,
                # Asia context
                'dist_asia_hi_atr':    dist_asia_hi_atr,
                'dist_asia_lo_atr':    dist_asia_lo_atr,
                'asia_range_atr':      asia_range_atr,
                'spike_vs_asia_hi':    spike_vs_asia_hi,
                'spike_vs_asia_lo':    spike_vs_asia_lo,
                # True open / PDH / PDL
                'above_true_open':     above_true_open,
                'dist_true_open':      dist_true_open,
                'dist_pdh_atr':        dist_pdh_atr,
                'dist_pdl_atr':        dist_pdl_atr,
                # VA / POC
                'inside_va':           sv(r0['inside_va']),
                'dist_poc_entry':      sv(r0['dist_poc']) / atr if atr > 0 else 0,
                'entry_in_pre_range':  int(pre_lo <= entry <= pre_hi),
                'dist_poc_atr':        sv(r0['dist_poc']) / atr if atr > 0 else 0,
                'dist_vwap_atr':       sv(r0['dist_vwap']) / atr if atr > 0 else 0,
                # Volume / delta
                'vol_ratio':           vol_ratio,
                'spike_delta':         spike_delta,
                'disp_delta':          disp_delta,
                'delta_at_high':       delta_at_high,
                'delta_at_low':        delta_at_low,
                'big_buy':             big_buy,
                'big_sell':            big_sell,
                'big_imbalance':       big_imbalance,
                'absorption':          absorption,
                'bar_delta_norm':      spike_delta / atr if atr > 0 else 0,
                'cum_delta_norm':      sv(disp_bar.get('cum_delta', 0)) / atr if atr > 0 else 0,
                'buy_sell_ratio':      bv / sv2 if sv2 > 0 else 1.0,
                'of_doi':              sv(ny_sess['of_doi'].mean()) if 'of_doi' in ny_sess.columns else 0,
                'stacked_bull':        int(ny_sess['stacked_bull'].any()) if 'stacked_bull' in ny_sess.columns else 0,
                'stacked_bear':        int(ny_sess['stacked_bear'].any()) if 'stacked_bear' in ny_sess.columns else 0,
                # FVG
                'htf_fvg_aligned':     htf_fvg_aligned,
                'ob_proxy_bull':       ob_proxy_bull,
                'ob_proxy_bear':       ob_proxy_bear,
                'ob_aligned':          ob_aligned,
                'vol_x_fvg_1h':        vol_x_fvg_1h,
                # Technical
                'adx':                 adx_v,
                'adx_strong':          1 if adx_v > 25 else 0,
                'hurst':               hurst_v,
                'fisher_transform':    sv(r0['fisher_transform']),
                'fisher_extreme':      1 if abs(sv(r0['fisher_transform'])) > 2 else 0,
                'acf_lag1':            sv(r0['acf_lag1']),
                'acf_lag5':            sv(r0['acf_lag5']),
                'kalman_smooth':       sv(r0['kalman_smooth']),
                'garch_vol':           sv(r0['garch_vol']),
                'rvol':                sv(r0['rvol'], 1.0),
                'vix_proxy_20d':       vix_proxy_20d,
                'atr_vs_10d':          atr_vs_10d,
                # New: Rolling regime features
                'vix_proxy_5d':        vix_proxy_5d,
                'vol_regime':          vol_regime,
                'atr_trend':           atr_trend,
                'adx_10d_mean':        adx_10d_mean,
                'atr_5d':              atr_5d,
                # New: MTF ICT features
                'dist_bear_5m':        dist_bear_5m,
                'dist_bull_5m':        dist_bull_5m,
                'dist_bear_15m':       dist_bear_15m,
                'dist_bull_15m':       dist_bull_15m,
                'dist_bear_1h':        dist_bear_1h,
                'dist_bull_1h':        dist_bull_1h,
                'htf_fvg_aligned_mtf': htf_fvg_aligned_mtf,
                # New: Persistent state features
                'rolling_5sess_wr':    rolling_5sess_wr,
                'days_since_win_dir':  float(days_since_win_dir),
                'days_since_win_any':  float(days_since_win_any),
                'week_setup_count':    float(week_setup_count),
                'prev_ny_dir':         float(prev_ny_dir),
                'prev_ny_aligned':     float(prev_ny_aligned),
                # New: Calendar proximity
                'fomc_proximity':      fomc_prox,
                # New: Inline computed NOM features
                'atr_tp_norm':         atr_tp_norm,
                'disp_range_atr':      disp_range_atr,
                'lon_asia_aligned':    float(lon_asia_aligned),
                'asia_dir_x_lon_dir':  asia_dir_x_lon_dir,
                'asia_range_vs_atr5d': asia_range_vs_atr5d,
                'is_pre_nfp':          float(is_pre_nfp),
                'drive_x_sweep':       drive_x_sweep,
                'triple_x_h4':         float(triple_x_h4),
                'smt_x_sweep':         smt_x_sweep,
                'nfp_post_x_sweep':    nfp_post_x_sweep,
                'vol_x_sweep':         vol_x_sweep,
                'vol_x_ts_close':      vol_x_ts_close,
                # Time
                'day_of_week':         int(r0['day_of_week']),
                'month':               int(r0['month']),
                'is_tuesday':          1 if int(r0['day_of_week']) == 1 else 0,
                'is_wednesday':        1 if int(r0['day_of_week']) == 2 else 0,
                'is_thursday':         1 if int(r0['day_of_week']) == 3 else 0,
                'is_friday':           1 if int(r0['day_of_week']) == 4 else 0,
                # Direction
                'direction_enc':       1 if direction == 'SHORT' else 0,
                # Features lipsă (adăugate)
                'sweep_x_h4':          sweep_x_h4,
                'dir_x_hurst':         dir_x_hurst,
                'dir_x_adx':           dir_x_adx,
                'sweep_time_late':     float(sweep_time_late),
                'ny_open_in_lon':      float(ny_open_in_lon),
                'prev_ny_opposite':    float(prev_ny_opp),
            }])

            row = row.reindex(columns=pkg['features'], fill_value=0).fillna(0)
            nom_score = _nom_predict(pkg, row)

        except Exception as e:
            logger.warning(f"NOM score error: {e}", exc_info=True)

    if nom_score < MODEL_THRESHOLD:
        logger.info(f"NOM: {direction} detectat, scor {nom_score:.2f} < {MODEL_THRESHOLD} → WAIT")
        return None

    # ── SL / TP ───────────────────────────────────────────────────────────────
    if direction == 'SHORT':
        recent_hi = float(day_df['high'].iloc[-5:].max())
        sl_dist   = max(4.0, min(recent_hi - entry + 1.0, MAX_SL_PT))
        sl_price  = entry + sl_dist
        tp_price  = entry - sl_dist * TP_MULT
    else:
        recent_lo = float(day_df['low'].iloc[-5:].min())
        sl_dist   = max(4.0, min(entry - recent_lo + 1.0, MAX_SL_PT))
        sl_price  = entry - sl_dist
        tp_price  = entry + sl_dist * TP_MULT

    # ── Staleness: preț s-a mișcat prea mult față de entry → setup expirat ──────
    current_close = float(day_df['close'].iloc[-1])
    dist_from_entry = abs(current_close - entry)
    if dist_from_entry > 1.5 * atr:
        logger.info(f"NOM: setup expirat (dist={dist_from_entry:.1f}pt > 1.5×ATR={1.5*atr:.1f}) → SKIP")
        return None

    result = {
        'direction':  direction,
        'setup_type': 'NY_NOM',
        'session':    'NY',
        'entry':      round(entry, 2),
        'sl':         round(sl_price, 2),
        'sl_pt':      round(sl_dist, 2),
        'tp':         round(tp_price, 2),
        'tp_pt':      round(sl_dist * TP_MULT, 2),
        'rr':         round(TP_MULT, 2),
        'nom_score':  round(nom_score, 3),
        'ml_score':   round(nom_score, 3),
        'nom_prob':   round(nom_score, 3),   # alias for stacking
        'regime_enc': int(regime_enc),
        'regime_str': regime or 'EXPANSION',
        'sweep_feats': _sweep_feats,
        'tp_level':   'NOM_TARGET',
        'swept_lon':  f"{'HIGH' if direction=='SHORT' else 'LOW'} London sweepuit cu {spike_mag:.1f}pt",
        'message':    (f"✅ NOM_{direction} | spike={spike_mag:.1f}pt | "
                       f"score={nom_score:.2f} | Entry {round(entry,2)} | "
                       f"SL {round(sl_dist,1)}pt | RR {TP_MULT:.1f}")
    }
    logger.info(f"🎯 NOM CONFIRMAT: {result['message']}")
    return result
