"""
London Open Manipulation (LOM) Checker v1.1 — EVENT-DRIVEN
===========================================================
v1.1: Sweep detectat dinamic pe toată sesiunea LON (04:00-07:00 ET),
      nu doar în fereastra fixă 08:00-08:30 UTC.
      Regimul (PRE_EXPANSION) înlocuiește fereastra ca filtru principal.

Pattern: sweep față de pre-London range (Asia session high/low)
         + displacement în direcție opusă
         + regime = PRE_EXPANSION (sau UNKNOWN ca fallback)

OOS (2025-2026, event-driven, full session LON 04:00-07:00 ET):
  AUC=0.677
  IS AUC=0.817 | 651 IS setups (765 zile) | best_iter=131
"""

import pickle, sqlite3, logging, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger("ALADIN.LOM")

# ── v1.1 EVENT-DRIVEN: ferestre ET (DB stochează ET, UTC-4) ──────────────────
_ET_H = 4

# ── Daily cache for regime features (computed once per day) ──────────────────
_LOM_DAILY = {
    'date': None,
    'regime': {},
    'mtf': None,
}

# Sesiunea London în ET: 04:00-07:00 ET = 08:00-11:00 UTC
LON_SESS_START_ET = 400    # 04:00 ET — London open
LON_SESS_END_ET   = 700    # 07:00 ET — London mid
# Pre-London = Asia (00:00-03:59 ET)
PRE_LON_END_ET    = 359    # 03:59 ET
ASIA_START_ET     = 0      # 00:00 ET
ASIA_END_ET       = 359    # 03:59 ET

# UTC gate: 08:00-11:00 UTC
LOM_UTC_START     = 800
LOM_UTC_END       = 1100

MIN_SPIKE_PT      = 5.0
MIN_DISP_PT       = 4.0
MAX_SL_PT         = 12.0
TP_MULT           = 1.5    # 1.5R pentru LOM (London mai puțin spațiu)
MODEL_THRESHOLD   = 0.65   # WR=65.1% la 0.65 (OOS 2025-2026, 83 setups)
import sys as _sys
_LOM_DIR = str(Path(__file__).parent)
if _LOM_DIR not in _sys.path:
    _sys.path.insert(0, _LOM_DIR)

# Prioritate: v4 ensemble → v1 single (fallback)
_MODEL_CANDIDATES = [
    Path(__file__).parent / "lom_model_v4.pkl",
    Path(__file__).parent / "lom_model_v1.pkl",
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if p.exists()), _MODEL_CANDIDATES[-1])

_LOM_MODEL = None


def _lom_predict(pkg, row_df: "pd.DataFrame") -> float:
    """Dispatch prediction: ensemble (N modele) sau single model."""
    import numpy as np
    if pkg.get('type') == 'ensemble':
        preds = [float(m.predict_proba(row_df)[0, 1]) for m in pkg['models']]
        return float(np.mean(preds))
    return float(pkg['model'].predict_proba(row_df)[0, 1])


def _load_lom_model():
    global _LOM_MODEL
    if _LOM_MODEL is not None:
        return _LOM_MODEL
    if not MODEL_PATH.exists():
        logger.warning(f"LOM model lipsă: {MODEL_PATH}")
        return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            _LOM_MODEL = pickle.load(f)
        mtype = _LOM_MODEL.get('type', 'single')
        nmod  = _LOM_MODEL.get('n_models', 1) if mtype == 'ensemble' else 1
        logger.info(f"✅ LOM loaded [{MODEL_PATH.name}]: type={mtype} n={nmod} "
                    f"OOS={_LOM_MODEL.get('oos_auc',0):.3f} | {_LOM_MODEL.get('n_features',0)} feats")
        return _LOM_MODEL
    except Exception as e:
        logger.error(f"LOM load error: {e}")
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


def _load_lom_state():
    """Load LOM persistent state (last wins, prev closes)."""
    state_path = Path(__file__).parent / "lom_checker_state.json"
    default = {
        'last_win_LONG': None,
        'last_win_SHORT': None,
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


def check_lom_setup(db_path: str, now_utc: datetime = None,
                    regime: str = None, regime_prob: float = 0.0,
                    regime_enc: int = 2) -> dict | None:
    """
    Checker LOM event-driven v1.1.

    Detectează sweep față de pre-London range ORICÂND în sesiunea LON (04:00-07:00 ET).
    Regimul înlocuiește fereastra fixă ca filtru principal.

    Args:
        regime:      regimul curent din classify_regime() — ex. 'PRE_EXPANSION'
        regime_prob: probabilitatea regimului (0-1)
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    hhmm_utc = now_utc.hour * 100 + now_utc.minute

    # Gate UTC: sesiunea London activă 08:00-11:00 UTC
    if not (LOM_UTC_START <= hhmm_utc <= LOM_UTC_END):
        return None

    # Regime gate: preferăm PRE_EXPANSION, acceptăm UNKNOWN ca fallback
    if regime is not None and regime_prob >= 0.65:
        if regime == 'CONSOLIDATION':
            logger.debug("LOM: CONSOLIDATION detectat → skip")
            return None
        if regime in ('EXPANSION', 'DISTRIBUTION') and regime_prob >= 0.75:
            logger.debug(f"LOM: {regime} → nu e momentul pentru sweep manipulation")
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
        logger.error(f"LOM DB error: {e}")
        return None

    if len(day_df) < 15:
        return None

    day_df['ts']   = pd.to_datetime(day_df['timestamp'])
    day_df['hhmm'] = day_df['ts'].dt.hour * 100 + day_df['ts'].dt.minute

    # ── v1.1: Sesiuni dinamice ────────────────────────────────────────────────
    asia_bars = day_df[day_df['hhmm'].between(ASIA_START_ET, ASIA_END_ET)]
    pre_lon   = day_df[day_df['hhmm'] <= PRE_LON_END_ET]   # tot ce e înainte de London open
    lon_sess  = day_df[day_df['hhmm'].between(LON_SESS_START_ET, LON_SESS_END_ET)]

    if len(pre_lon) < 5 or len(lon_sess) < 3:
        logger.debug(f"LOM: date insuficiente pre_lon={len(pre_lon)} lon_sess={len(lon_sess)}")
        return None

    # ── Pre-London range (Asia session) ──────────────────────────────────────
    pre_hi  = float(pre_lon['high'].max())
    pre_lo  = float(pre_lon['low'].min())
    pre_rng = pre_hi - pre_lo
    if pre_rng < 3:
        return None

    # ATR
    atr_vals = day_df['atr_14'].replace(0, np.nan).dropna()
    atr = float(atr_vals.iloc[-1]) if len(atr_vals) > 0 else 10.0

    # ── v1.1: Detectare sweep dinamic în toată sesiunea London ───────────────
    spike_hi  = float(lon_sess['high'].max())
    spike_lo  = float(lon_sess['low'].min())
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
        spike_bar_idx = lon_sess['high'].idxmax()
    elif spike_dn > spike_up and sweep_dn_valid and spike_dn >= MIN_SPIKE_PT * 0.5:
        direction     = 'LONG'
        spike_mag     = max(spike_dn, 0)
        spike_bar_idx = lon_sess['low'].idxmin()

    if direction is None:
        logger.debug(f"LOM: sweep insuficient (up={spike_up:.1f} dn={spike_dn:.1f} min={MIN_SPIKE_PT})")
        return None

    spike_bar   = day_df.loc[spike_bar_idx]
    after_spike = day_df[day_df.index > spike_bar_idx]
    # 45min după spike (generos — London are mai puțin spațiu)
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
        logger.debug(f"LOM: {direction} spike detectat dar fără displacement")
        return None

    entry       = float(disp_bar['close'])
    pkg         = _load_lom_model()
    lom_score   = 0.5
    _sweep_feats = {}   # compact features for sweep_scorer — populated inside try block

    if pkg is not None:
        try:
            def sv(v, d=0.0):
                try: x = float(v); return x if np.isfinite(x) else d
                except: return d

            r0      = day_df.iloc[-1]
            dir_num = -1 if direction == 'SHORT' else 1
            disp_body = abs(disp_bar['close'] - disp_bar['open'])
            spike_bar_range = sv(spike_bar['high'] - spike_bar['low'], 1.0)

            # ── anti-fakeout (ts_close_inside) ───────────────────────────────
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

            wick_pct          = wick * atr / spike_bar_range if spike_bar_range > 0 else 0
            ts_wick_dom       = 1 if ts_wick_pct > 0.6 else 0
            ts_combo_score    = ts_close_inside * ts_rejection_str

            # ── Sweep quality ─────────────────────────────────────────────────
            sweep_wick_clean  = 1 if wick_pct > 0.5 else 0
            sweep_depth_atr   = spike_mag / atr if atr > 0 else 0
            deep_sweep        = 1 if sweep_depth_atr > 1.5 else 0
            shallow_sweep     = 1 if sweep_depth_atr < 0.5 else 0
            sweep_quality     = ts_close_inside*0.4 + sweep_wick_clean*0.3 + deep_sweep*0.2 + 0.1

            # ── HTF bias ──────────────────────────────────────────────────────
            h4_hi = sv(r0['h4_hi']); h4_lo = sv(r0['h4_lo'])
            h1_hi = sv(r0['h1_hi']); h1_lo = sv(r0['h1_lo'])
            h4_mid = (h4_hi + h4_lo) / 2 if h4_hi > 0 and h4_lo > 0 else 0
            h1_mid = (h1_hi + h1_lo) / 2 if h1_hi > 0 and h1_lo > 0 else 0
            h4_bias = 1 if entry < h4_mid else (-1 if h4_mid > 0 else 0)
            h1_bias = 1 if entry < h1_mid else (-1 if h1_mid > 0 else 0)
            h4_h1_aligned   = 1 if h4_bias == h1_bias and h4_bias != 0 else 0
            h4_bias_aligned = 1 if h4_bias == dir_num else 0

            # ── Weekly premium/discount ───────────────────────────────────────
            lw_hi = sv(r0['lw_hi']); lw_lo = sv(r0['lw_lo'])
            lw_rng = lw_hi - lw_lo
            weekly_prem = (entry - lw_lo) / lw_rng if lw_rng > 0 else 0.5
            in_weekly_premium  = 1 if weekly_prem > 0.5 else 0
            in_weekly_discount = 1 if weekly_prem < 0.5 else 0
            weekly_prem_aligned = 1 if (direction == 'SHORT' and in_weekly_premium) or \
                                       (direction == 'LONG'  and in_weekly_discount) else 0
            h4_x_weekly = (1 if h4_bias == dir_num else 0) * weekly_prem_aligned
            lw_range_atr = lw_rng / atr if atr > 0 else 0
            week_range_so_far = (day_df['high'].max() - day_df['low'].min()) / atr if atr > 0 else 0
            dist_lw_hi   = abs(entry - lw_hi) / atr if atr > 0 else 0
            dist_lw_lo   = abs(entry - lw_lo) / atr if atr > 0 else 0
            dist_prev_wk_lo = abs(entry - lw_lo) / atr if atr > 0 else 0

            # ── Asia context ──────────────────────────────────────────────────
            asia_hi_v = sv(r0['asia_hi']); asia_lo_v = sv(r0['asia_lo'])
            asia_rng  = asia_hi_v - asia_lo_v
            asia_mid  = (asia_hi_v + asia_lo_v) / 2 if asia_hi_v > 0 else entry
            # Asia close = ultimul close din sesiunea Asia (hhmm 0-359 ET)
            _asia_bars = day_df[day_df['hhmm'] <= 359]
            asia_close = float(_asia_bars['close'].iloc[-1]) if len(_asia_bars) > 0 else asia_mid
            asia_dir   = 1 if asia_close > asia_mid else -1
            dist_asia_hi_atr = abs(entry - asia_hi_v) / atr if atr > 0 and asia_hi_v > 0 else 0
            dist_asia_lo_atr = abs(entry - asia_lo_v) / atr if atr > 0 and asia_lo_v > 0 else 0
            asia_range_atr   = asia_rng / atr if atr > 0 and asia_rng > 0 else 0
            spike_vs_asia_hi = (spike_hi - asia_hi_v) / atr if atr > 0 and asia_hi_v > 0 else 0
            spike_vs_asia_lo = (asia_lo_v - spike_lo) / atr if atr > 0 and asia_lo_v > 0 else 0

            # ── London first 15min range ──────────────────────────────────────
            lon15 = lon_sess[lon_sess['hhmm'].between(LON_SESS_START_ET, LON_SESS_START_ET + 15)]
            lon15_rng = (lon15['high'].max() - lon15['low'].min()) if len(lon15) > 0 else 0
            lon15_range_atr = lon15_rng / atr if atr > 0 else 0
            lon15_close = float(lon15['close'].iloc[-1]) if len(lon15) > 0 else entry
            lon15_mid   = (float(lon15['high'].max()) + float(lon15['low'].min())) / 2 if len(lon15) > 0 else entry
            lon15_bias  = 1 if lon15_close > lon15_mid else -1
            asia_close_vs_mid = (asia_close - asia_mid) / atr if atr > 0 else 0
            in_first_15     = 1 if spike_hhmm <= LON_SESS_START_ET + 15 else 0

            # ── Equal levels ──────────────────────────────────────────────────
            eq_tol = atr * 0.3
            pre_highs = pre_lon['high'].values; pre_lows = pre_lon['low'].values
            eq_hi = sum(1 for h in pre_highs if abs(h - pre_hi) <= eq_tol) - 1
            eq_lo = sum(1 for l in pre_lows  if abs(l - pre_lo) <= eq_tol) - 1
            equal_level_score = (eq_hi if direction == 'SHORT' else eq_lo) / max(len(pre_lon), 1)

            # ── True open & PDH/PDL ───────────────────────────────────────────
            above_true_open = 1 if entry > sv(r0['true_open']) else 0
            dist_true_open  = abs(entry - sv(r0['true_open'])) / atr if atr > 0 else 0
            dist_pdh_atr    = abs(entry - sv(r0['p_hi'])) / atr if atr > 0 else 0
            dist_pdl_atr    = abs(entry - sv(r0['p_lo'])) / atr if atr > 0 else 0

            # ── Volume / delta ────────────────────────────────────────────────
            pre_vol   = float(pre_lon['volume'].sum()) if len(pre_lon) > 0 else 1.0
            lon_vol   = float(lon_sess['volume'].sum()) if len(lon_sess) > 0 else 1.0
            vol_ratio = lon_vol / pre_vol if pre_vol > 0 else 1.0

            bv  = sv(lon_sess['bar_buy_vol'].sum())
            sv2 = sv(lon_sess['bar_sell_vol'].sum())

            spike_delta   = sv(lon_sess['bar_delta'].sum())
            disp_delta    = sv(after_early['bar_delta'].sum()) if len(after_early) > 0 else 0
            delta_at_high = sv(lon_sess['delta_at_high'].sum()) if 'delta_at_high' in lon_sess.columns else 0
            delta_at_low  = sv(lon_sess['delta_at_low'].sum())  if 'delta_at_low'  in lon_sess.columns else 0
            absorption    = sv(lon_sess['absorption_score'].mean()) if 'absorption_score' in lon_sess.columns else 0

            # ── FVG alignment ─────────────────────────────────────────────────
            fvg_up_v   = int(lon_sess['fvg_up'].any())   if 'fvg_up'   in lon_sess.columns else 0
            fvg_down_v = int(lon_sess['fvg_down'].any()) if 'fvg_down' in lon_sess.columns else 0
            htf_fvg_aligned = 1 if (direction == 'SHORT' and fvg_down_v) or \
                                    (direction == 'LONG'  and fvg_up_v)  else 0

            ob_proxy_bull = int(lon_sess['stacked_bull'].any()) if 'stacked_bull' in lon_sess.columns else 0
            ob_proxy_bear = int(lon_sess['stacked_bear'].any()) if 'stacked_bear' in lon_sess.columns else 0

            # ── Technical indicators ──────────────────────────────────────────
            adx_v   = sv(r0['adx_14'])
            hurst_v = sv(r0['hurst'], 0.5)

            # ── VIX proxy ─────────────────────────────────────────────────────
            rets = day_df['close'].pct_change().dropna()
            vix_proxy_20d = float(rets.rolling(20).std().iloc[-1]) if len(rets) > 20 else 0

            # ── Rolling regime features (cached daily) ────────────────────────
            global _LOM_DAILY
            today_str = today.isoformat()
            if _LOM_DAILY['date'] != today_str:
                _LOM_DAILY['date'] = today_str
                _LOM_DAILY['regime'] = {}
                try:
                    vix_5d = float(rets.rolling(5).std().iloc[-1]) if len(rets) > 5 else 0
                    vix_20d = vix_proxy_20d
                    vix_proxy_5d = vix_5d
                    vol_regime = (vix_proxy_5d / vix_20d) if vix_20d > 0 else 1.0

                    atr_5d = float(day_df['atr_14'].rolling(5).mean().iloc[-1]) if len(day_df) > 5 else atr
                    atr_10d = float(day_df['atr_14'].rolling(10).mean().iloc[-1]) if len(day_df) > 10 else atr
                    atr_trend = (atr_5d / atr_10d) if atr_10d > 0 else 1.0

                    adx_10d_mean = float(day_df['adx_14'].rolling(10).mean().iloc[-1]) if len(day_df) > 10 else sv(r0['adx_14'])

                    _LOM_DAILY['regime'] = {
                        'vix_proxy_5d': vix_proxy_5d,
                        'vol_regime': vol_regime,
                        'atr_trend': atr_trend,
                        'adx_10d_mean': adx_10d_mean,
                        'atr_5d': atr_5d,
                    }
                except Exception as e:
                    logger.debug(f"Regime computation error: {e}")
                    _LOM_DAILY['regime'] = {
                        'vix_proxy_5d': 0.0,
                        'vol_regime': 1.0,
                        'atr_trend': 1.0,
                        'adx_10d_mean': 20.0,
                        'atr_5d': atr,
                    }

            vix_proxy_5d = _LOM_DAILY['regime'].get('vix_proxy_5d', 0.0)
            vol_regime = _LOM_DAILY['regime'].get('vol_regime', 1.0)
            atr_trend = _LOM_DAILY['regime'].get('atr_trend', 1.0)
            adx_10d_mean = _LOM_DAILY['regime'].get('adx_10d_mean', 20.0)
            atr_5d = _LOM_DAILY['regime'].get('atr_5d', atr)

            # ── MTF ICT features (cached daily, computed once) ────────────────
            try:
                if _LOM_DAILY['mtf'] is None:
                    mtf_data = {}
                    for tf_name in ['5m', '15m', '1h', '4h']:
                        mtf_data[tf_name] = {}
                    _LOM_DAILY['mtf'] = mtf_data
                mtf_data = _LOM_DAILY['mtf']
            except:
                mtf_data = {}

            dist_bear_5m = mtf_data.get('5m', {}).get('dist_bear', 5.0)
            dist_bear_15m = mtf_data.get('15m', {}).get('dist_bear', 5.0)
            dist_bear_1h = mtf_data.get('1h', {}).get('dist_bear', 5.0)
            dist_bear_4h = mtf_data.get('4h', {}).get('dist_bear', 5.0)
            dist_bull_5m = mtf_data.get('5m', {}).get('dist_bull', 5.0)
            dist_bull_15m = mtf_data.get('15m', {}).get('dist_bull', 5.0)
            dist_bull_1h = mtf_data.get('1h', {}).get('dist_bull', 5.0)
            in_ifvg_b_1h = mtf_data.get('1h', {}).get('in_ifvg_b', 0)
            in_ifvg_s_15m = mtf_data.get('15m', {}).get('in_ifvg_s', 0)
            in_ifvg_s_1h = mtf_data.get('1h', {}).get('in_ifvg_s', 0)
            breaker_s_5m = mtf_data.get('5m', {}).get('breaker_s', 0)
            htf_fvg_aligned_mtf = max(mtf_data.get('1h', {}).get('fvg_aligned', 0),
                                      mtf_data.get('4h', {}).get('fvg_aligned', 0))

            # ── Persistent state (prev closes) ────────────────────────────────
            lom_state = _load_lom_state()
            prev_lon_dir = 0
            prev_lon_aligned = 0

            try:
                prev_close = lom_state.get('prev_dir_close', entry)
                prev_mid = lom_state.get('prev_dir_mid', entry)
                if prev_close and prev_mid:
                    prev_lon_dir = 1 if prev_close > prev_mid else -1
                    prev_lon_aligned = 1 if prev_lon_dir == dir_num else 0
            except:
                pass

            # ── Calendar proximity ───────────────────────────────────────────
            cal = _load_calendar()
            fomc_prox = _fomc_proximity(today_str, cal.get('fomc', []))

            # ── LOM-specific inline features ─────────────────────────────────
            asia_range_vs_atr5d = asia_rng / atr_5d if atr_5d > 0 else 0
            regime_trending     = 1 if adx_10d_mean > 22 else 0
            atr_vs_5d           = float(np.clip(atr / atr_5d if atr_5d > 0 else 1.0, 0.5, 2.0))
            sweep_x_htf_fvg     = sweep_quality * htf_fvg_aligned_mtf
            vol_x_htf_fvg       = vol_regime * htf_fvg_aligned_mtf
            # Features lipsă
            asia_dir_explicit   = float(asia_dir)  # +1=bullish Asia, -1=bearish
            asia_dir_x_h4       = float(asia_dir) * float(h4_bias)
            sweep_x_eq_level    = sweep_quality * equal_level_score
            sweep_aligned_eq    = sweep_quality * ((eq_hi if direction=='SHORT' else eq_lo) / max(1, len(pre_lon)))
            ts_sweep_depth_pts  = spike_mag  # alias
            bar_hhmm            = spike_hhmm  # sweep bar hhmm
            sweep_time_early    = 1 if bar_hhmm <= LON_SESS_START_ET + 30 else 0
            sweep_time_late     = 1 if bar_hhmm > LON_SESS_START_ET + 90 else 0
            vol_x_sweep         = vol_regime * sweep_quality
            fvg_up_any          = int(lon_sess['fvg_up'].any()) if 'fvg_up' in lon_sess.columns else 0
            fvg_dn_any          = int(lon_sess['fvg_down'].any()) if 'fvg_down' in lon_sess.columns else 0
            vol_x_fvg           = vol_ratio * (1 if (direction=='LONG' and fvg_up_any) or (direction=='SHORT' and fvg_dn_any) else 0)
            equal_hi_count      = float(eq_hi)

            # ── Sweep bar specific features (NUOVO) ──────────────────────────
            _sb_range = max(sv(spike_bar['high']) - sv(spike_bar['low']), 0.01)
            _sb_body  = abs(sv(spike_bar['close']) - sv(spike_bar['open']))
            sweep_bar_body_pct = _sb_body / _sb_range
            if direction == 'SHORT':
                sweep_bar_wick_pct = (sv(spike_bar['high']) - max(sv(spike_bar['close']), sv(spike_bar['open']))) / _sb_range
            else:
                sweep_bar_wick_pct = (min(sv(spike_bar['close']), sv(spike_bar['open'])) - sv(spike_bar['low'])) / _sb_range
            _sess_mean_vol = max(lon_sess['volume'].mean(), 1)
            sweep_bar_vol_ratio = sv(spike_bar['volume']) / _sess_mean_vol
            _lon_before = lon_sess[lon_sess['hhmm'] < bar_hhmm]
            if direction == 'SHORT':
                n_level_tests = len(_lon_before[_lon_before['high'] >= pre_hi - 1.0])
            else:
                n_level_tests = len(_lon_before[_lon_before['low'] <= pre_lo + 1.0])
            _entry_hhmm   = int(disp_bar['hhmm']) if 'hhmm' in disp_bar.index else bar_hhmm
            bars_to_disp  = float(_entry_hhmm - bar_hhmm)
            _pre5 = _lon_before.tail(5)
            pre5_mom = 0.0
            if len(_pre5) >= 3:
                _closes = _pre5['close'].astype(float).values
                _changes = np.sign(np.diff(_closes))
                pre5_mom = float(_changes.mean())

            # ── Compact sweep features (for sweep_scorer) ─────────────────────
            _sweep_feats = {
                'session_enc':         0,
                'spike_mag_atr':       spike_mag / atr if atr > 0 else 0,
                'pre_rng_atr':         pre_rng / atr if atr > 0 else 0,
                'ts_close_inside':     ts_close_inside,
                'ts_rejection_str':    ts_rejection_str,
                'sweep_wick_atr':      wick,
                'sweep_depth_atr':     sweep_depth_atr,
                'deep_sweep':          int(deep_sweep),
                'sweep_quality':       sweep_quality,
                'disp_body_atr':       abs(sv(disp_bar['close']) - sv(disp_bar['open'])) / atr if atr > 0 else 0,
                'h4_bias':             h4_bias,
                'h4_bias_aligned':     h4_bias_aligned,
                'weekly_premium_pct':  weekly_prem,
                'weekly_prem_aligned': weekly_prem_aligned,
                'lw_range_atr':        lw_range_atr,
                'dist_lw_hi':          dist_lw_hi,
                'dist_lw_lo':          dist_lw_lo,
                'dist_asia_hi_atr':    dist_asia_hi_atr,
                'dist_asia_lo_atr':    dist_asia_lo_atr,
                'asia_range_atr':      asia_range_atr,
                'asia_dir':            float(asia_dir),
                'asia_dir_aligned':    1 if asia_dir == dir_num else 0,
                'adx':                 adx_v,
                'hurst':               hurst_v,
                'garch_vol':           sv(r0['garch_vol']),
                'rvol':                sv(r0['rvol'], 1.0),
                'fisher_transform':    sv(r0['fisher_transform']),
                'acf_lag1':            sv(r0['acf_lag1']),
                'acf_lag5':            sv(r0['acf_lag5']),
                'kalman_smooth':       sv(r0['kalman_smooth']),
                'kalman_noise':        sv(r0.get('kalman_noise', 0.0)),
                'is_nfp_day':          0,
                'is_fomc_day':         1 if fomc_prox <= 1 else 0,
                'is_news_day':         1 if fomc_prox <= 1 else 0,
                'direction_enc':       1 if direction == 'SHORT' else 0,
                'day_of_week':         int(r0['day_of_week']),
                'is_thursday':         1 if int(r0['day_of_week']) == 3 else 0,
                'is_friday':           1 if int(r0['day_of_week']) == 4 else 0,
                'month':               int(r0['month']),
                'regime_enc':          regime_enc,
                'is_pre_expansion':    1 if (regime or '') == 'PRE_EXPANSION' else 0,
                'is_expansion':        1 if (regime or '') == 'EXPANSION' else 0,
                'is_retracement':      1 if (regime or '') == 'RETRACEMENT' else 0,
                'dir_x_adx':           dir_num * adx_v,
                'dir_x_hurst':         dir_num * hurst_v,
                'vol_x_sweep':         vol_x_sweep,
                'h4_x_weekly':         h4_x_weekly,
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
                'ts_htf_anti':         0,
                'ts_combo_score':      ts_combo_score,
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
                # Asia context (LOM-specific)
                'dist_asia_hi_atr':    dist_asia_hi_atr,
                'dist_asia_lo_atr':    dist_asia_lo_atr,
                'asia_range_atr':      asia_range_atr,
                'spike_vs_asia_hi':    spike_vs_asia_hi,
                'spike_vs_asia_lo':    spike_vs_asia_lo,
                # London first 15min
                'lon15_range_atr':     lon15_range_atr,
                'in_first_15':         in_first_15,
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
                'big_buy':             1 if vol_ratio > 2.0 and direction == 'LONG'  else 0,
                'big_sell':            1 if vol_ratio > 2.0 and direction == 'SHORT' else 0,
                'big_imbalance':       1 if vol_ratio > 2.0 else 0,
                'absorption':          absorption,
                'bar_delta_norm':      spike_delta / atr if atr > 0 else 0,
                'cum_delta_norm':      sv(disp_bar.get('cum_delta', 0)) / atr if atr > 0 else 0,
                'buy_sell_ratio':      bv / sv2 if sv2 > 0 else 1.0,
                'of_doi':              sv(lon_sess['of_doi'].mean()) if 'of_doi' in lon_sess.columns else 0,
                'stacked_bull':        ob_proxy_bull,
                'stacked_bear':        ob_proxy_bear,
                # FVG
                'fvg_up':              fvg_up_v,
                'fvg_down':            fvg_down_v,
                'htf_fvg_aligned':     htf_fvg_aligned,
                'ob_proxy_bull':       ob_proxy_bull,
                'ob_proxy_bear':       ob_proxy_bear,
                'ob_aligned':          0,
                # New: Rolling regime features
                'vix_proxy_5d':        vix_proxy_5d,
                'vol_regime':          vol_regime,
                'atr_trend':           atr_trend,
                'adx_10d_mean':        adx_10d_mean,
                'atr_5d':              atr_5d,
                # New: MTF ICT features
                'dist_bear_5m':        dist_bear_5m,
                'dist_bear_15m':       dist_bear_15m,
                'dist_bear_1h':        dist_bear_1h,
                'dist_bear_4h':        dist_bear_4h,
                'dist_bull_5m':        dist_bull_5m,
                'dist_bull_15m':       dist_bull_15m,
                'dist_bull_1h':        dist_bull_1h,
                'in_ifvg_b_1h':        float(in_ifvg_b_1h),
                'in_ifvg_s_15m':       float(in_ifvg_s_15m),
                'in_ifvg_s_1h':        float(in_ifvg_s_1h),
                'breaker_s_5m':        float(breaker_s_5m),
                'htf_fvg_aligned_mtf': htf_fvg_aligned_mtf,
                # New: Persistent state features
                'prev_lon_dir':        float(prev_lon_dir),
                'prev_lon_aligned':    float(prev_lon_aligned),
                # New: Calendar proximity
                'fomc_proximity':      fomc_prox,
                # New: Inline computed LOM features
                'asia_range_vs_atr5d': asia_range_vs_atr5d,
                'regime_trending':     float(regime_trending),
                'atr_vs_5d':           atr_vs_5d,
                'sweep_x_htf_fvg':     sweep_x_htf_fvg,
                'vol_x_htf_fvg':       vol_x_htf_fvg,
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
                'vol_high':            1 if sv(r0['rvol'], 1) > 1.5 else 0,
                'vol_low':             1 if sv(r0['rvol'], 1) < 0.7 else 0,
                'vix_proxy_20d':       vix_proxy_20d,
                # Time
                'day_of_week':         int(r0['day_of_week']),
                'is_thursday':         1 if int(r0['day_of_week']) == 3 else 0,
                'month':               int(r0['month']),
                # Interaction
                'dir_x_adx':           dir_num * adx_v,
                'dir_x_hurst':         dir_num * hurst_v,
                'h4_x_weekly':         h4_x_weekly,
                'sweep_x_h4':          sweep_quality * (1 if h4_bias == dir_num else 0),
                'ts_close_x_h4':       ts_close_inside * (1 if h4_bias == dir_num else 0),
                # Direction
                'direction_enc':       1 if direction == 'SHORT' else 0,
                # Features lipsă (adăugate)
                'lon15_bias':          float(lon15_bias),
                'asia_close_vs_mid':   asia_close_vs_mid,
                'asia_dir_explicit':   asia_dir_explicit,
                'asia_dir_x_h4':       asia_dir_x_h4,
                'sweep_x_eq_level':    sweep_x_eq_level,
                'sweep_aligned_eq':    sweep_aligned_eq,
                'ts_sweep_depth_pts':  ts_sweep_depth_pts,
                'sweep_time_early':    float(sweep_time_early),
                'sweep_time_late':     float(sweep_time_late),
                'vol_x_sweep':         vol_x_sweep,
                'vol_x_fvg':           vol_x_fvg,
                'equal_hi_count':      equal_hi_count,
            }])

            row = row.reindex(columns=pkg['features'], fill_value=0).fillna(0)
            lom_score = _lom_predict(pkg, row)

        except Exception as e:
            logger.warning(f"LOM score error: {e}", exc_info=True)

    if lom_score < MODEL_THRESHOLD:
        logger.info(f"LOM: {direction} detectat, scor {lom_score:.2f} < {MODEL_THRESHOLD} → WAIT")
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

    # ── Staleness: preț s-a mișcat prea mult față de entry → setup expirat ──
    current_close = float(day_df['close'].iloc[-1])
    dist_from_entry = abs(current_close - entry)
    if dist_from_entry > 1.5 * atr:
        logger.info(f"LOM: setup expirat (dist={dist_from_entry:.1f}pt > 1.5×ATR={1.5*atr:.1f}) → SKIP")
        return None

    result = {
        'direction':  direction,
        'setup_type': 'LON_LOM',
        'session':    'LON',
        'entry':      round(entry, 2),
        'sl':         round(sl_price, 2),
        'sl_pt':      round(sl_dist, 2),
        'tp':         round(tp_price, 2),
        'tp_pt':      round(sl_dist * TP_MULT, 2),
        'rr':         round(TP_MULT, 2),
        'lom_score':  round(lom_score, 3),
        'ml_score':   round(lom_score, 3),
        'lom_prob':   round(lom_score, 3),   # alias for stacking
        'regime_enc': int(regime_enc),
        'regime_str': regime or 'EXPANSION',
        'sweep_feats': _sweep_feats,
        'tp_level':   'LOM_TARGET',
        'swept_asia': f"{'HIGH' if direction=='SHORT' else 'LOW'} Asia sweepuit cu {spike_mag:.1f}pt",
        'message':    (f"✅ LOM_{direction} | spike={spike_mag:.1f}pt | "
                       f"score={lom_score:.2f} | Entry {round(entry,2)} | "
                       f"SL {round(sl_dist,1)}pt | RR {TP_MULT:.1f}")
    }
    logger.info(f"🎯 LOM CONFIRMAT: {result['message']}")
    return result
