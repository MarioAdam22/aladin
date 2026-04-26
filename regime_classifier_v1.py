"""
regime_classifier_v1.py — Market Regime Detector
==================================================
Detectează regimul curent al pieței NQ folosind un model XGBoost multi-clasă
antrenat pe features combinate din v2/v5/v6 + NOM/LOM/DSM.

Regimuri:
  CONSOLIDATION  — ADX scăzut, inside VA, skip tot
  PRE_EXPANSION  — sweep lichiditate la session open → NOM/LOM/DSM
  EXPANSION      — mișcare HTF activă → mario_rag + MTF
  RETRACEMENT    — pullback în expansion → ICT clasic, TS
  DISTRIBUTION   — extreme VWAP, momentum fading → TS reversal

Integrare în gate_verdict():
  regime, prob = classify_regime(db_path, now_utc)
  → routează ce checker e activ
"""

import sqlite3, joblib, logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("ALADIN")

_ET_H        = 4
_MODEL_PATH_V2 = Path(__file__).parent / "regime_classifier_v2.pkl"
_MODEL_PATH_V1 = Path(__file__).parent / "regime_classifier_v1.pkl"
_MODEL_PATH    = _MODEL_PATH_V2 if _MODEL_PATH_V2.exists() else _MODEL_PATH_V1
_CONTEXT_BARS  = 35   # bare istorice pentru rolling features

LON_OPEN_ET = (400,  700)
NY_OPEN_ET  = (900, 1130)

_pkg_cache = None


def _load_model():
    global _pkg_cache, _MODEL_PATH
    if _pkg_cache is not None:
        return _pkg_cache
    # Always prefer v2 if available (hot-reload after retrain)
    _MODEL_PATH = _MODEL_PATH_V2 if _MODEL_PATH_V2.exists() else _MODEL_PATH_V1
    if not _MODEL_PATH.exists():
        logger.warning("regime_classifier_v1.pkl / v2.pkl lipsesc")
        return None
    try:
        _pkg_cache = joblib.load(_MODEL_PATH)
        version    = _pkg_cache.get('version', 'v1')
        logger.info(f"✅ Regime classifier [{version}] încărcat | "
                    f"clase: {[_pkg_cache['regimes'][c] for c in _pkg_cache['classes']]}")
        return _pkg_cache
    except Exception as e:
        logger.error(f"Regime classifier load error: {e}")
        return None


def classify_regime(db_path: str, now_utc: datetime = None) -> tuple:
    """
    Returnează (regime_str, probability) pentru bara curentă.

    Exemple:
      'PRE_EXPANSION', 0.87
      'CONSOLIDATION', 0.92
      'EXPANSION',     0.78
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    pkg = _load_model()
    if pkg is None:
        return 'UNKNOWN', 0.0

    model    = pkg['model']
    le       = pkg['label_encoder']
    features = pkg['features']
    regimes  = pkg['regimes']

    now_et = now_utc - timedelta(hours=_ET_H)
    today  = now_et.date()

    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True,
                               timeout=30, check_same_thread=False)
        df = pd.read_sql(f"""
            SELECT timestamp, hour_min, open, high, low, close, volume,
                   adx_14, hurst, garch_vol, kalman_smooth,
                   acf_lag1, acf_lag5, fisher_transform, sample_entropy,
                   inside_va, dist_vwap, dist_poc, dist_pdh, dist_pdl,
                   has_displacement, body_size, rvol,
                   bar_delta, cum_delta,
                   delta_at_high, delta_at_low, big_buy_count, big_sell_count,
                   imbalance_pct, dom_ratio, fvg_up, fvg_down,
                   atr_14, true_open, h4_hi, h4_lo, h1_hi, h1_lo,
                   lon_hi, lon_lo, p_hi, p_lo,
                   day_of_week, month
            FROM market_data
            WHERE date = '{today}'
              AND hour_min <= '{now_et.strftime("%H:%M")}'
              AND adx_14 > 0 AND atr_14 > 0
            ORDER BY timestamp DESC
            LIMIT {_CONTEXT_BARS}
        """, conn)
        conn.close()
    except Exception as e:
        logger.debug(f"Regime DB error: {e}")
        return 'UNKNOWN', 0.0

    if df.empty or len(df) < 3:
        return 'UNKNOWN', 0.0

    df = df.iloc[::-1].reset_index(drop=True)
    row = df.iloc[-1]   # bara curentă

    def sv(x):
        try:
            v = float(x)
            return 0.0 if (np.isnan(v) or np.isinf(v)) else v
        except:
            return 0.0

    atr = max(sv(row['atr_14']), 0.01)

    # ── Feature engineering (identic cu train_regime.py) ──────────────────────
    feat = {}

    feat['adx_14']           = sv(row['adx_14'])
    feat['hurst']            = sv(row['hurst'])
    feat['garch_vol']        = sv(row['garch_vol'])
    feat['kalman_smooth']    = sv(row['kalman_smooth'])
    feat['acf_lag1']         = sv(row['acf_lag1'])
    feat['acf_lag5']         = sv(row['acf_lag5'])
    feat['fisher_transform'] = sv(row['fisher_transform'])
    feat['sample_entropy']   = sv(row['sample_entropy'])

    feat['inside_va']        = sv(row['inside_va'])
    feat['dist_vwap_atr']    = abs(sv(row['dist_vwap'])) / atr
    feat['dist_poc_atr']     = abs(sv(row['dist_poc'])) / atr
    feat['dist_pdh_atr']     = abs(sv(row['dist_pdh'])) / atr
    feat['dist_pdl_atr']     = abs(sv(row['dist_pdl'])) / atr

    feat['has_displacement'] = sv(row['has_displacement'])
    feat['body_size_atr']    = sv(row['body_size']) / atr
    feat['rvol']             = sv(row['rvol'])

    feat['bar_delta_atr']    = abs(sv(row['bar_delta'])) / max(sv(row['volume']), 1)
    # cum_delta rolling 20
    cum_d = df['cum_delta'].fillna(0).values
    feat['cum_delta_20_atr'] = float(np.sum(cum_d[-20:])) / atr

    feat['delta_at_high_atr'] = abs(sv(row['delta_at_high'])) / atr
    feat['delta_at_low_atr']  = abs(sv(row['delta_at_low'])) / atr
    feat['big_buy_count']     = sv(row['big_buy_count'])
    feat['big_sell_count']    = sv(row['big_sell_count'])
    feat['imbalance_pct']     = sv(row['imbalance_pct'])
    feat['dom_ratio']         = sv(row['dom_ratio'])

    # Session
    hhmm_str = str(row['hour_min']).replace(':', '')
    hhmm     = int(hhmm_str) if hhmm_str.isdigit() else 0
    feat['hhmm_enc']       = hhmm
    feat['is_session_open'] = int(
        (LON_OPEN_ET[0] <= hhmm <= LON_OPEN_ET[1]) or
        (NY_OPEN_ET[0]  <= hhmm <= NY_OPEN_ET[1])
    )

    sess_hi = max(sv(row['lon_hi']), sv(row['p_hi']))
    sess_lo_val = sv(row['lon_lo'])
    sess_lo_p   = sv(row['p_lo'])
    sess_lo = min(sess_lo_val if sess_lo_val > 0 else 999999,
                  sess_lo_p   if sess_lo_p   > 0 else 999999)
    if sess_lo == 999999:
        sess_lo = sv(row['close']) - atr * 5

    close = sv(row['close'])
    feat['dist_sess_hi_atr'] = abs(sess_hi - close) / atr if sess_hi > 0 else 0
    feat['dist_sess_lo_atr'] = abs(close - sess_lo) / atr

    h4_mid = (sv(row['h4_hi']) + sv(row['h4_lo'])) / 2
    h1_mid = (sv(row['h1_hi']) + sv(row['h1_lo'])) / 2
    feat['h4_bias_atr']         = (close - h4_mid) / atr if h4_mid > 0 else 0
    feat['h1_bias_atr']         = (close - h1_mid) / atr if h1_mid > 0 else 0
    feat['above_true_open_atr'] = (close - sv(row['true_open'])) / atr if sv(row['true_open']) > 0 else 0

    feat['day_of_week'] = sv(row['day_of_week'])
    feat['month']       = sv(row['month'])
    feat['fvg_up']      = sv(row['fvg_up'])
    feat['fvg_down']    = sv(row['fvg_down'])

    pre_hi  = max(sv(row['p_hi']), sv(row['lon_hi']))
    pre_lo_v = sv(row['p_lo'])
    pre_lo_l = sv(row['lon_lo'])
    pre_lo  = min(pre_lo_v if pre_lo_v > 0 else 999999,
                  pre_lo_l if pre_lo_l > 0 else 999999)
    if pre_lo == 999999:
        pre_lo = close - atr * 10
    pre_range = max(pre_hi - pre_lo, 0.01)
    feat['pre_range_atr'] = pre_range / atr

    bar_lo = sv(row['low'])
    bar_hi = sv(row['high'])
    feat['sweep_dn_atr'] = max(pre_lo - bar_lo, 0) / atr
    feat['sweep_up_atr'] = max(bar_hi - pre_hi, 0) / atr

    # ── Predicție ─────────────────────────────────────────────────────────────
    X = pd.DataFrame([{f: feat.get(f, 0.0) for f in features}]).fillna(0).astype(float)
    try:
        probs = model.predict_proba(X)[0]
        pred_enc  = int(np.argmax(probs))
        pred_prob = float(probs[pred_enc])
        pred_class = int(le.classes_[pred_enc])
        regime_str = regimes[pred_class]
        return regime_str, round(pred_prob, 3)
    except Exception as e:
        logger.debug(f"Regime predict error: {e}")
        return 'UNKNOWN', 0.0
