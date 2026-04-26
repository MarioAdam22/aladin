"""
regime_runtime.py — Unified Multi-Scale Regime Provider (Runtime)
=================================================================
Wrapper peste toate straturile de regim:
  1. Bar-level classifier  → regime_classifier_v1 (5 stări MESO)
  2. Hierarchical HMM      → hierarchical_regime_v1 (macro BULL/BEAR/SIDEWAYS)
  3. Bayesian intraday     → bayesian_regime_updater_v1 (update per bară)

Expune o singură funcție:
    from regime_runtime import get_regime
    r = get_regime(db_path, now_utc)
    # r = {
    #   'regime':      'PRE_EXPANSION',   # string MESO
    #   'regime_enc':  1,                 # 0=CONS 1=PRE 2=EXP 3=RET 4=DIST
    #   'regime_prob': 0.82,
    #   'entropy':     0.21,
    #   'macro':       'BULL',            # macro HMM
    #   'macro_prob':  0.74,
    #   'bayesian_n':  5,                 # n bare Bayesian update-uri
    #   'source':      'bayesian',        # bayesian / classifier / fallback
    # }

Usage în ict_gate_v3.py:
    regime_info = get_regime(db_path, now_utc)
    regime      = regime_info['regime']
    regime_enc  = regime_info['regime_enc']
    regime_prob = regime_info['regime_prob']
"""

import logging
import sqlite3
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("ALADIN.Regime")

BASE = Path(__file__).parent

MESO_STATES = ['CONSOLIDATION', 'PRE_EXPANSION', 'EXPANSION', 'RETRACEMENT', 'DISTRIBUTION']
MESO_ENC    = {s: i for i, s in enumerate(MESO_STATES)}
# MACRO HMM states → friendly names
MACRO_NAMES = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}

_ET_H = 4   # UTC → ET offset

# ── Module-level caches ────────────────────────────────────────────────────────
_hierarchical_pkg = None    # hierarchical_regime_v1.pkl
_bayesian_pkg     = None    # bayesian_regime_updater_v1.pkl (raw pkg dict)

# Per-session Bayesian state: resets at session open
_bayesian_state = {
    'date': None,       # 'YYYY-MM-DD' of current session
    'session': None,    # 'LON' / 'NY'
    'updater': None,    # BayesianRegimeUpdater instance
}


# ═══════════════════════════════════════════════════════════════════════════════
# Loaders (lazy, singleton)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_hierarchical():
    global _hierarchical_pkg
    if _hierarchical_pkg is not None:
        return _hierarchical_pkg
    p = BASE / 'hierarchical_regime_v1.pkl'
    if not p.exists():
        return None
    try:
        _hierarchical_pkg = pickle.load(open(p, 'rb'))
        logger.info("✅ HierarchicalRegime loaded")
        return _hierarchical_pkg
    except Exception as e:
        logger.warning(f"HierarchicalRegime load error: {e}")
        return None


def _load_bayesian_pkg():
    global _bayesian_pkg
    if _bayesian_pkg is not None:
        return _bayesian_pkg
    p = BASE / 'bayesian_regime_updater_v1.pkl'
    if not p.exists():
        return None
    try:
        _bayesian_pkg = pickle.load(open(p, 'rb'))
        logger.info("✅ BayesianRegimeUpdater pkg loaded")
        return _bayesian_pkg
    except Exception as e:
        logger.warning(f"BayesianUpdater load error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Macro regime from Hierarchical HMM
# ═══════════════════════════════════════════════════════════════════════════════

def _get_macro_regime(db_path: str, date_str: str) -> tuple:
    """
    Returnează (macro_str, macro_prob) folosind HierarchicalRegime pe date zilnice.
    Fallback: 'BULL', 0.5
    """
    pkg = _load_hierarchical()
    if pkg is None:
        return 'BULL', 0.5

    try:
        macro_model = pkg.get('macro_model')
        if macro_model is None:
            return 'BULL', 0.5

        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=20)
        sql = """
        SELECT date,
               AVG(adx_14) AS adx_14, AVG(hurst) AS hurst,
               AVG(garch_vol) AS garch_vol, AVG(atr_14) AS atr_14,
               AVG(kalman_smooth) AS kalman_smooth,
               MAX(close) AS close, MAX(true_open) AS true_open,
               MAX(h4_hi) AS h4_hi, MIN(h4_lo) AS h4_lo,
               MAX(p_hi) AS p_hi, MIN(p_lo) AS p_lo,
               AVG(dist_vwap) AS dist_vwap, AVG(dist_poc) AS dist_poc
        FROM market_data
        WHERE date <= ? AND adx_14 > 0 AND atr_14 > 0
          AND day_of_week BETWEEN 1 AND 5
        GROUP BY date ORDER BY date DESC LIMIT 20
        """
        df = pd.read_sql(sql, conn, params=(date_str,))
        conn.close()
        if df.empty:
            return 'BULL', 0.5

        df = df.iloc[::-1].reset_index(drop=True)
        macro_feats = pkg.get('macro_features', [])
        if not macro_feats:
            return 'BULL', 0.5

        atr = df['atr_14'].clip(lower=1)
        df['atr_5d'] = atr.rolling(5, min_periods=2).mean().shift(1).fillna(atr)
        df['atr_vs_5d'] = atr / df['atr_5d'].clip(lower=1)
        df['h4_range'] = (df['h4_hi'].fillna(df['close']) - df['h4_lo'].fillna(df['close']))
        df['h4_range_atr'] = df['h4_range'] / atr
        df['dist_vwap_atr'] = df['dist_vwap'].fillna(0).abs() / atr

        row = df.tail(1)[macro_feats].fillna(0)
        row = row.reindex(columns=macro_feats, fill_value=0)
        proba = macro_model.predict_proba(row)[0]
        cls   = int(np.argmax(proba))
        prob  = float(proba[cls])
        macro_str = MACRO_NAMES.get(cls, 'BULL')
        return macro_str, prob

    except Exception as e:
        logger.debug(f"Macro regime error: {e}")
        return 'BULL', 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Bayesian Intraday Update
# ═══════════════════════════════════════════════════════════════════════════════

def _get_bayesian_regime(bar_features: dict, date_str: str,
                         session: str, prior_probs: dict = None) -> dict:
    """
    Update-ează posterioară Bayesiană cu bara curentă.
    Resetează la fiecare sesiune nouă.
    Returnează dict cu toate câmpurile.
    """
    global _bayesian_state

    pkg = _load_bayesian_pkg()
    if pkg is None:
        return None

    try:
        from bayesian_regime_updater import BayesianRegimeUpdater

        # Reset la sesiune nouă
        sess_key = f"{date_str}_{session}"
        if _bayesian_state['date'] != sess_key or _bayesian_state['updater'] is None:
            updater = BayesianRegimeUpdater(pkg)
            updater.reset(prior_probs)
            _bayesian_state['date']    = sess_key
            _bayesian_state['session'] = session
            _bayesian_state['updater'] = updater
            logger.debug(f"Bayesian reset pentru sesiunea {sess_key}")

        result = _bayesian_state['updater'].update(bar_features)
        return result

    except Exception as e:
        logger.debug(f"Bayesian update error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Bar features extractor (pentru Bayesian updater)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_bar_features(db_path: str, now_et: datetime) -> dict:
    """
    Extrage features din ultima bară pentru Bayesian update.
    """
    date_str = now_et.date().isoformat()
    hhmm_str = now_et.strftime('%H:%M')
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=15)
        df = pd.read_sql(f"""
            SELECT adx_14, hurst, garch_vol, fisher_transform,
                   dist_vwap, atr_14, close, true_open,
                   h4_hi, h4_lo, asia_hi, asia_lo, p_hi, p_lo
            FROM market_data
            WHERE date = '{date_str}'
              AND hour_min <= '{hhmm_str}'
              AND adx_14 > 0 AND atr_14 > 0
            ORDER BY timestamp DESC LIMIT 5
        """, conn)
        conn.close()
        if df.empty:
            return {}
        row = df.iloc[0]
        atr = max(float(row['atr_14']), 1.0)
        def sv(x, d=0.0):
            try: v=float(x); return d if (np.isnan(v) or np.isinf(v)) else v
            except: return d
        h4_mid = (sv(row['h4_hi'], sv(row['close'])) + sv(row['h4_lo'], sv(row['close']))) / 2
        asia_hi = sv(row.get('asia_hi', 0))
        asia_lo = sv(row.get('asia_lo', 0))
        ref_hi  = max(sv(row.get('p_hi', 0)), asia_hi)
        return {
            'adx_14':              sv(row['adx_14']),
            'hurst':               sv(row['hurst'], 0.5),
            'garch_vol':           sv(row['garch_vol']),
            'fisher_transform':    sv(row['fisher_transform']),
            'dist_vwap_atr':       abs(sv(row['dist_vwap'])) / atr,
            'h4_bias_atr':         (sv(row['close']) - h4_mid) / atr,
            'above_true_open_atr': (sv(row['close']) - sv(row.get('true_open', sv(row['close'])))) / atr,
            'sweep_up_atr':        max(sv(row.get('close', 0)) - ref_hi, 0) / atr,
            'sweep_dn_atr':        0.0,
        }
    except Exception as e:
        logger.debug(f"Bar features error: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def get_regime(db_path: str, now_utc: datetime = None,
               prior_probs: dict = None) -> dict:
    """
    Returnează regimul curent complet.

    Flow:
      1. classify_regime() → bar-level XGBoost (primary)
      2. Bayesian update cu bara curentă → rafineaza probabilitățile
      3. Macro HMM → context zilnic

    Returns dict:
        regime:      str   — 'PRE_EXPANSION' etc.
        regime_enc:  int   — 0-4 (MESO encoding)
        regime_prob: float — probabilitatea regimului dominant
        entropy:     float — 0=sigur, 1=uniform
        macro:       str   — 'BULL'/'BEAR'/'SIDEWAYS'
        macro_prob:  float
        bayesian_n:  int   — n bare procesate Bayesian în sesiunea curentă
        source:      str   — 'bayesian'/'classifier'/'fallback'
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    now_et   = now_utc - timedelta(hours=_ET_H)
    date_str = now_et.date().isoformat()
    hhmm_et  = now_et.hour * 100 + now_et.minute
    session  = 'LON' if 400 <= hhmm_et < 700 else ('NY' if 900 <= hhmm_et < 1300 else 'OTHER')

    # ── 1. Bar-level classifier (primary) ─────────────────────────────────────
    try:
        from regime_classifier_v1 import classify_regime
        regime_str, regime_prob = classify_regime(db_path, now_utc)
    except Exception as e:
        logger.debug(f"classify_regime error: {e}")
        regime_str, regime_prob = 'UNKNOWN', 0.0

    regime_enc = MESO_ENC.get(regime_str, 2)   # default EXPANSION

    # ── 2. Bayesian refinement ─────────────────────────────────────────────────
    bayesian_n = 0
    entropy    = 0.5
    source     = 'classifier'

    bar_feats = _extract_bar_features(db_path, now_et)
    if bar_feats:
        bayes_result = _get_bayesian_regime(
            bar_feats, date_str, session, prior_probs
        )
        if bayes_result is not None:
            b_regime = bayes_result.get('regime', regime_str)
            b_prob   = bayes_result.get('prob', regime_prob)
            b_entropy= bayes_result.get('entropy', 0.5)
            bayesian_n = bayes_result.get('n_updates', 0)

            # Usa Bayesian se più sicuro O dopo 3+ update nel corso della sessione
            if b_entropy < 0.7 or bayesian_n >= 3:
                # Weighted blend: più aggiornamenti → più peso al Bayesian
                w_bayes = min(0.3 + bayesian_n * 0.05, 0.6)
                w_class = 1.0 - w_bayes
                if b_regime == regime_str:
                    regime_prob = w_class * regime_prob + w_bayes * b_prob
                else:
                    # Disaccordo: usa quello con probabilità più alta
                    if b_prob > regime_prob:
                        regime_str  = b_regime
                        regime_enc  = MESO_ENC.get(b_regime, regime_enc)
                        regime_prob = b_prob
                entropy = b_entropy
                source  = 'bayesian'

    # ── 3. Macro regime ────────────────────────────────────────────────────────
    macro_str, macro_prob = _get_macro_regime(db_path, date_str)

    # ── 4. Boost PRE_EXPANSION se macro = BULL ─────────────────────────────────
    if regime_str == 'PRE_EXPANSION' and macro_str == 'BULL':
        regime_prob = min(regime_prob * 1.10, 0.99)   # +10% confidence boost
    elif regime_str == 'PRE_EXPANSION' and macro_str == 'BEAR':
        regime_prob = regime_prob * 0.85               # slight penalty

    if regime_str == 'UNKNOWN':
        regime_str  = 'EXPANSION'
        regime_enc  = 2
        source      = 'fallback'

    result = {
        'regime':      regime_str,
        'regime_enc':  regime_enc,
        'regime_prob': round(float(regime_prob), 4),
        'entropy':     round(float(entropy), 4),
        'macro':       macro_str,
        'macro_prob':  round(float(macro_prob), 4),
        'bayesian_n':  bayesian_n,
        'source':      source,
    }
    logger.debug(
        f"Regime → {regime_str}({regime_enc}) p={regime_prob:.2f} "
        f"macro={macro_str} src={source} bayes_n={bayesian_n}"
    )
    return result


def reset_session(date_str: str = None, session: str = None):
    """Forțează reset Bayesian — apelat la fiecare sesiune nouă."""
    global _bayesian_state
    _bayesian_state['date']    = None
    _bayesian_state['session'] = None
    _bayesian_state['updater'] = None
    logger.info(f"Bayesian state reset ({date_str} {session})")
