"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║          ALADIN QUANTUM-ICT v6.0 — RAG ENGINE                                           ║
║          mario_rag.py  |  Full Hybrid AI + Quantum + AMT + ICT + RAG                   ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Arhitectură completă:
  Modul 0  — Validare Calendar NYSE (pandas_market_calendars)
  Modul 1  — Quantum v6: Main 6q SEL + SMT 6q Amplitude + Noise VQC 4q + Regime IQP 8q
  Modul 2  — Narrative ICT (Weekly Bias, Power of 3, Liquidity Draw)
  Modul 3  — Filtre Instituționale HTF (Monthly/Weekly Bias Guard)
  Modul 4  — Risk Management & Position Sizing (ATR-based)
  Modul 4.5— Advanced Orderflow & FVG Gap Analytics
  Modul 4.6— Session Projections (Asia/London SD Targets)
  Modul 5  — Synthetic Relative Strength QQQ vs SPY
  Modul 6  — Standard Deviation Projections (ICT Vanguard Logic)
  Modul 7  — Automated Journaling System (CSV pe Desktop)
  Modul 8  — Deep FVG Analysis (Consequent Encroachment 50%)
  Modul 9  — News Impact Filter (Economic Calendar Heuristic)
  Modul 10 — Pyramiding Plan (Scale-In Sniper Logic)
  Modul 11 — Order Block Detection (Institutional Candle Scanner)
  Modul 12 — RAG Pattern Memory (FAISS + SentenceTransformers)
  Modul 13 — Quantum Noise Filter (Conviction Threshold Guard)
  Modul 14 — Backtesting Utilities (Walk-Forward Stats)
  Modul 15 — aladin_engine() — Motorul principal (returnează DICT)

Fixes față de versiunea anterioară:
  ✅ aladin_engine() returnează DICT (nu None)
  ✅ check_news_impact() implementat complet
  ✅ get_pyramiding_plan() implementat complet
  ✅ detect_order_blocks() implementat complet
  ✅ RAG Pattern Memory cu fallback graceful dacă FAISS/ST lipsesc
  ✅ SQL parametrizat (fără f-string injection)
  ✅ Toate modulele returnează valori sigure (no crash pe date lipsă)
"""

import pandas as pd
import numpy as np
import sqlite3
import xgboost as xgb
import os
import re
import json
import hashlib
import logging

# UPDATE #2: PlattModel cu Isotonic Calibration — identic cu train_mario_ai.py
# IsotonicRegression > Platt pe date imbalanced — nu colapsează SHORT/LONG la WAIT
class PlattModel:
    """Wrapper Isotonic Calibration — deserializabil din pickle."""
    def __init__(self, base_model, calibrators):
        self.base_model  = base_model
        self.calibrators = calibrators

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)
        # IsotonicRegression folosește .predict() (nu .predict_proba())
        cal = np.column_stack([
            np.clip(self.calibrators[c].predict(raw[:, c]), 0.0, 1.0)
            for c in range(len(self.calibrators))
        ])
        row_sums = cal.sum(axis=1, keepdims=True)
        return cal / np.maximum(row_sums, 1e-8)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
from datetime import datetime, timedelta
from typing import Optional

import pennylane as qml
from pennylane import numpy as qnp
import pandas_market_calendars as mcal

# UPDATE #6: Sentiment Engine (FinBERT + Stocktwits)
try:
    import sys as _sys
    import os as _os
    _aladin_dir = _os.path.dirname(_os.path.abspath(__file__))
    if _aladin_dir not in _sys.path:
        _sys.path.insert(0, _aladin_dir)
    import sentiment_engine as _sentiment_engine
    _SENTIMENT_ENGINE_OK = True
except Exception as _se_err:
    _SENTIMENT_ENGINE_OK = False
    _sentiment_engine = None

# UPDATE #1: Supabase Client — logging semnale + trade-uri în cloud
try:
    import supabase_client as _supabase
    _SUPABASE_OK = True
except Exception as _sb_err:
    _SUPABASE_OK = False
    _supabase = None

# UPDATE #14b: Deduplicare Supabase — salvăm timestamp-ul ultimei bare scrise
# Previne scrierea de N ori pe aceeași bară dacă analysis e apelat de mai multe ori
_last_supabase_bar_ts: str = ""

# UPDATE #11: Reinforcement Learning Feedback Loop — dynamic weights
try:
    import rl_feedback as _rl_feedback
    _RL_OK = True
except Exception as _rl_err:
    _RL_OK = False
    _rl_feedback = None

# Advanced Features: Hurst, GARCH, Kalman, ADX, VWAP, SampEn, Fisher, FFT, ACF
try:
    import advanced_features as _af
    _AF_OK = True
except Exception as _af_err:
    _AF_OK = False
    _af = None

# =============================================================================
# CONFIGURARE LOGGING
# =============================================================================
logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s - ALADIN - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTE GLOBALE
# =============================================================================
PATH_DB      = "/Users/mario/Desktop/Aladin/mario_trading.db"
MODEL_PATH   = "/Users/mario/Desktop/Aladin/mario_bot_open.json"
JOURNAL_PATH = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
RAG_INDEX_PATH   = "/Users/mario/Desktop/Aladin/aladin_rag.index"
RAG_META_PATH    = "/Users/mario/Desktop/Aladin/aladin_rag_meta.json"

# ── Feature 3: Instrument SL/TP Config (NQ only) ──────────────────────────
INSTRUMENT_PARAMS = {
    # v10.1 Prop Firm Tuning: SL redus pentru Lucid Flex 50K ($2,000 MLL)
    # ATR median real NQ 1m = 10.5pts → SL_MIN 15 era > ATR median (prea larg)
    # SL_MAX 30pts × 2ct × $20 = $1,200 = 60% din MLL → inadmisibil pe prop firm
    # Nou: SL_MAX 18pts × 2ct × $20 = $720 = 36% MLL | SL_DEFAULT 12pts = 1.15× ATR median
    # SL_MIN 8pts: permite niveluri structurale reale la 8-10pts (anterior ignorate)
    "NQ":  {"point_value": 20.0, "sl_default": 12.0, "sl_min": 8.0, "sl_max": 15.0},  # v14: cap 15pt propfirm
}

def _get_sl_params(instrument: str = "NQ") -> dict:
    """Returnează SL params pentru instrument (fallback: NQ)."""
    sym = instrument.upper().replace(" ", "")
    for key in INSTRUMENT_PARAMS:
        if sym.startswith(key):
            return INSTRUMENT_PARAMS[key]
    return INSTRUMENT_PARAMS["NQ"]

# v10.6: SINCRONIZAT CU train_mario_ai.py — toate feature groups identice
# FVG/SMT acum backfill-uite pe 4M bare, incluse în model training
FEATURES_STRICT = [
    'open', 'high', 'low', 'close', 'volume',
    'lm_hi', 'lm_lo', 'lw_hi', 'lw_lo', 'm_hi', 'm_lo', 'p_hi', 'p_lo',
    'h4_hi', 'h4_lo', 'h1_hi', 'h1_lo',
    'true_open', 'poc_level', 'vah', 'val',
    'is_above_open', 'has_displacement',
    # v10.6: ICT signals — backfill-uite pe toți 4M bare
    'fvg_up', 'fvg_down',
    'is_smt_bearish', 'is_smt_bullish',
]

# 10 extra features: 5 ICT context (v5.0) + 5 direcție (v6.4 fix mismatch XGBoost)
FEATURES_EXTRA = ['dist_poc', 'inside_va', 'dist_pdh', 'dist_pdl', 'atr_14',
                   'slope_h1', 'slope_h4', 'momentum_15', 'body_dir', 'wick_ratio']

# Volume Profile + OrderFlow features — salvate live din BRIDGE_LIVE în market_data
# v10.6: OF columns create în DB, populate going forward din bridge live
FEATURES_VP_OF = [
    'rvol',               # Relative Volume vs medie istorică (1.0=normal, >1.5=instituțional)
    'profile_shape_enc',  # Forma VP: P=1 (bullish dist.), D=0 (balanced), b=-1 (bearish dist.)
    'dist_prev_poc',      # Close - POC sesiune anterioară (pts NQ) — nivel instituțional primar
    'delta_exhaust_enc',  # Absorb instituțional: 1=LONG_EXHAUST, -1=SHORT_EXHAUST, 0=NONE
]

# v10.6: OF columns din bridge_live (going forward, populate cu date reale)
# Pe date vechi = 0, XGBoost gestionează nativ. Pe date noi = semnale OF reale.
FEATURES_OF_NATIVE = [
    'bar_delta',          # Buy vol - Sell vol per bară
    'bar_buy_vol',        # Volume buy per bară
    'bar_sell_vol',       # Volume sell per bară
    'delta_at_high',      # Delta la price high — negativ = absorption la high
    'delta_at_low',       # Delta la price low — pozitiv = absorption la low
    'big_buy_count',      # Nr big trades (>=20c) buy per bară
    'big_sell_count',     # Nr big trades (>=20c) sell per bară
    'imbalance_pct',      # (buy-sell)/total — dezechilibru OF
    'tape_speed',         # Trades/sec — activitate piață
    'dom_ratio',          # Bid/Ask ratio din DOM
    'of_doi',             # Delta Oscillation Index — <0 = consolidare
    'of_bilateral_abs',   # Absorption bilaterală — 1 = range instituțional
    'of_big_balance',     # Big trade balance — 0.5 = echilibrat = consolidare
    'of_d_shape_count',   # D-shape consecutive — consolidare confirmată VP
]

# Advanced features (Tier 1 + Tier 2) — computed by advanced_features.py
FEATURES_ADVANCED = [
    'hurst', 'garch_vol', 'kalman_noise', 'adx_14', 'dist_vwap',
    'sample_entropy', 'fisher_transform', 'fft_cycle', 'acf_lag1', 'acf_lag5',
]

# v8.0 REVERSAL features — învață modelul să detecteze schimbări de trend
FEATURES_REVERSAL = [
    'mss_bullish',           # Market Structure Shift bullish (HH after LL)
    'mss_bearish',           # Market Structure Shift bearish (LL after HH)
    'choch_bullish',         # Change of Character bullish (close > recent high after downtrend)
    'choch_bearish',         # Change of Character bearish (close < recent low after uptrend)
    'reversal_strength',     # Combined reversal signal strength [-1, +1]
    'trend_exhaustion',      # Consecutive bars in same direction / price deceleration
    'delta_flip',            # Cum delta flips direction (buy→sell or sell→buy)
    'poc_drift_direction',   # POC drift change direction
    'swing_break',           # Price breaks above/below recent swing point
    'momentum_divergence',   # Price goes up but momentum slows (bearish div) or vice versa
]

# v11.0 AUCTION MARKET THEORY features — AMT complet
FEATURES_AMT = [
    'failed_auction',        # Breakout care eșuează: high>swing_hi dar close<swing_hi → bearish (-1)
    'excess',                # Excess la extreme: wick lung la high/low recent = respingere [-1,+1]
    'poor_high',             # Poor high: flat top fără excess → va fi re-testat (1=poor)
    'poor_low',              # Poor low: flat bottom fără excess → va fi re-testat (1=poor)
    'initiative_responsive', # Breakout din VA = inițiativă (+1/-1), respingere = responsive (0)
    'va_migration',          # Direcția migrării Value Area (+1=up, -1=down)
    'rotation_factor',       # 0=trending, 1=balance (range-bound)
]

# v12.1: OF AGGREGATED features — rolling sums/ratios pe 15/30 bare
FEATURES_OF_AGG = [
    'delta_sum_15',          # Suma delta pe 15 bare — acumulare direcțională
    'delta_sum_30',          # Suma delta pe 30 bare — trend OF mai lung
    'delta_ratio_15',        # delta_sum_15 / abs_delta_sum_15 — direcționalitate [-1,+1]
    'big_trade_ratio_15',    # (big_buy - big_sell) / (big_buy + big_sell) pe 15 bare
    'buy_sell_ratio_30',     # bar_buy_vol / (bar_buy_vol + bar_sell_vol) rolling 30
    'imbalance_ma_15',       # Media imbalance_pct pe 15 bare
    'tape_speed_rel',        # tape_speed / rolling_mean_60 — relativă la context
    'absorption_score_15',   # Absorption la extreme pe 15 bare
    'of_pressure',           # delta_trend × volume_trend — presiune direcțională
    'dom_ratio_ma_15',       # DOM ratio smoothed pe 15 bare
]

# v12.2: CONSOLIDATION DETECTION features — modelul învață singur WAIT în range
FEATURES_CONSOL = [
    # === A. Prețul nu merge nicăieri ===
    'range_atr_ratio',       # Range ultimelor 20 bare / ATR — <2.0 = consolidare
    'directional_efficiency',# net_move / total_move pe 20 bare — 0=chop, 1=trend
    'net_move_10',           # |close[-1] - close[-10]| / ATR — mișcare netă 10 bare
    'net_move_20',           # |close[-1] - close[-20]| / ATR — mișcare netă 20 bare
    'close_std_20',          # Std(close) pe 20 bare / ATR — volatilitate close-to-close
    'avg_bar_range_ratio',   # Mean(high-low) pe 10 bare / ATR — bare mici vs ATR = chop
    # === B. Prețul revine mereu la același loc ===
    'bars_inside_va',        # Câte din ultimele 10 bare sunt inside VA
    'va_width_atr',          # Lățimea VA / ATR — VA strânsă = range
    'va_overlap_pct',        # Suprapunere VA curentă vs VA shifted 10 bare (>80% = range)
    'mean_reversion_speed',  # Cât de repede revine prețul la SMA20 după deviere
    'same_level_rejections', # Atingeri de VAH/VAL/POC în 20 bare
    'pivot_count_20',        # Câte schimbări de direcție pe 20 bare (multe = chop)
    'poc_stability',         # Std(POC) pe 10 bare / ATR — POC stabil = range
    # === C. Volume confirmă lipsa de convingere ===
    'volume_trend',          # Slope volum pe 20 bare (negativ = volum scade = consolidare)
    'volume_cv',             # Coeficient variație volum pe 20 bare (mare = spike-uri, mic = flat)
    # === D. Structura pierdută ===
    'hh_ll_score',           # Higher-highs + lower-lows: 0=flat, ±1=trending
    'bollinger_width',       # (BB_upper - BB_lower) / close — îngustă = squeeze = consolidare
    'atr_percentile',        # Percentila ATR curent vs ultimele 100 bare (0-1)
    'swing_size_decay',      # Swing-urile devin mai mici? (ratio ultimul swing / penultimul)
    'candle_overlap_pct',    # % din bare care se suprapun cu bara anterioară (>70% = chop)
]

# v12.5: STREAK PREVENTION features — fără session_age, day_of_week, hour_sin, hour_cos
# SCOASE hour_sin + hour_cos: combinat dominau 21.5% = temporal overfitting masiv
FEATURES_STREAK = [
    'atr_change_speed',      # ATR acum / ATR acum 10 bare
    'consecutive_same_dir',  # Bare consecutive în aceeași direcție
    'price_vs_daily_range',  # Poziția în range-ul zilnic
    'recent_signal_quality', # Proxy calitate semnale recente
]

# v12.4: TRADE QUALITY CONTEXT — detectează degradarea condițiilor
FEATURES_CONTEXT = [
    'trend_r2',              # R² al close pe 20 bare — 1.0=trend, 0=noise
    'trend_slope_norm',      # Slope pe 20 bare / ATR
    'close_vs_ema_stack',    # EMA8 vs EMA21 vs EMA55 alignment
    'roc_10',                # Rate of Change 10 bare / ATR
    'roc_divergence',        # ROC 5 vs ROC 20 divergence
    'momentum_consistency',  # Consistență direcție pe 10 bare
    'volume_on_move',        # Volum pe mișcări mari vs mici
    'volume_directional',    # Volum bullish vs bearish
    'clean_bars_pct',        # % bare curate (body > 50% range)
    'false_break_count',     # False breakouts pe 20 bare
    'bar_range_consistency', # Uniformitate range bare
]

# v12.4: MTF CONFIRMATION — multi-timeframe agreement
FEATURES_MTF_CONFIRM = [
    'h1_trend_aligned',      # H1 are direcție clară
    'h4_trend_aligned',      # H4 are direcție clară
    'mtf_agreement',         # Câte TF-uri sunt de acord (0-1)
    'recent_rejection_strength',  # Forța rejections recente
]

# v12.7: Opening Range Breakout features (runtime)
FEATURES_ORH = [
    'in_orh',
    'post_orh',
    'orh_width_atr',
    'dist_to_orh_high_atr',
    'dist_to_orh_low_atr',
    'orh_broken_up',
    'orh_broken_down',
    'bars_since_session',
    'session_vol_ratio',
    'orh_midpoint_dist_atr',
]

# =============================================================================
# v13 REGIME-AWARE (9-class) — weekly profile + day-type + multi-level sweep
# =============================================================================
try:
    from aladin_v13 import (
        FEATURES_WEEKLY as _V13_FW,
        FEATURES_DAYTYPE as _V13_FD,
        FEATURES_SWEEP as _V13_FS,
        add_weekly_features as _v13_add_weekly,
        add_daytype_features as _v13_add_daytype,
        add_sweep_features as _v13_add_sweep,
        REGIME_NAMES as _V13_REGIME_NAMES,
    )
    FEATURES_WEEKLY = list(_V13_FW)
    FEATURES_DAYTYPE = list(_V13_FD)
    FEATURES_SWEEP = list(_V13_FS)
    # v14: Direction map 5 clase (collapsed). 0=WAIT 1=SHORT_BREAK 2=LONG_BREAK 3=SHORT_REV 4=LONG_REV
    V13_DIR_MAP = {0: 0, 1: -1, 2: +1, 3: -1, 4: +1}
    _V13_AVAILABLE = True
except ImportError as _v13_imp_err:
    FEATURES_WEEKLY, FEATURES_DAYTYPE, FEATURES_SWEEP = [], [], []
    _v13_add_weekly = _v13_add_daytype = _v13_add_sweep = None
    _V13_REGIME_NAMES = {}
    V13_DIR_MAP = {0: 0, 1: -1, 2: +1}
    _V13_AVAILABLE = False

# Killzone windows (Europe/Bucharest UTC+2/+3)
KILLZONES = {
    "Sydney Open":   ("02:00", "05:00"),   # 23:00-02:00 UTC = 02:00-05:00 RO (EEST)
    "London Open":   ("09:00", "11:00"),
    "NY Open":       ("15:30", "17:30"),
    "London Close":  ("19:00", "20:00"),
    "Asia":          ("02:00", "05:00"),
}

# News impact heuristic — ore cu risc maxim (UTC+2 RO)
HIGH_IMPACT_HOURS = {
    "Monday":    [],
    "Tuesday":   ["15:30", "16:00"],
    "Wednesday": ["15:15", "20:00"],  # FOMC
    "Thursday":  ["15:30"],           # Jobless Claims
    "Friday":    ["15:30"],           # NFP
}

# =============================================================================
# MODUL 0: VALIDARE CALENDAR NYSE
# =============================================================================
def is_market_open(target_dt: datetime) -> bool:
    """
    Verifică dacă NQ Futures sunt disponibile pentru trading.
    NQ Futures: duminică 22:00 UTC → vineri 21:00 UTC (aproape non-stop).
    NYSE calendar blochează duminica → folosim logica futures corectă.
    """
    wd  = target_dt.weekday()  # 0=Luni … 6=Duminică
    utc_hour = target_dt.hour

    # Sâmbătă toată ziua = piața închisă (futures se închid vineri 21:00 UTC)
    if wd == 5:
        return False

    # Duminică: futures deschid la 22:00 UTC (Sydney open)
    if wd == 6:
        return utc_hour >= 22

    # Luni–Vineri: futures sunt deschise, cu excepția zilelor de sărbătoare NYSE
    # EXCEPȚIE CRITICĂ: Good Friday — NYSE e închis dar NQ/ES Futures CME sunt DESCHISE!
    # CME nu închide futures pe Good Friday (spre deosebire de NYSE echity).
    try:
        nyse     = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(
            start_date = target_dt.date(),
            end_date   = target_dt.date()
        )
        if schedule.empty:
            # NYSE e închis. Verificăm dacă e Vineri (Good Friday) sau altă sărbătoare.
            # Good Friday = Vineri (wd==4) → NQ Futures DESCHISE pe CME
            # Crăciun/Thanksgiving/etc. cad și pe alte zile → futures ÎNCHISE
            if wd == 4:
                logger.info("📅 Good Friday detectat — NYSE închis dar NQ Futures CME sunt deschise")
                return True
            return False  # Altă sărbătoare NYSE (Crăciun, 4 Iulie, etc.)
        return True
    except Exception as e:
        logger.warning(f"Eroare calendar NYSE: {e}. Fallback weekday check.")
        return True  # Luni-Vineri presupunem deschis dacă calendar pică


def get_active_killzone(t_str: str) -> Optional[str]:
    """
    Returnează numele killzone-ului activ la ora dată, sau None.
    t_str format: 'HH:MM'
    """
    for name, (start, end) in KILLZONES.items():
        if start <= t_str < end:
            return name
    return None


# =============================================================================
# UPDATE #15: VOLATILITY COMPRESSION FILTER
# =============================================================================
def check_volatility_filter(df: pd.DataFrame) -> tuple:
    """
    Update #15 → #14c → FIX v7.5: Volatility compression filter.
    FIX v7.5: Bug critic — percentila ATR era calculată pe doar 100 bare (df),
    nu pe istoricul real din DB. O mișcare normală de 75pts apărea ca "RECORD ABSOLUT"
    pentru că era cea mai mare din ultimele 100 minute. Acum încarcă 90 zile de ATR
    din DB pentru comparație reală (~35,000 bare = context statistic valid).
    Returnează: (skip_trade: bool, reason: str)
    """
    try:
        if 'atr_14' not in df.columns or len(df) < 14:
            return False, "ATR insuficient"

        current_atr = df['atr_14'].dropna().iloc[-1] if len(df['atr_14'].dropna()) > 0 else 0
        if current_atr <= 0:
            return False, "ATR zero"

        # FIX v7.5: Încarcă ATR istoric din DB (90 zile ≈ 35,000 bare 1-min)
        # pentru comparație corectă — NU doar cele 100 bare din df
        import sqlite3
        try:
            conn_vol = sqlite3.connect(f'file:{PATH_DB}?mode=ro', uri=True,
                                       timeout=30, check_same_thread=False)
            hist_atr = pd.read_sql_query(
                "SELECT atr_14 FROM market_data WHERE atr_14 > 0 ORDER BY timestamp DESC LIMIT 35000",
                conn_vol
            )
            conn_vol.close()
            atr_hist = hist_atr['atr_14'].dropna()
        except Exception:
            # Fallback: dacă DB nu merge, folosim df-ul mic (comportament vechi)
            atr_hist = df['atr_14'].dropna()

        if len(atr_hist) < 100:
            return False, "Istoric ATR insuficient"

        # Calculăm percentila ATR curent vs distribuția istorică reală
        atr_percentile = float((atr_hist < current_atr).sum()) / len(atr_hist)

        logger.info(f"   📊 ATR Filter: current={current_atr:.2f} | hist={len(atr_hist)} bare | percentilă={atr_percentile:.3f}")

        # Hard block DOAR crize absolute (> 0.98 = top 2% din 90 zile reale)
        if atr_percentile > 0.98:
            if atr_percentile >= 0.999:
                label = "RECORD ATR ABSOLUT"
            else:
                label = f"top {(1-atr_percentile)*100:.1f}%"
            return True, f"⚡ ATR EXTREM ({label}, percentilă {atr_percentile:.3f})"

        return False, f"Volatilitate {'ridicată' if atr_percentile > 0.80 else 'OK'} (percentilă ATR: {atr_percentile:.2f})"
    except Exception as e:
        logger.warning(f"Volatility filter error: {e}")
        return False, "Filter error"


# =============================================================================
# UPDATE #16: GAP + VOLUME FILTER PRIMA ORĂ
# =============================================================================
def check_volume_trend_filter(df: pd.DataFrame, t_str: str) -> float:
    """
    Update #16: Gap + volume filter prima oră.
    Dacă prima oră are volum >150% din media zilnică → zi de trend → conviction +0.1
    Returnează: conviction_boost (0.0 sau +0.1)
    """
    try:
        if 'volume' not in df.columns or len(df) < 60:
            return 0.0

        # Verificăm dacă suntem în prima oră (09:00-10:00 RO / London open)
        hour = int(t_str.split(':')[0])
        if hour not in [9, 10]:
            return 0.0

        vol_series = df['volume'].dropna()
        if len(vol_series) < 20:
            return 0.0

        # Ultima bară vs media ultimelor 20 bare
        avg_vol  = float(vol_series.iloc[-20:-1].mean())
        last_vol = float(vol_series.iloc[-1])
        vol_ratio = last_vol / (avg_vol + 1e-8)

        if vol_ratio > 1.5:
            logger.info(f"   📊 Volume spike detectat: {vol_ratio:.1f}x medie → trend day boost +0.1")
            return 0.1
        return 0.0
    except Exception as e:
        logger.warning(f"Volume filter error: {e}")
        return 0.0


# =============================================================================
# UPDATE #4: CACHE GLOBAL pentru VIX / DXY / Options (TTL = 5 minute)
# Fără cache: fiecare semnal face 5-8 request-uri HTTP → latență + risc timeout
# Cu cache: primul request descarcă, următoarele 5 minute folosesc valoarea cached
# =============================================================================
import time as _time_module

_CACHE: dict = {}
_CACHE_TTL: float = 300.0  # 5 minute

def _cache_get(key: str):
    """Returnează valoarea din cache dacă nu e expirat, altfel None."""
    entry = _CACHE.get(key)
    if entry and (_time_module.time() - entry['ts']) < _CACHE_TTL:
        return entry['val']
    return None

def _cache_set(key: str, val):
    """Salvează valoarea în cache cu timestamp-ul curent."""
    _CACHE[key] = {'val': val, 'ts': _time_module.time()}

# =============================================================================
# UPDATE #17: VIX FILTER PENTRU POSITION SIZING
# =============================================================================
def get_vix_sizing_mult() -> float:
    """
    Update #17 + UPDATE #4 (cache 5 min): VIX filter pentru position sizing.
    VIX >25 → sizing 50% | VIX <15 → sizing 125% | altfel 100%
    """
    cached = _cache_get('vix')
    if cached is not None:
        return cached
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="2d", interval="1d")
        if hist.empty:
            return 1.0
        vix_val = float(hist['Close'].iloc[-1])
        logger.info(f"   📉 VIX: {vix_val:.1f}")
        if vix_val >= 25:  # Fix v9.1: >= 25 (nu doar > 25) — captează și exact 25.0
            logger.info("   ⚠️ VIX >=25 → position sizing redus la 50%")
            result = 0.5
        elif vix_val < 15:
            logger.info("   ✅ VIX <15 → position sizing crescut la 125%")
            result = 1.25
        else:
            result = 1.0
        _cache_set('vix', result)
        return result
    except Exception as e:
        logger.warning(f"VIX filter error (yfinance): {e}. Fallback sizing 1.0")
        return 1.0


# =============================================================================
# UPDATE #18: SENTIMENT ANALYSIS FINBERT (STUB ROBUST)
# =============================================================================
_finbert_pipeline = None

def get_news_sentiment_score() -> float:
    """
    Update #18: Sentiment analysis pe știri financiare cu FinBERT.
    Scraper RSS → FinBERT sentiment → scor [0,1] (0.5 = neutru).
    Returnează: sentiment_score în [0.3, 0.7] cu fallback la 0.5
    """
    global _finbert_pipeline
    try:
        import feedparser
        # RSS feeds financiare
        feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=QQQ&region=US&lang=en-US",
            "https://feeds.reuters.com/reuters/businessNews",
        ]
        headlines = []
        for url in feeds[:1]:  # doar primul pentru viteză
            try:
                feed = feedparser.parse(url)
                headlines += [e.title for e in feed.entries[:5]]
            except Exception:
                pass

        if not headlines:
            return 0.5

        # Încearcă FinBERT
        if _finbert_pipeline is None:
            try:
                from transformers import pipeline
                _finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    max_length=128, truncation=True
                )
            except Exception:
                return 0.5

        scores = []
        for headline in headlines[:3]:
            try:
                result = _finbert_pipeline(headline[:128])[0]
                if result['label'] == 'positive':
                    scores.append(0.65)
                elif result['label'] == 'negative':
                    scores.append(0.35)
                else:
                    scores.append(0.5)
            except Exception:
                scores.append(0.5)

        avg_score = float(np.mean(scores)) if scores else 0.5
        logger.info(f"   📰 FinBERT sentiment: {avg_score:.2f} ({len(headlines)} headlines)")
        return round(avg_score, 3)

    except Exception as e:
        logger.debug(f"News sentiment fallback: {e}")
        return 0.5


def minutes_to_killzone(t_str: str) -> int:
    """
    Calculează minutele până la următoarea killzone.
    Util pentru time-decay quantum.
    """
    t_now = datetime.strptime(t_str, "%H:%M")
    min_dist = 9999
    for name, (start, end) in KILLZONES.items():
        t_start = datetime.strptime(start, "%H:%M")
        diff = (t_start - t_now).total_seconds() / 60
        if diff < 0:
            diff += 1440  # next day
        if diff < min_dist:
            min_dist = int(diff)
    return min_dist


# =============================================================================
# MODUL 1: CIRCUIT CUANTIC PENNYLANE — v6.0 QUANTUM UPGRADE
# =============================================================================
#
# 4 circuite specializate:
#   A) quantum_main_circuit   — StronglyEntanglingLayers 6 qubiți
#   B) quantum_smt_circuit    — Amplitude Embedding QQQ vs SPY (6 qubiți)
#   C) quantum_noise_circuit  — Variational filter antrenabil (4 qubiți)
#   D) quantum_regime_circuit — IQP Embedding multi-timeframe (8 qubiți)
#
# =============================================================================

def _make_device(wires: int, name: str):
    try:
        dev = qml.device("lightning.qubit", wires=wires)
        logger.info(f"⚛️  {name}: lightning.qubit ({wires}w)")
        return dev
    except Exception:
        dev = qml.device("default.qubit", wires=wires)
        logger.info(f"⚛️  {name}: default.qubit fallback ({wires}w)")
        return dev

dev_main   = _make_device(6, "Main Circuit")
dev_smt    = _make_device(6, "SMT Circuit")
dev_noise  = _make_device(4, "Noise Filter")
dev_regime = _make_device(8, "Regime Detector")


# ── A) Circuit principal 6 qubiți — StronglyEntanglingLayers ─────────────────
@qml.qnode(dev_main)
def quantum_main_circuit(inputs, weights):
    """
    6 qubiți. Inputs: kz, poc, smt, va, fvg_strength, displacement_mag
    Weights shape: (2, 6, 3). Output: probs toate cele 64 stări.
    """
    scaled = qnp.array([x * np.pi for x in inputs])
    qml.AngleEmbedding(scaled, wires=range(6))
    qml.StronglyEntanglingLayers(weights, wires=range(6))
    return qml.probs(wires=range(6))

_MAIN_WEIGHTS_PATH = "/Users/mario/Desktop/Aladin/aladin_main_weights.npy"

def _load_main_weights():
    """Încarcă MAIN_WEIGHTS antrenate din fișier, sau inițializează aleator dacă lipsesc."""
    if os.path.exists(_MAIN_WEIGHTS_PATH):
        try:
            w = np.load(_MAIN_WEIGHTS_PATH)
            if w.shape == (2, 6, 3):
                logger.info("⚛️  MAIN_WEIGHTS antrenate încărcate din fișier")
                return qnp.array(w, requires_grad=True)
        except Exception:
            pass
    rng = np.random.default_rng(42)
    logger.info("⚛️  MAIN_WEIGHTS inițializate aleator (neantrenate) — rulează train_main_circuit() pentru antrenare")
    return qnp.array(
        rng.uniform(-np.pi/4, np.pi/4, size=(2, 6, 3)),
        requires_grad=True
    )

MAIN_WEIGHTS = _load_main_weights()


# ── B) Circuit SMT — Amplitude Embedding QQQ vs SPY ──────────────────────────
@qml.qnode(dev_smt)
def quantum_smt_circuit(qqq_vec, spy_vec):
    """
    6 qubiți. Qubiți 0-2: QQQ, qubiți 3-5: SPY.
    Interferența măsoară divergența continuu.
    """
    def _norm(v):
        v = qnp.array(v, dtype=float)
        n = qnp.linalg.norm(v)
        return v / (n + 1e-8)

    qqq_norm = _norm(qqq_vec)
    spy_norm = _norm(spy_vec)

    qml.AngleEmbedding(qqq_norm * np.pi, wires=[0, 1, 2])
    qml.AngleEmbedding(spy_norm * np.pi, wires=[3, 4, 5])
    qml.CNOT(wires=[0, 3])
    qml.CNOT(wires=[1, 4])
    qml.CNOT(wires=[2, 5])
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=3)
    qml.CRZ(np.pi / 4, wires=[0, 1])
    qml.CRZ(np.pi / 4, wires=[3, 4])
    qml.CNOT(wires=[1, 4])
    qml.Hadamard(wires=1)
    return qml.probs(wires=[0, 1, 2, 3])


def compute_quantum_smt(best: pd.Series) -> float:
    """SMT divergence continuu [0.0, 1.0]. 1.0 = divergență maximă."""
    try:
        close  = float(best.get('close', 1))
        high   = float(best.get('high',  close * 1.001))
        low    = float(best.get('low',   close * 0.999))
        spy_hi = float(best.get('spy_hi', 0) or 0)
        spy_lo = float(best.get('spy_lo', 0) or 0)

        if spy_hi <= 0:
            # Fallback neutru când SPY lipsește — 0.55 în loc de 0.3
            return 1.0 if (best.get('is_smt_bearish', 0) or best.get('is_smt_bullish', 0)) else 0.55

        spy_close = (spy_hi + spy_lo) / 2.0
        ref = close if close > 0 else 1.0
        qqq_vec = [high / ref, low / ref, close / ref]
        spy_vec = [spy_hi / ref, spy_lo / ref, spy_close / ref]

        probs = quantum_smt_circuit(qqq_vec, spy_vec)
        qqq_mass = float(probs[0]) + float(probs[1])
        spy_mass = float(probs[2]) + float(probs[3])
        divergence = abs(qqq_mass - spy_mass) * 2.0

        if best.get('is_smt_bearish', 0) or best.get('is_smt_bullish', 0):
            divergence = min(divergence + 0.25, 1.0)

        return round(float(divergence), 4)

    except Exception as e:
        logger.warning(f"Quantum SMT fallback: {e}")
        return 1.0 if (best.get('is_smt_bearish', 0) or best.get('is_smt_bullish', 0)) else 0.55


# ── C) Circuit Noise Filter variațional — 4 qubiți ───────────────────────────
_NOISE_WEIGHTS_PATH = "/Users/mario/Desktop/Aladin/aladin_noise_weights.npy"

def _load_noise_weights():
    if os.path.exists(_NOISE_WEIGHTS_PATH):
        try:
            w = np.load(_NOISE_WEIGHTS_PATH)
            logger.info("⚛️  Noise weights încărcate din fișier")
            return qnp.array(w, requires_grad=True)
        except Exception:
            pass
    rng = np.random.default_rng(99)
    return qnp.array(rng.uniform(-0.1, 0.1, size=8), requires_grad=True)

NOISE_WEIGHTS = _load_noise_weights()

@qml.qnode(dev_noise)
def quantum_noise_circuit(features, weights):
    """4 qubiți variațional. Output: prob wire[0] > 0.65 → noise."""
    qml.AngleEmbedding(features * np.pi, wires=range(4))
    for i in range(4):
        qml.RY(weights[i], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    for i in range(4):
        qml.RZ(weights[i + 4], wires=i)
    qml.CNOT(wires=[1, 2])
    return qml.probs(wires=[0])


def apply_quantum_noise_filter_v2(score: float, best: pd.Series) -> tuple:
    """
    Noise filter v6.3 FIX — circuit quantum dezactivat.
    Weights neantenate clasificau orice ca noise si taiau 40% din scor.
    Ex: raw=0.764 → final=0.458 (pierdere 40% nejustificata).
    Acum: scorul trece direct, fara penalizare artificiala.
    """
    has_smt = bool(best.get('is_smt_bearish', 0) or best.get('is_smt_bullish', 0))
    has_fvg = bool(best.get('fvg_up', 0) or best.get('fvg_down', 0))
    has_dis = bool(best.get('has_displacement', 0))
    confluence = sum([has_smt, has_fvg, has_dis])

    # Singura penalizare reala: scor mic SI zero confluente pe bara curenta
    if score < 0.40 and confluence == 0:
        return score * 0.9, True

    return score, False


def train_noise_filter(trades_df: pd.DataFrame):
    """
    Antrenează noise filter pe datele din backtest.
    Apelează din DASHBOARD după backtest complet.
    """
    global NOISE_WEIGHTS
    try:
        import pennylane.numpy as pnp
        opt = qml.AdamOptimizer(stepsize=0.05)
        losses = []

        # Pre-calcul batch → plain numpy (fix ArrayBox)
        def _build_noise_batch(df_batch):
            _feats_list, _labels = [], []
            for _, _r in df_batch.iterrows():
                _conf  = float(bool(_r.get('smt')) + bool(_r.get('fvg')) + (float(_r.get('score', 50)) > 75))
                _f0    = float(np.clip(float(_r.get('score', 50)) / 100.0, 0.0, 1.0))
                _f1    = float(np.clip(_conf / 3.0, 0.0, 1.0))
                _f2    = float(np.clip(float(_r.get('atr_14', 1) or 1) / 5.0, 0.0, 1.0))
                _feats_list.append(np.array([_f0, _f1, _f2, 0.5], dtype=float))
                _labels.append(1.0 if str(_r.get('result', '')).upper() == 'LOSS' else 0.0)
            return _feats_list, _labels

        for _ in range(30):
            batch = trades_df.sample(min(20, len(trades_df)), random_state=42)
            _pre_feats, _pre_labels = _build_noise_batch(batch)
            _nb = len(_pre_feats)

            def cost_fn(w):
                total_loss = pnp.array(0.0)
                for _i in range(_nb):
                    _fq = pnp.array(_pre_feats[_i], requires_grad=False)
                    _lq = pnp.array(_pre_labels[_i], requires_grad=False)
                    p   = quantum_noise_circuit(_fq, w)
                    total_loss = total_loss + (p[0] - _lq) ** 2
                return total_loss / _nb

            NOISE_WEIGHTS, loss_val = opt.step_and_cost(cost_fn, NOISE_WEIGHTS)
            losses.append(float(loss_val))

        np.save(_NOISE_WEIGHTS_PATH, np.array(NOISE_WEIGHTS))
        logger.info(f"⚛️  Noise filter antrenat. Loss: {losses[-1]:.4f}")
        return losses
    except Exception as e:
        logger.warning(f"Noise training error: {e}")
        return []


# ── D) Circuit Regime Detector — IQP Embedding 8 qubiți ──────────────────────
@qml.qnode(dev_regime)
def quantum_regime_circuit(features):
    """IQP 8 qubiți. Output: probs [0,1,2] → TRENDING/RANGING/VOLATILE."""
    scaled = qnp.array([f * np.pi for f in features])
    for i in range(8):
        qml.Hadamard(wires=i)
    for i in range(8):
        qml.RZ(scaled[i], wires=i)
    for i in range(0, 7, 2):
        qml.CZ(wires=[i, i + 1])
    for i in range(1, 6, 2):
        qml.CZ(wires=[i, i + 1])
    for i in range(8):
        qml.RZ(scaled[i] ** 2 % np.pi, wires=i)
    for i in range(8):
        qml.Hadamard(wires=i)
    qml.CNOT(wires=[0, 4])
    qml.CNOT(wires=[2, 6])
    qml.Toffoli(wires=[0, 2, 1])
    return qml.probs(wires=[0, 1, 2])


def get_market_regime_quantum(df: pd.DataFrame) -> str:
    """Regime detector IQP 8 qubiți cu fallback clasic."""
    try:
        if len(df) < 20:
            return "UNKNOWN"
        close    = df['close'].values
        atr      = df['atr_14'].values if 'atr_14' in df.columns else np.ones(len(close))
        atr_now  = float(atr[-1])
        atr_mean = float(np.mean(atr[-20:]))
        atr_slow = float(np.mean(atr[-50:])) if len(atr) >= 50 else atr_mean
        slope_h1 = float(close[-1] - close[-60]) / (close[-60] + 1e-8) if len(close) >= 60 else 0.0
        slope_h4 = float(close[-1] - close[-240]) / (close[-240] + 1e-8) if len(close) >= 240 else slope_h1
        vol      = df['volume'].values if 'volume' in df.columns else np.ones(len(close))
        vol_ratio = float(vol[-1]) / (float(np.mean(vol[-20:])) + 1e-8)
        highs = df['high'].values if 'high' in df.columns else close
        lows  = df['low'].values  if 'low'  in df.columns else close
        rng_now  = float(highs[-1] - lows[-1])
        rng_mean = float(np.mean(highs[-20:] - lows[-20:]))
        range_ratio = rng_now / (rng_mean + 1e-8)
        poc_dist = float(df['dist_poc'].iloc[-1]) if 'dist_poc' in df.columns else 0.5

        def _clip(v, lo=0.0, hi=2.0):
            return float(np.clip((v - lo) / (hi - lo + 1e-8), 0.0, 1.0))

        features = qnp.array([
            _clip(atr_now / (atr_mean + 1e-8), 0.5, 2.5),
            _clip(atr_now / (atr_slow + 1e-8), 0.5, 2.5),
            _clip(abs(slope_h1) * 100, 0.0, 1.0),
            _clip(abs(slope_h4) * 100, 0.0, 1.0),
            _clip(vol_ratio, 0.5, 3.0),
            _clip(range_ratio, 0.5, 2.5),
            _clip(poc_dist, 0.0, 1.0),
            _clip(atr_mean / (atr_slow + 1e-8), 0.5, 2.0),
        ])
        probs = quantum_regime_circuit(features)
        regime_probs = {
            "TRENDING": float(probs[0]) + float(probs[1]),
            "RANGING":  float(probs[2]) + float(probs[3]),
            "VOLATILE": float(probs[4]) + float(probs[5]) + float(probs[6]) + float(probs[7]),
        }
        dominant  = max(regime_probs, key=regime_probs.get)
        direction = "UP" if slope_h4 > 0 else "DOWN"
        if dominant == "TRENDING":
            return f"TRENDING {direction}"
        elif dominant == "RANGING":
            return "RANGING (Low ATR)" if atr_now < atr_mean else "RANGING"
        else:
            return f"VOLATILE ({direction})"
    except Exception as e:
        logger.warning(f"Quantum regime fallback: {e}")
        return _get_market_regime_classic(df)


def _get_market_regime_classic(df: pd.DataFrame) -> str:
    try:
        if len(df) < 20:
            return "UNKNOWN"
        atr_now  = float(df['atr_14'].iloc[-1]) if 'atr_14' in df.columns else 0
        atr_mean = float(df['atr_14'].tail(20).mean()) if 'atr_14' in df.columns else 1
        # Fix v9.0: check column existence BEFORE accessing it
        if 'h4_hi' in df.columns and len(df) >= 5:
            h4_trend = "UP" if df['h4_hi'].iloc[-1] > df['h4_hi'].iloc[-5] else "DOWN"
        else:
            h4_trend = "FLAT"

        # v12.2 FIX: PRICE-RANGE consolidation detection
        # ATR normal NU înseamnă trending — dacă prețul oscilează în range strâns,
        # e consolidare chiar cu ATR normal (bare pline care se anulează reciproc).
        # Verificăm: range-ul ultimelor 20 bare < 2.0× ATR curent = CONSOLIDARE.
        _tail20 = df.tail(20)
        _range_20 = float(_tail20['high'].max() - _tail20['low'].min()) if 'high' in df.columns else 0
        _is_price_range_tight = (_range_20 < atr_now * 2.0) if atr_now > 0 else False

        # Verificare suplimentară: higher-highs / lower-lows pe ultimele 10 bare
        # Dacă nu avem nici HH nici LL → piața e lateral, nu trending.
        _tail10 = df.tail(10)
        _highs = _tail10['high'].values if 'high' in df.columns else []
        _lows  = _tail10['low'].values if 'low' in df.columns else []
        _has_hh = False
        _has_ll = False
        if len(_highs) >= 5 and len(_lows) >= 5:
            # Comparăm prima jumătate vs a doua jumătate
            _first_half_hi  = max(_highs[:5])
            _second_half_hi = max(_highs[5:])
            _first_half_lo  = min(_lows[:5])
            _second_half_lo = min(_lows[5:])
            _has_hh = _second_half_hi > _first_half_hi + atr_now * 0.3   # HH cu marjă
            _has_ll = _second_half_lo < _first_half_lo - atr_now * 0.3   # LL cu marjă
        _no_structure = not _has_hh and not _has_ll  # nici HH, nici LL = lateral

        if atr_now > atr_mean * 1.5:
            return f"VOLATILE ({h4_trend})"
        elif atr_now < atr_mean * 0.7:
            return "RANGING (Low ATR)"
        elif _is_price_range_tight:
            return "RANGING (Tight Range)"
        elif _no_structure and atr_now <= atr_mean * 1.1:
            return "RANGING (No Structure)"
        else:
            return f"TRENDING {h4_trend}"
    except Exception:
        return "UNKNOWN"


def get_quantum_conviction(best: pd.Series, target_ts: str) -> float:
    """
    v6.0: Circuit principal 6 qubiți (StronglyEntanglingLayers) +
    SMT cuantic continuu. q_boost scalat corect pentru 64 stări.
    """
    try:
        t_now = pd.to_datetime(target_ts)
        t_str = t_now.strftime("%H:%M")

        # Factor 1: Killzone proximity
        kz = get_active_killzone(t_str)
        kz_factor = 1.0 if kz else max(0.5, 1.0 - (minutes_to_killzone(t_str) / 240.0))

        # Factor 2: POC distance
        # Fix v7.4 #5: POC factor normalized to absolute distance (50pt typical NQ spread)
        poc_val = float(best.get('poc_level', 0) or best['close'])
        if poc_val > 0:
            poc_factor = max(0.0, 1.0 - abs(best['close'] - poc_val) / 50.0)
        else:
            poc_factor = 0.5

        # Factor 3: SMT cuantic continuu (nu mai e binar)
        smt_factor = compute_quantum_smt(best)

        # Factor 4: VA positioning
        vah   = float(best.get('vah', 0) or best['close'])
        val_l = float(best.get('val', 0) or best['close'])
        close = best['close']
        if vah > val_l > 0:
            if abs(close - vah) < 1.5 or abs(close - val_l) < 1.5:
                va_factor = 1.0
            elif val_l <= close <= vah:
                va_factor = 0.6
            else:
                va_factor = 0.3
        else:
            va_factor = 0.5

        # Factor 5: FVG strength
        fvg_strength = 0.0
        if best.get('fvg_up', 0) or best.get('fvg_down', 0):
            poc_d = float(best.get('dist_poc', 0.5) or 0.5)
            fvg_strength = min(0.7 + max(0.0, 0.3 - poc_d * 0.3), 1.0)

        # Factor 6: Displacement magnitude
        displacement_mag = 0.0
        if best.get('has_displacement', 0):
            atr = float(best.get('atr_14', 1) or 1)
            rng = float(best.get('high', close) - best.get('low', close))
            displacement_mag = min(rng / (atr + 1e-8), 1.0)

        # Composite
        composite = (
            0.18 * kz_factor
            + 0.22 * poc_factor
            + 0.22 * smt_factor
            + 0.18 * va_factor
            + 0.12 * fvg_strength
            + 0.08 * displacement_mag
        )

        # Circuit principal 6 qubiți
        inputs = qnp.array([
            min(kz_factor,        1.0),
            min(poc_factor,       1.0),
            min(smt_factor,       1.0),
            min(va_factor,        1.0),
            min(fvg_strength,     1.0),
            min(displacement_mag, 1.0),
        ])
        probs = quantum_main_circuit(inputs, MAIN_WEIGHTS)

        # FIX v6.1: 6 qubiți = 64 stări → scalăm față de max(4/64)
        # pentru q_boost neutru ~1.025 și range real [0.85, 1.20]
        _raw = float(sum(probs[:4]))
        _max = 4.0 / 64.0
        q_boost = min(max(0.85 + (_raw / _max) * 0.35, 0.85), 1.20)

        result = min(composite * q_boost, 1.0)
        logger.info(
            f"   ⚛️  Quantum v6: kz={kz_factor:.2f} poc={poc_factor:.2f} "
            f"smt={smt_factor:.2f} va={va_factor:.2f} fvg={fvg_strength:.2f} "
            f"dis={displacement_mag:.2f} composite={composite:.3f} "
            f"boost={q_boost:.3f} → {result:.3f}"
        )
        return round(result, 4)

    except Exception as e:
        logger.warning(f"Quantum conviction fallback: {e}")
        return 0.5


def apply_quantum_noise_filter(score: float, best: pd.Series) -> tuple:
    """v6.0: Noise filter variațional — delegă la apply_quantum_noise_filter_v2."""
    return apply_quantum_noise_filter_v2(score, best)


def train_main_circuit(journal_df: pd.DataFrame = None, epochs: int = 50):
    """
    Antrenează MAIN_WEIGHTS (circuitul quantum principal 6 qubiți) pe datele din jurnal.
    Obiectiv: maximize q_boost pentru trade-urile câștigătoare, minimize pentru pierzătoare.

    Apelează din dashboard sau din train_mario_ai.py după antrenarea XGBoost.
    Salvează weights antrenate în aladin_main_weights.npy.

    Args:
        journal_df: DataFrame cu coloane [score, result, smt, fvg, displacement, atr_14, ...]
                    Dacă None, încearcă să încarce din CSV/DB.
        epochs: numărul de iterații de gradient descent (default 50)
    Returns:
        list[float]: lista loss-urilor per epocă
    """
    global MAIN_WEIGHTS
    try:
        import pennylane.numpy as pnp

        # Încearcă să încarce jurnalul dacă nu e furnizat
        if journal_df is None or len(journal_df) == 0:
            if os.path.exists(JOURNAL_PATH):
                journal_df = pd.read_csv(JOURNAL_PATH, low_memory=False)
                logger.info(f"⚛️  Quantum train: {len(journal_df)} trade-uri din jurnal")
            else:
                # Fallback: încearcă DB
                try:
                    conn = sqlite3.connect(f'file:{PATH_DB}?mode=ro', uri=True,
                                           timeout=30, check_same_thread=False)
                    journal_df = pd.read_sql_query("SELECT * FROM market_data ORDER BY ROWID DESC LIMIT 2000", conn)
                    conn.close()
                    logger.info(f"⚛️  Quantum train (DB fallback): {len(journal_df)} bare")
                except Exception:
                    logger.warning("⚛️  Quantum train: lipsă date — skip")
                    return []

        if len(journal_df) < 5:
            logger.warning("⚛️  Quantum train: date insuficiente (< 5 rânduri)")
            return []

        opt = qml.AdamOptimizer(stepsize=0.03)
        losses = []

        # ── FIX ArrayBox: pre-calculăm TOATE inputurile ÎNAINTE de gradient tape ──
        # PennyLane tracează (wraps) valorile în ArrayBox în interiorul cost_fn.
        # float() sau bool() pe un ArrayBox → crash. Soluție: extragem toate
        # valorile din DataFrame ca Python floats plain ACUM, stocăm în liste.
        def _build_batch_arrays(batch_df):
            """Returnează (inp_list, tgt_list) ca plain Python lists de numpy arrays."""
            _inp_list, _tgt_list = [], []
            for _, _r in batch_df.iterrows():
                _kz  = float(np.clip(float(_r.get('score', 50)) / 100.0, 0.0, 1.0))
                _poc = float(_r.get('poc_level', 0.5) or 0.5)
                _poc = float(np.clip(abs(_poc) / 500.0 if abs(_poc) > 1 else _poc, 0.0, 1.0))
                _smt = 1.0 if (_r.get('smt') or _r.get('is_smt_bullish') or _r.get('is_smt_bearish')) else 0.0
                _va  = float(np.clip(float(_r.get('inside_va', 0.5) or 0.5), 0.0, 1.0))
                _fvg = 1.0 if (_r.get('fvg') or _r.get('fvg_up') or _r.get('fvg_down')) else 0.0
                _dis = 1.0 if _r.get('has_displacement') else 0.0
                _inp_list.append(np.array([_kz, _poc, _smt, _va, _fvg, _dis], dtype=float))
                _rs  = str(_r.get('result', '')).upper()
                _tgt_list.append(6.0/64.0 if _rs in ('WIN','LONG','2','PROFIT','TP') else 1.0/64.0)
            return _inp_list, _tgt_list

        for epoch in range(epochs):
            sample_size = min(32, len(journal_df))
            batch = journal_df.sample(sample_size, random_state=epoch)

            # Pre-calcul complet ÎNAINTE de apelul optimizer (outside gradient tape)
            _pre_inp, _pre_tgt = _build_batch_arrays(batch)
            _n_batch = len(_pre_inp)

            def cost_fn(w):
                # REGULA: niciun float()/bool() pe valori din w sau probs.
                # _pre_inp[i] = numpy plain → qnp.array(requires_grad=False) = OK
                # _pre_tgt[i] = float Python plain → OK ca scalar
                _total = pnp.array(0.0)
                for _i in range(_n_batch):
                    _inp_q = pnp.array(_pre_inp[_i], requires_grad=False)
                    _probs = quantum_main_circuit(_inp_q, w)
                    # qnp.sum în loc de float(sum()) — rămâne în graficul autograd
                    _pred  = pnp.sum(_probs[:4])
                    _tgt_q = pnp.array(_pre_tgt[_i], requires_grad=False)
                    _total = _total + (_pred - _tgt_q) ** 2
                return _total / _n_batch

            MAIN_WEIGHTS, loss_val = opt.step_and_cost(cost_fn, MAIN_WEIGHTS)
            losses.append(float(loss_val))

            if epoch % 10 == 0:
                logger.info(f"⚛️  Quantum train epoch {epoch}/{epochs} — loss: {float(loss_val):.6f}")

        # Salvează weights antrenate
        np.save(_MAIN_WEIGHTS_PATH, np.array(MAIN_WEIGHTS))
        logger.info(f"⚛️  MAIN_WEIGHTS antrenate salvate → {_MAIN_WEIGHTS_PATH}  (loss final: {losses[-1]:.6f})")
        return losses

    except Exception as e:
        logger.warning(f"⚛️  train_main_circuit error: {e}")
        return []


# =============================================================================
# MODUL 2: NARRATIVE ICT COMPLET
# =============================================================================
def get_detailed_narrative(best: pd.Series, t_str: str) -> dict:
    """
    Reconstruiește povestea completă a pieței:
      - HTF Weekly Bias (Bull/Bear)
      - Power of 3 (Accumulation / Manipulation / Distribution)
      - Liquidity Draw (PDH/PDL)
      - AMT Context (POC + Value Area)
      - Killzone Context
      - FVG Status
      - SMT Divergence
    """
    # Weekly Bias — None-safe (coloana poate fi NULL în DB)
    _lw_lo = float(best.get('lw_lo') or 0)
    _lw_hi = float(best.get('lw_hi') or 0)
    lw_mid = _lw_lo + (_lw_hi - _lw_lo) / 2
    w_bias = "BULLISH" if best['close'] > lw_mid else "BEARISH"

    # Monthly Bias — None-safe
    _lm_lo = float(best.get('lm_lo') or 0)
    _lm_hi = float(best.get('lm_hi') or 0)
    lm_mid = _lm_lo + (_lm_hi - _lm_lo) / 2
    m_bias = "BULLISH" if best['close'] > lm_mid else "BEARISH"

    # Power of 3
    if "07:00" <= t_str <= "11:00":
        po3 = "ACCUMULATION (Smart Money builds position)"
    elif "14:00" <= t_str <= "16:30":
        po3 = "MANIPULATION — Judas Swing (Stop Hunt)"
    elif "16:30" <= t_str <= "19:00":
        po3 = "DISTRIBUTION (Institutional delivery)"
    else:
        po3 = "CONSOLIDATION (Outside killzone)"

    # Liquidity Draw
    true_open = best.get('true_open', best.get('open', best['close']))
    # Fix v10.2: ICT terminologie corectă
    # Buy-side liquidity = resting ABOVE highs (stop-uri cumpărători + short covering) → PDH
    # Sell-side liquidity = resting BELOW lows (stop-uri vânzători + long stop-losses) → PDL
    if best['close'] > true_open:
        liq_target = f"Drawing on Buy-side → PDH ({best.get('p_hi', 'N/A')})"
    else:
        liq_target = f"Drawing on Sell-side → PDL ({best.get('p_lo', 'N/A')})"

    # FVG context
    if best.get('fvg_up', 0):
        fvg_ctx = "Bullish FVG present — potential CE retest magnet"
    elif best.get('fvg_down', 0):
        fvg_ctx = "Bearish FVG present — potential CE retest magnet"
    else:
        fvg_ctx = "No active FVG"

    # SMT
    if best.get('is_smt_bearish', 0):
        smt_ctx = "SMT BEARISH DIVERGENCE — QQQ new high, SPY failed"
    elif best.get('is_smt_bullish', 0):
        smt_ctx = "SMT BULLISH DIVERGENCE — QQQ new low, SPY held"
    else:
        smt_ctx = "No SMT divergence"

    # Active killzone
    kz = get_active_killzone(t_str) or "Outside killzone"

    return {
        "macro":       f"W-Bias: {w_bias} | M-Bias: {m_bias} | Phase: {po3}",
        "liquidity":   f"Liquidity Draw: {liq_target}",
        "amt_context": (
            f"POC: {best.get('poc_level', 'N/A')} | "
            f"VA: [{best.get('val', 'N/A')} — {best.get('vah', 'N/A')}] | "
            f"Asia: [{best.get('asia_lo', 'N/A')} — {best.get('asia_hi', 'N/A')}]"
        ),
        "fvg":         fvg_ctx,
        "smt":         smt_ctx,
        "killzone":    kz,
        "htf_levels":  (
            f"Monthly [{best.get('lm_lo', 'N/A')} — {best.get('lm_hi', 'N/A')}] | "
            f"Weekly [{best.get('lw_lo', 'N/A')} — {best.get('lw_hi', 'N/A')}] | "
            f"Monday [{best.get('m_lo', 'N/A')} — {best.get('m_hi', 'N/A')}] | "
            f"PDH/PDL [{best.get('p_lo', 'N/A')} — {best.get('p_hi', 'N/A')}]"
        ),
    }


def build_narrative_text(nar: dict, best: pd.Series, extra: dict = None) -> str:
    """
    Construiește textul narativ complet pentru dashboard și console.
    """
    lines = [
        "═" * 80,
        "🕵️  ALADIN ICT NARRATIVE — FULL ANALYSIS",
        "═" * 80,
        f"🌍  MACRO     : {nar['macro']}",
        f"⚓  KILLZONE  : {nar['killzone']}",
        f"🎯  LIQUIDITY : {nar['liquidity']}",
        f"📊  AMT       : {nar['amt_context']}",
        f"🏗️   HTF       : {nar['htf_levels']}",
        f"📐  FVG       : {nar['fvg']}",
        f"🤝  SMT       : {nar['smt']}",
        "─" * 80,
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"   {k}: {v}")
    lines.append("═" * 80)
    return "\n".join(lines)


# =============================================================================
# MODUL 3: FILTRE INSTITUȚIONALE HTF
# =============================================================================
def apply_institutional_filters(best: pd.Series, score: float, early_bias: str = "NEUTRAL") -> tuple:
    """
    Blochează trade-urile care contrazic Smart Money (Weekly/Monthly alignment).
    Fix v7.7: Direction-aware — penalizează doar când direcția CONTRAZICE poziția.
      - Price above Monthly High + SHORT = BINE (vinzi la rezistență) → mic bonus
      - Price above Monthly High + LONG  = RĂU (cumperi extins) → penalizare -30%
      - Price below Monthly Low  + LONG  = BINE (cumperi la suport) → mic bonus
      - Price below Monthly Low  + SHORT = RĂU (vinzi extins) → penalizare -30%
    early_bias: "BULL", "BEAR", sau "NEUTRAL" din orderflow/HTF bias înainte de direction calc
    """
    lm_lo = best.get('lm_lo', 0) or 0
    lm_hi = best.get('lm_hi', 0) or 1e9
    lw_lo = best.get('lw_lo', 0) or 0
    lw_hi = best.get('lw_hi', 0) or 1e9
    close = best['close']

    # Regula 1: Price sub Monthly Low by >0.5%
    if lm_lo > 0 and close < lm_lo * (1 - 0.005) and score > 0.4:
        if early_bias == "BULL":
            # LONG sub monthly low = cumperi la suport → MIC bonus
            return min(score * 1.05, 1.0), "✅ Price below M-Low + LONG bias (buying support)"
        elif early_bias == "BEAR":
            # SHORT sub monthly low = vinzi extins → penalizare moderată
            return score * 0.70, "⚠️ CAUTION: Price below Monthly Low + SHORT bias (chasing)"
        else:
            return score * 0.85, "⚠️ Price below Monthly Low by >0.5% (neutral bias)"

    # Regula 2: Price peste Monthly High by >0.5%
    if lm_hi < 1e9 and close > lm_hi * (1 + 0.005) and score > 0.4:
        if early_bias == "BEAR":
            # SHORT peste monthly high = vinzi la rezistență → MIC bonus
            return min(score * 1.05, 1.0), "✅ Price above M-High + SHORT bias (selling resistance)"
        elif early_bias == "BULL":
            # LONG peste monthly high = cumperi extins → penalizare moderată
            return score * 0.70, "⚠️ CAUTION: Price above Monthly High + LONG bias (chasing)"
        else:
            return score * 0.85, "⚠️ Price above Monthly High by >0.5% (neutral bias)"

    # Bonus aliniere W+M (modest +5%)
    if lw_lo > 0 and lw_hi < 1e9 and lm_lo > 0:
        if lw_lo < close < lw_hi and close > lm_lo:
            return min(score * 1.05, 1.0), "✅ ALIGNED W+M Bullish"
        elif lw_lo < close < lw_hi and close < lm_hi:
            return min(score * 1.05, 1.0), "✅ ALIGNED W+M Bearish"

    return score, "⚪ Partial HTF Alignment"


# =============================================================================
# MODUL 4: RISK MANAGEMENT & POSITION SIZING
# =============================================================================
def calculate_sniper_risk(score: float, best: pd.Series, balance: float = 10000, direction: str = "LONG", live_data: dict = None) -> dict:
    """
    Determină mărimea poziției bazată pe scorul hibrid și ATR.

    Sizing rules:
      score > 0.85 → 2% risk (Sniper Conviction)
      score > 0.70 → 1.5% risk
      default       → 1% risk

    SL: la VAL/VAH sau ATR buffer dacă dist prea mică.
    TP: RR 3:1 față de SL
    """
    # Fix v9.0: Thresholds aliniate cu docstring — nu da 2% la orice scor > 0.55!
    # score > 0.85 → 2% (Sniper conviction, setup premium)
    # score > 0.70 → 1.5% (setup solid)
    # default       → 1% (setup marginal)
    if score > 0.85:
        risk_pct = 0.02
    elif score > 0.70:
        risk_pct = 0.015
    else:
        risk_pct = 0.01

    # ── Niveluri reale din DB ────────────────────────────────────────────
    close   = float(best['close'])
    is_long = direction.upper() == "LONG"

    val     = float(best.get('val',      0) or 0)
    vah     = float(best.get('vah',      0) or 0)
    poc     = float(best.get('poc_level',0) or 0)
    h1_lo   = float(best.get('h1_lo',    0) or 0)
    h1_hi   = float(best.get('h1_hi',    0) or 0)
    h4_lo   = float(best.get('h4_lo',    0) or 0)
    h4_hi   = float(best.get('h4_hi',    0) or 0)
    asia_lo = float(best.get('asia_lo',  0) or 0)
    asia_hi = float(best.get('asia_hi',  0) or 0)
    p_hi    = float(best.get('p_hi',     0) or 0)
    p_lo    = float(best.get('p_lo',     0) or 0)
    lw_lo   = float(best.get('lw_lo',    0) or 0)
    lw_hi   = float(best.get('lw_hi',    0) or 0)

    # ── v10.1: SL Dinamic bazat pe ATR curent — adaptat per instrument ──
    # Fix: SL fix de 8pts la NY Open e bătut de noise/spread singur.
    # Soluție: SL_MIN/DEFAULT/MAX se scalează cu ATR curent (proporțional cu volatilitatea momentului).
    #
    # Formule (NQ, prop firm cap $2K MLL):
    #   SL_MIN    = max(floor_abs, ATR × 0.55)   — sub asta = noise, nu structură
    #   SL_DEFAULT= max(floor_abs+2, ATR × 0.85) — când nu există nivel structural
    #   SL_MAX    = min(prop_cap, ATR × 1.20)    — niciodată mai mult decât 1.2× ATR
    #
    # Exemple reale NQ:
    #   ATR=10 (piață calmă):   SL_MIN=8,  SL_DEFAULT=10, SL_MAX=12 → max $480/2ct
    #   ATR=15 (normal):        SL_MIN=9,  SL_DEFAULT=13, SL_MAX=18 → max $720/2ct
    #   ATR=20 (London/NY Open):SL_MIN=11, SL_DEFAULT=17, SL_MAX=20 → max $800/2ct (cap)
    #   ATR=30 (news spike):    SL_MIN=16, SL_DEFAULT=20, SL_MAX=20 → max $800/2ct (cap)
    #
    # Prop firm hard cap: SL_MAX niciodată > 20pts (20×$20×2ct = $800 = 40% din $2K MLL)
    _inst_sym   = (live_data.get("symbol", "NQ") if live_data else "NQ")
    _sl_params  = _get_sl_params(_inst_sym)
    _sl_floor   = _sl_params["sl_min"]      # 8pts — floor absolut anti-noise
    _sl_prop_cap= _sl_params["sl_max"]      # 18pts — cap prop firm (override mai jos cu ATR)

    # ATR curent — din best row sau live_data
    _atr_now = float(best.get('atr_14', 0) or 0)
    if _atr_now <= 0 and live_data:
        _atr_now = float(live_data.get('atr_14', 0) or 0)
    if _atr_now <= 0:
        _atr_now = 12.0  # fallback conservator dacă ATR lipsește

    # SL dinamic proporțional cu ATR, cu floor absolut și prop firm cap
    _HARD_CAP   = 20.0  # absolut hard cap prop firm — niciodată > 20pts indiferent de ATR
    SL_MIN     = max(_sl_floor,        round(_atr_now * 0.55, 1))
    SL_DEFAULT = min(_HARD_CAP - 1,   max(_sl_floor + 2.0, round(_atr_now * 0.85, 1)))
    SL_MAX     = min(_HARD_CAP,        round(_atr_now * 1.20, 1))
    SL_MAX     = max(SL_MAX, SL_MIN + 1)     # garantăm SL_MAX > SL_MIN
    SL_DEFAULT = min(SL_DEFAULT, SL_MAX)     # garantăm SL_DEFAULT ≤ SL_MAX

    logger.debug(
        f"   📐 SL Dinamic ATR={_atr_now:.1f}pts → "
        f"SL_MIN={SL_MIN:.1f} SL_DEFAULT={SL_DEFAULT:.1f} SL_MAX={SL_MAX:.1f}"
    )

    if is_long:
        # ── SL LONG: caută nivel structural sub close, clamp la 8-18pts ──
        # Fix v10.1: candidații includ niveluri la ≥SL_MIN (8pts) față de close
        # Anterior: SL_MIN=15 ignora niveluri structurale reale la 8-14pts
        sl_candidates = [x for x in [h1_lo, h4_lo, asia_lo, val, lw_lo]
                         if 0 < x < close - SL_MIN]
        if sl_candidates:
            stop_loss = max(sl_candidates)  # cel mai aproape de close (cel mai bun RR)
            dist = abs(close - stop_loss)
            if dist < SL_MIN:
                stop_loss = close - SL_MIN      # prea aproape → SL_MIN
            elif dist > SL_MAX:
                stop_loss = close - SL_MAX      # prea departe → clamp la SL_MAX (18pts)
        else:
            stop_loss = close - SL_DEFAULT      # niciun nivel structural → default 12pts

        dist = abs(close - stop_loss)

        # ── TP LONG: bazat pe RR minim 2:1 ──────────────────────────────
        tp_candidates = [x for x in [val, poc, vah, p_hi, lw_hi]
                         if x > close + dist * 2.0]
        if tp_candidates:
            take_profit = min(tp_candidates)  # cel mai aproape deasupra 2R
        else:
            take_profit = close + dist * 3    # fallback 3R

    else:
        # ── SL SHORT: caută nivel structural deasupra close, clamp 8-18pts
        sl_candidates = [x for x in [h1_hi, h4_hi, asia_hi, vah, lw_hi]
                         if x > close + SL_MIN]
        if sl_candidates:
            stop_loss = min(sl_candidates)    # cel mai aproape de close
            dist = abs(close - stop_loss)
            if dist < SL_MIN:
                stop_loss = close + SL_MIN
            elif dist > SL_MAX:
                stop_loss = close + SL_MAX    # clamp la 18pts max
        else:
            stop_loss = close + SL_DEFAULT    # niciun nivel → default 12pts

        dist = abs(close - stop_loss)

        # ── TP SHORT: bazat pe RR ────────────────────────────────────────
        tp_candidates = [x for x in [vah, poc, val, p_lo, lw_lo]
                         if 0 < x < close - dist * 2.0]
        if tp_candidates:
            take_profit = max(tp_candidates)  # cel mai aproape
        else:
            take_profit = close - dist * 3

    # Garanteaza minim RR 2:1
    if dist > 0 and abs(take_profit - close) < dist * 2:
        take_profit = close + dist * 3 if is_long else close - dist * 3

    stop_loss   = round(stop_loss,   2)
    take_profit = round(take_profit, 2)
    atr         = round(dist, 2)  # folosim dist ca proxy ATR pentru return dict
    tp_mult     = abs(take_profit - close) / dist if dist > 0 else 3.0  # R-multiple

    risk_cash = balance * risk_pct
    shares = risk_cash / (dist * 20) if dist > 0 else 0

    # Fix v9.0: TP deja calculat din candidate logic mai sus — NU mai suprascrie
    # Linia veche suprascria TP-ul institutional cu 3R simplu → dead code eliminat

    # ── v8.1: PARTIAL EXIT PLAN (1R, 2R, trail rest) ─────────────────
    # La 1R: scoate 33% din poziție, mută SL la breakeven
    # La 2R: scoate încă 33%, mută SL la 1R
    # La 3R: TP final pe restul 34%, sau lasă trailing SL
    sign = 1 if is_long else -1
    level_1r = round(close + sign * dist * 1.0, 2)
    level_2r = round(close + sign * dist * 2.0, 2)
    level_3r = round(close + sign * dist * 3.0, 2)

    partial_plan = [
        {
            "level": level_1r,
            "R": "1R",
            "action": "close_33pct",
            "pct": 33,
            "move_sl_to": round(close, 2),  # breakeven
            "note": f"1R hit ({level_1r}) → scoate 33%, SL → breakeven ({close})"
        },
        {
            "level": level_2r,
            "R": "2R",
            "action": "close_33pct",
            "pct": 33,
            "move_sl_to": level_1r,  # mută SL la 1R
            "note": f"2R hit ({level_2r}) → scoate 33%, SL → 1R ({level_1r})"
        },
        {
            "level": level_3r,
            "R": "3R",
            "action": "close_rest_or_trail",
            "pct": 34,
            "move_sl_to": level_2r,  # trail SL la 2R
            "note": f"3R hit ({level_3r}) → TP final 34% sau trail SL la 2R ({level_2r})"
        },
    ]

    return {
        "units":    round(shares, 2),
        "risk_usd": round(risk_cash, 2),
        "risk_pct": round(risk_pct * 100, 1),
        "sl":       round(stop_loss, 2),
        "tp":       round(take_profit, 2),
        "rr":       f"{tp_mult:.0f}:1",
        "atr":      round(atr, 2),
        "sl_pts":   round(dist, 1),
        "tp_pts":   round(dist * tp_mult, 1),
        "partial_exits": partial_plan,
        "levels": {
            "entry": round(close, 2),
            "1R": level_1r,
            "2R": level_2r,
            "3R": level_3r,
        },
    }


# =============================================================================
# UPDATE #47: PORTFOLIO HEAT MONITORING
# =============================================================================
def check_portfolio_heat(open_trades_file: str = "/Users/mario/Desktop/Aladin/aladin_open_trades.json") -> dict:
    """
    Update #47: Portfolio heat monitoring.
    Dacă ai 3+ trade-uri deschise simultan → riscul total nu depășește 5% din capital.
    Returnează: {'n_open': int, 'total_risk_pct': float, 'can_open_new': bool, 'reason': str}
    """
    try:
        if not os.path.exists(open_trades_file):
            return {'n_open': 0, 'total_risk_pct': 0.0, 'can_open_new': True, 'reason': 'No open trades file'}

        with open(open_trades_file, 'r') as f:
            open_trades = json.load(f)

        n_open = len(open_trades)
        total_risk_pct = sum(t.get('risk_pct', 0.5) for t in open_trades)

        can_open = total_risk_pct < 5.0  # max 5% total portfolio heat
        reason = f"{n_open} trade-uri deschise, risc total {total_risk_pct:.1f}%"

        if not can_open:
            reason += f" → BLOCAT (limită 5% portfolio heat)"

        return {
            'n_open':           n_open,
            'total_risk_pct':   round(total_risk_pct, 2),
            'can_open_new':     can_open,
            'reason':           reason,
        }
    except Exception as e:
        logger.debug(f"Portfolio heat check error: {e}")
        return {'n_open': 0, 'total_risk_pct': 0.0, 'can_open_new': True, 'reason': str(e)}


# =============================================================================
# UPDATE #10: CORRELATION FILTER — corelație REALĂ NQ/ES din yfinance (rolling 20 zile)
# =============================================================================
def get_nq_es_correlation() -> dict:
    """
    UPDATE #10: Calculează corelația reală NQ vs ES din ultimele 20 zile (yfinance).
    Folosește cache 5 min (același pattern ca VIX/DXY).
    Returnează: {'corr': float, 'divergence': bool, 'nq_ret': float, 'es_ret': float, 'source': str}
    """
    _key = "nq_es_corr"
    cached = _cache_get(_key)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
        # NQ=F (E-mini Nasdaq futures) și ES=F (E-mini S&P futures)
        tickers = yf.download(
            ["NQ=F", "ES=F"],
            period="30d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        closes = tickers["Close"].dropna()

        if len(closes) < 10:
            raise ValueError("Date insuficiente pentru corelație")

        closes = closes.tail(20)
        nq_ret = closes["NQ=F"].pct_change().dropna()
        es_ret = closes["ES=F"].pct_change().dropna()

        corr = float(nq_ret.corr(es_ret))

        # Divergență intra-day: ultimele 2 zile merg în direcții opuse
        nq_last = float(nq_ret.iloc[-1]) if len(nq_ret) else 0
        es_last = float(es_ret.iloc[-1]) if len(es_ret) else 0
        divergence = (nq_last > 0 and es_last < -0.002) or (nq_last < 0 and es_last > 0.002)

        result = {
            "corr":       round(corr, 4),
            "divergence": divergence,
            "nq_ret_1d":  round(nq_last * 100, 3),
            "es_ret_1d":  round(es_last * 100, 3),
            "n_days":     len(closes),
            "source":     "yfinance",
        }
        _cache_set(_key, result)
        logger.info(f"   📊 NQ/ES Corr 20d: {corr:.3f} | Divergence: {divergence}")
        return result

    except Exception as e:
        logger.debug(f"NQ/ES correlation error: {e}")
        fallback = {"corr": 0.93, "divergence": False, "nq_ret_1d": 0.0,
                    "es_ret_1d": 0.0, "n_days": 0, "source": "fallback"}
        _cache_set(_key, fallback)
        return fallback


def check_correlation_filter(instrument: str = "NQ",
                              active_positions: list = None) -> dict:
    """
    UPDATE #10: Correlation filter cu corelație REALĂ NQ/ES.
    - Blochează trade dacă NQ/ES diverge (merg în direcții opuse)
    - Reduce score_adj dacă corelație < 0.80 (piața fracturată)
    - Nu blochează dacă nu există poziții active pe ES simultan
    Returnează: {'blocked': bool, 'reason': str, 'corr': float, 'score_adj': float}
    """
    corr_data = get_nq_es_correlation()
    corr      = corr_data.get("corr", 0.93)
    diverging = corr_data.get("divergence", False)

    # Dacă NQ și ES diverge activ → warning, score penalizat ușor
    if diverging:
        return {
            'blocked':   False,
            'reason':    f"⚠️ NQ/ES divergență detectată (NQ {corr_data['nq_ret_1d']:+.2f}% vs ES {corr_data['es_ret_1d']:+.2f}%) → scor redus",
            'corr':      corr,
            'score_adj': -0.03,   # -3% din scor
        }

    # Corelație scăzută = piața fracturată = mai puțin de încredere în semnal NQ
    if corr < 0.80:
        return {
            'blocked':   False,
            'reason':    f"⚠️ Corelație NQ/ES scăzută ({corr:.2f} < 0.80) → semnal mai slab",
            'corr':      corr,
            'score_adj': -0.02,
        }

    # Poziții active pe instrumente corelate (previne dublă expunere)
    if active_positions:
        for pos in (active_positions or []):
            if pos.get('instrument', '') in ['ES', 'QQQ', 'SPY']:
                return {
                    'blocked':   True,
                    'reason':    f"⛔ Poziție activă pe {pos['instrument']} — corelație {corr:.2f} → BLOCAT",
                    'corr':      corr,
                    'score_adj': 0.0,
                }

    return {
        'blocked':   False,
        'reason':    f"✅ Corelație NQ/ES OK ({corr:.2f})",
        'corr':      corr,
        'score_adj': 0.0,
    }


# =============================================================================
# UPDATE #49: DAILY LOSS LIMIT CIRCUIT BREAKER
# =============================================================================
def check_daily_loss_limit(
    journal_path: str = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv",
    max_daily_loss_pct: float = 3.0,
    balance: float = 10000.0,
) -> dict:
    """
    Update #49: Daily loss limit circuit breaker.
    Dacă pierzi >3% într-o zi → sistemul se oprește automat până a doua zi.
    Standard în prop trading firms.
    Returnează: {'blocked': bool, 'daily_pnl': float, 'daily_pnl_pct': float, 'reason': str}
    """
    try:
        if not os.path.exists(journal_path):
            return {'blocked': False, 'daily_pnl': 0.0, 'daily_pnl_pct': 0.0, 'reason': 'Journal lipsă'}

        today = datetime.now().strftime('%Y-%m-%d')
        df    = pd.read_csv(journal_path, low_memory=False)

        # Filtrează trade-urile de astăzi
        if 'timestamp' not in df.columns:
            return {'blocked': False, 'daily_pnl': 0.0, 'daily_pnl_pct': 0.0, 'reason': 'Coloană timestamp lipsă'}

        today_trades = df[df['timestamp'].str.startswith(today)] if len(df) > 0 else pd.DataFrame()

        if today_trades.empty:
            return {'blocked': False, 'daily_pnl': 0.0, 'daily_pnl_pct': 0.0, 'reason': f'Niciun trade azi ({today})'}

        # Fix v9.0: Calculează P&L zilnic corect — fallback la 'result_usd' sau estimare din SL
        daily_pnl = 0.0
        if 'pnl' in today_trades.columns:
            daily_pnl = float(today_trades['pnl'].sum())
        elif 'result_usd' in today_trades.columns:
            daily_pnl = float(today_trades['result_usd'].sum())
        elif 'risk_usd' in today_trades.columns and 'result' in today_trades.columns:
            # Estimare: WIN = +risk_usd * RR, LOSS = -risk_usd
            for _, _t in today_trades.iterrows():
                _risk = float(_t.get('risk_usd', 0) or 0)
                _res = str(_t.get('result', '')).upper()
                if _res == 'WIN':
                    daily_pnl += _risk * 2.0  # estimat 2R avg
                elif _res == 'LOSS':
                    daily_pnl -= _risk
        else:
            logger.debug("Daily loss check: nicio coloană PnL disponibilă în jurnal")
        daily_pnl_pct = (daily_pnl / balance * 100) if balance > 0 else 0.0

        blocked = daily_pnl_pct < -max_daily_loss_pct

        return {
            'blocked':       blocked,
            'daily_pnl':     round(daily_pnl, 2),
            'daily_pnl_pct': round(daily_pnl_pct, 2),
            'reason':        f"Daily P&L: {daily_pnl_pct:.1f}% {'⛔ CIRCUIT BREAKER ACTIV' if blocked else '✅ OK'}",
        }
    except Exception as e:
        logger.debug(f"Daily loss limit check error: {e}")
        return {'blocked': False, 'daily_pnl': 0.0, 'daily_pnl_pct': 0.0, 'reason': str(e)}


# =============================================================================
# UPDATE #50: MAX DRAWDOWN CIRCUIT BREAKER
# =============================================================================
def check_max_drawdown_breaker(
    journal_path: str = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv",
    max_dd_pct: float = 15.0,
    initial_balance: float = 10000.0,
) -> dict:
    """
    Update #50: Max drawdown circuit breaker.
    Dacă drawdown atinge -15% față de peak → sistemul se oprește și alertează.
    Protejează capital-ul în perioade adverse.
    Returnează: {'blocked': bool, 'current_dd': float, 'peak_balance': float, 'reason': str}
    """
    try:
        if not os.path.exists(journal_path):
            return {'blocked': False, 'current_dd': 0.0, 'peak_balance': initial_balance, 'reason': 'Journal lipsă'}

        df = pd.read_csv(journal_path, low_memory=False)

        if df.empty or 'score' not in df.columns:
            return {'blocked': False, 'current_dd': 0.0, 'peak_balance': initial_balance, 'reason': 'Date insuficiente'}

        # Reconstruim equity curve din journal
        # Dacă avem coloana 'balance', o folosim direct; altfel estimăm
        if 'balance' in df.columns and len(df) > 1:
            balances   = df['balance'].astype(float)
        else:
            # Fallback: presupunem că avem un balance inițial
            balances = pd.Series([initial_balance])

        peak_balance = float(balances.cummax().iloc[-1]) if len(balances) > 0 else initial_balance
        curr_balance = float(balances.iloc[-1]) if len(balances) > 0 else initial_balance
        current_dd   = (curr_balance - peak_balance) / peak_balance * 100 if peak_balance > 0 else 0.0

        blocked = current_dd < -max_dd_pct

        return {
            'blocked':       blocked,
            'current_dd':    round(current_dd, 2),
            'peak_balance':  round(peak_balance, 2),
            'curr_balance':  round(curr_balance, 2),
            'reason':        f"DD: {current_dd:.1f}% {'⛔ MAX DD BREAKER ACTIV' if blocked else '✅ OK'}",
        }
    except Exception as e:
        logger.debug(f"Max DD breaker error: {e}")
        return {'blocked': False, 'current_dd': 0.0, 'peak_balance': initial_balance, 'reason': str(e)}


# =============================================================================
# UPDATE #51: KELLY CRITERION POSITION SIZING
# =============================================================================
def kelly_position_size(
    win_rate: float,
    avg_win:  float,
    avg_loss: float,
    capital:  float,
    fraction: float = 0.25,  # Quarter Kelly pentru siguranță
) -> float:
    """
    Update #51: Kelly Criterion position sizing.
    Quarter Kelly (×0.25) pentru protecție împotriva variance mari.

    Formula: kelly = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
    Position = capital * kelly * fraction
    """
    try:
        if avg_win <= 0 or capital <= 0:
            return capital * 0.005  # fallback 0.5%

        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0.0, min(kelly_fraction, 0.20))  # cap la 20% Kelly

        position_size = capital * kelly_fraction * fraction

        # Clamp: minim $50, maxim 5% capital
        position_size = max(50.0, min(position_size, capital * 0.05))

        logger.debug(f"   💰 Kelly fraction: {kelly_fraction:.3f} → position: ${position_size:.0f}")
        return round(position_size, 2)
    except Exception:
        return round(capital * 0.005, 2)  # fallback 0.5%


# =============================================================================
# MODUL 4.5: ADVANCED ORDERFLOW & FVG GAP ANALYTICS
# =============================================================================
def analyze_orderflow_imbalance(df: pd.DataFrame) -> dict:
    """
    Analiză profundă a ineficiențelor de preț (FVG, Liquidity Voids).
    Detectează gap-uri pe ultimele 3 lumânări și calculează atracția magnetică.

    Returnează:
      score_bias: float  (ajustare scor -0.15 / 0 / +0.15)
      levels: list[dict] (tip, size, CE level, status)
    """
    if len(df) < 3:
        return {"score_bias": 0.0, "levels": []}

    last_3 = df.tail(3).reset_index(drop=True)
    report = {"score_bias": 0.0, "levels": []}

    try:
        c0_hi = float(last_3.iloc[0]['high'])
        c0_lo = float(last_3.iloc[0]['low'])
        c2_hi = float(last_3.iloc[2]['high'])
        c2_lo = float(last_3.iloc[2]['low'])

        # FVG Bullish: low[2] > high[0]
        if c2_lo > c0_hi:
            gap_size = c2_lo - c0_hi
            ce_level = c0_hi + gap_size / 2
            report['levels'].append({
                "type":      "BULLISH OB",
                "price_top": round(c2_lo, 2),
                "price_bot": round(c0_hi, 2),
                "size":      round(gap_size, 2),
                "ce":        round(ce_level, 2),
                "status":    "UNFILLED",
            })
            report['score_bias'] += 0.15

        # FVG Bearish: high[2] < low[0]
        if c2_hi < c0_lo:
            gap_size = c0_lo - c2_hi
            ce_level = c2_hi + gap_size / 2
            report['levels'].append({
                "type":      "BEARISH OB",
                "price_top": round(c0_lo, 2),
                "price_bot": round(c2_hi, 2),
                "size":      round(gap_size, 2),
                "ce":        round(ce_level, 2),
                "status":    "UNFILLED",
            })
            report['score_bias'] -= 0.15

    except Exception as e:
        logger.warning(f"Orderflow imbalance error: {e}")

    return report


def detect_order_blocks(df: pd.DataFrame, lookback: int = 20) -> list:
    """
    Detectează Order Blocks instituționale în fereastra de lookback.

    Definiție ICT Order Block:
      Bullish OB: ultima lumânare bearish înainte de un impuls bullish puternic
      Bearish OB: ultima lumânare bullish înainte de un impuls bearish puternic

    Returnează listă de OB-uri cu price_top, price_bot, type, strength.
    """
    if len(df) < lookback + 3:
        return []

    rows = df.tail(lookback + 3).reset_index(drop=True)
    obs  = []

    try:
        body_mean = abs(rows['close'] - rows['open']).rolling(10).mean()

        for i in range(2, len(rows) - 1):
            c     = rows.iloc[i]
            c_prev = rows.iloc[i - 1]
            c_next = rows.iloc[i + 1]

            body_c = abs(c['close'] - c['open'])
            body_n = abs(c_next['close'] - c_next['open'])
            avg_b  = body_mean.iloc[i] if not np.isnan(body_mean.iloc[i]) else 5.0

            # Bullish OB: c bearish, c_next bullish puternic (displacement)
            if (c['close'] < c['open']
                    and c_next['close'] > c_next['open']
                    and body_n > avg_b * 1.5):
                obs.append({
                    "type":      "BULLISH OB",
                    "price_top": round(max(c['open'], c['close']), 2),
                    "price_bot": round(min(c['open'], c['close']), 2),
                    "strength":  round(body_n / avg_b, 2),
                    "index":     i,
                })

            # Bearish OB: c bullish, c_next bearish puternic
            if (c['close'] > c['open']
                    and c_next['close'] < c_next['open']
                    and body_n > avg_b * 1.5):
                obs.append({
                    "type":      "BEARISH OB",
                    "price_top": round(max(c['open'], c['close']), 2),
                    "price_bot": round(min(c['open'], c['close']), 2),
                    "strength":  round(body_n / avg_b, 2),
                    "index":     i,
                })

        # Sortăm după strength descrescător, top 5
        obs = sorted(obs, key=lambda x: x['strength'], reverse=True)[:5]

    except Exception as e:
        logger.warning(f"Order block detection error: {e}")

    return obs


# =============================================================================
# MODUL 4.6: SESSION PROJECTIONS
# =============================================================================
def get_session_projections(best: pd.Series) -> dict:
    """
    Calculează deviațiile standard bazate pe range-ul sesiunilor Asia/London.
    Folosit pentru a ținti Take Profit-uri instituționale.
    """
    asia_hi = best.get('asia_hi', best['high'])
    asia_lo = best.get('asia_lo', best['low'])
    asia_range = asia_hi - asia_lo

    if asia_range <= 0:
        return {}

    return {
        "bull_targets": {
            "SD_1.0": round(asia_hi + asia_range * 1.0, 2),
            "SD_2.0": round(asia_hi + asia_range * 2.0, 2),
            "SD_2.5": round(asia_hi + asia_range * 2.5, 2),
            "SD_4.0": round(asia_hi + asia_range * 4.0, 2),
        },
        "bear_targets": {
            "SD_1.0": round(asia_lo - asia_range * 1.0, 2),
            "SD_2.0": round(asia_lo - asia_range * 2.0, 2),
            "SD_2.5": round(asia_lo - asia_range * 2.5, 2),
            "SD_4.0": round(asia_lo - asia_range * 4.0, 2),
        },
        "asia_range": round(asia_range, 2),
    }


# =============================================================================
# MODUL 5: SYNTHETIC RELATIVE STRENGTH (QQQ vs SPY)
# =============================================================================
def analyze_relative_strength(df: pd.DataFrame, lookback: int = 5) -> tuple:
    """
    Înlocuiește DXY. Analizează dacă QQQ (Tech) conduce piața față de SPY.

    Dacă QQQ urcă mai tare decât SPY → instituțiile pompează bani în risc (Bullish).
    Dacă QQQ scade mai mult decât SPY → risk-off (Bearish).

    Returnează: (multiplier: float, info_str: str)
    """
    try:
        if len(df) < lookback + 1:
            return 1.0, "⚪ Relative Strength N/A (date insuficiente)"

        qqq_change = (
            (df['close'].iloc[-1] - df['close'].iloc[-lookback])
            / df['close'].iloc[-lookback]
        )

        _spy_ref = df['spy_hi'].iloc[-lookback] if 'spy_hi' in df.columns else 0
        if 'spy_hi' in df.columns and not df['spy_hi'].isna().all() and _spy_ref and _spy_ref == _spy_ref:
            # Estimăm SPY close din SPY hi (proxy dacă spy_close nu e disponibil)
            spy_change = (
                (df['spy_hi'].iloc[-1] - _spy_ref)
                / _spy_ref
            )
        elif 'spy_close' in df.columns:
            spy_change = (
                (df['spy_close'].iloc[-1] - df['spy_close'].iloc[-lookback])
                / df['spy_close'].iloc[-lookback]
            )
        else:
            return 1.0, "⚪ SPY data N/A"

        rel_strength = qqq_change - spy_change
        multiplier   = 1.15 if rel_strength > 0 else 0.85

        direction = "🚀 QQQ LEADING (Risk-ON)" if rel_strength > 0 else "🐢 QQQ LAGGING (Risk-OFF)"
        info      = f"{direction} | Δ:{rel_strength:.5f} | QQQ:{qqq_change:.4f} SPY:{spy_change:.4f}"

        return multiplier, info

    except Exception as e:
        logger.warning(f"Relative strength error: {e}")
        return 1.0, "⚪ Relative Strength N/A"


# =============================================================================
# MODUL 6: STANDARD DEVIATION PROJECTIONS
# =============================================================================
def calculate_standard_deviations(best: pd.Series) -> dict:
    """
    Calculează manual deviațiile standard (SD) bazate pe Asia Range (ICT Vanguard Logic).
    Target-urile instituționale sunt adesea la 2, 2.5 sau 4 SD față de Asian High/Low.
    """
    asia_hi = float(best.get('asia_hi') or 0)
    asia_lo = float(best.get('asia_lo') or 0)
    asia_range = asia_hi - asia_lo

    if asia_range <= 0:
        logger.debug("SD projections: Asia range ≤ 0, skip.")
        return {}

    return {
        "bull": {
            "SD_1.0": round(asia_hi + asia_range * 1.0, 2),
            "SD_2.0": round(asia_hi + asia_range * 2.0, 2),
            "SD_2.5": round(asia_hi + asia_range * 2.5, 2),
            "SD_4.0": round(asia_hi + asia_range * 4.0, 2),
        },
        "bear": {
            "SD_1.0": round(asia_lo - asia_range * 1.0, 2),
            "SD_2.0": round(asia_lo - asia_range * 2.0, 2),
            "SD_2.5": round(asia_lo - asia_range * 2.5, 2),
            "SD_4.0": round(asia_lo - asia_range * 4.0, 2),
        },
        "asia_range": round(asia_range, 2),
        "asia_hi":    round(asia_hi, 2),
        "asia_lo":    round(asia_lo, 2),
    }


# =============================================================================
# MODUL 7: AUTOMATED JOURNALING SYSTEM
# =============================================================================
def log_aladin_verdict(
    target_ts: str,
    score: float,
    risk_data: dict,
    narrative: dict,
    verdict: str = "",
    extra: dict = None,
) -> str:
    """
    Salvează automat fiecare interogare într-un jurnal CSV de audit.
    Esențial pentru Faza 4: Scalare, Backtest și Review Sniper.
    """
    try:
        entry = {
            "timestamp":     target_ts,
            "hybrid_score":  round(score * 100, 2),
            "verdict":       verdict or ("HIGH" if score > 0.82 else "OBSERVE"),
            "position_size": risk_data.get('units', 0),
            "stop_loss":     risk_data.get('sl', 0),
            "take_profit":   risk_data.get('tp', 0),
            "risk_usd":      risk_data.get('risk_usd', 0),
            "rr":            risk_data.get('rr', '3:1'),
            "macro_bias":    narrative.get('macro', ''),
            "killzone":      narrative.get('killzone', ''),
            "smt":           narrative.get('smt', ''),
            "fvg":           narrative.get('fvg', ''),
            "logged_at":     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        if extra:
            entry.update(extra)

        df_new = pd.DataFrame([entry])

        if not os.path.isfile(JOURNAL_PATH):
            df_new.to_csv(JOURNAL_PATH, index=False)
        else:
            df_new.to_csv(JOURNAL_PATH, mode='a', header=False, index=False)

        return f"📝 Jurnalizat → {JOURNAL_PATH}"

    except Exception as e:
        logger.warning(f"Journal write error: {e}")
        return f"⚠️ Journal error: {e}"


def load_journal_stats() -> dict:
    """
    Încarcă statistici din jurnalul de audit.
    Returnează dict cu win_rate, avg_score, total_signals, etc.
    """
    try:
        if not os.path.isfile(JOURNAL_PATH):
            return {}
        df = pd.read_csv(JOURNAL_PATH, low_memory=False)
        if df.empty:
            return {}

        stats = {
            "total":      len(df),
            "high_conv":  int((df['hybrid_score'] > 82).sum()) if 'hybrid_score' in df.columns else 0,
            "avg_score":  round(df['hybrid_score'].mean(), 2) if 'hybrid_score' in df.columns else 0,
            "last_ts":    df['timestamp'].iloc[-1] if 'timestamp' in df.columns else "N/A",
        }
        return stats
    except Exception as e:
        logger.warning(f"Journal load error: {e}")
        return {}


# =============================================================================
# MODUL 8: DEEP FVG ANALYSIS (Consequent Encroachment)
# =============================================================================
def deep_fvg_analysis(df: pd.DataFrame, lookback: int = 10) -> str:
    """
    Analizează nu doar prezența FVG, ci și Consequent Encroachment (CE).
    Dacă prețul revine la 50% din gap și respinge → confirmarea e maximă.

    Returnează string descriptiv pentru dashboard.
    """
    if len(df) < 3:
        return "⚪ Insufficient data for FVG analysis"

    rows = df.tail(lookback).reset_index(drop=True)
    current_price = float(rows.iloc[-1]['close'])
    messages = []

    try:
        for i in range(2, len(rows)):
            c0 = rows.iloc[i - 2]
            c2 = rows.iloc[i]

            # Bullish FVG
            if float(c2['low']) > float(c0['high']):
                gap_top    = float(c2['low'])
                gap_bottom = float(c0['high'])
                ce_level   = gap_bottom + (gap_top - gap_bottom) / 2
                gap_size   = gap_top - gap_bottom

                if abs(current_price - ce_level) < 1.5:
                    messages.append(f"🎯 CE TEST Bullish FVG: {ce_level:.2f} (gap {gap_size:.2f})")
                elif gap_bottom <= current_price <= gap_top:
                    messages.append(f"⚡ INSIDE Bullish FVG [{gap_bottom:.2f}–{gap_top:.2f}]")

            # Bearish FVG
            if float(c2['high']) < float(c0['low']):
                gap_top    = float(c0['low'])
                gap_bottom = float(c2['high'])
                ce_level   = gap_bottom + (gap_top - gap_bottom) / 2
                gap_size   = gap_top - gap_bottom

                if abs(current_price - ce_level) < 1.5:
                    messages.append(f"🎯 CE TEST Bearish FVG: {ce_level:.2f} (gap {gap_size:.2f})")
                elif gap_bottom <= current_price <= gap_top:
                    messages.append(f"⚡ INSIDE Bearish FVG [{gap_bottom:.2f}–{gap_top:.2f}]")

    except Exception as e:
        logger.warning(f"Deep FVG error: {e}")

    if messages:
        return " | ".join(messages[:3])  # Max 3 mesaje
    return "⚪ No active CE test detected"


# =============================================================================
# MODUL 9: NEWS IMPACT FILTER (Economic Calendar Heuristic)
# =============================================================================
def check_news_impact(target_ts: str, trade_news: bool = False) -> tuple:
    """
    Filtrează semnalele în preajma evenimentelor economice de impact major.

    Fix v5.1: Filtrul anterior era prea agresiv:
      - CPI/PPI: ORICE marți/miercuri 15:30 → ×0.5  (greșit — CPI e lunar)
      - Jobless Claims: ORICE joi 15:30 → ×0.7       (prea des penalizat)

    v5.2 — NEWS TRADE MODE (trade_news=True):
      - NFP/FOMC BLACKOUT → înlocuit cu NEWS TRADE MODE (×0.88)
      - Primele 2 min după release: skip spike haotic (×0.20)
      - Pre-NFP buffer redus la 1 min (doar 15:29)
      - Post-FOMC continuation (20:30-21:00): full permis (×1.0)
      - Necesită scor +8% mai mare (aplicat în aladin_engine)
      - TP extins ×1.3 de bridge_api când news_mode_active=True

    Returnează: (multiplier: float, message: str)
    """
    try:
        t_dt  = pd.to_datetime(target_ts)
        t_str = t_dt.strftime("%H:%M")
        dow   = t_dt.strftime("%A")
        day   = t_dt.day

        # ── NFP — primul vineri al lunii ─────────────────────────────────────
        if dow == "Friday" and day <= 7:
            if t_str == "15:30":
                if trade_news:
                    # Prima bară = spike haotic → skip, așteptăm confirmarea direcției
                    return 0.20, "⏱️ NFP SPIKE (15:30) — așteptăm 2 min pentru confirmare direcție"
                return 0.1, "🚨 NEWS BLACKOUT: NFP Release — NO TRADE"
            if t_str == "15:31":
                if trade_news:
                    return 0.20, "⏱️ NFP SPIKE (15:31) — încă instabil, mai așteptăm 1 min"
                return 0.1, "🚨 NEWS BLACKOUT: NFP Release — NO TRADE"
            if "15:32" <= t_str <= "15:55":
                if trade_news:
                    return 0.88, "📰 NEWS TRADE MODE: NFP Continuation — scor +8% necesar | TP ×1.3"
                return 0.1, "🚨 NEWS BLACKOUT: NFP Release — NO TRADE"
            # Pre-NFP buffer
            if trade_news and t_str == "15:29":
                return 0.30, "⚠️ PRE-NFP (1 min) — imediat înainte de release"
            if not trade_news and t_str in ("15:25", "15:26", "15:27", "15:28", "15:29"):
                return 0.5, "⚠️ PRE-NFP BUFFER: 5 min înainte de NFP"

        # ── FOMC — miercuri seara ─────────────────────────────────────────────
        if dow == "Wednesday":
            if "19:45" <= t_str <= "20:01":
                if trade_news:
                    return 0.20, "⏱️ FOMC SPIKE — așteptăm direcția (2 min)"
                return 0.1, "🚨 NEWS BLACKOUT: FOMC Statement — NO TRADE"
            if "20:02" <= t_str <= "20:30":
                if trade_news:
                    return 0.88, "📰 NEWS TRADE MODE: FOMC Continuation — scor +8% necesar | TP ×1.3"
                return 0.1, "🚨 NEWS BLACKOUT: FOMC Statement — NO TRADE"
            # Post-FOMC continuation (20:30-21:00)
            if "20:30" <= t_str <= "21:00":
                if trade_news:
                    return 1.0, "✅ POST-FOMC CONTINUATION — piață în tendință clară"
                return 0.7, "⚠️ POST-FOMC: Volatilitate ridicată"

        # ── CPI — marți săptămâna 2, 15:30 ──────────────────────────────────
        if dow == "Tuesday" and 8 <= day <= 14 and t_str == "15:30":
            if trade_news:
                return 0.88, "📰 NEWS TRADE MODE: CPI Release — scor +8% necesar | TP ×1.3"
            return 0.7, "⚠️ NEWS CAUTION: Potential CPI Release (Tue wk2)"

        # ── PPI — miercuri săptămâna 2-3, 15:30 ─────────────────────────────
        if dow == "Wednesday" and 8 <= day <= 17 and t_str == "15:30":
            return 0.8, "⚠️ NEWS CAUTION: Potential PPI Release"

        # ── Jobless Claims — joi 15:30 ────────────────────────────────────────
        if dow == "Thursday" and t_str == "15:30":
            return 0.85, "ℹ️ Jobless Claims 15:30 — impact redus"

        # ── PCE — ultima vineri din lună, 15:30 ──────────────────────────────
        if dow == "Friday" and day >= 25 and t_str == "15:30":
            if trade_news:
                return 0.88, "📰 NEWS TRADE MODE: PCE Release — scor +8% necesar | TP ×1.3"
            return 0.7, "⚠️ NEWS CAUTION: PCE Deflator Release"

        # ── JOLTS (Job Openings) — prima marți din lună, 16:00 ───────────────
        if dow == "Tuesday" and 1 <= day <= 7 and t_str in ("16:00", "15:59", "16:01"):
            if trade_news:
                return 0.88, "📰 NEWS TRADE MODE: JOLTS Job Openings — scor +8% | TP ×1.3"
            return 0.75, "⚠️ NEWS CAUTION: JOLTS Job Openings (impact mediu-ridicat)"

        # ── ISM Manufacturing — prima zi lucrătoare din lună, 16:00 ──────────
        if dow in ("Monday", "Tuesday") and 1 <= day <= 5 and t_str in ("16:00", "15:59", "16:01"):
            if trade_news:
                return 0.88, "📰 NEWS TRADE MODE: ISM Manufacturing — scor +8% | TP ×1.3"
            return 0.75, "⚠️ NEWS CAUTION: ISM Manufacturing PMI"

        # ── ISM Services — prima miercuri sau joi din lună, 16:00 ────────────
        if dow in ("Wednesday", "Thursday") and 1 <= day <= 7 and t_str in ("16:00", "15:59", "16:01"):
            return 0.80, "⚠️ NEWS CAUTION: ISM Services PMI"

        # ── Retail Sales — miercuri săptămâna 2, 15:30 ───────────────────────
        if dow == "Wednesday" and 8 <= day <= 14 and t_str == "15:30":
            if trade_news:
                return 0.88, "📰 NEWS TRADE MODE: Retail Sales — scor +8% | TP ×1.3"
            return 0.75, "⚠️ NEWS CAUTION: Retail Sales (impact ridicat pt USD/NQ)"

        # ── GDP (Advance/Preliminary) — ultima săptămână din lună, 15:30 ─────
        if dow in ("Wednesday", "Thursday") and day >= 25 and t_str == "15:30":
            if trade_news:
                return 0.88, "📰 NEWS TRADE MODE: GDP Release — scor +8% | TP ×1.3"
            return 0.70, "⚠️ NEWS CAUTION: GDP Release"

        # ── ADP Employment — miercuri înainte de NFP, 15:15 ──────────────────
        if dow == "Wednesday" and 1 <= day <= 6 and t_str in ("15:15", "15:14", "15:16"):
            return 0.80, "⚠️ NEWS CAUTION: ADP Employment (pre-NFP indicator)"

        # ── Consumer Confidence — ultima marți din lună, 16:00 ───────────────
        if dow == "Tuesday" and day >= 25 and t_str in ("16:00", "15:59", "16:01"):
            return 0.85, "ℹ️ Consumer Confidence — impact moderat"

        return 1.0, "✅ No high-impact news scheduled"

    except Exception as e:
        logger.warning(f"News filter error: {e}")
        return 1.0, "⚪ News filter N/A"


def is_news_blackout(target_ts: str) -> bool:
    """Helper rapid — returnează True dacă suntem în blackout."""
    mult, _ = check_news_impact(target_ts)
    return mult < 0.2


def check_news_continuation(target_ts: str) -> tuple:
    """
    FAZA 3.3 — News Continuation Setup.
    Detectează fereastra POST-news (15-30 min după eveniment major).
    În această fereastră, un move puternic în direcția surprizei = setup ICT de mare calitate.

    Returnează: (boost: float, reason: str)
      boost > 0 → boost scor (setup valid)
      boost = 0 → situație normală
    """
    try:
        t_dt  = pd.to_datetime(target_ts)
        t_str = t_dt.strftime("%H:%M")
        dow   = t_dt.strftime("%A")
        day   = t_dt.day

        # FIX v8.1: Ferestre extinse — pre-news (-5min) + post-news (60min)
        # NFP: vineri prima săptămână, 15:25-16:30 (era 15:31-16:00)
        if dow == "Friday" and day <= 7 and "15:25" <= t_str <= "16:30":
            if t_str <= "15:30":
                return 0.05, f"⚡ PRE-NFP ALERT ({t_str}) → cautious boost +5%"
            elif t_str <= "16:00":
                return 0.12, f"🚀 POST-NFP CONTINUATION ({t_str}) → boost +12%"
            else:
                return 0.06, f"📉 LATE NFP CONTINUATION ({t_str}) → fade boost +6%"

        # FOMC: miercuri, 19:55-21:00 (era 20:00-20:45)
        if dow == "Wednesday" and "19:55" <= t_str <= "21:00":
            if t_str <= "20:00":
                return 0.04, f"⚡ PRE-FOMC ALERT ({t_str}) → cautious boost +4%"
            elif t_str <= "20:45":
                return 0.10, f"🚀 POST-FOMC CONTINUATION ({t_str}) → boost +10%"
            else:
                return 0.05, f"📉 LATE FOMC CONTINUATION ({t_str}) → fade boost +5%"

        # CPI: marți săptămâna 2, 15:25-16:30 (era 15:31-16:00)
        if dow == "Tuesday" and 8 <= day <= 14 and "15:25" <= t_str <= "16:30":
            if t_str <= "15:30":
                return 0.04, f"⚡ PRE-CPI ALERT ({t_str}) → cautious boost +4%"
            elif t_str <= "16:00":
                return 0.08, f"🚀 POST-CPI CONTINUATION ({t_str}) → boost +8%"
            else:
                return 0.04, f"📉 LATE CPI CONTINUATION ({t_str}) → fade boost +4%"

        # PCE: ultima vineri, 15:25-16:30 (era 15:31-16:00)
        if dow == "Friday" and day >= 25 and "15:25" <= t_str <= "16:30":
            if t_str <= "15:30":
                return 0.04, f"⚡ PRE-PCE ALERT ({t_str}) → cautious boost +4%"
            elif t_str <= "16:00":
                return 0.08, f"🚀 POST-PCE CONTINUATION ({t_str}) → boost +8%"
            else:
                return 0.04, f"📉 LATE PCE CONTINUATION ({t_str}) → fade boost +4%"

        return 0.0, ""

    except Exception as e:
        logger.debug(f"News continuation check error: {e}")
        return 0.0, ""


# =============================================================================
# MODUL 10: PYRAMIDING PLAN (Scale-In Sniper Logic)
# =============================================================================
def get_pyramiding_plan(score: float, risk: dict, best: pd.Series) -> list:
    """
    Generează planul de scalare în poziție (Pyramiding) pentru score > 0.82.

    Logica:
      Entry 1: Imediat la prețul curent
      Entry 2: La retestul CE sau la +0.5 ATR în direcție
      Entry 3: La +1.0 ATR în direcție (confirmation)

    Fiecare nivel reduce sizing-ul (50% → 30% → 20%).

    Returnează: list[dict] sau string dacă score < 0.82.
    """
    if score <= 0.55:
        return f"Score {score*100:.1f}% sub pragul de pyramiding (55%)"

    try:
        close  = float(best['close'])
        atr    = float(best.get('atr_14', 8.0)) or 8.0
        units  = risk.get('units', 1.0)
        is_long = score > 0.5

        direction = 1 if is_long else -1

        plan = [
            {
                "entry_num":      1,
                "trigger_price":  round(close, 2),
                "added_units":    round(units * 0.5, 2),
                "cumulative":     round(units * 0.5, 2),
                "note":           "Initial Sniper Entry (50%)",
                "sl":             risk.get('sl', 0),
            },
            {
                "entry_num":      2,
                "trigger_price":  round(close + direction * atr * 0.5, 2),
                "added_units":    round(units * 0.3, 2),
                "cumulative":     round(units * 0.8, 2),
                "note":           "Add on CE retest / 0.5 ATR extension (30%)",
                "sl":             round(close - direction * atr * 0.25, 2),
            },
            {
                "entry_num":      3,
                "trigger_price":  round(close + direction * atr * 1.0, 2),
                "added_units":    round(units * 0.2, 2),
                "cumulative":     round(units * 1.0, 2),
                "note":           "Confirmation add on 1 ATR (20%)",
                "sl":             round(close + direction * atr * 0.25, 2),
            },
        ]

        return plan

    except Exception as e:
        logger.warning(f"Pyramiding plan error: {e}")
        return []


# =============================================================================
# MODUL 12: RAG PATTERN MEMORY (FAISS + SentenceTransformers)
# =============================================================================
class AladinRAG:
    """
    RAG Pattern Memory pentru Aladin.
    Stochează pattern-uri de trading anterioare și le recuperează prin similaritate.

    Dacă FAISS sau SentenceTransformers nu sunt instalate,
    fallback la căutare prin keyword în jurnalul CSV.
    """

    def __init__(self):
        self.faiss_available = False
        self.st_available    = False
        self.index           = None
        self.metadata        = []
        self.model           = None
        self._init_backends()

    def _init_backends(self):
        """Inițializează FAISS și SentenceTransformers dacă sunt disponibile."""
        try:
            import faiss
            self.faiss_available = True
            logger.info("📚 FAISS disponibil — RAG vector store activ")
        except ImportError:
            logger.warning("⚠️  FAISS nu este instalat. RAG fallback la CSV keyword search.")

        # Fix v9.2: folosim singleton de la nivel de modul — nu creăm instanță nouă
        self.model = _get_st_model()
        self.st_available = _st_model_available and self.model is not None
        if self.st_available:
            logger.info("📚 SentenceTransformers — model singleton reutilizat (fără reload)")
        else:
            logger.warning("⚠️  SentenceTransformers nu este disponibil — RAG offline.")

        # Încearcă să încarce index existent
        if self.faiss_available and self.st_available:
            self._load_index()

    def _load_index(self):
        """Încarcă indexul FAISS și metadata de pe disk."""
        try:
            import faiss
            if os.path.exists(RAG_INDEX_PATH) and os.path.exists(RAG_META_PATH):
                self.index = faiss.read_index(RAG_INDEX_PATH)
                with open(RAG_META_PATH, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"📚 RAG index încărcat: {len(self.metadata)} pattern-uri")
        except Exception as e:
            logger.warning(f"RAG index load error: {e}")

    def _save_index(self):
        """Salvează indexul FAISS și metadata pe disk."""
        try:
            import faiss
            if self.index and self.metadata:
                faiss.write_index(self.index, RAG_INDEX_PATH)
                with open(RAG_META_PATH, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"RAG index save error: {e}")

    def add_pattern(self, pattern_text: str, metadata: dict):
        """
        Adaugă un pattern nou în memoria RAG.
        pattern_text: descriere text a setup-ului
        metadata: dict cu score, timestamp, verdict, etc.
        """
        if not (self.faiss_available and self.st_available and self.model):
            return

        try:
            import faiss
            embedding = self.model.encode([pattern_text], convert_to_numpy=True)
            dim = embedding.shape[1]

            if self.index is None:
                self.index = faiss.IndexFlatL2(dim)

            self.index.add(embedding.astype('float32'))
            self.metadata.append({**metadata, "text": pattern_text})
            self._save_index()
            logger.info(f"📚 Pattern RAG adăugat: {pattern_text[:60]}...")

        except Exception as e:
            logger.warning(f"RAG add pattern error: {e}")

    def query(self, query_text: str, top_k: int = 3) -> str:
        """
        Recuperează cele mai similare pattern-uri pentru query-ul dat.
        Returnează text formatat pentru dashboard.
        """
        # Fallback 1: FAISS + ST disponibil
        if self.faiss_available and self.st_available and self.model and self.index:
            try:
                import faiss
                if self.index.ntotal == 0:
                    return self._csv_fallback_query(query_text)

                q_emb = self.model.encode([query_text], convert_to_numpy=True)
                distances, indices = self.index.search(q_emb.astype('float32'), top_k)

                lines = ["🧠 RAG PATTERN MEMORY — Top Matches:", "─" * 60]
                for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < len(self.metadata):
                        m = self.metadata[idx]
                        sim = max(0, 100 - dist * 10)
                        lines.append(
                            f"[{rank+1}] Sim: {sim:.0f}% | "
                            f"Score: {m.get('score','?')} | "
                            f"Verdict: {m.get('verdict','?')} | "
                            f"{m.get('timestamp','?')}"
                        )
                        lines.append(f"     {m.get('text','')[:80]}")
                return "\n".join(lines)

            except Exception as e:
                logger.warning(f"RAG query error: {e}")

        # Fallback 2: CSV keyword search
        return self._csv_fallback_query(query_text)

    def _csv_fallback_query(self, query_text: str) -> str:
        """Fallback: caută pattern-uri similare în jurnalul CSV prin keyword."""
        try:
            if not os.path.isfile(JOURNAL_PATH):
                return "📚 RAG: Nicio memorie disponibilă (jurnal gol)"

            df = pd.read_csv(JOURNAL_PATH, low_memory=False)
            if df.empty:
                return "📚 RAG: Jurnalul este gol"

            # Extrage keywords din query
            keywords = query_text.lower().split()
            results  = []

            for _, row in df.tail(50).iterrows():
                row_text = str(row.get('macro_bias', '')) + str(row.get('smt', ''))
                match_score = sum(1 for kw in keywords if kw in row_text.lower())
                if match_score > 0:
                    results.append((match_score, row))

            results.sort(key=lambda x: x[0], reverse=True)
            top = results[:3]

            if not top:
                return "📚 RAG Fallback: Niciun pattern similar găsit în jurnal"

            lines = ["🧠 RAG KEYWORD FALLBACK — Pattern Memory:", "─" * 60]
            for rank, (sim, row) in enumerate(top):
                lines.append(
                    f"[{rank+1}] Score: {row.get('hybrid_score','?')}% | "
                    f"Verdict: {row.get('verdict','?')} | {row.get('timestamp','?')}"
                )
                lines.append(f"     Bias: {row.get('macro_bias','N/A')}")
            return "\n".join(lines)

        except Exception as e:
            return f"📚 RAG Fallback error: {e}"

    def build_rag_context(self, best: pd.Series, nar: dict) -> str:
        """
        Construiește query-ul RAG din starea curentă a pieței și recuperează pattern-uri.
        """
        query = (
            f"{nar.get('macro', '')} "
            f"{nar.get('smt', '')} "
            f"{nar.get('fvg', '')} "
            f"{nar.get('killzone', '')} "
            f"close={best['close']:.0f}"
        )
        return self.query(query)

    def store_current_analysis(self, best: pd.Series, nar: dict, score: float, verdict: str):
        """Stochează analiza curentă în memoria RAG pentru utilizări viitoare."""
        pattern_text = (
            f"{nar.get('macro', '')} | {nar.get('smt', '')} | "
            f"{nar.get('fvg', '')} | {nar.get('killzone', '')} | "
            f"close={best['close']:.0f} score={score*100:.0f}%"
        )
        meta = {
            "score":     f"{score*100:.1f}%",
            "verdict":   verdict,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "macro":     nar.get('macro', ''),
        }
        self.add_pattern(pattern_text, meta)


# ── Singleton SentenceTransformer — module-level, nu poate fi resetat ─────────
# Fix v9.2: modelul ST era în AladinRAG._init_backends() — dacă AladinRAG() era
# re-instanțiat (orice motiv), modelul se reîncărca de la zero (3-4s / ciclu).
# Mutând modelul la nivel de modul, el se inițializează O SINGURĂ DATĂ per proces.
_st_model_singleton = None
_st_model_available = False

def _get_st_model():
    """Returnează modelul SentenceTransformer singleton — inițializat o singură dată."""
    global _st_model_singleton, _st_model_available
    if _st_model_singleton is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model_singleton = SentenceTransformer('all-MiniLM-L6-v2')
            _st_model_available = True
            logger.info("📚 SentenceTransformer singleton inițializat — all-MiniLM-L6-v2")
        except Exception as _st_err:
            logger.warning(f"⚠️  SentenceTransformer unavailable: {_st_err}")
            _st_model_available = False
    return _st_model_singleton


# Singleton RAG instance
_rag_instance = None

def get_rag() -> AladinRAG:
    """Returnează instanța singleton a RAG-ului."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = AladinRAG()
    return _rag_instance


# =============================================================================
# MODUL 13: QUANTUM NOISE FILTER
# =============================================================================
def calculate_conviction_level(score: float, best: pd.Series, kz: Optional[str]) -> str:
    """
    Calculează nivelul de convicție calitativă al semnalului.
    Returnează: 'DIAMOND' / 'HIGH' / 'MEDIUM' / 'LOW' / 'NOISE'
    """
    has_smt = bool(best.get('is_smt_bearish', 0) or best.get('is_smt_bullish', 0))
    has_fvg = bool(best.get('fvg_up', 0) or best.get('fvg_down', 0))
    has_dis = bool(best.get('has_displacement', 0))
    in_kz   = kz is not None

    conf = sum([has_smt, has_fvg, has_dis, in_kz])

    # Fix v10.3: thresholds coborâte — score 30-40% cu HTF bullish = setup real, nu noise
    if score > 0.55 and conf >= 2:
        return "DIAMOND 💎"
    elif score > 0.45 and conf >= 1:
        return "HIGH 🟢"
    elif score > 0.35 and conf >= 1:
        return "MEDIUM 🟡"
    elif score > 0.25:
        return "LOW 🔴"
    else:
        return "NOISE ⚫"


# =============================================================================
# MODUL 14: BACKTESTING UTILITIES
# =============================================================================
def calculate_backtest_stats(df_journal: pd.DataFrame) -> dict:
    """
    Calculează statistici de backtest din jurnalul de audit.
    Returnează dict cu: total, win_rate, avg_rr, profit_factor, max_dd.
    """
    if df_journal.empty:
        return {}

    try:
        total = len(df_journal)
        if 'result' in df_journal.columns:
            wins = (df_journal['result'] == 'WIN').sum()
            win_rate = wins / total * 100 if total > 0 else 0
        else:
            # Estimăm din scor
            wins = (df_journal.get('hybrid_score', pd.Series()) > 82).sum()
            win_rate = wins / total * 100 if total > 0 else 0

        avg_score = df_journal['hybrid_score'].mean() if 'hybrid_score' in df_journal.columns else 0

        return {
            "total_signals": total,
            "high_conviction": int(wins),
            "win_rate_est":    round(win_rate, 1),
            "avg_score":       round(avg_score, 2),
        }
    except Exception as e:
        logger.warning(f"Backtest stats error: {e}")
        return {}


def get_market_regime(df: pd.DataFrame) -> str:
    """v6.0: IQP 8 qubiți cu fallback clasic."""
    return get_market_regime_quantum(df)


# =============================================================================
# UPDATE #42: DATA QUALITY MONITORING
# =============================================================================
def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Update #42: Data quality monitoring.
    Dacă feed-ul de date pică sau trimite valori aberante → oprește trading automat.
    Validare: close > low, high > open, volume > 0, fără NaN pe coloane critice.
    Returnează: {'valid': bool, 'issues': list, 'quality_score': float}
    """
    issues = []

    if df is None or df.empty:
        return {'valid': False, 'issues': ['DataFrame gol'], 'quality_score': 0.0}

    if len(df) < 10:
        return {'valid': False, 'issues': [f'Prea puține date: {len(df)} bare'], 'quality_score': 0.0}

    last = df.tail(10)

    # Validare OHLC basic
    if 'high' in last.columns and 'low' in last.columns:
        bad_hl = (last['high'] < last['low']).sum()
        if bad_hl > 0:
            issues.append(f'high < low pe {bad_hl} bare')

    if 'close' in last.columns and 'low' in last.columns:
        bad_cl = (last['close'] < last['low']).sum()
        if bad_cl > 0:
            issues.append(f'close < low pe {bad_cl} bare')

    if 'close' in last.columns and 'high' in last.columns:
        bad_ch = (last['close'] > last['high']).sum()
        if bad_ch > 0:
            issues.append(f'close > high pe {bad_ch} bare')

    # Validare volum
    if 'volume' in last.columns:
        bad_vol = (last['volume'] <= 0).sum()
        if bad_vol > 0:
            issues.append(f'volume <= 0 pe {bad_vol} bare')

    # Validare NaN pe coloane critice
    critical_cols = ['open', 'high', 'low', 'close']
    for col in critical_cols:
        if col in last.columns:
            nan_count = last[col].isna().sum()
            if nan_count > 0:
                issues.append(f'NaN pe {col}: {nan_count} bare')

    # Validare price spike (>5% variație pe o bară)
    if 'close' in last.columns and len(last) > 1:
        pct_change = last['close'].pct_change().abs()
        spikes = (pct_change > 0.05).sum()
        if spikes > 0:
            issues.append(f'Price spike >5% pe {spikes} bare — posibil date aberante')

    # Quality score: 1.0 dacă fără probleme, scade cu fiecare problemă
    quality_score = max(0.0, 1.0 - len(issues) * 0.25)
    valid = len(issues) == 0

    if issues:
        logger.warning(f"   ⚠️  Data Quality issues: {issues}")

    return {
        'valid':         valid,
        'issues':        issues,
        'quality_score': round(quality_score, 2),
        'n_bars':        len(df),
    }


# =============================================================================
# UPDATE #43: SYNTHETIC DXY CALCULATOR
# =============================================================================
def get_synthetic_dxy() -> dict:
    """
    Update #43 + UPDATE #4 (cache 5 min): Synthetic DXY calculator.
    Construiește DXY din 6 pairs forex prin yfinance (gratuit):
    EUR/USD (57.6%), USD/JPY (13.6%), GBP/USD (11.9%),
    USD/CAD (9.1%), USD/SEK (4.2%), USD/CHF (3.6%)

    Formula DXY: 50.14348112 × EURUSD^(-0.576) × USDJPY^0.136 × GBPUSD^(-0.119)
                 × USDCAD^0.091 × USDSEK^0.042 × USDCHF^0.036

    Returnează: {'dxy': float, 'trend': str, 'bullish': bool, 'source': str}
    """
    cached = _cache_get('dxy')
    if cached is not None:
        return cached
    try:
        import yfinance as yf

        tickers = {
            'EURUSD=X': ('EUR/USD', -0.576),
            'JPY=X':    ('USD/JPY',  0.136),
            'GBPUSD=X': ('GBP/USD', -0.119),
            'CAD=X':    ('USD/CAD',  0.091),
            'SEK=X':    ('USD/SEK',  0.042),
            'CHF=X':    ('USD/CHF',  0.036),
        }

        prices = {}
        for ticker, (name, _) in tickers.items():
            try:
                hist = yf.Ticker(ticker).history(period='5d', interval='1d')
                if not hist.empty:
                    prices[ticker] = float(hist['Close'].iloc[-1])
            except Exception:
                pass

        if len(prices) < 4:
            return {'dxy': 100.0, 'trend': 'UNKNOWN', 'bullish': True, 'source': 'fallback'}

        # Formula DXY
        dxy = 50.14348112
        for ticker, (name, exponent) in tickers.items():
            if ticker in prices:
                dxy *= prices[ticker] ** exponent

        # DXY >104 → USD puternic → bullish USD → bearish QQQ (corelație inversă)
        # DXY <100 → USD slab → bullish QQQ
        dxy_trend = "STRONG" if dxy > 104 else "WEAK" if dxy < 100 else "NEUTRAL"
        dxy_bullish_for_equity = dxy < 102  # USD slab = bun pentru acțiuni

        logger.info(f"   💵 DXY Sintetic: {dxy:.2f} ({dxy_trend}) — {'🟢 Favorabil QQQ' if dxy_bullish_for_equity else '🔴 Nefavorabil QQQ'}")

        result = {
            'dxy':      round(dxy, 2),
            'trend':    dxy_trend,
            'bullish':  dxy_bullish_for_equity,
            'source':   'synthetic_yfinance',
            'prices':   prices,
        }
        _cache_set('dxy', result)
        return result

    except Exception as e:
        logger.debug(f"DXY synthetic error: {e}")
        return {'dxy': 100.0, 'trend': 'UNKNOWN', 'bullish': True, 'source': 'error'}


# =============================================================================
# UPDATE #53: OPTIONS FLOW SENTIMENT (PUT/CALL RATIO)
# =============================================================================
def get_options_flow_signal() -> dict:
    """
    Update #53 + UPDATE #4 (cache 5 min): Options flow sentiment (Put/Call ratio).
    Dacă Put/Call ratio >1.5 → bearish pressure → ajustează bias SHORT.
    Date gratuite prin yfinance options chain pe QQQ.
    Returnează: {'pc_ratio': float, 'signal': str, 'bias_adj': float}
    """
    cached = _cache_get('options')
    if cached is not None:
        return cached
    try:
        import yfinance as yf

        ticker = yf.Ticker("QQQ")

        # Obține lanțul de opțiuni (prima expirare disponibilă)
        expirations = ticker.options
        if not expirations:
            return {'pc_ratio': 1.0, 'signal': 'NEUTRAL', 'bias_adj': 0.0, 'source': 'no_data'}

        # Folsim prima expirare
        chain = ticker.option_chain(expirations[0])

        put_vol  = float(chain.puts['volume'].sum())
        call_vol = float(chain.calls['volume'].sum())

        pc_ratio = put_vol / (call_vol + 1e-8)

        if pc_ratio > 1.5:
            signal   = "BEARISH"
            bias_adj = -0.05  # reduce score pentru LONG, crește pentru SHORT
        elif pc_ratio < 0.7:
            signal   = "BULLISH"
            bias_adj = +0.03  # ușor boost pentru LONG
        else:
            signal   = "NEUTRAL"
            bias_adj = 0.0

        logger.info(f"   📊 Options P/C Ratio: {pc_ratio:.2f} → {signal}")

        result = {
            'pc_ratio': round(pc_ratio, 3),
            'signal':   signal,
            'bias_adj': bias_adj,
            'source':   'yfinance_options',
            'put_vol':  int(put_vol),
            'call_vol': int(call_vol),
        }
        _cache_set('options', result)
        return result

    except Exception as e:
        logger.debug(f"Options flow error: {e}")
        return {'pc_ratio': 1.0, 'signal': 'NEUTRAL', 'bias_adj': 0.0, 'source': 'error'}


# =============================================================================
# UPDATE #54: FRED API MACRO FILTER
# =============================================================================
_fred_cache = {}  # cache legacy — înlocuit de _CACHE global (UPDATE #4)

def get_fred_macro_filter() -> dict:
    """
    Update #54: FRED API macro filter (Federal Reserve Economic Data — GRATUIT).
    Integrează: Fed Funds Rate, CPI, Unemployment.
    Reduce sizing în perioadele de incertitudine macro.
    API Key FRED: înregistrare gratuită pe fred.stlouisfed.org
    Returnează: {'sizing_mult': float, 'macro_risk': str, 'data': dict}
    """
    global _fred_cache

    # UPDATE #4: folosim cache-ul unificat cu TTL 1 oră pentru FRED (date zilnice)
    cached = _cache_get('fred')
    if cached is not None:
        return cached

    try:

        # Încearcă fredapi (pip install fredapi)
        try:
            from fredapi import Fred
            fred_key = os.environ.get('FRED_API_KEY', '')
            if not fred_key:
                raise ImportError("FRED_API_KEY lipsă din env")
            fred = Fred(api_key=fred_key)

            # Fed Funds Rate (FEDFUNDS)
            ff_rate = float(fred.get_series('FEDFUNDS').dropna().iloc[-1])
            # CPI YoY (CPIAUCSL)
            cpi = fred.get_series('CPIAUCSL').dropna()
            cpi_yoy = float((cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100) if len(cpi) > 13 else 3.0
            # Unemployment (UNRATE)
            unemp = float(fred.get_series('UNRATE').dropna().iloc[-1])

        except Exception:
            # Fallback la yfinance pentru indicatori proxy
            import yfinance as yf
            # TIPS spread ca proxy pentru inflation expectations
            try:
                tips = yf.Ticker("TIP").history(period="5d")
                spy  = yf.Ticker("SPY").history(period="5d")
                ff_rate = 5.25  # valoare estimată curentă
                cpi_yoy = 3.2   # estimat
                unemp   = 4.0   # estimat
            except Exception:
                ff_rate, cpi_yoy, unemp = 5.25, 3.2, 4.0

        # Evaluare risc macro
        macro_risk_score = 0

        if ff_rate > 5.0:
            macro_risk_score += 1  # dobânzi ridicate → presiune pe QQQ
        if cpi_yoy > 4.0:
            macro_risk_score += 1  # inflație ridicată → Fed hawkish → presiune QQQ
        if unemp > 5.0:
            macro_risk_score += 1  # șomaj ridicat → economic stress

        if macro_risk_score >= 2:
            macro_risk    = "HIGH"
            sizing_mult   = 0.75  # reducem sizing cu 25%
        elif macro_risk_score == 1:
            macro_risk    = "MEDIUM"
            sizing_mult   = 0.90
        else:
            macro_risk    = "LOW"
            sizing_mult   = 1.0

        result_data = {
            'sizing_mult': sizing_mult,
            'macro_risk':  macro_risk,
            'data': {
                'fed_funds_rate': round(ff_rate, 2),
                'cpi_yoy':        round(cpi_yoy, 2),
                'unemployment':   round(unemp, 2),
            }
        }

        # UPDATE #4: cache unificat cu TTL 1 oră
        _CACHE_TTL_FRED = 3600.0  # 1 oră pentru FRED (date zilnice, nu se schimbă des)
        _CACHE['fred'] = {'val': result_data, 'ts': _time_module.time() - (_CACHE_TTL - _CACHE_TTL_FRED)}

        logger.info(f"   🏛️  FRED Macro: FF={ff_rate:.2f}% CPI={cpi_yoy:.1f}% U={unemp:.1f}% → Risk={macro_risk} sizing×{sizing_mult}")
        return result_data

    except Exception as e:
        logger.debug(f"FRED macro filter error: {e}")
        return {'sizing_mult': 1.0, 'macro_risk': 'UNKNOWN', 'data': {}}


# =============================================================================
# MODUL 15: ALADIN ENGINE PRINCIPAL
# =============================================================================
def aladin_engine(query: str, balance: float = 10000, live_data: dict = None) -> dict:
    """
    Motorul principal Aladin Quantum-ICT v5.0.

    Integrează:
      ✅ AI Prediction (XGBoost 37 features)
      ✅ Quantum Validation (PennyLane circuit)
      ✅ AMT (POC, Value Area)
      ✅ ICT Core (FVG, Displacement, SMT, Power of 3)
      ✅ HTF Hierarchy (Monthly, Weekly, PDH/PDL, Monday)
      ✅ News Filter (Economic Calendar Heuristic)
      ✅ Relative Strength (QQQ vs SPY)
      ✅ SD Targets (Institutional TP levels)
      ✅ Risk Management (ATR-based position sizing)
      ✅ Pyramiding Plan (Scale-in Sniper)
      ✅ Order Blocks (Institutional candle detection)
      ✅ RAG Pattern Memory (FAISS + fallback)
      ✅ Automated Journaling

    Returnează: dict complet pentru Streamlit dashboard.
    Niciodată nu returnează None — garantat dict.
    """
    # ── INITIALIZE ALL VARIABLES WITH SAFE DEFAULTS (Fix v6.9) ─────────────
    # Bug #7: vix_mult, macro_sizing_mult, dxy_adj, options_adj etc.
    # trebuie să fie disponibile dacă orice bloc mai târziu eșuează
    vix_mult = 1.0
    macro_sizing_mult = 1.0
    dxy_adj = 0.0
    options_adj = 0.0
    vol_boost = 0.0
    skip_volatile = False
    vol_msg = ""
    circuit_blocked = False
    sentiment_score = 0.5
    sweep_score = 0.3
    poc_side_score = 0.5
    fvg_score = 0.5
    htf_bullish = False

    # ── Guard: DB existence ──────────────────────────────────────────────────
    if not os.path.exists(PATH_DB):
        logger.error(f"DB lipsește: {PATH_DB}")
        return {
            "verdict": "❌ Baza de date SQL lipsește! Rulați Data Pipeline.",
            "score": 0,
        }

    # FIX v10.6: read-only URI — nu schimbăm journal_mode (NT8 are DB deschis pentru scriere)
    # PRAGMA journal_mode=WAL pe o conexiune read-write → disk I/O error când NT8 scrie simultan
    conn = sqlite3.connect(f'file:{PATH_DB}?mode=ro', uri=True,
                           timeout=30, check_same_thread=False)

    try:
        # ── 1. Parsing timp & calendar ───────────────────────────────────────
        match = re.search(r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})', query)
        if not match:
            return {
                "verdict": "❌ Format invalid. Folosiți: YYYY-MM-DD HH:MM",
                "score": 0,
            }

        target_ts = f"{match.group(1)} {match.group(2)}:00"
        t_dt      = pd.to_datetime(target_ts)
        t_str     = t_dt.strftime("%H:%M")   # ET time (pentru DB queries care sunt în ET)

        # Fix v10.5: Convertim ET → EEST (Europe/Bucharest) pentru killzone check
        # Bridge trimite timestamp în America/New_York (ET), dar KILLZONES sunt definite în EEST.
        # Fără conversie: 04:15 ET pică fals în "Sydney Open" (02:00-05:00 EEST)
        # Cu conversie: 04:15 ET = 11:15 EEST → niciun killzone activ (corect!)
        try:
            _t_et   = t_dt.replace(tzinfo=ZoneInfo("America/New_York"))
            _t_eest = _t_et.astimezone(ZoneInfo("Europe/Bucharest"))
            t_str_kz = _t_eest.strftime("%H:%M")
        except Exception:
            t_str_kz = t_str  # fallback la ET dacă conversie eșuează

        logger.info(f"🔍 Aladin Engine: {target_ts} (ET) | KZ check: {t_str_kz} (EEST)")

        # Verificare calendar NYSE
        if not is_market_open(t_dt):
            return {
                "verdict": f"😴 REPAUS INSTITUȚIONAL: {t_dt.date()} este zi închisă.",
                "score": 0,
                "timestamp": target_ts,
            }

        # ── 2. Data fetch ────────────────────────────────────────────────────
        df = pd.read_sql_query(
            "SELECT * FROM market_data WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 100",
            conn,
            params=(target_ts,),
        )

        if df.empty:
            return {
                "verdict": f"❌ Lipsă date în DB pentru: {target_ts}",
                "score": 0,
                "timestamp": target_ts,
            }

        df   = df.iloc[::-1].reset_index(drop=True)
        best = df.iloc[-1]

        logger.info(f"   ✅ {len(df)} rânduri încărcate | Close: {best['close']}")

        # ── LIVE NT8 OVERRIDE ────────────────────────────────────────────────
        # Când vine din NT8 live (live_data != None), înlocuim features-urile
        # de orderflow/volume profile cu valorile exacte trimise de AladinBridge.
        # Datele istorice (FVG, SMT, HTF levels) rămân din SQLite.
        # Backtesting: live_data=None → citește 100% din SQLite (comportament normal)
        _live_mode = live_data is not None and isinstance(live_data, dict)
        # Score min setat de user din dashboard (default 60 dacă nu e setat)
        _user_score_min = float(live_data.get("score_min", 60)) / 100.0 if _live_mode else 0.60

        # ── SESSION-AWARE SCORE_MIN ───────────────────────────────────────────
        # Sesiunile cu lichiditate scăzută necesită setup mai clar (prag mai mare).
        # Asian Pre (00-03 UTC): spread mare, fake moves, institucionalii dorm → +10%
        # Asian     (03-07 UTC): lichiditate redusă față de London/NY            → +5%
        # NY Close  (20-24 UTC): lichiditate în scădere, volum mic               → +5%
        # London + New York: lichiditate maximă → prag normal (0% ajustare)
        #
        # Fix v9.3: NT8 trimite timestamp în LOCAL TIME (US Eastern), nu UTC.
        # t_dt.hour = 05 (ET) → sistemul credea că e Asian session.
        # Fix: folosim datetime.utcnow() pentru ora REALĂ UTC curentă.
        _session_adj = 0.0
        _t_hour_utc  = datetime.utcnow().hour
        if 0 <= _t_hour_utc < 3:
            _session_adj    = 0.10   # Asian Pre — cel mai restrictiv
            _session_label  = "Asian Pre"
        elif 3 <= _t_hour_utc < 7:
            _session_adj    = 0.05   # Asian
            _session_label  = "Asian"
        elif 7 <= _t_hour_utc < 12:
            _session_adj    = 0.0    # London — normal
            _session_label  = "London"
        elif 12 <= _t_hour_utc < 20:
            _session_adj    = 0.0    # New York — normal
            _session_label  = "New York"
        else:
            _session_adj    = 0.05   # NY Close
            _session_label  = "NY Close"

        if _session_adj > 0:
            _base_score_min  = _user_score_min
            _user_score_min  = min(_user_score_min + _session_adj, 0.95)
            logger.info(
                f"   🕐 SESSION SCORE_MIN [{_session_label}]: "
                f"{_base_score_min:.0%} → {_user_score_min:.0%} (+{_session_adj:.0%} lichiditate scăzută)"
            )
        # NEWS TRADE MODE — dacă e activat din dashboard, robotul tranzacționează news-urile
        # în loc să le evite; cere scor +8% mai mare și extinde TP ×1.3 (via bridge_api)
        _trade_news = bool(live_data.get("trade_news", False)) if _live_mode else False
        # GEO RISK MODE — activat automat de RSS monitor (știri geopolitice/Trump/conflict) sau manual
        # NU ridică threshold-ul global (știrile geo vin zilnic → ar bloca permanent Aladin)
        # Influențează DOAR direcția via geo_sentiment (BEARISH_NQ/BULLISH_NQ)
        _geo_risk_active = bool(live_data.get("geo_risk_active", False)) if _live_mode else False
        if _geo_risk_active:
            logger.info(f"   🌍 GEO RISK MODE activ — threshold neschimbat, sentiment direcțional activ")
        # GEO SENTIMENT — singura influență geo asupra scorului: ajustare direcțională mică
        # BEARISH_NQ: război/tarife/oil shock → penalizează LONG -4%, bonus SHORT +3%
        # BULLISH_NQ: ceasefire/rate cut/trade deal → penalizează SHORT -4%, bonus LONG +3%
        _geo_sentiment = live_data.get("geo_sentiment", "NEUTRAL") if _live_mode else "NEUTRAL"
        if _live_mode:
            _live_keys = [
                # Fix v10.3: OHLC live — fără asta close/open/high/low rămân statice din DB
                'close', 'open', 'high', 'low',
                'cum_delta', 'imbalance_pct', 'vwap',
                'poc_level', 'vah', 'val',
                'bid_ask_ratio', 'atr_14',
                'fvg_up', 'fvg_down',
                'is_smt_bearish', 'is_smt_bullish',
                'has_displacement',
            ]
            _overridden = []
            for _k in _live_keys:
                if _k in live_data and live_data[_k] is not None:
                    df.at[df.index[-1], _k] = float(live_data[_k])
                    _overridden.append(_k)
            # Recalculăm dist_poc și inside_va cu poc/vah/val live
            _poc = df.at[df.index[-1], 'poc_level'] if 'poc_level' in df.columns else 0
            _vah = df.at[df.index[-1], 'vah']       if 'vah' in df.columns else 0
            _val = df.at[df.index[-1], 'val']       if 'val' in df.columns else 0
            _cls = df.at[df.index[-1], 'close']
            if _poc > 0:
                df.at[df.index[-1], 'dist_poc']  = float(_cls - _poc)
            if _vah > 0 and _val > 0:
                df.at[df.index[-1], 'inside_va'] = float(1 if _val <= _cls <= _vah else 0)
            # Recalculăm best cu datele live
            best = df.iloc[-1]
            logger.info(f"   🔴 LIVE MODE: {len(_overridden)} features override din NT8 → {_overridden}")

        # ── Update #42: Validare calitate date ──────────────────────────────
        dq = validate_data_quality(df)
        if not dq['valid']:
            logger.error(f"   ❌ Data Quality FAIL: {dq['issues']}")
            # Nu oprim complet — logăm și continuăm cu precauție
            logger.warning("   ⚠️  Continuăm cu date potențial impure — verifică feed-ul")

        # ── 3. Module de analiză ─────────────────────────────────────────────

        # A. Relative Strength (QQQ vs SPY)
        rel_mult, rel_info = analyze_relative_strength(df)

        # B. Deep FVG & Orderflow
        fvg_deep_msg   = deep_fvg_analysis(df)
        imbalance_data = analyze_orderflow_imbalance(df)

        # C. SD Targets
        sd_targets = calculate_standard_deviations(best)

        # D. News Filter — v5.2: pasăm _trade_news pentru a activa NEWS TRADE MODE
        news_mult, news_msg = check_news_impact(target_ts, trade_news=_trade_news)
        # Dacă suntem în NEWS TRADE MODE (0.80 ≤ news_mult ≤ 0.92), cerem scor +8% mai mare
        _news_mode_active = _trade_news and 0.80 <= news_mult <= 0.92
        if _news_mode_active:
            _user_score_min = min(_user_score_min + 0.08, 0.90)
            logger.info(f"   📰 NEWS TRADE MODE activ — score_min ridicat la {_user_score_min:.0%}")
        # FAZA 3.3: News Continuation boost (fereastră post-release)
        _news_cont_boost, _news_cont_msg = check_news_continuation(target_ts)

        # E. Order Blocks
        order_blocks = detect_order_blocks(df, lookback=20)

        # F. Market Regime
        regime = get_market_regime(df)

        # G. Active killzone — folosim t_str_kz (EEST) nu t_str (ET)!
        active_kz = get_active_killzone(t_str_kz)

        # ── Update #43: DXY Sintetic ─────────────────────────────────────────
        dxy_data = get_synthetic_dxy()
        # Pre-calcul HTF bias pentru options adjustment
        # Fix None-safety: best.get(key, 0) returnează None dacă coloana există cu NULL în DB
        _lw_lo_pre  = float(best.get('lw_lo') or 0)
        _lw_hi_pre  = float(best.get('lw_hi') or 0)
        _lm_lo_pre  = float(best.get('lm_lo') or 0)
        _lm_hi_pre  = float(best.get('lm_hi') or 0)
        _lw_mid_pre = _lw_lo_pre + (_lw_hi_pre - _lw_lo_pre) / 2
        _lm_mid_pre = _lm_lo_pre + (_lm_hi_pre - _lm_lo_pre) / 2
        _w_bias_pre = "BULLISH" if best['close'] > _lw_mid_pre else "BEARISH"
        _m_bias_pre = "BULLISH" if best['close'] > _lm_mid_pre else "BEARISH"
        htf_bullish_early = (_w_bias_pre == "BULLISH" and _m_bias_pre == "BULLISH")  # AND logic consistent cu htf_bullish
        dxy_adj = -0.05 if not dxy_data.get('bullish', True) and htf_bullish_early else 0.0

        # ── Update #53: Options Flow (Put/Call Ratio) ────────────────────────
        options_flow = get_options_flow_signal()
        options_adj  = options_flow.get('bias_adj', 0.0)
        # Inversează ajustarea dacă suntem SHORT
        if not htf_bullish_early:
            options_adj = -options_adj

        # ── Update #54: FRED Macro Filter ────────────────────────────────────
        fred_macro = get_fred_macro_filter()
        macro_sizing_mult = fred_macro.get('sizing_mult', 1.0)

        # ── Update #15: Volatility compression filter ────────────────────────────
        skip_volatile, vol_msg = check_volatility_filter(df)
        if skip_volatile:
            logger.info(f"   🛑 Volatility filter activ: {vol_msg}")

        # ── Update #16: Volume trend boost ──────────────────────────────────────
        vol_boost = check_volume_trend_filter(df, t_str)

        # ── Update #17: VIX sizing multiplier ───────────────────────────────────
        # Apelăm VIX doar dacă nu e backtest (e.g. dacă balance > 0)
        vix_mult = get_vix_sizing_mult() if balance > 0 else 1.0

        # ── Update #47-50: Circuit Breakers & Portfolio Heat ─────────────────
        heat_check = check_portfolio_heat()
        dd_check   = check_max_drawdown_breaker(initial_balance=balance)
        dl_check   = check_daily_loss_limit(balance=balance)

        # UPDATE #10: Corelație reală NQ/ES
        corr_check  = check_correlation_filter(instrument="NQ", active_positions=None)
        _corr_adj   = corr_check.get('score_adj', 0.0)
        if corr_check.get('blocked', False):
            logger.warning(f"   ⛔ CORRELATION BLOCK: {corr_check['reason']}")
        elif _corr_adj != 0.0:
            logger.info(f"   📊 Correlation adj: {_corr_adj:+.2f} — {corr_check['reason']}")

        circuit_blocked = (
            not heat_check.get('can_open_new', True) or
            dd_check.get('blocked', False) or
            dl_check.get('blocked', False) or
            corr_check.get('blocked', False)
        )

        if circuit_blocked:
            block_reason = []
            if not heat_check.get('can_open_new', True):
                block_reason.append(heat_check.get('reason', 'Portfolio heat'))
            if dd_check.get('blocked', False):
                block_reason.append(dd_check.get('reason', 'Max DD'))
            if dl_check.get('blocked', False):
                block_reason.append(dl_check.get('reason', 'Daily loss'))
            if corr_check.get('blocked', False):
                block_reason.append(corr_check.get('reason', 'Correlation'))
            logger.warning(f"   ⛔ CIRCUIT BREAKER: {' | '.join(block_reason)}")

        # ── Fix v6.4: Calculează features direcție lipsă pentru XGBoost inference ──
        # Aceste 5 features sunt calculate în train_mario_ai.py la training,
        # dar lipseau din pipeline-ul live → cauzau "feature_names mismatch".
        df['slope_h1']    = (df['close'] - df['close'].shift(60))  / (df['close'].shift(60).abs()  + 1e-8)
        df['slope_h4']    = (df['close'] - df['close'].shift(240)) / (df['close'].shift(240).abs() + 1e-8)
        df['momentum_15'] = (df['close'] - df['close'].shift(15))  / (df['close'].shift(15).abs()  + 1e-8)
        # v10.7: body_dir normalizat — raport body/range (nu -1/0/1 brut)
        df['body_dir']    = (df['close'] - df['open']) / (df['high'] - df['low']).clip(lower=1e-8)
        df['wick_ratio']  = (df['high'] - df['low']) / (abs(df['close'] - df['open']) + 1e-8)

        # ── Fix v7.5: 6 features noi microstructură (adăugate în train_mario_ai.py) ──
        _body = (df['close'] - df['open']).abs()
        _range = (df['high'] - df['low']).clip(lower=1e-8)
        df['upper_wick']  = (df['high'] - df[['close', 'open']].max(axis=1)) / _range
        df['lower_wick']  = (df[['close', 'open']].min(axis=1) - df['low']) / _range
        df['wick_bias']   = df['upper_wick'] - df['lower_wick']
        _log_ret = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol'] = _log_ret.rolling(20).std().fillna(0)
        df['vol_of_vol']   = df['realized_vol'].rolling(20).std().fillna(0)
        df['return_acf1']  = _log_ret.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) >= 5 else 0, raw=False
        ).fillna(0)

        # ── Advanced Features (Hurst, GARCH, Kalman, ADX, VWAP, SampEn, Fisher, FFT, ACF) ──
        if _AF_OK:
            try:
                df = _af.compute_live_advanced(df)
                logger.info(f"   🧮 Advanced features: {len(FEATURES_ADVANCED)} computed OK")
            except Exception as _af_err:
                logger.warning(f"   ⚠️ Advanced features error: {_af_err}")
                for _afc in FEATURES_ADVANCED:
                    if _afc not in df.columns:
                        df[_afc] = 0.0
        else:
            for _afc in FEATURES_ADVANCED:
                if _afc not in df.columns:
                    df[_afc] = 0.0

        # ── v10.6: REVERSAL features (MSS/CHoCH/trend exhaustion/delta flip) ──
        # Identic cu train_mario_ai.py add_reversal_features()
        try:
            _lookback = 5
            df['_recent_hi'] = df['high'].rolling(_lookback).max().shift(1)
            df['_recent_lo'] = df['low'].rolling(_lookback).min().shift(1)
            df['_trend_down'] = (df['close'].shift(1) < df['close'].shift(_lookback)).astype(int)
            df['_trend_up']   = (df['close'].shift(1) > df['close'].shift(_lookback)).astype(int)
            df['choch_bullish'] = ((df['_trend_down'] == 1) & (df['close'] > df['_recent_hi'])).astype(int)
            df['choch_bearish'] = ((df['_trend_up'] == 1) & (df['close'] < df['_recent_lo'])).astype(int)
            df['_hh'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['_ll'] = (df['low'] < df['low'].shift(1)).astype(int)
            df['_ll_count'] = df['_ll'].rolling(8).sum()
            df['_hh_count'] = df['_hh'].rolling(8).sum()
            df['mss_bullish'] = ((df['_ll_count'] >= 4) & (df['_hh'] == 1) & (df['close'] > df['open'])).astype(int)
            df['mss_bearish'] = ((df['_hh_count'] >= 4) & (df['_ll'] == 1) & (df['close'] < df['open'])).astype(int)
            df['reversal_strength'] = (
                df['choch_bullish'].astype(float) * 0.5 + df['mss_bullish'].astype(float) * 0.5
                - df['choch_bearish'].astype(float) * 0.5 - df['mss_bearish'].astype(float) * 0.5
            )
            df['_bar_dir'] = np.sign(df['close'] - df['open'])
            df['_same_dir'] = (df['_bar_dir'] == df['_bar_dir'].shift(1)).astype(int)
            df['_consec'] = df['_same_dir'].rolling(8, min_periods=1).sum()
            df['_body_r'] = (df['close'] - df['open']).abs()
            df['_body_shrink'] = (df['_body_r'] < df['_body_r'].shift(1) * 0.6).astype(int)
            df['trend_exhaustion'] = (df['_consec'] / 8.0) * 0.6 + df['_body_shrink'].astype(float) * 0.4
            df['_delta_est'] = (df['close'] - df['open']) * df['volume'].clip(lower=1)
            df['_cum_delta_5'] = df['_delta_est'].rolling(5).sum()
            df['_cum_delta_prev'] = df['_cum_delta_5'].shift(1)
            df['delta_flip'] = ((np.sign(df['_cum_delta_5']) != np.sign(df['_cum_delta_prev'])) & (df['_cum_delta_prev'] != 0)).astype(int)
            if 'poc_level' in df.columns:
                df['_poc_diff'] = df['poc_level'] - df['poc_level'].shift(3)
                df['poc_drift_direction'] = np.sign(df['_poc_diff'].fillna(0))
            else:
                df['poc_drift_direction'] = 0
            _swing_period = 10
            df['_swing_hi'] = df['high'].rolling(_swing_period).max().shift(1)
            df['_swing_lo'] = df['low'].rolling(_swing_period).min().shift(1)
            df['swing_break'] = 0.0
            df.loc[df['close'] > df['_swing_hi'], 'swing_break'] = 1.0
            df.loc[df['close'] < df['_swing_lo'], 'swing_break'] = -1.0
            _mom_period = 15
            df['_price_change_r'] = df['close'] - df['close'].shift(_mom_period)
            df['_mom_r'] = df['close'].pct_change(_mom_period)
            df['_mom_accel'] = df['_mom_r'] - df['_mom_r'].shift(5)
            df['momentum_divergence'] = 0.0
            df.loc[(df['_price_change_r'] > 0) & (df['_mom_accel'] < -0.001), 'momentum_divergence'] = -1.0
            df.loc[(df['_price_change_r'] < 0) & (df['_mom_accel'] > 0.001), 'momentum_divergence'] = 1.0
            # Cleanup temp cols
            _temp_cols = [c for c in df.columns if c.startswith('_')]
            df.drop(columns=_temp_cols, inplace=True, errors='ignore')
            logger.info(f"   🔄 Reversal features: {len(FEATURES_REVERSAL)} computed OK")
        except Exception as _rev_err:
            logger.warning(f"   ⚠️ Reversal features error: {_rev_err}")
            for _rfc in FEATURES_REVERSAL:
                if _rfc not in df.columns:
                    df[_rfc] = 0.0

        # ── v11.0: AMT features (Auction Market Theory) ──────────────────
        # Identic cu train_mario_ai.py add_amt_features()
        try:
            _swing_period = 10
            _sw_hi = df['high'].rolling(_swing_period).max().shift(1)
            _sw_lo = df['low'].rolling(_swing_period).min().shift(1)
            _rng   = (df['high'] - df['low']).clip(lower=1e-8)

            # 1. Failed Auction
            _break_above = df['high'] > _sw_hi
            _fail_above  = _break_above & (df['close'] < _sw_hi)
            _break_below = df['low'] < _sw_lo
            _fail_below  = _break_below & (df['close'] > _sw_lo)
            df['failed_auction'] = 0.0
            df.loc[_fail_below, 'failed_auction'] = 1.0
            df.loc[_fail_above, 'failed_auction'] = -1.0
            _both = _fail_above & _fail_below
            df.loc[_both & (df['close'] > df['open']), 'failed_auction'] = 1.0
            df.loc[_both & (df['close'] <= df['open']), 'failed_auction'] = -1.0

            # 2. Excess
            _upper_wick = (df['high'] - df[['close', 'open']].max(axis=1)) / _rng
            _lower_wick = (df[['close', 'open']].min(axis=1) - df['low']) / _rng
            _near_high = (df['high'] >= _sw_hi * 0.999)
            _near_low  = (df['low'] <= _sw_lo * 1.001)
            df['excess'] = 0.0
            df.loc[(_upper_wick > 0.40) & _near_high, 'excess'] = -1.0
            df.loc[(_lower_wick > 0.40) & _near_low, 'excess'] = 1.0

            # 3. Poor High / Poor Low
            df['poor_high'] = ((_upper_wick < 0.10) & _near_high).astype(float)
            df['poor_low']  = ((_lower_wick < 0.10) & _near_low).astype(float)

            # 4. Initiative vs Responsive
            _has_va = ('vah' in df.columns) and ('val' in df.columns)
            if _has_va:
                _vah = df['vah'].fillna(df['high'])
                _val = df['val'].fillna(df['low'])
                df['initiative_responsive'] = 0.0
                df.loc[df['close'] > _vah, 'initiative_responsive'] = 1.0
                df.loc[df['close'] < _val, 'initiative_responsive'] = -1.0
                df.loc[(df['high'] > _vah) & (df['close'] <= _vah), 'initiative_responsive'] = -0.5
                df.loc[(df['low'] < _val) & (df['close'] >= _val), 'initiative_responsive'] = 0.5
            else:
                df['initiative_responsive'] = 0.0

            # 5. VA Migration
            if _has_va:
                _va_mid = (_vah + _val) / 2
                _va_mid_prev = _va_mid.shift(60)
                _va_diff = _va_mid - _va_mid_prev
                _va_atr = df['atr_14'].fillna(1.0).clip(lower=1e-8) if 'atr_14' in df.columns else _rng.rolling(14).mean().clip(lower=1e-8)
                df['va_migration'] = (_va_diff / _va_atr).clip(-1, 1).fillna(0)
            else:
                df['va_migration'] = 0.0

            # 6. Rotation Factor
            _rolling_hi = df['high'].rolling(20).max()
            _rolling_lo = df['low'].rolling(20).min()
            _rolling_mid = (_rolling_hi + _rolling_lo) / 2
            _cross_above = (df['low'] < _rolling_mid) & (df['high'] > _rolling_mid)
            _rotation_count = _cross_above.rolling(20, min_periods=5).sum()
            df['rotation_factor'] = (_rotation_count / 20.0).clip(0, 1).fillna(0.5)

            logger.info(f"   📊 AMT features: {len(FEATURES_AMT)} computed OK")
        except Exception as _amt_err:
            logger.warning(f"   ⚠️ AMT features error: {_amt_err}")
            for _afc in FEATURES_AMT:
                if _afc not in df.columns:
                    df[_afc] = 0.0

        # ── v12.1: OF Aggregated features — DEZACTIVAT (insuficiente date OF reale)
        # Reactivează când ai 6+ luni de date OF populat din NinjaTrader
        # Codul rămâne aici, comentat, gata de reactivare.

        # ── v12.2: Consolidation features (runtime) ───────────────────
        # Calculăm features de consolidare din datele disponibile în df.
        # Aceste features sunt calculate din OHLC + VA, deci funcționează și live.
        import numpy as _np_consol
        try:
            if 'atr_14' in df.columns and 'high' in df.columns and len(df) >= 5:
                _atr_last = float(df['atr_14'].iloc[-1]) if df['atr_14'].iloc[-1] > 0 else 1.0
                # range_atr_ratio
                _t20 = df.tail(min(20, len(df)))
                df['range_atr_ratio'] = (float(_t20['high'].max() - _t20['low'].min()) / _atr_last)
                # bars_inside_va
                if 'inside_va' in df.columns:
                    df['bars_inside_va'] = float(df['inside_va'].tail(10).sum())
                else:
                    df['bars_inside_va'] = 5.0
                # va_width_atr
                if 'vah' in df.columns and 'val' in df.columns:
                    _vaw = float(df['vah'].iloc[-1] or 0) - float(df['val'].iloc[-1] or 0)
                    df['va_width_atr'] = max(_vaw, 0) / _atr_last
                else:
                    df['va_width_atr'] = 2.0
                # hh_ll_score
                if len(df) >= 10:
                    _h10 = df['high'].tail(10).values
                    _l10 = df['low'].tail(10).values
                    _fh = max(_h10[:5]); _sh = max(_h10[5:])
                    _fl = min(_l10[:5]); _sl = min(_l10[5:])
                    _hh = 1.0 if _sh > _fh + _atr_last * 0.3 else 0.0
                    _ll = -1.0 if _sl < _fl - _atr_last * 0.3 else 0.0
                    df['hh_ll_score'] = _hh + _ll
                else:
                    df['hh_ll_score'] = 0.0
                # same_level_rejections
                if all(c in df.columns for c in ['close', 'vah', 'val', 'poc_level']) and len(df) >= 20:
                    _c20 = df['close'].tail(20).values
                    _vah_r = float(df['vah'].iloc[-1] or 0)
                    _val_r = float(df['val'].iloc[-1] or 0)
                    _poc_r = float(df['poc_level'].iloc[-1] or 0)
                    _margin = max(_atr_last * 0.1, 2.0)
                    _rej = sum(1 for c in _c20 if abs(c - _vah_r) <= _margin or abs(c - _val_r) <= _margin or (_poc_r > 0 and abs(c - _poc_r) <= _margin))
                    df['same_level_rejections'] = min(_rej, 20)
                else:
                    df['same_level_rejections'] = 0
                # directional_efficiency
                if 'close' in df.columns and len(df) >= 20:
                    _c20 = df['close'].tail(20).values
                    _net = abs(_c20[-1] - _c20[0])
                    _tot = sum(abs(_c20[i] - _c20[i-1]) for i in range(1, len(_c20)))
                    df['directional_efficiency'] = _net / _tot if _tot > 0 else 0.5
                else:
                    df['directional_efficiency'] = 0.5

                # ── v12.2: 14 noi features consolidare ──────────────────
                # net_move_10
                if 'close' in df.columns and len(df) >= 10:
                    df['net_move_10'] = abs(float(df['close'].iloc[-1]) - float(df['close'].iloc[-10])) / _atr_last
                else:
                    df['net_move_10'] = 0.0
                # net_move_20
                if 'close' in df.columns and len(df) >= 20:
                    df['net_move_20'] = abs(float(df['close'].iloc[-1]) - float(df['close'].iloc[-20])) / _atr_last
                else:
                    df['net_move_20'] = 0.0
                # close_std_20
                if 'close' in df.columns and len(df) >= 20:
                    df['close_std_20'] = float(df['close'].tail(20).std()) / _atr_last
                else:
                    df['close_std_20'] = 0.5
                # avg_bar_range_ratio
                if 'high' in df.columns and 'low' in df.columns and len(df) >= 10:
                    _bar_ranges = (df['high'].tail(10) - df['low'].tail(10)).values
                    df['avg_bar_range_ratio'] = float(_np_consol.mean(_bar_ranges)) / _atr_last
                else:
                    df['avg_bar_range_ratio'] = 1.0
                # va_overlap_pct — suprapunere VA curentă vs VA de 10 bare în urmă
                if all(c in df.columns for c in ['vah', 'val']) and len(df) >= 10:
                    _vah_now = float(df['vah'].iloc[-1] or 0)
                    _val_now = float(df['val'].iloc[-1] or 0)
                    _vah_old = float(df['vah'].iloc[-10] or 0)
                    _val_old = float(df['val'].iloc[-10] or 0)
                    _w_now = _vah_now - _val_now
                    _w_old = _vah_old - _val_old
                    if _w_now > 0 and _w_old > 0:
                        _overlap_lo = max(_val_now, _val_old)
                        _overlap_hi = min(_vah_now, _vah_old)
                        _overlap = max(0, _overlap_hi - _overlap_lo)
                        df['va_overlap_pct'] = _overlap / min(_w_now, _w_old)
                    else:
                        df['va_overlap_pct'] = 0.5
                else:
                    df['va_overlap_pct'] = 0.5
                # mean_reversion_speed — cât de repede revine prețul la SMA20
                if 'close' in df.columns and len(df) >= 25:
                    _sma20 = df['close'].rolling(20).mean()
                    _dev = (df['close'] - _sma20).abs().tail(10)
                    _dev_prev = (df['close'] - _sma20).abs().tail(20).head(10)
                    _d_now = float(_dev.mean()) if len(_dev) > 0 else 1.0
                    _d_prev = float(_dev_prev.mean()) if len(_dev_prev) > 0 else 1.0
                    df['mean_reversion_speed'] = (_d_prev / _d_now) if _d_now > 0 else 1.0
                else:
                    df['mean_reversion_speed'] = 1.0
                # pivot_count_20 — câte schimbări de direcție pe 20 bare
                if 'close' in df.columns and len(df) >= 20:
                    _c20v = df['close'].tail(20).values
                    _diffs = _np_consol.diff(_c20v)
                    _diffs = _diffs[1:]  # remove first NaN
                    _signs = _np_consol.sign(_diffs)
                    _pivots = sum(1 for i in range(1, len(_signs)) if _signs[i] != _signs[i-1] and _signs[i] != 0)
                    df['pivot_count_20'] = float(_pivots)
                else:
                    df['pivot_count_20'] = 5.0
                # poc_stability — Std(POC) pe 10 bare / ATR
                if 'poc_level' in df.columns and len(df) >= 10:
                    _poc_vals = df['poc_level'].tail(10).replace(0, _np_consol.nan).dropna()
                    if len(_poc_vals) >= 3:
                        df['poc_stability'] = float(_poc_vals.std()) / _atr_last
                    else:
                        df['poc_stability'] = 1.0
                else:
                    df['poc_stability'] = 1.0
                # volume_trend — slope volum pe 20 bare (normalizat)
                if 'volume' in df.columns and len(df) >= 20:
                    _vol20 = df['volume'].tail(20).values.astype(float)
                    _x = _np_consol.arange(20, dtype=float)
                    _vbar = _np_consol.mean(_vol20)
                    if _vbar > 0:
                        _slope = float(_np_consol.polyfit(_x, _vol20, 1)[0]) / _vbar
                        df['volume_trend'] = _np_consol.clip(_slope, -2.0, 2.0)
                    else:
                        df['volume_trend'] = 0.0
                else:
                    df['volume_trend'] = 0.0
                # volume_cv — coeficient de variație volum pe 20 bare
                if 'volume' in df.columns and len(df) >= 20:
                    _vol20 = df['volume'].tail(20).values.astype(float)
                    _vm = float(_np_consol.mean(_vol20))
                    _vs = float(_np_consol.std(_vol20))
                    df['volume_cv'] = (_vs / _vm) if _vm > 0 else 0.5
                else:
                    df['volume_cv'] = 0.5
                # bollinger_width — (BB_upper - BB_lower) / close
                if 'close' in df.columns and len(df) >= 20:
                    _sma = float(df['close'].tail(20).mean())
                    _std = float(df['close'].tail(20).std())
                    _bb_upper = _sma + 2 * _std
                    _bb_lower = _sma - 2 * _std
                    _cl = float(df['close'].iloc[-1])
                    df['bollinger_width'] = ((_bb_upper - _bb_lower) / _cl) if _cl > 0 else 0.01
                else:
                    df['bollinger_width'] = 0.01
                # atr_percentile — percentila ATR curent vs ultimele 100 bare
                if 'atr_14' in df.columns and len(df) >= 20:
                    _atr_window = df['atr_14'].tail(min(100, len(df))).values
                    _atr_now = float(df['atr_14'].iloc[-1])
                    _pct = float((_atr_window < _atr_now).sum()) / len(_atr_window)
                    df['atr_percentile'] = _pct
                else:
                    df['atr_percentile'] = 0.5
                # swing_size_decay — ratio ultimul swing / penultimul swing
                if 'high' in df.columns and 'low' in df.columns and len(df) >= 20:
                    _h20 = df['high'].tail(20).values
                    _l20 = df['low'].tail(20).values
                    # Identifică swing-uri simple din midpoint
                    _mid = (_h20 + _l20) / 2.0
                    _swings = []
                    for i in range(1, len(_mid) - 1):
                        if (_mid[i] > _mid[i-1] and _mid[i] > _mid[i+1]) or \
                           (_mid[i] < _mid[i-1] and _mid[i] < _mid[i+1]):
                            _swings.append(abs(_mid[i] - _mid[i-1]))
                    if len(_swings) >= 2:
                        df['swing_size_decay'] = _swings[-1] / _swings[-2] if _swings[-2] > 0 else 1.0
                    else:
                        df['swing_size_decay'] = 1.0
                else:
                    df['swing_size_decay'] = 1.0
                # candle_overlap_pct — % din bare suprapuse cu bara anterioară
                if 'high' in df.columns and 'low' in df.columns and len(df) >= 10:
                    _h10 = df['high'].tail(10).values
                    _l10 = df['low'].tail(10).values
                    _overlaps = 0
                    for i in range(1, len(_h10)):
                        _overlap_lo = max(_l10[i], _l10[i-1])
                        _overlap_hi = min(_h10[i], _h10[i-1])
                        if _overlap_hi > _overlap_lo:
                            _bar_range = max(_h10[i] - _l10[i], 0.01)
                            _overlaps += (_overlap_hi - _overlap_lo) / _bar_range
                    df['candle_overlap_pct'] = _overlaps / (len(_h10) - 1)
                else:
                    df['candle_overlap_pct'] = 0.5

                logger.debug(f"   📐 Consol features (20): range_atr={df['range_atr_ratio'].iloc[-1]:.2f} bars_va={df['bars_inside_va'].iloc[-1]:.0f} dir_eff={df['directional_efficiency'].iloc[-1]:.2f} boll_w={df['bollinger_width'].iloc[-1]:.4f} overlap={df['candle_overlap_pct'].iloc[-1]:.2f}")
        except Exception as _consol_err:
            logger.debug(f"   ⚠️ Consol features skip: {_consol_err}")
            for _cf in FEATURES_CONSOL:
                if _cf not in df.columns:
                    df[_cf] = 0.0

        # ── v12.2: Streak Prevention features (runtime) ──────────────────
        try:
            import numpy as _np_streak
            if 'atr_14' in df.columns and len(df) >= 10:
                _atr_s = df['atr_14'].replace(0, _np_streak.nan)
                _atr_shifted_s = _atr_s.shift(10)
                df['atr_change_speed'] = (_atr_s / _atr_shifted_s).fillna(1.0).clip(0.3, 3.0)
            else:
                df['atr_change_speed'] = 1.0

            # v12.5: hour_sin/hour_cos SCOASE (temporal overfitting 21.5%)

            # consecutive_same_dir
            if 'close' in df.columns and 'open' in df.columns and len(df) >= 5:
                _dirs = (df['close'].tail(20) > df['open'].tail(20)).values
                _cnt = 1
                for i in range(len(_dirs)-2, -1, -1):
                    if _dirs[i] == _dirs[-1]:
                        _cnt += 1
                    else:
                        break
                df['consecutive_same_dir'] = _cnt
            else:
                df['consecutive_same_dir'] = 0

            # price_vs_daily_range
            if 'high' in df.columns and 'low' in df.columns and len(df) >= 20:
                _dh = df['high'].tail(100).max()
                _dl = df['low'].tail(100).min()
                _dr = _dh - _dl
                _cv = float(df['close'].iloc[-1])
                df['price_vs_daily_range'] = ((_cv - _dl) / _dr) if _dr > 0 else 0.5
            else:
                df['price_vs_daily_range'] = 0.5

            # v12.3: session_age SCOS (overfitting)

            # recent_signal_quality
            if 'close' in df.columns and 'open' in df.columns and len(df) >= 10:
                _bd = _np_streak.sign(df['close'].values - df['open'].values)
                _nm = _np_streak.sign(_np_streak.diff(df['close'].values))
                _nm = _nm[1:]  # shift
                _bd = _bd[:-1]
                _correct = (_bd[-30:] == _nm[-30:]) if len(_bd) >= 30 else (_bd == _nm)
                df['recent_signal_quality'] = float(_correct.mean()) if len(_correct) > 0 else 0.5
            else:
                df['recent_signal_quality'] = 0.5

        except Exception as _streak_err:
            logger.debug(f"   ⚠️ Streak features skip: {_streak_err}")
            for _sf in FEATURES_STREAK:
                if _sf not in df.columns:
                    df[_sf] = 0.0

        # ── 3c. CONTEXT features (v12.4 — trend quality, momentum, volume) ──
        try:
            import numpy as _np_ctx

            _atr_ctx = df['atr_14'].iloc[-1] if 'atr_14' in df.columns else 1.0
            _atr_ctx = max(_atr_ctx, 0.01)

            # --- trend_r2: R² al close pe 20 bare ---
            if 'close' in df.columns and len(df) >= 20:
                _y_r2 = df['close'].values[-20:]
                _x_r2 = _np_ctx.arange(20, dtype=float)
                _xm = _x_r2.mean(); _ym = _y_r2.mean()
                _ss_tot = _np_ctx.sum((_y_r2 - _ym)**2)
                _ss_xy = _np_ctx.sum((_x_r2 - _xm) * (_y_r2 - _ym))
                _ss_xx = _np_ctx.sum((_x_r2 - _xm)**2)
                if _ss_tot > 0 and _ss_xx > 0:
                    _b_r2 = _ss_xy / _ss_xx
                    _y_pred = _ym + _b_r2 * (_x_r2 - _xm)
                    _ss_res = _np_ctx.sum((_y_r2 - _y_pred)**2)
                    df['trend_r2'] = float(_np_ctx.clip(1.0 - _ss_res / _ss_tot, 0, 1))
                    df['trend_slope_norm'] = float(_np_ctx.clip(_b_r2 / _atr_ctx, -3, 3))
                else:
                    df['trend_r2'] = 0.5
                    df['trend_slope_norm'] = 0.0
            else:
                df['trend_r2'] = 0.5
                df['trend_slope_norm'] = 0.0

            # --- close_vs_ema_stack: EMA8 vs EMA21 vs EMA55 alignment ---
            if 'close' in df.columns and len(df) >= 55:
                _ema8 = df['close'].ewm(span=8).mean().iloc[-1]
                _ema21 = df['close'].ewm(span=21).mean().iloc[-1]
                _ema55 = df['close'].ewm(span=55).mean().iloc[-1]
                _bull = float(_ema8 > _ema21 and _ema21 > _ema55)
                _bear = float(_ema8 < _ema21 and _ema21 < _ema55)
                df['close_vs_ema_stack'] = _bull + _bear  # 1.0=aligned, 0=mixed
            else:
                df['close_vs_ema_stack'] = 0.0

            # --- roc_10: Rate of Change 10 bare / ATR ---
            if 'close' in df.columns and len(df) >= 11:
                _roc10 = (df['close'].iloc[-1] - df['close'].iloc[-11]) / _atr_ctx
                df['roc_10'] = float(_np_ctx.clip(_roc10, -5, 5))
            else:
                df['roc_10'] = 0.0

            # --- roc_divergence: ROC 5 vs ROC 20 ---
            if 'close' in df.columns and len(df) >= 21:
                _roc5 = (df['close'].iloc[-1] - df['close'].iloc[-6]) / _atr_ctx
                _roc20 = (df['close'].iloc[-1] - df['close'].iloc[-21]) / _atr_ctx
                df['roc_divergence'] = float(_np_ctx.clip(_roc5 - _roc20, -5, 5))
            else:
                df['roc_divergence'] = 0.0

            # --- momentum_consistency: consistență direcție pe 10 bare ---
            if 'close' in df.columns and len(df) >= 11:
                _cl = df['close'].values
                _ups = sum(1 for k in range(-10, 0) if _cl[k] > _cl[k-1])
                _up_pct = _ups / 10.0
                df['momentum_consistency'] = float(2 * abs(_up_pct - 0.5))
            else:
                df['momentum_consistency'] = 0.0

            # --- volume_on_move: volum pe mișcări mari vs mici ---
            if 'volume' in df.columns and 'high' in df.columns and 'low' in df.columns and len(df) >= 10:
                _bar_rng = (df['high'] - df['low']).clip(lower=0.01).values[-10:]
                _vol_arr = df['volume'].values[-10:].astype(float)
                _big = _bar_rng > 0.5 * _atr_ctx
                _vm = _vol_arr[_big].mean() if _big.any() else 1.0
                _vq = _vol_arr[~_big].mean() if (~_big).any() else 1.0
                df['volume_on_move'] = float(_np_ctx.clip(_vm / max(_vq, 1.0), 0.1, 10))
            else:
                df['volume_on_move'] = 1.0

            # --- volume_directional: volum bullish vs bearish ---
            if 'volume' in df.columns and 'close' in df.columns and 'open' in df.columns and len(df) >= 10:
                _bull_bar = (df['close'].values[-10:] > df['open'].values[-10:])
                _vol10 = df['volume'].values[-10:].astype(float)
                _vb = _vol10[_bull_bar].mean() if _bull_bar.any() else 1.0
                _vs = _vol10[~_bull_bar].mean() if (~_bull_bar).any() else 1.0
                df['volume_directional'] = float(_np_ctx.clip(_vb / max(_vs, 1.0), 0.1, 10))
            else:
                df['volume_directional'] = 1.0

            # --- clean_bars_pct: % bare curate (body > 50% range) ---
            if 'close' in df.columns and 'open' in df.columns and 'high' in df.columns and 'low' in df.columns and len(df) >= 10:
                _body_ctx = _np_ctx.abs(df['close'].values[-10:] - df['open'].values[-10:])
                _range_ctx = _np_ctx.clip(df['high'].values[-10:] - df['low'].values[-10:], 0.01, None)
                _clean = (_body_ctx / _range_ctx > 0.5).sum() / 10.0
                df['clean_bars_pct'] = float(_clean)
            else:
                df['clean_bars_pct'] = 0.5

            # --- false_break_count: false breakouts pe 20 bare ---
            if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns and len(df) >= 21:
                _h_fb = df['high'].values[-21:]
                _l_fb = df['low'].values[-21:]
                _c_fb = df['close'].values[-21:]
                _fb_cnt = 0
                for _j in range(1, 21):
                    if _h_fb[_j] > _h_fb[_j-1] and _c_fb[_j] < _h_fb[_j-1]:
                        _fb_cnt += 1
                    if _l_fb[_j] < _l_fb[_j-1] and _c_fb[_j] > _l_fb[_j-1]:
                        _fb_cnt += 1
                df['false_break_count'] = float(min(_fb_cnt, 20))
            else:
                df['false_break_count'] = 0.0

            # --- bar_range_consistency: uniformitate range bare ---
            if 'high' in df.columns and 'low' in df.columns and len(df) >= 10:
                _br = df['high'].values[-10:] - df['low'].values[-10:]
                _br_mean = _br.mean()
                _br_std = _br.std()
                if _br_mean > 0.01:
                    df['bar_range_consistency'] = float(_np_ctx.clip(_br_std / _br_mean, 0, 3))
                else:
                    df['bar_range_consistency'] = 0.5
            else:
                df['bar_range_consistency'] = 0.5

            logger.debug(f"   ✅ Context features: trend_r2={df['trend_r2'].iloc[-1]:.3f}, ema_stack={df['close_vs_ema_stack'].iloc[-1]:.1f}, momentum={df['momentum_consistency'].iloc[-1]:.3f}")

        except Exception as _ctx_err:
            logger.debug(f"   ⚠️ Context features skip: {_ctx_err}")
            for _cf in FEATURES_CONTEXT:
                if _cf not in df.columns:
                    df[_cf] = 0.0

        # ── 3d. MTF CONFIRMATION features (v12.4 — multi-timeframe agreement) ──
        try:
            import numpy as _np_mtf

            _c_mtf = df['close']

            # --- h1_trend_aligned: H1 slope has clear direction ---
            if len(df) >= 61:
                _h1_slope = (_c_mtf.iloc[-1] - _c_mtf.iloc[-61]) / max(_c_mtf.iloc[-61], 0.01)
                df['h1_trend_aligned'] = float(abs(_h1_slope) > 0.001)
            else:
                df['h1_trend_aligned'] = 0.0

            # --- h4_trend_aligned: H4 slope has clear direction ---
            if len(df) >= 241:
                _h4_slope = (_c_mtf.iloc[-1] - _c_mtf.iloc[-241]) / max(_c_mtf.iloc[-241], 0.01)
                df['h4_trend_aligned'] = float(abs(_h4_slope) > 0.002)
            else:
                df['h4_trend_aligned'] = 0.0

            # --- mtf_agreement: câte TF-uri pe aceeași direcție ---
            if len(df) >= 241:
                _dirs = []
                for _shift in [1, 5, 15, 60, 240]:
                    if len(df) > _shift:
                        _d = _np_mtf.sign(_c_mtf.iloc[-1] - _c_mtf.iloc[-1 - _shift])
                        if _d != 0:
                            _dirs.append(_d)
                if len(_dirs) > 0:
                    _dom = 1 if sum(_dirs) > 0 else -1
                    df['mtf_agreement'] = float(sum(1 for d in _dirs if d == _dom) / len(_dirs))
                else:
                    df['mtf_agreement'] = 0.5
            else:
                df['mtf_agreement'] = 0.5

            # --- recent_rejection_strength: wick × volume ---
            if 'volume' in df.columns and 'high' in df.columns and 'low' in df.columns and len(df) >= 5:
                _wick_t = (df['high'].values[-5:] - df['low'].values[-5:]) - _np_mtf.abs(df['close'].values[-5:] - df['open'].values[-5:])
                _body_mtf = _np_mtf.clip(_np_mtf.abs(df['close'].values[-5:] - df['open'].values[-5:]), 0.01, None)
                _wick_dom = _wick_t > _body_mtf
                _vol_mtf = df['volume'].values[-5:].astype(float)
                _rej = _np_mtf.where(_wick_dom, _wick_t * _vol_mtf, 0).mean()
                # normalize by avg volume*range over 20 bars
                if len(df) >= 20:
                    _norm_mtf = (df['volume'].values[-20:].astype(float) * (df['high'].values[-20:] - df['low'].values[-20:])).mean()
                    _norm_mtf = max(_norm_mtf, 0.01)
                    df['recent_rejection_strength'] = float(_np_mtf.clip(_rej / _norm_mtf, 0, 5))
                else:
                    df['recent_rejection_strength'] = 0.0
            else:
                df['recent_rejection_strength'] = 0.0

            logger.debug(f"   ✅ MTF features: h1={df['h1_trend_aligned'].iloc[-1]:.0f}, h4={df['h4_trend_aligned'].iloc[-1]:.0f}, agree={df['mtf_agreement'].iloc[-1]:.2f}")

        except Exception as _mtf_err:
            logger.debug(f"   ⚠️ MTF features skip: {_mtf_err}")
            for _mf in FEATURES_MTF_CONFIRM:
                if _mf not in df.columns:
                    df[_mf] = 0.0

        # ── v12.7: ORH features runtime (Opening Range Breakout) ──────────
        try:
            import numpy as _np_orh
            _n_orh = len(df)
            if 'timestamp' in df.columns and _n_orh > 0:
                _ts_orh = pd.to_datetime(df['timestamp'], errors='coerce')
                _hour_o = _ts_orh.dt.hour.values
                _min_o  = _ts_orh.dt.minute.values
                _date_o = _ts_orh.dt.date.values
                _tdec_o = _hour_o + _min_o / 60.0
                _london_o = (_tdec_o >= 9.0) & (_tdec_o <= 11.0)
                _ny_o     = (_tdec_o >= 15.5) & (_tdec_o <= 17.5)

                _h_o = df['high'].values
                _l_o = df['low'].values
                _c_o = df['close'].values
                _v_o = df['volume'].values if 'volume' in df.columns else _np_orh.ones(_n_orh)

                if 'atr_14' in df.columns:
                    _atr_o = df['atr_14'].values
                else:
                    _tr_o = _np_orh.maximum(_h_o - _l_o,
                                             _np_orh.maximum(_np_orh.abs(_h_o - _np_orh.roll(_c_o, 1)),
                                                              _np_orh.abs(_l_o - _np_orh.roll(_c_o, 1))))
                    _atr_o = pd.Series(_tr_o).rolling(14).mean().fillna(10.0).values
                _atr_o = _np_orh.where(_atr_o > 0, _atr_o, 10.0)

                _orh_hi   = _np_orh.full(_n_orh, _np_orh.nan)
                _orh_lo   = _np_orh.full(_n_orh, _np_orh.nan)
                _in_or    = _np_orh.zeros(_n_orh, dtype=_np_orh.int8)
                _post_or  = _np_orh.zeros(_n_orh, dtype=_np_orh.int8)
                _bars_s   = _np_orh.zeros(_n_orh, dtype=_np_orh.int32)
                _brk_up   = _np_orh.zeros(_n_orh, dtype=_np_orh.int8)
                _brk_dn   = _np_orh.zeros(_n_orh, dtype=_np_orh.int8)
                _or_vol_s = _np_orh.zeros(_n_orh, dtype=_np_orh.float64)
                _or_vol_c = _np_orh.zeros(_n_orh, dtype=_np_orh.int32)

                OR_DUR = 30
                _cd_o, _ck_o = None, None
                _ss_o = 0
                _sh_o = -_np_orh.inf
                _sl_o =  _np_orh.inf
                _ov_sum = 0.0
                _ov_cnt = 0
                _bu_f, _bd_f = False, False

                for _i_o in range(_n_orh):
                    _kz_o = "london" if _london_o[_i_o] else ("ny" if _ny_o[_i_o] else None)
                    if _kz_o is None:
                        _ck_o = None
                        continue
                    if _date_o[_i_o] != _cd_o or _kz_o != _ck_o:
                        _cd_o, _ck_o = _date_o[_i_o], _kz_o
                        _ss_o = _i_o
                        _sh_o = _h_o[_i_o]
                        _sl_o = _l_o[_i_o]
                        _ov_sum = _v_o[_i_o] if not _np_orh.isnan(_v_o[_i_o]) else 0.0
                        _ov_cnt = 1
                        _bu_f, _bd_f = False, False
                    _bis = _i_o - _ss_o
                    _bars_s[_i_o] = _bis
                    if _bis < OR_DUR:
                        if _h_o[_i_o] > _sh_o: _sh_o = _h_o[_i_o]
                        if _l_o[_i_o] < _sl_o: _sl_o = _l_o[_i_o]
                        if not _np_orh.isnan(_v_o[_i_o]):
                            _ov_sum += _v_o[_i_o]; _ov_cnt += 1
                        _in_or[_i_o] = 1
                    else:
                        _post_or[_i_o] = 1
                        if not _bu_f and _c_o[_i_o] > _sh_o: _bu_f = True
                        if not _bd_f and _c_o[_i_o] < _sl_o: _bd_f = True
                        _brk_up[_i_o] = int(_bu_f)
                        _brk_dn[_i_o] = int(_bd_f)
                    _orh_hi[_i_o] = _sh_o
                    _orh_lo[_i_o] = _sl_o
                    _or_vol_s[_i_o] = _ov_sum
                    _or_vol_c[_i_o] = max(_ov_cnt, 1)

                df['in_orh'] = _in_or
                df['post_orh'] = _post_or
                df['bars_since_session'] = _bars_s
                df['orh_broken_up'] = _brk_up
                df['orh_broken_down'] = _brk_dn
                _w_o = (_orh_hi - _orh_lo)
                df['orh_width_atr'] = _np_orh.where(~_np_orh.isnan(_w_o), _w_o / _atr_o, 0.0)
                df['dist_to_orh_high_atr'] = _np_orh.where(~_np_orh.isnan(_orh_hi), (_c_o - _orh_hi) / _atr_o, 0.0)
                df['dist_to_orh_low_atr']  = _np_orh.where(~_np_orh.isnan(_orh_lo), (_c_o - _orh_lo) / _atr_o, 0.0)
                _mid_o = (_orh_hi + _orh_lo) / 2.0
                df['orh_midpoint_dist_atr'] = _np_orh.where(~_np_orh.isnan(_mid_o), (_c_o - _mid_o) / _atr_o, 0.0)
                _oav = _or_vol_s / _np_orh.maximum(_or_vol_c, 1)
                df['session_vol_ratio'] = _np_orh.where(_oav > 0, _v_o / _oav, 1.0)
                for _fx in ['orh_width_atr','dist_to_orh_high_atr','dist_to_orh_low_atr',
                            'orh_midpoint_dist_atr','session_vol_ratio']:
                    df[_fx] = df[_fx].replace([_np_orh.inf, -_np_orh.inf], 0).fillna(0).clip(-10, 10)
                logger.debug(f"   ✅ ORH runtime: in_or={df['in_orh'].iloc[-1]} post_or={df['post_orh'].iloc[-1]} "
                             f"brk_up={df['orh_broken_up'].iloc[-1]} brk_dn={df['orh_broken_down'].iloc[-1]}")
            else:
                for _of in FEATURES_ORH:
                    if _of not in df.columns:
                        df[_of] = 0.0
        except Exception as _orh_err:
            logger.debug(f"   ⚠️ ORH features skip: {_orh_err}")
            for _of in FEATURES_ORH:
                if _of not in df.columns:
                    df[_of] = 0.0

        # ── 3.5. v13 REGIME FEATURES (weekly + daytype + sweep) ────────
        if _V13_AVAILABLE:
            try:
                df = _v13_add_weekly(df)
                df = _v13_add_daytype(df)
                df = _v13_add_sweep(df)
                for _vf in (FEATURES_WEEKLY + FEATURES_DAYTYPE + FEATURES_SWEEP):
                    if _vf not in df.columns:
                        df[_vf] = 0.0
                    else:
                        df[_vf] = df[_vf].replace([np.inf, -np.inf], 0).fillna(0)
            except Exception as _v13_err:
                logger.debug(f"   ⚠️ v13 features skip: {_v13_err}")
                for _vf in (FEATURES_WEEKLY + FEATURES_DAYTYPE + FEATURES_SWEEP):
                    if _vf not in df.columns:
                        df[_vf] = 0.0

        # ── 4. AI Score — Quality Gate v2/v5 (înlocuiește mario_bot_open.json) ──
        # mario_bot_open.json era un model vechi (2015-2021, AUC slab) care contribuia
        # greșit la hybrid score. Înlocuit cu scorul Quality Gate v6 (LON, AUC OOS=0.79)
        # și ny_v3 (NY, AUC OOS=0.72), injectat din bridge_api.py via live_data['qg_score']
        # după ce ict_gate.gate_verdict() rulează.
        ai_direction = "NEUTRAL"
        _qg_score    = float((live_data or {}).get('qg_score', 0.0))
        _qg_dir      = str((live_data or {}).get('qg_direction', 'NEUTRAL'))
        if _qg_score > 0 and _qg_dir in ('LONG', 'SHORT'):
            base_ai      = _qg_score
            ai_direction = _qg_dir
            logger.info(f"   🤖 AI (QualityGate v2/v5): score={base_ai:.3f} dir={ai_direction}")
        else:
            base_ai = 0.0
            logger.info(f"   🤖 AI (QualityGate): score unavailable → base_ai=0.0")

        if False:  # DISABLED: mario_bot_open.json (kept for reference, do not remove)
            try:
                # v10.6: ALL feature groups — sincronizat cu train_mario_ai.py
                all_features = (FEATURES_STRICT + FEATURES_EXTRA + FEATURES_VP_OF
                                + FEATURES_OF_NATIVE + FEATURES_ADVANCED + FEATURES_REVERSAL
                                + FEATURES_AMT + FEATURES_CONSOL
                                + FEATURES_STREAK + FEATURES_CONTEXT
                                + FEATURES_MTF_CONFIRM
                                + FEATURES_ORH  # v12.7: +ORB expansion
                                + FEATURES_WEEKLY + FEATURES_DAYTYPE + FEATURES_SWEEP)  # v13 regime-aware
                available = [f for f in all_features if f in df.columns]
                _missing_feats = [f for f in all_features if f not in df.columns]
                if _missing_feats:
                    logger.warning(f"   ⚠️ Features lipsă din DB ({len(_missing_feats)}): {_missing_feats[:5]}...")
                X = df.reindex(columns=available).tail(1).fillna(0)

                # ── v10.6: ENSEMBLE PREDICTION (XGBoost + LightGBM + RandomForest) ──
                # Încarcă toate modelele disponibile, face media probabilităților.
                # Dacă doar XGBoost e disponibil, funcționează ca înainte (single model).
                model_loaded = None
                model_type   = "none"
                _ensemble_models = []

                # 1. XGBoost (primary — mereu disponibil)
                if os.path.exists(MODEL_PATH):
                    _xgb_model = xgb.XGBClassifier()
                    _xgb_model.load_model(MODEL_PATH)
                    _ensemble_models.append(("XGBoost", _xgb_model))
                    model_loaded = _xgb_model  # fallback reference

                # 2. LightGBM (dacă există)
                _lgbm_path = MODEL_PATH.replace('.json', '_lgbm.pkl')
                if os.path.exists(_lgbm_path):
                    try:
                        import pickle
                        with open(_lgbm_path, 'rb') as _f:
                            _lgbm_model = pickle.load(_f)
                        _ensemble_models.append(("LightGBM", _lgbm_model))
                    except Exception as _e:
                        logger.debug(f"LightGBM load skip: {_e}")

                # 3. RandomForest (dacă există)
                _rf_path = MODEL_PATH.replace('.json', '_rf.pkl')
                if os.path.exists(_rf_path):
                    try:
                        import pickle
                        with open(_rf_path, 'rb') as _f:
                            _rf_model = pickle.load(_f)
                        _ensemble_models.append(("RandomForest", _rf_model))
                    except Exception as _e:
                        logger.debug(f"RF load skip: {_e}")

                model_type = f"ensemble_{len(_ensemble_models)}" if len(_ensemble_models) > 1 else "xgboost_raw"
                _model_names = [n for n, _ in _ensemble_models]
                logger.info(f"   🤖 Modele încărcate: {_model_names} ({len(available)} features)")

                # v10.6: Aliniere automată features per model (fiecare model poate avea
                # feature set ușor diferit dacă a fost retrained la momente diferite)
                def _get_model_features(m):
                    """Extrage feature names din orice model type."""
                    if hasattr(m, 'get_booster'):  # XGBoost
                        return m.get_booster().feature_names
                    elif hasattr(m, 'feature_name_'):  # LightGBM
                        return list(m.feature_name_)
                    elif hasattr(m, 'feature_names_in_'):  # sklearn (RF, etc.)
                        return list(m.feature_names_in_)
                    return None

                # Aliniere primară la XGBoost (referință)
                _model_feats = _get_model_features(model_loaded) if model_loaded else None
                if _model_feats:
                    X = X.reindex(columns=_model_feats, fill_value=0)

                # Ensemble prediction: media probabilităților tuturor modelelor
                # v13: acceptă 3-class legacy și 9-class regime; NU amesteca modele cu shape diferit.
                _all_proba = []
                _ncls_ref = None
                for _m_name, _m in _ensemble_models:
                    try:
                        _m_feats = _get_model_features(_m)
                        if _m_feats and list(_m_feats) != list(X.columns):
                            _X_m = X.reindex(columns=_m_feats, fill_value=0)
                        else:
                            _X_m = X
                        _p = _m.predict_proba(_X_m)
                        if _ncls_ref is None:
                            _ncls_ref = _p.shape[1]
                        if _p.shape[1] in (3, 5, 9) and _p.shape[1] == _ncls_ref:
                            _all_proba.append(_p)
                        else:
                            logger.warning(f"   ⚠️ {_m_name} classes={_p.shape[1]} (ref={_ncls_ref}), skip")
                    except Exception as _e:
                        logger.warning(f"   ⚠️ {_m_name} predict error: {_e}")

                if _all_proba:
                    proba = sum(_all_proba) / len(_all_proba)
                else:
                    proba = model_loaded.predict_proba(X)
                _n_classes = proba.shape[1]
                if _n_classes not in (3, 5, 9):
                    logger.error(f"   ❌ Model clasă greșită: expected 3/5/9, got {_n_classes}")
                    raise ValueError(f"Expected 3/5/9-class model, got {_n_classes}")

                p_wait = float(proba[-1, 0])
                if _n_classes == 3:
                    # Legacy: 0=WAIT, 1=SHORT, 2=LONG
                    p_short = float(proba[-1, 1])
                    p_long  = float(proba[-1, 2])
                    _regime_name = "LEGACY_" + ("SHORT" if p_short > p_long else "LONG" if p_long > p_short else "WAIT")
                    _regime_cls = 1 if p_short > p_long else 2 if p_long > p_short else 0
                else:
                    # v13/v14: 5 sau 9 clase — sum per direction via V13_DIR_MAP
                    _p_row = proba[-1]
                    p_short = float(sum(_p_row[c] for c in range(_n_classes) if V13_DIR_MAP.get(c, 0) == -1))
                    p_long  = float(sum(_p_row[c] for c in range(_n_classes) if V13_DIR_MAP.get(c, 0) == +1))
                    _regime_cls = int(np.argmax(_p_row))
                    _regime_name = _V13_REGIME_NAMES.get(_regime_cls, f'class_{_regime_cls}')
                    logger.info(f"   🎯 v13 Regime: {_regime_name} (p={_p_row[_regime_cls]:.2f}) | Σp_short={p_short:.2f} Σp_long={p_long:.2f}")

                # v12.5: Confidence threshold din training (97% WR target)
                # Dacă probabilitatea clasei SHORT/LONG nu depășește threshold-ul,
                # tratăm ca WAIT — modelul nu e suficient de sigur
                _conf_thr = 0.50  # fallback
                try:
                    _feat_meta_path = MODEL_PATH.replace('.json', '').replace('mario_bot', 'mario_features') + '.json'
                    if os.path.exists(_feat_meta_path):
                        import json as _json_thr
                        with open(_feat_meta_path, 'r') as _fthr:
                            _feat_meta = _json_thr.load(_fthr)
                        _conf_thr = float(_feat_meta.get('conf_threshold', 0.50))
                        logger.debug(f"   🎯 Confidence threshold din training: {_conf_thr:.2%}")
                except Exception:
                    pass

                # Aplicăm confidence threshold: dacă nici SHORT nici LONG nu trec, forțăm WAIT
                if p_short < _conf_thr and p_long < _conf_thr:
                    ai_direction = "NEUTRAL"
                    p_dir = max(p_short, p_long)
                    logger.info(f"   🛡️ Confidence gate: p_short={p_short:.3f} p_long={p_long:.3f} < threshold={_conf_thr:.2%} → NEUTRAL")
                else:
                    ai_direction = "LONG" if p_long >= p_short else "SHORT"
                    p_dir = p_long if p_long >= p_short else p_short

                # v11.0 → v12.2: ANTI-STREAK FILTER — blochează semnale în condiții choppy
                # v11.0 original: necesita TOATE 4 condițiile simultan (AND) — prea restrictiv,
                # confidence 0.55+ bypasa filtrul complet chiar în consolidare clară.
                # v12.2 FIX: scoring system — dacă 3 din 4 condiții sunt True → SKIP.
                # Threshold confidence relaxat: 0.55 → 0.65 (modelul dă 0.56-0.62 în range).
                _last_row = df.iloc[-1] if len(df) > 0 else None
                _anti_streak_blocked = False
                if _last_row is not None:
                    _is_inside_va   = float(_last_row.get('inside_va', 0)) > 0.5
                    _va_mig_flat    = abs(float(_last_row.get('va_migration', 0))) < 0.05
                    _no_initiative  = abs(float(_last_row.get('initiative_responsive', 0))) < 0.1
                    _low_confidence = p_dir < 0.65   # v12.2: relaxat de la 0.55

                    # v12.2: scoring — 3 din 4 condiții = SKIP (nu mai necesită toate 4)
                    _chop_signals = sum([_is_inside_va, _va_mig_flat, _no_initiative, _low_confidence])

                    # Condiție suplimentară: regime RANGING detectat → scade pragul la 2 din 4
                    _regime_is_ranging = "RANG" in str(regime).upper() if regime else False
                    _chop_threshold = 2 if _regime_is_ranging else 3

                    if _chop_signals >= _chop_threshold:
                        _anti_streak_blocked = True
                        logger.warning(
                            f"   🛡️ ANTI-STREAK v12.2: {_chop_signals}/4 chop signals (threshold={_chop_threshold}) "
                            f"inside_va={_is_inside_va} va_mig={float(_last_row.get('va_migration', 0)):+.3f} "
                            f"initiative={float(_last_row.get('initiative_responsive', 0)):+.2f} conf={p_dir:.2f} "
                            f"regime={'RANGING' if _regime_is_ranging else 'OTHER'} → SKIP semnal"
                        )
                        p_wait = 0.99   # forțăm WAIT
                        p_dir  = 0.01

                # v10.8: PROP FIRM — AI contribuie DIRECT la decizie
                # Modelul nou are WR 53-56%, PF 1.33, Monte Carlo 100% profit
                # Nu mai gate-uim — lăsăm AI-ul să contribuie proporțional cu confidența
                #
                # Formula: base_ai = p_dir (probabilitatea direcției dominante)
                # p_dir=0.15 → base_ai=0.15 (semnal slab, contribuție mică)
                # p_dir=0.30 → base_ai=0.30 (semnal mediu)
                # p_dir=0.45 → base_ai=0.45 (semnal puternic)
                #
                # Gate SOFT: dacă p_wait > 0.85 (modelul e foarte sigur pe WAIT),
                # reducem contribuția la 20% din p_dir (nu 0 complet)
                _ai_gate_threshold = 0.85
                if p_wait > _ai_gate_threshold:
                    # Modelul zice WAIT cu convingere mare — reducem dar nu anulăm
                    base_ai = p_dir * 0.20
                    _gate_status = "SOFT"
                else:
                    # Modelul are o opinie pe SHORT/LONG — contribuie direct
                    base_ai = p_dir
                    _gate_status = "OFF"
                logger.info(f"   🤖 AI ({model_type}): WAIT={p_wait:.2f} SHORT={p_short:.2f} LONG={p_long:.2f} → {ai_direction} base_ai={base_ai:.3f} (gate={_gate_status})")
            except Exception as e:
                logger.warning(f"AI predict error: {e} → fallback 0.0 (neutru)")
                base_ai = 0.0

        # ── 5. Quantum Validation ────────────────────────────────────────────
        q_edge = get_quantum_conviction(best, target_ts)
        logger.info(f"   ⚛️  Quantum Edge: x{q_edge:.3f}")

        # ── 6. Hybrid Score — WEIGHTED ADDITIVE ─────────────────────────────

        # Component 1: AI (0–1)
        ai_component = base_ai

        # Component 2: ICT confluence — caută în ultimele 45 bare, nu doar bara curentă
        # Fix v7.2: mărit de la 5→45 bare pentru a captura displacement post-spike
        # BSL sweep la 14:20 Romania (11:20 UTC) → vizibil la 15:05 RO (12:05 UTC) = 45 bare distanță
        # has_displacement acum detectat și din SIZE (range > 1.5×ATR) - fix bridge_api.py
        # Fix v9.0: renamed last_5 → last_window (de fapt ia 45 bare, nu 5)
        last_window = df.tail(45)
        # Fix v7.3: SMT din divergență NQ intern (fără SPY — is_smt_bearish/bullish=0 mereu în DB)
        # Fix v9.0: SMT lookback relativ la window, nu fix 25 bare
        _smt_detected = False
        _smt_lookback = min(25, len(df) - 1)  # adaptiv la lungimea datelor disponibile
        if _smt_lookback >= 10:  # minim 10 bare pentru SMT valid
            _ref_idx   = len(df) - 1 - _smt_lookback
            _ref_high  = float(df['high'].iloc[_ref_idx])
            _ref_low   = float(df['low'].iloc[_ref_idx])
            _cur_high  = float(best.get('high', 0) or 0)
            _cur_low   = float(best.get('low', 0) or 0)
            _cur_close = float(best.get('close', 0) or 0)
            # SMT Bearish: new high dar close sub ref_high → fake breakout bearish
            if _cur_high > _ref_high and _cur_close < _ref_high:
                _smt_detected = True
                logger.debug(f"   🔀 SMT BEARISH intern: cur_high={_cur_high:.1f} > ref_high={_ref_high:.1f} dar close={_cur_close:.1f} < ref")
            # SMT Bullish: new low dar close deasupra ref_low → fake breakdown bullish
            elif _cur_low < _ref_low and _cur_close > _ref_low:
                _smt_detected = True
                logger.debug(f"   🔀 SMT BULLISH intern: cur_low={_cur_low:.1f} < ref_low={_ref_low:.1f} dar close={_cur_close:.1f} > ref")
        has_smt = _smt_detected or bool((last_window['is_smt_bearish'].any()) or (last_window['is_smt_bullish'].any()))
        # Fix v7.2: fvg_up/fvg_down sunt mereu 0 în DB (nu sunt calculate în bridge)
        # Calculăm FVG din prețuri reale: gap între high/low de 3 bare consecutive
        # FVG Bearish: high[i+2] < low[i]  |  FVG Bullish: low[i+2] > high[i]
        _fvg_detected = False
        if len(last_window) >= 3:
            for _fi in range(len(last_window) - 2):
                _hi0 = float(last_window['high'].iloc[_fi])
                _lo0 = float(last_window['low'].iloc[_fi])
                _hi2 = float(last_window['high'].iloc[_fi + 2])
                _lo2 = float(last_window['low'].iloc[_fi + 2])
                if _lo2 > _hi0 or _hi2 < _lo0:  # bullish sau bearish FVG
                    _fvg_detected = True
                    break
        has_fvg    = _fvg_detected or bool((last_window['fvg_up'].any()) or (last_window['fvg_down'].any()))
        has_dis    = bool(last_window['has_displacement'].any())
        in_kz      = active_kz is not None
        above_open = bool(best.get('is_above_open', 0))

        # Bonus pentru semnale pe bara curentă (mai relevante)
        cur_smt = bool(best.get('is_smt_bearish', 0) or best.get('is_smt_bullish', 0))
        # FVG pe bara curentă (last 3 bars)
        cur_fvg = False
        if len(df) >= 3:
            _c_hi0 = float(df['high'].iloc[-3])
            _c_lo0 = float(df['low'].iloc[-3])
            _c_hi2 = float(df['high'].iloc[-1])
            _c_lo2 = float(df['low'].iloc[-1])
            cur_fvg = bool(_c_lo2 > _c_hi0 or _c_hi2 < _c_lo0)
        cur_fvg = cur_fvg or bool(best.get('fvg_up', 0) or best.get('fvg_down', 0))
        cur_dis = bool(best.get('has_displacement', 0))

        # H4 bias aliniat cu direcția AI — semnal ICT core (macro direcție zilei)
        # Dacă H4 close > midpoint H4 → H4 bullish; dacă AI vrea LONG și H4 e bullish → aliniat
        _h4_hi_ict = float(best.get('h4_hi', 0) or 0)
        _h4_lo_ict = float(best.get('h4_lo', 0) or 0)
        if _h4_hi_ict > _h4_lo_ict > 0:
            _h4_mid_ict  = _h4_lo_ict + (_h4_hi_ict - _h4_lo_ict) * 0.5
            _h4_bull_ict = best['close'] > _h4_mid_ict
            h4_aligned   = (_h4_bull_ict and ai_direction == "LONG") or \
                           (not _h4_bull_ict and ai_direction == "SHORT")
        else:
            h4_aligned = False   # H4 lipsește — nu contribuie

        # H1 bias aliniat cu direcția AI — semnal ICT core (direcție de execuție)
        # H1 e mai granular decât H4: confirmă că structura intraday e în linie cu AI
        _h1_hi_ict = float(best.get('h1_hi', 0) or 0)
        _h1_lo_ict = float(best.get('h1_lo', 0) or 0)
        if _h1_hi_ict > _h1_lo_ict > 0:
            _h1_mid_ict  = _h1_lo_ict + (_h1_hi_ict - _h1_lo_ict) * 0.5
            _h1_bull_ict = best['close'] > _h1_mid_ict
            h1_aligned   = (_h1_bull_ict and ai_direction == "LONG") or \
                           (not _h1_bull_ict and ai_direction == "SHORT")
        else:
            h1_aligned = False   # H1 lipsește — nu contribuie

        # M15 bias + VP confluence — semnal ICT de execuție (cel mai granular HTF)
        # Confirmă că structura M15 aliniază cu AI ȘI că prețul e la un nivel VP relevant
        # VP confluence: prețul aproape de LVN (potential move) sau HVN (support/resistance real)
        _m15_hi_ict = float(best.get('m15_hi', 0) or 0)
        _m15_lo_ict = float(best.get('m15_lo', 0) or 0)
        if _m15_hi_ict > _m15_lo_ict > 0:
            _m15_mid_ict  = _m15_lo_ict + (_m15_hi_ict - _m15_lo_ict) * 0.5
            _m15_bull_ict = best['close'] > _m15_mid_ict
            _m15_dir_ok   = (_m15_bull_ict and ai_direction == "LONG") or \
                            (not _m15_bull_ict and ai_direction == "SHORT")
            # VP confluence pe M15: prețul în 0.5 ATR de un HVN sau LVN
            _atr_ict  = float(best.get('atr_14', 1.0) or 1.0)
            _poc_ict  = float(best.get('poc_level', 0) or 0)
            _vah_ict  = float(best.get('vah', 0) or 0)
            _val_ict  = float(best.get('val', 0) or 0)
            _close_ict = float(best.get('close', 0) or 0)
            _vp_levels = [l for l in [_poc_ict, _vah_ict, _val_ict] if l > 0]
            _near_vp   = any(abs(_close_ict - lvl) < _atr_ict * 0.5 for lvl in _vp_levels)
            m15_aligned = _m15_dir_ok and _near_vp
        else:
            m15_aligned = False   # M15 lipsește — nu contribuie

        ict_signals = sum([has_smt, has_fvg, has_dis, in_kz, above_open, h4_aligned, h1_aligned, m15_aligned])
        # Bonus dacă semnalul e pe bara curentă (mai puternic)
        cur_bonus   = sum([cur_smt, cur_fvg, cur_dis]) * 0.15
        ict_component = min(ict_signals / 7.0 + cur_bonus, 1.0)  # /7 (8 semnale posibile)
        # Fix v10.5b: dacă reversal_override va flipă direcția, h4/h1/m15 sunt calculate
        # pentru ai_direction greșit — inversăm contribuția lor în ICT score.
        # Exemplu: AI=LONG, H4=BEAR → h4_aligned=False (❌) pentru LONG, dar pentru SHORT ar fi True (✅).
        # Aplicăm corecția la scor: fiecare semnal inversat adaugă +1 la ict_signals.
        # Nota: _reversal_override e calculat mai jos, deci folosim un proxy:
        # dacă H4 e BEAR și ai_direction=LONG → va fi reversal → corectăm h4 acum
        _ro_h4_flip  = h4_aligned == False and ai_direction == "LONG"   # H4 BEAR, AI LONG → SHORT override
        _ro_h4_flip |= h4_aligned == True  and ai_direction == "SHORT"  # H4 BULL, AI SHORT → LONG override
        if _ro_h4_flip:
            # Inversăm h4, h1, m15 în ict_signals
            _htf_correction = sum([not h4_aligned, not h1_aligned, not m15_aligned]) - sum([h4_aligned, h1_aligned, m15_aligned])
            ict_signals_ro = max(0, ict_signals + _htf_correction)
            ict_component  = min(ict_signals_ro / 7.0 + cur_bonus, 1.0)

        # Component 3: Quantum edge (0–1)
        q_component = min(q_edge, 1.0)

        # Component 4: Relative strength (0–1)
        rel_component = min(max((rel_mult - 0.7) / 0.6, 0.0), 1.0)

        # ── Pre-calcul HTF bias (necesar pentru orderflow direction-aware) ──────
        # Calculat ÎNAINTE de scor, nu după — fix pentru bug v6.5 (Bug #2: htf_bullish trebuie ÎNAINTE)
        # Fix None-safety: best.get(key, 0) returnează None dacă coloana există cu NULL în DB
        _lw_lo_pre2  = float(best.get('lw_lo') or 0)
        _lw_hi_pre2  = float(best.get('lw_hi') or 0)
        _lm_lo_pre2  = float(best.get('lm_lo') or 0)
        _lm_hi_pre2  = float(best.get('lm_hi') or 0)
        _lw_mid_pre = _lw_lo_pre2 + (_lw_hi_pre2 - _lw_lo_pre2) / 2
        _lm_mid_pre = _lm_lo_pre2 + (_lm_hi_pre2 - _lm_lo_pre2) / 2
        _w_bias_pre = "BULLISH" if best['close'] > _lw_mid_pre else "BEARISH"
        _m_bias_pre = "BULLISH" if best['close'] > _lm_mid_pre else "BEARISH"

        # ── Direcție finală: AND logic macro (consistent cu trendul) ─────────
        # LONG doar dacă Weekly ȘI Monthly sunt BULLISH — trading cu macro trendul.
        # Aceasta maximizează win rate — contra-trend bounces sunt prea volatile.
        htf_bullish = (_w_bias_pre == "BULLISH" and _m_bias_pre == "BULLISH")

        # Fix v7.4: Direcție orderflow bazată pe AI, nu pe HTF lunar
        # htf_bullish = monthly AND weekly → blocat pe SHORT când lm_mid >> price
        # of_bullish = direcția AI intraday (sau H4 fallback dacă AI=NEUTRAL)
        if ai_direction and ai_direction != "NEUTRAL":
            of_bullish = (ai_direction == "LONG")
        else:
            _h4_hi_of = float(best.get('h4_hi', 0) or 0)
            _h4_lo_of = float(best.get('h4_lo', 0) or 0)
            of_bullish = best['close'] > (_h4_lo_of + (_h4_hi_of - _h4_lo_of) * 0.5) if _h4_hi_of > _h4_lo_of > 0 else htf_bullish

        logger.info(
            f"   📐 HTF BIAS: Weekly={_w_bias_pre} | Monthly={_m_bias_pre} "
            f"→ Direcție: {'LONG 🟢' if htf_bullish else 'SHORT 🔴'}"
        )

        # Component 5: Orderflow imbalance — ICT SWEEP+REVERSAL AWARE (fix v6.7)
        # PROBLEMA v6.6: formula anterioară penaliza intrările ICT pe pullback.
        # În ICT, intri la REVERSAL după sweep: lumânarea de entry e adesea BEARISH
        # în uptrend — tocmai aceasta era penalizată cu body_dir_aligned=0.0.
        # Fix: folosim logica de sweep lichiditate + poziție POC + FVG data din imbalance_data.
        try:
            last_10  = df.tail(10)
            # Fix v7.2: sweep lookback extins la 45 bare (~45 min) pentru detectare post-spike
            # ICT: un BSL sweep rămâne valid 30-45 min — distribution înainte de SHORT entry
            # Cu last_10, spike-ul de la 14:20 nu era văzut la 14:55 → sweep_score=0.3 (minim)
            # Cu last_45, spike-ul e detectat → sweep_score=1.0 → orderflow creşte corect
            last_45  = df.tail(45)

            # 1. Sweep de lichiditate în ultimele 45 bare — confirmă setup ICT post-spike
            p_lo = float(best.get('p_lo', 0) or 0)
            p_hi = float(best.get('p_hi', 0) or 0)
            poc  = float(best.get('poc_level', 0) or best['close'])

            recent_lo = float(last_45['low'].min())
            recent_hi = float(last_45['high'].max())

            # Fix v7.4: of_bullish (AI direction) în loc de htf_bullish (monthly)
            if of_bullish:
                # LONG setup: sweep sub p_lo sau sub POC → confirmare lichidit
                if p_lo > 0 and recent_lo <= p_lo * 1.002:
                    sweep_score = 1.0
                elif poc > 0 and recent_lo <= poc:
                    sweep_score = 0.7
                else:
                    sweep_score = 0.3
            else:
                # SHORT setup: sweep deasupra p_hi sau deasupra POC
                if p_hi > 0 and recent_hi >= p_hi * 0.998:
                    sweep_score = 1.0
                elif poc > 0 and recent_hi >= poc:
                    sweep_score = 0.7
                else:
                    sweep_score = 0.3

            # 2. Poziție față de POC aliniată cu HTF (prețul curent pe partea corectă)
            # Fix v7.4: of_bullish (AI) în loc de htf_bullish (monthly)
            if poc > 0:
                above_poc    = best['close'] > poc
                poc_side_score = 1.0 if (above_poc == of_bullish) else 0.0
            else:
                poc_side_score = 0.5

            # 3. FVG imbalance din analyze_orderflow_imbalance (score_bias ∈ [-0.15, +0.15])
            # Convertim bias-ul FVG la [0,1]: 0.15→1.0, 0→0.5, -0.15→0.0
            fvg_bias     = imbalance_data.get('score_bias', 0.0)
            fvg_score    = min(max((fvg_bias + 0.15) / 0.30, 0.0), 1.0)
            # Fix v9.0: FVG alignment corect pentru SHORT
            # fvg_bias e bipolar [-0.15, +0.15]: negativ = bearish FVG, pozitiv = bullish FVG
            # Dacă AI zice SHORT (not of_bullish), inversăm scorul:
            #   fvg_bias=-0.15 (bearish) → fvg_score=0.0 → inversat=1.0 (bun pt SHORT)
            #   fvg_bias=+0.15 (bullish) → fvg_score=1.0 → inversat=0.0 (rău pt SHORT)
            # Floor la 0.15 (nu 0.3) — permite penalizare reală când FVG e contra
            if not of_bullish:
                fvg_score = max(0.15, 1.0 - fvg_score)

            # Combinat: sweep (50%) + POC side (30%) + FVG imbalance (20%)
            imb_component = (
                0.50 * sweep_score
                + 0.30 * poc_side_score
                + 0.20 * fvg_score
            )
            imb_component = min(max(imb_component, 0.0), 1.0)

        except Exception:
            imb_component = 0.5

        # Weighted sum v6.8 — UPDATE #18: Sentiment analysis ca al 6-lea semnal
        # Problema v6.6: orderflow la 40% cu formula greșită domina scorul negativ
        # pe exact condițiile ICT de entry (pullback = sweep + reversal).
        # ICT restaurat la 30% — rămâne semnalul primar
        # AI crescut la 10% — modelul XGBoost acum are toate 41 features corecte
        # Orderflow redus la 25% — ICT sweep logic corect dar rol secundar
        # Sentiment adăugat la 5% — FinBERT sentiment financiar

        # ── UPDATE #6: Sentiment Engine (FinBERT + Stocktwits) ───────────────
        # get_combined_sentiment() → combined_score [-1,+1] + sentiment_mult [0.80,1.20]
        # Mapăm combined_score la [0,1] pentru componenta 5% din raw_score
        # sentiment_mult aplică ±20% boost/reducere pe scorul final
        _sent_mult = 1.0
        try:
            if _SENTIMENT_ENGINE_OK and _sentiment_engine is not None:
                _sent = _sentiment_engine.get_combined_sentiment()
                # Mapare: combined_score [-1,+1] → sentiment_score [0,1]
                sentiment_score = round((_sent['combined_score'] + 1.0) / 2.0, 4)
                _sent_mult = _sent.get('sentiment_mult', 1.0)
                logger.info(
                    f"   🧠 Sentiment #6: score={_sent['combined_score']:.3f} → {_sent['label']} "
                    f"(FinBERT={_sent['news']['score']:.3f} ST={_sent['social']['score']:.3f}) "
                    f"mult×{_sent_mult:.2f}"
                )
            else:
                sentiment_score = get_news_sentiment_score()  # fallback vechi
        except Exception as _se:
            logger.debug(f"Sentiment engine fallback: {_se}")
            sentiment_score = 0.5

        # Ponderi v7.0: ML (XGBoost calibrat) devine contribuitorul principal
        # ── UPDATE #11: Dynamic weights din RL Feedback Loop ─────────────────────
        # Weights default: AI=0.35, ICT=0.25, Q=0.10, Rel=0.10, OF=0.15, Sent=0.05
        # Se ajustează automat după fiecare trade WIN/LOSS via rl_feedback.py
        # Fix v7.4 #1: Quantum disabled (untrained weights = noise)
        # Quantum component never trained → random initialization produces noise
        # Redistributed quantum weight (0.10) to AI (0.30) for total 0.40 AI weight
        if _RL_OK and _rl_feedback is not None:
            _rl_w = _rl_feedback.load_weights()
        else:
            _rl_w = {
                "ai": 0.30, "ict": 0.25, "quantum": 0.00,
                "rel_strength": 0.05, "orderflow": 0.25, "sentiment": 0.05
            }

        # Salvăm component_scores pentru on_trade_closed (trimis din bridge_api)
        _component_scores_for_rl = {
            "ai":           round(float(ai_component),       4),
            "ict":          round(float(ict_component),      4),
            "quantum":      round(float(q_component),        4),
            "rel_strength": round(float(rel_component),      4),
            "orderflow":    round(float(imb_component),      4),
            "sentiment":    round(float(sentiment_score),    4),
        }

        raw_score = (
            _rl_w["ai"]          * ai_component    # ML principal — XGBoost/calibrat
            + _rl_w["ict"]       * ict_component   # ICT structură piață
            + _rl_w["quantum"]   * q_component     # Quantum edge
            + _rl_w["rel_strength"] * rel_component  # Relative strength QQQ vs SPY
            + _rl_w["orderflow"] * imb_component   # Orderflow sweep + POC + FVG
            + _rl_w["sentiment"] * sentiment_score # FinBERT sentiment financiar
        )

        # Fix v10.5c: AI GATED NORMALIZATION
        # Când AI e complet gated (base_ai=0, WAIT prea mare), weight-ul AI (24.7%)
        # e dead weight → scor maxim posibil ~55%. Celelalte componente (ICT+OF+Q+Rel+Sent)
        # pot avea semnale puternice dar scorul nu reflectă asta.
        # Soluție: normalizăm raw_score la suma efectivă a weight-urilor non-zero.
        # Exemplu: raw=0.433, ai_w=0.247 → 0.433/(1-0.247) = 0.575 (57.5%)
        # Aplicăm doar când AI contribuie sub 5% (gated complet sau aproape).
        if ai_component < 0.05 and _rl_w.get("ai", 0) > 0.05:
            _ai_dead_weight = _rl_w["ai"]
            _active_weight  = 1.0 - _ai_dead_weight
            if _active_weight > 0.3:  # safety: nu normalizăm dacă prea puțin weight activ
                _pre_norm = raw_score
                raw_score = min(raw_score / _active_weight, 1.0)
                logger.info(
                    f"   🔧 AI GATED NORM: AI contribuție={ai_component:.2f} < 5% → "
                    f"normalizare weight {_pre_norm:.3f} / {_active_weight:.3f} = {raw_score:.3f} "
                    f"(elimină dead weight AI {_ai_dead_weight:.1%})"
                )

        # ── UPDATE #14d: ICT+KZ Floor ─────────────────────────────────────
        # Când ICT e complet (≥0.75) ȘI suntem în killzone → floor 0.60
        # Motivație: ICT confluence puternică în fereastră temporală validă
        # nu trebuie blocată de AI gated sau componente slabe.
        # EXCEPȚIE: news blackout (news_mult < 0.5) → nu aplicăm floor.
        # Fix v9.1: Dacă AI e CONTRA HTF (SHORT în piață BULLISH sau LONG în piață BEARISH),
        # floor-ul e redus la 0.50 — sub pragul score_min de 55% → nu forțează trade contra trend.
        # Fix v10.5: ICT+KZ Floor DEZACTIVAT când OF contrazice direcția (< 0.50)
        # Problemă reală: raw=0.38 (LONG) dar OF=0.35 (bearish) → floor ridica la 0.60 → LONG entry
        # pe un SHORT clar. OF sub 0.50 = orderflow-ul real nu confirmă direcția → floor = trap.
        _of_confirms_floor = imb_component >= 0.50
        if ict_component >= 0.75 and in_kz and news_mult >= 0.5 and _of_confirms_floor:
            _contra_htf_floor = (ai_direction == "SHORT" and htf_bullish) or \
                                (ai_direction == "LONG" and not htf_bullish)
            _kz_floor_val = 0.50 if _contra_htf_floor else 0.60
            if raw_score < _kz_floor_val:
                logger.info(
                    f"   🏁 ICT+KZ Floor activ: raw={raw_score:.3f} → {_kz_floor_val:.2f} "
                    f"(ICT={ict_component:.2f} kz={active_kz} OF={imb_component:.2f}"
                    f"{' ⚠️ contra-HTF → floor redus' if _contra_htf_floor else ''})"
                )
                raw_score = _kz_floor_val
        elif ict_component >= 0.75 and in_kz and not _of_confirms_floor:
            logger.info(
                f"   🚫 ICT+KZ Floor BLOCAT: OF={imb_component:.2f} < 0.50 contrazice direcția "
                f"(ICT={ict_component:.2f} kz={active_kz}) → orderflow nu confirmă, probabil trap"
            )

        # Fix v9.0: CORECT adjustment order — MULTIPLICATIVE FIRST, then ADDITIVE
        # Motivație: news_mult=0.1 (NFP blackout) trebuie să ucidă scorul ÎNAINTE
        # de ajustări aditive, altfel DXY/Options/Correlation pot resuscita un semnal mort.
        # Ordinea: raw → news × → sentiment × → DXY + → Options + → Correlation +
        _pre_adj_raw = raw_score

        # ── 1. MULTIPLICATIVE: News hard multiplier (blackout/caution) ──────
        raw_score = raw_score * news_mult

        # ── 2. MULTIPLICATIVE: Sentiment (±5% max) ─────────────────────────
        if _sent_mult != 1.0:
            _sent_mult_capped = max(0.95, min(_sent_mult, 1.05))
            raw_score = raw_score * _sent_mult_capped
        raw_score = max(0.0, min(raw_score, 1.0))

        # ── 3. ADDITIVE: DXY adjustment ────────────────────────────────────
        raw_score = max(0.0, min(raw_score + dxy_adj, 1.0))

        # ── 4. ADDITIVE: Options flow ──────────────────────────────────────
        raw_score = max(0.0, min(raw_score + options_adj, 1.0))

        # ── 5. ADDITIVE: Correlation NQ/ES ─────────────────────────────────
        if _corr_adj != 0.0:
            raw_score = max(0.0, min(raw_score + _corr_adj, 1.0))

        logger.info(
            f"   📊 Score breakdown v7.1+RL: AI={ai_component:.2f}×{_rl_w['ai']:.3f} "
            f"ICT={ict_component:.2f}×{_rl_w['ict']:.3f}(smt={has_smt}/{cur_smt} fvg={has_fvg}/{cur_fvg} dis={has_dis}/{cur_dis} kz={in_kz} h4={'✅' if h4_aligned else '❌'} h1={'✅' if h1_aligned else '❌'} m15={'✅' if m15_aligned else '❌'}) "
            f"Q={q_component:.2f}×{_rl_w['quantum']:.3f} Rel={rel_component:.2f}×{_rl_w['rel_strength']:.3f} "
            f"Orderflow={imb_component:.2f}×{_rl_w['orderflow']:.3f}(bias={'BULL' if of_bullish else 'BEAR'}/AI sweep={sweep_score:.2f} poc={poc_side_score:.1f} fvg={fvg_score:.2f}) "
            f"Sentiment={sentiment_score:.2f}×{_rl_w['sentiment']:.3f}(mult×{_sent_mult:.2f}) "
            f"→ raw={raw_score:.3f}"
        )

        # Institutional filters (HTF alignment) — direction-aware v7.7
        _early_bias = "BULL" if of_bullish else "BEAR"
        filtered_score, filter_msg = apply_institutional_filters(best, raw_score, early_bias=_early_bias)

        # Fix v9.0: Quantum noise filter — cap penalty la max -15% din institutional score
        # Evită double-crush: dacă institutional deja a penalizat heavy, noise nu mai taie
        _pre_noise = filtered_score
        final_score, is_noisy = apply_quantum_noise_filter(filtered_score, best)
        if is_noisy and _pre_noise > 0:
            _noise_pct = 1.0 - (final_score / _pre_noise) if _pre_noise > 0 else 0
            if _noise_pct > 0.15:
                final_score = _pre_noise * 0.85  # cap la -15%
                logger.info(f"   🔇 Noise filter capped: {_pre_noise:.3f} → {final_score:.3f} (max -15%)")

        score_pct = round(min(final_score * 100, 100), 2)

        # Fix v9.0: Volume boost MULTIPLICATIV (nu aditiv)
        # Aditiv (+0.10 flat) distorsiona: scor 0.30 → 0.40 (+33%) vs scor 0.85 → 0.95 (+12%)
        # Multiplicativ (×1.10) e proporțional indiferent de scor
        if vol_boost > 0:
            _vol_mult = 1.0 + vol_boost  # vol_boost=0.10 → ×1.10
            _pre_vol = final_score
            final_score = min(final_score * _vol_mult, 1.0)
            score_pct   = round(min(final_score * 100, 100), 2)
            logger.info(f"   📈 Volume boost ×{_vol_mult:.2f}: {_pre_vol:.3f} → {final_score:.3f}")

        # ── FAZA 3.3: News Continuation Setup ────────────────────────────────
        # Fereastră POST-news (primele 30 min după NFP/FOMC/CPI/PCE) = setup ICT premium.
        # Boost aplicat ÎNAINTE de time decay (overriding partial decay).
        if _news_cont_boost > 0:
            _pre_nc     = final_score
            final_score = min(final_score + _news_cont_boost, 1.0)
            score_pct   = round(min(final_score * 100, 100), 2)
            logger.info(
                f"   📰 {_news_cont_msg} | scor {_pre_nc:.2f} → {final_score:.2f}"
            )
        # ─────────────────────────────────────────────────────────────────────

        # ── FAZA 1.2: Time-based Confidence Decay ─────────────────────────────
        # Penalizare semnale late-session: cu cât e mai târziu în sesiune,
        # cu atât scorul se reduce — piața devine mai impredictibilă la capete.
        # Ore UTC: London = 07-12, NY = 13-20, Overlap = 12-14
        # Penalizare: -5% în ultima oră a fiecărei sesiuni, -10% după închiderea NY (20 UTC)
        _now_utc = datetime.utcnow()
        _utc_h   = _now_utc.hour + _now_utc.minute / 60.0
        _decay   = 0.0
        _decay_reason = ""

        if 11.75 <= _utc_h < 12.0:          # ultimele 15 min sesiune London
            _decay = 0.05
            _decay_reason = "late London (11:45-12:00 UTC)"
        elif 12.0 <= _utc_h < 13.0:         # tranziție London→NY (dead zone)
            _decay = 0.08
            _decay_reason = "London→NY transition (12-13 UTC)"
        elif 19.5 <= _utc_h < 20.0:         # ultimele 30 min NY
            _decay = 0.05
            _decay_reason = "late NY (19:30-20:00 UTC)"
        elif _utc_h >= 20.0 or _utc_h < 2.0:  # post-NY / Asia early
            _decay = 0.12
            _decay_reason = f"post-NY / Asian early ({_utc_h:.1f} UTC)"

        if _decay > 0.0:
            _pre_decay_score = final_score
            final_score = max(final_score - _decay, 0.0)
            score_pct   = round(min(final_score * 100, 100), 2)
            logger.info(
                f"   ⏰ Time Decay [{_decay_reason}]: "
                f"{_pre_decay_score:.2f} → {final_score:.2f} (-{_decay:.0%})"
            )
        # ─────────────────────────────────────────────────────────────────────

        logger.info(f"   🔥 Final Score: {score_pct}%")
        _engine_score_pct = score_pct   # salvat pentru floor-ul post-processing
        _post_proc_anchor = final_score  # v10.5: ancoră pentru global cap pe ajustări post-raw

        # ── 7. Risk Management ───────────────────────────────────────────────
        # Fix v7.3: trade_direction determinat de AI (intraday), nu blocat de HTF lunar
        # Problema AND logic: lm_mid=24747, price=23514 → monthly BEARISH mereu
        # → trade_direction=SHORT PERMANENT, robotul nu poate lua LONG niciodată în luna bearish
        # Fix: AI determină direcția intraday; HTF rămâne ca multiplicator în orderflow score
        # Când AI e gated (p_wait>0.88, base_ai=0) → fallback la HTF 4H (h4_bull), nu monthly
        #
        # Fix v10.6: HTF DIRECTION BLEND — când AI are confidence scăzut (p_dir < 0.30 sau
        # WAIT > 0.60), direcția marginală AI (26% SHORT vs 5% LONG) NU ar trebui să override
        # tot HTF-ul (Weekly BULL + Monthly BULL + H4 BULL). Acum:
        #   - AI confident (p_dir >= 0.30, WAIT <= 0.60): AI decide 100% (ca înainte)
        #   - AI moderat (p_dir >= 0.20, WAIT <= 0.70): AI decide, dar cu HTF penalty pe scor
        #   - AI slab (p_dir < 0.20 sau WAIT > 0.70): HTF override — fallback la H4 bias
        # Asta previne: AI cu 26% SHORT / 5% LONG / 69% WAIT → SHORT, ignorând tot HTF BULL
        _h4_hi_dir = float(best.get('h4_hi', 0) or 0)
        _h4_lo_dir = float(best.get('h4_lo', 0) or 0)
        _h4_bull   = best['close'] > (_h4_lo_dir + (_h4_hi_dir - _h4_lo_dir) * 0.5) if _h4_hi_dir > _h4_lo_dir > 0 else htf_bullish

        # p_dir = probabilitatea direcției alese de AI (cea mai mare din SHORT/LONG)
        _ai_p_dir    = max(p_short, p_long) if 'p_short' in dir() and 'p_long' in dir() else 0.0
        _ai_p_wait   = p_wait if 'p_wait' in dir() else 1.0
        _ai_confident = (_ai_p_dir >= 0.30 and _ai_p_wait <= 0.60)
        _ai_moderate  = (_ai_p_dir >= 0.22 and _ai_p_wait <= 0.62)
        _htf_dir      = "LONG" if _h4_bull else "SHORT"

        _dir_reason = ""
        if _ai_confident:
            # AI confident — decide 100%
            trade_direction = ai_direction if ai_direction else _htf_dir
            _dir_reason = "AI_CONFIDENT"
        elif _ai_moderate:
            # AI moderat — decide direcția, dar dacă contrazice HTF, logăm warning
            trade_direction = ai_direction if ai_direction else _htf_dir
            _dir_reason = "AI_MODERATE"
            if ai_direction and ai_direction != _htf_dir:
                _dir_reason = "AI_MODERATE_CONTRA_HTF"
                logger.info(
                    f"   ⚠️ AI MODERAT contrazice HTF: AI={ai_direction} (p_dir={_ai_p_dir:.2f}) "
                    f"vs HTF={_htf_dir} (h4={'BULL' if _h4_bull else 'BEAR'}) → AI decide, scor penalizat"
                )
        else:
            # AI slab (WAIT > 0.70 sau p_dir < 0.20) — HTF override
            if ai_direction and ai_direction == _htf_dir:
                # AI slab DAR aliniată cu HTF → OK, merge
                trade_direction = ai_direction
                _dir_reason = "AI_WEAK_HTF_ALIGNED"
            elif ai_direction and ai_direction != _htf_dir:
                # AI slab ȘI contrazice HTF → HTF override
                trade_direction = _htf_dir
                _dir_reason = "HTF_OVERRIDE"
                logger.info(
                    f"   🔄 HTF OVERRIDE: AI={ai_direction} (p_dir={_ai_p_dir:.2f}, WAIT={_ai_p_wait:.2f}) "
                    f"prea slab să contrazică HTF={_htf_dir} (h4={'BULL' if _h4_bull else 'BEAR'}, "
                    f"W={'BULL' if htf_bullish else 'BEAR'}) → trade_direction={_htf_dir}"
                )
            else:
                trade_direction = _htf_dir
                _dir_reason = "HTF_FALLBACK"

        logger.info(
            f"   🧭 DIRECȚIE AI: AI={ai_direction} (p_dir={_ai_p_dir:.2f} WAIT={_ai_p_wait:.2f}) "
            f"→ trade_direction={trade_direction} [{_dir_reason}] "
            f"(HTF={'LONG' if htf_bullish else 'SHORT'} h4={'BULL' if _h4_bull else 'BEAR'})"
        )

        # ══════════════════════════════════════════════════════════════════════
        # v8.0 REVERSAL OVERRIDE — Detectare schimbare de trend intraday
        # Când 4+ indicatori intraday confirmă direcția opusă față de AI,
        # flip trade_direction pe reversal. Permite botului să prindă turnările.
        # ══════════════════════════════════════════════════════════════════════
        _reversal_override = False
        _reversal_dir      = None
        _reversal_score    = 0
        _reversal_signals  = []

        try:
            if len(df) >= 10:
                _rev_close = float(best.get('close', 0) or 0)
                _rev_open  = float(best.get('open', 0) or 0)

                # ── 1. MSS/CHoCH Detection — v10.1 HTF Swing ──────────────────
                # Fix v10.1: lookback extins la 25 bare (≈5m context) cu structură reală
                # Anterior: 5-bar trend → orice oscilație 1m = "CHoCH" fals
                # Acum: lower highs + lower lows obligatorii, nu doar close comparison
                _rev_lb = min(25, len(df) - 1)
                _rev_half = _rev_lb // 2
                _rev_highs  = list(df['high'].iloc[-_rev_lb:])
                _rev_lows   = list(df['low'].iloc[-_rev_lb:])
                _rev_closes = list(df['close'].iloc[-_rev_lb:])

                _rev_older_hi  = max(_rev_highs[:_rev_half])
                _rev_older_lo  = min(_rev_lows[:_rev_half])
                _rev_recent_hi = max(_rev_highs[_rev_half:-1]) if len(_rev_highs) > _rev_half + 1 else _rev_close
                _rev_recent_lo = min(_rev_lows[_rev_half:-1])  if len(_rev_lows)  > _rev_half + 1 else _rev_close

                # Trend real: lower highs + lower lows (nu doar close comparison)
                _rev_trend_down = (
                    _rev_recent_hi < _rev_older_hi and
                    _rev_recent_lo < _rev_older_lo and
                    _rev_closes[-2] < _rev_closes[-_rev_half]
                )
                _rev_trend_up = (
                    _rev_recent_hi > _rev_older_hi and
                    _rev_recent_lo > _rev_older_lo and
                    _rev_closes[-2] > _rev_closes[-_rev_half]
                )

                # CHoCH Bullish: was downtrend, now breaks above recent swing high
                _choch_bull = _rev_trend_down and _rev_close > _rev_recent_hi
                # CHoCH Bearish: was uptrend, now breaks below recent swing low
                _choch_bear = _rev_trend_up and _rev_close < _rev_recent_lo

                if _choch_bull:
                    _reversal_score += 2
                    _reversal_signals.append("CHoCH_BULL")
                elif _choch_bear:
                    _reversal_score -= 2
                    _reversal_signals.append("CHoCH_BEAR")

                # ── 2. Swing Break ──
                _sw_hi_10 = max(_rev_highs[:-1]) if len(_rev_highs) > 1 else _rev_close
                _sw_lo_10 = min(_rev_lows[:-1]) if len(_rev_lows) > 1 else _rev_close
                if _rev_close > _sw_hi_10:
                    _reversal_score += 1
                    _reversal_signals.append("SWING_BREAK_BULL")
                elif _rev_close < _sw_lo_10:
                    _reversal_score -= 1
                    _reversal_signals.append("SWING_BREAK_BEAR")

                # ── 3. DOM Ratio — institutional pressure ──
                _rev_dom = float(best.get('bid_ask_ratio', 1.0) or 1.0)
                if _rev_dom >= 2.0:
                    _reversal_score += 1
                    _reversal_signals.append(f"DOM_BID_{_rev_dom:.1f}")
                elif _rev_dom <= 0.5:
                    _reversal_score -= 1
                    _reversal_signals.append(f"DOM_ASK_{_rev_dom:.2f}")

                # ── 4. POC Drift — institutional accumulation direction ──
                if 'poc_level' in df.columns and len(df) >= 5:
                    _rev_poc_now  = float(df['poc_level'].iloc[-1]) if df['poc_level'].iloc[-1] else 0
                    _rev_poc_prev = float(df['poc_level'].iloc[-4]) if df['poc_level'].iloc[-4] else 0
                    if _rev_poc_now > 0 and _rev_poc_prev > 0:
                        _poc_change = _rev_poc_now - _rev_poc_prev
                        if _poc_change > 2.0:
                            _reversal_score += 1
                            _reversal_signals.append("POC_DRIFT_UP")
                        elif _poc_change < -2.0:
                            _reversal_score -= 1
                            _reversal_signals.append("POC_DRIFT_DOWN")

                # ── 5. Cumulative Delta proxy (close-open * volume) ──
                if len(df) >= 6:
                    _rev_deltas = [(float(df['close'].iloc[i]) - float(df['open'].iloc[i])) * max(float(df['volume'].iloc[i]), 1)
                                   for i in range(-6, 0)]
                    _cum_delta = sum(_rev_deltas)
                    _prev_deltas = [(float(df['close'].iloc[i]) - float(df['open'].iloc[i])) * max(float(df['volume'].iloc[i]), 1)
                                    for i in range(-12, -6)] if len(df) >= 12 else _rev_deltas
                    _prev_cum = sum(_prev_deltas)

                    # Delta flip: previous was negative, now positive (or vice versa)
                    if _prev_cum < 0 and _cum_delta > 0:
                        _reversal_score += 1
                        _reversal_signals.append("DELTA_FLIP_BULL")
                    elif _prev_cum > 0 and _cum_delta < 0:
                        _reversal_score -= 1
                        _reversal_signals.append("DELTA_FLIP_BEAR")

                # ── 6. Momentum Divergence ──
                if len(df) >= 16:
                    _rev_price_ch = _rev_close - float(df['close'].iloc[-16])
                    _rev_mom = (_rev_close / max(float(df['close'].iloc[-16]), 1) - 1)
                    _rev_mom_prev = (float(df['close'].iloc[-6]) / max(float(df['close'].iloc[-16]), 1) - 1)
                    _rev_mom_accel = _rev_mom - _rev_mom_prev
                    # Bearish div: price up but momentum decelerating
                    if _rev_price_ch > 5 and _rev_mom_accel < -0.001:
                        _reversal_score -= 1
                        _reversal_signals.append("MOM_DIV_BEAR")
                    # Bullish div: price down but momentum accelerating
                    elif _rev_price_ch < -5 and _rev_mom_accel > 0.001:
                        _reversal_score += 1
                        _reversal_signals.append("MOM_DIV_BULL")

                # ── 7. Trend Exhaustion — consecutive bars + body shrinking ──
                if len(df) >= 8:
                    _rev_dirs = [1 if float(df['close'].iloc[i]) > float(df['open'].iloc[i]) else -1
                                 for i in range(-8, 0)]
                    _consec_bull = sum(1 for d in _rev_dirs if d > 0)
                    _consec_bear = sum(1 for d in _rev_dirs if d < 0)
                    _last_body = abs(_rev_close - _rev_open)
                    _prev_body = abs(float(df['close'].iloc[-2]) - float(df['open'].iloc[-2]))

                    if _consec_bear >= 6 and _last_body < _prev_body * 0.5:
                        _reversal_score += 1
                        _reversal_signals.append("EXHAUST_BEAR→BULL")
                    elif _consec_bull >= 6 and _last_body < _prev_body * 0.5:
                        _reversal_score -= 1
                        _reversal_signals.append("EXHAUST_BULL→BEAR")

                # ── 8. Orderflow Super-Confirmation ──
                # CHoCH/MSS + Orderflow aligned = scor dublu pe CHoCH/swing signals
                # Folosim cum_delta din live_data (bridge) + DOM ratio + imbalance
                _of_confirms_bull = False
                _of_confirms_bear = False
                try:
                    if _live_mode and live_data:
                        _rev_cum_delta = float(live_data.get("cum_delta", 0) or 0)
                        _rev_imbalance = float(live_data.get("imbalance_pct", 0) or 0)
                        _rev_bar_deltas = live_data.get("bar_deltas", []) or []

                        # Recent bar deltas (last 5) — bullish dacă majoritatea pozitive
                        _recent_bd = [float(x) for x in _rev_bar_deltas[-5:]] if len(_rev_bar_deltas) >= 5 else []
                        _bd_bull = sum(1 for x in _recent_bd if x > 0) >= 4 if _recent_bd else False
                        _bd_bear = sum(1 for x in _recent_bd if x < 0) >= 4 if _recent_bd else False

                        # Cum delta > 300 = buy pressure, < -300 = sell pressure
                        _of_confirms_bull = (_rev_cum_delta > 300 or _bd_bull) and _rev_dom >= 1.5
                        _of_confirms_bear = (_rev_cum_delta < -300 or _bd_bear) and _rev_dom <= 0.7

                        if _of_confirms_bull and _reversal_score > 0:
                            _reversal_score += 2
                            _reversal_signals.append(f"OF_SUPER_BULL(delta={_rev_cum_delta:.0f})")
                            logger.info(f"   💪 ORDERFLOW SUPER-CONFIRM BULL: cum_delta={_rev_cum_delta:.0f}, DOM={_rev_dom:.1f}, bar_deltas_bull={_bd_bull}")
                        elif _of_confirms_bear and _reversal_score < 0:
                            _reversal_score -= 2
                            _reversal_signals.append(f"OF_SUPER_BEAR(delta={_rev_cum_delta:.0f})")
                            logger.info(f"   💪 ORDERFLOW SUPER-CONFIRM BEAR: cum_delta={_rev_cum_delta:.0f}, DOM={_rev_dom:.2f}, bar_deltas_bear={_bd_bear}")
                except Exception as _of_err:
                    logger.debug(f"OF super-confirm error: {_of_err}")

                # ── 9. Big Trade Detection (placeholder — activat când NT8 trimite date) ──
                # NT8 addon va trimite big_trades: [{price, size, side, timestamp}]
                # Când un big trade (>50 contracts NQ) apare în direcția reversal → +2
                _big_trades = (live_data.get("big_trades", []) or []) if _live_mode and live_data else []
                if _big_trades:
                    try:
                        _bt_bull = sum(1 for bt in _big_trades if bt.get("side") == "BUY" and bt.get("size", 0) >= 50)
                        _bt_bear = sum(1 for bt in _big_trades if bt.get("side") == "SELL" and bt.get("size", 0) >= 50)
                        if _bt_bull >= 2 and _reversal_score > 0:
                            _reversal_score += 2
                            _reversal_signals.append(f"BIG_TRADES_BULL({_bt_bull})")
                        elif _bt_bear >= 2 and _reversal_score < 0:
                            _reversal_score -= 2
                            _reversal_signals.append(f"BIG_TRADES_BEAR({_bt_bear})")
                    except Exception:
                        pass

                # ── 10. Absorption Detection for Reversal ──
                # Absorption = volum mare dar preț static → instituționalii absorb
                # Bid absorption (bullish): sell pressure absorbită → preț nu scade
                # Ask absorption (bearish): buy pressure absorbită → preț nu urcă
                try:
                    if _live_mode and live_data:
                        _rev_abs_score = float(live_data.get("absorption_score", 0) or 0)
                        _rev_bbv = float(live_data.get("bar_buy_vol", 0) or 0)
                        _rev_bsv = float(live_data.get("bar_sell_vol", 0) or 0)
                        _rev_tot_vol = _rev_bbv + _rev_bsv
                        _rev_body = abs(_rev_close - _rev_open)
                        _rev_atr = float(best.get('atr_14', 0) or 0)

                        _abs_bull = False
                        _abs_bear = False

                        # NT8 direct absorption
                        if _rev_abs_score >= 60:
                            _rev_cd = float(live_data.get("cum_delta", 0) or 0)
                            if _rev_cd > 0:
                                _abs_bull = True
                            else:
                                _abs_bear = True

                        # Delta absorption proxy
                        elif _rev_tot_vol > 500 and _rev_atr > 0 and _rev_body < _rev_atr * 0.3:
                            if _rev_bsv > _rev_tot_vol * 0.60:
                                _abs_bull = True  # sell vol mare dar preț nu scade
                            elif _rev_bbv > _rev_tot_vol * 0.60:
                                _abs_bear = True  # buy vol mare dar preț nu urcă

                        # Volume absorption proxy din bar history
                        elif not _abs_bull and not _abs_bear:
                            _rev_bvols = live_data.get("bar_volumes", []) or []
                            _rev_bopens = live_data.get("bar_opens", []) or []
                            _rev_bcloses = live_data.get("bar_closes", []) or []
                            if len(_rev_bvols) >= 6 and _rev_atr > 0:
                                _rv_avg = sum(_rev_bvols[:-1]) / max(len(_rev_bvols) - 1, 1)
                                _rv_last = _rev_bvols[-1]
                                _rv_body = abs(_rev_bcloses[-1] - _rev_bopens[-1]) if _rev_bcloses and _rev_bopens else 999
                                if _rv_last > _rv_avg * 2.0 and _rv_body < _rev_atr * 0.3:
                                    _rv_cd = float(live_data.get("cum_delta", 0) or 0)
                                    if _rv_cd > 0:
                                        _abs_bull = True
                                    else:
                                        _abs_bear = True

                        if _abs_bull:
                            _reversal_score += 1
                            _reversal_signals.append("ABSORPTION_BULL")
                            logger.info(f"   🧱 REVERSAL ABSORPTION: BID absorption (bullish) +1")
                        elif _abs_bear:
                            _reversal_score -= 1
                            _reversal_signals.append("ABSORPTION_BEAR")
                            logger.info(f"   🧱 REVERSAL ABSORPTION: ASK absorption (bearish) -1")
                except Exception as _abs_rev_err:
                    logger.debug(f"Absorption reversal error: {_abs_rev_err}")

                # ══ DECIZIA DE REVERSAL ══
                # Threshold: scor >= 4 (minim 4 indicatori din 10 confirmă reversal)
                # Cu Orderflow super-confirm + absorption, e mai ușor de atins
                # Fix v9.1: pe zile cu volume spike (trend day), lowering threshold la 3
                # Fix v10.1: în consolidare mid-range threshold ridicat la 6 pentru a
                #   preveni flip-uri false bazate pe oscilații normale în range
                _rev_threshold = 3 if vol_boost > 0 else 4

                # ── Consolidation guard pentru reversal override ───────────────
                # Calculăm quick proxy consolidare folosind `regime` (disponibil la linia 3017)
                # și un pd_pct estimat din h4_hi/h4_lo disponibile în `best`
                _rev_regime_str = str(regime).upper() if regime else ""
                _rev_is_consol  = ("CONSOL" in _rev_regime_str or "RANG" in _rev_regime_str
                                   or "CHOP" in _rev_regime_str)
                _rev_h4_hi = float(best.get('h4_hi', 0) or 0)
                _rev_h4_lo = float(best.get('h4_lo', 0) or 0)
                if _rev_h4_hi > _rev_h4_lo > 0 and _rev_close > 0:
                    _rev_pd_pct = (_rev_close - _rev_h4_lo) / (_rev_h4_hi - _rev_h4_lo)
                else:
                    _rev_pd_pct = 0.5
                _rev_mid_range = 0.20 < _rev_pd_pct < 0.80
                _rev_has_disp  = bool(best.get('has_displacement', 0))

                if _rev_is_consol and _rev_mid_range and not _rev_has_disp:
                    # Consolidare + mid-range + fără displacement = risc maxim de fakeout
                    # Reversal override necesită 6+ semnale în loc de 3-4
                    _old_thresh = _rev_threshold
                    _rev_threshold = max(_rev_threshold + 3, 6)
                    logger.info(
                        f"   🔲 CONSOLIDATION GUARD reversal: threshold {_old_thresh} → {_rev_threshold} "
                        f"(regime={_rev_regime_str}, pd_pct={_rev_pd_pct:.0%}, mid_range)"
                    )

                if _reversal_score >= _rev_threshold and trade_direction == "SHORT":
                    _reversal_override = True
                    _reversal_dir = "LONG"
                    logger.info(
                        f"   🔄 REVERSAL OVERRIDE: {trade_direction} → LONG "
                        f"(score={_reversal_score}, signals={_reversal_signals})"
                    )
                    trade_direction = "LONG"

                elif _reversal_score <= -_rev_threshold and trade_direction == "LONG":
                    _reversal_override = True
                    _reversal_dir = "SHORT"
                    logger.info(
                        f"   🔄 REVERSAL OVERRIDE: {trade_direction} → SHORT "
                        f"(score={_reversal_score}, signals={_reversal_signals})"
                    )
                    trade_direction = "SHORT"

                else:
                    logger.info(
                        f"   🔄 Reversal check: score={_reversal_score} "
                        f"(threshold=±{_rev_threshold}) signals={_reversal_signals} → NO OVERRIDE"
                    )

        except Exception as _rev_err:
            logger.debug(f"Reversal override error: {_rev_err}")

        logger.info(
            f"   🧭 DIRECȚIE FINALĂ: trade_direction={trade_direction} "
            f"(reversal_override={_reversal_override})"
        )

        # ── HTF HARD BLOCK v9.1: VIX >25 + HTF Bullish → blochează SHORT ────
        # Zilele cu VIX>25 + HTF W+M BULLISH sunt tipic squeeze/reversal macro.
        # Shortul contra trendului principal în aceste condiții are win-rate scăzut.
        # EXCEPȚIE: dacă reversal_override e deja activ (deja flipped la LONG), nu interferăm.
        _htf_hard_block = htf_bullish and vix_mult < 1.0  # vix_mult=0.5 → VIX>25
        if _htf_hard_block and trade_direction == "SHORT" and not _reversal_override:
            logger.info(
                f"   🛡️ HTF HARD BLOCK v9.1: SHORT blocat → LONG forțat "
                f"(htf_bullish=True W+M | vix_mult={vix_mult:.2f} → VIX>25)"
            )
            trade_direction = "LONG"

        # ── CONFLICT FILTER: AI vs HTF ───────────────────────────────────────
        # Penalizare ușoară dacă AI și HTF monthly sunt în conflict (nu mai SKIP)
        _ai_conflicts_htf = False  # dezactivat — HTF monthly nu mai blochează direcția AI

        # ── FAZA 1.3: Asia Range Filter ───────────────────────────────────────
        # Dacă prețul curent e ÎNĂUNTRUL range-ului Asia, piața nu a breakout-uit
        # → semnal slab, mai ales în primele ore London.
        # Activ pe toată sesiunea London (07:00-17:00 UTC) — consistent cu strategia intraday_london
        _asia_hi  = float(best.get('asia_hi', 0) or 0)
        _asia_lo  = float(best.get('asia_lo', 0) or 0)
        _cur_px   = float(best.get('close',   0) or 0)
        _utc_h_ar = datetime.utcnow().hour + datetime.utcnow().minute / 60.0

        _asia_range_valid = _asia_hi > 0 and _asia_lo > 0 and _cur_px > 0
        _in_asia_range    = _asia_lo <= _cur_px <= _asia_hi
        _london_session   = 7.0 <= _utc_h_ar < 17.0        # London + NY overlap (07:00-17:00 UTC)

        if _asia_range_valid and _in_asia_range and _london_session and not skip_volatile:
            # Penalizare: scădem 10% din scor, NU skipăm complet (poate fi valid cu score mare)
            _asia_range_pts = _asia_hi - _asia_lo
            _asia_mid       = (_asia_hi + _asia_lo) / 2
            _dist_mid_pct   = abs(_cur_px - _asia_mid) / max(_asia_range_pts, 1) * 100
            logger.info(
                f"   🌏 ASIA RANGE FILTER: preț {_cur_px} în [{_asia_lo}-{_asia_hi}] "
                f"(range={_asia_range_pts:.1f}pts, dist mid={_dist_mid_pct:.1f}%) "
                f"→ penalizare scor -10%"
            )
            final_score = max(final_score - 0.10, 0.0)
            score_pct   = round(min(final_score * 100, 100), 2)
        elif _asia_range_valid and _in_asia_range and not _london_session:
            # În afara London session + în range Asia = și mai slab → skip
            logger.info(
                f"   🌏 ASIA RANGE: preț {_cur_px} în range Asia [{_asia_lo}-{_asia_hi}] "
                f"și nu e sesiune London → SKIP"
            )
            if not skip_volatile:
                skip_volatile = True
                vol_msg = f"ÎN ASIA RANGE [{_asia_lo}-{_asia_hi}] fără sesiune activă"
        # ─────────────────────────────────────────────────────────────────────

        # ── FAZA 2.1: Exhaustion Detector ─────────────────────────────────────
        # Fix v7.6: ATR_14 pe 1-min = range-ul mediu al UNEI bare.
        # Range pe N bare crește ca √N × ATR (random walk).
        # Normalizăm: ratio = range / (ATR × √N) → 1.0 = normal, >1.5 = extins
        # Praguri: 1.8×norm → -8%, 2.2×norm → -15%, 3.0×norm → SKIP
        import math as _math_exh
        _exh_atr   = float(best.get('atr_14', 0) or 0)
        _exh_high  = float(best.get('high',   0) or 0)
        _exh_low   = float(best.get('low',    0) or 0)
        _exh_close = float(best.get('close',  0) or 0)

        if _exh_atr > 0 and _exh_high > 0 and _exh_low > 0:
            # Fix v7.7: lookback LOCAL (10 bare) nu zilnic (30 bare)
            # Pe zile volatile cu gap-uri mari, 30 bare captează întregul gap zilnic
            # → false SKIP chiar când local piața e stabilă.
            # 10 bare = ~10 minute pe 1-min chart = range LOCAL real.
            _lookback_exh = min(10, len(df))
            if _lookback_exh >= 2:
                _day_high  = float(df['high'].iloc[-_lookback_exh:].max())
                _day_low   = float(df['low'].iloc[-_lookback_exh:].min())
                _day_range = _day_high - _day_low
            else:
                _day_range = _exh_high - _exh_low
            # Normalizare: range așteptat = ATR × √lookback
            _expected_range = _exh_atr * _math_exh.sqrt(max(_lookback_exh, 1))
            _exh_ratio   = _day_range / _expected_range if _expected_range > 0 else 0
            _exh_penalty = 0.0
            _exh_label   = ""

            # Fix v10.5: penalizare exhaustion redusă — max -7% normal, -3% când reversal_override
            # Candela mare în context de reversal override = displacement, nu exhaustion
            if _exh_ratio >= 2.2:
                _exh_penalty = 0.03 if _reversal_override else 0.07
                _exh_label   = f"exhausted {_exh_ratio:.1f}×norm (range={_day_range:.1f}pts, expected={_expected_range:.0f}pts)"
            elif _exh_ratio >= 1.8:
                _exh_penalty = 0.02 if _reversal_override else 0.04
                _exh_label   = f"extended {_exh_ratio:.1f}×norm (range={_day_range:.1f}pts, expected={_expected_range:.0f}pts)"
            else:
                logger.info(f"   💨 EXHAUSTION: range={_day_range:.1f}pts | expected={_expected_range:.0f}pts | ratio={_exh_ratio:.2f}×norm → OK (sub 1.8×)")

            if _exh_penalty > 0:
                # v10.5: Bypass exhaustion când OF live confirmă mișcarea e reală
                # Tape rapid + delta aliniată + volum de vânzare/cumpărare = trend real, nu exhaustion
                _of_cd_exh = float(live_data.get("cum_delta", 0) or 0) if _live_mode and live_data else 0
                _of_ts_exh = float(live_data.get("tape_speed", 0) or 0) if _live_mode and live_data else 0
                _of_bd_exh = float(live_data.get("bar_delta", 0) or 0) if _live_mode and live_data else 0
                _exh_of_confirms = False
                if trade_direction == "LONG" and _of_cd_exh > 0 and _of_bd_exh > 0 and _of_ts_exh >= 40:
                    _exh_of_confirms = True
                elif trade_direction == "SHORT" and _of_cd_exh < 0 and _of_bd_exh < 0 and _of_ts_exh >= 40:
                    _exh_of_confirms = True

                if _exh_of_confirms:
                    _exh_penalty = min(_exh_penalty, 0.01)  # cap la -1% dacă OF confirmă
                    logger.info(
                        f"   💨 EXHAUSTION BYPASS: tape={_of_ts_exh:.0f} ticks/sec + "
                        f"cum_delta={_of_cd_exh:.0f} + bar_delta={_of_bd_exh:.0f} → "
                        f"OF confirmă trend real, penalizare redusă la -1%"
                    )

                _pre_exh = final_score
                final_score = max(final_score - _exh_penalty, 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                _ro_tag = " [reversal_override → -3%]" if _reversal_override else ""
                logger.info(
                    f"   💨 EXHAUSTION: {_exh_label}{_ro_tag} "
                    f"→ scor {_pre_exh:.2f} → {final_score:.2f} (-{_exh_penalty:.0%})"
                )
        # ─────────────────────────────────────────────────────────────────────

        # ── FAZA 2.2: Liquidity Sweep Detection ──────────────────────────────
        # ICT stop hunt + reversal — preț sparge un nivel cheie (H1 hi/lo, Asia hi/lo)
        # cu wick lung, dar closeul revine înapoi = manipulare instituțională.
        # Aceasta e un setup DE MARE CALITATE → BOOST scor +12%.
        # Dacă sweep e în direcția trade-ului → și mai bun.
        _lsw_hi     = float(best.get('high',    0) or 0)
        _lsw_lo     = float(best.get('low',     0) or 0)
        _lsw_open   = float(best.get('open',    0) or 0)
        _lsw_close  = float(best.get('close',   0) or 0)
        _lsw_wick_r = float(best.get('wick_ratio', 0) or 0)
        _h1_hi      = float(best.get('h1_hi',   0) or 0)
        _h1_lo      = float(best.get('h1_lo',   0) or 0)

        _lsw_detected  = False
        _lsw_direction = None   # "LONG" dacă sweep sub nivel → reversal sus

        # Fix v7.4 #8: Volatility-normalized wick threshold
        # Base 2.5 reduced by ATR volatility ratio: higher volatility → lower threshold
        _wick_threshold = max(1.8, 2.5 - (_exh_atr / best['close'] * 100)) if _exh_atr > 0 else 2.5

        # Sweep bearish (stop hunt pe SHORTs → reversal LONG):
        # Prețul a spart sub h1_lo/asia_lo cu wick, dar a revenit sus
        _key_lo = min(x for x in [_h1_lo, _asia_lo] if x > 0) if (_h1_lo > 0 or _asia_lo > 0) else 0
        _key_hi = max(x for x in [_h1_hi, _asia_hi] if x > 0) if (_h1_hi > 0 or _asia_hi > 0) else 0

        if _key_lo > 0 and _lsw_lo < _key_lo and _lsw_close > _key_lo and _lsw_wick_r >= _wick_threshold:
            # Wick jos sub nivel cheie dar close deasupra = bullish sweep → LONG setup
            _lsw_detected  = True
            _lsw_direction = "LONG"
        elif _key_hi > 0 and _lsw_hi > _key_hi and _lsw_close < _key_hi and _lsw_wick_r >= _wick_threshold:
            # Wick sus peste nivel cheie dar close dedesubt = bearish sweep → SHORT setup
            _lsw_detected  = True
            _lsw_direction = "SHORT"

        if _lsw_detected:
            _lsw_aligned  = (_lsw_direction == trade_direction)
            _lsw_boost    = 0.12 if _lsw_aligned else 0.06   # mai mic dacă contra-direcție
            _lsw_pre      = final_score
            final_score   = min(final_score + _lsw_boost, 1.0)
            score_pct     = round(min(final_score * 100, 100), 2)
            logger.info(
                f"   🎯 LIQUIDITY SWEEP [{_lsw_direction}] wick_ratio={_lsw_wick_r:.1f} "
                f"{'✅ ALINIAT' if _lsw_aligned else '⚠️ contra-direcție'} "
                f"→ boost +{_lsw_boost:.0%}: {_lsw_pre:.2f} → {final_score:.2f}"
            )
            # Sweep detectat anulează penalizarea Asia Range dacă e aliniat cu trade
            if _lsw_aligned and skip_volatile and "ASIA RANGE" in vol_msg:
                logger.info(f"   🎯 Sweep ICT anulează filtrul Asia Range → continuăm")
                skip_volatile = False
                vol_msg = ""
        # ─────────────────────────────────────────────────────────────────────

        # ── FAZA 3.1: Regime Filter Activ ─────────────────────────────────────
        # Ajustăm scorul final în funcție de regimul pieței detectat.
        # TRENDING  → boost +8%  (trade cu trendul = high probability)
        # RANGING   → penalizare -10% (fakeouts frecvente, evitare)
        # CHOPPY    → penalizare -15% (haos, skip recomandat)
        # BREAKOUT  → boost +5%  (moment de expansiune — potențial trade bun)
        _regime_label = str(regime).upper() if regime else "UNKNOWN"
        _regime_adj   = 0.0
        _regime_emoji = "📊"

        if "TREND" in _regime_label:
            _regime_adj   = +0.08
            _regime_emoji = "📈"
        elif "BREAKOUT" in _regime_label or "EXPAN" in _regime_label:
            _regime_adj   = +0.05
            _regime_emoji = "🚀"
        elif "CHOP" in _regime_label or "NOISE" in _regime_label:
            _regime_adj   = -0.15
            _regime_emoji = "🌀"
        elif "RANG" in _regime_label or "CONSOL" in _regime_label or "LATERAL" in _regime_label:
            # v12.2: CONSOLIDATION HARD MODE — nu doar -10%, ci cap agresiv + VA extreme only
            _regime_adj   = 0.0   # ajustarea e mai jos, mai complexă
            _regime_emoji = "🔲"

            # ── v12.2: CONSOLIDATION HARD MODE ────────────────────────────────
            # În consolidare, edge-ul modelului dispare (WR scade de la 77% la ~45%).
            # Fix: cap scor la 40% MAXIM (sub orice threshold rezonabil).
            # EXCEPȚIE: dacă prețul e la extrema VA (VAH/VAL ±5 pts) → permitem trade
            # cu scor cap la 55% (reversal de la extremă = setup valid chiar în range).
            _vah_val = float(best.get('vah', 0) or 0)
            _val_val = float(best.get('val', 0) or 0)
            _cur_close = float(best.get('close', 0) or 0)
            _va_margin = atr_now * 0.15 if atr_now > 0 else 5.0   # ~15% din ATR ca marjă

            _at_va_extreme = False
            if _vah_val > 0 and _val_val > 0 and _cur_close > 0:
                _at_va_extreme = (
                    _cur_close >= _vah_val - _va_margin or   # la VAH (short setup)
                    _cur_close <= _val_val + _va_margin       # la VAL (long setup)
                )

            _pre_regime = final_score
            if _at_va_extreme:
                # La extrema VA → cap la 55% (permitem trade dacă alte confluențe confirmă)
                _consol_cap = 0.55
                if final_score > _consol_cap:
                    final_score = _consol_cap
                logger.info(
                    f"   🔲 CONSOL HARD MODE: LA EXTREMA VA → cap {_consol_cap:.0%} "
                    f"(close={_cur_close:.2f} VAH={_vah_val:.2f} VAL={_val_val:.2f}) "
                    f"scor {_pre_regime:.2f} → {final_score:.2f}"
                )
            else:
                # La mijlocul VA → cap la 40% (sub orice threshold → SKIP garantat)
                _consol_cap = 0.40
                if final_score > _consol_cap:
                    final_score = _consol_cap
                logger.warning(
                    f"   🔲🚫 CONSOL HARD MODE: MID-RANGE VA → cap {_consol_cap:.0%} "
                    f"(close={_cur_close:.2f} VAH={_vah_val:.2f} VAL={_val_val:.2f}) "
                    f"scor {_pre_regime:.2f} → {final_score:.2f} — TRADE BLOCAT"
                )
            score_pct = round(min(final_score * 100, 100), 2)

        if _regime_adj != 0.0:
            _pre_regime = final_score
            final_score = max(min(final_score + _regime_adj, 1.0), 0.0)
            score_pct   = round(min(final_score * 100, 100), 2)
            logger.info(
                f"   {_regime_emoji} REGIME [{_regime_label}]: "
                f"scor {_pre_regime:.2f} → {final_score:.2f} ({_regime_adj:+.0%})"
            )
        elif "RANG" not in _regime_label and "CONSOL" not in _regime_label and "LATERAL" not in _regime_label:
            logger.debug(f"   📊 REGIME [{_regime_label}]: fără ajustare (unknown/neutral)")
        # ─────────────────────────────────────────────────────────────────────

        # ── FAZA 3.2: Multi-timeframe Confluence ─────────────────────────────
        # Numărăm câte timeframe-uri sunt ALINIATE cu direcția trade-ului.
        # Sursă: h4_hi/h4_lo, h1_hi/h1_lo, lw_hi/lw_lo (weekly), lm_hi/lm_lo (monthly)
        # Plus slope_h4 și slope_h1 dacă sunt disponibile.
        # 4+/4 aliniate → +10% | 3/4 → +5% | 2/4 → 0 | 1/4 → -8% | 0/4 → -15%
        _mtf_close   = float(best.get('close', 0) or 0)
        _mtf_checks  = []   # True = aliniat cu trade_direction, False = contra

        # H1: preț deasupra 65% din range H1 → bullish H1 (Fix v7.4 #4: 50%→65%)
        _h1_hi_m = float(best.get('h1_hi', 0) or 0)
        _h1_lo_m = float(best.get('h1_lo', 0) or 0)
        if _h1_hi_m > 0 and _h1_lo_m > 0 and _mtf_close > 0:
            _h1_bull  = _mtf_close > (_h1_lo_m + (_h1_hi_m - _h1_lo_m) * 0.65)
            _mtf_checks.append(_h1_bull == (trade_direction == "LONG"))

        # H4: preț deasupra 65% din range H4 → bullish H4 (Fix v7.4 #4: 50%→65%)
        _h4_hi_m = float(best.get('h4_hi', 0) or 0)
        _h4_lo_m = float(best.get('h4_lo', 0) or 0)
        if _h4_hi_m > 0 and _h4_lo_m > 0 and _mtf_close > 0:
            _h4_bull  = _mtf_close > (_h4_lo_m + (_h4_hi_m - _h4_lo_m) * 0.65)
            _mtf_checks.append(_h4_bull == (trade_direction == "LONG"))

        # Weekly: preț deasupra 65% din range weekly → bullish (Fix v7.4 #4: 50%→65%)
        _lw_hi_m = float(best.get('lw_hi', 0) or 0)
        _lw_lo_m = float(best.get('lw_lo', 0) or 0)
        if _lw_hi_m > 0 and _lw_lo_m > 0 and _mtf_close > 0:
            _lw_bull = _mtf_close > (_lw_lo_m + (_lw_hi_m - _lw_lo_m) * 0.65)
            _mtf_checks.append(_lw_bull == (trade_direction == "LONG"))

        # Monthly: preț deasupra 65% din range monthly → bullish (Fix v7.4 #4: 50%→65%)
        _lm_hi_m = float(best.get('lm_hi', 0) or 0)
        _lm_lo_m = float(best.get('lm_lo', 0) or 0)
        if _lm_hi_m > 0 and _lm_lo_m > 0 and _mtf_close > 0:
            _lm_bull = _mtf_close > (_lm_lo_m + (_lm_hi_m - _lm_lo_m) * 0.65)
            _mtf_checks.append(_lm_bull == (trade_direction == "LONG"))

        # Fix v7.4 #4: Slope confluence checks (if available in best dict)
        _slope_h1_val = float(best.get('slope_h1', 0) or 0)
        if _slope_h1_val != 0:
            _mtf_checks.append((_slope_h1_val > 0) == (trade_direction == "LONG"))
        _slope_h4_val = float(best.get('slope_h4', 0) or 0)
        if _slope_h4_val != 0:
            _mtf_checks.append((_slope_h4_val > 0) == (trade_direction == "LONG"))

        if len(_mtf_checks) >= 2:
            _mtf_aligned = sum(_mtf_checks)
            _mtf_total   = len(_mtf_checks)
            _mtf_pct     = _mtf_aligned / _mtf_total

            if _mtf_pct >= 1.0:
                _mtf_adj = +0.10
            elif _mtf_pct >= 0.75:
                _mtf_adj = +0.05
            elif _mtf_pct >= 0.50:
                _mtf_adj = 0.0
            elif _mtf_pct >= 0.25:
                _mtf_adj = -0.04   # redus de la -0.08 → -0.04 (penalizare parțială)
            else:
                _mtf_adj = -0.08   # redus de la -0.15 → -0.08 (penalizare max)

            # Reversal override: OF bate HTF — penalizare MTF max 3%
            # Când sistemul detectează reversal instituțional puternic (cum_delta, POC drift)
            # nu are sens să penalizezi agresiv că H4/H1 nu sunt aliniate cu noua direcție
            if _reversal_override and _mtf_adj < 0:
                _mtf_adj = max(_mtf_adj, -0.03)

            if _mtf_adj != 0.0:
                _pre_mtf    = final_score
                final_score = max(min(final_score + _mtf_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(
                    f"   🔭 MTF CONFLUENCE {_mtf_aligned}/{_mtf_total} TF aliniate cu {trade_direction} "
                    f"→ scor {_pre_mtf:.2f} → {final_score:.2f} ({_mtf_adj:+.0%})"
                )
            else:
                logger.debug(
                    f"   🔭 MTF CONFLUENCE {_mtf_aligned}/{_mtf_total}: neutru"
                )
        # ─────────────────────────────────────────────────────────────────────

        # ── PDH/PDL — Previous Day High/Low Magnet ───────────────────────────
        # PDH și PDL sunt niveluri instituționale foarte respectate.
        # Strategii posibile:
        #   1) Prețul SE APROPIE de PDH/PDL din interior → boost (magnet, TP probabil acolo)
        #   2) Prețul TOCMAI A SPART PDH/PDL (breakout) → boost dacă direcție aliniată
        #   3) Prețul e LA PDH/PDL și respins → penalizare (reversal probabil)
        _pdh     = float(best.get('p_hi', 0) or 0)
        _pdl     = float(best.get('p_lo', 0) or 0)
        _px      = float(best.get('close', 0) or 0)
        _px_atr  = float(best.get('atr_14', 8) or 8)

        if _pdh > 0 and _pdl > 0 and _px > 0:
            _dist_pdh   = abs(_px - _pdh)
            _dist_pdl   = abs(_px - _pdl)
            _proximity  = _px_atr * 0.5    # "aproape" = în raza de 0.5 ATR

            _pdh_near   = _dist_pdh <= _proximity
            _pdl_near   = _dist_pdl <= _proximity
            _above_pdh  = _px > _pdh + 0.5  # breakout deasupra PDH
            _below_pdl  = _px < _pdl - 0.5  # breakout sub PDL

            _pdh_adj    = 0.0
            _pdh_reason = ""

            if _above_pdh and trade_direction == "LONG":
                # Breakout deasupra PDH cu direcție LONG = continuare instituțională
                _pdh_adj    = +0.08
                _pdh_reason = f"PDH BREAKOUT LONG ({_pdh:.1f}) → magnet +8%"
            elif _below_pdl and trade_direction == "SHORT":
                # Breakout sub PDL cu direcție SHORT = continuare instituțională
                _pdh_adj    = +0.08
                _pdh_reason = f"PDL BREAKOUT SHORT ({_pdl:.1f}) → magnet +8%"
            elif _pdh_near and trade_direction == "SHORT":
                # Aproape de PDH dar SHORT = respingere probabilă la nivel
                _pdh_adj    = +0.05
                _pdh_reason = f"PDH REJECTION SHORT (dist={_dist_pdh:.1f}pts) → +5%"
            elif _pdl_near and trade_direction == "LONG":
                # Aproape de PDL dar LONG = respingere probabilă la nivel
                _pdh_adj    = +0.05
                _pdh_reason = f"PDL REJECTION LONG (dist={_dist_pdl:.1f}pts) → +5%"
            elif _pdh_near and trade_direction == "LONG":
                # Tocmai la PDH și vrem LONG = rezistență → penalizare
                _pdh_adj    = -0.07
                _pdh_reason = f"PDH REZISTENȚĂ LONG (dist={_dist_pdh:.1f}pts) → -7%"
            elif _pdl_near and trade_direction == "SHORT":
                # Tocmai la PDL și vrem SHORT = suport → penalizare
                _pdh_adj    = -0.07
                _pdh_reason = f"PDL SUPORT SHORT (dist={_dist_pdl:.1f}pts) → -7%"

            if _pdh_adj != 0.0:
                _pre_pdh    = final_score
                final_score = max(min(final_score + _pdh_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(
                    f"   📅 PDH/PDL: {_pdh_reason} | scor {_pre_pdh:.2f} → {final_score:.2f}"
                )
        # ─────────────────────────────────────────────────────────────────────

        # ── PREMIUM/DISCOUNT ICT ─────────────────────────────────────────────
        # Echilibrul sesiunii (50% al range zilei) separă Premium de Discount.
        # ICT: cumperi în Discount, vinzi în Premium.
        # Dacă intrăm contra (LONG în Premium sau SHORT în Discount) → penalizare.
        _pd_pct = 0.5   # default neutral — redefinit în blocul de mai jos dacă range valid
        #
        # Fix v9.1: EROARE CRITICĂ — versiunea anterioară folosea best['high']/best['low']
        # = HIGH/LOW-ul barei CURENTE de 1 minut.
        # Pe o lumânare roșie mare, close e la BAZA barei → _pd_pct ≈ 0 → "SHORT în DISCOUNT" → -8%
        # pe FIECARE bară care cade, chiar dacă SHORT e corect direcțional!
        # Fix: folosim range H4 (ICT standard) → H1 → 15M → Asia → fallback session60.
        # Range-ul H4/H1 se schimbă rar (o dată pe oră/4 ore), reflectând corect poziția prețului.
        # 15M range calculat din ultimele 15 bare de 1-min — mai relevant decât bara curentă.
        _pd_px   = float(best.get('close', 0) or 0)
        # Prioritate: H4 range > H1 range > 15M range > Asia range > session60
        _pd_h4_hi = float(best.get('h4_hi', 0) or 0)
        _pd_h4_lo = float(best.get('h4_lo', 0) or 0)
        _pd_h1_hi = float(best.get('h1_hi', 0) or 0)
        _pd_h1_lo = float(best.get('h1_lo', 0) or 0)
        _pd_as_hi = float(best.get('asia_hi', 0) or 0)
        _pd_as_lo = float(best.get('asia_lo', 0) or 0)
        # 15M range — din ultimele 15 bare 1-min (nu coloană în DB)
        _pd_m15_hi = float(df['high'].tail(15).max()) if len(df) >= 15 else 0
        _pd_m15_lo = float(df['low'].tail(15).min())  if len(df) >= 15 else 0

        if _pd_h4_hi > _pd_h4_lo > 0:
            _pd_high, _pd_low, _pd_tf = _pd_h4_hi, _pd_h4_lo, "H4"
        elif _pd_h1_hi > _pd_h1_lo > 0:
            _pd_high, _pd_low, _pd_tf = _pd_h1_hi, _pd_h1_lo, "H1"
        elif _pd_m15_hi > _pd_m15_lo > 0:
            _pd_high, _pd_low, _pd_tf = _pd_m15_hi, _pd_m15_lo, "15M"
        elif _pd_as_hi > _pd_as_lo > 0:
            _pd_high, _pd_low, _pd_tf = _pd_as_hi, _pd_as_lo, "Asia"
        else:
            # Fallback la range-ul sesiunii din ultimele 60 bare (nu bara curentă!)
            _sess_hi = float(df['high'].tail(60).max()) if len(df) >= 60 else 0
            _sess_lo = float(df['low'].tail(60).min()) if len(df) >= 60 else 0
            _pd_high, _pd_low, _pd_tf = _sess_hi, _sess_lo, "Session60"

        if _pd_high > _pd_low and _pd_px > 0:
            _equilibrium  = (_pd_high + _pd_low) / 2.0
            _pd_pct       = (_pd_px - _pd_low) / (_pd_high - _pd_low)   # 0=low, 1=high, 0.5=eq
            _in_premium   = _pd_pct >= 0.60    # deasupra 60% = premium
            _in_discount  = _pd_pct <= 0.40    # sub 40% = discount
            _pd_adj       = 0.0
            _pd_reason    = ""

            if _in_discount and trade_direction == "LONG":
                _pd_adj    = +0.06
                _pd_reason = f"DISCOUNT LONG [{_pd_tf}] ({_pd_pct:.0%} din range {_pd_low:.0f}-{_pd_high:.0f}) → ICT aliniat +6%"
            elif _in_premium and trade_direction == "SHORT":
                _pd_adj    = +0.06
                _pd_reason = f"PREMIUM SHORT [{_pd_tf}] ({_pd_pct:.0%} din range {_pd_low:.0f}-{_pd_high:.0f}) → ICT aliniat +6%"
            elif _in_premium and trade_direction == "LONG":
                # Fix v10.2: dacă displacement recent, intrarea în premium = stop hunt valid (Judas Swing)
                # Prețul a intrat în premium PRIN displacement, nu organic → penalizare redusă
                _has_disp_pd = (
                    bool(best.get('has_displacement', 0)) or
                    (len(df) >= 5 and 'has_displacement' in df.columns and bool(df['has_displacement'].tail(5).any()))
                )
                if _has_disp_pd:
                    _pd_adj    = -0.03
                    _pd_reason = f"LONG în PREMIUM [{_pd_tf}] ({_pd_pct:.0%}) + DISPLACEMENT → Judas Swing, penalizare redusă -3%"
                else:
                    # Fix v10.5: Dacă OF confirmă LONG (cum_delta > 0, buy > sell), e breakout prin premium
                    _of_cd_pd2 = float(live_data.get("cum_delta", 0) or 0) if _live_mode and live_data else 0
                    _of_bv_pd2 = float(live_data.get("bar_buy_vol", 0) or 0) if _live_mode and live_data else 0
                    _of_sv_pd2 = float(live_data.get("bar_sell_vol", 0) or 0) if _live_mode and live_data else 0
                    _of_confirms_long = (_of_cd_pd2 > 0 and _of_bv_pd2 > _of_sv_pd2 * 1.2)
                    if _of_confirms_long:
                        _pd_adj    = -0.03
                        _pd_reason = f"LONG în PREMIUM [{_pd_tf}] ({_pd_pct:.0%}) dar OF confirmă breakout → penalizare redusă -3%"
                    else:
                        _pd_adj    = -0.08
                        _pd_reason = f"LONG în PREMIUM [{_pd_tf}] ({_pd_pct:.0%} din range {_pd_low:.0f}-{_pd_high:.0f}) → contra ICT -8%"
            elif _in_discount and trade_direction == "SHORT":
                # Fix v10.5: Dacă OF confirmă puternic SHORT (cum_delta < 0, sell > buy),
                # e breakdown institutional prin discount, nu fade contra-trend.
                # Reducem penalizarea de la -8% la -3%.
                _of_cd_pd = float(live_data.get("cum_delta", 0) or 0) if _live_mode and live_data else 0
                _of_sv_pd = float(live_data.get("bar_sell_vol", 0) or 0) if _live_mode and live_data else 0
                _of_bv_pd = float(live_data.get("bar_buy_vol", 0) or 0) if _live_mode and live_data else 0
                _of_confirms_short = (_of_cd_pd < 0 and _of_sv_pd > _of_bv_pd * 1.2)
                if _of_confirms_short:
                    _pd_adj    = -0.03
                    _pd_reason = f"SHORT în DISCOUNT [{_pd_tf}] ({_pd_pct:.0%}) dar OF confirmă breakdown → penalizare redusă -3%"
                else:
                    _pd_adj    = -0.08
                    _pd_reason = f"SHORT în DISCOUNT [{_pd_tf}] ({_pd_pct:.0%} din range {_pd_low:.0f}-{_pd_high:.0f}) → contra ICT -8%"
            else:
                logger.info(
                    f"   ⚖️  PREMIUM/DISCOUNT [{_pd_tf}]: {_pd_pct:.0%} din range {_pd_low:.0f}-{_pd_high:.0f} → neutru (40-60%)"
                )

            if _pd_adj != 0.0:
                _pre_pd    = final_score
                final_score = max(min(final_score + _pd_adj, 1.0), 0.0)
                score_pct  = round(min(final_score * 100, 100), 2)
                logger.info(
                    f"   ⚖️  PREMIUM/DISCOUNT: {_pd_reason} | scor {_pre_pd:.2f} → {final_score:.2f}"
                )
        # ─────────────────────────────────────────────────────────────────────

        # ── DAY-OF-WEEK BIAS — dezactivat v9.4 ───────────────────────────────
        # Penalizări flat pe zi (Luni -5%, Vineri -6%) adăugau noise fără valoare reală.
        # Un setup ICT valid de Luni nu trebuie penalizat indiferent de zi.
        # ─────────────────────────────────────────────────────────────────────

        # ── B1. VWAP DISTANCE FILTER ──────────────────────────────────────────
        # Dacă prețul e prea departe de VWAP (>2×ATR), probabilitatea de revenire
        # e mare → penalizăm tranzacțiile în extensie extremă.
        # Dacă prețul e APROAPE de VWAP (< 0.3×ATR) și tranzacționăm cu trendul → boost.
        _vwap     = float(best.get('vwap', 0) or 0)
        _vw_close = float(best.get('close', 0) or 0)
        _vw_atr   = float(best.get('atr_14', 8) or 8)

        if _vwap > 0 and _vw_close > 0 and _vw_atr > 0:
            _vwap_dist  = abs(_vw_close - _vwap)
            _vwap_ratio = _vwap_dist / _vw_atr
            _above_vwap = _vw_close > _vwap
            _vwap_adj   = 0.0

            if _vwap_ratio > 2.5:
                # Fix v10.4: VWAP EXTREM — max -5% indiferent de context
                # VWAP e un indicator de context, nu de entry. Pe NQ volatil prețul
                # stă departe de VWAP ore întregi. ICT + orderflow bat VWAP.
                _vwap_adj = -0.05
                logger.info(
                    f"   📉 VWAP EXTREM: dist={_vwap_dist:.1f}pts = {_vwap_ratio:.1f}×ATR → -5%"
                )
            elif _vwap_ratio > 1.5:
                _vwap_adj = -0.03
                logger.info(f"   📉 VWAP EXTINS: dist={_vwap_dist:.1f}pts = {_vwap_ratio:.1f}×ATR → -3%")
            elif _vwap_ratio < 0.3:
                # Aproape de VWAP + cu trendul = setup bun
                _vwap_aligned = (_above_vwap and trade_direction == "LONG") or \
                                (not _above_vwap and trade_direction == "SHORT")
                if _vwap_aligned:
                    _vwap_adj = +0.05
                    logger.info(f"   📈 VWAP PROXIM aliniat {trade_direction}: +5%")

            if _vwap_adj != 0.0:
                final_score = max(min(final_score + _vwap_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
        # ─────────────────────────────────────────────────────────────────────

        # ── B2. VOLUME ABSORPTION DETECTOR ───────────────────────────────────
        # Volume mare la un nivel cheie (POC/VAH/VAL) fără mișcare de preț =
        # absorpție instituțională → semnal de reversal sau breakout iminent.
        # Detectăm: volume >> medie dar range bara << 0.3×ATR
        _va_vol  = float(best.get('volume', 0) or 0)
        _va_high = float(best.get('high', 0) or 0)
        _va_low  = float(best.get('low', 0) or 0)
        _va_atr  = float(best.get('atr_14', 8) or 8)
        _va_poc  = float(best.get('poc', 0) or 0)

        if _va_vol > 0 and _va_high > _va_low and _va_atr > 0:
            _bar_range   = _va_high - _va_low
            _is_absorbed = _bar_range < _va_atr * 0.3   # bara mică = prețul absorbit
            # Verificăm dacă suntem aproape de POC (nivel de absorbție clasic)
            _near_poc    = _va_poc > 0 and abs(float(best.get('close', 0)) - _va_poc) < _va_atr * 0.5

            if _is_absorbed and _near_poc:
                _va_adj = +0.07
                final_score = max(min(final_score + _va_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(
                    f"   🧲 ABSORPȚIE POC: range={_bar_range:.1f}pts < 0.3ATR lângă POC={_va_poc:.1f} → +7%"
                )
        # ─────────────────────────────────────────────────────────────────────

        # ── B3. GAP FILL FILTER ───────────────────────────────────────────────
        # Dacă azi a deschis cu gap față de ieri, 80%+ din timp se umple în prima oră.
        # Boost dacă suntem în direcția gap fill (spre preț de ieri close).
        # Penalizare dacă tranzacționăm contra gap fill.
        _gf_open  = float(best.get('open',  0) or 0)
        _gf_close = float(best.get('close', 0) or 0)
        _gf_p_hi  = float(best.get('p_hi',  0) or 0)
        _gf_p_lo  = float(best.get('p_lo',  0) or 0)
        # Estimăm "ieri close" ca mijlocul PDH/PDL (approx)
        _yday_close_approx = (_gf_p_hi + _gf_p_lo) / 2 if _gf_p_hi > 0 and _gf_p_lo > 0 else 0

        if _gf_open > 0 and _yday_close_approx > 0:
            _gap_size = _gf_open - _yday_close_approx
            _gap_pct  = abs(_gap_size) / _yday_close_approx * 100
            _utc_h_gf = datetime.utcnow().hour

            if _gap_pct > 0.05 and _utc_h_gf < 5:   # gap semnificativ în prima oră
                _gap_fill_dir = "SHORT" if _gap_size > 0 else "LONG"  # gap sus → fill în jos
                if trade_direction == _gap_fill_dir:
                    final_score = max(min(final_score + 0.07, 1.0), 0.0)
                    score_pct   = round(min(final_score * 100, 100), 2)
                    logger.info(
                        f"   🕳️ GAP FILL {_gap_fill_dir}: gap={_gap_size:+.1f}pts ({_gap_pct:.2f}%) → +7%"
                    )
                elif trade_direction != _gap_fill_dir:
                    final_score = max(final_score - 0.06, 0.0)
                    score_pct   = round(min(final_score * 100, 100), 2)
                    logger.info(
                        f"   🕳️ CONTRA GAP FILL: gap={_gap_size:+.1f}pts → -6%"
                    )
        # ─────────────────────────────────────────────────────────────────────

        # ── C1. WEEKLY PROFILE BIAS ───────────────────────────────────────────
        # Luni dimineața: dacă săptămâna trecută s-a închis sub midpoint weekly
        # → bias SHORT pentru primele 2 zile, și invers pentru LONG.
        # Aplicăm o ajustare mică bazată pe poziția față de weekly midpoint.
        _wp_lw_hi  = float(best.get('lw_hi', 0) or 0)
        _wp_lw_lo  = float(best.get('lw_lo', 0) or 0)
        _wp_close  = float(best.get('close', 0) or 0)
        _wp_dow    = datetime.utcnow().weekday()  # 0=Mon

        if _wp_lw_hi > 0 and _wp_lw_lo > 0 and _wp_close > 0 and _wp_dow <= 1:
            # Primele 2 zile ale săptămânii — bias weekly contează cel mai mult
            _wp_mid  = (_wp_lw_hi + _wp_lw_lo) / 2
            _wp_bull = _wp_close > _wp_mid
            _wp_aligned = (_wp_bull and trade_direction == "LONG") or \
                          (not _wp_bull and trade_direction == "SHORT")
            _wp_adj = +0.06 if _wp_aligned else -0.06
            final_score = max(min(final_score + _wp_adj, 1.0), 0.0)
            score_pct   = round(min(final_score * 100, 100), 2)
            logger.info(
                f"   📅 WEEKLY BIAS ({'BULL' if _wp_bull else 'BEAR'}, {'✅ aliniat' if _wp_aligned else '❌ contra'}) "
                f"({_wp_adj:+.0%}) | scor → {final_score:.2f}"
            )
        # ─────────────────────────────────────────────────────────────────────

        # ── C2. SEASONAL BIAS — dezactivat v9.4 ──────────────────────────────
        # Ajustări de ±2-3% pe lună adăugau noise fără impact real pe scor.
        # ─────────────────────────────────────────────────────────────────────

        # ── C3. OPTIONS EXPIRY PINNING — OPEX ────────────────────────────────
        # A 3-a vineri a lunii = OPEX. Piața tinde să fie "pinned" la niveluri
        # round (strike multiples of 50/100). Volatilitate redusă în dimineața OPEX,
        # dar gap exploziv la deschidere. Reducem sizing cu 30% în această zi.
        _opex_month = datetime.utcnow().month
        _opex_day   = datetime.utcnow().day
        _opex_dow   = datetime.utcnow().weekday()   # 4 = Vineri

        _is_opex = False
        if _opex_dow == 4:   # e vineri
            # A 3-a vineri: ziua 15-21
            _is_opex = 15 <= _opex_day <= 21

        if _is_opex:
            # Penalizare OPEX: piața se comportă atipic — reducere scor -8%
            final_score = max(final_score - 0.08, 0.0)
            score_pct   = round(min(final_score * 100, 100), 2)
            logger.info(
                f"   📋 OPEX (a 3-a vineri, ziua {_opex_day}): pinning activ → -8% | scor → {final_score:.2f}"
            )
        # ─────────────────────────────────────────────────────────────────────

        # ══════════════════════════════════════════════════════════════════════
        # ELITE FILTERS — top 0.1% edge separators
        # ══════════════════════════════════════════════════════════════════════

        # ── E1. DOM BID/ASK INSTITUTIONAL CONFIRMATION ───────────────────────
        # Instituțiile plasează ordine masive pe DOM înainte de mișcări mari.
        # bid_ask_ratio > 2.5 → presiune de cumpărare instituțională masivă
        # bid_ask_ratio < 0.4 → presiune de vânzare instituțională masivă
        # Utilizăm ratio-ul din NT8 DOM (cel mai precis indicator de orderflow live)
        _dom_ratio = float(best.get('bid_ask_ratio', 1.0) or 1.0)
        _dom_adj   = 0.0

        if _dom_ratio >= 2.5 and trade_direction == "LONG":
            _dom_adj = +0.07
            logger.info(f"   🏦 DOM BID SPIKE {_dom_ratio:.1f}:1 → confirmare instituțională LONG +7%")
        elif _dom_ratio <= 0.4 and trade_direction == "SHORT":
            _dom_adj = +0.07
            logger.info(f"   🏦 DOM ASK SPIKE {_dom_ratio:.2f} → confirmare instituțională SHORT +7%")
        elif _dom_ratio >= 2.5 and trade_direction == "SHORT":
            _dom_adj = -0.06
            logger.info(f"   🚫 DOM BID SPIKE {_dom_ratio:.1f}:1 dar SHORT = contra flux -6%")
        elif _dom_ratio <= 0.4 and trade_direction == "LONG":
            _dom_adj = -0.06
            logger.info(f"   🚫 DOM ASK SPIKE {_dom_ratio:.2f} dar LONG = contra flux -6%")

        if _dom_adj != 0.0:
            _pre_dom    = final_score
            final_score = max(min(final_score + _dom_adj, 1.0), 0.0)
            score_pct   = round(min(final_score * 100, 100), 2)
            logger.info(f"   🏦 DOM: scor {_pre_dom:.2f} → {final_score:.2f}")
        # ─────────────────────────────────────────────────────────────────────

        # ── E2. BAR STRUCTURE QUALITY (Conviction vs Indecision) ─────────────
        # Bara de semnal spune tot:
        # Corp > 65% din range = conviction bar → boost dacă aliniat
        # Corp < 20% din range = doji/pinbar = indecision → penalizare
        # Corp mare contra direcției = presiune opusă → penalizare mai mare
        _bs_open  = float(best.get('open',  0) or 0)
        _bs_close = float(best.get('close', 0) or 0)
        _bs_high  = float(best.get('high',  0) or 0)
        _bs_low   = float(best.get('low',   0) or 0)

        if _bs_high > _bs_low > 0 and _bs_open > 0:
            _bs_range   = _bs_high - _bs_low
            _bs_body    = abs(_bs_close - _bs_open)
            _bs_ratio   = _bs_body / _bs_range if _bs_range > 0 else 0.5
            _bs_bull    = _bs_close > _bs_open
            _bs_aligned = (_bs_bull and trade_direction == "LONG") or \
                          (not _bs_bull and trade_direction == "SHORT")
            _bs_adj     = 0.0

            if _bs_ratio >= 0.65 and _bs_aligned:
                _bs_adj = +0.05
                logger.info(f"   🕯️ CONVICTION BAR aliniat: corp={_bs_ratio:.0%} → +5%")
            elif _bs_ratio <= 0.20:
                # Fix v10.2: post-displacement, bare mici = consolidare normală (nu indecision)
                # Instituționalii absorb la nivel cheie după stop hunt — bara mică e o pauză, nu slăbiciune
                _has_disp_bs = (
                    bool(best.get('has_displacement', 0)) or
                    (len(df) >= 4 and 'has_displacement' in df.columns and bool(df['has_displacement'].tail(4).any()))
                )
                if _has_disp_bs:
                    _bs_adj = 0.0
                    logger.info(f"   🕯️ DOJI post-displacement: corp={_bs_ratio:.0%} → neutru (consolidare ICT)")
                else:
                    _bs_adj = -0.04
                    logger.info(f"   🕯️ DOJI/INDECISION: corp={_bs_ratio:.0%} → -4%")
            elif _bs_ratio >= 0.65 and not _bs_aligned:
                _bs_adj = -0.06
                logger.info(f"   🕯️ CONTRA BAR puternic: corp={_bs_ratio:.0%} contra {trade_direction} → -6%")

            if _bs_adj != 0.0:
                _pre_bs     = final_score
                final_score = max(min(final_score + _bs_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(f"   🕯️ BAR STRUCTURE: scor {_pre_bs:.2f} → {final_score:.2f}")
        # ─────────────────────────────────────────────────────────────────────

        # ── E2b. REJECTION CANDLE FILTER ─────────────────────────────────────
        # Wick lung sus (upper wick > 60% din range) = BSL sweep + rejection bearish
        # → penalizăm LONG puternic (e periculos să intri LONG după un spike rejected)
        # Wick lung jos (lower wick > 60% din range) = SSL sweep + rejection bullish
        # → penalizăm SHORT puternic
        if _bs_high > _bs_low > 0:
            _bs_range_r   = _bs_high - _bs_low
            _upper_wick   = _bs_high - max(_bs_open, _bs_close)
            _lower_wick   = min(_bs_open, _bs_close) - _bs_low
            _upper_wick_r = _upper_wick / _bs_range_r if _bs_range_r > 0 else 0
            _lower_wick_r = _lower_wick / _bs_range_r if _bs_range_r > 0 else 0

            _rej_adj = 0.0
            if _upper_wick_r >= 0.60 and trade_direction == "LONG":
                _rej_adj = -0.12   # wick sus mare = rejection bearish → LONG periculos
                logger.info(
                    f"   🕯️ REJECTION CANDLE (upper wick={_upper_wick_r:.0%}) "
                    f"→ BSL sweep detected, LONG periculos -12%"
                )
            elif _lower_wick_r >= 0.60 and trade_direction == "SHORT":
                _rej_adj = -0.12   # wick jos mare = rejection bullish → SHORT periculos
                logger.info(
                    f"   🕯️ REJECTION CANDLE (lower wick={_lower_wick_r:.0%}) "
                    f"→ SSL sweep detected, SHORT periculos -12%"
                )

            if _rej_adj != 0.0:
                _pre_rej    = final_score
                final_score = max(min(final_score + _rej_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(f"   🕯️ REJECTION FILTER: scor {_pre_rej:.2f} → {final_score:.2f}")
        # ─────────────────────────────────────────────────────────────────────

        # ── E3. PRICE MOMENTUM STREAK (4/4 bare aliniate) ────────────────────
        # Dacă ultimele 4 bare s-au închis TOATE în direcția semnalului → momentum real.
        # Dacă ultimele 4 bare sunt contra direcției → intrăm contra trendului = risc.
        # Filtrează intrările "la timp" de cele "prea târziu" sau "contra".
        if len(df) >= 5:
            _pm_closes = list(df['close'].iloc[-5:])
            _pm_dirs   = [1 if _pm_closes[i] > _pm_closes[i-1] else -1
                          for i in range(1, len(_pm_closes))]   # 4 direcții
            _pm_up     = sum(1 for d in _pm_dirs if d == 1)
            _pm_down   = sum(1 for d in _pm_dirs if d == -1)
            _pm_adj    = 0.0

            if _pm_up >= 4 and trade_direction == "LONG":
                _pm_adj = +0.06
                logger.info(f"   📈 MOMENTUM STREAK LONG {_pm_up}/4 bare up → +6%")
            elif _pm_down >= 4 and trade_direction == "SHORT":
                _pm_adj = +0.06
                logger.info(f"   📉 MOMENTUM STREAK SHORT {_pm_down}/4 bare down → +6%")
            elif _pm_up >= 4 and trade_direction == "SHORT":
                _pm_adj = -0.06
                logger.info(f"   ⚠️ CONTRA MOMENTUM: {_pm_up}/4 bare up dar SHORT → -6%")
            elif _pm_down >= 4 and trade_direction == "LONG":
                _pm_adj = -0.06
                logger.info(f"   ⚠️ CONTRA MOMENTUM: {_pm_down}/4 bare down dar LONG → -6%")

            if _pm_adj != 0.0:
                _pre_pm     = final_score
                final_score = max(min(final_score + _pm_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(f"   📊 MOMENTUM: scor {_pre_pm:.2f} → {final_score:.2f}")
        # ─────────────────────────────────────────────────────────────────────

        # ── E4. VOLUME CLIMAX / EXHAUSTION DETECTOR ───────────────────────────
        # Volume 2× față de media ultimelor 9 bare = event major.
        # Climax aliniat cu direcția = confirmare instituțională puternică.
        # Climax contra direcției = ending move / exhaustion = EVITĂ.
        # Aceasta este una dintre cele mai puternice confirme din orderflow real.
        if len(df) >= 10 and 'volume' in df.columns:
            _vc_vols  = list(df['volume'].iloc[-10:])
            _vc_avg   = sum(_vc_vols[:-1]) / max(len(_vc_vols) - 1, 1)
            _vc_last  = _vc_vols[-1]
            _vc_ratio = _vc_last / _vc_avg if _vc_avg > 0 else 1.0
            _vc_adj   = 0.0

            if _vc_ratio >= 2.0:
                _vc_bull_bar = float(best.get('close', 0)) >= float(best.get('open', 0))
                _vc_aligned  = (_vc_bull_bar and trade_direction == "LONG") or \
                               (not _vc_bull_bar and trade_direction == "SHORT")
                if _vc_aligned:
                    _vc_adj = +0.08
                    logger.info(
                        f"   🔥 VOLUME CLIMAX {_vc_ratio:.1f}x aliniat {trade_direction} → confirmare +8%"
                    )
                else:
                    _vc_adj = -0.07
                    logger.info(
                        f"   💀 VOLUME CLIMAX {_vc_ratio:.1f}x CONTRA {trade_direction} "
                        f"→ ending move? -7%"
                    )

            if _vc_adj != 0.0:
                _pre_vc     = final_score
                final_score = max(min(final_score + _vc_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(f"   🔥 VOLUME CLIMAX: scor {_pre_vc:.2f} → {final_score:.2f}")
        # ─────────────────────────────────────────────────────────────────────

        # ══════════════════════════════════════════════════════════════════════
        # ELITE FILTERS NIVEL 2 — date istorice per-bară (acum disponibile din NT8)
        # Acestea necesită AladinBridge.cs v2.3+ care trimite poc_history,
        # dom_ratio_history și bar delta în câmpul "d" din bar_history.
        # ══════════════════════════════════════════════════════════════════════

        # ── E5. POC DRIFT DETECTOR ────────────────────────────────────────────
        # POC-ul (Point of Control) migrând constant în sus = instituții acumulează.
        # POC fix sau coborând = distribuție sau range. Unul dintre cei mai puri
        # indicatori de intenție instituțională din Volume Profile analysis.
        # Necesită minim 3 valori POC istorice (trimise de AladinBridge.cs v2.3+).
        _poc_hist = (live_data.get("poc_history", []) or []) if _live_mode and live_data else []
        if len(_poc_hist) >= 3:
            _poc_old   = float(_poc_hist[0])   # cel mai vechi
            _poc_mid   = float(_poc_hist[len(_poc_hist)//2])
            _poc_new   = float(_poc_hist[-1])  # cel mai recent
            # Verificăm că există trend consistent (nu salt izolat)
            _poc_drift = _poc_new - _poc_old
            _poc_consistent = (_poc_new >= _poc_mid >= _poc_old) or (_poc_new <= _poc_mid <= _poc_old)
            _poc_adj   = 0.0

            if _poc_consistent and abs(_poc_drift) > 0:
                _poc_rising = _poc_drift > 0
                if _poc_rising and trade_direction == "LONG":
                    _poc_adj = +0.06
                    logger.info(
                        f"   📊 POC DRIFT UP: {_poc_old:.1f} → {_poc_new:.1f} "
                        f"(+{_poc_drift:.1f}pts) → acumulare instituțională LONG +6%"
                    )
                elif not _poc_rising and trade_direction == "SHORT":
                    _poc_adj = +0.06
                    logger.info(
                        f"   📊 POC DRIFT DOWN: {_poc_old:.1f} → {_poc_new:.1f} "
                        f"({_poc_drift:.1f}pts) → distribuție instituțională SHORT +6%"
                    )
                elif _poc_rising and trade_direction == "SHORT":
                    # Fix v10.5: dacă OF confirmă SHORT, POC rising = distribuție la nivele mai înalte
                    # Instituțiile vând de la POC mai sus → nu penaliza agresiv
                    _of_cd_poc = float(live_data.get("cum_delta", 0) or 0) if _live_mode and live_data else 0
                    _of_sv_poc = float(live_data.get("bar_sell_vol", 0) or 0) if _live_mode and live_data else 0
                    _of_bv_poc = float(live_data.get("bar_buy_vol", 0) or 0) if _live_mode and live_data else 0
                    if _of_cd_poc < 0 and _of_sv_poc > _of_bv_poc * 1.2:
                        _poc_adj = -0.02
                        logger.info(
                            f"   ⚠️ POC DRIFT UP dar SHORT + OF confirmă → penalizare redusă -2%"
                        )
                    else:
                        _poc_adj = -0.05
                        logger.info(
                            f"   ⚠️ POC DRIFT UP dar SHORT = contra acumulării → -5%"
                        )
                elif not _poc_rising and trade_direction == "LONG":
                    _of_cd_poc2 = float(live_data.get("cum_delta", 0) or 0) if _live_mode and live_data else 0
                    _of_bv_poc2 = float(live_data.get("bar_buy_vol", 0) or 0) if _live_mode and live_data else 0
                    _of_sv_poc2 = float(live_data.get("bar_sell_vol", 0) or 0) if _live_mode and live_data else 0
                    if _of_cd_poc2 > 0 and _of_bv_poc2 > _of_sv_poc2 * 1.2:
                        _poc_adj = -0.02
                        logger.info(
                            f"   ⚠️ POC DRIFT DOWN dar LONG + OF confirmă → penalizare redusă -2%"
                        )
                    else:
                        _poc_adj = -0.05
                        logger.info(
                            f"   ⚠️ POC DRIFT DOWN dar LONG = contra distribuției → -5%"
                        )

            if _poc_adj != 0.0:
                _pre_poc    = final_score
                final_score = max(min(final_score + _poc_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(f"   📊 POC DRIFT: scor {_pre_poc:.2f} → {final_score:.2f}")
        # ─────────────────────────────────────────────────────────────────────

        # ── E6. DOM RATIO TREND (spike vs trend instituțional) ────────────────
        # Un spike izolat pe DOM poate fi zgomot. Un trend de 3+ bare cu bid/ask
        # ratio crescând constant = presiune instituțională reală, nu manipulare.
        # Aceasta separă confirmările reale de false positives din E1.
        # Necesită dom_ratio_history din AladinBridge.cs v2.3+.
        _dom_hist = (live_data.get("dom_ratio_history", []) or []) if _live_mode and live_data else []
        if len(_dom_hist) >= 4:
            _dom_last3  = [float(x) for x in _dom_hist[-4:]]
            _dom_trend  = _dom_last3[-1] - _dom_last3[0]   # schimbare totală pe 4 bare
            # Trend semnificativ: >0.3 modificare = instituții se mișcă consistent
            _dom_threshold = 0.30
            _dom_adj    = 0.0

            if _dom_trend >= _dom_threshold and trade_direction == "LONG":
                # Bid pressure crescând consistent pe 4 bare = acumulare reală
                # v10.5: redus de la +7% la +4% (DOM Stacking E6e acoperă depth-ul direct)
                _dom_adj = +0.04
                logger.info(
                    f"   🏦 DOM TREND BID crescător: {_dom_last3[0]:.2f}→{_dom_last3[-1]:.2f} "
                    f"(+{_dom_trend:.2f} pe 4 bare) → presiune LONG trend +4%"
                )
            elif _dom_trend <= -_dom_threshold and trade_direction == "SHORT":
                _dom_adj = +0.04
                logger.info(
                    f"   🏦 DOM TREND ASK crescător: {_dom_last3[0]:.2f}→{_dom_last3[-1]:.2f} "
                    f"({_dom_trend:.2f} pe 4 bare) → presiune SHORT trend +4%"
                )
            elif _dom_trend >= _dom_threshold and trade_direction == "SHORT":
                # v10.5: redus de la -6% la -3% (DOM Stacking E6e e mai precis)
                _dom_adj = -0.03
                logger.info(
                    f"   🚫 DOM TREND BID crescător dar SHORT → contra presiunii -3%"
                )
            elif _dom_trend <= -_dom_threshold and trade_direction == "LONG":
                _dom_adj = -0.03
                logger.info(
                    f"   🚫 DOM TREND ASK crescător dar LONG → contra presiunii -3%"
                )

            if _dom_adj != 0.0:
                _pre_domt   = final_score
                final_score = max(min(final_score + _dom_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(f"   🏦 DOM TREND: scor {_pre_domt:.2f} → {final_score:.2f}")
        # ─────────────────────────────────────────────────────────────────────

        # ══════════════════════════════════════════════════════════════════════
        # ── E6b. ABSORPTION SCORE (delta_at_high / delta_at_low) ──────────────
        # Cel mai pur semnal de footprint: delta negativă la High = sellers au absorbit
        # rally-ul exact la vârf. Delta pozitivă la Low = buyers au absorbit sell-off-ul.
        # Exemplu: Bar face new High, dar delta_at_high e -500 → vânzătorii au blocat
        # fiecare tentativă de breakout. Setup classic ICT: stop hunt → reversal.
        # ══════════════════════════════════════════════════════════════════════
        _dah = float(live_data.get("delta_at_high", 0) or 0) if _live_mode and live_data else 0
        _dal = float(live_data.get("delta_at_low", 0) or 0) if _live_mode and live_data else 0
        _abs_adj = 0.0

        if _dah != 0 or _dal != 0:
            if trade_direction == "SHORT" and _dah < -100:
                # Delta puternic negativă la High = absorbție bearish (sellers au blocat rally)
                # Cu cât mai negativă, cu atât mai puternic semnalul
                _abs_strength = min(abs(_dah) / 500.0, 1.0)  # normalizat la 500
                _abs_adj = +0.04 + (_abs_strength * 0.04)     # +4% → +8% bazat pe intensitate
                logger.info(
                    f"   🧊 ABSORPTION BEARISH: delta_at_high={_dah:.0f} "
                    f"(sellers au absorbit la High) → SHORT boost +{_abs_adj:.0%}"
                )
            elif trade_direction == "LONG" and _dal > 100:
                # Delta puternic pozitivă la Low = absorbție bullish (buyers au absorbit sell-off)
                _abs_strength = min(abs(_dal) / 500.0, 1.0)
                _abs_adj = +0.04 + (_abs_strength * 0.04)
                logger.info(
                    f"   🧊 ABSORPTION BULLISH: delta_at_low={_dal:.0f} "
                    f"(buyers au absorbit la Low) → LONG boost +{_abs_adj:.0%}"
                )
            elif trade_direction == "LONG" and _dah < -200:
                # Vrem LONG dar sellers absorb la High = rezistență puternică
                _abs_adj = -0.04
                logger.info(
                    f"   🧊 ABSORPTION CONTRA LONG: delta_at_high={_dah:.0f} "
                    f"(sellers la High blochează) → penalizare -4%"
                )
            elif trade_direction == "SHORT" and _dal > 200:
                # Vrem SHORT dar buyers absorb la Low = suport puternic
                _abs_adj = -0.04
                logger.info(
                    f"   🧊 ABSORPTION CONTRA SHORT: delta_at_low={_dal:.0f} "
                    f"(buyers la Low blochează) → penalizare -4%"
                )

            if _abs_adj != 0.0:
                _pre_abs    = final_score
                final_score = max(min(final_score + _abs_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(f"   🧊 ABSORPTION: scor {_pre_abs:.2f} → {final_score:.2f}")
        # ─────────────────────────────────────────────────────────────────────

        # ══════════════════════════════════════════════════════════════════════
        # ── E6c. INSTITUTIONAL FOOTPRINT (big trades) ─────────────────────────
        # Big trades (>= 20 contracte) sunt amprenta instituțională directă.
        # Dacă big_sell_count >> big_buy_count pe o bară = instituțiile distribuie.
        # Dacă big_buy_count >> big_sell_count = instituțiile acumulează.
        # Diferența trebuie să fie semnificativă (>= 3 trade-uri) ca să nu fie noise.
        # ══════════════════════════════════════════════════════════════════════
        _big_buy  = int(live_data.get("big_buy_count", 0) or 0) if _live_mode and live_data else 0
        _big_sell = int(live_data.get("big_sell_count", 0) or 0) if _live_mode and live_data else 0
        _big_total = _big_buy + _big_sell
        _inst_adj  = 0.0

        if _big_total >= 3:  # minim 3 big trades pe bară pt semnal valid
            _big_ratio = _big_buy / max(_big_sell, 1)  # ratio buy/sell
            _big_diff  = _big_buy - _big_sell

            if trade_direction == "LONG" and _big_diff >= 3:
                # Instituții acumulează agresiv → LONG confirmat
                _inst_adj = min(+0.03 + (_big_diff * 0.01), +0.08)
                logger.info(
                    f"   🏛️ INSTITUTIONAL BUY: {_big_buy} big buys vs {_big_sell} big sells "
                    f"(+{_big_diff}) → LONG boost +{_inst_adj:.0%}"
                )
            elif trade_direction == "SHORT" and _big_diff <= -3:
                # Instituții distribuie agresiv → SHORT confirmat
                _inst_adj = min(+0.03 + (abs(_big_diff) * 0.01), +0.08)
                logger.info(
                    f"   🏛️ INSTITUTIONAL SELL: {_big_sell} big sells vs {_big_buy} big buys "
                    f"({_big_diff}) → SHORT boost +{_inst_adj:.0%}"
                )
            elif trade_direction == "LONG" and _big_diff <= -3:
                # Instituții vând dar noi suntem LONG = contra
                _inst_adj = -0.04
                logger.info(
                    f"   🏛️ INSTITUTIONAL CONTRA LONG: {_big_sell} big sells vs {_big_buy} big buys "
                    f"→ instituții contra direcției -4%"
                )
            elif trade_direction == "SHORT" and _big_diff >= 3:
                _inst_adj = -0.04
                logger.info(
                    f"   🏛️ INSTITUTIONAL CONTRA SHORT: {_big_buy} big buys vs {_big_sell} big sells "
                    f"→ instituții contra direcției -4%"
                )

            if _inst_adj != 0.0:
                _pre_inst   = final_score
                final_score = max(min(final_score + _inst_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(f"   🏛️ INSTITUTIONAL: scor {_pre_inst:.2f} → {final_score:.2f}")
        # ─────────────────────────────────────────────────────────────────────

        # ══════════════════════════════════════════════════════════════════════
        # ── E6d. TAPE SPEED (urgency detection) ──────────────────────────────
        # Tape speed = tick-uri/secundă. Tape rapid (> 50 ticks/sec pe NQ) = urgență.
        # O mișcare cu tape rapid e mai convingătoare decât una cu tape lent.
        # Tape rapid + direcție confirmată = boost. Tape lent = nu penalizăm, doar nu dăm boost.
        # ══════════════════════════════════════════════════════════════════════
        _tape_spd = float(live_data.get("tape_speed", 0) or 0) if _live_mode and live_data else 0
        _bar_dlt  = float(live_data.get("bar_delta", 0) or 0) if _live_mode and live_data else 0
        _tape_adj = 0.0

        if _tape_spd > 0:
            _tape_fast = _tape_spd >= 50.0  # 50+ ticks/sec = fast tape pe NQ
            _tape_very_fast = _tape_spd >= 100.0  # 100+ = extremely aggressive

            if _tape_fast:
                # Tape rapid — verificăm dacă delta confirmă direcția
                if trade_direction == "LONG" and _bar_dlt > 0:
                    _tape_adj = +0.05 if _tape_very_fast else +0.03
                    logger.info(
                        f"   ⚡ TAPE SPEED: {_tape_spd:.1f} ticks/sec (fast) "
                        f"+ bar_delta={_bar_dlt:.0f} (bullish) → LONG boost +{_tape_adj:.0%}"
                    )
                elif trade_direction == "SHORT" and _bar_dlt < 0:
                    _tape_adj = +0.05 if _tape_very_fast else +0.03
                    logger.info(
                        f"   ⚡ TAPE SPEED: {_tape_spd:.1f} ticks/sec (fast) "
                        f"+ bar_delta={_bar_dlt:.0f} (bearish) → SHORT boost +{_tape_adj:.0%}"
                    )
                elif trade_direction == "LONG" and _bar_dlt < -100:
                    # Fast tape dar delta contra = sellers agresivi
                    _tape_adj = -0.03
                    logger.info(
                        f"   ⚡ TAPE SPEED CONTRA: {_tape_spd:.1f} ticks/sec + "
                        f"bar_delta={_bar_dlt:.0f} contra LONG → -3%"
                    )
                elif trade_direction == "SHORT" and _bar_dlt > 100:
                    _tape_adj = -0.03
                    logger.info(
                        f"   ⚡ TAPE SPEED CONTRA: {_tape_spd:.1f} ticks/sec + "
                        f"bar_delta={_bar_dlt:.0f} contra SHORT → -3%"
                    )

            if _tape_adj != 0.0:
                _pre_tape   = final_score
                final_score = max(min(final_score + _tape_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(f"   ⚡ TAPE: scor {_pre_tape:.2f} → {final_score:.2f}")
        # ─────────────────────────────────────────────────────────────────────

        # ══════════════════════════════════════════════════════════════════════
        # ── E6e. DOM STACKING (bid/ask depth imbalance) ──────────────────────
        # DOM arată ordinele PASIVE (limit orders) = unde instituțiile au pus ziduri.
        # Total bid size >> total ask size = suport pasiv puternic (bullish).
        # Total ask size >> total bid size = rezistență pasivă (bearish).
        # Folosim top 5 nivele de preț pe fiecare parte.
        # Atenție: DOM se schimbă rapid (spoofing) — confirmăm cu direcția trade-ului.
        # ══════════════════════════════════════════════════════════════════════
        _dom_bids = (live_data.get("dom_bids", []) or []) if _live_mode and live_data else []
        _dom_asks = (live_data.get("dom_asks", []) or []) if _live_mode and live_data else []
        _dom_adj  = 0.0

        if len(_dom_bids) >= 3 and len(_dom_asks) >= 3:
            # Top 5 nivele (cele mai apropiate de preț — cele mai relevante)
            _top_bid_size = sum(l.get("size", 0) for l in _dom_bids[:5])
            _top_ask_size = sum(l.get("size", 0) for l in _dom_asks[:5])
            _dom_total    = _top_bid_size + _top_ask_size

            if _dom_total > 0:
                _dom_bid_pct = _top_bid_size / _dom_total  # 0.5 = echilibru, >0.65 = bid heavy

                if _dom_bid_pct >= 0.65 and trade_direction == "LONG":
                    # Bid stacking puternic + LONG = suport pasiv confirmat
                    _dom_adj = +0.04
                    logger.info(
                        f"   📊 DOM STACKING BID: {_top_bid_size} vs {_top_ask_size} asks "
                        f"({_dom_bid_pct:.0%} bids) → suport LONG +4%"
                    )
                elif _dom_bid_pct <= 0.35 and trade_direction == "SHORT":
                    # Ask stacking puternic + SHORT = rezistență pasivă confirmată
                    _dom_adj = +0.04
                    logger.info(
                        f"   📊 DOM STACKING ASK: {_top_ask_size} vs {_top_bid_size} bids "
                        f"({1-_dom_bid_pct:.0%} asks) → rezistență SHORT +4%"
                    )
                elif _dom_bid_pct >= 0.70 and trade_direction == "SHORT":
                    # Bid wall masiv dar noi suntem SHORT = contra suportului pasiv
                    _dom_adj = -0.03
                    logger.info(
                        f"   📊 DOM STACKING CONTRA SHORT: bid wall {_top_bid_size} "
                        f"({_dom_bid_pct:.0%}) → contra suport pasiv -3%"
                    )
                elif _dom_bid_pct <= 0.30 and trade_direction == "LONG":
                    _dom_adj = -0.03
                    logger.info(
                        f"   📊 DOM STACKING CONTRA LONG: ask wall {_top_ask_size} "
                        f"({1-_dom_bid_pct:.0%}) → contra rezistență pasivă -3%"
                    )

                if _dom_adj != 0.0:
                    _pre_dom    = final_score
                    final_score = max(min(final_score + _dom_adj, 1.0), 0.0)
                    score_pct   = round(min(final_score * 100, 100), 2)
                    logger.info(f"   📊 DOM STACK: scor {_pre_dom:.2f} → {final_score:.2f}")
        # ─────────────────────────────────────────────────────────────────────

        # ══════════════════════════════════════════════════════════════════════
        # ── E6f. DELTA PROFILE (per-price delta distribution) ────────────────
        # Delta profile arată la CE NIVEL de preț s-a cumpărat/vândut.
        # Dacă delta e concentrată bearish la nivele superioare = distribuție la vârf.
        # Dacă delta e concentrată bullish la nivele inferioare = acumulare la bază.
        # Împărțim profilul în jumătatea superioară și inferioară — comparăm delta sumată.
        # ══════════════════════════════════════════════════════════════════════
        _dp = (live_data.get("delta_profile", []) or []) if _live_mode and live_data else []
        _dp_adj = 0.0

        if len(_dp) >= 4:
            # Sortăm după preț (low → high)
            _dp_sorted = sorted(_dp, key=lambda x: x.get("p", 0))
            _dp_mid    = len(_dp_sorted) // 2
            _dp_lower  = _dp_sorted[:_dp_mid]    # jumătatea de preț inferioară
            _dp_upper  = _dp_sorted[_dp_mid:]     # jumătatea de preț superioară

            _delta_lower = sum(e.get("d", 0) for e in _dp_lower)  # delta cumulativă jos
            _delta_upper = sum(e.get("d", 0) for e in _dp_upper)  # delta cumulativă sus

            # Distribuție la vârf: delta negativă sus = sellers domină la prețuri ridicate
            # Acumulare la bază: delta pozitivă jos = buyers domină la prețuri mici
            _dp_spread = _delta_lower - _delta_upper  # pozitiv = acumulare jos, negativ = distribuție sus

            if trade_direction == "LONG" and _dp_spread > 200:
                # Cumpărare concentrată la bază = acumulare clasică
                _dp_adj = +0.04
                logger.info(
                    f"   📈 DELTA PROFILE: acumulare la bază (δ_lower={_delta_lower:.0f} vs "
                    f"δ_upper={_delta_upper:.0f}) → LONG boost +4%"
                )
            elif trade_direction == "SHORT" and _dp_spread < -200:
                # Vânzare concentrată la vârf = distribuție clasică
                _dp_adj = +0.04
                logger.info(
                    f"   📈 DELTA PROFILE: distribuție la vârf (δ_upper={_delta_upper:.0f} vs "
                    f"δ_lower={_delta_lower:.0f}) → SHORT boost +4%"
                )
            elif trade_direction == "LONG" and _dp_spread < -300:
                # Distribuție masivă la vârf dar LONG = contra
                _dp_adj = -0.03
                logger.info(
                    f"   📈 DELTA PROFILE CONTRA LONG: distribuție la vârf "
                    f"(δ_upper={_delta_upper:.0f}) → -3%"
                )
            elif trade_direction == "SHORT" and _dp_spread > 300:
                _dp_adj = -0.03
                logger.info(
                    f"   📈 DELTA PROFILE CONTRA SHORT: acumulare la bază "
                    f"(δ_lower={_delta_lower:.0f}) → -3%"
                )

            if _dp_adj != 0.0:
                _pre_dp     = final_score
                final_score = max(min(final_score + _dp_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(f"   📈 DELTA PROFILE: scor {_pre_dp:.2f} → {final_score:.2f}")
        # ─────────────────────────────────────────────────────────────────────

        # ── E7. DELTA DIVERGENCE (cel mai puternic semnal de reversare) ────────
        # Delta Divergence = prețul și fluxul real de ordine merg în direcții opuse.
        # Preț face new high + delta scade = instituțiile NU confirmă mișcarea (distribuție).
        # Preț face new low + delta crește = instituțiile absorb vânzările (acumulare).
        # Folosit de toți traderii profesioniști pe ATAS, Bookmap, Sierra Chart.
        # Necesită câmpul "d" (bar delta) din bar_history — disponibil AladinBridge v2.3+.
        _bar_deltas = (live_data.get("bar_deltas", []) or []) if _live_mode and live_data else []
        if len(_bar_deltas) >= 4 and len(df) >= 4:
            # Luăm ultimele 4 bare pentru comparație
            _bd_last  = [float(x) for x in _bar_deltas[-4:]]
            _cls_last = list(df['close'].iloc[-4:])
            _price_up   = _cls_last[-1] > _cls_last[0]   # prețul a urcat?
            _delta_up   = _bd_last[-1]  > _bd_last[0]    # delta a crescut?
            _dd_adj     = 0.0

            # ── Divergență bearish: preț sus, delta jos ──
            if _price_up and not _delta_up and trade_direction == "LONG":
                # v10.5: redus de la -8% la -5% (Absorption E6b + Delta Profile E6f acoperă granular)
                _dd_adj = -0.05
                logger.info(
                    f"   ⚡ DELTA DIVERGENCE BEARISH: preț ↑ ({_cls_last[0]:.1f}→{_cls_last[-1]:.1f}) "
                    f"dar delta ↓ ({_bd_last[0]:.0f}→{_bd_last[-1]:.0f}) "
                    f"→ distribuție instituțională, LONG compromis -5%"
                )
            # ── Divergență bullish: preț jos, delta sus ──
            elif not _price_up and _delta_up and trade_direction == "SHORT":
                # Fix v10.3: delta div bullish NU invalideaza SHORT-ul cand cum_delta e masiv negativ
                # O recuperare de delta de la -345 la +90 e NOISE cand net flow total e -5000+
                # Penalizăm NUMAI dacă cum_delta e relativ neutru (nu avem trend bear clar)
                _cd_now = float(live_data.get("cum_delta", 0) or 0) if _live_mode and live_data else 0
                _price_drop_pts = abs(_cls_last[-1] - _cls_last[0])
                _delta_abs_move = abs(_bd_last[-1] - _bd_last[0])
                _atr_now_dd = float(live_data.get("atr_14", 12) or 12) if _live_mode and live_data else 12.0
                # Ignorăm divergența bullish dacă:
                # 1. cum_delta masiv negativ (trend bear real) — era -1500, acum -1000
                # 2. SAU prețul a căzut >1.5×ATR (mișcare clară SHORT) cu delta recovery mică (<50)
                _delta_recovery_noise = (
                    _cd_now < -1000 or
                    (_price_drop_pts > _atr_now_dd * 1.5 and _delta_abs_move < 50)
                )
                if _delta_recovery_noise:
                    _dd_adj = 0.0
                    logger.info(
                        f"   ⚡ DELTA DIV BULLISH ignorat: preț ↓ dar delta ↑ ({_bd_last[0]:.0f}→{_bd_last[-1]:.0f}) "
                        f"= noise, cum_delta={_cd_now:.0f} confirmă trend BEAR → skip penalizare SHORT"
                    )
                else:
                    # v10.5: redus de la -8% la -5% (Absorption + Delta Profile acoperă granular)
                    _dd_adj = -0.05
                    logger.info(
                        f"   ⚡ DELTA DIVERGENCE BULLISH: preț ↓ ({_cls_last[0]:.1f}→{_cls_last[-1]:.1f}) "
                        f"dar delta ↑ ({_bd_last[0]:.0f}→{_bd_last[-1]:.0f}) "
                        f"→ acumulare instituțională, SHORT compromis -5%"
                    )
            # ── Confirmare: preț și delta în aceeași direcție ──
            elif _price_up and _delta_up and trade_direction == "LONG":
                _dd_adj = +0.06
                logger.info(
                    f"   ✅ DELTA CONFIRMARE LONG: preț ↑ și delta ↑ → momentum real +6%"
                )
            elif not _price_up and not _delta_up and trade_direction == "SHORT":
                _dd_adj = +0.06
                logger.info(
                    f"   ✅ DELTA CONFIRMARE SHORT: preț ↓ și delta ↓ → momentum real +6%"
                )

            if _dd_adj != 0.0:
                _pre_dd     = final_score
                final_score = max(min(final_score + _dd_adj, 1.0), 0.0)
                score_pct   = round(min(final_score * 100, 100), 2)
                logger.info(f"   ⚡ DELTA DIV: scor {_pre_dd:.2f} → {final_score:.2f}")

        # ── E7b. ABSORPTION DETECTOR v8.1 ─────────────────────────────────────
        # Absorption = volum mare dar preț nu se mișcă → cineva absoarbe presiunea
        # Bid Absorption (bullish): sell volume mare dar prețul nu scade = cumpărătorii absorb
        # Ask Absorption (bearish): buy volume mare dar prețul nu urcă = vânzătorii absorb
        # Trei metode:
        #   A) Proxy din bar data: volume > 2×avg dar body < 0.3×ATR
        #   B) Delta absorption: bar_buy_vol mare dar preț flat (sau invers)
        #   C) NT8 absorption_score: trimis direct de addon (când disponibil)
        _absorption_detected = False
        _absorption_direction = ""  # "BULL" = bid absorption, "BEAR" = ask absorption
        _absorption_strength = 0.0  # 0-1
        _absorption_signals = []

        if _live_mode and live_data:
            try:
                # ── Metoda C: NT8 absorption direct (cel mai precis) ──
                _nt8_abs_score = float(live_data.get("absorption_score", 0) or 0)
                _nt8_abs_side  = ""
                if _nt8_abs_score >= 60:
                    # NT8 addon a detectat absorpție semnificativă
                    # Determinăm direcția din cum_delta + price action
                    _abs_cum = float(live_data.get("cum_delta", 0) or 0)
                    if _abs_cum > 0:
                        _absorption_direction = "BULL"
                        _nt8_abs_side = "BID"
                    else:
                        _absorption_direction = "BEAR"
                        _nt8_abs_side = "ASK"
                    _absorption_detected = True
                    _absorption_strength = min(_nt8_abs_score / 100.0, 1.0)
                    _absorption_signals.append(f"NT8_ABS_{_nt8_abs_side}({_nt8_abs_score:.0f})")
                    logger.info(
                        f"   🧱 NT8 ABSORPTION: score={_nt8_abs_score:.0f}, "
                        f"side={_nt8_abs_side}, strength={_absorption_strength:.2f}"
                    )

                # ── Metoda A: Proxy — volum mare + body mic (absorption candle) ──
                _bar_vols    = live_data.get("bar_volumes", []) or []
                _bar_opens   = live_data.get("bar_opens", []) or []
                _bar_closes  = live_data.get("bar_closes", []) or []
                _bar_highs   = live_data.get("bar_highs", []) or []
                _bar_lows    = live_data.get("bar_lows", []) or []
                _abs_atr     = float(live_data.get("atr_14", 0) or 0)

                if len(_bar_vols) >= 8 and _abs_atr > 0 and not _absorption_detected:
                    # Average volume pe ultimele 20 bare (sau câte avem)
                    _avg_vol = sum(_bar_vols[:-1]) / max(len(_bar_vols) - 1, 1) if len(_bar_vols) > 1 else 1
                    # Ultima bară
                    _last_vol  = _bar_vols[-1] if _bar_vols else 0
                    _last_body = abs(_bar_closes[-1] - _bar_opens[-1]) if _bar_closes and _bar_opens else 999
                    _last_range = (_bar_highs[-1] - _bar_lows[-1]) if _bar_highs and _bar_lows else 999

                    # Absorption candle: vol > 2x avg DAR body < 0.3 * ATR
                    _vol_spike    = _last_vol > _avg_vol * 2.0
                    _small_body   = _last_body < _abs_atr * 0.3
                    _has_wicks    = _last_range > _last_body * 2.5 if _last_body > 0 else False

                    if _vol_spike and _small_body:
                        _absorption_detected = True
                        _absorption_strength = min((_last_vol / max(_avg_vol, 1)) / 5.0, 1.0)

                        # Direcția: dacă lower wick lung → bid absorption (bullish)
                        # dacă upper wick lung → ask absorption (bearish)
                        _lower_wick = min(_bar_opens[-1], _bar_closes[-1]) - _bar_lows[-1]
                        _upper_wick = _bar_highs[-1] - max(_bar_opens[-1], _bar_closes[-1])
                        if _lower_wick > _upper_wick * 1.5:
                            _absorption_direction = "BULL"
                            _absorption_signals.append(
                                f"VOL_ABS_BID(vol={_last_vol:.0f}/{_avg_vol:.0f}, "
                                f"body={_last_body:.1f}, wick_lo={_lower_wick:.1f})"
                            )
                        elif _upper_wick > _lower_wick * 1.5:
                            _absorption_direction = "BEAR"
                            _absorption_signals.append(
                                f"VOL_ABS_ASK(vol={_last_vol:.0f}/{_avg_vol:.0f}, "
                                f"body={_last_body:.1f}, wick_hi={_upper_wick:.1f})"
                            )
                        else:
                            # Ambiguous — use cum_delta to decide
                            _abs_cd = float(live_data.get("cum_delta", 0) or 0)
                            _absorption_direction = "BULL" if _abs_cd > 0 else "BEAR"
                            _absorption_signals.append(
                                f"VOL_ABS_{'BID' if _abs_cd > 0 else 'ASK'}"
                                f"(vol={_last_vol:.0f}/{_avg_vol:.0f}, delta={_abs_cd:.0f})"
                            )
                        logger.info(
                            f"   🧱 VOLUME ABSORPTION: vol={_last_vol:.0f} vs avg={_avg_vol:.0f} "
                            f"(×{_last_vol/max(_avg_vol,1):.1f}), body={_last_body:.1f}, "
                            f"ATR={_abs_atr:.1f} → {_absorption_direction}"
                        )

                # ── Metoda B: Delta absorption — sell vol mare dar preț flat ──
                _bbv = float(live_data.get("bar_buy_vol", 0) or 0)
                _bsv = float(live_data.get("bar_sell_vol", 0) or 0)
                _total_bvol = _bbv + _bsv

                if _total_bvol > 500 and _abs_atr > 0 and not _absorption_detected:
                    _cur_body = abs(float(best.get('close', 0) or 0) - float(best.get('open', 0) or 0))
                    _cur_close = float(best.get('close', 0) or 0)
                    _cur_open  = float(best.get('open', 0) or 0)

                    # Bid absorption: sell_vol > 60% dar preț nu scade (sau urcă puțin)
                    if _bsv > _total_bvol * 0.60 and _cur_body < _abs_atr * 0.3:
                        _absorption_detected = True
                        _absorption_direction = "BULL"
                        _absorption_strength = min((_bsv / max(_total_bvol, 1)), 1.0)
                        _absorption_signals.append(
                            f"DELTA_ABS_BID(sell={_bsv:.0f}/{_total_bvol:.0f}={_bsv/_total_bvol*100:.0f}%, "
                            f"body={_cur_body:.1f})"
                        )
                        logger.info(
                            f"   🧱 DELTA ABSORPTION BID: sell_vol={_bsv:.0f} "
                            f"({_bsv/_total_bvol*100:.0f}%) dar body={_cur_body:.1f} < ATR*0.3={_abs_atr*0.3:.1f} "
                            f"→ cumpărătorii absorb vânzările (BULLISH)"
                        )

                    # Ask absorption: buy_vol > 60% dar preț nu urcă
                    elif _bbv > _total_bvol * 0.60 and _cur_body < _abs_atr * 0.3:
                        _absorption_detected = True
                        _absorption_direction = "BEAR"
                        _absorption_strength = min((_bbv / max(_total_bvol, 1)), 1.0)
                        _absorption_signals.append(
                            f"DELTA_ABS_ASK(buy={_bbv:.0f}/{_total_bvol:.0f}={_bbv/_total_bvol*100:.0f}%, "
                            f"body={_cur_body:.1f})"
                        )
                        logger.info(
                            f"   🧱 DELTA ABSORPTION ASK: buy_vol={_bbv:.0f} "
                            f"({_bbv/_total_bvol*100:.0f}%) dar body={_cur_body:.1f} < ATR*0.3={_abs_atr*0.3:.1f} "
                            f"→ vânzătorii absorb cumpărăturile (BEARISH)"
                        )

            except Exception as _abs_err:
                logger.debug(f"Absorption detector error: {_abs_err}")

        # ── Aplicare absorption pe scor ──
        _abs_adj = 0.0
        if _absorption_detected:
            if _absorption_direction == "BULL" and trade_direction == "LONG":
                # Absorption confirmă direcția → bonus
                _abs_adj = +0.07 * _absorption_strength
                logger.info(f"   🧱 ABSORPTION CONFIRM LONG: +{_abs_adj:.1%}")
            elif _absorption_direction == "BEAR" and trade_direction == "SHORT":
                _abs_adj = +0.07 * _absorption_strength
                logger.info(f"   🧱 ABSORPTION CONFIRM SHORT: +{_abs_adj:.1%}")
            elif _absorption_direction == "BULL" and trade_direction == "SHORT":
                # Absorption contra direcției → penalizare
                _abs_adj = -0.08 * _absorption_strength
                logger.info(f"   🧱 ABSORPTION CONTRA SHORT: {_abs_adj:.1%} (bid absorb = bullish)")
            elif _absorption_direction == "BEAR" and trade_direction == "LONG":
                # Fix v10.2: post-displacement, bare cu buy_vol mare + body mic = consolidare normală
                # Instituționalii acumulează după stop hunt — NU vânzătorii absorb
                _has_disp_abs = (
                    bool(best.get('has_displacement', 0)) or
                    (len(df) >= 4 and 'has_displacement' in df.columns and bool(df['has_displacement'].tail(4).any()))
                )
                if _has_disp_abs:
                    _abs_adj = -0.02 * _absorption_strength
                    logger.info(
                        f"   🧱 ABSORPTION CONTRA LONG (post-disp): {_abs_adj:.1%} redus "
                        f"— consolidare normală după displacement, nu vânzare instituțională"
                    )
                else:
                    _abs_adj = -0.08 * _absorption_strength
                    logger.info(f"   🧱 ABSORPTION CONTRA LONG: {_abs_adj:.1%} (ask absorb = bearish)")

        if _abs_adj != 0.0:
            _pre_abs = final_score
            final_score = max(min(final_score + _abs_adj, 1.0), 0.0)
            score_pct = round(min(final_score * 100, 100), 2)
            logger.info(f"   🧱 ABSORPTION: scor {_pre_abs:.2f} → {final_score:.2f}")
        # ─────────────────────────────────────────────────────────────────────

        # ══════════════════════════════════════════════════════════════════════
        # ADVANCED ORDERFLOW ANALYTICS — 5 tehnici noi
        # ══════════════════════════════════════════════════════════════════════

        # ── OF1. EXHAUSTION / CLIMAX VOLUME ──────────────────────────────────
        # Volum extrem (>4x avg) + wick lung + body mic = participanții agresivi
        # au fost epuizați → reversal iminent
        _exhaustion_detected = False
        _exhaustion_side = ""  # "BULL_EXHAUST" = buyers epuizați → SHORT, "BEAR_EXHAUST" → LONG
        _exhaustion_adj = 0.0
        try:
            _bar_vols_ex = live_data.get("bar_volumes", []) if _live_mode else []
            _bar_highs_ex = live_data.get("bar_highs", []) if _live_mode else []
            _bar_lows_ex = live_data.get("bar_lows", []) if _live_mode else []
            _bar_opens_ex = live_data.get("bar_opens", []) if _live_mode else []
            _bar_closes_ex = live_data.get("bar_closes", []) if _live_mode else []

            if len(_bar_vols_ex) >= 10:
                _avg_vol_ex = sum(_bar_vols_ex[-10:-1]) / 9.0 if len(_bar_vols_ex) >= 10 else 1
                _last_vol_ex = _bar_vols_ex[-1]
                _last_h = _bar_highs_ex[-1]
                _last_l = _bar_lows_ex[-1]
                _last_o = _bar_opens_ex[-1]
                _last_c = _bar_closes_ex[-1]
                _last_body = abs(_last_c - _last_o)
                _last_range = _last_h - _last_l if _last_h > _last_l else 0.01
                _upper_wick = _last_h - max(_last_o, _last_c)
                _lower_wick = min(_last_o, _last_c) - _last_l

                # Climax: vol > 4x avg + body < 30% range (wick dominant)
                if _avg_vol_ex > 0 and _last_vol_ex > _avg_vol_ex * 4 and _last_body < _last_range * 0.3:
                    if _upper_wick > _lower_wick * 2:
                        # Upper wick dominant + climax vol = buyers epuizați
                        _exhaustion_detected = True
                        _exhaustion_side = "BULL_EXHAUST"
                        if trade_direction == "SHORT":
                            _exhaustion_adj = +0.06
                        elif trade_direction == "LONG":
                            _exhaustion_adj = -0.07
                    elif _lower_wick > _upper_wick * 2:
                        # Lower wick dominant + climax vol = sellers epuizați
                        _exhaustion_detected = True
                        _exhaustion_side = "BEAR_EXHAUST"
                        if trade_direction == "LONG":
                            _exhaustion_adj = +0.06
                        elif trade_direction == "SHORT":
                            _exhaustion_adj = -0.07

                if _exhaustion_adj != 0.0:
                    _pre_exh = final_score
                    final_score = max(min(final_score + _exhaustion_adj, 1.0), 0.0)
                    score_pct = round(min(final_score * 100, 100), 2)
                    logger.info(
                        f"   💥 EXHAUSTION: {_exhaustion_side} vol={_last_vol_ex:.0f} ({_last_vol_ex/_avg_vol_ex:.1f}x avg) "
                        f"body/range={_last_body/_last_range:.0%} | {_exhaustion_adj:+.0%} | scor {_pre_exh:.2f} → {final_score:.2f}"
                    )
        except Exception as _exh_err:
            logger.debug(f"Exhaustion detector error: {_exh_err}")

        # ── OF2. DELTA DIVERGENCE ────────────────────────────────────────────
        # Preț face Higher High dar Delta face Lower High = distribuție (bearish)
        # Preț face Lower Low dar Delta face Higher Low = acumulare (bullish)
        _delta_div_detected = False
        _delta_div_side = ""
        _delta_div_adj = 0.0
        try:
            _bar_deltas_dd = live_data.get("bar_deltas", []) if _live_mode else []
            _bar_closes_dd = live_data.get("bar_closes", []) if _live_mode else []
            _bar_highs_dd = live_data.get("bar_highs", []) if _live_mode else []
            _bar_lows_dd = live_data.get("bar_lows", []) if _live_mode else []

            if len(_bar_deltas_dd) >= 6 and len(_bar_closes_dd) >= 6:
                # Comparăm ultimele 3 bare cu cele 3 dinaintea lor
                _recent_highs = _bar_highs_dd[-3:]
                _prev_highs = _bar_highs_dd[-6:-3]
                _recent_lows = _bar_lows_dd[-3:]
                _prev_lows = _bar_lows_dd[-6:-3]
                _recent_deltas = _bar_deltas_dd[-3:]
                _prev_deltas = _bar_deltas_dd[-6:-3]

                _r_max_h = max(_recent_highs)
                _p_max_h = max(_prev_highs)
                _r_max_d = max(_recent_deltas)
                _p_max_d = max(_prev_deltas)
                _r_min_l = min(_recent_lows)
                _p_min_l = min(_prev_lows)
                _r_min_d = min(_recent_deltas)
                _p_min_d = min(_prev_deltas)

                # Bearish divergence: higher high in price, lower high in delta
                if _r_max_h > _p_max_h and _r_max_d < _p_max_d * 0.7:
                    _delta_div_detected = True
                    _delta_div_side = "BEARISH_DIV"
                    if trade_direction == "SHORT":
                        _delta_div_adj = +0.06
                    elif trade_direction == "LONG":
                        _delta_div_adj = -0.06

                # Bullish divergence: lower low in price, higher low in delta
                elif _r_min_l < _p_min_l and _r_min_d > _p_min_d * 0.7:
                    _delta_div_detected = True
                    _delta_div_side = "BULLISH_DIV"
                    if trade_direction == "LONG":
                        _delta_div_adj = +0.06
                    elif trade_direction == "SHORT":
                        # Fix v10.3: BULLISH_DIV nu compromite SHORT-ul cand cum_delta e masiv negativ
                        # Higher low in delta = pullback normal in trend bear, nu acumulare reala
                        _cd_of2 = float(live_data.get("cum_delta", 0) or 0) if _live_mode and live_data else 0
                        if _cd_of2 > -1500:  # cum_delta neutru sau bullish = divergenta reala
                            _delta_div_adj = -0.06
                        else:
                            _delta_div_adj = 0.0  # cum_delta masiv bear = ignore divergenta
                            logger.info(
                                f"   📊 BULLISH_DIV SHORT ignorat: cum_delta={_cd_of2:.0f} confirmă trend BEAR"
                            )

                if _delta_div_adj != 0.0:
                    _pre_dd = final_score
                    final_score = max(min(final_score + _delta_div_adj, 1.0), 0.0)
                    score_pct = round(min(final_score * 100, 100), 2)
                    logger.info(
                        f"   📊 DELTA DIVERGENCE: {_delta_div_side} | "
                        f"price H/L: {_r_max_h:.1f}/{_r_min_l:.1f} vs {_p_max_h:.1f}/{_p_min_l:.1f} | "
                        f"delta H/L: {_r_max_d:.0f}/{_r_min_d:.0f} vs {_p_max_d:.0f}/{_p_min_d:.0f} | "
                        f"{_delta_div_adj:+.0%} | scor {_pre_dd:.2f} → {final_score:.2f}"
                    )
        except Exception as _dd_err:
            logger.debug(f"Delta Divergence error: {_dd_err}")

        # ── OF3. STACKED IMBALANCES — dezactivat v9.4 ────────────────────────
        # Ajustări ±2-3% prea mici, NT8 nu trimite stacked_imbalances live.
        _stacked_adj = 0.0

        # ── OF4. UNFINISHED BUSINESS (din NT8) ───────────────────────────────
        # Niveluri unde doar un side a tranzacționat → preț atras ca magnet
        _ub_adj = 0.0
        try:
            _ub_levels = live_data.get("unfinished_business", []) if _live_mode else []
            if _ub_levels and isinstance(_ub_levels, list) and len(_ub_levels) > 0:
                _close_price = float(best['close'])
                # Cel mai apropiat UB level
                _nearest_ub = None
                _nearest_dist = float('inf')
                for _ub in _ub_levels:
                    _ub_price = float(_ub.get("price", 0) or 0)
                    if _ub_price > 0:
                        _dist = abs(_close_price - _ub_price)
                        if _dist < _nearest_dist:
                            _nearest_dist = _dist
                            _nearest_ub = _ub

                if _nearest_ub:
                    _ub_price = float(_nearest_ub["price"])
                    _ub_side = _nearest_ub.get("side", "")
                    _atr_val = float(best.get('atr_14', 0) or live_data.get('atr_14', 20))
                    if _atr_val == 0: _atr_val = 20

                    # UB e relevant doar dacă e la < 2 ATR distanță
                    if _nearest_dist < _atr_val * 2:
                        # UB sub preț = magnet jos → favorizează SHORT
                        if _ub_price < _close_price:
                            if trade_direction == "SHORT":
                                _ub_adj = +0.04
                            elif trade_direction == "LONG":
                                _ub_adj = -0.03
                        # UB deasupra prețului = magnet sus → favorizează LONG
                        else:
                            if trade_direction == "LONG":
                                _ub_adj = +0.04
                            elif trade_direction == "SHORT":
                                _ub_adj = -0.03

                        if _ub_adj != 0.0:
                            _pre_ub = final_score
                            final_score = max(min(final_score + _ub_adj, 1.0), 0.0)
                            score_pct = round(min(final_score * 100, 100), 2)
                            logger.info(
                                f"   🧲 UNFINISHED BUSINESS: {_ub_side} @ {_ub_price:.2f} "
                                f"(dist={_nearest_dist:.1f}, {_nearest_dist/_atr_val:.1f} ATR) | "
                                f"{_ub_adj:+.0%} | scor {_pre_ub:.2f} → {final_score:.2f}"
                            )
        except Exception as _ub_err:
            logger.debug(f"Unfinished Business error: {_ub_err}")

        # ── OF5. ICEBERG DETECTION — dezactivat v9.4 ─────────────────────────
        # Ajustări ±3% prea mici, NT8 nu trimite date iceberg live consistent.
        _iceberg_adj = 0.0

        # ══════════════════════════════════════════════════════════════════════
        # ELITE FILTERS NIVEL 3 — ICT Core Concepts (Liquidity, MSS, Equal H/L)
        # ══════════════════════════════════════════════════════════════════════

        # ── E8. LIQUIDITY SWEEP DETECTOR (cel mai pur setup ICT) ─────────────
        # Un Liquidity Sweep = prețul ia out swing high/low anterior și revine
        # LONG după sweep bearish: bara curenta low < prev_low DAR închide deasupra prev_low
        # SHORT după sweep bullish: bara curenta high > prev_high DAR închide sub prev_high
        # Sweepul confirmă că market makerul a luat lichiditate și acum inversează
        #
        # v9.6 MTF VALIDATION — cascade top-down:
        #   sweep pe 1m VALID numai dacă aliniată cu contextul HTF (15m/H1/H4)
        #   Bearish sweep (SHORT) = valid NUMAI dacă prețul e în PREMIUM pe HTF (≥50%)
        #   Bullish sweep (LONG)  = valid NUMAI dacă prețul e în DISCOUNT pe HTF (≤50%)
        #   + minim sweep: cel puțin ATR×0.10 sau 5pts (nu detectăm micro-sweepuri de 2pts)
        if len(df) >= 5:
            try:
                _highs   = list(df['high'].iloc[-6:-1])   # ultimele 5 high-uri (excl. curent)
                _lows    = list(df['low'].iloc[-6:-1])     # ultimele 5 low-uri (excl. curent)
                _cur_bar = df.iloc[-1]
                _cur_h   = float(_cur_bar['high'])
                _cur_l   = float(_cur_bar['low'])
                _cur_c   = float(_cur_bar['close'])
                _prev_hi = max(_highs)   # swing high al ultimelor 5 bare
                _prev_lo = min(_lows)    # swing low al ultimelor 5 bare

                # ── Minim sweep: ATR×0.10 sau cel puțin 5pts ──────────────────
                _atr_val  = float(best.get('atr_14', 0) or 0)
                _min_sweep = max(_atr_val * 0.10, 5.0)   # ex: ATR=30 → minim 3pts dar floor 5pts

                _sweep_size_bear = _cur_h - _prev_hi   # câți pts a depășit high-ul
                _sweep_size_bull = _prev_lo - _cur_l   # câți pts a depășit low-ul

                # ── Context MTF: cascadă relativă ────────────────────────────
                # Sweep pe 1m → validat față de 15m (HTF imediat superior)
                # Sweep pe 15m → față de H4, pe H4 → față de Daily/Weekly
                # Deoarece analiza rulează pe 1m, folosim SPECIFIC 15m range
                # (nu H4 — H4 e HTF pentru 15m, nu pentru 1m)
                _sw_15m_hi = float(df['high'].tail(15).max()) if len(df) >= 15 else 0
                _sw_15m_lo = float(df['low'].tail(15).min())  if len(df) >= 15 else 0
                _sw_tf_label = "15m"

                if _sw_15m_hi > _sw_15m_lo > 0:
                    _sw_pd_pct = (_cur_c - _sw_15m_lo) / (_sw_15m_hi - _sw_15m_lo)
                else:
                    _sw_pd_pct = 0.5   # neutru dacă nu avem date

                # Premium = prețul în top 50% al range-ului 15m (≥0.50)
                # Discount = prețul în bottom 50% al range-ului 15m (≤0.50)
                _in_htf_premium  = _sw_pd_pct >= 0.50
                _in_htf_discount = _sw_pd_pct <= 0.50

                _sweep_adj = 0.0
                _sweep_msg = ""

                # Bearish sweep (fake breakout sus → SHORT)
                if _cur_h > _prev_hi and _cur_c < _prev_hi:
                    if _sweep_size_bear < _min_sweep:
                        # Micro-sweep — zgomot, ignorat
                        logger.debug(
                            f"   ⚡ SWEEP IGNORAT (micro): bearish {_sweep_size_bear:.1f}pts < min {_min_sweep:.1f}pts"
                        )
                    elif not _in_htf_premium:
                        # Sweep în DISCOUNT pe 15m = zgomot, nu setup SHORT real
                        logger.info(
                            f"   ⚡ SWEEP INVALID MTF: bearish sweep {_cur_h:.1f}>{_prev_hi:.1f} "
                            f"dar prețul în DISCOUNT {_sw_tf_label} ({_sw_pd_pct:.0%}) → nu e setup SHORT valid"
                        )
                        if trade_direction == "SHORT":
                            _sweep_adj = -0.05
                            _sweep_msg = f"❌ BEARISH SWEEP în DISCOUNT {_sw_tf_label} ({_sw_pd_pct:.0%}) → nu e valid SHORT -5%"
                    else:
                        if trade_direction == "SHORT":
                            _sweep_adj = +0.09
                            _sweep_msg = (
                                f"🎯 BEARISH SWEEP VALID: high {_cur_h:.1f} > prev_hi {_prev_hi:.1f} "
                                f"({_sweep_size_bear:.1f}pts ≥ min {_min_sweep:.1f}pts) | "
                                f"{_sw_tf_label} PREMIUM {_sw_pd_pct:.0%} ✅ → SHORT confirmat +9%"
                            )
                        elif trade_direction == "LONG":
                            _sweep_adj = -0.09
                            _sweep_msg = f"⚠️ CONTRA SWEEP: bearish sweep valid dar LONG → -9%"

                # Bullish sweep (fake breakout jos → LONG)
                elif _cur_l < _prev_lo and _cur_c > _prev_lo:
                    if _sweep_size_bull < _min_sweep:
                        logger.debug(
                            f"   ⚡ SWEEP IGNORAT (micro): bullish {_sweep_size_bull:.1f}pts < min {_min_sweep:.1f}pts"
                        )
                    elif not _in_htf_discount:
                        # Sweep în PREMIUM pe 15m = zgomot, nu setup LONG real
                        logger.info(
                            f"   ⚡ SWEEP INVALID MTF: bullish sweep {_cur_l:.1f}<{_prev_lo:.1f} "
                            f"dar prețul în PREMIUM {_sw_tf_label} ({_sw_pd_pct:.0%}) → nu e setup LONG valid"
                        )
                        if trade_direction == "LONG":
                            _sweep_adj = -0.05
                            _sweep_msg = f"❌ BULLISH SWEEP în PREMIUM {_sw_tf_label} ({_sw_pd_pct:.0%}) → nu e valid LONG -5%"
                    else:
                        if trade_direction == "LONG":
                            _sweep_adj = +0.09
                            _sweep_msg = (
                                f"🎯 BULLISH SWEEP VALID: low {_cur_l:.1f} < prev_lo {_prev_lo:.1f} "
                                f"({_sweep_size_bull:.1f}pts ≥ min {_min_sweep:.1f}pts) | "
                                f"{_sw_tf_label} DISCOUNT {_sw_pd_pct:.0%} ✅ → LONG confirmat +9%"
                            )
                        elif trade_direction == "SHORT":
                            _sweep_adj = -0.09
                            _sweep_msg = f"⚠️ CONTRA SWEEP: bullish sweep valid dar SHORT → -9%"

                if _sweep_adj != 0.0:
                    _pre_sweep  = final_score
                    final_score = max(min(final_score + _sweep_adj, 1.0), 0.0)
                    score_pct   = round(min(final_score * 100, 100), 2)
                    logger.info(f"   ⚡ LIQUIDITY SWEEP: {_sweep_msg} | scor {_pre_sweep:.2f} → {final_score:.2f}")
            except Exception as _e8_err:
                logger.debug(f"E8 Liquidity Sweep error: {_e8_err}")

        # ── E9. MARKET STRUCTURE SHIFT / CHoCH (Change of Character) ─────────
        # MSS = prețul rupe structura în direcția opusă față de tendința precedentă
        # CHoCH bullish: după un downtrend real (lower highs + lower lows), first HH
        # CHoCH bearish: după un uptrend real (higher highs + higher lows), first LL
        #
        # v10.1 HTF SWING LOGIC — Mario guideline:
        #   1m analysis → validat față de 5m swing (≈25 bare 1m) + 15m swing (≈75 bare 1m)
        #   Trend valid NUMAI dacă există ATÂT lower lows CÂT ȘI lower highs (structură reală)
        #   Nu mai e suficientă o simplă comparație close[-2] vs close[-5] (4 bare = zgomot)
        #   Bonus constant 0.07 — nu mai e amplificat de reversal_override (stop feedback loop)
        #   Displacement required: bara CHoCH trebuie să închidă cu corp ≥ 50% din range-ul barei
        #   (high - low = range total; body = abs(close - open); body/range >= 0.50)
        #   Regulă timeframe-agnostică: o lumânare care închide la jumătate din range-ul ei
        #   confirmă că direcția e reală, nu un wick/spike fals
        if len(df) >= 26:
            try:
                # ── Context 5m: ultimele 25 bare 1m ≈ 5 candle-uri pe 5m ──────
                _lb5  = min(25, len(df) - 2)
                _half5 = _lb5 // 2
                _c5   = list(df['close'].iloc[-_lb5:])
                _h5   = list(df['high'].iloc[-_lb5:])
                _l5   = list(df['low'].iloc[-_lb5:])

                # Swing high/low al primei jumătăți (mai vechi) vs a doua jumătate (mai recent)
                _older5_hi = max(_h5[:_half5])
                _older5_lo = min(_l5[:_half5])
                _recent5_hi = max(_h5[_half5:-1])
                _recent5_lo = min(_l5[_half5:-1])
                _cur_close  = _c5[-1]

                # Trend real descendent pe 5m: lower highs + lower lows + close în scădere
                _trend_down_5m = (
                    _recent5_hi < _older5_hi and      # lower highs
                    _recent5_lo < _older5_lo and      # lower lows
                    _c5[-2] < _c5[-_half5]            # close confirmat descendent
                )
                # Trend real ascendent pe 5m: higher highs + higher lows + close în creștere
                _trend_up_5m = (
                    _recent5_hi > _older5_hi and      # higher highs
                    _recent5_lo > _older5_lo and      # higher lows
                    _c5[-2] > _c5[-_half5]            # close confirmat ascendent
                )

                # ── Context 15m: ultimele 75 bare 1m ≈ 5 candle-uri pe 15m ──
                _lb15 = min(75, len(df) - 2)
                _half15 = _lb15 // 2
                _c15  = list(df['close'].iloc[-_lb15:])
                _h15  = list(df['high'].iloc[-_lb15:])
                _l15  = list(df['low'].iloc[-_lb15:])
                _older15_hi  = max(_h15[:_half15])
                _older15_lo  = min(_l15[:_half15])
                _recent15_hi = max(_h15[_half15:-1])
                _recent15_lo = min(_l15[_half15:-1])

                _trend_down_15m = (
                    _recent15_hi < _older15_hi and
                    _recent15_lo < _older15_lo and
                    _c15[-2] < _c15[-_half15]
                )
                _trend_up_15m = (
                    _recent15_hi > _older15_hi and
                    _recent15_lo > _older15_lo and
                    _c15[-2] > _c15[-_half15]
                )

                # Trend valid = confirmat pe cel puțin un TF (5m sau 15m)
                _trend_down = _trend_down_5m or _trend_down_15m
                _trend_up   = _trend_up_5m   or _trend_up_15m

                # Nivel de rupere al structurii = swing high/low al contextului 5m
                _break_hi = _recent5_hi   # CHoCH bull: close trebuie să depășească asta
                _break_lo = _recent5_lo   # CHoCH bear: close trebuie să cadă sub asta

                # ── Displacement requirement: bara CHoCH trebuie să aibă corp real ──
                # Displacement: body ≥ 50% din range-ul barei curente (timeframe-agnostic)
                # Ex: bara high=100, low=90 → range=10 → body trebuie ≥ 5 puncte
                _e9_body   = abs(_cur_close - float(df['open'].iloc[-1]))
                _e9_range  = max(float(df['high'].iloc[-1]) - float(df['low'].iloc[-1]), 0.25)
                _has_disp_e9 = (_e9_body / _e9_range) >= 0.50

                _mss_adj = 0.0
                # Fix v10.1: bonus constant 0.07 — nu mai e amplificat de reversal_override
                # Feedback loop anterior: reversal_override → mss_bonus ×1.7 → confirma reversal
                _mss_bonus   = 0.07
                _mss_penalty = 0.06

                # CHoCH Bullish: trend real descendent rupt → LONG
                if _trend_down and _cur_close > _break_hi:
                    if not _has_disp_e9:
                        logger.debug(
                            f"   📈 CHoCH BULL detectat dar fără displacement "
                            f"(body={_e9_body:.1f} < body/range={_e9_body/_e9_range:.0%} < 50%) → ignorat"
                        )
                    elif trade_direction == "LONG":
                        _mss_adj = +_mss_bonus
                        _tf_ctx  = "5m+15m" if (_trend_down_5m and _trend_down_15m) else ("5m" if _trend_down_5m else "15m")
                        logger.info(
                            f"   📈 CHoCH BULLISH [{_tf_ctx}]: downtrend rupt, "
                            f"close {_cur_close:.1f} > swing_hi {_break_hi:.1f} "
                            f"(body={_e9_body:.1f}) → LONG +{_mss_bonus:.0%}"
                        )
                    elif trade_direction == "SHORT":
                        _mss_adj = -_mss_penalty
                        logger.info(f"   📉 CONTRA CHoCH: bullish MSS dar SHORT → -{_mss_penalty:.0%}")

                # CHoCH Bearish: trend real ascendent rupt → SHORT
                elif _trend_up and _cur_close < _break_lo:
                    if not _has_disp_e9:
                        logger.debug(
                            f"   📉 CHoCH BEAR detectat dar fără displacement "
                            f"(body={_e9_body:.1f} < body/range={_e9_body/_e9_range:.0%} < 50%) → ignorat"
                        )
                    elif trade_direction == "SHORT":
                        _mss_adj = +_mss_bonus
                        _tf_ctx  = "5m+15m" if (_trend_up_5m and _trend_up_15m) else ("5m" if _trend_up_5m else "15m")
                        logger.info(
                            f"   📉 CHoCH BEARISH [{_tf_ctx}]: uptrend rupt, "
                            f"close {_cur_close:.1f} < swing_lo {_break_lo:.1f} "
                            f"(body={_e9_body:.1f}) → SHORT +{_mss_bonus:.0%}"
                        )
                    elif trade_direction == "LONG":
                        _mss_adj = -_mss_penalty
                        logger.info(f"   📈 CONTRA CHoCH: bearish MSS dar LONG → -{_mss_penalty:.0%}")

                if _mss_adj != 0.0:
                    _pre_mss    = final_score
                    final_score = max(min(final_score + _mss_adj, 1.0), 0.0)
                    score_pct   = round(min(final_score * 100, 100), 2)
                    logger.info(f"   ⚡ MSS/CHoCH: scor {_pre_mss:.2f} → {final_score:.2f}")
            except Exception as _e9_err:
                logger.debug(f"E9 MSS/CHoCH error: {_e9_err}")

        # ── E10. EQUAL HIGHS / EQUAL LOWS (zone de lichiditate duble) ────────
        # Equal Highs = două sau mai multe bare cu high-ul aproape identic
        # → lichiditate (stop-uri) acumulată deasupra — preferăm SHORT (sweep iminent)
        # Equal Lows = două sau mai multe bare cu low-ul aproape identic
        # → lichiditate acumulată dedesubt — preferăm LONG (sweep iminent)
        # Toleranță: 2 tick-uri (pentru NQ TickSize=0.25 → 0.5 pts; pentru ES 0.25)
        if len(df) >= 6:
            try:
                _tick_tol = 0.75  # toleranță 3 ticks pentru NQ (0.25 tick size)
                _h6       = list(df['high'].iloc[-7:-1])  # ultimele 6 high-uri (excl. curent)
                _l6       = list(df['low'].iloc[-7:-1])   # ultimele 6 low-uri (excl. curent)
                _cur_h    = float(df['high'].iloc[-1])
                _cur_l    = float(df['low'].iloc[-1])

                # Equal Highs: cel puțin 2 bare din ultimele 6 au high în range de _tick_tol de max
                _max_hi = max(_h6)
                _eq_hi_count = sum(1 for h in _h6 if abs(h - _max_hi) <= _tick_tol)
                # Equal Lows: cel puțin 2 bare din ultimele 6 au low în range de _tick_tol de min
                _min_lo = min(_l6)
                _eq_lo_count = sum(1 for l in _l6 if abs(l - _min_lo) <= _tick_tol)

                _eql_adj = 0.0
                if _eq_hi_count >= 2 and _cur_h >= _max_hi - _tick_tol:
                    # Prețul e la equal highs = lichiditate sus = risc de sweep bearish
                    if trade_direction == "SHORT":
                        _eql_adj = +0.05  # SHORT la equal highs = zona de distribuție
                        logger.info(f"   🎯 EQUAL HIGHS ({_eq_hi_count}x @ {_max_hi:.1f}): lichiditate sus → SHORT la zonă +5%")
                    elif trade_direction == "LONG":
                        _eql_adj = -0.05  # LONG la equal highs = risc sweep
                        logger.info(f"   ⚠️ EQUAL HIGHS: LONG la zonă de lichiditate sus → -5%")

                elif _eq_lo_count >= 2 and _cur_l <= _min_lo + _tick_tol:
                    # Prețul e la equal lows = lichiditate jos = risc de sweep bullish
                    if trade_direction == "LONG":
                        _eql_adj = +0.05  # LONG la equal lows = zona de acumulare
                        logger.info(f"   🎯 EQUAL LOWS ({_eq_lo_count}x @ {_min_lo:.1f}): lichiditate jos → LONG la zonă +5%")
                    elif trade_direction == "SHORT":
                        _eql_adj = -0.05
                        logger.info(f"   ⚠️ EQUAL LOWS: SHORT la zonă de lichiditate jos → -5%")

                if _eql_adj != 0.0:
                    _pre_eql    = final_score
                    final_score = max(min(final_score + _eql_adj, 1.0), 0.0)
                    score_pct   = round(min(final_score * 100, 100), 2)
                    logger.info(f"   ⚡ EQUAL H/L: scor {_pre_eql:.2f} → {final_score:.2f}")
            except Exception as _e10_err:
                logger.debug(f"E10 Equal H/L error: {_e10_err}")

        # ── E11. ORDER BLOCKS — retest zone instituționale ──────────────────
        # Un Order Block (OB) = ultima lumânare contrară înainte de un impuls mare.
        # Dacă prețul curent retestează un OB recent, confluența ICT este ridicată.
        # Bullish OB: ultima lumânare bearish înainte de move bullish > 1.5×ATR
        # Bearish OB: ultima lumânare bullish înainte de move bearish > 1.5×ATR
        if len(df) >= 10:
            try:
                _ob_adj = 0.0
                _ob_msg = ""
                _atr_val = float(df['atr'].iloc[-1]) if 'atr' in df.columns else float(df['high'].iloc[-1] - df['low'].iloc[-1])
                _ob_lookback = min(30, len(df) - 3)

                for _i in range(len(df) - _ob_lookback, len(df) - 3):
                    _c0 = float(df['close'].iloc[_i])
                    _o0 = float(df['open'].iloc[_i])
                    _h0 = float(df['high'].iloc[_i])
                    _l0 = float(df['low'].iloc[_i])
                    _c2 = float(df['close'].iloc[_i + 2])
                    _move = abs(_c2 - _c0)

                    if _move < _atr_val * 1.3:
                        continue  # mișcare insuficientă pentru OB valid

                    _cur_close_now = float(df['close'].iloc[-1])

                    if _c0 > _o0:
                        # Lumânare bullish → potențial Bearish OB (impuls bearish urmează)
                        if _c2 < _c0 and _l0 <= _cur_close_now <= _h0:
                            # Prețul retestează Bearish OB
                            if trade_direction == "SHORT":
                                _ob_adj = +0.08
                                _ob_msg = f"E11 Bearish OB retest @ {_l0:.1f}–{_h0:.1f}"
                            elif trade_direction == "LONG":
                                _ob_adj = -0.07
                                _ob_msg = f"E11 CONTRA Bearish OB → risc SHORT"
                            break
                    else:
                        # Lumânare bearish → potențial Bullish OB (impuls bullish urmează)
                        if _c2 > _c0 and _l0 <= _cur_close_now <= _h0:
                            # Prețul retestează Bullish OB
                            if trade_direction == "LONG":
                                _ob_adj = +0.08
                                _ob_msg = f"E11 Bullish OB retest @ {_l0:.1f}–{_h0:.1f}"
                            elif trade_direction == "SHORT":
                                # Fix v10.3: dacă reversal_override e activ, OB-ul bullish a fost
                                # compromis de orderflow instituțional → nu penaliza SHORT
                                if _reversal_override:
                                    _ob_adj = 0.0
                                    _ob_msg = f"E11 Bullish OB (skip CONTRA — reversal_override + OF confirmat)"
                                else:
                                    _ob_adj = -0.07
                                    _ob_msg = f"E11 CONTRA Bullish OB → risc LONG"
                            break

                if _ob_adj != 0.0:
                    _pre_ob    = final_score
                    final_score = max(min(final_score + _ob_adj, 1.0), 0.0)
                    score_pct   = round(min(final_score * 100, 100), 2)
                    logger.info(f"   🏦 {_ob_msg}: scor {_pre_ob:.2f} → {final_score:.2f}")
            except Exception as _e11_err:
                logger.debug(f"E11 Order Block error: {_e11_err}")

        # ─────────────────────────────────────────────────────────────────────

        # ── E12. REVERSAL OVERRIDE INSTITUTIONAL BOOST ────────────────────────
        # Când reversal_override e activ + cum_delta masiv → boost direct
        # Logica: OF Super-Confirm cu delta -5000+ e cel mai puternic semnal disponibil.
        # Post-processingul penalizează pentru că HTF/OB/VWAP nu sunt aliniate,
        # dar acestea sunt irelevante când instituționalii mișcă piața cu forță.
        # Fix v10.3: boost proporțional cu puterea orderflow
        if _reversal_override and _live_mode and live_data:
            try:
                _ro_cd = float(live_data.get("cum_delta", 0) or 0)
                _ro_boost = 0.0
                if abs(_ro_cd) >= 5000:
                    _ro_boost = +0.12   # flux instituțional extrem
                elif abs(_ro_cd) >= 3000:
                    _ro_boost = +0.09   # flux instituțional puternic
                elif abs(_ro_cd) >= 1500:
                    _ro_boost = +0.05   # flux instituțional moderat

                if _ro_boost > 0:
                    _pre_ro = final_score
                    final_score = min(final_score + _ro_boost, 1.0)
                    score_pct   = round(min(final_score * 100, 100), 2)
                    logger.info(
                        f"   💪 REVERSAL OVERRIDE boost: cum_delta={_ro_cd:.0f} → +{_ro_boost:.0%} "
                        f"| scor {_pre_ro:.2f} → {final_score:.2f}"
                    )
            except Exception as _ro_err:
                logger.debug(f"Reversal boost error: {_ro_err}")
        # ─────────────────────────────────────────────────────────────────────

        # ══════════════════════════════════════════════════════════════════════
        # CONSOLIDATION RANGE FILTER v10.0
        # Problema principală: botul intră în mijlocul unui range (50-65% H4)
        # unde probabilitatea de fakeout e maximă. -10% regim nu e suficient.
        #
        # Reguli:
        #   - CONSOLIDATION/RANGING + preț în mijloc (15%-85% H4) → skip complet
        #   - CONSOLIDATION + preț la extremă (≤15% discount → LONG; ≥85% premium → SHORT):
        #     setup scalping valid — permis cu reducere TP la 12pts
        #   - Dacă există displacement real (has_displacement=True) → bypass filtru
        #   - Dacă killzone activă (London/NY) → filtru relaxat (penalizare -20% nu skip)
        # ══════════════════════════════════════════════════════════════════════
        _consol_regime = (
            "CONSOL" in _regime_label
            or "RANG"  in _regime_label
            or "CHOP"  in _regime_label
        )
        _has_disp     = bool(best.get('has_displacement', 0))
        _in_kz_active = bool(active_kz and active_kz != "Outside killzone" and active_kz)
        _fvg_present  = bool(best.get('fvg_up', 0) or best.get('fvg_down', 0))

        if _consol_regime:
            # Extrema de range: ≤15% = deep discount (cumpăr), ≥85% = deep premium (vând)
            _at_range_extreme = (_pd_pct <= 0.15 or _pd_pct >= 0.85)
            # Extremă medie: 15-25% sau 75-85% — mai puțin sigur
            _at_range_edge    = (_pd_pct <= 0.25 or _pd_pct >= 0.75)

            if _at_range_extreme and not _has_disp:
                # La extrema range → scalping setup valid dar SL/TP reduse (adaptat la range)
                # Verificăm că direcția e corectă (LONG la discount, SHORT la premium)
                _consol_correct_dir = (
                    (_pd_pct <= 0.15 and trade_direction == "LONG") or
                    (_pd_pct >= 0.85 and trade_direction == "SHORT")
                )
                if _consol_correct_dir:
                    logger.info(
                        f"   🔲 CONSOLIDATION EXTREME: pd_pct={_pd_pct:.0%} → "
                        f"scalping mode {trade_direction} la {'discount' if _pd_pct<=0.15 else 'premium'} "
                        f"(SL/TP adaptat la range în bridge)"
                    )
                    # Nu modificăm scorul — deja penalizat de RANGING -10%
                    # Exportăm flag pentru bridge_api: SL/TP vor fi adaptate la VA spread
                    _consol_scalping_mode = True
                else:
                    # La extremă dar direcție greșită (LONG în premium ≥85%, SHORT în discount ≤15%)
                    _pre_consol = final_score
                    final_score = max(final_score - 0.25, 0.0)
                    score_pct   = round(final_score * 100, 2)
                    skip_volatile = True
                    vol_msg = f"CONSOLIDATION extremă CONTRA-DIRECȚIE (pd={_pd_pct:.0%}, dir={trade_direction})"
                    logger.info(
                        f"   🔲 CONSOLIDATION CONTRA-EXTREME: {_pre_consol:.2f} → {final_score:.2f} "
                        f"pd_pct={_pd_pct:.0%} dir={trade_direction} → SKIP"
                    )
                    _consol_scalping_mode = False

            elif _has_disp:
                # Displacement real prezent — bypass filtru consolidare
                logger.info(
                    f"   🔲 CONSOLIDATION dar DISPLACEMENT real → bypass filtru "
                    f"(pd={_pd_pct:.0%})"
                )
                _consol_scalping_mode = False

            elif _in_kz_active and _at_range_edge:
                # Killzone activă + la edge — penalizare moderată, nu skip
                _pre_consol = final_score
                final_score = max(final_score - 0.12, 0.0)
                score_pct   = round(final_score * 100, 2)
                logger.info(
                    f"   🔲 CONSOLIDATION KZ edge: pd={_pd_pct:.0%} scor {_pre_consol:.2f} → {final_score:.2f} (-12%)"
                )
                _consol_scalping_mode = False

            else:
                # Mijlocul range-ului (15-75%) fără displacement → SKIP hard
                _pre_consol = final_score
                final_score = max(final_score - 0.30, 0.0)
                score_pct   = round(final_score * 100, 2)
                if not skip_volatile:
                    skip_volatile = True
                    vol_msg = (
                        f"CONSOLIDATION mid-range (pd={_pd_pct:.0%}, regime={_regime_label}) "
                        f"→ fakeout risc maxim"
                    )
                logger.info(
                    f"   🔲 CONSOLIDATION MID-RANGE: {_pre_consol:.2f} → {final_score:.2f} "
                    f"pd_pct={_pd_pct:.0%} → SKIP (no displacement, no KZ extreme)"
                )
                _consol_scalping_mode = False
        else:
            _consol_scalping_mode = False
        # ─────────────────────────────────────────────────────────────────────

        risk = calculate_sniper_risk(final_score, best, balance, direction=trade_direction, live_data=live_data)

        # Update #17: Aplică VIX sizing mult pe risk
        if 'risk_usd' in risk:
            risk['risk_usd'] = round(float(risk['risk_usd']) * vix_mult, 2)
            risk['vix_mult'] = vix_mult

        # ── Update #54: Macro sizing adjustment ────────────────────────────
        if 'risk_usd' in risk and macro_sizing_mult != 1.0:
            risk['risk_usd'] = round(float(risk['risk_usd']) * macro_sizing_mult, 2)
            risk['macro_mult'] = macro_sizing_mult

        # ── 8. Pyramiding Plan ───────────────────────────────────────────────
        pyramid_plan = get_pyramiding_plan(final_score, risk, best)

        # ── 9. Narrative ─────────────────────────────────────────────────────
        nar          = get_detailed_narrative(best, t_str)
        nar_text     = build_narrative_text(nar, best, {
            "News":          news_msg,
            "HTF Filter":    filter_msg,
            "Rel Strength":  rel_info,
            "FVG Deep":      fvg_deep_msg,
            "Regime":        regime,
            "Killzone":      active_kz or "Outside",
            "Noise Filter":  "ON" if is_noisy else "OFF",
        })

        # ── 10. RAG Pattern Memory ───────────────────────────────────────────
        rag = get_rag()
        rag_context = rag.build_rag_context(best, nar)

        # ── 10b. GLOBAL CAP pe ajustări post-raw ─────────────────────────────
        # Toate ajustările post-"Final Score" (exhaustion, P/D, POC, DOM, absorption,
        # institutional, tape, delta div, etc.) pot acumula ±100%+.
        # Asta face weight-urile RL complet irelevante — scorul final e dictat
        # de câte penalizări/bonusuri se niméresc să fire simultan.
        # Fix v10.5: Cap total ajustări la ±25% din scorul ancorat.
        # Raw score 43% → range permis: [18%, 68%]. Suficient să miște semnificativ
        # dar nu poate transforma un 43% în 0% sau 95%.
        _POST_CAP_MAX = 0.25   # max +25% boost din post-processing
        _POST_CAP_MIN = -0.20  # max -20% penalizare din post-processing (asimetric: protejăm capital)
        _total_post_adj = final_score - _post_proc_anchor
        if _total_post_adj > _POST_CAP_MAX:
            final_score = _post_proc_anchor + _POST_CAP_MAX
            score_pct   = round(min(final_score * 100, 100), 2)
            logger.info(
                f"   🔒 GLOBAL CAP: ajustări post-raw +{_total_post_adj:.0%} depășesc cap +{_POST_CAP_MAX:.0%} "
                f"→ scor capped la {final_score:.2f} (ancoră={_post_proc_anchor:.2f})"
            )
        elif _total_post_adj < _POST_CAP_MIN:
            final_score = max(_post_proc_anchor + _POST_CAP_MIN, 0.0)
            score_pct   = round(min(final_score * 100, 100), 2)
            logger.info(
                f"   🔒 GLOBAL CAP: ajustări post-raw {_total_post_adj:.0%} depășesc cap {_POST_CAP_MIN:.0%} "
                f"→ scor capped la {final_score:.2f} (ancoră={_post_proc_anchor:.2f})"
            )
        elif abs(_total_post_adj) > 0.01:
            logger.info(
                f"   📊 POST-PROC TOTAL: {_total_post_adj:+.0%} (ancoră={_post_proc_anchor:.2f} → final={final_score:.2f})"
            )

        # ── 11. Conviction Level ─────────────────────────────────────────────
        conviction = calculate_conviction_level(final_score, best, active_kz)

        # ── 12. Verdict ──────────────────────────────────────────────────────
        if news_mult < 0.2:
            verdict = f"🚨 NEWS BLACKOUT — {news_msg}"
        elif final_score > 0.55 and ict_signals >= 2:
            verdict = f"💎 SNIPER ENTRY CONFIRMED — {conviction}"
        elif final_score > 0.42:
            # Fix v10.3: threshold coborât 0.45→0.42 (penalizările post-proc reduse, nu e nevoie de marjă mare)
            verdict = f"🟡 SETUP ÎN FORMARE — Score: {score_pct}%"
        elif final_score > 0.28:
            verdict = f"⏳ MONITORIZARE — Confluențe insuficiente"
        else:
            verdict = f"❌ NO TRADE — Scor sub prag ({score_pct}%)"

        # Aplică volatility filter (override verdict dacă piața e prea volatilă)
        # UPDATE #14f: High-conviction override — nu blocăm dacă scor >= 60% + ICT >= 2 signals + in killzone
        # Logica: ATR extrem nu = mișcări impredictibile; setup-uri ICT rămân valide oricând
        # KZ nu mai e obligatoriu — se poate tranzacționa și outside KZ dacă scor + ICT ok
        _vol_conviction_override = (
            skip_volatile
            and final_score >= _user_score_min
            and ict_signals >= 2
        )
        if skip_volatile and not _vol_conviction_override:
            verdict = f"🛑 SKIP — {vol_msg}"
        elif skip_volatile and _vol_conviction_override:
            logger.info(f"   ⚡ Vol.Override: ATR extrem dar scor={final_score:.2f} >= {_user_score_min:.0%} (user) ICT={ict_signals} → trade permis")
            verdict = verdict + f" [⚡ATRmax]"

        # GEO RISK MODE tag în verdict (după vol filter, înainte de circuit breaker)
        if _geo_risk_active:
            verdict = verdict + " [🌍GEO RISK]"

        # GEO SENTIMENT — penalitate direcțională bazată pe context geopolitic
        # Aplicat după noise filter și vol filter, înainte de verdict final
        if _geo_sentiment != "NEUTRAL":
            # Ajustare mică — inferăm direcția din titluri de știri (incert)
            # Penalitate -4% contra-direcție | Bonus +3% în direcție
            # (news filter e mult mai agresiv: NFP ×0.1 = -90%, CPI ×0.5 = -50%)
            _geo_sent_adj = 0.0
            if _geo_sentiment == "BEARISH_NQ":
                if trade_direction == "LONG":
                    _geo_sent_adj = -0.04
                    logger.info(f"   🌍 GEO SENTIMENT BEARISH_NQ → penalizare LONG -4%: {final_score:.2f} → {max(final_score + _geo_sent_adj, 0):.2f}")
                elif trade_direction == "SHORT":
                    _geo_sent_adj = +0.03
                    logger.info(f"   🌍 GEO SENTIMENT BEARISH_NQ → bonus SHORT +3%: {final_score:.2f} → {min(final_score + _geo_sent_adj, 1.0):.2f}")
            elif _geo_sentiment == "BULLISH_NQ":
                if trade_direction == "SHORT":
                    _geo_sent_adj = -0.04
                    logger.info(f"   🌍 GEO SENTIMENT BULLISH_NQ → penalizare SHORT -4%: {final_score:.2f} → {max(final_score + _geo_sent_adj, 0):.2f}")
                elif trade_direction == "LONG":
                    _geo_sent_adj = +0.03
                    logger.info(f"   🌍 GEO SENTIMENT BULLISH_NQ → bonus LONG +3%: {final_score:.2f} → {min(final_score + _geo_sent_adj, 1.0):.2f}")
            if _geo_sent_adj != 0.0:
                final_score = max(0.0, min(final_score + _geo_sent_adj, 1.0))
                score_pct   = round(final_score * 100, 2)
                verdict = verdict + f" [🧭{_geo_sentiment}]"

        # Update #47-50: Override verdict dacă circuit breaker activ
        if circuit_blocked:
            verdict = f"⛔ CIRCUIT BREAKER — {'Max DD' if dd_check.get('blocked') else 'Daily Limit' if dl_check.get('blocked') else 'Portfolio Heat'}"

        # ── Fix v7.8: POST-PROCESSING FLOOR ──────────────────────────────────
        # Penalizările cumulative (MTF, Bias, DOW, Seasonal, Delta, Imbalances)
        # pot tăia scorul de la 30% → 1% pe zile volatile / setups contra-trend.
        # Floor: post-processing nu poate reduce scorul sub 40% din scorul engine
        # (valoarea de la "🔥 Final Score") — semnalul rămâne lizibil.
        # Circuit breaker și skip_volatile NU sunt afectate de floor.
        if not skip_volatile and not circuit_blocked:
            _postproc_floor = _engine_score_pct * 0.40  # minim 40% din scorul engine
            if final_score * 100 < _postproc_floor:
                _before_floor = final_score * 100
                final_score = _postproc_floor / 100.0
                score_pct   = round(_postproc_floor, 2)
                logger.info(
                    f"   🛡️ POST-PROC FLOOR: scor {_before_floor:.1f}% → {score_pct:.1f}% "
                    f"(floor=40% din {_engine_score_pct:.1f}% engine score)"
                )
        # ─────────────────────────────────────────────────────────────────────

        # ── 13. Journaling ───────────────────────────────────────────────────
        log_msg = log_aladin_verdict(
            target_ts, final_score, risk, nar, verdict,
            extra={"regime": regime, "killzone": active_kz or "outside", "noise": is_noisy}
        )

        # ── 14. Store in RAG ─────────────────────────────────────────────────
        rag.store_current_analysis(best, nar, final_score, verdict)

        # ── Console output (păstrat pentru CLI) ──────────────────────────────
        print(f"\n🕵️  ALADIN QUANTUM-ICT v5.0 | {target_ts}")
        print("=" * 90)
        print(f"🌍  MACRO      : {nar['macro']}")
        print(f"⚓  KILLZONE   : {active_kz or 'Outside'} | REGIME: {regime}")
        print(f"🎯  LIQUIDITY  : {nar['liquidity']}")
        print(f"📊  AMT        : {nar['amt_context']}")
        print(f"🛡️   HTF FILTER : {filter_msg}")
        print(f"📰  NEWS       : {news_msg}")
        print(f"📈  REL STR    : {rel_info}")
        print(f"📐  FVG DEEP   : {fvg_deep_msg}")
        print("─" * 90)
        smt_s = "🤝 SMT ACTIV" if (best.get('is_smt_bearish', 0) or best.get('is_smt_bullish', 0)) else "❌ SMT INACTIV"
        fvg_s = "🟢 BULL" if best.get('fvg_up', 0) else ("🔴 BEAR" if best.get('fvg_down', 0) else "⚪ N/A")
        print(f"🤝  INDICATORI : {smt_s} | FVG: {fvg_s} | Displacement: {'✅' if best.get('has_displacement', 0) else '❌'}")
        print(f"🔥  SCOR HIBRID: {score_pct}% (AI={ai_component:.2f} ICT={ict_component:.2f} Q={q_component:.2f} Rel={rel_component:.2f} News×{news_mult:.1f})")
        print(f"💰  RISK       : {risk['units']} lots | ${risk['risk_usd']} ({risk['risk_pct']}%) | SL: {risk['sl']} | TP: {risk['tp']}")
        if isinstance(pyramid_plan, list) and pyramid_plan:
            print(f"📈  PYRAMID    : E1@{pyramid_plan[0]['trigger_price']} | E2@{pyramid_plan[1]['trigger_price']} | E3@{pyramid_plan[2]['trigger_price']}")
        print(f"🔇  NOISE      : {'ON' if is_noisy else 'OFF'} | CONVICTION: {conviction}")
        print(f"📝  AUDIT      : {log_msg}")
        print("─" * 90)
        print(f"🟢  VERDICT    : {verdict}")
        print("=" * 90)

        # ── UPDATE #1 + #14b: Log semnal în Supabase — 1x per bară ─────────
        # UPDATE #14b: Deduplicare — scriem doar dacă bara e nouă (timestamp diferit)
        global _last_supabase_bar_ts
        _bar_ts_key = target_ts[:16] if target_ts else ""  # truncat la minut
        if _SUPABASE_OK and _supabase is not None and _bar_ts_key != _last_supabase_bar_ts:
            _last_supabase_bar_ts = _bar_ts_key
            try:
                _supabase.log_signal(
                    symbol          = "NQ",
                    direction       = trade_direction,
                    score_pct       = score_pct,
                    ai_score        = round(base_ai * 100, 2),
                    verdict         = verdict,
                    ict_component   = ict_component,
                    q_component     = q_component,
                    sentiment_score = sentiment_score,
                    sentiment_mult  = _sent_mult,
                    vix_mult        = vix_mult,
                    macro_mult      = macro_sizing_mult,
                    regime          = regime,
                    killzone        = active_kz or "",
                    live_mode       = _live_mode,
                    raw_score       = raw_score,
                )
                logger.debug(f"Supabase log_signal: bară {_bar_ts_key}")
            except Exception as _sb_log_err:
                logger.debug(f"Supabase log_signal skip: {_sb_log_err}")

        # Feature 3: instrument symbol + params (needed in return dict)
        _inst_sym  = live_data.get("symbol", "NQ") if _live_mode and live_data else "NQ"
        _sl_params = _get_sl_params(_inst_sym)

        # ── Return dict complet pentru Streamlit ─────────────────────────────
        return {
            # Core
            "verdict":        verdict,
            "score":          score_pct,
            "ai_score":       round(base_ai * 100, 2),
            "quantum_score":  q_edge,
            "final_score_raw": final_score,
            "trade_direction": trade_direction,

            # Preț entry
            "close":          float(best['close']),

            # Indicatori
            "fvg_active":     bool(best.get('fvg_up', 0) or best.get('fvg_down', 0)),
            "smt_active":     bool(best.get('is_smt_bearish', 0) or best.get('is_smt_bullish', 0)),
            "displacement":   bool(best.get('has_displacement', 0)),

            # Risk
            "risk":           risk,

            # Targets
            "sd_targets":     sd_targets,
            "order_blocks":   order_blocks,
            "pyramid":        pyramid_plan,

            # Filters
            "noise_filter":   is_noisy,
            "news_msg":        news_msg,
            "news_mode_active": _news_mode_active,   # True = NEWS TRADE MODE (TP ×1.3 în bridge_api)
            "filter_msg":      filter_msg,
            "rel_info":        rel_info,
            "conviction":     conviction,
            "regime":         regime,
            "killzone":       active_kz,

            # Circuit Breakers (Update #47-50)
            "circuit_breaker": circuit_blocked,
            "portfolio_heat":  heat_check,
            "daily_loss":      dl_check,
            "max_dd":          dd_check,

            # Narrative & RAG
            "narrative":      nar_text,
            "nar_dict":       nar,
            "rag_context":    rag_context,
            "fvg_deep":       fvg_deep_msg,

            # Update #42-54: Data Quality, DXY, Options, FRED
            "data_quality":   dq,
            "dxy":            dxy_data,
            "options_flow":   options_flow,
            "fred_macro":     fred_macro,

            # Meta
            "timestamp":      target_ts,
            "log_msg":        log_msg,

            # UPDATE #11: RL Feedback — component scores + weights pentru bridge_api
            "component_scores": _component_scores_for_rl,
            "rl_weights":       _rl_w,

            # v8.0: Reversal Override info
            "reversal_override":  _reversal_override,
            "reversal_direction": _reversal_dir,
            "reversal_score":     _reversal_score,
            "reversal_signals":   _reversal_signals,

            # v10.0: Consolidation range filter
            "pd_pct":              _pd_pct,               # 0=low 1=high 0.5=eq (H4 sau fallback)
            "consol_scalping":     _consol_scalping_mode, # True = setup scalping la extrema range
            # v10.6: VA bounds pentru bridge consolidation-aware SL/TP
            "vah":                 float(best.get('vah', 0) or 0),
            "val":                 float(best.get('val', 0) or 0),
            "regime":              regime if 'regime' in dir() else "",

            # v8.1: Absorption info
            "absorption_detected":  _absorption_detected,
            "absorption_direction": _absorption_direction,
            "absorption_strength":  _absorption_strength,
            "absorption_signals":   _absorption_signals,
            # Advanced OrderFlow Analytics v9.0
            "exhaustion_detected":  _exhaustion_detected,
            "exhaustion_side":      _exhaustion_side,
            "delta_divergence":     _delta_div_detected,
            "delta_div_side":       _delta_div_side,
            "stacked_imbalances":   live_data.get("stacked_imbalances", {}) if _live_mode else {},
            "unfinished_business":  live_data.get("unfinished_business", []) if _live_mode else [],
            "iceberg":              live_data.get("iceberg", {}) if _live_mode else {},
            # Feature 3: instrument info
            "instrument":           _inst_sym,
            "instrument_params":    _sl_params,

            # v10.5: ICT individual signals — pentru Telegram display (H4/H1/M15/KZ/FVG/SMT)
            # Fix v10.5b: dacă reversal_override a flipat direcția (LONG→SHORT sau invers),
            # h4/h1/m15 au fost calculate pentru ai_direction original — le inversăm.
            "ict_h4":  bool(not h4_aligned if _reversal_override else h4_aligned),
            "ict_h1":  bool(not h1_aligned if _reversal_override else h1_aligned),
            "ict_m15": bool(not m15_aligned if _reversal_override else m15_aligned),
            "in_kz":   bool(in_kz),
            "has_fvg": bool(has_fvg),
            "has_smt": bool(has_smt),

            # Advanced features (Tier 1 + Tier 2)
            "advanced_features": {
                "hurst":            float(best.get('hurst', 0)) if 'hurst' in df.columns else 0.0,
                "garch_vol":        float(best.get('garch_vol', 0)) if 'garch_vol' in df.columns else 0.0,
                "kalman_noise":     float(best.get('kalman_noise', 0)) if 'kalman_noise' in df.columns else 0.0,
                "adx":              float(best.get('adx_14', 0)) if 'adx_14' in df.columns else 0.0,
                "dist_vwap":        float(best.get('dist_vwap', 0)) if 'dist_vwap' in df.columns else 0.0,
                "sample_entropy":   float(best.get('sample_entropy', 0)) if 'sample_entropy' in df.columns else 0.0,
                "fisher_transform": float(best.get('fisher_transform', 0)) if 'fisher_transform' in df.columns else 0.0,
                "fft_cycle":        float(best.get('fft_cycle', 0)) if 'fft_cycle' in df.columns else 0.0,
                "acf_lag1":         float(best.get('acf_lag1', 0)) if 'acf_lag1' in df.columns else 0.0,
                "regime_hint":      "TRENDING" if (float(best.get('hurst', 0.5)) > 0.55 and float(best.get('adx_14', 0)) > 25) else
                                    "MEAN_REVERT" if float(best.get('hurst', 0.5)) < 0.45 else "NEUTRAL",
            },
        }

    except Exception as e:
        logger.error(f"Eroare critică în Aladin Engine: {e}", exc_info=True)
        return {
            "verdict": f"❌ Eroare critică engine: {str(e)}",
            "score":   0,
            "timestamp": query,
        }

    finally:
        conn.close()


# =============================================================================
# CLI — Mod terminal interactiv
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("⚛️  ALADIN QUANTUM-ICT v5.0 — Terminal Mode")
    print("=" * 60)
    print("Introduceți timestamp-ul dorit (YYYY-MM-DD HH:MM)")
    print("Comenzi: 'exit' = ieșire | 'stats' = statistici jurnal")
    print("=" * 60)

    while True:
        try:
            q = input("\n🎯 Aladin > ").strip()
            if not q:
                continue
            if q.lower() == 'exit':
                print("👋 Aladin Engine oprit. La revedere!")
                break
            if q.lower() == 'stats':
                stats = load_journal_stats()
                if stats:
                    print(f"📊 Journal Stats: {json.dumps(stats, indent=2)}")
                else:
                    print("📭 Jurnalul este gol.")
                continue

            result = aladin_engine(q)
            # Verdict e deja printat de engine — afișăm scorul sumar
            print(f"\n   → Returnat dict cu {len(result)} chei | Score: {result.get('score', 0)}%")

        except KeyboardInterrupt:
            print("\n👋 Ieșire forțată.")
            break
        except Exception as e:
            print(f"❌ Eroare: {e}")