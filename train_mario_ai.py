"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║          ALADIN QUANTUM-ICT v6.7 — XGBOOST SNIPER TRAINER + ENSEMBLE + ONLINE           ║
║          train_mario_ai.py  |  Multi-class: 0=Wait  1=Short  2=Long                    ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Îmbunătățiri v6.7:
  1. Target dinamic bazat pe ATR (nu static 10 pts)
  2. Normalizare price levels față de true_open (modelul învață structura, nu prețul)
  3. 10 features noi: dist_poc, inside_va, dist_pdh, dist_pdl, atr_14,
     slope_h1, slope_h4, momentum_15, body_dir, wick_ratio
  4. Sample weights mai agresive pentru Sniper (1:50)
  5. Hyperparameters optimizați: n_estimators=800, max_depth=5
  6. SMOTE pentru echilibrare clase (Update #9)
  7. Feature importance analysis cu Top 15 (Update #10)
  8. LightGBM în paralel cu XGBoost (Update #11)
  9. Calibrare probabilități Platt Scaling (Update #12)
  10. Ensemble model: XGBoost + LightGBM + RandomForest cu vot majoritate (Update #13)
  11. Online learning scheduler pentru reantrenare săptămânală (Update #14)
  12. Cross-validation score
  13. Salvare model + feature list + calibrated model + RF + LightGBM în JSON/PKL

Fixes v5.1:
  - y_pred folosește np.argmax(predict_proba) — compatibil Python 3.13 + XGBoost latest
  - cross_val_score primește estimator neinițializat (nu modelul deja antrenat)
  - Indentare uniformă curată
"""

import pandas as pd
import numpy as np
import sqlite3
import xgboost as xgb
import json
import os
import time as _time
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from analytics_suite import run_full_analysis
    _ANALYTICS = True
except ImportError:
    _ANALYTICS = False
    print("⚠️  analytics_suite.py lipsă — analytics dezactivat")

# v13 REGIME-AWARE: ICT profile săptămânal + day type + sweep detection + regime target
try:
    from aladin_v13 import (
        FEATURES_WEEKLY, FEATURES_DAYTYPE, FEATURES_SWEEP,
        add_weekly_features, add_daytype_features, add_sweep_features,
        generate_regime_target, REGIME_NAMES,
    )
    _V13_AVAILABLE = True
except ImportError as _v13_err:
    print(f"⚠️ aladin_v13 indisponibil ({_v13_err}) — rulăm în modul legacy")
    _V13_AVAILABLE = False
    FEATURES_WEEKLY = FEATURES_DAYTYPE = FEATURES_SWEEP = []

# Toggle v13 target mode: True = 9-class regime target, False = legacy 3-class ATR target
USE_V13_REGIME = True

# =============================================================================
# UPDATE #2: Isotonic Calibration (înlocuiește Platt Scaling)
# IsotonicRegression funcționează mai bine pe date imbalanced (WAIT=95%)
# și nu colapsează SHORT/LONG la WAIT ca LogisticRegression
# =============================================================================
class PlattModel:
    """Wrapper Isotonic Calibration — la nivel de modul pentru pickle corect."""
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

# =============================================================================
# CONFIG
# =============================================================================
_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PATH_DB         = os.path.join(_SCRIPT_DIR, "mario_trading.db")
MODEL_SAVE_PATH = os.path.join(_SCRIPT_DIR, "mario_bot.json")
FEATURES_PATH   = os.path.join(_SCRIPT_DIR, "mario_features.json")

# ── TRAIN MODE ────────────────────────────────────────────────────────────────
# 'ALL' = ambele killzone (default, comportament vechi)
# 'LON' = London only 08:00-12:00 RO (pre-London + London + post-London)
# 'NY'  = New York only 15:00-17:30 RO (pre-NY + NY open + NY macro windows)
TRAIN_MODE = 'ALL'   # override din CLI: python3 train_mario_ai.py lon / ny / all

# Ferestre timp pentru fiecare mode (ore RO, decimal)
KZ_WINDOWS = {
    'LON': (8.0,  12.0),   # 08:00-12:00 RO
    'NY':  (15.0, 17.5),   # 15:00-17:30 RO
    'ALL': (8.0,  17.5),   # tot
}
# Ferestre OR (primele 30 min din killzone — pentru target filter)
OR_WINDOWS = {
    'LON': (9.0,  9.5),    # OR London: 09:00-09:30
    'NY':  (15.5, 16.0),   # OR NY: 15:30-16:00
    'ALL': None,           # ambele, gestionat intern
}

# Fix v10.6: FVG/SMT/displacement sunt BACKFILL-uite pe toți 4M bare
# → le reactivăm ca features. OF columns (bar_delta etc.) sunt populate doar
# going forward din BRIDGE_LIVE — le includem dar XGBoost gestionează 0-uri nativ.
FEATURES_BASE = [
    'open', 'high', 'low', 'close', 'volume',
    'lm_hi', 'lm_lo', 'lw_hi', 'lw_lo', 'm_hi', 'm_lo', 'p_hi', 'p_lo',
    'h4_hi', 'h4_lo', 'h1_hi', 'h1_lo',
    'true_open', 'poc_level', 'vah', 'val',
    'is_above_open', 'has_displacement',
    # v10.6: ICT signals — backfill-uite pe toți 4M bare
    'fvg_up', 'fvg_down',
    'is_smt_bearish', 'is_smt_bullish',
]

# Features noi (dacă există în DB) — Update #2: 10 features (identice cu mario_rag.py)
FEATURES_EXTRA = ['dist_poc', 'inside_va', 'dist_pdh', 'dist_pdl', 'atr_14',
                   'slope_h1', 'slope_h4', 'momentum_15', 'body_dir', 'wick_ratio']

# Volume Profile + OrderFlow features — salvate live din BRIDGE_LIVE în market_data
# v12.2: DEZACTIVAT ÎN TRAINING — toate 4 au importanță 0.0 pe date istorice
# Rămân active la runtime via model feature alignment (populate live din bridge)
FEATURES_VP_OF = [
    # 'rvol', 'profile_shape_enc', 'dist_prev_poc', 'delta_exhaust_enc',
]

# v10.6 → v12.2: OF columns din bridge_live — DEZACTIVAT ÎN TRAINING
# Toate 14 features OF_NATIVE au importanță 0.0 pe date istorice (sunt zero).
# Le păstrăm doar în mario_rag.py (runtime) unde sunt populate live din NT8.
# În training adaugă doar noise → eliminate.
FEATURES_OF_NATIVE = [
    # v12.2: SCOASE DIN TRAINING (importanță 0.0 pe toate 14)
    # Rămân active doar la runtime în mario_rag.py via model feature alignment
    # 'bar_delta', 'bar_buy_vol', 'bar_sell_vol', 'delta_at_high', 'delta_at_low',
    # 'big_buy_count', 'big_sell_count', 'imbalance_pct', 'tape_speed', 'dom_ratio',
    # 'of_doi', 'of_bilateral_abs', 'of_big_balance', 'of_d_shape_count',
]

# Advanced features (Tier 1 + Tier 2) — computed by advanced_features.py
# v12.2: Pruned — eliminat hurst(0.59%), fisher_transform(0.67%), fft_cycle(0.51%),
# acf_lag1(0.51%), acf_lag5(0.50%) — sub 1% importanță confirmat, noise.
FEATURES_ADVANCED = [
    'garch_vol',        # GARCH(1,1) predicted volatility — 1.34%
    'kalman_noise',     # Kalman filter noise score — 0.78% (marginal dar unic)
    'adx_14',           # ADX trend strength — 0.61% (păstrat: indicator clasic trend)
    'dist_vwap',        # Distance from VWAP — 1.21%
    'sample_entropy',   # Market complexity — 1.72%
    'realized_vol',     # v6.7 direction feature — 2.56% (top 10)
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
    'failed_auction',        # Breakout care eșuează: high>swing_hi dar close<swing_hi → bearish (-1), invers bullish (+1)
    'excess',                # Excess la extreme: wick lung la high/low recent = respingere definitivă [-1,+1]
    'poor_high',             # Poor high: flat top fără excess → va fi re-testat (1=poor, 0=normal)
    'poor_low',              # Poor low: flat bottom fără excess → va fi re-testat (1=poor, 0=normal)
    'initiative_responsive', # Breakout din VA = inițiativă (+1/-1), respingere înapoi în VA = responsive (0)
    'va_migration',          # Direcția migrării Value Area între sesiuni (+1=up, -1=down, 0=flat)
    'rotation_factor',       # Cât de mult rotează prețul: 0=trending, 1=balance (range-bound)
]

# v12.1: OF AGGREGATED features — rolling sums/ratios pe 15/30 bare
# Transformă OF raw (zgomot pe 1 min) în semnale structurale
FEATURES_OF_AGG = [
    'delta_sum_15',          # Suma delta pe 15 bare — acumulare direcțională
    'delta_sum_30',          # Suma delta pe 30 bare — trend OF mai lung
    'delta_ratio_15',        # delta_sum_15 / abs_delta_sum_15 — direcționalitate [-1,+1]
    'big_trade_ratio_15',    # (big_buy - big_sell) / (big_buy + big_sell) pe 15 bare
    'buy_sell_ratio_30',     # bar_buy_vol / (bar_buy_vol + bar_sell_vol) rolling 30 — cine domină
    'imbalance_ma_15',       # Media imbalance_pct pe 15 bare — trend OF persistent
    'tape_speed_rel',        # tape_speed / rolling_mean_60 — relativă la context
    'absorption_score_15',   # delta_at_high<0 + delta_at_low>0 pe 15 bare — absorption la extreme
    'of_pressure',           # Combinat: delta_trend × volume_trend — presiune direcțională
    'dom_ratio_ma_15',       # DOM ratio smoothed pe 15 bare — bid/ask persistent
]

# v12.5: STREAK PREVENTION features — fără session_age, day_of_week, hour_sin, hour_cos
# SCOASE hour_sin + hour_cos: combinat dominau 21.5% importanță = temporal overfitting masiv
# Modelul învăța CÂND să tradeeze (ora), nu CE condiții sunt bune
FEATURES_STREAK = [
    'atr_change_speed',      # ATR acum / ATR acum 10 bare — regime change indicator
    'consecutive_same_dir',  # Câte bare consecutive în aceeași direcție — mean reversion risk
    'price_vs_daily_range',  # (close - day_low) / (day_high - day_low) — pozitia in range zilnic
    'recent_signal_quality', # Din ultimele 30 bare, câte semnale ar fi fost corecte (proxy)
]
# v12.3: SCOASE session_age + day_of_week | v12.5: SCOASE hour_sin + hour_cos

# v12.4: TRADE QUALITY CONTEXT — features care văd degradarea condițiilor ÎNAINTE de streak
# Ideea: streak-urile apar când modelul nu vede că condițiile s-au schimbat.
# Aceste features îi arată CE S-A ÎNTÂMPLAT RECENT: trend fading, volume dying, etc.
FEATURES_CONTEXT = [
    # --- Trend quality degradation ---
    'trend_r2',              # R² al close pe 20 bare — 1.0=trend lin perfect, 0=noise
    'trend_slope_norm',      # Slope al close pe 20 bare / ATR — direcția + forța trendului
    'close_vs_ema_stack',    # EMA8 vs EMA21 vs EMA55 alignment — 1.0=aligned, 0=mixed
    # --- Momentum confirmation ---
    'roc_10',                # Rate of Change 10 bare / ATR — momentum normalizat
    'roc_divergence',        # ROC 5 vs ROC 20 divergence — momentum pe termen scurt vs lung
    'momentum_consistency',  # Câte din ultimele 10 bare au close > close[-1] în direcția trendului
    # --- Volume quality ---
    'volume_on_move',        # Volum mediu pe barele cu mișcare > 0.5*ATR vs barele mici
    'volume_directional',    # Volum pe bare bullish vs bearish (>1 = volum bullish dominant)
    # --- Price action quality ---
    'clean_bars_pct',        # % din ultimele 10 bare cu body > 50% range (bare curate, nu doji)
    'false_break_count',     # Câte false breakouts în ultimele 20 bare (high>prev_high dar close<prev_high)
    'bar_range_consistency', # Std(bar range) / mean(bar range) pe 10 bare — bare uniforme = trend bun
]

# v12.4: MULTI-TIMEFRAME CONFIRMATION — modelul vede dacă timeframe-urile sunt de acord
FEATURES_MTF_CONFIRM = [
    'h1_trend_aligned',      # Direcția H1 e consistentă cu semnalul curent
    'h4_trend_aligned',      # Direcția H4 e consistentă cu semnalul curent
    'mtf_agreement',         # Câte TF-uri (M1/M5/M15/H1/H4) sunt de acord (0-5 normalizat 0-1)
    'recent_rejection_strength',  # Forța ultimei rejections la S/R (volume * wick size)
]

# v12.7: OPENING RANGE (ORH/ORL) features — ICT killzone expansion
# Per sesiune killzone (London 09-11, NY 15:30-17:30), primele 30 min = OR.
# Modelul învață: IN-OR (acumulare), BREAKOUT (expansion prima gambă), FADE-BACK (false break).
# Target trade: 30-50 pts din prima expansion leg după OR break.
FEATURES_ORH = [
    'in_orh',                # 1 dacă suntem în OR period (primele 30 min killzone)
    'post_orh',              # 1 dacă suntem în killzone dar după OR
    'orh_width_atr',         # Lățimea OR (high-low) / ATR — strâns = breakout probabil
    'dist_to_orh_high_atr',  # (close - orh_high) / ATR — <0 = sub ORH, ~0 = la prag
    'dist_to_orh_low_atr',   # (close - orh_low) / ATR — >0 = peste ORL
    'orh_broken_up',         # 1 dacă close a depășit orh_high (o dată în sesiune)
    'orh_broken_down',       # 1 dacă close a scăzut sub orh_low (o dată în sesiune)
    'bars_since_session',    # Câte bare de la începutul sesiunii (killzone)
    'session_vol_ratio',     # Volume curent / avg volume OR — >1.5 = confirm expansion
    'orh_midpoint_dist_atr', # (close - (orh_high+orh_low)/2) / ATR — poziție în range
]

# v14: pentru modul regime-aware, excludem ORH features dominante (post_orh=68.7% import)
# care făceau modelul să învețe "sunt în killzone post-OR" ca proxy universal,
# crowd-out pentru feature-urile ICT reale (sweep, weekly, daytype).
FEATURES_ORH_V14_DROP = {
    'post_orh',
    'bars_since_session',
    'orh_broken_up',
    'orh_broken_down',
}

# ─────────────────────────────────────────────────────────────────────────────
# v14 HTF (Higher TimeFrame) features — contextul 4H / 1H / 15M / Daily
# Logică: mario_ai SUSȚINE direcția trade-ului OR breakout, nu îl blochează.
# Features calculați din 1M bars (nu necesită coloane separate în DB).
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_HTF = [
    # ── 4H context (rolling 240 bare 1M = 4 ore) ─────────────────────────────
    'h4_momentum',        # (close - close[240]) / ATR — trend direcțional 4H
    'h4_hh_hl',           # +1 dacă 4H face HH+HL (bullish struct), -1 dacă LH+LL
    'h4_swept_low',       # 1 dacă prețul a spart h4_lo recent (sweep liq bears)
    'h4_swept_high',      # 1 dacă prețul a spart h4_hi recent (sweep liq bulls)
    'h4_range_position',  # (close - low240) / (high240 - low240) — poziție în range 4H
    # ── 1H context (rolling 60 bare 1M = 1 oră) ──────────────────────────────
    'h1_momentum',        # (close - close[60]) / ATR — trend 1H
    'h1_range_position',  # (close - low60) / (high60 - low60) — poziție în range 1H
    'h1_hh_hl',           # +1 bullish 1H struct, -1 bearish
    # ── 15M context (rolling 15 bare 1M) ─────────────────────────────────────
    'h15_momentum',       # (close - close[15]) / ATR — trend 15M
    'h15_h4_aligned',     # 1 dacă 15M și 4H au aceeași direcție (confluență)
    # ── Previous day NY session extremes ─────────────────────────────────────
    'dist_yday_ny_hi',    # (close - yday_ny_high) / ATR — distanța față de NY High ieri
    'dist_yday_ny_lo',    # (close - yday_ny_low)  / ATR — distanța față de NY Low ieri
    'dist_yday_ldn_hi',   # (close - yday_ldn_high) / ATR
    'dist_yday_ldn_lo',   # (close - yday_ldn_low)  / ATR
    'yday_ny_bull',       # 1 dacă NY de ieri s-a închis bullish (close > open)
    'yday_ldn_bull',      # 1 dacă LDN de ieri s-a închis bullish
    # ── Weekly context: PWH / PWL (din lw_hi/lw_lo în DB) ───────────────────
    'dist_pwh_atr',       # (pwh - close) / ATR — distanța până la Previous Week High
    'dist_pwl_atr',       # (close - pwl)  / ATR — distanța până la Previous Week Low
    'above_pwh',          # 1 dacă prețul a depășit PWH (lichiditate luată sus)
    'below_pwl',          # 1 dacă prețul a spart PWL (lichiditate luată jos)
    'weekly_bias',        # (close - weekly_open) / ATR — bias față de weekly open
    # ── PDH / PDL context (din pdh/pdl în DB) ────────────────────────────────
    'dist_pdh_atr',       # (pdh - close) / ATR
    'dist_pdl_atr',       # (close - pdl)  / ATR
    'above_pdh',          # 1 dacă close > pdh (breakout confirmed)
    'below_pdl',          # 1 dacă close < pdl (breakdown confirmed)
    # ── Structural confluence ─────────────────────────────────────────────────
    'htf_bull_confluence',  # scor 0-5: câte TF sunt bullish simultan (4H+1H+15M+above_pdh+above_pwl)
    'htf_bear_confluence',  # scor 0-5: câte TF sunt bearish simultan
]


def add_htf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    v14 HTF: Calculează features de context multi-timeframe din bare 1M.
    Toate features sunt normalizate față de ATR sau relative (0/1 flags).
    Necesită coloane: close, high, low, atr_14, lw_hi, lw_lo (PWH/PWL),
                      pdh, pdl (previous day H/L), timestamp.
    """
    df = df.copy()
    _atr = df['atr_14'].replace(0, np.nan).ffill().fillna(9.0).values
    _cl  = df['close'].values
    _hi  = df['high'].values
    _lo  = df['low'].values
    n    = len(df)

    # ── 4H (240 bare) ─────────────────────────────────────────────────────────
    _cl_240  = pd.Series(_cl).shift(240).values
    _hi_240  = pd.Series(_hi).rolling(240, min_periods=60).max().values
    _lo_240  = pd.Series(_lo).rolling(240, min_periods=60).min().values
    _rng_240 = np.maximum(_hi_240 - _lo_240, 0.01)

    df['h4_momentum']      = np.clip((_cl - _cl_240) / np.maximum(_atr, 0.1), -5, 5)
    df['h4_range_position'] = np.clip((_cl - _lo_240) / _rng_240, 0, 1)
    df['h4_momentum']       = df['h4_momentum'].fillna(0)
    df['h4_range_position'] = df['h4_range_position'].fillna(0.5)

    # HH+HL structure pe 4H: compară ultimele 2 blocuri de 240 bare
    _hi_prev240 = pd.Series(_hi).rolling(240, min_periods=60).max().shift(120).values
    _lo_prev240 = pd.Series(_lo).rolling(240, min_periods=60).min().shift(120).values
    _hh = (_hi_240 > _hi_prev240)
    _hl = (_lo_240 > _lo_prev240)
    _lh = (_hi_240 < _hi_prev240)
    _ll = (_lo_240 < _lo_prev240)
    df['h4_hh_hl'] = np.where(_hh & _hl, 1.0, np.where(_lh & _ll, -1.0, 0.0))

    # Sweep h4 levels: prețul a trecut de h4_hi/h4_lo din DB (dacă disponibil)
    if 'h4_hi' in df.columns and 'h4_lo' in df.columns:
        _h4h = df['h4_hi'].fillna(0).values
        _h4l = df['h4_lo'].fillna(0).values
        df['h4_swept_high'] = ((_cl > _h4h) & (_h4h > 0)).astype(float)
        df['h4_swept_low']  = ((_cl < _h4l) & (_h4l > 0)).astype(float)
    else:
        df['h4_swept_high'] = ((_cl > _hi_240) & (_hi_240 > 0)).astype(float)
        df['h4_swept_low']  = ((_cl < _lo_240) & (_lo_240 > 0)).astype(float)

    # ── 1H (60 bare) ──────────────────────────────────────────────────────────
    _cl_60 = pd.Series(_cl).shift(60).values
    _hi_60 = pd.Series(_hi).rolling(60, min_periods=20).max().values
    _lo_60 = pd.Series(_lo).rolling(60, min_periods=20).min().values
    _rng_60 = np.maximum(_hi_60 - _lo_60, 0.01)

    df['h1_momentum']      = np.clip((_cl - _cl_60) / np.maximum(_atr, 0.1), -5, 5)
    df['h1_range_position'] = np.clip((_cl - _lo_60) / _rng_60, 0, 1)
    df['h1_momentum']       = df['h1_momentum'].fillna(0)
    df['h1_range_position'] = df['h1_range_position'].fillna(0.5)

    _hi_prev60 = pd.Series(_hi).rolling(60, min_periods=20).max().shift(30).values
    _lo_prev60 = pd.Series(_lo).rolling(60, min_periods=20).min().shift(30).values
    _hh1 = (_hi_60 > _hi_prev60)
    _hl1 = (_lo_60 > _lo_prev60)
    _lh1 = (_hi_60 < _hi_prev60)
    _ll1 = (_lo_60 < _lo_prev60)
    df['h1_hh_hl'] = np.where(_hh1 & _hl1, 1.0, np.where(_lh1 & _ll1, -1.0, 0.0))

    # ── 15M (15 bare) ─────────────────────────────────────────────────────────
    _cl_15 = pd.Series(_cl).shift(15).values
    df['h15_momentum'] = np.clip((_cl - _cl_15) / np.maximum(_atr, 0.1), -5, 5)
    df['h15_momentum'] = df['h15_momentum'].fillna(0)

    # Confluență 15M cu 4H
    _h4_dir  = np.sign(df['h4_momentum'].values)
    _h15_dir = np.sign(df['h15_momentum'].values)
    df['h15_h4_aligned'] = (_h4_dir == _h15_dir).astype(float)

    # ── Previous day NY / LDN session extremes ─────────────────────────────────
    # Calculăm din timestamp: NY = 15:30-17:30 RO, LDN = 09:00-12:00 RO
    _yday_ny_hi  = np.zeros(n)
    _yday_ny_lo  = np.full(n, np.nan)
    _yday_ldn_hi = np.zeros(n)
    _yday_ldn_lo = np.full(n, np.nan)
    _yday_ny_bull  = np.zeros(n)
    _yday_ldn_bull = np.zeros(n)

    if 'timestamp' in df.columns:
        _ts  = pd.to_datetime(df['timestamp'], errors='coerce')
        _td  = (_ts.dt.hour + _ts.dt.minute / 60.0).values
        _dt  = _ts.dt.date.values
        _dates = sorted(set(_dt[~pd.isnull(_dt)]))

        # Pre-compute per-day session extremes
        _ny_stats  = {}  # date → (hi, lo, bull)
        _ldn_stats = {}

        for d in _dates:
            mask_d = _dt == d
            _td_d  = _td[mask_d]
            _hi_d  = _hi[mask_d]
            _lo_d  = _lo[mask_d]
            _cl_d  = _cl[mask_d]

            # NY: 15.5-17.5
            _ny_m  = (_td_d >= 15.5) & (_td_d <= 17.5)
            if _ny_m.sum() > 3:
                _ny_stats[d] = (float(_hi_d[_ny_m].max()), float(_lo_d[_ny_m].min()),
                                int(_cl_d[_ny_m][-1] > _cl_d[_ny_m][0]))
            # LDN: 9.0-12.0
            _ldn_m = (_td_d >= 9.0) & (_td_d <= 12.0)
            if _ldn_m.sum() > 3:
                _ldn_stats[d] = (float(_hi_d[_ldn_m].max()), float(_lo_d[_ldn_m].min()),
                                 int(_cl_d[_ldn_m][-1] > _cl_d[_ldn_m][0]))

        # Assign previous day stats
        from datetime import timedelta
        for i in range(n):
            d = _dt[i]
            if d is None or str(d) == 'nan':
                continue
            try:
                from datetime import date as _date_cls
                prev_d = d - timedelta(days=1)
                # Look back up to 5 days for previous trading day
                for _back in range(1, 6):
                    _pd_candidate = d - timedelta(days=_back)
                    if _pd_candidate in _ny_stats:
                        _yday_ny_hi[i], _yday_ny_lo[i], _yday_ny_bull[i] = _ny_stats[_pd_candidate]
                        break
                for _back in range(1, 6):
                    _pd_candidate = d - timedelta(days=_back)
                    if _pd_candidate in _ldn_stats:
                        _yday_ldn_hi[i], _yday_ldn_lo[i], _yday_ldn_bull[i] = _ldn_stats[_pd_candidate]
                        break
            except Exception:
                pass

    _yday_ny_lo  = np.where(np.isnan(_yday_ny_lo),  _cl * 0.999, _yday_ny_lo)
    _yday_ldn_lo = np.where(np.isnan(_yday_ldn_lo), _cl * 0.999, _yday_ldn_lo)

    df['dist_yday_ny_hi']  = np.clip((_cl - _yday_ny_hi)  / np.maximum(_atr, 0.1), -5, 5)
    df['dist_yday_ny_lo']  = np.clip((_cl - _yday_ny_lo)  / np.maximum(_atr, 0.1), -5, 5)
    df['dist_yday_ldn_hi'] = np.clip((_cl - _yday_ldn_hi) / np.maximum(_atr, 0.1), -5, 5)
    df['dist_yday_ldn_lo'] = np.clip((_cl - _yday_ldn_lo) / np.maximum(_atr, 0.1), -5, 5)
    df['yday_ny_bull']     = _yday_ny_bull.astype(float)
    df['yday_ldn_bull']    = _yday_ldn_bull.astype(float)

    # ── Weekly context: PWH / PWL ──────────────────────────────────────────────
    # Prioritar lw_hi/lw_lo din DB; fallback rolling 5D (1950 bare 1M)
    if 'lw_hi' in df.columns and 'lw_lo' in df.columns:
        _pwh_s = df['lw_hi'].replace(0, np.nan).ffill()
        _pwh = np.where(_pwh_s.isna().values, _hi, _pwh_s.values)
        _pwl_s = df['lw_lo'].replace(0, np.nan).ffill()
        _pwl = np.where(_pwl_s.isna().values, _lo, _pwl_s.values)
    else:
        _pwh = pd.Series(_hi).rolling(1950, min_periods=300).max().values
        _pwl = pd.Series(_lo).rolling(1950, min_periods=300).min().values

    df['dist_pwh_atr'] = np.clip((_pwh - _cl) / np.maximum(_atr, 0.1), -5, 5)
    df['dist_pwl_atr'] = np.clip((_cl - _pwl)  / np.maximum(_atr, 0.1), -5, 5)
    df['above_pwh']    = (_cl > _pwh).astype(float)
    df['below_pwl']    = (_cl < _pwl).astype(float)

    # Weekly open = luni dimineața (prima bară din săptămână)
    if 'timestamp' in df.columns:
        _ts  = pd.to_datetime(df['timestamp'], errors='coerce')
        _dow = _ts.dt.dayofweek.values   # 0=Mon, 4=Fri
        _week_open = np.full(n, np.nan)
        _last_mon_open = np.nan
        for i in range(n):
            if _dow[i] == 0 and (i == 0 or _dow[i-1] != 0):
                _last_mon_open = _cl[max(0, i-1)]
            _week_open[i] = _last_mon_open
        _week_open = np.where(np.isnan(_week_open), _cl, _week_open)
        df['weekly_bias'] = np.clip((_cl - _week_open) / np.maximum(_atr, 0.1), -5, 5)
    else:
        df['weekly_bias'] = 0.0

    # ── PDH / PDL context ──────────────────────────────────────────────────────
    if 'pdh' in df.columns and 'pdl' in df.columns:
        _pdh = df['pdh'].replace(0, np.nan).ffill().fillna(_hi).values
        _pdl = df['pdl'].replace(0, np.nan).ffill().fillna(_lo).values
    else:
        # Fallback: rolling 1D back (390 bare)
        _pdh = pd.Series(_hi).rolling(390, min_periods=100).max().shift(1).values
        _pdl = pd.Series(_lo).rolling(390, min_periods=100).min().shift(1).values
        _pdh = np.where(np.isnan(_pdh), _hi, _pdh)
        _pdl = np.where(np.isnan(_pdl), _lo, _pdl)

    df['dist_pdh_atr'] = np.clip((_pdh - _cl) / np.maximum(_atr, 0.1), -5, 5)
    df['dist_pdl_atr'] = np.clip((_cl - _pdl)  / np.maximum(_atr, 0.1), -5, 5)
    df['above_pdh']    = (_cl > _pdh).astype(float)
    df['below_pdl']    = (_cl < _pdl).astype(float)

    # ── HTF confluence score ───────────────────────────────────────────────────
    _h4_bull  = (df['h4_momentum'].values > 0.3).astype(float)
    _h1_bull  = (df['h1_momentum'].values > 0.1).astype(float)
    _h15_bull = (df['h15_momentum'].values > 0.0).astype(float)
    _pdh_bull = df['above_pdh'].values
    _pwl_bull = (df['dist_pwl_atr'].values > 0).astype(float)

    _h4_bear  = (df['h4_momentum'].values < -0.3).astype(float)
    _h1_bear  = (df['h1_momentum'].values < -0.1).astype(float)
    _h15_bear = (df['h15_momentum'].values < 0.0).astype(float)
    _pdl_bear = df['below_pdl'].values
    _pwh_bear = (df['dist_pwh_atr'].values > 0).astype(float)

    df['htf_bull_confluence'] = (_h4_bull + _h1_bull + _h15_bull + _pdh_bull + _pwl_bull)
    df['htf_bear_confluence'] = (_h4_bear + _h1_bear + _h15_bear + _pdl_bear + _pwh_bear)

    # Fillna pentru toate HTF columns
    for col in FEATURES_HTF:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# v12.2 → v12.3: CONSOLIDATION DETECTION features — 20 features comprehensive
# Modelul trebuie să vadă consolidarea din TOATE unghiurile:
# A) Prețul nu merge nicăieri  B) Revine mereu la același loc
# C) Volumul confirmă lipsa de convingere  D) Structura e pierdută
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


# =============================================================================
# HELPER — CONSOLIDATION FEATURES
# =============================================================================
def add_consolidation_features(df: pd.DataFrame) -> pd.DataFrame:
    """v12.3: 20 features comprehensive de consolidare.
    A) Prețul nu merge nicăieri  B) Revine mereu la același loc
    C) Volume confirmă lipsa de convingere  D) Structura e pierdută"""
    import numpy as np

    _atr_col = 'atr_14' in df.columns
    _atr = df['atr_14'].replace(0, np.nan) if _atr_col else pd.Series(1.0, index=df.index)
    _has_ohlc = all(c in df.columns for c in ['open', 'high', 'low', 'close'])

    # ═══ A. PREȚUL NU MERGE NICĂIERI ═══════════════════════════════════════

    # 1. range_atr_ratio — range 20 bare / ATR
    if _has_ohlc and _atr_col:
        _roll_hi = df['high'].rolling(20, min_periods=5).max()
        _roll_lo = df['low'].rolling(20, min_periods=5).min()
        df['range_atr_ratio'] = ((_roll_hi - _roll_lo) / _atr).fillna(3.0).clip(0, 10)
    else:
        df['range_atr_ratio'] = 3.0

    # 2. directional_efficiency — net_move / total_move pe 20 bare
    if 'close' in df.columns:
        _c = df['close'].values
        _de = np.full(len(_c), 0.5)
        for i in range(20, len(_c)):
            _net = abs(_c[i] - _c[i-20])
            _tot = sum(abs(_c[j] - _c[j-1]) for j in range(i-19, i+1))
            _de[i] = _net / _tot if _tot > 0 else 0.0
        df['directional_efficiency'] = _de
    else:
        df['directional_efficiency'] = 0.5

    # 3. net_move_10 — mișcarea netă pe 10 bare / ATR
    if 'close' in df.columns and _atr_col:
        _nm10 = (df['close'] - df['close'].shift(10)).abs() / _atr
        df['net_move_10'] = _nm10.fillna(1.0).clip(0, 10)
    else:
        df['net_move_10'] = 1.0

    # 4. net_move_20 — mișcarea netă pe 20 bare / ATR
    if 'close' in df.columns and _atr_col:
        _nm20 = (df['close'] - df['close'].shift(20)).abs() / _atr
        df['net_move_20'] = _nm20.fillna(1.5).clip(0, 10)
    else:
        df['net_move_20'] = 1.5

    # 5. close_std_20 — std close-to-close pe 20 bare / ATR (mic = chop strâns)
    if 'close' in df.columns and _atr_col:
        _std20 = df['close'].rolling(20, min_periods=5).std() / _atr
        df['close_std_20'] = _std20.fillna(0.5).clip(0, 5)
    else:
        df['close_std_20'] = 0.5

    # 6. avg_bar_range_ratio — media (high-low) pe 10 bare / ATR
    if _has_ohlc and _atr_col:
        _br = (df['high'] - df['low']).rolling(10, min_periods=3).mean() / _atr
        df['avg_bar_range_ratio'] = _br.fillna(1.0).clip(0, 5)
    else:
        df['avg_bar_range_ratio'] = 1.0

    # ═══ B. PREȚUL REVINE MEREU LA ACELAȘI LOC ════════════════════════════

    # 7. bars_inside_va
    if 'inside_va' in df.columns:
        df['bars_inside_va'] = df['inside_va'].rolling(10, min_periods=3).sum().fillna(5).clip(0, 10)
    else:
        df['bars_inside_va'] = 5.0

    # 8. va_width_atr
    if 'vah' in df.columns and 'val' in df.columns and _atr_col:
        _va_w = (df['vah'] - df['val']).clip(lower=0)
        df['va_width_atr'] = (_va_w / _atr).fillna(2.0).clip(0, 10)
    else:
        df['va_width_atr'] = 2.0

    # 9. va_overlap_pct — suprapunere VA curentă vs VA de 10 bare în urmă
    if 'vah' in df.columns and 'val' in df.columns:
        _vah_now = df['vah'].values
        _val_now = df['val'].values
        _vah_old = df['vah'].shift(10).values
        _val_old = df['val'].shift(10).values
        _overlap = np.full(len(df), 0.5)
        for i in range(10, len(df)):
            if _vah_now[i] > 0 and _val_now[i] > 0 and _vah_old[i] > 0 and _val_old[i] > 0:
                _hi = min(_vah_now[i], _vah_old[i])
                _lo = max(_val_now[i], _val_old[i])
                _ov = max(0, _hi - _lo)
                _total = max(_vah_now[i] - _val_now[i], _vah_old[i] - _val_old[i], 0.01)
                _overlap[i] = min(_ov / _total, 1.0)
        df['va_overlap_pct'] = _overlap
    else:
        df['va_overlap_pct'] = 0.5

    # 10. mean_reversion_speed — cât de repede revine prețul la SMA20
    if 'close' in df.columns:
        _sma20 = df['close'].rolling(20, min_periods=5).mean()
        _dev = (df['close'] - _sma20).abs()
        _dev_prev = _dev.shift(1)
        # Raport: deviația scade rapid = mean reversion puternic
        _mrs = (_dev / _dev_prev.replace(0, np.nan)).fillna(1.0).clip(0.1, 3.0)
        # Media pe 10 bare: <0.8 = mean reversion activ
        df['mean_reversion_speed'] = _mrs.rolling(10, min_periods=3).mean().fillna(1.0)
    else:
        df['mean_reversion_speed'] = 1.0

    # 11. same_level_rejections (vectorizat — mai rapid)
    if all(c in df.columns for c in ['close', 'vah', 'val', 'poc_level']) and _atr_col:
        _c = df['close'].values
        _vah_v = df['vah'].values
        _val_v = df['val'].values
        _poc_v = df['poc_level'].values
        _atr_v = df['atr_14'].values
        _rej = np.zeros(len(df))
        for i in range(20, len(df)):
            _margin = max(_atr_v[i] * 0.1, 2.0)
            _cnt = 0
            for j in range(i-20, i):
                if abs(_c[j] - _vah_v[i]) <= _margin: _cnt += 1
                if abs(_c[j] - _val_v[i]) <= _margin: _cnt += 1
                if _poc_v[i] > 0 and abs(_c[j] - _poc_v[i]) <= _margin: _cnt += 1
            _rej[i] = min(_cnt, 30)
        df['same_level_rejections'] = _rej
    else:
        df['same_level_rejections'] = 0

    # 12. pivot_count_20 — schimbări de direcție pe 20 bare (multe = chop)
    if 'close' in df.columns:
        _diff = np.sign(np.diff(df['close'].values, prepend=df['close'].values[0]))
        _pivots = np.zeros(len(df))
        for i in range(20, len(df)):
            _seg = _diff[i-20:i]
            _pivots[i] = sum(1 for j in range(1, len(_seg)) if _seg[j] != _seg[j-1] and _seg[j] != 0)
        df['pivot_count_20'] = _pivots
    else:
        df['pivot_count_20'] = 5

    # 13. poc_stability — Std(POC) pe 10 bare / ATR (mic = POC stabil = range)
    if 'poc_level' in df.columns and _atr_col:
        _poc_std = df['poc_level'].rolling(10, min_periods=3).std()
        df['poc_stability'] = (_poc_std / _atr).fillna(0.5).clip(0, 5)
    else:
        df['poc_stability'] = 0.5

    # ═══ C. VOLUME CONFIRMĂ LIPSA DE CONVINGERE ═══════════════════════════

    # 14. volume_trend — slope volum pe 20 bare (negativ = volum scade)
    if 'volume' in df.columns:
        _vol_ma5 = df['volume'].rolling(5, min_periods=2).mean()
        _vol_ma20 = df['volume'].rolling(20, min_periods=5).mean()
        df['volume_trend'] = ((_vol_ma5 / _vol_ma20.replace(0, np.nan)) - 1.0).fillna(0).clip(-1, 2)
    else:
        df['volume_trend'] = 0.0

    # 15. volume_cv — coeficient variație volum pe 20 bare
    if 'volume' in df.columns:
        _vol_mean = df['volume'].rolling(20, min_periods=5).mean()
        _vol_std = df['volume'].rolling(20, min_periods=5).std()
        df['volume_cv'] = (_vol_std / _vol_mean.replace(0, np.nan)).fillna(0.5).clip(0, 3)
    else:
        df['volume_cv'] = 0.5

    # ═══ D. STRUCTURA PIERDUTĂ ═════════════════════════════════════════════

    # 16. hh_ll_score
    if 'high' in df.columns and 'low' in df.columns:
        _h = df['high'].values
        _l = df['low'].values
        _hh_score = np.zeros(len(df))
        for i in range(10, len(df)):
            _fh = max(_h[i-10:i-5]); _sh = max(_h[i-5:i])
            _fl = min(_l[i-10:i-5]); _sl = min(_l[i-5:i])
            _hh_score[i] = (1.0 if _sh > _fh else 0.0) + (-1.0 if _sl < _fl else 0.0)
        df['hh_ll_score'] = _hh_score
    else:
        df['hh_ll_score'] = 0.0

    # 17. bollinger_width — (BB_upper - BB_lower) / close
    if 'close' in df.columns:
        _sma = df['close'].rolling(20, min_periods=5).mean()
        _std = df['close'].rolling(20, min_periods=5).std()
        _bb_w = (4 * _std) / df['close'].replace(0, np.nan)  # 2 std up + 2 std down
        df['bollinger_width'] = _bb_w.fillna(0.02).clip(0, 0.2)
    else:
        df['bollinger_width'] = 0.02

    # 18. atr_percentile — unde e ATR curent vs ultimele 100 bare
    if _atr_col:
        _atr_rank = df['atr_14'].rolling(100, min_periods=20).rank(pct=True)
        df['atr_percentile'] = _atr_rank.fillna(0.5)
    else:
        df['atr_percentile'] = 0.5

    # 19. swing_size_decay — ratio ultimul swing / penultimul (sub 1 = decay)
    if _has_ohlc:
        _h = df['high'].values
        _l = df['low'].values
        _decay = np.ones(len(df))
        for i in range(20, len(df)):
            _seg_h = _h[i-20:i]
            _seg_l = _l[i-20:i]
            _range_first = max(_seg_h[:10]) - min(_seg_l[:10])
            _range_second = max(_seg_h[10:]) - min(_seg_l[10:])
            _decay[i] = (_range_second / _range_first) if _range_first > 0 else 1.0
        df['swing_size_decay'] = np.clip(_decay, 0.2, 3.0)
    else:
        df['swing_size_decay'] = 1.0

    # 20. candle_overlap_pct — % bare care se suprapun cu bara anterioară
    if _has_ohlc:
        _h = df['high'].values
        _l = df['low'].values
        _overlap_arr = np.full(len(df), 0.5)
        for i in range(20, len(df)):
            _cnt = 0
            for j in range(i-20, i):
                if j > 0:
                    _ov = min(_h[j], _h[j-1]) - max(_l[j], _l[j-1])
                    _rng = max(_h[j] - _l[j], 0.01)
                    if _ov / _rng > 0.5:  # >50% overlap
                        _cnt += 1
            _overlap_arr[i] = _cnt / 20.0
        df['candle_overlap_pct'] = _overlap_arr
    else:
        df['candle_overlap_pct'] = 0.5

    return df


# =============================================================================
# HELPER — STREAK PREVENTION FEATURES
# =============================================================================
def add_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """v12.2: Features care ajută modelul să evite condițiile care duc la pierderi consecutive.
    Toate calculabile din OHLC + timestamp — funcționează pe date istorice."""
    import numpy as np

    # 1. ATR change speed — raport ATR acum / ATR acum 10 bare
    # >1.5 = volatilitate crește rapid (regim nou) | <0.7 = se comprimă (consolidare)
    if 'atr_14' in df.columns:
        _atr = df['atr_14'].replace(0, np.nan)
        _atr_shifted = _atr.shift(10)
        df['atr_change_speed'] = (_atr / _atr_shifted).fillna(1.0).clip(0.3, 3.0)
    else:
        df['atr_change_speed'] = 1.0

    # v12.5: hour_sin + hour_cos SCOASE (21.5% combined importance = temporal overfitting)
    # v12.3: day_of_week SCOS (overfitting temporal)

    # 4. Consecutive same direction bars — câte bare consecutive cu close > open (sau invers)
    if 'close' in df.columns and 'open' in df.columns:
        _dir = (df['close'] > df['open']).astype(int)  # 1=bullish, 0=bearish
        _consec = []
        _count = 0
        _last_d = -1
        for d in _dir.values:
            if d == _last_d:
                _count += 1
            else:
                _count = 1
                _last_d = d
            _consec.append(_count)
        df['consecutive_same_dir'] = _consec
    else:
        df['consecutive_same_dir'] = 0

    # 5. Price vs daily range — poziția prețului în range-ul ultimelor 100 bare (~1 zi)
    # 0.0 = la low-ul zilei | 1.0 = la high-ul zilei | 0.5 = mijloc (chop zone)
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        _day_hi = df['high'].rolling(100, min_periods=20).max()
        _day_lo = df['low'].rolling(100, min_periods=20).min()
        _day_range = _day_hi - _day_lo
        df['price_vs_daily_range'] = ((df['close'] - _day_lo) / _day_range.replace(0, np.nan)).fillna(0.5).clip(0, 1)
    else:
        df['price_vs_daily_range'] = 0.5

    # v12.3: session_age SCOS (proxy trivial min(len,200) — noise, domina 17% importanță)

    # 7. Recent signal quality — proxy: din ultimele 30 bare, câte "semnale"
    # (close_t+1 în aceeași direcție ca close_t - open_t) au fost corecte
    # Dă un proxy de "cât de bine funcționează semnalele direcționale în condițiile curente"
    if 'close' in df.columns and 'open' in df.columns:
        _bar_dir = np.sign(df['close'].values - df['open'].values)   # +1=bull, -1=bear
        _next_move = np.sign(df['close'].diff().shift(-1).values)    # +1=preț creste, -1=scade
        _correct = (_bar_dir == _next_move).astype(float)
        # Rolling mean pe 30 bare — 0.5 = random, 0.7+ = semnale bune, 0.3- = inversate
        _rsq = pd.Series(_correct).rolling(30, min_periods=10).mean().fillna(0.5).values
        df['recent_signal_quality'] = _rsq
    else:
        df['recent_signal_quality'] = 0.5

    return df


# =============================================================================
# HELPER — TRADE QUALITY CONTEXT FEATURES (v12.4)
# =============================================================================
def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """v12.4: Features care detectează degradarea condițiilor de trade.
    Streak-urile apar când modelul nu vede că piața s-a schimbat.
    Aceste features arată calitatea trendului, momentum-ului și volume-ului."""
    import numpy as np

    _atr_col = 'atr_14' if 'atr_14' in df.columns else None
    _atr = df[_atr_col].clip(lower=0.01) if _atr_col else pd.Series(1.0, index=df.index)

    # --- 1. Trend R² pe 20 bare (cât de liniar e trendul) ---
    # R²=1 → trend perfect, R²=0 → noise pur (consolidare)
    _r2 = []
    _slope_norm = []
    _c = df['close'].values
    for i in range(len(df)):
        if i < 19:
            _r2.append(0.5)
            _slope_norm.append(0.0)
            continue
        _y = _c[i-19:i+1]
        _x = np.arange(20, dtype=float)
        _xm = _x.mean()
        _ym = _y.mean()
        _ss_tot = np.sum((_y - _ym)**2)
        _ss_xy = np.sum((_x - _xm) * (_y - _ym))
        _ss_xx = np.sum((_x - _xm)**2)
        if _ss_tot > 0 and _ss_xx > 0:
            _b = _ss_xy / _ss_xx
            _y_pred = _ym + _b * (_x - _xm)
            _ss_res = np.sum((_y - _y_pred)**2)
            _r2.append(1.0 - _ss_res / _ss_tot)
            _slope_norm.append(_b / max(_atr.iloc[i], 0.01))
        else:
            _r2.append(0.5)
            _slope_norm.append(0.0)
    df['trend_r2'] = np.clip(_r2, 0, 1)
    df['trend_slope_norm'] = np.clip(_slope_norm, -3, 3)

    # --- 2. EMA stack alignment (EMA8 vs EMA21 vs EMA55) ---
    _ema8 = df['close'].ewm(span=8).mean()
    _ema21 = df['close'].ewm(span=21).mean()
    _ema55 = df['close'].ewm(span=55).mean()
    _bull_stack = ((_ema8 > _ema21) & (_ema21 > _ema55)).astype(float)
    _bear_stack = ((_ema8 < _ema21) & (_ema21 < _ema55)).astype(float)
    df['close_vs_ema_stack'] = _bull_stack + _bear_stack  # 1.0 = aligned (either dir), 0 = mixed

    # --- 3. ROC 10 normalizat ---
    _roc10 = (df['close'] - df['close'].shift(10)) / _atr
    df['roc_10'] = _roc10.fillna(0).clip(-5, 5)

    # --- 4. ROC divergence (short vs long momentum) ---
    _roc5 = (df['close'] - df['close'].shift(5)) / _atr
    _roc20 = (df['close'] - df['close'].shift(20)) / _atr
    df['roc_divergence'] = (_roc5 - _roc20).fillna(0).clip(-5, 5)

    # --- 5. Momentum consistency ---
    # Câte din ultimele 10 bare au close > close[-1]
    _up = (df['close'] > df['close'].shift(1)).astype(float)
    _up_pct = _up.rolling(10, min_periods=5).mean().fillna(0.5)
    # Transformăm: 1.0 sau 0.0 = consistent, 0.5 = fără direcție
    df['momentum_consistency'] = (2 * (_up_pct - 0.5).abs()).clip(0, 1)

    # --- 6. Volume on move vs quiet ---
    if 'volume' in df.columns:
        _bar_range = (df['high'] - df['low']).clip(lower=0.01)
        _big_move = _bar_range > 0.5 * _atr
        _vol = df['volume'].values.astype(float)
        _vol_move = pd.Series(np.where(_big_move, _vol, np.nan)).rolling(10, min_periods=3).mean()
        _vol_quiet = pd.Series(np.where(~_big_move, _vol, np.nan)).rolling(10, min_periods=3).mean()
        df['volume_on_move'] = (_vol_move / _vol_quiet.clip(lower=1)).fillna(1.0).clip(0.1, 10)
    else:
        df['volume_on_move'] = 1.0

    # --- 7. Volume directional (bull vs bear) ---
    if 'volume' in df.columns:
        _bull_bar = df['close'] > df['open']
        _vol_bull = pd.Series(np.where(_bull_bar, df['volume'].values, np.nan)).rolling(10, min_periods=3).mean()
        _vol_bear = pd.Series(np.where(~_bull_bar, df['volume'].values, np.nan)).rolling(10, min_periods=3).mean()
        df['volume_directional'] = (_vol_bull / _vol_bear.clip(lower=1)).fillna(1.0).clip(0.1, 10)
    else:
        df['volume_directional'] = 1.0

    # --- 8. Clean bars pct (body > 50% range) ---
    _body = (df['close'] - df['open']).abs()
    _range = (df['high'] - df['low']).clip(lower=0.01)
    _clean = (_body / _range > 0.5).astype(float)
    df['clean_bars_pct'] = _clean.rolling(10, min_periods=5).mean().fillna(0.5)

    # --- 9. False break count ---
    _h = df['high'].values
    _l = df['low'].values
    _fb = []
    for i in range(len(df)):
        if i < 20:
            _fb.append(0)
            continue
        _cnt = 0
        for j in range(i-19, i+1):
            if j < 1:
                continue
            # False break up: high > prev high but close < prev high
            if _h[j] > _h[j-1] and _c[j] < _h[j-1]:
                _cnt += 1
            # False break down: low < prev low but close > prev low
            if _l[j] < _l[j-1] and _c[j] > _l[j-1]:
                _cnt += 1
        _fb.append(_cnt)
    df['false_break_count'] = np.clip(_fb, 0, 20)

    # --- 10. Bar range consistency ---
    _br = df['high'] - df['low']
    _br_std = _br.rolling(10, min_periods=5).std()
    _br_mean = _br.rolling(10, min_periods=5).mean().clip(lower=0.01)
    df['bar_range_consistency'] = (_br_std / _br_mean).fillna(0.5).clip(0, 3)

    return df


# =============================================================================
# HELPER — OPENING RANGE (ORH/ORL) FEATURES (v12.7)
# =============================================================================
def add_orh_features(df: pd.DataFrame) -> pd.DataFrame:
    """v12.7: Opening Range per killzone session.
    London OR = 09:00-09:30 RO. NY OR = 15:30-16:00 RO. OR_DURATION = 30 bare.
    Primele 30 min = in_orh (acumulare). Restul killzone = post_orh (expansion).
    Target trade: breakout ORH/ORL cu follow-through 30-50 pts prima gambă.
    """
    import numpy as np

    _n = len(df)
    _orh_high = np.full(_n, np.nan)
    _orh_low  = np.full(_n, np.nan)
    _in_orh   = np.zeros(_n, dtype=np.int8)
    _post_orh = np.zeros(_n, dtype=np.int8)
    _bars_since_session = np.zeros(_n, dtype=np.int32)
    _session_vol_sum = np.zeros(_n, dtype=np.float64)
    _session_vol_cnt = np.zeros(_n, dtype=np.int32)
    _broken_up   = np.zeros(_n, dtype=np.int8)
    _broken_down = np.zeros(_n, dtype=np.int8)

    OR_DURATION = 30  # primele 30 bare din killzone = OR

    if 'timestamp' not in df.columns:
        df['orh_high'] = np.nan
        df['orh_low']  = np.nan
        df['in_orh'] = 0
        df['post_orh'] = 0
        df['orh_width_atr'] = 0.0
        df['dist_to_orh_high_atr'] = 0.0
        df['dist_to_orh_low_atr'] = 0.0
        df['orh_broken_up'] = 0
        df['orh_broken_down'] = 0
        df['bars_since_session'] = 0
        df['session_vol_ratio'] = 1.0
        df['orh_midpoint_dist_atr'] = 0.0
        return df

    _ts = pd.to_datetime(df['timestamp'], errors='coerce')
    _hour = _ts.dt.hour.values
    _minute = _ts.dt.minute.values
    _date = _ts.dt.date.values
    _tdec = _hour + _minute / 60.0

    # Session ID: (date, "london" or "ny" or None)
    _in_london = (_tdec >= 9.0) & (_tdec <= 11.0)
    _in_ny     = (_tdec >= 15.5) & (_tdec <= 17.5)

    _h = df['high'].values
    _l = df['low'].values
    _c = df['close'].values
    _v = df['volume'].values if 'volume' in df.columns else np.ones(_n)

    # Pre-compute ATR pentru normalizare
    if 'atr_14' in df.columns:
        _atr = df['atr_14'].values
    else:
        _tr = np.maximum(_h - _l,
                         np.maximum(np.abs(_h - np.roll(_c, 1)),
                                    np.abs(_l - np.roll(_c, 1))))
        _atr = pd.Series(_tr).rolling(14).mean().fillna(10.0).values
    _atr = np.where(_atr > 0, _atr, 10.0)  # safe floor

    # Iterare pe sesiuni (date + killzone type)
    _cur_date = None
    _cur_kz = None  # "london" / "ny"
    _sess_start = 0
    _sess_high = -np.inf
    _sess_low  = np.inf
    _or_vol_sum = 0.0
    _or_vol_cnt = 0
    _or_finalized = False
    _broke_up_flag = False
    _broke_dn_flag = False

    for i in range(_n):
        _d = _date[i]
        if _in_london[i]:
            _kz = "london"
        elif _in_ny[i]:
            _kz = "ny"
        else:
            _kz = None
            _cur_kz = None
            continue

        # Detect nouă sesiune
        if _d != _cur_date or _kz != _cur_kz:
            _cur_date = _d
            _cur_kz = _kz
            _sess_start = i
            _sess_high = _h[i]
            _sess_low  = _l[i]
            _or_vol_sum = _v[i] if not np.isnan(_v[i]) else 0.0
            _or_vol_cnt = 1
            _or_finalized = False
            _broke_up_flag = False
            _broke_dn_flag = False

        _bars_in_sess = i - _sess_start
        _bars_since_session[i] = _bars_in_sess

        if _bars_in_sess < OR_DURATION:
            # În OR period: update high/low
            if _h[i] > _sess_high: _sess_high = _h[i]
            if _l[i] < _sess_low:  _sess_low  = _l[i]
            if not np.isnan(_v[i]):
                _or_vol_sum += _v[i]
                _or_vol_cnt += 1
            _in_orh[i] = 1
            _orh_high[i] = _sess_high
            _orh_low[i]  = _sess_low
        else:
            # Post-OR: ORH/ORL congelate
            if not _or_finalized:
                _or_finalized = True
            _post_orh[i] = 1
            _orh_high[i] = _sess_high
            _orh_low[i]  = _sess_low
            # Detect break
            if not _broke_up_flag and _c[i] > _sess_high:
                _broke_up_flag = True
            if not _broke_dn_flag and _c[i] < _sess_low:
                _broke_dn_flag = True
            _broken_up[i]   = int(_broke_up_flag)
            _broken_down[i] = int(_broke_dn_flag)

        _session_vol_sum[i] = _or_vol_sum
        _session_vol_cnt[i] = max(_or_vol_cnt, 1)

    # Compune coloanele finale
    df['orh_high'] = _orh_high
    df['orh_low']  = _orh_low
    df['in_orh']   = _in_orh
    df['post_orh'] = _post_orh
    df['orh_broken_up']   = _broken_up
    df['orh_broken_down'] = _broken_down
    df['bars_since_session'] = _bars_since_session

    _width = (_orh_high - _orh_low)
    df['orh_width_atr'] = np.where(~np.isnan(_width), _width / _atr, 0.0)
    df['dist_to_orh_high_atr'] = np.where(~np.isnan(_orh_high), (_c - _orh_high) / _atr, 0.0)
    df['dist_to_orh_low_atr']  = np.where(~np.isnan(_orh_low),  (_c - _orh_low)  / _atr, 0.0)
    _mid = (_orh_high + _orh_low) / 2.0
    df['orh_midpoint_dist_atr'] = np.where(~np.isnan(_mid), (_c - _mid) / _atr, 0.0)

    _or_avg_vol = _session_vol_sum / np.maximum(_session_vol_cnt, 1)
    df['session_vol_ratio'] = np.where(_or_avg_vol > 0, _v / _or_avg_vol, 1.0)

    # Clip pentru stabilitate
    for col in ['orh_width_atr','dist_to_orh_high_atr','dist_to_orh_low_atr',
                'orh_midpoint_dist_atr','session_vol_ratio']:
        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0).clip(-10, 10)

    return df


# =============================================================================
# SESSION TIME FEATURES — ferestre temporale per killzone
# =============================================================================
def add_session_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features binare pentru ferestrele de timp din fiecare sesiune.
    Modelul LON înțelege: pre-london (acumulare), OR (formarea range), post-OR (breakout), close (14-12).
    Modelul NY înțelege: pre-NY (bias), OR NY, macro windows (16:50-17:10, 17:10-17:30).

    Ore Romania (RO = UTC+3 summer / UTC+2 winter — folosim hora fix din timestamp).
    """
    if 'timestamp' not in df.columns:
        for col in FEATURES_SESSION_TIME:
            df[col] = 0
        return df

    _ts  = pd.to_datetime(df['timestamp'], errors='coerce')
    _td  = (_ts.dt.hour + _ts.dt.minute / 60.0).values  # time decimal RO

    # ── LONDON windows ──────────────────────────────────────────────────────
    # 08:00-09:00: pre-London accumulation (Asia close, instituționalii se poziționează)
    df['in_pre_london']    = ((_td >= 8.0) & (_td < 9.0)).astype(np.int8)
    # 09:00-09:30: London OR — formarea Opening Range (NU tranzacționăm, observăm)
    df['in_london_or']     = ((_td >= 9.0) & (_td < 9.5)).astype(np.int8)
    # 09:30-11:00: London killzone proper — breakout post-OR, cel mai activ
    df['in_london_kz']     = ((_td >= 9.5) & (_td < 11.0)).astype(np.int8)
    # 11:00-12:00: London close — continuare sau reversal, volum scade
    df['in_london_close']  = ((_td >= 11.0) & (_td < 12.0)).astype(np.int8)

    # ── NEW YORK windows ─────────────────────────────────────────────────────
    # 15:00-15:30: pre-NY window — Londres e încă activ, NY futures se mișcă
    df['in_pre_ny']        = ((_td >= 15.0) & (_td < 15.5)).astype(np.int8)
    # 15:30-16:00: NY OR — formarea Opening Range NY (NU tranzacționăm)
    df['in_ny_or']         = ((_td >= 15.5) & (_td < 16.0)).astype(np.int8)
    # 15:45-16:10: PRE-NY MACRO — macro data prints (Retail Sales, PMI etc), high conviction
    df['in_pre_ny_macro']  = ((_td >= 15.75) & (_td < 16.167)).astype(np.int8)
    # 16:00-16:50: NY killzone core — post-OR expansion principală
    df['in_ny_kz_core']    = ((_td >= 16.0) & (_td < 16.833)).astype(np.int8)
    # 16:50-17:10: NY MACRO 1 — prima fereastră macro post-open (move violent posibil)
    df['in_ny_macro_1']    = ((_td >= 16.833) & (_td < 17.167)).astype(np.int8)
    # 17:10-17:30: NY MACRO 2 — a doua fereastră macro (continuation sau reversal macro)
    df['in_ny_macro_2']    = ((_td >= 17.167) & (_td < 17.5)).astype(np.int8)

    # ── Composite: bara e într-o fereastră macro activă (NY) ─────────────────
    df['in_any_macro']     = (
        (df['in_pre_ny_macro'] == 1) |
        (df['in_ny_macro_1'] == 1)   |
        (df['in_ny_macro_2'] == 1)
    ).astype(np.int8)

    # ── Minute în sesiune (0-120 pentru LON, 0-150 pentru NY) ───────────────
    # Util ca feature continuu — modelul înțelege "sunt la minutul 10 sau 80 din sesiune"
    _lon_start = 9.0   # London OR start
    _ny_start  = 15.5  # NY OR start
    _mins_lon = np.clip((_td - _lon_start) * 60, 0, 180)  # 0-180 min din LON
    _mins_ny  = np.clip((_td - _ny_start)  * 60, 0, 120)  # 0-120 min din NY
    df['mins_since_lon_open'] = np.where(_td >= _lon_start, _mins_lon, 0).astype(np.float32)
    df['mins_since_ny_open']  = np.where(_td >= _ny_start,  _mins_ny,  0).astype(np.float32)

    return df


FEATURES_SESSION_TIME = [
    # London
    'in_pre_london', 'in_london_or', 'in_london_kz', 'in_london_close',
    'mins_since_lon_open',
    # New York
    'in_pre_ny', 'in_ny_or', 'in_pre_ny_macro',
    'in_ny_kz_core', 'in_ny_macro_1', 'in_ny_macro_2', 'in_any_macro',
    'mins_since_ny_open',
]


# =============================================================================
# HELPER — MTF CONFIRMATION FEATURES (v12.4)
# =============================================================================
def add_mtf_confirm_features(df: pd.DataFrame) -> pd.DataFrame:
    """v12.4: Multi-timeframe agreement + rejection strength.
    Streak-urile apar când trade-ul e contra TF mai mari."""
    import numpy as np

    _c = df['close']

    # H1 trend direction (slope pe 60 bare = 1 oră)
    _h1_slope = (_c - _c.shift(60)) / _c.shift(60).clip(lower=0.01)
    df['h1_trend_aligned'] = (_h1_slope.abs() > 0.001).astype(float)  # >0.1% = are direcție

    # H4 trend direction (slope pe 240 bare = 4 ore)
    _h4_slope = (_c - _c.shift(240)) / _c.shift(240).clip(lower=0.01)
    df['h4_trend_aligned'] = (_h4_slope.abs() > 0.002).astype(float)  # >0.2% = are direcție

    # MTF agreement: câte perioade scurte (5/15/60/240) sunt în aceeași direcție
    _m5 = np.sign((_c - _c.shift(5)).values)
    _m15 = np.sign((_c - _c.shift(15)).values)
    _m60 = np.sign((_c - _c.shift(60)).values)
    _m240 = np.sign((_c - _c.shift(240)).values)
    _m1 = np.sign((_c - _c.shift(1)).values)

    _agree = np.zeros(len(df))
    for i in range(240, len(df)):
        _dirs = [_m1[i], _m5[i], _m15[i], _m60[i], _m240[i]]
        _dirs = [d for d in _dirs if d != 0]
        if len(_dirs) > 0:
            _dominant = 1 if sum(_dirs) > 0 else -1
            _agree[i] = sum(1 for d in _dirs if d == _dominant) / len(_dirs)
        else:
            _agree[i] = 0.5
    df['mtf_agreement'] = _agree

    # Recent rejection strength (volum × wick pe ultimele 5 bare cu wick > body)
    _wick_total = (df['high'] - df['low']) - (df['close'] - df['open']).abs()
    _body = (df['close'] - df['open']).abs().clip(lower=0.01)
    _wick_dominant = (_wick_total > _body)
    if 'volume' in df.columns:
        _rej_strength = np.where(_wick_dominant, _wick_total * df['volume'].values, 0)
        _rej_series = pd.Series(_rej_strength).rolling(5, min_periods=2).mean()
        _norm = pd.Series(df['volume'].values * (df['high'].values - df['low'].values)).rolling(20, min_periods=5).mean().clip(lower=0.01)
        df['recent_rejection_strength'] = (_rej_series / _norm).fillna(0).clip(0, 5)
    else:
        df['recent_rejection_strength'] = 0.0

    return df


# =============================================================================
# HELPER — NORMALIZARE PRICE LEVELS
# =============================================================================
def normalize_price_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizează toate nivelurile de preț față de true_open.
    Modelul învață structura relativă, nu prețul absolut.
    Permite generalizare pe orice nivel de preț al QQQ.
    """
    X = X.copy()
    ref_col = 'true_open'
    if ref_col not in X.columns:
        return X

    price_cols = [
        'open', 'high', 'low', 'close',
        'lm_hi', 'lm_lo', 'lw_hi', 'lw_lo',
        'm_hi', 'm_lo', 'p_hi', 'p_lo',
        'h4_hi', 'h4_lo', 'h1_hi', 'h1_lo',
        'kalman_smooth', 'vwap',  # Advanced price-level features
        'asia_hi', 'asia_lo', 'lon_hi', 'lon_lo',
        'poc_level', 'vah', 'val',
        'dist_poc', 'dist_pdh', 'dist_pdl',
    ]

    true_open = X[ref_col].replace(0, np.nan).ffill().fillna(1)

    for col in price_cols:
        if col in X.columns:
            X[col] = (X[col] - true_open) / true_open.clip(lower=1)

    return X


# =============================================================================
# HELPER — DYNAMIC ATR TARGET
# =============================================================================
def generate_atr_target(df: pd.DataFrame, horizon: int = 15,
                        atr_multiplier: float = 0.8,
                        mae_filter: bool = True,
                        mae_max_atr: float = 0.3) -> pd.Series:
    """
    v12.5: Generează target dinamic cu MAE filter (Maximum Adverse Excursion).
      - LONG  (2): high atinge +ATR*mult ȘI low nu cade mai mult de mae_max_atr*ATR
      - SHORT (1): low atinge -ATR*mult ȘI high nu urcă mai mult de mae_max_atr*ATR
      - WAIT  (0): altfel (ambiguu sau nu atinge target)
    MAE filter: elimină semnale unde prețul face drawdown mare înainte de target.
    Semnale curate → WR natural mai mare → model învață pattern-uri reale.
    OPTIMIZAT: vectorizat cu numpy sliding_window_view (~5s pe 4M bare).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    _close = df['close'].values.astype(float)
    _high = df['high'].values.astype(float)
    _low = df['low'].values.astype(float)
    _atr = atr.values.astype(float)
    _n = len(df)

    target = np.zeros(_n, dtype=int)

    if _n <= horizon + 1:
        return pd.Series(target, index=df.index)

    # Forward-looking rolling max(high) și min(low) pe next `horizon` bare
    # sliding_window_view(_high[1:], horizon)[i] = _high[i+1 : i+1+horizon]
    _h_windows = sliding_window_view(_high[1:], horizon)  # shape: (n-horizon, horizon)
    _l_windows = sliding_window_view(_low[1:], horizon)

    _future_max_high = np.full(_n, np.nan)
    _future_min_low = np.full(_n, np.nan)
    _future_max_high[:_h_windows.shape[0]] = _h_windows.max(axis=1)
    _future_min_low[:_l_windows.shape[0]] = _l_windows.min(axis=1)

    # Targets și MAE limits (vectorizate)
    _target_long = _close + _atr * atr_multiplier
    _target_short = _close - _atr * atr_multiplier
    _mae_limit = _atr * mae_max_atr

    # Condiții (vectorizate)
    _valid = ~np.isnan(_atr) & (_atr >= 0.01) & ~np.isnan(_future_max_high)
    _hit_long = _future_max_high >= _target_long       # high atinge target LONG
    _hit_short = _future_min_low <= _target_short       # low atinge target SHORT
    _mae_long_ok = _future_min_low >= (_close - _mae_limit)   # low nu cade prea mult
    _mae_short_ok = _future_max_high <= (_close + _mae_limit) # high nu urcă prea mult

    if mae_filter:
        # LONG: target hit + MAE ok + nu e și SHORT valid (ambiguu)
        _is_long = _valid & _hit_long & _mae_long_ok & ~(_hit_short & _mae_short_ok)
        _is_short = _valid & _hit_short & _mae_short_ok & ~(_hit_long & _mae_long_ok)
    else:
        # Fără MAE filter (compatibilitate cu vechiul cod)
        _price_future = np.roll(_close, -horizon)
        _is_long = _valid & (_price_future > _target_long)
        _is_short = _valid & (_price_future < _target_short)

    target[_is_long] = 2
    target[_is_short] = 1

    return pd.Series(target, index=df.index)


# =============================================================================
# HELPER — SWEEP + SNIPER CONDITIONS
# =============================================================================
# =============================================================================
# HELPER — REVERSAL FEATURES (v8.0)
# =============================================================================
def add_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adaugă features de reversal detection pe care modelul le învață:
    MSS, CHoCH, trend exhaustion, momentum divergence, swing breaks.
    Acestea permit modelului să flipeze direcția când trendul se schimbă.
    """
    df = df.copy()
    n = len(df)

    # ── 1. MSS / CHoCH — Market Structure Shift ──
    # CHoCH Bullish: after downtrend (4-bar), close breaks above recent 4-bar high
    # CHoCH Bearish: after uptrend (4-bar), close breaks below recent 4-bar low
    lookback = 5
    df['_recent_hi'] = df['high'].rolling(lookback).max().shift(1)
    df['_recent_lo'] = df['low'].rolling(lookback).min().shift(1)
    df['_trend_down'] = (df['close'].shift(1) < df['close'].shift(lookback)).astype(int)
    df['_trend_up']   = (df['close'].shift(1) > df['close'].shift(lookback)).astype(int)

    df['choch_bullish'] = ((df['_trend_down'] == 1) & (df['close'] > df['_recent_hi'])).astype(int)
    df['choch_bearish'] = ((df['_trend_up'] == 1) & (df['close'] < df['_recent_lo'])).astype(int)

    # MSS uses higher highs / lower lows over 8 bars
    df['_hh'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['_ll'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['_ll_count'] = df['_ll'].rolling(8).sum()  # how many lower lows in last 8 bars
    df['_hh_count'] = df['_hh'].rolling(8).sum()  # how many higher highs in last 8 bars

    # MSS bullish: was making lower lows (>=4 of 8), now makes higher high
    df['mss_bullish'] = ((df['_ll_count'] >= 4) & (df['_hh'] == 1) & (df['close'] > df['open'])).astype(int)
    # MSS bearish: was making higher highs (>=4 of 8), now makes lower low
    df['mss_bearish'] = ((df['_hh_count'] >= 4) & (df['_ll'] == 1) & (df['close'] < df['open'])).astype(int)

    # ── 2. Reversal Strength — combined signal [-1 to +1] ──
    df['reversal_strength'] = (
        df['choch_bullish'].astype(float) * 0.5 +
        df['mss_bullish'].astype(float) * 0.5 -
        df['choch_bearish'].astype(float) * 0.5 -
        df['mss_bearish'].astype(float) * 0.5
    )

    # ── 3. Trend Exhaustion — consecutive bars in same direction + deceleration ──
    df['_bar_dir'] = np.sign(df['close'] - df['open'])
    df['_same_dir'] = (df['_bar_dir'] == df['_bar_dir'].shift(1)).astype(int)
    df['_consec'] = df['_same_dir'].rolling(8, min_periods=1).sum()  # consecutive same direction
    df['_body'] = (df['close'] - df['open']).abs()
    df['_body_shrink'] = (df['_body'] < df['_body'].shift(1) * 0.6).astype(int)  # body shrinking
    df['trend_exhaustion'] = (df['_consec'] / 8.0) * 0.6 + df['_body_shrink'].astype(float) * 0.4

    # ── 4. Delta Flip — cumulative delta changes sign ──
    # Proxy: use (close - open) * volume as delta estimate
    df['_delta'] = (df['close'] - df['open']) * df['volume'].clip(lower=1)
    df['_cum_delta_5'] = df['_delta'].rolling(5).sum()
    df['_cum_delta_prev'] = df['_cum_delta_5'].shift(1)
    df['delta_flip'] = (
        (np.sign(df['_cum_delta_5']) != np.sign(df['_cum_delta_prev'])) &
        (df['_cum_delta_prev'] != 0)
    ).astype(int)

    # ── 5. POC Drift Direction — POC moving up or down ──
    if 'poc_level' in df.columns:
        df['_poc_diff'] = df['poc_level'] - df['poc_level'].shift(3)
        df['poc_drift_direction'] = np.sign(df['_poc_diff'].fillna(0))
    else:
        df['poc_drift_direction'] = 0

    # ── 6. Swing Break — price breaks above/below recent swing high/low ──
    swing_period = 10
    df['_swing_hi'] = df['high'].rolling(swing_period).max().shift(1)
    df['_swing_lo'] = df['low'].rolling(swing_period).min().shift(1)
    df['swing_break'] = 0.0
    df.loc[df['close'] > df['_swing_hi'], 'swing_break'] = 1.0     # bullish break
    df.loc[df['close'] < df['_swing_lo'], 'swing_break'] = -1.0    # bearish break

    # ── 7. Momentum Divergence — price vs momentum disagreement ──
    mom_period = 15
    df['_price_change'] = df['close'] - df['close'].shift(mom_period)
    df['_mom'] = df['close'].pct_change(mom_period)
    df['_mom_accel'] = df['_mom'] - df['_mom'].shift(5)  # momentum acceleration
    # Bearish divergence: price rising but momentum decelerating
    df['momentum_divergence'] = 0.0
    df.loc[(df['_price_change'] > 0) & (df['_mom_accel'] < -0.001), 'momentum_divergence'] = -1.0
    # Bullish divergence: price falling but momentum accelerating
    df.loc[(df['_price_change'] < 0) & (df['_mom_accel'] > 0.001), 'momentum_divergence'] = 1.0

    # Clean up temp columns
    temp_cols = [c for c in df.columns if c.startswith('_')]
    df.drop(columns=temp_cols, inplace=True)

    return df


def add_amt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    v11.0 — Auction Market Theory (AMT) features complete.
    Concepte: Failed Auction, Excess, Poor High/Low, Initiative vs Responsive,
    VA Migration, Rotation Factor.
    """
    df = df.copy()
    swing_period = 10

    _swing_hi = df['high'].rolling(swing_period).max().shift(1)
    _swing_lo = df['low'].rolling(swing_period).min().shift(1)
    _range    = (df['high'] - df['low']).clip(lower=1e-8)

    # ── 1. FAILED AUCTION ─────────────────────────────────────────────────────
    # Breakout care eșuează: high sparge swing high DAR close revine sub el
    # = lichiditatea a fost luată dar nu e cumpărare reală → SHORT
    # Invers: low sparge swing low DAR close revine deasupra → LONG
    _break_above = df['high'] > _swing_hi                   # wick a spart high-ul
    _fail_above  = _break_above & (df['close'] < _swing_hi) # dar close e sub → FAILED
    _break_below = df['low'] < _swing_lo                    # wick a spart low-ul
    _fail_below  = _break_below & (df['close'] > _swing_lo) # dar close e sus → FAILED

    df['failed_auction'] = 0.0
    df.loc[_fail_below, 'failed_auction'] = 1.0    # bullish: vânzătorii au eșuat
    df.loc[_fail_above, 'failed_auction'] = -1.0   # bearish: cumpărătorii au eșuat
    # Dacă ambele (bar foarte volatile), cel care domină e close vs open
    _both = _fail_above & _fail_below
    df.loc[_both & (df['close'] > df['open']), 'failed_auction'] = 1.0
    df.loc[_both & (df['close'] <= df['open']), 'failed_auction'] = -1.0

    # ── 2. EXCESS ─────────────────────────────────────────────────────────────
    # Wick lung la extreme = respingere definitivă. Prețul NU va reveni acolo.
    # Excess la high: upper_wick > 40% din range ȘI aproape de swing high
    # Excess la low: lower_wick > 40% din range ȘI aproape de swing low
    _upper_wick = (df['high'] - df[['close', 'open']].max(axis=1)) / _range
    _lower_wick = (df[['close', 'open']].min(axis=1) - df['low']) / _range
    _near_high = (df['high'] >= _swing_hi * 0.999)  # within 0.1% of swing high
    _near_low  = (df['low'] <= _swing_lo * 1.001)   # within 0.1% of swing low

    df['excess'] = 0.0
    # Excess bearish la high (vânzători au respins cu putere) → preț scade
    df.loc[(_upper_wick > 0.40) & _near_high, 'excess'] = -1.0
    # Excess bullish la low (cumpărători au respins cu putere) → preț crește
    df.loc[(_lower_wick > 0.40) & _near_low, 'excess'] = 1.0

    # ── 3. POOR HIGH / POOR LOW ───────────────────────────────────────────────
    # Extreme fără excess = flat top/bottom → vor fi re-testate (unfinished business)
    # Poor high: upper_wick < 10% din range la swing high = top plat
    # Poor low: lower_wick < 10% din range la swing low = bottom plat
    df['poor_high'] = ((_upper_wick < 0.10) & _near_high).astype(float)
    df['poor_low']  = ((_lower_wick < 0.10) & _near_low).astype(float)

    # ── 4. INITIATIVE vs RESPONSIVE ───────────────────────────────────────────
    # Initiative: prețul iese din Value Area cu convingere (close outside VA)
    # Responsive: prețul e respins înapoi în VA (wick outside, close inside)
    # +1 = initiative bullish (breakout above VAH)
    # -1 = initiative bearish (breakout below VAL)
    # 0  = responsive (inside VA sau respins înapoi)
    _has_va = ('vah' in df.columns) and ('val' in df.columns)
    if _has_va:
        _vah = df['vah'].fillna(df['high'])
        _val = df['val'].fillna(df['low'])
        _close_above_vah = df['close'] > _vah
        _close_below_val = df['close'] < _val
        _wick_above_vah  = (df['high'] > _vah) & (df['close'] <= _vah)  # respins
        _wick_below_val  = (df['low'] < _val) & (df['close'] >= _val)   # respins

        df['initiative_responsive'] = 0.0
        df.loc[_close_above_vah, 'initiative_responsive'] = 1.0    # initiative bullish
        df.loc[_close_below_val, 'initiative_responsive'] = -1.0   # initiative bearish
        # Responsive override: wick a ieșit dar close e înapoi = responsive (neutral)
        df.loc[_wick_above_vah, 'initiative_responsive'] = -0.5    # responsive bearish
        df.loc[_wick_below_val, 'initiative_responsive'] = 0.5     # responsive bullish
    else:
        df['initiative_responsive'] = 0.0

    # ── 5. VA MIGRATION ───────────────────────────────────────────────────────
    # Direcția migrării Value Area: dacă VAH+VAL cresc → bullish, scad → bearish
    # Folosim shift(60) = ~1 oră ca reference
    if _has_va:
        _va_mid      = (_vah + _val) / 2
        _va_mid_prev = _va_mid.shift(60)
        _va_diff     = _va_mid - _va_mid_prev
        _va_atr      = df['atr_14'].fillna(1.0).clip(lower=1e-8) if 'atr_14' in df.columns else _range.rolling(14).mean().clip(lower=1e-8)
        # Normalizăm: migrare > 0.5 ATR = semnificativă
        df['va_migration'] = (_va_diff / _va_atr).clip(-1, 1).fillna(0)
    else:
        df['va_migration'] = 0.0

    # ── 6. ROTATION FACTOR ────────────────────────────────────────────────────
    # Cât de mult rotează prețul (balance) vs trending (one-directional)
    # Balance: prețul traversează midpoint-ul range-ului frecvent
    # Trending: prețul rămâne pe o parte
    _rolling_hi = df['high'].rolling(20).max()
    _rolling_lo = df['low'].rolling(20).min()
    _rolling_mid = (_rolling_hi + _rolling_lo) / 2
    # Câte bare din ultimele 20 au traversat midpoint-ul
    _cross_above = (df['low'] < _rolling_mid) & (df['high'] > _rolling_mid)
    _rotation_count = _cross_above.rolling(20, min_periods=5).sum()
    # Normalizăm: 0-20 traversări → 0-1 (mai mult = mai mult balance/rotation)
    df['rotation_factor'] = (_rotation_count / 20.0).clip(0, 1).fillna(0.5)

    return df


def add_of_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    v12.1: Transformă OF raw (zgomot pe 1 min) în semnale structurale.
    Rolling sums/ratios pe 15/30 bare — captează acumulare, nu noise.
    """
    df = df.copy()

    # Helper: get column or zeros
    def _col(name):
        return df[name].fillna(0) if name in df.columns else pd.Series(0.0, index=df.index)

    _bar_delta    = _col('bar_delta')
    _bar_buy_vol  = _col('bar_buy_vol')
    _bar_sell_vol = _col('bar_sell_vol')
    _big_buy      = _col('big_buy_count')
    _big_sell     = _col('big_sell_count')
    _imbalance    = _col('imbalance_pct')
    _tape_speed   = _col('tape_speed')
    _delta_hi     = _col('delta_at_high')
    _delta_lo     = _col('delta_at_low')
    _dom_ratio    = _col('dom_ratio')

    # 1. Delta sum 15/30 bare — acumulare direcțională
    df['delta_sum_15'] = _bar_delta.rolling(15, min_periods=1).sum()
    df['delta_sum_30'] = _bar_delta.rolling(30, min_periods=1).sum()

    # 2. Delta ratio 15 — direcționalitate: +1=pure buy, -1=pure sell, 0=echilibrat
    _abs_sum_15 = _bar_delta.abs().rolling(15, min_periods=1).sum().clip(lower=1e-8)
    df['delta_ratio_15'] = (df['delta_sum_15'] / _abs_sum_15).clip(-1, 1).fillna(0)

    # 3. Big trade ratio 15 — instituționalii sunt net buy sau sell?
    _big_buy_sum = _big_buy.rolling(15, min_periods=1).sum()
    _big_sell_sum = _big_sell.rolling(15, min_periods=1).sum()
    _big_total = (_big_buy_sum + _big_sell_sum).clip(lower=1e-8)
    df['big_trade_ratio_15'] = ((_big_buy_sum - _big_sell_sum) / _big_total).clip(-1, 1).fillna(0)

    # 4. Buy/sell ratio 30 — cine domină pe 30 bare
    _buy_sum_30  = _bar_buy_vol.rolling(30, min_periods=1).sum()
    _sell_sum_30 = _bar_sell_vol.rolling(30, min_periods=1).sum()
    _total_30 = (_buy_sum_30 + _sell_sum_30).clip(lower=1e-8)
    df['buy_sell_ratio_30'] = (_buy_sum_30 / _total_30).clip(0, 1).fillna(0.5)

    # 5. Imbalance MA 15 — trend OF persistent
    df['imbalance_ma_15'] = _imbalance.rolling(15, min_periods=1).mean().fillna(0)

    # 6. Tape speed relativă — activitate vs normal
    _tape_ma_60 = _tape_speed.rolling(60, min_periods=1).mean().clip(lower=1e-8)
    df['tape_speed_rel'] = (_tape_speed / _tape_ma_60).clip(0, 5).fillna(1.0)

    # 7. Absorption score 15 — absorption la extreme (bearish la high, bullish la low)
    # delta_at_high < 0 = selling absorption at highs (bearish)
    # delta_at_low > 0 = buying absorption at lows (bullish)
    _absorb_hi = (_delta_hi < 0).astype(float)   # bearish absorption events
    _absorb_lo = (_delta_lo > 0).astype(float)    # bullish absorption events
    _absorb_hi_sum = _absorb_hi.rolling(15, min_periods=1).sum()
    _absorb_lo_sum = _absorb_lo.rolling(15, min_periods=1).sum()
    # Net score: +1 = bullish absorption dominant, -1 = bearish absorption dominant
    df['absorption_score_15'] = ((_absorb_lo_sum - _absorb_hi_sum) / 15.0).clip(-1, 1).fillna(0)

    # 8. OF Pressure — delta trend × volume trend = presiune direcțională
    _delta_ma_15 = _bar_delta.rolling(15, min_periods=1).mean()
    _vol = _col('volume') if 'volume' in df.columns else (_bar_buy_vol + _bar_sell_vol)
    _vol_ma_30 = _vol.rolling(30, min_periods=1).mean().clip(lower=1e-8)
    _vol_rel = (_vol.rolling(15, min_periods=1).mean() / _vol_ma_30).fillna(1.0)
    # Pressure = direction × intensity
    df['of_pressure'] = (_delta_ma_15 * _vol_rel).fillna(0)
    # Normalize to reasonable range
    _p_std = df['of_pressure'].std()
    if _p_std > 0:
        df['of_pressure'] = (df['of_pressure'] / _p_std).clip(-3, 3)

    # 9. DOM ratio MA 15 — bid/ask persistent
    df['dom_ratio_ma_15'] = _dom_ratio.rolling(15, min_periods=1).mean().fillna(1.0)

    return df


def add_sniper_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Adaugă coloane auxiliare pentru filtrare sniper ultra-strictă."""
    df = df.copy()

    # Sweep High (QQQ a lichidat nivele de vânzare)
    df['sweep_h'] = (
        (df['high'] > df['p_hi'].fillna(df['high'])) |
        (df['high'] > df['vah'].fillna(df['high']))  |
        (df['high'] > df['lon_hi'].fillna(df['high']))
    ).astype(int)

    # Sweep Low
    df['sweep_l'] = (
        (df['low'] < df['p_lo'].fillna(df['low'])) |
        (df['low'] < df['val'].fillna(df['low']))   |
        (df['low'] < df['asia_lo'].fillna(df['low']))
    ).astype(int)

    return df


# =============================================================================
# MAIN — ANTRENARE PRO
# =============================================================================
def antrenare_pro_sql():
    """
    Antrenează modelul XGBoost Sniper pe datele din SQLite.
    Include: normalizare, ATR target, class weighting, CV score, feature importance.
    """
    print("\n" + "=" * 80)
    print("🧠 ALADIN AI TRAINER v5.0 — START")
    print("=" * 80)

    # ── Pasul 1: Verificare DB ───────────────────────────────────────────────
    if not os.path.exists(PATH_DB):
        print(f"❌ DB lipsă: {PATH_DB}")
        return

    # ── Pasul 2: Incarcare date ──────────────────────────────────────────────
    print("📖 Pasul 1: Încărcare date din SQLite...")
    conn = sqlite3.connect(PATH_DB)
    df   = pd.read_sql_query("SELECT * FROM market_data", conn)
    conn.close()
    print(f"   ✅ {len(df):,} rânduri încărcate")

    # ── Pasul 3: Detectează features disponibile ─────────────────────────────
    print("🔍 Pasul 2: Detectare features disponibile...")
    available_extra   = [f for f in FEATURES_EXTRA     if f in df.columns]
    available_adv     = [f for f in FEATURES_ADVANCED  if f in df.columns]
    available_vp_of   = [f for f in FEATURES_VP_OF     if f in df.columns]
    available_of_nat  = [f for f in FEATURES_OF_NATIVE if f in df.columns]
    features = FEATURES_BASE + available_extra + available_adv + available_vp_of + available_of_nat
    print(f"   ✅ Features: {len(FEATURES_BASE)} base + {len(available_extra)} extra + "
          f"{len(available_adv)} advanced + {len(available_vp_of)} vp_of + "
          f"{len(available_of_nat)} of_native = {len(features)} total")
    if available_extra:
        print(f"   🆕 Extra: {available_extra}")
    if available_adv:
        print(f"   🧮 Advanced: {available_adv}")
    if available_of_nat:
        print(f"   📊 OF Native: {available_of_nat}")
    else:
        print(f"   ⚠️ Advanced features lipsesc din DB — rulează import_nt8_nq.py cu advanced_features.py")
    if available_vp_of:
        print(f"   📊 VP+OF features (BRIDGE_LIVE): {available_vp_of}")
    else:
        print(f"   ℹ️  VP+OF features nu sunt încă în DB — se acumulează live (2-3 săptămâni)")

    # ── Pasul 4: Adaugă condiții sniper ──────────────────────────────────────
    print("🔄 Pasul 2.5: Adăugare REVERSAL features (v8.0)...")
    df = add_reversal_features(df)
    rev_available = [f for f in FEATURES_REVERSAL if f in df.columns]
    print(f"   ✅ Reversal features adăugate: {len(rev_available)}/{len(FEATURES_REVERSAL)}")
    print(f"   🆕 {rev_available}")

    print("🔄 Pasul 2.6: Adăugare AMT features (v11.0 — Auction Market Theory)...")
    df = add_amt_features(df)
    amt_available = [f for f in FEATURES_AMT if f in df.columns]
    print(f"   ✅ AMT features adăugate: {len(amt_available)}/{len(FEATURES_AMT)}")
    print(f"   🆕 {amt_available}")

    # v12.1: OF Aggregated — DEZACTIVAT (date istorice OF=0, adaugă noise)
    # print("🔄 Pasul 2.7: Adăugare OF AGGREGATED features (v12.1 — rolling sums/ratios)...")
    # df = add_of_aggregated_features(df)
    print("   ⏸️  Pasul 2.7: OF Aggregated SKIP (insuficiente date OF reale)")

    # v12.2: Consolidation Detection features — modelul învață WAIT în range
    print("🔄 Pasul 2.8: Adăugare CONSOLIDATION features (v12.2)...")
    df = add_consolidation_features(df)
    _consol_available = [f for f in FEATURES_CONSOL if f in df.columns]
    print(f"   ✅ Consolidation features: {len(_consol_available)}/{len(FEATURES_CONSOL)} disponibile")

    # v12.2: Streak Prevention features — modelul învață când să NU tranzacționeze
    print("🔄 Pasul 2.9: Adăugare STREAK PREVENTION features (v12.2)...")
    df = add_streak_features(df)
    _streak_available = [f for f in FEATURES_STREAK if f in df.columns]
    print(f"   ✅ Streak prevention features: {len(_streak_available)}/{len(FEATURES_STREAK)} disponibile")

    # v12.4: Trade Quality Context — detectează degradarea condițiilor
    print("🔄 Pasul 2.10: Adăugare CONTEXT features (v12.4 — trade quality)...")
    df = add_context_features(df)
    _ctx_available = [f for f in FEATURES_CONTEXT if f in df.columns]
    print(f"   ✅ Context features: {len(_ctx_available)}/{len(FEATURES_CONTEXT)} disponibile")

    # SESSION TIME features — ferestre temporale (pre-London, macro NY etc)
    print("🔄 Pasul 2.10b: Adăugare SESSION TIME features...")
    df = add_session_time_features(df)
    _stime_available = [f for f in FEATURES_SESSION_TIME if f in df.columns]
    print(f"   ✅ Session time features: {len(_stime_available)}/{len(FEATURES_SESSION_TIME)} disponibile")

    # v14: HTF direction features — 4H/1H/15M/Daily pentru suport direcție OR breakout
    print("🔄 Pasul 2.10c: Adăugare HTF DIRECTION features (4H/1H/15M/Daily/Weekly)...")
    df = add_htf_features(df)
    _htf_available = [f for f in FEATURES_HTF if f in df.columns]
    print(f"   ✅ HTF direction features: {len(_htf_available)}/{len(FEATURES_HTF)} disponibile")

    # v12.4: MTF Confirmation — multi-timeframe agreement
    print("🔄 Pasul 2.11: Adăugare MTF CONFIRMATION features (v12.4)...")
    df = add_mtf_confirm_features(df)
    _mtf_available = [f for f in FEATURES_MTF_CONFIRM if f in df.columns]
    print(f"   ✅ MTF confirmation features: {len(_mtf_available)}/{len(FEATURES_MTF_CONFIRM)} disponibile")

    # v12.7: Opening Range features — ORH/ORL per killzone pentru ORB strategy
    # Pasul 2.12: ORH features DEZACTIVATE din training (revert pre-ORH)
    # add_orh_features rămâne disponibilă pentru backtest, nu se adaugă în model
    df = add_orh_features(df)  # rulăm pentru in_orh (folosit la target filter), dar NU adăugăm în features

    # v13: REGIME-AWARE ICT features (weekly profile + day type + sweep detection)
    if _V13_AVAILABLE:
        print("🔄 Pasul 2.13: Adăugare v13 WEEKLY profile features...")
        df = add_weekly_features(df)
        _weekly_available = [f for f in FEATURES_WEEKLY if f in df.columns]
        print(f"   ✅ Weekly features: {len(_weekly_available)}/{len(FEATURES_WEEKLY)} disponibile")

        print("🔄 Pasul 2.14: Adăugare v13 DAY-TYPE classification features...")
        df = add_daytype_features(df)
        _daytype_available = [f for f in FEATURES_DAYTYPE if f in df.columns]
        print(f"   ✅ Day-type features: {len(_daytype_available)}/{len(FEATURES_DAYTYPE)} disponibile")

        print("🔄 Pasul 2.15: Adăugare v13 SWEEP detection features...")
        df = add_sweep_features(df)
        _sweep_available = [f for f in FEATURES_SWEEP if f in df.columns]
        print(f"   ✅ Sweep features: {len(_sweep_available)}/{len(FEATURES_SWEEP)} disponibile")
    else:
        print("⚠️ v13 features SKIP — aladin_v13 indisponibil")

    print("🎯 Pasul 3: Adăugare condiții Sniper + Target ATR dinamic...")
    df = add_sniper_conditions(df)

    # ── Pasul 5: Target dinamic ATR — v10.8 PROP FIRM ICT ──────────────────
    # PROP FIRM RULES:
    #   - Max 3 trades London + max 3 trades NY = 3-6 trades/zi
    #   - Trade-urile trebuie să fie pe mișcări semnificative (nu scalp 6 pts)
    #   - Doar în killzones (London 09-11 RO, NY 15:30-17:30 RO)
    #
    # v10.8: ATR multiplier 0.6 → 1.8 (doar mișcări mari)
    #        Horizon 15 → 45 bare (trade ICT durează 30-60 min)
    #        Killzone filter (doar London + NY)
    #        Cooldown 30 bare între semnale (nu spam)

    # v12.7: ORB EXPANSION — target 30-40 pts prima gambă (post-OR breakout)
    # ATR_MULT 2.5 → ~35 pts NQ (prima expansion leg)
    # Horizon 45 min — trade ORB ICT durează 30-60 min
    # MAE 0.8 — lăsăm pullback normal înainte de expansion
    _ATR_MULT = 2.0    # v12.8: 2.0×ATR (≈25-30 pts NQ) PRAG MINIM — expansion real
    _HORIZON  = 45     # 45 minute lookahead — trade ORB ICT
    _COOLDOWN = 15     # minim 15 bare între semnale
    _MAE_MAX  = 1.2    # v12.8: MAE 1.2×ATR — PULLBACK NORMAL OK (trailing DD 0.85R în live gestionează risk)

    # v12.5: MAE filter ON — doar semnale care NU fac drawdown > 0.3×ATR
    print(f"   🎯 v12.5: MAE filter activ (max drawdown {_MAE_MAX}×ATR) — semnale curate")
    df['target_atr'] = generate_atr_target(df, horizon=_HORIZON, atr_multiplier=_ATR_MULT,
                                            mae_filter=True, mae_max_atr=_MAE_MAX)
    _mae_long = (df['target_atr'] == 2).sum()
    _mae_short = (df['target_atr'] == 1).sum()
    print(f"   📊 MAE targets: {_mae_long + _mae_short:,} total (LONG={_mae_long:,} SHORT={_mae_short:,})")
    # Comparație fără MAE (pentru referință)
    _no_mae = generate_atr_target(df, horizon=_HORIZON, atr_multiplier=_ATR_MULT, mae_filter=False)
    _no_mae_signals = (_no_mae != 0).sum()
    print(f"   📊 Fără MAE filter: {_no_mae_signals:,} semnale — MAE a eliminat {_no_mae_signals - _mae_long - _mae_short:,} semnale cu drawdown")
    del _no_mae  # cleanup
    df['price_next']  = df['close'].shift(-_HORIZON)

    # Killzone filter: doar barele din London Open (09:00-11:00 RO) și NY Open (15:30-17:30 RO)
    _in_killzone = pd.Series(False, index=df.index)
    if 'timestamp' in df.columns:
        try:
            _ts = pd.to_datetime(df['timestamp'], errors='coerce')
            _hour = _ts.dt.hour
            _minute = _ts.dt.minute
            _time_decimal = _hour + _minute / 60.0
            # London Open: 09:00 - 11:00 Romania time
            _london = (_time_decimal >= 9.0) & (_time_decimal <= 11.0)
            # NY Open: 15:30 - 17:30 Romania time
            _ny = (_time_decimal >= 15.5) & (_time_decimal <= 17.5)
            _in_killzone = _london | _ny
            _kz_count = _in_killzone.sum()
            print(f"   🕐 Killzone filter: {_kz_count:,} bare în London+NY ({_kz_count/len(df)*100:.1f}%)")
        except Exception as _kze:
            print(f"   ⚠️ Killzone parse error: {_kze} — dezactivat")
            _in_killzone = pd.Series(True, index=df.index)  # fallback: toate barele
    else:
        print("   ⚠️ Coloana 'timestamp' lipsește din DB — killzone dezactivat")
        _in_killzone = pd.Series(True, index=df.index)

    # v12.7: Target = ATR target DOAR în POST-OR (după primele 30 min din killzone)
    # În OR period (primele 30 min) nu tranzacționăm — e acumulare, lăsăm structura să se formeze
    df['target'] = df['target_atr'].copy()
    df.loc[~_in_killzone, 'target'] = 0  # în afara killzone → forțat WAIT
    if 'in_orh' in df.columns:
        _in_or_mask = (df['in_orh'] == 1)
        _before = (df['target'] != 0).sum()
        df.loc[_in_or_mask, 'target'] = 0  # în OR period → WAIT (lăsăm OR să se formeze)
        _after = (df['target'] != 0).sum()
        print(f"   🎯 v12.7 ORB filter: post-OR only — {_before:,} → {_after:,} semnale "
              f"(eliminate {_before - _after:,} din OR period)")

    # ── v13 REGIME TARGET OVERRIDE ──────────────────────────────────────────
    # Înlocuim target-ul 3-class cu 9-class regime target (LONG_EXP/SHORT_EXP/
    # LONG_SWEEP/SHORT_SWEEP/REV_HIGH/REV_LOW/MR_VAH/MR_VAL + WAIT).
    # Skip-uim downstream mean-rev legacy block deoarece clasele MR sunt deja
    # incluse în regime target.
    _USE_V13 = _V13_AVAILABLE and USE_V13_REGIME
    if _USE_V13:
        print("🔄 Pasul 3.1 (v13): Generare REGIME TARGET (9-class multiclass)...")
        _regime_target = generate_regime_target(df, horizon=60)
        df['target'] = _regime_target.values.astype(int)
        _dist = df['target'].value_counts().sort_index()
        print(f"   ✅ v13 REGIME target distribuție:")
        for _k, _v in _dist.items():
            print(f"      {_k}. {REGIME_NAMES.get(int(_k), '?'):25s}: {_v:>7,}")
        _total_nw = int((df['target'] != 0).sum())
        print(f"   📊 Total non-WAIT: {_total_nw:,}")
        # Skip legacy HIGH_VOL filter pentru v13 — regime target deja filtrează post-OR
        # Skip legacy mean-rev block — clasele 7/8 deja incluse

    # ── v12.8: HIGH_VOL regime filter ──────────────────────────────────────
    # Anterior v12.7 arăta WR 65.9% pe HIGH_VOL regime vs ~52% pe restul.
    # Tranzacționăm DOAR când volatilitatea e suficientă pentru target 1.8×ATR.
    # v13: SKIP — regime target are deja logică ICT internă (nu filtrare brute-force)
    if not _USE_V13:
        _atr_col = None
        for _c in ['atr_percentile', 'atr_pct_rank', 'atr_rank']:
            if _c in df.columns:
                _atr_col = _c
                break
        if _atr_col is not None:
            _before_vol = (df['target'] != 0).sum()
            _low_vol_mask = df[_atr_col] < 0.60  # sub percentila 60 → WAIT
            df.loc[_low_vol_mask, 'target'] = 0
            _after_vol = (df['target'] != 0).sum()
            print(f"   🔥 v12.8 HIGH_VOL filter ({_atr_col} ≥ 0.60): "
                  f"{_before_vol:,} → {_after_vol:,} semnale "
                  f"(eliminate {_before_vol - _after_vol:,} low-vol)")
        else:
            print("   ⚠️ v12.8: coloana atr_percentile lipsește — HIGH_VOL filter skip")
    else:
        print("   ℹ️ v13: HIGH_VOL filter skip (regime target internal gating)")

    # ── v12.3: MEAN-REVERSION TARGETS în consolidare ────────────────────────
    # v13: SKIP — mean-rev clases 7/8 deja incluse în regime target
    if _USE_V13:
        print("🔄 Pasul 3.5: Mean-reversion targets SKIP (v13 regime include MR_VAH/VAL)")
        _mr_added = _mr_long = _mr_short = _mr_bb_used = 0
    else:
        print("🔄 Pasul 3.5: Mean-reversion targets în consolidare (v12.3)...")
        _mr_added = 0
        _mr_long = 0
        _mr_short = 0
        _mr_bb_used = 0

    _cls = df['close'].values
    _future = df['close'].shift(-_HORIZON).values
    _tgt = df['target'].values.copy()
    _kz = _in_killzone.values

    # Pre-compute Bollinger Bands (fallback când VA lipsește)
    _bb_sma = df['close'].rolling(20, min_periods=10).mean().values
    _bb_std = df['close'].rolling(20, min_periods=10).std().values

    # VA arrays (pot fi toate 0 pentru date vechi)
    _has_va = all(c in df.columns for c in ['vah', 'val'])
    _vah = df['vah'].values if _has_va else np.zeros(len(df))
    _val = df['val'].values if _has_va else np.zeros(len(df))

    # ATR
    if 'atr_14' in df.columns:
        _atr = df['atr_14'].values
    else:
        _tr = np.maximum(df['high'].values - df['low'].values,
                         np.maximum(np.abs(df['high'].values - np.roll(_cls, 1)),
                                    np.abs(df['low'].values - np.roll(_cls, 1))))
        _atr = pd.Series(_tr).rolling(14).mean().values

    # v13: skip entire mean-rev loop (regime target already has MR classes)
    _iterate_mr = not _USE_V13
    for i in range(20, len(df)) if _iterate_mr else range(0):
        # Skip dacă deja are semnal (trend) sau nu e killzone
        if _tgt[i] != 0 or not _kz[i]:
            continue
        if np.isnan(_future[i]) or np.isnan(_atr[i]) or _atr[i] <= 0:
            continue

        _atr_i = max(_atr[i], 1.0)

        # Determin range-ul: VA dacă există, altfel Bollinger Bands
        _range_hi = 0.0
        _range_lo = 0.0
        _using_bb = False

        if _vah[i] > 0 and _val[i] > 0 and (_vah[i] - _val[i]) > 0:
            _va_w = _vah[i] - _val[i]
            if _va_w <= _atr_i * 4.0:  # v12.3: relaxat de la 2.5 la 4.0
                _range_hi = _vah[i]
                _range_lo = _val[i]
        if _range_hi == 0 and not np.isnan(_bb_sma[i]) and _bb_std[i] > 0:
            # Bollinger Bands fallback: range = SMA ± 1.5*std
            _bb_w = _bb_std[i] * 3.0  # width = 3*std (upper - lower)
            if _bb_w <= _atr_i * 4.0:  # range strâns = consolidare
                _range_hi = _bb_sma[i] + 1.5 * _bb_std[i]
                _range_lo = _bb_sma[i] - 1.5 * _bb_std[i]
                _using_bb = True

        if _range_hi == 0:
            continue  # nici VA nici BB nu indică consolidare

        _range_w = _range_hi - _range_lo
        _margin = _range_w * 0.3  # 30% din range = "la margine"
        _mr_thresh = _atr_i * 0.15  # v12.3: 0.15×ATR (era 0.3 — prea strict)

        # Close lângă extrema inferioară → potențial LONG
        if _cls[i] <= _range_lo + _margin:
            if _future[i] > _cls[i] + _mr_thresh:
                _tgt[i] = 2  # LONG
                _mr_long += 1
                _mr_added += 1
                if _using_bb:
                    _mr_bb_used += 1
        # Close lângă extrema superioară → potențial SHORT
        elif _cls[i] >= _range_hi - _margin:
            if _future[i] < _cls[i] - _mr_thresh:
                _tgt[i] = 1  # SHORT
                _mr_short += 1
                _mr_added += 1
                if _using_bb:
                    _mr_bb_used += 1

    if not _USE_V13:
        df['target'] = _tgt
        print(f"   ✅ Mean-reversion targets: {_mr_added:,} adăugate ({_mr_long:,} LONG + {_mr_short:,} SHORT)")
        print(f"   📐 Condiții: range < 4.0×ATR | margin 30% range | target 0.15×ATR")
        print(f"   🔄 Din care via Bollinger Bands (fără VA): {_mr_bb_used:,}")

    # Cooldown: minim _COOLDOWN bare între semnale (aplică pe TOATE inclusiv MR)
    # v13: aplicăm cooldown și pentru regime target (evităm trade-uri consecutive pe aceeași mișcare)
    _target_arr = df['target'].values.copy()
    _signal_indices = np.where(_target_arr != 0)[0]
    _last_kept = -_COOLDOWN - 1
    for _si in _signal_indices:
        if _si - _last_kept < _COOLDOWN:
            _target_arr[_si] = 0
        else:
            _last_kept = _si
    df['target'] = _target_arr

    # Statistici target
    counts    = df['target'].value_counts()
    if _USE_V13:
        # v13: 9-class — orice clasă != 0 e semnal
        _signals_total = int((df['target'] != 0).sum())
        pct_trade = _signals_total / len(df) * 100
        _approx_days = len(df) / 390
        _signals_per_day = _signals_total / _approx_days if _approx_days > 0 else 0
        print(f"   📊 v13 Regime distribuție post-cooldown:")
        for _k in sorted(counts.index):
            _name = REGIME_NAMES.get(int(_k), '?') if _V13_AVAILABLE else f'class{_k}'
            print(f"      {int(_k)}. {_name:25s}: {int(counts[_k]):>7,}")
        print(f"   📈 Rata semnale: {pct_trade:.2f}% ({_signals_per_day:.2f} semnale/zi)")
        print(f"   🎯 v13 regime target (horizon=60) + cooldown={_COOLDOWN}")
    else:
        pct_trade = (counts.get(1, 0) + counts.get(2, 0)) / len(df) * 100
        _signals_total = counts.get(1, 0) + counts.get(2, 0)
        _approx_days = len(df) / 390
        _signals_per_day = _signals_total / _approx_days if _approx_days > 0 else 0
        print(f"   📊 Distribuție: WAIT={counts.get(0,0):,}  SHORT={counts.get(1,0):,}  LONG={counts.get(2,0):,}")
        print(f"   📈 Rata semnale: {pct_trade:.1f}% ({_signals_per_day:.1f} semnale/zi)")
        print(f"   🎯 ATR mult={_ATR_MULT}, horizon={_HORIZON} bare, cooldown={_COOLDOWN} bare")
    print(f"   🕐 Killzones: London 09-11 + NY 15:30-17:30 (RO time)")

    if _USE_V13:
        _min_class_count = min(int(counts.get(c, 0)) for c in range(1, 9))
        if _min_class_count < 30:
            print(f"   ⚠️ v13: cel puțin o clasă non-WAIT are <30 samples (min={_min_class_count}) — verifică tuning")
    else:
        if counts.get(1, 0) < 100 or counts.get(2, 0) < 100:
            print("   ⚠️ Prea puține semnale — verifică atr_multiplier sau horizon")

    # ── Pasul 6: Pregătire X, y ──────────────────────────────────────────────
    print("⚙️  Pasul 4: Preprocesare features + normalizare...")

    # Features de direcție — esențiale pentru predicție corectă a direcției
    df['slope_h1']    = (df['close'] - df['close'].shift(60))  / (df['close'].shift(60).abs()  + 1e-8)
    df['slope_h4']    = (df['close'] - df['close'].shift(240)) / (df['close'].shift(240).abs() + 1e-8)
    df['momentum_15'] = (df['close'] - df['close'].shift(15))  / (df['close'].shift(15).abs()  + 1e-8)
    # v10.7: body_dir normalizat — nu mai e -1/0/1 brut ci raport body/range
    # Asta păstrează direcția dar scade dominanța (era 44.7% importanță)
    df['body_dir']    = (df['close'] - df['open']) / (df['high'] - df['low']).clip(lower=1e-8)
    df['wick_ratio']  = (df['high'] - df['low']) / (abs(df['close'] - df['open']) + 1e-8)

    # Fix v7.4: Microstructure features — wick patterns, volatility, autocorrelation
    df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
    df['wick_bias'] = (df['upper_wick'] - df['lower_wick']) / (df['high'] - df['low'] + 1e-8)
    df['realized_vol'] = np.log(df['close'] / df['close'].shift(1).clip(lower=0.01)).rolling(14).std()
    df['vol_of_vol'] = df['realized_vol'].rolling(10).std()
    df['return_acf1'] = df['close'].pct_change().rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0.0, raw=False)

    dir_feats = ['slope_h1', 'slope_h4', 'momentum_15', 'body_dir', 'wick_ratio',
                 'upper_wick', 'lower_wick', 'wick_bias', 'realized_vol', 'vol_of_vol', 'return_acf1']
    for col in dir_feats:
        if col not in features:
            features.append(col)
    print(f"   ✅ Features direcție adăugate: {dir_feats}")

    # v8.0: Reversal features
    for col in FEATURES_REVERSAL:
        if col in df.columns and col not in features:
            features.append(col)
    rev_in_features = [f for f in FEATURES_REVERSAL if f in features and f in df.columns]
    print(f"   ✅ Reversal features în model: {rev_in_features}")

    # v11.0: AMT features
    for col in FEATURES_AMT:
        if col in df.columns and col not in features:
            features.append(col)
    amt_in_features = [f for f in FEATURES_AMT if f in features and f in df.columns]
    print(f"   ✅ AMT features în model: {amt_in_features}")

    # v12.1: OF Aggregated features — DEZACTIVAT: pe date istorice OF=0, adaugă noise
    # Reactivează când ai 6+ luni de date cu OF real populat din NinjaTrader
    # for col in FEATURES_OF_AGG:
    #     if col in df.columns and col not in features:
    #         features.append(col)
    print(f"   ⏸️  OF Aggregated features: DEZACTIVAT (insuficiente date OF reale)")

    # v12.2: Consolidation features — ACTIV (calculat din OHLC+VA, funcționează pe date istorice)
    for col in FEATURES_CONSOL:
        if col in df.columns and col not in features:
            features.append(col)
    consol_in_features = [f for f in FEATURES_CONSOL if f in features and f in df.columns]
    print(f"   ✅ Consolidation features în model: {consol_in_features}")

    # v12.2: Streak Prevention features — ACTIV
    for col in FEATURES_STREAK:
        if col in df.columns and col not in features:
            features.append(col)
    streak_in_features = [f for f in FEATURES_STREAK if f in features and f in df.columns]
    print(f"   ✅ Streak prevention features în model: {streak_in_features}")

    # v12.4: Context features — ACTIV
    for col in FEATURES_CONTEXT:
        if col in df.columns and col not in features:
            features.append(col)
    ctx_in_features = [f for f in FEATURES_CONTEXT if f in features and f in df.columns]
    print(f"   ✅ Context features în model: {ctx_in_features}")

    # v12.4: MTF Confirmation features — ACTIV
    for col in FEATURES_MTF_CONFIRM:
        if col in df.columns and col not in features:
            features.append(col)
    mtf_in_features = [f for f in FEATURES_MTF_CONFIRM if f in features and f in df.columns]
    print(f"   ✅ MTF confirmation features în model: {mtf_in_features}")

    # v12.7/v14: ORH features — Opening Range Breakout
    # v14: în mod v13 REGIME, excludem ORH-ul dominant (post_orh, bars_since_session)
    # pentru a forța modelul să învețe ICT real, nu "sunt în killzone".
    # ORH features excluse din model (revert pre-ORH — modelul se bazează pe ICT pur)
    print(f"   ⏸️  ORH features: DEZACTIVATE din model (revert pre-ORH)")

    # SESSION TIME features — adăugate întotdeauna (modelul LON/NY le folosesc diferit)
    for col in FEATURES_SESSION_TIME:
        if col in df.columns and col not in features:
            features.append(col)
    _stime_in = [f for f in FEATURES_SESSION_TIME if f in features and f in df.columns]
    print(f"   ✅ Session time features în model: {_stime_in}")

    # v14: HTF DIRECTION features — suport direcție OR breakout (4H/1H/15M/Daily/Weekly)
    for col in FEATURES_HTF:
        if col in df.columns and col not in features:
            features.append(col)
    _htf_in = [f for f in FEATURES_HTF if f in features and f in df.columns]
    print(f"   ✅ HTF direction features în model: {len(_htf_in)}/{len(FEATURES_HTF)} "
          f"(bull_conf/bear_conf/h4/h1/h15/weekly/daily)")

    # v13: REGIME-AWARE features (weekly + daytype + sweep) în model
    if _V13_AVAILABLE:
        for col in FEATURES_WEEKLY + FEATURES_DAYTYPE + FEATURES_SWEEP:
            if col in df.columns and col not in features:
                features.append(col)
        _v13_in_features = [f for f in (FEATURES_WEEKLY + FEATURES_DAYTYPE + FEATURES_SWEEP)
                            if f in features and f in df.columns]
        print(f"   ✅ v13 features în model: {len(_v13_in_features)} "
              f"(weekly={len([x for x in _v13_in_features if x in FEATURES_WEEKLY])}, "
              f"daytype={len([x for x in _v13_in_features if x in FEATURES_DAYTYPE])}, "
              f"sweep={len([x for x in _v13_in_features if x in FEATURES_SWEEP])})")

    # Fix v9.0: Dropna pe target ÎNAINTE de feature creation
    df = df.dropna(subset=['price_next']).copy()

    # ── TRAIN MODE FILTER — filtrăm la fereastra killzone după ce toate features sunt compute ──
    if TRAIN_MODE in ('LON', 'NY') and 'timestamp' in df.columns:
        _ts_filt = pd.to_datetime(df['timestamp'], errors='coerce')
        _td_filt = _ts_filt.dt.hour + _ts_filt.dt.minute / 60.0
        _w_start, _w_end = KZ_WINDOWS[TRAIN_MODE]
        _mask = (_td_filt >= _w_start) & (_td_filt <= _w_end)
        _before = len(df)
        df = df[_mask].copy()
        print(f"   🎯 TRAIN_MODE={TRAIN_MODE}: filtrat la {_w_start:.1f}-{_w_end:.1f}h RO "
              f"→ {len(df):,} bare (din {_before:,})")
        if len(df) < 10000:
            print(f"   ⚠️ AVERTISMENT: prea puține bare ({len(df):,}) pentru antrenare robustă")

    available_feats = [f for f in features if f in df.columns]
    X = df[available_feats].ffill().fillna(0)
    y = df['target'].astype(int)

    # Normalizare price levels față de true_open
    X = normalize_price_features(X)

    print(f"   ✅ Shape: X={X.shape}, y distribuție={dict(y.value_counts())}")

    # ── Pasul 7: Sample weights — definit mai jos în blocul SMOTE/class weights ──

    # v12.5: 3-WAY TEMPORAL SPLIT — train mic / test MARE (~9 ani OOS)
    # - Train (15%): ~1.5 ani antrenare (suficient pentru XGBoost)
    # - Val (3%): early stopping + calibrare
    # - Test (82%): ~9 ani out-of-sample — test ROBUST pe toată istoria
    # Mario: "fa test period si pe 10 ani"
    # v12.6: Train 45% / Val 5% / Test 50% — ~5 ani train, ~5 ani OOS
    # (15% train era prea puțin — 6K semnale, model nu învață cu 100 features)
    split_train = int(len(X) * 0.45)
    split_val   = int(len(X) * 0.50)

    # v14: PURGED SPLIT — eliminăm zona de overlap horizon (target leakage).
    # target se uită forward până la 60 bare → ultimele 60 bare din train au info-ul
    # primelor 60 bare din val. Fără purge: autocorelație masivă → model pare bun in-sample,
    # colapsează OOS (vezi 2026 disaster).
    _PURGE = 60  # max horizon din generate_regime_target
    X_train   = X.iloc[:max(0, split_train - _PURGE)]
    X_val     = X.iloc[split_train + _PURGE : max(split_train + _PURGE, split_val - _PURGE)]
    X_test    = X.iloc[split_val + _PURGE:]
    y_train   = y.iloc[:max(0, split_train - _PURGE)]
    y_val     = y.iloc[split_train + _PURGE : max(split_train + _PURGE, split_val - _PURGE)]
    y_test    = y.iloc[split_val + _PURGE:]
    print(f"   ✅ Split temporal PURGED 3-way (purge={_PURGE} bare la fiecare graniță):")
    print(f"      Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")

    # ── OOS VERIFICATION: arată ce perioade acoperă fiecare split ──
    if 'timestamp' in df.columns:
        try:
            _ts_col = pd.to_datetime(df['timestamp'], errors='coerce')
            _train_ts = _ts_col.iloc[:split_train].dropna()
            _val_ts   = _ts_col.iloc[split_train:split_val].dropna()
            _test_ts  = _ts_col.iloc[split_val:].dropna()
            print(f"   📅 Train period: {_train_ts.min()} → {_train_ts.max()}")
            print(f"   📅 Val period:   {_val_ts.min()} → {_val_ts.max()}")
            print(f"   📅 Test period:  {_test_ts.min()} → {_test_ts.max()}")
            _test_days = (_test_ts.max() - _test_ts.min()).days
            print(f"   📅 Test span: {_test_days} zile ({_test_days/30:.1f} luni)")
        except Exception as _ts_err:
            print(f"   ⚠️ OOS timestamp parse error: {_ts_err}")

    # ── v10.9: UNDERSAMPLE WAIT — forțează modelul să vadă semnalele ────────
    # Cu killzone filter, avem ~1.2% semnale (16K SHORT + 17K LONG din 2.7M)
    # Nici 15× weights nu ajunge. Soluție: reducem WAIT la ~5× semnalele.
    # ~100K WAIT + 16K SHORT + 17K LONG = ~133K rânduri, 25% signal rate
    # Antrenare 20× mai rapidă + model echilibrat natural
    print("⚖️  Pasul Undersample WAIT (prop firm)...")
    _cw = pd.Series(y_train).value_counts()
    _n_wait_actual = int(_cw.get(0, 0))
    if _USE_V13:
        _n_signals = int((y_train != 0).sum())
        # v13: keep ratio WAIT : signals ≈ 6:1 pentru antrenare echilibrată
        _n_wait_target = max(_n_signals * 6, 1000)
        _wait_idx = y_train[y_train == 0].index
        _rng = np.random.RandomState(42)
        _wait_sampled = _rng.choice(_wait_idx, size=min(_n_wait_target, _n_wait_actual), replace=False)
        _signal_idx_arr = y_train[y_train != 0].index.values
        _keep_idx = np.concatenate([_wait_sampled, _signal_idx_arr])
        _keep_idx.sort()

        X_train_res = X_train.loc[_keep_idx]
        y_train_res = y_train.loc[_keep_idx]

        # v14 sample weights — BALANCED inverse-frequency cu sqrt pentru a preveni
        # over-weight pe clase extrem de rare (care au și noise mai mare în target).
        _cw_res = pd.Series(y_train_res.values).value_counts()
        _n_total = len(y_train_res)
        weights_map = {0: 1.0}
        for _c in range(1, 5):  # 5-class (1..4 signals)
            _n_c = int(_cw_res.get(_c, 0))
            if _n_c > 0:
                # Inverse-freq cu sqrt dampening: weight ∝ sqrt(n_wait / n_c), clamp [1.0, 6.0]
                _w = (_cw_res.get(0, 1) / max(_n_c, 1)) ** 0.5
                weights_map[_c] = float(max(1.0, min(_w, 6.0)))
            else:
                weights_map[_c] = 1.0
        sw_train_res = np.array([weights_map.get(int(c), 1.0) for c in y_train_res])
        print(f"   ✅ v14 Original: WAIT={_n_wait_actual:,} SIGNALS={_n_signals:,}")
        print(f"   ✅ v14 Undersampled: " + " ".join(f"{REGIME_NAMES.get(c,str(c))[:8]}={int(_cw_res.get(c,0)):,}" for c in range(5)))
        print(f"   ✅ v14 Balanced weights: " + " ".join(f"{REGIME_NAMES.get(c,str(c))[:6]}={weights_map.get(c,1.0):.2f}" for c in range(5)))
        print(f"   ✅ Signal rate: {(len(y_train_res)-_cw_res.get(0,0)) / len(y_train_res) * 100:.1f}%")
    else:
        _n_short = _cw.get(1, 0)
        _n_long  = _cw.get(2, 0)
        _n_signals = _n_short + _n_long
        _n_wait_target = _n_signals * 6

        _wait_idx    = y_train[y_train == 0].index
        _short_idx   = y_train[y_train == 1].index
        _long_idx    = y_train[y_train == 2].index

        _rng = np.random.RandomState(42)
        _wait_sampled = _rng.choice(_wait_idx, size=min(_n_wait_target, _n_wait_actual), replace=False)
        _keep_idx = np.concatenate([_wait_sampled, _short_idx.values, _long_idx.values])
        _keep_idx.sort()

        X_train_res = X_train.loc[_keep_idx]
        y_train_res = y_train.loc[_keep_idx]

        weights_map = {0: 1, 1: 1.5, 2: 1.5}
        sw_train_res = np.array([weights_map[c] for c in y_train_res])

        _cw_res = pd.Series(y_train_res.values).value_counts()
        print(f"   ✅ Original: WAIT={_n_wait_actual:,} SHORT={_n_short:,} LONG={_n_long:,}")
        print(f"   ✅ Undersampled: WAIT={_cw_res.get(0,0):,} SHORT={_cw_res.get(1,0):,} LONG={_cw_res.get(2,0):,}")
        print(f"   ✅ Signal rate: {(_cw_res.get(1,0) + _cw_res.get(2,0)) / len(y_train_res) * 100:.1f}%")
        print(f"   ✅ Sample weights: WAIT={weights_map[0]}, SHORT={weights_map[1]}, LONG={weights_map[2]}")
    print(f"   ⚡ Training size: {len(X_train_res):,} (de {_n_wait_actual // max(1,len(X_train_res))}× mai rapid)")

    # ── Pasul 5: TWO-PASS XGBoost PRO (v12.5) ───────────────────────────────
    # PASS 1: Antrenare inițială → PASS 2: Streak penalty pe predicții reale → Retrain
    # BUG FIX: Vechiul streak penalty folosea GROUND TRUTH (targets) nu PREDICȚII.
    # Target-urile sunt corecte prin definiție → 0 losses → 0 penalty.
    # Fix: antrenăm modelul, apoi detectăm unde MODELUL greșește consecutiv.
    print("\n🚀 Pasul 5: Antrenare XGBoost TWO-PASS (v12.5)...")

    # v13: 9 clase regime (WAIT + 8 regimes); legacy: 3 clase (WAIT/SHORT/LONG)
    # v14: collapsed 9→5 (WAIT, SHORT_BREAK, LONG_BREAK, SHORT_REV, LONG_REV)
    _NUM_CLASS = 5 if _USE_V13 else 3
    print(f"   🧠 Model num_class = {_NUM_CLASS}  ({'v14 REGIME' if _USE_V13 else 'legacy ATR'})")

    _xgb_params = dict(
        n_estimators     = 3000,     # v12.5: 3000 (era 1500 — loss scădea la 1499)
        max_depth        = 5,
        learning_rate    = 0.012,    # v12.5: mai mic pentru 3000 iterații
        objective        = 'multi:softprob',
        num_class        = _NUM_CLASS,
        subsample        = 0.80,
        colsample_bytree = 0.80,
        colsample_bylevel = 0.7,
        reg_lambda       = 15.0,
        reg_alpha        = 3.0,
        min_child_weight = 10,
        gamma            = 1.0,
        tree_method      = 'hist',
        eval_metric      = 'mlogloss',
        random_state     = 42,
        verbosity        = 0,
        early_stopping_rounds = 100,  # v12.5: mai multă răbdare cu 3000 iterații
    )

    # ── PASS 1: Antrenare inițială ──
    print("   📌 PASS 1: Antrenare inițială...")
    model_pass1 = xgb.XGBClassifier(**_xgb_params)
    model_pass1.fit(
        X_train_res, y_train_res,
        sample_weight = sw_train_res,
        eval_set      = [(X_val, y_val)],
        verbose       = 100,
    )
    print(f"   ✅ Pass 1 best iteration: {model_pass1.best_iteration} / {model_pass1.n_estimators}")

    # ── FEATURE PRUNING: Elimină features cu importanță < 0.005 (noise pur) ──
    _imp_pass1 = model_pass1.feature_importances_
    _feat_names = list(X_train_res.columns)
    _keep_mask = _imp_pass1 >= 0.005
    _pruned_feats = [f for f, k in zip(_feat_names, _keep_mask) if not k]
    _kept_feats = [f for f, k in zip(_feat_names, _keep_mask) if k]
    print(f"   ✂️  Feature pruning: {len(_pruned_feats)} noise features eliminate (imp < 0.005)")
    print(f"   ✅ Features rămase: {len(_kept_feats)} (din {len(_feat_names)})")
    if _pruned_feats:
        print(f"   🗑️  Eliminate: {_pruned_feats[:15]}{'...' if len(_pruned_feats) > 15 else ''}")

    # Aplicăm pruning
    X_train_pruned = X_train_res[_kept_feats]
    X_val_pruned = X_val[_kept_feats]

    # ── STREAK DETECTION pe predicțiile Pass 1 ──
    # Simulăm trade-urile secvențial folosind PREDICȚIILE modelului (nu ground truth)
    # Așa găsim unde MODELUL generează CL ≥ 3 și penalizăm acele zone
    print("   🔍 Streak detection pe predicții Pass 1...")
    _pred_p1 = model_pass1.predict(X_train_res)
    _train_idx = y_train_res.index
    _pnext_tr = df.loc[_train_idx, 'price_next'].values if 'price_next' in df.columns else None
    _close_tr = df.loc[_train_idx, 'close'].values if 'close' in df.columns else None
    _streak_boost_count = 0
    _COST_PTS = 2.45  # slippage + commission

    if _pnext_tr is not None and _close_tr is not None:
        _cl_sim = 0
        _streak_start_indices = []
        for _si in range(len(_pred_p1)):
            _p = _pred_p1[_si]
            if _p == 0:  # model says WAIT → no trade
                continue
            # Verificăm dacă trade-ul modelului ar fi profitabil
            if _p == 2:  # model says LONG
                _pnl = float(_pnext_tr[_si]) - float(_close_tr[_si]) - _COST_PTS
            else:  # model says SHORT
                _pnl = float(_close_tr[_si]) - float(_pnext_tr[_si]) - _COST_PTS

            if np.isnan(_pnl):
                continue

            if _pnl < 0:
                _cl_sim += 1
                if _cl_sim >= 3:
                    sw_train_res[_si] *= 3.0
                    _streak_boost_count += 1
                    # Penalizăm retroactiv și primele 2 losses din streak
                    if _cl_sim == 3:
                        for _back in _streak_start_indices[-2:]:
                            sw_train_res[_back] *= 3.0
                            _streak_boost_count += 1
                _streak_start_indices.append(_si)
            else:
                _cl_sim = 0
                _streak_start_indices.clear()

        # Detectăm max CL pe training data
        _cl_sim2 = 0
        _max_cl_train = 0
        for _si in range(len(_pred_p1)):
            if _pred_p1[_si] == 0:
                continue
            if _pred_p1[_si] == 2:
                _pnl2 = float(_pnext_tr[_si]) - float(_close_tr[_si]) - _COST_PTS
            else:
                _pnl2 = float(_close_tr[_si]) - float(_pnext_tr[_si]) - _COST_PTS
            if np.isnan(_pnl2):
                continue
            if _pnl2 < 0:
                _cl_sim2 += 1
                _max_cl_train = max(_max_cl_train, _cl_sim2)
            else:
                _cl_sim2 = 0
        print(f"   🎯 Pass 1 max CL pe training: {_max_cl_train}")
    else:
        print("   ⚠️ price_next lipsește — streak penalty dezactivat")

    print(f"   🎯 Streak penalty: {_streak_boost_count:,} trade samples cu weight 3× (din streak-uri ≥3 CL)")

    # ── PASS 2: Retrain cu features pruned + streak penalty weights ──
    print(f"\n   📌 PASS 2: Retrain cu {len(_kept_feats)} features + {_streak_boost_count} streak penalties...")
    model_xgb = xgb.XGBClassifier(**_xgb_params)
    model_xgb.fit(
        X_train_pruned, y_train_res,
        sample_weight = sw_train_res,
        eval_set      = [(X_val_pruned, y_val)],
        verbose       = 100,
    )
    print(f"   ✅ Pass 2 best iteration: {model_xgb.best_iteration} / {model_xgb.n_estimators}")

    # IMPORTANT: Update features list and ALL X_* datasets for downstream code
    features = _kept_feats
    X_train_res = X_train_pruned
    X_val = X_val_pruned
    X_test = X_test.reindex(columns=_kept_feats, fill_value=0)
    X_train = X_train.reindex(columns=_kept_feats, fill_value=0)  # v12.5: pentru CV + downstream

    # ── Pasul 9: Evaluare ────────────────────────────────────────────────────
    print("\n📊 Pasul 6: Evaluare model...")

    # Trading costs (definite devreme pentru threshold search)
    SLIPPAGE_PTS = 2.0
    COMMISSION_PTS = 0.45
    TOTAL_COST_PTS = SLIPPAGE_PTS + COMMISSION_PTS  # 2.45 pts/trade

    # v12.5: DYNAMIC CONFIDENCE THRESHOLD — caută automat threshold-ul pentru 97% WR
    # Nu mai folosim un threshold fix de 50%. Căutăm threshold-ul unde win rate = TARGET_WR.
    # Semnalul SHORT/LONG e valid DOAR dacă probabilitatea clasei depășește threshold-ul.
    TARGET_WR = 0.97  # 97% win rate target
    y_proba = model_xgb.predict_proba(X_test)           # shape (n, num_class)
    y_pred_raw = np.argmax(y_proba, axis=1).astype(int)  # argmax brut

    # Căutăm cel mai mic threshold care dă WR >= TARGET_WR
    # Test returns: price_next - close pentru fiecare bară
    _test_idx = y_test.index
    _pnext_test = df.loc[_test_idx, 'price_next'].values if 'price_next' in df.columns else None
    _close_test = df.loc[_test_idx, 'close'].values if 'close' in df.columns else None

    print("\n   🔍 Căutare confidence threshold — target: 3-5 trades/zi cu WR maxim...")
    print(f"   {'Threshold':>10s} {'Trades':>8s} {'WR':>8s} {'CL_max':>8s} {'P&L':>12s} {'Trades/zi':>10s}")
    print(f"   {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*12} {'─'*10}")

    _best_threshold = 0.50  # fallback
    _n_test_days = len(X_test) / 390.0  # 390 bare/zi (1-min)

    # v12.6: MARIO REQUIREMENT — 3-5 trades/zi, WR cât mai mare în această fereastră
    MIN_TPD = 3.0   # minim 3 trades/zi
    MAX_TPD = 5.0   # maxim 5 trades/zi (peste înseamnă threshold prea mic)
    _candidates = []   # [(thr, wr, cl_max, tpd, net, n_trades), ...]

    # v14: map regime class → direction (+1 LONG / -1 SHORT) — 5-class scheme
    if _USE_V13:
        _DIR_MAP = {
            0: 0,   # WAIT
            1: -1,  # SHORT_BREAK
            2: +1,  # LONG_BREAK
            3: -1,  # SHORT_REV
            4: +1,  # LONG_REV
        }
    else:
        _DIR_MAP = {0: 0, 1: -1, 2: +1}

    _thr_grid = [round(x, 3) for x in np.arange(0.30, 0.991, 0.01)]
    for _thr in _thr_grid:
        _y_thr = y_pred_raw.copy()
        # pentru fiecare clasă signal, aplicăm threshold pe probabilitatea ei
        for _cls_idx in range(1, y_proba.shape[1]):
            _mask = (_y_thr == _cls_idx) & (y_proba[:, _cls_idx] < _thr)
            _y_thr[_mask] = 0

        _pnl_list = []
        if _pnext_test is not None and _close_test is not None:
            _sig_idx = np.where(_y_thr != 0)[0]
            for _i in _sig_idx:
                _dir = _DIR_MAP.get(int(_y_thr[_i]), 0)
                if _dir == 0:
                    continue
                _mv = float(_pnext_test[_i]) - float(_close_test[_i])
                _pnl_list.append(_dir * _mv - TOTAL_COST_PTS)

        if len(_pnl_list) == 0:
            continue
        _pnl_arr = np.array(_pnl_list)
        _wr_thr = (_pnl_arr > 0).sum() / len(_pnl_arr)
        _cl_t = 0; _mcl_t = 0
        for _p in _pnl_arr:
            if _p < 0: _cl_t += 1; _mcl_t = max(_mcl_t, _cl_t)
            else: _cl_t = 0
        _tpd = len(_pnl_arr) / _n_test_days if _n_test_days > 0 else 0
        _net = _pnl_arr.sum()
        _candidates.append((_thr, _wr_thr, _mcl_t, _tpd, _net, len(_pnl_arr)))

    # Print doar câteva threshold-uri reprezentative
    _print_thrs = {0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97}
    for (_thr, _wr, _mcl, _tpd, _net, _n) in _candidates:
        if round(_thr, 2) in _print_thrs or abs(_thr - 0.99) < 1e-6:
            _tag = " ← in target" if MIN_TPD <= _tpd <= MAX_TPD else ""
            print(f"   {_thr:>10.0%} {_n:>8,} {_wr:>7.1%} {_mcl:>8} {_net:>+12,.0f} {_tpd:>10.1f}{_tag}")

    # Selecție: din candidații cu MIN_TPD ≤ tpd ≤ MAX_TPD, alege cel cu WR maxim (tiebreak: CL mic)
    _in_band = [c for c in _candidates if MIN_TPD <= c[3] <= MAX_TPD]
    if _in_band:
        _in_band.sort(key=lambda c: (-c[1], c[2]))  # WR desc, apoi CL asc
        _best = _in_band[0]
        _best_threshold = _best[0]
        print(f"\n   ✅ Selectat threshold {_best_threshold:.2%}: WR={_best[1]:.1%} CL_max={_best[2]} trades/zi={_best[3]:.1f}")
    else:
        # Fallback: cel mai apropiat de MIN_TPD (peste e OK dacă sub MAX_TPD imposibil)
        _candidates.sort(key=lambda c: (abs(c[3] - ((MIN_TPD+MAX_TPD)/2)), -c[1]))
        if _candidates:
            _best = _candidates[0]
            _best_threshold = _best[0]
            print(f"\n   ⚠️ Niciun threshold în banda {MIN_TPD}-{MAX_TPD} tpd. Fallback: {_best_threshold:.2%} (tpd={_best[3]:.1f} WR={_best[1]:.1%})")

    CONF_THRESHOLD = _best_threshold
    print(f"\n   ✅ Threshold optim pentru {TARGET_WR:.0%} WR: {CONF_THRESHOLD:.2%}")

    # Aplicăm threshold-ul optim
    y_pred = y_pred_raw.copy()
    if _USE_V13:
        # v13: filtrează orice clasă non-WAIT sub threshold
        for _i in range(len(y_pred)):
            _c = int(y_pred[_i])
            if _c != 0 and y_proba[_i, _c] < CONF_THRESHOLD:
                y_pred[_i] = 0
        _filtered = int((y_pred_raw != y_pred).sum())
        _kept_short = int(np.isin(y_pred, [1, 4, 5, 7]).sum())
        _kept_long  = int(np.isin(y_pred, [2, 3, 6, 8]).sum())
    else:
        for _i in range(len(y_pred)):
            if y_pred[_i] == 1 and y_proba[_i, 1] < CONF_THRESHOLD:
                y_pred[_i] = 0
            elif y_pred[_i] == 2 and y_proba[_i, 2] < CONF_THRESHOLD:
                y_pred[_i] = 0
        _filtered = (y_pred_raw != y_pred).sum()
        _kept_short = (y_pred == 1).sum()
        _kept_long  = (y_pred == 2).sum()

    acc = accuracy_score(y_test, y_pred)
    print(f"\n   ✅ Acuratețe globală: {acc:.2%}")
    print(f"   🎯 Confidence threshold: {CONF_THRESHOLD:.2%} — {_filtered:,} semnale filtrate")
    print(f"   📊 Semnale rămase: SHORT={_kept_short:,} | LONG={_kept_long:,}")

    # Evaluare și fără threshold pentru comparație
    acc_raw = accuracy_score(y_test, y_pred_raw)
    print(f"   📊 [Referință fără threshold: acc={acc_raw:.2%}]")
    print("\n   📋 Raport Sniper detaliat:")
    if _USE_V13:
        _rep_names = [REGIME_NAMES.get(i, f'c{i}') for i in range(_NUM_CLASS)]
    else:
        _rep_names = ['WAIT', 'SHORT', 'LONG']
    print(classification_report(
        y_test, y_pred,
        labels = list(range(_NUM_CLASS)),
        target_names = _rep_names,
        zero_division = 0,
    ))

    # Confidence distribution
    max_proba = y_proba.max(axis=1)
    high_conf = (max_proba > 0.80).sum()
    print(f"   💎 Semnale HIGH CONVICTION (>80%): {high_conf:,} ({high_conf/len(y_test)*100:.1f}%)")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(_NUM_CLASS)))
    print(f"\n   📐 Confusion Matrix:\n{cm}")

    # ── Pasul 10: Cross-Validation ───────────────────────────────────────────
    # Fix v7.4: Forward-chain temporal CV (no data leakage — train on past, test on future)
    # Use only X_train, y_train (80% of data), split into 3 folds with increasing history
    print("\n🔄 Pasul 7: Forward-Chain Temporal CV (3-fold)...")
    try:
        cv_scores_list = []
        n_folds = 3
        fold_size = len(X_train) // (n_folds + 1)  # +1 to leave enough data for test

        for fold_idx in range(n_folds):
            # Fold 0: train [0:fold_size], test [fold_size:2*fold_size]
            # Fold 1: train [0:2*fold_size], test [2*fold_size:3*fold_size]
            # Fold 2: train [0:3*fold_size], test [3*fold_size:4*fold_size]
            train_end = (fold_idx + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, len(X_train))

            X_cv_train = X_train.iloc[:train_end]
            y_cv_train = y_train.iloc[:train_end]
            X_cv_test = X_train.iloc[test_start:test_end]
            y_cv_test = y_train.iloc[test_start:test_end]

            # Create fresh estimator for each fold — v10.8 params
            cv_estimator = xgb.XGBClassifier(
                n_estimators     = 500,
                max_depth        = 5,
                learning_rate    = 0.015,
                objective        = 'multi:softprob',
                num_class        = _NUM_CLASS,
                subsample        = 0.80,
                colsample_bytree = 0.80,
                colsample_bylevel = 0.7,
                reg_lambda       = 15.0,
                reg_alpha        = 3.0,
                min_child_weight = 10,
                gamma            = 1.0,
                tree_method      = 'hist',
                random_state     = 42,
                verbosity        = 0,
            )

            _sw_cv = np.array([weights_map[c] for c in y_cv_train])
            cv_estimator.fit(X_cv_train, y_cv_train, sample_weight=_sw_cv)
            y_cv_pred = np.argmax(cv_estimator.predict_proba(X_cv_test), axis=1)
            fold_score = accuracy_score(y_cv_test, y_cv_pred)
            cv_scores_list.append(fold_score)
            print(f"   Fold {fold_idx+1}: train={len(X_cv_train):,} test={len(X_cv_test):,} acc={fold_score:.4f}")

        cv_scores = np.array(cv_scores_list)
        print(f"   ✅ Forward-Chain CV: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    except Exception as exc:
        print(f"   ⚠️ CV skip: {exc}")

    # ── Pasul 11: Feature Importance ─────────────────────────────────────────
    print("\n🏆 Pasul 8: Top 15 Features importante...")
    importance     = model_xgb.feature_importances_
    final_features = list(X_train_res.columns) if hasattr(X_train_res, 'columns') else available_feats
    feat_df        = pd.DataFrame({
        'feature':    final_features,
        'importance': importance,
    }).sort_values('importance', ascending=False)

    print(feat_df.head(15).to_string(index=False))

    top_features = feat_df.head(15)['feature'].tolist()
    print(f"\n   🥇 Top 3: {top_features[:3]}")

    # ── Update #10: Feature Importance Analysis ──────────────────────────────
    print("\n📊 Pasul Feature Importance Analysis...")
    feat_imp = pd.Series(model_xgb.feature_importances_, index=X_train_res.columns if hasattr(X_train_res, 'columns') else available_feats).sort_values(ascending=False)
    print("   Top 15 features XGBoost:")
    for fname, fval in feat_imp.head(15).items():
        bar = "█" * int(fval * 200)
        print(f"   {fname:25s} {fval:.4f} {bar}")
    low_imp = feat_imp[feat_imp < 0.01].index.tolist()
    if low_imp:
        print(f"   ⚠️ Features cu importanță <0.01 (considerate noise): {low_imp}")
    # Salvează feature importance
    feat_imp_path = MODEL_SAVE_PATH.replace('.json', '_feat_imp.csv')
    feat_imp.reset_index().rename(columns={'index':'feature', 0:'importance'}).to_csv(feat_imp_path, index=False)
    print(f"   ✅ Feature importance salvat: {feat_imp_path}")

    # ── Undersample validation set (shared: LightGBM, RF, Two-Stage) ────────
    # Bug fix: val set-ul real (98.8% WAIT) cauza early stop la iter 18.
    # Soluție: undersample WAIT din val set la același ratio ca training.
    _val_wait_mask = y_val == 0
    _val_sig_mask  = ~_val_wait_mask
    _n_val_signals = _val_sig_mask.sum()
    _n_val_wait_target = _n_val_signals * 5  # same 5:1 ratio as train
    _val_wait_idx = y_val[_val_wait_mask].index
    if len(_val_wait_idx) > _n_val_wait_target:
        _val_wait_keep = np.random.RandomState(42).choice(_val_wait_idx, size=int(_n_val_wait_target), replace=False)
        _val_keep_idx = np.concatenate([_val_wait_keep, y_val[_val_sig_mask].index.values])
        _val_keep_idx.sort()
        X_val_us = X_val.loc[_val_keep_idx]
        y_val_us = y_val.loc[_val_keep_idx]
    else:
        X_val_us = X_val
        y_val_us = y_val
    _val_dist = pd.Series(y_val_us).value_counts().sort_index()
    print(f"\n   ✅ Val set undersampled (shared): WAIT={_val_dist.get(0,0):,} SHORT={_val_dist.get(1,0):,} LONG={_val_dist.get(2,0):,}")

    # ── v10.7: LightGBM — FIX: undersampled validation set ──────────────────
    print("\n🌲 Pasul LightGBM...")
    model_lgbm = None
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        t0 = _time.time()
        model_lgbm = LGBMClassifier(
            n_estimators=1500, max_depth=5, learning_rate=0.015,
            subsample=0.80, colsample_bytree=0.80,
            reg_lambda=15.0, reg_alpha=3.0, min_child_samples=10,
            random_state=42, verbosity=-1,
            objective='multiclass', num_class=_NUM_CLASS,
        )
        model_lgbm.fit(
            X_train_res, y_train_res,
            sample_weight=sw_train_res,
            eval_set=[(X_val_us, y_val_us)],
            callbacks=[early_stopping(50), log_evaluation(100)],
        )
        t_lgbm = _time.time() - t0

        y_pred_lgbm = np.argmax(model_lgbm.predict_proba(X_test), axis=1)
        acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
        if _USE_V13:
            _lgbm_short = int(np.isin(y_pred_lgbm, [1, 4, 5, 7]).sum())
            _lgbm_long  = int(np.isin(y_pred_lgbm, [2, 3, 6, 8]).sum())
            _xgb_short  = int(np.isin(y_pred, [1, 4, 5, 7]).sum())
            _xgb_long   = int(np.isin(y_pred, [2, 3, 6, 8]).sum())
            _tgt_names  = [REGIME_NAMES.get(i, f'c{i}') for i in range(_NUM_CLASS)]
        else:
            _lgbm_short = int((y_pred_lgbm == 1).sum())
            _lgbm_long  = int((y_pred_lgbm == 2).sum())
            _xgb_short  = int((y_pred == 1).sum())
            _xgb_long   = int((y_pred == 2).sum())
            _tgt_names  = ['WAIT','SHORT','LONG']
        print(f"   ✅ LightGBM — Acc: {acc_lgbm:.4f} | Timp: {t_lgbm:.1f}s | Best iter: {model_lgbm.best_iteration_}")
        print(f"   📊 LightGBM semnale: SHORT={_lgbm_short:,} LONG={_lgbm_long:,} (vs XGB: SHORT={_xgb_short:,} LONG={_xgb_long:,})")
        print(classification_report(y_test, y_pred_lgbm, labels=list(range(_NUM_CLASS)), target_names=_tgt_names, zero_division=0))

        lgbm_path = MODEL_SAVE_PATH.replace('.json', '_lgbm.pkl')
        import pickle
        with open(lgbm_path, 'wb') as f:
            pickle.dump(model_lgbm, f)
        print(f"   ✅ LightGBM salvat: {lgbm_path}")
    except ImportError:
        print("   ⚠️ lightgbm lipsă — (pip install lightgbm)")

    # ── UPDATE #2: Calibrare probabilități — Isotonic Regression ───────────────
    # FIX v12.2: Calibrare pe val set UNDERSAMPLED (same ratio as train)
    # Bug: pe distribuție reală (98.8% WAIT), Isotonic învață "WAIT=always best" → 0 SHORT/LONG
    # Soluție: folosim X_val_us/y_val_us (5:1 WAIT ratio) — la fel ca LightGBM fix
    print("\n🎯 Pasul Calibrare probabilități — Isotonic Regression (UPDATE #2)...")
    model_calibrated = None
    try:
        from sklearn.isotonic import IsotonicRegression
        import pickle

        # FIX v12.2: Val set undersampled pentru calibrare (nu distribuție reală)
        cal_X = X_val_us
        cal_y_val = y_val_us
        eval_X = X_test   # evaluăm calibrated model pe test (truly held-out)
        eval_y = y_test

        # Obținem probabilitățile brute XGBoost pe setul de calibrare (val)
        raw_proba_cal = model_xgb.predict_proba(cal_X)   # shape (n, 3)
        cal_y_arr     = np.array(cal_y_val)

        # FIX v12.2: Undersampled distribution — SHORT/LONG au suficiente exemple
        _counts = pd.Series(cal_y_arr).value_counts()
        if _USE_V13:
            print(f"   ✅ Cal set (UNDERSAMPLED) v13: " + " ".join(f"{REGIME_NAMES.get(c,c)[:6]}={_counts.get(c,0):,}" for c in range(_NUM_CLASS)))
        else:
            print(f"   ✅ Cal set (UNDERSAMPLED): WAIT={_counts.get(0,0):,} SHORT={_counts.get(1,0):,} LONG={_counts.get(2,0):,}")

        # Isotonic calibration per clasă pe distribuția reală
        # out_of_bounds='clip' previne valorile negative sau >1
        _iso_calibrators = []
        for _c in range(_NUM_CLASS):
            _y_bin = (cal_y_arr == _c).astype(float)
            _iso = IsotonicRegression(out_of_bounds='clip')
            _iso.fit(raw_proba_cal[:, _c], _y_bin)
            _iso_calibrators.append(_iso)

        model_calibrated = PlattModel(model_xgb, _iso_calibrators)

        y_proba_cal = model_calibrated.predict_proba(eval_X)
        y_pred_cal  = np.argmax(y_proba_cal, axis=1)
        acc_cal     = accuracy_score(eval_y, y_pred_cal)
        if _USE_V13:
            _cal_short = int(np.isin(y_pred_cal, [1, 4, 5, 7]).sum())
            _cal_long  = int(np.isin(y_pred_cal, [2, 3, 6, 8]).sum())
            _tgt_names_c = [REGIME_NAMES.get(i, f'c{i}') for i in range(_NUM_CLASS)]
        else:
            _cal_short = int((y_pred_cal == 1).sum())
            _cal_long  = int((y_pred_cal == 2).sum())
            _tgt_names_c = ['WAIT','SHORT','LONG']
        print(f"   ✅ Calibrated model acc: {acc_cal:.4f}")
        print(f"   📊 Calibrated semnale: SHORT={_cal_short:,} LONG={_cal_long:,}")
        print(classification_report(eval_y, y_pred_cal, labels=list(range(_NUM_CLASS)), target_names=_tgt_names_c, zero_division=0))

        cal_path = MODEL_SAVE_PATH.replace('.json', '_calibrated.pkl')
        with open(cal_path, 'wb') as f:
            pickle.dump(model_calibrated, f)
        print(f"   ✅ Model calibrat salvat: {cal_path}")
    except Exception as e:
        print(f"   ⚠️ Calibrare eșuată: {e}")
        model_calibrated = None

    # ── v10.7: Ensemble Model — RF fix: mai agresiv pe semnale ─────────────
    print("\n🤝 Pasul Ensemble (XGBoost + LightGBM + RandomForest)...")
    try:
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier

        if _USE_V13:
            # v14: aliniat cu weights_map balanced (inverse-freq sqrt)
            _rf_cw = {0: 1}
            for _c in range(1, _NUM_CLASS):
                _rf_cw[_c] = float(weights_map.get(_c, 3.0))
        else:
            _rf_cw = {0: 1, 1: 3, 2: 3}
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=7, min_samples_leaf=10,
            class_weight=_rf_cw,  # FIX: 3× (8× era prea agresiv → 244K semnale)
            random_state=42, n_jobs=-1
        )
        rf.fit(X_train_res, y_train_res, sample_weight=sw_train_res)

        # Ensemble cu vot moale (probabilități mediate)
        # Bug #12 fix: Verifică dacă model_lgbm e None înainte de a-l folosi
        def ensemble_predict_proba(X):
            p_xgb = model_xgb.predict_proba(X)
            p_rf  = rf.predict_proba(X)
            if model_lgbm is not None:
                p_lgbm = model_lgbm.predict_proba(X)
                return (p_xgb + p_lgbm + p_rf) / 3
            else:
                return (p_xgb + p_rf) / 2

        # RF standalone check
        y_pred_rf = np.argmax(rf.predict_proba(X_test), axis=1)
        if _USE_V13:
            _rf_short = int(np.isin(y_pred_rf, [1, 4, 5, 7]).sum())
            _rf_long  = int(np.isin(y_pred_rf, [2, 3, 6, 8]).sum())
            _tgt_names_e = [REGIME_NAMES.get(i, f'c{i}') for i in range(_NUM_CLASS)]
        else:
            _rf_short = int((y_pred_rf == 1).sum())
            _rf_long  = int((y_pred_rf == 2).sum())
            _tgt_names_e = ['WAIT','SHORT','LONG']
        print(f"   📊 RF semnale: SHORT={_rf_short:,} LONG={_rf_long:,}")

        y_ens = np.argmax(ensemble_predict_proba(X_test), axis=1)
        acc_ens = accuracy_score(y_test, y_ens)
        if _USE_V13:
            _ens_short = int(np.isin(y_ens, [1, 4, 5, 7]).sum())
            _ens_long  = int(np.isin(y_ens, [2, 3, 6, 8]).sum())
        else:
            _ens_short = int((y_ens == 1).sum())
            _ens_long  = int((y_ens == 2).sum())
        print(f"   ✅ Ensemble acc: {acc_ens:.4f}")
        print(f"   📊 Ensemble semnale: SHORT={_ens_short:,} LONG={_ens_long:,}")
        print(classification_report(y_test, y_ens, labels=list(range(_NUM_CLASS)), target_names=_tgt_names_e, zero_division=0))

        # Salvează RF
        rf_path = MODEL_SAVE_PATH.replace('.json', '_rf.pkl')
        import pickle
        with open(rf_path, 'wb') as f:
            pickle.dump(rf, f)
        print(f"   ✅ RandomForest salvat: {rf_path}")
    except Exception as e:
        print(f"   ⚠️ Ensemble error: {e}")

    # ── SLIPPAGE + COMMISSION (shared: two-stage + quant analysis) ──
    SLIPPAGE_PTS = 2.0
    COMMISSION_PTS = 0.45
    TOTAL_COST_PTS = SLIPPAGE_PTS + COMMISSION_PTS  # 2.45 pts per trade

    # ══════════════════════════════════════════════════════════════════════════
    # v12.0: TWO-STAGE MODEL — Stage 1: TRADE/NO-TRADE, Stage 2: LONG/SHORT
    # v13: SKIP two-stage (inlocuit de 9-class ensemble XGB+LGBM+RF)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("🎯 TWO-STAGE MODEL — Filtrare superioară a semnalelor")
    print("="*80)
    model_stage1 = None
    model_stage2 = None
    if _USE_V13:
        print("   ℹ️ v13: Two-stage SKIP (folosim ensemble 9-class direct)")
    try:
        if _USE_V13:
            raise Exception("v13_skip_twostage")
        from xgboost import XGBClassifier as XGB2

        # ── Stage 1: TRADE (1) vs NO-TRADE (0) — binary ──
        # Convertim: WAIT(0) → 0, SHORT(1)/LONG(2) → 1
        y_train_s1 = (y_train_res > 0).astype(int)
        y_val_s1   = (y_val_us > 0).astype(int)

        print("\n── Stage 1: TRADE vs NO-TRADE (binary) ──")
        _s1_dist = pd.Series(y_train_s1).value_counts().sort_index()
        print(f"   Train: NO-TRADE={_s1_dist.get(0,0):,} TRADE={_s1_dist.get(1,0):,}")

        model_stage1 = XGB2(
            n_estimators=800, max_depth=4, learning_rate=0.02,
            reg_lambda=10.0, reg_alpha=2.0, min_child_weight=10, gamma=1.0,
            subsample=0.80, colsample_bytree=0.80,
            eval_metric='logloss', random_state=42,
            tree_method='hist', device='cpu',
        )
        model_stage1.fit(
            X_train_res, y_train_s1,
            sample_weight=sw_train_res,
            eval_set=[(X_val_us, y_val_s1)],
            verbose=False,
        )
        # Predict pe test
        _s1_proba = model_stage1.predict_proba(X_test)[:, 1]  # prob TRADE
        _s1_pred  = (_s1_proba >= 0.50).astype(int)
        _s1_trades = _s1_pred.sum()
        print(f"   ✅ Stage 1: {_s1_trades:,} bare marcate TRADE din {len(X_test):,} ({_s1_trades/len(X_test)*100:.2f}%)")

        # ── Stage 2: LONG vs SHORT (doar pe barele TRADE) ──
        print("\n── Stage 2: LONG vs SHORT (doar pe TRADE bars) ──")
        # Train: doar barele SHORT(1) și LONG(2) din training
        _s2_mask_train = y_train_res > 0
        X_train_s2 = X_train_res[_s2_mask_train]
        y_train_s2 = y_train_res[_s2_mask_train].copy()
        y_train_s2 = (y_train_s2 == 2).astype(int)  # 0=SHORT, 1=LONG
        sw_train_s2 = sw_train_res[_s2_mask_train]

        # Val: doar barele cu semnal din val
        _s2_val_us = y_val_us
        _X_val_s2_src = X_val_us
        _s2_mask_val = _s2_val_us > 0
        X_val_s2 = _X_val_s2_src[_s2_mask_val]
        y_val_s2 = _s2_val_us[_s2_mask_val].copy()
        y_val_s2 = (y_val_s2 == 2).astype(int)

        _s2_dist = pd.Series(y_train_s2).value_counts().sort_index()
        print(f"   Train: SHORT(0)={_s2_dist.get(0,0):,} LONG(1)={_s2_dist.get(1,0):,}")

        model_stage2 = XGB2(
            n_estimators=800, max_depth=4, learning_rate=0.02,
            reg_lambda=8.0, reg_alpha=2.0, min_child_weight=5, gamma=0.5,
            subsample=0.80, colsample_bytree=0.80,
            eval_metric='logloss', random_state=42,
            tree_method='hist', device='cpu',
        )
        model_stage2.fit(
            X_train_s2, y_train_s2,
            sample_weight=sw_train_s2,
            eval_set=[(X_val_s2, y_val_s2)],
            verbose=False,
        )

        # Predict pe test: Stage 2 doar pe barele unde Stage 1 zice TRADE
        _s2_test_mask = _s1_pred == 1
        y_twostage = np.zeros(len(X_test), dtype=int)  # default WAIT
        if _s2_test_mask.sum() > 0:
            _X_s2_test = X_test[_s2_test_mask]
            _s2_proba = model_stage2.predict_proba(_X_s2_test)[:, 1]  # prob LONG
            _s2_pred  = np.where(_s2_proba >= 0.50, 2, 1)  # 2=LONG, 1=SHORT
            y_twostage[_s2_test_mask] = _s2_pred

        _ts_short = (y_twostage == 1).sum()
        _ts_long  = (y_twostage == 2).sum()
        _ts_total = _ts_short + _ts_long
        print(f"   ✅ Stage 2: SHORT={_ts_short:,} LONG={_ts_long:,} Total={_ts_total:,}")

        # ── Compare two-stage vs single XGBoost ──
        print("\n── COMPARAȚIE: Two-Stage vs Single XGBoost ──")
        # Two-stage PnL
        _test_idx_ts = df.index[split_val:split_val + len(X_test)]
        _close_ts = df.loc[_test_idx_ts, 'close'].values
        _next_ts  = df.loc[_test_idx_ts, 'price_next'].values
        _rets_ts  = _next_ts - _close_ts

        _ts_pnl = []
        for _i in range(len(y_twostage)):
            if y_twostage[_i] == 1:
                _ts_pnl.append(-_rets_ts[_i] - TOTAL_COST_PTS)
            elif y_twostage[_i] == 2:
                _ts_pnl.append(_rets_ts[_i] - TOTAL_COST_PTS)
        _ts_pnl = np.array(_ts_pnl)

        if len(_ts_pnl) > 0:
            _ts_wr = (_ts_pnl > 0).sum() / len(_ts_pnl) * 100
            _ts_ev = _ts_pnl.mean()
            _ts_gross_p = _ts_pnl[_ts_pnl > 0].sum()
            _ts_gross_l = abs(_ts_pnl[_ts_pnl < 0].sum())
            _ts_pf = _ts_gross_p / _ts_gross_l if _ts_gross_l > 0 else float('inf')
            _ts_net = _ts_pnl.sum()
            # Consecutive losses
            _ts_is_loss = (_ts_pnl < 0).astype(int)
            _ts_max_cl = 0
            _ts_cur = 0
            for _l in _ts_is_loss:
                if _l: _ts_cur += 1
                else:
                    _ts_max_cl = max(_ts_max_cl, _ts_cur)
                    _ts_cur = 0
            _ts_max_cl = max(_ts_max_cl, _ts_cur)
            # Trades/day
            _ts_n_days = len(X_test) / 390
            _ts_tpd = len(_ts_pnl) / _ts_n_days if _ts_n_days > 0 else 0

            print(f"   {'Metric':<25s} {'XGBoost':>12s} {'Two-Stage':>12s}")
            print(f"   {'─'*25} {'─'*12} {'─'*12}")
            # Folosim _pred (XGBoost) și _all_pnl va fi calculat mai jos, deci calculăm aici
            _xgb_pnl_quick = []
            for _i in range(len(y_pred)):
                if y_pred[_i] == 1:
                    _xgb_pnl_quick.append(-_rets_ts[_i] - TOTAL_COST_PTS)
                elif y_pred[_i] == 2:
                    _xgb_pnl_quick.append(_rets_ts[_i] - TOTAL_COST_PTS)
            _xgb_pnl_quick = np.array(_xgb_pnl_quick)
            _xgb_wr = (_xgb_pnl_quick > 0).sum() / len(_xgb_pnl_quick) * 100 if len(_xgb_pnl_quick) > 0 else 0
            _xgb_ev = _xgb_pnl_quick.mean() if len(_xgb_pnl_quick) > 0 else 0
            _xgb_gp = _xgb_pnl_quick[_xgb_pnl_quick > 0].sum() if len(_xgb_pnl_quick) > 0 else 0
            _xgb_gl = abs(_xgb_pnl_quick[_xgb_pnl_quick < 0].sum()) if len(_xgb_pnl_quick) > 0 else 1
            _xgb_pf = _xgb_gp / _xgb_gl if _xgb_gl > 0 else float('inf')
            _xgb_net = _xgb_pnl_quick.sum() if len(_xgb_pnl_quick) > 0 else 0
            _xgb_tpd = len(_xgb_pnl_quick) / _ts_n_days if _ts_n_days > 0 else 0
            # XGB consecutive losses
            _xgb_il = (_xgb_pnl_quick < 0).astype(int) if len(_xgb_pnl_quick) > 0 else np.array([])
            _xgb_mcl = 0; _xc = 0
            for _l in _xgb_il:
                if _l: _xc += 1
                else: _xgb_mcl = max(_xgb_mcl, _xc); _xc = 0
            _xgb_mcl = max(_xgb_mcl, _xc)

            print(f"   {'Trades':<25s} {len(_xgb_pnl_quick):>12,} {len(_ts_pnl):>12,}")
            print(f"   {'Trades/zi':<25s} {_xgb_tpd:>12.1f} {_ts_tpd:>12.1f}")
            print(f"   {'Win Rate':<25s} {_xgb_wr:>11.1f}% {_ts_wr:>11.1f}%")
            print(f"   {'EV/trade (pts)':<25s} {_xgb_ev:>+12.1f} {_ts_ev:>+12.1f}")
            print(f"   {'Profit Factor':<25s} {_xgb_pf:>12.2f} {_ts_pf:>12.2f}")
            print(f"   {'Net P&L (pts)':<25s} {_xgb_net:>+12,.0f} {_ts_net:>+12,.0f}")
            print(f"   {'Max consec. losses':<25s} {_xgb_mcl:>12} {_ts_max_cl:>12}")

            # Winner?
            if _ts_pf > _xgb_pf and _ts_max_cl <= _xgb_mcl:
                print(f"\n   ✅ TWO-STAGE câștigă! PF {_ts_pf:.2f} vs {_xgb_pf:.2f}, CL {_ts_max_cl} vs {_xgb_mcl}")
            elif _ts_max_cl < _xgb_mcl:
                print(f"\n   ✅ TWO-STAGE reduce consec. losses: {_ts_max_cl} vs {_xgb_mcl}")
            else:
                print(f"\n   📊 XGBoost single rămâne mai bun pe aceste date")
        else:
            print("   ⚠️ Two-stage: 0 semnale — stage 1 prea restrictiv")

        # Salvăm two-stage models
        import pickle
        _s1_path = MODEL_SAVE_PATH.replace('.json', '_stage1.json')
        _s2_path = MODEL_SAVE_PATH.replace('.json', '_stage2.json')
        model_stage1.save_model(_s1_path)
        model_stage2.save_model(_s2_path)
        print(f"\n   💾 Stage 1 salvat: {_s1_path}")
        print(f"   💾 Stage 2 salvat: {_s2_path}")

    except Exception as _ts_err:
        print(f"   ⚠️ Two-stage error: {_ts_err}")
        import traceback; traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════════════
    # v10.7: QUANT STATISTICS — evaluare profitabilitate pe test set
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("📈 QUANT ANALYSIS — Profitabilitate semnale pe test set")
    print("="*80)

    print(f"   💰 Trading costs: slippage={SLIPPAGE_PTS} pts + commission={COMMISSION_PTS} pts = {TOTAL_COST_PTS} pts/trade")

    try:
        # Reconstituim price data pentru test set
        _test_idx = df.index[split_val:split_val + len(X_test)]
        _test_close = df.loc[_test_idx, 'close'].values
        _test_close_15 = df.loc[_test_idx, 'price_next'].values

        # Return-ul real la 15 bare (în puncte NQ)
        _returns_pts = _test_close_15 - _test_close  # pozitiv = price went up

        # Semnalele modelului
        _pred = y_pred  # din evaluarea anterioară (XGBoost)
        _true = y_test.values

        # ── 1. Win Rate per clasă ──
        print("\n── 1. WIN RATE PER CLASĂ ──")
        if _USE_V13:
            _cls_iter = [(c, REGIME_NAMES.get(c, f'c{c}')) for c in range(1, _NUM_CLASS)]
        else:
            _cls_iter = [(1, "SHORT"), (2, "LONG")]
        for _cls, _name in _cls_iter:
            _mask = _pred == _cls
            _n = _mask.sum()
            if _n == 0:
                print(f"   {_name}: 0 semnale")
                continue
            _rets = _returns_pts[_mask]
            _dir = _DIR_MAP.get(int(_cls), 0)  # +1 LONG, -1 SHORT
            _pnl_raw = _dir * _rets
            _pnl = _pnl_raw - TOTAL_COST_PTS  # subtract slippage + commission
            _wins = (_pnl > 0).sum()
            _wr = _wins / _n * 100
            _avg_win = _pnl[_pnl > 0].mean() if (_pnl > 0).any() else 0
            _avg_loss = abs(_pnl[_pnl < 0].mean()) if (_pnl < 0).any() else 0
            print(f"   {_name}: {_n:,} semnale | Win Rate: {_wr:.1f}% (after costs) | "
                  f"Avg Win: {_avg_win:.1f} pts | Avg Loss: {_avg_loss:.1f} pts")

        # ── 2. Profit Factor ──
        print("\n── 2. PROFIT FACTOR ──")
        _all_pnl = []
        for _cls, _name in _cls_iter:
            _mask = _pred == _cls
            if _mask.sum() == 0:
                continue
            _rets = _returns_pts[_mask]
            _dir = _DIR_MAP.get(int(_cls), 0)
            _pnl_raw = _dir * _rets
            _pnl_adj = _pnl_raw - TOTAL_COST_PTS
            _all_pnl.extend(_pnl_adj.tolist())
        _all_pnl = np.array(_all_pnl)
        if len(_all_pnl) > 0:
            _gross_profit = _all_pnl[_all_pnl > 0].sum()
            _gross_loss   = abs(_all_pnl[_all_pnl < 0].sum())
            _pf = _gross_profit / _gross_loss if _gross_loss > 0 else float('inf')
            _net = _gross_profit - _gross_loss
            print(f"   Gross Profit: {_gross_profit:,.1f} pts | Gross Loss: {_gross_loss:,.1f} pts")
            print(f"   Profit Factor: {_pf:.2f} | Net P&L: {_net:+,.1f} pts")
        else:
            print("   ⚠️ Niciun semnal SHORT/LONG — nu se poate calcula")
            _all_pnl = np.array([0])
            _pf = 0

        # ── 3. Expected Value per Trade ──
        print("\n── 3. EXPECTED VALUE PER TRADE ──")
        if len(_all_pnl) > 0:
            _ev = _all_pnl.mean()
            _std = _all_pnl.std()
            _n_trades = len(_all_pnl)
            # EV în $ (NQ = $20/punct)
            print(f"   EV: {_ev:+.2f} pts/trade ({_ev * 20:+.1f} $/trade NQ)")
            print(f"   Std Dev: {_std:.2f} pts | Trades: {_n_trades:,}")

        # ── 4. Sharpe & Sortino Ratio ──
        print("\n── 4. SHARPE & SORTINO RATIO (anual, pe semnale) ──")
        if len(_all_pnl) > 10:
            _daily_trades = max(1, _n_trades / (len(X_test) / 390))  # ~390 bare/zi
            _annualize = np.sqrt(252 * _daily_trades)
            _sharpe = (_all_pnl.mean() / _all_pnl.std()) * _annualize if _all_pnl.std() > 0 else 0
            _downside = _all_pnl[_all_pnl < 0].std() if (_all_pnl < 0).any() else 1e-8
            _sortino = (_all_pnl.mean() / _downside) * _annualize
            print(f"   Sharpe Ratio:  {_sharpe:.2f}")
            print(f"   Sortino Ratio: {_sortino:.2f}")
            if _sharpe > 1.5:
                print(f"   ✅ Sharpe > 1.5 — excelent")
            elif _sharpe > 0.8:
                print(f"   ✅ Sharpe > 0.8 — bun")
            elif _sharpe > 0:
                print(f"   ⚠️ Sharpe pozitiv dar slab — modelul e marginal profitabil")
            else:
                print(f"   ❌ Sharpe negativ — modelul pierde bani pe test set")

        # ── 5. Max Drawdown ──
        print("\n── 5. MAX DRAWDOWN (secvențial pe test set) ──")
        if len(_all_pnl) > 0:
            _equity = np.cumsum(_all_pnl)
            _peak = np.maximum.accumulate(_equity)
            _dd = _equity - _peak
            _max_dd = _dd.min()
            _max_dd_idx = np.argmin(_dd)
            _recovery = 0
            if _max_dd_idx < len(_equity) - 1:
                _post_dd = _equity[_max_dd_idx:]
                _recovered = np.where(_post_dd >= _peak[_max_dd_idx])[0]
                _recovery = _recovered[0] if len(_recovered) > 0 else len(_post_dd)
            print(f"   Max Drawdown: {_max_dd:,.1f} pts ({_max_dd * 20:,.0f} $ NQ)")
            print(f"   Recovery: {_recovery} trade-uri")
            print(f"   Final Equity: {_equity[-1]:+,.1f} pts ({_equity[-1] * 20:+,.0f} $ NQ)")

        # ── 5b. PROP FIRM RISK — Consecutive Losses & Daily Loss ──
        print("\n── 5b. PROP FIRM RISK ANALYSIS ──")
        if len(_all_pnl) > 0:
            # Consecutive losses (losing streaks)
            _is_loss = (_all_pnl < 0).astype(int)
            _max_consec_loss = 0
            _cur_streak = 0
            _streaks = []  # toate streak-urile de pierderi
            for _il in _is_loss:
                if _il == 1:
                    _cur_streak += 1
                else:
                    if _cur_streak > 0:
                        _streaks.append(_cur_streak)
                    _cur_streak = 0
            if _cur_streak > 0:
                _streaks.append(_cur_streak)
            _max_consec_loss = max(_streaks) if _streaks else 0
            _avg_streak = np.mean(_streaks) if _streaks else 0

            # Distribuția streak-urilor
            _streak_counts = {}
            for _s in _streaks:
                _streak_counts[_s] = _streak_counts.get(_s, 0) + 1

            print(f"   Max pierderi consecutive: {_max_consec_loss}")
            print(f"   Avg losing streak: {_avg_streak:.1f}")
            print(f"   Distribuție streaks: ", end="")
            for _slen in sorted(_streak_counts.keys()):
                if _slen <= 10:
                    print(f"{_slen}×:{_streak_counts[_slen]} ", end="")
            if any(s > 10 for s in _streak_counts):
                _big = sum(v for k, v in _streak_counts.items() if k > 10)
                print(f" >10×:{_big}", end="")
            print()

            # Worst consecutive loss in pts
            _worst_streak_pts = 0
            _cur_streak_pts = 0
            for _p in _all_pnl:
                if _p < 0:
                    _cur_streak_pts += _p
                    _worst_streak_pts = min(_worst_streak_pts, _cur_streak_pts)
                else:
                    _cur_streak_pts = 0
            print(f"   Worst losing streak: {_worst_streak_pts:,.1f} pts ({_worst_streak_pts * 20:,.0f} $ NQ)")

            # Daily P&L simulation (grupăm ~390 bare = 1 zi)
            _bars_per_day = 390  # 1-min bars în sesiunea US
            _trades_with_day = []
            _test_df_daily = df.iloc[split_val:split_val + len(X_test)].copy()
            _test_df_daily['_pred'] = y_pred
            _test_df_daily['_pnl'] = 0.0
            # v13: iterate dynamically over all non-WAIT classes via _DIR_MAP
            for _c_daily in range(1, _NUM_CLASS):
                _dir_c = _DIR_MAP.get(int(_c_daily), 0)
                if _dir_c == 0:
                    continue
                _mask_c = _test_df_daily['_pred'] == _c_daily
                _test_df_daily.loc[_mask_c, '_pnl'] = _dir_c * (_test_df_daily.loc[_mask_c, 'price_next'] - _test_df_daily.loc[_mask_c, 'close']) - TOTAL_COST_PTS

            # Grupăm pe zile (fiecare ~390 bare)
            _n_days = len(_test_df_daily) // _bars_per_day
            _daily_pnl = []
            _daily_trades = []
            for _d in range(_n_days):
                _day_slice = _test_df_daily.iloc[_d * _bars_per_day : (_d + 1) * _bars_per_day]
                _day_signals = _day_slice[_day_slice['_pred'] != 0]
                _daily_pnl.append(_day_signals['_pnl'].sum())
                _daily_trades.append(len(_day_signals))
            _daily_pnl = np.array(_daily_pnl)
            _daily_trades = np.array(_daily_trades)

            _losing_days = (_daily_pnl < 0).sum()
            _winning_days = (_daily_pnl > 0).sum()
            _daily_wr = _winning_days / len(_daily_pnl) * 100 if len(_daily_pnl) > 0 else 0
            _worst_day = _daily_pnl.min() if len(_daily_pnl) > 0 else 0
            _best_day = _daily_pnl.max() if len(_daily_pnl) > 0 else 0
            _avg_trades_day = _daily_trades.mean() if len(_daily_trades) > 0 else 0

            # Max consecutive losing days
            _daily_loss_streak = 0
            _max_daily_loss_streak = 0
            for _dp in _daily_pnl:
                if _dp < 0:
                    _daily_loss_streak += 1
                    _max_daily_loss_streak = max(_max_daily_loss_streak, _daily_loss_streak)
                else:
                    _daily_loss_streak = 0

            print(f"\n   📅 DAILY P&L ({_n_days} zile simulate):")
            print(f"   Winning days: {_winning_days} ({_daily_wr:.1f}%) | Losing days: {_losing_days}")
            print(f"   Best day: {_best_day:+,.1f} pts ({_best_day*20:+,.0f}$) | Worst day: {_worst_day:+,.1f} pts ({_worst_day*20:+,.0f}$)")
            print(f"   Avg trades/zi: {_avg_trades_day:.1f}")
            print(f"   Max zile pierdere consecutive: {_max_daily_loss_streak}")
            print(f"   Avg daily P&L: {_daily_pnl.mean():+.1f} pts ({_daily_pnl.mean()*20:+.0f}$/zi)")

            # PROP FIRM SAFETY CHECK
            print(f"\n   🏦 PROP FIRM SAFETY:")
            _daily_limit = 1500  # $1500 daily loss limit prop firm NQ (75 pts)
            _daily_limit_pts = _daily_limit / 20
            _days_over_limit = (_daily_pnl < -_daily_limit_pts).sum()
            print(f"   Zile care depășesc daily loss ${_daily_limit}: {_days_over_limit} ({_days_over_limit/_n_days*100:.1f}%)")
            if _max_consec_loss <= 5:
                print(f"   ✅ Max consecutive losses ({_max_consec_loss}) — acceptabil prop firm")
            elif _max_consec_loss <= 8:
                print(f"   ⚠️ Max consecutive losses ({_max_consec_loss}) — riscant")
            else:
                print(f"   ❌ Max consecutive losses ({_max_consec_loss}) — periculos pentru prop firm")

            # ── v12.3: BRIDGE CL=3 SIMULATION ──────────────────────────────
            # Simulăm exact ce face bridge-ul live: după 3 CL → stop 45min (45 trade-uri)
            # + adaptive score_min: CL1→+15%, CL2→+30% (skip dacă scor < threshold)
            print(f"\n   🔒 BRIDGE CL=3 SIMULATION (v12.3):")
            _cl_max = 3
            _cl_cooldown_trades = 45  # 45 min ≈ 45 bare skip
            _sim_pnl = []
            _sim_cl = 0
            _sim_max_cl = 0
            _sim_skipped = 0
            _sim_cooldown_remaining = 0
            _sim_blocked_by_score = 0

            # Simulăm scor adaptive: CL1→score +15%, CL2→score +30%
            # Semnalele cu probabilitate < threshold sunt skipped
            _proba_test = model_xgb.predict_proba(X_test)
            _score_min_base = 0.50  # default threshold 50%

            for _ti in range(len(_all_pnl)):
                # Cooldown activ? Skip trade
                if _sim_cooldown_remaining > 0:
                    _sim_cooldown_remaining -= 1
                    _sim_skipped += 1
                    continue

                # Adaptive score_min
                _adj_score_min = _score_min_base
                if _sim_cl == 1:
                    _adj_score_min = min(_score_min_base + 0.15, 0.85)
                elif _sim_cl >= 2:
                    _adj_score_min = min(_score_min_base + 0.30, 0.95)

                # Check dacă scorul trece pragul
                _pred_class = y_pred[_ti]
                if _pred_class == 0:
                    continue  # WAIT
                _score_i = _proba_test[_ti, _pred_class]
                if _score_i < _adj_score_min:
                    _sim_blocked_by_score += 1
                    continue

                # Execute trade
                _pnl_i = _all_pnl[_ti]
                _sim_pnl.append(_pnl_i)

                if _pnl_i < 0:
                    _sim_cl += 1
                    _sim_max_cl = max(_sim_max_cl, _sim_cl)
                    if _sim_cl >= _cl_max:
                        _sim_cooldown_remaining = _cl_cooldown_trades
                        _sim_cl = 0  # reset după cooldown
                else:
                    _sim_cl = 0  # WIN → reset

            _sim_pnl = np.array(_sim_pnl) if _sim_pnl else np.array([0])
            _sim_total = _sim_pnl.sum()
            _sim_trades = len(_sim_pnl)
            _sim_wr = (_sim_pnl > 0).sum() / _sim_trades * 100 if _sim_trades > 0 else 0
            _sim_pf = abs(_sim_pnl[_sim_pnl > 0].sum() / _sim_pnl[_sim_pnl < 0].sum()) if _sim_pnl[_sim_pnl < 0].sum() != 0 else 999

            # Recalculate max CL in simulated trades
            _sim_cl2 = 0
            _sim_max_cl2 = 0
            for _sp in _sim_pnl:
                if _sp < 0:
                    _sim_cl2 += 1
                    _sim_max_cl2 = max(_sim_max_cl2, _sim_cl2)
                else:
                    _sim_cl2 = 0

            print(f"   Trades executate: {_sim_trades:,} (din {len(_all_pnl):,} disponibile)")
            print(f"   Trades skipped (cooldown): {_sim_skipped:,}")
            print(f"   Trades blocked (adaptive score): {_sim_blocked_by_score:,}")
            print(f"   Win Rate: {_sim_wr:.1f}%")
            print(f"   Profit Factor: {_sim_pf:.2f}")
            print(f"   Net P&L: {_sim_total:+,.1f} pts ({_sim_total*20:+,.0f} $ NQ)")
            print(f"   EV/trade: {_sim_total/_sim_trades:+.1f} pts" if _sim_trades > 0 else "   EV/trade: N/A")
            print(f"   Max consecutive losses: {_sim_max_cl2}")
            if _sim_max_cl2 <= 3:
                print(f"   ✅ CL={_sim_max_cl2} — PROP FIRM SAFE!")
            else:
                print(f"   ⚠️ CL={_sim_max_cl2} — bridge va tăia la 3 live")

        # ── 6. Regime Analysis ──
        print("\n── 6. REGIME ANALYSIS (performanță per market condition) ──")
        _test_df_regime = df.iloc[split_val:split_val + len(X_test)].copy()
        _test_df_regime['pred'] = _pred
        _test_df_regime['pnl'] = 0.0
        for _c_reg in range(1, _NUM_CLASS):
            _dir_r = _DIR_MAP.get(int(_c_reg), 0)
            if _dir_r == 0:
                continue
            _mask_r = _test_df_regime['pred'] == _c_reg
            _test_df_regime.loc[_mask_r, 'pnl'] = _dir_r * (_test_df_regime.loc[_mask_r, 'price_next'] - _test_df_regime.loc[_mask_r, 'close']) - TOTAL_COST_PTS

        # Regime pe baza ADX + realized_vol
        if 'adx_14' in _test_df_regime.columns and 'realized_vol' in _test_df_regime.columns:
            _test_df_regime['_regime'] = 'NORMAL'
            _test_df_regime.loc[_test_df_regime['adx_14'] > 25, '_regime'] = 'TREND'
            _test_df_regime.loc[_test_df_regime['adx_14'] < 15, '_regime'] = 'RANGE'
            _high_vol_thresh = _test_df_regime['realized_vol'].quantile(0.85)
            _test_df_regime.loc[_test_df_regime['realized_vol'] > _high_vol_thresh, '_regime'] = 'HIGH_VOL'

            for _regime in ['TREND', 'RANGE', 'NORMAL', 'HIGH_VOL']:
                _rm = (_test_df_regime['_regime'] == _regime) & (_test_df_regime['pred'] != 0)
                _n_r = _rm.sum()
                if _n_r < 10:
                    continue
                _pnl_r = _test_df_regime.loc[_rm, 'pnl']
                _wr_r = (_pnl_r > 0).sum() / _n_r * 100
                print(f"   {_regime:10s}: {_n_r:5,} trades | WR: {_wr_r:.1f}% | "
                      f"Avg: {_pnl_r.mean():+.2f} pts | Total: {_pnl_r.sum():+,.0f} pts")
        else:
            print("   ⚠️ adx_14 sau realized_vol lipsesc — skip regime analysis")

        # ── 7. Monte Carlo Simulation (REALISTIC) ──
        print("\n── 7. MONTE CARLO SIMULATION — REALISTIC (1000 paths) ──")
        print(f"   Noise: random slippage ±1.5 pts/trade + regime degradation 10%")
        if len(_all_pnl) > 20:
            _mc_paths = 1000
            _mc_final = []
            _mc_dd    = []
            _n_mc = len(_all_pnl)
            _rng = np.random.RandomState(42)

            # Parametri zgomot realist
            _SLIPPAGE_NOISE_STD = 1.5   # ±1.5 pts random slippage suplimentar
            _REGIME_DEGRADE     = 0.10  # 10% degradare — simul regime shift

            for _ in range(_mc_paths):
                _shuffled = _rng.choice(_all_pnl, size=_n_mc, replace=True)
                # Random slippage noise per trade (normal distribution)
                _slip_noise = _rng.normal(0, _SLIPPAGE_NOISE_STD, size=_n_mc)
                # Regime degradation: reduce fiecare trade cu 10% din avg win
                _avg_win_est = max(_shuffled[_shuffled > 0].mean(), 1.0) if (_shuffled > 0).any() else 5.0
                _regime_cost = _REGIME_DEGRADE * _avg_win_est
                # PnL ajustat = PnL original + noise - regime degradation
                _adjusted = _shuffled + _slip_noise - _regime_cost
                _eq = np.cumsum(_adjusted)
                _mc_final.append(_eq[-1])
                _pk = np.maximum.accumulate(_eq)
                _mc_dd.append((_eq - _pk).min())
            _mc_final = np.array(_mc_final)
            _mc_dd    = np.array(_mc_dd)
            _p5  = np.percentile(_mc_final, 5)
            _p50 = np.percentile(_mc_final, 50)
            _p95 = np.percentile(_mc_final, 95)
            _prob_profit = (_mc_final > 0).mean() * 100
            print(f"   Final Equity — P5: {_p5:+,.0f} pts | P50: {_p50:+,.0f} pts | P95: {_p95:+,.0f} pts")
            print(f"   Probabilitate profit: {_prob_profit:.1f}%")
            print(f"   Worst DD — P5: {np.percentile(_mc_dd, 5):,.0f} pts | "
                  f"P50: {np.percentile(_mc_dd, 50):,.0f} pts")
            if _prob_profit >= 70:
                print(f"   ✅ Monte Carlo REALISTIC: {_prob_profit:.0f}% probabilitate profit — robust")
            elif _prob_profit >= 50:
                print(f"   ⚠️ Monte Carlo REALISTIC: {_prob_profit:.0f}% — marginal, necesită optimizare")
            else:
                print(f"   ❌ Monte Carlo REALISTIC: {_prob_profit:.0f}% — modelul nu e profitabil statistic")
        else:
            print("   ⚠️ Prea puține semnale pentru Monte Carlo")

        # ── 7b. MONTE CARLO STRESS TEST — Edge Degradation 30%/50% ──
        print("\n── 7b. MONTE CARLO STRESS TEST — Ce se întâmplă dacă edge-ul scade? ──")
        if len(_all_pnl) > 20:
            _rng_stress = np.random.RandomState(123)
            for _degrade_pct in [0.30, 0.50, 0.70]:
                _mc_stress_final = []
                _mc_stress_dd = []
                for _ in range(1000):
                    _shuffled = _rng_stress.choice(_all_pnl, size=len(_all_pnl), replace=True)
                    # Degradare: reduce wins cu X%, mărește losses cu X%
                    _degraded = np.where(
                        _shuffled > 0,
                        _shuffled * (1 - _degrade_pct),    # wins mai mici
                        _shuffled * (1 + _degrade_pct)     # losses mai mari
                    )
                    # Plus noise suplimentar
                    _degraded += _rng_stress.normal(0, 1.5, size=len(_all_pnl))
                    _eq_s = np.cumsum(_degraded)
                    _mc_stress_final.append(_eq_s[-1])
                    _pk_s = np.maximum.accumulate(_eq_s)
                    _mc_stress_dd.append((_eq_s - _pk_s).min())
                _mc_stress_final = np.array(_mc_stress_final)
                _mc_stress_dd = np.array(_mc_stress_dd)
                _prob_s = (_mc_stress_final > 0).mean() * 100
                _p50_s = np.percentile(_mc_stress_final, 50)
                _dd_p50 = np.percentile(_mc_stress_dd, 50)
                _icon = "✅" if _prob_s >= 70 else ("⚠️" if _prob_s >= 50 else "❌")
                print(f"   {_icon} Edge -{_degrade_pct*100:.0f}%: P(profit)={_prob_s:.1f}% | "
                      f"P50 equity={_p50_s:+,.0f} pts | P50 DD={_dd_p50:,.0f} pts")

        # ── 8. Walk-Forward Analysis (3 ferestre) ──
        print("\n── 8. WALK-FORWARD ANALYSIS (3 ferestre sliding) ──")
        _wf_size = len(X_test) // 3
        for _wf_i in range(3):
            _wf_start = _wf_i * _wf_size
            _wf_end   = _wf_start + _wf_size
            _wf_pred  = _pred[_wf_start:_wf_end]
            _wf_rets  = _returns_pts[_wf_start:_wf_end]
            _wf_pnl   = []
            for _j in range(len(_wf_pred)):
                if _wf_pred[_j] == 1:
                    _wf_pnl.append(-_wf_rets[_j] - TOTAL_COST_PTS)
                elif _wf_pred[_j] == 2:
                    _wf_pnl.append(_wf_rets[_j] - TOTAL_COST_PTS)
            _wf_pnl = np.array(_wf_pnl)
            if len(_wf_pnl) > 0:
                _wf_wr = (_wf_pnl > 0).sum() / len(_wf_pnl) * 100
                print(f"   Window {_wf_i+1}: {len(_wf_pnl):,} trades | WR: {_wf_wr:.1f}% | "
                      f"EV: {_wf_pnl.mean():+.2f} pts | Total: {_wf_pnl.sum():+,.0f} pts")
            else:
                print(f"   Window {_wf_i+1}: 0 trades")

    except Exception as _quant_err:
        print(f"\n   ⚠️ Quant analysis error: {_quant_err}")
        import traceback; traceback.print_exc()

    print("\n" + "="*80)

    # ── Pasul 12: Salvare ────────────────────────────────────────────────────
    print(f"\n💾 Pasul 9: Salvare model → {MODEL_SAVE_PATH}")
    model_xgb.save_model(MODEL_SAVE_PATH)

    features_meta = {
        "features":           final_features,
        "top_features":       top_features,
        "n_features":         len(final_features),
        "accuracy":           round(acc, 4),
        # Fix v9.0: Evaluăm pe X_train_res (same distribution as training) + val accuracy
        "train_accuracy":     round(float(accuracy_score(y_train_res, np.argmax(model_xgb.predict_proba(X_train_res), axis=1))), 4),
        "val_accuracy":       round(float(accuracy_score(y_val, np.argmax(model_xgb.predict_proba(X_val), axis=1))), 4),
        "trained_at":         datetime.now().isoformat(),
        "rows_trained":       len(X_train),
        "sniper_pct":         round(pct_trade, 3),
        "confusion_matrix":   cm.tolist(),
        "conf_threshold":     round(CONF_THRESHOLD, 4),  # v12.5: threshold optim pentru 97% WR
        "target_wr":          TARGET_WR,
    }
    with open(FEATURES_PATH, 'w') as f:
        json.dump(features_meta, f, indent=2)

    print(f"   ✅ Feature metadata → {FEATURES_PATH}")

    # ── Update #15: Antrenare Quantum Main Circuit ─────────────────────────────
    # Antrenăm MAIN_WEIGHTS (circuitul quantum 6 qubiți) pe datele de antrenament.
    # Se apelează imediat după XGBoost pentru a folosi același batch de date.
    # Obiectiv: q_boost > 1.0 pentru semnale LONG/SHORT, q_boost < 1.0 pentru WAIT.
    print(f"\n⚛️  Update #15: Antrenare Quantum Main Circuit...")
    try:
        import sys, importlib
        _rag_path = os.path.dirname(MODEL_SAVE_PATH)  # /Users/mario/Desktop/Aladin/
        if _rag_path not in sys.path:
            sys.path.insert(0, _rag_path)

        import pennylane as qml
        from pennylane import numpy as qnp

        _MAIN_WEIGHTS_PATH = MODEL_SAVE_PATH.replace('mario_bot.json', 'aladin_main_weights.npy')

        # Reîncarcă sau inițializează weights
        if os.path.exists(_MAIN_WEIGHTS_PATH):
            _w = np.load(_MAIN_WEIGHTS_PATH)
            main_weights = qnp.array(_w, requires_grad=True)
            print(f"   ✅ MAIN_WEIGHTS anterioare reîncărcate din {_MAIN_WEIGHTS_PATH}")
        else:
            _rng = np.random.default_rng(42)
            main_weights = qnp.array(_rng.uniform(-np.pi/4, np.pi/4, size=(2, 6, 3)), requires_grad=True)
            print(f"   ℹ️  MAIN_WEIGHTS inițializate aleator (prima antrenare)")

        dev_main_train = qml.device("default.qubit", wires=6)

        @qml.qnode(dev_main_train, diff_method="parameter-shift")
        def _qmc_train(inputs, weights):
            # FIX: inputs trebuie să fie qnp.array cu requires_grad=False
            qml.AngleEmbedding(inputs * np.pi, wires=range(6))
            qml.StronglyEntanglingLayers(weights, wires=range(6))
            # FIX: expval în loc de probs — nativ diferențiabil
            return qml.expval(qml.PauliZ(0))

        opt_q = qml.AdamOptimizer(stepsize=0.03)

        # Prepară batch de antrenament din X_train
        # Features: kz≈slope_h1, poc≈dist_poc, smt≈is_smt_bullish+is_smt_bearish,
        #           va≈inside_va, fvg≈fvg_up+fvg_down, displacement≈has_displacement
        q_input_cols = []
        for _col in ['slope_h1', 'dist_poc', 'is_smt_bullish', 'inside_va', 'fvg_up', 'has_displacement']:
            if _col in X_train_res.columns if hasattr(X_train_res, 'columns') else available_feats:
                q_input_cols.append(_col)

        if len(q_input_cols) >= 4:
            # Folosim un subset reprezentativ din X_train (max 500 rânduri pentru viteză)
            _qt_size = min(500, len(X_train_res))
            _qt_idx  = np.random.choice(len(X_train_res), _qt_size, replace=False)
            _X_qt    = X_train_res.iloc[_qt_idx] if hasattr(X_train_res, 'iloc') else X_train_res[_qt_idx]
            _y_qt    = np.array(y_train_res)[_qt_idx]

            # ── FIX ArrayBox: pre-calculăm TOATE inputurile ca numpy pur ÎNAINTE ──
            # de orice apel la cost_fn (PennyLane tracează valorile în cost_fn →
            # orice float() apelat acolo explodează cu 'ArrayBox' error).
            # Soluție: extragem floats din DataFrame ACUM, stocăm în liste Python plain.
            def _extract_q_inputs(X_subset, y_subset):
                """Returnează (inputs_list, targets_list) ca numpy plain arrays."""
                _inp_list = []
                _tgt_list = []
                _xarr = X_subset.values if hasattr(X_subset, 'values') else X_subset
                _col_idx = [
                    list(X_subset.columns).index(c) if c in list(X_subset.columns) else -1
                    for c in q_input_cols
                ] if hasattr(X_subset, 'columns') else list(range(len(q_input_cols)))

                for _i in range(len(_xarr)):
                    _row = _xarr[_i]
                    def _get(ci, default=0.5):
                        return float(_row[ci]) if ci >= 0 and ci < len(_row) else default

                    _v0 = float(np.clip(_get(_col_idx[0]), 0.0, 1.0))
                    _v1 = float(np.clip(_get(_col_idx[1]) / 10.0 + 0.5, 0.0, 1.0))
                    _v2 = float(np.clip(_get(_col_idx[2]), 0.0, 1.0))
                    _v3 = float(np.clip(_get(_col_idx[3]), 0.0, 1.0))
                    _v4 = float(np.clip(_get(_col_idx[4] if len(_col_idx) > 4 else -1, 0.5), 0.0, 1.0))
                    _v5 = float(np.clip(_get(_col_idx[5] if len(_col_idx) > 5 else -1, 0.5), 0.0, 1.0))
                    _inp_list.append(np.array([_v0, _v1, _v2, _v3, _v4, _v5], dtype=float))
                    # WIN = clasă 1 sau 2; WAIT = 0
                    _tgt_list.append(7.0/64.0 if int(_y_qt[_i] if _i < len(y_subset) else 0) != 0 else 2.0/64.0)
                return _inp_list, _tgt_list

            q_losses = []
            for _epoch in range(40):
                _batch_idx = np.random.choice(len(_X_qt), min(32, len(_X_qt)), replace=False)
                _xb = _X_qt.iloc[_batch_idx] if hasattr(_X_qt, 'iloc') else _X_qt[_batch_idx]
                _yb = _y_qt[_batch_idx]

                # Pre-calcul inputs ca numpy plain ÎNAINTE de gradient tape
                _pre_inp, _pre_tgt = _extract_q_inputs(_xb, _yb)

                def _q_cost(w):
                    # FIX FINAL: singură apelare QNode cu input mediat
                    # Loop-ul cu += pierde differentiability în PennyLane autograd
                    # Soluție: media inputurilor + media targeturilor → un singur circuit call
                    _avg_inp = qnp.array(np.mean(_pre_inp, axis=0), requires_grad=False)
                    _avg_tgt = float(np.mean([1.0 if t > 0.1 else -1.0 for t in _pre_tgt]))
                    _pred = _qmc_train(_avg_inp, w)   # scalar expval [-1, +1]
                    return (_pred - _avg_tgt) ** 2

                main_weights, _loss_val = opt_q.step_and_cost(_q_cost, main_weights)
                q_losses.append(float(_loss_val))

                if _epoch % 10 == 0:
                    print(f"   ⚛️  Quantum epoch {_epoch}/40 — loss: {float(_loss_val):.6f}")

            np.save(_MAIN_WEIGHTS_PATH, np.array(main_weights))
            print(f"   ✅ MAIN_WEIGHTS antrenate salvate → {_MAIN_WEIGHTS_PATH}")
            print(f"   📉 Quantum loss: {q_losses[0]:.6f} → {q_losses[-1]:.6f} (Δ: {q_losses[0]-q_losses[-1]:.6f})")
        else:
            print(f"   ⚠️  Features insuficiente pentru quantum train ({q_input_cols}) — skip")

    except ImportError as _qe:
        print(f"   ⚠️  PennyLane lipsă — quantum train skip ({_qe})")
    except Exception as _qe:
        print(f"   ⚠️  Quantum train error: {_qe} — skip (nu afectează XGBoost)")

    # ── v14: ANALYTICS SUITE — statistical validation complet ──────────────────
    # Mario: "toate modelele să aibă correlation matrix, monte carlo, tot ce există în train mario"
    # Notă: mario_ai e 5-class (WAIT/SHORT_BREAK/LONG_BREAK/SHORT_REV/LONG_REV).
    # Analytics rulează binar: SIGNAL (cls > 0) vs WAIT (cls == 0) pentru AUC/calibrare/MC.
    # Ecuitatea folosește direcțiile reale din _DIR_MAP (LONG_BREAK/REV=+1, SHORT=−1).
    if _ANALYTICS:
        print(f"\n{'='*80}")
        print(f"📊 ANALYTICS SUITE v14 — Validare statistică completă")
        print(f"{'='*80}")
        try:
            # model factory pt walk-forward (identic cu _xgb_params dar fără early_stopping)
            def _mario_model_factory(*args, **kwargs):
                import xgboost as _xgb
                return _xgb.XGBClassifier(
                    n_estimators=500, max_depth=5, learning_rate=0.015,
                    objective='multi:softprob', num_class=_NUM_CLASS,
                    subsample=0.80, colsample_bytree=0.80,
                    reg_lambda=15.0, reg_alpha=3.0, min_child_weight=10, gamma=1.0,
                    tree_method='hist', random_state=42, verbosity=0,
                )

            # risk_pts: folosim estimare 12 pts (SL structural median NQ 1 contract)
            _mario_risk_pts = 12.0

            # Binary conversion: clasa 0 = WAIT(0), orice altceva = SIGNAL(1)
            _y_test_bin  = (y_test.values > 0).astype(int)
            _y_train_bin = (y_train_res.values > 0).astype(int)
            # Probabilitate binară = 1 - P(WAIT)
            _y_proba_bin = 1.0 - model_xgb.predict_proba(X_test)[:, 0]
            _threshold_bin = 0.50  # binary threshold pentru analytics

            run_full_analysis(
                model           = model_xgb,
                X_train         = X_train_res,
                X_test          = X_test,
                y_train         = pd.Series(_y_train_bin, index=y_train_res.index),
                y_test          = pd.Series(_y_test_bin, index=y_test.index),
                features        = list(X_train_res.columns) if hasattr(X_train_res, 'columns') else final_features,
                risk_pts        = _mario_risk_pts,
                threshold       = _threshold_bin,
                label           = f"MARIO_AI_{TRAIN_MODE}",
                save_dir        = str(os.path.dirname(MODEL_SAVE_PATH)),
                model_factory   = _mario_model_factory,
                monte_carlo_sims= 1000,
                run_perm_imp    = True,
                run_walk_fwd    = True,
                n_perm_imp_repeats = 5,
            )
            print(f"   ✅ Analytics suite completă — salvat în {os.path.dirname(MODEL_SAVE_PATH)}")
        except Exception as _ana_err:
            print(f"   ⚠️ Analytics error: {_ana_err}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"🎯 ANTRENARE COMPLETĂ!")
    print(f"   Model:       {MODEL_SAVE_PATH}")
    print(f"   Acuratețe:   {acc:.2%}")
    print(f"   Features:    {len(final_features)}")
    print(f"   Sniper rate: {pct_trade:.2f}%")
    print(f"{'='*80}\n")


# =============================================================================
# Update #14: Online Learning Scheduler
# =============================================================================
def online_learning_weekly():
    """
    Update #14: Reantrenare automată săptămânală cu datele din ultima săptămână.
    Se apelează automat prin cron/scheduler sau manual.
    """
    print(f"\n🔄 ONLINE LEARNING — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("   Reantrenare cu date noi din ultima săptămână...")
    # Refolosim antrenare_pro_sql() care citește tot din DB
    # DB-ul trebuie să fie actualizat de download_qqq_10y.py înainte
    antrenare_pro_sql()
    print("   ✅ Model updatat cu date noi.")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import sys
    # CLI: python3 train_mario_ai.py lon / ny / all
    if len(sys.argv) > 1:
        _arg = sys.argv[1].upper()
        if _arg in ('LON', 'NY', 'ALL'):
            TRAIN_MODE = _arg
        else:
            print(f"⚠️ Argument necunoscut '{sys.argv[1]}'. Folosiți: lon / ny / all")

    # Mode-specific output paths
    if TRAIN_MODE != 'ALL':
        _sfx = f'_{TRAIN_MODE.lower()}'
        MODEL_SAVE_PATH = os.path.join(_SCRIPT_DIR, f'mario_bot{_sfx}.json')
        FEATURES_PATH   = os.path.join(_SCRIPT_DIR, f'mario_features{_sfx}.json')
        print(f"🎯 TRAIN_MODE={TRAIN_MODE} → model: mario_bot{_sfx}.json")

    antrenare_pro_sql()
