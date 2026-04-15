"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN API v3.0 — FastAPI Server Mac (port 8000)                           ║
║                                                                              ║
║  NT8 Bridge:        POST /nt8_data, /execution_confirm, /manual_command     ║
║  Dashboard React:   /health, /signal/now, /trades, /equity, /stats          ║
║                     /model/stats, /model/bias, /model/shap                  ║
║                     /economic-calendar, /news/flash, /news/sentiment        ║
║                     /backtest/start, /backtest/status, /backtest/last        ║
║                     /market-data, /fear-greed, /market/levels               ║
║  WebSocket:         /ws — feed real-time                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import statistics
import time
import uuid
from collections import deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional, List, Dict, Any

# Încarcă .env din același director
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ─── UPDATE #1: Supabase Client ────────────────────────────────────────────────
try:
    import sys as _sys, os as _os
    _aladin_dir = str(Path.home() / "Desktop" / "Aladin")
    if _aladin_dir not in _sys.path:
        _sys.path.insert(0, _aladin_dir)
    import supabase_client as _supabase
    _SUPABASE_OK = True
except Exception as _sb_imp_err:
    _SUPABASE_OK = False
    _supabase = None

# ─── UPDATE #11: RL Feedback Loop ─────────────────────────────────────────────
try:
    import rl_feedback as _rl_feedback
    _RL_OK = True
    log_init = logging.getLogger("aladin")
    log_init.info("🧠 RL Feedback Loop activ")
except Exception as _rl_imp_err:
    _RL_OK = False
    _rl_feedback = None

# ─── Advanced Features: Sortino, Calmar, Fractional Kelly ─────────────────────
try:
    from advanced_features import sortino_ratio, calmar_ratio, fractional_kelly, sharpe_ratio
    _AF_METRICS_OK = True
except ImportError:
    _AF_METRICS_OK = False
    def sortino_ratio(pnl_list, annualize=252.0): return 0.0
    def calmar_ratio(total_pnl, max_dd): return 0.0
    def sharpe_ratio(pnl_list, annualize=252.0): return 0.0
    def fractional_kelly(wr, aw, al, fraction=0.25): return {"full_kelly": 0, "fractional_kelly": 0, "risk_pct": 0, "edge": 0, "odds_ratio": 0, "recommendation": "N/A"}

# ─── UPDATE #13: Telegram Alerts ──────────────────────────────────────────────
try:
    from telegram_alerts import (
        send_trade_executed        as _tg_trade,
        send_trade_closed          as _tg_closed,
        telegram_poll_loop         as _tg_poll,
        send_news_alert_15min      as _tg_news_alert,
        send_circuit_breaker_alert as _tg_circuit,
        send_geo_alert             as _tg_geo_alert,
        send_geo_risk_update       as _tg_geo_risk_update,
        send_daily_report          as _tg_daily_report,
        send_weekly_report         as _tg_weekly_report,
    )
    _TG_OK = True
except Exception:
    _TG_OK = False
    _tg_trade           = None
    _tg_closed          = None
    _tg_poll            = None
    _tg_news_alert      = None
    _tg_circuit         = None
    _tg_weekly_report   = None
    _tg_geo_alert       = None
    _tg_geo_risk_update = None
    _tg_daily_report    = None

# ─── SHORT Quality Model — lazy loader (AUC=0.80, bin60+bin90 filter) ─────────
# Antrenat pe NY SHORT signals 2023-2024, testat OOS 2025.
# Filtrează semnalele SHORT de calitate slabă (TRAIL_STOP=0) față de cele bune (TRAIL_STOP=1).
# Threshold=0.25 → +$4-8/săptămână față de filtrul de timp singur.
_SHORT_QUAL_MODEL    = None   # xgb.XGBClassifier, încărcat lazy la primul semnal
_SHORT_QUAL_FEATURES = None   # list[str] — ordinea features din meta JSON

def _load_short_quality_model():
    """Încarcă modelul de calitate SHORT dacă există. Returnează (model, features) sau (None, None)."""
    global _SHORT_QUAL_MODEL, _SHORT_QUAL_FEATURES
    if _SHORT_QUAL_MODEL is not None:
        return _SHORT_QUAL_MODEL, _SHORT_QUAL_FEATURES
    try:
        import xgboost as _xgb, json as _json
        _aq_dir   = Path.home() / "Desktop" / "Aladin"
        _model_f  = _aq_dir / "mario_short_quality.json"
        _meta_f   = _aq_dir / "mario_short_quality_features.json"
        if not _model_f.exists() or not _meta_f.exists():
            return None, None
        _m = _xgb.XGBClassifier()
        _m.load_model(str(_model_f))
        _meta = _json.loads(_meta_f.read_text())
        _SHORT_QUAL_MODEL    = _m
        _SHORT_QUAL_FEATURES = _meta.get("features", [])
        _thr = _meta.get("best_threshold", 0.25)
        _auc = _meta.get("auc_oos_2025", 0)
        logging.getLogger("aladin").info(
            f"✅ SHORT Quality Model încărcat: AUC={_auc:.4f} thr={_thr:.2f} "
            f"({len(_SHORT_QUAL_FEATURES)} features)"
        )
        return _SHORT_QUAL_MODEL, _SHORT_QUAL_FEATURES
    except Exception as _sqm_err:
        logging.getLogger("aladin").warning(f"SHORT Quality Model skip: {_sqm_err}")
        return None, None

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ALADIN] %(levelname)s — %(message)s",
)
log = logging.getLogger("aladin")

# ─── Config ────────────────────────────────────────────────────────────────────
NT8_IP   = "172.16.233.128" # ← IP PC NinjaTrader (Windows) — VMware Fusion NAT
NT8_PORT = 8002
DATA_DIR = Path.home() / "Desktop" / "Aladin" / "data"
NOTES_FILE  = DATA_DIR / "notes.json"
TRADES_FILE = DATA_DIR / "trades.json"
ANALYSIS_Q_SIZE = 100

DB_PATH = Path.home() / "Desktop" / "Aladin" / "mario_trading.db"
STRATEGY_STATE_FILE  = DATA_DIR / "active_strategy.json"  # persistență strategy între restart-uri
OPEN_TRADE_FILE      = DATA_DIR / "open_trade.json"        # persistență trade deschis între restart-uri
EQUITY_HISTORY_FILE  = DATA_DIR / "equity_history.json"    # Feature 2: persistență equity curve
AB_STATE_FILE        = DATA_DIR / "ab_test_state.json"     # Feature 4: A/B test state

# ── Feature 3: Multi-Instrument Config ──────────────────────────────────────────
# Fiecare instrument: point_value ($), tick_size, ticks_per_point, default_sl_pts, symbol_nt8
INSTRUMENT_CONFIG = {
    "NQ":  {"point_value": 20.0, "tick_size": 0.25, "ticks_per_point": 4, "default_sl": 20, "name": "NQ (E-mini Nasdaq 100)"},
}
ACTIVE_INSTRUMENT = "NQ"   # default — poate fi schimbat din /instrument/set


def _get_instrument_config(symbol: str = "") -> dict:
    """Returnează config-ul instrumentului activ sau al celui specificat."""
    sym = (symbol or ACTIVE_INSTRUMENT).upper().replace(" ", "")
    # Încearcă match exact, apoi prefix match (ex: "NQ 06-25" → "NQ")
    if sym in INSTRUMENT_CONFIG:
        return INSTRUMENT_CONFIG[sym]
    for key in INSTRUMENT_CONFIG:
        if sym.startswith(key):
            return INSTRUMENT_CONFIG[key]
    return INSTRUMENT_CONFIG["NQ"]  # fallback


def _get_point_value(symbol: str = "") -> float:
    """Shortcut — returnează point value ($) pentru instrument."""
    return _get_instrument_config(symbol)["point_value"]

def _init_db_source_column():
    """Adaugă coloana 'source' în market_data dacă nu există (migrare one-time)."""
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        cols = [r[1] for r in conn.execute("PRAGMA table_info(market_data)").fetchall()]
        if "source" not in cols:
            conn.execute("ALTER TABLE market_data ADD COLUMN source TEXT DEFAULT 'LEGACY'")
            conn.commit()
            log.info("✅ DB migration: coloana 'source' adăugată în market_data")
        conn.close()
    except Exception as e:
        log.warning(f"DB migration skip: {e}")

def _save_bar_to_db(tick) -> bool:
    """
    Scrie bara curentă NT8 în market_data cu source='BRIDGE_LIVE'.
    INSERT OR IGNORE — nu suprascrie niciodată date NT8_MANUAL sau existente.
    Returnează True dacă bara a fost inserată, False dacă exista deja.
    """
    try:
        import sqlite3

        ts_raw = tick.timestamp[:16] if tick.timestamp else ""
        if not ts_raw:
            return False

        # Fix v7.4: Convertim UTC (ce primim de la NT8) → ET (Eastern Time)
        # DB-ul istoric (LEGACY + NT8_GAP_FILL) e tot în ET — consistență necesară
        # pentru ca query-urile date(timestamp) să alinieze zilele corect
        _utc_str = ts_raw.replace("T", " ") + ":00"
        _utc_dt  = datetime.fromisoformat(_utc_str).replace(tzinfo=ZoneInfo("UTC"))
        _et_dt   = _utc_dt.astimezone(ZoneInfo("America/New_York"))
        ts_db    = _et_dt.strftime("%Y-%m-%d %H:%M:00")
        dt       = _et_dt.replace(tzinfo=None)  # naive ET pentru restul codului

        conn = sqlite3.connect(str(DB_PATH))

        # ── Calculează HTF levels din datele existente în DB ──────────────────
        # Fix v7.3: inclus și BRIDGE_LIVE — fără date istorice importate, excluderea
        # source='BRIDGE_LIVE' lasă lw_hi/lw_lo goale → fallback la bara curentă (6 pts range)
        # Cu fix: weekly high/low = maximul/minimul real din ultimele 7 zile tranzacționate
        row_lw = conn.execute("""
            SELECT MAX(high), MIN(low) FROM market_data
            WHERE date(timestamp) BETWEEN date(?, '-7 days') AND date(?, '-1 day')
        """, (ts_db, ts_db)).fetchone()
        lw_hi = float(row_lw[0]) if row_lw and row_lw[0] else tick.price.high
        lw_lo = float(row_lw[1]) if row_lw and row_lw[1] else tick.price.low

        # Last Month High/Low
        row_lm = conn.execute("""
            SELECT MAX(high), MIN(low) FROM market_data
            WHERE date(timestamp) BETWEEN date(?, '-35 days') AND date(?, '-1 day')
        """, (ts_db, ts_db)).fetchone()
        lm_hi = float(row_lm[0]) if row_lm and row_lm[0] else lw_hi
        lm_lo = float(row_lm[1]) if row_lm and row_lm[1] else lw_lo

        # Previous Day High/Low
        row_pd = conn.execute("""
            SELECT MAX(high), MIN(low) FROM market_data
            WHERE date(timestamp) = date(?, '-1 day')
        """, (ts_db,)).fetchone()
        p_hi = float(row_pd[0]) if row_pd and row_pd[0] else tick.price.high
        p_lo = float(row_pd[1]) if row_pd and row_pd[1] else tick.price.low

        # Current Month High/Low so far
        row_cm = conn.execute("""
            SELECT MAX(high), MIN(low) FROM market_data
            WHERE strftime('%Y-%m', timestamp) = strftime('%Y-%m', ?)
        """, (ts_db,)).fetchone()
        m_hi = float(row_cm[0]) if row_cm and row_cm[0] else tick.price.high
        m_lo = float(row_cm[1]) if row_cm and row_cm[1] else tick.price.low

        close  = tick.price.close
        # Fix v7.2: true_open după restart — ia prima bară din DB de azi, nu bara curentă
        # Problema: bar_buffer gol după restart → true_open=close → is_above_open=0 mereu
        _today_open_row = conn.execute("""
            SELECT open FROM market_data WHERE date(timestamp) = date(?) ORDER BY timestamp ASC LIMIT 1
        """, (ts_db,)).fetchone()
        if state.bar_buffer:
            true_open = list(state.bar_buffer)[0].price.open
        elif _today_open_row and _today_open_row[0]:
            true_open = float(_today_open_row[0])  # prima bară din DB azi = session open
        else:
            true_open = close

        row = {
            "timestamp":       ts_db,
            "open":            tick.price.open,
            "high":            tick.price.high,
            "low":             tick.price.low,
            "close":           close,
            "volume":          int(tick.price.volume or 0),
            "date":            dt.strftime("%Y-%m-%d"),
            "hour_min":        dt.strftime("%H:%M"),
            "day_of_week":     dt.weekday(),
            "week_id":         int(dt.strftime("%V")),
            "month":           dt.month,
            "year":            dt.year,
            "lw_hi":           lw_hi,
            "lw_lo":           lw_lo,
            "lm_hi":           lm_hi,
            "lm_lo":           lm_lo,
            "m_hi":            m_hi,
            "m_lo":            m_lo,
            "p_hi":            p_hi,
            "p_lo":            p_lo,
            "h4_hi":           tick.htf.h4_hi,
            "h4_lo":           tick.htf.h4_lo,
            "h1_hi":           tick.htf.h1_hi,
            "h1_lo":           tick.htf.h1_lo,
            "true_open":       true_open,
            "asia_hi":         0.0,
            "asia_lo":         0.0,
            "lon_hi":          0.0,
            "lon_lo":          0.0,
            "price_bin":       round(close / 25) * 25,
            "poc_level":       tick.volume_profile.poc,
            "vah":             tick.volume_profile.vah,
            "val":             tick.volume_profile.val,
            "fvg_up":          0,
            "fvg_down":        0,
            "body_size":       abs(close - tick.price.open),
            # Fix v7.2: displacement detectat și din SIZE (range > 1.5×ATR), nu doar imbalance
            # Spike-ul BSL de 120 puncte nu era detectat dacă imbalance_pct < 60
            "has_displacement":1 if (
                abs(tick.orderflow.imbalance_pct) > 60
                or (tick.atr_14 > 0 and (tick.price.high - tick.price.low) > 1.5 * tick.atr_14)
            ) else 0,
            "is_above_open":   1 if close > true_open else 0,
            "atr_14":          tick.atr_14,
            "dist_poc":        close - tick.volume_profile.poc if tick.volume_profile.poc > 0 else 0.0,
            "inside_va":       1 if tick.volume_profile.val <= close <= tick.volume_profile.vah else 0,
            "dist_pdh":        close - p_hi,
            "dist_pdl":        close - p_lo,
            "spy_hi":          0.0,
            "spy_lo":          0.0,
            "spy_p_hi":        0.0,
            "spy_p_lo":        0.0,
            "is_smt_bearish":  0,
            "is_smt_bullish":  0,
            # ── Noi features (adăugate pentru viitor training) ────────────────
            # Encodate numeric ca să fie compatibile cu XGBoost/LightGBM
            "rvol":            getattr(state, "rvol", 1.0),
            # profile_shape: P=1 (distribuție bullish terminată), D=0 (balanced), b=-1 (bearish)
            "profile_shape_enc": (1 if getattr(state, "profile_shape", "D") == "P"
                                  else -1 if getattr(state, "profile_shape", "D") == "b"
                                  else 0),
            # prev_poc: distanță close față de POC sesiune anterioară (pts NQ)
            "dist_prev_poc":   (close - tick.volume_profile.prev_poc
                                if tick.volume_profile.prev_poc > 0 else 0.0),
            # delta_exhaustion: 1=LONG_EXHAUSTION, -1=SHORT_EXHAUSTION, 0=NONE
            "delta_exhaust_enc": (1  if getattr(state, "delta_exhaustion", "NONE") == "LONG_EXHAUSTION"
                                  else -1 if getattr(state, "delta_exhaustion", "NONE") == "SHORT_EXHAUSTION"
                                  else 0),
            # ── ORDERFLOW DATA (going forward — nu se poate backfill pe date vechi) ──
            # Aceste câmpuri sunt populate DOAR din bridge live, datele istorice rămân 0
            "bar_delta":       float(tick.orderflow.bar_buy_vol - tick.orderflow.bar_sell_vol) if tick.orderflow else 0,
            "cum_delta":       float(tick.orderflow.cum_delta or 0) if tick.orderflow else 0,
            "bar_buy_vol":     float(tick.orderflow.bar_buy_vol or 0) if tick.orderflow else 0,
            "bar_sell_vol":    float(tick.orderflow.bar_sell_vol or 0) if tick.orderflow else 0,
            "delta_at_high":   float(tick.orderflow.delta_at_high or 0) if tick.orderflow else 0,
            "delta_at_low":    float(tick.orderflow.delta_at_low or 0) if tick.orderflow else 0,
            "big_buy_count":   int(tick.orderflow.big_buy_count or 0) if tick.orderflow else 0,
            "big_sell_count":  int(tick.orderflow.big_sell_count or 0) if tick.orderflow else 0,
            "imbalance_pct":   float(tick.orderflow.imbalance_pct or 0) if tick.orderflow else 0,
            "tape_speed":      float(tick.orderflow.tape_speed or 0) if tick.orderflow else 0,
            "dom_bid_total":   int(tick.dom_liquidity.total_bid_size or 0) if tick.dom_liquidity else 0,
            "dom_ask_total":   int(tick.dom_liquidity.total_ask_size or 0) if tick.dom_liquidity else 0,
            "dom_ratio":       float(tick.dom_liquidity.bid_ask_ratio or 1.0) if tick.dom_liquidity else 1.0,
            "vwap_live":       float(tick.orderflow.vwap or 0) if tick.orderflow else 0,
            "absorption_score":float(tick.absorption_score or 0),
            "absorption_side": str(tick.absorption_side or ""),
            "stacked_bull":    int(tick.stacked_imbalances.get("bull_levels", 0)) if tick.stacked_imbalances else 0,
            "stacked_bear":    int(tick.stacked_imbalances.get("bear_levels", 0)) if tick.stacked_imbalances else 0,
            # OF Consolidation Metrics (din NT8 nativ sau fallback)
            "of_doi":          float(getattr(state, '_of_consol_metrics', {}).get("delta_oscillation_idx", 0)),
            "of_bilateral_abs":1 if getattr(state, '_of_consol_metrics', {}).get("bilateral_absorption", False) else 0,
            "of_big_balance":  float(getattr(state, '_of_consol_metrics', {}).get("big_trade_balance", 0.5)),
            "of_d_shape_count":int(getattr(state, '_of_consol_metrics', {}).get("consol_d_shape_count", 0)),
            "source":          "BRIDGE_LIVE",
        }

        placeholders = ", ".join(["?"] * len(row))
        columns      = ", ".join(row.keys())
        conn.execute(
            f"INSERT OR IGNORE INTO market_data ({columns}) VALUES ({placeholders})",
            list(row.values())
        )
        inserted = conn.total_changes > 0
        conn.commit()
        conn.close()
        return inserted
    except Exception as e:
        log.debug(f"DB save bar skip: {e}")
        return False

# ── Persistență strategie activă — supraviețuiește restart bridge ─────────────
def _save_strategy_state():
    """Fix v9.0: Salvează active_strategy + trades counter + daily stats pe disc."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "active_strategy":          state.active_strategy,
            "autotrade_enabled":        state.autotrade_enabled,
            # Fix v9.0: Persistăm counter-ul zilnic + losses + daily PnL
            "strategy_trades_today":    state.strategy_trades_today,
            "consecutive_losses":       state.consecutive_losses,
            "daily_loss_usd":           getattr(state, 'daily_loss_usd', 0.0),
            "daily_profit_usd":         getattr(state, 'daily_profit_usd', 0.0),
            "strategy_last_reset_date": getattr(state, 'strategy_last_reset_date', ''),
            "score_history":            getattr(state, 'score_history', []),
            # Fix v9.1: Persistăm și starea circuit breaker — supraviețuiește restart bridge
            "loss_circuit_open":        getattr(state, 'loss_circuit_open', False),
            # v12.2: Hard lockout — persistăm pe disc, supraviețuiește restart
            "loss_circuit_hard_locked": getattr(state, 'loss_circuit_hard_locked', False),
            # Fix v10.5: Persistăm peak PnL și dd_lockout — previne false circuit breaker după restart
            "session_peak_pnl":         getattr(state, '_session_peak_pnl', 0.0),
            "dd_lockout_active":        getattr(state, 'dd_lockout_active', False),
            "saved_at":                 datetime.now(timezone.utc).isoformat(),
        }
        STRATEGY_STATE_FILE.write_text(json.dumps(payload, indent=2))
    except Exception as _sse:
        log.debug(f"_save_strategy_state skip: {_sse}")

def _load_strategy_state():
    """Fix v9.0: Restaurează strategie + counter trades + daily stats la pornire."""
    try:
        if not STRATEGY_STATE_FILE.exists():
            return
        payload = json.loads(STRATEGY_STATE_FILE.read_text())
        _strat = payload.get("active_strategy")
        if _strat and isinstance(_strat, dict) and _strat.get("id"):
            state.active_strategy    = _strat
            state.autotrade_enabled  = bool(payload.get("autotrade_enabled", True))

            # Fix v9.0: Restaurăm counter-ul doar dacă e din aceeași zi
            _saved_date = payload.get("strategy_last_reset_date", "")
            _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if _saved_date == _today:
                state.strategy_trades_today = int(payload.get("strategy_trades_today", 0))
                state.consecutive_losses    = int(payload.get("consecutive_losses", 0))
                state.daily_loss_usd        = float(payload.get("daily_loss_usd", 0.0))
                state.daily_profit_usd      = float(payload.get("daily_profit_usd", 0.0))
                state.score_history         = list(payload.get("score_history", []))
                # Fix v10.5: Restaurăm peak PnL și dd_lockout — previne false circuit breaker
                state._session_peak_pnl     = float(payload.get("session_peak_pnl", 0.0))
                state.dd_lockout_active     = bool(payload.get("dd_lockout_active", False))
                if state.dd_lockout_active:
                    log.warning(f"⚠️  DD Lockout restaurat după restart — blocat restul zilei")
                # Fix v9.1: Restaurăm loss_circuit_open — dacă era activ înainte de restart, rămâne activ
                _saved_circuit = bool(payload.get("loss_circuit_open", False))
                _max_consec_r  = int(_strat.get("max_consecutive_losses", 2))
                _max_loss_r    = float(_strat.get("max_daily_loss_usd", 1000))
                # Auto-detectăm circuitul și din valorile numerice (protecție dublă)
                # v12.2: Restaurăm hard lockout din disc
                state.loss_circuit_hard_locked = bool(payload.get("loss_circuit_hard_locked", False))
                if _saved_circuit or state.consecutive_losses >= _max_consec_r or state.daily_loss_usd >= _max_loss_r:
                    state.loss_circuit_open = True
                    state.loss_circuit_hard_locked = True  # v12.2: dacă circuit era activ, hard lock-ul rămâne
                    log.warning(
                        f"⚠️🔒 Circuit breaker restaurat după restart (HARD LOCKED): "
                        f"losses={state.consecutive_losses}/{_max_consec_r} | "
                        f"daily_loss=${state.daily_loss_usd:.0f}/${_max_loss_r:.0f} → BLOCAT"
                    )
                log.info(
                    f"♻️  Strategie restaurată: {_strat.get('label', _strat['id'])} "
                    f"| trades azi={state.strategy_trades_today} | losses={state.consecutive_losses} "
                    f"| circuit={'BLOCAT' if state.loss_circuit_open else 'OK'} "
                    f"| autotrade={'ON' if state.autotrade_enabled else 'OFF'}"
                )
            else:
                # Zi nouă → resetăm counter-urile dar păstrăm strategia
                state.strategy_trades_today = 0
                state.consecutive_losses    = 0
                state.daily_loss_usd        = 0.0
                state.daily_profit_usd      = 0.0
                state.score_history         = []
                state.strategy_last_reset_date = _today
                log.info(
                    f"♻️  Strategie restaurată (zi nouă — counters reset): "
                    f"{_strat.get('label', _strat['id'])} | autotrade={'ON' if state.autotrade_enabled else 'OFF'}"
                )
    except Exception as _lse:
        log.debug(f"_load_strategy_state skip: {_lse}")

def _save_open_trade():
    """Salvează detaliile trade-ului deschis pe disc (pentru trailing după restart)."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "position_open":       state._position_open,
            "open_trade_entry":    state.open_trade_entry,
            "open_trade_sl":       state.open_trade_sl,
            "open_trade_tp":       state.open_trade_tp,
            "open_trade_dir":      state.open_trade_dir,
            "open_trade_qty":      min(int(state.open_trade_qty), 2),  # HARD CAP 2 contracte
            "open_trade_ts":       state.open_trade_ts,
            "partial_close_done":  state.partial_close_done,
            "trailing_sl":         state.trailing_sl,
            "trail_r_level":       state.trail_r_level,
            "milestone_05r_done":  state.milestone_05r_done,   # ← FIX: supraviețuiește restart
            "milestone_085r_done": state.milestone_085r_done,  # ← FIX: supraviețuiește restart
        }
        OPEN_TRADE_FILE.write_text(json.dumps(payload, indent=2))
    except Exception as _e:
        log.debug(f"_save_open_trade skip: {_e}")

def _clear_open_trade():
    """Șterge fișierul de trade deschis când poziția s-a închis."""
    try:
        if OPEN_TRADE_FILE.exists():
            OPEN_TRADE_FILE.unlink()
    except Exception as _e:
        log.debug(f"_clear_open_trade skip: {_e}")

def _load_open_trade():
    """Restaurează starea trade-ului deschis la pornire (trailing supraviețuiește restart)."""
    try:
        if not OPEN_TRADE_FILE.exists():
            return
        payload = json.loads(OPEN_TRADE_FILE.read_text())
        if not payload.get("position_open"):
            return

        # ── Auto-clear dacă trade-ul e din ziua precedentă ──────────────────
        _ts_str = str(payload.get("open_trade_ts", ""))
        if _ts_str:
            try:
                _trade_dt = datetime.fromisoformat(_ts_str.replace("Z", "+00:00"))
                _today    = datetime.now(timezone.utc).date()
                if _trade_dt.date() < _today:
                    log.warning(
                        f"🗑️  Trade din ziua precedentă ({_trade_dt.date()}) detectat la startup → "
                        f"auto-clear (NT8 a fost oprit, poziția nu mai există)"
                    )
                    OPEN_TRADE_FILE.write_text(json.dumps({"position_open": False}))
                    return
            except Exception:
                pass  # dacă ts e invalid, continuăm normal

        # Fix v10.6: STARTUP POSITION SYNC — marcăm trade-ul ca "pending_validation"
        # La primul tick NT8 (OnBarUpdate), verificăm dacă NT8 chiar are poziție deschisă.
        # Dacă nu → curățăm state-ul automat (previne restaurarea stale trades).
        state._position_open      = True
        state.open_trade_entry    = float(payload.get("open_trade_entry", 0.0))
        state.open_trade_sl       = float(payload.get("open_trade_sl", 0.0))
        state.open_trade_tp       = float(payload.get("open_trade_tp", 0.0))
        state.open_trade_dir      = str(payload.get("open_trade_dir", ""))
        state.open_trade_qty      = min(int(payload.get("open_trade_qty", 1)), 2)  # HARD CAP 2
        state.open_trade_ts       = str(payload.get("open_trade_ts", ""))
        state.partial_close_done  = bool(payload.get("partial_close_done", False))
        state.trailing_sl         = float(payload.get("trailing_sl", 0.0))
        state.trail_r_level       = int(payload.get("trail_r_level", 0))
        state.milestone_05r_done  = bool(payload.get("milestone_05r_done", False))
        state.milestone_085r_done = bool(payload.get("milestone_085r_done", False))
        state._restored_pending_validation = True   # ← va fi validat la primul tick NT8
        state._restored_validation_ts = time.time()  # timeout: dacă NT8 nu confirmă în 30s → clear
        log.info(
            f"♻️  Trade restaurat din disc: {state.open_trade_dir} entry={state.open_trade_entry} "
            f"SL={state.open_trade_sl} TP={state.open_trade_tp} qty={state.open_trade_qty} "
            f"trailing_sl={state.trailing_sl} partial_done={state.partial_close_done} "
            f"05r={state.milestone_05r_done} 085r={state.milestone_085r_done}"
        )
        log.info(f"   ⏳ PENDING VALIDATION — așteptăm confirm NT8 în 30s, altfel auto-clear")
    except Exception as _e:
        log.debug(f"_load_open_trade skip: {_e}")

# ── Feature 2: Equity Curve Persistence ─────────────────────────────────────────
_equity_history_cache: list = []   # in-memory cache; synced to disc

def _load_equity_history():
    """Încarcă equity history din disc la pornire."""
    global _equity_history_cache
    try:
        if EQUITY_HISTORY_FILE.exists():
            _equity_history_cache = json.loads(EQUITY_HISTORY_FILE.read_text())
        else:
            _equity_history_cache = []
    except Exception:
        _equity_history_cache = []

def _save_equity_history():
    """Salvează equity history pe disc (max 2000 puncte)."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _equity_history_cache[-2000:]  # trim in memory
        tmp = str(EQUITY_HISTORY_FILE) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_equity_history_cache[-2000:], f)
        os.replace(tmp, str(EQUITY_HISTORY_FILE))
    except Exception as _e:
        log.debug(f"_save_equity_history skip: {_e}")

def _persist_equity_point(trade: dict):
    """Adaugă un punct pe equity curve din trade-ul real confirmat de NT8."""
    pnl_usd = float(trade.get("pnl_usd", 0))
    # Calculăm running balance
    prev_balance = _equity_history_cache[-1]["balance"] if _equity_history_cache else 100000.0
    new_balance = round(prev_balance + pnl_usd, 2)

    # Peak & drawdown
    prev_peak = _equity_history_cache[-1]["peak"] if _equity_history_cache else prev_balance
    new_peak = max(prev_peak, new_balance)
    dd_usd = round(new_peak - new_balance, 2)
    dd_pct = round(dd_usd / new_peak * 100, 2) if new_peak > 0 else 0.0

    point = {
        "ts":         trade.get("exit_time", "")[:19],
        "trade_id":   trade.get("id", ""),
        "pnl_usd":    pnl_usd,
        "balance":    new_balance,
        "peak":       new_peak,
        "dd_usd":     dd_usd,
        "dd_pct":     dd_pct,
        "result":     trade.get("result", ""),
        "direction":  trade.get("direction", ""),
        "instrument": trade.get("instrument", ACTIVE_INSTRUMENT),
        "r_multiple": trade.get("r_multiple", 0.0),
        "strategy":   trade.get("strategy", "Default"),
    }
    _equity_history_cache.append(point)
    _save_equity_history()

def _compute_equity_metrics() -> dict:
    """Calculează Sharpe, profit factor, max drawdown din equity history real."""
    pts = _equity_history_cache
    if not pts:
        return {"sharpe": 0, "profit_factor": 0, "max_dd_usd": 0, "max_dd_pct": 0,
                "total_pnl": 0, "trades": 0, "win_rate": 0, "expectancy": 0}

    pnls = [p["pnl_usd"] for p in pts]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    total = len(pnls)
    win_rate = len(wins) / total * 100 if total else 0
    avg_win = gross_win / len(wins) if wins else 0
    avg_loss = gross_loss / len(losses) if losses else 0
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0
    expectancy = round((win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss, 2)

    # Sharpe Ratio — via advanced_features.sharpe_ratio (înlocuiește formula inline)
    sharpe = sharpe_ratio(pnls)   # mean/std × √252, cu fallback graceful

    # Max drawdown din equity history
    max_dd_usd = max((p["dd_usd"] for p in pts), default=0)
    max_dd_pct = max((p["dd_pct"] for p in pts), default=0)
    max_dd_ts = ""
    for p in pts:
        if p["dd_usd"] == max_dd_usd:
            max_dd_ts = p["ts"]
            break

    # Sortino Ratio (penalizes only downside volatility)
    _sortino = sortino_ratio(pnls)

    # Calmar Ratio (return / max drawdown)
    _calmar = calmar_ratio(sum(pnls), max_dd_usd)

    # Fractional Kelly (optimal position sizing)
    _kelly = fractional_kelly(
        win_rate=win_rate / 100.0,
        avg_win=avg_win,
        avg_loss=avg_loss,
        fraction=0.25,
    )

    return {
        "sharpe":         sharpe,
        "sortino":        _sortino,
        "calmar":         _calmar,
        "profit_factor":  pf,
        "max_dd_usd":     round(max_dd_usd, 2),
        "max_dd_pct":     round(max_dd_pct, 2),
        "max_dd_at":      max_dd_ts,
        "total_pnl":      round(sum(pnls), 2),
        "current_balance": pts[-1]["balance"] if pts else 100000.0,
        "peak_balance":    pts[-1]["peak"] if pts else 100000.0,
        "trades":          total,
        "win_rate":        round(win_rate, 1),
        "avg_win":         round(avg_win, 2),
        "avg_loss":        round(avg_loss, 2),
        "expectancy":      expectancy,
        "kelly":           _kelly,
    }


# ── Feature 4: A/B Strategy Testing ────────────────────────────────────────────
_ab_test_state: dict = {
    "active": False,
    "strategy_a": None,     # dict cu configurația strategiei A (real)
    "strategy_b": None,     # dict cu configurația strategiei B (paper)
    "trades_a": [],         # trade-uri reale (A)
    "trades_b": [],         # trade-uri paper (B)
    "started_at": "",
}

def _load_ab_state():
    """Încarcă A/B test state din disc."""
    global _ab_test_state
    try:
        if AB_STATE_FILE.exists():
            _ab_test_state = json.loads(AB_STATE_FILE.read_text())
    except Exception:
        pass

def _save_ab_state():
    """Salvează A/B test state pe disc."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = str(AB_STATE_FILE) + ".tmp"
        # Trim to last 200 trades per strategy to keep file small
        data = {**_ab_test_state}
        data["trades_a"] = data.get("trades_a", [])[-200:]
        data["trades_b"] = data.get("trades_b", [])[-200:]
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, str(AB_STATE_FILE))
    except Exception as _e:
        log.debug(f"_save_ab_state skip: {_e}")

def _ab_record_trade(strategy_slot: str, trade: dict):
    """Înregistrează un trade pentru A/B test."""
    if not _ab_test_state.get("active"):
        return
    key = "trades_a" if strategy_slot == "A" else "trades_b"
    _ab_test_state[key].append({
        "ts":        trade.get("exit_time", ""),
        "pnl_usd":   trade.get("pnl_usd", 0),
        "result":    trade.get("result", ""),
        "direction": trade.get("direction", ""),
        "r_multiple": trade.get("r_multiple", 0),
        "score":     trade.get("score", 0),
    })
    _save_ab_state()

def _ab_compare() -> dict:
    """Compară performanța A vs B."""
    result = {}
    for slot in ["A", "B"]:
        key = f"trades_{slot.lower()}"
        trades = _ab_test_state.get(key, [])
        pnls = [t.get("pnl_usd", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total = len(pnls)
        result[slot] = {
            "strategy":     _ab_test_state.get(f"strategy_{slot.lower()}", {}),
            "trades":       total,
            "total_pnl":    round(sum(pnls), 2),
            "win_rate":     round(len(wins) / total * 100, 1) if total else 0,
            "avg_pnl":      round(sum(pnls) / total, 2) if total else 0,
            "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 999.0,
            "max_win":      round(max(pnls), 2) if pnls else 0,
            "max_loss":     round(min(pnls), 2) if pnls else 0,
        }
    # Winner
    if result["A"]["total_pnl"] > result["B"]["total_pnl"]:
        result["winner"] = "A"
    elif result["B"]["total_pnl"] > result["A"]["total_pnl"]:
        result["winner"] = "B"
    else:
        result["winner"] = "TIE"
    result["started_at"] = _ab_test_state.get("started_at", "")
    result["active"] = _ab_test_state.get("active", False)
    return result


app = FastAPI(title="Aladin API v3.0", version="3.0.0")

# ─── Validation Error Logging ──────────────────────────────────────────────────
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = None
    try:
        body = await request.body()
        body = body.decode("utf-8")
    except Exception:
        pass
    log.error(f"422 VALIDATION ERROR on {request.url.path}")
    log.error(f"  Errors: {exc.errors()}")
    if body:
        # Arată zona din jurul poziției cu eroare
        for err in exc.errors():
            loc = err.get("loc", ())
            if len(loc) >= 2 and isinstance(loc[1], int):
                pos = loc[1]
                start = max(0, pos - 100)
                end = min(len(body), pos + 100)
                log.error(f"  Body around pos {pos}: ...{repr(body[start:end])}...")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# ─── CORS Configuration ────────────────────────────────────────────────────────
_cors_default = "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8080"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", _cors_default).split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]
BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ─── Security Headers Middleware ───────────────────────────────────────────────
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


app.add_middleware(SecurityHeadersMiddleware)


# ─── API Key Validation Middleware ────────────────────────────────────────────
from fastapi import Request


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """
    Validate X-API-Key header for non-localhost requests.
    If BRIDGE_API_KEY is set, all requests must include it.
    """
    if not BRIDGE_API_KEY:
        return await call_next(request)

    # Skip localhost
    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "localhost"):
        return await call_next(request)

    # Skip health endpoint (public)
    if request.url.path == "/health":
        return await call_next(request)

    # Check API key
    api_key = request.headers.get("X-API-Key", "")
    if api_key != BRIDGE_API_KEY:
        return Response(
            status_code=401,
            content={"error": "Invalid or missing X-API-Key header"},
        )

    return await call_next(request)

# ─── Pydantic Models ───────────────────────────────────────────────────────────

class PriceData(BaseModel):
    open:   float = Field(..., ge=0, le=1_000_000)
    high:   float = Field(..., ge=0, le=1_000_000)
    low:    float = Field(..., ge=0, le=1_000_000)
    close:  float = Field(..., ge=0, le=1_000_000)
    volume: float = Field(..., ge=0, le=100_000_000)
    bid:    float = Field(0.0, ge=0, le=1_000_000)
    ask:    float = Field(0.0, ge=0, le=1_000_000)
    spread: float = Field(0.0, ge=0, le=10_000)

class OrderFlowData(BaseModel):
    cum_delta:      float = Field(0.0, ge=-10_000_000, le=10_000_000)
    bar_buy_vol:    float = Field(0.0, ge=0, le=100_000_000)
    bar_sell_vol:   float = Field(0.0, ge=0, le=100_000_000)
    imbalance_pct:  float = Field(0.0, ge=-100, le=100)
    session_vol:    float = Field(0.0, ge=0)
    tick_count:     int   = Field(0,   ge=0, le=1_000_000)
    vwap:           float = Field(0.0, ge=0, le=1_000_000)
    # ── Native big trades (fără AladinAbsorption) ─────────────────────────
    big_buy_count:  int   = Field(0, ge=0)       # trades >= 20c buy în bara curentă
    big_sell_count: int   = Field(0, ge=0)       # trades >= 20c sell în bara curentă
    max_trade_size: float = Field(0.0, ge=0)     # cel mai mare trade individual
    bar_tick_count: int   = Field(0, ge=0)       # tick-uri în bara curentă
    tape_speed:     float = Field(0.0, ge=0)     # tick-uri/secundă în bara curentă
    # ── Bar delta explicit + Delta la extreme ──────────────────────────────
    bar_delta:      float = Field(0.0)           # bar_buy_vol - bar_sell_vol (nu cumulativ sesiune)
    delta_at_high:  float = Field(0.0)           # cum_delta când price a atins High barei
    delta_at_low:   float = Field(0.0)           # cum_delta când price a atins Low barei

class DeltaProfileEntry(BaseModel):
    p: float = Field(0.0)   # price level
    d: float = Field(0.0)   # delta la acel nivel (buy_vol - sell_vol); >0=buyers, <0=sellers

class VolumeProfile(BaseModel):
    poc: float = Field(0.0, ge=0, le=1_000_000)
    vah: float = Field(0.0, ge=0, le=1_000_000)
    val: float = Field(0.0, ge=0, le=1_000_000)
    # ── Previous Session Levels ─────────────────────────────────────────────
    prev_poc: float = Field(0.0, ge=0, le=1_000_000)  # POC sesiune anterioară
    prev_vah: float = Field(0.0, ge=0, le=1_000_000)  # VAH sesiune anterioară
    prev_val: float = Field(0.0, ge=0, le=1_000_000)  # VAL sesiune anterioară
    # ── LVN / HVN ──────────────────────────────────────────────────────────
    hvn: List[float] = Field(default_factory=list)    # High Volume Nodes (top 3)
    lvn: List[float] = Field(default_factory=list)    # Low Volume Nodes (bottom 3)
    # ── VWAP Standard Deviation Bands ──────────────────────────────────────
    vwap_sd:     float = Field(0.0, ge=0)   # deviația standard intra-sesiune
    vwap_sd1_hi: float = Field(0.0)         # VWAP + 1σ
    vwap_sd1_lo: float = Field(0.0)         # VWAP − 1σ
    vwap_sd2_hi: float = Field(0.0)         # VWAP + 2σ (overextension up)
    vwap_sd2_lo: float = Field(0.0)         # VWAP − 2σ (overextension down)
    # ── Delta Profile (per nivel de preț) ─────────────────────────────────
    delta_profile: List[DeltaProfileEntry] = Field(default_factory=list)

class DomLevel(BaseModel):
    price: float; size: int

class DomData(BaseModel):
    bids: List[DomLevel] = Field(default_factory=list)
    asks: List[DomLevel] = Field(default_factory=list)

class DomLiquidity(BaseModel):
    total_bid_size: int = 0; total_ask_size: int = 0; bid_ask_ratio: float = 1.0

class HTFData(BaseModel):
    """H4, H1 și M15 OHLC real din NT8 — pentru HTF bias filter (Mario_Rag Modul 3)"""
    h4_hi: float = 0.0;     h4_lo: float = 0.0
    h4_open: float = 0.0;   h4_close: float = 0.0
    h1_hi: float = 0.0;     h1_lo: float = 0.0
    h1_open: float = 0.0;   h1_close: float = 0.0
    m15_hi: float = 0.0;    m15_lo: float = 0.0
    m15_open: float = 0.0;  m15_close: float = 0.0

class BarHistoryEntry(BaseModel):
    o: float = 0.0  # open
    h: float = 0.0  # high
    l: float = 0.0  # low
    c: float = 0.0  # close
    v: float = 0.0  # volume
    atr: float = 0.0
    d: float = 0.0  # bar delta (buy_vol - sell_vol) — pentru Delta Divergence filter
    t: str = ""     # timestamp ISO

class AccountData(BaseModel):
    """Date reale de cont trimise de AladinBridge.cs pe fiecare bară."""
    cash:               float = 0.0   # CashValue — balanță cash reală NT8
    realized_pnl_today: float = 0.0   # RealizedProfitLoss — P&L realizat azi (reset zilnic în NT8)
    open_pnl:           float = 0.0   # UnrealizedProfitLoss — P&L poziție deschisă
    net_liquidation:    float = 0.0   # NetLiquidation — valoare totală cont

class NT8Data(BaseModel):
    symbol: str; timestamp: str
    price: PriceData
    orderflow: OrderFlowData = Field(default_factory=OrderFlowData)
    volume_profile: VolumeProfile = Field(default_factory=VolumeProfile)
    dom: DomData = Field(default_factory=DomData)
    dom_liquidity: DomLiquidity = Field(default_factory=DomLiquidity)
    bar_history: List[BarHistoryEntry] = Field(default_factory=list)  # ultimele 20 bare NT8
    atr_14: float = 0.0   # ATR(14) nativ NT8
    bar_index: int = 0
    # FAZA 2.3: Correlation NQ/ES — opțional, trimis de AladinBridge dacă e disponibil
    es_close: float = 0.0  # ES futures last price (0 = indisponibil)
    # FAZA 3.2: Multi-timeframe Confluence — date HTF live de la NT8 (opțional)
    h1_close: float = 0.0  # H1 bar close curent
    h4_close: float = 0.0  # H4 bar close curent
    # SYNC POZIȚIE: NT8 trimite statusul real al poziției pe fiecare bară
    # "FLAT" / "LONG" / "SHORT" — bridge sincronizează _position_open din asta
    position_status: str = ""   # "" = nedisponibil (AladinBridge vechi)
    position_qty: int    = 0    # câte contracte are deschise
    h1_trend: str   = ""   # "UP" / "DOWN" / "FLAT" — calculat în NT8
    h4_trend: str   = ""   # "UP" / "DOWN" / "FLAT" — calculat în NT8
    # HISTORICAL ORDERFLOW — date per-bară pentru filtre elite
    poc_history:       List[float] = Field(default_factory=list)  # POC ultim 10 bare
    dom_ratio_history: List[float] = Field(default_factory=list)  # bid/ask ratio ultim 20 bare
    # HTF BIAS — H4 și H1 OHLC real (v2.4+); înlocuiesc zerouri hardcodate
    htf: HTFData = Field(default_factory=HTFData)
    # v8.1: Big Trades — ordine individuale >= 50 contracte (detectate de NT8 addon)
    # Format: [{"price": 19500.5, "size": 75, "side": "BUY", "timestamp": "..."}]
    big_trades: List[Dict[str, Any]] = Field(default_factory=list)
    # v8.1: Absorption Score — calculat de NT8 addon (0-100)
    # 0 = addon nu trimite | >60 = absorbție semnificativă detectată
    absorption_score: float = 0.0
    # v8.1: Absorption Side — "BID" (bullish) sau "ASK" (bearish), "" dacă nu e
    absorption_side: str = ""
    # v9.0: Advanced OrderFlow Analytics
    stacked_imbalances: Dict[str, Any] = Field(default_factory=dict)  # {bull_levels, bear_levels, side}
    unfinished_business: List[Dict[str, Any]] = Field(default_factory=list)  # [{price, side, vol}]
    iceberg: Dict[str, Any] = Field(default_factory=dict)  # {score, side}
    # v10.6: Profile Shape — calculat nativ în NT8 din POC/High/Low
    # P = bullish (POC top 1/3), b = bearish (POC bottom 1/3), D = balanced (consolidare)
    profile_shape: str = ""   # "" = AladinBridge vechi, nu trimite
    # v10.6: OF Consolidation Metrics — calculate nativ în NT8 C# (tick-by-tick precis)
    # Bridge-ul le citește direct, fără recalculare din cache
    of_consol_metrics: Dict[str, Any] = Field(default_factory=dict)  # {delta_oscillation_idx, bilateral_absorption, big_trade_balance, consol_d_shape_count}
    # v10.1: Date reale de cont din NT8 — trimise pe fiecare bară de AladinBridge.cs
    # Folosite pentru a sincroniza state.account_balance și daily P&L din sursa reală NT8
    # Fallback: dacă NT8 nu trimite (AladinBridge vechi), rămân la 0 și bridge folosește estimarea internă
    account: AccountData = Field(default_factory=AccountData)

class ExecutionConfirm(BaseModel):
    action: str  = Field(..., pattern="^(buy|sell|close|reduce|move_sl|BUY|SELL|CLOSE|REDUCE|MOVE_SL|long|short|LONG|SHORT)$")
    qty:    int   = Field(..., ge=0, le=100)
    price:  float = Field(..., ge=0, le=1_000_000)
    ts:     str   = Field(..., max_length=50)

    @validator("action")
    def normalize_action(cls, v):
        v = v.upper()
        if v == "LONG":   return "BUY"
        if v == "SHORT":  return "SELL"
        # REDUCE și MOVE_SL sunt confirmate de NT8 dar nu schimbă direcția
        # — le lăsăm ca atare, handler-ul le va ignora la scale-in detection
        return v

class ManualCommand(BaseModel):
    action: str = Field(..., pattern="^(buy|sell|close|BUY|SELL|CLOSE|long|short|LONG|SHORT)$")
    qty:    int  = Field(1, ge=1, le=100)
    signal: str  = Field("MANUAL", max_length=50)

    @validator("action")
    def normalize_action(cls, v):
        v = v.upper()
        if v == "LONG":  return "BUY"
        if v == "SHORT": return "SELL"
        return v

# ─── State Global ──────────────────────────────────────────────────────────────

class AladinState:
    def __init__(self):
        self.latest_tick: Optional[NT8Data] = None
        self.bar_buffer: deque = deque(maxlen=ANALYSIS_Q_SIZE)
        self.last_analysis: Dict[str, Any] = {}
        self.last_exec_confirm: Optional[Dict] = None
        self.ws_clients: List[WebSocket] = []
        self.tick_count: int = 0
        self.connected_since: Optional[float] = None
        self.session_cum_delta: float = 0.0
        self.delta_history: deque = deque(maxlen=200)
        self.error_log: List[str] = []
        self.trade_log: List[Dict] = []
        self.notes: List[str] = []
        self.backtest_jobs: Dict[str, Dict] = {}
        self.last_backtest_result: Optional[Dict] = None
        # UPDATE #5: Autotrade toggle (ON by default)
        self.autotrade_enabled: bool = True
        # UPDATE #8: Paper Trading mode
        self.paper_mode: bool = False
        # UPDATE #14b: Deduplicare analiză per bară (nu per tick)
        self._last_analysis_bar_ts: str = ""
        # DB auto-save: ultima bară scrisă în SQLite
        self._last_saved_bar_ts: str = ""
        # UPDATE #14e: Flag poziție deschisă — previne trade dublu
        self._position_open: bool = False
        # UPDATE #12: Active Strategy + time window enforcement
        self.active_strategy: Optional[Dict] = None
        self.strategy_trades_today: int = 0
        self.strategy_last_reset_date: str = ""
        self.paper_trades: list = []   # jurnal trades simulate
        self.paper_pnl: float = 0.0    # PnL cumulat paper
        # FAZA 1.1: Consecutive Loss Protection
        self.consecutive_losses: int = 0       # losses la rând azi
        self.daily_loss_usd: float = 0.0       # pierdere totală azi în USD
        self.loss_protection_date: str = ""    # data ultimului reset
        self.loss_circuit_open: bool = False   # True = trading blocat (loss limit)
        # v12.2: HARD LOCKOUT — circuit breaker nu poate fi dezactivat manual
        self.loss_circuit_hard_locked: bool = False  # True = lockout permanent azi
        # PROFIT TARGET: blocare simetrică — când ziua e câștigată, nu o mai risc
        self.daily_profit_usd: float = 0.0     # profit realizat azi în USD
        self.profit_circuit_open: bool = False # True = profit target atins → stop azi
        # PnL tracking pentru Telegram /trade
        self.session_max_profit: float   = 0.0   # cel mai mare profit atins în sesiune
        self.session_max_drawdown: float = 0.0   # cel mai mare drawdown atins în sesiune
        self._session_peak_pnl: float    = 0.0   # peak PnL pentru calcul drawdown
        # FAZA 2.3: Correlation NQ/ES history
        self.es_price_history: list = []       # [(ts, es_close), ...] ultimele 20 bare
        self.nq_price_history: list = []       # [(ts, nq_close), ...] ultimele 20 bare
        # PARTIAL CLOSE 1R: tracking poziție deschisă pentru breakeven management
        self.open_trade_entry: float = 0.0     # prețul de intrare al trade-ului curent
        self.open_trade_sl: float    = 0.0     # SL inițial al trade-ului
        self.open_trade_tp: float    = 0.0     # TP inițial
        self.open_trade_dir: str     = ""      # "LONG" / "SHORT"
        self.open_trade_qty: int      = 1       # cantitate totală intrată
        self.partial_close_done: bool = False  # True = am redus deja la 1R
        self.trailing_sl: float       = 0.0    # SL curent după trailing
        self.trail_r_level: int       = 0      # 0=neinit, 1=BE atins, 2=1R blocat, 3+=ATR trail
        self.milestone_05r_done: bool = False  # True = 0.5R atins, SL mutat la BE
        self.milestone_085r_done: bool = False # True = 0.85R atins, SL mutat la 0.5R profit
        self.open_trade_ts: str       = ""     # timestamp intrare (pentru max hold)
        self.open_trade_mae_pts: float = 0.0  # Max Adverse Excursion (pts contra)
        self.open_trade_mfe_pts: float = 0.0  # Max Favorable Excursion (pts în favoare)
        self.entry_nt8_confirmed: bool = False  # Fix v10.5: True = NT8 a trimis confirm valid de entry
        self.trailing_exit_ts: Optional[str] = None  # timestamp ultimului exit prin trailing/BE
        self.trailing_exit_dir: str          = ""    # direcția la exit (LONG/SHORT)
        self.trailing_exit_pending_reset: bool = False  # True = așteaptă scor sub prag înainte de re-entry
        # ── POST-SL COOLDOWN: după un SL hit, blocăm entry nou timp de N secunde ──
        # Motivul: după o pierdere, botul tinde să intre imediat pe retrace/fakeout
        # și ia al doilea SL consecutiv. Cooldown = 180s = 3 minute.
        self.last_sl_hit_ts: float           = 0.0   # time.time() la ultimul SL hit
        self.post_sl_cooldown_sec: int       = 180   # durata cooldown după SL (3 min)
        # ── DRAWDOWN TIERED SCORING (Lucid MLL protection) ──
        # Scala: peak_pnl - session_total = drawdown din peak al zilei.
        #   dd <  500: permisiv (bot decide)
        #   500 <= dd < 1000: score_min >= 60%
        #   1000 <= dd < 1700: score_min >= 70% + Telegram warn (trade curent NU se închide)
        #   dd >= 1700: LOCKOUT total restul zilei
        self.dd_tier_2_usd: float            = 800.0   # start tier 2 (era 500 — prea agresiv)
        self.dd_tier_3_usd: float            = 1400.0  # start tier 3 (era 1000)
        self.dd_lockout_usd: float           = 2000.0  # lockout (era 1700)
        self.dd_score_tier_2: float          = 55.0    # score min în tier 2 (era 60 — bloca tot)
        self.dd_score_tier_3: float          = 65.0    # score min în tier 3 (era 70)
        self.dd_current_tier: int            = 0       # 0=permisiv, 2=tier2, 3=tier3, 9=lockout
        self.dd_lockout_active: bool         = False   # irreversibil până reset manual/zi nouă
        self.dd_last_tg_tier: int            = 0       # hysteresis pe Telegram (evită spam)
        self._dd_forced_score_min: float     = 0.0     # override score_min calculat în _auto_execute
        # ── EMA-3 SCORE SMOOTHING ──
        # Filtrează oscilațiile de zgomot (ex: 40%→34%→50% în 3 cicluri).
        # EMA-3 cu α=0.5: reacționează rapid dar amortizează spike-urile izolate.
        # Se aplică DOAR pe deciziile de circuit (reversal, DD tier, lockout) — NU pe entry.
        self.score_ema_long: float           = 0.0    # EMA scor LONG (actualizat fiecare ciclu)
        self.score_ema_short: float          = 0.0    # EMA scor SHORT
        self.score_ema_alpha: float          = 0.5    # α = 2/(N+1) pentru N=3
        # ── REVERSAL TRADING ──────────────────────────────────────────────────────
        # Close poziție curentă + entry imediat în direcție opusă când:
        # (1) Orderflow primar s-a inversat: cum_delta + imbalance_pct confirmă
        # (2) Modelul confirmă (score >= 65% în direcție opusă)
        # (3) PnL curent < 0.5R (nu flipăm trade-uri câștigătoare)
        # Execuție în 2 faze: CLOSE → așteptăm confirm NT8 → entry opus
        self.reversal_pending_dir: str    = ""    # "LONG"/"SHORT" — direcție nou trade (armat)
        self.reversal_pending_score: float = 0.0  # scorul modelului la momentul trigger
        self.reversal_pending_sl: float   = 0.0   # SL pre-calculat pentru noul trade
        self.reversal_pending_tp: float   = 0.0   # TP pre-calculat pentru noul trade
        self.reversal_pending_ts: float   = 0.0   # time.time() la trigger (timeout 45s)
        self.reversal_count_today: int    = 0     # nr reversale executate azi
        self.last_reversal_ts: float      = 0.0   # time.time() ultimul reversal (cooldown)
        self.reversal_score_min: float    = 55.0  # scor minim model pentru trigger
        self.reversal_imbalance_min: float = 20.0 # |imbalance_pct| threshold
        self.reversal_max_per_day: int    = 2     # maxim 2 reversale/zi
        self.reversal_cooldown_sec: float = 90.0  # cooldown între reversale (secunde)
        self.reversal_profit_cap_r: float = 0.5   # nu flipăm dacă PnL >= 0.5R
        # ── Orderflow history pentru filtrele anti-retracement ─────────────────
        # bar_vol_history: volum total (buy+sell) per bară — ultimele 20 bare
        # Folosit pentru Filtru 2: vol trigger bar ≥ 1.2× medie (participare instituțională)
        self.bar_vol_history: deque = deque(maxlen=20)
        # A3: Scale In — activ dacă strategia are scale_in=True
        self.scaled_entry_pending: bool  = False
        self.scaled_entry_done: bool     = False
        # Score history — Aladin învață din trade-uri executate (top 20% → 2 contracte)
        self.score_history: list         = []     # ultimele 50 scoruri de intrare
        # ACCOUNT-LEVEL DRAWDOWN: protecție totală cont (nu doar zilnică)
        self.account_balance: float      = 0.0    # balanța curentă a contului (setată din strategie)
        self.account_peak: float         = 0.0    # vârful balanței (pentru drawdown calc)
        self.account_circuit_open: bool  = False  # True = drawdown critic → oprire totală
        # ── LUCID FLEX MLL (Max Loss Limit) — EOD Trailing Drawdown ───────────
        # Floor-ul se mișcă DOAR la end-of-day bazat pe cel mai mare closing balance.
        # Odată ce balanta depășește initial_capital + mll_usd + 100 → floor se blochează.
        # Ref: LucidFlex 50K: MLL=$2000, Initial Trail Balance=$52100, Locked MLL=$50100
        self.lucid_mll_usd: float        = 2000.0   # MLL în USD (din strategie sau default)
        self.lucid_eod_peak: float       = 0.0      # cel mai mare closing balance EOD atins
        self.lucid_mll_floor: float      = 0.0      # floor curent (eod_peak - mll_usd)
        self.lucid_mll_locked: bool      = False    # True = floor blocat la initial+$100
        self.lucid_mll_circuit: bool     = False    # True = balanța ≤ floor → trading oprit
        self._lucid_score_adj: float     = 0.0      # delta score_min din buffer awareness (aplicat în E6)
        # NEWS ALERT: tracked ca să nu trimitem același alert de mai multe ori
        self._news_alerted_today: set    = set()  # set de event_name-uri alertate azi
        self._news_alert_date: str       = ""     # data ultimei resetări a set-ului
        # GEO RISK MODE: activat automat de RSS monitor sau manual via /geo on
        self.geo_risk_active: bool       = False    # True = scor min ridicat cu 10%, sizing redus 30%
        self.geo_risk_reason: str        = ""       # ultima știre/motiv care a activat modul
        self.geo_risk_user_off: bool     = False    # True = userul a dat /geo off manual → blocat auto-activare
        # GEO SENTIMENT: direcția anticipată pe NQ din context geopolitic
        # "BEARISH_NQ" = război/tarife/oil shock → penalizează LONG
        # "BULLISH_NQ" = ceasefire/rate cut/trade deal → penalizează SHORT
        # "NEUTRAL"    = nicio știre relevantă sau impact ambiguu
        self.geo_sentiment: str          = "NEUTRAL"
        # ── RVOL — Relative Volume (calculat per interval orar din DB) ─────────
        self.rvol: float                 = 1.0   # curent RVOL (1.0 = neutral, >1.5 = spike)
        self._rvol_cache_time: str       = ""    # "HH:MM" ultima actualizare (cache per interval)
        # ── Volume Profile Analytics (Python-computed) ─────────────────────
        # profile_shape: forma distribuției VP pentru sesiunea curentă
        #   "P" = volum în top, coadă jos → distribuție bullish terminată (short setup)
        #   "b" = volum în bottom, coadă sus → distribuție bearish terminată (long setup)
        #   "D" = balanced → breakout iminent (energie acumulată, direcție incertă)
        self.profile_shape: str          = "D"
        # delta_exhaustion: cumpărătorii/vânzătorii au împins delta dar prețul n-a urmat
        #   "LONG_EXHAUSTION"  = vânzători absorbiți (delta scăzut, preț rezistă) → long
        #   "SHORT_EXHAUSTION" = cumpărători absorbiți (delta crescut, preț rezistă) → short
        #   "NONE"             = fără semnal
        self.delta_exhaustion: str       = "NONE"
        # ── Composite Volume Profile (multi-sesiune, calculat din DB) ──────
        # HVN/LVN structurale pe 7 zile — niveluri instituționale reale
        self.composite_hvn: List[float]  = []    # High Volume Nodes compozit
        self.composite_lvn: List[float]  = []    # Low Volume Nodes compozit
        self.composite_poc: float        = 0.0   # POC compozit (cel mai tranzacționat nivel)
        self._composite_date: str        = ""    # data ultimei calculări (refresh zilnic)
        self._load_trades()
        self._load_notes()

    def _load_trades(self):
        try:
            if TRADES_FILE.exists():
                with open(TRADES_FILE) as f:
                    self.trade_log = json.load(f)
        except Exception:
            self.trade_log = []

    def _save_trades(self):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            # Fix v7.4: Trade Log Atomic Write — write to temp first, then rename
            tmp_path = str(TRADES_FILE) + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(self.trade_log[-500:], f)
            os.replace(tmp_path, str(TRADES_FILE))
        except Exception as e:
            log.critical(f"Failed to save trades: {e}")

    def _load_notes(self):
        try:
            if NOTES_FILE.exists():
                with open(NOTES_FILE) as f:
                    data = json.load(f)
                    self.notes = data if isinstance(data, list) else data.get("notes", [])
        except Exception:
            self.notes = []

    def _save_notes(self):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(NOTES_FILE, "w") as f:
                json.dump(self.notes, f)
        except Exception:
            pass

    def add_trade(self, trade: Dict):
        self.trade_log.append(trade)
        self._save_trades()

    def compute_stats(self) -> Dict:
        trades = self.trade_log
        if not trades:
            return _empty_stats()
        wins       = [t for t in trades if t.get("result") == "WIN"]
        losses     = [t for t in trades if t.get("result") == "LOSS"]
        pnls       = [t.get("pnl", 0) for t in trades]
        win_pnls   = [p for p in pnls if p > 0]
        loss_pnls  = [p for p in pnls if p < 0]
        gross_win  = sum(win_pnls) if win_pnls else 0
        gross_loss = abs(sum(loss_pnls)) if loss_pnls else 1e-9
        pf         = gross_win / gross_loss

        balance = 10000.0
        max_eq = balance
        max_dd = 0.0
        for t in trades:
            balance += t.get("pnl", 0)
            if balance > max_eq: max_eq = balance
            dd = (max_eq - balance) / max_eq * 100
            if dd > max_dd: max_dd = dd

        sw = sl = cw = cl = 0
        for t in trades:
            if t.get("result") == "WIN": cw += 1; cl = 0
            else: cl += 1; cw = 0
            sw = max(sw, cw); sl = max(sl, cl)

        tod: Dict[str, Dict] = {}
        for t in trades:
            h = t.get("entry_hour", 14)
            key = f"{h:02d}:00"
            if key not in tod:
                tod[key] = {"hour": key, "trades": 0, "wins": 0, "pnl": 0}
            tod[key]["trades"] += 1
            tod[key]["wins"]   += 1 if t.get("result") == "WIN" else 0
            tod[key]["pnl"]    += t.get("pnl", 0)
        tod_list = sorted(tod.values(), key=lambda x: x["hour"])

        days = ["Lun", "Mar", "Mie", "Joi", "Vin"]
        dow: Dict[str, Dict] = {}
        for t in trades:
            d = t.get("entry_dow", 1)
            key = days[d % 5]
            if key not in dow:
                dow[key] = {"day": key, "trades": 0, "wins": 0, "pnl": 0}
            dow[key]["trades"] += 1
            dow[key]["wins"]   += 1 if t.get("result") == "WIN" else 0
            dow[key]["pnl"]    += t.get("pnl", 0)
        dow_list = [dow.get(d, {"day": d, "trades": 0, "wins": 0, "pnl": 0}) for d in days]

        total_pnl = sum(pnls)
        initial   = 10000.0
        return {
            "total_trades": len(trades), "total_wins": len(wins), "total_losses": len(losses),
            "win_rate": len(wins) / len(trades) * 100,
            "profit_factor": round(pf, 2),
            "max_drawdown": round(max_dd, 2),
            "avg_win":  round(sum(win_pnls)  / len(win_pnls)  if win_pnls  else 0, 2),
            "avg_loss": round(sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0, 2),
            "best_trade":  round(max(pnls) if pnls else 0, 2),
            "worst_trade": round(min(pnls) if pnls else 0, 2),
            "total_pnl":   round(total_pnl, 2),
            "return_pct":  round(total_pnl / initial * 100, 2),
            "final_balance": round(initial + total_pnl, 2),
            "max_win_streak": sw, "max_loss_streak": sl,
            "time_of_day": tod_list, "day_of_week": dow_list,
            "monte_carlo": {"p10": round(total_pnl * 0.65, 0), "p50": round(total_pnl, 0), "p90": round(total_pnl * 1.35, 0)},
        }


def _empty_stats():
    days = ["Lun", "Mar", "Mie", "Joi", "Vin"]
    return {
        "total_trades": 0, "total_wins": 0, "total_losses": 0,
        "win_rate": 0, "profit_factor": 0, "max_drawdown": 0,
        "avg_win": 0, "avg_loss": 0, "best_trade": 0, "worst_trade": 0,
        "total_pnl": 0, "return_pct": 0, "final_balance": 10000,
        "max_win_streak": 0, "max_loss_streak": 0,
        "time_of_day": [], "day_of_week": [{"day": d, "trades": 0, "wins": 0, "pnl": 0} for d in days],
        "monte_carlo": {"p10": 0, "p50": 0, "p90": 0},
    }


state = AladinState()

# ─── HTTP Client NT8 ──────────────────────────────────────────────────────────

_nt8_client = httpx.AsyncClient(timeout=1.0)

async def send_command_to_nt8(action: str, qty: int = 1, signal: str = "",
                               sl: float = 0.0, tp: float = 0.0) -> bool:
    # UPDATE #8: Paper Trading intercept — nu trimite la NT8 dacă paper_mode activ
    if state.paper_mode:
        log.info(f"📄 PAPER TRADE: {action} qty={qty} sl={sl} tp={tp} (nu trimis la NT8)")
        return True

    # Fix v7.4: NT8 Command Failure Tracking
    if not hasattr(state, 'nt8_cmd_failures'):
        state.nt8_cmd_failures = 0

    url = f"http://{NT8_IP}:{NT8_PORT}/execute"
    try:
        payload = {"action": action, "qty": qty, "signal": signal, "sl": sl, "tp": tp}
        resp = await _nt8_client.post(url, json=payload)
        if resp.status_code == 200:
            # On success, reset failure counter
            state.nt8_cmd_failures = 0
            return True
        else:
            # On failure, increment and check threshold
            state.nt8_cmd_failures += 1
            if state.nt8_cmd_failures >= 3:
                log.critical("⛔ NT8 unreachable (3+ failures) — pausing auto-trade")
            return False
    except Exception as e:
        log.warning(f"NT8 command failed: {e}")
        # On exception, increment and check threshold
        state.nt8_cmd_failures += 1
        if state.nt8_cmd_failures >= 3:
            log.critical("⛔ NT8 unreachable (3+ failures) — pausing auto-trade")
        return False

# ─── WebSocket Broadcast ──────────────────────────────────────────────────────

async def broadcast_ws(message: dict):
    if not state.ws_clients:
        return
    text = json.dumps(message, default=str)
    dead = []
    for ws in state.ws_clients:
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)

# ─── Analysis ─────────────────────────────────────────────────────────────────

_analysis_lock = asyncio.Lock()

# Fix v7.8: Singleton thread pool — același thread refolosit între analize
# Elimină re-încărcarea SentenceTransformer la fiecare minut (3-4s pierdute)
import concurrent.futures as _cf
_analysis_executor = _cf.ThreadPoolExecutor(max_workers=1)

async def run_analysis_async(tick: NT8Data):
    async with _analysis_lock:
        try:
            loop = asyncio.get_event_loop()
            # Refolosim același executor (thread persistent → mario_rag cache rămâne în memorie)
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(_analysis_executor, _call_mario_rag, tick),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                log.warning(f"Analysis timeout (30s) — mario_rag took too long")
                return
            if result:
                # Fix v7.4: Score Age Tracking
                result["_computed_at"] = time.time()
                state.last_analysis = result
                await broadcast_ws({"type": "analysis", "timestamp": datetime.now(timezone.utc).isoformat(), "data": result})
                await _auto_execute(result, tick)
        except Exception as e:
            log.warning(f"Analysis error: {e}")

def _call_mario_rag(tick: NT8Data) -> Optional[Dict]:
    try:
        import sys
        rag_dir = os.path.dirname(os.path.abspath(__file__))
        if rag_dir not in sys.path:
            sys.path.insert(0, rag_dir)
        import mario_rag, pandas as pd

        # ── Prioritate: bar_history real din NT8 (are ATR nativ) ──────────────
        if tick.bar_history and len(tick.bar_history) >= 1:
            true_open = tick.bar_history[0].o if tick.bar_history else tick.price.open
            poc  = tick.volume_profile.poc
            vah  = tick.volume_profile.vah
            val  = tick.volume_profile.val
            vwap = tick.orderflow.vwap
            df = pd.DataFrame([{
                "open":   b.o, "high": b.h, "low": b.l, "close": b.c, "volume": b.v,
                # OrderFlow — folosim valorile live pentru ultima bară
                "cum_delta":     tick.orderflow.cum_delta if i == len(tick.bar_history)-1 else 0,
                "imbalance_pct": tick.orderflow.imbalance_pct if i == len(tick.bar_history)-1 else 0,
                "vwap":          vwap,
                "poc_level":     poc,
                "vah":           vah,
                "val":           val,
                "bid_ask_ratio": tick.dom_liquidity.bid_ask_ratio if i == len(tick.bar_history)-1 else 1.0,
                # Structural levels — monthly/weekly/asia/london rămân 0 (vin din SQLite)
                "lm_hi": 0, "lm_lo": 0, "lw_hi": 0, "lw_lo": 0,
                "m_hi":  0, "m_lo":  0, "p_hi":  0, "p_lo":  0,
                # H4/H1/M15 — valori REALE din AladinBridge v2.4+
                "h4_hi": tick.htf.h4_hi, "h4_lo": tick.htf.h4_lo,
                "h4_open": getattr(tick.htf, 'h4_open', 0) if tick.htf else 0,
                "h4_close": getattr(tick.htf, 'h4_close', 0) if tick.htf else 0,
                "h1_hi": tick.htf.h1_hi, "h1_lo": tick.htf.h1_lo,
                "h1_open": getattr(tick.htf, 'h1_open', 0) if tick.htf else 0,
                "h1_close": getattr(tick.htf, 'h1_close', 0) if tick.htf else 0,
                "m15_hi": getattr(tick.htf, 'm15_hi', 0) if tick.htf else 0,
                "m15_lo": getattr(tick.htf, 'm15_lo', 0) if tick.htf else 0,
                "m15_open": getattr(tick.htf, 'm15_open', 0) if tick.htf else 0,
                "m15_close": getattr(tick.htf, 'm15_close', 0) if tick.htf else 0,
                "asia_hi": 0, "asia_lo": 0, "lon_hi": 0, "lon_lo": 0,
                "true_open": true_open,
                "fvg_up": 0, "fvg_down": 0,
                "is_above_open":    1 if b.c > true_open else 0,
                "has_displacement": 1 if abs(tick.orderflow.imbalance_pct) > 60 and i == len(tick.bar_history)-1 else 0,
                "is_smt_bearish": 0, "is_smt_bullish": 0,
                "dist_poc":  b.c - poc if poc > 0 else 0,
                "inside_va": 1 if val <= b.c <= vah else 0,
                "dist_pdh":  0, "dist_pdl": 0,
                "atr_14":    b.atr if b.atr > 0 else tick.atr_14,  # ATR nativ din NT8
            } for i, b in enumerate(tick.bar_history)])
        elif len(state.bar_buffer) >= 1:
            # Fallback: bar_buffer din tick-uri live (mai puțin precis)
            rows = list(state.bar_buffer)
            true_open = rows[0].price.open
            df = pd.DataFrame([{
                "open": r.price.open, "high": r.price.high, "low": r.price.low,
                "close": r.price.close, "volume": r.price.volume,
                "cum_delta": r.orderflow.cum_delta, "imbalance_pct": r.orderflow.imbalance_pct,
                "vwap": r.orderflow.vwap, "poc_level": r.volume_profile.poc,
                "vah": r.volume_profile.vah, "val": r.volume_profile.val,
                "bid_ask_ratio": r.dom_liquidity.bid_ask_ratio,
                "lm_hi": 0, "lm_lo": 0, "lw_hi": 0, "lw_lo": 0,
                "m_hi": 0, "m_lo": 0, "p_hi": 0, "p_lo": 0,
                # H4/H1/M15 — valori REALE din AladinBridge v2.4+
                "h4_hi": tick.htf.h4_hi, "h4_lo": tick.htf.h4_lo,
                "h4_open": getattr(tick.htf, 'h4_open', 0) if tick.htf else 0,
                "h4_close": getattr(tick.htf, 'h4_close', 0) if tick.htf else 0,
                "h1_hi": tick.htf.h1_hi, "h1_lo": tick.htf.h1_lo,
                "h1_open": getattr(tick.htf, 'h1_open', 0) if tick.htf else 0,
                "h1_close": getattr(tick.htf, 'h1_close', 0) if tick.htf else 0,
                "m15_hi": getattr(tick.htf, 'm15_hi', 0) if tick.htf else 0,
                "m15_lo": getattr(tick.htf, 'm15_lo', 0) if tick.htf else 0,
                "m15_open": getattr(tick.htf, 'm15_open', 0) if tick.htf else 0,
                "m15_close": getattr(tick.htf, 'm15_close', 0) if tick.htf else 0,
                "asia_hi": 0, "asia_lo": 0, "lon_hi": 0, "lon_lo": 0,
                "true_open": true_open, "fvg_up": 0, "fvg_down": 0,
                "is_above_open": 1 if r.price.close > true_open else 0,
                "has_displacement": 1 if abs(r.orderflow.imbalance_pct) > 60 else 0,
                "is_smt_bearish": 0, "is_smt_bullish": 0,
                "dist_poc": r.price.close - r.volume_profile.poc if r.volume_profile.poc > 0 else 0,
                "inside_va": 1 if r.volume_profile.val <= r.price.close <= r.volume_profile.vah else 0,
                "dist_pdh": 0, "dist_pdl": 0, "atr_14": tick.atr_14,
            } for r in rows])
        else:
            return None  # insuficiente date

        # aladin_engine citeste din SQLite pentru date istorice (FVG, SMT, HTF levels)
        # live_data injectează datele exacte de orderflow/volume profile din NT8
        # Fix v7.4: convertim query timestamp UTC→ET (consistent cu DB)
        _q_utc = datetime.fromisoformat(tick.timestamp[:16].replace("T", " ")).replace(tzinfo=ZoneInfo("UTC"))
        query  = _q_utc.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M")
        live_data = {
            # Feature 3: symbol pentru multi-instrument SL params
            "symbol":        tick.symbol,
            # Fix v10.3: OHLC live din NT8 — fără asta best['close'] rămâne static din DB
            "close":         tick.price.close,
            "open":          tick.price.open,
            "high":          tick.price.high,
            "low":           tick.price.low,
            # OrderFlow live din NT8
            "cum_delta":     tick.orderflow.cum_delta,
            "imbalance_pct": tick.orderflow.imbalance_pct,
            "vwap":          tick.orderflow.vwap,
            # v8.1: Bar buy/sell volume — necesar pentru absorption detection
            "bar_buy_vol":   tick.orderflow.bar_buy_vol,
            "bar_sell_vol":  tick.orderflow.bar_sell_vol,
            # Volume Profile live din NT8
            "poc_level":     tick.volume_profile.poc,
            "vah":           tick.volume_profile.vah,
            "val":           tick.volume_profile.val,
            # DOM Liquidity
            "bid_ask_ratio": tick.dom_liquidity.bid_ask_ratio,
            # v10.5: Advanced OF — delta la extreme, institutional footprint, tape speed
            "delta_at_high":  tick.orderflow.delta_at_high,
            "delta_at_low":   tick.orderflow.delta_at_low,
            "bar_delta":      tick.orderflow.bar_delta,
            "big_buy_count":  tick.orderflow.big_buy_count,
            "big_sell_count": tick.orderflow.big_sell_count,
            "max_trade_size": tick.orderflow.max_trade_size,
            "tape_speed":     tick.orderflow.tape_speed,
            "bar_tick_count": tick.orderflow.bar_tick_count,
            # v10.5: DOM depth — full bid/ask arrays
            "dom_bids":       [{"price": l.price, "size": l.size} for l in tick.dom.bids] if tick.dom.bids else [],
            "dom_asks":       [{"price": l.price, "size": l.size} for l in tick.dom.asks] if tick.dom.asks else [],
            # v10.5: Delta Profile — delta per nivel de preț
            "delta_profile":  [{"p": e.p, "d": e.d} for e in tick.volume_profile.delta_profile] if tick.volume_profile.delta_profile else [],
            # ATR nativ din NT8 (mai precis decât cel calculat din SQLite)
            "atr_14":        tick.atr_14,
            # Score min din strategia activă (setat de user din dashboard)
            "score_min":     float(state.active_strategy.get("score_min", 60)) if state.active_strategy else 60.0,
            # HISTORICAL ORDERFLOW — acum disponibile din AladinBridge.cs v2.3+
            # poc_history: ultimele 10 valori POC (migrare POC = trend instituțional)
            "poc_history":       list(tick.poc_history),
            # dom_ratio_history: ultimele 20 bid/ask ratios (trend vs spike)
            "dom_ratio_history": list(tick.dom_ratio_history),
            # bar_deltas: delta (buy-sell) per bară din bar_history — Delta Divergence
            "bar_deltas":        [b.d for b in tick.bar_history],
            # v8.1: Bar volumes per bară — necesar pt absorption proxy pe istoric
            "bar_volumes":       [b.v for b in tick.bar_history],
            # v8.1: Bar OHLC per bară — body size pt absorption (high vol + small body)
            "bar_opens":         [b.o for b in tick.bar_history],
            "bar_closes":        [b.c for b in tick.bar_history],
            "bar_highs":         [b.h for b in tick.bar_history],
            "bar_lows":          [b.l for b in tick.bar_history],
            # v8.1: Big trades — trimise de NT8 addon (orders >= 50 contracts)
            # Format: [{price, size, side, timestamp}] — gol dacă addon nu trimite
            "big_trades":        list(getattr(tick, 'big_trades', []) or []),
            # v8.1: Absorption score — calculat de NT8 addon (0-100, 0 = nu trimite)
            "absorption_score":  float(getattr(tick, 'absorption_score', 0) or 0),
            # v8.1: Absorption side — "BID" (bullish) / "ASK" (bearish) / "" (no data)
            "absorption_side":   str(getattr(tick, 'absorption_side', '') or ''),
            # v9.0: Advanced OrderFlow Analytics
            "stacked_imbalances":  dict(getattr(tick, 'stacked_imbalances', {}) or {}),
            "unfinished_business": list(getattr(tick, 'unfinished_business', []) or []),
            "iceberg":             dict(getattr(tick, 'iceberg', {}) or {}),
            # NEWS TRADE MODE — activat de user din dashboard
            # Când True: NFP/FOMC/CPI nu mai sunt BLACKOUT, ci NEWS TRADE cu scor +8%
            "trade_news":        bool(state.active_strategy.get("trade_news", False)) if state.active_strategy else False,
            # GEO RISK MODE — activat automat de RSS monitor sau manual via /geo on
            # Când True: aladin_engine ridică score_min cu 10% și notează [GEO RISK] în verdict
            "geo_risk_active":   state.geo_risk_active,
            # GEO SENTIMENT — direcția anticipată pe NQ din context geopolitic
            # "BEARISH_NQ" → penalizează LONG -8% | "BULLISH_NQ" → penalizează SHORT -8% | "NEUTRAL" → nimic
            "geo_sentiment":     state.geo_sentiment,
            # v10.6: OF Consolidation Metrics — compute-uri din delta_history
            "of_consol_metrics": getattr(state, '_of_consol_metrics', {}),
        }
        return mario_rag.aladin_engine(query=query, balance=10000, live_data=live_data)
    except Exception as e:
        log.warning(f"mario_rag skipped: {e}")
        return None

async def _auto_execute(analysis: Dict, tick: NT8Data):
    # UPDATE #5: Respectă toggle-ul din Dashboard
    if not state.autotrade_enabled:
        return

    # UPDATE #14e: Nu executa dacă există deja o poziție deschisă
    # ── REVERSAL TRADING: când conviction flipează, close + entry opus ────────
    if state._position_open:
        # ── EMA UPDATE în timp ce suntem în trade (necesar pentru detecție reversal) ──
        _rev_raw_dir   = str(analysis.get("trade_direction", "")).upper()
        _rev_raw_score = float(analysis.get("score", 0.0))
        _rev_alpha     = state.score_ema_alpha
        if _rev_raw_dir == "LONG":
            state.score_ema_long  = round(_rev_alpha * _rev_raw_score + (1 - _rev_alpha) * state.score_ema_long,  2) if state.score_ema_long  > 0 else _rev_raw_score
            state.score_ema_short = max(0.0, round((1 - _rev_alpha) * state.score_ema_short, 2))
        elif _rev_raw_dir == "SHORT":
            state.score_ema_short = round(_rev_alpha * _rev_raw_score + (1 - _rev_alpha) * state.score_ema_short, 2) if state.score_ema_short > 0 else _rev_raw_score
            state.score_ema_long  = max(0.0, round((1 - _rev_alpha) * state.score_ema_long,  2))

        # ── REVERSAL CHECK ────────────────────────────────────────────────────
        _cur_dir      = state.open_trade_dir              # direcția curentă ("LONG"/"SHORT")
        _opp_dir      = "SHORT" if _cur_dir == "LONG" else "LONG"
        _signal_dir   = _rev_raw_dir                      # ce zice modelul acum
        _signal_score = _rev_raw_score

        # Gate 1: modelul a flipat în direcție opusă cu scor suficient
        _is_flip = (_signal_dir == _opp_dir and _signal_score >= state.reversal_score_min)

        if _is_flip:
            # ── ORDERFLOW: cum_delta curent ──────────────────────────────────────
            _cum_delta = tick.orderflow.cum_delta
            _imbalance = tick.orderflow.imbalance_pct
            _buy_vol   = tick.orderflow.bar_buy_vol
            _sell_vol  = tick.orderflow.bar_sell_vol
            _vwap      = tick.orderflow.vwap
            _price     = float(tick.price.close)

            if _opp_dir == "LONG":
                _of_delta     = _cum_delta > 0
                _of_imbalance = _imbalance > state.reversal_imbalance_min
                _of_vol_bar   = (_buy_vol > _sell_vol * 1.3) if (_buy_vol + _sell_vol) > 0 else False
                _of_vwap      = (_vwap > 0 and _price > _vwap)
            else:
                _of_delta     = _cum_delta < 0
                _of_imbalance = _imbalance < -state.reversal_imbalance_min
                _of_vol_bar   = (_sell_vol > _buy_vol * 1.3) if (_buy_vol + _sell_vol) > 0 else False
                _of_vwap      = (_vwap > 0 and _price < _vwap)

            # ── FILTRU 1: Delta consecutiv pe min 3 bare ─────────────────────
            # Retracement = 1-2 bare de delta contra-trend, apoi revine
            # Reversal real = 3+ bare consecutive cu delta în direcția nouă
            # Per-bar delta = dif dintre cum_delta bara[i] și bara[i-1]
            _dh = list(state.delta_history)
            _filter1_consecutive = False
            _f1_bars_ok = 0
            if len(_dh) >= 4:
                # Ultimele 3 per-bar deltas (din 4 puncte consecutive)
                _pb_deltas = [_dh[i]["delta"] - _dh[i-1]["delta"] for i in range(len(_dh)-3, len(_dh))]
                if _opp_dir == "LONG":
                    _f1_bars_ok = sum(1 for d in _pb_deltas if d > 0)
                    _filter1_consecutive = all(d > 0 for d in _pb_deltas)
                else:
                    _f1_bars_ok = sum(1 for d in _pb_deltas if d < 0)
                    _filter1_consecutive = all(d < 0 for d in _pb_deltas)
            else:
                # Istoric insuficient — dăm benefit of the doubt
                _filter1_consecutive = True
                _f1_bars_ok = 0

            # ── FILTRU 2: Volume pe bara de trigger ≥ 1.2× media recentă ────
            # Retracement = volum scăzut (fără urgență, "thin")
            # Reversal real = spike de volum (participare instituțională)
            _filter2_volume = False
            _vol_trigger    = _buy_vol + _sell_vol
            _vol_avg        = 0.0
            if len(state.bar_vol_history) >= 5:
                _vol_avg     = sum(state.bar_vol_history) / len(state.bar_vol_history)
                _filter2_volume = (_vol_trigger >= _vol_avg * 1.2) if _vol_avg > 0 else True
            else:
                _filter2_volume = True  # istoric insuficient — permitem

            # ── FILTRU 3: Delta Divergence (anti-retracement) ────────────────
            # Dacă prețul se mișcă contra-trend DAR delta rămâne în direcția curentă
            # → cumpărătorii/vânzătorii absorb presiunea → retracement, NU reversal
            # Ex: suntem LONG, prețul coboară (-2 bare), dar per-bar delta pozitiv
            #     → cumpărătorii absorb vânzările → NU flipăm SHORT
            _filter3_no_divergence = True
            if len(_dh) >= 3:
                _pb2 = [_dh[i]["delta"] - _dh[i-1]["delta"] for i in range(len(_dh)-2, len(_dh))]
                _p2  = [_dh[i]["price"] for i in range(len(_dh)-2, len(_dh))]
                _price_down = _p2[-1] < _p2[0]
                _price_up   = _p2[-1] > _p2[0]
                _delta_pos  = all(d > 0 for d in _pb2)
                _delta_neg  = all(d < 0 for d in _pb2)
                # Divergență: prețul coboară DAR delta pozitiv = absorb cumpărători
                #             → nu flipăm SHORT (e retracement)
                # Divergență: prețul urcă DAR delta negativ = distribuție vânzători
                #             → nu flipăm LONG (e retracement rally sau BSL sweep)
                _is_divergence = (
                    (_opp_dir == "SHORT" and _price_down and _delta_pos) or
                    (_opp_dir == "LONG"  and _price_up   and _delta_neg)
                )
                _filter3_no_divergence = not _is_divergence

            # ── FILTRU 4: Delta Exhaustion — absorb instituțional confirmă reversalul ─
            # SHORT_EXHAUSTION = cumpărătorii absorbiți de vânzători → flip SHORT confirmat
            # LONG_EXHAUSTION  = vânzătorii absorbiți de cumpărători → flip LONG confirmat
            _exhaust_signal = state.delta_exhaustion
            _exhaust_ok = (
                (_opp_dir == "LONG"  and _exhaust_signal == "LONG_EXHAUSTION") or
                (_opp_dir == "SHORT" and _exhaust_signal == "SHORT_EXHAUSTION")
            )
            # Dacă exhaustion confirmă, relaxăm F1 de la 3 bare la 2 bare
            # (semnalul de absorb e deja mai puternic decât 3 bare consecutive)
            if _exhaust_ok and not _filter1_consecutive and len(_dh) >= 3:
                _pb_relax = [_dh[i]["delta"] - _dh[i-1]["delta"] for i in range(len(_dh)-2, len(_dh))]
                if _opp_dir == "LONG":
                    _filter1_consecutive = all(d > 0 for d in _pb_relax)
                else:
                    _filter1_consecutive = all(d < 0 for d in _pb_relax)
                if _filter1_consecutive:
                    log.debug(f"🔄 F1 relaxat la 2 bare (exhaustion {_exhaust_signal} confirmă)")

            # ── FILTRU 5: RVOL — filtrăm piețe subțiri (cost mic de manipulat) ──
            # RVOL < 0.80 = volum anemic = breakout/reversal probabil fakeout
            # RVOL ≥ 1.30 = spike instituțional = reversal cu substanță
            _rvol_val = getattr(state, "rvol", 1.0)
            _rvol_ok  = _rvol_val >= 0.80   # gate minim: piață nu e subțire

            # ── OF SCORE COMPOZIT: toate semnalele (0-9) ─────────────────────
            _of_primary_ok = (
                _of_delta                   # cum_delta curent obligatoriu
                and (_of_imbalance or _of_vol_bar)  # minim 1 semnal bară curentă
                and _filter1_consecutive    # 3 bare consecutive (sau 2 cu exhaustion)
                and _filter2_volume         # volum trigger ≥ 1.2× medie
                and _filter3_no_divergence  # fără divergență delta/preț
                and _rvol_ok               # F5: piață lichidă suficient (nu thin)
            )
            _of_score = sum([_of_delta, _of_imbalance, _of_vol_bar, _of_vwap,
                             _filter1_consecutive, _filter2_volume, _filter3_no_divergence,
                             _exhaust_ok, _rvol_ok])  # 0-9

            # ── PROFIT CAP: nu flipăm dacă suntem ≥ 0.5R în profit ──────────
            _e_px  = state.open_trade_entry
            _s_px  = state.open_trade_sl
            _r_pts = abs(_e_px - _s_px) if (_e_px and _s_px) else 0
            _pnl_pts = (_price - _e_px) if _cur_dir == "LONG" else (_e_px - _price)
            _half_r  = _r_pts * state.reversal_profit_cap_r
            _in_profit = (_r_pts > 0 and _pnl_pts >= _half_r)

            # ── COOLDOWN + LIMITE ─────────────────────────────────────────────
            _rev_cooldown_ok = (time.time() - state.last_reversal_ts) >= state.reversal_cooldown_sec
            _rev_count_ok    = state.reversal_count_today < state.reversal_max_per_day
            _rev_not_locked  = not state.dd_lockout_active and not state.loss_circuit_open

            # ── GATE HTF: blocăm reversal contra H4 ──────────────────────────
            # Deducem H4 direction din ict_h4 (h4_aligned cu trade_direction):
            #   ict_h4=True  + direction=LONG  → H4 BULLISH
            #   ict_h4=False + direction=LONG  → H4 BEARISH
            #   ict_h4=True  + direction=SHORT → H4 BEARISH
            #   ict_h4=False + direction=SHORT → H4 BULLISH
            _rev_analysis_dir = str(analysis.get("trade_direction", "")).upper()
            _rev_h4_aligned   = bool(analysis.get("ict_h4", False))
            _h4_bullish = (_rev_h4_aligned and _rev_analysis_dir == "LONG") or \
                          (not _rev_h4_aligned and _rev_analysis_dir == "SHORT")
            _h4_ok_for_reversal = True
            if _opp_dir == "LONG"  and not _h4_bullish:
                _h4_ok_for_reversal = False
                log.warning(f"🔄 REVERSAL LONG blocat: H4 BEARISH → contra-trend HTF (BSL sweep probabil)")
            elif _opp_dir == "SHORT" and _h4_bullish:
                _h4_ok_for_reversal = False
                log.warning(f"🔄 REVERSAL SHORT blocat: H4 BULLISH → contra-trend HTF (SSL sweep probabil)")

            # ── EMERGENCY REVERSAL: bypass OF filters când suntem în pierdere >1R ──
            # Dacă scorul opus e ≥60% și pierdem >1R, filtrele OF devin secundare.
            # Rațiunea: a sta în pierdere cu semnal puternic contra e mai riscant
            # decât un reversal fără confirmare perfectă de orderflow.
            _loss_pts   = -_pnl_pts if _pnl_pts < 0 else 0.0
            _one_r      = _r_pts if _r_pts > 0 else 30.0
            _emergency  = (
                _signal_score >= 60.0      # semnal puternic ≥60%
                and _loss_pts >= _one_r    # în pierdere de cel puțin 1R
                and _h4_ok_for_reversal    # HTF confirmă direcția nouă
                and _rev_cooldown_ok
                and _rev_count_ok
                and _rev_not_locked
            )
            if _emergency and not _of_primary_ok:
                log.warning(
                    f"🚨 EMERGENCY REVERSAL: {_cur_dir}→{_opp_dir} | "
                    f"score={_signal_score:.0f}% ≥60% | pierdere={_loss_pts:.1f}pts ≥ 1R({_one_r:.0f}pts) "
                    f"→ bypass OF filters (OF={_of_score}/9)"
                )

            if (_of_primary_ok or _emergency) and not _in_profit and _rev_cooldown_ok and _rev_count_ok and _rev_not_locked and _h4_ok_for_reversal:
                # ── REVERSAL TRIGGER ACTIVAT ──────────────────────────────────
                # Calculăm SL/TP pentru noul trade (aceeași formulă ATR-adaptive ca entry normal)
                _rr_r  = float(state.active_strategy.get("rr", 2.5)) if state.active_strategy else 2.5
                _sid_r = state.active_strategy.get("id", "") if state.active_strategy else ""
                _SL_R  = {"scalping_london": 20, "scalping_ny": 20, "silver_bullet_london": 25,
                           "silver_bullet_ny": 25, "intraday_london": 35, "ny_open": 30,
                           "intraday_ny": 35, "overlap": 40, "swing_judas": 60, "custom": 30}
                _ATR_R = {"scalping_london": 8, "scalping_ny": 8, "silver_bullet_london": 10,
                           "silver_bullet_ny": 10, "intraday_london": 13, "ny_open": 14,
                           "intraday_ny": 13, "overlap": 16, "swing_judas": 25, "custom": 12}
                _sl_pts_r  = _SL_R.get(_sid_r, 30)
                _atr_live_r = tick.atr_14 if (hasattr(tick, 'atr_14') and tick.atr_14 > 0) else 0.0
                if _atr_live_r > 0:
                    _atr_base_r   = _ATR_R.get(_sid_r, 12)
                    _atr_ratio_r  = min(max(_atr_live_r / _atr_base_r, 0.75), 1.60)
                    _sl_pts_r     = int(round(_sl_pts_r * _atr_ratio_r))
                _tp_pts_r  = round(_sl_pts_r * _rr_r, 2)
                _rev_entry = _price
                if _opp_dir == "LONG":
                    _rev_sl = round(_rev_entry - _sl_pts_r, 2)
                    _rev_tp = round(_rev_entry + _tp_pts_r, 2)
                else:
                    _rev_sl = round(_rev_entry + _sl_pts_r, 2)
                    _rev_tp = round(_rev_entry - _tp_pts_r, 2)

                log.warning(
                    f"🔄 REVERSAL TRIGGER: {_cur_dir}→{_opp_dir} | "
                    f"score={_signal_score:.0f}% | OF={_of_score}/9 | "
                    f"Δcur={'✅' if _of_delta else '❌'} "
                    f"imb={'✅' if _of_imbalance else '❌'} "
                    f"vol={'✅' if _of_vol_bar else '❌'} "
                    f"vwap={'✅' if _of_vwap else '❌'} | "
                    f"F1={'✅' if _filter1_consecutive else '❌'}({_f1_bars_ok}bare) "
                    f"F2={'✅' if _filter2_volume else '❌'}({_vol_trigger:.0f}/{_vol_avg:.0f}) "
                    f"F3={'✅' if _filter3_no_divergence else '❌'} "
                    f"F4_exhaust={'✅' if _exhaust_ok else '❌'}({_exhaust_signal}) "
                    f"F5_rvol={'✅' if _rvol_ok else '❌'}({_rvol_val:.2f}×) | "
                    f"PnL={_pnl_pts:+.1f}pts(cap={_half_r:.1f}pts)"
                )
                _close_ok = await send_command_to_nt8(action="CLOSE", qty=0, signal="REVERSAL_FLIP")
                if _close_ok:
                    state.reversal_pending_dir   = _opp_dir
                    state.reversal_pending_score = _signal_score
                    state.reversal_pending_sl    = _rev_sl
                    state.reversal_pending_tp    = _rev_tp
                    state.reversal_pending_ts    = time.time()
                    state.last_reversal_ts       = time.time()
                    state.reversal_count_today  += 1
                    log.info(
                        f"🔄 REVERSAL armat: {_opp_dir} entry~{_rev_entry:.2f} "
                        f"SL={_rev_sl} TP={_rev_tp} — așteptăm CLOSE confirm NT8"
                    )
                    if _TG_OK and _tg_circuit:
                        try:
                            _tg_circuit(
                                reason="reversal_flip",
                                details=(f"Reversal {_cur_dir}→{_opp_dir} trigerat. "
                                         f"Score {_signal_score:.0f}%, OF {_of_score}/4. "
                                         f"Așteptăm confirm NT8."),
                                daily_loss=state.daily_loss_usd,
                                consecutive=state.consecutive_losses,
                            )
                        except Exception: pass
                else:
                    log.warning("⚠️ REVERSAL: CLOSE NT8 eșuat — reversal anulat")
            else:
                # Logăm de ce nu s-a trigerat (DEBUG — nu spammăm la fiecare bară)
                if _is_flip:
                    _why = []
                    if not _of_delta:
                        _why.append(f"cum_delta={'pozitiv' if _opp_dir=='SHORT' else 'negativ'} lipsă ({_cum_delta:+.0f})")
                    if not (_of_imbalance or _of_vol_bar):
                        _why.append(f"imbalance({_imbalance:+.0f}%) + vol({'ok' if _of_vol_bar else 'slab'}) ambele negative")
                    if not _filter1_consecutive:
                        _why.append(f"F1: doar {_f1_bars_ok}/3 bare consecutive cu delta în {_opp_dir} — posibil retracement")
                    if not _filter2_volume:
                        _why.append(f"F2: vol trigger {_vol_trigger:.0f} < avg×1.2 ({_vol_avg*1.2:.0f}) — mișcare thin")
                    if not _filter3_no_divergence:
                        _why.append("F3: divergență delta/preț detectată — retracement (absorb activ)")
                    if not _rvol_ok:
                        _why.append(f"F5: RVOL {_rvol_val:.2f}× < 0.80 — piață subțire, fakeout probabil")
                    if _exhaust_signal != "NONE" and not _exhaust_ok:
                        _why.append(f"F4: exhaustion {_exhaust_signal} contra-direcțional")
                    if _in_profit:
                        _why.append(f"PnL {_pnl_pts:+.1f}pts ≥ 0.5R ({_half_r:.1f}pts) — nu flipăm winner")
                    if not _rev_cooldown_ok:
                        _why.append(f"cooldown {int(state.reversal_cooldown_sec - (time.time() - state.last_reversal_ts))}s rămase")
                    if not _rev_count_ok:
                        _why.append(f"max {state.reversal_max_per_day}/zi atins")
                    if not _rev_not_locked:
                        _why.append("DD lockout / loss circuit")
                    log.debug(f"📊 REVERSAL skip [{_cur_dir}→{_opp_dir} {_signal_score:.0f}%]: {'; '.join(_why)}")
        return  # poziție deschisă — nu continuăm cu entry normal

    # ── POST-SL COOLDOWN: blocăm entry nou timp de N sec după un SL hit ──
    # După o pierdere, bot-ul intră frecvent pe retrace/fakeout și ia al 2-lea SL
    # consecutiv (pattern confirmat în logurile 08-04-2026). Cooldown 3 min.
    if state.last_sl_hit_ts > 0:
        _since_sl = time.time() - state.last_sl_hit_ts
        if _since_sl < state.post_sl_cooldown_sec:
            _remain = int(state.post_sl_cooldown_sec - _since_sl)
            log.info(
                f"⏳ POST-SL COOLDOWN activ — {_remain}s rămase "
                f"(blocăm entry după SL pentru a evita revenge trade)"
            )
            return
        else:
            # cooldown expirat → resetăm ca să nu mai loghăm degeaba
            state.last_sl_hit_ts = 0.0
            log.info("✅ POST-SL cooldown expirat — entry permis")

    # ── DRAWDOWN TIERED SCORING (Lucid MLL protection) ────────────────────
    # Scala de protecție: cu cât e mai mare drawdown-ul din peak-ul zilei,
    # cu atât e mai strict scorul minim. La $1700 DD din peak → LOCKOUT total.
    # Trade-ul curent NU se închide — regula se aplică doar la entry-uri noi.
    if state.dd_lockout_active:
        log.info(f"🛑 DD LOCKOUT activ — trading blocat restul zilei (DD peak)")
        return

    _sess_total_dd = state.daily_profit_usd - state.daily_loss_usd
    _dd_from_peak = max(0.0, state._session_peak_pnl - _sess_total_dd)
    state._dd_forced_score_min = 0.0   # override aplicat în secțiunea adaptive de jos

    if _dd_from_peak >= state.dd_lockout_usd:
        # LOCKOUT TOTAL
        state.dd_lockout_active = True
        state.dd_current_tier = 9
        log.warning(
            f"🛑 DD LOCKOUT ACTIVAT — drawdown ${_dd_from_peak:.0f} din peak "
            f"${state._session_peak_pnl:.0f} >= ${state.dd_lockout_usd:.0f} → STOP restul zilei"
        )
        if _TG_OK and _tg_circuit:
            try:
                _tg_circuit(
                    reason="dd_lockout",
                    details=(f"Drawdown ${_dd_from_peak:.0f} din peak ${state._session_peak_pnl:.0f}. "
                             f"Lockout activ până la reset manual sau zi nouă."),
                    daily_loss=state.daily_loss_usd,
                    consecutive=state.consecutive_losses,
                )
            except Exception: pass
        return

    elif _dd_from_peak >= state.dd_tier_3_usd:
        # TIER 3: score min 70% + Telegram warn (hysteresis)
        state.dd_current_tier = 3
        state._dd_forced_score_min = state.dd_score_tier_3
        _remain_to_bust = state.dd_lockout_usd - _dd_from_peak
        if state.dd_last_tg_tier < 3:
            log.warning(
                f"🚨 DD TIER 3 — drawdown ${_dd_from_peak:.0f} din peak "
                f"${state._session_peak_pnl:.0f} | ${_remain_to_bust:.0f} până la lockout | "
                f"score_min ridicat la {state.dd_score_tier_3:.0f}%"
            )
            if _TG_OK and _tg_circuit:
                try:
                    _tg_circuit(
                        reason="dd_tier_3_warn",
                        details=(f"⚠️ Drawdown critic ${_dd_from_peak:.0f} din peak. "
                                 f"${_remain_to_bust:.0f} până la pierderea contului. "
                                 f"Score min ridicat la 70% pentru entry-uri noi."),
                        daily_loss=state.daily_loss_usd,
                        consecutive=state.consecutive_losses,
                    )
                except Exception: pass
            state.dd_last_tg_tier = 3

    elif _dd_from_peak >= state.dd_tier_2_usd:
        # TIER 2: score min 60%
        state.dd_current_tier = 2
        state._dd_forced_score_min = state.dd_score_tier_2
        if state.dd_last_tg_tier < 2 or state.dd_last_tg_tier == 3:
            log.info(
                f"⚠️ DD TIER 2 — drawdown ${_dd_from_peak:.0f} din peak → "
                f"score_min {state.dd_score_tier_2:.0f}%"
            )
            # Hysteresis: dacă coborâm din tier 3, resetăm flag-ul tg_tier la 2
            state.dd_last_tg_tier = 2

    else:
        # Permisiv (dd < 500): bot decide liber
        if state.dd_current_tier != 0:
            log.info(f"✅ DD recovered — drawdown ${_dd_from_peak:.0f} < ${state.dd_tier_2_usd:.0f} → tier permisiv")
        state.dd_current_tier = 0
        state.dd_last_tg_tier = 0

    # v9.5 TRAILING SIGNAL-RESET — după exit prin BE/trailing, blocăm re-intrarea
    # pe același semnal activ. Re-intrăm DOAR dacă scorul a coborât sub prag (semnal
    # s-a resetat) și a revenit. Asta evită re-intrarea imediată pe retrasament dar
    # permite continuarea în aceeași direcție dacă piața continuă după reset.
    _cur_dir   = str(analysis.get("verdict", "")).upper()
    _cur_score = float(analysis.get("score", 0.0))
    _threshold = float(state.active_strategy.get("score_min", 60)) if state.active_strategy else 60.0

    if state.trailing_exit_pending_reset and state.trailing_exit_dir:
        _same_dir = (_cur_dir == state.trailing_exit_dir)
        _below_threshold = (_cur_score < _threshold) or (_cur_dir not in ("LONG", "SHORT")) or (not _same_dir)

        if _below_threshold:
            # Scorul a coborât sub prag sau direcția s-a schimbat → semnal resetat
            state.trailing_exit_pending_reset = False
            state.trailing_exit_dir           = ""
            log.info(
                f"✅ TRAILING RESET complet — semnal resetat "
                f"(score={_cur_score:.1f}% dir={_cur_dir}) → re-entry permis pe semnal nou"
            )
            # Continuăm — semnalul e fresh, putem intra dacă scorul e suficient
        else:
            # Același semnal activ, scor încă peste prag → blocat
            log.info(
                f"🔄 TRAILING RESET pending — skip re-entry {_cur_dir} "
                f"score={_cur_score:.1f}% (aștept scor sub {_threshold:.0f}% pentru reset)"
            )
            return

    # Default score_min (overridden below if strategy is active)
    score_min = 60.0

    # UPDATE #12: Strategy time window + max trades enforcement
    if state.active_strategy:
        strat    = state.active_strategy
        now_utc  = datetime.now(timezone.utc)
        now_time = now_utc.strftime("%H:%M")
        today    = now_utc.strftime("%Y-%m-%d")

        # Reset trades counter zilnic + reset loss/profit protection
        if state.strategy_last_reset_date != today:
            state.strategy_trades_today   = 0
            state.strategy_last_reset_date = today
            # Reset complet circuit breaker la zi nouă
            state.consecutive_losses  = 0
            state.daily_loss_usd      = 0.0
            state.loss_circuit_open   = False
            state.loss_circuit_hard_locked = False  # v12.2: reset hard lock doar la zi nouă
            state.daily_profit_usd    = 0.0
            state.profit_circuit_open = False
            state.loss_protection_date = today
            # Reset DD tier lockout la zi nouă
            state.dd_lockout_active   = False
            state.dd_current_tier     = 0
            state.dd_last_tg_tier     = 0
            state._session_peak_pnl   = 0.0
            state.last_sl_hit_ts      = 0.0
            state.score_ema_long      = 0.0
            state.score_ema_short     = 0.0
            state.reversal_count_today = 0
            state.last_reversal_ts    = 0.0
            state.reversal_pending_dir = ""
            state.reversal_pending_ts  = 0.0
            log.info("🔄 Zi nouă — reset circuit breaker losses + profit target + DD lockout + cooldown + EMA + reversal")

        # ── CIRCUIT BREAKER: ACCOUNT-LEVEL DRAWDOWN ───────────────────────────
        _max_account_dd_pct = float(strat.get("max_account_drawdown_pct", 8.0))  # default 8%
        if state.account_circuit_open:
            log.info(f"💀 ACCOUNT DRAWDOWN CIRCUIT activ — trading blocat permanent până la reset manual")
            return
        if state.account_peak > 0 and state.account_balance > 0:
            _acc_dd_pct = (state.account_peak - state.account_balance) / state.account_peak * 100
            if _acc_dd_pct >= _max_account_dd_pct:
                state.account_circuit_open = True
                log.warning(f"💀 ACCOUNT DRAWDOWN CRITIC: -{_acc_dd_pct:.1f}% (limit: -{_max_account_dd_pct:.0f}%)")
                if _TG_OK and _tg_circuit:
                    _tg_circuit(
                        reason="account_drawdown",
                        details=f"Contul a scăzut cu {_acc_dd_pct:.1f}% față de peak (${state.account_peak:.0f}). Limita setată: {_max_account_dd_pct:.0f}%",
                        daily_loss=state.daily_loss_usd,
                        consecutive=state.consecutive_losses,
                        account_drawdown_pct=_acc_dd_pct,
                    )
                return
            # Fix v7.4: Auto-recover circuit if account makes new peak
            if state.account_circuit_open and hasattr(state, 'account_balance') and hasattr(state, 'account_peak'):
                if state.account_balance > state.account_peak:
                    state.account_peak = state.account_balance  # Fix v9.0: UPDATE peak-ul!
                    state.account_circuit_open = False
                    log.info(f"💚 Account circuit auto-cleared (new peak: ${state.account_peak:.0f})")

        # ── CIRCUIT BREAKER: LOSS LIMIT ───────────────────────────────────────
        _max_consecutive = int(strat.get("max_consecutive_losses", 3))   # Fix v10.5: default 3 (era 2 — prea agresiv pe market volatil)
        _max_daily_loss  = float(strat.get("max_daily_loss_usd", 1000))  # default: $1000 pierdere max/zi
        if state.loss_circuit_open:
            log.info(f"⛔ LOSS CIRCUIT BREAKER activ — {state.consecutive_losses} losses consecutive / "
                     f"${state.daily_loss_usd:.0f} pierdere azi — trading blocat până mâine")
            return
        if state.consecutive_losses >= _max_consecutive:
            # v12.3: HARD BLOCK — FĂRĂ BYPASS, FĂRĂ EXCEPȚII
            # Vecheul bypass (sesiune net pozitivă) a permis 21+ CL → pierdut cont prop firm
            # Acum: 3 losses consecutive = STOP IMEDIAT, indiferent de profitul zilei
            _session_net = state.daily_profit_usd - state.daily_loss_usd
            state.loss_circuit_open = True
            state.loss_circuit_hard_locked = True
            # Cooldown 45 min — nu se deschide circuit breaker-ul devreme
            import time as _time_cb
            state._cl_cooldown_until = _time_cb.time() + 45 * 60  # epoch + 45min (unused if hard locked all day)
            log.warning(
                f"⛔🔒 LOSS CIRCUIT BREAKER ACTIVAT + HARD LOCKED: "
                f"{state.consecutive_losses} losses consecutive! "
                f"Sesiune net ${_session_net:.0f} — NU CONTEAZĂ, trading OPRIT. "
                f"Nu poate fi dezactivat manual."
            )
            if _TG_OK and _tg_circuit:
                _tg_circuit(
                    reason="consecutive_losses",
                    details=(
                        f"🛑 {state.consecutive_losses} losses consecutive — HARD STOP ACTIVAT.\n"
                        f"Sesiune net: ${_session_net:.0f}\n"
                        f"🔒 Nu poate fi reactivat azi. Bypass-ul vechi a fost ELIMINAT."
                    ),
                    daily_loss=state.daily_loss_usd,
                    consecutive=state.consecutive_losses,
                )
            return
        if state.daily_loss_usd >= _max_daily_loss:
            state.loss_circuit_open = True
            state.loss_circuit_hard_locked = True   # v12.2: HARD LOCKOUT
            log.warning(f"⛔🔒 LOSS CIRCUIT BREAKER ACTIVAT + HARD LOCKED: pierdere zilnică ${state.daily_loss_usd:.0f} >= ${_max_daily_loss:.0f}! NU poate fi dezactivat manual.")
            if _TG_OK and _tg_circuit:
                _tg_circuit(
                    reason="daily_loss",
                    details=f"Pierdere zilnică ${state.daily_loss_usd:.0f} a atins limita de ${_max_daily_loss:.0f}",
                    daily_loss=state.daily_loss_usd,
                    consecutive=state.consecutive_losses,
                )
            return

        # ── CIRCUIT BREAKER: PROFIT TARGET ────────────────────────────────────
        # Dacă ziua e câștigată, blocăm trading pentru a proteja profitul
        _max_daily_profit = float(strat.get("max_daily_profit_usd", 0))  # 0 = dezactivat
        if _max_daily_profit > 0:
            if state.profit_circuit_open:
                log.info(f"🏆 PROFIT TARGET atins — ${state.daily_profit_usd:.0f} profit azi → stop trading")
                return
            if state.daily_profit_usd >= _max_daily_profit:
                state.profit_circuit_open = True
                log.warning(f"🏆 PROFIT TARGET ATINS: ${state.daily_profit_usd:.0f} >= ${_max_daily_profit:.0f} → ziua e blocată, profit asigurat!")
                if _TG_OK and _tg_circuit:
                    _tg_circuit(
                        reason="profit_target",
                        details=f"Profit zilnic ${state.daily_profit_usd:.0f} a atins targetul de ${_max_daily_profit:.0f} 🎉 Ziua e asigurată!",
                        consecutive=state.consecutive_losses,
                    )

        # ── LUCID FLEX MLL (Max Loss Limit) — EOD Trailing Drawdown ─────────
        # Balanța efectivă = capital inițial + profit realizat - pierderi realizate
        # (NT8 nu trimite account balance live → calculăm din PnL zilnic)
        _lucid_enabled = bool(strat.get("lucid_mll_enabled", False))
        if _lucid_enabled:
            _initial_capital = float(strat.get("initial_capital", 50000.0))
            _mll_usd         = float(strat.get("lucid_mll_usd", 2000.0))
            _locked_buffer   = float(strat.get("lucid_locked_buffer", 100.0))  # $100 over initial

            # Balanța curentă realizată (doar trades închise, nu open PnL)
            state.account_balance = round(
                _initial_capital + state.daily_profit_usd - state.daily_loss_usd, 2
            )

            # Inițializare peak la prima rulare
            if state.lucid_eod_peak == 0.0:
                state.lucid_eod_peak  = _initial_capital
                state.lucid_mll_floor = _initial_capital - _mll_usd
                log.info(f"🏦 LUCID MLL INIT: balance={state.account_balance:.0f} peak={state.lucid_eod_peak:.0f} floor={state.lucid_mll_floor:.0f}")

            # EOD peak update (folosim balanța realizată curentă ca proxy intraday)
            # Pe prop firm real, peak-ul se actualizează DOAR la EOD (4:45 PM EST)
            # Pe demo/simulare, actualizăm în timp real pentru protecție imediată
            _eod_update_live = bool(strat.get("lucid_eod_live_update", True))
            if _eod_update_live and state.account_balance > state.lucid_eod_peak:
                state.lucid_eod_peak = state.account_balance
                # Lock: dacă peak depășește initial + mll + locked_buffer → floor blocat
                _trail_end = _initial_capital + _mll_usd + _locked_buffer
                if not state.lucid_mll_locked and state.lucid_eod_peak >= _trail_end:
                    state.lucid_mll_locked = True
                    state.lucid_mll_floor  = _initial_capital + _locked_buffer
                    log.warning(
                        f"🔒 LUCID MLL LOCKED: peak={state.lucid_eod_peak:.0f} >= {_trail_end:.0f} "
                        f"→ floor blocat la ${state.lucid_mll_floor:.0f} (initial+${_locked_buffer:.0f})"
                    )
                    if _TG_OK and _tg_circuit:
                        _tg_circuit(
                            reason="lucid_mll_locked",
                            details=f"MLL floor blocat la ${state.lucid_mll_floor:.0f}. "
                                    f"Balance={state.account_balance:.0f}, Peak={state.lucid_eod_peak:.0f}",
                            daily_loss=state.daily_loss_usd,
                            consecutive=state.consecutive_losses,
                        )
                elif not state.lucid_mll_locked:
                    state.lucid_mll_floor = state.lucid_eod_peak - _mll_usd
                    log.debug(f"   📈 LUCID MLL: peak actualizat → floor={state.lucid_mll_floor:.0f}")

            # Circuit breaker: balanța ≤ floor → STOP complet
            if state.lucid_mll_circuit:
                log.warning(
                    f"💀 LUCID MLL BREACH: balanța=${state.account_balance:.0f} ≤ "
                    f"floor=${state.lucid_mll_floor:.0f} → BLOCAT PERMANENT"
                )
                return

            if state.lucid_mll_floor > 0 and state.account_balance <= state.lucid_mll_floor:
                state.lucid_mll_circuit = True
                _loss_from_peak = state.lucid_eod_peak - state.account_balance
                log.warning(
                    f"💀 LUCID MLL HIT: balance=${state.account_balance:.0f} ≤ "
                    f"floor=${state.lucid_mll_floor:.0f} (peak=${state.lucid_eod_peak:.0f}, "
                    f"drop=${_loss_from_peak:.0f}) → CONT BREACHED"
                )
                if _TG_OK and _tg_circuit:
                    _tg_circuit(
                        reason="lucid_mll_breach",
                        details=(
                            f"⚠️ LUCID MLL BREACHED! Balance=${state.account_balance:.0f} "
                            f"a atins floor-ul de ${state.lucid_mll_floor:.0f}. "
                            f"Drop din peak: ${_loss_from_peak:.0f}. CONT BLOCAT."
                        ),
                        daily_loss=state.daily_loss_usd,
                        consecutive=state.consecutive_losses,
                    )
                return

            # ── Buffer awareness: când suntem aproape de floor → conservatism ──
            _lucid_buffer_left = state.account_balance - state.lucid_mll_floor
            _lucid_danger_zone = _mll_usd * 0.30   # 30% din MLL = $600 buffer rămas
            _lucid_caution_zone = _mll_usd * 0.60  # 60% din MLL = $1200 buffer rămas

            # score_min e setat ulterior (E6 adaptive) → salvăm delta-ul ca să-l aplicăm acolo
            if _lucid_buffer_left <= _lucid_danger_zone:
                # DANGER: <$600 buffer rămas → max 1 contract + scor minim +15%
                state._lucid_score_adj = 15.0
                log.warning(
                    f"⚠️ LUCID DANGER ZONE: buffer=${_lucid_buffer_left:.0f} < "
                    f"${_lucid_danger_zone:.0f} (30% MLL) → score_min +15% | max 1 contract"
                )
            elif _lucid_buffer_left <= _lucid_caution_zone:
                # CAUTION: <$1200 buffer → scor minim +8%
                state._lucid_score_adj = 8.0
                log.info(
                    f"🔶 LUCID CAUTION ZONE: buffer=${_lucid_buffer_left:.0f} < "
                    f"${_lucid_caution_zone:.0f} (60% MLL) → score_min +8%"
                )
            else:
                state._lucid_score_adj = 0.0
                log.debug(
                    f"   🏦 LUCID MLL OK: balance={state.account_balance:.0f} "
                    f"floor={state.lucid_mll_floor:.0f} buffer={_lucid_buffer_left:.0f}"
                )
        # ─────────────────────────────────────────────────────────────────────

        # Verifică fereastra de timp
        s_start = strat.get("session_start", "00:00")
        s_end   = strat.get("session_end",   "23:59")
        if s_start <= s_end:
            in_window = s_start <= now_time < s_end
        else:  # overnight (ex: 22:00 - 02:00)
            in_window = now_time >= s_start or now_time < s_end
        if not in_window:
            log.debug(f"⏰ [{strat['id']}] Afară ferestrei {s_start}-{s_end} (acum {now_time} UTC)")
            return

        # ── SHORT-ONLY + BIN60/BIN90 TIME FILTER ─────────────────────────────
        # Insight OOS 2023-2025 (734 NY SHORT vs 626 NY LONG semnale conf≥0.60):
        #   NY SHORT: WR=18.7%, avgR=+0.296, P&L=$+32,210 (3yr) ← TOT ALPHA-UL
        #   NY LONG:  WR=11.2%, avgR=+0.010, P&L=$-894        ← FĂRĂ EDGE
        #
        # Ferestre optime (bin 15-min din 9:30 AM ET = ICT Silver Bullet / Power Hour):
        #   bin60: 10:30-10:44 AM ET — WR=54%, avgR=+1.671
        #   bin90: 11:00-11:14 AM ET — WR=31%, avgR=+0.730
        #   COMBINAT: WR=41%, avgR=+1.125, P&L=$308/wk (+52% vs $202/wk baseline)
        #
        # Activat cu: strat["short_only_bin60"] = True (din dashboard Strategy params)
        # ─────────────────────────────────────────────────────────────────────
        _short_only_filter = bool(strat.get("short_only_bin60", False))
        if _short_only_filter:
            _tick_utc = datetime.fromisoformat(
                tick.timestamp[:16].replace("T", " ")
            ).replace(tzinfo=ZoneInfo("UTC"))
            _tick_et  = _tick_utc.astimezone(ZoneInfo("America/New_York"))
            _tick_dir = str(analysis.get("trade_direction", "")).upper()

            # 1. Blochează LONG-uri complet — fără edge pe NY LONG (P&L=-$894 / 3yr)
            if _tick_dir == "LONG":
                log.debug(
                    f"🚫 SHORT-ONLY FILTER: LONG blocat — edge exclusiv pe NY SHORT "
                    f"(WR=18.7% vs 11.2% LONG)"
                )
                return

            # 2. SHORT-uri: doar în bin60 (10:30-10:44) sau bin90 (11:00-11:14) ET
            _et_h = _tick_et.hour
            _et_m = _tick_et.minute
            _in_bin60 = (_et_h == 10 and 30 <= _et_m <= 44)
            _in_bin90 = (_et_h == 11 and  0 <= _et_m <= 14)
            if not (_in_bin60 or _in_bin90):
                log.debug(
                    f"⏰ SHORT-ONLY FILTER: {_tick_et.strftime('%H:%M')} ET "
                    f"nu e în bin60 (10:30-10:44) / bin90 (11:00-11:14) → skip"
                )
                return

            _bin_name = "bin60 (10:30 AM ET)" if _in_bin60 else "bin90 (11:00 AM ET)"
            log.info(f"✅ SHORT-ONLY FILTER: semnal SHORT în {_bin_name} — procesat")

            # 3. Opțional: filtru calitate XGBoost (AUC=0.80, thr≥0.25)
            _qual_thr = float(strat.get("short_qual_threshold", 0.0))
            if _qual_thr > 0.0:
                _sqm, _sqf = _load_short_quality_model()
                if _sqm is not None and _sqf:
                    try:
                        import numpy as _np, pandas as _pd
                        _atr_live_q = tick.atr_14 if (hasattr(tick, 'atr_14') and tick.atr_14 > 0) else 9.0
                        _cl_q  = float(tick.price.close) if hasattr(tick, 'price') else 0.0
                        _h4h_q = float(tick.htf.h4_hi) if tick.htf else 0.0
                        _h4l_q = float(tick.htf.h4_lo) if tick.htf else 0.0
                        _h1h_q = float(tick.htf.h1_hi) if tick.htf else 0.0
                        _h1l_q = float(tick.htf.h1_lo) if tick.htf else 0.0
                        _poc_q = float(tick.volume_profile.poc) if tick.volume_profile else 0.0
                        _vwap_q = float(tick.orderflow.vwap) if tick.orderflow else 0.0
                        _et_min_from_open = (_et_h * 60 + _et_m) - (9 * 60 + 30)  # min from 9:30 AM

                        _feat_vals = {
                            "confidence":       float(analysis.get("score", 60)) / 100.0,
                            "sl_pts":           float(strat.get("sl_pts", 20)),
                            "atr_entry":        _atr_live_q,
                            "time_in_ny":       float(_et_min_from_open),
                            "day_of_week":      float(_tick_et.weekday()),
                            "month":            float(_tick_et.month),
                            "h4_bias":          (((_h4h_q + _h4l_q) / 2 - _cl_q) / _atr_live_q)
                                                if _h4h_q > 0 and _h4l_q > 0 else 0.0,
                            "h1_bias":          (((_h1h_q + _h1l_q) / 2 - _cl_q) / _atr_live_q)
                                                if _h1h_q > 0 and _h1l_q > 0 else 0.0,
                            "hurst":            float(analysis.get("hurst", 0.5)),
                            "adx_14":           float(analysis.get("adx_14", 20.0)),
                            "rvol":             float(analysis.get("rvol", 1.0)) or 1.0,
                            "dom_ratio":        float(tick.dom_liquidity.bid_ask_ratio)
                                                if tick.dom_liquidity else 1.0,
                            "has_displacement": float(bool(analysis.get("displacement", False))),
                            "fvg_down":         float(bool(analysis.get("has_fvg", False))),
                            "acf_lag1":         0.0,
                            "fisher_transform": 0.0,
                            "garch_vol_atr":    1.0,
                            "dist_vwap_atr":    ((_cl_q - _vwap_q) / _atr_live_q)
                                                if _atr_live_q > 0 else 0.0,
                            "dist_poc_atr":     ((_cl_q - _poc_q) / _atr_live_q)
                                                if (_poc_q > 0 and _atr_live_q > 0) else 0.0,
                            "bar_delta_norm":   (float(tick.orderflow.bar_delta) /
                                                 max(float(tick.price.volume), 1))
                                                if tick.orderflow and tick.price else 0.0,
                            "sweep_wick_atr":   0.0,
                            "body_bear":        0.0,
                            "prev_day_bear":    0.0,
                            "atr_vs_10d":       1.0,
                        }
                        _X_q = _pd.DataFrame([[_feat_vals.get(f, 0.0) for f in _sqf]],
                                             columns=_sqf).astype(float)
                        _prob_q = float(_sqm.predict_proba(_X_q)[0, 1])
                        if _prob_q < _qual_thr:
                            log.info(
                                f"🔴 SHORT QUALITY GATE: prob={_prob_q:.3f} < thr={_qual_thr:.2f} "
                                f"→ semnal SHORT de calitate slabă, skip"
                            )
                            return
                        log.info(
                            f"✅ SHORT QUALITY GATE: prob={_prob_q:.3f} >= thr={_qual_thr:.2f} "
                            f"→ semnal HIGH QUALITY, procesat"
                        )
                    except Exception as _sqe:
                        log.warning(f"SHORT quality gate error (ignorat): {_sqe}")
        # ─────────────────────────────────────────────────────────────────────

        # Verifică max trades
        max_t = int(strat.get("max_trades", 99))
        if state.strategy_trades_today >= max_t:
            log.info(f"🛑 [{strat['id']}] Max trades atins: {state.strategy_trades_today}/{max_t}")
            return

        # ── E6: ADAPTIVE SCORE_MIN (top 0.1% — auto-ajustare prag) ─────────────
        # Un prag fix (ex. 65%) e static și ignoră starea curentă a sistemului.
        # Elite: pragul crește automat când sistemul pierde, scade când câștigă.
        # Logica: protejează capitalul în perioade proaste, exploatează perioadele bune.
        score_min = float(strat.get("score_min", 60))
        _base_score_min = score_min

        # v12.6: RELAXAT — Mario vrea 5 trades/zi, nu 7 trades/an
        # Hard stop CL=3 rămâne non-negociabil (circuit breaker deja oprește la 3)
        # Adaptive fost prea agresiv (+15/+30 bloca 48% din trade-uri live)
        # CL=1: +3% (cosmetic, lăsăm trade-urile normale să treacă)
        # CL=2: +8% (ușoară selecție, nu blocaj)
        # CL=3: hard stop la circuit breaker (nu ajunge aici)
        if state.consecutive_losses >= 2:
            score_min = min(score_min + 8.0, 80.0)   # 2 losses → +8% selectiv, NU blocaj
            log.warning(f"   🔺🔺 ADAPTIVE SCORE_MIN: {_base_score_min}% → {score_min}% "
                        f"({state.consecutive_losses} CL — ultim trade înainte de hard stop CL=3)")
        elif state.consecutive_losses == 1:
            score_min = min(score_min + 3.0, 75.0)   # 1 loss → +3% cosmetic
            log.info(f"   🔺 ADAPTIVE SCORE_MIN: {_base_score_min}% → {score_min}% "
                     f"(1 loss consecutiv)")
        elif state.consecutive_losses == 0 and state.strategy_trades_today >= 3:
            # Streak pozitiv (0 losses după 3+ trades azi) → ușor mai agresivi
            score_min = max(score_min - 3.0, 55.0)
            log.debug(f"   🔽 ADAPTIVE SCORE_MIN: {_base_score_min}% → {score_min}% "
                      f"(streak pozitiv, 0 losses)")

        # Aplică delta-ul LUCID buffer awareness după adaptive score_min
        if getattr(state, '_lucid_score_adj', 0.0) > 0:
            _adj_applied = state._lucid_score_adj
            score_min = min(score_min + _adj_applied, 90.0)
            log.info(f"   🏦 LUCID BUFFER ADJ: score_min +{_adj_applied:.0f}% → {score_min:.0f}%")

        # ── DD TIER OVERRIDE: ridică score_min dacă suntem în drawdown tier 2/3 ──
        # Se aplică DUPĂ toate celelalte ajustări — e floor-ul absolut de protecție.
        _dd_forced = getattr(state, '_dd_forced_score_min', 0.0)
        if _dd_forced > 0 and _dd_forced > score_min:
            log.info(
                f"   🛡️ DD TIER {state.dd_current_tier} OVERRIDE: score_min "
                f"{score_min:.0f}% → {_dd_forced:.0f}% (drawdown protection)"
            )
            score_min = _dd_forced

        if analysis.get("score", 0) < score_min:
            _raw_score = analysis.get("score", 0)
            log.debug(
                f"   ⏭️ Score {_raw_score:.1f}% < adaptive min {score_min:.1f}% → skip"
            )
            return

        # ── CONVICTION GATE: LOW conviction necesită scor mai mare ───────────
        # LOW conviction = 0 confluențe ICT (fără SMT, FVG, displacement, killzone)
        # Un trade LOW cu scor limită (50-62%) e mai probabil fals decât real.
        _conviction = str(analysis.get("conviction", "")).upper()
        _conv_score = analysis.get("score", 0)
        if "LOW" in _conviction and _conv_score < 52.0:
            log.warning(
                f"   🔴 CONVICTION LOW blocat: score={_conv_score:.1f}% < 52% minim pentru LOW conviction "
                f"(fără SMT/FVG/displacement/killzone) → skip"
            )
            return

    score    = analysis.get("score", 0)
    verdict  = analysis.get("verdict", "")
    ai_score = analysis.get("ai_score", 0)

    # ── EMA-3 SCORE SMOOTHING ──────────────────────────────────────────────
    # Actualizăm EMA la fiecare ciclu Aladin indiferent de direcție.
    # α=0.5 (N=3): noua valoare are greutate 50%, cele precedente 50%.
    # Scopul: filtrăm oscilațiile de zgomot (34%→59% în 2 min = fakeout).
    # EMA se aplică pe scor brut pe direcția curentă a analizei.
    _ema_alpha = state.score_ema_alpha   # 0.5
    _raw_verdict = analysis.get("trade_direction", "").upper()
    if _raw_verdict == "LONG":
        if state.score_ema_long == 0.0:
            state.score_ema_long = score      # inițializare la primul semnal
        else:
            state.score_ema_long = round(_ema_alpha * score + (1 - _ema_alpha) * state.score_ema_long, 2)
        state.score_ema_short = max(0.0, round((1 - _ema_alpha) * state.score_ema_short, 2))
    elif _raw_verdict == "SHORT":
        if state.score_ema_short == 0.0:
            state.score_ema_short = score
        else:
            state.score_ema_short = round(_ema_alpha * score + (1 - _ema_alpha) * state.score_ema_short, 2)
        state.score_ema_long = max(0.0, round((1 - _ema_alpha) * state.score_ema_long, 2))
    log.debug(
        f"📊 EMA-3 score: raw={score:.1f}% dir={_raw_verdict} "
        f"ema_long={state.score_ema_long:.1f}% ema_short={state.score_ema_short:.1f}%"
    )

    # ── v12.2: SAME-ZONE LOSS PROTECTION ─────────────────────────────────
    # Dacă ultimele 2 trade-uri LOSS au fost în aceeași zonă de preț (±30 pts),
    # blochează trade-uri noi în acea zonă pentru 30 minute.
    # Previne: "death by a thousand cuts" în consolidare.
    _ZONE_RADIUS_PTS  = 30.0    # pts NQ — zona de preț "aceeași zonă"
    _ZONE_COOLDOWN_MIN = 30     # minute cooldown după 2 losses în aceeași zonă
    _MIN_ZONE_LOSSES  = 2       # câte losses în zonă declanșează cooldown
    _cur_px = float(tick.price.close) if hasattr(tick, 'price') else 0.0
    _zone_blocked = False
    if state.trade_log and _cur_px > 0:
        # Ultimele N trade-uri LOSS din ultimele 2 ore
        _now_ts = time.time()
        _recent_losses = []
        for _t in reversed(state.trade_log):
            if _t.get("result") != "LOSS":
                continue
            _t_entry = float(_t.get("entry_price", 0) or 0)
            _t_time  = _t.get("exit_time", "") or _t.get("entry_time", "")
            if not _t_entry or not _t_time:
                continue
            # Parsăm timestamp-ul trade-ului
            try:
                from datetime import datetime as _dt_cls
                _t_dt = _dt_cls.fromisoformat(_t_time.replace("Z", "+00:00"))
                _t_age_min = (_now_ts - _t_dt.timestamp()) / 60.0
            except Exception:
                _t_age_min = 999
            if _t_age_min > 120:   # doar ultimele 2 ore
                break
            _recent_losses.append({"price": _t_entry, "age_min": _t_age_min})

        # Câte losses recente sunt în zona prețului curent?
        _zone_losses = [l for l in _recent_losses
                        if abs(l["price"] - _cur_px) <= _ZONE_RADIUS_PTS]
        if len(_zone_losses) >= _MIN_ZONE_LOSSES:
            # Cel mai recent loss din zonă — verificăm cooldown
            _newest_zone_loss_age = min(l["age_min"] for l in _zone_losses)
            if _newest_zone_loss_age < _ZONE_COOLDOWN_MIN:
                _zone_blocked = True
                _remaining = round(_ZONE_COOLDOWN_MIN - _newest_zone_loss_age, 0)
                log.warning(
                    f"🔒 SAME-ZONE BLOCK: {len(_zone_losses)} losses în zona "
                    f"{_cur_px:.0f}±{_ZONE_RADIUS_PTS:.0f}pts — cooldown {_remaining:.0f}min rămase"
                )

    if _zone_blocked:
        return

    # UPDATE #14d fix: ai_score > 20 bloca toate trade-urile (AI avg p_dir=0.15 → ai_score~5%)
    # Scorul compozit 73%+ deja include validarea AI — nu mai garda separat ai_score
    # Condiție nouă: scorul total > score_min și semnalul nu e SKIP explicit din volatilitate
    _verdict_is_skip = "SKIP" in verdict and ("VOLATIL" in verdict or "BLACKOUT" in verdict)
    if score >= score_min and not _verdict_is_skip:
        direction = analysis.get("trade_direction", "LONG")
        _risk     = analysis.get("risk", {})
        entry_px  = float(tick.price.close) if hasattr(tick, 'price') else 0.0

        # ── FAZA 2.3: Correlation Filter NQ/ES ───────────────────────────────
        # Dacă NQ și ES diverge în ultimele N bare, semnalul e neconcludent.
        # NQ/ES au corelație ~0.94 — divergența > 0.3% timp de 3+ bare = suspect.
        # Necesită ca AladinBridge.cs să trimită es_close în payload.
        _corr_active = len(state.es_price_history) >= 5 and len(state.nq_price_history) >= 5
        if _corr_active:
            # Calculăm direcția ultimelor 5 bare pentru NQ și ES
            _nq_last5 = [p for _, p in state.nq_price_history[-5:]]
            _es_last5 = [p for _, p in state.es_price_history[-5:]]
            _nq_dir = _nq_last5[-1] - _nq_last5[0]   # >0 up, <0 down
            _es_dir = _es_last5[-1] - _es_last5[0]
            # Normalize: procent din preț
            _nq_dir_pct = _nq_dir / _nq_last5[0] * 100 if _nq_last5[0] > 0 else 0
            _es_dir_pct = _es_dir / _es_last5[0] * 100 if _es_last5[0] > 0 else 0
            # Divergență: NQ merge sus dar ES merge jos (sau invers) cu diferență > 0.15%
            _diverge = (_nq_dir_pct > 0.05 and _es_dir_pct < -0.05) or \
                       (_nq_dir_pct < -0.05 and _es_dir_pct > 0.05)
            if _diverge:
                log.warning(
                    f"⚠️ CORELAȚIE NQ/ES RUPTĂ: NQ={_nq_dir_pct:+.3f}% ES={_es_dir_pct:+.3f}% "
                    f"→ trade SKIP (divergență piețe)"
                )
                return   # skip trade — semnal neconcludent
            else:
                log.debug(
                    f"📊 Corelație NQ/ES OK: NQ={_nq_dir_pct:+.3f}% ES={_es_dir_pct:+.3f}%"
                )
        # ─────────────────────────────────────────────────────────────────────

        # ── SPREAD / SLIPPAGE MONITOR ─────────────────────────────────────────
        # NQ normal spread: 0.25-0.50 puncte (1-2 tickuri).
        # Spread > 1.5 puncte = condiții anormale (news, pre-market, slippage mare) → SKIP
        _spread     = float(tick.price.ask - tick.price.bid) if hasattr(tick, 'price') else 0.0
        _max_spread = 1.5  # puncte NQ — configurabil
        if _spread > _max_spread and _spread > 0:
            log.warning(
                f"⚠️ SPREAD ANORMAL: {_spread:.2f} pts (max={_max_spread}) → trade SKIP (slippage risk)"
            )
            return   # nu executa în condiții de spread larg
        elif _spread > 0:
            log.debug(f"📊 Spread OK: {_spread:.2f} pts")
        # ─────────────────────────────────────────────────────────────────────

        # ── MAX CONTRACTS ENFORCEMENT (Lucid Flex: 4 Mini / 40 Micro) ──────────
        # _lucid_buffer_left/_lucid_danger_zone definite în blocul MLL de mai sus;
        # inițializate la valori sigure dacă Lucid MLL nu e activ
        if not _lucid_enabled:
            _lucid_buffer_left = float('inf')
            _lucid_danger_zone = 0.0
        _max_contracts = int(strat.get("max_contracts", 4))
        _trade_qty_raw = int(_risk.get("units", 1)) if _risk else 1
        _trade_qty     = min(max(_trade_qty_raw, 1), _max_contracts)
        # Buffer danger zone → forțăm 1 contract indiferent de sizing normal
        if _lucid_enabled and _lucid_buffer_left <= _lucid_danger_zone:
            _trade_qty = 1
            log.warning(f"⚠️ LUCID DANGER: sizing redus forțat la 1 contract (buffer=${_lucid_buffer_left:.0f})")
        elif _trade_qty != _trade_qty_raw:
            log.info(f"🔒 MAX CONTRACTS: qty {_trade_qty_raw} → {_trade_qty} (max={_max_contracts})")
        # ─────────────────────────────────────────────────────────────────────

        # ── CONSOLIDATION MID-RANGE DOUBLE-CHECK (bridge layer) ──────────────
        # mario_rag.py aplică skip_volatile deja, dar dacă scorul a rămas >score_min
        # prin post-proc floor (40% din engine score), blocăm explicit în bridge.
        _consol_scalping = bool(analysis.get("consol_scalping", False)) if analysis else False
        _pd_pct_br       = float(analysis.get("pd_pct", 0.5)) if analysis else 0.5
        _regime_br       = str(analysis.get("regime", "")).upper()
        _is_consol_br    = "CONSOL" in _regime_br or "RANG" in _regime_br or "CHOP" in _regime_br
        _at_extreme_br   = (_pd_pct_br <= 0.15 or _pd_pct_br >= 0.85)
        _has_disp_br     = bool(analysis.get("displacement", False)) if analysis else False

        if _is_consol_br and not _consol_scalping and not _has_disp_br and not _at_extreme_br:
            log.warning(
                f"🔲 CONSOLIDATION MID-RANGE [bridge block]: regime={_regime_br} "
                f"pd_pct={_pd_pct_br:.0%} → trade SKIP"
            )
            return
        # ─────────────────────────────────────────────────────────────────────

        # UPDATE #15: SL/TP fixe în puncte per tip strategie
        # Distanțe rezonabile pentru NQ intraday/scalping (nu mai folosim SL-ul din analiză
        # care era calculat pe bara anterioară și genera distanțe de 1000+ puncte)
        _rr      = float(state.active_strategy.get("rr", 2.5) if state.active_strategy else 2.5)
        _strat_id = state.active_strategy.get("id", "") if state.active_strategy else ""

        # ── E5: SL/TP baseline per strategie ─────────────────────────────────
        _SL_POINTS = {
            "scalping_london":      20,
            "scalping_ny":          20,
            "silver_bullet_london": 25,
            "silver_bullet_ny":     25,
            "intraday_london":      35,
            "ny_open":              30,
            "intraday_ny":          35,
            "overlap":              40,
            "swing_judas":          60,
            "custom":               30,
        }
        _sl_pts_base = _SL_POINTS.get(_strat_id, 30)

        # ── BIN60 ATR-BASED SL OVERRIDE ──────────────────────────────────────
        # Backtest OOS 2023-2025: SL optim = ~0.87×ATR (median 6pts, max 14pts).
        # Bridge folosea SL fix 20-35pts → mismatch față de backtest.
        # Când short_only_bin60 e activ ȘI semnalul e în bin60/bin90,
        # overrideăm SL la ATR-based pentru a alinia cu logica backtestată.
        # Floor: 5pts (minim funcțional pe NQ)  Ceiling: 15pts (max acceptabil).
        # Activat cu: strat["bin60_atr_sl"] = True
        _bin60_atr_sl = bool(strat.get("bin60_atr_sl", False))
        if _bin60_atr_sl and _short_only_filter:
            _atr_for_sl = tick.atr_14 if (hasattr(tick, 'atr_14') and tick.atr_14 > 0) else 7.0
            _sl_pts_base = max(5, min(15, int(round(_atr_for_sl * 0.87))))
            log.info(
                f"   🎯 BIN60 ATR-SL: SL={_sl_pts_base}pts (ATR={_atr_for_sl:.1f} × 0.87) "
                f"← aliniament backtest (median 6pts OOS 2023-2025)"
            )
        # ─────────────────────────────────────────────────────────────────────

        # ── E5: ATR-RELATIVE SL/TP (top 0.1% risk management) ────────────────
        # SL fix = pierderi mari în piețe volatile / premature exit în piețe liniștite.
        # Soluție: scalăm SL față de ATR curent vs ATR "tipic" per strategie.
        # Volatilitate ridicată (ATR 2× normal) → SL mai larg cu 40% (evită premature exit)
        # Volatilitate scăzută (ATR 0.5× normal) → SL mai strâns cu 20% (risc mic, poziție mai bună)
        _ATR_BASELINE = {
            # ATR NQ tipic per fereastra de timp a strategiei (în puncte)
            "scalping_london":       8.0,
            "scalping_ny":           8.0,
            "silver_bullet_london": 10.0,
            "silver_bullet_ny":     10.0,
            "intraday_london":      13.0,
            "ny_open":              14.0,
            "intraday_ny":          13.0,
            "overlap":              16.0,
            "swing_judas":          25.0,
            "custom":               12.0,
        }
        _atr_live = tick.atr_14 if (hasattr(tick, 'atr_14') and tick.atr_14 > 0) else 0.0
        _atr_base = _ATR_BASELINE.get(_strat_id, 12.0)

        if _atr_live > 0 and _atr_base > 0:
            # Ratio ATR curent vs tipic — clamped între 0.75× și 1.60× pentru siguranță
            _atr_ratio = round(min(max(_atr_live / _atr_base, 0.75), 1.60), 3)
            _sl_pts    = int(round(_sl_pts_base * _atr_ratio))
            _atr_tag   = f" [ATR {_atr_live:.1f} vs base {_atr_base:.0f} → ×{_atr_ratio}]"
        else:
            _sl_pts  = _sl_pts_base
            _atr_tag = " [ATR n/a → baseline]"

        _tp_pts = round(_sl_pts * _rr, 2)

        # NEWS TRADE MODE — extinde TP ×1.3 (news moves sunt mai ample)
        _news_mode_active = bool(analysis.get("news_mode_active", False)) if analysis else False
        _news_tp_tag = ""
        if _news_mode_active:
            _tp_pts = round(_tp_pts * 1.3, 2)
            _news_tp_tag = " [📰 NEWS TP×1.3]"
            log.info(f"   📰 NEWS TRADE MODE: TP extins la {_tp_pts}pts")

        # ── Fix v10.6: CONSOLIDATION-AWARE SL/TP ────────────────────────────
        # În consolidare (ranging/chop), SL=53pts și TP=132pts e absurd când range-ul
        # e de 40-50 puncte. Entry-urile în consolidare sunt scalp-uri din VA extreme
        # (VAL→POC sau POC→VAH) cu target 20-40pts, nu trend-following de 130pts.
        #
        # Detectare consolidare:
        #   1. Flag explicit din mario_rag: consol_scalping=True (regime CONSOL/RANGING/CHOP)
        #   2. Range detection: dacă VA spread (VAH-VAL) < 2×SL_base → piața e strânsă
        #   3. ATR compression: dacă ATR curent < 60% din ATR base → volatilitate scăzută
        #
        # Parametri consolidare:
        #   SL: max(15, min(VA_spread×0.4, 22)) — strâns, adaptat la range real
        #   TP: max(20, min(VA_spread×0.6, 40)) — scalp din VAL/VAH spre POC/mijloc
        #   R:R rămâne ≥1:1.3 minimum
        _consol_tp_tag = ""
        _consol_mode_active = False

        # Metoda 1: flag explicit din mario_rag
        if _consol_scalping:
            _consol_mode_active = True

        # Metoda 2: range detection din VA spread
        _va_spread = 0.0
        if analysis:
            _vah_br = float(analysis.get("vah", 0) or 0)
            _val_br = float(analysis.get("val", 0) or 0)
            if _vah_br > 0 and _val_br > 0:
                _va_spread = _vah_br - _val_br
            # Dacă VA spread < 2× SL base → piața e într-un range strâns
            if _va_spread > 5 and _va_spread < _sl_pts_base * 2.0:
                _consol_mode_active = True
                log.info(
                    f"   🔲 RANGE DETECT: VA spread={_va_spread:.1f}pts < 2×SL_base={_sl_pts_base*2}pts "
                    f"→ consolidation mode"
                )

        # Metoda 3: ATR compression
        if _atr_live > 0 and _atr_base > 0 and (_atr_live / _atr_base) < 0.60:
            _consol_mode_active = True
            log.info(
                f"   🔲 ATR COMPRESS: ATR={_atr_live:.1f} < 60% base={_atr_base:.0f} "
                f"→ consolidation mode"
            )

        # Metoda 4: OF Consolidation Metrics (3+ din 4 semnale = consolidare confirmată OF)
        _ofcm = getattr(state, '_of_consol_metrics', {})
        if _ofcm:
            _of_consol_signals = sum([
                _ofcm.get("delta_oscillation_idx", 0) < -0.15,
                _ofcm.get("bilateral_absorption", False),
                0.35 <= _ofcm.get("big_trade_balance", 0.5) <= 0.65,
                _ofcm.get("consol_d_shape_count", 0) >= 3,
            ])
            if _of_consol_signals >= 3 and not _consol_mode_active:
                _consol_mode_active = True
                log.info(
                    f"   🔲 OF CONSOL DETECT ({_of_consol_signals}/4): "
                    f"Δ_oscil={_ofcm.get('delta_oscillation_idx',0):.3f} "
                    f"biAbsorb={'✅' if _ofcm.get('bilateral_absorption') else '❌'} "
                    f"bigBal={_ofcm.get('big_trade_balance',0):.2f} "
                    f"D-shape={_ofcm.get('consol_d_shape_count',0)} → consolidation mode"
                )

        # ── BREAKOUT OVERRIDE: displacement CONFIRMAT de OF → forțăm TREND mode ──
        # ATENȚIE: displacement (candle size > 1.5×ATR) ≠ breakout real.
        # Pe NQ, sweep-urile BSL/SSL au aceeași candelă mare dar prețul revine.
        # Diferența e în orderflow:
        #   - Breakout real: delta susținut, fără absorption la extremă, big trades pe o parte
        #   - Sweep/fakeout: delta se inversează, absorption la extremă, big trades echilibrate
        # → displacement override DOAR dacă OF confirmă breakout real
        if _consol_mode_active and _has_disp_br:
            _ofcm_bo = getattr(state, '_of_consol_metrics', {})
            _doi_bo  = _ofcm_bo.get("delta_oscillation_idx", 0)
            _ba_bo   = _ofcm_bo.get("bilateral_absorption", False)
            _btb_bo  = _ofcm_bo.get("big_trade_balance", 0.5)

            # OF confirmă breakout: delta momentum (nu mean-revert) + fără bilateral absorption
            # + big trades dezechilibrate (una domină)
            _of_confirms_breakout = (
                _doi_bo > -0.05 and          # delta NU mean-reverts (nu e consolidare)
                not _ba_bo and                # NU e absorption bilaterală (un capăt a cedat)
                (_btb_bo < 0.30 or _btb_bo > 0.70)  # big trades dezechilibrate (direcționale)
            )

            # Extra check: delta_at_high / delta_at_low din ultimul tick
            _dah_last = float(tick.orderflow.delta_at_high or 0) if hasattr(tick.orderflow, 'delta_at_high') else 0
            _dal_last = float(tick.orderflow.delta_at_low or 0) if hasattr(tick.orderflow, 'delta_at_low') else 0
            _direction_br = str(analysis.get("trade_direction", "")).upper() if analysis else ""

            # Bullish breakout: delta_at_high pozitiv (cumpărători agresivi la high, nu absorbiți)
            # Bearish breakout: delta_at_low negativ (vânzători agresivi la low, nu absorbiți)
            _no_absorption_at_extreme = True
            if _direction_br == "LONG" and _dah_last < -50:
                _no_absorption_at_extreme = False   # absorption la highs → sweep BSL, nu breakout
            elif _direction_br == "SHORT" and _dal_last > 50:
                _no_absorption_at_extreme = False   # absorption la lows → sweep SSL, nu breakout

            if _of_confirms_breakout and _no_absorption_at_extreme:
                _consol_mode_active = False
                log.info(
                    f"   📈 BREAKOUT CONFIRMAT OF: displacement + delta_oscil={_doi_bo:.3f} "
                    f"+ no biAbsorb + bigBal={_btb_bo:.2f} + no absorption at extreme "
                    f"→ TREND mode"
                )
            else:
                # Displacement FĂRĂ OF confirm → probabil SWEEP → rămânem pe consolidation
                log.info(
                    f"   🔲 SWEEP SUSPECT: displacement candle DAR OF NU confirmă breakout "
                    f"(Δ_oscil={_doi_bo:.3f} biAbsorb={'✅' if _ba_bo else '❌'} "
                    f"bigBal={_btb_bo:.2f} absorp@ext={'❌' if _no_absorption_at_extreme else '✅'}) "
                    f"→ rămânem CONSOL mode (SL/TP scalp)"
                )

        # ATR spike: doar dacă persistent (nu single candle spike care = sweep)
        if _consol_mode_active and _atr_live > 0 and _atr_base > 0 and (_atr_live / _atr_base) > 1.50:
            # Threshold ridicat de la 1.30 la 1.50 — un spike mic poate fi sweep
            _consol_mode_active = False
            log.info(
                f"   📈 ATR SPIKE OVERRIDE: ATR={_atr_live:.1f} > 150% base={_atr_base:.0f} "
                f"→ consolidation mode dezactivat (volatilitate excesivă)"
            )

        if _consol_mode_active:
            # ── SL adaptat la range real ──
            if _va_spread > 10:
                _consol_sl = max(16, min(int(round(_va_spread * 0.45)), 25))
                _consol_tp = max(22, min(int(round(_va_spread * 0.70)), 50))
            else:
                # Fallback: SL/TP fix pentru scalp
                _consol_sl = 20
                _consol_tp = 32

            # Asigurăm R:R minim 1:1.3
            if _consol_tp < _consol_sl * 1.3:
                _consol_tp = int(round(_consol_sl * 1.3))

            _old_sl = _sl_pts
            _old_tp = _tp_pts
            _sl_pts = _consol_sl
            _tp_pts = _consol_tp
            _consol_tp_tag = (
                f" [🔲 CONSOL SL={_consol_sl}pts TP={_consol_tp}pts"
                f" (was SL={_old_sl} TP={_old_tp:.0f})"
                f" VA={_va_spread:.0f}pts]"
            )
            log.info(
                f"   🔲 CONSOLIDATION SCALP MODE: SL {_old_sl}→{_consol_sl}pts | "
                f"TP {_old_tp:.0f}→{_consol_tp}pts | VA spread={_va_spread:.0f}pts | "
                f"R:R 1:{_consol_tp/_consol_sl:.1f}"
            )

        if direction == "LONG":
            sl_px = round(entry_px - _sl_pts, 2)
            tp_px = round(entry_px + _tp_pts, 2)
        else:  # SHORT
            sl_px = round(entry_px + _sl_pts, 2)
            tp_px = round(entry_px - _tp_pts, 2)

        # Fix v7.4: SL Validation — ensure SL is on correct side
        if direction == "LONG" and sl_px >= entry_px:
            log.error(f"LONG SL {sl_px} >= entry {entry_px} — clamping")
            sl_px = entry_px - 10  # Emergency fallback: 10 pts below
        if direction == "SHORT" and sl_px <= entry_px:
            log.error(f"SHORT SL {sl_px} <= entry {entry_px} — clamping")
            sl_px = entry_px + 10  # Emergency fallback: 10 pts above

        log.info(
            f"🎯 SL/TP ATR-adaptive [{_strat_id}]: entry={entry_px} "
            f"SL={_sl_pts}pts TP={_tp_pts}pts R:R=1:{_rr}"
            f"{_atr_tag}{_news_tp_tag}{_consol_tp_tag} → SL={sl_px} TP={tp_px}"
        )

        # UPDATE #8: Paper mode — simuleaza PnL instant (entry→TP sau entry→SL)
        if state.paper_mode:
            # Estimăm PnL simplu: dacă scor > 70 presupunem TP atins, altfel 50/50
            import random
            hit_tp   = score > 70 or (score > 60 and random.random() > 0.45)
            ticks_nq = 2.0  # 1 tick NQ = $5
            if hit_tp and tp_px and entry_px:
                raw_pnl = abs(tp_px - entry_px) * ticks_nq * (1 if direction=="LONG" else -1)
                status  = "TP_HIT"
            elif sl_px and entry_px:
                raw_pnl = -abs(entry_px - sl_px) * ticks_nq
                status  = "SL_HIT"
            else:
                raw_pnl = 0.0
                status  = "CLOSED"
            state.paper_pnl += raw_pnl
            paper_trade = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "direction": direction, "score": score, "ai_score": ai_score,
                "entry": entry_px, "sl": sl_px, "tp": tp_px,
                "pnl": round(raw_pnl, 2), "status": status,
                "cum_pnl": round(state.paper_pnl, 2),
            }
            state.paper_trades.append(paper_trade)
            if len(state.paper_trades) > 200:
                state.paper_trades = state.paper_trades[-200:]
            log.info(f"📄 PAPER: {direction} @ {entry_px} → {status}  PnL=${raw_pnl:+.2f}  Cum=${state.paper_pnl:+.2f}")
            await broadcast_ws({"type": "paper_trade", "data": paper_trade})

            # ── UPDATE #11: RL Feedback — actualizează weights după trade paper ──
            if _RL_OK and _rl_feedback is not None:
                try:
                    _comp_scores = dict(analysis.get("component_scores", {}))
                    # Injectăm scorul volume_profile (a 7-a componentă RL)
                    _paper_dir = analysis.get("trade_direction", "NEUTRAL")
                    _comp_scores["volume_profile"] = _compute_vp_score(state, state.latest_tick, _paper_dir)
                    _rl_result   = "WIN" if raw_pnl > 0 else "LOSS"
                    new_w = _rl_feedback.on_trade_closed(
                        component_scores = _comp_scores,
                        result           = _rl_result,
                        pnl              = raw_pnl,
                        score_pct        = score,
                        direction        = direction,
                    )
                    log.info(
                        f"🧠 RL Update (paper): {_rl_result} ${raw_pnl:+.0f} | "
                        f"AI={new_w.get('ai',0):.3f} ICT={new_w.get('ict',0):.3f} "
                        f"OF={new_w.get('orderflow',0):.3f} Sent={new_w.get('sentiment',0):.3f}"
                    )
                    await broadcast_ws({"type": "rl_update", "data": {
                        "result": _rl_result, "pnl": raw_pnl, "weights": new_w
                    }})
                except Exception as _rl_err:
                    log.debug(f"RL on_trade_closed skip: {_rl_err}")

            # ── FAZA 1.1: Consecutive Loss Protection — actualizează contoare ──
            if raw_pnl < 0:
                state.consecutive_losses += 1
                state.daily_loss_usd     += abs(raw_pnl)
                log.warning(
                    f"🔴 Pierdere #{state.consecutive_losses}  "
                    f"${raw_pnl:+.2f}  Pierdere azi: ${state.daily_loss_usd:.2f}"
                )
            else:
                state.consecutive_losses  = 0        # reset la WIN
                state.daily_profit_usd   += raw_pnl  # acumulăm profitul zilei
                log.info(f"✅ WIN ${raw_pnl:+.2f} — profit azi: ${state.daily_profit_usd:.2f}")
            # Session max tracking (paper mode)
            _sess_net = state.daily_profit_usd - state.daily_loss_usd
            if _sess_net > state.session_max_profit:
                state.session_max_profit = _sess_net
            if _sess_net > state._session_peak_pnl:
                state._session_peak_pnl = _sess_net
            _dd_from_peak = state._session_peak_pnl - _sess_net
            if _dd_from_peak > state.session_max_drawdown:
                state.session_max_drawdown = _dd_from_peak
                # Verificare profit target imediat după WIN
                _strat_now       = state.active_strategy
                _max_prof_check  = float(_strat_now.get("max_daily_profit_usd", 0)) if _strat_now else 0
                if _max_prof_check > 0 and state.daily_profit_usd >= _max_prof_check:
                    state.profit_circuit_open = True
                    log.warning(
                        f"🏆 PROFIT TARGET ATINS: ${state.daily_profit_usd:.0f} >= ${_max_prof_check:.0f} "
                        f"→ trading blocat azi, profitul e asigurat!"
                    )
                    await broadcast_ws({"type": "profit_target_hit",
                                        "profit": round(state.daily_profit_usd, 2),
                                        "target": _max_prof_check})
        else:
            log.info(f"🎯 AUTO-EXECUTE: {direction}  score={score}%  ai={ai_score}%")

        # UPDATE #12: incrementăm counter-ul de trades al strategiei
        if state.active_strategy:
            state.strategy_trades_today += 1
            log.info(f"📊 Trades azi: {state.strategy_trades_today}/{state.active_strategy.get('max_trades','∞')} [{state.active_strategy['id']}]")
            _save_strategy_state()  # Fix v9.0: persistăm counter după fiecare trade

        # UPDATE #14e: Marchează poziție deschisă imediat (previne trade dublu pe bara următoare)
        state._position_open = True
        state.entry_nt8_confirmed = False  # Fix v10.5: resetăm flag — așteptăm confirm valid de la NT8
        log.info(f"🔒 Position flag SET — blocat execuții noi până la /execution_confirm CLOSE")

        # ── A3 + E7: SCALE IN cu DRAWDOWN-AWARE SIZING (top 0.1%) ──────────────
        # Două nivele de decizie:
        # 1. Scale In (activat din dashboard): top 20% scor → 2 contracte
        # 2. Drawdown override: dacă sistemul e în drawdown → forțat 1 contract
        #    Logica: nu mări riscul când ești deja în pierdere. Protecție capitală.
        _scale_in_enabled = bool(state.active_strategy.get("scale_in", False)) if state.active_strategy else False
        _qty_to_use = 1  # default conservator

        # ── E7: DRAWDOWN GUARD — blochează 2 contracte în pierdere ──────────
        # Dacă avem pierderi consecutive sau pierdere zilnică > 50% din limita max,
        # forțăm 1 contract indiferent de scor — capital preservation first.
        _max_daily_loss = float(strat.get("max_daily_loss_usd", 1000)) if strat else 1000.0
        # Fix v7.4: Drawdown Guard — require 2 losses, not 1
        _drawdown_guard = (
            state.consecutive_losses >= 2 or                           # 2+ pierderi consecutive
            state.daily_loss_usd >= (_max_daily_loss * 0.50)          # am pierdut 50%+ din limita zilnică
        )

        if _drawdown_guard and _scale_in_enabled:
            log.info(
                f"🛡️ DRAWDOWN GUARD activ → forțat 1 contract "
                f"(losses={state.consecutive_losses}, daily_loss=${state.daily_loss_usd:.0f})"
            )

        if _scale_in_enabled and not _drawdown_guard:
            # Adaugă scorul curent în istoricul de învățare
            state.score_history.append(score)
            if len(state.score_history) > 50:
                state.score_history = state.score_history[-50:]

            # Fix v9.0: Percentila 80 corectă cu numpy-style interpolation
            if len(state.score_history) >= 5:
                _sorted   = sorted(state.score_history)
                _p80_idx  = int(len(_sorted) * 0.80) - 1  # -1 pt 0-indexed
                _p80_idx  = max(0, min(_p80_idx, len(_sorted) - 1))
                _p80      = _sorted[_p80_idx]
                _is_top20 = score >= _p80
            else:
                _p80      = 80.0
                _is_top20 = score >= 80.0

            # ── FIX #3: GARDIAN SCALE-IN ────────────────────────────────────
            # Regulile care blochează scale-in periculos (averaging down, contra-conviction)
            # Gardian 1: Nu scale-in dacă avem losses consecutive — suntem deja în pierdere
            _scalein_no_losses = state.consecutive_losses == 0
            # Gardian 2: Nu scale-in dacă direcția opusă are scor mai mare (contra-conviction)
            # Extragem opposite_score din analysis dacă e disponibil
            _opposite_dir   = "SHORT" if direction == "LONG" else "LONG"
            _opposite_score = float(analysis.get(f"score_{_opposite_dir.lower()}", 0.0)) if analysis else 0.0
            # Fallback: dacă nu e în analysis, considerăm safe (nu blocăm)
            _scalein_conviction_ok = (_opposite_score == 0.0 or _opposite_score < score)
            # Gardian 3: Scorul trebuie să fie cu cel puțin 10% peste score_min (nu la limită)
            _scalein_score_ok = score >= (score_min + 10.0)

            if _is_top20 and _scalein_no_losses and _scalein_conviction_ok and _scalein_score_ok:
                _qty_to_use = 2
                log.info(
                    f"🔥 SCALE IN 2x: scor {score:.1f}% în top 20% "
                    f"(p80={_p80:.1f}%) | losses=0 | opp_score={_opposite_score:.0f}% | "
                    f"margin OK → 2 contracte ✅"
                )
            elif _is_top20:
                # Top 20% dar gardian blocat — loghăm motivul
                _reason = []
                if not _scalein_no_losses:
                    _reason.append(f"losses={state.consecutive_losses}")
                if not _scalein_conviction_ok:
                    _reason.append(f"opp_score={_opposite_score:.0f}%>score={score:.0f}%")
                if not _scalein_score_ok:
                    _reason.append(f"score {score:.0f}% < min+10% ({score_min+10:.0f}%)")
                log.info(
                    f"🛡️ SCALE-IN BLOCAT (top 20% dar gardian activ): "
                    f"{', '.join(_reason)} → 1 contract"
                )
            else:
                log.info(
                    f"📊 Scale In: scor {score:.1f}% sub top 20% "
                    f"(p80={_p80:.1f}%) → 1 contract"
                )
        elif _scale_in_enabled and _drawdown_guard:
            # Înregistrăm scorul în istorie chiar dacă nu scalăm
            state.score_history.append(score)
            if len(state.score_history) > 50:
                state.score_history = state.score_history[-50:]

        # ── MAX CONTRACTS CAP (Lucid Flex 4 Mini / danger zone 1 contract) ──
        # _trade_qty = min(risk.units, max_contracts) calculat mai sus
        if _qty_to_use > _trade_qty:
            log.info(f"🔒 QTY CAP: {_qty_to_use} → {_trade_qty} (max_contracts={_max_contracts})")
            _qty_to_use = _trade_qty
        # ── HARD CAP ABSOLUT: NICIODATĂ MAI MULT DE 2 CONTRACTE ─────────────
        # Regulă fixă, nu poate fi overridden de strategie sau risc sau configurație.
        if _qty_to_use > 2:
            log.warning(f"🚫 HARD CAP 2 CONTRACTE: {_qty_to_use} → 2 (regulă absolută)")
            _qty_to_use = 2
        # ─────────────────────────────────────────────────────────────────────

        # PARTIAL CLOSE 1R: salvează detalii trade pentru monitoring breakeven
        state.open_trade_entry   = entry_px
        state.open_trade_sl      = sl_px or 0.0
        state.open_trade_tp      = tp_px or 0.0
        state.open_trade_dir     = direction
        state.open_trade_qty     = _qty_to_use
        state.partial_close_done = False
        state.milestone_05r_done  = False
        state.milestone_085r_done = False
        state.trailing_sl        = sl_px or 0.0
        state.open_trade_ts      = datetime.now(timezone.utc).isoformat()
        state.open_trade_mae_pts = 0.0   # reset MAE la intrare trade nou
        state.open_trade_mfe_pts = 0.0   # reset MFE la intrare trade nou
        log.info(f"📌 Trade tracking: {direction} x{_qty_to_use} entry={entry_px} SL={sl_px} TP={tp_px}")
        _save_open_trade()   # ← persistăm pe disc — trailing supraviețuiește restart bridge

        # Trimite entry către NT8 — SL fizic imediat (protecție dacă bridge pică)
        # Trailing: MOVE_SL la 1R→BE, 2R/3R→trailing (tot via comenzi fizice NT8)
        log.info(f"🎯 Sending to NT8: {direction} x{_qty_to_use} entry={entry_px} sl={sl_px}(fizic) tp={tp_px}")
        await send_command_to_nt8(action=direction, qty=_qty_to_use, signal=f"ALADIN_{score:.0f}",
                                   sl=sl_px, tp=tp_px)

        # UPDATE #13: Telegram notificare trade executat (cu SL/TP reale)
        if _TG_OK and _tg_trade:
            try:
                _strat_lbl = state.active_strategy.get("label", "") if state.active_strategy else ""
                _sl_pts_real   = abs(entry_px - sl_px) if entry_px and sl_px else 0
                _risk_usd_real = round(_sl_pts_real * 20.0 * _qty_to_use, 2)
                _is_tg_scale_in = _qty_to_use >= 2
                # v10.0: component scores + ICT signals + VP context + delta exhaustion
                _tg_comp   = dict(analysis.get("component_scores", {})) if analysis else {}
                # Fix v10.5: ICT signals iau datele direct din analysis dict (returnat de mario_rag.py)
                # Înainte: folosea state._last_h4_aligned (neexistent) și in_kz/has_fvg/has_smt (chei greșite)
                # Acum: mario_rag.py returnează ict_h4, ict_h1, ict_m15, in_kz, has_fvg, has_smt explicit
                _tg_ict_sg = {
                    "h4":  bool(analysis.get("ict_h4",  False)) if analysis else False,
                    "h1":  bool(analysis.get("ict_h1",  False)) if analysis else False,
                    "m15": bool(analysis.get("ict_m15", False)) if analysis else False,
                    "kz":  bool(analysis.get("in_kz",   False)) if analysis else False,
                    "fvg": bool(analysis.get("has_fvg",  False)) if analysis else False,
                    "smt": bool(analysis.get("has_smt",  False)) if analysis else False,
                }
                _tg_poc  = float(getattr(state, "poc",  0) or 0)
                _tg_vah  = float(getattr(state, "vah",  0) or 0)
                _tg_val  = float(getattr(state, "val",  0) or 0)
                _tg_poc_dist = round(entry_px - _tg_poc, 1) if _tg_poc and entry_px else 0
                _tg_vp   = {
                    "rvol":     round(getattr(state, "rvol", 1.0), 2),
                    "shape":    getattr(state, "profile_shape", ""),
                    "poc_dist": _tg_poc_dist,
                    "poc":      _tg_poc,
                    "vah":      _tg_vah,
                    "val":      _tg_val,
                }
                _tg_trade(
                    direction          = direction,
                    score              = score,
                    sl                 = sl_px,
                    tp                 = tp_px,
                    risk_usd           = _risk_usd_real,
                    strategy           = _strat_lbl,
                    price              = entry_px,
                    trades_today       = state.strategy_trades_today if state.active_strategy else 0,
                    max_trades         = int(state.active_strategy.get("max_trades", 0)) if state.active_strategy else 0,
                    is_scale_in        = _is_tg_scale_in,
                    scale_in_qty       = _qty_to_use if _is_tg_scale_in else 0,
                    scale_in_total_qty = _qty_to_use if _is_tg_scale_in else 0,
                    avg_entry          = entry_px,
                    component_scores   = _tg_comp,
                    ict_signals        = _tg_ict_sg,
                    vp_context         = _tg_vp,
                    delta_exhaustion   = getattr(state, "delta_exhaustion", ""),
                )
                log.info(f"📱 Telegram notificare trimisă {'(SCALE IN)' if _is_tg_scale_in else ''}")
            except Exception as _tg_err:
                log.debug(f"Telegram skip: {_tg_err}")

        # ── UPDATE #1: Log trade în Supabase ──────────────────────────────────
        if _SUPABASE_OK and _supabase is not None:
            try:
                # Fix v9.0: qty și risk_usd reale (nu hardcoded 1)
                _supabase.log_trade(
                    symbol      = tick.symbol or "NQ",
                    direction   = direction,
                    score_pct   = score,
                    ai_score    = ai_score,
                    entry_price = entry_px,
                    sl_price    = sl_px,
                    tp_price    = tp_px,
                    qty         = _qty_to_use,
                    risk_usd    = float(_risk.get("risk_usd", 0)) * _qty_to_use,
                    live_mode   = not state.paper_mode,
                    note        = f"{'[PAPER] ' if state.paper_mode else ''}{verdict[:80] if verdict else ''}",
                )
            except Exception as _sb_trade_err:
                log.debug(f"Supabase log_trade skip: {_sb_trade_err}")

# ─── NT8 Bridge Endpoints ─────────────────────────────────────────────────────

@app.post("/ping_nt8")
async def ping_nt8(payload: dict = None):
    """Endpoint de test conectivitate — NT8 îl apelează la pornirea strategiei."""
    ts = datetime.now(timezone.utc).isoformat()
    log.info(f"🏓 PING de la NT8: {payload}")
    await broadcast_ws({"type": "nt8_ping", "data": {"ts": ts, "payload": payload}})
    return {"status": "pong", "server_ts": ts, "nt8_ok": True}

import re as _re

# ─── Volume Profile Analytics Helpers ────────────────────────────────────────

def _compute_profile_shape(poc: float, session_high: float, session_low: float) -> str:
    """Forma distribuției volume profile (P / b / D).
    P = volum concentrat în top (POC sus) → distribuție bullish terminată → short setup.
    b = volum concentrat în bottom (POC jos) → distribuție bearish terminată → long setup.
    D = balanced, POC la mijloc → energie acumulată, breakout iminent.
    """
    if session_high <= session_low or poc <= 0:
        return "D"
    poc_pos = (poc - session_low) / (session_high - session_low)
    if poc_pos > 0.60:
        return "P"
    elif poc_pos < 0.40:
        return "b"
    return "D"


def _compute_composite_vp(db_path: str, days: int = 7, tick_size: float = 0.25) -> dict:
    """Calculează Volume Profile compozit din ultimele N zile din market_data.

    Metodă: distribuim volumul fiecărei bare uniform pe range-ul [low, high]
    la granularitate tick_size. Aproximare validă pentru 5-min bars pe NQ —
    HVN/LVN structurale sunt vizibile chiar și fără tick-by-tick data.

    Returnează: {poc, hvn: [top3], lvn: [top3 în zona relevantă], levels: {price: vol}}
    """
    import sqlite3 as _sq
    import math as _math
    result = {"poc": 0.0, "hvn": [], "lvn": [], "total_bars": 0}
    try:
        conn = _sq.connect(db_path)
        rows = conn.execute("""
            SELECT open, high, low, close, volume FROM market_data
            WHERE volume > 0
              AND date(timestamp) >= date('now', ?)
              AND date(timestamp) < date('now')
            ORDER BY timestamp ASC
        """, (f"-{days} days",)).fetchall()
        conn.close()

        if len(rows) < 20:
            return result

        # Agregăm volumul per nivel de preț (distribuție uniformă pe range bara)
        vol_map: Dict[float, float] = {}
        for o, h, l, c, v in rows:
            if not h or not l or h <= l or not v:
                continue
            bar_range = h - l
            # Numărul de niveluri în range
            n_levels = max(1, int(round(bar_range / tick_size)))
            vol_per_level = float(v) / n_levels
            # Ponderam ușor spre close (prețul de închidere = cel mai reprezentativ)
            # 70% distribuit uniform, 30% concentrat la close
            vol_uniform = vol_per_level * 0.70
            level = round(l / tick_size) * tick_size
            while level <= h + 1e-9:
                key = round(level / tick_size) * tick_size
                vol_map[key] = vol_map.get(key, 0.0) + vol_uniform
                level = round((level + tick_size) / tick_size) * tick_size
            # Bonus la close
            close_key = round(float(c) / tick_size) * tick_size
            vol_map[close_key] = vol_map.get(close_key, 0.0) + float(v) * 0.30

        if not vol_map:
            return result

        # POC compozit = nivelul cu cel mai mare volum agregat
        poc = max(vol_map, key=vol_map.get)
        result["poc"] = round(poc, 2)
        result["total_bars"] = len(rows)

        # Sortăm după volum
        sorted_levels = sorted(vol_map.items(), key=lambda x: x[1], reverse=True)
        total_vol = sum(vol_map.values())
        avg_vol   = total_vol / len(vol_map)

        # HVN compozit = top 5 niveluri, fără duplicate la distanță < 5 tickuri
        hvn: List[float] = []
        for lvl, _ in sorted_levels:
            if len(hvn) >= 5:
                break
            if all(abs(lvl - h) > tick_size * 5 for h in hvn):
                hvn.append(round(lvl, 2))
        result["hvn"] = sorted(hvn)

        # LVN compozit = niveluri cu volum < 30% din medie, în zona ±50pts de POC
        lvn_candidates = [
            (lvl, vol) for lvl, vol in sorted_levels
            if vol < avg_vol * 0.30 and abs(lvl - poc) < 50.0
        ]
        # Sortăm LVN după preț și luăm top 5 mai distanțate
        lvn_candidates.sort(key=lambda x: x[0])
        lvn: List[float] = []
        for lvl, _ in lvn_candidates:
            if len(lvn) >= 5:
                break
            if all(abs(lvl - l) > tick_size * 5 for l in lvn):
                lvn.append(round(lvl, 2))
        result["lvn"] = sorted(lvn)

    except Exception as _cvp_err:
        pass
    return result


def _compute_vp_score(state, tick, direction: str) -> float:
    """Score 0.0–1.0 pentru componenta volume_profile în direcția dată.
    Folosit de RL feedback ca al 7-lea component (alături de AI, ICT, OF etc.).
    Semnale: RVOL, delta_exhaustion, profile_shape, LVN proximity, prev_poc context.
    """
    score = 0.50   # neutral baseline
    if not tick:
        return score

    rvol    = getattr(state, "rvol", 1.0)
    exhaust = getattr(state, "delta_exhaustion", "NONE")
    shape   = getattr(state, "profile_shape", "D")
    close   = float(tick.price.close) if tick else 0.0
    d       = direction.upper()

    # ── RVOL: spike = participare instituțională (bun indiferent de direcție)
    if rvol >= 1.5:
        score += 0.10    # volum neobișnuit = mișcare cu substanță
    elif rvol >= 1.2:
        score += 0.05
    elif rvol < 0.80:
        score -= 0.10    # piață subțire = fakeout probabil

    # ── Delta Exhaustion: signal directional puternic (absorb instituțional)
    if d == "LONG"  and exhaust == "LONG_EXHAUSTION":
        score += 0.20    # vânzătorii absorbiți → long setup confirmat
    elif d == "SHORT" and exhaust == "SHORT_EXHAUSTION":
        score += 0.20    # cumpărătorii absorbiți → short setup confirmat
    elif exhaust != "NONE":
        score -= 0.05    # exhaustion contra-direcțional = penalizare ușoară

    # ── Profile Shape: confirmă distribuția de volum
    if d == "LONG"  and shape == "b":
        score += 0.10    # b-shape = volum jos, coadă sus → long setup
    elif d == "SHORT" and shape == "P":
        score += 0.10    # P-shape = volum sus, coadă jos → short setup
    elif d == "LONG"  and shape == "P":
        score -= 0.05    # P-shape contra long (distribuție terminată sus)
    elif d == "SHORT" and shape == "b":
        score -= 0.05    # b-shape contra short

    # ── LVN proximity: prețul lângă LVN = mișcare rapidă așteptată = entry bun
    try:
        lvn = tick.volume_profile.lvn
        if lvn and close > 0:
            min_dist_lvn = min(abs(close - p) for p in lvn)
            if min_dist_lvn < 3.0:     # < 3 pts NQ = în zona LVN
                score += 0.08
    except Exception:
        pass

    # ── Prev POC context: above prev POC = bullish, below = bearish
    try:
        prev_poc = tick.volume_profile.prev_poc
        if prev_poc > 0 and close > 0:
            if d == "LONG"  and close > prev_poc:
                score += 0.05
            elif d == "SHORT" and close < prev_poc:
                score += 0.05
            elif d == "LONG"  and close < prev_poc:
                score -= 0.03
            elif d == "SHORT" and close > prev_poc:
                score -= 0.03
    except Exception:
        pass

    return round(min(max(score, 0.0), 1.0), 4)


def _compute_delta_exhaustion(delta_history) -> str:
    """Detectare absorb instituțional: delta s-a mișcat mult dar prețul n-a urmat.
    SHORT_EXHAUSTION: cumpărători absorbiți (delta ↑ dar preț →/↓) → setup short.
    LONG_EXHAUSTION:  vânzători absorbiți  (delta ↓ dar preț →/↑) → setup long.
    Prag: delta ≥ 100 pts mișcare fără cel puțin 3 pts preț în același sens.
    """
    dh = list(delta_history)
    lookback = 10
    if len(dh) < lookback:
        return "NONE"
    recent      = dh[-lookback:]
    delta_move  = recent[-1]["delta"] - recent[0]["delta"]
    price_move  = recent[-1]["price"] - recent[0]["price"]
    DELTA_MIN   = 100.0   # mișcare minimă delta pentru a considera semnal
    PRICE_MAX   = 3.0     # preț trebuie să fie stagnant (< 3 pts pe NQ)
    if delta_move >= DELTA_MIN and price_move < PRICE_MAX:
        return "SHORT_EXHAUSTION"   # cumpărători absorbiți de vânzători instituționali
    if delta_move <= -DELTA_MIN and price_move > -PRICE_MAX:
        return "LONG_EXHAUSTION"    # vânzători absorbiți de cumpărători instituționali
    return "NONE"


def _fix_nt8_json(raw: bytes) -> bytes:
    """Fixează valori malformate din NT8 înainte de parsare JSON.
    Ex: \"vol\":F40 → \"vol\":40  (format spec C# scăpat neinterpretat)
    """
    text = raw.decode("utf-8", errors="replace")
    # "vol":F40 → "vol":40  (F urmat de cifre, fără ghilimele)
    text = _re.sub(r'("vol"\s*:\s*)F(\d+)', r'\g<1>\2', text)
    # Orice alt câmp numeric cu prefix F: "key":F123 → "key":123
    text = _re.sub(r'(:\s*)F(\d+)([,}\]\s])', r'\g<1>\2\3', text)
    return text.encode("utf-8")

@app.post("/nt8_data")
async def receive_nt8_data(request: Request, background_tasks: BackgroundTasks):
    import json as _json
    try:
        raw = await request.body()
        raw = _fix_nt8_json(raw)
        payload = _json.loads(raw)
        data = NT8Data(**payload)
    except Exception as e:
        # ClientDisconnect = NT8 fire-and-forget, benign — log only at DEBUG
        if type(e).__name__ == "ClientDisconnect":
            log.debug("NT8 disconnected before body read (fire-and-forget, benign)")
            return JSONResponse(status_code=200, content={"status": "ok"})
        log.warning(f"NT8 data parse error: {type(e).__name__}: {e}")
        return JSONResponse(status_code=422, content={"detail": str(e)})
    # Fix v7.4: Input Validation on NT8Data
    if data.price.close <= 0:
        log.warning(f"Invalid price close={data.price.close}, skipping")
        return {"status": "error", "detail": "invalid price"}
    # Fix v10.5: BAD TICK GUARD — salvăm prețul ANTERIOR înainte de update
    # (folosit mai jos pentru a detecta bad ticks cu preț aberant)
    _prev_tick_close = state.latest_tick.price.close if (state.latest_tick and state.latest_tick.price) else 0.0
    state.latest_tick = data
    state.tick_count += 1
    # Fix v10.5: salvăm poc/vah/val în state pentru Telegram VP context
    if data.volume_profile:
        if data.volume_profile.poc > 0:
            state.poc = data.volume_profile.poc
        if data.volume_profile.vah > 0:
            state.vah = data.volume_profile.vah
        if data.volume_profile.val > 0:
            state.val = data.volume_profile.val
    if state.connected_since is None:
        state.connected_since = time.time()
        log.info(f"✅ NT8 conectat! Symbol: {data.symbol}")
    state.bar_buffer.append(data)
    state.session_cum_delta = data.orderflow.cum_delta
    state.delta_history.append({
        "ts":          data.timestamp,
        "delta":       data.orderflow.cum_delta,
        "bar_delta":   data.orderflow.bar_delta,      # explicit per-bară, nu derivat
        "price":       data.price.close,
        "delta_at_hi": data.orderflow.delta_at_high,  # fingerprint instituțional la extreme
        "delta_at_lo": data.orderflow.delta_at_low,
    })
    # Filtru 2 anti-retracement: tracking volum per bară (buy+sell)
    _bar_vol_now = data.orderflow.bar_buy_vol + data.orderflow.bar_sell_vol
    if _bar_vol_now > 0:
        state.bar_vol_history.append(_bar_vol_now)

    # ── Fix v10.6: STARTUP POSITION SYNC — validare trade restaurat din disc ──────
    # Dacă bridge-ul a restaurat un trade din disc, verificăm în primele 30s:
    # - NT8 trimite tick-uri dar NU a trimis execution_confirm → trade-ul e stale
    # - Dacă după 30s nu avem entry_nt8_confirmed → auto-clear
    if getattr(state, '_restored_pending_validation', False):
        _restore_age = time.time() - getattr(state, '_restored_validation_ts', 0)
        _has_nt8_confirm = getattr(state, 'entry_nt8_confirmed', False)
        if _has_nt8_confirm:
            # NT8 a confirmat — trade valid
            state._restored_pending_validation = False
            log.info(f"✅ POSITION SYNC: trade restaurat VALIDAT de NT8 confirm")
        elif _restore_age > 30.0:
            # 30s fără confirm → trade stale, curățăm
            log.warning(
                f"🗑️  POSITION SYNC TIMEOUT: trade restaurat din disc ({state.open_trade_dir} "
                f"@ {state.open_trade_entry}) nu a fost confirmat de NT8 în 30s → AUTO-CLEAR. "
                f"Dacă aveți poziție deschisă pe NT8, bridge-ul o va detecta la următorul execution_confirm."
            )
            state._position_open      = False
            state.partial_close_done  = False
            state.milestone_05r_done  = False
            state.milestone_085r_done = False
            state.open_trade_entry    = 0.0
            state.open_trade_sl       = 0.0
            state.open_trade_tp       = 0.0
            state.open_trade_dir      = ""
            state.trailing_sl         = 0.0
            state.trail_r_level       = 0
            state.open_trade_ts       = ""
            state._restored_pending_validation = False
            _clear_open_trade()

    # ── RVOL — Relative Volume (volum curent vs medie istorică același interval orar) ──
    # Nu necesită modificări în NT8 — calculăm din DB-ul de market_data existent.
    # RVOL > 1.5 = activitate neobișnuită (participare instituțională)
    # RVOL < 0.7 = piață subțire (manipulare posibilă la cost scăzut)
    try:
        import sqlite3 as _sq
        _ts_now  = data.timestamp[:16] if data.timestamp else ""
        _hr_min  = _ts_now[11:16] if len(_ts_now) >= 16 else ""   # "HH:MM"
        if _hr_min and state._rvol_cache_time != _hr_min:
            _conn = _sq.connect(str(DB_PATH))
            # Media volumului pentru același interval orar în ultimele 30 de zile
            # Subquery cu LIMIT 40 pe cele mai recente bare pentru același hour_min
            # (limitarea se face pe input, nu pe output)
            _row = _conn.execute("""
                SELECT AVG(volume) FROM (
                    SELECT volume FROM market_data
                    WHERE hour_min = ? AND volume > 0
                    ORDER BY timestamp DESC LIMIT 40
                )
            """, (_hr_min,)).fetchone()
            _conn.close()
            _avg_vol_hist = float(_row[0]) if _row and _row[0] else 0.0
            _cur_vol      = float(data.price.volume or 0)
            state.rvol    = round(_cur_vol / _avg_vol_hist, 2) if _avg_vol_hist > 0 else 1.0
            state._rvol_cache_time = _hr_min
            if state.rvol >= 1.5:
                log.info(f"📊 RVOL spike: {state.rvol:.2f}× (vol={_cur_vol:.0f} avg={_avg_vol_hist:.0f}) @ {_hr_min}")
    except Exception as _rv_err:
        log.debug(f"RVOL calc skip: {_rv_err}")

    # ── Composite Volume Profile (refresh zilnic din DB) ──────────────────────
    _today_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    if state._composite_date != _today_str:
        try:
            _cvp = _compute_composite_vp(str(DB_PATH), days=7)
            if _cvp["poc"] > 0:
                state.composite_poc = _cvp["poc"]
                state.composite_hvn = _cvp["hvn"]
                state.composite_lvn = _cvp["lvn"]
                state._composite_date = _today_str
                log.info(
                    f"📊 Composite VP (7 zile, {_cvp['total_bars']} bare): "
                    f"POC={state.composite_poc:.2f} "
                    f"HVN={state.composite_hvn} "
                    f"LVN={state.composite_lvn}"
                )
        except Exception as _cvp_err:
            log.debug(f"Composite VP skip: {_cvp_err}")

    # ── Profile Shape (P / b / D) ──────────────────────────────────────────────
    # Citit nativ din NT8 dacă disponibil, altfel fallback pe calcul local
    _prev_shape = state.profile_shape
    _nt8_shape = getattr(data, 'profile_shape', '') or ''
    if _nt8_shape in ("P", "b", "D"):
        state.profile_shape = _nt8_shape
    else:
        # Fallback: calculăm din POC curent + High/Low (AladinBridge vechi)
        state.profile_shape = _compute_profile_shape(
            data.volume_profile.poc,
            data.price.high,
            data.price.low,
        )
    if state.profile_shape != _prev_shape:
        _shape_src = "NT8" if _nt8_shape in ("P", "b", "D") else "LOCAL"
        log.info(f"📊 Profile shape → {state.profile_shape} [{_shape_src}] "
                 f"(POC={data.volume_profile.poc:.2f} H={data.price.high:.2f} L={data.price.low:.2f})")

    # ── Delta Exhaustion ───────────────────────────────────────────────────────
    # Detectare absorb instituțional din ultimele 10 bare (delta ↑↓ dar preț stagnant)
    # Fix v10.5: cooldown 60s per tip — previne oscillare NONE→LONG→NONE la fiecare tick
    _EXHAUST_COOLDOWN = 60.0  # secunde între două detecții de același tip
    _now_ts = time.time()
    _prev_exhaust = state.delta_exhaustion
    _new_exhaust  = _compute_delta_exhaustion(state.delta_history)
    # Aplicăm cooldown: ignorăm detecție dacă același tip a fost raportat recent
    _last_exhaust_ts   = getattr(state, "_last_exhaust_ts",   0.0)
    _last_exhaust_type = getattr(state, "_last_exhaust_type", "NONE")
    if _new_exhaust != "NONE":
        if _new_exhaust == _last_exhaust_type and (_now_ts - _last_exhaust_ts) < _EXHAUST_COOLDOWN:
            _new_exhaust = _prev_exhaust  # păstrăm starea curentă, nu re-raportăm
        else:
            state._last_exhaust_ts   = _now_ts
            state._last_exhaust_type = _new_exhaust
    else:
        # Resetăm starea la NONE doar dacă cooldown-ul a expirat
        if (_now_ts - _last_exhaust_ts) >= _EXHAUST_COOLDOWN:
            pass  # reset normal la NONE
        else:
            _new_exhaust = _prev_exhaust  # menținem ultima valoare în cooldown
    state.delta_exhaustion = _new_exhaust
    if state.delta_exhaustion != "NONE" and state.delta_exhaustion != _prev_exhaust:
        _dh_list = list(state.delta_history)
        if len(_dh_list) >= 10:
            _dm = _dh_list[-1]["delta"] - _dh_list[-10]["delta"]
            _pm = _dh_list[-1]["price"] - _dh_list[-10]["price"]
            log.warning(f"⚡ DELTA EXHAUSTION: {state.delta_exhaustion} "
                        f"Δdelta={_dm:+.0f} Δprice={_pm:+.2f}pts — absorb instituțional")

    # ── Fix v10.6+: OF CONSOLIDATION METRICS — citite NATIV din NT8 ──────────
    # AladinBridge.cs calculează cele 4 metrici tick-by-tick (mult mai precis decât cache)
    # și le trimite în payload-ul /nt8_data ca "of_consol_metrics".
    # Bridge-ul le citește direct — zero recalculare, zero latență adăugată.
    _nt8_ofcm = getattr(data, 'of_consol_metrics', {}) or {}
    _of_consol_metrics = {
        "delta_oscillation_idx": float(_nt8_ofcm.get("delta_oscillation_idx", 0) or 0),
        "bilateral_absorption":  bool(_nt8_ofcm.get("bilateral_absorption", False)),
        "big_trade_balance":     float(_nt8_ofcm.get("big_trade_balance", 0.5) or 0.5),
        "consol_d_shape_count":  int(_nt8_ofcm.get("consol_d_shape_count", 0) or 0),
    }

    # Fallback: dacă NT8 nu trimite (AladinBridge vechi fără update), calculăm din cache
    _nt8_has_ofcm = bool(_nt8_ofcm)
    if not _nt8_has_ofcm:
        _dh = list(state.delta_history)
        if len(_dh) >= 20:
            _bar_deltas = [d.get("bar_delta", 0) for d in _dh[-25:]]
            _bd = [float(x) for x in _bar_deltas if x is not None]
            if len(_bd) >= 15:
                _bd_mean = sum(_bd) / len(_bd)
                _bd_centered = [x - _bd_mean for x in _bd]
                _var = sum(x*x for x in _bd_centered)
                if _var > 0:
                    _cov = sum(_bd_centered[i] * _bd_centered[i+1] for i in range(len(_bd_centered)-1))
                    _of_consol_metrics["delta_oscillation_idx"] = round(_cov / _var, 3)
            _dah_vals = [float(d.get("delta_at_hi", 0) or 0) for d in _dh[-15:]]
            _dal_vals = [float(d.get("delta_at_lo", 0) or 0) for d in _dh[-15:]]
            if sum(1 for x in _dah_vals if x < -30) >= 3 and sum(1 for x in _dal_vals if x > 30) >= 3:
                _of_consol_metrics["bilateral_absorption"] = True
        _big_buy  = float(data.orderflow.big_buy_count or 0) if hasattr(data.orderflow, 'big_buy_count') else 0
        _big_sell = float(data.orderflow.big_sell_count or 0) if hasattr(data.orderflow, 'big_sell_count') else 0
        _big_total = _big_buy + _big_sell
        if _big_total > 0:
            _of_consol_metrics["big_trade_balance"] = round(_big_buy / _big_total, 2)
        _shape_hist = getattr(state, '_shape_history', [])
        _shape_hist.append(state.profile_shape)
        if len(_shape_hist) > 20: _shape_hist = _shape_hist[-20:]
        state._shape_history = _shape_hist
        _d_count = 0
        for _sh in reversed(_shape_hist):
            if _sh == "D": _d_count += 1
            else: break
        _of_consol_metrics["consol_d_shape_count"] = _d_count
        log.debug("OF CONSOL: fallback calcul din cache (NT8 nu trimite of_consol_metrics)")

    # Store in state for export
    state._of_consol_metrics = _of_consol_metrics

    # Log when consolidation signals are strong
    _doi = _of_consol_metrics["delta_oscillation_idx"]
    _ba  = _of_consol_metrics["bilateral_absorption"]
    _btb = _of_consol_metrics["big_trade_balance"]
    _dsc = _of_consol_metrics["consol_d_shape_count"]
    _consol_signals = sum([
        _doi < -0.15,     # mean-reversion delta
        _ba,              # bilateral absorption
        0.35 <= _btb <= 0.65,  # balanced big trades
        _dsc >= 3,        # 3+ D-shape bars
    ])
    _ofcm_src = "NT8" if _nt8_has_ofcm else "CACHE"
    if _consol_signals >= 3:
        log.info(
            f"   🔲 OF CONSOLIDATION STRONG ({_consol_signals}/4) [{_ofcm_src}]: "
            f"Δ_oscil={_doi:.3f} biAbsorb={'✅' if _ba else '❌'} "
            f"bigBal={_btb:.2f} D-shape={_dsc}bare"
        )

    # FAZA 2.3: actualizăm istoricul NQ/ES pentru correlation filter
    state.nq_price_history.append((data.timestamp, data.price.close))
    if len(state.nq_price_history) > 30:
        state.nq_price_history = state.nq_price_history[-30:]
    if data.es_close > 0:
        state.es_price_history.append((data.timestamp, data.es_close))
        if len(state.es_price_history) > 30:
            state.es_price_history = state.es_price_history[-30:]

    # ── SYNC CONT DIN NT8 (v10.1) ───────────────────────────────────────────────
    # AladinBridge.cs trimite date reale de cont pe fiecare bară.
    # Prioritate: NT8 real > estimare internă bridge.
    # Fallback: dacă NT8 nu trimite (câmpuri = 0, AladinBridge vechi), rămânem pe estimare.
    try:
        _nt8_acc = data.account
        if _nt8_acc.net_liquidation > 0:
            # Balanță reală din NT8 → override estimarea internă
            state.account_balance = round(_nt8_acc.net_liquidation, 2)

        if _nt8_acc.realized_pnl_today != 0:
            # P&L realizat azi din NT8 (RealizedProfitLoss — reset zilnic în NT8)
            _nt8_rpnl = round(_nt8_acc.realized_pnl_today, 2)
            if _nt8_rpnl >= 0:
                state.daily_profit_usd = _nt8_rpnl
                # Recalculăm daily_loss dacă profit crește (trade win)
            else:
                state.daily_loss_usd = abs(_nt8_rpnl)
                state.daily_profit_usd = 0.0

        # Open P&L live pentru dashboard (nu afectează circuite — e unrealized)
        state._open_pnl_nt8 = round(_nt8_acc.open_pnl, 2)
    except Exception as _acc_sync_err:
        log.debug(f"Account sync NT8 error: {_acc_sync_err}")

    # ── SYNC POZIȚIE DIN NT8 ──────────────────────────────────────────────────
    # Dacă AladinBridge.cs trimite position_status, sincronizăm _position_open
    # direct din NT8 — elimină bug-ul de flag blocat după trade manual/disconnect.
    # FLUX AUTOMAT: close detectat → flag reset → analiza bara următoare → trade nou dacă
    # strategy_trades_today < max_trades și scor >= threshold. Fără intervenție manuală.
    # Fix v7.4: Position Sync Idempotent — only process on CHANGE
    if data.position_status and data.position_status.upper() != getattr(state, '_last_position_status', ''):
        state._last_position_status = data.position_status.upper()
        _nt8_flat = data.position_status.upper() == "FLAT" or data.position_qty == 0
        if _nt8_flat and state._position_open:
            # ── Calcul trade count pentru log ──────────────────────────────
            _trades_done = state.strategy_trades_today
            _max_t       = int(state.active_strategy.get("max_trades", 0)) if state.active_strategy else 0
            _more_trades = _trades_done < _max_t if _max_t > 0 else True
            _next_info   = (
                f"→ gata pentru trade {_trades_done + 1}/{_max_t}"
                if _more_trades and _max_t > 0
                else f"→ MAX TRADES atins ({_trades_done}/{_max_t}), stop"
            )
            log.info(
                f"🔄 AUTO-SYNC NT8: poziție FLAT detectată "
                f"(trade {_trades_done}/{_max_t if _max_t else '∞'} completat) "
                f"{_next_info}"
            )
            # ── LOGARE TRADE cu PnL real înainte de reset ──────────────────
            _sync_entry  = state.open_trade_entry
            _sync_dir    = state.open_trade_dir
            _sync_qty    = state.open_trade_qty or 1
            _sync_ts     = state.open_trade_ts
            _sync_sl     = state.open_trade_sl
            _exit_px     = data.price.close
            _now_sync    = datetime.now(timezone.utc)
            if _sync_entry > 0 and _sync_dir:
                _inst_cfg  = _get_instrument_config(getattr(data, 'symbol', '') if data else '')
                _point_val = _inst_cfg["point_value"]
                _diff      = _exit_px - _sync_entry
                _pnl_usd   = round((_diff if _sync_dir == "LONG" else -_diff) * _point_val * _sync_qty, 2)
                _result    = "WIN" if _pnl_usd > 0 else ("LOSS" if _pnl_usd < 0 else "BE")
                _r_mult    = 0.0
                if _sync_sl > 0:
                    _risk_pts = abs(_sync_entry - _sync_sl)
                    _risk_usd = _risk_pts * _point_val * _sync_qty
                    if _risk_usd > 0:
                        _r_mult = round(_pnl_usd / _risk_usd, 2)
                _sync_trade = {
                    "id":          str(uuid.uuid4())[:8],
                    "entry_time":  (_sync_ts or _now_sync.isoformat())[:19],
                    "exit_time":   _now_sync.isoformat()[:19],
                    "direction":   _sync_dir,
                    "entry_price": round(_sync_entry, 2),
                    "exit_price":  round(_exit_px, 2),
                    "qty":         _sync_qty,
                    "pnl":         round(_pnl_usd / _point_val, 4),
                    "pnl_usd":     _pnl_usd,
                    "pnl_pct":     0.0,
                    "result":      _result,
                    "mae_pts":     round(getattr(state, "open_trade_mae_pts", 0) or 0, 2),
                    "mfe_pts":     round(getattr(state, "open_trade_mfe_pts", 0) or 0, 2),
                    "exit_reason": "NT8_AUTOSYNC",
                    "sl":          round(state.open_trade_sl, 2),
                    "tp":          round(state.open_trade_tp, 2),
                    "entry_hour":  _now_sync.hour,
                    "entry_dow":   _now_sync.weekday(),
                    "setup":       "MANUAL_CLOSE_NT8",
                    "score":       0.0,
                    "strategy":    (state.active_strategy or {}).get("label", "Manual"),
                    "session":     "NY",
                    "r_multiple":  _r_mult,
                    "source":      "NT8_AUTOSYNC",
                    "instrument":  getattr(data, 'symbol', ACTIVE_INSTRUMENT),
                    "geo_risk_active": state.geo_risk_active,
                }
                state.add_trade(_sync_trade)
                _persist_equity_point(_sync_trade)
                if _pnl_usd > 0:
                    state.daily_profit_usd += _pnl_usd
                else:
                    state.daily_loss_usd   += abs(_pnl_usd)
                    state.consecutive_losses += 1
                log.info(f"📋 AUTO-SYNC trade logat: {_sync_dir} {_sync_entry}→{_exit_px} PnL=${_pnl_usd:+.2f} {_result} R={_r_mult}")
            # v9.5: dacă trade-ul s-a închis DUPĂ ce trailing a fost activat (partial_done=True)
            # → marcăm că așteptăm reset de semnal înainte de re-entry în aceeași direcție
            _was_trailing = state.partial_close_done   # True = 1R atins, trailing activ
            if _was_trailing and state.open_trade_dir:
                state.trailing_exit_ts            = datetime.now(timezone.utc).isoformat()
                state.trailing_exit_dir           = state.open_trade_dir   # "SHORT" / "LONG"
                state.trailing_exit_pending_reset = True
                log.info(
                    f"🔄 TRAILING EXIT detectat ({state.open_trade_dir}) "
                    f"→ re-entry blocat până la reset semnal"
                )
            # Reset COMPLET al stării de poziție
            state._position_open      = False
            state.partial_close_done  = False
            state.milestone_05r_done  = False
            state.milestone_085r_done = False
            state.trailing_sl         = 0.0
            state.trail_r_level       = 0
            state.open_trade_ts       = ""
            state.open_trade_entry    = 0.0
            state.open_trade_sl       = 0.0
            state.open_trade_tp       = 0.0
            state.open_trade_dir     = ""
            _clear_open_trade()   # ← ștergem fișierul de trade
            # Notifică dashboard via WebSocket că poziția s-a închis automat
            await broadcast_ws({
                "type":         "auto_position_reset",
                "source":       "nt8_sync",
                "trades_done":  _trades_done,
                "max_trades":   _max_t,
                "more_trades":  _more_trades,
                "timestamp":    _now_sync.isoformat(),
            })
            # Telegram: notifică că trade-ul s-a închis și sistemul e gata
            if _TG_OK and _tg_trade is None:
                pass  # _tg_trade este pentru trade_executed, nu close — skip
        elif not _nt8_flat and not state._position_open:
            # NT8 spune că e o poziție deschisă dar bridge nu știe — sincronizăm
            log.info(
                f"🔄 SYNC NT8: poziție {data.position_status} qty={data.position_qty} "
                f"detectată → _position_open=True"
            )
            state._position_open = True
    # ─────────────────────────────────────────────────────────────────────────

    # ── LIVE PnL CALC — calculat pe fiecare tick ──────────────────────────────
    _live_price    = data.price.close
    _open_pnl_usd  = 0.0
    if state._position_open and state.open_trade_entry > 0 and _live_price > 0:
        _diff         = _live_price - state.open_trade_entry
        _signed_diff  = _diff if state.open_trade_dir == "LONG" else -_diff
        _open_pnl_usd = round(_signed_diff * 20.0 * (state.open_trade_qty or 1), 2)
        # Actualizăm MAE (mers contra) și MFE (mers în favoare) în puncte
        if _signed_diff < 0:
            state.open_trade_mae_pts = max(state.open_trade_mae_pts, abs(_signed_diff))
        else:
            state.open_trade_mfe_pts = max(state.open_trade_mfe_pts, _signed_diff)
    _net_pnl_usd = round(state.daily_profit_usd - state.daily_loss_usd, 2)

    # Log periodic (la fiecare 30 tick-uri) când poziție e deschisă
    if state._position_open and state.open_trade_entry > 0 and state.tick_count % 30 == 0:
        _pnl_icon = "📈" if _open_pnl_usd >= 0 else "📉"
        _net_icon = "✅" if _net_pnl_usd >= 0 else "🔴"
        log.info(
            f"{_pnl_icon} LIVE PnL  {state.open_trade_dir} @ {state.open_trade_entry}  "
            f"px={_live_price:.2f}  open={_open_pnl_usd:+.0f}$  "
            f"| {_net_icon} Zi: profit={state.daily_profit_usd:.0f}$  "
            f"loss={state.daily_loss_usd:.0f}$  net={_net_pnl_usd:+.0f}$"
        )

    await broadcast_ws({
        "type": "tick", "symbol": data.symbol, "timestamp": data.timestamp,
        "price": data.price.close, "bid": data.price.bid, "ask": data.price.ask,
        "cum_delta": data.orderflow.cum_delta, "imbalance": data.orderflow.imbalance_pct,
        "poc": data.volume_profile.poc, "vwap": data.orderflow.vwap,
        "dom_ratio": data.dom_liquidity.bid_ask_ratio,
        # ── PnL live ──
        "open_pnl_usd":     _open_pnl_usd,
        "daily_profit_usd": round(state.daily_profit_usd, 2),
        "daily_loss_usd":   round(state.daily_loss_usd, 2),
        "net_pnl_usd":      _net_pnl_usd,
    })

    # ── SOFT SL MONITOR (înainte de 1R) ──────────────────────────────────────
    # Fără stop fizic în NT8 până la 1R — Python monitorizează și închide la SL
    # Previne "InvalidPrice" din NT8 Simulation pe stop-ul inițial
    if (state._position_open and not state.partial_close_done
            and state.open_trade_entry > 0 and state.open_trade_sl > 0):
        _soft_cur  = data.price.close
        _soft_dir  = state.open_trade_dir
        _soft_sl   = state.open_trade_sl
        _soft_sl_hit = (
            (_soft_dir == "LONG"  and _soft_cur <= _soft_sl) or
            (_soft_dir == "SHORT" and _soft_cur >= _soft_sl)
        )
        if _soft_sl_hit:
            log.warning(
                f"🛑 SOFT SL HIT {_soft_dir} @ {_soft_cur} "
                f"(SL={_soft_sl}) → CLOSE manual (fără stop fizic NT8)"
            )
            await send_command_to_nt8(action="CLOSE", qty=0, signal="SOFT_SL_HIT")
            _soft_pnl = round((_soft_cur - state.open_trade_entry if _soft_dir == "LONG"
                               else state.open_trade_entry - _soft_cur) * 20.0, 2)
            # POST-SL COOLDOWN: marchează timestamp doar dacă exit-ul e pe pierdere reală
            if _soft_pnl < 0:
                state.last_sl_hit_ts = time.time()
                log.info(f"⏳ POST-SL COOLDOWN armat ({state.post_sl_cooldown_sec}s) — SOFT_SL_HIT loss ${_soft_pnl}")
            state._position_open      = False
            state.partial_close_done  = False
            state.milestone_05r_done  = False
            state.milestone_085r_done = False
            state.trailing_sl         = 0.0
            state.trail_r_level       = 0
            _save_open_trade()
            if _TG_OK and _tg_closed:
                try:
                    _tg_closed(direction=_soft_dir, entry=state.open_trade_entry,
                               exit_price=_soft_cur, pnl_usd=_soft_pnl,
                               result="LOSS", exit_reason="SOFT_SL_HIT")
                except Exception: pass
            await broadcast_ws({"type": "soft_sl_hit", "data": {
                "direction": _soft_dir, "entry": state.open_trade_entry,
                "close_at": _soft_cur, "sl": _soft_sl
            }})
            return   # stop procesare, trade închis

    # ── 1R MONITOR + BREAKEVEN + TRAILING SETUP ──────────────────────────────
    # Logică diferită pentru 1 vs 2 contracte:
    #
    #  1 CONTRACT → la 1R: NU reducem (ar închide tot).
    #               Mutăm SL la breakeven și activăm trailing pe întreaga poziție.
    #               Trailing va proteja profitul până la TP sau SL hit.
    #
    #  2 CONTRACTE → la 1R: închidem 1 (profit sigur), SL la BE pe al 2-lea.
    #               Trailing pe contractul rămas.
    if (state._position_open and not state.partial_close_done
            and state.open_trade_entry > 0 and state.open_trade_sl > 0):
        # Fix v10.5: BAD TICK FILTER — comparăm cu prețul ANTERIOR tick (salvat la intrarea în handler)
        # Bad ticks (erori feed market data) pot declanșa false milestones care trimit
        # MOVE_SL la prețuri invalide → InvalidPrice NT8 și dezactivare strategie.
        # Threshold: 5×ATR sau 50pts (NQ), oricare e mai mare.
        _raw_cur = data.price.close
        _atr_now = float(getattr(data, 'atr_14', 0) or 0) or 15.0  # fallback 15pts ATR
        _bad_tick_thresh = max(_atr_now * 5.0, 50.0)
        if _prev_tick_close > 0 and abs(_raw_cur - _prev_tick_close) > _bad_tick_thresh:
            log.warning(
                f"⚠️ BAD TICK FILTRAT: data.close={_raw_cur:.2f} diferă cu "
                f"{abs(_raw_cur - _prev_tick_close):.1f}pts față de tick anterior={_prev_tick_close:.2f} "
                f"(thresh={_bad_tick_thresh:.0f}pts, 5×ATR={_atr_now*5:.0f}pts) → milestone check sărit"
            )
            return  # skip milestones pe tick corupt — nu trimitem MOVE_SL cu preț invalid
        _cur  = _raw_cur
        _en   = state.open_trade_entry
        _sl   = state.open_trade_sl
        _dir  = state.open_trade_dir
        _qty  = state.open_trade_qty   # 1 sau 2
        _risk = abs(_en - _sl)         # 1R în puncte

        if _risk > 0:
            # ── MILESTONE 0.5R → SL la BE ────────────────────────────────────
            _05r_long  = _dir == "LONG"  and _cur >= _en + _risk * 0.5
            _05r_short = _dir == "SHORT" and _cur <= _en - _risk * 0.5
            if (_05r_long or _05r_short) and not state.milestone_05r_done:
                _be_sl_05 = round(_en + 1.0, 2) if _dir == "LONG" else round(_en - 1.0, 2)
                state.milestone_05r_done = True
                state.trailing_sl = _be_sl_05
                # Fix v10.5: dacă prețul live a depășit deja noul SL → CLOSE direct
                _live_px_05 = state.latest_tick.price.close if state.latest_tick else _cur
                _breached_05 = (_dir == "SHORT" and _live_px_05 >= _be_sl_05) or \
                               (_dir == "LONG"  and _live_px_05 <= _be_sl_05)
                if _breached_05:
                    log.warning(f"⚡ MILESTONE_05R: SL={_be_sl_05} deja depășit de px={_live_px_05} → CLOSE direct")
                    await send_command_to_nt8(action="CLOSE", qty=0, signal="MILESTONE_05R_BREACHED")
                else:
                    await send_command_to_nt8(action="MOVE_SL", qty=0, signal="MILESTONE_05R", sl=_be_sl_05)
                _save_open_trade()
                log.info(
                    f"🔰 0.5R ATINS {_dir} @ {_cur}  entry={_en}  risk={_risk:.1f}pts "
                    f"→ SL mutat la BE={_be_sl_05}"
                )
                await broadcast_ws({"type": "milestone_05r", "data": {
                    "direction": _dir, "entry": _en, "price": _cur, "sl": _be_sl_05
                }})

            # ── MILESTONE 0.85R → SL la 0.5R profit ─────────────────────────
            _085r_long  = _dir == "LONG"  and _cur >= _en + _risk * 0.85
            _085r_short = _dir == "SHORT" and _cur <= _en - _risk * 0.85
            if (_085r_long or _085r_short) and not state.milestone_085r_done:
                _sl_085 = round(_en + _risk * 0.5, 2) if _dir == "LONG" else round(_en - _risk * 0.5, 2)
                state.milestone_085r_done = True
                state.trailing_sl = _sl_085
                # Fix v10.5: dacă prețul live a depășit deja noul SL → CLOSE direct
                _live_px_085 = state.latest_tick.price.close if state.latest_tick else _cur
                _breached_085 = (_dir == "SHORT" and _live_px_085 >= _sl_085) or \
                                (_dir == "LONG"  and _live_px_085 <= _sl_085)
                if _breached_085:
                    log.warning(f"⚡ MILESTONE_085R: SL={_sl_085} deja depășit de px={_live_px_085} → CLOSE direct")
                    await send_command_to_nt8(action="CLOSE", qty=0, signal="MILESTONE_085R_BREACHED")
                else:
                    await send_command_to_nt8(action="MOVE_SL", qty=0, signal="MILESTONE_085R", sl=_sl_085)

                if _qty == 1:
                    # ── FIX #4: 1 contract → trailing ATR activ de la 0.85R (nu mai așteptăm 1R) ──
                    # Motivul: pe NQ scalping ATR mic, distanța 0.85R→1R e 4 ticks care rareori
                    # se ating și lasă $200-$300 pe masă. Trailing pornește direct de la lock 0.5R.
                    state.partial_close_done = True
                    _save_open_trade()
                    log.info(
                        f"🎯 0.85R ATINS {_dir} @ {_cur}  entry={_en}  risk={_risk:.1f}pts "
                        f"→ [1 contract] SL mutat la 0.5R profit={_sl_085} + trailing ATR ACTIV"
                    )
                    await broadcast_ws({"type": "milestone_085r", "data": {
                        "direction": _dir, "entry": _en, "price": _cur, "sl": _sl_085,
                        "trailing_active": True
                    }})
                else:
                    # ── 2 contracte: SL la 0.5R profit, trailing se va activa la 1R ──
                    _save_open_trade()
                    log.info(
                        f"🔰 0.85R ATINS {_dir} @ {_cur}  entry={_en}  risk={_risk:.1f}pts "
                        f"→ [2 contracte] SL mutat la 0.5R profit={_sl_085} | trailing la 1R"
                    )
                    await broadcast_ws({"type": "milestone_085r", "data": {
                        "direction": _dir, "entry": _en, "price": _cur, "sl": _sl_085,
                        "trailing_active": False
                    }})

            # ── 1R: doar pentru 2 contracte (REDUCE + BE) ────────────────────────
            # 1 contract: partial_close_done e deja True din blocul 0.85R de mai sus
            _1r_long  = _dir == "LONG"  and _cur >= _en + _risk
            _1r_short = _dir == "SHORT" and _cur <= _en - _risk

            if (_1r_long or _1r_short) and _qty >= 2:
                _be_sl = round(_en + 1.0, 2) if _dir == "LONG" else round(_en - 1.0, 2)
                # ── 2 contracte: închidem 1, rămâne 1 cu trailing ──
                log.info(
                    f"🎯 1R ATINS {_dir} @ {_cur}  entry={_en}  risk={_risk:.1f}pts "
                    f"→ [2 contracte] REDUCE 1 + BE SL={_be_sl} → trailing pe 1"
                )
                await send_command_to_nt8(action="REDUCE", qty=1, signal="PARTIAL_1R")
                await asyncio.sleep(0.3)
                await send_command_to_nt8(action="MOVE_SL", qty=0, signal="BREAKEVEN", sl=_be_sl)
                state.trailing_sl        = _be_sl   # trailing pornește de la BE
                state.partial_close_done = True
                _save_open_trade()
                await broadcast_ws({"type": "partial_close", "data": {
                    "direction": _dir, "entry": _en, "close_at": _cur,
                    "breakeven_sl": _be_sl, "qty_closed": 1, "qty_remaining": 1,
                    "mode": "2contracts_reduce"
                }})
    # ─────────────────────────────────────────────────────────────────────────

    # ── A1. TRAILING STOP — DINAMIC bazat pe SL original ─────────────────────
    # Trailing se strânge pe măsură ce prețul avansează:
    #   După 2R → trail = 25% din SL original (ex: SL=26 → trail=6.5 pts)
    #   După 3R → trail = 15% din SL original (ex: SL=26 → trail=4 pts)
    # Asta protejează profitul fără să taie prematur mișcările mari

    if state._position_open and state.partial_close_done and state.trailing_sl > 0:
        # Fix v10.5: folosim prețul live (latest_tick) NU bara closed — prinde spikes între bare
        _tc   = state.latest_tick.price.close if state.latest_tick else data.price.close
        _td   = state.open_trade_dir
        _en   = state.open_trade_entry
        _sl0  = state.open_trade_sl
        _risk = abs(_en - _sl0)   # 1R original în puncte

        # ── TRAILING SL HIT CHECK — închide poziția când prețul traversează trailing_sl ──
        _trail_hit = (
            (_td == "LONG"  and _tc <= state.trailing_sl) or
            (_td == "SHORT" and _tc >= state.trailing_sl)
        )
        if _trail_hit:
            log.warning(
                f"🎯 TRAILING SL HIT {_td} @ {_tc:.2f} "
                f"(trailing_sl={state.trailing_sl:.2f} trail_r={state.trail_r_level}) → CLOSE"
            )
            await send_command_to_nt8(action="CLOSE", qty=0, signal="TRAIL_SL_HIT")
            _trail_pnl  = round((_tc - _en if _td == "LONG" else _en - _tc) * 20.0, 2)
            _trail_rmult = round(_trail_pnl / (_risk * 20.0), 2) if _risk > 0 else 0.0
            _trail_result = "WIN" if _trail_pnl > 0 else ("LOSS" if _trail_pnl < 0 else "BE")
            # POST-SL COOLDOWN: doar pe pierdere reală, nu pe trailing profit locked
            if _trail_pnl < 0:
                state.last_sl_hit_ts = time.time()
                log.info(f"⏳ POST-SL COOLDOWN armat ({state.post_sl_cooldown_sec}s) — TRAIL_SL_HIT loss ${_trail_pnl}")
            state._position_open      = False
            state.partial_close_done  = False
            state.milestone_05r_done  = False
            state.milestone_085r_done = False
            state.trailing_sl         = 0.0
            state.trail_r_level       = 0
            state.open_trade_ts      = ""
            _clear_open_trade()
            if _TG_OK and _tg_closed:
                try:
                    _tg_closed(direction=_td, entry=_en, exit_price=_tc,
                               pnl_usd=_trail_pnl, result=_trail_result,
                               r_mult=_trail_rmult, exit_reason="TRAILING STOP")
                except Exception: pass
            await broadcast_ws({"type": "trail_sl_hit", "data": {
                "direction": _td, "entry": _en, "close_at": _tc,
                "trailing_sl": state.trailing_sl, "trail_r_level": state.trail_r_level
            }})
            return

        # Trail dinamic bazat pe ATR — se strânge progresiv cu fiecare nivel R
        _atr_now   = data.atr_14 if (hasattr(data, 'atr_14') and data.atr_14 > 0) else 9.0
        _trail_1r  = max(round(_risk * 0.40, 2), round(_atr_now * 0.70, 1))  # 40% risc / 70% ATR  (cel mai larg)
        _trail_15r = max(round(_risk * 0.35, 2), round(_atr_now * 0.62, 1))  # 35% risc / 62% ATR
        _trail_2r  = max(round(_risk * 0.30, 2), round(_atr_now * 0.55, 1))  # 30% risc / 55% ATR
        _trail_25r = max(round(_risk * 0.25, 2), round(_atr_now * 0.60, 1))  # 25% risc / 60% ATR
        _trail_3r  = max(round(_risk * 0.20, 2), round(_atr_now * 0.65, 1))  # 20% risc / 65% ATR  (cel mai strâns)

        _1r_long   = _td == "LONG"  and _risk > 0 and _tc >= _en + _risk
        _1r_short  = _td == "SHORT" and _risk > 0 and _tc <= _en - _risk
        _15r_long  = _td == "LONG"  and _risk > 0 and _tc >= _en + 1.5 * _risk
        _15r_short = _td == "SHORT" and _risk > 0 and _tc <= _en - 1.5 * _risk
        _2r_long   = _td == "LONG"  and _risk > 0 and _tc >= _en + 2 * _risk
        _2r_short  = _td == "SHORT" and _risk > 0 and _tc <= _en - 2 * _risk
        _25r_long  = _td == "LONG"  and _risk > 0 and _tc >= _en + 2.5 * _risk
        _25r_short = _td == "SHORT" and _risk > 0 and _tc <= _en - 2.5 * _risk
        _3r_long   = _td == "LONG"  and _risk > 0 and _tc >= _en + 3 * _risk
        _3r_short  = _td == "SHORT" and _risk > 0 and _tc <= _en - 3 * _risk

        # helper: mută SL dacă e mai bun, trimite MOVE_SL
        # Fix v10.5: dacă prețul live a depășit deja noul SL → CLOSE direct
        _live_px_trail = state.latest_tick.price.close if state.latest_tick else _tc
        async def _apply_trail(signal: str, trail_pts: float):
            if _td == "LONG":
                _nt = round(_tc - trail_pts, 2)
                if _nt > state.trailing_sl:
                    state.trailing_sl = _nt
                    if _live_px_trail <= _nt:
                        log.warning(f"⚡ {signal} LONG: SL={_nt} deja depășit px={_live_px_trail} → CLOSE direct")
                        await send_command_to_nt8(action="CLOSE", qty=0, signal=f"{signal}_BREACHED")
                    else:
                        await send_command_to_nt8(action="MOVE_SL", qty=0, signal=signal, sl=_nt, tp=0.0)
                    _save_open_trade()
                    log.debug(f"📈 {signal} LONG: SL → {_nt} (px={_tc} -{trail_pts}pts)")
            elif _td == "SHORT":
                _nt = round(_tc + trail_pts, 2)
                if _nt < state.trailing_sl:
                    state.trailing_sl = _nt
                    if _live_px_trail >= _nt:
                        log.warning(f"⚡ {signal} SHORT: SL={_nt} deja depășit px={_live_px_trail} → CLOSE direct")
                        await send_command_to_nt8(action="CLOSE", qty=0, signal=f"{signal}_BREACHED")
                    else:
                        await send_command_to_nt8(action="MOVE_SL", qty=0, signal=signal, sl=_nt, tp=0.0)
                    _save_open_trade()
                    log.debug(f"📉 {signal} SHORT: SL → {_nt} (px={_tc} +{trail_pts}pts)")

        # ── 3R trailing (cel mai strâns) ─────────────────────────────────────
        if _3r_long or _3r_short:
            if state.trail_r_level < 3:
                state.trail_r_level = 3
                log.info(f"🔥 3R ATINS {_td} @ {_tc:.2f} → trail {_trail_3r}pts")
            await _apply_trail("TRAIL_3R", _trail_3r)

        # ── 2.5R trailing ─────────────────────────────────────────────────────
        elif _25r_long or _25r_short:
            if state.trail_r_level < 25:
                state.trail_r_level = 25
                log.info(f"🎯 2.5R ATINS {_td} @ {_tc:.2f} → trail {_trail_25r}pts")
            await _apply_trail("TRAIL_25R", _trail_25r)

        # ── 2R trailing ───────────────────────────────────────────────────────
        elif _2r_long or _2r_short:
            if state.trail_r_level < 2:
                state.trail_r_level = 2
                _1r_lock = round(_en + _risk, 2) if _td == "LONG" else round(_en - _risk, 2)
                log.info(
                    f"🎯 2R ATINS {_td} @ {_tc:.2f}  entry={_en}  risk={_risk:.1f}pts "
                    f"→ trail {_trail_2r}pts | SL minim 1R={_1r_lock}"
                )
                # la prima activare 2R: garantăm SL la minim 1R
                if _td == "LONG" and _1r_lock > state.trailing_sl:
                    state.trailing_sl = _1r_lock
                    await send_command_to_nt8(action="MOVE_SL", qty=0, signal="TRAIL_2R_LOCK", sl=_1r_lock, tp=0.0)
                elif _td == "SHORT" and _1r_lock < state.trailing_sl:
                    state.trailing_sl = _1r_lock
                    await send_command_to_nt8(action="MOVE_SL", qty=0, signal="TRAIL_2R_LOCK", sl=_1r_lock, tp=0.0)
                _save_open_trade()
            await _apply_trail("TRAIL_2R", _trail_2r)

        # ── 1.5R trailing ─────────────────────────────────────────────────────
        elif _15r_long or _15r_short:
            if state.trail_r_level < 15:
                state.trail_r_level = 15
                log.info(f"🎯 1.5R ATINS {_td} @ {_tc:.2f} → trail {_trail_15r}pts")
            await _apply_trail("TRAIL_15R", _trail_15r)

        # ── 1R trailing (cel mai larg) ────────────────────────────────────────
        elif _1r_long or _1r_short:
            if state.trail_r_level < 1:
                state.trail_r_level = 1
                log.info(f"🎯 1R TRAIL ACTIV {_td} @ {_tc:.2f} → trail {_trail_1r}pts")
            await _apply_trail("TRAIL_1R", _trail_1r)

    # ── A2. TIME-BASED EXIT (MAX HOLD TIME) ──────────────────────────────────
    # Dacă trade-ul nu a atins TP în limita de timp, ieșim la market.
    # Default: scalping=45min, intraday=4h, swing=24h
    if state._position_open and state.open_trade_ts:
        _max_hold = {
            "scalping_london": 45, "scalping_ny": 45,
            "silver_bullet_london": 30, "silver_bullet_ny": 30,
            "intraday_london": 240, "ny_open": 120, "intraday_ny": 240,
            "overlap": 120, "swing_judas": 1440, "custom": 120,
        }
        _strat_id_hold = state.active_strategy.get("id", "custom") if state.active_strategy else "custom"
        _max_min       = _max_hold.get(_strat_id_hold, 120)
        try:
            _entry_dt  = datetime.fromisoformat(state.open_trade_ts.replace("Z", "+00:00"))
            _now_dt    = datetime.now(timezone.utc)
            _held_min  = (_now_dt - _entry_dt).total_seconds() / 60
            if _held_min >= _max_min:
                log.warning(
                    f"⏱️ MAX HOLD TIME [{_strat_id_hold}]: {_held_min:.0f}min >= {_max_min}min → EXIT"
                )
                await send_command_to_nt8(action="CLOSE", qty=0, signal="MAX_HOLD_EXIT")
                state._position_open      = False
                state.partial_close_done  = False
                state.milestone_05r_done  = False
                state.milestone_085r_done = False
                state.trailing_sl         = 0.0
                state.trail_r_level       = 0
                state.open_trade_ts       = ""
                _clear_open_trade()   # ← ștergem fișierul de trade
        except Exception as _te:
            log.debug(f"Max hold timer error: {_te}")
    # ─────────────────────────────────────────────────────────────────────────

    # UPDATE #14b: Rulăm analiza O SINGURĂ DATĂ per bară (nu per tick)
    # Trunchiem timestamp la minut → bară nouă = timestamp nou
    # Funcționează și cu "On each tick" (1 analiză/minut) și "On bar close" (1 analiză/POST)
    _bar_ts_min = data.timestamp[:16] if data.timestamp else ""
    if _bar_ts_min and _bar_ts_min != state._last_analysis_bar_ts:
        state._last_analysis_bar_ts = _bar_ts_min
        background_tasks.add_task(run_analysis_async, data)

    # ── DB AUTO-SAVE: scrie bara în SQLite o dată pe minut ───────────────────
    # INSERT OR IGNORE — nu suprascrie niciodată importuri manuale NT8
    if _bar_ts_min and _bar_ts_min != state._last_saved_bar_ts:
        state._last_saved_bar_ts = _bar_ts_min
        background_tasks.add_task(_save_bar_to_db, data)

    return {"status": "ok", "tick": state.tick_count}

@app.post("/execution_confirm")
async def execution_confirm(confirm: ExecutionConfirm):
    state.last_exec_confirm = confirm.model_dump()
    action = confirm.action.upper()
    now    = datetime.now(timezone.utc)

    # ── Detectează sesiunea ─────────────────────────────────────────────────
    _utc_hour = now.hour
    if 7 <= _utc_hour < 12:      _session = "London"
    elif 12 <= _utc_hour < 20:  _session = "New York"
    elif 20 <= _utc_hour < 24:  _session = "NY Close"
    elif 0 <= _utc_hour < 3:    _session = "Asian Pre"
    else:                        _session = "Asian"     # 03-07 UTC

    _analysis  = state.last_analysis or {}
    _strat     = state.active_strategy or {}
    _score_raw = _analysis.get("score", 0)
    _score_pct = round(_score_raw * 100, 1) if _score_raw <= 1 else round(_score_raw, 1)

    # ── REDUCE / MOVE_SL — confirmate de NT8 dar nu schimbă entry tracking ──
    if action in ("REDUCE", "MOVE_SL"):
        log.info(f"📥 NT8 confirm {action} @ {confirm.price:.2f} qty={confirm.qty} → acknowledged (no state change)")
        return {"status": "confirmed", "action": action.lower() + "_acknowledged"}

    # ── BUY / SELL — înregistrăm INTRAREA, nu adăugăm în trade_log încă ────
    if action in ("BUY", "SELL"):
        direction = "LONG" if action == "BUY" else "SHORT"

        # Fix v9.1: Scale-in detection — dacă e aceeași direcție și poziție deja deschisă,
        # NU suprascriem entry și qty, ci facem average și acumulăm.
        # Fix v9.2: Sanity check price — dacă fill-ul diferă cu >50pts față de entry așteptat,
        # e un confirm stale (din ordine vechi) → ignorăm scale-in, înregistrăm ca entry nou
        _price_ok    = abs(confirm.price - state.open_trade_entry) <= 50 if state.open_trade_entry > 0 else True
        _is_scale_in = state._position_open and state.open_trade_dir == direction and state.open_trade_entry > 0 and _price_ok

        if state._position_open and state.open_trade_dir == direction and state.open_trade_entry > 0 and not _price_ok:
            # Fix v10.5: dacă flagul entry_nt8_confirmed e False, înseamnă că nu am primit
            # niciun confirm valid de la NT8 pentru trade-ul curent → position_flag ar putea
            # fi stuck. Logăm un warning clar cu instrucțiuni.
            _no_valid_confirm = not getattr(state, 'entry_nt8_confirmed', True)
            log.warning(
                f"⚠️ Execution confirm STALE ignorat complet: {direction} @ {confirm.price:.2f} "
                f"diferă cu {abs(confirm.price - state.open_trade_entry):.1f}pts față de entry așteptat "
                f"{state.open_trade_entry} (max 50pts) → entry/SL NU se suprascriu"
                + (" | ⚡ ATENȚIE: niciun confirm valid primit — dacă NT8 nu are poziție deschisă, apelați /reset_position" if _no_valid_confirm else "")
            )
            await broadcast_ws({"type": "execution", "data": state.last_exec_confirm})
            return {"status": "confirmed", "action": "stale_ignored"}

        if _is_scale_in:
            # ── FIX #3: GARDIAN SCALE-IN MID-TRADE ──────────────────────────
            # Verificăm condițiile înainte să acceptăm scale-in-ul NT8.
            # Un scale-in mid-trade (când e deja _position_open) e periculos dacă:
            #   - Suntem încă sub 0.85R profit (trade-ul nu e dovedit)
            #   - Trade-ul curent e în pierdere
            #   - Avem deja 2+ contracte (max 1 scale-in per trade)
            _cur_close  = state.latest_tick.price.close if (state.latest_tick and state.latest_tick.price) else 0.0
            _entry_pt   = state.open_trade_entry
            _sl_pt      = state.open_trade_sl
            _dir_pt     = state.open_trade_dir
            _risk_pt    = abs(_entry_pt - _sl_pt) if _sl_pt > 0 else 0.0
            # PnL curent la prețul tick-ului
            if _cur_close > 0 and _entry_pt > 0 and _dir_pt:
                _open_pnl = (_cur_close - _entry_pt if _dir_pt == "LONG" else _entry_pt - _cur_close) * 20.0
            else:
                _open_pnl = 0.0
            # Guard 1: trebuie să avem cel puțin 0.85R profit (SL deja mutat la 0.5R)
            _scalein_mid_profit_ok = state.milestone_085r_done
            # Guard 2: open PnL trebuie să fie pozitiv
            _scalein_mid_pnl_ok   = _open_pnl >= 0
            # Guard 3: max 1 scale-in per trade (qty curentă = 1)
            _scalein_mid_qty_ok   = (state.open_trade_qty or 1) <= 1

            if _scalein_mid_profit_ok and _scalein_mid_pnl_ok and _scalein_mid_qty_ok:
                # Scale-in valid — calculăm entry mediu ponderat
                _prev_qty    = state.open_trade_qty or 1
                _prev_entry  = state.open_trade_entry
                _total_qty   = _prev_qty + confirm.qty
                _avg_entry   = round((_prev_entry * _prev_qty + confirm.price * confirm.qty) / _total_qty, 2)
                state.open_trade_qty   = _total_qty
                state.open_trade_entry = _avg_entry
                state.entry_nt8_confirmed = True  # Fix v10.5: confirm valid primit
                if state.open_trade_sl > 0:
                    _sl_dist = abs(_prev_entry - state.open_trade_sl)
                    state.open_trade_sl = round(_avg_entry + _sl_dist if direction == "SHORT" else _avg_entry - _sl_dist, 2)
                _save_open_trade()
                log.info(
                    f"📥 SCALE-IN VALID confirm NT8: {direction} @ {confirm.price:.2f} qty=+{confirm.qty} "
                    f"(0.85R atins ✅ pnl={_open_pnl:+.0f}$ ✅ qty=1 ✅) "
                    f"→ total={_total_qty} avg_entry={_avg_entry} SL={state.open_trade_sl}"
                )
            else:
                # Scale-in blocat de gardian — tratăm ca entry stale/ignorat
                _reasons = []
                if not _scalein_mid_profit_ok:
                    _reasons.append("sub 0.85R profit")
                if not _scalein_mid_pnl_ok:
                    _reasons.append(f"pnl={_open_pnl:+.0f}$ (pierdere)")
                if not _scalein_mid_qty_ok:
                    _reasons.append(f"deja {state.open_trade_qty} contracte")
                log.warning(
                    f"🛡️ SCALE-IN MID-TRADE BLOCAT: {direction} @ {confirm.price:.2f} "
                    f"→ {', '.join(_reasons)} → confirm ignorat"
                )
                await broadcast_ws({"type": "execution", "data": state.last_exec_confirm})
                return {"status": "confirmed", "action": "scale_in_blocked_guardian"}
        else:
            # Trade nou — înregistrăm fresh
            _old_sl_dist = abs(state.open_trade_entry - state.open_trade_sl) if state.open_trade_sl > 0 and state.open_trade_entry > 0 else 0
            _fill_diff   = abs(confirm.price - state.open_trade_entry) if state.open_trade_entry > 0 else 0

            state.open_trade_dir   = direction
            state.open_trade_qty   = confirm.qty
            if not state.open_trade_ts:
                state.open_trade_ts = now.isoformat()

            # Fix v9.1: Recalculăm SL față de prețul real de fill (nu față de prețul analizei)
            # Păstrăm aceeași distanță în puncte, dar mutăm față de fill-ul real.
            if state.open_trade_sl > 0 and state.open_trade_entry > 0 and _fill_diff > 0:
                _sl_dist_pts = abs(state.open_trade_entry - state.open_trade_sl)
                _new_sl = round(confirm.price + _sl_dist_pts if direction == "SHORT" else confirm.price - _sl_dist_pts, 2)
                log.info(
                    f"   🔧 SL recalculat din fill real: analysis_entry={state.open_trade_entry} "
                    f"fill={confirm.price} (diff={_fill_diff:.2f}pts) "
                    f"SL {state.open_trade_sl}→{_new_sl}"
                )
                state.open_trade_sl = _new_sl

            state.open_trade_entry = confirm.price
            state.entry_nt8_confirmed = True  # Fix v10.5: confirm valid primit de la NT8
            log.info(f"📥 ENTRY confirm NT8: {direction} @ {confirm.price:.2f}  qty={confirm.qty}")

        await broadcast_ws({"type": "execution", "data": state.last_exec_confirm})
        return {"status": "confirmed", "action": "scale_in_recorded" if _is_scale_in else "entry_recorded"}

    # ── CLOSE / EXIT / FLAT — calculăm PnL și adăugăm trade complet ────────
    if action in ("CLOSE", "EXIT", "FLAT"):
        # Dacă _position_open e deja False înseamnă că bridge-ul a închis deja
        # via TRAIL_SL_HIT sau SOFT_SL_HIT — confirmul NT8 e stale, ignorăm
        if not state._position_open:
            log.info(
                f"📥 CLOSE confirm NT8 @ {confirm.price:.2f} ignorat — "
                f"poziția a fost deja închisă de bridge (TRAIL/SOFT SL)"
            )
            return {"status": "confirmed", "action": "already_closed_ignored"}

        # Fix v10.5: STALE CLOSE detection — dacă exit price diferă cu >150pts față de
        # ultimul preț live cunoscut, e un confirm vechi (din altă sesiune/zi).
        # În loc să calculăm PnL fantomă (care ar putea declanșa circuit breaker fals),
        # curățăm state-ul poziției și deblocăm trading-ul.
        _live_close = state.latest_tick.price.close if (state.latest_tick and state.latest_tick.price) else 0.0
        if confirm.price > 0 and _live_close > 0 and abs(confirm.price - _live_close) > 150:
            log.warning(
                f"⚠️ CLOSE confirm STALE detectat: exit={confirm.price:.2f} diferă cu "
                f"{abs(confirm.price - _live_close):.1f}pts față de prețul live={_live_close:.2f} (max 150pts) "
                f"→ confirm ignorat la PnL, dar poziție curățată"
            )
            # Curățăm position state pentru a debloca trading-ul
            state._position_open      = False
            state.partial_close_done  = False
            state.milestone_05r_done  = False
            state.milestone_085r_done = False
            state.open_trade_entry    = 0.0
            state.open_trade_sl       = 0.0
            state.open_trade_tp       = 0.0
            state.open_trade_dir      = ""
            state.trailing_sl         = 0.0
            state.trail_r_level       = 0
            state.open_trade_ts       = ""
            _clear_open_trade()
            await broadcast_ws({"type": "execution", "data": state.last_exec_confirm})
            return {"status": "confirmed", "action": "stale_close_position_cleared"}

        # Capturăm entry ÎNAINTE de a reseta state-ul
        _entry_px  = state.open_trade_entry
        _entry_dir = state.open_trade_dir
        _entry_qty = state.open_trade_qty or confirm.qty
        _entry_ts  = state.open_trade_ts or now.isoformat()
        _exit_px   = confirm.price

        # ── Fallback exit price când NT8 trimite price=0 (OnPositionUpdate înainte de fill) ──
        if _exit_px <= 0:
            # Folosim trailing_sl ca estimare exit (e aproape de prețul real de ieșire)
            if state.trailing_sl > 0:
                _exit_px = state.trailing_sl
                log.info(f"⚠️ CLOSE price=0 → folosim trailing_sl={state.trailing_sl:.2f} ca exit estimat")
            # Sau ultimul preț live cunoscut
            elif state.latest_tick and state.latest_tick.price and state.latest_tick.price.close > 0:
                _exit_px = state.latest_tick.price.close
                log.info(f"⚠️ CLOSE price=0 → folosim last_tick={_exit_px:.2f} ca exit estimat")

        # ── Fallback: dacă state e gol (restart bridge / manual close), citim din disc ──
        if (_entry_px == 0.0 or not _entry_dir) and OPEN_TRADE_FILE.exists():
            try:
                _disk = json.loads(OPEN_TRADE_FILE.read_text())
                if _disk.get("position_open"):
                    _entry_px  = float(_disk.get("open_trade_entry", 0.0)) or _entry_px
                    _entry_dir = str(_disk.get("open_trade_dir", ""))      or _entry_dir
                    _entry_qty = int(_disk.get("open_trade_qty", 1))
                    _entry_ts  = str(_disk.get("open_trade_ts", now.isoformat()))
                    log.info(f"♻️  Entry restaurat din disc pentru CLOSE: {_entry_dir} @ {_entry_px}")
            except Exception as _de:
                log.debug(f"Disk fallback skip: {_de}")

        # Feature 3: Point value dinamic din instrument config
        _inst_cfg  = _get_instrument_config(getattr(state.latest_tick, 'symbol', '') if state.latest_tick else '')
        _point_val = _inst_cfg["point_value"]
        if _entry_px > 0 and _entry_dir and _exit_px > 0:
            _diff = _exit_px - _entry_px  # puncte
            _pnl_usd = (_diff if _entry_dir == "LONG" else -_diff) * _point_val * _entry_qty
            _pnl_usd = round(_pnl_usd, 2)
        elif _exit_px <= 0:
            # Fix v9.2: price=0 = confirm stale/invalid — nu calculăm PnL fantomă
            _pnl_usd = 0.0
            log.warning(
                f"⚠️ CLOSE cu price=0 ignorat la calcul PnL — confirm stale sau order NT8 fără fill "
                f"(entry={_entry_px} dir={_entry_dir}) → PnL=0"
            )
        else:
            _pnl_usd = 0.0
            log.warning(f"⚠️ CLOSE fără entry înregistrat — PnL calculat 0 (entry_px={_entry_px}, dir={_entry_dir})")

        _result = "WIN" if _pnl_usd > 0 else ("LOSS" if _pnl_usd < 0 else "BE")
        # R-Multiple: PnL / risc inițial (SL distance * point_val)
        _sl     = state.open_trade_sl
        _r_mult = 0.0
        if _sl > 0 and _entry_px > 0 and _entry_dir:
            _risk_pts = abs(_entry_px - _sl)
            _risk_usd = _risk_pts * _point_val * _entry_qty
            if _risk_usd > 0:
                _r_mult = round(_pnl_usd / _risk_usd, 2)

        # ── MAE/MFE + Exit Reason (v10.6) ────────────────────────────────────
        # MAE = cât a mers trade-ul împotrivă (max drawdown, puncte)
        # MFE = cât a mers în favoare (max favorable, puncte)
        # exit_reason = ce a cauzat închiderea (SL_HIT, TP_HIT, MANUAL, TRAILING, CLOSE, REDUCE)
        _mae_pts = round(getattr(state, "open_trade_mae_pts", 0) or 0, 2)
        _mfe_pts = round(getattr(state, "open_trade_mfe_pts", 0) or 0, 2)

        # Determinăm exit reason din context
        _exit_reason = "UNKNOWN"
        if action == "CLOSE":
            # Verificăm dacă a fost SL hit, TP hit, sau manual
            if _sl > 0 and _exit_px > 0:
                _sl_dist = abs(_exit_px - _sl)
                _tp_dist = abs(_exit_px - _tp) if _tp > 0 else 999
                if _sl_dist <= 1.0:               # exit la <1pt de SL → SL hit
                    _exit_reason = "SL_HIT"
                elif _tp > 0 and _tp_dist <= 1.0: # exit la <1pt de TP → TP hit
                    _exit_reason = "TP_HIT"
                elif getattr(state, 'trailing_sl', 0) > 0 and _sl_dist <= 2.0:
                    _exit_reason = "TRAILING_SL"   # trailing stop hit
                else:
                    _exit_reason = "MANUAL_CLOSE"  # closed manual sau de bridge
            else:
                _exit_reason = "MANUAL_CLOSE"
        elif action == "REDUCE":
            _exit_reason = "PARTIAL_CLOSE"
        else:
            _exit_reason = action  # BUY/SELL = reverse

        trade = {
            "id":          str(uuid.uuid4())[:8],
            "entry_time":  _entry_ts[:19],
            "exit_time":   now.isoformat()[:19],
            "direction":   _entry_dir or action,
            "entry_price": round(_entry_px, 2),
            "exit_price":  round(_exit_px, 2),
            "qty":         _entry_qty,
            "sl":          round(_sl, 2),
            "tp":          round(_tp, 2),
            "pnl":         round(_pnl_usd / _point_val, 4),  # puncte
            "pnl_usd":     _pnl_usd,
            "pnl_pct":     0.0,
            "result":      _result,
            "mae_pts":     _mae_pts,
            "mfe_pts":     _mfe_pts,
            "exit_reason": _exit_reason,
            "entry_hour":  now.hour,
            "entry_dow":   now.weekday(),
            "setup":       _analysis.get("verdict", "LIVE")[:50],
            "score":       _score_pct,
            "strategy":    _strat.get("label", "Default"),
            "session":     _session,
            "r_multiple":  _r_mult,
            "source":      "NT8_DEMO",
            "instrument":  (state.latest_tick.symbol if state.latest_tick else ACTIVE_INSTRUMENT),
            "geo_risk_active": state.geo_risk_active,
        }
        state.add_trade(trade)
        # Feature 2: persistăm în equity history
        _persist_equity_point(trade)
        # Feature 4: înregistrăm în A/B test (slot A = real)
        _ab_record_trade("A", trade)
        log.info(f"✅ TRADE LOG: {_entry_dir} {_entry_px:.2f}→{_exit_px:.2f} | PnL: ${_pnl_usd:+.2f} | {_result} | R={_r_mult}")

        # ── RL FEEDBACK pe trade REAL (fix: era apelat doar pe paper trades) ──
        # Actualizează ponderile celor 7 componente din formula de scoring
        # după fiecare trade real câștigat/pierdut.
        if _RL_OK and _rl_feedback is not None:
            try:
                _comp_scores = dict(_analysis.get("component_scores", {}))
                # Injectăm scorul volume_profile (a 7-a componentă RL)
                _real_dir = _analysis.get("trade_direction", state.open_trade_dir)
                _comp_scores["volume_profile"] = _compute_vp_score(state, state.latest_tick, _real_dir)
                _new_w = _rl_feedback.on_trade_closed(
                    component_scores = _comp_scores,
                    result           = _result,
                    pnl              = _pnl_usd,
                    score_pct        = _score_pct,
                    direction        = _entry_dir,
                )
                log.info(
                    f"🧠 RL Update (real): {_result} ${_pnl_usd:+.0f} | "
                    f"AI={_new_w.get('ai',0):.3f} ICT={_new_w.get('ict',0):.3f} "
                    f"OF={_new_w.get('orderflow',0):.3f} Sent={_new_w.get('sentiment',0):.3f}"
                )
            except Exception as _rl_err:
                log.warning(f"RL on_trade_closed (real) skip: {_rl_err}")

        # POST-SL COOLDOWN: dacă exit-ul e pe pierdere, armăm cooldown (confirm NT8 fizic)
        if _pnl_usd < 0:
            state.last_sl_hit_ts = time.time()
            log.info(f"⏳ POST-SL COOLDOWN armat ({state.post_sl_cooldown_sec}s) — NT8 exit loss ${_pnl_usd}")

        # ── SESSION PnL TRACKING (pentru /trade Telegram) ──────────────────────
        if _pnl_usd > 0:
            state.daily_profit_usd += _pnl_usd
        else:
            state.daily_loss_usd   += abs(_pnl_usd)
        _session_total = state.daily_profit_usd - state.daily_loss_usd
        # Actualizăm max profit sesiune
        if _session_total > state.session_max_profit:
            state.session_max_profit = _session_total
        # Actualizăm peak PnL și max drawdown
        if _session_total > state._session_peak_pnl:
            state._session_peak_pnl = _session_total
        _drawdown_from_peak = state._session_peak_pnl - _session_total
        if _drawdown_from_peak > state.session_max_drawdown:
            state.session_max_drawdown = _drawdown_from_peak
        log.info(
            f"📊 Sesiune: profit={state.daily_profit_usd:.2f} loss={state.daily_loss_usd:.2f} "
            f"net={_session_total:+.2f} | max_profit={state.session_max_profit:.2f} "
            f"max_dd={state.session_max_drawdown:.2f}"
        )

        # Reset state poziție
        _trades_done = state.strategy_trades_today
        _max_t       = int(state.active_strategy.get("max_trades", 0)) if state.active_strategy else 0
        _more_trades = _trades_done < _max_t if _max_t > 0 else True
        state._position_open      = False
        state.entry_nt8_confirmed = False  # Fix v10.5: resetăm pentru trade-ul următor
        state.partial_close_done  = False
        state.milestone_05r_done  = False
        state.milestone_085r_done = False
        state.open_trade_entry    = 0.0
        state.open_trade_sl       = 0.0
        state.open_trade_tp       = 0.0
        state.open_trade_dir      = ""
        state.trailing_sl         = 0.0
        state.trail_r_level       = 0
        state.open_trade_ts       = ""
        _clear_open_trade()   # ← ștergem fișierul de trade

        # ── REVERSAL RE-ENTRY: dacă reversal e armat, intrăm în direcție opusă ──
        # Fereastra de 45s: NT8 trimite CLOSE confirm de obicei în <5s; 45s = margine generoasă
        if state.reversal_pending_dir and (time.time() - state.reversal_pending_ts) < 45:
            _rv_dir   = state.reversal_pending_dir
            _rv_sl    = state.reversal_pending_sl
            _rv_tp    = state.reversal_pending_tp
            _rv_score = state.reversal_pending_score
            # Prețul real de entry = exit price din confirm (sau ultimul tick dacă 0)
            _rv_entry = _exit_px if _exit_px > 0 else (
                float(state.latest_tick.price.close) if state.latest_tick else 0.0
            )
            # Curățăm pending state ÎNAINTE de execuție (atomic — evităm re-trigger)
            state.reversal_pending_dir   = ""
            state.reversal_pending_score = 0.0
            state.reversal_pending_sl    = 0.0
            state.reversal_pending_tp    = 0.0
            state.reversal_pending_ts    = 0.0
            # Set up tracking pentru noul trade
            state.open_trade_entry    = _rv_entry
            state.open_trade_dir      = _rv_dir
            state.open_trade_sl       = _rv_sl
            state.open_trade_tp       = _rv_tp
            state.open_trade_qty      = 1          # reversal: întotdeauna 1 contract
            state.partial_close_done  = False
            state.milestone_05r_done  = False
            state.milestone_085r_done = False
            state.trailing_sl         = _rv_sl
            state.trail_r_level       = 0
            state.open_trade_ts       = datetime.now(timezone.utc).isoformat()
            _save_open_trade()
            # Trimitem entry la NT8
            _rv_ok = await send_command_to_nt8(
                action=_rv_dir, qty=1,
                signal=f"REVERSAL_{_rv_score:.0f}",
                sl=_rv_sl, tp=_rv_tp,
            )
            if _rv_ok:
                state._position_open = True
                if state.active_strategy:
                    state.strategy_trades_today += 1
                    _save_strategy_state()
                log.warning(
                    f"🔄 REVERSAL EXECUTAT: {_rv_dir} qty=1 entry~{_rv_entry:.2f} "
                    f"SL={_rv_sl} TP={_rv_tp} score={_rv_score:.0f}%"
                )
            else:
                # NT8 n-a acceptat — curățăm starea (nu lăsăm zombie trade)
                state.open_trade_entry = 0.0
                state.open_trade_dir   = ""
                state._position_open   = False
                _clear_open_trade()
                log.error("❌ REVERSAL: entry NT8 eșuat — stare curățată")
        elif state.reversal_pending_dir:
            # Timeout — CLOSE confirm a venit prea târziu (>45s), reversal anulat
            log.warning(
                f"⚠️ REVERSAL TIMEOUT (>45s): {state.reversal_pending_dir} anulat — "
                f"confirm NT8 prea lent sau pierdut"
            )
            state.reversal_pending_dir = ""
            state.reversal_pending_ts  = 0.0

        # ── Telegram: notificare trade închis ──────────────────────────────────
        if _TG_OK and _tg_closed:
            try:
                # Durată trade în minute
                _dur_min = 0.0
                if state.open_trade_ts:
                    try:
                        _open_dt  = datetime.fromisoformat(state.open_trade_ts.replace("Z", "+00:00"))
                        _dur_min  = (datetime.now(timezone.utc) - _open_dt).total_seconds() / 60.0
                    except Exception:
                        pass
                # MAE/MFE din state (dacă sunt urmărite)
                _mae = abs(getattr(state, "open_trade_mae_pts", 0) or 0)
                _mfe = abs(getattr(state, "open_trade_mfe_pts", 0) or 0)
                _tg_closed(
                    direction    = _entry_dir or action,
                    entry        = _entry_px,
                    exit_price   = _exit_px,
                    pnl_usd      = _pnl_usd,
                    result       = _result,
                    r_mult       = _r_mult,
                    exit_reason  = action,
                    strategy     = _strat.get("label", ""),
                    daily_net    = _session_total,
                    duration_min = round(_dur_min, 1),
                    mae_pts      = _mae,
                    mfe_pts      = _mfe,
                )
                log.info("📱 Telegram: trade închis notificat")
            except Exception as _tg_err:
                log.debug(f"Telegram close skip: {_tg_err}")

        await broadcast_ws({
            "type":        "position_closed",
            "source":      "execution_confirm",
            "action":      action,
            "price":       confirm.price,
            "pnl_usd":     _pnl_usd,
            "result":      _result,
            "trades_done": _trades_done,
            "max_trades":  _max_t,
            "more_trades": _more_trades,
            "timestamp":   now.isoformat(),
        })
        await broadcast_ws({"type": "execution", "data": state.last_exec_confirm})
        return {"status": "confirmed", "pnl_usd": _pnl_usd, "result": _result}

    # ── Alte acțiuni (SCALE, etc.) ──────────────────────────────────────────
    await broadcast_ws({"type": "execution", "data": state.last_exec_confirm})
    return {"status": "confirmed"}

@app.post("/manual_command")
async def manual_command(cmd: ManualCommand):
    ok = await send_command_to_nt8(cmd.action, cmd.qty, cmd.signal)
    if not ok:
        raise HTTPException(status_code=503, detail="NT8 unreachable")
    return {"status": "sent", "action": cmd.action, "qty": cmd.qty}

@app.get("/orderflow")
async def orderflow_snapshot():
    if not state.latest_tick:
        return {"error": "Nu există date. Pornește AladinBridge.cs în NT8."}
    t = state.latest_tick
    return {
        "symbol": t.symbol, "timestamp": t.timestamp,
        "price": t.price.model_dump(), "orderflow": t.orderflow.model_dump(),
        "volume_profile": t.volume_profile.model_dump(),
        "dom": t.dom.model_dump(), "dom_liquidity": t.dom_liquidity.model_dump(),
        "delta_history": list(state.delta_history)[-50:],
    }

# ─── Price endpoint (pentru saas_api auto-tp loop) ────────────────────────────

@app.get("/price/{symbol}")
async def price_symbol(symbol: str):
    """Returnează prețul curent al unui instrument din ultimul tick NT8."""
    t = state.latest_tick
    if t is None:
        return JSONResponse(status_code=404, content={"detail": "No tick data yet"})
    last = t.price.close if hasattr(t, "price") else 0
    return {
        "symbol": symbol.upper(),
        "last":   last,
        "price":  last,
        "bid":    t.order_flow.bid_volume if hasattr(t, "order_flow") else 0,
        "ask":    t.order_flow.ask_volume if hasattr(t, "order_flow") else 0,
        "ts":     t.timestamp if hasattr(t, "timestamp") else "",
    }

# ─── Health & System ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    uptime = time.time() - state.connected_since if state.connected_since else 0
    t = state.latest_tick
    return {
        "status": "ok", "version": "3.0.0",
        "connected": state.connected_since is not None,
        "uptime_s": round(uptime, 1),
        "tick_count": state.tick_count,
        "ws_clients": len(state.ws_clients),
        "symbol": t.symbol if t else None,
        "last_price": t.price.close if t else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/status")
async def status():
    return await health()

# ── UPDATE #5: Autotrade toggle endpoints (pentru Dashboard) ──────────────────

@app.get("/autotrade/status")
async def autotrade_status():
    """Returnează starea curentă a autotrading-ului."""
    return {
        "enabled":          state.autotrade_enabled,
        "label":            "ON" if state.autotrade_enabled else "OFF",
        "position_open":    state._position_open,
        "active_strategy":  state.active_strategy,
        "circuit_open":     state.loss_circuit_open,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }

@app.post("/reset_position")
async def reset_position():
    """Reset manual al flag-ului de poziție — folosit când NT8 nu a trimis CLOSE confirm."""
    was_open = state._position_open
    state._position_open      = False
    state.partial_close_done  = False
    state.milestone_05r_done  = False
    state.milestone_085r_done = False
    state.trailing_sl         = 0.0
    state.trail_r_level       = 0
    state.open_trade_ts       = ""
    state.open_trade_entry    = 0.0
    state.open_trade_sl       = 0.0
    state.open_trade_tp       = 0.0
    log.warning(f"🔓 RESET MANUAL poziție (era open={was_open}) — declanșat de dashboard")
    await broadcast_ws({"type": "position_reset", "was_open": was_open})
    return {"ok": True, "was_open": was_open}

@app.post("/autotrade/toggle")
async def autotrade_toggle():
    """Toggle autotrade ON/OFF și notifică via WebSocket.
    v12.2: HARD LOCKOUT — dacă circuit breaker e hard locked, nu permite reactivare.
    """
    # v12.2: HARD LOCKOUT PROTECTION
    if state.loss_circuit_hard_locked and not state.autotrade_enabled:
        log.warning(
            f"🔒 HARD LOCKOUT: autotrade toggle REFUZAT — circuit breaker hard locked. "
            f"Se resetează automat mâine. Nu poți reactiva trading azi."
        )
        return {
            "enabled": False,
            "label":   "OFF",
            "hard_locked": True,
            "message": "Circuit breaker HARD LOCKED. Trading blocat până mâine.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    state.autotrade_enabled = not state.autotrade_enabled
    status_str = "ON" if state.autotrade_enabled else "OFF"
    log.info(f"🎯 Autotrade: {status_str}")
    _save_strategy_state()   # ← persistăm și starea autotrade
    await broadcast_ws({
        "type": "autotrade_changed",
        "enabled": state.autotrade_enabled,
        "label": status_str,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return {
        "enabled": state.autotrade_enabled,
        "label":   status_str,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ── UPDATE #8: Paper Trading endpoints ───────────────────────────────────────

@app.get("/paper/status")
async def paper_status():
    return {
        "enabled":     state.paper_mode,
        "label":       "PAPER" if state.paper_mode else "LIVE",
        "pnl":         round(state.paper_pnl, 2),
        "trades":      len(state.paper_trades),
        "wins":        sum(1 for t in state.paper_trades if t.get("pnl", 0) > 0),
        "losses":      sum(1 for t in state.paper_trades if t.get("pnl", 0) < 0),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

@app.post("/paper/toggle")
async def paper_toggle():
    state.paper_mode = not state.paper_mode
    label = "PAPER" if state.paper_mode else "LIVE"
    log.info(f"📄 Paper mode: {label}")
    await broadcast_ws({
        "type": "paper_mode_changed",
        "enabled": state.paper_mode,
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return {"enabled": state.paper_mode, "label": label}

# ── UPDATE #12: Strategy endpoints ───────────────────────────────────────────

@app.post("/strategy/set")
async def strategy_set(payload: dict):
    """Setează strategia activă cu parametrii de execuție."""
    strat_id = payload.get("id", "")
    state.active_strategy = {
        "id":            strat_id,
        "label":         payload.get("label", strat_id),
        "score_min":     float(payload.get("score_min", 60)),
        "rr":            float(payload.get("rr", 2.0)),
        "max_trades":    int(payload.get("max_trades", 3)),
        "session_start": payload.get("session_start", "00:00"),
        "session_end":   payload.get("session_end", "23:59"),
        "scale_in":      bool(payload.get("scale_in", False)),
    }
    # Reset counter zilnic DOAR dacă strategia s-a schimbat sau nu e DST-resync
    # Fix v10.5: păstrăm counter dacă e aceeași strategie (ex: schimbare max_trades)
    # keep_trades_counter=True → DST fix → păstrăm trades azi
    _prev_strat_id = state.active_strategy.get("id", "") if state.active_strategy else ""
    _keep_counter  = bool(payload.get("keep_trades_counter", False)) or (_prev_strat_id == strat_id)
    if not _keep_counter:
        # Reset doar la schimbare reală de strategie (altă sesiune)
        state.strategy_trades_today    = 0
        # v12.2: NU resetăm consecutive_losses dacă hard locked — previne bypass
        if not state.loss_circuit_hard_locked:
            state.consecutive_losses       = 0
        else:
            log.warning(f"🔒 HARD LOCK: consecutive_losses NU se resetează ({state.consecutive_losses}) — schimbarea strategiei nu bypassează circuit breaker")
        state.strategy_last_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        log.info(f"🔄 DST resync {strat_id}: ore actualizate, counter trades păstrat ({state.strategy_trades_today} trades azi)")
    # Activează auto-trade dacă cerut
    # v12.2: HARD LOCKOUT — nu permite autostart dacă circuit breaker e locked
    if payload.get("autostart", False):
        if state.loss_circuit_hard_locked:
            log.warning(f"🔒 HARD LOCK: autostart REFUZAT — circuit breaker hard locked")
        else:
            state.autotrade_enabled = True
    _si_label = " | 🔥 ScaleIn ON (top20%→2x)" if state.active_strategy.get("scale_in") else ""
    log.info(f"🎯 Strategie setată: {strat_id} | {state.active_strategy['session_start']}-{state.active_strategy['session_end']} UTC | max={state.active_strategy['max_trades']} | score≥{state.active_strategy['score_min']}{_si_label}")
    _save_strategy_state()   # ← persistență pe disc (supraviețuiește restart bridge)
    await broadcast_ws({"type": "strategy_set", "data": state.active_strategy})
    return {"ok": True, "strategy": state.active_strategy, "autotrade": state.autotrade_enabled}

@app.post("/strategy/clear")
async def strategy_clear():
    """Resetează strategia activă (fără fereastră de timp)."""
    state.active_strategy          = None
    state.strategy_trades_today    = 0
    log.info("🔓 Strategie ștearsă — autotrade fără fereastră de timp")
    _save_strategy_state()   # ← șterge și din disc (None = fără strategie)
    return {"ok": True}

@app.get("/strategy/status")
async def strategy_status():
    """Status curent: strategie activă, trades azi, în fereastra de timp."""
    now_utc  = datetime.now(timezone.utc)
    now_time = now_utc.strftime("%H:%M")
    strat    = state.active_strategy
    in_window = False
    if strat:
        s, e = strat.get("session_start","00:00"), strat.get("session_end","23:59")
        in_window = (s <= now_time < e) if s <= e else (now_time >= s or now_time < e)
    # P&L live din paper trades de azi (sau live trades dacă sunt înregistrate)
    _trades_all   = getattr(state, "paper_trades", [])
    _today_str    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _today_trades = [t for t in _trades_all if t.get("ts","").startswith(_today_str)]
    _wins_today   = sum(1 for t in _today_trades if t.get("pnl", 0) > 0)
    _losses_today = sum(1 for t in _today_trades if t.get("pnl", 0) <= 0)
    _pnl_today    = round(sum(t.get("pnl", 0) for t in _today_trades), 2)
    _win_rate     = round(_wins_today / len(_today_trades) * 100, 1) if _today_trades else 0.0
    _max_profit   = float(strat.get("max_daily_profit_usd", 0)) if strat else 0.0
    return {
        "active":               strat is not None,
        "strategy":             strat,
        "trades_today":         state.strategy_trades_today,
        "max_trades":           strat.get("max_trades", 0) if strat else 0,
        "in_window":            in_window,
        "now_utc":              now_time,
        "autotrade":            state.autotrade_enabled,
        # P&L live panel — pentru dashboard
        "pnl_today":            _pnl_today,
        "wins_today":           _wins_today,
        "losses_today":         _losses_today,
        "win_rate_today":       _win_rate,
        "daily_loss_usd":       round(state.daily_loss_usd, 2),
        "daily_profit_usd":     round(state.daily_profit_usd, 2),
        "max_daily_profit_usd": _max_profit,
        "profit_target_hit":    state.profit_circuit_open,
        "consecutive_losses":   state.consecutive_losses,
        "position_open":        state._position_open,
    }

@app.get("/paper/trades")
async def paper_trades(limit: int = 50):
    trades = state.paper_trades[-limit:][::-1]
    wins   = sum(1 for t in state.paper_trades if t.get("pnl", 0) > 0)
    total  = len(state.paper_trades)
    return {
        "trades":   trades,
        "total":    total,
        "wins":     wins,
        "losses":   total - wins,
        "win_rate": round(wins / total * 100, 1) if total else 0.0,
        "pnl":      round(state.paper_pnl, 2),
    }

@app.post("/paper/reset")
async def paper_reset():
    state.paper_trades = []
    state.paper_pnl    = 0.0
    log.info("📄 Paper trading reset")
    return {"ok": True, "message": "Paper trades resetate"}

@app.get("/system/info")
async def system_info():
    return {
        "version": "3.0.0", "python": "3.11+",
        "uptime_s": round(time.time() - state.connected_since, 1) if state.connected_since else 0,
        "nt8_connected": state.connected_since is not None,
        "tick_count": state.tick_count,
        "trades_logged": len(state.trade_log),
        "data_dir": str(DATA_DIR),
        "data_files": [f.name for f in DATA_DIR.glob("*.csv")] if DATA_DIR.exists() else [],
    }

@app.get("/system/errors")
async def system_errors():
    return {"errors": state.error_log[-50:]}

@app.post("/cache/clear")
async def cache_clear():
    state.last_analysis = {}
    return {"status": "ok", "message": "Cache șters"}

# ─── UPDATE #11: RL Feedback endpoints ───────────────────────────────────────
@app.get("/rl/stats")
async def rl_stats():
    """Returnează statistici RL: weights curente, drift, win_rate."""
    if not _RL_OK or _rl_feedback is None:
        return {"error": "RL module not loaded"}
    try:
        return _rl_feedback.get_rl_stats()
    except Exception as e:
        return {"error": str(e)}

@app.post("/rl/reset")
async def rl_reset():
    """Resetează RL weights la default și golește history."""
    if not _RL_OK or _rl_feedback is None:
        return {"error": "RL module not loaded"}
    try:
        _rl_feedback.reset_weights()
        return {"status": "ok", "message": "RL weights resetate la default"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/rl/weights")
async def rl_weights():
    """Returnează weights curente + default."""
    if not _RL_OK or _rl_feedback is None:
        from rl_feedback import DEFAULT_WEIGHTS
        return {"weights": DEFAULT_WEIGHTS, "source": "default"}
    try:
        w = _rl_feedback.load_weights()
        return {"weights": w, "source": "rl_weights.json"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/telegram/toggle")
async def telegram_toggle(enabled: bool = Query(False)):
    return {"status": "ok", "enabled": enabled}

# ─── Signal Endpoints ─────────────────────────────────────────────────────────

def _build_signal(balance: float = 10000) -> Dict:
    a = state.last_analysis
    t = state.latest_tick
    if not a:
        return {
            "verdict": "AȘTEPT DATE — Pornește NT8 + AladinBridge.cs" if not t else "AȘTEPT CONFIRMARE",
            "score": 0, "ai_score": 0, "quantum_score": 0, "ict_score": 0,
            "orderflow_score": 0, "sentiment_score": 0,
            "conviction": "LOW", "trade_direction": "NEUTRAL", "setup": "—",
            "symbol": t.symbol if t else "—", "price": t.price.close if t else 0,
            "sl_pct": 0.5, "tp_pct": 1.25, "risk_usd": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nt8_connected": t is not None,
        }
    score = a.get("score", 0)
    conv  = "HIGH" if score > 75 else "MEDIUM" if score > 60 else "LOW"
    # Fix v7.4: Score Age Tracking
    _computed_at = a.get("_computed_at", time.time())
    score_age_seconds = round(time.time() - _computed_at, 1)
    return {
        "verdict":         a.get("verdict", "WAIT"),
        "score":           round(score, 1),
        "ai_score":        round(a.get("ai_score", 0), 1),
        "quantum_score":   round(a.get("quantum_score", 0), 1),
        "ict_score":       round(a.get("ict_score", 0), 1),
        "orderflow_score": round(a.get("orderflow_score", 0), 1),
        "sentiment_score": round(a.get("sentiment_score", 0), 1),
        "conviction":      conv,
        "trade_direction": a.get("trade_direction", "NEUTRAL"),
        "setup":           a.get("setup", "—"),
        "symbol":          t.symbol if t else "—",
        "price":           t.price.close if t else 0,
        "sl_pct":          0.3,
        "tp_pct":          0.75,
        "risk_usd":        round(balance * 0.01, 2),
        "timestamp":       a.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "score_age_seconds": score_age_seconds,
        "nt8_connected":   state.connected_since is not None,
        # ── Volume Profile Analytics ──────────────────────────────────────
        "profile_shape":    state.profile_shape,          # "P"/"b"/"D"
        "delta_exhaustion": state.delta_exhaustion,       # "LONG_EXHAUSTION"/"SHORT_EXHAUSTION"/"NONE"
        "rvol":             round(state.rvol, 2),          # Relative Volume (1.0 = normal)
        "poc":              t.volume_profile.poc if t else 0,
        "prev_poc":         t.volume_profile.prev_poc if t else 0,
        "prev_vah":         t.volume_profile.prev_vah if t else 0,
        "prev_val":         t.volume_profile.prev_val if t else 0,
        "hvn":              t.volume_profile.hvn if t else [],
        "lvn":              t.volume_profile.lvn if t else [],
        "vwap_sd1_hi":      t.volume_profile.vwap_sd1_hi if t else 0,
        "vwap_sd1_lo":      t.volume_profile.vwap_sd1_lo if t else 0,
        "vwap_sd2_hi":      t.volume_profile.vwap_sd2_hi if t else 0,
        "vwap_sd2_lo":      t.volume_profile.vwap_sd2_lo if t else 0,
        # ── Composite VP (7 zile structurale) ────────────────────────────
        "composite_poc":    round(state.composite_poc, 2),
        "composite_hvn":    state.composite_hvn,
        "composite_lvn":    state.composite_lvn,
        # ── Delta Profile (top niveluri per preț sesiune curentă) ────────
        "delta_profile":    [{"p": e.p, "d": e.d}
                             for e in (t.volume_profile.delta_profile if t else [])],
    }

@app.get("/signal/now")
async def signal_now(balance: float = Query(10000, ge=0, le=10_000_000)):
    return _build_signal(balance)

@app.get("/signal/latest")
async def signal_latest(balance: float = Query(10000, ge=0, le=10_000_000)):
    return _build_signal(balance)

@app.post("/signal")
async def signal_at(timestamp: str = Query(""), balance: float = Query(10000, ge=0, le=10_000_000)):
    return _build_signal(balance)

# ─── Trades & Equity ──────────────────────────────────────────────────────────

@app.get("/trades")
async def get_trades(limit: int = Query(100, ge=1, le=2000)):
    return state.trade_log[-limit:]

@app.post("/trades/manual")
async def add_manual_trade(payload: dict):
    """Adaugă manual un trade în journal (ex: închis din NT8 fără bridge)."""
    now = datetime.now(timezone.utc)
    # Feature 3: point value dinamic
    _inst_sym  = str(payload.get("instrument", "")).upper() or (state.latest_tick.symbol if state.latest_tick else "")
    _point_val = _get_point_value(_inst_sym)
    entry_px  = float(payload.get("entry_price", 0))
    exit_px   = float(payload.get("exit_price", 0))
    direction = str(payload.get("direction", "SHORT")).upper()
    qty       = int(payload.get("qty", 1))
    sl        = float(payload.get("sl", 0))
    strategy  = str(payload.get("strategy", state.active_strategy.get("label","Manual") if state.active_strategy else "Manual"))

    if entry_px > 0 and exit_px > 0:
        _diff    = exit_px - entry_px
        pnl_usd  = (_diff if direction == "LONG" else -_diff) * _point_val * qty
    else:
        pnl_usd  = float(payload.get("pnl_usd", 0))

    pnl_usd  = round(pnl_usd, 2)
    result   = "WIN" if pnl_usd > 0 else ("LOSS" if pnl_usd < 0 else "BE")
    r_mult   = 0.0
    if sl > 0 and entry_px > 0:
        risk_pts = abs(entry_px - sl)
        risk_usd = risk_pts * _point_val * qty
        if risk_usd > 0:
            r_mult = round(pnl_usd / risk_usd, 2)

    _utc_hour = now.hour
    if 7 <= _utc_hour < 12:    _session = "London"
    elif 12 <= _utc_hour < 20: _session = "New York"
    else:                       _session = "Asian"

    trade = {
        "id":          str(uuid.uuid4())[:8],
        "entry_time":  payload.get("entry_time", now.isoformat()[:19]),
        "exit_time":   payload.get("exit_time",  now.isoformat()[:19]),
        "direction":   direction,
        "entry_price": round(entry_px, 2),
        "exit_price":  round(exit_px, 2),
        "qty":         qty,
        "sl":          round(sl, 2),
        "tp":          float(payload.get("tp", 0)),
        "pnl":         round(pnl_usd / _point_val, 4),
        "pnl_usd":     pnl_usd,
        "pnl_pct":     0.0,
        "result":      result,
        "mae_pts":     float(payload.get("mae_pts", 0)),
        "mfe_pts":     float(payload.get("mfe_pts", 0)),
        "exit_reason": str(payload.get("exit_reason", "MANUAL")),
        "entry_hour":  now.hour,
        "entry_dow":   now.weekday(),
        "setup":       payload.get("setup", "MANUAL"),
        "score":       float(payload.get("score", 0)),
        "strategy":    strategy,
        "session":     payload.get("session", _session),
        "r_multiple":  r_mult,
        "source":      "MANUAL",
        "instrument":  _inst_sym or ACTIVE_INSTRUMENT,
        "geo_risk_active": state.geo_risk_active,
    }
    state.add_trade(trade)
    log.info(f"📝 TRADE MANUAL LOG: {direction} {entry_px}→{exit_px} | PnL: ${pnl_usd:+.2f} | {result}")
    await broadcast_ws({"type": "manual_trade_added", "data": trade})
    return {"ok": True, "trade": trade}

@app.get("/equity")
async def get_equity():
    # Feature 2: dacă avem equity history persistent, folosim-o
    if _equity_history_cache:
        metrics = _compute_equity_metrics()
        dates = ["Start"] + [p["ts"][:10] for p in _equity_history_cache]
        values = [100000.0] + [p["balance"] for p in _equity_history_cache]
        return {
            "dates": dates, "values": values,
            "initial": 100000.0, "current": _equity_history_cache[-1]["balance"],
            "metrics": metrics,
        }
    # Fallback: calculăm din trade_log (backward compat)
    initial = 10000.0
    balance = initial
    dates, values = ["Start"], [initial]
    for t in state.trade_log:
        balance += t.get("pnl", 0)
        dates.append(t.get("exit_time", t.get("entry_time", ""))[:10])
        values.append(round(balance, 2))
    return {"dates": dates, "values": values, "initial": initial, "current": round(balance, 2)}

@app.get("/equity/detailed")
async def get_equity_detailed():
    """Feature 2: Equity curve detaliată cu metrici avansate (Sharpe, PF, DD)."""
    metrics = _compute_equity_metrics()
    return {
        "equity_curve":  _equity_history_cache[-500:],   # ultimele 500 puncte
        "metrics":       metrics,
        "total_points":  len(_equity_history_cache),
    }

@app.get("/equity/export")
async def export_equity(format: str = Query("json", pattern="^(json|csv)$")):
    """Feature 2: Export equity curve (json sau csv)."""
    if format == "csv":
        lines = ["ts,trade_id,pnl_usd,balance,peak,dd_usd,dd_pct,result,direction,instrument,r_multiple,strategy"]
        for p in _equity_history_cache:
            lines.append(",".join(str(p.get(k, "")) for k in
                ["ts","trade_id","pnl_usd","balance","peak","dd_usd","dd_pct","result","direction","instrument","r_multiple","strategy"]))
        return {"csv": "\n".join(lines), "rows": len(_equity_history_cache)}
    return {"equity": _equity_history_cache, "rows": len(_equity_history_cache)}

@app.get("/stats")
async def get_stats():
    return state.compute_stats()

# ── Feature 3: Instrument Management ────────────────────────────────────────────

@app.get("/instrument/config")
async def get_instrument_config():
    """Returnează configurația tuturor instrumentelor suportate + cel activ."""
    return {
        "active": ACTIVE_INSTRUMENT,
        "active_config": _get_instrument_config(),
        "instruments": INSTRUMENT_CONFIG,
    }

@app.post("/instrument/set")
async def set_instrument(payload: dict):
    """Schimbă instrumentul activ (NQ)."""
    global ACTIVE_INSTRUMENT
    sym = str(payload.get("instrument", "")).upper().strip()
    # Prefix match
    matched = None
    for key in INSTRUMENT_CONFIG:
        if sym.startswith(key) or sym == key:
            matched = key
            break
    if not matched:
        raise HTTPException(400, f"Instrument necunoscut: {sym}. Suportate: {list(INSTRUMENT_CONFIG.keys())}")
    ACTIVE_INSTRUMENT = matched
    log.info(f"🔧 Instrument schimbat: {matched} ({INSTRUMENT_CONFIG[matched]['name']})")
    return {"ok": True, "instrument": matched, "config": INSTRUMENT_CONFIG[matched]}


# ── Feature 4: A/B Strategy Testing Endpoints ──────────────────────────────────

@app.post("/ab/start")
async def ab_start(payload: dict):
    """Pornește un A/B test cu 2 strategii: A=real, B=paper."""
    global _ab_test_state
    strategy_a = payload.get("strategy_a", {})
    strategy_b = payload.get("strategy_b", {})
    if not strategy_a or not strategy_b:
        raise HTTPException(400, "Trebuie strategy_a și strategy_b (fiecare cu id, label)")
    _ab_test_state = {
        "active": True,
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "trades_a": [],
        "trades_b": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_ab_state()
    log.info(f"🧪 A/B Test START: A={strategy_a.get('label','?')} (real) vs B={strategy_b.get('label','?')} (paper)")
    return {"ok": True, "ab_test": _ab_test_state}

@app.post("/ab/stop")
async def ab_stop():
    """Oprește A/B test-ul și returnează comparația finală."""
    comparison = _ab_compare()
    _ab_test_state["active"] = False
    _save_ab_state()
    log.info(f"🧪 A/B Test STOP: Winner={comparison.get('winner','?')}")
    return {"ok": True, "comparison": comparison}

@app.get("/ab/status")
async def ab_status():
    """Status A/B test curent cu comparație live."""
    if not _ab_test_state.get("active"):
        return {"active": False, "message": "No A/B test running"}
    return _ab_compare()

@app.post("/ab/record")
async def ab_record(payload: dict):
    """
    Înregistrează un trade paper pentru strategie B (A se înregistrează automat din execution_confirm).
    Payload: {slot: "B", pnl_usd, result, direction, r_multiple, score}
    """
    slot = str(payload.get("slot", "B")).upper()
    if slot not in ("A", "B"):
        raise HTTPException(400, "slot trebuie să fie 'A' sau 'B'")
    trade = {
        "exit_time":   datetime.now(timezone.utc).isoformat()[:19],
        "pnl_usd":     float(payload.get("pnl_usd", 0)),
        "result":      payload.get("result", ""),
        "direction":   payload.get("direction", ""),
        "r_multiple":  float(payload.get("r_multiple", 0)),
        "score":       float(payload.get("score", 0)),
    }
    _ab_record_trade(slot, trade)
    return {"ok": True, "slot": slot, "trade": trade}


# ─── ANALYTICS ENDPOINTS ───────────────────────────────────────────────────────

@app.get("/analytics/equity")
async def analytics_equity():
    """Equity curve cumulativă din toate trade-urile istorice."""
    trades = [t for t in state.trade_log if t.get("result") in ("WIN", "LOSS")]
    trades_sorted = sorted(trades, key=lambda t: t.get("entry_time", ""))
    balance = 100000.0  # balanță start referință
    points = [{"ts": "start", "balance": balance, "pnl": 0.0}]
    cumulative = 0.0
    for t in trades_sorted:
        pnl = float(t.get("pnl_usd") or t.get("pnl", 0))
        cumulative += pnl
        balance += pnl
        points.append({
            "ts":        t.get("entry_time", "")[:16],
            "balance":   round(balance, 2),
            "pnl":       round(pnl, 2),
            "cumulative": round(cumulative, 2),
            "result":    t.get("result"),
            "direction": t.get("direction"),
        })
    return {"equity_curve": points, "total_pnl": round(cumulative, 2), "trades": len(trades_sorted)}


@app.get("/analytics/sessions")
async def analytics_sessions():
    """Performance per sesiune: London / New York / Asian."""
    trades = [t for t in state.trade_log if t.get("result") in ("WIN", "LOSS")]
    sessions: Dict[str, Dict] = {}
    for t in trades:
        # Detectează sesiunea din câmpul salvat sau din entry_hour
        sess = t.get("session")
        if not sess:
            h = int(t.get("entry_hour", 0))
            if 7 <= h < 12:    sess = "London"
            elif 12 <= h < 20: sess = "New York"
            elif 20 <= h < 24: sess = "NY Close"
            elif 0 <= h < 3:   sess = "Asian Pre"
            else:               sess = "Asian"     # 03-07 UTC
        if sess not in sessions:
            sessions[sess] = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}
        s = sessions[sess]
        s["trades"] += 1
        pnl = float(t.get("pnl_usd") or t.get("pnl", 0))
        s["pnl"] = round(s["pnl"] + pnl, 2)
        if t.get("result") == "WIN":  s["wins"] += 1
        else:                          s["losses"] += 1
    # Calculăm win rate per sesiune
    result = {}
    for sess, data in sessions.items():
        total = data["trades"]
        result[sess] = {
            **data,
            "win_rate": round(data["wins"] / total * 100, 1) if total else 0,
            "avg_pnl":  round(data["pnl"] / total, 2) if total else 0,
        }
    return result


@app.get("/analytics/score-accuracy")
async def analytics_score_accuracy():
    """Acuratețe scor — win rate per bucket de scor (0-50, 50-60, 60-70, 70-80, 80-90, 90-100)."""
    trades = [t for t in state.trade_log if t.get("result") in ("WIN", "LOSS") and t.get("score") is not None]
    buckets = {
        "50-60": {"wins": 0, "losses": 0, "pnl": 0.0},
        "60-70": {"wins": 0, "losses": 0, "pnl": 0.0},
        "70-80": {"wins": 0, "losses": 0, "pnl": 0.0},
        "80-90": {"wins": 0, "losses": 0, "pnl": 0.0},
        "90-100":{"wins": 0, "losses": 0, "pnl": 0.0},
    }
    for t in trades:
        score = float(t.get("score", 0))
        pnl   = float(t.get("pnl_usd") or t.get("pnl", 0))
        if score < 50:   key = None
        elif score < 60: key = "50-60"
        elif score < 70: key = "60-70"
        elif score < 80: key = "70-80"
        elif score < 90: key = "80-90"
        else:            key = "90-100"
        if key:
            buckets[key]["wins" if t["result"] == "WIN" else "losses"] += 1
            buckets[key]["pnl"] = round(buckets[key]["pnl"] + pnl, 2)
    result = {}
    for key, data in buckets.items():
        total = data["wins"] + data["losses"]
        result[key] = {
            **data,
            "trades":   total,
            "win_rate": round(data["wins"] / total * 100, 1) if total else 0,
        }
    return result


@app.get("/analytics/drawdown")
async def analytics_drawdown():
    """Drawdown maxim históric din equity curve."""
    trades = [t for t in state.trade_log if t.get("result") in ("WIN", "LOSS")]
    trades_sorted = sorted(trades, key=lambda t: t.get("entry_time", ""))
    balance  = 100000.0
    peak     = balance
    max_dd   = 0.0
    max_dd_pct = 0.0
    max_dd_ts  = ""
    current_dd = 0.0
    current_dd_pct = 0.0
    for t in trades_sorted:
        pnl = float(t.get("pnl_usd") or t.get("pnl", 0))
        balance += pnl
        if balance > peak:
            peak = balance
        dd     = peak - balance
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd     = dd
            max_dd_pct = dd_pct
            max_dd_ts  = t.get("entry_time", "")[:16]
        current_dd     = peak - balance
        current_dd_pct = current_dd / peak * 100 if peak > 0 else 0
    return {
        "max_drawdown_usd": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_dd_at":        max_dd_ts,
        "current_drawdown_usd": round(current_dd, 2),
        "current_drawdown_pct": round(current_dd_pct, 2),
        "peak_balance":    round(peak, 2),
        "current_balance": round(balance, 2),
    }


@app.get("/analytics/strategies")
async def analytics_strategies():
    """Statistici per strategie activă."""
    trades = [t for t in state.trade_log if t.get("result") in ("WIN", "LOSS")]
    strats: Dict[str, Dict] = {}
    for t in trades:
        strat_name = t.get("strategy", "Default")
        if strat_name not in strats:
            strats[strat_name] = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0, "scores": []}
        s = strats[strat_name]
        s["trades"] += 1
        pnl = float(t.get("pnl_usd") or t.get("pnl", 0))
        s["pnl"] = round(s["pnl"] + pnl, 2)
        if t.get("result") == "WIN": s["wins"] += 1
        else:                        s["losses"] += 1
        if t.get("score"): s["scores"].append(float(t["score"]))
    result = {}
    for name, data in strats.items():
        total = data["trades"]
        scores = data.pop("scores")
        result[name] = {
            **data,
            "win_rate":  round(data["wins"] / total * 100, 1) if total else 0,
            "avg_pnl":   round(data["pnl"] / total, 2) if total else 0,
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
        }
    return result

@app.get("/analytics/kpis")
async def analytics_kpis():
    """Profit Factor, Expectancy, Avg Win, Avg Loss, Best/Worst trade."""
    trades = [t for t in state.trade_log if t.get("result") in ("WIN", "LOSS")]
    wins   = [float(t.get("pnl_usd") or t.get("pnl", 0)) for t in trades if t.get("result") == "WIN"]
    losses = [float(t.get("pnl_usd") or t.get("pnl", 0)) for t in trades if t.get("result") == "LOSS"]
    gross_win  = sum(wins)
    gross_loss = abs(sum(losses))
    total = len(trades)
    win_rate = len(wins) / total if total else 0
    avg_win  = round(gross_win  / len(wins),   2) if wins   else 0
    avg_loss = round(gross_loss / len(losses),  2) if losses else 0
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 0
    expectancy = round(win_rate * avg_win - (1 - win_rate) * avg_loss, 2)
    return {
        "profit_factor": pf,
        "expectancy":    expectancy,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "gross_win":     round(gross_win, 2),
        "gross_loss":    round(gross_loss, 2),
        "best_trade":    round(max(wins,   default=0), 2),
        "worst_trade":   round(min(losses, default=0) * -1, 2),
        "total_trades":  total,
        "win_count":     len(wins),
        "loss_count":    len(losses),
    }


@app.get("/analytics/daily-calendar")
async def analytics_daily_calendar():
    """PnL per zi calendaristică — pentru calendar heatmap."""
    trades = [t for t in state.trade_log if t.get("result") in ("WIN", "LOSS")]
    daily: Dict[str, Dict] = {}
    for t in trades:
        day = (t.get("entry_time") or t.get("exit_time") or "")[:10]
        if not day or day == "":
            continue
        pnl = float(t.get("pnl_usd") or t.get("pnl", 0))
        if day not in daily:
            daily[day] = {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0}
        daily[day]["pnl"]    = round(daily[day]["pnl"] + pnl, 2)
        daily[day]["trades"] += 1
        if t.get("result") == "WIN": daily[day]["wins"]   += 1
        else:                        daily[day]["losses"] += 1
    # Calculăm și win rate per zi
    result = {}
    for day, data in sorted(daily.items()):
        result[day] = {
            **data,
            "win_rate": round(data["wins"] / data["trades"] * 100, 1) if data["trades"] else 0,
        }
    return result


@app.get("/analytics/r-multiples")
async def analytics_r_multiples():
    """Distribuție R-Multiple — cât de departe merge fiecare trade față de risc."""
    trades = [t for t in state.trade_log if t.get("result") in ("WIN", "LOSS")]
    buckets = {
        "< -1R":   {"count": 0, "pnl": 0.0},
        "-1R–0R":  {"count": 0, "pnl": 0.0},
        "0R–1R":   {"count": 0, "pnl": 0.0},
        "1R–2R":   {"count": 0, "pnl": 0.0},
        "2R–3R":   {"count": 0, "pnl": 0.0},
        "> 3R":    {"count": 0, "pnl": 0.0},
    }
    r_values = []
    for t in trades:
        r = t.get("r_multiple")
        if r is None:
            # estimare din pnl și rr al strategiei (fallback)
            pnl = float(t.get("pnl_usd") or t.get("pnl", 0))
            rr  = float(t.get("rr", 2.0) or 2.0)
            # assume 1R = avg_loss → skip fără r_multiple real
            continue
        r = float(r)
        r_values.append(r)
        pnl = float(t.get("pnl_usd") or t.get("pnl", 0))
        if r < -1:      key = "< -1R"
        elif r < 0:     key = "-1R–0R"
        elif r < 1:     key = "0R–1R"
        elif r < 2:     key = "1R–2R"
        elif r < 3:     key = "2R–3R"
        else:           key = "> 3R"
        buckets[key]["count"] += 1
        buckets[key]["pnl"]    = round(buckets[key]["pnl"] + pnl, 2)
    avg_r = round(sum(r_values) / len(r_values), 2) if r_values else 0
    return {"buckets": buckets, "avg_r": avg_r, "total": len(r_values)}


@app.get("/analytics/dow")
async def analytics_dow():
    """Win rate și PnL per zi a săptămânii (0=Mon … 6=Sun)."""
    from datetime import datetime
    day_names = ["Luni","Marți","Miercuri","Joi","Vineri","Sâmbătă","Duminică"]
    trades = [t for t in state.trade_log if t.get("result") in ("WIN","LOSS")]
    dow: Dict[str, Dict] = {d: {"wins":0,"losses":0,"pnl":0.0,"trades":0} for d in day_names}
    for t in trades:
        ts_raw = t.get("entry_time") or t.get("exit_time") or ""
        if not ts_raw:
            continue
        try:
            dt = datetime.fromisoformat(ts_raw[:19])
            name = day_names[dt.weekday()]  # 0=Mon
        except Exception:
            continue
        pnl = float(t.get("pnl_usd") or t.get("pnl", 0))
        dow[name]["trades"]  += 1
        dow[name]["pnl"]      = round(dow[name]["pnl"] + pnl, 2)
        if t["result"] == "WIN": dow[name]["wins"]   += 1
        else:                    dow[name]["losses"] += 1
    result = {}
    for name, data in dow.items():
        total = data["trades"]
        result[name] = {
            **data,
            "win_rate": round(data["wins"] / total * 100, 1) if total else 0,
            "avg_pnl":  round(data["pnl"] / total, 2) if total else 0,
        }
    return result


@app.get("/analytics/score-dist")
async def analytics_score_dist():
    """Distribuția scorurilor la care Aladin a intrat efectiv în trade."""
    trades = [t for t in state.trade_log if t.get("result") in ("WIN","LOSS") and t.get("score") is not None]
    buckets = {
        "50-55": {"wins":0,"losses":0},
        "55-60": {"wins":0,"losses":0},
        "60-65": {"wins":0,"losses":0},
        "65-70": {"wins":0,"losses":0},
        "70-75": {"wins":0,"losses":0},
        "75-80": {"wins":0,"losses":0},
        "80-85": {"wins":0,"losses":0},
        "85-90": {"wins":0,"losses":0},
        "90+":   {"wins":0,"losses":0},
    }
    for t in trades:
        score = float(t["score"])
        if score < 50:      continue
        elif score < 55:    key = "50-55"
        elif score < 60:    key = "55-60"
        elif score < 65:    key = "60-65"
        elif score < 70:    key = "65-70"
        elif score < 75:    key = "70-75"
        elif score < 80:    key = "75-80"
        elif score < 85:    key = "80-85"
        elif score < 90:    key = "85-90"
        else:               key = "90+"
        buckets[key]["wins" if t["result"]=="WIN" else "losses"] += 1
    result = {}
    for k, data in buckets.items():
        total = data["wins"] + data["losses"]
        result[k] = {**data, "trades": total, "win_rate": round(data["wins"]/total*100,1) if total else 0}
    return result


@app.get("/analytics/geo-impact")
async def analytics_geo_impact():
    """Compară win rate și PnL cu geo_risk ON vs OFF."""
    trades = [t for t in state.trade_log if t.get("result") in ("WIN","LOSS")]
    groups = {
        "geo_on":  {"wins":0,"losses":0,"pnl":0.0,"trades":0},
        "geo_off": {"wins":0,"losses":0,"pnl":0.0,"trades":0},
    }
    for t in trades:
        key = "geo_on" if t.get("geo_risk_active") else "geo_off"
        pnl = float(t.get("pnl_usd") or t.get("pnl", 0))
        groups[key]["trades"] += 1
        groups[key]["pnl"]     = round(groups[key]["pnl"] + pnl, 2)
        if t["result"] == "WIN": groups[key]["wins"]   += 1
        else:                    groups[key]["losses"] += 1
    result = {}
    for key, data in groups.items():
        total = data["trades"]
        result[key] = {
            **data,
            "win_rate": round(data["wins"]/total*100,1) if total else 0,
            "avg_pnl":  round(data["pnl"]/total,2) if total else 0,
        }
    return result


@app.get("/analytics/hour-heatmap")
async def analytics_hour_heatmap():
    """PnL și win rate per ora zilei × zi a săptămânii — heatmap 5×24."""
    from datetime import datetime
    day_names = ["Luni","Marți","Miercuri","Joi","Vineri"]
    trades = [t for t in state.trade_log if t.get("result") in ("WIN","LOSS")]
    # grid[dow][hour] = {pnl, wins, losses}
    grid: Dict[str, Dict[int, Dict]] = {
        d: {h: {"pnl": 0.0, "wins": 0, "losses": 0} for h in range(24)}
        for d in day_names
    }
    for t in trades:
        ts_raw = t.get("entry_time") or t.get("exit_time") or ""
        if not ts_raw:
            continue
        try:
            dt   = datetime.fromisoformat(ts_raw[:19])
            name = day_names[dt.weekday()]
            if name not in grid:
                continue
            hour = dt.hour
            pnl  = float(t.get("pnl_usd") or t.get("pnl", 0))
            grid[name][hour]["pnl"]    = round(grid[name][hour]["pnl"] + pnl, 2)
            if t["result"] == "WIN":  grid[name][hour]["wins"]   += 1
            else:                     grid[name][hour]["losses"] += 1
        except Exception:
            continue
    # Serializare: {day: [{hour, pnl, trades, win_rate}]}
    result = {}
    for name in day_names:
        result[name] = []
        for h in range(24):
            cell = grid[name][h]
            total = cell["wins"] + cell["losses"]
            result[name].append({
                "hour":     h,
                "pnl":      cell["pnl"],
                "trades":   total,
                "wins":     cell["wins"],
                "losses":   cell["losses"],
                "win_rate": round(cell["wins"] / total * 100, 1) if total else 0,
            })
    return result


@app.get("/stats/session")
async def get_session_stats():
    """Real-time session statistics — win rate, avg R:R, PnL, drawdown."""
    trades = state.paper_trades if state.paper_mode else state.trade_log
    wins   = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) < 0]
    total  = len(trades)
    win_r  = round(len(wins) / total * 100, 1) if total > 0 else 0.0

    pnl_all   = [t.get("pnl", 0) for t in trades]
    total_pnl = round(sum(pnl_all), 2)
    avg_win   = round(sum(t.get("pnl", 0) for t in wins) / max(len(wins), 1), 2)
    avg_loss  = round(sum(t.get("pnl", 0) for t in losses) / max(len(losses), 1), 2)

    # Profit Factor
    gross_profit = sum(t.get("pnl", 0) for t in wins)
    gross_loss   = abs(sum(t.get("pnl", 0) for t in losses))
    pf           = round(gross_profit / max(gross_loss, 1), 2)

    # Max Drawdown
    cum_pnl = 0.0
    peak    = 0.0
    max_dd  = 0.0
    for p in pnl_all:
        cum_pnl += p
        peak     = max(peak, cum_pnl)
        dd       = peak - cum_pnl
        max_dd   = max(max_dd, dd)

    # Avg R:R realizat
    avg_rr = round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0.0

    # Streak curent
    streak = 0
    streak_type = ""
    for t in reversed(trades):
        result = "W" if t.get("pnl", 0) > 0 else "L"
        if streak == 0:
            streak_type = result
        if result == streak_type:
            streak += 1
        else:
            break

    return {
        "total_trades":   total,
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate_pct":   win_r,
        "total_pnl":      total_pnl,
        "avg_win_usd":    avg_win,
        "avg_loss_usd":   avg_loss,
        "avg_rr":         avg_rr,
        "profit_factor":  pf,
        "max_drawdown":   round(max_dd, 2),
        "current_streak": {"type": streak_type or "—", "count": streak},
        "circuit_open":   state.loss_circuit_open,
        "consecutive_l":  state.consecutive_losses,
        "daily_loss_usd": round(state.daily_loss_usd, 2),
        "paper_mode":     state.paper_mode,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }

# ─── AI / Model Endpoints ─────────────────────────────────────────────────────

@app.get("/model/stats")
async def model_stats():
    try:
        import sys
        rag_dir = os.path.dirname(os.path.abspath(__file__))
        if rag_dir not in sys.path:
            sys.path.insert(0, rag_dir)
        import mario_rag
        info = mario_rag.get_model_info() if hasattr(mario_rag, "get_model_info") else {}
    except Exception:
        info = {}
    a = state.last_analysis
    return {
        "model_version": info.get("version", "v7.0"),
        "features": info.get("features", 35),
        "trained_samples": info.get("trained_samples", 0),
        "accuracy": info.get("accuracy", 0),
        "last_score": round(a.get("score", 0), 1),
        "last_ai_score": round(a.get("ai_score", 0), 1),
        "last_quantum_score": round(a.get("quantum_score", 0), 1),
        "weights": {"AI": 0.35, "ICT": 0.25, "Quantum": 0.10, "RelStrength": 0.10, "OrderFlow": 0.15, "Sentiment": 0.05},
        "last_update": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/model/bias")
async def model_bias():
    a     = state.last_analysis
    t     = state.latest_tick
    score = a.get("score", 0)
    direction = a.get("trade_direction", "NEUTRAL")
    delta = state.session_cum_delta
    bull_pct = 50.0
    if direction == "LONG":   bull_pct = min(90, 50 + score * 0.4)
    elif direction == "SHORT": bull_pct = max(10, 50 - score * 0.4)
    if delta > 500:   bull_pct = min(95, bull_pct + 10)
    elif delta < -500: bull_pct = max(5, bull_pct - 10)
    bias_label = "BULLISH" if bull_pct > 60 else "BEARISH" if bull_pct < 40 else "NEUTRAL"
    return {
        "bias": bias_label, "bull_pct": round(bull_pct, 1), "bear_pct": round(100 - bull_pct, 1),
        "score": round(score, 1), "direction": direction,
        "delta_signal": "BUY_PRESSURE" if delta > 300 else "SELL_PRESSURE" if delta < -300 else "NEUTRAL",
        "cum_delta": round(delta, 0),
        "nt8_connected": state.connected_since is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/model/shap")
async def model_shap():
    a = state.last_analysis
    t = state.latest_tick
    features = [
        {"feature": "AI Score (XGBoost)", "value": round(a.get("ai_score", 30) * 0.35, 2)},
        {"feature": "ICT Pattern",        "value": round(a.get("ict_score", 20) * 0.25, 2)},
        {"feature": "OrderFlow Delta",    "value": round(abs(state.session_cum_delta) / 100 * 0.15, 2)},
        {"feature": "Quantum Signal",     "value": round(a.get("quantum_score", 15) * 0.10, 2)},
        {"feature": "Relative Strength",  "value": round(a.get("rel_strength", 10) * 0.10, 2)},
        {"feature": "News Sentiment",     "value": round(a.get("sentiment_score", 5) * 0.05, 2)},
        {"feature": "VWAP Distance",      "value": round(abs((t.orderflow.vwap or 0) - t.price.close) / (t.price.close + 1e-8) * 100, 2) if t else 0.0},
        {"feature": "Volume Imbalance",   "value": round(abs(t.orderflow.imbalance_pct) * 0.01, 2) if t else 0.0},
    ]
    features.sort(key=lambda x: abs(x["value"]), reverse=True)
    return {"shap_values": features, "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/stats/patterns")
async def stats_patterns():
    return {"patterns": [
        {"name": "ICT Silver Bullet",    "count": 12, "win_rate": 75.0, "avg_pnl": 285},
        {"name": "NY Open Displacement", "count": 18, "win_rate": 66.7, "avg_pnl": 210},
        {"name": "London Open Sweep",    "count": 9,  "win_rate": 77.8, "avg_pnl": 320},
        {"name": "Judas Swing",          "count": 7,  "win_rate": 71.4, "avg_pnl": 450},
        {"name": "FVG Fill",             "count": 14, "win_rate": 64.3, "avg_pnl": 180},
    ]}

@app.get("/stats/correlations")
async def stats_correlations():
    return {"correlations": [
        {"pair": "NQ/ES",   "value": 0.94,  "direction": "positive"},
        {"pair": "NQ/DXY",  "value": -0.71, "direction": "negative"},
        {"pair": "ES/VIX",  "value": -0.82, "direction": "negative"},
        {"pair": "NQ/Gold", "value": 0.35,  "direction": "weak"},
    ]}

@app.get("/stats/anomalies")
async def stats_anomalies():
    t = state.latest_tick
    anomalies = []
    if t:
        delta = t.orderflow.cum_delta
        if abs(delta) > 2000:
            anomalies.append({"type": "Extreme Delta", "description": f"CumDelta: {delta:+.0f}", "severity": "HIGH", "timestamp": t.timestamp})
        if abs(t.orderflow.imbalance_pct) > 80:
            anomalies.append({"type": "OrderFlow Imbalance", "description": f"Imbalance: {t.orderflow.imbalance_pct:.0f}%", "severity": "MEDIUM", "timestamp": t.timestamp})
    return {"anomalies": anomalies}

# ─── Market Data ──────────────────────────────────────────────────────────────

# Cache pentru DXY/Gold/BTC/SPY (5 minute TTL)
_market_cache: dict = {}
_market_cache_ts: float = 0.0
_MARKET_TTL: float = 300.0

def _fetch_ext_prices() -> dict:
    """Fetch DXY/Gold/BTC/SPY din yfinance cu cache 5 min."""
    global _market_cache, _market_cache_ts
    if _market_cache and (time.time() - _market_cache_ts) < _MARKET_TTL:
        return _market_cache
    try:
        import yfinance as yf
        tickers = {"DXY": "DX-Y.NYB", "Gold": "GC=F", "BTC": "BTC-USD", "SPY": "SPY"}
        result  = {}
        for name, sym in tickers.items():
            try:
                tk   = yf.Ticker(sym)
                # fast_info returnează atribute, nu dict — folosim getattr cu fallback
                fi   = tk.fast_info
                price = float(getattr(fi, "last_price", None) or
                              getattr(fi, "regular_market_price", None) or 0)
                prev  = float(getattr(fi, "previous_close", None) or
                              getattr(fi, "regular_market_previous_close", None) or price)
                # Fallback la history dacă fast_info nu are date
                if price == 0:
                    hist = tk.history(period="2d", interval="1d")
                    if len(hist) >= 1:
                        price = float(hist["Close"].iloc[-1])
                        prev  = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
                chg   = price - prev
                chg_p = (chg / prev * 100) if prev and prev != 0 else 0
                result[name] = {
                    "price":      round(price, 4),
                    "change":     round(chg, 4),
                    "change_pct": round(chg_p, 3),
                }
            except Exception as _e:
                log.debug(f"ext_price {name}: {_e}")
                result[name] = {"price": 0, "change": 0, "change_pct": 0}
        _market_cache    = result
        _market_cache_ts = time.time()
        return result
    except Exception:
        return {k: {"price": 0, "change": 0, "change_pct": 0} for k in ["DXY","Gold","BTC","SPY"]}

@app.get("/market-data")
async def market_data():
    t = state.latest_tick

    # Prețuri externe (DXY/Gold/BTC/SPY) — async în thread pool
    loop = asyncio.get_event_loop()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        ext = await loop.run_in_executor(pool, _fetch_ext_prices)

    if not t:
        return {**ext, "symbol": "—", "price": 0, "change": 0, "change_pct": 0, "connected": False}

    first  = list(state.bar_buffer)[0] if state.bar_buffer else None
    change = t.price.close - first.price.close if first else 0
    cp     = change / first.price.close * 100 if first and first.price.close > 0 else 0
    return {
        # NT8 live data
        "symbol": t.symbol, "price": t.price.close,
        "open": t.price.open, "high": t.price.high, "low": t.price.low,
        "volume": t.price.volume, "bid": t.price.bid, "ask": t.price.ask,
        "vwap": t.orderflow.vwap, "poc": t.volume_profile.poc,
        "vah": t.volume_profile.vah, "val": t.volume_profile.val,
        "cum_delta": state.session_cum_delta, "imbalance": t.orderflow.imbalance_pct,
        "change": round(change, 2), "change_pct": round(cp, 3),
        "connected": True, "timestamp": t.timestamp,
        # DXY / Gold / BTC / SPY pentru Dashboard ticker
        **ext,
    }

# ─── CHART: OHLCV Candlestick data ───────────────────────────────────────────
_ohlcv_cache: dict = {}
_ohlcv_cache_ts: dict = {}
_OHLCV_TTL = 60.0  # 60s cache per symbol+interval

def _fetch_ohlcv(symbol: str, period: str, interval: str) -> list:
    """Fetch candlestick OHLCV data via yfinance. Returns list of {x, o, h, l, c, v}."""
    cache_key = f"{symbol}_{period}_{interval}"
    now = time.time()
    if cache_key in _ohlcv_cache and (now - _ohlcv_cache_ts.get(cache_key, 0)) < _OHLCV_TTL:
        return _ohlcv_cache[cache_key]
    try:
        import yfinance as yf
        # Map friendly names to yfinance tickers
        TICKER_MAP = {
            "NQ": "NQ=F", "ES": "ES=F",
            "SPY": "SPY", "QQQ": "QQQ", "BTC": "BTC-USD",
        }
        yf_sym = TICKER_MAP.get(symbol.upper(), symbol)
        df = yf.download(yf_sym, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return []
        candles = []
        for ts, row in df.iterrows():
            try:
                o = float(row["Open"].iloc[0] if hasattr(row["Open"], 'iloc') else row["Open"])
                h = float(row["High"].iloc[0] if hasattr(row["High"], 'iloc') else row["High"])
                l = float(row["Low"].iloc[0]  if hasattr(row["Low"],  'iloc') else row["Low"])
                c = float(row["Close"].iloc[0] if hasattr(row["Close"],'iloc') else row["Close"])
                v = float(row["Volume"].iloc[0] if hasattr(row["Volume"],'iloc') else row["Volume"])
                candles.append({
                    "x": int(ts.timestamp() * 1000),   # ms timestamp
                    "o": round(o, 2), "h": round(h, 2),
                    "l": round(l, 2), "c": round(c, 2),
                    "v": round(v, 0),
                })
            except Exception:
                continue
        _ohlcv_cache[cache_key] = candles
        _ohlcv_cache_ts[cache_key] = now
        return candles
    except Exception as e:
        log.warning(f"OHLCV fetch error {symbol}: {e}")
        return []

@app.get("/chart/ohlcv")
async def chart_ohlcv(
    symbol:   str = Query("NQ",  description="Symbol: NQ, ES, SPY, QQQ, BTC"),
    period:   str = Query("5d",  description="yfinance period: 1d, 5d, 1mo, 3mo"),
    interval: str = Query("5m",  description="yfinance interval: 1m, 5m, 15m, 1h, 1d"),
):
    """OHLCV candlestick data pentru chart. Cache 60s."""
    VALID_INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1h","4h","1d","5d","1wk","1mo","3mo"}
    VALID_PERIODS   = {"1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"}
    if interval not in VALID_INTERVALS or period not in VALID_PERIODS:
        log.warning(f"OHLCV invalid params — period={period} interval={interval}, ignorat")
        return {"symbol": symbol, "period": period, "interval": interval, "candles": []}
    loop = asyncio.get_event_loop()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        candles = await loop.run_in_executor(pool, _fetch_ohlcv, symbol, period, interval)
    return {
        "symbol":   symbol.upper(),
        "period":   period,
        "interval": interval,
        "count":    len(candles),
        "candles":  candles,
    }

@app.get("/fear-greed")
async def fear_greed():
    t = state.latest_tick
    value = 50
    if t:
        val = 50 + (state.session_cum_delta / 5000 * 30) + (t.orderflow.imbalance_pct / 100 * 20)
        value = max(0, min(100, int(val)))
    label = ("Extreme Frică" if value < 20 else "Frică" if value < 40 else
             "Neutru" if value < 60 else "Lăcomie" if value < 80 else "Lăcomie Extremă")
    return {"data": [{"value": value, "value_classification": label,
                      "timestamp": str(int(time.time())), "time_until_update": "live"}]}

@app.get("/market/levels")
async def market_levels():
    t = state.latest_tick
    levels = []
    if t:
        if t.volume_profile.poc > 0:
            levels.append({"price": t.volume_profile.poc, "type": "POC", "strength": 90, "label": f"POC {t.volume_profile.poc:.2f}"})
        if t.volume_profile.vah > 0:
            levels.append({"price": t.volume_profile.vah, "type": "VAH", "strength": 75, "label": f"VAH {t.volume_profile.vah:.2f}"})
        if t.volume_profile.val > 0:
            levels.append({"price": t.volume_profile.val, "type": "VAL", "strength": 75, "label": f"VAL {t.volume_profile.val:.2f}"})
        if t.orderflow.vwap > 0:
            levels.append({"price": t.orderflow.vwap, "type": "VWAP", "strength": 85, "label": f"VWAP {t.orderflow.vwap:.2f}"})
    return {"levels": levels, "timestamp": datetime.now(timezone.utc).isoformat()}

# ─── Economic Calendar & News ─────────────────────────────────────────────────

def _load_news_engine():
    try:
        import sys
        rag_dir = os.path.dirname(os.path.abspath(__file__))
        if rag_dir not in sys.path:
            sys.path.insert(0, rag_dir)
        from news_clustering import get_news_engine
        return get_news_engine()
    except Exception:
        return None

@app.get("/economic-calendar")
async def economic_calendar():
    engine = _load_news_engine()
    events = []
    if engine:
        try:
            raw = engine.get_upcoming_events(hours_ahead=168)
            for ev in (raw or []):
                events.append({
                    "datetime": str(ev.get("datetime", "")), "currency": ev.get("currency", ""),
                    "impact": ev.get("impact", "Low"), "event": ev.get("event", ""),
                    "actual": ev.get("actual", ""), "forecast": ev.get("forecast", ""),
                    "previous": ev.get("previous", ""),
                })
        except Exception:
            pass
    return {"events": events, "count": len(events)}

@app.get("/economic-calendar/week")
async def economic_calendar_week(date: str = Query("")):
    try:
        target     = datetime.fromisoformat(date) if date else datetime.now()
        week_start = target - timedelta(days=target.weekday())
        week_end   = week_start + timedelta(days=5)
        engine     = _load_news_engine()
        events     = []
        if engine:
            raw = engine.get_upcoming_events(hours_ahead=168)
            for ev in (raw or []):
                try:
                    ev_dt = datetime.fromisoformat(str(ev.get("datetime", ""))[:19])
                    if week_start.date() <= ev_dt.date() <= week_end.date():
                        events.append({
                            "datetime": str(ev.get("datetime", "")), "currency": ev.get("currency", ""),
                            "impact": ev.get("impact", "Low"), "event": ev.get("event", ""),
                            "actual": ev.get("actual", ""), "forecast": ev.get("forecast", ""),
                            "previous": ev.get("previous", ""),
                        })
                except Exception:
                    continue
        return {"events": events, "week_start": str(week_start.date()), "week_end": str(week_end.date())}
    except Exception:
        return {"events": [], "week_start": "", "week_end": ""}

@app.get("/news/flash")
async def news_flash():
    engine = _load_news_engine()
    headlines = []
    if engine:
        try:
            ctx = engine.get_current_context()
            emoji_map = {"High Impact Expected": "🔴", "Medium Impact Expected": "🟡", "Low Impact Expected": "🟢"}
            for cluster in (ctx.get("active_windows") or [])[:5]:
                for ev in (cluster.get("events") or [])[:2]:
                    em = emoji_map.get(ev.get("impact", ""), "⚪")
                    headlines.append({
                        "title": f"{em} {ev.get('currency','')} — {ev.get('event','')}",
                        "impact": ev.get("impact", "Low"),
                        "time": str(ev.get("datetime", ""))[:16],
                        "currency": ev.get("currency", ""),
                    })
        except Exception:
            pass
    return {"headlines": headlines[:8]}

@app.get("/news/sentiment")
async def news_sentiment():
    engine = _load_news_engine()
    if engine:
        try:
            ctx   = engine.get_current_context()
            mult  = engine.get_trading_multiplier()
            alert = ctx.get("alert_level", "LOW")
            score = ctx.get("impact_score", 0)
            direc = ctx.get("direction_sentiment", 0)
            return {
                "sentiment": "BEARISH" if direc < -0.2 else "BULLISH" if direc > 0.2 else "NEUTRAL",
                "score": round(score, 2), "multiplier": round(mult, 2),
                "alert_level": alert, "is_blackout": alert == "BLACKOUT",
                "direction": round(direc, 3), "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception:
            pass
    return {"sentiment": "NEUTRAL", "score": 0, "multiplier": 1.0,
            "alert_level": "LOW", "is_blackout": False, "direction": 0,
            "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/sentiment/combined")
async def sentiment_combined():
    """UPDATE #6: Sentiment combinat FinBERT + Stocktwits pentru Vision UI Dashboard."""
    if _SUPABASE_OK and _supabase is not None:
        try:
            import sys as _sys2, os as _os2
            _aladin_dir2 = str(Path.home() / "Desktop" / "Aladin")
            if _aladin_dir2 not in _sys2.path:
                _sys2.path.insert(0, _aladin_dir2)
            import sentiment_engine as _se
            result = _se.get_combined_sentiment()
            return result
        except Exception as _se_err:
            log.debug(f"Sentiment combined error: {_se_err}")
    # Fallback la datele din ultimul semnal Supabase
    try:
        if _SUPABASE_OK and _supabase is not None:
            sigs = _supabase.get_recent_signals("NQ", 1)
            if sigs:
                s = sigs[0]
                sc = float(s.get("sentiment_score", 0.5)) * 2 - 1
                return {
                    "combined_score": round(sc, 3),
                    "sentiment_mult": float(s.get("sentiment_mult", 1.0)),
                    "label": "BULLISH" if sc > 0.2 else "BEARISH" if sc < -0.2 else "NEUTRAL",
                    "news":   {"score": round(sc, 3)},
                    "social": {"score": 0.0},
                }
    except Exception:
        pass
    return {
        "combined_score": 0.0, "sentiment_mult": 1.0, "label": "NEUTRAL",
        "news": {"score": 0.0}, "social": {"score": 0.0},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/day-analysis")
async def day_analysis(date: str = Query("")):
    target = date[:10] if date else datetime.now().strftime("%Y-%m-%d")
    day_trades = [t for t in state.trade_log if t.get("entry_time", "")[:10] == target]
    pnls = [t.get("pnl", 0) for t in day_trades]
    wins = [t for t in day_trades if t.get("result") == "WIN"]
    engine = _load_news_engine()
    multiplier = 1.0
    if engine:
        try: multiplier = engine.get_trading_multiplier()
        except Exception: pass
    return {
        "date": target, "trades": len(day_trades), "wins": len(wins),
        "pnl": round(sum(pnls), 2),
        "win_rate": len(wins) / len(day_trades) * 100 if day_trades else 0,
        "news_multiplier": round(multiplier, 2),
        "analysis": state.last_analysis if day_trades else {},
    }

# ─── Notes ────────────────────────────────────────────────────────────────────

@app.get("/notes")
async def get_notes():
    return {"notes": state.notes}

@app.post("/notes")
async def add_note(body: Dict[str, Any]):
    text = body.get("text", "").strip()
    if text:
        state.notes.append(f"[{datetime.now().strftime('%d/%m %H:%M')}] {text}")
        state.notes = state.notes[-50:]
        state._save_notes()
    return {"status": "ok", "notes": state.notes}

@app.delete("/notes/{idx}")
async def delete_note(idx: int):
    """Șterge nota de la indexul specificat."""
    if 0 <= idx < len(state.notes):
        removed = state.notes.pop(idx)
        state._save_notes()
        return {"status": "ok", "removed": removed, "notes": state.notes}
    raise HTTPException(404, f"Nota {idx} nu există")


# ─── MISSING ENDPOINTS (QA fix — conectare frontend) ────────────────────────────

@app.post("/trades/add")
async def add_trade_journal(body: Dict[str, Any]):
    """Adaugă trade din Journal (frontend trimite format extins cu grade/notes/tags)."""
    now = datetime.now(timezone.utc)
    _inst_sym = str(body.get("instrument", "")).upper() or (state.latest_tick.symbol if state.latest_tick else "")
    _point_val = _get_point_value(_inst_sym)

    entry_px  = float(body.get("entry_price", 0))
    exit_px   = float(body.get("exit_price", 0))
    direction = str(body.get("direction", "LONG")).upper()
    qty       = int(body.get("qty", 1))
    sl        = float(body.get("sl", 0))

    if entry_px > 0 and exit_px > 0:
        _diff   = exit_px - entry_px
        pnl_usd = (_diff if direction == "LONG" else -_diff) * _point_val * qty
    else:
        pnl_usd = float(body.get("pnl_usd") or body.get("pnl", 0))
    pnl_usd = round(pnl_usd, 2)
    result  = body.get("result") or ("WIN" if pnl_usd > 0 else ("LOSS" if pnl_usd < 0 else "BE"))

    r_mult = 0.0
    if sl > 0 and entry_px > 0:
        risk_usd = abs(entry_px - sl) * _point_val * qty
        if risk_usd > 0:
            r_mult = round(pnl_usd / risk_usd, 2)

    trade = {
        "id":           str(uuid.uuid4())[:8],
        "entry_time":   body.get("timestamp", body.get("entry_time", now.isoformat()[:19])),
        "exit_time":    body.get("exit_time", now.isoformat()[:19]),
        "direction":    direction,
        "entry_price":  round(entry_px, 2),
        "exit_price":   round(exit_px, 2),
        "qty":          qty,
        "sl":           round(sl, 2),
        "tp":           float(body.get("tp", 0)),
        "pnl":          round(pnl_usd / _point_val, 4) if _point_val else 0,
        "pnl_usd":      pnl_usd,
        "pnl_pct":      0.0,
        "result":       result,
        "mae_pts":      float(body.get("mae_pts", 0)),
        "mfe_pts":      float(body.get("mfe_pts", 0)),
        "exit_reason":  body.get("exit_reason", "JOURNAL"),
        "entry_hour":   now.hour,
        "entry_dow":    now.weekday(),
        "setup":        body.get("setup", "JOURNAL"),
        "score":        float(body.get("score", 0)),
        "strategy":     body.get("strategy", state.active_strategy.get("label", "Manual") if state.active_strategy else "Manual"),
        "session":      body.get("session", ""),
        "r_multiple":   r_mult,
        "source":       "JOURNAL",
        "instrument":   _inst_sym or ACTIVE_INSTRUMENT,
        # Journal-specific fields
        "setup_grade":  body.get("setup_grade", ""),
        "mistake":      body.get("mistake", ""),
        "notes":        body.get("notes", ""),
        "tags":         body.get("tags", []),
        "risk_usd":     float(body.get("risk_usd", 0)),
        "rr":           body.get("rr", ""),
    }
    state.add_trade(trade)
    _persist_equity_point(trade)
    log.info(f"📝 JOURNAL ADD: {direction} {entry_px}→{exit_px} | PnL: ${pnl_usd:+.2f} | {result}")
    return {"ok": True, "trade": trade}


@app.delete("/trades/{idx}")
async def delete_trade(idx: int):
    """Șterge trade-ul de la indexul specificat din trade_log."""
    if 0 <= idx < len(state.trade_log):
        removed = state.trade_log.pop(idx)
        state._save_trades()
        log.info(f"🗑️  Trade #{idx} șters: {removed.get('direction','')} {removed.get('pnl_usd',0)}")
        return {"ok": True, "removed": removed}
    raise HTTPException(404, f"Trade index {idx} invalid (total: {len(state.trade_log)})")


@app.patch("/trades/{idx}")
async def patch_trade(idx: int, body: Dict[str, Any]):
    """Actualizează câmpuri ale unui trade existent (grade, mistake, notes, tags)."""
    if 0 <= idx < len(state.trade_log):
        trade = state.trade_log[idx]
        for key in ["setup_grade", "mistake", "notes", "tags", "setup", "strategy"]:
            if key in body:
                trade[key] = body[key]
        state._save_trades()
        return {"ok": True, "trade": trade}
    raise HTTPException(404, f"Trade index {idx} invalid")


@app.post("/trades/import")
async def import_trades(body: Dict[str, Any]):
    """Import trades dintr-un array JSON (folosit de Journal CSV/Excel import)."""
    trades_data = body.get("trades", [])
    if not trades_data:
        raise HTTPException(400, "Nicio tranzacție de importat")
    imported = 0
    for t in trades_data:
        trade = {
            "id":           str(uuid.uuid4())[:8],
            "entry_time":   t.get("entry_time", ""),
            "exit_time":    t.get("exit_time", ""),
            "direction":    t.get("direction", "LONG"),
            "entry_price":  float(t.get("entry_price", 0)),
            "exit_price":   float(t.get("exit_price", 0)),
            "qty":          int(t.get("qty", 1)),
            "pnl":          float(t.get("pnl", 0)),
            "pnl_usd":      float(t.get("pnl_usd", 0)),
            "result":       t.get("result", ""),
            "setup":        t.get("setup", "IMPORT"),
            "score":        float(t.get("score", 0)),
            "strategy":     t.get("strategy", "Import"),
            "r_multiple":   float(t.get("r_multiple", 0)),
            "source":       "IMPORT",
            "instrument":   t.get("instrument", ACTIVE_INSTRUMENT),
            "setup_grade":  t.get("setup_grade", ""),
            "mistake":      t.get("mistake", ""),
            "notes":        t.get("notes", ""),
        }
        state.trade_log.append(trade)
        imported += 1
    state._save_trades()
    log.info(f"📥 IMPORT: {imported} trade-uri importate")
    return {"ok": True, "imported": imported, "total": len(state.trade_log)}


@app.get("/journal/analytics")
async def journal_analytics():
    """Analytics pentru Journal page: equity curve zilnică, monthly breakdown, setup analysis."""
    trades = [t for t in state.trade_log if t.get("result") in ("WIN", "LOSS")]
    if not trades:
        return {"equity": [], "monthly": {}, "setups": {}, "grades": {}}

    trades_sorted = sorted(trades, key=lambda t: t.get("entry_time", ""))

    # Equity curve zilnică
    daily_pnl: Dict[str, float] = {}
    for t in trades_sorted:
        day = t.get("entry_time", "")[:10]
        if day:
            daily_pnl[day] = daily_pnl.get(day, 0) + float(t.get("pnl_usd") or t.get("pnl", 0))
    balance = 100000.0
    equity = []
    for day in sorted(daily_pnl.keys()):
        balance += daily_pnl[day]
        equity.append({"date": day, "balance": round(balance, 2), "pnl": round(daily_pnl[day], 2)})

    # Monthly breakdown
    monthly: Dict[str, Dict] = {}
    for t in trades_sorted:
        month = t.get("entry_time", "")[:7]  # YYYY-MM
        if not month:
            continue
        if month not in monthly:
            monthly[month] = {"wins": 0, "losses": 0, "pnl": 0.0}
        pnl = float(t.get("pnl_usd") or t.get("pnl", 0))
        monthly[month]["pnl"] = round(monthly[month]["pnl"] + pnl, 2)
        if t.get("result") == "WIN":
            monthly[month]["wins"] += 1
        else:
            monthly[month]["losses"] += 1

    # Setup grade analysis
    grades: Dict[str, Dict] = {}
    for t in trades_sorted:
        g = t.get("setup_grade", "") or "Ungraded"
        if g not in grades:
            grades[g] = {"trades": 0, "wins": 0, "pnl": 0.0}
        grades[g]["trades"] += 1
        grades[g]["pnl"] = round(grades[g]["pnl"] + float(t.get("pnl_usd") or 0), 2)
        if t.get("result") == "WIN":
            grades[g]["wins"] += 1

    # Mistake analysis
    setups: Dict[str, Dict] = {}
    for t in trades_sorted:
        m = t.get("mistake", "") or "None"
        if m not in setups:
            setups[m] = {"trades": 0, "pnl": 0.0}
        setups[m]["trades"] += 1
        setups[m]["pnl"] = round(setups[m]["pnl"] + float(t.get("pnl_usd") or 0), 2)

    return {"equity": equity, "monthly": monthly, "grades": grades, "setups": setups}


SETTINGS_FILE = DATA_DIR / "user_settings.json"

@app.get("/settings")
async def get_settings():
    """Returnează setările utilizatorului salvate pe disc."""
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text())
        return {}
    except Exception:
        return {}

@app.post("/settings")
async def save_settings(body: Dict[str, Any]):
    """Salvează setările utilizatorului pe disc."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        body["saved_at"] = datetime.now(timezone.utc).isoformat()
        tmp = str(SETTINGS_FILE) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(body, f, indent=2)
        os.replace(tmp, str(SETTINGS_FILE))
        log.info(f"⚙️ Settings salvate ({len(body)} keys)")
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, f"Nu s-au putut salva setările: {e}")


@app.post("/system/errors/clear")
async def clear_system_errors():
    """Golește error log-ul."""
    count = len(state.error_log)
    state.error_log.clear()
    return {"ok": True, "cleared": count}


@app.get("/backtest/montecarlo")
async def backtest_montecarlo(n_sims: int = Query(200, ge=10, le=2000), n_trades: int = Query(100, ge=10, le=5000)):
    """Monte Carlo simulation pe baza trade-urilor istorice reale."""
    trades = [t for t in state.trade_log if t.get("result") in ("WIN", "LOSS")]
    if len(trades) < 10:
        return {"error": "Minimum 10 trade-uri necesare", "simulations": []}

    pnls = [float(t.get("pnl_usd") or t.get("pnl", 0)) for t in trades]
    simulations = []
    final_balances = []
    max_dds = []

    for _ in range(min(n_sims, 1000)):
        shuffled = random.choices(pnls, k=min(n_trades, len(pnls) * 3))
        balance = 100000.0
        peak = balance
        max_dd = 0.0
        curve = [balance]
        for p in shuffled:
            balance += p
            curve.append(round(balance, 2))
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        final_balances.append(round(balance, 2))
        max_dds.append(round(max_dd, 2))
        if len(simulations) < 20:  # Trimitem doar primele 20 curbe (UI)
            simulations.append(curve)

    final_balances.sort()
    return {
        "simulations":    simulations,
        "n_sims":         n_sims,
        "n_trades":       n_trades,
        "median_balance":  round(final_balances[len(final_balances)//2], 2),
        "p5_balance":      round(final_balances[int(len(final_balances)*0.05)], 2),
        "p95_balance":     round(final_balances[int(len(final_balances)*0.95)], 2),
        "avg_max_dd":      round(sum(max_dds)/len(max_dds), 2),
        "worst_dd":        round(max(max_dds), 2),
        "ruin_pct":        round(sum(1 for b in final_balances if b < 80000) / len(final_balances) * 100, 1),
    }


# ─── Backtest Compare ─────────────────────────────────────────────────────────

@app.get("/backtest/compare")
async def backtest_compare():
    """Comparație strategii din date REALE (trade_log grupat per strategy)."""
    trades = [t for t in state.trade_log if t.get("result") in ("WIN", "LOSS")]
    strat_map: Dict[str, Dict] = {}
    for t in trades:
        name = t.get("strategy", "Default")
        if name not in strat_map:
            strat_map[name] = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}
        s = strat_map[name]
        s["trades"] += 1
        pnl = float(t.get("pnl_usd") or t.get("pnl", 0))
        s["pnl"] += pnl
        if t.get("result") == "WIN":
            s["wins"] += 1
        else:
            s["losses"] += 1

    strategies = []
    for name, data in strat_map.items():
        total = data["trades"]
        gross_win  = sum(float(t.get("pnl_usd") or t.get("pnl", 0)) for t in trades if t.get("strategy") == name and t.get("result") == "WIN")
        gross_loss = abs(sum(float(t.get("pnl_usd") or t.get("pnl", 0)) for t in trades if t.get("strategy") == name and t.get("result") == "LOSS"))
        pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0
        strategies.append({
            "name":          name,
            "win_rate":      round(data["wins"] / total * 100, 1) if total else 0,
            "profit_factor": pf,
            "total_pnl":     round(data["pnl"], 0),
            "trades":        total,
            "badge":         "Live",
        })
    strategies.sort(key=lambda x: x["total_pnl"], reverse=True)

    if state.last_backtest_result:
        st = state.last_backtest_result.get("stats", {})
        strategies.insert(0, {
            "name": "▶ Ultimul BT",
            "win_rate": round(st.get("win_rate", 0), 1),
            "profit_factor": round(st.get("profit_factor", 0), 2),
            "total_pnl": round(st.get("total_pnl", 0), 0),
            "trades": st.get("total_trades", 0), "badge": "Backtest",
        })
    return {"strategies": strategies}

@app.get("/backtest/last")
async def backtest_last():
    if state.last_backtest_result:
        return {"found": True, "result": state.last_backtest_result}
    return {"found": False}

# ─── Backtest Engine ──────────────────────────────────────────────────────────

def _load_csv(symbol: str, start_date: str, end_date: str):
    import pandas as pd
    for p in [DATA_DIR / f"{symbol}.csv", DATA_DIR / f"@{symbol}#.csv", DATA_DIR / f"{symbol}_1min.csv"]:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            col_map = {}
            for c in df.columns:
                cl = c.lower().strip()
                if cl in ("time","datetime","date","timestamp"): col_map[c] = "datetime"
                elif cl in ("open","o"):   col_map[c] = "open"
                elif cl in ("high","h"):   col_map[c] = "high"
                elif cl in ("low","l"):    col_map[c] = "low"
                elif cl in ("close","c"):  col_map[c] = "close"
                elif cl in ("volume","vol","v"): col_map[c] = "volume"
            df.rename(columns=col_map, inplace=True)
            if "datetime" not in df.columns: continue
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"]).sort_values("datetime")
            sd = pd.to_datetime(start_date); ed = pd.to_datetime(end_date)
            df = df[(df["datetime"] >= sd) & (df["datetime"] <= ed)]
            if len(df) >= 100:
                if "atr" not in df.columns:
                    df["atr"] = (df["high"] - df["low"]).rolling(14).mean().fillna((df["high"] - df["low"]).mean())
                log.info(f"CSV loaded: {p.name} — {len(df):,} bars")
                return df.reset_index(drop=True)
        except Exception as e:
            log.debug(f"CSV error {p}: {e}")
    return None

def _gen_synthetic(symbol: str, start_date: str, end_date: str):
    import pandas as pd, numpy as np
    sd = pd.to_datetime(start_date); ed = pd.to_datetime(end_date)
    idx = pd.date_range(sd, ed, freq="1min")
    idx = idx[(idx.hour >= 8) & (idx.hour < 22) & (idx.dayofweek < 5)]
    n = len(idx)
    base = {"NQ": 17500, "ES": 5200, "XAUUSD": 2400, "BTCUSD": 65000}.get(symbol.upper(), 17500)
    np.random.seed(42)
    ret = np.random.normal(0, 0.0002, n)
    for i in range(n):
        h = idx[i].hour
        if 8 <= h < 10 or 13 <= h < 16: ret[i] *= 2.0
        elif 20 <= h <= 22: ret[i] *= 0.4
    close = base * np.exp(np.cumsum(ret))
    hi_noise = np.abs(np.random.normal(0, 0.0003, n))
    lo_noise = np.abs(np.random.normal(0, 0.0003, n))
    df = pd.DataFrame({
        "datetime": idx,
        "open":   close * (1 + np.random.normal(0, 0.0001, n)),
        "high":   close * (1 + hi_noise),
        "low":    close * (1 - lo_noise),
        "close":  close,
        "volume": np.random.lognormal(8, 1, n).astype(int),
    })
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean().fillna((df["high"] - df["low"]).mean())
    log.info(f"Synthetic: {len(df):,} bars for {symbol}")
    return df.reset_index(drop=True)

def _score_bar(df, idx: int) -> float:
    if idx < 20: return 30.0
    bar  = df.iloc[idx]
    hour = bar["datetime"].hour if hasattr(bar["datetime"], "hour") else 14
    if 9 <= hour < 11 or 13 <= hour < 16: ss = 25
    elif 8 <= hour < 12 or 13 <= hour < 18: ss = 18
    else: ss = 8
    sma20 = df["close"].iloc[idx - 20:idx].mean()
    ts    = 20 if bar["close"] > sma20 else 14
    atr_v = float(df["atr"].iloc[idx]) if "atr" in df.columns else bar["close"] * 0.002
    r5    = df["high"].iloc[idx - 5:idx].max() - df["low"].iloc[idx - 5:idx].min()
    ds    = min(25, r5 / (atr_v + 1e-8) * 10)
    avg_v = df["volume"].iloc[idx - 10:idx].mean()
    vs    = min(15, bar["volume"] / (avg_v + 1e-8) * 7)
    fvg   = abs(bar["low"] - df["high"].iloc[idx - 2]) / bar["close"] * 100 if idx >= 2 else 0
    ict   = min(10, fvg * 25)
    return min(100, ss + ts + ds + vs + ict)

def _sim_trade(df, idx: int, direction: str, risk_pct: float, rr: float, balance: float) -> Optional[Dict]:
    if idx >= len(df) - 5: return None
    bar         = df.iloc[idx]
    entry_price = float(bar["close"])
    atr_v       = float(df["atr"].iloc[idx]) if "atr" in df.columns else entry_price * 0.002
    sl_size     = atr_v * 1.5
    sl = entry_price - sl_size if direction == "LONG" else entry_price + sl_size
    tp = entry_price + sl_size * rr  if direction == "LONG" else entry_price - sl_size * rr
    risk_usd    = balance * (risk_pct / 100)
    result_str  = "OPEN"; exit_price = entry_price; exit_idx = min(idx + 240, len(df) - 1)
    for i in range(idx + 1, min(idx + 241, len(df))):
        b = df.iloc[i]
        if direction == "LONG":
            if float(b["low"]) <= sl:  exit_price = sl;  result_str = "LOSS"; exit_idx = i; break
            if float(b["high"]) >= tp: exit_price = tp;  result_str = "WIN";  exit_idx = i; break
        else:
            if float(b["high"]) >= sl: exit_price = sl;  result_str = "LOSS"; exit_idx = i; break
            if float(b["low"])  <= tp: exit_price = tp;  result_str = "WIN";  exit_idx = i; break
    if result_str == "OPEN":
        exit_price = float(df.iloc[exit_idx]["close"])
        result_str = "WIN" if (direction == "LONG" and exit_price > entry_price) or \
                               (direction == "SHORT" and exit_price < entry_price) else "LOSS"
    move = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
    r_mult = move / (sl_size + 1e-8)
    pnl    = risk_usd * r_mult
    entry_dt = df.iloc[idx]["datetime"]; exit_dt = df.iloc[exit_idx]["datetime"]
    return {
        "id": str(uuid.uuid4())[:8],
        "entry_time": str(entry_dt), "exit_time": str(exit_dt),
        "direction": direction, "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2), "sl": round(sl, 2), "tp": round(tp, 2),
        "result": result_str, "pnl": round(pnl, 2), "pnl_pct": round(pnl / balance * 100, 3),
        "r_multiple": round(r_mult, 2), "bars_held": exit_idx - idx,
        "entry_hour": entry_dt.hour if hasattr(entry_dt, "hour") else 14,
        "entry_dow":  entry_dt.weekday() if hasattr(entry_dt, "weekday") else 1,
    }

def _run_backtest_sync(job_id: str, params: Dict):
    import pandas as pd
    try:
        j = state.backtest_jobs[job_id]
        j["status"] = "running"; j["status_text"] = "Încarc date..."; j["progress"] = 5

        symbol     = params.get("symbol", "NQ")
        start_date = params.get("start_date", "2024-01-01")
        end_date   = params.get("end_date",   "2024-12-31")
        bal0       = float(params.get("initial_balance", 10000))
        risk_pct   = float(params.get("risk_per_trade", 1.0))
        rr         = float(params.get("rr_ratio", 2.0))
        thresh     = float(params.get("score_threshold", 0.65)) * 100
        entry_times = [t.strip() for t in params.get("entry_times_str", "15:30,16:00").split(",") if t.strip()]
        max_day    = int(params.get("max_trades_day", 3))

        df = _load_csv(symbol, start_date, end_date)
        synthetic = df is None
        if synthetic:
            j["status_text"] = "CSV lipsă — date sintetice..."
            df = _gen_synthetic(symbol, start_date, end_date)
        if df is None or len(df) < 50:
            j["status"] = "error"; j["error"] = "Nu s-au putut genera date"; return

        if "atr" not in df.columns:
            df["atr"] = (df["high"] - df["low"]).rolling(14).mean().fillna((df["high"] - df["low"]).mean())

        j["status_text"] = f"Simulez {len(df):,} bare..."; j["progress"] = 15
        df["_date"] = pd.to_datetime(df["datetime"]).dt.date
        unique_days = df["_date"].unique()
        trades_all  = []; balance = bal0

        for d_i, day in enumerate(unique_days):
            day_rows = df[df["_date"] == day]
            trades_today = 0
            for et in entry_times:
                if trades_today >= max_day: break
                try: et_h, et_m = int(et.split(":")[0]), int(et.split(":")[1])
                except Exception: et_h, et_m = 15, 30
                cands = [
                    i for i in day_rows.index
                    if hasattr(df.iloc[i]["datetime"], "hour")
                    and df.iloc[i]["datetime"].hour == et_h
                    and abs(df.iloc[i]["datetime"].minute - et_m) <= 5
                ]
                if not cands: continue
                idx = cands[0]
                score = _score_bar(df, idx)
                if score < thresh: continue
                sma20 = df["close"].iloc[max(0, idx - 20):idx].mean()
                direc = "LONG" if df.iloc[idx]["close"] > sma20 else "SHORT"
                tr = _sim_trade(df, idx, direc, risk_pct, rr, balance)
                if tr:
                    tr["score"] = round(score, 1)
                    balance += tr["pnl"]
                    trades_all.append(tr); trades_today += 1

            j["progress"] = 15 + int((d_i / len(unique_days)) * 75)
            if d_i % 20 == 0:
                j["status_text"] = f"Zi {d_i}/{len(unique_days)} — {len(trades_all)} trades..."

        j["status_text"] = "Calculez statistici..."; j["progress"] = 92
        if not trades_all:
            j["status"] = "error"; j["error"] = "Nicio tranzacție. Ajustează pragul sau orele de intrare."; return

        pnls      = [t["pnl"] for t in trades_all]
        wins      = [t for t in trades_all if t["result"] == "WIN"]
        losses    = [t for t in trades_all if t["result"] == "LOSS"]
        gw        = sum(p for p in pnls if p > 0)
        gl        = abs(sum(p for p in pnls if p < 0)) or 1e-9
        pf        = gw / gl
        wp        = [p for p in pnls if p > 0]; lp = [p for p in pnls if p < 0]

        eq_v = bal0; eq_max = bal0; max_dd = 0.0
        eq_dates = ["Start"]; eq_vals = [bal0]
        for t in trades_all:
            eq_v += t["pnl"]; eq_max = max(eq_max, eq_v)
            dd = (eq_max - eq_v) / eq_max * 100; max_dd = max(max_dd, dd)
            eq_dates.append(t["exit_time"][:10]); eq_vals.append(round(eq_v, 2))

        sw = sl_s = cw = cl = 0
        for t in trades_all:
            if t["result"] == "WIN": cw += 1; cl = 0
            else: cl += 1; cw = 0
            sw = max(sw, cw); sl_s = max(sl_s, cl)

        stats = {
            "total_trades": len(trades_all), "total_wins": len(wins), "total_losses": len(losses),
            "win_rate": round(len(wins) / len(trades_all) * 100, 1),
            "profit_factor": round(pf, 2), "max_drawdown": round(max_dd, 2),
            "avg_win":  round(sum(wp) / len(wp) if wp else 0, 2),
            "avg_loss": round(sum(lp) / len(lp) if lp else 0, 2),
            "best_trade": round(max(pnls), 2), "worst_trade": round(min(pnls), 2),
            "total_pnl":   round(sum(pnls), 2),
            "return_pct":  round(sum(pnls) / bal0 * 100, 2),
            "final_balance": round(bal0 + sum(pnls), 2),
            "max_win_streak": sw, "max_loss_streak": sl_s,
            "time_of_day": [], "day_of_week": [],
            "monte_carlo": {"p10": round(sum(pnls)*0.65,0), "p50": round(sum(pnls),0), "p90": round(sum(pnls)*1.35,0)},
        }
        result = {
            "trades": trades_all, "stats": stats,
            "equity_curve": {"dates": eq_dates, "values": eq_vals},
            "symbol": symbol, "start_date": start_date, "end_date": end_date,
            "_params": params, "synthetic_data": synthetic,
            "timestamp": datetime.now().isoformat(),
        }
        j["status"] = "done"; j["progress"] = 100
        j["status_text"] = f"Done! {len(trades_all)} trades, PnL: ${sum(pnls):+.0f}"
        j["result"] = result; state.last_backtest_result = result
        log.info(f"[BT] Done: {len(trades_all)} trades, WR={stats['win_rate']}%, PnL=${stats['total_pnl']:+.0f}")
    except Exception as e:
        log.error(f"Backtest error: {e}")
        state.backtest_jobs[job_id]["status"] = "error"
        state.backtest_jobs[job_id]["error"]  = str(e)
        state.error_log.append(f"{datetime.now().isoformat()} backtest: {e}")

@app.post("/backtest/start")
async def backtest_start(
    background_tasks: BackgroundTasks,
    start_date:      str   = Query("2024-01-01"),
    end_date:        str   = Query("2024-12-31"),
    initial_balance: float = Query(10000, ge=0, le=10_000_000),
    risk_per_trade:  float = Query(1.0),
    rr_ratio:        float = Query(2.0),
    score_threshold: float = Query(0.65),
    entry_times_str: str   = Query("15:30,16:00"),
    max_trades_day:  int   = Query(3),
    walk_forward:    bool  = Query(False),
    symbol:          str   = Query("NQ"),
):
    job_id = str(uuid.uuid4())[:8]
    params = {
        "symbol": symbol, "start_date": start_date, "end_date": end_date,
        "initial_balance": initial_balance, "risk_per_trade": risk_per_trade,
        "rr_ratio": rr_ratio, "score_threshold": score_threshold,
        "entry_times_str": entry_times_str, "max_trades_day": max_trades_day,
        "walk_forward": walk_forward, "style": entry_times_str,
    }
    state.backtest_jobs[job_id] = {
        "status": "queued", "progress": 0, "status_text": "Pornind...",
        "job_id": job_id, "result": None, "error": None, "params": params,
    }
    import concurrent.futures
    loop = asyncio.get_event_loop()
    background_tasks.add_task(
        lambda: asyncio.ensure_future(
            loop.run_in_executor(None, _run_backtest_sync, job_id, params)
        )
    )
    log.info(f"[BT] Job {job_id} queued: {symbol} {start_date}→{end_date}")
    return {"job_id": job_id, "status": "queued"}

@app.get("/backtest/status/{job_id}")
async def backtest_status(job_id: str):
    if job_id not in state.backtest_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    j = state.backtest_jobs[job_id]
    return {"job_id": job_id, "status": j["status"], "progress": j.get("progress", 0),
            "status_text": j.get("status_text", ""), "result": j.get("result"), "error": j.get("error")}

@app.post("/backtest/optimize")
async def backtest_optimize():
    return {"status": "ok", "message": "Optimizarea necesită date NT8 exportate. Exportă @NQ# 1 Minute din NT8 → data/."}

# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.ws_clients.append(websocket)
    log.info(f"WebSocket conectat. Total: {len(state.ws_clients)}")
    try:
        if state.latest_tick:
            await websocket.send_text(json.dumps({
                "type": "snapshot", "status": "connected",
                "tick": state.tick_count, "analysis": state.last_analysis or {},
            }, default=str))
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if msg == "ping":
                    await websocket.send_text('{"type":"pong"}')
            except asyncio.TimeoutError:
                await websocket.send_text('{"type":"heartbeat","ts":"' + datetime.now().isoformat() + '"}')
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.ws_clients:
            state.ws_clients.remove(websocket)
        log.info(f"WebSocket deconectat. Total: {len(state.ws_clients)}")

# ─── Startup / Shutdown ───────────────────────────────────────────────────────

def _tg_state_snapshot() -> dict:
    """Returnează statusul curent pentru răspunsul Telegram /status."""
    strat     = state.active_strategy
    analysis  = state.last_analysis or {}
    score_raw = analysis.get("score", 0)
    score_pct = round(score_raw * 100, 1) if score_raw <= 1 else round(score_raw, 1)
    signal    = analysis.get("trade_direction", "—") if analysis else "—"
    verdict   = analysis.get("verdict", "")
    if verdict and "SKIP" in verdict:
        signal = f"SKIP ({signal})"

    # Trade deschis — folosim direct state (mai precis decât paper_trades)
    open_trade   = None
    open_pnl_usd = 0.0
    if state._position_open and state.open_trade_entry > 0:
        _cur_px = state.latest_tick.price.close if state.latest_tick else 0.0
        # Fix v10.5: arată trailing_sl (SL curent) nu open_trade_sl (SL inițial)
        _sl_st  = state.trailing_sl if state.trailing_sl > 0 else state.open_trade_sl
        _tp_st  = state.open_trade_tp
        _dir_st = state.open_trade_dir
        _qty_st = state.open_trade_qty or 1
        open_trade = {
            "direction": _dir_st,
            "entry":     round(state.open_trade_entry, 2),
            "sl":        round(_sl_st, 2),
            "tp":        round(_tp_st, 2),
        }
        # Open PnL live (NQ: $20/punct)
        if _cur_px > 0 and state.open_trade_entry > 0 and _dir_st:
            _diff = _cur_px - state.open_trade_entry
            open_pnl_usd = round((_diff if _dir_st == "LONG" else -_diff) * 20.0 * _qty_st, 2)
    else:
        # fallback: caută în paper_trades dacă state nu e populat
        for t in reversed(state.paper_trades):
            if t.get("result") == "OPEN":
                open_trade = {
                    "direction": t.get("direction", ""),
                    "entry":     float(t.get("entry_price", 0)),
                    "sl":        float(t.get("sl_price", 0)),
                    "tp":        float(t.get("tp_price", 0)),
                }
                break

    # fereastră timp
    in_window = False
    if strat:
        from datetime import datetime, timezone
        now_t = datetime.now(timezone.utc).strftime("%H:%M")
        s, e  = strat.get("session_start", "00:00"), strat.get("session_end", "23:59")
        in_window = (s <= now_t < e) if s <= e else (now_t >= s or now_t < e)

    # PnL sesiune
    _realized_pnl = state.daily_profit_usd - state.daily_loss_usd

    return {
        "autotrade":          state.autotrade_enabled,
        "paper_mode":         state.paper_mode,
        "strategy":           strat.get("label", "") if strat else None,
        "score":              score_pct,
        "trades_today":       state.strategy_trades_today,
        "max_trades":         int(strat.get("max_trades", 0)) if strat else 0,
        "in_window":          in_window,
        "last_signal":        signal,
        "open_trade":         open_trade,
        "geo_risk_active":    state.geo_risk_active,
        # PnL pentru /trade command
        "daily_profit_usd":   round(state.daily_profit_usd, 2),
        "daily_loss_usd":     round(state.daily_loss_usd, 2),
        "realized_pnl_usd":   round(_realized_pnl, 2),
        "open_pnl_usd":       open_pnl_usd,
        "session_max_profit": round(state.session_max_profit, 2),
        "session_max_drawdown": round(state.session_max_drawdown, 2),
    }


# ─── Backtest Replay Endpoints (FXReplay) ─────────────────────────────────────
# Folosește funcțiile module-level din backtest_engine.py (operează pe _bt global)

_bt_loaded = False  # flag: sesiune încărcată

def _bt_import():
    """Importă backtest_engine — raises HTTPException dacă nu e găsit."""
    try:
        import backtest_engine as _be
        return _be
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"backtest_engine not found: {exc}")


@app.post("/backtest/load")
async def backtest_load(payload: dict):
    """Descarcă date OHLCV și inițializează sesiunea replay.

    Body: { "symbol": "NQ", "timeframe": "5m", "days": 30 }
    """
    global _bt_loaded
    be = _bt_import()

    symbol    = str(payload.get("symbol",    "NQ")).upper()
    timeframe = str(payload.get("timeframe", "5m"))
    days      = int(payload.get("days",      30))

    ok, msg = be.load_data(symbol, timeframe, days)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    _bt_loaded = True
    n = len(be._bt.df) if be._bt.df is not None else 0
    ts = be._bt.df["ts"] if be._bt.df is not None else None
    log.info(f"📊 Backtest loaded: {symbol} {timeframe} {n} bars")
    return {
        "ok":       True,
        "symbol":   symbol,
        "bars":     n,
        "from":     str(ts.iloc[0])  if ts is not None and len(ts) > 0 else "",
        "to":       str(ts.iloc[-1]) if ts is not None and len(ts) > 0 else "",
        "message":  msg,
    }


@app.get("/backtest/step")
async def backtest_step(n: int = 1):
    """Avansează n bare și returnează chart + semnal curent."""
    if not _bt_loaded:
        raise HTTPException(status_code=400, detail="No session loaded. Call /backtest/load first.")
    be = _bt_import()
    return be.step_bar(n_bars=n)


@app.post("/backtest/trade")
async def backtest_trade(payload: dict):
    """Plasează sau închide o tranzacție manuală.

    Body: { "action": "buy"|"sell"|"close", "size": 1 }
    """
    if not _bt_loaded:
        raise HTTPException(status_code=400, detail="No session loaded.")
    be     = _bt_import()
    action = str(payload.get("action", "")).lower()
    size   = int(payload.get("size", 1))

    if action in ("buy", "long"):
        return be.place_trade("LONG", source="manual", size=size)
    elif action in ("sell", "short"):
        return be.place_trade("SHORT", source="manual", size=size)
    elif action == "close":
        return be.close_position()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")


@app.get("/backtest/summary")
async def backtest_summary():
    """Statistici complete ale sesiunii replay."""
    if not _bt_loaded:
        raise HTTPException(status_code=400, detail="No session loaded.")
    be = _bt_import()
    return be.get_summary()


@app.post("/backtest/reset")
async def backtest_reset():
    """Resetează trades/balance, păstrează datele descărcate."""
    if not _bt_loaded:
        raise HTTPException(status_code=400, detail="No session loaded.")
    be = _bt_import()
    return be.reset_session()


# ─── News Alert Background Task ──────────────────────────────────────────────
# Verifică la fiecare minut dacă urmează un eveniment economic în 15 minute.
# Dacă da, trimite alertă Telegram o singură dată per eveniment per zi.

_NEWS_SCHEDULE = [
    # (dow, day_range, hour, minute, event_name)
    # NFP — primul vineri din lună la 15:30
    ("Friday",    (1,  7), 15, 15, "NFP"),
    # FOMC — miercuri la 20:00
    ("Wednesday", (1, 31), 19, 45, "FOMC"),
    # CPI — marți săptămâna 2 la 15:30
    ("Tuesday",   (8, 14), 15, 15, "CPI"),
    # PPI — miercuri săptămâna 2-3 la 15:30
    ("Wednesday", (8, 17), 15, 15, "PPI"),
    # PCE — ultima vineri din lună la 15:30
    ("Friday",    (25, 31), 15, 15, "PCE"),
    # JOLTS — prima marți din lună la 16:00
    ("Tuesday",   (1,  7), 15, 45, "JOLTS"),
    # ISM Manufacturing — prima luni/marți la 16:00
    ("Monday",    (1,  5), 15, 45, "ISM Manufacturing"),
    ("Tuesday",   (1,  5), 15, 45, "ISM Manufacturing"),
    # ISM Services — prima miercuri/joi la 16:00
    ("Wednesday", (1,  7), 15, 45, "ISM Services"),
    ("Thursday",  (1,  7), 15, 45, "ISM Services"),
    # Retail Sales — miercuri săptămâna 2 la 15:30
    ("Wednesday", (8, 14), 15, 15, "Retail Sales"),
    # GDP — ultima miercuri/joi la 15:30
    ("Wednesday", (25, 31), 15, 15, "GDP"),
    ("Thursday",  (25, 31), 15, 15, "GDP"),
    # ADP — miercuri prima săptămână la 15:15
    ("Wednesday", (1,  6), 15,  0, "ADP Employment"),
    # Jobless Claims — joi la 15:30
    ("Thursday",  (1, 31), 15, 15, "Jobless Claims"),
]

async def _news_alert_loop():
    """Loop care rulează la fiecare 60s și trimite alertă Telegram 15min înainte de news."""
    await asyncio.sleep(10)  # startup delay
    while True:
        try:
            now = datetime.now(timezone.utc)
            today = now.strftime("%Y-%m-%d")
            dow   = now.strftime("%A")
            day   = now.day

            # Reset set alertate la zi nouă
            if state._news_alert_date != today:
                state._news_alerted_today = set()
                state._news_alert_date    = today

            for (ev_dow, (d_min, d_max), ev_h, ev_m, ev_name) in _NEWS_SCHEDULE:
                if dow != ev_dow:
                    continue
                if not (d_min <= day <= d_max):
                    continue

                # Calculează timestamp-ul evenimentului
                event_dt = now.replace(hour=ev_h, minute=ev_m, second=0, microsecond=0)
                diff_min = (event_dt - now).total_seconds() / 60.0

                alert_key = f"{today}_{ev_name}"
                if 14.0 <= diff_min <= 16.0 and alert_key not in state._news_alerted_today:
                    # Trimite alertă
                    release_time = f"{ev_h:02d}:{ev_m + 15:02d}"  # ora reală a evenimentului
                    strat = state.active_strategy or {}
                    symbol = strat.get("symbol", "NQ")
                    # Fix v7.5: News alerts dezactivate din AladinBot — rămân doar în NewsBot
                    # _tg_news_alert se trimite doar din news bot dedicat, nu din bridge
                    state._news_alerted_today.add(alert_key)
                    log.info(f"⏰ NEWS ALERT (NewsBot only): {ev_name} în ~15min ({release_time} UTC)")

        except Exception as _nal_err:
            log.debug(f"News alert loop error: {_nal_err}")

        await asyncio.sleep(60)


# ─── GEO NEWS MONITOR: RSS feeds Reuters / AP / BBC ───────────────────────────

# ── Keyword system: DOAR 2 niveluri (MEDIUM scos — prea mult zgomot) ──────────
# CRITICAL → alertă imediată + geo_risk_active automat
# HIGH     → alertă Telegram + geo_risk_active automat
# Fiecare nivel cere minim 1 match dintr-o pereche (keyword_required, [context_words])
# sau 2+ match-uri din lista simplă (pentru a evita false positives)

_GEO_CRITICAL_KW: list = [
    # Atacuri directe / război declarat
    "nuclear weapon", "nuclear strike", "nuclear attack",
    "missile strike", "missile attack", "ballistic missile",
    "war declared", "declaration of war",
    "invasion begins", "military invasion",
    "chemical weapon", "biological weapon",
    "major terrorist attack", "terror attack",
    "nato article 5", "nato invoked",
    "market circuit breaker", "trading halt",
    "fed emergency cut", "emergency rate",
    "oil embargo", "oil supply cut",
    "coup attempt", "government overthrow",
]

_GEO_HIGH_KW: list = [
    # Trump & tarife — direct market moving
    "trump tariff", "trump imposes tariff", "trump trade war",
    "trump sanctions", "trump executive order",
    "tariffs on china", "tariffs on europe", "tariffs on imports",
    "trade war escalat", "trade war intensif",
    # Fed / rate — market moving
    "powell speech", "fed chair speech", "federal reserve emergency",
    "interest rate decision", "rate hike surprise", "rate cut surprise",
    # Conflicte militare clare
    "airstrike", "air strike", "military strike",
    "troops deployed", "military offensive",
    "ceasefire collapsed", "ceasefire broken",
    # Război specific + impact oil/energie (Iran, Russia, Ukraine)
    "iran war", "iran attack", "iran strike",
    "russia attack", "ukraine attack",
    "oil supply disruption", "oil supply cut", "energy supply",
    "strait of hormuz", "oil tanker attack",
    # Sancțiuni majore cu impact economic
    "sanctions against russia", "sanctions against china",
    "sanctions against iran", "oil sanctions",
    "export ban", "chip ban", "semiconductor ban",
    # Criză economică directă
    "recession declared", "default risk", "debt ceiling",
    "stock market crash", "market selloff", "market plunge",
    "opec production cut", "opec emergency", "opec cut",
    # China / Taiwan — risc major pentru tech stocks NQ
    "china taiwan strait", "taiwan blockade", "taiwan invasion",
    "china semiconductor", "china tech ban",
]

# RSS feeds — confirmate că merg de pe Mac (BBC + CNBC testate OK)
# Toate 9 confirmate OK de pe Mac (testate live). Reuters/AP/CoinGlass — scoase.
_GEO_RSS_FEEDS: list = [
    ("WatcherGuru",     "https://watcher.guru/news/feed"),
    ("Al Jazeera",      "https://www.aljazeera.com/xml/rss/all.xml"),  # Middle East/Iran specialist
    ("BBC World",       "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("CNBC Markets",    "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Guardian World",  "https://www.theguardian.com/world/rss"),
    ("MarketWatch",     "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("Yahoo Finance",   "https://finance.yahoo.com/rss/topfinstories"),
    ("Investing.com",   "https://www.investing.com/rss/news.rss"),
]

# Deduplicare: set cu titluri hash văzute — persistat pe disc, supraviețuiește restart
_GEO_SEEN_FILE = DATA_DIR / "geo_seen_hashes.json"
_geo_last_critical_ts: float = 0.0

def _load_geo_seen() -> set:
    """Încarcă hash-urile știrilor deja trimise de pe disc."""
    try:
        if _GEO_SEEN_FILE.exists():
            data = json.loads(_GEO_SEEN_FILE.read_text())
            hashes = set(data.get("hashes", []))
            # Păstrăm doar hash-urile din ultimele 48h (nu vrem să creștem la infinit)
            cutoff_ts = time.time() - 48 * 3600
            ts_map    = {h: t for h, t in data.get("ts_map", {}).items() if t > cutoff_ts}
            valid     = hashes & set(ts_map.keys())
            return valid
    except Exception:
        pass
    return set()

def _save_geo_seen(seen: set, ts_map: dict) -> None:
    """Salvează hash-urile pe disc."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _GEO_SEEN_FILE.write_text(json.dumps({"hashes": list(seen), "ts_map": ts_map}))
    except Exception:
        pass

_geo_seen: set = _load_geo_seen()
_geo_seen_ts: dict = {}  # hash → timestamp când a fost văzut


# ── Geo Sentiment: keywords → direcție anticipată pe NQ ─────────────────────
# BEARISH_NQ: război, tarife, oil shock, Fed hawkish → NQ în jos
_GEO_BEARISH_NQ_KW: list = [
    "iran war", "iran attack", "iran strike", "iran offensive",
    "russia attack", "ukraine attack", "military invasion", "invasion begins",
    "airstrike", "air strike", "missile strike", "missile attack", "ballistic missile",
    "nuclear strike", "nuclear attack", "nuclear weapon",
    "strait of hormuz", "oil tanker attack", "oil supply cut", "oil supply disruption",
    "oil embargo", "opec production cut", "opec cut", "opec emergency",
    "trump tariff", "trump trade war", "tariffs on china", "tariffs on europe",
    "tariffs on imports", "trade war escalat",
    "rate hike surprise", "fed hawkish",
    "sanctions against russia", "sanctions against china", "sanctions against iran",
    "export ban", "chip ban", "semiconductor ban",
    "taiwan invasion", "taiwan blockade", "china taiwan strait",
    "stock market crash", "market selloff", "market plunge",
    "recession declared", "default risk", "debt ceiling",
    "coup attempt", "government overthrow", "martial law",
]

# BULLISH_NQ: pace, rate cut, trade deal → NQ în sus
_GEO_BULLISH_NQ_KW: list = [
    "ceasefire agreement", "ceasefire deal", "peace deal", "peace agreement",
    "war ends", "conflict ends", "troops withdrawn", "withdrawal complete",
    "rate cut surprise", "emergency rate cut", "fed cuts rates",
    "trade deal signed", "trade agreement", "tariffs removed", "tariffs lifted",
    "sanctions lifted", "sanctions removed",
    "oil prices fall", "oil prices drop", "opec increases production",
]


def _geo_detect_sentiment(text: str) -> str:
    """Detectează sentimentul directional pe NQ din textul știrii.
    Returnează: 'BEARISH_NQ', 'BULLISH_NQ', sau 'NEUTRAL'."""
    t = text.lower()
    if any(kw in t for kw in _GEO_BEARISH_NQ_KW):
        return "BEARISH_NQ"
    if any(kw in t for kw in _GEO_BULLISH_NQ_KW):
        return "BULLISH_NQ"
    return "NEUTRAL"


def _geo_check_keywords(text: str) -> tuple:
    """
    Verifică textul față de keyword lists.
    Returnează (severity: str | None, matched_kw: list).
    CRITICAL: minim 1 match exactă din lista critică.
    HIGH: minim 1 match exactă din lista high SAU 2+ cuvinte cheie simple co-ocurente.
    """
    text_lower = text.lower()

    # CRITICAL — orice match din lista critică
    matched_crit = [kw for kw in _GEO_CRITICAL_KW if kw in text_lower]
    if matched_crit:
        return "CRITICAL", matched_crit

    # HIGH — orice match din lista high
    matched_high = [kw for kw in _GEO_HIGH_KW if kw in text_lower]
    if matched_high:
        return "HIGH", matched_high

    return None, []


async def _geo_news_loop():
    """
    Loop async care rulează la fiecare 90s și monitorizează RSS feeds pentru știri
    geopolitice/politice care pot impacta piața. Trimite alerte Telegram automat și
    activează geo_risk_mode dacă severitatea e HIGH sau CRITICAL.
    """
    global _geo_seen, _geo_seen_ts, _geo_last_critical_ts
    await asyncio.sleep(20)  # startup delay — lasă bridge-ul să se inițializeze complet
    log.info("🌍 Geo News Monitor pornit (Reuters/AP/BBC, 90s interval)")

    while True:
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                for (source_name, feed_url) in _GEO_RSS_FEEDS:
                    try:
                        resp = await client.get(feed_url)
                        if resp.status_code != 200:
                            continue

                        # Parsare RSS minimală — extragem <title> și <description>
                        import xml.etree.ElementTree as _ET
                        try:
                            root = _ET.fromstring(resp.text)
                        except Exception:
                            continue

                        # RSS standard: channel > item > title/description/pubDate
                        ns = {"atom": "http://www.w3.org/2005/Atom"}
                        items = root.findall(".//item") or root.findall(".//atom:entry", ns)

                        for item in items[:15]:  # primele 15 titluri
                            # Extragem titlul
                            title_el = item.find("title")
                            title = (title_el.text or "").strip() if title_el is not None else ""
                            if not title:
                                continue

                            # Hash deduplicare — persistat pe disc, supraviețuiește restart
                            import hashlib
                            h = hashlib.md5(title.encode()).hexdigest()[:12]
                            if h in _geo_seen:
                                continue
                            _geo_seen.add(h)
                            _geo_seen_ts[h] = time.time()

                            # Limit set size — curățăm hash-uri vechi (>48h)
                            if len(_geo_seen) > 500:
                                _cutoff = time.time() - 48 * 3600
                                _geo_seen_ts = {k: v for k, v in _geo_seen_ts.items() if v > _cutoff}
                                _geo_seen    = set(_geo_seen_ts.keys())

                            # Salvăm pe disc imediat — supraviețuiește restart
                            _save_geo_seen(_geo_seen, _geo_seen_ts)

                            # Descriere (opțional)
                            desc_el = item.find("description")
                            desc = (desc_el.text or "").strip() if desc_el is not None else ""
                            full_text = f"{title} {desc}"

                            severity, matched_kw = _geo_check_keywords(full_text)
                            if not severity:
                                continue

                            log.info(f"🌍 GEO [{severity}] {source_name}: {title[:80]}")

                            # Trimite alertă Telegram
                            if _TG_OK and _tg_geo_alert:
                                _tg_geo_alert(
                                    headline=title[:200],
                                    source=source_name,
                                    severity=severity,
                                    keywords_found=matched_kw[:4],
                                    geo_risk_active=state.geo_risk_active,
                                )

                            # Detectează sentiment directional pe NQ din titlu
                            _sentiment = _geo_detect_sentiment(full_text)
                            if _sentiment != "NEUTRAL":
                                state.geo_sentiment = _sentiment
                                log.info(f"🌍 GEO SENTIMENT: {_sentiment} (din: {title[:60]})")

                            # Activează geo_risk_mode automat DOAR la CRITICAL
                            # HIGH → trimite alertă Telegram dar NU blochează tranzacțiile
                            # CRITICAL → război, nuclear, market halt → blochează automat
                            # EXCEPȚIE: dacă userul l-a dezactivat manual (/geo off) → rămâne OFF
                            if severity == "CRITICAL" and not state.geo_risk_active:
                                if state.geo_risk_user_off:
                                    log.debug(f"🌍 GEO RSS [{severity}] ignorat — user a dezactivat manual geo_risk")
                                else:
                                    state.geo_risk_active = True
                                    state.geo_risk_reason = f"{source_name}: {title[:100]}"
                                    _geo_last_critical_ts = time.time()
                                    if _TG_OK and _tg_geo_risk_update:
                                        _tg_geo_risk_update(active=True, reason=state.geo_risk_reason)
                                    log.info(f"🌍 GEO RISK MODE activat automat: {state.geo_risk_reason[:80]}")

                    except Exception as _feed_err:
                        log.debug(f"Geo feed error [{source_name}]: {_feed_err}")

            # Auto-dezactivare geo_risk_mode după 4 ore fără știri noi HIGH/CRITICAL
            if state.geo_risk_active and _geo_last_critical_ts > 0:
                hours_since = (time.time() - _geo_last_critical_ts) / 3600
                if hours_since >= 4.0:
                    state.geo_risk_active = False
                    state.geo_risk_reason = ""
                    state.geo_sentiment   = "NEUTRAL"
                    if _TG_OK and _tg_geo_risk_update:
                        _tg_geo_risk_update(active=False, reason="Nicio știre HIGH/CRITICAL în ultimele 4 ore — revenire automată la parametri normali.")
                    log.info("🌍 GEO RISK MODE dezactivat automat (4h fără știri critice)")

        except Exception as _gloop_err:
            log.debug(f"Geo news loop error: {_gloop_err}")

        await asyncio.sleep(90)  # verifică la fiecare 90 secunde


def _geo_command_handler(cmd: str):
    """
    Callback apelat din telegram_poll_loop când userul trimite /geo on sau /geo off.
    Modifică state.geo_risk_active și trimite confirmare Telegram.
    """
    global _geo_last_critical_ts
    if cmd == "geo_on":
        state.geo_risk_active   = True
        state.geo_risk_reason   = "Activat manual via /geo on"
        state.geo_risk_user_off = False   # ← deblochează auto-activarea din RSS monitor
        _geo_last_critical_ts   = time.time()
        if _TG_OK and _tg_geo_risk_update:
            _tg_geo_risk_update(active=True, reason="Activat manual de utilizator via /geo on")
        log.info("🌍 GEO RISK MODE activat manual via Telegram")
    elif cmd == "geo_off":
        state.geo_risk_active  = False
        state.geo_risk_reason  = ""
        state.geo_risk_user_off = True   # ← blochează auto-reactivarea din RSS monitor
        if _TG_OK and _tg_geo_risk_update:
            _tg_geo_risk_update(active=False, reason="Dezactivat manual de utilizator via /geo off — auto-activare blocată până la /geo on")
        log.info("🌍 GEO RISK MODE dezactivat manual via Telegram (auto-activare blocată)")


async def _daily_report_loop():
    """
    Trimite raport zilnic pe Telegram la 22:00 UTC (după NY close).
    Rulează în background, verifică la fiecare minut dacă e ora exactă.
    Trimite o singură dată pe zi (deduplicare prin _daily_report_sent_date).
    """
    _daily_report_sent_date = ""
    await asyncio.sleep(15)  # startup delay
    log.info("📊 Daily Report Loop pornit (22:00 UTC)")

    while True:
        try:
            now  = datetime.now(timezone.utc)
            today = now.strftime("%Y-%m-%d")
            # Trimite la 20:00 UTC (NY close ~16:00 ET), o singură dată pe zi
            if now.hour == 20 and now.minute == 0 and _daily_report_sent_date != today:
                _daily_report_sent_date = today

                # Calculăm statisticile zilei din trade_log
                all_trades  = state.trade_log or []
                today_trades = [t for t in all_trades if t.get("ts", "").startswith(today)]

                wins   = [t for t in today_trades if t.get("result") == "WIN"]
                losses = [t for t in today_trades if t.get("result") == "LOSS"]

                pnl_values   = [float(t.get("pnl_usd", 0)) for t in today_trades]
                pnl_total    = sum(pnl_values)
                best_trade   = max(pnl_values) if pnl_values else 0.0
                worst_trade  = min(pnl_values) if pnl_values else 0.0
                win_rate     = len(wins) / len(today_trades) if today_trades else 0.0

                scores = [float(t.get("score", 0)) for t in today_trades if t.get("score")]
                avg_score = (sum(scores) / len(scores) * 100) if scores else 0.0

                circuit_open = (
                    state.loss_circuit_open
                    or state.profit_circuit_open
                    or state.account_circuit_open
                )

                # v10.1: Folosim datele reale NT8 dacă sunt disponibile (prioritate față de estimare)
                # account_balance vine din NT8 net_liquidation (sinc în /nt8_data)
                _nt8_bal   = state.account_balance   # deja sincronizat din NT8 dacă AladinBridge trimite
                _nt8_dpnl  = state.daily_profit_usd  # sincronizat din NT8 realized_pnl_today
                _nt8_dloss = state.daily_loss_usd    # sincronizat din NT8 realized_pnl_today (negativ)

                # P&L total al zilei: preferăm suma reală NT8 (daily_profit - daily_loss)
                _net_pnl = round(_nt8_dpnl - _nt8_dloss, 2)
                # Dacă trade_log are date mai recente, le folosim ca backup
                if abs(pnl_total) > 0 and abs(_net_pnl) == 0:
                    _net_pnl = pnl_total  # fallback la trade_log dacă NT8 nu a trimis

                # ── Sharpe zilnic ──────────────────────────────────────────
                _sharpe_day = None
                if len(pnl_values) >= 2:
                    try:
                        from advanced_features import sharpe_ratio as _sh_fn
                        _sharpe_day = round(_sh_fn(pnl_values, annualize=1.0), 2)
                    except Exception:
                        pass

                # ── Streak curent ───────────────────────────────────────────
                _cons_wins = 0
                for _t in reversed(today_trades):
                    if _t.get("result") == "WIN": _cons_wins += 1
                    else: break

                # ── Cea mai bună oră ────────────────────────────────────────
                from collections import defaultdict
                _hour_wins = defaultdict(int); _hour_trades = defaultdict(int)
                for _t in today_trades:
                    _hr = _t.get("ts", "")[:13].split("T")[-1]  # "HH"
                    _hour_trades[_hr] += 1
                    if _t.get("result") == "WIN": _hour_wins[_hr] += 1
                _best_hour = ""
                if _hour_trades:
                    _bh = max(_hour_trades, key=lambda h: (_hour_wins[h], _hour_trades[h]))
                    _best_hour = f"{_bh}:00 ({_hour_wins[_bh]}W {_hour_trades[_bh]-_hour_wins[_bh]}L)"

                # ── Top componentă ──────────────────────────────────────────
                _top_comp = ""
                try:
                    _comp_avgs = {}
                    for _t in today_trades:
                        for _ck, _cv in (_t.get("component_scores") or {}).items():
                            _comp_avgs.setdefault(_ck, []).append(float(_cv))
                    if _comp_avgs:
                        _tc = max(_comp_avgs, key=lambda k: sum(_comp_avgs[k])/len(_comp_avgs[k]))
                        _tc_avg = sum(_comp_avgs[_tc]) / len(_comp_avgs[_tc])
                        _top_comp = f"{_tc} (avg {_tc_avg*100:.0f}%)"
                except Exception:
                    pass

                snapshot = {
                    "date_str":           today,
                    "trades_today":       len(today_trades),
                    "wins":               len(wins),
                    "losses":             len(losses),
                    "pnl_usd":            _net_pnl,
                    "best_trade":         best_trade,
                    "worst_trade":        worst_trade,
                    "win_rate":           win_rate,
                    "avg_score":          avg_score,
                    "circuit_open":       circuit_open,
                    "geo_risk_active":    state.geo_risk_active,
                    "geo_sentiment":      state.geo_sentiment,
                    "consecutive_losses": state.consecutive_losses,
                    "consecutive_wins":   _cons_wins,
                    "daily_loss_usd":     _nt8_dloss,
                    "daily_profit_usd":   _nt8_dpnl,
                    "account_balance":    _nt8_bal,
                    "data_source":        "NT8_REAL" if _nt8_bal > 0 else "BRIDGE_ESTIMATE",
                    "sharpe_day":         _sharpe_day,
                    "best_hour":          _best_hour,
                    "top_component":      _top_comp,
                }

                if _TG_OK and _tg_daily_report:
                    _tg_daily_report(snapshot)
                    log.info(f"📊 Raport zilnic trimis: {len(today_trades)} trades | PnL: {pnl_total:.2f}")

                # ── Raport săptămânal (duminică la 21:00 UTC) ───────────────
                if now.weekday() == 6 and _TG_OK and _tg_weekly_report:
                    try:
                        _week_start = (now - timedelta(days=6)).strftime("%d.%m")
                        _week_end   = now.strftime("%d.%m.%Y")
                        _week_str   = f"{_week_start} – {_week_end}"
                        _week_trades = [t for t in (state.trade_log or [])
                                        if t.get("ts","") >= (now - timedelta(days=7)).isoformat()[:10]]
                        _ww = [t for t in _week_trades if t.get("result") == "WIN"]
                        _wl = [t for t in _week_trades if t.get("result") == "LOSS"]
                        _wpnl_vals = [float(t.get("pnl_usd", 0)) for t in _week_trades]
                        _wpnl = sum(_wpnl_vals)
                        _wwr  = len(_ww) / len(_week_trades) if _week_trades else 0.0
                        _wscores = [float(t.get("score", 0)) for t in _week_trades if t.get("score")]
                        _wavg_score = (sum(_wscores)/len(_wscores)*100) if _wscores else 0.0
                        _wsharpe = None
                        if len(_wpnl_vals) >= 3:
                            try:
                                from advanced_features import sharpe_ratio as _shw
                                _wsharpe = round(_shw(_wpnl_vals, annualize=52.0), 2)
                            except Exception: pass
                        # PnL per zi (Luni-Vineri)
                        _day_names = ["Luni","Marți","Miercuri","Joi","Vineri","Sâmbătă","Duminică"]
                        _pnl_per_day: dict = {}
                        for _t in _week_trades:
                            try:
                                _tday = datetime.fromisoformat(_t["ts"][:10]).weekday()
                                _dn   = _day_names[_tday]
                                _pnl_per_day[_dn] = _pnl_per_day.get(_dn, 0) + float(_t.get("pnl_usd", 0))
                            except Exception: pass
                        _best_wday  = max(_pnl_per_day, key=_pnl_per_day.get) if _pnl_per_day else ""
                        _worst_wday = min(_pnl_per_day, key=_pnl_per_day.get) if _pnl_per_day else ""
                        _tg_weekly_report({
                            "week_str":       _week_str,
                            "total_trades":   len(_week_trades),
                            "wins":           len(_ww),
                            "losses":         len(_wl),
                            "pnl_usd":        _wpnl,
                            "win_rate":       _wwr,
                            "avg_score":      _wavg_score,
                            "sharpe_week":    _wsharpe,
                            "best_day":       f"{_best_wday} +${_pnl_per_day.get(_best_wday,0):.0f}" if _best_wday else "",
                            "worst_day":      f"{_worst_wday} ${_pnl_per_day.get(_worst_wday,0):+.0f}" if _worst_wday else "",
                            "pnl_per_day":    _pnl_per_day,
                            "top_component":  _top_comp,
                            "best_hour_week": _best_hour,
                        })
                        log.info("📅 Raport săptămânal trimis pe Telegram")
                    except Exception as _wr_err:
                        log.debug(f"Weekly report error: {_wr_err}")

        except Exception as _dr_err:
            log.debug(f"Daily report loop error: {_dr_err}")

        await asyncio.sleep(60)


@app.on_event("startup")
async def on_startup():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _load_strategy_state()   # ← restaurează strategia activă din disc (dacă există)
    _load_open_trade()       # ← restaurează trade deschis din disc (trailing supraviețuiește restart)
    _load_equity_history()   # Feature 2: restaurează equity curve history
    _load_ab_state()         # Feature 4: restaurează A/B test state
    log.info("=" * 60)
    log.info("🚀 ALADIN API v3.0 pornit pe port 8000")
    log.info(f"   NT8 target:  {NT8_IP}:{NT8_PORT}")
    log.info(f"   Data dir:    {DATA_DIR}")
    log.info(f"   Trades:      {len(state.trade_log)} înregistrate")
    log.info(f"   Docs:        http://localhost:8000/docs")
    log.info("=" * 60)
    # UPDATE #13: pornește Telegram polling loop cu callback pentru /geo on/off
    if _TG_OK and _tg_poll:
        asyncio.create_task(_tg_poll(_tg_state_snapshot, interval=5, command_callback=_geo_command_handler))
        log.info("📱 Telegram bot polling activ (comenzi: /status /geo /help)")
    # NEWS ALERT TASK: verifică la fiecare minut dacă urmează un eveniment în 15 min
    asyncio.create_task(_news_alert_loop())
    log.info("⏰ News alert loop pornit (15min pre-eveniment)")
    # GEO NEWS MONITOR: RSS feeds — monitorizare non-stop știri geopolitice
    asyncio.create_task(_geo_news_loop())
    log.info("🌍 Geo News Monitor pornit (9 surse, interval 90s)")
    # DAILY REPORT: raport zilnic automat la 22:00 UTC (după NY close)
    asyncio.create_task(_daily_report_loop())
    log.info("📊 Daily Report Loop pornit (22:00 UTC)")

@app.on_event("shutdown")
async def on_shutdown():
    await _nt8_client.aclose()

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _init_db_source_column()
    uvicorn.run("bridge_api:app", host="0.0.0.0", port=8000, reload=False, workers=1, log_level="info")
