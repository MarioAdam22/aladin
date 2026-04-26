"""
quality_gate_live.py — Quality Gate Live v1.0
Înlocuiește ict_setup_scorer_v4.1 în ict_gate_v3.py

Modele:
  v2 : mario_quality_v6_calibrated.pkl  (LON)  +  mario_quality_ny_v3_calibrated.pkl  (NY)
  v5 : max(v6, ts_lon_v1) pentru LON reversale  +  max(ny_v3, ts_ny_v1) pentru NY reversale

Config: model_config.json  →  {"model":"v2"|"v5", "lon_thr":0.20, "ny_thr":0.20, "enabled":true}
"""

import json
import math
import pickle
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("quality_gate_live")

# ─── Paths ───────────────────────────────────────────────────────────────────
_DIR        = Path(__file__).parent
DB_PATH     = _DIR / "mario_trading.db"
CONFIG_PATH = _DIR / "model_config.json"
TRADES_FILE = Path.home() / "Desktop" / "Aladin" / "data" / "trades.json"

# ─── Economic Calendar ───────────────────────────────────────────────────────
_CAL_PATH = _DIR / "data" / "economic_calendar.json"
try:
    _cal = json.loads(_CAL_PATH.read_text())
    _FOMC_DATES   = set(_cal.get("fomc",    []))
    _NFP_DATES    = set(_cal.get("nfp",     []))
    _CPI_DATES    = set(_cal.get("cpi",     []))
    _PPI_DATES    = set(_cal.get("ppi",     []))
    _RETAIL_DATES = set(_cal.get("retail",  []))
    _ISM_DATES    = set(_cal.get("ism",     []))
    _ANY_HIGH     = set(_cal.get("any_high",[]))
    _NEWS_DAYS    = _FOMC_DATES | _NFP_DATES | _CPI_DATES | _PPI_DATES
except Exception as _cal_e:
    logger.warning(f"Economic calendar load error: {_cal_e}")
    _FOMC_DATES = _NFP_DATES = _CPI_DATES = _PPI_DATES = _RETAIL_DATES = _ISM_DATES = _ANY_HIGH = _NEWS_DAYS = set()

def _fomc_proximity(date_str: str) -> float:
    """Days to nearest FOMC event (for fomc_proximity feature)."""
    try:
        d = pd.Timestamp(date_str).date()
        diffs = [abs((d - pd.Timestamp(x).date()).days) for x in _FOMC_DATES]
        return float(min(diffs)) if diffs else 30.0
    except Exception:
        return 30.0

_MODEL_FILES = {
    "v6":     _DIR / "mario_quality_v6_calibrated.pkl",
    "ny_v3":  _DIR / "mario_quality_ny_v3_calibrated.pkl",
    "ts_lon": _DIR / "mario_quality_ts_lon_v1_calibrated.pkl",
    "ts_ny":  _DIR / "mario_quality_ts_ny_v1_calibrated.pkl",
}
_FEAT_FILES = {
    "v6":     _DIR / "mario_quality_v6_features.json",
    "ny_v3":  _DIR / "mario_quality_ny_v3_features.json",
    "ts_lon": _DIR / "mario_quality_ts_lon_v1_features.json",
    "ts_ny":  _DIR / "mario_quality_ts_ny_v1_features.json",
}

# ─── Lazy singletons ─────────────────────────────────────────────────────────
_MODELS:   dict = {}
_FEATURES: dict = {}
_CONFIG_CACHE: dict = {}
_CONFIG_MTIME: float = 0.0

# Regime-specific model cache: {"v6_EXPANSION": model, "ny_v3_RETRACEMENT": model, ...}
_REGIME_MODELS: dict = {}

# Mapping: model_name → PKL filename prefix
_MODEL_PREFIXES = {
    "v6": "mario_quality_v6",
    "ts_lon": "mario_quality_ts_lon_v1",
    "ny_v3": "mario_quality_ny_v3",
    "ts_ny": "mario_quality_ts_ny_v1",
}
_ACTIVE_REGIMES = ["CONSOLIDATION", "PRE_EXPANSION", "EXPANSION", "RETRACEMENT", "DISTRIBUTION"]


def _load_models():
    global _MODELS, _FEATURES, _REGIME_MODELS
    if _MODELS:
        return
    for name in ("v6", "ny_v3", "ts_lon", "ts_ny"):
        try:
            with open(_MODEL_FILES[name], "rb") as f:
                _MODELS[name] = pickle.load(f)
            with open(_FEAT_FILES[name]) as f:
                _FEATURES[name] = json.load(f)["features"]
            logger.info(f"✅ QualityGate '{name}' loaded ({len(_FEATURES[name])} features)")
        except Exception as e:
            logger.warning(f"QualityGate '{name}' load error: {e}")
        # Try to load regime-specific models (_v2 preferred, fallback to original)
        prefix = _MODEL_PREFIXES.get(name, '')
        for regime in _ACTIVE_REGIMES:
            rpath_v2  = _DIR / f"{prefix}_{regime}_v2_calibrated.pkl"
            rpath_v1  = _DIR / f"{prefix}_{regime}_calibrated.pkl"
            rpath = rpath_v2 if rpath_v2.exists() else rpath_v1
            if rpath.exists():
                try:
                    with open(rpath, "rb") as f:
                        _REGIME_MODELS[f"{name}_{regime}"] = pickle.load(f)
                    tag = "_v2" if rpath == rpath_v2 else ""
                    logger.info(f"  ✅ Regime model: {name}_{regime}{tag}")
                except Exception as re:
                    logger.debug(f"  Regime model {name}_{regime}: {re}")


def load_config() -> dict:
    """Citește model_config.json cu hot-reload la fiecare schimbare."""
    global _CONFIG_CACHE, _CONFIG_MTIME
    try:
        mtime = CONFIG_PATH.stat().st_mtime
        if mtime != _CONFIG_MTIME:
            _CONFIG_CACHE = json.loads(CONFIG_PATH.read_text())
            _CONFIG_MTIME = mtime
    except Exception:
        pass
    return _CONFIG_CACHE or {
        "model": "v2", "lon_thr": 0.20, "ny_thr": 0.20, "enabled": True
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _sf(val, default=0.0) -> float:
    """Safe float conversion."""
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _get_db_row(conn: sqlite3.Connection, now_utc: datetime) -> dict:
    """Cea mai recentă bară din market_data ≤ now_utc."""
    ts = now_utc.strftime("%Y-%m-%d %H:%M:00")
    row = conn.execute(
        "SELECT * FROM market_data WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
        (ts,)
    ).fetchone()
    if not row:
        return {}
    cols = [d[0] for d in conn.execute("PRAGMA table_info(market_data)").fetchall()]
    return dict(zip(cols, row))


def _get_rolling_regime(conn: sqlite3.Connection, now_date: str) -> dict:
    """ADX 10d, Hurst 20d, ATR 5d/20d pentru regime features."""
    try:
        rows = conn.execute("""
            SELECT date, AVG(adx_14) adx, AVG(hurst) hurst, AVG(atr_14) atr
            FROM market_data WHERE date < ?
            GROUP BY date ORDER BY date DESC LIMIT 20
        """, (now_date,)).fetchall()
        if not rows:
            return {"adx_10d": 25.0, "hurst_20d": 0.52, "atr_5d": 20.0, "atr_20d": 20.0}
        adx   = [_sf(r[1], 25.0) for r in rows]
        hurst = [_sf(r[2], 0.52) for r in rows]
        atr   = [_sf(r[3], 20.0) for r in rows]
        return {
            "adx_10d":  float(np.mean(adx[:10])),
            "hurst_20d": float(np.mean(hurst)),
            "atr_5d":   float(np.mean(atr[:5])),
            "atr_20d":  float(np.mean(atr)),
        }
    except Exception:
        return {"adx_10d": 25.0, "hurst_20d": 0.52, "atr_5d": 20.0, "atr_20d": 20.0}


def _get_week_extremes(conn: sqlite3.Connection, now_utc: datetime) -> dict:
    """Hi/Lo din săptămâna curentă + range de Luni."""
    monday = now_utc - timedelta(days=now_utc.weekday())
    mon_str = monday.strftime("%Y-%m-%d")
    today   = now_utc.strftime("%Y-%m-%d")
    try:
        r = conn.execute(
            "SELECT MAX(high), MIN(low) FROM market_data WHERE date BETWEEN ? AND ?",
            (mon_str, today)
        ).fetchone()
        wk_hi, wk_lo = _sf(r[0]), _sf(r[1])
        rm = conn.execute(
            "SELECT MAX(high), MIN(low) FROM market_data WHERE date = ?",
            (mon_str,)
        ).fetchone()
        mon_hi, mon_lo = _sf(rm[0]), _sf(rm[1])
        return {"week_hi": wk_hi, "week_lo": wk_lo, "mon_hi": mon_hi, "mon_lo": mon_lo}
    except Exception:
        return {"week_hi": 0.0, "week_lo": 0.0, "mon_hi": 0.0, "mon_lo": 0.0}


def _get_rolling_wr(session: str, now_date: str) -> float:
    """Rolling 5-session win rate din trades.json. Default 0.25 dacă nu e istoric."""
    try:
        if not TRADES_FILE.exists():
            return 0.25
        trades = json.loads(TRADES_FILE.read_text())
        sess_map: dict = {}
        for t in trades:
            if t.get("session", "") != session:
                continue
            d = str(t.get("date", ""))[:10]
            if d >= now_date:
                continue
            if d not in sess_map:
                sess_map[d] = {"w": 0, "n": 0}
            sess_map[d]["n"] += 1
            if t.get("result") in ("TRAIL", "WIN", "TP"):
                sess_map[d]["w"] += 1
        if not sess_map:
            return 0.25
        last5 = sorted(sess_map)[-5:]
        tot = sum(sess_map[d]["n"] for d in last5)
        win = sum(sess_map[d]["w"] for d in last5)
        return win / tot if tot > 0 else 0.25
    except Exception:
        return 0.25


# ─── MTF FVG / IFVG / Breaker ────────────────────────────────────────────────

def _ict_zones_on_tf(df_tf: pd.DataFrame, entry: float, lookback: int = 30) -> dict:
    """Detectează FVG/IFVG/Breaker/Rejection pe un timeframe și returnează valorile la entry."""
    zero = {"in_bull": 0.0, "in_bear": 0.0, "dist_bull": 9.9, "dist_bear": 9.9,
            "in_ifvg_b": 0.0, "in_ifvg_s": 0.0, "breaker_b": 0.0, "breaker_s": 0.0, "rejection": 0.0}
    if len(df_tf) < 5:
        return zero

    H = df_tf["high"].values.astype(float)
    L = df_tf["low"].values.astype(float)
    C = df_tf["close"].values.astype(float)
    O = df_tf["open"].values.astype(float)
    A = np.maximum(df_tf["atr"].fillna(20.0).values.astype(float), 1.0)
    n = len(H)

    bull_top = np.zeros(n); bull_bot = np.zeros(n)
    bear_top = np.zeros(n); bear_bot = np.zeros(n)
    for i in range(2, n):
        if H[i-2] < L[i] and (L[i] - H[i-2]) > 0.5:
            bull_top[i] = L[i]; bull_bot[i] = H[i-2]
        if L[i-2] > H[i] and (L[i-2] - H[i]) > 0.5:
            bear_top[i] = L[i-2]; bear_bot[i] = H[i]

    active_bull: list = []; active_bear: list = []
    inv_bull: list = []; inv_bear: list = []
    bull_obs: list = []; bear_obs: list = []

    in_bull_f = 0.0; in_bear_f = 0.0; db_f = 9.9; dbe_f = 9.9
    in_ifvg_b = 0.0; in_ifvg_s = 0.0; brk_b = 0.0; brk_s = 0.0; rej = 0.0

    for i in range(n):
        c = C[i]; l = L[i]; h = H[i]; a = A[i]

        # Expire/migrate active bull FVGs
        new_b = []
        for top, bot, j in active_bull:
            if i - j > lookback: continue
            if l < bot: inv_bull.append((top, bot, i))
            else: new_b.append((top, bot, j))
        active_bull = new_b

        new_b = []
        for top, bot, j in active_bear:
            if i - j > lookback: continue
            if h > top: inv_bear.append((top, bot, i))
            else: new_b.append((top, bot, j))
        active_bear = new_b

        if bull_top[i] > 0: active_bull.append((bull_top[i], bull_bot[i], i))
        if bear_top[i] > 0: active_bear.append((bear_top[i], bear_bot[i], i))

        if i >= 2:
            pb = C[i-1] - O[i-1]; pr = max(H[i-1] - L[i-1], 0.01)
            if pb >  0.55 * pr and pb > 1.0:  bull_obs.append((C[i-1], O[i-1], i-1))
            if pb < -0.55 * pr and abs(pb) > 1.0: bear_obs.append((O[i-1], C[i-1], i-1))

        if i == n - 1:
            for top, bot, _ in active_bull:
                if bot <= entry <= top: in_bull_f = 1.0
                db_f = min(db_f, min(abs(entry - top), abs(entry - bot)) / a)
            for top, bot, _ in active_bear:
                if bot <= entry <= top: in_bear_f = 1.0
                dbe_f = min(dbe_f, min(abs(entry - top), abs(entry - bot)) / a)
            for top, bot, k in inv_bull[-15:]:
                if i - k <= lookback * 2 and bot <= entry <= top: in_ifvg_b = 1.0
            for top, bot, k in inv_bear[-15:]:
                if i - k <= lookback * 2 and bot <= entry <= top: in_ifvg_s = 1.0
            for top, bot, j in bull_obs[-20:]:
                if i - j <= lookback and entry < min(bot, O[j]) - a * 0.05:
                    if abs(entry - top) / a < 0.8 or abs(entry - bot) / a < 0.8: brk_s = 1.0
            for top, bot, j in bear_obs[-20:]:
                if i - j <= lookback and entry > max(top, O[j]) + a * 0.05:
                    if abs(entry - top) / a < 0.8 or abs(entry - bot) / a < 0.8: brk_b = 1.0
            if i >= 2:
                wick_u = H[i-1] - max(C[i-1], O[i-1])
                wick_d = min(C[i-1], O[i-1]) - L[i-1]
                body_  = abs(C[i-1] - O[i-1]); al = A[i-1]
                if wick_u > 2.5 * max(body_, 0.5) and wick_u > al * 0.3:
                    rt = H[i-1]; rb = max(C[i-1], O[i-1])
                    if abs(entry - rt) / al < 0.6 or abs(entry - rb) / al < 0.6: rej = 1.0
                if wick_d > 2.5 * max(body_, 0.5) and wick_d > al * 0.3:
                    rt = min(C[i-1], O[i-1]); rb = L[i-1]
                    if abs(entry - rt) / al < 0.6 or abs(entry - rb) / al < 0.6: rej = 1.0

    return {"in_bull": in_bull_f, "in_bear": in_bear_f,
            "dist_bull": min(db_f, 9.9), "dist_bear": min(dbe_f, 9.9),
            "in_ifvg_b": in_ifvg_b, "in_ifvg_s": in_ifvg_s,
            "breaker_b": brk_b, "breaker_s": brk_s, "rejection": rej}


def _compute_mtf(conn: sqlite3.Connection, now_utc: datetime,
                 entry: float, atr: float, direction: str) -> dict:
    """Calculează toate MTF features (5m/15m/1h/4h) la momentul semnalului."""
    dir_num = -1 if direction == "SHORT" else 1
    ts = now_utc.strftime("%Y-%m-%d %H:%M:00")
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, atr_14
        FROM market_data WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 600
    """, (ts,)).fetchall()

    blank = {}
    for tf in ["5m", "15m", "1h", "4h"]:
        for k in [f"in_bull_fvg_{tf}", f"in_bear_fvg_{tf}", f"dist_bull_fvg_{tf}",
                  f"dist_bear_fvg_{tf}", f"fvg_aligned_{tf}", f"in_ifvg_{tf}",
                  f"ifvg_aligned_{tf}", f"breaker_aligned_{tf}", f"rejection_{tf}"]:
            blank[k] = 0.0
    blank.update({"fvg_tf_confluence": 0.0, "htf_fvg_aligned": 0.0,
                  "ifvg_htf_aligned": 0.0, "vol_x_fvg_1h": 0.0,
                  "vol_x_htf_fvg": 0.0, "regime_x_htf_fvg": 0.0})
    if not rows:
        return blank

    df1 = pd.DataFrame(rows[::-1], columns=["ts", "open", "high", "low", "close", "atr"])
    df1["ts"] = pd.to_datetime(df1["ts"])
    df1 = df1.set_index("ts").sort_index()
    for col in ["open", "high", "low", "close"]:
        df1[col] = df1[col].astype(float)
    df1["atr"] = df1["atr"].astype(float).ffill().fillna(atr)

    result = dict(blank)
    for tf_min, tf_label in [(5, "5m"), (15, "15m"), (60, "1h"), (240, "4h")]:
        df_tf = df1.resample(f"{tf_min}min").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"), atr=("atr", "mean")
        ).dropna(subset=["open"])
        df_tf["atr"] = df_tf["atr"].fillna(atr)
        z = _ict_zones_on_tf(df_tf, entry)
        result[f"in_bull_fvg_{tf_label}"]    = z["in_bull"]
        result[f"in_bear_fvg_{tf_label}"]    = z["in_bear"]
        result[f"dist_bull_fvg_{tf_label}"]  = z["dist_bull"]
        result[f"dist_bear_fvg_{tf_label}"]  = z["dist_bear"]
        result[f"fvg_aligned_{tf_label}"]    = z["in_bear"] if dir_num == -1 else z["in_bull"]
        result[f"in_ifvg_{tf_label}"]        = max(z["in_ifvg_b"], z["in_ifvg_s"])
        result[f"ifvg_aligned_{tf_label}"]   = z["in_ifvg_s"] if dir_num == -1 else z["in_ifvg_b"]
        result[f"breaker_aligned_{tf_label}"]= z["breaker_s"] if dir_num == -1 else z["breaker_b"]
        result[f"rejection_{tf_label}"]      = z["rejection"]

    result["fvg_tf_confluence"] = float(sum(
        1 for tf in ["5m", "15m", "1h", "4h"] if result.get(f"fvg_aligned_{tf}", 0) > 0
    ))
    result["htf_fvg_aligned"]  = max(result["fvg_aligned_1h"], result["fvg_aligned_4h"])
    result["ifvg_htf_aligned"] = max(result["ifvg_aligned_1h"], result["ifvg_aligned_4h"])
    return result


# ─── Feature Builders ─────────────────────────────────────────────────────────

def _build_v6(sig: dict, db: dict, mtf: dict, reg: dict, wk: dict,
              rolling_wr: float, now_utc: datetime) -> dict:
    """Construiește toți cei 147 de features v6 (LON)."""
    f: dict = {}
    entry   = _sf(sig.get("entry", 0))
    dire    = sig.get("direction", "LONG")
    dir_num = -1 if dire == "SHORT" else 1
    atr     = max(_sf(db.get("atr_14", 20.0), 20.0), 1.0)

    # ── Time
    f["dir_short"]    = 1.0 if dire == "SHORT" else 0.0
    f["hour_utc"]     = float(now_utc.hour)
    f["min_in_lon"]   = float(max(0.0, (now_utc.hour - 7) * 60 + now_utc.minute))
    f["day_of_week"]  = float(now_utc.weekday())
    f["month"]        = float(now_utc.month)
    f["is_monday"]    = 1.0 if now_utc.weekday() == 0 else 0.0
    f["is_tuesday"]   = 1.0 if now_utc.weekday() == 1 else 0.0
    f["is_wednesday"] = 1.0 if now_utc.weekday() == 2 else 0.0
    f["is_thursday"]  = 1.0 if now_utc.weekday() == 3 else 0.0
    f["is_friday"]    = 1.0 if now_utc.weekday() == 4 else 0.0
    f["year_norm"]    = (now_utc.year - 2020) / 5.0

    # ── Signal
    f["confidence"]  = _sf(sig.get("confluence_score", sig.get("confidence", 0.5)))
    f["atr_entry"]   = atr
    atr_20d = max(reg.get("atr_20d", atr), 1.0)
    f["atr_vs_10d"]  = atr / atr_20d

    # ── Asia sweep
    ah = _sf(db.get("asia_hi", entry)); al = _sf(db.get("asia_lo", entry))
    f["dist_asia_hi_atr"] = (entry - ah) / atr
    f["dist_asia_lo_atr"] = (entry - al) / atr
    f["asia_range_atr"]   = (ah - al) / atr
    f["swept_asia_hi"]    = 1.0 if dire == "SHORT" else 0.0
    f["swept_asia_lo"]    = 1.0 if dire == "LONG"  else 0.0
    f["asia_midpoint"]    = (entry - (ah + al) / 2) / atr if ah > 0 else 0.0

    # ── Previous day
    pdh = _sf(db.get("p_hi", entry)); pdl = _sf(db.get("p_lo", entry))
    f["dist_pdh_atr"]   = (entry - pdh) / atr
    f["dist_pdl_atr"]   = (entry - pdl) / atr
    to = _sf(db.get("true_open", entry))
    f["above_true_open"] = 1.0 if entry > to else 0.0
    f["dist_true_open"]  = (entry - to) / atr

    # ── HTF bias
    h4h = _sf(db.get("h4_hi", 0)); h4l = _sf(db.get("h4_lo", 0))
    h1h = _sf(db.get("h1_hi", 0)); h1l = _sf(db.get("h1_lo", 0))
    h4b = (1 if entry > (h4h + h4l) / 2 else -1) if h4h > 0 else 0
    h1b = (1 if entry > (h1h + h1l) / 2 else -1) if h1h > 0 else 0
    f["h4_bias"]      = float(h4b)
    f["h1_bias"]      = float(h1b)
    f["h4_h1_aligned"]= 1.0 if (h4b == h1b and h4b != 0) else 0.0

    # ── Weekly levels
    lwh = _sf(db.get("lw_hi", entry)); lwl = _sf(db.get("lw_lo", entry))
    lw_rng = max(lwh - lwl, 0.01)
    wp = (entry - lwl) / lw_rng if lwh != lwl else 0.5
    f["weekly_premium_pct"]  = wp
    f["in_weekly_premium"]   = 1.0 if wp > 0.5 else 0.0
    f["in_weekly_discount"]  = 1.0 if wp < 0.5 else 0.0
    f["weekly_prem_aligned"] = 1.0 if (dire == "SHORT" and wp > 0.5) or (dire == "LONG" and wp < 0.5) else 0.0
    f["dist_prev_wk_hi"]     = (entry - lwh) / atr
    f["dist_prev_wk_lo"]     = (entry - lwl) / atr
    f["lw_range_atr"]        = lw_rng / atr
    f["dist_lw_hi"]          = f["dist_prev_wk_hi"]
    f["dist_lw_lo"]          = f["dist_prev_wk_lo"]

    wk_hi = wk.get("week_hi", entry); wk_lo = wk.get("week_lo", entry)
    f["week_range_so_far"] = (wk_hi - wk_lo) / atr if wk_hi > wk_lo else 0.5
    f["week_hi_taken"]     = 1.0 if wk_hi > lwh else 0.0
    f["week_lo_taken"]     = 1.0 if wk_lo < lwl and lwl > 0 else 0.0
    mon_hi = wk.get("mon_hi", 0); mon_lo = wk.get("mon_lo", 0)
    f["monday_range_pt"]   = (mon_hi - mon_lo) / atr if mon_hi > mon_lo else 0.5
    f["monday_consol"]     = 1.0 if (mon_hi - mon_lo) < atr * 1.5 else 0.0
    f["tuesday_rev_ctx"]   = 1.0 if now_utc.weekday() == 1 else 0.0
    f["wednesday_rev_ctx"] = 1.0 if now_utc.weekday() == 2 else 0.0

    # ── Volume Profile
    poc = _sf(db.get("poc_level", entry))
    vah = _sf(db.get("vah", entry + atr)); val_ = _sf(db.get("val", entry - atr))
    f["inside_va"]    = float(_sf(db.get("inside_va", 0)))
    f["dist_poc_atr"] = abs(entry - poc) / atr
    f["dist_vwap_atr"]= abs(_sf(db.get("dist_vwap", 0))) / atr
    f["vah_dist"]     = (entry - vah) / atr
    f["val_dist"]     = (entry - val_) / atr

    # ── ICT structure
    f["has_displacement"] = float(_sf(db.get("has_displacement", 1)))
    f["fvg_up"]           = float(_sf(db.get("fvg_up",  0)))
    f["fvg_down"]         = float(_sf(db.get("fvg_down", 0)))
    f["is_smt_bearish"]   = float(_sf(db.get("is_smt_bearish", 0)))
    f["is_smt_bullish"]   = float(_sf(db.get("is_smt_bullish", 0)))

    # ── Market regime
    hurst  = _sf(db.get("hurst",  0.52), 0.52)
    adx    = _sf(db.get("adx_14", 25.0), 25.0)
    acf1   = _sf(db.get("acf_lag1", 0.0))
    acf5   = _sf(db.get("acf_lag5", 0.0))
    fsh    = _sf(db.get("fisher_transform", 0.0))
    fft    = _sf(db.get("fft_cycle", 0.0))
    ks     = _sf(db.get("kalman_smooth", entry), entry)
    kn     = _sf(db.get("kalman_noise", 0.0))
    garch  = _sf(db.get("garch_vol", atr), atr)
    se     = _sf(db.get("sample_entropy", 0.0))

    f["hurst"]           = hurst
    f["adx_14"]          = adx
    f["adx_strong"]      = 1.0 if adx > 25 else 0.0
    f["acf_lag1"]        = acf1
    f["acf_lag5"]        = acf5
    f["fisher_transform"]= fsh
    f["fisher_extreme"]  = 1.0 if abs(fsh) > 2.0 else 0.0
    f["fft_cycle"]       = fft
    f["kalman_smooth"]   = (entry - ks) / atr
    f["kalman_noise"]    = kn / atr
    f["garch_vol_atr"]   = garch / atr
    f["sample_entropy"]  = se

    # ── Order flow
    bd  = _sf(db.get("bar_delta", 0))
    cd  = _sf(db.get("cum_delta", 0))
    bv  = max(_sf(db.get("bar_buy_vol",  0.5)), 0.001)
    sv  = max(_sf(db.get("bar_sell_vol", 0.5)), 0.001)
    tv  = bv + sv
    f["bar_delta_norm"]   = bd / max(tv * atr, 1.0)
    f["cum_delta_norm"]   = cd / max(tv * atr, 1.0)
    f["buy_sell_ratio"]   = bv / tv
    f["absorption_score"] = _sf(db.get("absorption_score", 0))
    f["stacked_bull"]     = float(_sf(db.get("stacked_bull", 0)))
    f["stacked_bear"]     = float(_sf(db.get("stacked_bear", 0)))
    f["of_doi"]           = _sf(db.get("of_doi", 0))

    # ── Bar features
    bo = _sf(db.get("open",  entry)); bc = _sf(db.get("close", entry))
    bh = _sf(db.get("high",  entry)); bl = _sf(db.get("low",   entry))
    rng = max(bh - bl, 0.01); body = bc - bo
    f["body_bear"]     = 1.0 if body < 0 else 0.0
    f["body_pct"]      = abs(body) / rng
    sw = (bh - max(bc, bo)) if dire == "SHORT" else (min(bc, bo) - bl)
    f["sweep_wick_atr"]= max(sw, 0.0) / atr

    # ── Interaction
    f["dir_x_adx"]        = f["dir_short"] * adx
    f["dir_x_hurst"]      = f["dir_short"] * hurst
    f["confidence_x_adx"] = f["confidence"] * adx
    f["hour_x_dir"]       = f["hour_utc"] * f["dir_short"]
    f["year_x_adx"]       = f["year_norm"] * adx
    f["year_x_hurst"]     = f["year_norm"] * hurst

    # ── MTF FVG/IFVG/Breaker
    for key in mtf:
        if key in [
            "in_bull_fvg_5m","in_bear_fvg_5m","dist_bull_fvg_5m","dist_bear_fvg_5m","fvg_aligned_5m",
            "in_ifvg_5m","ifvg_aligned_5m","breaker_aligned_5m","rejection_5m",
            "in_bull_fvg_15m","in_bear_fvg_15m","dist_bull_fvg_15m","dist_bear_fvg_15m","fvg_aligned_15m",
            "in_ifvg_15m","ifvg_aligned_15m","breaker_aligned_15m","rejection_15m",
            "in_bull_fvg_1h","in_bear_fvg_1h","dist_bull_fvg_1h","dist_bear_fvg_1h","fvg_aligned_1h",
            "in_ifvg_1h","ifvg_aligned_1h","breaker_aligned_1h","rejection_1h",
            "in_bull_fvg_4h","in_bear_fvg_4h","dist_bull_fvg_4h","dist_bear_fvg_4h","fvg_aligned_4h",
            "in_ifvg_4h","ifvg_aligned_4h","breaker_aligned_4h","rejection_4h",
            "fvg_tf_confluence","htf_fvg_aligned","ifvg_htf_aligned",
        ]:
            f[key] = mtf[key]

    # ── Volatility regime
    atr_5d = reg.get("atr_5d", atr); atr_20d2 = reg.get("atr_20d", atr)
    f["vix_proxy_5d"]   = atr_5d / max(atr_20d2, 1.0)
    f["vix_proxy_20d"]  = 1.0
    vol_r = float(np.clip(atr / max(atr_20d2, 1.0), 0.3, 3.0))
    f["vol_regime"]     = vol_r
    f["vol_high"]       = 1.0 if atr > atr_20d2 * 1.3 else 0.0
    f["vol_low"]        = 1.0 if atr < atr_20d2 * 0.7 else 0.0
    f["atr_trend"]      = 1.0 if atr > atr_20d2 else -1.0
    f["atr_expanding"]  = 1.0 if atr > atr_5d else 0.0
    f["atr_contracting"]= 1.0 if atr < atr_5d * 0.85 else 0.0
    f["vol_x_fvg_1h"]   = vol_r * f.get("fvg_aligned_1h",  0.0)
    f["vol_x_htf_fvg"]  = vol_r * f.get("htf_fvg_aligned", 0.0)

    # ── Sweep quality
    swept_lvl = ah if dire == "SHORT" else al
    sd_atr = abs(entry - swept_lvl) / atr if swept_lvl > 0 else 0.5
    f["equal_level_score"]   = 1.0 if sd_atr < 0.3 else 0.0
    f["sweep_depth_atr"]     = sd_atr
    f["deep_sweep"]          = 1.0 if sd_atr > 1.5 else 0.0
    f["shallow_sweep"]       = 1.0 if sd_atr < 0.5 else 0.0
    f["sweep_wick_clean"]    = 1.0 if f["sweep_wick_atr"] > 0.5 else 0.0
    f["sweep_with_disp"]     = f["has_displacement"]
    f["sweep_quality_score"] = float(np.clip(
        0.3 * (1 - f["shallow_sweep"]) +
        0.3 * f["sweep_wick_clean"] +
        0.2 * f["has_displacement"] +
        0.2 * (1.0 if h4b == dir_num else 0.0),
        0.0, 1.0
    ))

    # ── Rolling / regime
    adx_10d   = reg.get("adx_10d", 25.0)
    hurst_20d = reg.get("hurst_20d", 0.52)
    f["rolling_5sess_wr"]   = rolling_wr
    f["adx_10d_mean"]       = adx_10d
    f["hurst_20d_mean"]     = hurst_20d
    f["regime_trending"]    = 1.0 if adx_10d > 25 else 0.0
    f["regime_hurst_trend"] = 1.0 if hurst_20d > 0.52 else 0.0
    f["recent_wr_high"]     = 1.0 if rolling_wr > 0.35 else 0.0
    f["recent_wr_low"]      = 1.0 if rolling_wr < 0.15 else 0.0
    f["regime_score"]       = float(np.clip(
        (adx_10d - 20) / 20 * 0.4 +
        (hurst_20d - 0.45) / 0.1 * 0.3 +
        (rolling_wr - 0.1) / 0.3 * 0.3,
        0.0, 1.0
    ))
    f["regime_x_htf_fvg"]   = f["regime_score"] * f.get("htf_fvg_aligned", 0.0)
    f["adx_x_sweep_quality"]= (adx / 50.0) * f["sweep_quality_score"]

    return f


def _build_ny_v3(sig: dict, db: dict, mtf: dict, reg: dict, wk: dict,
                 rolling_wr: float, now_utc: datetime, conn: sqlite3.Connection) -> dict:
    """Construiește toți cei 206 de features ny_v3 (NY). Reutilizează v6 + adaugă NY-specific."""
    # Pornire de la v6 base (features comune)
    f = _build_v6(sig, db, mtf, reg, wk, rolling_wr, now_utc)
    # Suprascrie min_in_lon → min_in_ny
    entry   = _sf(sig.get("entry", 0))
    dire    = sig.get("direction", "LONG")
    dir_num = -1 if dire == "SHORT" else 1
    atr     = max(_sf(db.get("atr_14", 20.0), 20.0), 1.0)

    # ── NY time
    f["min_in_ny"]  = float(max(0.0, (now_utc.hour - 13) * 60 + now_utc.minute))
    f["session_pct"]= float(np.clip(f["min_in_ny"] / 120.0, 0.0, 1.0))

    # ── Economic calendar (real dates din economic_calendar.json)
    _d = now_utc.strftime("%Y-%m-%d")
    f["is_fomc_day"]     = 1.0 if _d in _FOMC_DATES   else 0.0
    f["is_nfp_day"]      = 1.0 if _d in _NFP_DATES    else 0.0
    f["is_cpi_day"]      = 1.0 if _d in _CPI_DATES    else 0.0
    f["is_ppi_day"]      = 1.0 if _d in _PPI_DATES    else 0.0
    f["is_retail_day"]   = 1.0 if _d in _RETAIL_DATES else 0.0
    f["is_ism_day"]      = 1.0 if _d in _ISM_DATES    else 0.0
    f["is_any_high_day"] = 1.0 if _d in _ANY_HIGH     else 0.0
    f["is_news_day"]     = 1.0 if _d in _NEWS_DAYS    else 0.0
    f["fomc_proximity"]  = float(np.clip(_fomc_proximity(_d) / 14.0, 0.0, 1.0))

    # ── LON hi/lo (swept levels pentru NY)
    lon_hi = _sf(db.get("lon_hi", entry)); lon_lo = _sf(db.get("lon_lo", entry))
    f["dist_lon_hi_atr"] = (entry - lon_hi) / atr
    f["dist_lon_lo_atr"] = (entry - lon_lo) / atr
    f["lon_range_atr"]   = (lon_hi - lon_lo) / atr if lon_hi > lon_lo else 0.5
    f["swept_lon_hi"]    = 1.0 if dire == "SHORT" else 0.0
    f["swept_lon_lo"]    = 1.0 if dire == "LONG"  else 0.0
    lon_mid = (lon_hi + lon_lo) / 2 if lon_hi > 0 else entry
    f["lon_midpoint"]    = (entry - lon_mid) / atr
    f["lon_range_narrow"]= 1.0 if (lon_hi - lon_lo) < atr else 0.0

    # ── NY open vs LON range
    # NY open = prețul la 13:00 UTC (aproximat din entry)
    ny_open_in_lon  = 1.0 if lon_lo <= entry <= lon_hi else 0.0
    ny_open_ab_mid  = 1.0 if entry > lon_mid else 0.0
    ny_open_bw_mid  = 1.0 if entry < lon_mid else 0.0
    f["ny_open_in_lon"]      = ny_open_in_lon
    f["ny_open_above_lon_mid"]= ny_open_ab_mid
    f["ny_open_below_lon_mid"]= ny_open_bw_mid

    # ── LON close context (12:00-12:30 UTC bars)
    try:
        today_str = now_utc.strftime("%Y-%m-%d")
        lon_close_rows = conn.execute("""
            SELECT AVG(close) FROM market_data
            WHERE date = ? AND hour_min BETWEEN '11:30' AND '12:30'
        """, (today_str,)).fetchone()
        lon_close_px = _sf(lon_close_rows[0], lon_mid) if lon_close_rows else lon_mid
    except Exception:
        lon_close_px = lon_mid
    f["lon_close_vs_mid"] = (lon_close_px - lon_mid) / atr if lon_hi > 0 else 0.0
    f["lon_closed_weak"]  = 1.0 if abs(lon_close_px - lon_mid) / atr < 0.3 else 0.0
    # LON direction explicit (missing vs ny_v2)
    f["lon_dir_explicit"]     = 1.0 if lon_close_px > lon_mid else -1.0
    f["lon_dir_aligned"]      = 1.0 if f["lon_dir_explicit"] == dir_num else 0.0
    f["lon_dir_opposite"]     = 1.0 if f["lon_dir_explicit"] != dir_num else 0.0
    f["ny_open_dist_lon_mid"] = (entry - lon_mid) / atr  # signed: + = entry above LON mid

    # ── LON vs Asia alignment
    ah = _sf(db.get("asia_hi", entry)); al = _sf(db.get("asia_lo", entry))
    asia_mid = (ah + al) / 2 if ah > 0 else entry
    lon_vs_asia = 1 if lon_mid > asia_mid else -1
    h4h = _sf(db.get("h4_hi", 0)); h4l = _sf(db.get("h4_lo", 0))
    h4b = (1 if entry > (h4h + h4l) / 2 else -1) if h4h > 0 else 0
    f["lon_vs_asia_aligned"] = 1.0 if lon_vs_asia == h4b else 0.0
    f["triple_sess_aligned"] = 1.0 if lon_vs_asia == dir_num and h4b == dir_num else 0.0

    # ── NY open drive (primele 15 min: 13:00-13:15 UTC)
    try:
        today_str = now_utc.strftime("%Y-%m-%d")
        drive_rows = conn.execute("""
            SELECT open, high, low, close FROM market_data
            WHERE date = ? AND hour_min BETWEEN '13:00' AND '13:14'
            ORDER BY timestamp
        """, (today_str,)).fetchall()
        if drive_rows:
            drive_open  = _sf(drive_rows[0][0])
            drive_close = _sf(drive_rows[-1][3])
            drive_hi    = max(_sf(r[1]) for r in drive_rows)
            drive_lo    = min(_sf(r[2]) for r in drive_rows)
            drive_rng   = max(drive_hi - drive_lo, 0.01)
            drive_move  = drive_close - drive_open
            is_bull = 1.0 if drive_move >  atr * 0.15 else 0.0
            is_bear = 1.0 if drive_move < -atr * 0.15 else 0.0
            is_neut = 1.0 if abs(drive_move) <= atr * 0.15 else 0.0
            f["ny_open_drive_bull"]   = is_bull
            f["ny_open_drive_bear"]   = is_bear
            f["ny_open_drive_neutral"]= is_neut
            f["ny15_range_atr"]       = drive_rng / atr
            f["drive_aligned_dir"]    = 1.0 if (is_bull and dir_num == 1) or (is_bear and dir_num == -1) else 0.0
        else:
            f["ny_open_drive_bull"] = f["ny_open_drive_bear"] = 0.0
            f["ny_open_drive_neutral"] = 1.0
            f["ny15_range_atr"] = 0.5
            f["drive_aligned_dir"] = 0.0
    except Exception:
        f["ny_open_drive_bull"] = f["ny_open_drive_bear"] = 0.0
        f["ny_open_drive_neutral"] = 1.0
        f["ny15_range_atr"] = 0.5
        f["drive_aligned_dir"] = 0.0

    # ── Gap vs LON close
    f["gap_vs_lon_close_atr"] = (entry - lon_close_px) / atr
    f["gap_up"]   = 1.0 if entry > lon_close_px + atr * 0.2 else 0.0
    f["gap_down"] = 1.0 if entry < lon_close_px - atr * 0.2 else 0.0
    f["gap_aligned"] = 1.0 if (f["gap_down"] and dir_num == -1) or (f["gap_up"] and dir_num == 1) else 0.0

    # ── Prev NY push direction (approximation from prev day close vs open)
    pdh = _sf(db.get("p_hi", entry)); pdl = _sf(db.get("p_lo", entry))
    f["prev_ny_push_dir"]  = 0.0  # neutral default
    f["prev_ny_aligned"]   = 0.0
    f["prev_ny_opposite"]  = 0.0

    # ── PDH/PDL direct distance (non-ATR normalized)
    f["dist_pdh_direct"] = entry - pdh
    f["dist_pdl_direct"] = entry - pdl

    # ── H4 structure
    f["h4_bias_aligned"]    = 1.0 if (h4b == dir_num) else 0.0
    h4_mid = (h4h + h4l) / 2 if h4h > 0 else entry
    f["h4_bullish_struct"]  = 1.0 if entry > h4_mid else 0.0
    f["h4_bearish_struct"]  = 1.0 if entry < h4_mid else 0.0
    f["h4_struct_aligned"]  = 1.0 if (dir_num == 1 and entry > h4_mid) or (dir_num == -1 and entry < h4_mid) else 0.0

    # ── Weekly premium direction
    wp = f.get("weekly_premium_pct", 0.5)
    f["weekly_prem_direction"] = 1.0 if wp > 0.7 else (-1.0 if wp < 0.3 else 0.0)

    # ── ATR ratio week (current ATR vs weekly average)
    atr_5d = reg.get("atr_5d", atr)
    f["atr_ratio_week"] = atr / max(atr_5d, 1.0)
    # LON range vs 5d ATR (big/small LON day)
    lon_range_raw = lon_hi - lon_lo
    f["lon_range_vs_atr5d"] = float(np.clip(lon_range_raw / max(atr_5d, 1.0), 0.0, 10.0))
    f["lon_big_day"]        = 1.0 if lon_range_raw > atr * 1.5 else 0.0
    f["lon_small_day"]      = 1.0 if lon_range_raw < atr * 0.7 else 0.0

    # ── Thursday consolidation
    f["thursday_consol"] = 1.0 if now_utc.weekday() == 3 else 0.0

    # ── SMT aligned
    f["smt_aligned"] = float(max(_sf(db.get("is_smt_bearish", 0)), _sf(db.get("is_smt_bullish", 0))))

    # ── OB proxy (approximation from stacked bars)
    f["ob_proxy_bull"] = float(_sf(db.get("stacked_bull", 0)))
    f["ob_proxy_bear"] = float(_sf(db.get("stacked_bear", 0)))
    f["ob_aligned"]    = f["ob_proxy_bear"] if dir_num == -1 else f["ob_proxy_bull"]

    # ── Day type
    day_rng = (pdh - pdl) / atr if pdh > pdl else 1.0
    f["inside_day"] = 1.0 if day_rng < 1.0 else 0.0
    f["trend_day"]  = 1.0 if day_rng > 2.5 else 0.0
    f["range_day"]  = 1.0 if 0.8 < day_rng < 2.0 else 0.0

    # ── NY sub-session split
    _min_abs = now_utc.hour * 60 + now_utc.minute
    f["is_ny_open"]      = 1.0 if (780 <= _min_abs < 870) else 0.0   # 13:00-14:30
    f["is_ny_afternoon"] = 1.0 if _min_abs >= 870 else 0.0            # 14:30+
    f["min_in_ny_open"]  = float(np.clip(_min_abs - 780, 0, 90))
    f["ny_open_early"]   = 1.0 if (780 <= _min_abs < 810) else 0.0   # 13:00-13:30
    # Pre/post release timing (NFP/CPI/PPI la 8:30 AM ET = 13:30 UTC = 810 min)
    # FOMC la 14:00 ET = 19:00 UTC → toată fereastra NY quality gate e pre-anunț
    f["is_pre_nfp"]   = 1.0 if (f["is_nfp_day"] > 0 and _min_abs < 810) else 0.0
    f["is_post_nfp"]  = 1.0 if (f["is_nfp_day"] > 0 and _min_abs >= 810) else 0.0
    f["is_pre_cpi"]   = 1.0 if (f["is_cpi_day"] > 0 and _min_abs < 810) else 0.0
    f["is_post_cpi"]  = 1.0 if (f["is_cpi_day"] > 0 and _min_abs >= 810) else 0.0
    f["is_pre_ppi"]   = 1.0 if (f["is_ppi_day"] > 0 and _min_abs < 810) else 0.0
    f["is_post_ppi"]  = 1.0 if (f["is_ppi_day"] > 0 and _min_abs >= 810) else 0.0
    f["is_fomc_wait"] = f["is_fomc_day"]  # FOMC la 19:00 UTC → toată fereastra NY e pre-anunț

    # ── Interaction features NY (with real calendar)
    vol_r = f.get("vol_regime", 1.0)
    f["news_x_dir"]         = f["is_news_day"]  * f["dir_short"]
    f["swept_lon_x_dir"]    = f["swept_lon_hi"] if dir_num == -1 else f["swept_lon_lo"]
    f["fomc_x_weekly_prem"] = f["is_fomc_day"]  * f.get("weekly_premium_pct", 0.0)
    f["drive_x_swept"]      = f["drive_aligned_dir"] * f["swept_lon_x_dir"]
    f["vol_x_fomc"]         = vol_r * f["is_fomc_day"]
    f["vol_x_nfp"]          = vol_r * f["is_nfp_day"]
    f["sweep_depth_lon_atr"]= abs(entry - (lon_hi if dire == "SHORT" else lon_lo)) / atr
    f["deep_sweep_lon"]     = 1.0 if f["sweep_depth_lon_atr"] > 1.5 else 0.0
    f["ny_open_outside_lon"]= 1.0 if not ny_open_in_lon else 0.0
    f["fomc_x_fvg_1h"]      = f["is_fomc_day"] * f.get("fvg_aligned_1h", 0.0)
    f["fomc_x_open"]        = f["is_fomc_day"] * f["is_ny_open"]
    f["news_x_open"]        = f["is_news_day"] * f["is_ny_open"]
    f["nfp_pre_x_open"]     = f["is_pre_nfp"]  * f["is_ny_open"]
    f["nfp_post_x_open"]    = f["is_post_nfp"] * f["is_ny_open"]
    f["lon_dir_x_vol"]      = f["lon_dir_explicit"] * vol_r / 3.0
    f["lon_dir_x_fvg_1h"]   = f["lon_dir_explicit"] * f.get("fvg_aligned_1h", 0.0) / 2.0
    f["drive_x_fvg_15m"]    = f["drive_aligned_dir"] * f.get("fvg_aligned_15m", 0.0)
    f["triple_x_fvg_1h"]    = f["triple_sess_aligned"] * f.get("fvg_aligned_1h", 0.0)
    f["news_x_vol"]         = f["is_news_day"] * vol_r

    return f


def _add_ts_lon_features(f: dict, sig: dict, db: dict, atr: float) -> dict:
    """Adaugă cele 17 TS-specific features pentru LON."""
    entry   = _sf(sig.get("entry", 0))
    dire    = sig.get("direction", "LONG")
    dir_num = -1 if dire == "SHORT" else 1
    atr     = max(atr, 1.0)

    ah = _sf(db.get("asia_hi", entry)); al = _sf(db.get("asia_lo", entry))
    swept_lvl = ah if dire == "SHORT" else al
    asia_rng  = max(ah - al, 0.01)

    h4h = _sf(db.get("h4_hi", 0)); h4l = _sf(db.get("h4_lo", 0))
    h1h = _sf(db.get("h1_hi", 0)); h1l = _sf(db.get("h1_lo", 0))
    h4b = (1 if entry > (h4h + h4l) / 2 else -1) if h4h > 0 else 0
    h1b = (1 if entry > (h1h + h1l) / 2 else -1) if h1h > 0 else 0

    bar_o = _sf(db.get("open", entry)); bar_c = _sf(db.get("close", entry))
    bar_h = _sf(db.get("high", entry)); bar_l = _sf(db.get("low",  entry))
    bar_rng = max(bar_h - bar_l, 0.01)
    body    = bar_c - bar_o

    # Sweep wick: wick che ha superato il livello
    sweep_wick = (bar_h - max(bar_c, bar_o)) if dire == "SHORT" else (min(bar_c, bar_o) - bar_l)
    opp_dist   = abs(entry - (al if dire == "SHORT" else ah))

    f["ts_sweep_depth_pts"] = abs(entry - swept_lvl)
    f["ts_sweep_depth_atr"] = abs(entry - swept_lvl) / atr
    f["ts_close_inside"]    = 1.0 if (dire == "SHORT" and bar_c < ah) or (dire == "LONG" and bar_c > al) else 0.0
    f["ts_rejection_str"]   = max(sweep_wick, 0.0) / atr
    f["ts_wick_pct"]        = max(sweep_wick, 0.0) / bar_rng
    f["ts_body_pct"]        = abs(body) / bar_rng
    f["ts_close_quality"]   = 1.0 if f["ts_close_inside"] and abs(body) / bar_rng > 0.4 else 0.0
    f["ts_wick_dom"]        = 1.0 if max(sweep_wick, 0.0) > abs(body) else 0.0
    f["ts_asia_rng_atr"]    = asia_rng / atr
    f["ts_sweep_pct_asia"]  = abs(entry - swept_lvl) / max(asia_rng, 0.01)
    f["ts_opp_dist_atr"]    = opp_dist / atr
    rr = _sf(sig.get("rr", 2.0))
    f["ts_rr_impl"]         = rr
    f["ts_sharp"]           = 1.0 if rr > 2.5 and f["ts_close_inside"] else 0.0
    # entry proximity to swept level
    f["ts_entry_prox"]      = 1.0 if abs(entry - swept_lvl) / atr < 0.5 else 0.0
    # near weekly extreme
    lwh = _sf(db.get("lw_hi", entry)); lwl = _sf(db.get("lw_lo", entry))
    near_wk = (abs(swept_lvl - lwh) / atr < 0.5) or (abs(swept_lvl - lwl) / atr < 0.5)
    f["ts_near_wk_extreme"] = 1.0 if near_wk else 0.0
    # anti-HTF (direction opposta al bias → contrarian, più pericoloso = 0)
    f["ts_htf_anti"]        = 1.0 if (h4b == dir_num or h1b == dir_num) else 0.0
    f["ts_combo_score"]     = float(np.clip(
        f["ts_close_quality"] * 0.3 +
        f["ts_htf_anti"] * 0.25 +
        f["ts_sweep_depth_atr"] / 3.0 * 0.25 +
        f["ts_entry_prox"] * 0.2,
        0.0, 1.0
    ))
    return f


def _add_ts_ny_features(f: dict, sig: dict, db: dict, atr: float) -> dict:
    """Adaugă cele 25 TS-specific features pentru NY (swept levels = LON hi/lo)."""
    entry   = _sf(sig.get("entry", 0))
    dire    = sig.get("direction", "LONG")
    dir_num = -1 if dire == "SHORT" else 1
    atr     = max(atr, 1.0)

    lon_hi = _sf(db.get("lon_hi", entry)); lon_lo = _sf(db.get("lon_lo", entry))
    lon_rng   = max(lon_hi - lon_lo, 0.01)
    swept_lvl = lon_hi if dire == "SHORT" else lon_lo

    ah = _sf(db.get("asia_hi", entry)); al = _sf(db.get("asia_lo", entry))
    asia_mid = (ah + al) / 2 if ah > 0 else entry

    h4h = _sf(db.get("h4_hi", 0)); h4l = _sf(db.get("h4_lo", 0))
    h1h = _sf(db.get("h1_hi", 0)); h1l = _sf(db.get("h1_lo", 0))
    h4b = (1 if entry > (h4h + h4l) / 2 else -1) if h4h > 0 else 0
    h1b = (1 if entry > (h1h + h1l) / 2 else -1) if h1h > 0 else 0

    bar_o = _sf(db.get("open", entry)); bar_c = _sf(db.get("close", entry))
    bar_h = _sf(db.get("high", entry)); bar_l = _sf(db.get("low",  entry))
    bar_rng = max(bar_h - bar_l, 0.01)
    body    = bar_c - bar_o
    sweep_wick = (bar_h - max(bar_c, bar_o)) if dire == "SHORT" else (min(bar_c, bar_o) - bar_l)

    # TS features identice cu LON dar bazate pe LON levels
    f["ts_sweep_depth_pts"] = abs(entry - swept_lvl)
    f["ts_sweep_depth_atr"] = abs(entry - swept_lvl) / atr
    f["ts_close_inside"]    = 1.0 if (dire == "SHORT" and bar_c < lon_hi) or (dire == "LONG" and bar_c > lon_lo) else 0.0
    f["ts_rejection_str"]   = max(sweep_wick, 0.0) / atr
    f["ts_wick_pct"]        = max(sweep_wick, 0.0) / bar_rng
    f["ts_body_pct"]        = abs(body) / bar_rng
    f["ts_close_quality"]   = 1.0 if f["ts_close_inside"] and abs(body) / bar_rng > 0.4 else 0.0
    f["ts_wick_dom"]        = 1.0 if max(sweep_wick, 0.0) > abs(body) else 0.0
    f["ts_lon_rng_atr"]     = lon_rng / atr
    f["ts_sweep_pct_lon"]   = abs(entry - swept_lvl) / max(lon_rng, 0.01)
    opp_dist = abs(entry - (lon_lo if dire == "SHORT" else lon_hi))
    f["ts_opp_dist_atr"]    = opp_dist / atr
    rr = _sf(sig.get("rr", 2.0))
    f["ts_rr_impl"]         = rr
    f["ts_sharp"]           = 1.0 if rr > 2.5 and f["ts_close_inside"] else 0.0
    f["ts_entry_prox"]      = 1.0 if abs(entry - swept_lvl) / atr < 0.5 else 0.0
    lwh = _sf(db.get("lw_hi", entry)); lwl = _sf(db.get("lw_lo", entry))
    near_wk = (abs(swept_lvl - lwh) / atr < 0.5) or (abs(swept_lvl - lwl) / atr < 0.5)
    f["ts_near_wk_extreme"] = 1.0 if near_wk else 0.0
    f["ts_htf_anti"]        = 1.0 if (h4b == dir_num or h1b == dir_num) else 0.0
    f["ts_combo_score"]     = float(np.clip(
        f["ts_close_quality"] * 0.3 +
        f["ts_htf_anti"] * 0.25 +
        f["ts_sweep_depth_atr"] / 3.0 * 0.25 +
        f["ts_entry_prox"] * 0.2,
        0.0, 1.0
    ))
    f["ts_lon_mid_dist"]    = (entry - (lon_hi + lon_lo) / 2) / atr if lon_hi > 0 else 0.0
    f["ts_lon_above_asia"]  = 1.0 if lon_hi > asia_mid else 0.0

    # Extra NY interaction features
    f["lon_hi_x_dir"] = lon_hi * f.get("dir_short", 0.0)
    f["lon_lo_x_dir"] = lon_lo * (1.0 - f.get("dir_short", 0.0))
    vol_r = f.get("vol_regime", 1.0)
    f["vol_x_lon_sweep"] = vol_r * f["ts_sweep_depth_atr"]
    f["htf_x_lon_aligned"]= float(h4b == dir_num) * f.get("triple_sess_aligned", 0.0)

    return f


# ─── Feature vector → numpy array ────────────────────────────────────────────

def _to_array(feat_dict: dict, feature_list: list) -> np.ndarray:
    """Convertește dict de features în numpy array în ordinea corectă."""
    return np.array([feat_dict.get(k, 0.0) for k in feature_list], dtype=np.float32).reshape(1, -1)


def _predict(model_name: str, feat_dict: dict, regime: str = None) -> float:
    """Score calibrat [0, 1] din modelul dat.
    Dacă regime e specificat și există un model regim-specific, îl folosește.
    Fallback: modelul ALL (default)."""
    fl = _FEATURES.get(model_name)
    if fl is None:
        return 0.5

    # Încearcă modelul regim-specific
    m = None
    if regime and regime != 'UNKNOWN':
        regime_key = f"{model_name}_{regime}"
        m = _REGIME_MODELS.get(regime_key)
        if m is not None:
            logger.debug(f"_predict: using regime model {regime_key}")

    # Fallback la modelul ALL
    if m is None:
        m = _MODELS.get(model_name)
    if m is None:
        return 0.5

    try:
        X = _to_array(feat_dict, fl)
        clf = m if hasattr(m, "predict_proba") else m.get("model", m)
        return float(clf.predict_proba(X)[0, 1])
    except Exception as e:
        logger.warning(f"_predict({model_name}): {e}")
        return 0.5


# ─── Main entry point ─────────────────────────────────────────────────────────

def score_quality(signal: dict, now_utc: datetime,
                  conn: Optional[sqlite3.Connection] = None) -> dict:
    """
    Calculează quality score pentru un semnal.

    Args:
        signal: dict cu direction, entry, sl_pt, tp_pt, rr, session, confluence_score
        now_utc: timestamp UTC al semnalului
        conn: conexiune SQLite la mario_trading.db (opțional, deschisă dacă None)

    Returns:
        dict: {
            "score": float,          # scor final [0, 1]
            "score_v6": float,       # scor v6/ny_v3 de bază
            "score_ts": float,       # scor ts model (0.0 dacă nu e v5 sau non-reversal)
            "passed": bool,          # True dacă trece threshold-ul
            "threshold": float,      # threshold aplicat
            "model": str,            # "v2" sau "v5"
            "session": str,          # "LON" sau "NY"
        }
    """
    _load_models()
    cfg     = load_config()
    model   = cfg.get("model", "v2")
    enabled = cfg.get("enabled", True)
    session = signal.get("session", "LON").upper()
    lon_thr = cfg.get("lon_thr", 0.20)
    ny_thr  = cfg.get("ny_thr",  0.20)
    thr     = lon_thr if session == "LON" else ny_thr

    if not enabled:
        return {"score": 1.0, "score_v6": 1.0, "score_ts": 0.0,
                "passed": True, "threshold": thr, "model": model, "session": session}

    # Deschide conexiune DB dacă nu există
    _own_conn = conn is None
    if _own_conn:
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=5)
        except Exception as e:
            logger.warning(f"score_quality: DB open error: {e}")
            return {"score": 0.5, "score_v6": 0.5, "score_ts": 0.0,
                    "passed": True, "threshold": thr, "model": model, "session": session}

    try:
        now_date = now_utc.strftime("%Y-%m-%d")
        db_row   = _get_db_row(conn, now_utc)
        reg      = _get_rolling_regime(conn, now_date)
        wk       = _get_week_extremes(conn, now_utc)
        mtf      = _compute_mtf(conn, now_utc, _sf(signal.get("entry", 0)),
                                 max(_sf(db_row.get("atr_14", 20.0), 20.0), 1.0),
                                 signal.get("direction", "LONG"))
        roll_wr  = _get_rolling_wr(session, now_date)

        atr = max(_sf(db_row.get("atr_14", 20.0), 20.0), 1.0)

        # Regimul curent (din signal sau fallback UNKNOWN)
        current_regime = signal.get("regime", "UNKNOWN")
        # Normalizare: acceptăm și variante scurte
        if current_regime in ("PRE_EXP", "pre_expansion"): current_regime = "PRE_EXPANSION"
        elif current_regime in ("EXP", "expansion"): current_regime = "EXPANSION"
        elif current_regime in ("RET", "retracement"): current_regime = "RETRACEMENT"
        logger.debug(f"QualityGate: regime={current_regime}")

        if session == "LON":
            feat_base = _build_v6(signal, db_row, mtf, reg, wk, roll_wr, now_utc)
            qs_base   = _predict("v6", feat_base, regime=current_regime)

            if model == "v5":
                is_ts_reversal = ("REV" in str(current_regime).upper() or
                                  signal.get("is_reversal", True))
                if is_ts_reversal:
                    feat_ts = dict(feat_base)
                    feat_ts = _add_ts_lon_features(feat_ts, signal, db_row, atr)
                    qs_ts   = _predict("ts_lon", feat_ts, regime=current_regime)
                    qs_final= max(qs_base, qs_ts)
                else:
                    qs_ts    = 0.0
                    qs_final = qs_base
            else:
                qs_ts    = 0.0
                qs_final = qs_base

        else:  # NY
            feat_base = _build_ny_v3(signal, db_row, mtf, reg, wk, roll_wr, now_utc, conn)
            qs_base   = _predict("ny_v3", feat_base, regime=current_regime)

            if model == "v5":
                is_ts_reversal = ("REV" in str(current_regime).upper() or
                                  signal.get("is_reversal", True))
                if is_ts_reversal:
                    feat_ts = dict(feat_base)
                    feat_ts = _add_ts_ny_features(feat_ts, signal, db_row, atr)
                    qs_ts   = _predict("ts_ny", feat_ts, regime=current_regime)
                    qs_final= max(qs_base, qs_ts)
                else:
                    qs_ts    = 0.0
                    qs_final = qs_base
            else:
                qs_ts    = 0.0
                qs_final = qs_base

        passed = qs_final >= thr
        logger.info(
            f"QualityGate {session} | model={model} | "
            f"base={qs_base:.3f} ts={qs_ts:.3f} final={qs_final:.3f} thr={thr} → {'✅' if passed else '❌'}"
        )
        return {
            "score":    qs_final,
            "score_v6": qs_base,
            "score_ts": qs_ts,
            "passed":   passed,
            "threshold":thr,
            "model":    model,
            "session":  session,
        }

    except Exception as e:
        logger.error(f"score_quality error: {e}", exc_info=True)
        return {"score": 0.5, "score_v6": 0.5, "score_ts": 0.0,
                "passed": True, "threshold": thr, "model": model, "session": session}
    finally:
        if _own_conn and conn:
            conn.close()
