"""
ict_gate_v3.py — ICT Setup Gate v3  (integrat în bridge_api.py)
================================================================
Verifică condiţii ICT setup_v3 în timp real înainte de orice trade.

Reguli:
  LON_SHORT : sweep asian high (pre-LON) + displacement bearish + FVG + retrace + 15m bearish
  LON_LONG  : sweep asian low  (pre-LON) + displacement bullish + FVG + retrace + 15m bullish + asia_sweep
  NY_SHORT  : sweep London high (NY open) + displacement bearish + FVG + retrace + 15m bearish
  NY_LONG   : sweep London low  (NY open) + displacement bullish + FVG + retrace + 15m bullish

Filtre propfirm (v3.1 — optimizat payout-speed):
  - SL ≤ 12pt   (risc max $240/trade pe 1 contract NQ)
  - Max 2 trades per zi (ambele dacă setup valid)
  - Min RR 1.5

Backtest OOS 2024-2025: 18 payouturi în 3 ani, NEVER BLOWN
  2nd payout în ~100 zile, net=$32,251 ✅
"""

import sqlite3
import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger("ict_gate_v3")

# ─────────────────────────────────────────────────────────────────────────────
# QUALITY GATE v1  (înlocuiește ict_setup_scorer_v4.1)
# Modele: mario_quality_v6 + ny_v3 + ts_lon_v1 + ts_ny_v1
# Config: model_config.json → selectabil v2 (v6+ny_v3) sau v5 (ensemble TS)
# ─────────────────────────────────────────────────────────────────────────────

_QUALITY_GATE_OK = False
try:
    import quality_gate_live as _qg
    _QUALITY_GATE_OK = True
    logger.info("✅ quality_gate_live importat cu succes")
except Exception as _qg_err:
    logger.warning(f"quality_gate_live import error: {_qg_err} — fallback la scorer v4")
    _qg = None  # type: ignore

# Fallback: scorer vechi v4.1 (păstrat ca backup)
ML_BLOCK_THRESHOLD = 0.35
_SCORER = None

def _load_scorer():
    """Încarcă modelul ML scorer v4 (lazy, singleton). Returnează None dacă fișierul lipsește."""
    global _SCORER
    if _SCORER is not None:
        return _SCORER
    try:
        import pickle
        scorer_path = Path(__file__).parent / "ict_setup_scorer_v4_1.pkl"
        if not scorer_path.exists():
            # fallback la v4
            scorer_path = Path(__file__).parent / "ict_setup_scorer_v4.pkl"
        if not scorer_path.exists():
            # fallback la v3
            scorer_path = Path(__file__).parent / "ict_setup_scorer_v3.pkl"
        if not scorer_path.exists():
            logger.warning(f"ML scorer v4.1 nu există — scorer dezactivat")
            return None
        with open(scorer_path, "rb") as f:
            _SCORER = pickle.load(f)
        ver = "v4.1" if "v4_1" in str(scorer_path) else "v4"
        logger.info(f"✅ ML scorer {ver} încărcat ({len(_SCORER.get('features',[]))} features, "
                    f"WF_AUC={_SCORER.get('wf_auc_v41', _SCORER.get('wf_auc_v4', 0)):.4f}, "
                    f"block_thr={ML_BLOCK_THRESHOLD})")
        return _SCORER
    except Exception as e:
        logger.warning(f"ML scorer eroare încărcare: {e}")
        return None


def _load_daily_conditions(date_str: str) -> dict:
    """
    Citește condițiile de piață (Tier 1-4 + v3) din market_conditions.db pentru ziua curentă.
    Returnează dict gol dacă DB-ul nu e disponibil sau data nu e în el.
    """
    try:
        cond_db = Path(__file__).parent / "market_conditions.db"
        if not cond_db.exists():
            return {}
        con = sqlite3.connect(f'file:{cond_db}?mode=ro', uri=True, timeout=10)
        row = pd.read_sql(
            f"SELECT * FROM conditions WHERE date = '{date_str}' LIMIT 1", con
        )
        con.close()
        if len(row) == 0:
            return {}
        return row.iloc[0].to_dict()
    except Exception as e:
        logger.debug(f"market_conditions.db read error: {e}")
        return {}


def _check_rule_signals(signal: dict, cond: dict, df_today: pd.DataFrame,
                        atr: float, now_utc: "datetime") -> dict:
    """
    Evaluează pattern-uri rule-based (nu ML) pentru contextul tradeului.
    Returnează dict cu semnale și un 'confluence_score' agregat (0.0–1.0).

    Semnale verificate:
      TP confluences  — câte niveluri cheie coincid cu TP-ul
      Turtle Soup     — pattern 2AM→6AM→10AM confirmat
      Big Wick        — wick masiv în sesiunea NY anterioară
      Weekly Profile  — profilul săptămânal aliniat cu tradeul
      15m Trend       — trendul pe 15m aliniat cu direcția
      VWAP            — entry sub/deasupra VWAP corect
    """
    def _cv(key, default=0.0):
        v = cond.get(key, default)
        try:
            return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default
        except:
            return default

    direction = signal.get("direction", "LONG")
    dir_num   = 1 if direction == "LONG" else -1
    entry     = float(signal.get("entry", 0))
    tp_price  = float(signal.get("tp",    0))

    # ── 1. TP Confluence check ─────────────────────────────────────────────────
    def tp_near(level, thr=12.0):
        return int(level > 0 and abs(tp_price - level) <= thr)

    tp_lvls = {
        "POC":       tp_near(_cv("poc_lvl")),
        "VAH":       tp_near(_cv("vah")),
        "VAL":       tp_near(_cv("val")),
        "PDH":       tp_near(_cv("p_hi")),
        "PDL":       tp_near(_cv("p_lo")),
        "PWH":       tp_near(_cv("lw_hi")),
        "PWL":       tp_near(_cv("lw_lo")),
        "LonHi":     tp_near(_cv("lon_hi_day")),
        "LonLo":     tp_near(_cv("lon_lo_day")),
        "NyHi":      tp_near(_cv("ny_hi_day")),
        "NyLo":      tp_near(_cv("ny_lo_day")),
    }
    tp_hits   = [k for k, v in tp_lvls.items() if v]
    n_tp_conf = len(tp_hits)

    # ── 2. Turtle Soup pattern ─────────────────────────────────────────────────
    ts_bull = int(_cv("ts_pattern_bull"))  # ieri/azi: 2AM bear + 6AM sweep low + 10AM push sus
    ts_bear = int(_cv("ts_pattern_bear"))
    ts_aligned = ts_bull if dir_num == 1 else ts_bear

    c6am_sweep_bull = int(_cv("c6am_sweep_2am_bull"))
    c6am_sweep_bear = int(_cv("c6am_sweep_2am_bear"))
    c6am_sweep_aligned = c6am_sweep_bull if dir_num == 1 else c6am_sweep_bear

    # ── 3. Big Wick semnale NY ─────────────────────────────────────────────────
    bw_bull = int(_cv("big_wick_bull_ny"))  # câte big wick bull azi în NY
    bw_bear = int(_cv("big_wick_bear_ny"))
    bw_aligned = min(bw_bull, 1) if dir_num == 1 else min(bw_bear, 1)

    # ── 4. Weekly Profile ─────────────────────────────────────────────────────
    weekly_profile  = int(_cv("weekly_profile", 0))
    weekly_day_rank = int(_cv("weekly_day_rank", 3))
    monday_is_low   = int(_cv("monday_is_low", 0))
    tuesday_extreme = int(_cv("tuesday_extreme", 0))
    seek_destroy_b  = int(_cv("seek_destroy_bull", 0))
    seek_destroy_s  = int(_cv("seek_destroy_bear", 0))
    seek_aligned    = seek_destroy_b if dir_num == 1 else seek_destroy_s

    # Profile names
    profile_names = {0:"unknown",1:"TueLow",2:"TueHigh",3:"WedLow",4:"WedHigh",
                     5:"ThuBullRev",6:"ThuBearRev",7:"SeekDestBull",8:"SeekDestBear"}
    profile_str = profile_names.get(weekly_profile, "?")

    # Profile aliniat cu tradeul
    bull_profiles = {1, 3, 5, 7}  # profiluri bullish
    bear_profiles = {2, 4, 6, 8}  # profiluri bearish
    profile_aligned = int(
        (dir_num == 1 and weekly_profile in bull_profiles) or
        (dir_num == -1 and weekly_profile in bear_profiles)
    )

    # ── 5. 15m trend din df_today ─────────────────────────────────────────────
    try:
        hhmm_now = int(now_utc.strftime("%H%M"))
        recent_bars = df_today[df_today['hhmm'] <= hhmm_now].tail(30)
        if len(recent_bars) >= 15:
            atr_safe = max(atr, 1.0)
            slope = float(recent_bars['close'].iloc[-1] - recent_bars['close'].iloc[0])
            if slope > atr_safe * 0.3:
                trend_15m = 1
            elif slope < -atr_safe * 0.3:
                trend_15m = -1
            else:
                trend_15m = 0
        else:
            trend_15m = 0
    except:
        trend_15m = 0

    trend_15m_aligned = int(trend_15m == dir_num)

    # ── 6. VWAP alignment ─────────────────────────────────────────────────────
    try:
        vwap_live = float(df_today.iloc[-1].get("vwap", 0)) if len(df_today) > 0 else 0
        vwap_dist_pt = abs(entry - vwap_live) if vwap_live > 0 else 0
        above_vwap   = int(entry > vwap_live) if vwap_live > 0 else 0
        vwap_ok = int(
            (dir_num == 1 and above_vwap == 0) or  # LONG sub VWAP = discount
            (dir_num == -1 and above_vwap == 1)     # SHORT deasupra VWAP = premium
        )
    except:
        vwap_dist_pt = 0; above_vwap = 0; vwap_ok = 0

    # ── Confluence Score ──────────────────────────────────────────────────────
    # Numără câte semnale pozitive
    positives = [
        min(n_tp_conf, 2),    # max 2pt pentru TP confluences
        ts_aligned,            # 1pt dacă TS pattern confirmat
        c6am_sweep_aligned,    # 1pt dacă 6AM sweep confirmat
        bw_aligned,            # 1pt dacă big wick aliniat
        profile_aligned,       # 1pt dacă weekly profile aliniat
        seek_aligned,          # 1pt dacă seek&destroy
        trend_15m_aligned,     # 1pt dacă 15m trend aliniat
        vwap_ok,               # 1pt dacă VWAP corect
    ]
    max_pts = 9.0
    raw_score = sum(positives)
    confluence_score = round(raw_score / max_pts, 3)

    return {
        "n_tp_confluences": n_tp_conf,
        "tp_hits": tp_hits,
        "ts_aligned": ts_aligned,
        "c6am_sweep_aligned": c6am_sweep_aligned,
        "bw_aligned": bw_aligned,
        "profile_str": profile_str,
        "profile_aligned": profile_aligned,
        "seek_aligned": seek_aligned,
        "trend_15m": trend_15m,
        "trend_15m_aligned": trend_15m_aligned,
        "vwap_ok": vwap_ok,
        "vwap_dist_pt": vwap_dist_pt,
        "confluence_score": confluence_score,
        "raw_pts": raw_score,
        "monday_is_low": monday_is_low,
        "tuesday_extreme": tuesday_extreme,
        "weekly_day_rank": weekly_day_rank,
    }


def _score_setup(signal: dict, df_today: pd.DataFrame, pre_bars: pd.DataFrame,
                 vah: float, val: float, poc: float,
                 atr: float, hhmm: int, now_utc: datetime) -> float:
    """
    Calculează scorul ML (0.0–1.0) pentru un setup ICT valid.
    Score ≥ 0.35 → WR estimat ~57%  |  Score ≥ 0.40 → WR estimat ~61%
    Returnează 0.0 la eroare (scorer absent sau date insuficiente).
    """
    scorer = _load_scorer()
    if scorer is None:
        return 0.0

    try:
        direction = signal["direction"]
        tp_pt     = float(signal["tp_pt"])
        sl_pt     = float(signal["sl_pt"])
        rr        = float(signal["rr"])
        entry     = float(signal["entry"])
        dir_num   = 1 if direction == "LONG" else -1
        atr_safe  = max(atr, 1.0)

        # Bara curentă (ultima bară disponibilă)
        cb = df_today.iloc[-1]

        def _fv(col, default=0.0):
            """Safe float extragere din Series row."""
            v = cb.get(col, default)
            try:
                v = float(v)
                return v if not np.isnan(v) else default
            except Exception:
                return default

        # ── VP / distance features ────────────────────────────────────────────
        val_live     = _fv("val", val)
        vah_live     = _fv("vah", vah)
        poc_live     = _fv("poc_level", poc)
        dist_val_pt  = entry - val_live
        dist_vah_pt  = entry - vah_live
        dist_val_atr = dist_val_pt  / atr_safe
        dist_vah_atr = dist_vah_pt  / atr_safe
        dist_poc_atr = abs(entry - poc_live) / atr_safe
        near_poc     = 1 if dist_poc_atr < 0.5 else 0
        inside_va    = int(_fv("inside_va", 0))
        outside_va   = 1 - inside_va

        dist_vwap      = _fv("dist_vwap", 0.0)
        dist_vwap_norm = dist_vwap / atr_safe

        # ── Pre-London range features ─────────────────────────────────────────
        if len(pre_bars) >= 3:
            pre_hi            = float(pre_bars["high"].max())
            pre_lo            = float(pre_bars["low"].min())
            pre_rng           = max(pre_hi - pre_lo, 0.01)
            pre_lon_pos       = (entry - pre_lo) / pre_rng
            pre_lon_range_atr = pre_rng / atr_safe
        else:
            pre_lon_pos       = 0.5
            pre_lon_range_atr = 1.0
        sweep_extreme = 1 if (pre_lon_pos > 0.8 or pre_lon_pos < 0.2) else 0

        # ── TP features ───────────────────────────────────────────────────────
        tp_atr_ratio = tp_pt / atr_safe
        tp_reachable = 1 if tp_pt < atr_safe * 3 else 0
        tp_in_2atr   = 1 if tp_atr_ratio < 2.0 else 0
        sl_atr_ratio = sl_pt / atr_safe

        pdh = _fv("p_hi", 0.0)
        pdl = _fv("p_lo", 0.0)
        if direction == "LONG" and pdh > entry and (pdh - entry) < tp_pt:
            tp_at_pd_level = 1
        elif direction == "SHORT" and pdl < entry and (entry - pdl) < tp_pt:
            tp_at_pd_level = 1
        else:
            tp_at_pd_level = 0

        # ── Bias / alignment ──────────────────────────────────────────────────
        h4_hi = _fv("h4_hi", 0.0);  h4_lo = _fv("h4_lo", 0.0)
        h1_hi = _fv("h1_hi", 0.0);  h1_lo = _fv("h1_lo", 0.0)
        h4_bias = (1 if entry > (h4_hi + h4_lo) / 2 else -1) if h4_hi > 0 else 0
        h1_bias = (1 if entry > (h1_hi + h1_lo) / 2 else -1) if h1_hi > 0 else 0
        h4_aligned  = 1 if h4_bias == dir_num else 0
        htf_aligned = 1 if (h4_bias == h1_bias and h4_bias != 0) else 0

        # ── Premium / Discount ────────────────────────────────────────────────
        if   direction == "LONG"  and dist_val_pt < 0: premium_discount =  1
        elif direction == "SHORT" and dist_vah_pt > 0: premium_discount = -1
        else:                                           premium_discount =  0

        # ── VWAP alignment ────────────────────────────────────────────────────
        vwap_aligned = 1 if (direction == "LONG"  and dist_vwap < 0) or \
                            (direction == "SHORT" and dist_vwap > 0) else 0

        # ── VA width ─────────────────────────────────────────────────────────
        va_width_atr = (abs(dist_vah_pt) + abs(dist_val_pt)) / atr_safe

        # ── ICT setup quality ─────────────────────────────────────────────────
        fvg_up   = int(_fv("fvg_up",   0))
        fvg_down = int(_fv("fvg_down",  0))
        has_disp = int(_fv("has_displacement", 1))
        full_setup = has_disp * min(fvg_up + fvg_down, 1)

        # ── Bar features ──────────────────────────────────────────────────────
        bar_hi        = _fv("high", entry);  bar_lo = _fv("low", entry)
        bar_open      = _fv("open", entry);  bar_close = _fv("close", entry)
        bar_range_atr = (bar_hi - bar_lo) / atr_safe
        body_dir_val  = 1 if bar_close > bar_open else -1
        body_with_trade = float(body_dir_val * dir_num)

        # ── Market regime ─────────────────────────────────────────────────────
        adx    = _fv("adx_14",           25.0)
        hurst  = _fv("hurst",            0.52)
        fisher = _fv("fisher_transform", 0.0)
        acf1   = _fv("acf_lag1",         0.0)
        adx_strong     = 1 if adx   > 25  else 0
        adx_weak       = 1 if adx   < 18  else 0
        hurst_trending = 1 if hurst > 0.52 else 0
        fisher_extreme = 1 if abs(fisher) > 2.0 else 0
        acf_trending   = 1 if acf1  > 0.3 else 0

        # ── Calendar ─────────────────────────────────────────────────────────
        dow       = now_utc.weekday()
        is_monday = 1 if dow == 0 else 0
        is_friday = 1 if dow == 4 else 0
        is_tuesday  = 1 if dow == 1 else 0
        is_thursday = 1 if dow == 3 else 0

        # ── Timing ───────────────────────────────────────────────────────────
        in_first_15   = 1 if (800 <= hhmm <= 815) or (1330 <= hhmm <= 1345) else 0
        in_open30     = 1 if (800 <= hhmm <= 830) or (1330 <= hhmm <= 1400) else 0
        cur_min       = (hhmm // 100) * 60 + (hhmm % 100)
        lon_open_min  = 8 * 60
        min_since_lon = max(0, cur_min - lon_open_min)
        early_lon     = 1 if min_since_lon <= 30 else 0
        late_lon      = 1 if min_since_lon >= 90 else 0

        # ── Session prev-close features (v3 new) ─────────────────────────────
        # prev NY close și LON close din pre_bars/df_today
        try:
            ny_bars  = df_today[df_today.get("hour_min", pd.Series()).between("19:00","20:00")] \
                       if "hour_min" in df_today.columns else pd.DataFrame()
            lon_bars_p = df_today[df_today.get("hour_min", pd.Series()).between("11:00","12:00")] \
                         if "hour_min" in df_today.columns else pd.DataFrame()
            pny_close  = float(ny_bars["close"].iloc[-1])  if len(ny_bars)  else entry
            plon_close = float(lon_bars_p["close"].iloc[-1]) if len(lon_bars_p) else entry
        except Exception:
            pny_close = entry; plon_close = entry

        sweep_dist_pny_close_atr  = abs(entry - pny_close)  / atr_safe
        sweep_dist_plon_close_atr = abs(entry - plon_close) / atr_safe
        sweep_near_pny_close      = 1 if sweep_dist_pny_close_atr  <= 2.0 else 0
        sweep_near_plon_close     = 1 if abs(entry - plon_close)   <= 10  else 0

        pdh = _fv("p_hi", 0.0); pdl = _fv("p_lo", 0.0)
        sweep_dist_pdh_atr = abs(entry - pdh) / atr_safe if pdh > 0 else 5.0
        sweep_dist_pdl_atr = abs(entry - pdl) / atr_safe if pdl > 0 else 5.0
        sweep_near_pd       = 1 if min(sweep_dist_pdh_atr, sweep_dist_pdl_atr) <= 1.5 else 0
        sweep_near_pd_open30 = 1 if sweep_near_pd and in_open30 else 0
        prev_ny_push_dir    = 0  # neutru implicit (nu avem date prev-day live ușor)

        # ── Tier 1-4: condiții noi din market_conditions.db ──────────────────
        date_str = now_utc.strftime("%Y-%m-%d")
        cond     = _load_daily_conditions(date_str)

        def _cv(key, default=0.0):
            v = cond.get(key, default)
            try: return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default
            except: return default

        # Tier 1: HTF
        lw_hi = _cv("lw_hi", 0.0); lw_lo = _cv("lw_lo", 0.0)
        lw_rng = max(lw_hi - lw_lo, 1.0)
        weekly_premium_pct   = np.clip((entry - lw_lo) / lw_rng, 0, 1) if lw_hi > 0 else 0.5
        weekly_prem_direction = float(np.clip(
            (0.5 - weekly_premium_pct) if dir_num == 1 else (weekly_premium_pct - 0.5),
            -0.5, 0.5
        ))
        h4_structure     = _cv("h4_structure",  0.0)
        h4_struct_aligned= float(np.clip(h4_structure * dir_num, -1, 1))
        daily_bias       = _cv("daily_bias",    0.0)
        daily_bias_aligned = float(np.clip(daily_bias * dir_num, -1, 1))
        h4_x_daily       = h4_structure * daily_bias

        # Tier 2: ICT microstructure
        ob_bull_active  = _cv("ob_bull_active", 0)
        ob_bear_active  = _cv("ob_bear_active", 0)
        ob_aligned      = ob_bull_active if dir_num == 1 else ob_bear_active
        eq_highs_above  = _cv("eq_highs_above", 0)
        eq_lows_below   = _cv("eq_lows_below",  0)
        eq_sweep_aligned= eq_lows_below if dir_num == 1 else eq_highs_above
        fvg_active_bull = _cv("fvg_active_bull", 0)
        fvg_active_bear = _cv("fvg_active_bear", 0)
        fvg_aligned     = fvg_active_bull if dir_num == 1 else fvg_active_bear

        # Tier 3: session character
        day_type_str    = str(cond.get("day_type", "normal"))
        day_type_trend  = 1 if day_type_str == "trend"  else 0
        day_type_inside = 1 if day_type_str == "inside" else 0
        day_type_range  = 1 if day_type_str == "range"  else 0
        ny_open_in_lon  = _cv("ny_open_in_lon", 0)
        lon_range_pt    = _cv("lon_range", 80.0)
        lon_range_narrow= 1 if lon_range_pt < 60 else 0

        # Tier 4: macro & timing
        fomc_proximity  = _cv("fomc_proximity", 0)
        dow_mon         = is_monday
        dow_fri         = is_friday

        # ── Assemble feature row ──────────────────────────────────────────────
        feat_vals = {
            # v3 original features
            "tp_atr_ratio":      tp_atr_ratio,
            "tp_reachable":      tp_reachable,
            "tp_in_2atr":        tp_in_2atr,
            "premium_discount":  premium_discount,
            "h4_aligned":        h4_aligned,
            "va_width_atr":      va_width_atr,
            "sweep_extreme":     sweep_extreme,
            "full_setup":        full_setup,
            "htf_aligned":       htf_aligned,
            "in_first_15":       in_first_15,
            "dist_val_atr":      dist_val_atr,
            "dist_vah_atr":      dist_vah_atr,
            "pre_lon_pos":       pre_lon_pos,
            "bar_range_atr":     bar_range_atr,
            "rr":                rr,
            "fvg_up":            fvg_up,
            "fvg_down":          fvg_down,
            "h4_bias":           h4_bias,
            "h1_bias":           h1_bias,
            "sl_atr_ratio":      sl_atr_ratio,
            "vwap_aligned":      vwap_aligned,
            "dist_vwap_norm":    dist_vwap_norm,
            "dist_poc_atr":      dist_poc_atr,
            "near_poc":          near_poc,
            "adx_14_val":        adx,
            "adx_strong":        adx_strong,
            "adx_weak":          adx_weak,
            "hurst_val":         hurst,
            "hurst_trending":    hurst_trending,
            "is_monday":         is_monday,
            "is_friday":         is_friday,
            "body_with_trade":   body_with_trade,
            "pre_lon_range_atr": pre_lon_range_atr,
            "tp_at_pd_level":    tp_at_pd_level,
            "outside_va":        outside_va,
            "early_lon":         early_lon,
            "late_lon":          late_lon,
            "fisher_extreme":    fisher_extreme,
            "acf_trending":      acf_trending,
            # v3 session features
            "sweep_dist_pny_close_atr":  sweep_dist_pny_close_atr,
            "sweep_near_pny_close":      sweep_near_pny_close,
            "sweep_dist_plon_close_atr": sweep_dist_plon_close_atr,
            "sweep_near_plon_close":     sweep_near_plon_close,
            "sweep_dist_pdh_atr":        sweep_dist_pdh_atr,
            "sweep_dist_pdl_atr":        sweep_dist_pdl_atr,
            "sweep_near_pd":             sweep_near_pd,
            "in_open30":                 in_open30,
            "sweep_near_pd_open30":      sweep_near_pd_open30,
            "prev_ny_push_dir":          prev_ny_push_dir,
            "is_tuesday":                is_tuesday,
            "is_thursday":               is_thursday,
            # v4 Tier 1-4 features noi
            "weekly_premium_pct":   weekly_premium_pct,
            "weekly_prem_direction":weekly_prem_direction,
            "h4_structure":         h4_structure,
            "h4_struct_aligned":    h4_struct_aligned,
            "daily_bias":           daily_bias,
            "daily_bias_aligned":   daily_bias_aligned,
            "h4_x_daily":           h4_x_daily,
            "ob_bull_active":       ob_bull_active,
            "ob_bear_active":       ob_bear_active,
            "ob_aligned":           ob_aligned,
            "eq_highs_above":       eq_highs_above,
            "eq_lows_below":        eq_lows_below,
            "eq_sweep_aligned":     eq_sweep_aligned,
            "fvg_active_bull":      fvg_active_bull,
            "fvg_active_bear":      fvg_active_bear,
            "fvg_aligned":          fvg_aligned,
            "day_type_trend":       day_type_trend,
            "day_type_inside":      day_type_inside,
            "day_type_range":       day_type_range,
            "ny_open_in_lon":       ny_open_in_lon,
            "lon_range_pt":         lon_range_pt,
            "lon_range_narrow":     lon_range_narrow,
            "fomc_proximity":       fomc_proximity,
            "dow_mon":              dow_mon,
            "dow_fri":              dow_fri,
        }

        features = scorer["features"]
        row = pd.DataFrame([[feat_vals.get(f, 0.0) for f in features]], columns=features)
        score = round(float(scorer["model"].predict_proba(row)[0, 1]), 3)
        logger.info(f"🤖 ML scorer: {direction} | score={score:.3f} "
                    f"(rr={rr:.2f}, tp_atr={tp_atr_ratio:.2f}, "
                    f"premium_discount={premium_discount}, h4_aligned={h4_aligned})")
        return score

    except Exception as exc:
        logger.warning(f"ML scorer compute error: {exc}")
        return 0.0


# ── Constante (identice cu setup_detector_v3.py) ──────────────────────────────
DISPL_ATR_MULT = 1.0      # multiplicator ATR pt displacement bar
MIN_RR         = 1.5      # RR minim
MIN_TP_PT      = 10.0     # TP minim în puncte
MAX_TP_PT      = 55.0     # TP maxim în puncte
MAX_SL_PT      = 12.0     # SL maxim în puncte (filtru propfirm v3.1)
MIN_SL_PT      = 4.0      # SL minim structural

# Ferestre sesiune UTC
LON_DISP_START = 800;   LON_DISP_END  = 1030   # Londra: 08:00-10:30 UTC
NY_DISP_START  = 1330;  NY_DISP_END   = 1430   # New York open: 13:30-14:30 UTC
PRE_LON_START  = 500;   PRE_LON_END   = 759    # pre-Londra: 05:00-07:59 UTC
PRE_NY_START   = 800;   PRE_NY_END    = 1259   # sesiunea Londra completă = baza pt NY sweep

# ── Track zilnic ───────────────────────────────────────────────────────────────
_gate_state_file = Path(__file__).parent / "ict_gate_state.json"

def _load_state() -> dict:
    """Încarcă starea zilnică (trade count, best_signal) din fişier JSON."""
    today = date.today().isoformat()
    if _gate_state_file.exists():
        try:
            s = json.loads(_gate_state_file.read_text())
            if s.get("date") == today:
                return s
        except Exception:
            pass
    return {"date": today, "trades_taken": 0, "best_rr": 0.0, "signal_taken": None}

def _save_state(state: dict):
    _gate_state_file.write_text(json.dumps(state, default=str))

def mark_trade_taken(signal: dict):
    """Chemat de bridge după ce trade-ul e executat."""
    s = _load_state()
    s["trades_taken"] += 1
    s["signal_taken"] = signal
    _save_state(s)
    logger.info(f"ICT Gate: trade marcat → {s['trades_taken']}/2 maxim azi")

def daily_trades_taken() -> int:
    return _load_state()["trades_taken"]

# ── Helpers ────────────────────────────────────────────────────────────────────
def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _hhmm(ts: pd.Timestamp) -> int:
    return ts.hour * 100 + ts.minute

def _nearest_above(entry: float, levels: dict) -> tuple:
    """Cel mai aproape nivel DEASUPRA entry — returnează (price, name, dist)."""
    cands = [(v, k, v - entry)
             for k, v in levels.items()
             if not np.isnan(v) and MIN_TP_PT <= v - entry <= MAX_TP_PT]
    return sorted(cands, key=lambda x: x[2])[0] if cands else (None, None, 0.)

def _nearest_below(entry: float, levels: dict) -> tuple:
    """Cel mai aproape nivel SUB entry."""
    cands = [(v, k, entry - v)
             for k, v in levels.items()
             if not np.isnan(v) and MIN_TP_PT <= entry - v <= MAX_TP_PT]
    return sorted(cands, key=lambda x: x[2])[0] if cands else (None, None, 0.)

def _detect_sweep(bars: pd.DataFrame, level: float, direction: str, lookback: int = 3) -> bool:
    """
    Detectează sweep: price a depăşit `level` în direcţia opusă, apoi s-a închis înapoi.
    direction='above': high > level şi close < level (sweep high)
    direction='below': low  < level şi close > level (sweep low)
    """
    if level is None or np.isnan(level):
        return False
    for i in range(min(lookback, len(bars))):
        row = bars.iloc[-(i+1)]
        if direction == 'above' and row['high'] > level and row['close'] < level:
            return True
        if direction == 'below' and row['low'] < level and row['close'] > level:
            return True
    return False

def _detect_displacement(bars: pd.DataFrame, direction: str, atr: float) -> int:
    """
    Returnează indexul (de la sfârşit) al celei mai recente bare de displacement.
    displacement = |close - open| >= DISPL_ATR_MULT * ATR şi direcţia corectă.
    Returnează -1 dacă nu există.
    """
    threshold = DISPL_ATR_MULT * atr
    for i in range(min(10, len(bars))):
        row = bars.iloc[-(i+1)]
        body = row['close'] - row['open']
        if direction == 'SHORT' and body < -threshold:
            return i
        if direction == 'LONG'  and body > threshold:
            return i
    return -1

def _detect_fvg(bars: pd.DataFrame, disp_idx: int, direction: str) -> tuple:
    """
    FVG după bara de displacement.
    SHORT FVG: bara[i+1].low > bara[i-1].high  (gap în jos)
    LONG  FVG: bara[i+1].high < bara[i-1].low  (gap în sus)
    Returnează (fvg_hi, fvg_lo) sau (None, None).
    """
    # bara de displacement e la index -(disp_idx+1) de la sfârşit
    # bara înainte de displacement: -(disp_idx+2)
    # bara după displacement: -(disp_idx) dacă există
    if disp_idx < 0:
        return None, None
    b_idx  = len(bars) - (disp_idx + 1)  # bara displacement
    if b_idx < 1 or b_idx >= len(bars):
        return None, None
    b_disp  = bars.iloc[b_idx]
    b_prev  = bars.iloc[b_idx - 1] if b_idx > 0 else None
    b_after = bars.iloc[b_idx + 1] if b_idx + 1 < len(bars) else None

    if b_prev is None or b_after is None:
        return None, None

    if direction == 'SHORT':
        # FVG bearish: high-ul barei după < low-ul barei înainte
        if b_after['high'] < b_prev['low']:
            return b_prev['low'], b_after['high']  # (fvg_hi, fvg_lo)
    elif direction == 'LONG':
        # FVG bullish: low-ul barei după > high-ul barei înainte
        if b_after['low'] > b_prev['high']:
            return b_after['low'], b_prev['high']  # (fvg_hi, fvg_lo) → entry în mijloc
    return None, None

def _detect_retrace(bars: pd.DataFrame, fvg_hi: float, fvg_lo: float,
                    direction: str, after_idx: int) -> bool:
    """
    Verifică dacă price a retrasat în FVG după bara after_idx.
    """
    if fvg_hi is None:
        return False
    fvg_mid = (fvg_hi + fvg_lo) / 2
    for i in range(after_idx, min(after_idx + 30, len(bars))):
        row = bars.iloc[i]
        if direction == 'SHORT' and row['high'] >= fvg_lo and row['close'] <= fvg_hi:
            return True
        if direction == 'LONG'  and row['low']  <= fvg_hi and row['close'] >= fvg_lo:
            return True
    return False

def _trend_15m(bars_1m: pd.DataFrame, n_ema: int = 20) -> int:
    """
    Trend 15m calculat din barele 1m prin resample.
    Returnează 1 (bullish), -1 (bearish), 0 (neutru).
    """
    try:
        b15 = bars_1m.resample('15min', on='ts').agg(
            open=('open','first'), high=('high','max'),
            low=('low','min'), close=('close','last')
        ).dropna()
        if len(b15) < n_ema + 2:
            return 0
        ema = _ema(b15['close'], n_ema)
        last_close = b15['close'].iloc[-1]
        last_ema   = ema.iloc[-1]
        prev_ema   = ema.iloc[-2]
        if last_close > last_ema and last_ema > prev_ema:
            return 1
        if last_close < last_ema and last_ema < prev_ema:
            return -1
        return 0
    except Exception:
        return 0

# ── Turtle Soup Setup (check dedicat, separat de check_setup) ─────────────────
def check_ts_setup(db_path: str, now_utc: datetime = None) -> dict | None:
    """
    Detectează Turtle Soup pattern în timp real pe bare 1m din DB.

    Template ICT (4H windows):
      W1: c22pm (22-01:59 EST) setează range → c2am (02-05:59 EST) sweepează → MSS → entry
      W2: c2am  (02-05:59 EST) setează range → c6am (06-09:59 EST) sweepează → MSS → entry

    Entry logic pe 1m:
      1. Prima bara 1m din fereastra de sweep care depășește nivelul N cu >1pt
      2. Maxim 20 bare după sweep → swing high/low (intra-high) se formează
      3. Maxim 60 bare de la sweep → close dincolo de intra-high → ENTRY (MSS confirmat)

    SL: low/high al candelei 4H de sweep - 1pt, cap la 20pt de la entry
    TP: extrema opusă a candelei N (4H)
    Max SL: 20pt | Min RR: 1.5

    Fereastra activă (UTC, winter EST):
      W1 active: 07:00-10:59 UTC (imediat după c2am → înainte de c10am)
      W2 active: 11:00-16:00 UTC (imediat după c6am → end NY open zone)
    """
    if daily_trades_taken() >= 2:
        return None

    if now_utc is None:
        now_utc = datetime.now(ZoneInfo("UTC"))

    # ── EST/EDT offset ──────────────────────────────────────────────────────────
    today_utc  = now_utc.date()
    hhmm_now   = now_utc.hour * 100 + now_utc.minute

    # DST: 2nd Sunday March → 1st Sunday November = EDT (UTC-4), else EST (UTC-5)
    yr    = today_utc.year
    mar1  = datetime(yr, 3, 1, tzinfo=ZoneInfo("UTC"))
    edt_s = mar1 + timedelta(days=(6 - mar1.weekday()) % 7) + timedelta(weeks=1)
    nov1  = datetime(yr, 11, 1, tzinfo=ZoneInfo("UTC"))
    edt_e = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    off   = 4 if edt_s.date() <= today_utc < edt_e.date() else 5

    # ── Ferestre UTC pentru fiecare candelă 4H ──────────────────────────────────
    def win(h_est):
        hs = (h_est + off) % 24
        return hs * 100, (hs + 3) * 100 + 59

    w22_s, w22_e = win(22)   # c22pm window
    w2_s,  w2_e  = win(2)    # c2am  window
    w6_s,  w6_e  = win(6)    # c6am  window

    # Fereastra activă pentru TS:
    # W1 (sweep la c2am): MSS căutat după ce c2am a sweepat → de la w2_s la w6_e
    # W2 (sweep la c6am): MSS căutat după ce c6am a sweepat → de la w6_s la 1600 UTC
    ts_active_w1 = w2_s <= hhmm_now <= w6_e
    ts_active_w2 = w6_s <= hhmm_now <= 1600

    if not ts_active_w1 and not ts_active_w2:
        logger.debug(f"TS Gate: {hhmm_now} UTC în afara ferestrei TS active")
        return None

    # ── Încărcare bare 1m din DB (azi + ieri pentru c22pm) ─────────────────────
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True,
                               timeout=30, check_same_thread=False)
        df_raw = pd.read_sql(f"""
            SELECT timestamp, open, high, low, close, date
            FROM market_data
            WHERE date >= '{today_utc - timedelta(days=1)}'
            ORDER BY timestamp
        """, conn)
        conn.close()
    except Exception as e:
        logger.error(f"TS Gate DB error: {e}")
        return None

    if len(df_raw) < 30:
        return None

    df_raw['ts']   = pd.to_datetime(df_raw['timestamp'])
    df_raw['hhmm'] = df_raw['ts'].dt.hour * 100 + df_raw['ts'].dt.minute
    df_today       = df_raw[df_raw['date'] == str(today_utc)].reset_index(drop=True)

    if len(df_today) < 20:
        return None

    # ── Helper: agregare candelă 4H ─────────────────────────────────────────────
    def candle_4h(h_utc_start):
        hs = h_utc_start * 100
        he = (h_utc_start + 3) * 100 + 59
        bars = df_today[(df_today['hhmm'] >= hs) & (df_today['hhmm'] <= he)]
        if len(bars) < 2:
            return None
        return {
            'high':  float(bars['high'].max()),
            'low':   float(bars['low'].min()),
            'close': float(bars['close'].iloc[-1]),
        }

    # ── Helper: MSS detection pe 1m ────────────────────────────────────────────
    def find_mss(h_utc_start, direction, sweep_level, min_sweep=1.0):
        """
        Scanează fereastra 4H pentru sweep + swing high/low + MSS break.
        Returnează (entry, sl_1m_low, entry_hhmm) sau None.
        sl_1m_low = low-ul barei de sweep pe 1m (SL calculat în caller).
        """
        hs = h_utc_start * 100
        he = (h_utc_start + 3) * 100 + 59
        bars = df_today[(df_today['hhmm'] >= hs) & (df_today['hhmm'] <= he)].reset_index(drop=True)

        if len(bars) < 5:
            return None

        highs  = bars['high'].values
        lows   = bars['low'].values
        closes = bars['close'].values
        hhmms  = bars['hhmm'].values
        n      = len(bars)

        # 1. Prima bara de sweep
        if direction == 'LONG':
            sw_hits = np.where(lows < sweep_level - min_sweep)[0]
        else:
            sw_hits = np.where(highs > sweep_level + min_sweep)[0]

        if len(sw_hits) == 0:
            return None

        sw_i         = sw_hits[0]
        sweep_1m_ext = lows[sw_i] if direction == 'LONG' else highs[sw_i]

        # 2. Swing high/low în MAX 20 bare după sweep
        max_b = min(sw_i + 21, n - 1)
        intra_level = None
        intra_j     = None

        for j in range(sw_i + 1, max_b):
            if j + 1 >= n:
                break
            if direction == 'LONG':
                if highs[j] > highs[j - 1] and highs[j] > highs[j + 1]:
                    intra_level = highs[j]
                    intra_j     = j
                    break
            else:
                if lows[j] < lows[j - 1] and lows[j] < lows[j + 1]:
                    intra_level = lows[j]
                    intra_j     = j
                    break

        if intra_level is None:
            return None

        # 3. MSS break în MAX 60 bare de la sweep
        max_mss = min(sw_i + 61, n)
        for k in range(intra_j + 1, max_mss):
            if direction == 'LONG' and closes[k] > intra_level:
                return closes[k], sweep_1m_ext, hhmms[k]
            if direction == 'SHORT' and closes[k] < intra_level:
                return closes[k], sweep_1m_ext, hhmms[k]

        return None

    # ── Helper: calculează SL (4H sweep candle extreme, cap 20pt) ──────────────
    def build_sl(entry, sweep_4h_extreme, direction, cap=20.0):
        if direction == 'LONG':
            sl = sweep_4h_extreme - 1.0
            if entry - sl > cap:
                sl = entry - cap
        else:
            sl = sweep_4h_extreme + 1.0
            if sl - entry > cap:
                sl = entry + cap
        return round(sl, 2)

    # ── Candelele 4H necesare ───────────────────────────────────────────────────
    h22_utc = (22 + off) % 24
    h2_utc  = ( 2 + off) % 24
    h6_utc  = ( 6 + off) % 24

    c22pm = candle_4h(h22_utc)
    c2am  = candle_4h(h2_utc)
    c6am  = candle_4h(h6_utc)

    candidates = []

    # ── W1: c2am sweepează c22pm → MSS → TP=c22pm.{high/low} ──────────────────
    if ts_active_w1 and c22pm and c2am:

        # W1 LONG: c2am swept sub c22pm.low
        if c2am['low'] < c22pm['low'] - 1.0:
            res = find_mss(h2_utc, 'LONG', c22pm['low'])
            if res:
                entry, sw_ext, ehhmm = res
                sl      = build_sl(entry, c6am['low'] if c6am else c2am['low'], 'LONG')
                tp      = c22pm['high']
                sl_dist = round(entry - sl, 2)
                tp_dist = round(tp - entry, 2)
                rr      = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0
                if tp_dist > 0 and rr >= 1.5:
                    candidates.append({
                        'direction':  'LONG',
                        'setup_type': 'TS_W1',
                        'entry':      round(entry, 2),
                        'sl':         sl,
                        'sl_pt':      sl_dist,
                        'tp':         round(tp, 2),
                        'tp_pt':      tp_dist,
                        'rr':         rr,
                        'session':    'LON',
                        'entry_hhmm': ehhmm,
                        'message':    f"🐢 TS_W1_LONG | c2am swept c22pm.low | MSS@{ehhmm} | entry={entry:.1f} sl={sl:.1f} tp={tp:.1f} RR={rr:.2f}",
                    })

        # W1 SHORT: c2am swept peste c22pm.high
        if c2am['high'] > c22pm['high'] + 1.0:
            res = find_mss(h2_utc, 'SHORT', c22pm['high'])
            if res:
                entry, sw_ext, ehhmm = res
                sl      = build_sl(entry, c6am['high'] if c6am else c2am['high'], 'SHORT')
                tp      = c22pm['low']
                sl_dist = round(sl - entry, 2)
                tp_dist = round(entry - tp, 2)
                rr      = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0
                if tp_dist > 0 and rr >= 1.5:
                    candidates.append({
                        'direction':  'SHORT',
                        'setup_type': 'TS_W1',
                        'entry':      round(entry, 2),
                        'sl':         sl,
                        'sl_pt':      sl_dist,
                        'tp':         round(tp, 2),
                        'tp_pt':      tp_dist,
                        'rr':         rr,
                        'session':    'LON',
                        'entry_hhmm': ehhmm,
                        'message':    f"🐢 TS_W1_SHORT | c2am swept c22pm.high | MSS@{ehhmm} | entry={entry:.1f} sl={sl:.1f} tp={tp:.1f} RR={rr:.2f}",
                    })

    # ── W2: c6am sweepează c2am → MSS → TP=c2am.{high/low} ────────────────────
    if ts_active_w2 and c2am and c6am:

        # W2 LONG: c6am swept sub c2am.low
        if c6am['low'] < c2am['low'] - 1.0:
            res = find_mss(h6_utc, 'LONG', c2am['low'])
            if res:
                entry, sw_ext, ehhmm = res
                sl      = build_sl(entry, c6am['low'], 'LONG')
                tp      = c2am['high']
                sl_dist = round(entry - sl, 2)
                tp_dist = round(tp - entry, 2)
                rr      = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0
                if tp_dist > 0 and rr >= 1.5:
                    candidates.append({
                        'direction':  'LONG',
                        'setup_type': 'TS_W2',
                        'entry':      round(entry, 2),
                        'sl':         sl,
                        'sl_pt':      sl_dist,
                        'tp':         round(tp, 2),
                        'tp_pt':      tp_dist,
                        'rr':         rr,
                        'session':    'NY',
                        'entry_hhmm': ehhmm,
                        'message':    f"🐢 TS_W2_LONG | c6am swept c2am.low | MSS@{ehhmm} | entry={entry:.1f} sl={sl:.1f} tp={tp:.1f} RR={rr:.2f}",
                    })

        # W2 SHORT: c6am swept peste c2am.high
        if c6am['high'] > c2am['high'] + 1.0:
            res = find_mss(h6_utc, 'SHORT', c2am['high'])
            if res:
                entry, sw_ext, ehhmm = res
                sl      = build_sl(entry, c6am['high'], 'SHORT')
                tp      = c2am['low']
                sl_dist = round(sl - entry, 2)
                tp_dist = round(entry - tp, 2)
                rr      = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0
                if tp_dist > 0 and rr >= 1.5:
                    candidates.append({
                        'direction':  'SHORT',
                        'setup_type': 'TS_W2',
                        'entry':      round(entry, 2),
                        'sl':         sl,
                        'sl_pt':      sl_dist,
                        'tp':         round(tp, 2),
                        'tp_pt':      tp_dist,
                        'rr':         rr,
                        'session':    'NY',
                        'entry_hhmm': ehhmm,
                        'message':    f"🐢 TS_W2_SHORT | c6am swept c2am.high | MSS@{ehhmm} | entry={entry:.1f} sl={sl:.1f} tp={tp:.1f} RR={rr:.2f}",
                    })

    if not candidates:
        return None

    # ── Alege cel mai bun candidat (RR maxim) ──────────────────────────────────
    best_ts = max(candidates, key=lambda x: x['rr'])
    logger.info(f"TS Gate: SEMNAL → {best_ts['message']}")
    return best_ts


# ── Funcţia principală ─────────────────────────────────────────────────────────
def check_setup(db_path: str, now_utc: datetime = None) -> dict | None:
    """
    Verifică dacă există un setup ICT valid la momentul curent.

    Returnează dict cu câmpuri de semnal sau None dacă nu există semnal valid.
    Dict keys: direction, setup_type, entry, sl, sl_pt, tp, tp_pt, rr,
               tp_level, session, message
    """
    # ── Guards ──────────────────────────────────────────────────────────────────
    if daily_trades_taken() >= 2:
        logger.debug("ICT Gate: deja 2 trades azi → SKIP")
        return None

    if now_utc is None:
        now_utc = datetime.now(ZoneInfo("UTC"))

    today = now_utc.date()
    hhmm  = now_utc.hour * 100 + now_utc.minute

    # Determinăm sesiunea curentă
    if LON_DISP_START <= hhmm <= LON_DISP_END:
        session = "LON"
    elif NY_DISP_START <= hhmm <= NY_DISP_END:
        session = "NY"
    else:
        logger.debug(f"ICT Gate: {hhmm} UTC în afara sesiunilor (LON 08-10:30, NY 13:30-14:30)")
        return None

    # ── Încărcare date ───────────────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True,
                               timeout=30, check_same_thread=False)
        # Luăm barele de azi + ieri (pentru nivele pre-sesiune)
        df = pd.read_sql(f"""
            SELECT timestamp, date, open, high, low, close, volume,
                   asia_hi, asia_lo, p_hi, p_lo, vah, val, poc_level,
                   atr_14, bar_delta,
                   h4_hi, h4_lo, h1_hi, h1_lo,
                   dist_vwap, adx_14, hurst,
                   fisher_transform, acf_lag1,
                   dist_poc, inside_va,
                   fvg_up, fvg_down, has_displacement
            FROM market_data
            WHERE date >= '{today - timedelta(days=1)}'
              AND CAST(strftime('%H', timestamp) AS INT) BETWEEN 4 AND 18
            ORDER BY timestamp
        """, conn)
        conn.close()
    except Exception as e:
        logger.error(f"ICT Gate DB error: {e}")
        return None

    if len(df) < 30:
        logger.debug("ICT Gate: date insuficiente")
        return None

    df['ts']   = pd.to_datetime(df['timestamp'])
    df['hhmm'] = df['ts'].dt.hour * 100 + df['ts'].dt.minute
    df_today   = df[df['date'] == str(today)].copy()

    if len(df_today) < 10:
        logger.debug("ICT Gate: prea puţine bare azi")
        return None

    # ── Niveluri zilnice ─────────────────────────────────────────────────────────
    r0    = df_today.iloc[0]
    def fv(col): return float(r0[col]) if not pd.isna(r0[col]) else np.nan

    asia_hi  = fv('asia_hi')
    asia_lo  = fv('asia_lo')
    pdh      = fv('p_hi')
    pdl      = fv('p_lo')
    vah      = fv('vah')
    val      = fv('val')
    poc      = fv('poc_level')

    # ATR curent
    atr_vals = df_today['atr_14'].replace(0, np.nan).dropna()
    atr = float(atr_vals.iloc[-1]) if len(atr_vals) > 0 else 10.0

    # ── Bare sesiune curentă ─────────────────────────────────────────────────────
    if session == "LON":
        pre_bars  = df_today[df_today['hhmm'].between(PRE_LON_START, PRE_LON_END)]
        sess_bars = df_today[df_today['hhmm'].between(LON_DISP_START, hhmm)]
        sweep_levels_short = {'asia_hi': asia_hi, 'pdh': pdh, 'vah': vah}
        sweep_levels_long  = {'asia_lo': asia_lo, 'pdl': pdl, 'val': val}
        tp_levels_short    = {'val': val, 'poc': poc, 'pdl': pdl, 'asia_lo': asia_lo}
        tp_levels_long     = {'vah': vah, 'poc': poc, 'pdh': pdh, 'asia_hi': asia_hi}
    else:  # NY
        # London high/low computat din barele 08:00-11:00 UTC de azi
        lon_bars   = df_today[df_today['hhmm'].between(800, 1059)]
        lon_hi_day = float(lon_bars['high'].max()) if len(lon_bars) > 0 else np.nan
        lon_lo_day = float(lon_bars['low'].min())  if len(lon_bars) > 0 else np.nan
        pre_bars   = df_today[df_today['hhmm'].between(1200, 1329)]
        sess_bars  = df_today[df_today['hhmm'].between(NY_DISP_START, hhmm)]
        sweep_levels_short = {'lon_hi': lon_hi_day, 'pdh': pdh, 'vah': vah}
        sweep_levels_long  = {'lon_lo': lon_lo_day, 'pdl': pdl, 'val': val}
        tp_levels_short    = {'val': val, 'poc': poc, 'lon_lo': lon_lo_day, 'pdl': pdl}
        tp_levels_long     = {'vah': vah, 'poc': poc, 'lon_hi': lon_hi_day, 'pdh': pdh}

    if len(sess_bars) < 3:
        logger.debug(f"ICT Gate {session}: prea puţine bare sesiune ({len(sess_bars)})")
        return None

    # ── Verific setups în ambele direcţii ────────────────────────────────────────
    candidates = []

    for direction in ['SHORT', 'LONG']:
        if direction == 'SHORT':
            sweep_lvls = sweep_levels_short
            tp_lvls    = tp_levels_short
            sweep_dir  = 'above'
        else:
            sweep_lvls = sweep_levels_long
            tp_lvls    = tp_levels_long
            sweep_dir  = 'below'

        # ── 1. Filtru 15m trend ──────────────────────────────────────────────────
        bars_for_trend = df_today[df_today['hhmm'] <= hhmm].copy()
        trend = _trend_15m(bars_for_trend)

        if direction == 'SHORT':
            if session == 'LON' and trend != -1:
                continue  # LON SHORT necesită 15m bearish
            if session == 'NY' and trend != -1:
                continue
        else:  # LONG
            if trend != 1:
                continue  # LONG necesită 15m bullish

        # LON LONG necesită şi asia_sweep (confirmare extra)
        if session == 'LON' and direction == 'LONG':
            sl_asia = _detect_sweep(
                df_today[df_today['hhmm'] <= hhmm], asia_lo, 'below', lookback=6
            )
            if not sl_asia:
                continue

        # ── 2. Sweep în pre-sesiune ──────────────────────────────────────────────
        sweep_found = False
        for lvl_name, lvl_val in sweep_lvls.items():
            if np.isnan(lvl_val):
                continue
            bars_pre = pre_bars if len(pre_bars) > 0 else df_today[df_today['hhmm'] < hhmm - 100]
            if _detect_sweep(bars_pre, lvl_val, sweep_dir, lookback=len(bars_pre)):
                sweep_found = True
                break
        if not sweep_found:
            continue

        # ── 3. Displacement în sesiune ───────────────────────────────────────────
        disp_idx = _detect_displacement(sess_bars, direction, atr)
        if disp_idx < 0:
            continue

        # ── 4. FVG ──────────────────────────────────────────────────────────────
        fvg_hi, fvg_lo = _detect_fvg(sess_bars, disp_idx, direction)
        if fvg_hi is None:
            continue

        # ── 5. Retrace la FVG ───────────────────────────────────────────────────
        after_disp_idx = len(sess_bars) - disp_idx
        retrace_ok = _detect_retrace(sess_bars, fvg_hi, fvg_lo, direction, after_disp_idx)
        if not retrace_ok:
            continue

        # ── 6. Entry, SL, TP ────────────────────────────────────────────────────
        entry = float(sess_bars['close'].iloc[-1])

        # SL structural: swing recent contrar
        if direction == 'SHORT':
            recent_hi = float(sess_bars['high'].iloc[-5:].max())
            sl_dist   = max(MIN_SL_PT, min(recent_hi - entry + 1.0, MAX_SL_PT))
            sl_price  = entry + sl_dist
            tp_price, tp_name, tp_dist = _nearest_below(entry, tp_lvls)
        else:
            recent_lo = float(sess_bars['low'].iloc[-5:].min())
            sl_dist   = max(MIN_SL_PT, min(entry - recent_lo + 1.0, MAX_SL_PT))
            sl_price  = entry - sl_dist
            tp_price, tp_name, tp_dist = _nearest_above(entry, tp_lvls)

        # Filtre calitate
        if tp_price is None:
            continue
        if sl_dist > MAX_SL_PT:
            continue  # filtru propfirm: max 15pt SL
        rr = tp_dist / sl_dist
        if rr < MIN_RR:
            continue

        setup_type = f"{session}_{direction}"
        candidates.append({
            "direction":  direction,
            "setup_type": setup_type,
            "session":    session,
            "entry":      round(entry, 2),
            "sl":         round(sl_price, 2),
            "sl_pt":      round(sl_dist, 2),
            "tp":         round(tp_price, 2),
            "tp_pt":      round(tp_dist, 2),
            "tp_level":   tp_name,
            "rr":         round(rr, 2),
            "trend_15m":  trend,
            "message":    (f"✅ {setup_type} | Entry {round(entry,2)} | "
                           f"SL {round(sl_dist,1)}pt | TP {round(tp_dist,1)}pt | RR {round(rr,2)}")
        })

    if not candidates:
        return None

    # ── Alege cel mai bun (RR maxim) ────────────────────────────────────────────
    best = max(candidates, key=lambda x: x['rr'])

    # ── Quality Gate v1 (v6/ny_v3/ts) — înlocuiește scorer v4.1 ────────────────
    if _QUALITY_GATE_OK and _qg is not None:
        try:
            # Conectare DB pentru quality gate
            _qg_conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True,
                                       timeout=30, check_same_thread=False)
            best['session'] = session
            _qg_result = _qg.score_quality(
                signal  = best,
                now_utc = now_utc,
                conn    = _qg_conn,
            )
            _qg_conn.close()
            _sc     = _qg_result["score"]
            _sc_v6  = _qg_result["score_v6"]
            _sc_ts  = _qg_result["score_ts"]
            _model  = _qg_result["model"]
            _thr    = _qg_result["threshold"]
            _passed = _qg_result["passed"]
            best['ml_score']    = _sc
            best['ml_score_v6'] = _sc_v6
            best['ml_score_ts'] = _sc_ts
            best['ml_model']    = _model
            _sc_emoji = "🟢" if _sc >= 0.50 else ("🟡" if _sc >= 0.30 else "🔴")
            ts_str = f" ts={_sc_ts:.3f}" if _sc_ts > 0 else ""
            best['message'] += f" | QG[{_model}]={_sc:.3f}{_sc_emoji}{ts_str}"
            if not _passed:
                logger.info(
                    f"❌ QualityGate[{_model}]: scor {_sc:.3f} < {_thr} "
                    f"→ WAIT (setup slab, eliminat)"
                )
                return None
        except Exception as _qg_exc:
            logger.warning(f"quality_gate_live error: {_qg_exc} — fallback scorer v4")
            # Fallback la scorer vechi
            best['ml_score'] = _score_setup(
                signal=best, df_today=df_today, pre_bars=pre_bars,
                vah=vah, val=val, poc=poc, atr=atr, hhmm=hhmm, now_utc=now_utc,
            )
            _sc = best['ml_score']
            best['message'] += f" | ML_v4={_sc:.3f}"
            if _sc < ML_BLOCK_THRESHOLD:
                return None
    else:
        # ── Fallback: ML Setup Score v4 ─────────────────────────────────────────
        best['ml_score'] = _score_setup(
            signal   = best,
            df_today = df_today,
            pre_bars = pre_bars,
            vah      = vah,
            val      = val,
            poc      = poc,
            atr      = atr,
            hhmm     = hhmm,
            now_utc  = now_utc,
        )
        _sc = best['ml_score']
        _sc_emoji = "🟢" if _sc >= 0.55 else ("🟡" if _sc >= 0.35 else "🔴")
        best['message'] += f" | ML={_sc:.3f}{_sc_emoji}"
        if _sc < ML_BLOCK_THRESHOLD:
            logger.info(
                f"❌ ML scorer v4: scor {_sc:.3f} < {ML_BLOCK_THRESHOLD} "
                f"→ WAIT (setup slab, eliminat de scorer)"
            )
            return None

    # ── Rule Signals v3 (TP confluence, TS, Big Wick, Weekly Profile, 15m) ────
    date_str_gate = now_utc.strftime("%Y-%m-%d")
    cond_gate     = _load_daily_conditions(date_str_gate)

    rs = _check_rule_signals(
        signal   = best,
        cond     = cond_gate,
        df_today = df_today,
        atr      = atr,
        now_utc  = now_utc,
    )
    best['rule_signals'] = rs

    # ── Build rule log line ───────────────────────────────────────────────────
    rule_parts = []

    # TP confluences
    if rs['n_tp_confluences'] >= 2:
        rule_parts.append(f"🎯TP×{rs['n_tp_confluences']}({','.join(rs['tp_hits'])})")
    elif rs['n_tp_confluences'] == 1:
        rule_parts.append(f"🎯TP({','.join(rs['tp_hits'])})")
    else:
        rule_parts.append("⚠️TP_noLevel")

    # Turtle Soup
    if rs['ts_aligned']:
        rule_parts.append("🐢TS✅")
    elif rs['c6am_sweep_aligned']:
        rule_parts.append("🐢6AM_sweep")

    # Big wick
    if rs['bw_aligned']:
        rule_parts.append("⚡BigWick")

    # Weekly profile
    wp_emoji = "📈" if rs['profile_aligned'] else "⚖️"
    rule_parts.append(f"{wp_emoji}W={rs['profile_str']}")
    if rs['seek_aligned']:
        rule_parts.append("💥S&D")
    if rs['monday_is_low']:
        rule_parts.append("Mon=LOW")
    if rs['tuesday_extreme']:
        rule_parts.append("Tue=EXT")

    # Trend 15m
    t15_emoji = "📈" if rs['trend_15m'] == 1 else ("📉" if rs['trend_15m'] == -1 else "➡️")
    t15_align = "✅" if rs['trend_15m_aligned'] else "❌"
    rule_parts.append(f"{t15_emoji}15m{t15_align}")

    # VWAP
    vwap_emoji = "✅" if rs['vwap_ok'] else "❌"
    rule_parts.append(f"VWAP{vwap_emoji}({rs['vwap_dist_pt']:.0f}pt)")

    # Confluence score final
    conf_emoji = "🔥" if rs['confluence_score'] >= 0.67 else ("💪" if rs['confluence_score'] >= 0.44 else "ℹ️")
    rule_parts.append(f"{conf_emoji}CONF={rs['confluence_score']:.2f}({rs['raw_pts']}/9)")

    best['message'] += " | " + " ".join(rule_parts)

    # Log rezumat complet
    logger.info(f"ICT Gate: SEMNAL VALID → {best['message']}")

    # ── Optional: WAIT dacă profilul săptămânal și TS sunt ambele contra ──────
    # (nu blochează singur, doar avertizare)
    if rs['profile_aligned'] == 0 and rs['trend_15m_aligned'] == 0 and _sc < 0.45:
        logger.warning(
            f"⚠️ Profile săptămânal contra ({rs['profile_str']}) + "
            f"15m contra + ML={_sc:.3f} — trade cu risc crescut"
        )

    return best


# ── Import Regime Runtime (multi-scale: classifier + Bayesian + HMM macro) ────
try:
    from regime_runtime import get_regime as _get_regime_full
    logger.info("✅ regime_runtime importat (classifier + Bayesian + HMM macro)")
    def classify_regime(db_path, now_utc=None):
        r = _get_regime_full(db_path, now_utc)
        return r['regime'], r['regime_prob']
except Exception as _e:
    logger.warning(f"regime_runtime import failed ({_e}), fallback la classifier")
    try:
        from regime_classifier_v1 import classify_regime
        logger.info("✅ Regime classifier v1 importat (fallback)")
    except Exception as _e2:
        logger.warning(f"Regime classifier import failed: {_e2}")
        def classify_regime(db_path, now_utc=None): return 'UNKNOWN', 0.0
    def _get_regime_full(db_path, now_utc=None):
        r, p = classify_regime(db_path, now_utc)
        from regime_classifier_v1 import _MESO_ENC  # poate lipsi
        enc = {'CONSOLIDATION':0,'PRE_EXPANSION':1,'EXPANSION':2,'RETRACEMENT':3,'DISTRIBUTION':4}.get(r,2)
        return {'regime':r,'regime_enc':enc,'regime_prob':p,'entropy':0.5,
                'macro':'BULL','macro_prob':0.5,'bayesian_n':0,'source':'classifier'}

# ── Import Stacking Pipeline ───────────────────────────────────────────────────
_STACKING = None
try:
    import pickle as _pkl
    import importlib.util as _ilu
    import sys as _sys
    from pathlib import Path as _Path
    # Ensure train_stacking_advanced is importat (necesar pentru depickling StackingPipeline)
    _stk_mod_path = _Path(__file__).parent / 'train_stacking_advanced.py'
    if _stk_mod_path.exists() and 'train_stacking_advanced' not in _sys.modules:
        _spec = _ilu.spec_from_file_location('train_stacking_advanced', _stk_mod_path)
        _stk_mod = _ilu.module_from_spec(_spec)
        _sys.modules['train_stacking_advanced'] = _stk_mod
        _spec.loader.exec_module(_stk_mod)
    _stk_path = _Path(__file__).parent / 'stacking_advanced_v1.pkl'
    if _stk_path.exists():
        _STACKING = _pkl.load(open(_stk_path, 'rb'))
        logger.info(f"✅ StackingPipeline v1 încărcat (fitted={getattr(_STACKING,'is_fitted',False)})")
    else:
        logger.warning("stacking_advanced_v1.pkl lipsă — stacking dezactivat")
except Exception as _e:
    logger.warning(f"Stacking load failed: {_e}")
    _STACKING = None

_RECENT_WR = []   # ultimele etichete win/loss (0/1) pentru contextul RL

# ── Import Sweep Scorer (regime-specific sweep models) ─────────────────────────
_SWEEP_SCORER_OK = False
try:
    from sweep_scorer import (
        score_sweep as _score_sweep,
        score_sweep_gated as _score_sweep_gated,
        get_sweep_threshold as _get_sweep_threshold,
        preload_all as _sweep_preload,
        SWEEP_THRESHOLDS as _SWEEP_THRESHOLDS,
    )
    _SWEEP_SCORER_OK = True
    _sweep_preload()   # pre-load toate cele 4 modele la startup
    logger.info(
        "✅ SweepScorer importat (PRE_EXPANSION / EXPANSION / RETRACEMENT / ALL) "
        f"| thresholds={_SWEEP_THRESHOLDS}"
    )
except Exception as _e:
    logger.warning(f"SweepScorer import failed: {_e} — sweep dezactivat")
    def _score_sweep(feats, regime='ALL'): return 0.5
    def _score_sweep_gated(feats, regime='ALL'): return 0.5
    def _get_sweep_threshold(regime='ALL'): return 0.55

# ── Import LOM / NOM / DSM checkers ───────────────────────────────────────────
try:
    from lom_checker_v1 import check_lom_setup
    logger.info("✅ LOM checker v1 importat")
except Exception as _e:
    logger.warning(f"LOM checker import failed: {_e}")
    def check_lom_setup(db_path, now_utc=None, regime=None, regime_prob=0.0, regime_enc=2): return None

try:
    from nom_checker_v1 import check_nom_setup
    logger.info("✅ NOM checker v1 importat")
except Exception as _e:
    logger.warning(f"NOM checker import failed: {_e}")
    def check_nom_setup(db_path, now_utc=None, regime=None, regime_prob=0.0, regime_enc=2): return None

try:
    from dsm_checker_v1 import check_dsm_setup
    logger.info("✅ DSM checker v1 importat (LON+NY | AUC=0.713 | WR=76.4%@0.65)")
except Exception as _e:
    logger.warning(f"DSM checker import failed: {_e}")
    def check_dsm_setup(db_path, now_utc=None): return None


def _apply_stacking(sig: dict, sig_type: str) -> dict:
    """
    Scor compozit final = 0.45 × ml_score (quality gate / checker)
                        + 0.25 × stacking_score (StackingPipeline calibrat)
                        + 0.30 × sweep_prob (sweep model regime-specific)

    sizing_mult = 1.0 mereu (1 contract fix, nu se scalează dinamic).
    RL action 'skip' → override la 'standard' (checker e filtrul primar).
    """
    orig_score = float(sig.get('ml_score', 0.5))

    # ── 1. Sweep score (regime-specific, gated per threshold) ────────────────
    sweep_feats  = sig.get('sweep_feats', {})
    regime_str   = sig.get('regime_str', 'ALL')
    sweep_prob   = _score_sweep_gated(sweep_feats, regime_str)   # 0.5 neutru dacă sub threshold
    sweep_raw    = _score_sweep(sweep_feats, regime_str)          # raw prob pentru logging
    sweep_thr    = _get_sweep_threshold(regime_str)

    # ── 2. Stacking score ────────────────────────────────────────────────────
    stacking_score = orig_score   # fallback dacă stacking indisponibil
    stacking_action = 'standard'
    uncertainty = 0.1

    if _STACKING is not None:
        try:
            lom_p = float(sig.get('lom_prob', sig.get('lom_score', sig.get('nom_prob', sig.get('nom_score', 0.5)))))
            nom_p = float(sig.get('nom_prob', sig.get('nom_score', sig.get('lom_prob', sig.get('lom_score', 0.5)))))
            enc   = int(sig.get('regime_enc', 2))
            res   = _STACKING.predict(
                lom_prob=lom_p, nom_prob=nom_p,
                regime_enc=enc, recent_wr=_RECENT_WR[-10:] if _RECENT_WR else None
            )
            stacking_score  = float(res.get('final_prob', (lom_p + nom_p) / 2))
            stacking_action = res.get('action', 'standard')
            if stacking_action == 'skip':
                stacking_action = 'standard'   # checker e filtrul primar
            uncertainty = float(res.get('uncertainty', 0.1))
        except Exception as e:
            logger.warning(f"Stacking predict error: {e}")

    # ── 3. Composite score ───────────────────────────────────────────────────
    composite = round(0.45 * orig_score + 0.25 * stacking_score + 0.30 * sweep_prob, 3)

    sweep_confirmed = sweep_raw >= sweep_thr   # Fix: definit ÎNAINTE de a fi folosit

    sig['ml_score']         = composite
    sig['stacking_score']   = round(stacking_score, 3)
    sig['sweep_prob']       = round(sweep_raw, 3)       # raw prob (pentru dashboard/logs)
    sig['sweep_confirmed']  = sweep_confirmed           # bool: raw >= threshold
    sig['sweep_threshold']  = sweep_thr                 # threshold aplicat
    sig['stacking_action']  = stacking_action
    sig['sizing_mult']      = 1.0   # mereu 1 contract
    sig['uncertainty']      = round(uncertainty, 3)
    logger.info(
        f"Composite [{sig_type}]: ml={orig_score:.3f} stk={stacking_score:.3f} "
        f"swp={sweep_raw:.3f}[{regime_str}] thr={sweep_thr} "
        f"{'✅confirmed' if sweep_confirmed else '⬜below_thr→0.5'} "
        f"→ final={composite:.3f} action={stacking_action}"
    )
    return sig


def record_trade_outcome(won: bool):
    """
    Înregistrează rezultatul tranzacției pentru contextul RL.
    Apelat din bridge_api.py când tranzacția se închide.
    """
    global _RECENT_WR
    _RECENT_WR.append(1 if won else 0)
    if len(_RECENT_WR) > 50:
        _RECENT_WR = _RECENT_WR[-50:]


# ── Integrare în bridge_api.py ─────────────────────────────────────────────────
def gate_verdict(db_path: str, now_utc: datetime = None) -> tuple[str, dict | None]:
    """
    Returnează (verdict_str, signal_dict).

    verdict_str: "TRADE" sau "WAIT"
    signal_dict: câmpuri semnal sau None

    Verifică în ordine prioritate:
      1. check_setup()     — ICT LON/NY setup clasic (FVG + displacement + sweep)
      2. check_ts_setup()  — Turtle Soup (4H sweep + MSS pe 1m, W1 și W2)
      3. check_lom_setup() — London Open Manipulation ML (AUC=0.890, WR=99%)
      4. check_nom_setup() — NY Open Manipulation ML    (AUC=0.829, WR=100%)
      5. check_dsm_setup() — Double Sweep Model LON+NY  (AUC=0.713, WR=76.4%)

    Dacă mai multe returnează semnal → se alege cel cu ML score / RR mai mare.
    DSM are prioritate mai mică decât NOM/LOM (scor mai mic) dar acoperă
    setup-urile de continuare post-sweep pe care celelalte nu le prind.
    """
    if daily_trades_taken() >= 2:
        return "WAIT", None

    if now_utc is None:
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)

    # ── Regime routing (multi-scale: classifier + Bayesian + HMM macro) ─────────
    _hhmm = now_utc.hour * 100 + now_utc.minute
    try:
        _r_info = _get_regime_full(db_path, now_utc)
    except Exception as _re:
        logger.warning(f"get_regime error: {_re}")
        _r_info = {'regime':'UNKNOWN','regime_enc':2,'regime_prob':0.0,
                   'entropy':0.5,'macro':'BULL','macro_prob':0.5,
                   'bayesian_n':0,'source':'fallback'}

    regime      = _r_info['regime']
    regime_prob = _r_info['regime_prob']
    regime_enc  = _r_info['regime_enc']
    macro       = _r_info.get('macro', 'BULL')
    bayes_n     = _r_info.get('bayesian_n', 0)

    logger.info(
        f"🌐 Regime: {regime}({regime_enc}) p={regime_prob:.2f} "
        f"macro={macro} bayes_n={bayes_n} src={_r_info.get('source','?')} "
        f"@ {_hhmm:04d} UTC"
    )

    # CONSOLIDATION: skip automat
    if regime == 'CONSOLIDATION' and regime_prob >= 0.80:
        logger.info("⏸️  Gate: CONSOLIDATION detectat → WAIT automat")
        return "WAIT", None

    # Routing cu regime_enc transmis la checkere
    if regime == 'PRE_EXPANSION' and regime_prob >= 0.65:
        sig_ict = None
        sig_ts  = None
        sig_lom = check_lom_setup(db_path, now_utc, regime=regime, regime_prob=regime_prob, regime_enc=regime_enc)
        sig_nom = check_nom_setup(db_path, now_utc, regime=regime, regime_prob=regime_prob, regime_enc=regime_enc)
        sig_dsm = check_dsm_setup(db_path, now_utc)
    elif regime in ('EXPANSION', 'RETRACEMENT') and regime_prob >= 0.65:
        sig_ict = check_setup(db_path, now_utc)
        sig_ts  = check_ts_setup(db_path, now_utc)
        sig_lom = None
        sig_nom = None
        sig_dsm = check_dsm_setup(db_path, now_utc)
    elif regime == 'DISTRIBUTION' and regime_prob >= 0.65:
        sig_ict = check_setup(db_path, now_utc)
        sig_ts  = check_ts_setup(db_path, now_utc)
        sig_lom = None
        sig_nom = None
        sig_dsm = None
    else:
        # UNKNOWN sau prob mică → rulează tot (fallback)
        sig_ict = check_setup(db_path, now_utc)
        sig_ts  = check_ts_setup(db_path, now_utc)
        sig_lom = check_lom_setup(db_path, now_utc, regime=regime, regime_prob=regime_prob, regime_enc=regime_enc)
        sig_nom = check_nom_setup(db_path, now_utc, regime=regime, regime_prob=regime_prob, regime_enc=regime_enc)
        sig_dsm = check_dsm_setup(db_path, now_utc)

    # ── Stacking: aplică pipeline LOM/NOM → final_prob ───────────────────────
    if sig_lom is not None:
        sig_lom.setdefault('nom_prob', sig_lom.get('lom_prob', sig_lom.get('lom_score', 0.5)))
        sig_lom = _apply_stacking(sig_lom, 'LOM')
    if sig_nom is not None:
        sig_nom.setdefault('lom_prob', sig_nom.get('nom_prob', sig_nom.get('nom_score', 0.5)))
        sig_nom = _apply_stacking(sig_nom, 'NOM')

    # Log diagnostic
    logger.info(
        f"Gate semnale @ {_hhmm:04d} UTC [{regime}]: "
        f"ICT={'✅' if sig_ict else '❌'} "
        f"TS={'✅' if sig_ts else '❌'} "
        f"LOM={'✅' if sig_lom else '❌'} "
        f"NOM={'✅' if sig_nom else '❌'} "
        f"DSM={'✅' if sig_dsm else '❌'}"
    )

    # Colectează toate semnalele valide
    candidates = []
    if sig_ict is not None:
        sig_ict.setdefault('ml_score', float(sig_ict.get('rr', 1.5)))
        candidates.append(('ICT', sig_ict))
    if sig_ts is not None:
        sig_ts.setdefault('ml_score', float(sig_ts.get('rr', 1.5)))
        candidates.append(('TS', sig_ts))
    if sig_lom is not None:
        candidates.append(('LOM', sig_lom))
    if sig_nom is not None:
        candidates.append(('NOM', sig_nom))
    if sig_dsm is not None:
        candidates.append(('DSM', sig_dsm))

    if not candidates:
        return "WAIT", None

    if len(candidates) == 1:
        name, sig = candidates[0]
        logger.info(f"Gate: semnal {name} ales (singurul valid)")
        sig['regime']       = regime
        sig['regime_enc']   = regime_enc
        sig['regime_prob']  = round(regime_prob, 3)
        sig['macro_regime'] = macro
        sig['checker_name'] = name
        return "TRADE", sig

    # Alege semnalul cu ml_score cel mai mare
    # LOM/NOM/DSM: folosesc ml_score direct (0-1)
    # ICT/TS: RR normalizat la ~0.5-0.8 range
    def score_key(item):
        name, sig = item
        if name in ('LOM', 'NOM', 'DSM'):
            return float(sig.get('ml_score', 0.5))
        return float(sig.get('rr', 1.5)) / 3.0

    candidates.sort(key=score_key, reverse=True)
    best_name, chosen = candidates[0]
    names_str = ', '.join(f"{n}(score={s.get('ml_score', s.get('rr','?'))})" for n, s in candidates)
    logger.info(f"Gate: {len(candidates)} semnale [{names_str}] → ales {best_name}")
    # Injectăm regime info în semnal → meta_scorer_v1 îl folosește
    chosen['regime']       = regime
    chosen['regime_enc']   = regime_enc
    chosen['regime_prob']  = round(regime_prob, 3)
    chosen['macro_regime'] = macro
    chosen['checker_name'] = best_name
    return "TRADE", chosen


def _gate_single_chosen(candidates, regime, regime_prob):
    """Helper: ales singurul semnal + injectare regime."""
    name, sig = candidates[0]
    logger.info(f"Gate: semnal {name} ales (singurul valid)")
    sig['regime']       = regime
    sig['regime_prob']  = round(regime_prob, 3)
    sig['checker_name'] = name
    return "TRADE", sig


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "/Users/mario/Desktop/Aladin/mario_trading.db"
    from datetime import timezone
    now = datetime.now(timezone.utc)
    print(f"\n🔍 ICT Gate v3 — check la {now.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"   Trades azi: {daily_trades_taken()}")
    v, sig = gate_verdict(db, now)
    if v == "TRADE":
        print(f"\n   ✅ SEMNAL VALID:")
        for k, val in sig.items():
            print(f"      {k}: {val}")
    else:
        print(f"\n   ⏳ WAIT — condiţii ICT v3 neîndeplinite")
