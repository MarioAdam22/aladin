"""
ALADIN — REVERSAL MODEL BACKTEST 12 ANI
════════════════════════════════════════════════════════════════════════════════
Simulează reversal_model.json pe datele istorice NQ 1M (2015-2026).
Sesiuni: LON (09:00-12:00 CET) + NY (15:00-18:00 CET)

Logica:
  1. Încarcă market_data an cu an (evitare OOM)
  2. Calculează features reversal pentru fiecare bară din sesiune
  3. Rulează reversal_model.json pe fiecare bară
  4. Entry când: P(REVERSAL) >= PROBA_THR + swept_above/swept_below confirmed
  5. Exit: SL / BE la 0.5R / trail de la 2R (același RM ca OR backtest)
  6. Max 1 trade activ per sesiune (LON / NY independent)

Key times CET (= DB timezone):
  LON: 08:00-11:00 CET  (= 09:00-12:00 Romania)
  NY:  14:00-17:00 CET  (= 15:00-18:00 Romania)

Key times pentru reversal (CET):
  16:10 CET = NY Macro (17:10 Romania)
  15:50 CET = LDN-NY overlap (16:50 Romania)
  15:00 CET = NY Open (16:00 Romania)
  13:00 CET = LDN Close (14:00 Romania)
  10:00 CET = LON KZ End (11:00 Romania)
"""

import sqlite3, json, time, sys
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

DIR = Path(__file__).parent
DB  = DIR / "mario_trading.db"
MODEL_PATH    = DIR / "reversal_model.json"
FEATURES_PATH = DIR / "reversal_features.json"
OUT_CSV       = DIR / "backtest_reversal_10y_trades.csv"

# ── CONFIG ───────────────────────────────────────────────────────────────────
PROBA_THR         = 0.60   # threshold probabilitate reversal
SL_ATR_MULT       = 0.65   # SL = 0.65 × ATR
SL_MIN_PTS        = 8.0
SL_MAX_PTS        = 20.0
MAX_HOLD_BARS     = 120    # max 2 ore = 120 bare 1M
TRAIL_START_R     = 2.0    # start trail la 2R
BE_AT_R           = 0.5    # SL→BE la 0.5R
POINT_VALUE       = 20.0   # $20/punct NQ

# Sesiuni în CET (ora din DB)
SESSIONS = {
    "LON": (8.0,  11.0),   # 08:00-11:00 CET = 09:00-12:00 Romania
    "NY":  (14.0, 17.0),   # 14:00-17:00 CET = 15:00-18:00 Romania
}

# Key level threshold (în ATR)
KEY_LEVEL_ATR_THR = 0.30

# Key times CET
KEY_TIMES_CET = [16 + 10/60, 15 + 50/60, 15.0, 13.0, 10.0, 9.5]


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
def load_reversal_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    with open(FEATURES_PATH) as f:
        meta = json.load(f)
    features = meta["features"]
    print(f"✅ Reversal model: {len(features)} features | AUC={meta.get('auc', '?')}")
    return model, features


# ── LOAD BARS ─────────────────────────────────────────────────────────────────
def _load_year(year: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    avail = {r[1] for r in conn.execute("PRAGMA table_info(market_data)")}
    want = [
        "timestamp","open","high","low","close","volume","atr_14",
        "h1_hi","h1_lo","h4_hi","h4_lo","asia_hi","asia_lo",
        "val","vah","poc_level","lw_lo","lw_hi",
        "p_hi","p_lo","lon_hi","lon_lo",
        "dist_pdh","dist_pdl","dist_poc","inside_va",
        "has_displacement","rvol","adx_14","hurst","dom_ratio",
        "fisher_transform","fft_cycle","garch_vol",
        "is_smt_bearish","is_smt_bullish","true_open",
    ]
    cols = [c for c in want if c in avail]
    df = pd.read_sql_query(
        f"SELECT {','.join(cols)} FROM market_data "
        f"WHERE timestamp >= '{year-1}-12-01' AND timestamp < '{year+1}-02-01' "
        f"ORDER BY timestamp", conn)
    conn.close()

    df["ts"]       = pd.to_datetime(df["timestamp"])
    df["date"]     = df["ts"].dt.date
    df["hour_dec"] = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    df.drop(columns=["timestamp"], inplace=True)

    # Fill defaults
    df["atr_14"] = df["atr_14"].fillna(10.0).replace(0, 10.0)
    df["volume"] = df["volume"].fillna(0)
    for c in ["h1_hi","h1_lo","h4_hi","h4_lo","asia_hi","asia_lo",
              "val","vah","poc_level","lw_lo","lw_hi","p_hi","p_lo",
              "lon_hi","lon_lo","dist_pdh","dist_pdl","dist_poc",
              "inside_va","has_displacement","rvol","adx_14","hurst",
              "dom_ratio","fisher_transform","fft_cycle","garch_vol",
              "is_smt_bearish","is_smt_bullish","true_open"]:
        if c not in df.columns: df[c] = 0.0
        else: df[c] = df[c].fillna(0.0)
    df["rvol"]      = df["rvol"].replace(0, 1.0)
    df["dom_ratio"] = df["dom_ratio"].replace(0, 1.0)

    # PDH / PDL / prev week hi/lo
    daily = (df.groupby("date", sort=True)
               .agg(day_hi=("high","max"), day_lo=("low","min"), avg_atr=("atr_14","mean"))
               .reset_index())
    daily["pdh"]     = daily["day_hi"].shift(1)
    daily["pdl"]     = daily["day_lo"].shift(1)
    daily["atr_10d"] = daily["avg_atr"].rolling(10, min_periods=3).mean()
    df = df.merge(daily[["date","pdh","pdl","atr_10d"]], on="date", how="left")
    df["pdh"] = df["pdh"].fillna(0)
    df["pdl"] = df["pdl"].fillna(0)
    return df


# ── FEATURE ENGINEERING (replicat din train_reversal_model.py) ───────────────
def compute_reversal_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculează features reversal pentru toate barele simultan (vectorizat).
    """
    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    open_  = df["open"].values.astype(float)
    atr_v  = df["atr_14"].values.astype(float)
    vol_v  = df["volume"].values.astype(float)
    n      = len(df)

    feat = {}

    # ── KEY LEVELS ────────────────────────────────────────────────────────────
    pdh  = df["pdh"].values.astype(float)
    pdl  = df["pdl"].values.astype(float)
    pwh  = df["p_hi"].values.astype(float)
    pwl  = df["p_lo"].values.astype(float)
    vah  = df["vah"].values.astype(float)
    val  = df["val"].values.astype(float)
    poc  = df["poc_level"].values.astype(float)
    ldh  = df["lon_hi"].values.astype(float)
    ldl  = df["lon_lo"].values.astype(float)
    ash  = df["asia_hi"].values.astype(float)
    asl  = df["asia_lo"].values.astype(float)

    safe_atr = np.maximum(atr_v, 0.1)

    # Distanțe la nivele (în ATR)
    feat["dist_pdh_atr"]    = np.abs(close - pdh) / safe_atr
    feat["dist_pdl_atr"]    = np.abs(close - pdl) / safe_atr
    feat["dist_pwh_atr"]    = np.abs(close - np.where(pwh > 0, pwh, close)) / safe_atr
    feat["dist_pwl_atr"]    = np.abs(close - np.where(pwl > 0, pwl, close)) / safe_atr
    feat["dist_vah_atr"]    = np.abs(close - np.where(vah > 0, vah, close)) / safe_atr
    feat["dist_val_atr"]    = np.abs(close - np.where(val > 0, val, close)) / safe_atr
    feat["dist_ldn_hi_atr"] = np.abs(close - np.where(ldh > 0, ldh, close)) / safe_atr
    feat["dist_ldn_lo_atr"] = np.abs(close - np.where(ldl > 0, ldl, close)) / safe_atr
    feat["dist_asia_hi_atr"]= np.abs(close - np.where(ash > 0, ash, close)) / safe_atr
    feat["dist_asia_lo_atr"]= np.abs(close - np.where(asl > 0, asl, close)) / safe_atr
    feat["dist_poc_atr"]    = np.abs(close - np.where(poc > 0, poc, close)) / safe_atr

    # Near any level
    all_dists = np.stack([
        feat["dist_pdh_atr"], feat["dist_pdl_atr"],
        feat["dist_pwh_atr"], feat["dist_pwl_atr"],
        feat["dist_vah_atr"], feat["dist_val_atr"],
        feat["dist_ldn_hi_atr"], feat["dist_ldn_lo_atr"],
    ], axis=1)
    feat["near_any_level"] = (all_dists.min(axis=1) <= KEY_LEVEL_ATR_THR).astype(float)

    # Level type (care nivel e cel mai apropiat)
    level_prices = np.stack([pdh, pdl, pwh, pwl, vah, val, ldh, ldl, ash, asl, poc], axis=1)
    level_dists  = np.abs(close[:, None] - level_prices) / safe_atr[:, None]
    nearest_idx  = level_dists.argmin(axis=1)
    # 0=pdh,1=pdl,2=pwh,3=pwl → type 1; 4=vah → 3; 5=val → 4; 6=ldh,7=ldl → 5; 8=ash,9=asl → 6
    type_map = {0:1, 1:2, 2:1, 3:2, 4:3, 5:4, 6:5, 7:5, 8:6, 9:6, 10:0}
    feat["level_type"] = np.array([type_map.get(int(idx),0) for idx in nearest_idx], dtype=float)

    feat["above_level"] = (close > np.where(pdh > 0, pdh, close - 999)).astype(float)
    feat["below_level"] = (close < np.where(pdl > 0, pdl, close + 999)).astype(float)

    # ── KEY TIMES ─────────────────────────────────────────────────────────────
    hour_dec = df["hour_dec"].values
    mins_to_key = np.full(n, 999.0)
    key_type_arr = np.zeros(n)
    for kt_i, kt in enumerate(KEY_TIMES_CET):
        diff = np.abs(hour_dec - kt) * 60.0
        mask = diff < mins_to_key
        mins_to_key = np.where(mask, diff, mins_to_key)
        key_type_arr = np.where(mask, kt_i + 1, key_type_arr)

    feat["mins_to_key_time"] = np.clip(mins_to_key, 0, 120)
    feat["near_key_time"]    = (mins_to_key <= 15).astype(float)
    feat["key_time_type"]    = key_type_arr
    feat["time_in_window"]   = np.where(mins_to_key <= 15, 15 - mins_to_key, 0)

    # ── SWEEP DETECTION ───────────────────────────────────────────────────────
    swept_above = np.zeros(n)
    swept_below = np.zeros(n)

    all_levels = np.stack([pdh, pwh, vah, ldh, ash], axis=1)
    all_lows   = np.stack([pdl, pwl, val, ldl, asl], axis=1)

    for j in range(2, min(n, n)):
        for lv in range(all_levels.shape[1]):
            lv_price = all_levels[j, lv]
            if lv_price <= 0:
                continue
            # Swept above: high a depășit nivelul dar close a revenit sub el
            if high[j] > lv_price and close[j] < lv_price:
                swept_above[j] = 1.0
            # Verifică și 1-2 bare în urmă
            if j >= 2:
                if high[j-1] > lv_price and close[j] < lv_price:
                    swept_above[j] = 1.0
                if high[j-2] > lv_price and close[j] < lv_price:
                    swept_above[j] = 1.0

        for lv in range(all_lows.shape[1]):
            lv_price = all_lows[j, lv]
            if lv_price <= 0:
                continue
            if low[j] < lv_price and close[j] > lv_price:
                swept_below[j] = 1.0
            if j >= 2:
                if low[j-1] < lv_price and close[j] > lv_price:
                    swept_below[j] = 1.0
                if low[j-2] < lv_price and close[j] > lv_price:
                    swept_below[j] = 1.0

    feat["swept_above_level"] = swept_above
    feat["swept_below_level"] = swept_below

    # Sweep wick ratio
    wick_above = np.maximum(high - np.maximum(close, open_), 0)
    wick_below = np.maximum(np.minimum(close, open_) - low, 0)
    feat["sweep_wick_ratio"]   = np.clip(np.where(swept_above > 0, wick_above, wick_below) / safe_atr, 0, 5)
    feat["bars_since_sweep"]   = np.zeros(n)  # simplificat
    feat["reclaimed_after_sweep"] = np.where((swept_above + swept_below) > 0, 1.0, 0.0)

    # ── CANDLE STRUCTURE ─────────────────────────────────────────────────────
    rng = np.maximum(high - low, 0.01)
    body = close - open_
    feat["body_dir"]           = np.clip(body / rng, -1, 1)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    feat["wick_bias"]          = np.clip((upper_wick - lower_wick) / rng, -1, 1)
    feat["upper_wick_atr"]     = np.clip(upper_wick / safe_atr, 0, 5)
    feat["lower_wick_atr"]     = np.clip(lower_wick / safe_atr, 0, 5)

    # Sharp reversal / rejection
    sharp = np.zeros(n)
    for i in range(1, n):
        if body[i] * body[i-1] < 0:  # direction changed
            if abs(body[i]) > abs(body[i-1]) * 0.8:
                sharp[i] = 1.0
    feat["sharp_reversal_bar"] = sharp

    rej = np.zeros(n)
    for i in range(n):
        max_wick = max(upper_wick[i], lower_wick[i])
        if max_wick > 2 * abs(body[i]) and rng[i] > 0.5 * safe_atr[i]:
            rej[i] = 1.0
    feat["rejection_candle"] = rej

    outside = np.zeros(n)
    for i in range(1, n):
        if high[i] > high[i-1] and low[i] < low[i-1]:
            outside[i] = 1.0
    feat["outside_bar"] = outside

    # ── VOLATILITY ───────────────────────────────────────────────────────────
    # ATR percentile pe rolling 252 bare (approx 1 an)
    atr_series = pd.Series(atr_v)
    feat["atr_percentile"]  = atr_series.rank(pct=True).values
    log_ret = np.log(np.maximum(close[1:] / np.maximum(close[:-1], 0.01), 1e-8))
    log_ret = np.concatenate([[0], log_ret])
    rv = pd.Series(np.abs(log_ret)).rolling(14, min_periods=3).std().fillna(0).values
    feat["realized_vol"]    = np.clip(rv * 100, 0, 5)
    vol_ma20 = pd.Series(vol_v).rolling(20, min_periods=5).mean().fillna(vol_v.mean() + 1)
    feat["vol_spike"]       = (vol_v > 1.5 * vol_ma20.values).astype(float)
    ranges5 = pd.Series(high - low).rolling(5, min_periods=1).sum().values
    feat["range_atr_ratio"] = np.clip(ranges5 / (5 * safe_atr), 0.1, 3)

    # ── MOMENTUM ─────────────────────────────────────────────────────────────
    close_s = pd.Series(close)
    feat["momentum_5"]  = np.clip((close - close_s.shift(5).fillna(close[0]).values) / safe_atr, -10, 10)
    feat["momentum_15"] = np.clip((close - close_s.shift(15).fillna(close[0]).values) / safe_atr, -10, 10)
    feat["slope_h1"]    = np.clip((close - close_s.shift(60).fillna(close[0]).values) / (close_s.shift(60).fillna(close[0]).abs().values + 1e-8), -0.1, 0.1)
    feat["h4_momentum"] = np.clip((close - close_s.shift(240).fillna(close[0]).values) / (close_s.shift(240).fillna(close[0]).abs().values + 1e-8), -0.05, 0.05)

    # H4 HH/HL
    h4_hh_hl = np.zeros(n)
    for i in range(240, n):
        seg = close[max(0,i-240):i]
        if len(seg) >= 2:
            if seg[-1] > seg[-120:].max() * 0.999 and seg[-1] > seg[0]:
                h4_hh_hl[i] = 1.0
            elif seg[-1] < seg[-120:].min() * 1.001 and seg[-1] < seg[0]:
                h4_hh_hl[i] = -1.0
    feat["h4_hh_hl"] = h4_hh_hl

    # ── VOLUME PROFILE ───────────────────────────────────────────────────────
    feat["inside_va"]      = df["inside_va"].values.astype(float)
    feat["above_vah"]      = (close > np.where(vah > 0, vah, close + 999)).astype(float)
    feat["below_val"]      = (close < np.where(val > 0, val, close - 999)).astype(float)
    poc_s = np.where(poc > 0, poc, close)
    feat["dist_poc_signed"]= np.clip((close - poc_s) / safe_atr, -10, 10)

    # ── SESSION CONTEXT ───────────────────────────────────────────────────────
    feat["is_london"]  = ((hour_dec >= 8.0)  & (hour_dec < 11.0)).astype(float)
    feat["is_ny"]      = ((hour_dec >= 14.0) & (hour_dec < 17.0)).astype(float)
    feat["is_pre_ny"]  = ((hour_dec >= 15.0) & (hour_dec < 15.5)).astype(float)
    feat["hour_sin"]   = np.sin(2 * np.pi * hour_dec / 24.0)
    feat["hour_cos"]   = np.cos(2 * np.pi * hour_dec / 24.0)

    # ── STRUCTURE BREAK ──────────────────────────────────────────────────────
    feat["broke_pdh"]  = (close > np.where(pdh > 0, pdh, close - 999)).astype(float)
    feat["broke_pdl"]  = (close < np.where(pdl > 0, pdl, close + 999)).astype(float)
    feat["broke_pwh"]  = (close > np.where(pwh > 0, pwh, close - 999)).astype(float)
    feat["broke_pwl"]  = (close < np.where(pwl > 0, pwl, close + 999)).astype(float)
    feat["broke_vah"]  = (close > np.where(vah > 0, vah, close - 999)).astype(float)
    feat["broke_val"]  = (close < np.where(val > 0, val, close - 999)).astype(float)

    liq_above = (feat["broke_pdh"] + feat["broke_pwh"] + feat["broke_vah"])
    liq_below = (feat["broke_pdl"] + feat["broke_pwl"] + feat["broke_val"])
    feat["liq_above_count"] = np.clip(liq_above, 0, 5)
    feat["liq_below_count"] = np.clip(liq_below, 0, 5)

    return pd.DataFrame(feat, index=df.index)


# ── SIMULATE TRADE ────────────────────────────────────────────────────────────
def simulate_trade(bars_future: pd.DataFrame, direction: str,
                   entry_px: float, atr: float) -> dict:
    """Simulare trade cu RM: SL→BE la 0.5R, trail de la 2R."""
    risk = float(np.clip(atr * SL_ATR_MULT, SL_MIN_PTS, SL_MAX_PTS))
    sl   = entry_px - risk if direction == "LONG" else entry_px + risk
    be_trigger = entry_px + 0.5*risk if direction == "LONG" else entry_px - 0.5*risk
    be_moved = False
    mae = 0.0
    mfe = 0.0

    for bar_idx, (_, bar) in enumerate(bars_future.iterrows()):
        if bar_idx >= MAX_HOLD_BARS:
            exit_px = float(bar["close"])
            reason  = "MAX_HOLD"
            break

        bh = float(bar["high"])
        bl = float(bar["low"])
        bc = float(bar["close"])

        # MAE / MFE
        if direction == "LONG":
            mfe = max(mfe, bh - entry_px)
            mae = max(mae, entry_px - bl)
        else:
            mfe = max(mfe, entry_px - bl)
            mae = max(mae, bh - entry_px)

        # BE logic
        if not be_moved:
            if direction == "LONG" and bh >= be_trigger:
                sl = entry_px; be_moved = True
            elif direction == "SHORT" and bl <= be_trigger:
                sl = entry_px; be_moved = True

        # Trail logic (de la 2R)
        trail_atr_pct = {2.0: 1.5, 3.0: 1.2, 4.0: 1.0, 5.0: 0.8}
        for r_level, atr_pct in sorted(trail_atr_pct.items()):
            r_reached = entry_px + r_level * risk if direction == "LONG" else entry_px - r_level * risk
            if direction == "LONG" and bh >= r_reached:
                new_sl = bh - atr * atr_pct
                sl = max(sl, new_sl)
            elif direction == "SHORT" and bl <= r_reached:
                new_sl = bl + atr * atr_pct
                sl = min(sl, new_sl)

        # SL check
        if direction == "LONG" and bl <= sl:
            exit_px = sl
            pts = exit_px - entry_px
            reason = "BE" if be_moved and pts == 0 else ("SL_INITIAL" if pts < 0 else f"TRAIL_{pts/risk:.1f}R")
            break
        elif direction == "SHORT" and bh >= sl:
            exit_px = sl
            pts = entry_px - exit_px
            reason = "BE" if be_moved and pts == 0 else ("SL_INITIAL" if pts < 0 else f"TRAIL_{pts/risk:.1f}R")
            break
    else:
        exit_px = float(bars_future.iloc[-1]["close"]) if len(bars_future) > 0 else entry_px
        reason = "MAX_HOLD"

    pts = (exit_px - entry_px) if direction == "LONG" else (entry_px - exit_px)
    r_mult = pts / risk if risk > 0 else 0

    # Reason fix
    if r_mult >= 2.0:
        reason = f"TRAIL_{r_mult:.1f}R"
    elif r_mult > 0:
        reason = "WIN_SMALL"
    elif r_mult == 0:
        reason = "BE"
    elif reason != "MAX_HOLD":
        reason = "SL_INITIAL"

    return {
        "exit_px": round(exit_px, 2),
        "pts":     round(pts, 2),
        "r_mult":  round(r_mult, 3),
        "reason":  reason,
        "mae":     round(mae, 2),
        "mfe":     round(mfe, 2),
        "risk":    round(risk, 2),
        "bars_held": min(bar_idx + 1, MAX_HOLD_BARS),
    }


# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────
def run_backtest():
    print("═" * 60)
    print("  REVERSAL MODEL BACKTEST — 12 ANI NQ")
    print("═" * 60)

    model, features = load_reversal_model()

    conn = sqlite3.connect(DB)
    years = [r[0] for r in conn.execute(
        "SELECT DISTINCT CAST(strftime('%Y',timestamp) AS INT) "
        "FROM market_data ORDER BY 1"
    ).fetchall()]
    conn.close()
    print(f"📅 Ani: {years[0]}–{years[-1]} ({len(years)} ani)")

    all_trades = []

    for yr in years:
        t0 = time.time()
        df_yr = _load_year(yr)

        # Filtrăm doar barele din sesiunile LON + NY
        session_mask = np.zeros(len(df_yr), dtype=bool)
        for sess, (s_start, s_end) in SESSIONS.items():
            session_mask |= ((df_yr["hour_dec"] >= s_start) & (df_yr["hour_dec"] <= s_end))
        df_sess = df_yr[session_mask].reset_index(drop=True)
        if len(df_sess) < 100:
            continue

        # Calculăm features vectorizat
        feat_df = compute_reversal_features_vectorized(df_sess)

        # Asigurăm că avem toate features
        for f in features:
            if f not in feat_df.columns:
                feat_df[f] = 0.0

        X = feat_df[features].fillna(0).values
        proba = model.predict_proba(X)[:, 1]

        # Entry conditions: P >= THR + (swept sau near key level + near key time)
        entry_mask = (
            (proba >= PROBA_THR) &
            (
                (feat_df["swept_above_level"].values > 0) |
                (feat_df["swept_below_level"].values > 0) |
                (
                    (feat_df["near_any_level"].values > 0) &
                    (feat_df["near_key_time"].values > 0)
                )
            )
        )

        # Simulare trade per zi + sesiune
        df_sess["_proba"]      = proba
        df_sess["_entry_mask"] = entry_mask
        df_sess["_swept_up"]   = feat_df["swept_above_level"].values
        df_sess["_swept_dn"]   = feat_df["swept_below_level"].values

        trades_yr = []
        for date, day_df in df_sess.groupby("date"):
            for sess_name, (s_start, s_end) in SESSIONS.items():
                sess_df = day_df[
                    (day_df["hour_dec"] >= s_start) & (day_df["hour_dec"] <= s_end)
                ].reset_index(drop=True)
                if len(sess_df) < 10:
                    continue

                in_trade = False
                for i, row in sess_df.iterrows():
                    if in_trade:
                        continue
                    if not row["_entry_mask"]:
                        continue

                    # Direcție: swept_above → SHORT (reversal jos), swept_below → LONG
                    if row["_swept_up"] > 0:
                        direction = "SHORT"
                    elif row["_swept_dn"] > 0:
                        direction = "LONG"
                    else:
                        # near level + near key time: direcție bazată pe care parte suntem
                        direction = "SHORT" if row["close"] > row["poc_level"] else "LONG"

                    entry_px = float(row["close"])
                    entry_ts = str(row["ts"])
                    atr      = float(row["atr_14"])

                    # Barele viitoare din sesiune pentru simulare
                    future = sess_df.iloc[i+1:].reset_index(drop=True)
                    if len(future) < 3:
                        continue

                    result = simulate_trade(future, direction, entry_px, atr)
                    in_trade = True  # 1 trade / sesiune

                    trades_yr.append({
                        "date":      str(date),
                        "session":   sess_name,
                        "direction": direction,
                        "entry_ts":  entry_ts,
                        "entry_px":  entry_px,
                        "proba":     round(float(row["_proba"]), 4),
                        "atr":       round(atr, 2),
                        **result,
                    })

        all_trades.extend(trades_yr)
        n_yr = len(trades_yr)
        pnl_yr = sum(t["pts"] * POINT_VALUE for t in trades_yr)
        print(f"   ✓ {yr}: {n_yr:4d} trades | P&L ${pnl_yr:+,.0f} ({time.time()-t0:.1f}s)")

    # ── REZULTATE ─────────────────────────────────────────────────────────────
    if not all_trades:
        print("❌ 0 trades generate — verifică threshold sau features")
        return

    df_out = pd.DataFrame(all_trades)
    df_out["pnl_usd"] = df_out["pts"] * POINT_VALUE
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\n{'═'*60}")
    print(f"  REZULTATE FINALE — REVERSAL MODEL 12 ANI")
    print(f"{'═'*60}")
    n_days  = df_out["date"].nunique()
    n_total = len(df_out)
    print(f"  Total trades:     {n_total:,}")
    print(f"  Zile cu trade:    {n_days:,}")
    print(f"  Trades/zi:        {n_total/n_days:.2f}")
    print(f"\n  P&L total:        ${df_out.pnl_usd.sum():+,.0f}")
    print(f"  P&L mediu/trade:  ${df_out.pnl_usd.mean():+.0f}")
    print(f"  P&L mediu/zi:     ${df_out.groupby('date')['pnl_usd'].sum().mean():+.0f}")
    print(f"\n  WR:               {(df_out.r_mult > 0).mean():.1%}")
    print(f"  BE rate:          {(df_out.r_mult == 0).mean():.1%}")
    print(f"  SL rate:          {(df_out.r_mult < 0).mean():.1%}")
    print(f"  EV mediu/trade:   {df_out.r_mult.mean():+.3f}R")

    print(f"\n  DISTRIBUȚIE REASON:")
    for r, cnt in df_out["reason"].value_counts().items():
        print(f"    {r:20s}: {cnt:5d} ({100*cnt/n_total:.1f}%)")

    print(f"\n  PER SESIUNE:")
    for sess in ["LON","NY"]:
        d = df_out[df_out["session"]==sess]
        if len(d) == 0: continue
        print(f"    {sess}: {len(d):,} trades | EV={d.r_mult.mean():+.3f}R | "
              f"WR={(d.r_mult>0).mean():.1%} | P&L/zi ${d.groupby('date')['pnl_usd'].sum().mean():+.0f}")

    print(f"\n  TOP 5 CELE MAI BUNE ZILE:")
    daily_pnl = df_out.groupby("date")["pnl_usd"].sum().sort_values(ascending=False)
    for date, pnl in daily_pnl.head(5).items():
        print(f"    {date}: ${pnl:+,.0f}")

    print(f"\n  ZILE CU P&L > $500:  {(daily_pnl > 500).sum()} ({100*(daily_pnl>500).mean():.1f}%)")
    print(f"  ZILE CU P&L > $1000: {(daily_pnl > 1000).sum()} ({100*(daily_pnl>1000).mean():.1f}%)")
    print(f"\n💾 Salvat: {OUT_CSV.name}")


if __name__ == "__main__":
    run_backtest()
