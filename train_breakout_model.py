"""
ALADIN v14.5 — BINARY BREAKOUT MODEL (label v14.3 + threshold coborât)
═══════════════════════════════════════════════════════════════════════════════
LABEL: identic cu v14.3 — REAL = reached 1R (mfe≥1R) without SL_INITIAL first

  REAL (30.6%): r_mult mean=+1.000R | WR=52.9% — breakout autentic
  FAKE (69.4%): r_mult mean=-0.482R | WR=3.9%  — reversă rapid la SL

SCHIMBARE față de v14: INFERENCE THRESHOLD coborât 0.65 → 0.45
  La 0.45 modelul lasă să treacă ~50% din breakouts (vs 26% la 0.65)
  Trades cu P(REAL) 0.45-0.65 sunt "mai puțin sigure" dar cu EV pozitiv
  → mai mult de 1 trade/zi pe prop firm

LOGICA: NQ face mișcări de 30+ pts zilnic (30 pts = 1.5-3R).
  Nu trebuie să filtram 74% — modelul știe și tipurile de breakout mai lente.
═══════════════════════════════════════════════════════════════════════════════
"""

import sqlite3, pandas as pd, numpy as np
import xgboost as xgb, json, sys, time
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report

DB  = Path(__file__).parent / "mario_trading.db"
DIR = Path(__file__).parent

# CSV cu rezultatele backtest NAKED ("let it run" RM) — sursa labelelor
BACKTEST_CSV = DIR / "backtest_orh_10y_trades.csv"

# Label: REAL = mfe_r >= 1.0 AND reason != 'SL_INITIAL'
# = prețul a ajuns 1R în direcție fără să bată SL-ul inițial
MFE_THR_R = 1.0     # intern pentru label computation

# Threshold probabilitate la inferență: 0.45 → mai multe trades (vs 0.65 în v14)
# La 0.45: ~50% din breakouts trec (vs 26% la 0.65)
# Borderline trades cu P=0.45-0.65 au EV pozitiv dar WR mai mic
PROBA_THR = 0.45

# Ore CET (UTC+1 iarna) = ora DB. LON Open Romania=10:30=CET 09:30
KZ = {
    "LON": {"start": 9.5,  "end": 11.5, "or_end": 10.0},  # OR: 09:30-10:00 CET = 10:30-11:00 Romania
    "NY":  {"start": 15.5, "end": 17.5, "or_end": 16.0},  # OR: 15:30-16:00 CET = 16:30-17:00 Romania
}

BREAKOUT_FEATURES = [
    # ── Candle structure ──────────────────────────────────────────────────────
    "body_pct", "upper_wick_pct", "lower_wick_pct", "close_strength",
    "is_strong_close", "bar_range_atr",
    # ── Volume ───────────────────────────────────────────────────────────────
    "vol_vs_or_avg",
    # ── OR context ───────────────────────────────────────────────────────────
    "or_width_atr", "or_width_pts",
    "min_after_or", "min_after_or_bin",
    # ── Structural breaks ────────────────────────────────────────────────────
    "broke_pdh", "broke_pdl",
    "broke_asia_hi", "broke_asia_lo",
    "or_swept_above", "or_swept_below",
    # ── Trend / momentum ─────────────────────────────────────────────────────
    "momentum_pre_or",
    "h1_trend_vs_break", "h4_trend_vs_break",
    # ── ATR / volatility ─────────────────────────────────────────────────────
    "atr_abs", "atr_vs_10d_avg",
    # ── Entry distance ───────────────────────────────────────────────────────
    "dist_above_orh_atr", "dist_below_orl_atr",
    # ── OR quality ───────────────────────────────────────────────────────────
    "or_candle_count", "or_close_vs_mid",
    # ── VP / structural (v14.1) ───────────────────────────────────────────────
    "inside_va", "dist_poc_atr", "sl_dist_atr", "above_vah", "below_val",
    # ── Noi features v14.5 ───────────────────────────────────────────────────
    "or_approach_count",   # de câte ori s-a apropiat OR de ORH/ORL (compresie)
    "vol_trend_in_or",     # slope volum în OR (crește = acumulare)
    "range_compression",   # ultimele 5 bare OR vs OR total (strângere)
    # ── v15.0 — order flow + regim de piață din DB ───────────────────────────
    "rvol_break",          # relative volume la bara de breakout
    "dom_ratio_break",     # DOM bid/ask ratio la breakout
    "adx_break",           # ADX — trend strength la breakout
    "hurst_break",         # Hurst exponent (>0.5=trending, <0.5=ranging)
    "fisher_break",        # Fisher transform — momentum/overbought
    "garch_ratio_break",   # GARCH vol / ATR — regim volatilitate
    "sample_entropy_brk",  # sample entropy — zgomot piață
    "acf_lag1_break",      # autocorrelation lag1 — persistență momentum
    "kalman_noise_break",  # Kalman noise — smoothness piață
    "fft_cycle_break",     # FFT cycle dominant
    "dist_vwap_atr",       # distanță față de VWAP / ATR (signed)
    "vwap_aligned",        # 1 = breakout în direcția VWAP
    "rvol_or_avg",         # medie relative volume în OR
    "dom_ratio_or",        # medie DOM ratio în OR
    "adx_or_avg",          # medie ADX în OR
    "bigbal_or",           # big trade balance în OR
    "above_true_open",     # 1 = entry > true open
    "dist_true_op_atr",    # distanță față de true open / ATR (signed)
    "broke_pw_hi",         # 1 = OR a depășit prev week high
    "broke_pw_lo",         # 1 = OR a spart prev week low
    "lon_swept",           # 1 = OR a luat lichiditatea London
    "has_disp_in_or",      # 1 = displacement în OR
    "dist_pdh_atr_db",     # dist PDH din DB / ATR
    "dist_pdl_atr_db",     # dist PDL din DB / ATR
]

# SL structural (copie din backtest_orh_10y.py)
_HARD_CAP_SL = 20.0

def _structural_sl_dist(direction, entry, atr, h1_lo, h1_hi, h4_lo, h4_hi,
                         asia_lo, asia_hi, val, vah, lw_lo, lw_hi):
    SL_MIN     = max(8.0, round(atr * 0.55, 1))
    SL_DEFAULT = min(_HARD_CAP_SL-1, max(SL_MIN+2.0, round(atr*0.85,1)))
    SL_MAX     = max(min(_HARD_CAP_SL, round(atr*1.20,1)), SL_MIN+1)
    SL_DEFAULT = min(SL_DEFAULT, SL_MAX)
    if direction == "LONG":
        cands = [x for x in [h1_lo,h4_lo,asia_lo,val,lw_lo] if 0 < x < entry-SL_MIN]
        dist  = (entry - max(cands)) if cands else SL_DEFAULT
    else:
        cands = [x for x in [h1_hi,h4_hi,asia_hi,vah,lw_hi] if x > entry+SL_MIN]
        dist  = (min(cands) - entry) if cands else SL_DEFAULT
    return float(np.clip(dist, SL_MIN, SL_MAX))


# ─────────────────────────────────────────────────────────────────────────────
# LOAD BARS — an cu an (evitare OOM în sandbox 3.9GB)
# ─────────────────────────────────────────────────────────────────────────────
def _load_year(year: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    avail = {r[1] for r in conn.execute("PRAGMA table_info(market_data)")}
    want  = ["timestamp","open","high","low","close","volume","atr_14",
             "h1_hi","h1_lo","h4_hi","h4_lo","asia_hi","asia_lo",
             "val","vah","poc_level","lw_lo","lw_hi",
             # ── features noi din DB (order flow + regim + VP extins) ──────────
             "rvol",             # relative volume (100% coverage)
             "dom_ratio",        # DOM bid/ask ratio (100% coverage)
             "adx_14",           # trend strength ADX (100% coverage)
             "dist_vwap",        # distanță față de VWAP în pts (100% coverage)
             "hurst",            # Hurst exponent — trending(>0.5) vs ranging(<0.5) (100%)
             "fisher_transform", # Fisher transform — momentum/overbought (100%)
             "garch_vol",        # GARCH volatility estimate (100% coverage)
             "sample_entropy",   # sample entropy — market noise level (98%)
             "acf_lag1",         # autocorrelation lag1 — momentum persistence (100%)
             "of_big_balance",   # big trade balance 0-1 (100% coverage)
             "true_open",        # true open price (100% coverage)
             "p_hi","p_lo",      # prev week high/low (99.9% coverage)
             "lon_hi","lon_lo",  # London session high/low (99.8% — util pt NY KZ)
             "has_displacement", # displacement bar (24.9% — binary ok)
             "dist_pdh","dist_pdl",  # distanță PDH/PDL din DB în pts (99.6%)
             "kalman_noise",     # Kalman noise level (100% coverage)
             "fft_cycle",        # FFT dominant cycle (100% coverage)
             ]
    cols  = [c for c in want if c in avail]
    df = pd.read_sql_query(
        f"SELECT {','.join(cols)} FROM market_data "
        f"WHERE timestamp >= '{year-1}-12-01' AND timestamp < '{year+1}-02-01' "
        f"ORDER BY timestamp", conn)
    conn.close()

    df["ts"]            = pd.to_datetime(df["timestamp"])
    df["date"]          = df["ts"].dt.date
    df["hour_dec"]      = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    df["minute_of_day"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df.drop(columns=["timestamp"], inplace=True)
    df["atr_14"] = df["atr_14"].fillna(9.0).replace(0, 9.0)
    df["volume"] = df["volume"].fillna(0)
    for c in ["h1_hi","h1_lo","h4_hi","h4_lo","asia_hi","asia_lo",
              "val","vah","poc_level","lw_lo","lw_hi",
              "rvol","dom_ratio","adx_14","dist_vwap","hurst",
              "fisher_transform","garch_vol","sample_entropy","acf_lag1",
              "of_big_balance","true_open","p_hi","p_lo",
              "lon_hi","lon_lo","has_displacement","dist_pdh","dist_pdl",
              "kalman_noise","fft_cycle"]:
        if c not in df.columns: df[c] = 0.0
        else: df[c] = df[c].fillna(0)
    # rvol și dom_ratio: valoare default 1.0 (neutru) nu 0
    df["rvol"]      = df["rvol"].replace(0, 1.0)
    df["dom_ratio"] = df["dom_ratio"].replace(0, 1.0)

    # PDH/PDL și ATR 10-day
    daily = (df.groupby("date", sort=True)
               .agg(day_hi=("high","max"), day_lo=("low","min"), avg_atr=("atr_14","mean"))
               .reset_index())
    daily["pdh"]     = daily["day_hi"].shift(1)
    daily["pdl"]     = daily["day_lo"].shift(1)
    daily["atr_10d"] = daily["avg_atr"].rolling(10, min_periods=3).mean()
    df = df.merge(daily[["date","pdh","pdl","atr_10d"]], on="date", how="left")
    return df


def load_all_yearly() -> pd.DataFrame:
    """Încarcă toate barele an cu an, combină. ~50MB/an = OK în sandbox."""
    conn = sqlite3.connect(DB)
    years = [r[0] for r in conn.execute(
        "SELECT DISTINCT CAST(strftime('%Y',timestamp) AS INT) FROM market_data ORDER BY 1"
    ).fetchall()]
    conn.close()
    print(f"📅 Ani: {years[0]}–{years[-1]} ({len(years)} ani)")
    frames = []
    for yr in years:
        t0 = time.time()
        frames.append(_load_year(yr))
        print(f"   ✓ {yr}: {len(frames[-1]):,} bare ({time.time()-t0:.1f}s)")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("ts").reset_index(drop=True)
    # De-duplicate (overlap Dec/Ian din ferestre adiacente)
    df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
    print(f"✅ Total: {len(df):,} bare unice | RAM: ~{df.memory_usage(deep=True).sum()/1e6:.0f}MB")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT BREAKOUTS — features fără label (labelul vine din backtest CSV)
# ─────────────────────────────────────────────────────────────────────────────
def extract_breakouts(df: pd.DataFrame, kz_name: str) -> pd.DataFrame:
    kz      = KZ[kz_name]
    records = []
    groups  = list(df.groupby("date", sort=True))
    total   = len(groups)
    print(f"\n🔍 {kz_name}: procesez {total} zile...")
    t0 = time.time()

    for i, (date, day_df) in enumerate(groups):
        if i % 300 == 0:
            print(f"   {i}/{total} zile | {len(records)} breakout-uri", flush=True)

        kz_bars = day_df[(day_df["hour_dec"] >= kz["start"]) &
                         (day_df["hour_dec"] <= kz["end"])]
        if len(kz_bars) < 15: continue

        or_bars = kz_bars[kz_bars["hour_dec"] < kz["or_end"]]
        post_or = kz_bars[kz_bars["hour_dec"] >= kz["or_end"]].reset_index(drop=True)
        if len(or_bars) < 5 or len(post_or) < 3: continue

        orh      = float(or_bars["high"].max())
        orl      = float(or_bars["low"].min())
        or_width = orh - orl
        atr_ref  = float(or_bars["atr_14"].median())
        if or_width < 1.0 or atr_ref < 1.0: continue

        # Primul breakout
        brk_idx = brk_dir = None
        for j, row in post_or.iterrows():
            if row["close"] > orh:   brk_idx=j; brk_dir="LONG";  break
            elif row["close"] < orl: brk_idx=j; brk_dir="SHORT"; break
        if brk_idx is None: continue

        brk = post_or.iloc[brk_idx]
        entry = float(brk["close"])

        # ── Features candle ───────────────────────────────────────────────────
        rng    = max(float(brk["high"]-brk["low"]), 0.1)
        body   = abs(float(brk["close"]-brk["open"]))
        cl_str = (float(brk["close"])-float(brk["low"]))/rng
        is_str = int(cl_str>=0.75) if brk_dir=="LONG" else int(cl_str<=0.25)
        or_vol = max(float(or_bars["volume"].mean()), 1.0)
        or_mid = (orh+orl)/2.0
        min_after = max(0, int(brk["minute_of_day"])-int(kz["or_end"]*60))
        tb = 0 if min_after<5 else(1 if min_after<15 else(2 if min_after<30 else(3 if min_after<45 else 4)))

        # ── Structural ───────────────────────────────────────────────────────
        pdh=float(brk.get("pdh",np.nan) or np.nan)
        pdl=float(brk.get("pdl",np.nan) or np.nan)
        ahi=float(brk.get("asia_hi",0) or 0)
        alo=float(brk.get("asia_lo",0) or 0)
        orhi=float(or_bars["high"].max()); orlo=float(or_bars["low"].min())
        b_pdh=int(not np.isnan(pdh) and orhi>pdh)
        b_pdl=int(not np.isnan(pdl) and orlo<pdl)
        b_ahi=int(ahi>0 and orhi>ahi)
        b_alo=int(alo>0 and orlo<alo)
        mom=float(np.clip((float(or_bars.iloc[-1]["close"])-float(or_bars.iloc[0]["close"]))/atr_ref,-5,5))

        # ── H1/H4 trend ──────────────────────────────────────────────────────
        h1h=float(brk.get("h1_hi",0) or 0); h1l=float(brk.get("h1_lo",0) or 0); h1t=0
        if h1h>0 and h1l>0:
            oh=float(or_bars.iloc[-1].get("h1_hi",0) or 0)+float(or_bars.iloc[-1].get("h1_lo",0) or 0)
            sl1=((h1h+h1l)/2-oh/2)/atr_ref if atr_ref>0 else 0
            h1t=1 if(brk_dir=="LONG" and sl1>0.1)or(brk_dir=="SHORT" and sl1<-0.1)else 0
        h4h=float(brk.get("h4_hi",0) or 0); h4l=float(brk.get("h4_lo",0) or 0); h4t=0
        if h4h>0 and h4l>0:
            oh4=float(or_bars.iloc[-1].get("h4_hi",0) or 0)+float(or_bars.iloc[-1].get("h4_lo",0) or 0)
            sl4=((h4h+h4l)/2-oh4/2)/atr_ref if atr_ref>0 else 0
            h4t=1 if(brk_dir=="LONG" and sl4>0.05)or(brk_dir=="SHORT" and sl4<-0.05)else 0

        a10=float(brk.get("atr_10d",atr_ref) or atr_ref)
        last_or=float(or_bars.iloc[-1]["close"])

        # ── VP features ──────────────────────────────────────────────────────
        val_px = float(brk.get("val",0) or 0)
        vah_px = float(brk.get("vah",0) or 0)
        poc_px = float(brk.get("poc_level",0) or 0)
        lw_lo  = float(brk.get("lw_lo",0) or 0)
        lw_hi  = float(brk.get("lw_hi",0) or 0)

        inside_va  = int(val_px>0 and vah_px>0 and val_px<=entry<=vah_px)
        dist_poc   = round(float(np.clip((entry-poc_px)/atr_ref if poc_px>0 else 0,-10,10)),4)
        above_vah  = int(brk_dir=="LONG"  and vah_px>0 and entry>vah_px)
        below_val  = int(brk_dir=="SHORT" and val_px>0 and entry<val_px)

        sl_dist = _structural_sl_dist(brk_dir, entry, atr_ref,
                                       h1l,h1h,h4l,h4h,alo,ahi,val_px,vah_px,lw_lo,lw_hi)
        sl_dist_atr = round(float(np.clip(sl_dist/max(atr_ref,0.1),0.5,5.0)),4)

        # ── Noi features v14.5 ────────────────────────────────────────────────
        # De câte ori high OR s-a apropiat de ORH în ultimele 10 bare (compresie/acumulare)
        last10 = or_bars.tail(10)
        _thresh = orh - 0.3 * or_width if brk_dir == "LONG" else orl + 0.3 * or_width
        if brk_dir == "LONG":
            approach_count = int((last10["high"] >= _thresh).sum())
        else:
            approach_count = int((last10["low"] <= _thresh).sum())

        # Trend volum în OR: coeff normalizat (pozitiv = volum crește spre breakout)
        _or_vols = or_bars["volume"].values.astype(float)
        if len(_or_vols) >= 4:
            _x = np.arange(len(_or_vols))
            _coeff = float(np.polyfit(_x, _or_vols, 1)[0])
            vol_trend = float(np.clip(_coeff / max(np.mean(_or_vols), 1.0), -5.0, 5.0))
        else:
            vol_trend = 0.0

        # Compresie range: ultimele 5 bare OR vs OR total (< 1 = compresie)
        last5 = or_bars.tail(5)
        _last5_range = float(last5["high"].max() - last5["low"].min())
        range_compression = float(np.clip(_last5_range / max(or_width, 0.1), 0.1, 2.0))

        # ── Features DB: order flow + regim + VP extins ─────────────────────
        # La bara de breakout
        rvol_brk   = float(np.clip(brk.get("rvol", 1.0) or 1.0, 0.1, 10.0))
        dom_brk    = float(np.clip(brk.get("dom_ratio", 1.0) or 1.0, 0.1, 5.0))
        adx_brk    = float(np.clip(brk.get("adx_14", 20.0) or 20.0, 0.0, 100.0))
        hurst_brk  = float(np.clip(brk.get("hurst", 0.5) or 0.5, 0.0, 1.0))
        fish_brk   = float(np.clip(brk.get("fisher_transform", 0.0) or 0.0, -3.0, 3.0))
        garch_brk  = float(np.clip(brk.get("garch_vol", atr_ref) or atr_ref, 0.1, 200.0))
        sentropy   = float(np.clip(brk.get("sample_entropy", 0.5) or 0.5, 0.0, 3.0))
        acf1_brk   = float(np.clip(brk.get("acf_lag1", 0.0) or 0.0, -1.0, 1.0))
        knoise_brk = float(np.clip(brk.get("kalman_noise", 0.0) or 0.0, 0.0, 100.0))
        fft_brk    = float(np.clip(brk.get("fft_cycle", 20.0) or 20.0, 2.0, 200.0))

        # dist_vwap: în pts, directional față de direcția breakout
        _dv = float(brk.get("dist_vwap", 0.0) or 0.0)
        dist_vwap_atr = float(np.clip(_dv / max(atr_ref, 0.1), -10.0, 10.0))
        # +1 dacă breakout în direcția corectă față de VWAP
        vwap_aligned  = int(
            (brk_dir == "LONG"  and _dv > 0) or
            (brk_dir == "SHORT" and _dv < 0)
        )

        # GARCH ratio față de ATR (>1 = volatilitate mare)
        garch_ratio = float(np.clip(garch_brk / max(atr_ref, 0.1), 0.1, 5.0))

        # OR aggregated — medii pe barele OR
        rvol_or   = float(np.clip(or_bars["rvol"].mean(), 0.1, 10.0))
        dom_or    = float(np.clip(or_bars["dom_ratio"].mean(), 0.1, 5.0))
        adx_or    = float(np.clip(or_bars["adx_14"].mean(), 0.0, 100.0))
        bigbal_or = float(np.clip(or_bars["of_big_balance"].mean(), 0.0, 1.0))

        # True open context
        true_op = float(brk.get("true_open", 0.0) or 0.0)
        if true_op > 0:
            above_true_open   = int(entry > true_op)
            dist_true_op_atr  = float(np.clip((entry - true_op) / max(atr_ref, 0.1), -10.0, 10.0))
        else:
            above_true_open  = 0
            dist_true_op_atr = 0.0

        # Prev week high/low — a depășit structura săptămânală?
        pw_hi = float(brk.get("p_hi", 0.0) or 0.0)
        pw_lo = float(brk.get("p_lo", 0.0) or 0.0)
        broke_pw_hi = int(pw_hi > 0 and float(or_bars["high"].max()) > pw_hi)
        broke_pw_lo = int(pw_lo > 0 and float(or_bars["low"].min())  < pw_lo)

        # London high/low — util pentru NY KZ (a luat lichiditate LON?)
        lon_h = float(brk.get("lon_hi", 0.0) or 0.0)
        lon_l = float(brk.get("lon_lo", 0.0) or 0.0)
        lon_swept = int(
            (lon_h > 0 and float(or_bars["high"].max()) > lon_h) or
            (lon_l > 0 and float(or_bars["low"].min())  < lon_l)
        )

        # Displacement în OR (24.9% coverage — binary, 0 dacă lipsă)
        has_disp_or = int(or_bars["has_displacement"].sum() > 0)

        # PDH/PDL distanță mai precisă din DB (normalizat ATR)
        _pdh_dist_db = float(brk.get("dist_pdh", 0.0) or 0.0)
        _pdl_dist_db = float(brk.get("dist_pdl", 0.0) or 0.0)
        dist_pdh_atr = float(np.clip(_pdh_dist_db / max(atr_ref, 0.1), -10.0, 10.0))
        dist_pdl_atr = float(np.clip(_pdl_dist_db / max(atr_ref, 0.1), -10.0, 10.0))

        # ── Label INLINE v14.3 ───────────────────────────────────────────────
        # REAL = prețul ajunge la 1R (= sl_dist în direcție) înainte de SL_INITIAL
        # Verificăm barele VIITOARE din sesiune (post breakout bar)
        sl_level_long  = entry - sl_dist   # SL pt LONG (cu 1R = sl_dist)
        sl_level_short = entry + sl_dist   # SL pt SHORT
        target_long    = entry + sl_dist   # 1R target pt LONG
        target_short   = entry - sl_dist   # 1R target pt SHORT

        is_real = 0
        future_bars = post_or.iloc[brk_idx+1:brk_idx+121]  # max 120 bare (2 ore)
        for _, fb in future_bars.iterrows():
            fh = float(fb["high"]); fl = float(fb["low"])
            if brk_dir == "LONG":
                if fl <= sl_level_long:   break          # SL hit → FAKE
                if fh >= target_long:     is_real = 1; break  # 1R → REAL
            else:
                if fh >= sl_level_short:  break          # SL hit → FAKE
                if fl <= target_short:    is_real = 1; break  # 1R → REAL

        records.append({
            "body_pct":           round(body/rng,4),
            "upper_wick_pct":     round((float(brk["high"])-max(float(brk["open"]),float(brk["close"])))/rng,4),
            "lower_wick_pct":     round((min(float(brk["open"]),float(brk["close"]))-float(brk["low"]))/rng,4),
            "close_strength":     round(cl_str,4),
            "is_strong_close":    is_str,
            "bar_range_atr":      round(min(rng/atr_ref,10.0),4),
            "vol_vs_or_avg":      round(min(float(brk["volume"])/or_vol,20.0),4),
            "or_width_atr":       round(min(or_width/atr_ref,20.0),4),
            "or_width_pts":       round(or_width,2),
            "min_after_or":       min_after,
            "min_after_or_bin":   tb,
            "broke_pdh":          b_pdh,
            "broke_pdl":          b_pdl,
            "broke_asia_hi":      b_ahi,
            "broke_asia_lo":      b_alo,
            "or_swept_above":     int(b_pdh and brk_dir=="SHORT"),
            "or_swept_below":     int(b_pdl and brk_dir=="LONG"),
            "momentum_pre_or":    round(mom,4),
            "h1_trend_vs_break":  h1t,
            "h4_trend_vs_break":  h4t,
            "atr_abs":            round(atr_ref,2),
            "atr_vs_10d_avg":     round(float(np.clip(atr_ref/max(a10,1.0),0.3,3.0)),4),
            "dist_above_orh_atr": round(float(np.clip((entry-orh)/atr_ref if brk_dir=="LONG" else 0,0,10)),4),
            "dist_below_orl_atr": round(float(np.clip((orl-entry)/atr_ref if brk_dir=="SHORT" else 0,0,10)),4),
            "or_candle_count":    len(or_bars),
            "or_close_vs_mid":    round(float(np.clip((last_or-or_mid)/or_width if or_width>0 else 0,-1,1)),4),
            "inside_va":          inside_va,
            "dist_poc_atr":       dist_poc,
            "sl_dist_atr":        sl_dist_atr,
            "above_vah":          above_vah,
            "below_val":          below_val,
            # v14.5 noi
            "or_approach_count":  approach_count,
            "vol_trend_in_or":    round(vol_trend, 4),
            "range_compression":  round(range_compression, 4),
            # ── v15.0 — DB order flow + regim + VP extins ────────────────────
            # La bara de breakout
            "rvol_break":         round(rvol_brk, 4),
            "dom_ratio_break":    round(dom_brk, 4),
            "adx_break":          round(adx_brk, 2),
            "hurst_break":        round(hurst_brk, 4),
            "fisher_break":       round(fish_brk, 4),
            "garch_ratio_break":  round(garch_ratio, 4),
            "sample_entropy_brk": round(sentropy, 4),
            "acf_lag1_break":     round(acf1_brk, 4),
            "kalman_noise_break": round(knoise_brk, 4),
            "fft_cycle_break":    round(fft_brk, 2),
            # VWAP context
            "dist_vwap_atr":      round(dist_vwap_atr, 4),
            "vwap_aligned":       vwap_aligned,
            # OR aggregated
            "rvol_or_avg":        round(rvol_or, 4),
            "dom_ratio_or":       round(dom_or, 4),
            "adx_or_avg":         round(adx_or, 2),
            "bigbal_or":          round(bigbal_or, 4),
            # Market context
            "above_true_open":    above_true_open,
            "dist_true_op_atr":   round(dist_true_op_atr, 4),
            "broke_pw_hi":        broke_pw_hi,
            "broke_pw_lo":        broke_pw_lo,
            "lon_swept":          lon_swept,
            "has_disp_in_or":     has_disp_or,
            "dist_pdh_atr_db":    round(dist_pdh_atr, 4),
            "dist_pdl_atr_db":    round(dist_pdl_atr, 4),
            # metadata pt join cu backtest CSV + label inline
            "date":          str(date),
            "killzone":      kz_name,
            "direction":     brk_dir,
            "label_inline":  is_real,   # labelul v14.3 calculat inline
        })

    elapsed = time.time()-t0
    print(f"✅ {kz_name}: {len(records):,} breakout-uri în {elapsed:.0f}s")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# ADD BACKTEST LABELS — join cu naked backtest CSV → r_mult real
# ─────────────────────────────────────────────────────────────────────────────
def add_backtest_labels(data: pd.DataFrame, bt_csv: Path) -> pd.DataFrame:
    """
    Join breakout features cu rezultatele simulate din backtest naked.
    Labelul de regresie = r_mult (capped ±4R) din simularea reală.

    De ce merge join-ul pe (date, killzone):
    - naked backtest ia PRIMUL breakout per sesiune (LON/NY)
    - extract_breakouts ia și el PRIMUL breakout per sesiune
    → 1:1 match garantat
    """
    if not bt_csv.exists():
        raise FileNotFoundError(
            f"Lipsă: {bt_csv}\n"
            f"Rulează mai întâi: NO_PARTIAL=1 python3 backtest_orh_10y.py\n"
            f"(produce {bt_csv.name} cu r_mult real pentru fiecare trade)"
        )

    bt = pd.read_csv(bt_csv)
    bt["date"] = bt["date"].astype(str)

    # Join OPȚIONAL cu CSV — DOAR pentru r_mult real în EV table
    # Labelul de antrenament vine din label_inline (calculat inline)
    merged = data.merge(
        bt[["date","killzone","r_mult","pts","risk","reason"]],
        on=["date","killzone"], how="left"   # left: păstrăm toate breakout-urile
    )

    # Label de training = label_inline (v14.3 calculat inline)
    merged["label_r"] = merged["label_inline"].astype(int)

    # Statistici
    n_total  = len(data)
    pos = int(merged["label_r"].sum())
    neg = n_total - pos
    print(f"\n📊 Label INLINE v14.3 (REAL = ajunge 1R fără SL_INITIAL):")
    print(f"   Total breakouts: {n_total:,}")
    print(f"   REAL (1): {pos:,} ({100*pos/n_total:.1f}%) — prețul a confirmat 1R")
    print(f"   FAKE (0): {neg:,} ({100*neg/n_total:.1f}%) — SL rapid sau nu a ajuns 1R")
    # EV din backtest CSV pentru REAL vs FAKE (dacă avem r_mult)
    has_r = merged["r_mult"].notna()
    if has_r.sum() > 0:
        real_r = merged.loc[(merged.label_r==1) & has_r, "r_mult"]
        fake_r = merged.loc[(merged.label_r==0) & has_r, "r_mult"]
        if len(real_r) > 0:
            print(f"   r_mult REAL: mean={real_r.mean():+.3f}R | WR={(real_r>0).mean():.1%}")
        if len(fake_r) > 0:
            print(f"   r_mult FAKE: mean={fake_r.mean():+.3f}R | WR={(fake_r>0).mean():.1%}")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN XGBRegressor
# ─────────────────────────────────────────────────────────────────────────────
_CLF_PARAMS = dict(
    n_estimators=600, max_depth=5, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85,
    min_child_weight=3, gamma=0.2, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=1.0,   # ajustat dinamic în train_and_save
    random_state=42, n_jobs=-1, verbosity=0,
    eval_metric="auc", early_stopping_rounds=40,
    use_label_encoder=False if hasattr(xgb.XGBClassifier(), 'use_label_encoder') else None,
)

def train_and_save(data: pd.DataFrame, kz_name: str):
    print(f"\n{'═'*60}")
    print(f"🧠 BINARY CLASSIFIER v14.5 — {kz_name} | {len(data):,} breakout-uri")
    print(f"   Label: REAL = mfe≥{MFE_THR_R}R fără SL_INITIAL | Thr inferență={PROBA_THR}")
    print(f"{'═'*60}")

    # Asigurăm că toate featurile există
    for f in BREAKOUT_FEATURES:
        if f not in data.columns: data[f] = 0.0

    X = data[BREAKOUT_FEATURES].fillna(0).values
    y = data["label_r"].values.astype(int)          # binar: 1=REAL (ajunge 1R), 0=FAKE
    r_actual = data["r_mult"].fillna(0).values       # r_mult real pentru EV table (0 dacă lipsă)

    # Split temporal 80/20
    split = int(len(X) * 0.80)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    rtr_actual, rte_actual = r_actual[:split], r_actual[split:]

    pos_rate_tr = ytr.mean()
    pos_rate_te = yte.mean()
    spw = max(1.0, (1-pos_rate_tr)/max(pos_rate_tr, 0.01))  # scale_pos_weight

    print(f"   Train={len(Xtr):,} | Test={len(Xte):,}")
    print(f"   Train REAL={ytr.sum():,} ({pos_rate_tr:.1%}) | Test REAL={yte.sum():,} ({pos_rate_te:.1%})")
    print(f"   scale_pos_weight={spw:.2f}")

    params = dict(_CLF_PARAMS)
    params["scale_pos_weight"] = spw
    # Eliminăm use_label_encoder None dacă nu suportat
    if params.get("use_label_encoder") is None:
        params.pop("use_label_encoder", None)

    model = xgb.XGBClassifier(**params)
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

    proba = model.predict_proba(Xte)[:, 1]   # P(mfe≥0.8R)

    # ── Metrici clasificare ────────────────────────────────────────────────
    auc = roc_auc_score(yte, proba)
    pred_class = (proba >= PROBA_THR).astype(int)
    print(f"\n📊 METRICI CLASIFICARE (test set, label REAL=mfe≥{MFE_THR_R}R):")
    print(f"   AUC:          {auc:.4f}  (>0.65 = bun, >0.70 = foarte bun)")
    print(f"   Pred P(REAL): [{proba.min():.3f}, {proba.max():.3f}]  (trebuie spread >0.3)")
    print(f"   Selectate la thr={PROBA_THR}: {pred_class.sum():,} ({100*pred_class.mean():.1f}%)")

    # ── EV de profit REAL @ diferite threshold-uri de probabilitate ──────
    # ACEASTA E CHEIA: la ce threshold e EV pozitiv și câte trades?
    print(f"\n📈 EV (r_mult real = PROFIT) @ threshold P(REAL):")
    print(f"   {'Thr':>6} {'N trades':>9} {'% total':>8} {'E[r_mult]':>10} {'WR':>7} {'E[win]':>9}")
    for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = proba >= thr
        n = mask.sum()
        if n < 5: continue
        actual_profit = rte_actual[mask]
        ev  = actual_profit.mean()
        wr  = (actual_profit > 0).mean()
        be  = (actual_profit == 0).mean()
        avg_win = actual_profit[actual_profit > 0].mean() if (actual_profit > 0).any() else 0
        print(f"   {thr:>6.2f} {n:>9,} {100*n/len(yte):>7.1f}% "
              f"{ev:>+10.3f}R {wr:>7.1%} {avg_win:>+9.3f}R")

    # ── Baseline + summary ───────────────────────────────────────────────────
    print(f"\n   Baseline (fără filtru): E[r_mult]={rte_actual.mean():+.3f}R | WR={(rte_actual>0).mean():.1%}")
    thr_mask = proba >= PROBA_THR
    if thr_mask.sum() > 0:
        thr_ev  = rte_actual[thr_mask].mean()
        thr_wr  = (rte_actual[thr_mask] > 0).mean()
        thr_be  = (rte_actual[thr_mask] == 0).mean()
        n_thr = thr_mask.sum()
        est_trades_yr = int(n_thr / len(yte) * (len(y) / 12))  # extrapolat/an
        print(f"   → SELECȚIE FINALĂ P≥{PROBA_THR}: {n_thr:,} trades ({100*thr_mask.mean():.1f}%)")
        print(f"     E[r_mult]={thr_ev:+.3f}R | WR={thr_wr:.1%} | BE={thr_be:.1%}")
        print(f"     Estimare ~{est_trades_yr}/an ({est_trades_yr/252:.2f} trades/zi)")
        pnl_est = thr_ev * 200 * (len(y) / len(yte)) * (12/12)   # $200/R, scaling
        print(f"     P&L estimat (12 ani): ${pnl_est:,.0f}")

    # ── Top features ─────────────────────────────────────────────────────────
    print(f"\n🔑 TOP 10 FEATURES (importance pentru P(mfe≥{MFE_THR_R}R)):")
    imp_pairs = sorted(zip(BREAKOUT_FEATURES, model.feature_importances_), key=lambda x:-x[1])
    for feat, imp in imp_pairs[:10]:
        print(f"   {feat:35s} {imp:.4f} {'█'*int(imp*120)}")

    # ── Salvare model ─────────────────────────────────────────────────────────
    mj = DIR / f"mario_bot_breakout_{kz_name.lower()}.json"
    mm = DIR / f"mario_bot_breakout_{kz_name.lower()}_meta.json"
    model.save_model(str(mj))

    _atr_med   = float(data["atr_abs"].median()) if "atr_abs" in data.columns else 10.0
    _risk_pts  = float(np.clip(max(8.0, _atr_med * 0.85), 8.0, 20.0))

    with open(mm, "w") as f:
        json.dump({
            "killzone":       kz_name,
            "features":       BREAKOUT_FEATURES,
            "model_type":     "binary_breakout_v14.5",
            "version":        "v14.5",
            "label":          f"REAL = mfe_r >= {MFE_THR_R} AND reason != SL_INITIAL",
            "risk_pts":       round(_risk_pts, 1),
            "min_expected_r": PROBA_THR,    # threshold probabilitate la inferență
            "auc_test":       round(auc, 4),
            "n_train":        int(len(Xtr)),
            "n_test":         int(len(Xte)),
            "pos_rate_train": round(float(pos_rate_tr), 4),
        }, f, indent=2)
    print(f"\n💾 Salvat: {mj.name}  +  {mm.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    mode    = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    kz_list = ["LON","NY"] if mode in ("ALL","BOTH") else [mode]

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  ALADIN v14.5 — BINARY BREAKOUT TRAINER             ║")
    print(f"║  Label: REAL = mfe≥{MFE_THR_R}R fără SL_INITIAL | Thr={PROBA_THR}  ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    if not BACKTEST_CSV.exists():
        print(f"❌ Lipsă {BACKTEST_CSV.name}")
        print("   Rulează: NO_PARTIAL=1 python3 backtest_orh_10y.py")
        sys.exit(1)

    print(f"📊 Backtest CSV: {BACKTEST_CSV.name}")
    bt_preview = pd.read_csv(BACKTEST_CSV)
    print(f"   {len(bt_preview):,} trades | r_mult mean={bt_preview['r_mult'].mean():+.3f}")

    # ── Extragem features AN CU AN pentru a evita OOM ─────────────────────────
    # Barele (~400K/an) se încarcă, features se extrag (~500 rows/an/kz),
    # barele se șterg. Combinăm doar features (~5K rows total, sub 5MB).
    conn = sqlite3.connect(DB)
    years = [r[0] for r in conn.execute(
        "SELECT DISTINCT CAST(strftime('%Y',timestamp) AS INT) FROM market_data ORDER BY 1"
    ).fetchall()]
    conn.close()
    print(f"📅 Procesez {len(years)} ani an cu an (fiecare ~50MB, eliberat după extracție)")

    # Colectăm features per killzone
    kz_records = {kz: [] for kz in kz_list}

    for yr in years:
        print(f"\n── Încărcare {yr} ──────────────────────────────────")
        df_yr = _load_year(yr)
        print(f"   {len(df_yr):,} bare | {df_yr['ts'].min()} → {df_yr['ts'].max()}")

        for kz in kz_list:
            if kz not in KZ: continue
            recs = extract_breakouts(df_yr, kz)
            # Filtrăm la doar ANUL curent (excludem Dec prev / Ian next)
            recs['_yr'] = pd.to_datetime(recs['date']).dt.year
            recs = recs[recs['_yr'] == yr].drop(columns=['_yr'])
            kz_records[kz].append(recs)
            print(f"   {kz}: {len(recs)} breakouts în {yr}")

        del df_yr  # eliberăm RAM

    # ── Train per killzone ───────────────────────────────────────────────────
    for kz in kz_list:
        if kz not in KZ: continue
        frames = kz_records[kz]
        if not frames:
            print(f"⚠️ Nicio dată pentru {kz}"); continue

        data = pd.concat(frames, ignore_index=True)
        print(f"\n📊 {kz}: {len(data):,} breakouts totale extrase")

        if len(data) < 50:
            print(f"⚠️ Prea puține date: {len(data)}"); continue

        # Add labels + join opțional cu CSV pentru EV table
        data = add_backtest_labels(data, BACKTEST_CSV)
        if len(data) < 50 or "label_r" not in data.columns:
            print(f"⚠️ Prea puțin după labels: {len(data)}"); continue

        # Salvează dataset
        csv_out = DIR / f"breakout_dataset_{kz.lower()}_v145.csv"
        data.to_csv(csv_out, index=False)
        print(f"💾 Dataset: {csv_out.name} ({len(data):,} rows)")

        # Antrenează
        train_and_save(data, kz)

    print("\n✅ GATA! Rulează: NO_PARTIAL=1 python3 backtest_orh_10y.py v14")


if __name__ == "__main__":
    main()
