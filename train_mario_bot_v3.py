"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MARIO BOT v3 — SIMULATION-BASED TARGET LABELS                              ║
║  Fix fundamental v2 bug: target generation via SL/TP simulation,            ║
║  nu lookahead direction. Fiecare bara din KZ primeste label DOAR daca       ║
║  un trade simulat (entry=close, SL=0.5×ATR, TP=2.0×ATR) ar fi câștigat.   ║
║  Rezultat așteptat: signal rate 5-15% (față de 99% în v2)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import sqlite3, json, pathlib, sys, warnings, time, gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
warnings.filterwarnings("ignore")

DIR          = pathlib.Path(__file__).parent
PATH_DB      = DIR / "mario_trading.db"
OUT_MODEL    = DIR / "mario_bot_v3.json"
OUT_FEATURES = DIR / "mario_bot_v3_features.json"
DATASET_CSV  = DIR / "mario_bot_v3_dataset.csv"

# ── CONFIG ────────────────────────────────────────────────────────────────────
YEARS_ALL    = list(range(2015, 2026))
TRAIN_YEARS  = list(range(2015, 2022))   # 2015-2021 train
VAL_YEARS    = [2022]
TEST_YEARS   = list(range(2023, 2026))   # 2023-2025 OOS

LON_START, LON_END = 9.0,  11.0
NY_START,  NY_END  = 15.5, 17.5
HORIZON            = 80          # bare max de verificat pentru SL/TP
SL_ATR             = 0.5         # stop loss = 0.5×ATR sub/deasupra entry
TP_ATR             = 2.0         # take profit = 2.0×ATR (4R trade)
REV_PULLBACK_ATR   = 0.3         # dacă preț trage >0.3×ATR înainte de TP → e reversare
CONTEXT_BARS       = 300         # bare extra din an precedent pt rolling features

REGIME_NAMES = {0:'WAIT',1:'SHORT_BREAK',2:'LONG_BREAK',3:'SHORT_REV',4:'LONG_REV'}

FEATURES = [
    # Session timing
    'in_london_kz','in_london_or','in_london_close','in_pre_ny',
    'in_ny_or','in_ny_kz_core','in_ny_macro_1','in_ny_macro_2','in_any_macro',
    'mins_since_lon_open','mins_since_ny_open','hour_sin','hour_cos',
    # Level distances (ATR-norm)
    'dist_pdh_atr','dist_pdl_atr','dist_asia_hi_atr','dist_asia_lo_atr',
    'dist_lon_hi_atr','dist_lon_lo_atr','dist_vah_atr','dist_val_atr','dist_poc_atr',
    'above_vah','below_val','inside_va',
    # Level breaks
    'broke_pdh','broke_pdl','broke_asia_hi','broke_asia_lo',
    'broke_lon_hi','broke_lon_lo','broke_vah','broke_val',
    # Sweep
    'sweep_direction','swept_above','swept_below',
    'sweep_wick_ratio','upper_wick_atr','lower_wick_atr',
    'bars_since_sweep','reclaimed_after_sweep',
    # Candle
    'body_dir','wick_bias','range_atr_ratio','body_pct','close_strength',
    'sharp_reversal','rejection_candle',
    # Momentum
    'momentum_5','momentum_15','slope_h1','h4_momentum',
    # Volatility
    'atr_percentile','vol_spike','rvol','adx_14','hurst','garch_vol',
    'fisher_transform','acf_lag1',
    # Liquidity
    'liq_above','liq_below','range_day_score',
    # Order Flow DB
    'bar_delta_n','cum_delta_n','dom_ratio','of_big_balance',
    'of_doi','absorption_score','stacked_bull','stacked_bear',
    # VP + SMC
    'dist_vwap_atr','fvg_up','fvg_down','has_displacement',
]


def safe_div(a, b, fill=0.0):
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.asarray(a, float) / np.where(np.asarray(b, float) != 0, np.asarray(b, float), np.nan)
    return np.where(np.isfinite(r), r, fill)

def clip5(x): return np.clip(np.asarray(x, float), -5, 5)


def simulate_trade(hi_fut, lo_fut, entry, sl_dist, tp_dist, direction):
    """
    Simulează un trade bar cu bar.
    direction: +1 = LONG, -1 = SHORT
    Returnează: ('win', bars_to_close, max_adverse) sau ('loss', bars, max_adverse)
    """
    if direction == 1:
        sl_price = entry - sl_dist
        tp_price = entry + tp_dist
    else:
        sl_price = entry + sl_dist
        tp_price = entry - tp_dist

    max_adverse = 0.0
    for h in range(len(hi_fut)):
        bar_hi = hi_fut[h]
        bar_lo = lo_fut[h]

        if direction == 1:
            adverse = entry - bar_lo  # cât a coborât împotriva long
            max_adverse = max(max_adverse, adverse)
            # Check SL (low atinge sl_price)
            sl_hit = bar_lo <= sl_price
            tp_hit = bar_hi >= tp_price
        else:
            adverse = bar_hi - entry  # cât a urcat împotriva short
            max_adverse = max(max_adverse, adverse)
            sl_hit = bar_hi >= sl_price
            tp_hit = bar_lo <= tp_price

        if sl_hit and tp_hit:
            # Ambele în aceeași bară — considerăm SL hit primul (conservator)
            return ('loss', h+1, max_adverse)
        if tp_hit:
            return ('win', h+1, max_adverse)
        if sl_hit:
            return ('loss', h+1, max_adverse)

    return ('none', len(hi_fut), max_adverse)


def compute_features_and_target(df: pd.DataFrame, start_idx: int = 0) -> pd.DataFrame:
    """
    Compute features + target.
    start_idx = prima bara care apartine anului curent
    (barele 0..start_idx-1 sunt context din an precedent).

    TARGET v3 — Simulation-based:
    - LONG simulat (entry=close, SL=close-0.5ATR, TP=close+2ATR):
        → WIN + max_adverse ≤ REV_PULLBACK×ATR → LONG_BREAK (2)
        → WIN + max_adverse > REV_PULLBACK×ATR  → LONG_REV   (4)
    - SHORT simulat:
        → WIN + max_adverse ≤ REV_PULLBACK×ATR → SHORT_BREAK (1)
        → WIN + max_adverse > REV_PULLBACK×ATR  → SHORT_REV  (3)
    - Ambele câștigă → cel cu mai puțin drawdown adverse
    - Niciuna nu câștigă → WAIT (0)
    """
    cl  = df['close'].values.astype(float)
    hi  = df['high'].values.astype(float)
    lo  = df['low'].values.astype(float)
    op  = df['open'].values.astype(float)
    vol = df['volume'].values.astype(float)
    n   = len(df)

    atr  = np.where(df['atr_14'].values > 0, df['atr_14'].values, 9.0).astype(float)
    pdh  = pd.Series(df['p_hi'].values).ffill().values.astype(float)
    pdl  = pd.Series(df['p_lo'].values).ffill().values.astype(float)
    a_hi = pd.Series(df['asia_hi'].values).ffill().values.astype(float)
    a_lo = pd.Series(df['asia_lo'].values).ffill().values.astype(float)
    l_hi = pd.Series(df['lon_hi'].values).ffill().values.astype(float)
    l_lo = pd.Series(df['lon_lo'].values).ffill().values.astype(float)
    h4h  = pd.Series(df['h4_hi'].values).ffill().values.astype(float)
    h4l  = pd.Series(df['h4_lo'].values).ffill().values.astype(float)
    vah  = pd.Series(df['vah'].values).ffill().values.astype(float)
    val  = pd.Series(df['val'].values).ffill().values.astype(float)
    poc  = pd.Series(df['poc_level'].values).ffill().values.astype(float)
    lw_h = pd.Series(df['lw_hi'].values).ffill().values.astype(float)
    lw_l = pd.Series(df['lw_lo'].values).ffill().values.astype(float)

    ts = pd.to_datetime(df['timestamp'])
    td = (ts.dt.hour + ts.dt.minute / 60.0).values

    out = {}

    # Session windows
    out['in_london_kz']    = ((td >= 9.5)  & (td < 11.0)).astype(np.int8)
    out['in_london_or']    = ((td >= 9.0)  & (td < 9.5)).astype(np.int8)
    out['in_london_close'] = ((td >= 11.0) & (td < 12.0)).astype(np.int8)
    out['in_pre_ny']       = ((td >= 15.0) & (td < 15.5)).astype(np.int8)
    out['in_ny_or']        = ((td >= 15.5) & (td < 16.0)).astype(np.int8)
    out['in_ny_kz_core']   = ((td >= 16.0) & (td < 16.833)).astype(np.int8)
    out['in_ny_macro_1']   = ((td >= 16.833) & (td < 17.167)).astype(np.int8)
    out['in_ny_macro_2']   = ((td >= 17.167) & (td < 17.5)).astype(np.int8)
    out['in_any_macro']    = ((out['in_ny_macro_1']==1)|(out['in_ny_macro_2']==1)).astype(np.int8)
    out['mins_since_lon_open'] = np.clip((td-9.0)*60, 0, 180).astype(np.float32)
    out['mins_since_ny_open']  = np.clip((td-15.5)*60, 0, 120).astype(np.float32)
    out['hour_sin'] = np.sin(2*np.pi*td/24).astype(np.float32)
    out['hour_cos'] = np.cos(2*np.pi*td/24).astype(np.float32)

    # Level distances
    out['dist_pdh_atr']     = clip5(safe_div(pdh-cl, atr)).astype(np.float32)
    out['dist_pdl_atr']     = clip5(safe_div(cl-pdl, atr)).astype(np.float32)
    out['dist_asia_hi_atr'] = clip5(safe_div(a_hi-cl, atr)).astype(np.float32)
    out['dist_asia_lo_atr'] = clip5(safe_div(cl-a_lo, atr)).astype(np.float32)
    out['dist_lon_hi_atr']  = clip5(safe_div(l_hi-cl, atr)).astype(np.float32)
    out['dist_lon_lo_atr']  = clip5(safe_div(cl-l_lo, atr)).astype(np.float32)
    out['dist_vah_atr']     = clip5(safe_div(vah-cl, atr)).astype(np.float32)
    out['dist_val_atr']     = clip5(safe_div(cl-val, atr)).astype(np.float32)
    out['dist_poc_atr']     = clip5(safe_div(cl-poc, atr)).astype(np.float32)
    out['above_vah']        = (cl > vah).astype(np.int8)
    out['below_val']        = (cl < val).astype(np.int8)
    out['inside_va']        = df['inside_va'].fillna(0).values.astype(np.int8)

    # Breaks
    out['broke_pdh']     = (hi > pdh).astype(np.int8)
    out['broke_pdl']     = (lo < pdl).astype(np.int8)
    out['broke_asia_hi'] = (hi > a_hi).astype(np.int8)
    out['broke_asia_lo'] = (lo < a_lo).astype(np.int8)
    out['broke_lon_hi']  = (hi > l_hi).astype(np.int8)
    out['broke_lon_lo']  = (lo < l_lo).astype(np.int8)
    out['broke_vah']     = (hi > vah).astype(np.int8)
    out['broke_val']     = (lo < val).astype(np.int8)

    # Sweep
    sw_up = ((hi>a_hi)+(hi>pdh)+(hi>l_hi)+(hi>h4h)).astype(int)
    sw_dn = ((lo<a_lo)+(lo<pdl)+(lo<l_lo)+(lo<h4l)).astype(int)
    out['sweep_direction'] = np.where(sw_up>sw_dn,1,np.where(sw_dn>sw_up,-1,0)).astype(np.int8)
    out['swept_above']     = (sw_up > 0).astype(np.int8)
    out['swept_below']     = (sw_dn > 0).astype(np.int8)

    bar_range = np.maximum(hi-lo, 0.01)
    uw = hi - np.maximum(cl, op)
    lw = np.minimum(cl, op) - lo
    out['sweep_wick_ratio']  = clip5(safe_div(np.maximum(uw,lw), bar_range)).astype(np.float32)
    out['upper_wick_atr']    = clip5(safe_div(uw, atr)).astype(np.float32)
    out['lower_wick_atr']    = clip5(safe_div(lw, atr)).astype(np.float32)

    any_sw = ((sw_up + sw_dn) > 0).astype(int)
    bs = []
    cnt = 99
    for v in any_sw:
        cnt = 0 if v else cnt + 1
        bs.append(min(cnt, 99))
    out['bars_since_sweep'] = np.array(bs, dtype=np.float32)

    sw_up_10 = pd.Series(sw_up).rolling(10,min_periods=1).max().values
    sw_dn_10 = pd.Series(sw_dn).rolling(10,min_periods=1).max().values
    out['reclaimed_after_sweep'] = (
        ((sw_up_10>0) & (cl<a_hi) & (cl<pdh)) |
        ((sw_dn_10>0) & (cl>a_lo) & (cl>pdl))
    ).astype(np.int8)

    # Candle
    body = np.abs(cl - op)
    out['body_dir']       = np.sign(cl-op).astype(np.int8)
    out['wick_bias']      = clip5(safe_div(uw-lw, bar_range)).astype(np.float32)
    out['range_atr_ratio']= clip5(safe_div(bar_range, atr)).astype(np.float32)
    out['body_pct']       = clip5(safe_div(body, bar_range)).astype(np.float32)
    out['close_strength'] = safe_div(cl-lo, bar_range).astype(np.float32)
    out['sharp_reversal'] = ((out['range_atr_ratio']>1.5) & (out['close_strength']>0.7)).astype(np.int8)
    out['rejection_candle']= ((out['sweep_wick_ratio']>0.6) & (body<0.4*bar_range)).astype(np.int8)

    # Momentum
    cl_s = pd.Series(cl)
    out['momentum_5']  = clip5(safe_div((cl_s-cl_s.shift(5)).values,  atr)).astype(np.float32)
    out['momentum_15'] = clip5(safe_div((cl_s-cl_s.shift(15)).values, atr)).astype(np.float32)
    out['slope_h1']    = clip5(safe_div((cl_s-cl_s.shift(60)).values, atr)).astype(np.float32)
    out['h4_momentum'] = clip5(safe_div((cl_s-cl_s.shift(240)).values,atr)).astype(np.float32)

    # Volatility
    atr_s = pd.Series(atr)
    out['atr_percentile'] = atr_s.rolling(100,min_periods=10).rank(pct=True).fillna(0.5).values.astype(np.float32)
    out['vol_spike']      = (atr_s > atr_s.rolling(20).mean()*1.5).astype(np.int8).values
    out['rvol']           = df['rvol'].fillna(1.0).values.astype(np.float32)
    out['adx_14']         = df['adx_14'].fillna(20.0).values.astype(np.float32)
    out['hurst']          = df['hurst'].fillna(0.5).values.astype(np.float32)
    out['garch_vol']      = df['garch_vol'].fillna(0.0).values.astype(np.float32)
    out['fisher_transform']= df['fisher_transform'].fillna(0.0).values.astype(np.float32)
    out['acf_lag1']       = df['acf_lag1'].fillna(0.0).values.astype(np.float32)

    # Liquidity
    out['liq_above'] = ((a_hi>cl)+(pdh>cl)+(h4h>cl)+(lw_h>cl)).astype(np.int8)
    out['liq_below'] = ((a_lo<cl)+(pdl<cl)+(h4l<cl)+(lw_l<cl)).astype(np.int8)

    # Day range score
    dates = ts.dt.date
    day_grp_hi = df.groupby(dates)['high'].cummax().values
    day_grp_lo = df.groupby(dates)['low'].cummin().values
    out['range_day_score'] = (1.0/np.maximum(safe_div(day_grp_hi-day_grp_lo,atr,1.0),1.0)).astype(np.float32)

    # Order Flow (DB)
    bar_d = df['bar_delta'].fillna(0).values.astype(float)
    cum_d = df['cum_delta'].fillna(0).values.astype(float)
    vol_atr = vol * atr + 1
    out['bar_delta_n']     = clip5(safe_div(bar_d, vol_atr)).astype(np.float32)
    cum_abs_20 = pd.Series(np.abs(cum_d)).rolling(20).mean().fillna(1).values
    out['cum_delta_n']     = clip5(safe_div(cum_d, cum_abs_20)).astype(np.float32)
    out['dom_ratio']       = df['dom_ratio'].fillna(1.0).values.astype(np.float32)
    out['of_big_balance']  = df['of_big_balance'].fillna(0).values.astype(np.float32)
    out['of_doi']          = df['of_doi'].fillna(0).values.astype(np.float32)
    out['absorption_score']= df['absorption_score'].fillna(0).values.astype(np.float32)
    out['stacked_bull']    = df['stacked_bull'].fillna(0).values.astype(np.int8)
    out['stacked_bear']    = df['stacked_bear'].fillna(0).values.astype(np.int8)

    # VP + SMC
    dv = df['dist_vwap'].fillna(0).values.astype(float)
    out['dist_vwap_atr']   = clip5(safe_div(dv, atr)).astype(np.float32)
    out['fvg_up']          = df['fvg_up'].fillna(0).values.astype(np.int8)
    out['fvg_down']        = df['fvg_down'].fillna(0).values.astype(np.int8)
    out['has_displacement']= df['has_displacement'].fillna(0).values.astype(np.int8)

    feat_df = pd.DataFrame(out, index=df.index)

    # ── TARGET GENERATION v3 — ICT SETUP FILTER + SIMULATION ────────────────
    #
    # Doar barele care îndeplinesc un setup ICT specific sunt candidate.
    # Simulăm trade (SL/TP) numai pentru direcția corespunzătoare setup-ului.
    # Label = clasa IF simulare → WIN, altfel WAIT.
    #
    # SETUP-URI ACCEPTATE:
    #
    # LONG_REV (4): Sweep-and-Reverse Bullish
    #   • bara curentă: lo < min(a_lo, pdl, l_lo) - 0.05×ATR  (a atins/depasit un low major)
    #   • SI close > open  (bara se inchide bullish = respingere)
    #   • SI (close - lo) / range > 0.60  (inchidere in treimea superioara = rejection wick)
    #   • SI swept_below acum (calculat mai sus)
    #   → Simuleaza LONG (entry=close, SL=close-0.5×ATR, TP=close+2×ATR)
    #   → Daca WIN → LONG_REV
    #
    # SHORT_REV (3): Sweep-and-Reverse Bearish
    #   • bara curentă: hi > max(a_hi, pdh, l_hi) + 0.05×ATR  (a atins/depasit un high major)
    #   • SI close < open  (bara se inchide bearish = respingere)
    #   • SI (hi - close) / range > 0.60  (inchidere in treimea inferioara = wick sus)
    #   • SI swept_above acum
    #   → Simuleaza SHORT (entry=close, SL=close+0.5×ATR, TP=close-2×ATR)
    #   → Daca WIN → SHORT_REV
    #
    # LONG_BREAK (2): Bullish Breakout Bar
    #   • bara curentă: hi > max(a_hi, l_hi, pdh)  (rupe un high de sesiune)
    #   • SI close > max(a_hi, l_hi) - 0.05×ATR  (inchide deasupra nivelului rupt)
    #   • SI (close - lo) / range > 0.55  (momentum bullish = inchidere puternica)
    #   • SI range > 0.8×ATR  (bara de impuls, nu doji)
    #   → Simuleaza LONG (entry=close, SL=close-0.5×ATR, TP=close+2×ATR)
    #   → Daca WIN → LONG_BREAK
    #
    # SHORT_BREAK (1): Bearish Breakdown Bar
    #   • bara curentă: lo < min(a_lo, l_lo, pdl)  (rupe un low de sesiune)
    #   • SI close < min(a_lo, l_lo) + 0.05×ATR  (inchide sub nivelul rupt)
    #   • SI (hi - close) / range > 0.55  (momentum bearish = inchidere slaba)
    #   • SI range > 0.8×ATR  (bara de impuls)
    #   → Simuleaza SHORT (entry=close, SL=close+0.5×ATR, TP=close-2×ATR)
    #   → Daca WIN → SHORT_BREAK
    #
    # Daca o bara califica pentru mai multe setup-uri → prioritate REV > BREAK
    # ─────────────────────────────────────────────────────────────────────────

    in_kz = ((td >= 9.5) & (td <= 11.0)) | ((td >= 16.0) & (td <= 17.5))
    qualifying = np.where(in_kz)[0]

    target = np.zeros(n, dtype=np.int8)

    for idx in qualifying:
        if idx < start_idx:
            continue
        if idx + 2 >= n:
            continue

        a = atr[idx]
        if a <= 0:
            continue

        c   = cl[idx]
        o   = op[idx]
        h   = hi[idx]
        l   = lo[idx]
        br  = max(h - l, 0.01)  # bar range

        # Key levels la momentul barei
        key_hi = max(
            a_hi[idx] if not np.isnan(a_hi[idx]) else -np.inf,
            l_hi[idx] if not np.isnan(l_hi[idx]) else -np.inf,
            pdh[idx]  if not np.isnan(pdh[idx])  else -np.inf,
        )
        key_lo = min(
            a_lo[idx] if not np.isnan(a_lo[idx]) else np.inf,
            l_lo[idx] if not np.isnan(l_lo[idx]) else np.inf,
            pdl[idx]  if not np.isnan(pdl[idx])  else np.inf,
        )

        sl_dist = SL_ATR * a
        tp_dist = TP_ATR * a

        end_idx = min(idx + HORIZON + 1, n)
        hi_fut  = hi[idx+1:end_idx]
        lo_fut  = lo[idx+1:end_idx]

        if len(hi_fut) == 0:
            continue

        candidate_class = 0  # WAIT by default

        # ─── LONG_REV: sweep low + bullish rejection ─────────────────────────
        rev_long_ok = (
            l <= key_lo + 0.05 * a        # bara a atins/depasit un key low
            and c > o                      # bara bullish
            and (c - l) / br > 0.60        # inchidere in treimea superioara
        )
        if rev_long_ok and candidate_class == 0:
            res = simulate_trade(hi_fut, lo_fut, c, sl_dist, tp_dist, 1)
            if res[0] == 'win':
                candidate_class = 4   # LONG_REV

        # ─── SHORT_REV: sweep high + bearish rejection ────────────────────────
        rev_short_ok = (
            h >= key_hi - 0.05 * a         # bara a atins/depasit un key high
            and c < o                       # bara bearish
            and (h - c) / br > 0.60         # inchidere in treimea inferioara
        )
        if rev_short_ok and candidate_class == 0:
            res = simulate_trade(hi_fut, lo_fut, c, sl_dist, tp_dist, -1)
            if res[0] == 'win':
                candidate_class = 3   # SHORT_REV

        # ─── LONG_BREAK: rupe deasupra key high, inchide puternic ────────────
        brk_long_ok = (
            h > key_hi                     # rupe un high major
            and c >= key_hi - 0.05 * a    # inchide deasupra sau la nivel
            and (c - l) / br > 0.55        # inchidere in jumatatea superioara
            and br > 0.8 * a               # bara de impuls (nu doji)
            and candidate_class == 0
        )
        if brk_long_ok:
            res = simulate_trade(hi_fut, lo_fut, c, sl_dist, tp_dist, 1)
            if res[0] == 'win':
                candidate_class = 2   # LONG_BREAK

        # ─── SHORT_BREAK: rupe sub key low, inchide slab ─────────────────────
        brk_short_ok = (
            l < key_lo                     # rupe un low major
            and c <= key_lo + 0.05 * a    # inchide sub sau la nivel
            and (h - c) / br > 0.55        # inchidere in jumatatea inferioara (slab)
            and br > 0.8 * a               # bara de impuls
            and candidate_class == 0
        )
        if brk_short_ok:
            res = simulate_trade(hi_fut, lo_fut, c, sl_dist, tp_dist, -1)
            if res[0] == 'win':
                candidate_class = 1   # SHORT_BREAK

        target[idx] = candidate_class

    feat_df['target']    = target
    feat_df['timestamp'] = df['timestamp'].values
    feat_df['year']      = ts.dt.year.values

    # Returnează doar barele killzone din anul curent
    kz_mask = in_kz & (np.arange(n) >= start_idx)
    return feat_df[kz_mask].copy()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("═"*65)
    print("  MARIO BOT v3 — TRAINING (simulation-based targets)")
    print("═"*65)
    print(f"  Features: {len(FEATURES)}")
    print(f"  SL={SL_ATR}×ATR  TP={TP_ATR}×ATR  Horizon={HORIZON} bare")
    print(f"  REV threshold: adverse > {REV_PULLBACK_ATR}×ATR")

    conn = sqlite3.connect(str(PATH_DB))

    prev_tail  = None
    all_chunks = []

    for year in YEARS_ALL:
        print(f"\n📅 Procesez {year}...", flush=True)
        q = f"""
            SELECT timestamp, open, high, low, close, volume,
                   p_hi, p_lo, asia_hi, asia_lo, lon_hi, lon_lo,
                   lw_hi, lw_lo, h4_hi, h4_lo, h1_hi, h1_lo,
                   atr_14, vah, val, poc_level, inside_va,
                   bar_delta, cum_delta, dom_ratio, of_big_balance, of_doi,
                   absorption_score, stacked_bull, stacked_bear,
                   rvol, adx_14, hurst, garch_vol,
                   fisher_transform, acf_lag1, dist_vwap,
                   fvg_up, fvg_down, has_displacement
            FROM market_data WHERE year = {year} ORDER BY timestamp
        """
        df_year = pd.read_sql(q, conn)
        if len(df_year) == 0:
            print(f"   Sărit (no data)")
            continue

        if prev_tail is not None:
            df_combined = pd.concat([prev_tail, df_year], ignore_index=True)
            start_idx   = len(prev_tail)
        else:
            df_combined = df_year
            start_idx   = 0

        chunk = compute_features_and_target(df_combined, start_idx=start_idx)
        all_chunks.append(chunk)

        dist     = chunk['target'].value_counts().sort_index()
        sig_cnt  = (chunk['target'] != 0).sum()
        kz_total = len(chunk)
        sig_pct  = 100*sig_cnt/kz_total if kz_total > 0 else 0
        print(f"   ✓ {len(df_year):,} rows → {kz_total:,} KZ bars, "
              f"signals={sig_cnt} ({sig_pct:.1f}%) "
              f"({', '.join(f'{REGIME_NAMES[k]}={v}' for k,v in dist.items() if k!=0)})")

        prev_tail = df_year.tail(CONTEXT_BARS).copy()
        del df_combined, df_year, chunk
        gc.collect()

    conn.close()

    print(f"\n📊 Combinez dataset...", flush=True)
    full = pd.concat(all_chunks, ignore_index=True)
    del all_chunks; gc.collect()

    full['target'] = full['target'].astype(int)
    full[FEATURES] = full[FEATURES].fillna(0)

    print(f"   Dataset total: {len(full):,} bare KZ")
    dist_full = full['target'].value_counts().sort_index()
    sig_total = (full['target'] != 0).sum()
    print(f"   Signal rate total: {sig_total:,} / {len(full):,} = {100*sig_total/len(full):.1f}%")
    print(f"   Distributie: {dict(dist_full)}")

    # Salvam dataset pentru inspecție
    full.to_csv(str(DATASET_CSV), index=False)
    print(f"   💾 Salvat {DATASET_CSV.name}")

    # ── SPLIT ─────────────────────────────────────────────────────────────────
    df_train = full[full['year'].isin(TRAIN_YEARS)].copy()
    df_val   = full[full['year'].isin(VAL_YEARS)].copy()
    df_test  = full[full['year'].isin(TEST_YEARS)].copy()

    print(f"\n  Split: train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

    for label, df_s in [("TRAIN",df_train),("VAL",df_val),("TEST",df_test)]:
        d = df_s['target'].value_counts().sort_index()
        sig = (df_s['target'] != 0).sum()
        print(f"  {label}: signals={sig}/{len(df_s)} ({100*sig/len(df_s):.1f}%)")

    X_train = df_train[FEATURES].values
    y_train = df_train['target'].values
    X_val   = df_val[FEATURES].values
    y_val   = df_val['target'].values
    X_test  = df_test[FEATURES].values
    y_test  = df_test['target'].values

    # ── CLASS WEIGHTS ─────────────────────────────────────────────────────────
    # WAIT domina → da weight mai mare semnalelor
    from collections import Counter
    counts    = Counter(y_train)
    n_train   = len(y_train)
    n_classes = 5
    weights   = np.array([
        n_train / (n_classes * max(counts.get(c, 1), 1))
        for c in range(n_classes)
    ])
    # Boost signal classes × 3 față de auto-balance
    for c in range(1, 5):
        weights[c] *= 3.0
    sample_weights = np.array([weights[y] for y in y_train])

    print(f"\n  Class weights: { {REGIME_NAMES[i]: f'{weights[i]:.2f}' for i in range(5)} }")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    params = dict(
        objective          = 'multi:softprob',
        num_class          = 5,
        max_depth          = 6,
        learning_rate      = 0.05,
        n_estimators       = 600,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 30,
        gamma              = 2,
        reg_alpha          = 0.5,
        reg_lambda         = 2.0,
        eval_metric        = 'mlogloss',
        early_stopping_rounds = 40,
        n_jobs             = -1,
        random_state       = 42,
        tree_method        = 'hist',
    )

    print(f"\n⚙️  Training XGBoost...")
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight         = sample_weights,
        eval_set              = [(X_val, y_val)],
        verbose               = 50,
    )

    # ── EVALUARE ──────────────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  EVALUARE")
    print("═"*65)

    for label, X, y in [("VAL", X_val, y_val), ("OOS 2023-2025", X_test, y_test)]:
        proba = model.predict_proba(X)
        pred  = proba.argmax(axis=1)
        acc   = accuracy_score(y, pred)
        print(f"\n  [{label}]  Accuracy: {acc:.4f}")

        # Signal-only accuracy (ignorăm WAIT prediction pe bare WAIT)
        sig_mask = y != 0
        if sig_mask.sum() > 0:
            pred_sig = pred[sig_mask]
            y_sig    = y[sig_mask]
            acc_sig  = accuracy_score(y_sig, pred_sig)
            print(f"  [{label}]  Accuracy pe semnale reale: {acc_sig:.4f}")

        print(classification_report(y, pred, target_names=list(REGIME_NAMES.values()),
                                    zero_division=0))

        # Analiza confidence thresholds pe semnale reale
        print(f"  Analiza confidence (target ≠ WAIT):")
        for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            max_p    = proba.max(axis=1)
            pred_cls = proba.argmax(axis=1)
            mask     = (max_p >= thr) & (pred_cls != 0)
            if mask.sum() == 0:
                continue
            y_m      = y[mask]
            p_m      = pred_cls[mask]
            correct  = (y_m == p_m).sum()
            # Și din câte "non-WAIT" reale am prins
            real_sig = (y != 0).sum()
            recall   = mask.sum() / max(real_sig, 1)
            print(f"    conf≥{thr:.2f}:  N={mask.sum():5d}  "
                  f"Precision={correct/mask.sum():.3f}  "
                  f"Recall_sig={recall:.3f}")

    # ── SAVE ──────────────────────────────────────────────────────────────────
    model.save_model(str(OUT_MODEL))
    meta = dict(
        version        = 'v3',
        features       = FEATURES,
        regime_names   = REGIME_NAMES,
        sl_atr         = SL_ATR,
        tp_atr         = TP_ATR,
        horizon        = HORIZON,
        rev_pullback   = REV_PULLBACK_ATR,
        train_years    = TRAIN_YEARS,
        val_years      = VAL_YEARS,
        test_years     = TEST_YEARS,
        n_train        = int(len(df_train)),
        n_val          = int(len(df_val)),
        n_test         = int(len(df_test)),
    )
    with open(str(OUT_FEATURES), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✅ Model salvat: {OUT_MODEL.name}")
    print(f"  ✅ Meta salvat:  {OUT_FEATURES.name}")
    print(f"  ⏱️  Timp total: {(time.time()-t0)/60:.1f} min")
    print("═"*65)


if __name__ == "__main__":
    main()
