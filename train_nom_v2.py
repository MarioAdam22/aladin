"""
train_nom_v2.py — NY Open Manipulation v2 Enhanced
====================================================
vs v1.1 (OOS=0.610, IS=0.873, 125 features, fixed params):
  ✅ Optuna 80 trials + BorderlineSMOTE + year weights
  ✅ MTF ICT: FVG/IFVG/Breaker/Rejection pe 5m/15m/1h/4h (era lipsă complet)
  ✅ Rolling regime: vix_proxy_5d/20d, vol_regime, atr_trend, adx_10d_mean, hurst_20d_mean
  ✅ Rolling 5-session WR
  ✅ ny_open_drive_bull/bear/neutral (direcția primelor 15min NY)
  ✅ triple_sess_aligned (Asia → LON → NY direction chain)
  ✅ is_smt_bullish/bearish (Smart Money Trap din DB)
  ✅ gap_vs_lon_close_atr (deja exista), fvg_tf_confluence (suma MTF)
  ✅ prev_ny_dir (direcția NY din ziua precedentă)
  ✅ Feature selection top-100 (din ~220 total)

Output: nom_model_v1.pkl
"""

import sqlite3, pickle, logging, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import BorderlineSMOTE
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("NOM_V2")

DB            = Path(__file__).parent / "mario_trading.db"
OUT           = Path(__file__).parent / "nom_model_v1.pkl"
OPTUNA_TRIALS = 80
YEAR_WEIGHTS  = {2023: 0.85, 2024: 1.00}
TOP_N_FEATURES = 70   # mai strict → mai puțin overfit

MIN_SPIKE_PT   = 5.0
MIN_DISP_PT    = 4.0
TP_MULT        = 2.0    # TP = ATR × 2.0 (era 24pt fix → elimină label drift)
LABEL_WINDOW   = 60

TRAIN_YEARS    = [2023, 2024]
TEST_YEARS     = [2025, 2026]
N_WF_FOLDS     = 4

NY_SESS_START_ET = 900
NY_SESS_END_ET   = 1300
PRE_NY_END_ET    = 859
LON_START_ET     = 400
LON_END_ET       = 630
ASIA_START_ET    = 0
ASIA_END_ET      = 359

# ── Economic Calendar ─────────────────────────────────────────────────────────
_CAL_PATH = Path(__file__).parent / "data" / "economic_calendar.json"
try:
    _cal = json.loads(_CAL_PATH.read_text())
    FOMC_DATES   = set(_cal.get('fomc',   []))
    NFP_DATES    = set(_cal.get('nfp',    []))
    CPI_DATES    = set(_cal.get('cpi',    []))
    PPI_DATES    = set(_cal.get('ppi',    []))
    RETAIL_DATES = set(_cal.get('retail', []))
    ISM_DATES    = set(_cal.get('ism',    []))
    ANY_HIGH     = set(_cal.get('any_high', []))
    NEWS_DAYS    = FOMC_DATES | NFP_DATES | CPI_DATES | PPI_DATES
    log.info(f"Calendar: NFP={len(NFP_DATES)}, FOMC={len(FOMC_DATES)}, CPI={len(CPI_DATES)}")
except Exception as _e:
    log.warning(f"Calendar: {_e}")
    FOMC_DATES = NFP_DATES = CPI_DATES = PPI_DATES = RETAIL_DATES = ISM_DATES = ANY_HIGH = NEWS_DAYS = set()

def _fomc_prox(date_str):
    try:
        d = pd.Timestamp(date_str).date()
        diffs = [abs((d - pd.Timestamp(x).date()).days) for x in FOMC_DATES]
        return float(min(diffs)) if diffs else 30.0
    except: return 30.0

def sv(v, d=0.0):
    try: x = float(v); return x if np.isfinite(x) else d
    except: return d


# ════════════════════════════════════════════════════════════════════════════
# MTF ICT (identic cu LOM v2)
# ════════════════════════════════════════════════════════════════════════════
def compute_ict_on_tf(df_tf, lookback=20):
    H = df_tf['high'].values.astype(float); L = df_tf['low'].values.astype(float)
    C = df_tf['close'].values.astype(float); O = df_tf['open'].values.astype(float)
    A = np.maximum(df_tf['atr'].values.astype(float), 1.0)
    n = len(H)
    bull_top = np.zeros(n); bull_bot = np.zeros(n)
    bear_top = np.zeros(n); bear_bot = np.zeros(n)
    for i in range(2, n):
        if H[i-2] < L[i] and (L[i]-H[i-2]) > 0.5: bull_top[i]=L[i]; bull_bot[i]=H[i-2]
        if L[i-2] > H[i] and (L[i-2]-H[i]) > 0.5: bear_top[i]=L[i-2]; bear_bot[i]=H[i]
    in_bull=np.zeros(n); in_bear=np.zeros(n); dist_bull=np.full(n,9.9); dist_bear=np.full(n,9.9)
    in_ifvg_b=np.zeros(n); in_ifvg_s=np.zeros(n); breaker_b=np.zeros(n); breaker_s=np.zeros(n); rejection=np.zeros(n)
    active_bull=[]; active_bear=[]; inv_bull=[]; inv_bear=[]; bull_obs=[]; bear_obs=[]
    for i in range(n):
        c=C[i]; l=L[i]; h=H[i]; a=A[i]
        new_ab=[]
        for top,bot,j in active_bull:
            if i-j>lookback: continue
            if l<bot: inv_bull.append((top,bot,i))
            else: new_ab.append((top,bot,j))
        active_bull=new_ab
        new_ab2=[]
        for top,bot,j in active_bear:
            if i-j>lookback: continue
            if h>top: inv_bear.append((top,bot,i))
            else: new_ab2.append((top,bot,j))
        active_bear=new_ab2
        if bull_top[i]>0: active_bull.append((bull_top[i],bull_bot[i],i))
        if bear_top[i]>0: active_bear.append((bear_top[i],bear_bot[i],i))
        if i>=2:
            pb=C[i-1]-O[i-1]; pr=max(H[i-1]-L[i-1],0.01)
            if pb>0.55*pr and pb>1.0: bull_obs.append((C[i-1],O[i-1],i-1))
            if pb<-0.55*pr and abs(pb)>1.0: bear_obs.append((O[i-1],C[i-1],i-1))
        for top,bot,j in active_bull:
            if bot<=c<=top: in_bull[i]=1.0
            dist_bull[i]=min(dist_bull[i],min(abs(c-top),abs(c-bot))/a)
        for top,bot,j in active_bear:
            if bot<=c<=top: in_bear[i]=1.0
            dist_bear[i]=min(dist_bear[i],min(abs(c-top),abs(c-bot))/a)
        for top,bot,k in inv_bull[-15:]:
            if i-k<=lookback*2 and bot<=c<=top: in_ifvg_b[i]=1.0
        for top,bot,k in inv_bear[-15:]:
            if i-k<=lookback*2 and bot<=c<=top: in_ifvg_s[i]=1.0
        for top,bot,j in bull_obs[-20:]:
            if i-j<=lookback and c<min(bot,O[j])-a*0.05:
                if abs(c-top)/a<0.8 or abs(c-bot)/a<0.8: breaker_s[i]=1.0
        for top,bot,j in bear_obs[-20:]:
            if i-j<=lookback and c>max(top,O[j])+a*0.05:
                if abs(c-top)/a<0.8 or abs(c-bot)/a<0.8: breaker_b[i]=1.0
        if i>=2:
            wu=H[i-1]-max(C[i-1],O[i-1]); wd=min(C[i-1],O[i-1])-L[i-1]; bz=abs(C[i-1]-O[i-1])
            if wu>2.5*max(bz,0.5) and wu>a*0.3:
                rt=H[i-1]; rb=max(C[i-1],O[i-1])
                if abs(c-rt)/a<0.6 or abs(c-rb)/a<0.6: rejection[i]=1.0
            if wd>2.5*max(bz,0.5) and wd>a*0.3:
                rt=min(C[i-1],O[i-1]); rb=L[i-1]
                if abs(c-rt)/a<0.6 or abs(c-rb)/a<0.6: rejection[i]=1.0
    return pd.DataFrame({'in_bull':in_bull,'in_bear':in_bear,
        'dist_bull':np.clip(dist_bull,0,9.9),'dist_bear':np.clip(dist_bear,0,9.9),
        'in_ifvg_b':in_ifvg_b,'in_ifvg_s':in_ifvg_s,
        'breaker_b':breaker_b,'breaker_s':breaker_s,'rejection':rejection}, index=df_tf.index)


def compute_mtf_features(conn, setup_dates):
    min_d=min(setup_dates); max_d=max(setup_dates)
    warmup=(pd.Timestamp(min_d)-pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    log.info(f"   MTF: {warmup} → {max_d} ...")
    df1m=pd.read_sql(f"""SELECT timestamp,open,high,low,close,atr_14
        FROM market_data WHERE timestamp>='{warmup} 00:00:00' AND timestamp<='{max_d} 23:59:59'
        ORDER BY timestamp""", conn)
    df1m['ts']=pd.to_datetime(df1m['timestamp'])
    df1m=df1m.set_index('ts').rename(columns={'atr_14':'atr'})
    df1m['atr']=df1m['atr'].ffill().fillna(9.0)
    log.info(f"   1-min bars: {len(df1m):,}")
    all_features=pd.DataFrame(index=df1m.index)
    for tf_label,tf_rule,lookback in [('5m','5min',25),('15m','15min',20),('1h','1h',20),('4h','4h',15)]:
        df_tf=df1m.resample(tf_rule,label='left',closed='left').agg(
            open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),atr=('atr','last')
        ).dropna(subset=['open'])
        df_tf['atr']=df_tf['atr'].ffill().fillna(9.0)
        ict=compute_ict_on_tf(df_tf,lookback=lookback)
        ict_ff=ict.reindex(df1m.index,method='ffill')
        for col in ict.columns:
            all_features[f'{col}_{tf_label}']=ict_ff[col]
    all_features=all_features.fillna(0.0)
    all_features['ts_str']=all_features.index.strftime('%Y-%m-%d %H:%M:%S')
    log.info(f"   MTF: {all_features.shape[1]-1} cols × {len(all_features):,} rows")
    return all_features


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════
def load_day(conn, date_str):
    df = pd.read_sql(f"""
        SELECT timestamp, open, high, low, close, volume,
               atr_14, bar_delta, cum_delta, fvg_up, fvg_down, has_displacement,
               body_size, adx_14, hurst, dist_poc, inside_va, dist_vwap,
               delta_at_high, delta_at_low, big_buy_count, big_sell_count,
               absorption_score, stacked_bull, stacked_bear, of_doi, of_big_balance,
               bar_buy_vol, bar_sell_vol, garch_vol, kalman_smooth,
               fisher_transform, acf_lag1, acf_lag5, rvol,
               vah, val, poc_level, p_hi, p_lo, lw_hi, lw_lo,
               h4_hi, h4_lo, h1_hi, h1_lo, true_open, asia_hi, asia_lo,
               lon_hi, lon_lo, is_smt_bullish, is_smt_bearish,
               day_of_week, month
        FROM market_data WHERE date='{date_str}' ORDER BY timestamp
    """, conn)
    if len(df) < 30: return None
    df['ts']   = pd.to_datetime(df['timestamp'])
    df['hhmm'] = df['ts'].dt.hour * 100 + df['ts'].dt.minute
    return df


# ════════════════════════════════════════════════════════════════════════════
# Daily rolling context
# ════════════════════════════════════════════════════════════════════════════
def build_daily_context(conn, dates):
    log.info("   Rolling daily regime ...")
    min_d=min(dates); max_d=max(dates)
    warmup=(pd.Timestamp(min_d)-pd.Timedelta(days=40)).strftime('%Y-%m-%d')
    dr=pd.read_sql(f"""SELECT date(timestamp) as date,
               (MAX(high)-MIN(low)) as daily_range,
               AVG(atr_14) as avg_atr, AVG(adx_14) as avg_adx, AVG(hurst) as avg_hurst
        FROM market_data
        WHERE date(timestamp)>='{warmup}' AND date(timestamp)<='{max_d}'
        GROUP BY date(timestamp) ORDER BY date""", conn)
    dr['date']=dr['date'].astype(str)
    dr['avg_atr']=dr['avg_atr'].ffill().fillna(9.0)
    dr['daily_range']=dr['daily_range'].fillna(dr['avg_atr']*2)
    dr['range_atr_ratio']=dr['daily_range']/dr['avg_atr'].clip(lower=1)
    dr['vix_proxy_5d'] =dr['range_atr_ratio'].rolling(5, min_periods=2).mean().shift(1)
    dr['vix_proxy_20d']=dr['range_atr_ratio'].rolling(20,min_periods=5).mean().shift(1)
    dr['vol_regime']   =(dr['vix_proxy_5d']/dr['vix_proxy_20d'].clip(lower=0.5)).clip(upper=3)
    dr['adx_10d_mean'] =dr['avg_adx'].rolling(10,min_periods=3).mean().shift(1)
    dr['hurst_20d_mean']=dr['avg_hurst'].rolling(20,min_periods=5).mean().shift(1)
    dr['atr_5d']       =dr['avg_atr'].rolling(5, min_periods=2).mean().shift(1)
    dr['atr_10d']      =dr['avg_atr'].rolling(10,min_periods=3).mean().shift(1)
    dr['atr_trend']    =(dr['atr_5d']/dr['atr_10d'].clip(lower=1)).clip(upper=3)
    dr=dr.ffill().fillna(1.0)
    return {r['date']:r.to_dict() for _,r in dr.iterrows()}


# ════════════════════════════════════════════════════════════════════════════
# Setup extraction — event-driven NOM cu features complete
# ════════════════════════════════════════════════════════════════════════════
def _days_since_win(win_list, current_date_str):
    cd = pd.Timestamp(current_date_str)
    wins = [pd.Timestamp(d) for d, lbl in reversed(win_list) if lbl == 1]
    return float((cd - wins[0]).days) if wins else 30.0


def extract_setups(df, date_str, daily_ctx, cross_ctx=None):
    setups = []
    pre_ny   = df[df['hhmm'] <= PRE_NY_END_ET]
    ny_sess  = df[df['hhmm'].between(NY_SESS_START_ET, NY_SESS_END_ET)]
    london   = df[df['hhmm'].between(LON_START_ET, LON_END_ET)]
    asia_df  = df[df['hhmm'].between(ASIA_START_ET, ASIA_END_ET)]

    if len(pre_ny) < 20 or len(ny_sess) < 5: return setups

    pre_hi  = float(pre_ny['high'].max())
    pre_lo  = float(pre_ny['low'].min())
    pre_rng = pre_hi - pre_lo
    if pre_rng < 5: return setups

    atr = float(df['atr_14'].replace(0, np.nan).dropna().iloc[-1]) if len(df) > 0 else 10.0
    if atr <= 0: atr = 10.0

    # LON context (NOM: swept level = LON hi/lo)
    if len(london) > 0:
        lon_hi  = float(london['high'].max());  lon_lo   = float(london['low'].min())
        lon_rng = lon_hi - lon_lo;              lon_mid  = (lon_hi + lon_lo) / 2
        lon_close= float(london['close'].iloc[-1])
        lon_open = float(london['open'].iloc[0])
    else:
        lon_hi=pre_hi; lon_lo=pre_lo; lon_rng=pre_rng; lon_mid=(pre_hi+pre_lo)/2
        lon_close=lon_mid; lon_open=lon_mid

    # Asia context
    if len(asia_df) > 0:
        asia_hi_v  = float(asia_df['high'].max());  asia_lo_v  = float(asia_df['low'].min())
        asia_close = float(asia_df['close'].iloc[-1]); asia_open  = float(asia_df['open'].iloc[0])
        asia_rng   = asia_hi_v - asia_lo_v;  asia_mid  = (asia_hi_v + asia_lo_v) / 2
        asia_dir   = 1 if asia_close > asia_mid else -1
    else:
        asia_hi_v=pre_hi; asia_lo_v=pre_lo; asia_rng=pre_rng; asia_mid=pre_hi
        asia_close=pre_lo; asia_open=pre_hi; asia_dir=0

    # NY open drive (first 15min)
    ny15 = ny_sess[ny_sess['hhmm'].between(NY_SESS_START_ET, NY_SESS_START_ET + 15)]
    if len(ny15) > 0:
        ny15_hi = float(ny15['high'].max()); ny15_lo = float(ny15['low'].min())
        ny15_open = float(ny15['open'].iloc[0]); ny15_close = float(ny15['close'].iloc[-1])
        ny15_rng  = ny15_hi - ny15_lo
        ny15_move = ny15_close - ny15_open
        ny_drive_bull   = 1 if ny15_move >  atr * 0.15 else 0
        ny_drive_bear   = 1 if ny15_move < -atr * 0.15 else 0
        ny_drive_neutral= 1 if abs(ny15_move) <= atr * 0.15 else 0
        ny_drive_rng_atr= ny15_rng / atr
    else:
        ny_drive_bull=0; ny_drive_bear=0; ny_drive_neutral=1; ny_drive_rng_atr=0.5

    # Equal highs/lows in pre-NY range
    eq_tol    = atr * 0.3
    pre_highs = pre_ny['high'].values; pre_lows = pre_ny['low'].values
    eq_hi = max(0, sum(1 for h in pre_highs if abs(h - pre_hi) <= eq_tol) - 1)
    eq_lo = max(0, sum(1 for l in pre_lows  if abs(l - pre_lo) <= eq_tol) - 1)

    # NY open price
    ny_open_price = float(ny_sess['open'].iloc[0]) if len(ny_sess) > 0 else pre_hi

    partial_thresh = pre_rng * 0.50
    lon_reset = ny_sess.reset_index(drop=False)
    last_setup_hhmm = {'LONG': -999, 'SHORT': -999}

    for i in range(1, len(lon_reset) - 2):
        bar      = lon_reset.iloc[i]
        bar_hi   = sv(bar['high']); bar_lo = sv(bar['low']); bar_hhmm = int(bar['hhmm'])

        sweep_up = bar_hi - pre_hi;  sweep_dn = pre_lo - bar_lo

        for direction, spike_mag_raw, is_valid in [
            ('SHORT', max(sweep_up, 0), sweep_up >= MIN_SPIKE_PT or (sweep_up > 0 and sweep_up >= partial_thresh)),
            ('LONG',  max(sweep_dn, 0), sweep_dn >= MIN_SPIKE_PT or (sweep_dn > 0 and sweep_dn >= partial_thresh)),
        ]:
            if not is_valid or (bar_hhmm - last_setup_hhmm[direction]) < 30: continue

            spike_mag = spike_mag_raw; spike_hi_val = bar_hi; spike_lo_val = bar_lo

            after_spike = lon_reset[lon_reset['hhmm'].between(bar_hhmm + 1, bar_hhmm + 45)]
            disp_bar = None
            for _, ab in after_spike.iterrows():
                ab_body = abs(sv(ab['close']) - sv(ab['open']))
                if direction == 'SHORT' and sv(ab['close']) < sv(ab['open']) and ab_body >= MIN_DISP_PT:
                    disp_bar = ab; break
                elif direction == 'LONG' and sv(ab['close']) > sv(ab['open']) and ab_body >= MIN_DISP_PT:
                    disp_bar = ab; break
            if disp_bar is None: continue

            entry_price = sv(disp_bar['close'])
            entry_hhmm  = int(disp_bar['hhmm'])
            entry_ts    = str(disp_bar['timestamp'])
            dir_num     = 1 if direction == 'LONG' else -1

            future = df[df['hhmm'] > entry_hhmm].head(LABEL_WINDOW)
            if len(future) < 3: continue
            atr_tp = atr * TP_MULT   # TP ATR-relativ → elimină label drift
            if direction == 'LONG':
                reached_tp = float(future['high'].max()) >= entry_price + atr_tp
                max_fwd    = float(future['high'].max() - entry_price)
            else:
                reached_tp = float(future['low'].min()) <= entry_price - atr_tp
                max_fwd    = float(entry_price - future['low'].min())
            label = 1 if reached_tp else 0

            r0 = df.iloc[-1]
            spike_bar_range = max(sv(bar['high'] - bar['low']), 0.01)
            after_early = lon_reset[lon_reset['hhmm'].between(bar_hhmm + 1, bar_hhmm + 45)]

            if direction == 'SHORT':
                ts_close_inside  = 1 if sv(bar['close']) <= pre_hi else 0
                wick             = (sv(bar['high']) - max(sv(bar['close']), sv(bar['open']))) / atr
                ts_rejection_str = (spike_hi_val - sv(bar['close'])) / spike_mag if spike_mag > 0 else 0
                ts_wick_pct      = (spike_hi_val - sv(bar['close'])) / spike_bar_range
                ts_body_pct      = abs(sv(bar['open']) - sv(bar['close'])) / spike_bar_range
                ts_close_quality = max(0, (pre_hi - sv(bar['close'])) / pre_rng) if pre_rng > 0 else 0
            else:
                ts_close_inside  = 1 if sv(bar['close']) >= pre_lo else 0
                wick             = (min(sv(bar['close']), sv(bar['open'])) - sv(bar['low'])) / atr
                ts_rejection_str = (sv(bar['close']) - spike_lo_val) / spike_mag if spike_mag > 0 else 0
                ts_wick_pct      = (sv(bar['close']) - spike_lo_val) / spike_bar_range
                ts_body_pct      = abs(sv(bar['open']) - sv(bar['close'])) / spike_bar_range
                ts_close_quality = max(0, (sv(bar['close']) - pre_lo) / pre_rng) if pre_rng > 0 else 0

            wick_pct        = wick * atr / spike_bar_range
            sweep_wick_clean= 1 if wick_pct > 0.5 else 0
            sweep_depth_atr = spike_mag / atr
            deep_sweep      = 1 if sweep_depth_atr > 1.5 else 0
            sweep_quality   = ts_close_inside*0.4 + sweep_wick_clean*0.3 + deep_sweep*0.2 + 0.1

            disp_body = abs(sv(disp_bar['close']) - sv(disp_bar['open']))

            h4_hi=sv(r0['h4_hi']); h4_lo=sv(r0['h4_lo'])
            h1_hi=sv(r0['h1_hi']); h1_lo=sv(r0['h1_lo'])
            h4_mid=(h4_hi+h4_lo)/2 if h4_hi>0 and h4_lo>0 else 0
            h4_bias= 1 if entry_price < h4_mid else (-1 if h4_mid > 0 else 0)
            h1_mid = (h1_hi+h1_lo)/2 if h1_hi>0 and h1_lo>0 else 0
            h1_bias= 1 if entry_price < h1_mid else (-1 if h1_mid > 0 else 0)

            lw_hi=sv(r0['lw_hi']); lw_lo=sv(r0['lw_lo']); lw_rng=lw_hi-lw_lo
            weekly_prem=(entry_price-lw_lo)/lw_rng if lw_rng>0 else 0.5

            pre_vol  = float(pre_ny['volume'].sum()) if len(pre_ny)>0 else 1.0
            ny_vol   = float(ny_sess['volume'].sum()) if len(ny_sess)>0 else 1.0
            vol_ratio= ny_vol/pre_vol if pre_vol>0 else 1.0

            spike_delta= sv(ny_sess['bar_delta'].sum()) if len(ny_sess)>0 else 0
            fvg_up_v  = int(ny_sess['fvg_up'].any())   if 'fvg_up'   in ny_sess.columns else 0
            fvg_down_v= int(ny_sess['fvg_down'].any()) if 'fvg_down' in ny_sess.columns else 0
            adx_v     = sv(r0['adx_14']); hurst_v = sv(r0['hurst'], 0.5)

            # LON direction for triple alignment
            lon_dir = 1 if lon_close > lon_mid else -1
            # Triple session alignment: Asia → LON → entry direction
            triple_aligned = 1 if (asia_dir == dir_num and lon_dir != dir_num) else 0

            # Rolling regime
            dctx   = daily_ctx.get(date_str, {})
            vix5   = dctx.get('vix_proxy_5d',   2.0)
            vix20  = dctx.get('vix_proxy_20d',  2.0)
            vol_rg = dctx.get('vol_regime',     1.0)
            atr_tr = dctx.get('atr_trend',      1.0)
            adx10  = dctx.get('adx_10d_mean',   20.0)
            hst20  = dctx.get('hurst_20d_mean', 0.5)
            atr5d  = dctx.get('atr_5d',         atr)
            roll_wr= dctx.get('rolling_wr',     0.5)
            prev_ny_dir = dctx.get('prev_ny_dir', 0)
            cctx   = cross_ctx or {}
            dsw_dir= cctx.get('dsw_L' if direction=='LONG' else 'dsw_S', 30.0)
            dsw_any= min(cctx.get('dsw_L',30.0), cctx.get('dsw_S',30.0))
            wk_cnt = float(cctx.get('week_cnt', 0))
            td_cnt = float(cctx.get('td_L' if direction=='LONG' else 'td_S', 0))

            feat = {
                # ── Spike ───────────────────────────────────────────────────
                'spike_mag':            spike_mag,
                'spike_mag_atr':        spike_mag / atr,
                'spike_vs_range':       spike_mag / pre_rng if pre_rng > 0 else 0,
                'pre_rng_atr':          pre_rng / atr,
                # ── TS anti-fakeout ─────────────────────────────────────────
                'ts_close_inside':      ts_close_inside,
                'ts_rejection_str':     ts_rejection_str,
                'ts_wick_pct':          ts_wick_pct,
                'ts_body_pct':          ts_body_pct,
                'ts_close_quality':     ts_close_quality,
                'ts_wick_dom':          1 if ts_wick_pct > 0.6 else 0,
                'ts_htf_anti':          1 if h4_bias == dir_num else 0,
                'ts_combo_score':       ts_close_inside * ts_rejection_str,
                'ts_sweep_depth_pts':   spike_mag,
                'ts_sweep_depth_atr':   sweep_depth_atr,
                'ts_sweep_pct_lon':     spike_mag / lon_rng if lon_rng > 0 else 0,
                'ts_lon_mid_dist':      (entry_price - lon_mid) / atr,
                # ── Sweep quality ────────────────────────────────────────────
                'sweep_wick_atr':       wick,
                'sweep_wick_pct':       wick_pct,
                'sweep_wick_clean':     sweep_wick_clean,
                'sweep_depth_atr':      sweep_depth_atr,
                'deep_sweep':           deep_sweep,
                'shallow_sweep':        1 if sweep_depth_atr < 0.5 else 0,
                'sweep_with_disp':      1,
                'sweep_quality_score':  sweep_quality,
                'equal_level_score':    (eq_hi if direction=='SHORT' else eq_lo)/max(len(pre_ny),1),
                'equal_hi_count':       float(eq_hi),
                'equal_lo_count':       float(eq_lo),
                # ── Displacement ─────────────────────────────────────────────
                'disp_body':            disp_body,
                'disp_body_atr':        disp_body / atr,
                'disp_range':           sv(disp_bar['high'] - disp_bar['low']),
                'disp_wick_ratio':      (sv(disp_bar['high']-disp_bar['low'])-disp_body)/max(disp_body,0.01),
                'has_disp':             1,
                'body_pct':             disp_body/max(sv(disp_bar['high']-disp_bar['low']),0.01),
                'body_bear':            1 if direction == 'SHORT' else 0,
                # ── HTF ──────────────────────────────────────────────────────
                'h4_bias':              h4_bias,
                'h1_bias':              h1_bias,
                'h4_h1_aligned':        1 if h4_bias==h1_bias and h4_bias!=0 else 0,
                'h4_bias_aligned':      1 if h4_bias == dir_num else 0,
                # ── Weekly ───────────────────────────────────────────────────
                'weekly_premium_pct':   weekly_prem,
                'in_weekly_premium':    1 if weekly_prem > 0.5 else 0,
                'in_weekly_discount':   1 if weekly_prem < 0.5 else 0,
                'weekly_prem_aligned':  1 if (direction=='SHORT' and weekly_prem>0.5) or (direction=='LONG' and weekly_prem<0.5) else 0,
                'h4_x_weekly':          (1 if h4_bias==dir_num else 0)*(1 if (direction=='SHORT' and weekly_prem>0.5) or (direction=='LONG' and weekly_prem<0.5) else 0),
                'lw_range_atr':         lw_rng/atr if atr>0 else 0,
                'dist_lw_hi':           abs(entry_price-lw_hi)/atr,
                'dist_lw_lo':           abs(entry_price-lw_lo)/atr,
                # ── LON context (NOM: swept levels = LON hi/lo) ──────────────
                'lon_range_atr':        lon_rng / atr,
                'dist_lon_hi_atr':      abs(entry_price - lon_hi) / atr,
                'dist_lon_lo_atr':      abs(entry_price - lon_lo) / atr,
                'lon_close_vs_mid':     (lon_close - lon_mid) / lon_rng if lon_rng > 0 else 0,
                'gap_vs_lon_close_atr': (ny_open_price - lon_close) / atr,
                'ny15_range_atr':       ny_drive_rng_atr,
                'lon_dir_explicit':     float(lon_dir),
                'lon_dir_aligned':      1 if lon_dir == dir_num else 0,
                'lon_big_day':          1 if lon_rng > atr * 1.5 else 0,
                'lon_small_day':        1 if lon_rng < atr * 0.7 else 0,
                'ny_open_in_lon':       1 if lon_lo <= ny_open_price <= lon_hi else 0,
                'ny_open_above_lon_mid':1 if ny_open_price > lon_mid else 0,
                # ── NY open drive (NEW) ──────────────────────────────────────
                'ny_open_drive_bull':   float(ny_drive_bull),
                'ny_open_drive_bear':   float(ny_drive_bear),
                'ny_open_drive_neutral':float(ny_drive_neutral),
                'drive_aligned_dir':    1.0 if (ny_drive_bull and dir_num==1) or (ny_drive_bear and dir_num==-1) else 0.0,
                # ── Asia context ─────────────────────────────────────────────
                'dist_asia_hi_atr':     abs(entry_price-asia_hi_v)/atr if asia_hi_v>0 else 0,
                'dist_asia_lo_atr':     abs(entry_price-asia_lo_v)/atr if asia_lo_v>0 else 0,
                'asia_range_atr':       asia_rng/atr if asia_rng>0 else 0,
                'spike_vs_asia_hi':     (spike_hi_val-asia_hi_v)/atr if asia_hi_v>0 else 0,
                'spike_vs_asia_lo':     (asia_lo_v-spike_lo_val)/atr if asia_lo_v>0 else 0,
                'asia_dir_explicit':    float(asia_dir),
                'asia_dir_aligned':     1 if asia_dir == dir_num else 0,
                'asia_range_vs_atr5d':  float(np.clip(asia_rng/max(atr5d,1.0),0,10)),
                # ── Triple session alignment (NEW) ────────────────────────────
                'triple_sess_aligned':  float(triple_aligned),
                'lon_asia_aligned':     1 if lon_dir == asia_dir else 0,
                'full_alignment':       1 if lon_dir==asia_dir==dir_num else 0,
                # ── SMT (Smart Money Trap — NEW) ─────────────────────────────
                'is_smt_bullish':       sv(r0.get('is_smt_bullish', 0)),
                'is_smt_bearish':       sv(r0.get('is_smt_bearish', 0)),
                'smt_aligned':          sv(r0.get('is_smt_bullish',0)) if dir_num==1 else sv(r0.get('is_smt_bearish',0)),
                # ── PDH/PDL ──────────────────────────────────────────────────
                'above_true_open':      1 if entry_price > sv(r0['true_open']) else 0,
                'dist_true_open':       abs(entry_price-sv(r0['true_open']))/atr,
                'dist_pdh_atr':         abs(entry_price-sv(r0['p_hi']))/atr,
                'dist_pdl_atr':         abs(entry_price-sv(r0['p_lo']))/atr,
                # ── VA / POC ─────────────────────────────────────────────────
                'inside_va':            sv(r0['inside_va']),
                'dist_poc_atr':         sv(r0['dist_poc'])/atr,
                'dist_vwap_atr':        sv(r0['dist_vwap'])/atr,
                'entry_in_pre_range':   int(pre_lo <= entry_price <= pre_hi),
                # ── Volume / delta ────────────────────────────────────────────
                'vol_ratio':            vol_ratio,
                'spike_delta':          spike_delta,
                'disp_delta':           sv(after_early['bar_delta'].sum()) if len(after_early)>0 else 0,
                'delta_at_high':        sv(ny_sess['delta_at_high'].sum()) if 'delta_at_high' in ny_sess.columns else 0,
                'delta_at_low':         sv(ny_sess['delta_at_low'].sum())  if 'delta_at_low'  in ny_sess.columns else 0,
                'big_buy':              1 if vol_ratio>2 and direction=='LONG' else 0,
                'big_sell':             1 if vol_ratio>2 and direction=='SHORT' else 0,
                'big_imbalance':        1 if vol_ratio>2 else 0,
                'absorption':           sv(ny_sess['absorption_score'].mean()) if 'absorption_score' in ny_sess.columns else 0,
                'bar_delta_norm':       spike_delta/atr,
                'buy_sell_ratio':       sv(ny_sess['bar_buy_vol'].sum())/max(sv(ny_sess['bar_sell_vol'].sum()),1),
                'of_doi':               sv(ny_sess['of_doi'].mean()) if 'of_doi' in ny_sess.columns else 0,
                'stacked_bull':         int(ny_sess['stacked_bull'].any()) if 'stacked_bull' in ny_sess.columns else 0,
                'stacked_bear':         int(ny_sess['stacked_bear'].any()) if 'stacked_bear' in ny_sess.columns else 0,
                'fvg_up':               fvg_up_v,
                'fvg_down':             fvg_down_v,
                'htf_fvg_aligned':      1 if (direction=='SHORT' and fvg_down_v) or (direction=='LONG' and fvg_up_v) else 0,
                'vol_x_fvg':            vol_ratio*(1 if (direction=='LONG' and fvg_up_v) or (direction=='SHORT' and fvg_down_v) else 0),
                # ── Technical ────────────────────────────────────────────────
                'adx':                  adx_v,
                'adx_strong':           1 if adx_v > 25 else 0,
                'hurst':                hurst_v,
                'fisher_transform':     sv(r0['fisher_transform']),
                'fisher_extreme':       1 if abs(sv(r0['fisher_transform']))>2 else 0,
                'acf_lag1':             sv(r0['acf_lag1']),
                'acf_lag5':             sv(r0['acf_lag5']),
                'kalman_smooth':        sv(r0['kalman_smooth']),
                'garch_vol':            sv(r0['garch_vol']),
                'rvol':                 sv(r0['rvol'], 1.0),
                # ── Rolling regime (NEW) ──────────────────────────────────────
                'vix_proxy_5d':         float(vix5),
                'vix_proxy_20d':        float(vix20),
                'vol_regime':           float(vol_rg),
                'vol_high':             1 if vol_rg > 1.2 else 0,
                'vol_low':              1 if vol_rg < 0.8 else 0,
                'atr_trend':            float(atr_tr),
                'atr_expanding':        1 if atr_tr > 1.15 else 0,
                'adx_10d_mean':         float(adx10),
                'hurst_20d_mean':       float(hst20),
                'atr_vs_5d':            float(np.clip(atr/max(atr5d,1.0),0,3)),
                'regime_trending':      1 if adx10>22 and hst20>0.52 else 0,
                'rolling_5sess_wr':     float(roll_wr),
                'recent_wr_high':       1 if roll_wr > 0.35 else 0,
                # ── Calendar ─────────────────────────────────────────────────
                'is_nfp_day':           1 if date_str in NFP_DATES    else 0,
                'is_fomc_day':          1 if date_str in FOMC_DATES   else 0,
                'is_cpi_day':           1 if date_str in CPI_DATES    else 0,
                'is_ppi_day':           1 if date_str in PPI_DATES    else 0,
                'is_retail_day':        1 if date_str in RETAIL_DATES else 0,
                'is_ism_day':           1 if date_str in ISM_DATES    else 0,
                'is_news_day':          1 if date_str in NEWS_DAYS    else 0,
                'fomc_proximity':       float(np.clip(_fomc_prox(date_str)/14.0,0,1)),
                'is_pre_nfp':           1 if (date_str in NFP_DATES   and entry_hhmm < 830) else 0,
                'is_post_nfp':          1 if (date_str in NFP_DATES   and entry_hhmm >= 830) else 0,
                'is_pre_cpi':           1 if (date_str in CPI_DATES   and entry_hhmm < 830) else 0,
                'is_post_cpi':          1 if (date_str in CPI_DATES   and entry_hhmm >= 830) else 0,
                'is_pre_ppi':           1 if (date_str in PPI_DATES   and entry_hhmm < 830) else 0,
                'is_post_ppi':          1 if (date_str in PPI_DATES   and entry_hhmm >= 830) else 0,
                'is_fomc_wait':         1 if date_str in FOMC_DATES   else 0,
                # ── Time ─────────────────────────────────────────────────────
                'day_of_week':          sv(r0['day_of_week']),
                'is_monday':            1 if int(sv(r0['day_of_week']))==0 else 0,
                'is_tuesday':           1 if int(sv(r0['day_of_week']))==1 else 0,
                'is_wednesday':         1 if int(sv(r0['day_of_week']))==2 else 0,
                'is_thursday':          1 if int(sv(r0['day_of_week']))==3 else 0,
                'is_friday':            1 if int(sv(r0['day_of_week']))==4 else 0,
                'month':                sv(r0['month']),
                'sweep_time_early':     1 if bar_hhmm <= NY_SESS_START_ET + 30 else 0,
                'sweep_time_mid':       1 if NY_SESS_START_ET+30 < bar_hhmm <= NY_SESS_START_ET+120 else 0,
                'sweep_time_late':      1 if bar_hhmm > NY_SESS_START_ET+120 else 0,
                # ── Previous NY direction (NEW) ───────────────────────────────
                'prev_ny_dir':          float(prev_ny_dir),
                'prev_ny_aligned':      1 if prev_ny_dir == dir_num else 0,
                'prev_ny_opposite':     1 if prev_ny_dir == -dir_num else 0,
                # ── Interactions ─────────────────────────────────────────────
                'dir_x_adx':            dir_num * adx_v,
                'dir_x_hurst':          dir_num * hurst_v,
                'sweep_x_h4':           sweep_quality * (1 if h4_bias==dir_num else 0),
                'vol_x_sweep':          vol_rg * sweep_quality,
                'vol_x_ts_close':       vol_rg * ts_close_inside,
                'drive_x_sweep':        (1.0 if (ny_drive_bull and dir_num==1) or (ny_drive_bear and dir_num==-1) else 0.0) * sweep_quality,
                'triple_x_h4':          float(triple_aligned) * (1 if h4_bias==dir_num else 0),
                'smt_x_sweep':          sv(r0.get('is_smt_bullish',0) if dir_num==1 else r0.get('is_smt_bearish',0)) * sweep_quality,
                'nfp_post_x_sweep':     (1 if (date_str in NFP_DATES and entry_hhmm>=830) else 0) * sweep_quality,
                'asia_dir_x_lon_dir':   float(asia_dir) * float(lon_dir),
                # ── Direction ────────────────────────────────────────────────
                # ── Normalizare completă ──────────────────────────────────────
                'disp_range_atr':       sv(disp_bar['high']-disp_bar['low'])/atr,
                'atr_tp_norm':          float(np.clip(atr/20.0, 0.5, 3.0)),
                # ── Cross-setup context ───────────────────────────────────────
                'days_since_win_dir':   float(np.clip(dsw_dir,0,30)),
                'days_since_win_any':   float(np.clip(dsw_any,0,30)),
                'week_setup_count':     float(np.clip(wk_cnt,0,10)),
                'today_same_dir_cnt':   float(np.clip(td_cnt,0,5)),
                'hot_streak':           1 if dsw_dir<=2 else 0,
                'cold_streak':          1 if dsw_dir>=7 else 0,
                'first_today':          1 if td_cnt==0 else 0,
                # ── Direction ────────────────────────────────────────────────
                'direction_enc':        1 if direction == 'SHORT' else 0,
                # ── Meta ─────────────────────────────────────────────────────
                '_label':      label,
                '_direction':  direction,
                '_date':       str(date_str),
                '_entry_px':   entry_price,
                '_max_fwd':    max_fwd,
                '_entry_hhmm': entry_hhmm,
                '_entry_ts':   entry_ts,
            }
            setups.append(feat)
            last_setup_hhmm[direction] = bar_hhmm
            break
    return setups


# ════════════════════════════════════════════════════════════════════════════
# Dataset build
# ════════════════════════════════════════════════════════════════════════════
def build_dataset(years):
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    days = pd.read_sql(f"""SELECT DISTINCT date FROM market_data
        WHERE year IN ({','.join(map(str,years))}) AND day_of_week BETWEEN 1 AND 5
        ORDER BY date""", conn)['date'].tolist()

    daily_ctx = build_daily_context(conn, days)

    all_setups   = []
    wr_window    = []
    prev_ny_dir  = 0
    win_hist     = {'LONG': [], 'SHORT': []}
    week_counts  = {}
    for date_str in days:
        df = load_day(conn, date_str)
        if df is None: continue

        roll_wr = float(np.mean(wr_window[-5:])) if wr_window else 0.5
        if date_str in daily_ctx:
            daily_ctx[date_str]['rolling_wr']  = roll_wr
            daily_ctx[date_str]['prev_ny_dir'] = prev_ny_dir

        wk = pd.Timestamp(date_str).isocalendar()
        week_str = f"{wk.year}_{wk.week}"
        cross_ctx = {
            'dsw_L':   _days_since_win(win_hist['LONG'],  date_str),
            'dsw_S':   _days_since_win(win_hist['SHORT'], date_str),
            'week_cnt': float(week_counts.get(week_str, 0)),
            'td_L': 0.0, 'td_S': 0.0,
        }
        setups = extract_setups(df, date_str, daily_ctx, cross_ctx)
        all_setups.extend(setups)

        ny_bars = df[df['hhmm'].between(NY_SESS_START_ET, NY_SESS_END_ET)]
        if len(ny_bars) >= 5:
            nhi=float(ny_bars['high'].max()); nlo=float(ny_bars['low'].min())
            ncl=float(ny_bars['close'].iloc[-1]); nmid=(nhi+nlo)/2
            prev_ny_dir = 1 if ncl > nmid else -1

        for s in setups:
            d = s['_direction']
            win_hist[d].append((date_str, s['_label']))
            week_counts[week_str] = week_counts.get(week_str, 0) + 1
            wr_window.append(s['_label'])

    conn.close()
    log.info(f"  {years}: {len(days)} zile → {len(all_setups)} setups")

    if not all_setups: return pd.DataFrame()
    df_out = pd.DataFrame(all_setups)

    # MTF join
    log.info("   MTF join ...")
    conn2 = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    mtf = compute_mtf_features(conn2, sorted(df_out['_date'].unique()))
    conn2.close()

    df_out = df_out.merge(
        mtf.drop_duplicates('ts_str')[['ts_str']+[c for c in mtf.columns if c!='ts_str']],
        left_on='_entry_ts', right_on='ts_str', how='left'
    )
    for c in [c for c in mtf.columns if c!='ts_str']:
        df_out[c] = df_out[c].fillna(0.0)

    # Derived MTF features
    for tf_label in ['5m','15m','1h','4h']:
        in_bull=df_out[f'in_bull_{tf_label}'].values; in_bear=df_out[f'in_bear_{tf_label}'].values
        in_ifvg_b=df_out[f'in_ifvg_b_{tf_label}'].values; in_ifvg_s=df_out[f'in_ifvg_s_{tf_label}'].values
        brk_b=df_out[f'breaker_b_{tf_label}'].values; brk_s=df_out[f'breaker_s_{tf_label}'].values
        dir_n=np.where(df_out['direction_enc'].values==0,1.0,-1.0)
        df_out[f'fvg_aligned_{tf_label}']=np.where(dir_n==1,in_bull,in_bear)
        df_out[f'ifvg_aligned_{tf_label}']=np.where(dir_n==1,in_ifvg_s,in_ifvg_b)
        df_out[f'breaker_aligned_{tf_label}']=np.where(dir_n==1,brk_b,brk_s)

    df_out['fvg_tf_confluence']=(df_out.get('fvg_aligned_5m',pd.Series(0,index=df_out.index)).values+
        df_out.get('fvg_aligned_15m',pd.Series(0,index=df_out.index)).values+
        df_out.get('fvg_aligned_1h',pd.Series(0,index=df_out.index)).values+
        df_out.get('fvg_aligned_4h',pd.Series(0,index=df_out.index)).values)
    df_out['htf_fvg_aligned_mtf']=np.maximum(
        df_out.get('fvg_aligned_1h',pd.Series(0,index=df_out.index)).values,
        df_out.get('fvg_aligned_4h',pd.Series(0,index=df_out.index)).values)
    df_out['vol_x_htf_fvg']=df_out['vol_regime'].values*df_out['htf_fvg_aligned_mtf'].values
    df_out['drive_x_fvg_1h']=df_out['drive_aligned_dir'].values*df_out.get('fvg_aligned_1h',pd.Series(0,index=df_out.index)).values

    # ── Synthetic Order Flow features ─────────────────────────────────────
    _OF_PATH = Path(__file__).parent / "data" / "orderflow_features.parquet"
    if _OF_PATH.exists():
        import pandas as _pd2
        _of = _pd2.read_parquet(_OF_PATH)
        _of = _of[_of['session_type'] == 'NY'].copy()
        _of['date'] = _of['date'].astype(str)
        _OF_COLS = [c for c in _of.columns if c not in ['session_id','date','session_type',
                    'session_open','session_close','session_high','session_low','total_vol']]
        _of_m = _of[['date'] + _OF_COLS].rename(columns={'date': '_date'})
        df_out = df_out.merge(_of_m, on='_date', how='left')
        for _c in _OF_COLS:
            df_out[_c] = df_out[_c].fillna(0.0)
        log.info(f"   Order flow: {len(_OF_COLS)} features merged (NY)")
    log.info(f"   Total columns after MTF: {df_out.shape[1]}")
    return df_out


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════
def train_and_save():
    log.info("═"*60)
    log.info("NOM TRAIN v2 — Enhanced (Optuna+MTF+Rolling+SMT+Drive)")
    log.info("═"*60)

    log.info(f"IS ({TRAIN_YEARS})...")
    df_tr = build_dataset(TRAIN_YEARS)
    log.info(f"OOS ({TEST_YEARS})...")
    df_te = build_dataset(TEST_YEARS)

    meta_cols    = [c for c in df_tr.columns if c.startswith('_') or c=='ts_str']
    feature_cols = [c for c in df_tr.columns if c not in meta_cols]
    log.info(f"\nIS: {len(df_tr)} setups | {len(feature_cols)} features")
    log.info(f"OOS: {len(df_te)} setups | label: {df_te['_label'].value_counts().to_dict()}")
    if len(df_tr) < 50:
        log.error("Prea puțin data IS"); return

    X_tr = df_tr[feature_cols].fillna(0)
    y_tr = df_tr['_label']
    yr_  = df_tr['_date'].apply(lambda d: int(d[:4]))
    sw_  = np.array([YEAR_WEIGHTS.get(yr,1.0) for yr in yr_])
    X_te = df_te[feature_cols].fillna(0).reindex(columns=feature_cols, fill_value=0)
    y_te = df_te['_label']

    # Feature selection
    log.info(f"\n▶  Feature selection (top {TOP_N_FEATURES}) ...")
    neg,pos=(y_tr==0).sum(),(y_tr==1).sum(); _spw=neg/max(pos,1)
    _pre=xgb.XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,
        subsample=0.7,colsample_bytree=0.6,min_child_weight=25,gamma=1.5,
        reg_alpha=2.0,reg_lambda=4.0,scale_pos_weight=_spw,
        random_state=42,n_jobs=-1,use_label_encoder=False,eval_metric='logloss',verbosity=0)
    _pre.fit(X_tr,y_tr,sample_weight=sw_,verbose=False)
    _imp=pd.Series(_pre.feature_importances_,index=feature_cols).sort_values(ascending=False)
    selected=_imp.head(TOP_N_FEATURES).index.tolist()
    log.info(f"   Top5: {selected[:5]}")
    X_tr=X_tr[selected]; X_te=X_te.reindex(columns=selected,fill_value=0); feature_cols=selected

    # Walk-forward CV folds
    ts_tr  = pd.DatetimeIndex(pd.to_datetime(df_tr['_date']))
    y_tr_arr = y_tr.values

    def make_wf_folds(dates, n_folds, min_train_m=8, val_m=4):
        min_d=dates.min(); folds=[]
        for i in range(n_folds):
            tr_end=min_d+pd.DateOffset(months=min_train_m+i*val_m)
            vl_end=tr_end+pd.DateOffset(months=val_m)
            tm=np.array(dates<tr_end); vm=np.array((dates>=tr_end)&(dates<vl_end))
            if tm.sum()>=50 and vm.sum()>=20: folds.append((tm,vm))
        log.info(f"   Walk-forward: {len(folds)} folds"); return folds

    wf_folds=make_wf_folds(ts_tr, N_WF_FOLDS)
    if not wf_folds:
        val_cut=int(len(X_tr)*0.80)
        wf_folds=[(np.array([True]*val_cut+[False]*(len(X_tr)-val_cut)),
                   np.array([False]*val_cut+[True]*(len(X_tr)-val_cut)))]
        log.warning("   Fallback la split 80/20")

    def objective(trial):
        params={
            'n_estimators':     trial.suggest_int('n_estimators',150,800),
            'max_depth':        trial.suggest_int('max_depth',2,3),        # max 3 — dur anti-overfit
            'learning_rate':    trial.suggest_float('learning_rate',0.005,0.05,log=True),
            'subsample':        trial.suggest_float('subsample',0.5,0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree',0.35,0.75),
            'min_child_weight': trial.suggest_int('min_child_weight',20,80),
            'gamma':            trial.suggest_float('gamma',1.0,8.0),
            'reg_alpha':        trial.suggest_float('reg_alpha',1.0,8.0),
            'reg_lambda':       trial.suggest_float('reg_lambda',3.0,10.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight',3.0,12.0),
        }
        smote_r=trial.suggest_float('smote_ratio',0.10,0.40)
        fold_aucs=[]
        for tm,vm in wf_folds:
            Xf=X_tr[tm]; yf=y_tr_arr[tm]; swf=sw_[tm]
            Xv=X_tr[vm]; yv=y_tr_arr[vm]
            try:
                sm=BorderlineSMOTE(sampling_strategy=smote_r,random_state=42,k_neighbors=5)
                Xs,ys=sm.fit_resample(Xf,yf)
                sws=np.concatenate([swf,np.ones(len(Xs)-len(Xf))])
            except: Xs,ys,sws=Xf,yf,swf
            m=xgb.XGBClassifier(**params,use_label_encoder=False,eval_metric='logloss',
                                 random_state=42,n_jobs=-1,tree_method='hist',early_stopping_rounds=30)
            m.fit(Xs,ys,sample_weight=sws,eval_set=[(Xv,yv)],verbose=False)
            if yv.sum()>0 and yv.sum()<len(yv):
                fold_aucs.append(roc_auc_score(yv,m.predict_proba(Xv)[:,1]))
        return float(np.mean(fold_aucs)) if fold_aucs else 0.5

    log.info(f"\n▶  Optuna ({OPTUNA_TRIALS} trials) ...")
    study=optuna.create_study(direction='maximize')
    study.optimize(objective,n_trials=OPTUNA_TRIALS,show_progress_bar=False,n_jobs=1)
    bp=study.best_params; smote_best=bp.pop('smote_ratio')
    log.info(f"   Best val AUC: {study.best_value:.4f}")

    # Final train
    try:
        sm=BorderlineSMOTE(sampling_strategy=smote_best,random_state=42,k_neighbors=5)
        Xs,ys=sm.fit_resample(X_tr,y_tr)
        sws=np.concatenate([sw_,np.ones(len(Xs)-len(X_tr))])
    except:
        Xs,ys,sws=X_tr,y_tr,sw_
    model=xgb.XGBClassifier(**bp,use_label_encoder=False,eval_metric='logloss',
                             random_state=42,n_jobs=-1,tree_method='hist')
    model.fit(Xs,ys,sample_weight=sws,verbose=False)

    is_auc=roc_auc_score(y_tr,model.predict_proba(X_tr)[:,1])
    log.info(f"   IS AUC = {is_auc:.4f}")

    te_auc=0.0
    if len(df_te)>20:
        te_proba=model.predict_proba(X_te)[:,1]
        te_auc=roc_auc_score(y_te,te_proba)
        log.info(f"   OOS AUC = {te_auc:.4f}")
        for thr in [0.55,0.60,0.65,0.70]:
            mask=te_proba>=thr
            if mask.sum()>5:
                log.info(f"   threshold={thr}: {int(mask.sum())} setups, WR={float(y_te[mask].mean()):.1%}")

    try:
        imp=pd.Series(model.feature_importances_,index=feature_cols).sort_values(ascending=False)
        log.info(f"\nTop 15:\n{imp.head(15).to_string()}")
    except: pass

    old_auc=0.610
    if OUT.exists():
        try: old_auc=pickle.load(open(OUT,'rb')).get('oos_auc',0.0)
        except: pass

    if te_auc >= old_auc - 0.005:
        pkg={'model':model,'features':feature_cols,'is_auc':round(is_auc,4),
             'oos_auc':round(te_auc,4),'n_features':len(feature_cols),
             'train_years':TRAIN_YEARS,'test_years':TEST_YEARS,
             'version':'v2_enhanced_mtf_optuna_wf','label_tp_mult':TP_MULT,'label_window_min':LABEL_WINDOW}
        with open(OUT,'wb') as f: pickle.dump(pkg,f)
        log.info(f"\n💾 Salvat: {OUT}")
        log.info(f"   IS AUC={is_auc:.4f} | OOS AUC={te_auc:.4f} (was {old_auc:.4f})")
    else:
        log.warning(f"\n⚠️  OOS regresie ({te_auc:.4f} < {old_auc:.4f} - 0.005) — model vechi păstrat")

    df_tr.drop(columns=meta_cols,errors='ignore').to_pickle(Path(__file__).parent/"nom_dataset_train.pkl")
    df_te.drop(columns=meta_cols,errors='ignore').to_pickle(Path(__file__).parent/"nom_dataset_test.pkl")
    log.info("   Datasets salvate.")

if __name__ == "__main__":
    train_and_save()
