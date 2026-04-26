"""
train_nom_v3.py — NY Open Manipulation v3
==========================================
vs v2 (OOS=0.7273):
  ✅ regime_enc embedded as numeric feature (NOT split criterion)
  ✅ Lagged OF session features: same-day LON session CVD/absorption/opening_drive
     (LON OF predicts NY opening regime)
  ✅ Quantile regression sub-models: P10/P25/P50/P75/P90 of MFE
  ✅ Survival model: time_to_tp regression
  ✅ Removed use_label_encoder=False deprecation

PKL: nom_model_v3.pkl
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
log = logging.getLogger("NOM_V3")

DB            = Path(__file__).parent / "mario_trading.db"
OUT           = Path(__file__).parent / "nom_model_v3.pkl"
OPTUNA_TRIALS = 80

MIN_SPIKE_PT   = 5.0
MIN_DISP_PT    = 4.0
TP_MULT        = 2.0
LABEL_WINDOW   = 60

TRAIN_YEARS    = [2022, 2023, 2024]
TEST_YEARS     = [2025, 2026]
YEAR_WEIGHTS   = {2022: 0.75, 2023: 0.90, 2024: 1.00}
TOP_N_FEATURES = 75
N_WF_FOLDS     = 4

QUANTILE_ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]

NY_SESS_START_ET = 900
NY_SESS_END_ET   = 1300
PRE_NY_END_ET    = 859
LON_START_ET     = 400
LON_END_ET       = 630
ASIA_START_ET    = 0
ASIA_END_ET      = 359

_CAL_PATH = Path(__file__).parent / "data" / "economic_calendar.json"
try:
    _cal = json.loads(_CAL_PATH.read_text())
    FOMC_DATES   = set(_cal.get('fomc',   []))
    NFP_DATES    = set(_cal.get('nfp',    []))
    CPI_DATES    = set(_cal.get('cpi',    []))
    PPI_DATES    = set(_cal.get('ppi',    []))
    RETAIL_DATES = set(_cal.get('retail', []))
    ISM_DATES    = set(_cal.get('ism',    []))
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


# ── Helpers (same as LOM v3) ─────────────────────────────────────────────────
def load_regime_classifier():
    rc_path = Path(__file__).parent / "regime_classifier_v1.pkl"
    if not rc_path.exists():
        log.warning("  regime_classifier_v1.pkl not found"); return None
    try:
        import joblib
        pkg = joblib.load(rc_path); log.info("  Loaded regime_classifier_v1.pkl"); return pkg
    except Exception as e:
        log.warning(f"  Failed to load regime_classifier: {e}"); return None

def predict_regime_enc(regime_pkg, bar_features_dict):
    if regime_pkg is None: return 2
    try:
        model = regime_pkg['model']; feats = regime_pkg['features']; le = regime_pkg['label_encoder']
        x = pd.DataFrame([{f: bar_features_dict.get(f, 0.0) for f in feats}]).fillna(0)
        enc = model.predict(x)[0]
        return int(le.inverse_transform([enc])[0])
    except Exception: return 2

def load_of_lag_features():
    """
    For NY session on day D: use LON session OF from same day D.
    LON opens before NY → LON OF available when NY trades start.
    """
    of_path = Path(__file__).parent / "data" / "orderflow_features.parquet"
    if not of_path.exists(): return {}
    of = pd.read_parquet(of_path)
    of['date'] = of['date'].astype(str)
    lon = of[of['session_type'] == 'LON'].sort_values('date').reset_index(drop=True)
    def scol(df, c): return df[c].values if c in df.columns else np.zeros(len(df))
    lag_lookup = {}
    for _, row in lon.iterrows():
        d = row['date']
        lag_lookup[d] = {
            'of_cvd_lag1':    float(row.get('cvd_final', 0.0)),
            'of_abs_lag1':    float(row.get('absorption_score_mean', 0.0)),
            'of_od_lag1':     float(row.get('opening_drive_dir', 0.0)),
            'of_cvdz_lag1':   float(row.get('cvd_zscore_20d', 0.0)),
            'of_si_lag1':     float(row.get('stacked_imbalance_count', 0.0)),
            'of_or_lag1':     float(row.get('opening_range', 0.0)),
        }
    log.info(f"  NY OF lag lookup (LON same-day): {len(lag_lookup)} dates")
    return lag_lookup


# ── MTF ICT (identical to v2) ─────────────────────────────────────────────────
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
        active_bull=new_ab; new_ab2=[]
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
            d=min(abs(c-top),abs(c-bot))/a; dist_bull[i]=min(dist_bull[i],d)
        for top,bot,j in active_bear:
            if bot<=c<=top: in_bear[i]=1.0
            d=min(abs(c-top),abs(c-bot))/a; dist_bear[i]=min(dist_bear[i],d)
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
                if abs(c-H[i-1])/a<0.6 or abs(c-max(C[i-1],O[i-1]))/a<0.6: rejection[i]=1.0
            if wd>2.5*max(bz,0.5) and wd>a*0.3:
                if abs(c-min(C[i-1],O[i-1]))/a<0.6 or abs(c-L[i-1])/a<0.6: rejection[i]=1.0
    return pd.DataFrame({'in_bull':in_bull,'in_bear':in_bear,
        'dist_bull':np.clip(dist_bull,0,9.9),'dist_bear':np.clip(dist_bear,0,9.9),
        'in_ifvg_b':in_ifvg_b,'in_ifvg_s':in_ifvg_s,'breaker_b':breaker_b,'breaker_s':breaker_s,'rejection':rejection
    }, index=df_tf.index)


def compute_mtf_features(conn, setup_dates):
    min_d=min(setup_dates); max_d=max(setup_dates)
    warmup=(pd.Timestamp(min_d)-pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    df1m=pd.read_sql(f"""
        SELECT timestamp,open,high,low,close,atr_14 FROM market_data
        WHERE timestamp>='{warmup} 00:00:00' AND timestamp<='{max_d} 23:59:59'
        ORDER BY timestamp
    """, conn)
    df1m['ts']=pd.to_datetime(df1m['timestamp'])
    df1m=df1m.set_index('ts').rename(columns={'atr_14':'atr'})
    df1m['atr']=df1m['atr'].ffill().fillna(9.0)
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
    return all_features


# ── Data loading ──────────────────────────────────────────────────────────────
def load_day(conn, date_str):
    df = pd.read_sql(f"""
        SELECT timestamp, open, high, low, close, volume, atr_14, bar_delta, cum_delta,
               fvg_up, fvg_down, has_displacement, body_size, adx_14, hurst, dist_poc,
               inside_va, dist_vwap, delta_at_high, delta_at_low, big_buy_count, big_sell_count,
               absorption_score, stacked_bull, stacked_bear, of_doi, of_big_balance,
               bar_buy_vol, bar_sell_vol, garch_vol, kalman_smooth, fisher_transform,
               acf_lag1, acf_lag5, rvol, vah, val, poc_level, p_hi, p_lo, lw_hi, lw_lo,
               h4_hi, h4_lo, h1_hi, h1_lo, true_open, asia_hi, asia_lo,
               is_smt_bullish, is_smt_bearish, day_of_week, month
        FROM market_data WHERE date='{date_str}' ORDER BY timestamp
    """, conn)
    if len(df) < 15: return None
    df['ts']   = pd.to_datetime(df['timestamp'])
    df['hhmm'] = df['ts'].dt.hour * 100 + df['ts'].dt.minute
    return df


def _days_since_win(win_list, current_date_str):
    cd = pd.Timestamp(current_date_str)
    wins = [pd.Timestamp(d) for d, lbl in reversed(win_list) if lbl == 1]
    return float((cd - wins[0]).days) if wins else 30.0


def build_daily_context(conn, dates):
    min_d=min(dates); max_d=max(dates)
    warmup=(pd.Timestamp(min_d)-pd.Timedelta(days=40)).strftime('%Y-%m-%d')
    dr=pd.read_sql(f"""
        SELECT date(timestamp) as date,(MAX(high)-MIN(low)) as daily_range,
               AVG(atr_14) as avg_atr,AVG(adx_14) as avg_adx,AVG(hurst) as avg_hurst
        FROM market_data
        WHERE date(timestamp)>='{warmup}' AND date(timestamp)<='{max_d}'
        GROUP BY date(timestamp) ORDER BY date
    """, conn)
    dr['date']=dr['date'].astype(str)
    dr['avg_atr']=dr['avg_atr'].ffill().fillna(9.0)
    dr['daily_range']=dr['daily_range'].fillna(dr['avg_atr']*2)
    dr['range_atr_ratio']=dr['daily_range']/dr['avg_atr'].clip(lower=1)
    dr['vix_proxy_5d'] =dr['range_atr_ratio'].rolling(5,min_periods=2).mean().shift(1)
    dr['vix_proxy_20d']=dr['range_atr_ratio'].rolling(20,min_periods=5).mean().shift(1)
    dr['vol_regime']   =(dr['vix_proxy_5d']/dr['vix_proxy_20d'].clip(lower=0.5)).clip(upper=3)
    dr['adx_10d_mean'] =dr['avg_adx'].rolling(10,min_periods=3).mean().shift(1)
    dr['hurst_20d_mean']=dr['avg_hurst'].rolling(20,min_periods=5).mean().shift(1)
    dr['atr_5d']       =dr['avg_atr'].rolling(5,min_periods=2).mean().shift(1)
    dr['atr_10d']      =dr['avg_atr'].rolling(10,min_periods=3).mean().shift(1)
    dr['atr_trend']    =(dr['atr_5d']/dr['atr_10d'].clip(lower=1)).clip(upper=3)
    dr=dr.ffill().fillna(1.0)
    return {r['date']:r.to_dict() for _,r in dr.iterrows()}


# ── Extract setups — v3 additions ─────────────────────────────────────────────
def extract_setups(df, date_str, daily_ctx, cross_ctx=None,
                   regime_pkg=None, of_lag_lookup=None):
    setups = []
    pre_ny  = df[df['hhmm'] <= PRE_NY_END_ET]
    ny_sess = df[df['hhmm'].between(NY_SESS_START_ET, NY_SESS_END_ET)]
    london  = df[df['hhmm'].between(LON_START_ET, LON_END_ET)]
    asia_df = df[df['hhmm'].between(ASIA_START_ET, ASIA_END_ET)]

    if len(pre_ny) < 20 or len(ny_sess) < 5: return setups

    pre_hi = float(pre_ny['high'].max()); pre_lo = float(pre_ny['low'].min())
    pre_rng = pre_hi - pre_lo
    if pre_rng < 5: return setups

    atr = float(df['atr_14'].replace(0, np.nan).dropna().iloc[-1]) if len(df) > 0 else 10.0
    if atr <= 0: atr = 10.0

    if len(london) > 0:
        lon_hi=float(london['high'].max()); lon_lo=float(london['low'].min())
        lon_rng=lon_hi-lon_lo; lon_mid=(lon_hi+lon_lo)/2
        lon_close=float(london['close'].iloc[-1]); lon_open=float(london['open'].iloc[0])
    else:
        lon_hi=pre_hi; lon_lo=pre_lo; lon_rng=pre_rng; lon_mid=(pre_hi+pre_lo)/2
        lon_close=lon_mid; lon_open=lon_mid

    if len(asia_df) > 0:
        asia_hi_v=float(asia_df['high'].max()); asia_lo_v=float(asia_df['low'].min())
        asia_close=float(asia_df['close'].iloc[-1]); asia_open=float(asia_df['open'].iloc[0])
        asia_rng=asia_hi_v-asia_lo_v; asia_mid=(asia_hi_v+asia_lo_v)/2
        asia_dir=1 if asia_close>asia_mid else -1
    else:
        asia_hi_v=pre_hi; asia_lo_v=pre_lo; asia_rng=pre_rng; asia_mid=pre_hi
        asia_close=pre_lo; asia_open=pre_hi; asia_dir=0

    ny15=ny_sess[ny_sess['hhmm'].between(NY_SESS_START_ET, NY_SESS_START_ET+15)]
    if len(ny15) > 0:
        ny15_hi=float(ny15['high'].max()); ny15_lo=float(ny15['low'].min())
        ny15_open=float(ny15['open'].iloc[0]); ny15_close=float(ny15['close'].iloc[-1])
        ny15_rng=ny15_hi-ny15_lo; ny15_move=ny15_close-ny15_open
        ny_drive_bull=1 if ny15_move>atr*0.15 else 0
        ny_drive_bear=1 if ny15_move<-atr*0.15 else 0
        ny_drive_neutral=1 if abs(ny15_move)<=atr*0.15 else 0
        ny_drive_rng_atr=ny15_rng/atr
    else:
        ny_drive_bull=0; ny_drive_bear=0; ny_drive_neutral=1; ny_drive_rng_atr=0.5

    eq_tol=atr*0.3
    pre_highs=pre_ny['high'].values; pre_lows=pre_ny['low'].values
    eq_hi=max(0,sum(1 for h in pre_highs if abs(h-pre_hi)<=eq_tol)-1)
    eq_lo=max(0,sum(1 for l in pre_lows  if abs(l-pre_lo)<=eq_tol)-1)

    ny_open_price=float(ny_sess['open'].iloc[0]) if len(ny_sess)>0 else pre_hi
    partial_thresh=pre_rng*0.50
    lon_reset=ny_sess.reset_index(drop=False)
    last_setup_hhmm={'LONG':-999,'SHORT':-999}

    # Lagged OF (LON same day → NY)
    of_lag=(of_lag_lookup or {}).get(date_str,{})

    # Regime: use NY open bar
    dctx=daily_ctx.get(date_str,{})
    regime_bar_features={
        'adx_14': float(ny_sess['adx_14'].iloc[0]) if len(ny_sess)>0 else 20.0,
        'hurst':  float(ny_sess['hurst'].iloc[0])  if len(ny_sess)>0 else 0.5,
        'garch_vol': float(ny_sess['garch_vol'].iloc[0]) if len(ny_sess)>0 else 1.0,
        'inside_va': 0.0, 'dist_vwap': 5.0, 'has_displacement': 0,
        'rvol': float(ny_sess['rvol'].iloc[0]) if len(ny_sess)>0 else 1.0,
        'bar_delta': 0.0, 'cum_delta': 0.0, 'imbalance_pct': 0.0, 'dom_ratio': 1.0,
        'hhmm_enc': NY_SESS_START_ET, 'is_session_open': 1,
        'is_lon_session': 0, 'is_ny_session': 1,
        'sweep_dn_atr': 0.0, 'sweep_up_atr': 0.0, 'dist_poc': 5.0,
        'dist_pdh': 0.0, 'dist_pdl': 0.0, 'body_size': 0.0, 'fvg_up': 0, 'fvg_down': 0,
        'acf_lag1': float(ny_sess['acf_lag1'].iloc[0]) if len(ny_sess)>0 else 0.0,
        'acf_lag5': float(ny_sess['acf_lag5'].iloc[0]) if len(ny_sess)>0 else 0.0,
        'fisher_transform': float(ny_sess['fisher_transform'].iloc[0]) if len(ny_sess)>0 else 0.0,
        'sample_entropy': 0.5, 'day_of_week': float(df['day_of_week'].iloc[0]) if len(df)>0 else 0.0,
        'month': float(df['month'].iloc[0]) if len(df)>0 else 1.0,
        'of_cvd_lag1': of_lag.get('of_cvd_lag1',0.0), 'of_absorption_lag1': of_lag.get('of_abs_lag1',0.0),
        'of_opening_drive_lag1': of_lag.get('of_od_lag1',0.0), 'of_cvd_zscore_lag1': of_lag.get('of_cvdz_lag1',0.0),
        'of_stacked_imbalance_lag1': of_lag.get('of_si_lag1',0.0), 'of_opening_range_lag1': of_lag.get('of_or_lag1',0.0),
    }
    regime_enc=predict_regime_enc(regime_pkg, regime_bar_features)

    for i in range(1, len(lon_reset)-2):
        bar=lon_reset.iloc[i]; bar_hi=sv(bar['high']); bar_lo=sv(bar['low']); bar_hhmm=int(bar['hhmm'])
        sweep_up=bar_hi-pre_hi; sweep_dn=pre_lo-bar_lo

        for direction, spike_mag_raw, is_valid in [
            ('SHORT',max(sweep_up,0), sweep_up>=MIN_SPIKE_PT or (sweep_up>0 and sweep_up>=partial_thresh)),
            ('LONG', max(sweep_dn,0), sweep_dn>=MIN_SPIKE_PT or (sweep_dn>0 and sweep_dn>=partial_thresh)),
        ]:
            if not is_valid or (bar_hhmm-last_setup_hhmm[direction])<30: continue
            spike_mag=spike_mag_raw; spike_hi_val=bar_hi; spike_lo_val=bar_lo

            after_spike=lon_reset[lon_reset['hhmm'].between(bar_hhmm+1, bar_hhmm+45)]
            disp_bar=None
            for _,ab in after_spike.iterrows():
                ab_body=abs(sv(ab['close'])-sv(ab['open']))
                if direction=='SHORT' and sv(ab['close'])<sv(ab['open']) and ab_body>=MIN_DISP_PT:
                    disp_bar=ab; break
                elif direction=='LONG' and sv(ab['close'])>sv(ab['open']) and ab_body>=MIN_DISP_PT:
                    disp_bar=ab; break
            if disp_bar is None: continue

            entry_price=sv(disp_bar['close']); entry_hhmm=int(disp_bar['hhmm'])
            entry_ts=str(disp_bar['timestamp']); dir_num=1 if direction=='LONG' else -1

            future=df[df['hhmm']>entry_hhmm].head(LABEL_WINDOW)
            if len(future)<3: continue
            atr_tp=atr*TP_MULT
            if direction=='LONG':
                reached_tp=float(future['high'].max())>=entry_price+atr_tp
                max_fwd=float(future['high'].max()-entry_price)
                mae_raw=float(entry_price-future['low'].min())
            else:
                reached_tp=float(future['low'].min())<=entry_price-atr_tp
                max_fwd=float(entry_price-future['low'].min())
                mae_raw=float(future['high'].max()-entry_price)
            label=1 if reached_tp else 0
            max_fwd=max(max_fwd,0.0); mae_raw=max(mae_raw,0.0)

            # Survival labels
            tp_lvl=entry_price+atr_tp if direction=='LONG' else entry_price-atr_tp
            sl_pts=atr_tp*0.6
            sl_lvl=entry_price-sl_pts if direction=='LONG' else entry_price+sl_pts
            hit_tp=0; hit_sl=0; time_to_event=LABEL_WINDOW; is_censored=1
            for _,fb in future.iterrows():
                fh=sv(fb['high']); fl=sv(fb['low'])
                tp_h=(fh>=tp_lvl if direction=='LONG' else fl<=tp_lvl)
                sl_h=(fl<=sl_lvl if direction=='LONG' else fh>=sl_lvl)
                if tp_h or sl_h:
                    time_to_event=min(int(fb['hhmm'])-entry_hhmm, LABEL_WINDOW)
                    is_censored=0; hit_sl=1 if sl_h else 0; hit_tp=1 if tp_h and not sl_h else 0; break

            r0=df.iloc[-1]; spike_bar_range=max(sv(bar['high']-bar['low']),0.01)
            after_early=lon_reset[lon_reset['hhmm'].between(bar_hhmm+1, bar_hhmm+45)]

            if direction=='SHORT':
                ts_close_inside=1 if sv(bar['close'])<=pre_hi else 0
                wick=(sv(bar['high'])-max(sv(bar['close']),sv(bar['open'])))/atr
                ts_rejection_str=(spike_hi_val-sv(bar['close']))/spike_mag if spike_mag>0 else 0
                ts_wick_pct=(spike_hi_val-sv(bar['close']))/spike_bar_range
                ts_body_pct=abs(sv(bar['open'])-sv(bar['close']))/spike_bar_range
                ts_close_quality=max(0,(pre_hi-sv(bar['close']))/pre_rng) if pre_rng>0 else 0
            else:
                ts_close_inside=1 if sv(bar['close'])>=pre_lo else 0
                wick=(min(sv(bar['close']),sv(bar['open']))-sv(bar['low']))/atr
                ts_rejection_str=(sv(bar['close'])-spike_lo_val)/spike_mag if spike_mag>0 else 0
                ts_wick_pct=(sv(bar['close'])-spike_lo_val)/spike_bar_range
                ts_body_pct=abs(sv(bar['open'])-sv(bar['close']))/spike_bar_range
                ts_close_quality=max(0,(sv(bar['close'])-pre_lo)/pre_rng) if pre_rng>0 else 0

            wick_pct=wick*atr/spike_bar_range; sweep_wick_clean=1 if wick_pct>0.5 else 0
            sweep_depth_atr=spike_mag/atr; deep_sweep=1 if sweep_depth_atr>1.5 else 0
            sweep_quality=ts_close_inside*0.4+sweep_wick_clean*0.3+deep_sweep*0.2+0.1
            disp_body=abs(sv(disp_bar['close'])-sv(disp_bar['open']))
            h4_hi=sv(r0['h4_hi']); h4_lo=sv(r0['h4_lo'])
            h4_mid=(h4_hi+h4_lo)/2 if h4_hi>0 and h4_lo>0 else 0
            h4_bias=1 if entry_price<h4_mid else (-1 if h4_mid>0 else 0)
            h1_hi=sv(r0['h1_hi']); h1_lo=sv(r0['h1_lo'])
            h1_mid=(h1_hi+h1_lo)/2 if h1_hi>0 and h1_lo>0 else 0
            h1_bias=1 if entry_price<h1_mid else (-1 if h1_mid>0 else 0)
            lw_hi=sv(r0['lw_hi']); lw_lo=sv(r0['lw_lo']); lw_rng=lw_hi-lw_lo
            weekly_prem=(entry_price-lw_lo)/lw_rng if lw_rng>0 else 0.5
            pre_vol=float(pre_ny['volume'].sum()) if len(pre_ny)>0 else 1.0
            ny_vol=float(ny_sess['volume'].sum()) if len(ny_sess)>0 else 1.0
            vol_ratio=ny_vol/pre_vol if pre_vol>0 else 1.0
            spike_delta=sv(ny_sess['bar_delta'].sum()) if len(ny_sess)>0 else 0
            fvg_up_v=int(ny_sess['fvg_up'].any()); fvg_down_v=int(ny_sess['fvg_down'].any())
            adx_v=sv(r0['adx_14']); hurst_v=sv(r0['hurst'],0.5)
            lon_dir=1 if lon_close>lon_mid else -1
            triple_aligned=1 if (asia_dir==dir_num and lon_dir!=dir_num) else 0
            vix5=dctx.get('vix_proxy_5d',2.0); vix20=dctx.get('vix_proxy_20d',2.0)
            vol_rg=dctx.get('vol_regime',1.0); atr_tr=dctx.get('atr_trend',1.0)
            adx10=dctx.get('adx_10d_mean',20.0); hst20=dctx.get('hurst_20d_mean',0.5)
            atr5d=dctx.get('atr_5d',atr); roll_wr=dctx.get('rolling_wr',0.5)
            prev_ny_dir=dctx.get('prev_ny_dir',0)
            cctx=cross_ctx or {}
            dsw_dir=cctx.get('dsw_L' if direction=='LONG' else 'dsw_S',30.0)
            dsw_any=min(cctx.get('dsw_L',30.0),cctx.get('dsw_S',30.0))
            wk_cnt=float(cctx.get('week_cnt',0)); td_cnt=float(cctx.get('td_L' if direction=='LONG' else 'td_S',0))

            feat = {
                # ── Core sweep/TS features ──────────────────────────────────
                'spike_mag':spike_mag,'spike_mag_atr':spike_mag/atr,
                'spike_vs_range':spike_mag/pre_rng if pre_rng>0 else 0,
                'pre_rng_atr':pre_rng/atr,
                'ts_close_inside':ts_close_inside,'ts_rejection_str':ts_rejection_str,
                'ts_wick_pct':ts_wick_pct,'ts_body_pct':ts_body_pct,'ts_close_quality':ts_close_quality,
                'ts_wick_dom':1 if ts_wick_pct>0.6 else 0,'ts_htf_anti':1 if h4_bias==dir_num else 0,
                'ts_combo_score':ts_close_inside*ts_rejection_str,
                'ts_sweep_depth_pts':spike_mag,'ts_sweep_depth_atr':sweep_depth_atr,
                'ts_sweep_pct_lon':spike_mag/lon_rng if lon_rng>0 else 0,
                'ts_lon_mid_dist':(entry_price-lon_mid)/atr,
                # ── Sweep quality ──────────────────────────────────────────
                'sweep_wick_atr':wick,'sweep_wick_pct':wick_pct,'sweep_wick_clean':sweep_wick_clean,
                'sweep_depth_atr':sweep_depth_atr,'deep_sweep':deep_sweep,
                'shallow_sweep':1 if sweep_depth_atr<0.5 else 0,
                'sweep_with_disp':1,'sweep_quality_score':sweep_quality,
                'equal_level_score':(eq_hi if direction=='SHORT' else eq_lo)/max(len(pre_ny),1),
                'equal_hi_count':float(eq_hi),'equal_lo_count':float(eq_lo),
                # ── Displacement ──────────────────────────────────────────
                'disp_body':disp_body,'disp_body_atr':disp_body/atr,
                'disp_range':sv(disp_bar['high']-disp_bar['low']),
                'has_disp':1,'body_pct':disp_body/max(sv(disp_bar['high']-disp_bar['low']),0.01),
                'body_bear':1 if direction=='SHORT' else 0,
                'disp_range_atr':sv(disp_bar['high']-disp_bar['low'])/atr,
                # ── HTF ───────────────────────────────────────────────────
                'h4_bias':h4_bias,'h1_bias':h1_bias,
                'h4_h1_aligned':1 if h4_bias==h1_bias and h4_bias!=0 else 0,
                'h4_bias_aligned':1 if h4_bias==dir_num else 0,
                # ── Weekly ────────────────────────────────────────────────
                'weekly_premium_pct':weekly_prem,
                'in_weekly_premium':1 if weekly_prem>0.5 else 0,
                'in_weekly_discount':1 if weekly_prem<0.5 else 0,
                'weekly_prem_aligned':1 if (direction=='SHORT' and weekly_prem>0.5) or (direction=='LONG' and weekly_prem<0.5) else 0,
                'lw_range_atr':lw_rng/atr if atr>0 else 0,
                'dist_lw_hi':abs(entry_price-lw_hi)/atr,'dist_lw_lo':abs(entry_price-lw_lo)/atr,
                # ── LON context ───────────────────────────────────────────
                'lon_range_atr':lon_rng/atr,'dist_lon_hi_atr':abs(entry_price-lon_hi)/atr,
                'dist_lon_lo_atr':abs(entry_price-lon_lo)/atr,
                'lon_close_vs_mid':(lon_close-lon_mid)/lon_rng if lon_rng>0 else 0,
                'gap_vs_lon_close_atr':(ny_open_price-lon_close)/atr,
                'ny15_range_atr':ny_drive_rng_atr,'lon_dir_explicit':float(lon_dir),
                'lon_dir_aligned':1 if lon_dir==dir_num else 0,
                'lon_big_day':1 if lon_rng>atr*1.5 else 0,'lon_small_day':1 if lon_rng<atr*0.7 else 0,
                'ny_open_in_lon':1 if lon_lo<=ny_open_price<=lon_hi else 0,
                'ny_open_above_lon_mid':1 if ny_open_price>lon_mid else 0,
                # ── NY open drive ─────────────────────────────────────────
                'ny_open_drive_bull':float(ny_drive_bull),'ny_open_drive_bear':float(ny_drive_bear),
                'ny_open_drive_neutral':float(ny_drive_neutral),
                'drive_aligned_dir':1.0 if (ny_drive_bull and dir_num==1) or (ny_drive_bear and dir_num==-1) else 0.0,
                # ── Asia ─────────────────────────────────────────────────
                'dist_asia_hi_atr':abs(entry_price-asia_hi_v)/atr if asia_hi_v>0 else 0,
                'dist_asia_lo_atr':abs(entry_price-asia_lo_v)/atr if asia_lo_v>0 else 0,
                'asia_range_atr':asia_rng/atr if asia_rng>0 else 0,
                'asia_dir_explicit':float(asia_dir),'asia_dir_aligned':1 if asia_dir==dir_num else 0,
                'asia_range_vs_atr5d':float(np.clip(asia_rng/max(atr5d,1.0),0,10)),
                # ── Triple session alignment ──────────────────────────────
                'triple_sess_aligned':float(triple_aligned),
                'lon_asia_aligned':1 if lon_dir==asia_dir else 0,
                'full_alignment':1 if lon_dir==asia_dir==dir_num else 0,
                # ── SMT ───────────────────────────────────────────────────
                'is_smt_bullish':sv(r0.get('is_smt_bullish',0)),'is_smt_bearish':sv(r0.get('is_smt_bearish',0)),
                'smt_aligned':sv(r0.get('is_smt_bullish',0)) if dir_num==1 else sv(r0.get('is_smt_bearish',0)),
                # ── PDH/PDL ───────────────────────────────────────────────
                'above_true_open':1 if entry_price>sv(r0['true_open']) else 0,
                'dist_pdh_atr':abs(entry_price-sv(r0['p_hi']))/atr,
                'dist_pdl_atr':abs(entry_price-sv(r0['p_lo']))/atr,
                # ── VA/POC ────────────────────────────────────────────────
                'inside_va':sv(r0['inside_va']),'dist_poc_atr':sv(r0['dist_poc'])/atr,
                'dist_vwap_atr':sv(r0['dist_vwap'])/atr,'entry_in_pre_range':int(pre_lo<=entry_price<=pre_hi),
                # ── Volume/delta ──────────────────────────────────────────
                'vol_ratio':vol_ratio,'spike_delta':spike_delta,
                'disp_delta':sv(after_early['bar_delta'].sum()) if len(after_early)>0 else 0,
                'delta_at_high':sv(ny_sess['delta_at_high'].sum()) if 'delta_at_high' in ny_sess.columns else 0,
                'delta_at_low':sv(ny_sess['delta_at_low'].sum()) if 'delta_at_low' in ny_sess.columns else 0,
                'absorption':sv(ny_sess['absorption_score'].mean()) if 'absorption_score' in ny_sess.columns else 0,
                'buy_sell_ratio':sv(ny_sess['bar_buy_vol'].sum())/max(sv(ny_sess['bar_sell_vol'].sum()),1),
                'fvg_up':fvg_up_v,'fvg_down':fvg_down_v,
                'htf_fvg_aligned':1 if (direction=='SHORT' and fvg_down_v) or (direction=='LONG' and fvg_up_v) else 0,
                # ── Technical ─────────────────────────────────────────────
                'adx':adx_v,'adx_strong':1 if adx_v>25 else 0,'hurst':hurst_v,
                'fisher_transform':sv(r0['fisher_transform']),'acf_lag1':sv(r0['acf_lag1']),'acf_lag5':sv(r0['acf_lag5']),
                'garch_vol':sv(r0['garch_vol']),'rvol':sv(r0['rvol'],1.0),
                'kalman_smooth':sv(r0.get('kalman_smooth',0.0)),'kalman_noise':sv(r0.get('kalman_noise',0.0)),
                # ── Rolling regime ────────────────────────────────────────
                'vix_proxy_5d':float(vix5),'vix_proxy_20d':float(vix20),'vol_regime':float(vol_rg),
                'vol_high':1 if vol_rg>1.2 else 0,'atr_trend':float(atr_tr),
                'adx_10d_mean':float(adx10),'hurst_20d_mean':float(hst20),
                'atr_vs_5d':float(np.clip(atr/max(atr5d,1.0),0,3)),
                'rolling_5sess_wr':float(roll_wr),'atr_tp_norm':float(np.clip(atr/20.0,0.5,3.0)),
                # ── Calendar ─────────────────────────────────────────────
                'is_nfp_day':1 if date_str in NFP_DATES else 0,
                'is_fomc_day':1 if date_str in FOMC_DATES else 0,
                'is_cpi_day':1 if date_str in CPI_DATES else 0,
                'is_news_day':1 if date_str in NEWS_DAYS else 0,
                'fomc_proximity':float(np.clip(_fomc_prox(date_str)/14.0,0,1)),
                'is_pre_nfp':1 if (date_str in NFP_DATES and entry_hhmm<830) else 0,
                'is_post_nfp':1 if (date_str in NFP_DATES and entry_hhmm>=830) else 0,
                'sweep_time_early':1 if bar_hhmm<=NY_SESS_START_ET+30 else 0,
                'sweep_time_mid':1 if NY_SESS_START_ET+30<bar_hhmm<=NY_SESS_START_ET+120 else 0,
                'sweep_time_late':1 if bar_hhmm>NY_SESS_START_ET+120 else 0,
                # ── Time ─────────────────────────────────────────────────
                'day_of_week':sv(r0['day_of_week']),
                'is_monday':1 if int(sv(r0['day_of_week']))==0 else 0,
                'month':sv(r0['month']),
                # ── Prev NY dir ───────────────────────────────────────────
                'prev_ny_dir':float(prev_ny_dir),'prev_ny_aligned':1 if prev_ny_dir==dir_num else 0,
                # ── Cross-setup ───────────────────────────────────────────
                'days_since_win_dir':float(np.clip(dsw_dir,0,30)),
                'days_since_win_any':float(np.clip(dsw_any,0,30)),
                'week_setup_count':float(np.clip(wk_cnt,0,10)),
                'hot_streak':1 if dsw_dir<=2 else 0,'cold_streak':1 if dsw_dir>=7 else 0,
                # ── Interactions ──────────────────────────────────────────
                'dir_x_adx':dir_num*adx_v,'dir_x_hurst':dir_num*hurst_v,
                'sweep_x_h4':sweep_quality*(1 if h4_bias==dir_num else 0),
                'vol_x_sweep':vol_rg*sweep_quality,
                'drive_x_sweep':(1.0 if (ny_drive_bull and dir_num==1) or (ny_drive_bear and dir_num==-1) else 0.0)*sweep_quality,
                'triple_x_h4':float(triple_aligned)*(1 if h4_bias==dir_num else 0),
                # ── NUEVO: Regime enc ─────────────────────────────────────
                'regime_enc':float(regime_enc),
                'regime_is_pre':1 if regime_enc==1 else 0,
                'regime_is_exp':1 if regime_enc==2 else 0,
                'regime_is_ret':1 if regime_enc==3 else 0,
                'regime_aligned':1 if (regime_enc in [1,2] and h4_bias==dir_num) else 0,
                # ── NUEVO: Lagged OF features (LON → NY) ──────────────────
                'of_cvd_lag1':of_lag.get('of_cvd_lag1',0.0),
                'of_abs_lag1':of_lag.get('of_abs_lag1',0.0),
                'of_od_lag1':of_lag.get('of_od_lag1',0.0),
                'of_cvdz_lag1':of_lag.get('of_cvdz_lag1',0.0),
                'of_si_lag1':of_lag.get('of_si_lag1',0.0),
                'of_or_lag1':of_lag.get('of_or_lag1',0.0),
                'of_lon_cvd_x_dir':of_lag.get('of_cvd_lag1',0.0)*float(dir_num),
                'of_regime_x_sweep':of_lag.get('of_cvd_lag1',0.0)*float(regime_enc),
                # ── Cross-model features (LOM→NOM pollination) ───────────
                'asia_dir_x_h4':float(asia_dir)*float(h4_bias),
                'disp_wick_ratio':(sv(disp_bar['high']-disp_bar['low'])-disp_body)/max(disp_body,0.01),
                'spike_vs_asia_hi':(spike_hi_val-asia_hi_v)/atr if asia_hi_v>0 else 0.0,
                'spike_vs_asia_lo':(asia_lo_v-spike_lo_val)/atr if asia_lo_v>0 else 0.0,
                'h4_x_weekly':float(1 if h4_bias==dir_num else 0)*float(1 if (direction=='SHORT' and weekly_prem>0.5) or (direction=='LONG' and weekly_prem<0.5) else 0),
                'is_thursday':1 if int(sv(r0['day_of_week']))==3 else 0,
                'is_friday':1 if int(sv(r0['day_of_week']))==4 else 0,
                'pre_rng_vs_lw':pre_rng/max(lw_rng,0.01),
                'sweep_vs_lw_rng':spike_mag/max(lw_rng,0.01),
                # ── Direction ─────────────────────────────────────────────
                'direction_enc':1 if direction=='SHORT' else 0,
                # ── Meta ─────────────────────────────────────────────────
                '_label':label,'_direction':direction,'_date':str(date_str),
                '_entry_px':entry_price,'_max_fwd':max_fwd,'_mae_raw':mae_raw,
                '_entry_hhmm':entry_hhmm,'_entry_ts':entry_ts,
                '_hit_tp':hit_tp,'_hit_sl':hit_sl,'_time_to_event':time_to_event,'_is_censored':is_censored,
            }
            setups.append(feat)
            last_setup_hhmm[direction]=bar_hhmm
            break
    return setups


def build_dataset(years, regime_pkg=None, of_lag_lookup=None):
    conn=sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    days=pd.read_sql(f"""
        SELECT DISTINCT date FROM market_data
        WHERE year IN ({','.join(map(str,years))}) AND day_of_week BETWEEN 1 AND 5
        ORDER BY date
    """, conn)['date'].tolist()

    daily_ctx=build_daily_context(conn, days)
    all_setups=[]; wr_window=[]; prev_ny_dir=0
    win_hist={'LONG':[],'SHORT':[]}; week_counts={}

    for date_str in days:
        df=load_day(conn, date_str)
        if df is None: continue
        roll_wr=float(np.mean(wr_window[-5:])) if wr_window else 0.5
        if date_str in daily_ctx:
            daily_ctx[date_str]['rolling_wr']=roll_wr
            daily_ctx[date_str]['prev_ny_dir']=prev_ny_dir
        wk=pd.Timestamp(date_str).isocalendar(); week_str=f"{wk.year}_{wk.week}"
        cross_ctx={'dsw_L':_days_since_win(win_hist['LONG'],date_str),
                   'dsw_S':_days_since_win(win_hist['SHORT'],date_str),
                   'week_cnt':float(week_counts.get(week_str,0)),'td_L':0.0,'td_S':0.0}
        setups=extract_setups(df, date_str, daily_ctx, cross_ctx,
                              regime_pkg=regime_pkg, of_lag_lookup=of_lag_lookup)
        all_setups.extend(setups)
        ny_bars=df[df['hhmm'].between(NY_SESS_START_ET, NY_SESS_END_ET)]
        if len(ny_bars)>=3:
            ncl=float(ny_bars['close'].iloc[-1]); nmid=(ny_bars['high'].max()+ny_bars['low'].min())/2
            prev_ny_dir=1 if ncl>nmid else -1
        for s in setups:
            d=s['_direction']; win_hist[d].append((date_str,s['_label']))
            week_counts[week_str]=week_counts.get(week_str,0)+1; wr_window.append(s['_label'])

    conn.close()
    log.info(f"  {years}: {len(days)} days → {len(all_setups)} setups")
    if not all_setups: return pd.DataFrame()
    df_out=pd.DataFrame(all_setups)

    # MTF ICT
    log.info("   Joining MTF ICT ...")
    conn2=sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    mtf=compute_mtf_features(conn2, sorted(df_out['_date'].unique()))
    conn2.close()
    df_out=df_out.merge(
        mtf.drop_duplicates('ts_str')[['ts_str']+[c for c in mtf.columns if c!='ts_str']],
        left_on='_entry_ts', right_on='ts_str', how='left')
    for c in [c for c in mtf.columns if c!='ts_str']:
        df_out[c]=df_out[c].fillna(0.0)
    for tf in ['5m','15m','1h','4h']:
        dir_n=np.where(df_out['direction_enc'].values==0,1.0,-1.0)
        df_out[f'fvg_aligned_{tf}']=np.where(dir_n==1,df_out[f'in_bull_{tf}'],df_out[f'in_bear_{tf}'])
        df_out[f'ifvg_aligned_{tf}']=np.where(dir_n==1,df_out[f'in_ifvg_s_{tf}'],df_out[f'in_ifvg_b_{tf}'])
        df_out[f'breaker_aligned_{tf}']=np.where(dir_n==1,df_out[f'breaker_b_{tf}'],df_out[f'breaker_s_{tf}'])
    df_out['fvg_tf_confluence']=sum(df_out.get(f'fvg_aligned_{tf}',pd.Series(0,index=df_out.index)).values for tf in ['5m','15m','1h','4h'])
    df_out['htf_fvg_aligned_mtf']=np.maximum(
        df_out.get('fvg_aligned_1h',pd.Series(0,index=df_out.index)).values,
        df_out.get('fvg_aligned_4h',pd.Series(0,index=df_out.index)).values)

    # Synthetic OF
    _OF=Path(__file__).parent/"data"/"orderflow_features.parquet"
    if _OF.exists():
        _of=pd.read_parquet(_OF); _of=_of[_of['session_type']=='NY'].copy()
        _of['date']=_of['date'].astype(str)
        _OF_COLS=[c for c in _of.columns if c not in ['session_id','date','session_type',
                  'session_open','session_close','session_high','session_low','total_vol']]
        df_out=df_out.merge(_of[['date']+_OF_COLS].rename(columns={'date':'_date'}),on='_date',how='left')
        for _c in _OF_COLS: df_out[_c]=df_out[_c].fillna(0.0)
        log.info(f"   OF features: {len(_OF_COLS)} merged (NY)")

    # ── Correct regime_enc from precomputed regime_labels.parquet (NY session) ─
    # predict_regime_enc() always returned 2 (sweep_dn/up hardcoded=0). Fix: join.
    _RL = Path(__file__).parent / "data" / "regime_labels.parquet"
    if _RL.exists():
        try:
            rl = pd.read_parquet(_RL)
            rl_ny = rl[rl['session'] == 'NY'][['date', 'regime_enc', 'regime']].copy()
            rl_ny['date'] = pd.to_datetime(rl_ny['date']).dt.strftime('%Y-%m-%d')
            df_out = df_out.merge(rl_ny.rename(columns={
                'date': '_date', 'regime_enc': '_rl_enc', 'regime': '_rl_regime'
            }), on='_date', how='left')
            df_out['regime_enc']    = df_out['_rl_enc'].fillna(2).astype(int)
            re = df_out['regime_enc'].values
            df_out['regime_is_pre'] = (re == 1).astype(float)
            df_out['regime_is_exp'] = (re == 2).astype(float)
            df_out['regime_is_ret'] = (re == 3).astype(float)
            df_out['regime_aligned'] = np.where(
                np.isin(re, [1, 2]),
                (df_out.get('h4_bias_aligned', pd.Series(0, index=df_out.index)).values == 1).astype(float),
                0.0)
            if 'of_cvd_lag1' in df_out.columns:
                df_out['of_cvd_regime'] = df_out['of_cvd_lag1'].fillna(0) * re.astype(float)
            df_out.drop(columns=['_rl_enc', '_rl_regime'], errors='ignore', inplace=True)
            dist = dict(pd.Series(re).value_counts().sort_index())
            log.info(f"   regime_enc from precomputed labels (NY): {dist}")
        except Exception as e:
            log.warning(f"   regime_labels join failed: {e}")
    else:
        log.warning("   regime_labels.parquet not found — regime_enc may be inaccurate")

    log.info(f"   Total columns: {df_out.shape[1]}")
    return df_out


# ── Training (same structure as LOM v3) ───────────────────────────────────────
def train_main_model(X_tr, y_tr, X_te, y_te, sw_, feature_cols):
    ts_tr=pd.DatetimeIndex(pd.to_datetime(X_tr.index.map(lambda i: X_tr.index[i] if False else '2023-01-01')))
    val_cut=int(len(X_tr)*0.75)
    wf_folds=[(np.array([True]*val_cut+[False]*(len(X_tr)-val_cut)),
               np.array([False]*val_cut+[True]*(len(X_tr)-val_cut)))]

    def objective(trial):
        p={'n_estimators':trial.suggest_int('n_estimators',150,800),
           'max_depth':trial.suggest_int('max_depth',2,4),
           'learning_rate':trial.suggest_float('learning_rate',0.005,0.05,log=True),
           'subsample':trial.suggest_float('subsample',0.5,0.85),
           'colsample_bytree':trial.suggest_float('colsample_bytree',0.35,0.75),
           'min_child_weight':trial.suggest_int('min_child_weight',15,60),
           'gamma':trial.suggest_float('gamma',0.5,6.0),
           'reg_alpha':trial.suggest_float('reg_alpha',0.5,8.0),
           'reg_lambda':trial.suggest_float('reg_lambda',2.0,10.0),
           'scale_pos_weight':trial.suggest_float('scale_pos_weight',2.0,10.0)}
        smote_r=trial.suggest_float('smote',0.10,0.40)
        aucs=[]
        for tm,vm in wf_folds:
            Xf=X_tr[tm]; yf=y_tr.values[tm]; swf=sw_[tm]
            Xv=X_tr[vm]; yv=y_tr.values[vm]
            try:
                sm=BorderlineSMOTE(sampling_strategy=smote_r,random_state=42,k_neighbors=5)
                Xs,ys=sm.fit_resample(Xf,yf); sws=np.concatenate([swf,np.ones(len(Xs)-len(Xf))])
            except Exception: Xs,ys,sws=Xf,yf,swf
            m=xgb.XGBClassifier(**p,eval_metric='logloss',random_state=42,n_jobs=-1,
                                 tree_method='hist',early_stopping_rounds=30)
            m.fit(Xs,ys,sample_weight=sws,eval_set=[(Xv,yv)],verbose=False)
            if yv.sum()>0 and yv.sum()<len(yv): aucs.append(roc_auc_score(yv,m.predict_proba(Xv)[:,1]))
        return float(np.mean(aucs)) if aucs else 0.5

    log.info(f"▶  Optuna ({OPTUNA_TRIALS} trials) ...")
    study=optuna.create_study(direction='maximize')
    study.optimize(objective,n_trials=OPTUNA_TRIALS,show_progress_bar=False,n_jobs=1)
    bp=study.best_params; smote_r=bp.pop('smote')
    log.info(f"   Best val AUC: {study.best_value:.4f}")
    try:
        sm=BorderlineSMOTE(sampling_strategy=smote_r,random_state=42,k_neighbors=5)
        X_sm,y_sm=sm.fit_resample(X_tr,y_tr); sw_sm=np.concatenate([sw_,np.ones(len(X_sm)-len(X_tr))])
    except Exception: X_sm,y_sm=X_tr,y_tr; sw_sm=sw_
    model=xgb.XGBClassifier(**bp,eval_metric='logloss',random_state=42,n_jobs=-1,tree_method='hist')
    model.fit(X_sm,y_sm,sample_weight=sw_sm,verbose=False)
    is_auc=roc_auc_score(y_tr,model.predict_proba(X_tr)[:,1])
    oos_auc=0.0
    if len(X_te)>20: oos_auc=roc_auc_score(y_te,model.predict_proba(X_te)[:,1])
    log.info(f"   IS={is_auc:.4f}  OOS={oos_auc:.4f}")
    return model, is_auc, oos_auc


def train_quantile_models(X_tr, X_te, df_tr, df_te, feature_cols):
    if '_max_fwd' not in df_tr.columns:
        log.warning("  _max_fwd not found — skipping quantile models"); return {}
    y_tr_mfe=df_tr['_max_fwd'].values.astype(float)
    y_te_mfe=df_te['_max_fwd'].values.astype(float) if len(df_te)>0 else None
    log.info(f"▶  Quantile models (MFE mean={y_tr_mfe.mean():.1f}pt)")
    q_models={}
    for alpha in QUANTILE_ALPHAS:
        qm=xgb.XGBRegressor(objective='reg:quantileerror',quantile_alpha=alpha,
                             n_estimators=300,max_depth=3,learning_rate=0.05,
                             subsample=0.8,colsample_bytree=0.6,min_child_weight=15,
                             reg_alpha=1.0,reg_lambda=3.0,random_state=42,n_jobs=-1,tree_method='hist')
        qm.fit(X_tr,y_tr_mfe,verbose=False)
        q_models[alpha]=qm
        if y_te_mfe is not None and len(X_te)>0:
            preds=qm.predict(X_te); diff=y_te_mfe-preds
            pinball=np.mean(np.where(diff>=0,alpha*diff,(alpha-1)*diff))
            log.info(f"   Q{int(alpha*100)} pinball={pinball:.3f}")
    return q_models


def train_survival_model(X_tr, X_te, df_tr, df_te, feature_cols):
    if '_time_to_event' not in df_tr.columns:
        log.warning("  _time_to_event not found — skipping survival model"); return None
    y_tr_surv=np.log1p(df_tr['_time_to_event'].values.astype(float))
    sm=xgb.XGBRegressor(objective='reg:squarederror',n_estimators=300,max_depth=3,
                         learning_rate=0.05,subsample=0.8,colsample_bytree=0.6,min_child_weight=15,
                         reg_alpha=1.0,reg_lambda=3.0,random_state=42,n_jobs=-1,tree_method='hist')
    sm.fit(X_tr,y_tr_surv,verbose=False)
    if len(df_te)>0 and '_time_to_event' in df_te.columns:
        y_te_surv=np.log1p(df_te['_time_to_event'].values.astype(float))
        mae=np.mean(np.abs(np.expm1(sm.predict(X_te))-np.expm1(y_te_surv)))
        log.info(f"   Survival OOS MAE={mae:.1f} bars")
    return sm


def train_and_save():
    log.info("═"*60); log.info("NOM TRAIN v3 — regime_enc + quantile + survival"); log.info("═"*60)
    regime_pkg=load_regime_classifier(); of_lag_lookup=load_of_lag_features()
    log.info(f"Extrag IS ({TRAIN_YEARS}) ...")
    df_tr=build_dataset(TRAIN_YEARS, regime_pkg=regime_pkg, of_lag_lookup=of_lag_lookup)
    log.info(f"Extrag OOS ({TEST_YEARS}) ...")
    df_te=build_dataset(TEST_YEARS, regime_pkg=regime_pkg, of_lag_lookup=of_lag_lookup)

    meta_cols=[c for c in df_tr.columns if c.startswith('_') or c=='ts_str']
    feature_cols=[c for c in df_tr.columns if c not in meta_cols]
    log.info(f"\nIS: {len(df_tr)} setups | features: {len(feature_cols)}")
    log.info(f"OOS: {len(df_te)} setups")
    log.info(f"Label IS: {df_tr['_label'].value_counts().to_dict()}")
    log.info(f"Regime IS: {df_tr['regime_enc'].value_counts().to_dict()}")
    if len(df_tr)<50: log.error("Prea puțin data IS"); return

    X_tr=df_tr[feature_cols].fillna(0); y_tr=df_tr['_label']
    yr_=df_tr['_date'].apply(lambda d:int(d[:4]))
    sw_=np.array([YEAR_WEIGHTS.get(yr,1.0) for yr in yr_])
    X_te=df_te[feature_cols].fillna(0).reindex(columns=feature_cols,fill_value=0) if len(df_te)>0 else pd.DataFrame(columns=feature_cols)
    y_te=df_te['_label'] if len(df_te)>0 else pd.Series(dtype=int)

    # Feature selection
    log.info(f"▶  Feature selection (top {TOP_N_FEATURES}) ...")
    neg,pos=(y_tr==0).sum(),(y_tr==1).sum(); _spw=neg/max(pos,1)
    _pre=xgb.XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.7,
                            colsample_bytree=0.6,min_child_weight=25,gamma=1.5,reg_alpha=2.0,
                            reg_lambda=4.0,scale_pos_weight=_spw,random_state=42,n_jobs=-1,
                            eval_metric='logloss',verbosity=0)
    _pre.fit(X_tr,y_tr,sample_weight=sw_,verbose=False)
    _imp=pd.Series(_pre.feature_importances_,index=feature_cols).sort_values(ascending=False)
    must_keep=[c for c in [
        # Regime
        'regime_enc','regime_is_pre','regime_is_exp','regime_is_ret','regime_aligned',
        # Lagged OF
        'of_cvd_lag1','of_abs_lag1','of_od_lag1','of_lon_cvd_x_dir','of_regime_x_sweep',
        # High-importance from NOM v1
        'equal_lo_count','rolling_5sess_wr','fisher_transform','fomc_proximity',
        'drive_x_sweep','ts_sweep_pct_lon','adx_10d_mean','disp_body',
        'kalman_smooth',
        # Cross-model from LOM
        'garch_vol','atr_vs_5d','dir_x_hurst','vol_x_sweep',
        'asia_dir_x_h4','h4_x_weekly','spike_vs_asia_hi','spike_vs_asia_lo',
        'dist_lw_hi','dist_lw_lo','pre_rng_vs_lw','sweep_vs_lw_rng',
        'is_thursday','is_friday',
        # Core structural
        'pre_rng_atr','asia_dir_explicit','asia_range_vs_atr5d',
        'dist_asia_hi_atr','dist_asia_lo_atr','sweep_quality_score',
    ] if c in feature_cols]
    selected=list(dict.fromkeys(must_keep+_imp.head(TOP_N_FEATURES).index.tolist()))
    log.info(f"   Selectate {len(selected)} | top5: {_imp.head(5).index.tolist()}")
    X_tr=X_tr[selected]; X_te=X_te.reindex(columns=selected,fill_value=0); feature_cols=selected

    log.info("\n── Binary classifier ──")
    main_model,is_auc,oos_auc=train_main_model(X_tr,y_tr,X_te,y_te,sw_,feature_cols)
    log.info("\n── Quantile regression ──")
    q_models=train_quantile_models(X_tr,X_te,df_tr,df_te,feature_cols)
    log.info("\n── Survival model ──")
    surv_model=train_survival_model(X_tr,X_te,df_tr,df_te,feature_cols)

    old_auc=0.0
    if OUT.exists():
        try: old_pkg=pickle.load(open(OUT,'rb')); old_auc=old_pkg.get('oos_auc',0.0)
        except Exception: pass

    if oos_auc>=old_auc-0.005:
        pkg={'model':main_model,'quantile_models':q_models,'survival_model':surv_model,
             'features':feature_cols,'is_auc':round(is_auc,4),'oos_auc':round(oos_auc,4),
             'n_features':len(feature_cols),'train_years':TRAIN_YEARS,'test_years':TEST_YEARS,
             'version':'v3_regime_enc_quantile_survival','tp_mult':TP_MULT,'label_window':LABEL_WINDOW,
             'quantile_alphas':QUANTILE_ALPHAS,'has_regime_enc':True,'has_lagged_of':True}
        with open(OUT,'wb') as f: pickle.dump(pkg,f)
        log.info(f"\n💾 Salvat: {OUT}")
        log.info(f"   IS={is_auc:.4f}  OOS={oos_auc:.4f}  (prev={old_auc:.4f})")
    else:
        log.warning(f"\n⚠️  OOS regression ({oos_auc:.4f} < {old_auc:.4f} - 0.005) — old model kept")

    imp=pd.Series(main_model.feature_importances_,index=feature_cols).sort_values(ascending=False)
    log.info(f"\nTop 15 features:\n{imp.head(15).to_string()}")


if __name__=='__main__':
    train_and_save()
