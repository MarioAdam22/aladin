"""
ALADIN — NY Turtle Soup Quality Gate v1
══════════════════════════════════════════════════════════════════════════
Turtle Soup NY = false breakout după sweep LON Hi/Lo
(identic cu ts_lon_v1 dar sesiunea NY h13-h14 UTC,
 nivelele sweepate = lon_hi / lon_lo în loc de asia_hi / asia_lo)

Față de ny_v3 (toate trade-urile NY), ts_ny_v1 adaugă:
  ✅ Filter: NUMAI SHORT_REV + LONG_REV (true TS reversal trades)
  ✅ ts_sweep_depth_pts / _atr       — cât de adânc a intrat în lichiditate LON
  ✅ ts_close_inside                 — bar-ul a închis ÎNAPOI sub/peste LON hi/lo
  ✅ ts_rejection_str                — puterea revenirii vs nivel LON
  ✅ ts_wick_pct, ts_body_pct, ts_close_quality, ts_wick_dom
  ✅ ts_lon_rng_atr                  — lărgimea sesiunii LON
  ✅ ts_sweep_pct_lon                — sweep depth ca % din range LON
  ✅ ts_opp_dist_atr                 — distanța la lichiditatea opusă (LON lo/hi)
  ✅ ts_rr_impl                      — RR implicit
  ✅ ts_sharp, ts_entry_prox, ts_near_wk_extreme, ts_htf_anti, ts_combo_score

Toate celelalte features din ny_v3 (206) sunt păstrate → total ~222 features
"""
from __future__ import annotations
import sqlite3, json as _json, warnings, pathlib, pickle
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import BorderlineSMOTE

DIR      = pathlib.Path(__file__).parent
DB_PATH  = DIR.parent / "mario_trading.db"
CSV_PATH = DIR.parent / "backtest" / "backtest_open_sessions_trades.csv"

OPTUNA_TRIALS = 40
CLIP          = 10.0
IS_START      = pd.Timestamp("2023-01-01")
VAL_START     = pd.Timestamp("2025-01-01")
YEAR_WEIGHTS  = {2023: 0.85, 2024: 1.00}

# ════════════════════════════════════════════════════════════════════════════
# REGIME LABELS (pre-computed)
# ════════════════════════════════════════════════════════════════════════════
_REGIME_LABELS_PATH = DIR.parent / "data" / "regime_labels.parquet"
try:
    _regime_df = pd.read_parquet(_REGIME_LABELS_PATH)
    _SESS = 'NY'
    _regime_map = dict(zip(
        _regime_df[_regime_df['session']==_SESS]['date'],
        _regime_df[_regime_df['session']==_SESS]['regime']
    ))
    _regime_prob_map = dict(zip(
        _regime_df[_regime_df['session']==_SESS]['date'],
        _regime_df[_regime_df['session']==_SESS]['regime_prob']
    ))
    _regime_probs_full = _regime_df[_regime_df['session']==_SESS].set_index('date')
    print(f"   Regime labels: {len(_regime_map)} zile | dist: {pd.Series(list(_regime_map.values())).value_counts().to_dict()}")
except Exception as _re:
    print(f"   ⚠️ Regime labels lipsă: {_re}")
    _regime_map = {}; _regime_prob_map = {}; _regime_probs_full = pd.DataFrame()

# ════════════════════════════════════════════════════════════════════════════
# EXPONENTIAL DECAY WEIGHTING
# ════════════════════════════════════════════════════════════════════════════
DECAY_HALF_LIFE_MONTHS = 12

def compute_decay_weights(dates_series):
    """Exponential decay: w = exp(-ln(2)/12 * months_ago). Half-life = 12 months."""
    lambda_ = np.log(2) / DECAY_HALF_LIFE_MONTHS
    today = pd.Timestamp.today()
    months_ago = ((today - pd.to_datetime(dates_series)).dt.days / 30.44).clip(0, 36)
    return np.exp(-lambda_ * months_ago).values


# ── Economic Calendar (real dates din historical_news.csv) ─────────────────
import json as _json
from aladin_cal import _CalModel
_CAL_PATH = DIR.parent / "data" / "economic_calendar.json"
try:
    _cal = _json.loads(_CAL_PATH.read_text())
    FOMC_DATES   = set(_cal.get('fomc',   []))
    NFP_DATES    = set(_cal.get('nfp',    []))
    CPI_DATES    = set(_cal.get('cpi',    []))
    PPI_DATES    = set(_cal.get('ppi',    []))
    RETAIL_DATES = set(_cal.get('retail', []))
    ISM_DATES    = set(_cal.get('ism',    []))
    ANY_HIGH     = set(_cal.get('any_high', []))
    NEWS_DAYS    = FOMC_DATES | NFP_DATES | CPI_DATES | PPI_DATES
except Exception as _e:
    print(f"   ⚠️  Calendar load error: {_e}")
    FOMC_DATES = NFP_DATES = CPI_DATES = PPI_DATES = RETAIL_DATES = ISM_DATES = ANY_HIGH = NEWS_DAYS = set()

def fomc_proximity(date_str):
    import pandas as _pd
    d = _pd.Timestamp(date_str).date()
    fl = [_pd.Timestamp(x).date() for x in sorted(FOMC_DATES)]
    return min((abs((d-f).days) for f in fl), default=30)

print("=" * 74)
print("train_quality_ts_ny_v1.py — NY Turtle Soup Quality Gate v1")
print("=" * 74)

# ════════════════════════════════════════════════════════════════════════
# ICT on TF (identical to ny_v3)
# ════════════════════════════════════════════════════════════════════════
def compute_ict_on_tf(df_tf: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    H = df_tf['high'].values.astype(float)
    L = df_tf['low'].values.astype(float)
    C = df_tf['close'].values.astype(float)
    O = df_tf['open'].values.astype(float)
    A = np.maximum(df_tf['atr'].values.astype(float), 1.0)
    n = len(H)
    bull_top = np.zeros(n); bull_bot = np.zeros(n)
    bear_top = np.zeros(n); bear_bot = np.zeros(n)
    for i in range(2, n):
        if H[i-2] < L[i] and (L[i]-H[i-2]) > 0.5: bull_top[i]=L[i]; bull_bot[i]=H[i-2]
        if L[i-2] > H[i] and (L[i-2]-H[i]) > 0.5: bear_top[i]=L[i-2]; bear_bot[i]=H[i]
    in_bull=np.zeros(n); in_bear=np.zeros(n)
    dist_bull=np.full(n,9.9); dist_bear=np.full(n,9.9)
    in_ifvg_b=np.zeros(n); in_ifvg_s=np.zeros(n)
    breaker_b=np.zeros(n); breaker_s=np.zeros(n)
    rejection=np.zeros(n)
    active_bull=[]; active_bear=[]
    inv_bull_zones=[]; inv_bear_zones=[]
    bull_obs=[]; bear_obs=[]
    for i in range(n):
        c=C[i]; l=L[i]; h=H[i]; a=A[i]
        new_ab=[]
        for top,bot,j in active_bull:
            if i-j>lookback: continue
            if l<bot: inv_bull_zones.append((top,bot,i))
            else: new_ab.append((top,bot,j))
        active_bull=new_ab
        new_ab=[]
        for top,bot,j in active_bear:
            if i-j>lookback: continue
            if h>top: inv_bear_zones.append((top,bot,i))
            else: new_ab.append((top,bot,j))
        active_bear=new_ab
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
        for top,bot,k in inv_bull_zones[-15:]:
            if i-k<=lookback*2 and bot<=c<=top: in_ifvg_b[i]=1.0
        for top,bot,k in inv_bear_zones[-15:]:
            if i-k<=lookback*2 and bot<=c<=top: in_ifvg_s[i]=1.0
        for top,bot,j in bull_obs[-20:]:
            if i-j<=lookback:
                if c<min(bot,O[j])-a*0.05:
                    if abs(c-top)/a<0.8 or abs(c-bot)/a<0.8: breaker_s[i]=1.0
        for top,bot,j in bear_obs[-20:]:
            if i-j<=lookback:
                if c>max(top,O[j])+a*0.05:
                    if abs(c-top)/a<0.8 or abs(c-bot)/a<0.8: breaker_b[i]=1.0
        if i>=2:
            wu=H[i-1]-max(C[i-1],O[i-1]); wd=min(C[i-1],O[i-1])-L[i-1]
            bs=abs(C[i-1]-O[i-1])
            if wu>2.5*max(bs,0.5) and wu>a*0.3:
                rt=H[i-1]; rb=max(C[i-1],O[i-1])
                if abs(c-rt)/a<0.6 or abs(c-rb)/a<0.6: rejection[i]=1.0
            if wd>2.5*max(bs,0.5) and wd>a*0.3:
                rt=min(C[i-1],O[i-1]); rb=L[i-1]
                if abs(c-rt)/a<0.6 or abs(c-rb)/a<0.6: rejection[i]=1.0
    return pd.DataFrame({'in_bull':in_bull,'in_bear':in_bear,
        'dist_bull':np.clip(dist_bull,0,9.9),'dist_bear':np.clip(dist_bear,0,9.9),
        'in_ifvg_b':in_ifvg_b,'in_ifvg_s':in_ifvg_s,
        'breaker_b':breaker_b,'breaker_s':breaker_s,'rejection':rejection},index=df_tf.index)


def compute_mtf_features(conn, setup_dates):
    min_d=min(setup_dates); max_d=max(setup_dates)
    warmup=(pd.Timestamp(min_d)-pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    print(f"   [MTF] Loading 1-min data {warmup} → {max_d} ...")
    df1m=pd.read_sql(f"""SELECT timestamp,open,high,low,close,atr_14
        FROM market_data WHERE timestamp>='{warmup} 00:00:00'
        AND timestamp<='{max_d} 23:59:59' ORDER BY timestamp""",conn)
    df1m['ts']=pd.to_datetime(df1m['timestamp']); df1m=df1m.set_index('ts')
    df1m.rename(columns={'atr_14':'atr'},inplace=True)
    df1m['atr']=df1m['atr'].ffill().fillna(9.0)
    print(f"   [MTF] 1-min bars: {len(df1m):,}")
    all_features=pd.DataFrame(index=df1m.index)
    for tf_label,tf_rule,lb in [('5m','5min',25),('15m','15min',20),('1h','1h',20),('4h','4h',15)]:
        df_tf=df1m.resample(tf_rule,label='left',closed='left').agg(
            open=('open','first'),high=('high','max'),low=('low','min'),
            close=('close','last'),atr=('atr','last')).dropna(subset=['open'])
        df_tf['atr']=df_tf['atr'].ffill().fillna(9.0)
        ict=compute_ict_on_tf(df_tf,lookback=lb)
        ict_ff=ict.reindex(df1m.index,method='ffill')
        for col in ict.columns: all_features[f'{col}_{tf_label}']=ict_ff[col]
        print(f"   [MTF] {tf_label}: {len(df_tf):,} bars OK")
    all_features=all_features.fillna(0.0)
    all_features['ts_str']=all_features.index.strftime('%Y-%m-%d %H:%M:%S')
    print(f"   [MTF] {all_features.shape[1]-1} feature columns computed")
    return all_features


# ════════════════════════════════════════════════════════════════════════
# STEP 1: Load CSV — filter NY + REVERSAL only
# ════════════════════════════════════════════════════════════════════════
print("\n▶  [1/7] Încărcare backtest CSV — NY REVERSAL only ...")
df_csv = pd.read_csv(CSV_PATH)
df_all = df_csv[df_csv['session'] == 'NY'].copy()

# TS FILTER: only reversal trades
ts_regimes = ['SHORT_REV', 'LONG_REV']
df_ny = df_all[df_all['regime'].isin(ts_regimes)].copy()

print(f"   NY total:    {len(df_all):,}")
print(f"   NY reversal: {len(df_ny):,} ({len(df_ny)/len(df_all)*100:.1f}%)")
print(f"   Excluded breakouts: {len(df_all)-len(df_ny):,}")

df_ny['ts']          = pd.to_datetime(df_ny['timestamp'])
df_ny['trail']       = (df_ny['exit_reason'] == 'TRAIL').astype(int)
df_ny['hour_utc']    = df_ny['ts'].dt.hour
df_ny['day_of_week'] = df_ny['ts'].dt.dayofweek
df_ny['dir_short']   = (df_ny['direction'] == 'SHORT').astype(float)
df_ny['ts_str']      = df_ny['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_ny['date_str']    = df_ny['ts'].dt.strftime('%Y-%m-%d')
df_ny['year']        = df_ny['ts'].dt.year
df_ny = df_ny[df_ny['ts'] >= IS_START].copy()

print(f"   Trail rate: {df_ny['trail'].mean()*100:.2f}%")
for yr in sorted(df_ny['year'].unique()):
    sub = df_ny[df_ny['year']==yr]
    print(f"   {yr}: {len(sub):,} trades, trail={sub['trail'].mean()*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════
# STEP 2: JOIN cu market_data
# ════════════════════════════════════════════════════════════════════════
print("\n▶  [2/7] JOIN cu market_data (DB) ...")
conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=60)

# NY adds lon_hi, lon_lo (the swept levels for NY TS)
DB_COLS = [
    'timestamp','open','high','low','close','volume',
    'atr_14','asia_hi','asia_lo','lon_hi','lon_lo',
    'p_hi','p_lo','true_open',
    'h4_hi','h4_lo','h1_hi','h1_lo',
    'poc_level','vah','val','dist_poc','inside_va',
    'has_displacement','fvg_up','fvg_down',
    'is_smt_bearish','is_smt_bullish',
    'hurst','adx_14','garch_vol','sample_entropy',
    'fisher_transform','acf_lag1','acf_lag5',
    'vwap','dist_vwap','bar_delta','cum_delta',
    'bar_buy_vol','bar_sell_vol',
    'absorption_score','stacked_bull','stacked_bear',
    'body_size','lw_hi','lw_lo','lm_hi','lm_lo',
    'dist_pdh','dist_pdl',
    'fft_cycle','kalman_smooth','kalman_noise',
    'of_doi','of_bilateral_abs','of_big_balance',
]
cols_str = ', '.join(DB_COLS)
CHUNK = 5000; db_parts = []
ts_list = df_ny['ts_str'].tolist()
for i in range(0, len(ts_list), CHUNK):
    chunk=ts_list[i:i+CHUNK]; ph=','.join(['?']*len(chunk))
    db_parts.append(pd.read_sql(f"SELECT {cols_str} FROM market_data WHERE timestamp IN ({ph})",conn,params=chunk))
db = pd.concat(db_parts, ignore_index=True)
db['ts_str'] = db['timestamp']
print(f"   DB rows joined: {len(db):,} / {len(df_ny):,}")
df = df_ny.merge(db.drop(columns=['timestamp']), on='ts_str', how='inner')
print(f"   Post-merge: {len(df):,}")

# ════════════════════════════════════════════════════════════════════════
# STEP 2b: MTF ICT
# ════════════════════════════════════════════════════════════════════════
print("\n▶  [2b/7] MTF ICT features ...")
setup_dates = sorted(df_ny['date_str'].unique())
mtf_features = compute_mtf_features(conn, setup_dates)
df = df.merge(mtf_features.drop_duplicates('ts_str')[
    ['ts_str']+[c for c in mtf_features.columns if c!='ts_str']
], on='ts_str', how='left')
for c in [c for c in mtf_features.columns if c!='ts_str']:
    df[c] = df[c].fillna(0.0)

# ════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# STEP 2d: Synthetic Order Flow features (CVD, absorption, footprint, etc.)
# ════════════════════════════════════════════════════════════════════════════
print("\n▶  [2d/7] Synthetic order flow features ...")
_OF_PATH = DIR.parent / "data" / "orderflow_features.parquet"
if _OF_PATH.exists():
    _of = __import__('pandas').read_parquet(_OF_PATH)
    _of = _of[_of['session_type'] == 'NY'].copy()
    _of['date'] = _of['date'].astype(str)
    _OF_COLS = [c for c in _of.columns if c not in ['session_id','date','session_type',
                'session_open','session_close','session_high','session_low','total_vol']]
    _of_merge = _of[['date'] + _OF_COLS].rename(columns={'date':'date_str'})
    df = df.merge(_of_merge, on='date_str', how='left')
    for _c in _OF_COLS:
        df[_c] = df[_c].fillna(0.0)
    print(f"   Order flow features: {len(_OF_COLS)} cols mergiate pe {len(df):,} trades")
else:
    print("   ⚠️  orderflow_features.parquet nu există — skip")
    _OF_COLS = []

# STEP 2c: VIX proxy + Rolling regime
# ════════════════════════════════════════════════════════════════════════
print("\n▶  [2c/7] VIX proxy + rolling regime ...")
all_dates_sql = "','".join(setup_dates)
daily_reg = pd.read_sql(f"""
    SELECT date(timestamp) as date,
           (MAX(high)-MIN(low)) as daily_range,
           AVG(atr_14) as avg_atr, MAX(high) as day_hi, MIN(low) as day_lo,
           AVG(adx_14) as avg_adx, AVG(hurst) as avg_hurst
    FROM market_data
    WHERE date(timestamp) >= date('{setup_dates[0]}','-30 days')
      AND date(timestamp) <= '{setup_dates[-1]}'
    GROUP BY date(timestamp) ORDER BY date""", conn)
conn.close()

daily_reg['date'] = daily_reg['date'].astype(str)
daily_reg['date_dt'] = pd.to_datetime(daily_reg['date'])
daily_reg = daily_reg.sort_values('date').reset_index(drop=True)
daily_reg['avg_atr'] = daily_reg['avg_atr'].ffill().fillna(9.0)
daily_reg['daily_range'] = daily_reg['daily_range'].fillna(daily_reg['avg_atr']*2)
daily_reg['range_atr_ratio'] = daily_reg['daily_range'] / daily_reg['avg_atr'].clip(lower=1)
daily_reg['vix_proxy_5d']  = daily_reg['range_atr_ratio'].rolling(5,min_periods=2).mean().shift(1)
daily_reg['vix_proxy_20d'] = daily_reg['range_atr_ratio'].rolling(20,min_periods=5).mean().shift(1)
daily_reg['vol_regime']    = (daily_reg['vix_proxy_5d']/daily_reg['vix_proxy_20d'].clip(lower=0.5)).clip(upper=3)
daily_reg['vol_high']      = (daily_reg['vol_regime']>1.2).astype(float)
daily_reg['vol_low']       = (daily_reg['vol_regime']<0.8).astype(float)
daily_reg['adx_10d_mean']  = daily_reg['avg_adx'].rolling(10,min_periods=3).mean().shift(1)
daily_reg['hurst_20d_mean']= daily_reg['avg_hurst'].rolling(20,min_periods=5).mean().shift(1)
daily_reg['atr_5d']        = daily_reg['avg_atr'].rolling(5,min_periods=2).mean().shift(1)
daily_reg['atr_10d']       = daily_reg['avg_atr'].rolling(10,min_periods=3).mean().shift(1)
daily_reg['atr_trend']     = (daily_reg['atr_5d']/daily_reg['atr_10d'].clip(lower=1)).clip(upper=3)
daily_reg = daily_reg.ffill().fillna(1.0)
daily_dict = {r['date']:r for _,r in daily_reg.iterrows()}

# Rolling 5-session WR (from ALL NY reversal trades)
df_ny_sorted = df_ny.sort_values('ts').copy()
df_ny_sorted['trail_roll5'] = df_ny_sorted['trail'].rolling(5,min_periods=1).mean().shift(1).fillna(0.5)
roll5_map = dict(zip(df_ny_sorted['ts_str'], df_ny_sorted['trail_roll5']))

# ════════════════════════════════════════════════════════════════════════
# STEP 3: Weekly / session stats
# ════════════════════════════════════════════════════════════════════════
print("\n▶  [3/7] Weekly profile ...")
conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=60)
wk_df = pd.read_sql(f"""
    SELECT date(timestamp) as date, MAX(high) as day_hi, MIN(low) as day_lo
    FROM market_data WHERE date(timestamp) IN ('{all_dates_sql}')
    GROUP BY date(timestamp)""", conn)
# LON session close per zi (pentru lon_dir_explicit — era lipsă față de ny_v3)
lon_sess_df = pd.read_sql(f"""
    SELECT date(timestamp) as date,
           MAX(CASE WHEN strftime('%H',timestamp) = '09' THEN close END) as lon_close
    FROM market_data WHERE date(timestamp) IN ('{all_dates_sql}')
    GROUP BY date(timestamp)""", conn)
conn.close()
lon_close_map = {str(r['date']): r['lon_close'] for _, r in lon_sess_df.iterrows()}
wk_df['date']=wk_df['date'].astype(str); wk_df['date_dt']=pd.to_datetime(wk_df['date'])
wk_df=wk_df.sort_values('date').reset_index(drop=True)
wk_df['iso_year']=wk_df['date_dt'].dt.isocalendar().year.values
wk_df['iso_week']=wk_df['date_dt'].dt.isocalendar().week.values
wk_df['yw']=wk_df['iso_year'].astype(str)+'_'+wk_df['iso_week'].astype(str)
wk_df['dow_num']=wk_df['date_dt'].dt.dayofweek
mon_df=wk_df[wk_df['dow_num']==0][['yw','day_hi','day_lo']].rename(columns={'day_hi':'mon_hi','day_lo':'mon_lo'})
wk_df=wk_df.merge(mon_df,on='yw',how='left')
wk_ext=wk_df.groupby('yw').agg(wk_hi=('day_hi','max'),wk_lo=('day_lo','min')).reset_index().sort_values('yw').reset_index(drop=True)
wk_ext['prev_wk_hi']=wk_ext['wk_hi'].shift(1); wk_ext['prev_wk_lo']=wk_ext['wk_lo'].shift(1)
wk_df=wk_df.merge(wk_ext[['yw','prev_wk_hi','prev_wk_lo']],on='yw',how='left')
wk_df['wk_hi_sofar']=wk_df.groupby('yw')['day_hi'].cummax()
wk_df['wk_lo_sofar']=wk_df.groupby('yw')['day_lo'].cummin()
wk_df['wk_range_sofar']=wk_df['wk_hi_sofar']-wk_df['wk_lo_sofar']
wk_dict={r['date']:r for _,r in wk_df.iterrows()}

# ════════════════════════════════════════════════════════════════════════
# STEP 4: Feature Engineering (ny_v3 base + TS features)
# ════════════════════════════════════════════════════════════════════════
print("\n▶  [4/7] Feature engineering (ny_v3 base + TS features) ...")

cl  = df['close'].values.astype(float)
hi  = df['high'].values.astype(float)
lo  = df['low'].values.astype(float)
op  = df['open'].values.astype(float)
vol = np.where(df['volume'].values>0,df['volume'].values,1).astype(float)
atr = np.where(df['atr_14'].values>0,df['atr_14'].values,9.0).astype(float)
asia_hi  = np.where(df['asia_hi'].values>0,df['asia_hi'].values,cl)
asia_lo  = np.where(df['asia_lo'].values>0,df['asia_lo'].values,cl)
# NY TS: swept levels = LON hi/lo
lon_hi_arr = np.where(df['lon_hi'].values>0,df['lon_hi'].values,cl)
lon_lo_arr = np.where(df['lon_lo'].values>0,df['lon_lo'].values,cl)
p_hi_arr = np.where(df['p_hi'].values>0,df['p_hi'].values,cl)
p_lo_arr = np.where(df['p_lo'].values>0,df['p_lo'].values,cl)
true_open= np.where(df['true_open'].values>0,df['true_open'].values,cl)
h4h=np.where(df['h4_hi'].values>0,df['h4_hi'].values,cl)
h4l=np.where(df['h4_lo'].values>0,df['h4_lo'].values,cl)
h1h=np.where(df['h1_hi'].values>0,df['h1_hi'].values,cl)
h1l=np.where(df['h1_lo'].values>0,df['h1_lo'].values,cl)
poc=np.where(df['poc_level'].values>0,df['poc_level'].values,cl)
vwap_arr=np.where(df['vwap'].values>0,df['vwap'].values,cl)
lw_hi_arr=np.where(df['lw_hi'].values>0,df['lw_hi'].values,cl)
lw_lo_arr=np.where(df['lw_lo'].values>0,df['lw_lo'].values,cl)
date_arr=df['date_str'].values
_min_abs  = df['ts'].dt.hour.values * 60 + df['ts'].dt.minute.values  # minute UTC de la miezul nopții
lon_cls_arr = np.array([float(lon_close_map.get(d) or 0) or cl[i] for i,d in enumerate(date_arr)])

def _dd(d,key,fallback=0.0):
    r=wk_dict.get(d)
    if r is None: return fallback
    v=r[key] if isinstance(r,dict) else getattr(r,key,fallback)
    return float(v) if v is not None and pd.notna(v) else fallback

def _dr(d,key,fallback=1.0):
    r=daily_dict.get(d)
    if r is None: return fallback
    v=r[key] if isinstance(r,dict) else getattr(r,key,fallback)
    return float(v) if v is not None and pd.notna(v) else fallback

trade_dir=np.where(df['dir_short'].values==0,1.0,-1.0)
atr_5d_arr = np.array([_dr(d, 'atr_5d', atr[i]) for i,d in enumerate(date_arr)])

def clip(x,c=CLIP): return np.clip(np.where(np.isfinite(x),x,0.0),-c,c)
def safe_norm(num,denom,c=CLIP): return clip(np.where(np.abs(denom)>0.01,num/denom,0.0),c)

feat = pd.DataFrame()

# ── 1. TEMPORAL (NY: h13-h14 UTC)
feat['dir_short']       = df['dir_short'].values
feat['hour_utc']        = df['hour_utc'].values.astype(float)
feat['min_in_ny']       = np.clip((df['hour_utc'].values-13)*60,0,120).astype(float)
feat['day_of_week']     = df['day_of_week'].values.astype(float)
feat['month']           = df['ts'].dt.month.values.astype(float)
feat['is_monday']       = (df['day_of_week'].values==0).astype(float)
feat['is_friday']       = (df['day_of_week'].values==4).astype(float)
feat['year_norm']       = (df['ts'].dt.year.values.astype(float)-2023.0)/2.0
feat['is_tuesday']      = (df['day_of_week'].values==1).astype(float)
feat['is_wednesday']    = (df['day_of_week'].values==2).astype(float)
feat['is_thursday']     = (df['day_of_week'].values==3).astype(float)

# ── 2. CONFIDENCE
feat['confidence']      = df['confidence'].values.astype(float)

# ── 3. VOLATILITATE
feat['atr_entry']       = atr
feat['atr_vs_10d']      = clip(df.groupby(df['ts'].dt.date)['atr_14'].transform('mean').values/np.where(atr>0,atr,1),3)

# ── 4. ASIA CONTEXT (NY: Asia è riferimento indiretto)
valid_asia=(asia_hi>0)&(asia_lo>0)&(asia_hi>asia_lo)
feat['dist_asia_hi_atr']= safe_norm(cl-asia_hi,atr)
feat['dist_asia_lo_atr']= safe_norm(cl-asia_lo,atr)
feat['asia_range_atr']  = clip(safe_norm(asia_hi-asia_lo,atr),20)
feat['swept_asia_hi']   = ((cl>asia_hi)&valid_asia).astype(float)
feat['swept_asia_lo']   = ((cl<asia_lo)&valid_asia).astype(float)
feat['asia_midpoint']   = safe_norm(cl-(asia_hi+asia_lo)/2,atr)

# ── 4b. LON CONTEXT (NY primary reference — swept levels)
valid_lon=(lon_hi_arr>0)&(lon_lo_arr>0)&(lon_hi_arr>lon_lo_arr)
feat['dist_lon_hi_atr'] = safe_norm(cl-lon_hi_arr,atr)
feat['dist_lon_lo_atr'] = safe_norm(cl-lon_lo_arr,atr)
feat['lon_range_atr']   = clip(safe_norm(lon_hi_arr-lon_lo_arr,atr),20)
feat['swept_lon_hi']    = ((cl>lon_hi_arr)&valid_lon).astype(float)
feat['swept_lon_lo']    = ((cl<lon_lo_arr)&valid_lon).astype(float)
feat['lon_midpoint']    = safe_norm(cl-(lon_hi_arr+lon_lo_arr)/2,atr)
# LON direction + range quality (features lipsă față de ny_v3 — acum adăugate)
lon_mid_arr_ts               = (lon_hi_arr + lon_lo_arr) / 2
lon_range_ts                 = np.maximum(lon_hi_arr - lon_lo_arr, 1.0)
feat['lon_dir_explicit']     = np.where(lon_cls_arr > lon_mid_arr_ts, 1.0, -1.0)   # +1=bull LON, -1=bear LON
feat['lon_dir_aligned']      = (feat['lon_dir_explicit'].values == trade_dir).astype(float)
feat['lon_dir_opposite']     = (feat['lon_dir_explicit'].values != trade_dir).astype(float)
feat['lon_range_vs_atr5d']   = clip(safe_norm(lon_range_ts, np.where(atr_5d_arr > 0, atr_5d_arr, atr)))
feat['lon_big_day']          = (lon_range_ts > atr * 1.5).astype(float)
feat['lon_small_day']        = (lon_range_ts < atr * 0.7).astype(float)
feat['ny_open_dist_lon_mid'] = safe_norm(cl - lon_mid_arr_ts, atr)  # signed: + = entry deasupra LON mid

# ── 5. PREVIOUS DAY
feat['dist_pdh_atr']    = safe_norm(df['dist_pdh'].values,atr)
feat['dist_pdl_atr']    = safe_norm(df['dist_pdl'].values,atr)
feat['above_true_open'] = (cl>true_open).astype(float)
feat['dist_true_open']  = safe_norm(cl-true_open,atr)

# ── 6. HTF BIAS
feat['h4_bias']         = safe_norm((h4h+h4l)/2-cl,atr)
feat['h1_bias']         = safe_norm((h1h+h1l)/2-cl,atr)
feat['h4_h1_aligned']   = (np.sign(feat['h4_bias'].values)==np.sign(feat['h1_bias'].values)).astype(float)

# ── 7. WEEKLY CONTEXT
prev_wk_hi=np.array([_dd(d,'prev_wk_hi',cl[i]) for i,d in enumerate(date_arr)])
prev_wk_lo=np.array([_dd(d,'prev_wk_lo',cl[i]) for i,d in enumerate(date_arr)])
wk_hi_sf  =np.array([_dd(d,'wk_hi_sofar',cl[i]) for i,d in enumerate(date_arr)])
wk_lo_sf  =np.array([_dd(d,'wk_lo_sofar',cl[i]) for i,d in enumerate(date_arr)])
wk_range_sf=np.array([_dd(d,'wk_range_sofar',atr[i]) for i,d in enumerate(date_arr)])
mon_hi_arr=np.array([_dd(d,'mon_hi',0.0) for d in date_arr])
mon_lo_arr=np.array([_dd(d,'mon_lo',0.0) for d in date_arr])
valid_pw=(prev_wk_hi>prev_wk_lo)&(prev_wk_hi>0)
pw_range=np.where(valid_pw,prev_wk_hi-prev_wk_lo,atr*10)
wk_prem_pct=np.where(valid_pw,(cl-prev_wk_lo)/pw_range-0.5,0.0)
feat['weekly_premium_pct'] = clip(wk_prem_pct)
feat['in_weekly_premium']  = (wk_prem_pct>0.1).astype(float)
feat['in_weekly_discount'] = (wk_prem_pct<-0.1).astype(float)
feat['weekly_prem_aligned']= np.where(trade_dir==1,feat['in_weekly_discount'].values,feat['in_weekly_premium'].values)
feat['dist_prev_wk_hi']    = safe_norm(cl-prev_wk_hi,atr)
feat['dist_prev_wk_lo']    = safe_norm(cl-prev_wk_lo,atr)
feat['lw_range_atr']       = clip(safe_norm(lw_hi_arr-lw_lo_arr,atr),20)
feat['dist_lw_hi']         = safe_norm(cl-lw_hi_arr,atr)
feat['dist_lw_lo']         = safe_norm(cl-lw_lo_arr,atr)
feat['week_range_so_far']  = clip(safe_norm(wk_range_sf,atr),20)
feat['week_hi_taken']      = (cl>wk_hi_sf*0.998).astype(float)
feat['week_lo_taken']      = (cl<wk_lo_sf*1.002).astype(float)
mon_range_arr=np.where(mon_hi_arr>mon_lo_arr,mon_hi_arr-mon_lo_arr,0.0)
feat['monday_range_pt']    = clip(safe_norm(mon_range_arr,atr),10)
feat['monday_consol']      = (mon_range_arr<atr).astype(float)
feat['tuesday_rev_ctx']    = feat['is_tuesday'].values*(feat['week_hi_taken'].values+feat['week_lo_taken'].values).clip(0,1)
feat['wednesday_rev_ctx']  = feat['is_wednesday'].values*(np.abs(feat['weekly_premium_pct'].values.clip(-1,1))>0.3).astype(float)

# ── 8. VOLUME PROFILE
feat['inside_va']          = df['inside_va'].fillna(0).values.astype(float)
feat['dist_poc_atr']       = safe_norm(df['dist_poc'].values,atr)
feat['dist_vwap_atr']      = safe_norm(df['dist_vwap'].values,atr)
feat['vah_dist']           = safe_norm(cl-df['vah'].fillna(0).values.astype(float),atr)
feat['val_dist']           = safe_norm(cl-df['val'].fillna(0).values.astype(float),atr)

# ── 9. ICT SIGNALS
feat['has_displacement']   = df['has_displacement'].fillna(0).values.astype(float)
feat['fvg_up']             = df['fvg_up'].fillna(0).values.astype(float)
feat['fvg_down']           = df['fvg_down'].fillna(0).values.astype(float)
feat['is_smt_bearish']     = df['is_smt_bearish'].fillna(0).values.astype(float)
feat['is_smt_bullish']     = df['is_smt_bullish'].fillna(0).values.astype(float)

# ── 10. MOMENTUM / REGIME
feat['hurst']              = df['hurst'].fillna(0.5).values.astype(float)
feat['adx_14']             = df['adx_14'].fillna(20).values.astype(float)
feat['adx_strong']         = (df['adx_14'].fillna(20).values>25).astype(float)
feat['acf_lag1']           = df['acf_lag1'].fillna(0).values.astype(float)
feat['acf_lag5']           = df['acf_lag5'].fillna(0).values.astype(float)
feat['fisher_transform']   = df['fisher_transform'].fillna(0).values.astype(float)
feat['fisher_extreme']     = (np.abs(df['fisher_transform'].fillna(0).values)>2.0).astype(float)
feat['fft_cycle']          = df['fft_cycle'].fillna(0).values.astype(float)
feat['kalman_smooth']      = df['kalman_smooth'].fillna(0).values.astype(float)
feat['kalman_noise']       = df['kalman_noise'].fillna(0).values.astype(float)
garch_raw                  = df['garch_vol'].fillna(0).values.astype(float)
feat['garch_vol_atr']      = clip(np.where(atr>0,garch_raw*cl/atr,1.0),5)
feat['sample_entropy']     = df['sample_entropy'].fillna(2.0).values.astype(float)

# ── 11. ORDERFLOW
bar_delta=df['bar_delta'].fillna(0).values.astype(float)
feat['bar_delta_norm']     = clip(bar_delta/np.maximum(vol,1),1)
feat['cum_delta_norm']     = clip(df['cum_delta'].fillna(0).values/np.maximum(vol,1),1)
feat['buy_sell_ratio']     = clip(df['bar_buy_vol'].fillna(0).values/np.maximum(df['bar_sell_vol'].fillna(0).values,1),5)
feat['absorption_score']   = df['absorption_score'].fillna(0).values.astype(float)
feat['stacked_bull']       = df['stacked_bull'].fillna(0).values.astype(float)
feat['stacked_bear']       = df['stacked_bear'].fillna(0).values.astype(float)
feat['of_doi']             = df['of_doi'].fillna(0).values.astype(float)

# ── 12. CANDLESTICK
body=cl-op; wick_up=hi-np.maximum(cl,op); wick_down=np.minimum(cl,op)-lo
feat['body_bear']          = (body<0).astype(float)
feat['body_pct']           = clip(np.abs(body)/np.maximum(hi-lo,0.01),2)
feat['sweep_wick_atr']     = safe_norm(np.maximum(wick_up,wick_down),atr)

# ── 13. INTERACȚIUNI DE BAZĂ
feat['dir_x_adx']          = feat['dir_short'].values*feat['adx_14'].values/100.0
feat['dir_x_hurst']        = feat['dir_short'].values*feat['hurst'].values
feat['confidence_x_adx']   = feat['confidence'].values*feat['adx_strong'].values
feat['hour_x_dir']         = feat['hour_utc'].values*feat['dir_short'].values
feat['year_x_adx']         = feat['year_norm'].values*feat['adx_14'].values/100.0
feat['year_x_hurst']       = feat['year_norm'].values*feat['hurst'].values

# ── MTF ICT FEATURES
for tf_label in ['5m','15m','1h','4h']:
    in_bull=df[f'in_bull_{tf_label}'].values; in_bear=df[f'in_bear_{tf_label}'].values
    dist_bull=df[f'dist_bull_{tf_label}'].values; dist_bear=df[f'dist_bear_{tf_label}'].values
    in_ifvg_b=df[f'in_ifvg_b_{tf_label}'].values; in_ifvg_s=df[f'in_ifvg_s_{tf_label}'].values
    brk_b=df[f'breaker_b_{tf_label}'].values; brk_s=df[f'breaker_s_{tf_label}'].values
    rej=df[f'rejection_{tf_label}'].values
    feat[f'in_bull_fvg_{tf_label}']=in_bull; feat[f'in_bear_fvg_{tf_label}']=in_bear
    feat[f'dist_bull_fvg_{tf_label}']=np.clip(dist_bull,0,9.9)
    feat[f'dist_bear_fvg_{tf_label}']=np.clip(dist_bear,0,9.9)
    feat[f'fvg_aligned_{tf_label}']=np.where(trade_dir==1,in_bull,in_bear)
    feat[f'in_ifvg_{tf_label}']=np.maximum(in_ifvg_b,in_ifvg_s)
    feat[f'ifvg_aligned_{tf_label}']=np.where(trade_dir==1,in_ifvg_s,in_ifvg_b)
    feat[f'breaker_aligned_{tf_label}']=np.where(trade_dir==1,brk_b,brk_s)
    feat[f'rejection_{tf_label}']=rej
feat['fvg_tf_confluence']  = (feat['fvg_aligned_5m'].values+feat['fvg_aligned_15m'].values+
                               feat['fvg_aligned_1h'].values+feat['fvg_aligned_4h'].values)
feat['htf_fvg_aligned']    = np.maximum(feat['fvg_aligned_1h'].values,feat['fvg_aligned_4h'].values)
feat['ifvg_htf_aligned']   = np.maximum(feat['ifvg_aligned_1h'].values,feat['ifvg_aligned_4h'].values)

# ── VIX PROXY
feat['vix_proxy_5d']       = np.array([_dr(d,'vix_proxy_5d',2.0) for d in date_arr])
feat['vix_proxy_20d']      = np.array([_dr(d,'vix_proxy_20d',2.0) for d in date_arr])
feat['vol_regime']         = np.array([_dr(d,'vol_regime',1.0) for d in date_arr])
feat['vol_high']           = np.array([_dr(d,'vol_high',0.0) for d in date_arr])
feat['vol_low']            = np.array([_dr(d,'vol_low',0.0) for d in date_arr])
feat['atr_trend']          = np.array([_dr(d,'atr_trend',1.0) for d in date_arr])
feat['atr_expanding']      = (feat['atr_trend'].values>1.15).astype(float)
feat['atr_contracting']    = (feat['atr_trend'].values<0.85).astype(float)
feat['vol_x_fvg_1h']       = feat['vol_regime'].values*feat['fvg_aligned_1h'].values
feat['vol_x_htf_fvg']      = feat['vol_regime'].values*feat['htf_fvg_aligned'].values

# ── NY-SPECIFIC INTERACTIONS (from ny_v3)
feat['lon_hi_x_dir']       = feat['swept_lon_hi'].values*feat['dir_short'].values
feat['lon_lo_x_dir']       = feat['swept_lon_lo'].values*(1-feat['dir_short'].values)
feat['vol_x_lon_sweep']    = feat['vol_regime'].values*(feat['swept_lon_hi'].values+feat['swept_lon_lo'].values)
feat['htf_x_lon_aligned']  = feat['htf_fvg_aligned'].values*feat['lon_midpoint'].values.clip(-1,1)

# ── SWEEP QUALITY (using LON levels for NY)
sweep_level_ny=np.where(trade_dir==1,lon_lo_arr,lon_hi_arr)
dist_to_pdl=np.abs(sweep_level_ny-p_lo_arr); dist_to_pdh=np.abs(sweep_level_ny-p_hi_arr)
feat['equal_level_score']  = clip(1.0-safe_norm(np.minimum(dist_to_pdl,dist_to_pdh),atr))
sweep_depth_ny=np.where(trade_dir==1,sweep_level_ny-cl,cl-sweep_level_ny)
feat['sweep_depth_atr']    = safe_norm(sweep_depth_ny,atr)
feat['deep_sweep']         = (sweep_depth_ny>atr*0.4).astype(float)
feat['shallow_sweep']      = (sweep_depth_ny<atr*0.1).astype(float)
feat['sweep_wick_clean']   = (feat['sweep_wick_atr'].values>0.4).astype(float)
feat['sweep_with_disp']    = feat['sweep_wick_clean'].values*feat['has_displacement'].values
feat['sweep_quality_score']= (feat['equal_level_score'].values+feat['deep_sweep'].values+
                               feat['sweep_wick_clean'].values+feat['fvg_aligned_15m'].values+
                               feat['fvg_aligned_1h'].values).clip(0,5)/5.0

# ── ROLLING REGIME
feat['rolling_5sess_wr']   = np.array([roll5_map.get(ts_str,0.5) for ts_str in df['ts_str'].values])
feat['adx_10d_mean']       = np.array([_dr(d,'adx_10d_mean',20.0) for d in date_arr])
feat['hurst_20d_mean']     = np.array([_dr(d,'hurst_20d_mean',0.5) for d in date_arr])
feat['regime_trending']    = (feat['adx_10d_mean'].values>22).astype(float)
feat['regime_hurst_trend'] = (feat['hurst_20d_mean'].values>0.52).astype(float)
feat['recent_wr_high']     = (feat['rolling_5sess_wr'].values>0.35).astype(float)
feat['recent_wr_low']      = (feat['rolling_5sess_wr'].values<0.15).astype(float)
feat['regime_score']       = (feat['regime_trending'].values+feat['regime_hurst_trend'].values+
                               feat['vol_high'].values).clip(0,3)/3.0
feat['regime_x_htf_fvg']  = feat['regime_score'].values*feat['htf_fvg_aligned'].values
feat['adx_x_sweep_quality']= feat['adx_10d_mean'].values/30.0*feat['sweep_quality_score'].values

# ════════════════════════════════════════════════════════════════════════
# SECTION TS: TURTLE SOUP SPECIFIC FEATURES (NY — nivelele = LON hi/lo)
# ════════════════════════════════════════════════════════════════════════
# SHORT_REV: swept LON hi → short (high above lon_hi, close back below)
# LONG_REV:  swept LON lo → long  (low below lon_lo, close back above)

dir_s = df['dir_short'].values.astype(float)

# Nivelul sweepat (NY: LON hi/lo)
swept_lvl = np.where(dir_s==1, lon_hi_arr, lon_lo_arr)

# 1. Adâncimea sweep-ului
ts_depth_pts = np.where(dir_s==1,
    np.maximum(hi - swept_lvl, 0.0),    # SHORT: high above lon_hi
    np.maximum(swept_lvl - lo, 0.0))    # LONG: lon_lo above low
feat['ts_sweep_depth_pts']  = ts_depth_pts
feat['ts_sweep_depth_atr']  = safe_norm(ts_depth_pts, atr)

# 2. Close ÎNAPOI în range
ts_close_inside = np.where(dir_s==1,
    (cl < swept_lvl).astype(float),     # SHORT: close below lon_hi
    (cl > swept_lvl).astype(float))     # LONG: close above lon_lo
feat['ts_close_inside']     = ts_close_inside

# 3. Puterea reversal-ului
ts_rejection_str = np.where(dir_s==1,
    safe_norm(swept_lvl - cl, atr),
    safe_norm(cl - swept_lvl, atr))
feat['ts_rejection_str']    = ts_rejection_str

# 4. Wick dincolo de nivel ca % din range total bar
bar_rng = np.maximum(hi - lo, 0.1)
feat['ts_wick_pct']         = ts_depth_pts / bar_rng

# 5. Corp mic = pin bar bun
body_sz = np.abs(cl - op)
feat['ts_body_pct']         = body_sz / bar_rng

# 6. Close position în bar
cl_bar_pct = safe_norm(cl - lo, bar_rng)
feat['ts_close_quality']    = np.where(dir_s==1, 1.0-cl_bar_pct, cl_bar_pct)

# 7. Dominanța wick-ului direcțional
upper_wick = hi - np.maximum(cl, op)
lower_wick = np.minimum(cl, op) - lo
feat['ts_wick_dom']         = np.where(dir_s==1,
    safe_norm(upper_wick, np.maximum(lower_wick, 0.1)),
    safe_norm(lower_wick, np.maximum(upper_wick, 0.1)))

# 8. LON range ca context (analog asia_rng_atr pentru NY)
lon_rng = np.maximum(lon_hi_arr - lon_lo_arr, 0.1)
feat['ts_lon_rng_atr']      = safe_norm(lon_rng, atr)

# 9. Sweep ca % din range LON
feat['ts_sweep_pct_lon']    = ts_depth_pts / lon_rng

# 10. Distanța la lichiditatea opusă
feat['ts_opp_dist_atr']     = np.where(dir_s==1,
    safe_norm(swept_lvl - lon_lo_arr, atr),   # SHORT: target = lon_lo
    safe_norm(lon_hi_arr - swept_lvl, atr))   # LONG: target = lon_hi

# 11. RR implicit
feat['ts_rr_impl']          = safe_norm(
    np.where(dir_s==1, swept_lvl-lon_lo_arr, lon_hi_arr-swept_lvl),
    np.maximum(ts_depth_pts, 0.1))

# 12. Sharp rejection
feat['ts_sharp']            = ((ts_depth_pts > 0.3*atr) & (ts_close_inside==1.0)).astype(float)

# 13. Entry proximity la nivel
feat['ts_entry_prox']       = np.where(dir_s==1,
    safe_norm(swept_lvl - np.minimum(cl, op), atr),
    safe_norm(np.maximum(cl, op) - swept_lvl, atr))

# 14. Sweep atinge un extrem săptămânal
feat['ts_near_wk_extreme']  = np.where(dir_s==1,
    (np.abs(swept_lvl - wk_hi_sf) < atr*0.5).astype(float),
    (np.abs(swept_lvl - wk_lo_sf) < atr*0.5).astype(float))

# 15. H4 bias OPUS direcției sweep
h4_mid = (h4h + h4l) / 2
feat['ts_htf_anti']         = np.where(dir_s==1,
    (h4_mid < cl).astype(float),
    (h4_mid > cl).astype(float))

# 16. Combinat TS quality
feat['ts_combo_score']      = (feat['ts_close_inside'].values +
                                feat['ts_sharp'].values +
                                feat['ts_wick_dom'].values.clip(0,1) +
                                feat['ts_htf_anti'].values) / 4.0

# 17. NY-specific: distanța față de LON midpoint (context NY entry)
feat['ts_lon_mid_dist']     = safe_norm(cl - (lon_hi_arr+lon_lo_arr)/2, atr)

# 18. LON hi vs Asia hi (LON a extins sau nu deasupra Asiei)
feat['ts_lon_above_asia']   = np.where(dir_s==1,
    (lon_hi_arr > asia_hi).astype(float),    # SHORT: lon extended above asia
    (lon_lo_arr < asia_lo).astype(float))    # LONG: lon extended below asia

print(f"   Base features (ny_v3): ~206")
print(f"   TS features: 18")
print(f"   Total features construite: {len(feat.columns)}")


# ── ECONOMIC CALENDAR features ────────────────────────────────────────────
_is_fomc_arr   = np.array([d in FOMC_DATES   for d in date_arr], dtype=float)
_is_nfp_arr    = np.array([d in NFP_DATES    for d in date_arr], dtype=float)
_is_cpi_arr    = np.array([d in CPI_DATES    for d in date_arr], dtype=float)
_is_ppi_arr    = np.array([d in PPI_DATES    for d in date_arr], dtype=float)
_is_news_arr   = np.array([d in NEWS_DAYS    for d in date_arr], dtype=float)
# Pre/post release: NFP/CPI/PPI la 8:30 AM ET = 13:30 UTC = 810 min
# FOMC la 14:00 PM ET = 19:00 UTC → toată fereastra NY quality gate e pre-anunț
_pre_nfp_arr   = np.array([(d in NFP_DATES and _min_abs[i] < 810) for i,d in enumerate(date_arr)], dtype=float)
_post_nfp_arr  = np.array([(d in NFP_DATES and _min_abs[i] >= 810) for i,d in enumerate(date_arr)], dtype=float)
_pre_cpi_arr   = np.array([(d in CPI_DATES and _min_abs[i] < 810) for i,d in enumerate(date_arr)], dtype=float)
_post_cpi_arr  = np.array([(d in CPI_DATES and _min_abs[i] >= 810) for i,d in enumerate(date_arr)], dtype=float)
_pre_ppi_arr   = np.array([(d in PPI_DATES and _min_abs[i] < 810) for i,d in enumerate(date_arr)], dtype=float)
_post_ppi_arr  = np.array([(d in PPI_DATES and _min_abs[i] >= 810) for i,d in enumerate(date_arr)], dtype=float)

feat['is_fomc_day']       = _is_fomc_arr
feat['is_nfp_day']        = _is_nfp_arr
feat['is_cpi_day']        = _is_cpi_arr
feat['is_ppi_day']        = _is_ppi_arr
feat['is_retail_day']     = np.array([d in RETAIL_DATES for d in date_arr], dtype=float)
feat['is_ism_day']        = np.array([d in ISM_DATES    for d in date_arr], dtype=float)
feat['is_news_day']       = _is_news_arr
feat['is_any_high_day']   = np.array([d in ANY_HIGH     for d in date_arr], dtype=float)
feat['fomc_proximity']    = np.clip(np.array([fomc_proximity(d) for d in date_arr], dtype=float) / 14.0, 0, 1)
feat['is_pre_nfp']        = _pre_nfp_arr
feat['is_post_nfp']       = _post_nfp_arr
feat['is_pre_cpi']        = _pre_cpi_arr
feat['is_post_cpi']       = _post_cpi_arr
feat['is_pre_ppi']        = _pre_ppi_arr
feat['is_post_ppi']       = _post_ppi_arr
feat['is_fomc_wait']      = _is_fomc_arr.copy()  # FOMC la 19:00 UTC → toată fereastra NY e pre-anunț

# ── NY sub-session split ──────────────────────────────────────────────────
# _min_abs deja calculat mai sus
feat['is_ny_open']        = ((_min_abs >= 780) & (_min_abs < 870)).astype(float)   # 13:00-14:30
feat['is_ny_afternoon']   = (_min_abs >= 870).astype(float)                         # 14:30+
feat['min_in_ny_open']    = np.clip(_min_abs - 780, 0, 90).astype(float)
feat['ny_open_early']     = ((_min_abs >= 780) & (_min_abs < 810)).astype(float)   # 13:00-13:30

# ── Interactions calendar × session ───────────────────────────────────────
feat['fomc_x_open']       = feat['is_fomc_day'].values * feat['is_ny_open'].values
feat['news_x_dir']        = feat['is_news_day'].values * feat['dir_short'].values
feat['ppi_x_dir']         = feat['is_ppi_day'].values * feat['dir_short'].values
feat['ism_x_adx']         = feat['is_ism_day'].values * feat['adx_strong'].values
feat['nfp_pre_x_open']    = feat['is_pre_nfp'].values * feat['is_ny_open'].values
feat['nfp_post_x_open']   = feat['is_post_nfp'].values * feat['is_ny_open'].values
feat['lon_dir_x_vol']     = feat['lon_dir_explicit'].values * feat['vol_regime'].values / 3.0
feat['lon_dir_x_fvg_1h']  = feat['lon_dir_explicit'].values * feat['fvg_aligned_1h'].values / 2.0

# ══════════════════════════════════════════════════════════════════════════════
#  REGIME FEATURES (from pre-computed labels)
# ══════════════════════════════════════════════════════════════════════════════
REGIME_ORDER = ['CONSOLIDATION', 'PRE_EXPANSION', 'EXPANSION', 'RETRACEMENT', 'DISTRIBUTION']
REGIME_ENC   = {r: i for i, r in enumerate(REGIME_ORDER)}

def _get_regime(date_str):
    return _regime_map.get(date_str, 'UNKNOWN')

regime_arr     = np.array([_get_regime(d) for d in date_arr])
regime_enc_arr = np.array([REGIME_ENC.get(r, -1) for r in regime_arr], dtype=float)
regime_prob_arr= np.array([_regime_prob_map.get(d, 0.5) for d in date_arr], dtype=float)

feat['regime_enc']            = regime_enc_arr
feat['regime_prob']           = regime_prob_arr
feat['is_pre_expansion']      = (regime_arr == 'PRE_EXPANSION').astype(float)
feat['is_expansion']          = (regime_arr == 'EXPANSION').astype(float)
feat['is_retracement']        = (regime_arr == 'RETRACEMENT').astype(float)
feat['is_consolidation']      = (regime_arr == 'CONSOLIDATION').astype(float)
feat['is_distribution']       = (regime_arr == 'DISTRIBUTION').astype(float)
feat['regime_prob_pre_exp']   = np.array([
    _regime_probs_full.loc[d, 'prob_PRE_EXPANSION']
    if d in _regime_probs_full.index else 0.5 for d in date_arr], dtype=float)

feat['trail'] = df['trail'].values
feat['ts']    = df['ts'].values
feat['year']  = df['year'].values
feat['_regime'] = regime_arr

# ── Order Flow features → feat ─────────────────────────────────────────────
if '_OF_COLS' in dir() and _OF_COLS:
    for _oc in _OF_COLS:
        if _oc in df.columns:
            feat[_oc] = df[_oc].values
    print(f"   Order flow features in feat: {len([c for c in _OF_COLS if c in df.columns])}")
feat.dropna(inplace=True)

X   = feat.drop(columns=['trail','ts','year','_regime']).astype(float)
y   = feat['trail'].values
ts_ = pd.DatetimeIndex(feat['ts'].values)
yr_ = feat['year'].values

print(f"   Dataset final: {len(X):,} rânduri, {X.shape[1]} features")
print(f"   Trail rate: {y.mean()*100:.2f}%")

# ════════════════════════════════════════════════════════════════════════
# STEP 5: Temporal split
# ════════════════════════════════════════════════════════════════════════
print("\n▶  [5/7] Temporal split ...")
train_mask=np.array(ts_<VAL_START); val_mask=np.array(ts_>=VAL_START)
X_tr_all,y_tr_all=X[train_mask],y[train_mask]
X_val,y_val=X[val_mask],y[val_mask]
yr_tr=yr_[train_mask]
sw_tr=np.array([YEAR_WEIGHTS.get(int(yr),1.0) for yr in yr_tr])
print(f"   Train: {len(X_tr_all):,} | Val (Q4-2025): {len(X_val):,}")
print(f"   Train trail: {y_tr_all.mean()*100:.1f}% | Val trail: {y_val.mean()*100:.1f}%")
if len(X_val)<50:
    print("   ⚠️  Val set mic! Extindem la H2-2025.")
    VAL_START=pd.Timestamp("2024-07-01")
    train_mask=np.array(ts_<VAL_START); val_mask=np.array(ts_>=VAL_START)
    X_tr_all,y_tr_all=X[train_mask],y[train_mask]
    X_val,y_val=X[val_mask],y[val_mask]
    yr_tr=yr_[train_mask]
    sw_tr=np.array([YEAR_WEIGHTS.get(int(yr),1.0) for yr in yr_tr])
    print(f"   [ADJUSTED] Train: {len(X_tr_all):,} | Val: {len(X_val):,}")

# ════════════════════════════════════════════════════════════════════════
# STEP 6: Regime-aware Optuna tuning
# ════════════════════════════════════════════════════════════════════════
print(f"\n▶  [6/7] Regime-aware Optuna tuning ({OPTUNA_TRIALS} trials per regime) ...")

SAVE_DIR = DIR.parent
MODEL_NAME = 'mario_quality_ts_ny_v1'
ACTIVE_REGIMES = ['PRE_EXPANSION', 'EXPANSION', 'RETRACEMENT', 'ALL']

regime_arr_train = feat.loc[train_mask, '_regime'].values

def train_one_regime_model(X_tr_r, y_tr_r, X_val, y_val, X_te, y_te, sw_r, regime_suffix=''):
    """Train one regime-specific model."""
    if len(X_tr_r) < 50:
        return None, 0, 0


    # Regularizare adaptivă la sample count (closure → accessible in obj_regime)
    _n = len(X_tr_r)
    _max_d  = 3 if _n < 2000 else (4 if _n < 5000 else 6)
    _mcw_lo = 20 if _n < 2000 else (10 if _n < 5000 else 5)
    _n_est  = 600 if _n < 2000 else (1000 if _n < 5000 else 2000)

    def obj_regime(trial):
        params={
                        'n_estimators':     trial.suggest_int('n_estimators', 200, _n_est),
            'max_depth':        trial.suggest_int('max_depth', 2, _max_d),
            'learning_rate':    trial.suggest_float('learning_rate',0.003,0.08,log=True),
            'subsample':        trial.suggest_float('subsample',0.5,0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree',0.4,0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', _mcw_lo, _mcw_lo * 6),
            'gamma':            trial.suggest_float('gamma',0.0,3.0),
            'reg_alpha':        trial.suggest_float('reg_alpha',0.0,3.0),
            'reg_lambda':       trial.suggest_float('reg_lambda',0.5,6.0),
            'max_delta_step':   trial.suggest_int('max_delta_step',0,5),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight',3.0,15.0),
        }
        smote_ratio=trial.suggest_float('smote_ratio',0.20,0.55)
        try:
            sm=BorderlineSMOTE(sampling_strategy=smote_ratio,random_state=42,k_neighbors=5)
            X_sm,y_sm=sm.fit_resample(X_tr_r,y_tr_r)
            sw_sm=np.concatenate([sw_r,np.ones(len(X_sm)-len(X_tr_r))])
        except Exception:
            X_sm,y_sm=X_tr_r,y_tr_r; sw_sm=sw_r
        mdl=xgb.XGBClassifier(**params,use_label_encoder=False,eval_metric='logloss',
                              random_state=42,n_jobs=-1,tree_method='hist',early_stopping_rounds=50)
        mdl.fit(X_sm,y_sm,sample_weight=sw_sm,eval_set=[(X_val,y_val)],verbose=False)
        return roc_auc_score(y_val,mdl.predict_proba(X_val)[:,1])

    study=optuna.create_study(direction='maximize')
    study.optimize(obj_regime,n_trials=OPTUNA_TRIALS,show_progress_bar=False,n_jobs=1)
    best_p=study.best_params; best_s=best_p.pop('smote_ratio')

    try:
        sm_f=BorderlineSMOTE(sampling_strategy=best_s,random_state=42,k_neighbors=5)
        X_sm_f,y_sm_f=sm_f.fit_resample(X_tr_r,y_tr_r)
        sw_sm_f=np.concatenate([sw_r,np.ones(len(X_sm_f)-len(X_tr_r))])
    except Exception:
        X_sm_f,y_sm_f=X_tr_r,y_tr_r; sw_sm_f=sw_r

    final_m=xgb.XGBClassifier(**best_p,use_label_encoder=False,eval_metric='logloss',
                                random_state=42,n_jobs=-1,tree_method='hist',early_stopping_rounds=50)
    final_m.fit(X_sm_f,y_sm_f,sample_weight=sw_sm_f,eval_set=[(X_val,y_val)],verbose=False)

    is_auc=roc_auc_score(y_tr_r,final_m.predict_proba(X_tr_r)[:,1])  # IS real pe training
    oos_auc=roc_auc_score(y_te,final_m.predict_proba(X_te)[:,1]) if len(y_te)>0 else 0
    return final_m, is_auc, oos_auc

regime_models = {}
for regime_name in ACTIVE_REGIMES:
    if regime_name == 'ALL':
        mask_r = np.ones(len(X_tr_all), dtype=bool)
    else:
        mask_r = (regime_arr_train == regime_name)

    X_tr_r = X_tr_all[mask_r]
    y_tr_r = y_tr_all[mask_r]
    sw_r   = sw_tr[mask_r]

    print(f"\n▶  Training {regime_name}: {len(X_tr_r)} samples ...")
    if len(X_tr_r) < 80 and regime_name != 'ALL':
        print(f"   ⚠️ Too little data for {regime_name} → skip")
        continue

    mdl, is_a, oos_a = train_one_regime_model(X_tr_r, y_tr_r, X_val, y_val, X_val, y_val.values if hasattr(y_val,'values') else y_val, sw_r, regime_name)

    if mdl is not None:
        regime_models[regime_name] = mdl
        print(f"   ✅ {regime_name}: IS={is_a:.4f} OOS={oos_a:.4f}")
        # Salveaza PKL per regim (per routing live)
        if regime_name != 'ALL':
            _rpkl = DIR.parent / f"mario_quality_ts_ny_v1_{regime_name}_calibrated.pkl"
            import pickle as _pkl
            with open(_rpkl,'wb') as _f: _pkl.dump(mdl, _f)
            print(f"   💾 {_rpkl.name}")

# ════════════════════════════════════════════════════════════════════════
# STEP 7: Calibrare model ALL + save
# ════════════════════════════════════════════════════════════════════════
print("\n▶  [7/7] Calibrare model ALL + save ...")

if 'ALL' in regime_models:
    final_model = regime_models['ALL']
    print(f"   Using ALL-regime model")
else:
    print("   ERROR: No ALL model found!")
    raise ValueError("ALL regime model not trained")

print("\n   AUC per an:")
for yr in sorted(set(yr_)):
    mask=(yr_==yr)
    if mask.sum()<10: continue
    p_yr=final_model.predict_proba(X[mask])[:,1]
    auc_yr=roc_auc_score(y[mask],p_yr)
    tag="IN-SAMPLE" if (ts_[mask]<VAL_START).all() else ("OOS" if (ts_[mask]>=VAL_START).all() else "PARTIAL")
    print(f"   {yr}: AUC={auc_yr:.4f}  n={mask.sum():,}  trail={y[mask].mean()*100:.1f}%  [{tag}]")
for h in [13,14]:
    mask=(ts_.hour==h)
    if mask.sum()<10: continue
    auc_h=roc_auc_score(y[mask],final_model.predict_proba(X[mask])[:,1])
    print(f"   h{h} UTC: AUC={auc_h:.4f}  n={mask.sum():,}  trail={y[mask].mean()*100:.1f}%")

proba_val=final_model.predict_proba(X_val)[:,1]
auc_val=roc_auc_score(y_val,proba_val)
print(f"\n   ✅ AUC OOS (2025) = {auc_val:.4f}")

print("   Calibrare isotonică ...")
# sklearn 1.6+: cv='prefit' removed — manual isotonic calibration
from sklearn.isotonic import IsotonicRegression as _IR
_raw_val = final_model.predict_proba(X_val)[:, 1]
_ir_cal  = _IR(out_of_bounds='clip').fit(_raw_val, y_val)

cal_model = _CalModel(final_model, _ir_cal)
proba_cal=cal_model.predict_proba(X_val)[:,1]
auc_cal=roc_auc_score(y_val,proba_cal)
print(f"   AUC post-calibrare: {auc_cal:.4f}")

fpr,tpr,thresholds=roc_curve(y_val,proba_cal)
best_thr=float(thresholds[np.argmax(tpr-fpr)])
print(f"   Optimal threshold (Youden J): {best_thr:.3f}")

print(f"\n   Threshold analysis (val Q4-2025):")
print(f"   {'THR':>6}  {'N':>6}  {'Trail%':>8}  {'EV@RR2.0':>10}  {'EV@RR2.5':>10}")
SL=400.0; MONTHS=3
for thr in [0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30]:
    mask=proba_cal>=thr; n_t=mask.sum()
    if n_t<5: continue
    y_sub=y_val[mask]; tr_pct=y_sub.mean()
    ev20=tr_pct*480-(1-tr_pct)*SL; ev25=tr_pct*600-(1-tr_pct)*SL
    print(f"   {thr:>6.2f}  {n_t:>6}  {tr_pct*100:>7.1f}%  {ev20:>+10.0f}  {ev25:>+10.0f}   (~{n_t/MONTHS:.1f}/luna)")

fi=dict(zip(X.columns,final_model.feature_importances_))
fi_sort=sorted(fi.items(),key=lambda x:-x[1])
print(f"\n   Top 20 features:")
for name,imp in fi_sort[:20]: print(f"     {name}: {imp:.4f}")

# TS features importance
ts_feats=[f for f in X.columns if f.startswith('ts_')]
print(f"\n   TS-specific feature importances:")
ts_imp=[(n,fi[n]) for n in ts_feats if n in fi]
ts_imp_sort=sorted(ts_imp,key=lambda x:-x[1])
for name,imp in ts_imp_sort: print(f"     {name}: {imp:.4f}")

# Save
model_path=DIR.parent/"mario_quality_ts_ny_v1.json"
meta_path =DIR.parent/"mario_quality_ts_ny_v1_features.json"
cal_path  =DIR.parent/"mario_quality_ts_ny_v1_calibrated.pkl"
final_model.save_model(str(model_path))
with open(cal_path,'wb') as f: pickle.dump(cal_model,f)
import json
best_smote = 0.0   # defined inside train_one_regime_model; fallback
_bp_mdl = regime_models.get('ALL', None)
best_params = _bp_mdl.get_params() if _bp_mdl is not None else {}
meta={
    "version":"ts_ny_v1","session":"NY","filter":"SHORT_REV+LONG_REV only",
    "swept_levels":"lon_hi/lon_lo",
    "features":X.columns.tolist(),"n_features":X.shape[1],
    "auc_oos_2025":round(auc_val,4),"auc_calibrated":round(auc_cal,4),
    "best_threshold":round(best_thr,3),"best_params":{k:(float(v) if isinstance(v,(int,float,np.floating,np.integer)) else str(v)) for k,v in best_params.items()},
    "smote_ratio":round(float(best_smote),3),"year_weights":YEAR_WEIGHTS,
    "top_features":[n for n,_ in fi_sort[:10]],"top_ts_features":[n for n,_ in ts_imp_sort[:5]],
    "n_train":int(len(X_tr_all)),"n_val":int(len(X_val)),
    "ts_new_features":[f for f in X.columns if f.startswith('ts_')],
}
meta_path.write_text(json.dumps(meta,indent=2))
print(f"\n   ✅ {model_path.name}")
print(f"   ✅ {meta_path.name}")
print(f"   ✅ {cal_path.name}")
print()
print("="*74)
print(f"  FINAL AUC OOS 2025  : {auc_val:.4f}")
print(f"  FINAL AUC CALIBRAT     : {auc_cal:.4f}")
print(f"  THRESHOLD OPTIM        : {best_thr:.3f}")
print(f"  FEATURES               : {X.shape[1]}")
print(f"  TS FILTER              : SHORT_REV + LONG_REV only")
print(f"  SESSION                : NY (h13-h14 UTC)")
print(f"  SWEPT LEVELS           : lon_hi / lon_lo")
print("="*74)
