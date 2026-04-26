"""
train_v6_pre_expansion.py
━━━━━━━━━━━━━━━━━━━━━━━━
Antrenează DOAR mario_quality_v6_PRE_EXPANSION_calibrated.pkl.
Nu suprascrie nimic altceva (ALL, EXPANSION, RETRACEMENT, json, etc.).

GAP_PENALTY = 2.5  →  obj = val_auc - 2.5 * max(0, is_auc - val_auc - 0.06)
Dacă IS-OOS gap > 0.20 sau val_logloss > 0.693 → REJECT (nu salvează PKL).
"""
from __future__ import annotations
import sys, sqlite3, json as _json, warnings, pathlib, pickle
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from aladin_cal import _CalModel

DIR      = pathlib.Path(__file__).parent
DB_PATH  = DIR / "mario_trading.db"
CSV_PATH = DIR / "backtest" / "backtest_open_sessions_trades.csv"

OPTUNA_TRIALS  = int(__import__('os').environ.get('OPTUNA_TRIALS_OVERRIDE', '40'))
CLIP_VAL       = 10.0
IS_START       = pd.Timestamp("2023-01-01")
VAL_START      = pd.Timestamp("2025-01-01")
YEAR_WEIGHTS   = {2023: 0.85, 2024: 1.00}
GAP_PENALTY    = 2.5
TARGET_REGIME  = 'PRE_EXPANSION'
OUT_PKL        = DIR / f"mario_quality_v6_{TARGET_REGIME}_calibrated.pkl"

MAX_GAP        = 0.20   # IS-OOS gap max acceptat
MAX_LL         = 0.693  # val_logloss max acceptat (= mai rău ca random)

print("=" * 70)
print(f"train_v6_pre_expansion.py  GAP_PENALTY={GAP_PENALTY}")
print(f"Target: {OUT_PKL.name}")
print("=" * 70)

# ── Regime labels ────────────────────────────────────────────────────────────
_rl_path = DIR / "data" / "regime_labels.parquet"
_rl = pd.read_parquet(_rl_path)
_lon_rl = _rl[_rl['session'] == 'LON']
_regime_map      = dict(zip(_lon_rl['date'].astype(str), _lon_rl['regime']))
_regime_prob_map = dict(zip(_lon_rl['date'].astype(str), _lon_rl['regime_prob']))
_regime_probs_full = _lon_rl.set_index('date')
print(f"Regime labels: {len(_regime_map)} zile LON  |  PRE_EXPANSION: {sum(1 for v in _regime_map.values() if v=='PRE_EXPANSION')}")

# ── ICT / MTF helpers (copiate din train_quality_v6.py) ──────────────────────
def compute_ict_on_tf(df_tf, lookback=20):
    H = df_tf['high'].values.astype(float)
    L = df_tf['low'].values.astype(float)
    C = df_tf['close'].values.astype(float)
    O = df_tf['open'].values.astype(float)
    A = np.maximum(df_tf['atr'].values.astype(float), 1.0)
    n = len(H)
    bull_top = np.zeros(n); bull_bot = np.zeros(n)
    bear_top = np.zeros(n); bear_bot = np.zeros(n)
    for i in range(2, n):
        if H[i-2] < L[i] and (L[i] - H[i-2]) > 0.5:
            bull_top[i] = L[i]; bull_bot[i] = H[i-2]
        if L[i-2] > H[i] and (L[i-2] - H[i]) > 0.5:
            bear_top[i] = L[i-2]; bear_bot[i] = H[i]
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
            if i-j<=lookback and c<min(bot,O[j])-a*0.05:
                if abs(c-top)/a<0.8 or abs(c-bot)/a<0.8: breaker_s[i]=1.0
        for top,bot,j in bear_obs[-20:]:
            if i-j<=lookback and c>max(top,O[j])+a*0.05:
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
        'breaker_b':breaker_b,'breaker_s':breaker_s,'rejection':rejection},
        index=df_tf.index)

def compute_mtf_features(conn, setup_dates):
    min_d=min(setup_dates); max_d=max(setup_dates)
    ws=(pd.Timestamp(min_d)-pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    print(f"   Loading 1-min: {ws} → {max_d}")
    df1m=pd.read_sql(f"SELECT timestamp,open,high,low,close,atr_14 FROM market_data WHERE timestamp>='{ws} 00:00:00' AND timestamp<='{max_d} 23:59:59' ORDER BY timestamp",conn)
    df1m['ts']=pd.to_datetime(df1m['timestamp']); df1m=df1m.set_index('ts')
    df1m.rename(columns={'atr_14':'atr'},inplace=True); df1m['atr']=df1m['atr'].ffill().fillna(9.0)
    print(f"   1-min bars: {len(df1m):,}")
    all_feat=pd.DataFrame(index=df1m.index)
    for tfl,tfr,lb in [('5m','5min',25),('15m','15min',20),('1h','1h',20),('4h','4h',15)]:
        print(f"   ICT {tfl}...")
        dft=df1m.resample(tfr,label='left',closed='left').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),atr=('atr','last')).dropna(subset=['open'])
        dft['atr']=dft['atr'].ffill().fillna(9.0)
        ict=compute_ict_on_tf(dft,lookback=lb)
        ict_ff=ict.reindex(df1m.index,method='ffill')
        for col in ict.columns: all_feat[f'{col}_{tfl}']=ict_ff[col]
    all_feat=all_feat.fillna(0.0)
    all_feat['ts_str']=all_feat.index.strftime('%Y-%m-%d %H:%M:%S')
    return all_feat

# ── STEP 1: Load backtest CSV ────────────────────────────────────────────────
print("\n[1/6] Load backtest CSV ...")
df_csv = pd.read_csv(CSV_PATH)
df_lon = df_csv[df_csv['session']=='LON'].copy()
df_lon['ts']          = pd.to_datetime(df_lon['timestamp'])
df_lon['trail']       = (df_lon['exit_reason']=='TRAIL').astype(int)
df_lon['hour_utc']    = df_lon['ts'].dt.hour
df_lon['day_of_week'] = df_lon['ts'].dt.dayofweek
df_lon['dir_short']   = (df_lon['direction']=='SHORT').astype(float)
df_lon['ts_str']      = df_lon['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_lon['date_str']    = df_lon['ts'].dt.strftime('%Y-%m-%d')
df_lon['year']        = df_lon['ts'].dt.year
df_lon = df_lon[df_lon['ts'] >= IS_START].copy()

# Filter to PRE_EXPANSION only
df_lon['_regime_tmp'] = df_lon['date_str'].map(_regime_map)
df_pre = df_lon[df_lon['_regime_tmp']==TARGET_REGIME].copy()
print(f"LON total: {len(df_lon):,}  |  PRE_EXPANSION: {len(df_pre):,}")
print(f"PRE_EXP trail rate: {df_pre['trail'].mean()*100:.1f}%")
IS_mask_pre  = df_pre['ts'] <  VAL_START
VAL_mask_pre = df_pre['ts'] >= VAL_START
print(f"IS (2023-2024): {IS_mask_pre.sum():,}  |  VAL (2025+): {VAL_mask_pre.sum():,}")

# ── STEP 2: JOIN market_data ─────────────────────────────────────────────────
print("\n[2/6] JOIN market_data ...")
conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=60)
DB_COLS = ['timestamp','open','high','low','close','volume','atr_14','asia_hi','asia_lo',
    'p_hi','p_lo','true_open','h4_hi','h4_lo','h1_hi','h1_lo','poc_level','vah','val',
    'dist_poc','inside_va','has_displacement','fvg_up','fvg_down','is_smt_bearish','is_smt_bullish',
    'hurst','adx_14','garch_vol','sample_entropy','fisher_transform','acf_lag1','acf_lag5',
    'vwap','dist_vwap','bar_delta','cum_delta','bar_buy_vol','bar_sell_vol',
    'absorption_score','stacked_bull','stacked_bear','body_size','lw_hi','lw_lo','lm_hi','lm_lo',
    'dist_pdh','dist_pdl','fft_cycle','kalman_smooth','kalman_noise',
    'of_doi','of_bilateral_abs','of_big_balance']
cols_str=', '.join(DB_COLS)
CHUNK=5000; db_parts=[]; ts_list=df_pre['ts_str'].tolist()
for i in range(0,len(ts_list),CHUNK):
    chunk=ts_list[i:i+CHUNK]; ph=','.join(['?']*len(chunk))
    db_parts.append(pd.read_sql(f"SELECT {cols_str} FROM market_data WHERE timestamp IN ({ph})",conn,params=chunk))
db=pd.concat(db_parts,ignore_index=True)
db['ts_str']=db['timestamp']
print(f"DB rows: {len(db):,} / {len(df_pre):,} ({len(db)/len(df_pre)*100:.1f}%)")
df=df_pre.merge(db.drop(columns=['timestamp']),on='ts_str',how='inner')
print(f"Post-merge: {len(df):,}")

# MTF features
print("\n[2b/6] MTF ICT features ...")
setup_dates=sorted(df_pre['date_str'].unique())
mtf=compute_mtf_features(conn,setup_dates)
df=df.merge(mtf.drop_duplicates('ts_str')[['ts_str']+[c for c in mtf.columns if c!='ts_str']],on='ts_str',how='left')
for c in [c for c in mtf.columns if c!='ts_str']: df[c]=df[c].fillna(0.0)

# Orderflow
_of_path=DIR/"data"/"orderflow_features.parquet"
_OF_COLS=[]
if _of_path.exists():
    _of=pd.read_parquet(_of_path); _of=_of[_of['session_type']=='LON'].copy()
    _of['date']=_of['date'].astype(str)
    _OF_COLS=[c for c in _of.columns if c not in ['session_id','date','session_type','session_open','session_close','session_high','session_low','total_vol']]
    df=df.merge(_of[['date']+_OF_COLS].rename(columns={'date':'date_str'}),on='date_str',how='left')
    for c in _OF_COLS: df[c]=df[c].fillna(0.0)
    print(f"OF features: {len(_OF_COLS)}")

# VIX proxy + daily
print("\n[2c/6] VIX proxy ...")
all_dates_sql="','".join(setup_dates)
daily_reg=pd.read_sql(f"""SELECT date(timestamp) as date,(MAX(high)-MIN(low)) as daily_range,
    AVG(atr_14) as avg_atr,MAX(high) as day_hi,MIN(low) as day_lo,
    AVG(adx_14) as avg_adx,AVG(hurst) as avg_hurst
    FROM market_data WHERE date(timestamp)>=date('{setup_dates[0]}','-30 days')
    AND date(timestamp)<='{setup_dates[-1]}' GROUP BY date(timestamp) ORDER BY date""",conn)
conn.close()
daily_reg['date']=daily_reg['date'].astype(str)
daily_reg['date_dt']=pd.to_datetime(daily_reg['date'])
daily_reg=daily_reg.sort_values('date').reset_index(drop=True)
daily_reg['avg_atr']=daily_reg['avg_atr'].ffill().fillna(9.0)
daily_reg['daily_range']=daily_reg['daily_range'].fillna(daily_reg['avg_atr']*2)
daily_reg['range_atr_ratio']=daily_reg['daily_range']/daily_reg['avg_atr'].clip(lower=1)
daily_reg['vix_proxy_5d']=daily_reg['range_atr_ratio'].rolling(5,min_periods=2).mean().shift(1)
daily_reg['vix_proxy_20d']=daily_reg['range_atr_ratio'].rolling(20,min_periods=5).mean().shift(1)
daily_reg['vol_regime']=(daily_reg['vix_proxy_5d']/daily_reg['vix_proxy_20d'].clip(lower=0.5)).clip(upper=3)
daily_reg['vol_high']=(daily_reg['vol_regime']>1.2).astype(float)
daily_reg['vol_low']=(daily_reg['vol_regime']<0.8).astype(float)
daily_reg['adx_10d_mean']=daily_reg['avg_adx'].rolling(10,min_periods=3).mean().shift(1)
daily_reg['hurst_20d_mean']=daily_reg['avg_hurst'].rolling(20,min_periods=5).mean().shift(1)
daily_reg['atr_5d']=daily_reg['avg_atr'].rolling(5,min_periods=2).mean().shift(1)
daily_reg['atr_10d']=daily_reg['avg_atr'].rolling(10,min_periods=3).mean().shift(1)
daily_reg['atr_trend']=(daily_reg['atr_5d']/daily_reg['atr_10d'].clip(lower=1)).clip(upper=3)
daily_reg=daily_reg.fillna(method='ffill').fillna(1.0)
daily_dict={r['date']:r for _,r in daily_reg.iterrows()}

# Weekly context
conn2=sqlite3.connect(f"file:{DB_PATH}?mode=ro",uri=True,timeout=60)
wk_df=pd.read_sql(f"SELECT date(timestamp) as date,MAX(high) as day_hi,MIN(low) as day_lo FROM market_data WHERE date(timestamp) IN ('{all_dates_sql}') GROUP BY date(timestamp)",conn2)
conn2.close()
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

df_lon_sorted=df_pre.sort_values('ts').copy()
df_lon_sorted['trail_roll5']=(df_lon_sorted['trail'].rolling(5,min_periods=1).mean().shift(1).fillna(0.5))
roll5_map=dict(zip(df_lon_sorted['ts_str'],df_lon_sorted['trail_roll5']))

# ── STEP 3: Feature engineering (identic cu v6) ──────────────────────────────
print("\n[3/6] Feature engineering ...")
def _dd(d,key,fb=0.0):
    r=wk_dict.get(d);
    if r is None: return fb
    v=r[key] if isinstance(r,dict) else getattr(r,key,fb)
    return float(v) if v is not None and pd.notna(v) else fb
def _dr(d,key,fb=1.0):
    r=daily_dict.get(d)
    if r is None: return fb
    v=r[key] if isinstance(r,dict) else getattr(r,key,fb)
    return float(v) if v is not None and pd.notna(v) else fb
def clip(x,c=CLIP_VAL): return np.clip(np.where(np.isfinite(x),x,0.0),-c,c)
def safe_norm(num,denom,c=CLIP_VAL): return clip(np.where(np.abs(denom)>0.01,num/denom,0.0),c)

cl=df['close'].values.astype(float); hi=df['high'].values.astype(float)
lo=df['low'].values.astype(float);   op=df['open'].values.astype(float)
vol=np.where(df['volume'].values>0,df['volume'].values,1).astype(float)
atr=np.where(df['atr_14'].values>0,df['atr_14'].values,9.0).astype(float)
asia_hi=np.where(df['asia_hi'].values>0,df['asia_hi'].values,cl)
asia_lo=np.where(df['asia_lo'].values>0,df['asia_lo'].values,cl)
p_hi_arr=np.where(df['p_hi'].values>0,df['p_hi'].values,cl)
p_lo_arr=np.where(df['p_lo'].values>0,df['p_lo'].values,cl)
true_open=np.where(df['true_open'].values>0,df['true_open'].values,cl)
h4h=np.where(df['h4_hi'].values>0,df['h4_hi'].values,cl); h4l=np.where(df['h4_lo'].values>0,df['h4_lo'].values,cl)
h1h=np.where(df['h1_hi'].values>0,df['h1_hi'].values,cl); h1l=np.where(df['h1_lo'].values>0,df['h1_lo'].values,cl)
vwap_arr=np.where(df['vwap'].values>0,df['vwap'].values,cl)
lw_hi_arr=np.where(df['lw_hi'].values>0,df['lw_hi'].values,cl)
lw_lo_arr=np.where(df['lw_lo'].values>0,df['lw_lo'].values,cl)
date_arr=df['date_str'].values
trade_dir=np.where(df['dir_short'].values==0,1.0,-1.0)

feat=pd.DataFrame()
feat['dir_short']=df['dir_short'].values; feat['hour_utc']=df['hour_utc'].values.astype(float)
feat['min_in_lon']=np.clip((df['hour_utc'].values-7)*60,0,180).astype(float)
feat['day_of_week']=df['day_of_week'].values.astype(float)
feat['month']=df['ts'].dt.month.values.astype(float)
feat['is_monday']=(df['day_of_week'].values==0).astype(float); feat['is_friday']=(df['day_of_week'].values==4).astype(float)
feat['year_norm']=(df['ts'].dt.year.values.astype(float)-2023.0)/2.0
feat['confidence']=df['confidence'].values.astype(float)
feat['atr_entry']=atr
feat['atr_vs_10d']=clip(df.groupby(df['ts'].dt.date)['atr_14'].transform('mean').values/np.where(atr>0,atr,1),3)
valid_asia=(asia_hi>0)&(asia_lo>0)&(asia_hi>asia_lo)
feat['dist_asia_hi_atr']=safe_norm(cl-asia_hi,atr); feat['dist_asia_lo_atr']=safe_norm(cl-asia_lo,atr)
feat['asia_range_atr']=clip(safe_norm(asia_hi-asia_lo,atr),20)
feat['swept_asia_hi']=((cl>asia_hi)&valid_asia).astype(float); feat['swept_asia_lo']=((cl<asia_lo)&valid_asia).astype(float)
feat['asia_midpoint']=safe_norm(cl-(asia_hi+asia_lo)/2,atr)
feat['dist_pdh_atr']=safe_norm(df['dist_pdh'].values,atr); feat['dist_pdl_atr']=safe_norm(df['dist_pdl'].values,atr)
feat['above_true_open']=(cl>true_open).astype(float); feat['dist_true_open']=safe_norm(cl-true_open,atr)
feat['h4_bias']=safe_norm((h4h+h4l)/2-cl,atr); feat['h1_bias']=safe_norm((h1h+h1l)/2-cl,atr)
feat['h4_h1_aligned']=(np.sign(feat['h4_bias'].values)==np.sign(feat['h1_bias'].values)).astype(float)
prev_wk_hi=np.array([_dd(d,'prev_wk_hi',cl[i]) for i,d in enumerate(date_arr)])
prev_wk_lo=np.array([_dd(d,'prev_wk_lo',cl[i]) for i,d in enumerate(date_arr)])
wk_hi_sf=np.array([_dd(d,'wk_hi_sofar',cl[i]) for i,d in enumerate(date_arr)])
wk_lo_sf=np.array([_dd(d,'wk_lo_sofar',cl[i]) for i,d in enumerate(date_arr)])
wk_range_sf=np.array([_dd(d,'wk_range_sofar',atr[i]) for i,d in enumerate(date_arr)])
mon_hi_arr=np.array([_dd(d,'mon_hi',0.0) for d in date_arr]); mon_lo_arr=np.array([_dd(d,'mon_lo',0.0) for d in date_arr])
valid_pw=(prev_wk_hi>prev_wk_lo)&(prev_wk_hi>0); pw_range=np.where(valid_pw,prev_wk_hi-prev_wk_lo,atr*10)
wk_prem_pct=np.where(valid_pw,(cl-prev_wk_lo)/pw_range-0.5,0.0)
feat['weekly_premium_pct']=clip(wk_prem_pct); feat['in_weekly_premium']=(wk_prem_pct>0.1).astype(float)
feat['in_weekly_discount']=(wk_prem_pct<-0.1).astype(float)
feat['weekly_prem_aligned']=np.where(trade_dir==1,feat['in_weekly_discount'].values,feat['in_weekly_premium'].values)
feat['dist_prev_wk_hi']=safe_norm(cl-prev_wk_hi,atr); feat['dist_prev_wk_lo']=safe_norm(cl-prev_wk_lo,atr)
feat['lw_range_atr']=clip(safe_norm(lw_hi_arr-lw_lo_arr,atr),20)
feat['dist_lw_hi']=safe_norm(cl-lw_hi_arr,atr); feat['dist_lw_lo']=safe_norm(cl-lw_lo_arr,atr)
feat['week_range_so_far']=clip(safe_norm(wk_range_sf,atr),20)
feat['week_hi_taken']=(cl>wk_hi_sf*0.998).astype(float); feat['week_lo_taken']=(cl<wk_lo_sf*1.002).astype(float)
mon_range_arr=np.where(mon_hi_arr>mon_lo_arr,mon_hi_arr-mon_lo_arr,0.0)
feat['monday_range_pt']=clip(safe_norm(mon_range_arr,atr),10); feat['monday_consol']=(mon_range_arr<atr).astype(float)
feat['is_tuesday']=(df['day_of_week'].values==1).astype(float); feat['is_wednesday']=(df['day_of_week'].values==2).astype(float)
feat['is_thursday']=(df['day_of_week'].values==3).astype(float)
feat['tuesday_rev_ctx']=feat['is_tuesday'].values*(feat['week_hi_taken'].values+feat['week_lo_taken'].values).clip(0,1)
feat['wednesday_rev_ctx']=feat['is_wednesday'].values*(np.abs(feat['weekly_premium_pct'].values.clip(-1,1))>0.3).astype(float)
feat['inside_va']=df['inside_va'].fillna(0).values.astype(float)
feat['dist_poc_atr']=safe_norm(df['dist_poc'].values,atr); feat['dist_vwap_atr']=safe_norm(df['dist_vwap'].values,atr)
feat['vah_dist']=safe_norm(cl-df['vah'].fillna(0).values.astype(float),atr)
feat['val_dist']=safe_norm(cl-df['val'].fillna(0).values.astype(float),atr)
feat['has_displacement']=df['has_displacement'].fillna(0).values.astype(float)
feat['fvg_up']=df['fvg_up'].fillna(0).values.astype(float); feat['fvg_down']=df['fvg_down'].fillna(0).values.astype(float)
feat['is_smt_bearish']=df['is_smt_bearish'].fillna(0).values.astype(float); feat['is_smt_bullish']=df['is_smt_bullish'].fillna(0).values.astype(float)
feat['hurst']=df['hurst'].fillna(0.5).values.astype(float); feat['adx_14']=df['adx_14'].fillna(20).values.astype(float)
feat['adx_strong']=(df['adx_14'].fillna(20).values>25).astype(float)
feat['acf_lag1']=df['acf_lag1'].fillna(0).values.astype(float); feat['acf_lag5']=df['acf_lag5'].fillna(0).values.astype(float)
feat['fisher_transform']=df['fisher_transform'].fillna(0).values.astype(float)
feat['fisher_extreme']=(np.abs(df['fisher_transform'].fillna(0).values)>2.0).astype(float)
feat['fft_cycle']=df['fft_cycle'].fillna(0).values.astype(float)
feat['kalman_smooth']=df['kalman_smooth'].fillna(0).values.astype(float); feat['kalman_noise']=df['kalman_noise'].fillna(0).values.astype(float)
garch_raw=df['garch_vol'].fillna(0).values.astype(float)
feat['garch_vol_atr']=clip(np.where(atr>0,garch_raw*cl/atr,1.0),5)
feat['sample_entropy']=df['sample_entropy'].fillna(2.0).values.astype(float)
bar_delta=df['bar_delta'].fillna(0).values.astype(float)
feat['bar_delta_norm']=clip(bar_delta/np.maximum(vol,1),1); feat['cum_delta_norm']=clip(df['cum_delta'].fillna(0).values/np.maximum(vol,1),1)
feat['buy_sell_ratio']=clip(df['bar_buy_vol'].fillna(0).values/np.maximum(df['bar_sell_vol'].fillna(0).values,1),5)
feat['absorption_score']=df['absorption_score'].fillna(0).values.astype(float)
feat['stacked_bull']=df['stacked_bull'].fillna(0).values.astype(float); feat['stacked_bear']=df['stacked_bear'].fillna(0).values.astype(float)
feat['of_doi']=df['of_doi'].fillna(0).values.astype(float)
body=cl-op; wick_up=hi-np.maximum(cl,op); wick_down=np.minimum(cl,op)-lo
feat['body_bear']=(body<0).astype(float); feat['body_pct']=clip(np.abs(body)/np.maximum(hi-lo,0.01),2)
feat['sweep_wick_atr']=safe_norm(np.maximum(wick_up,wick_down),atr)
feat['dir_x_adx']=feat['dir_short'].values*feat['adx_14'].values/100.0
feat['dir_x_hurst']=feat['dir_short'].values*feat['hurst'].values
feat['confidence_x_adx']=feat['confidence'].values*feat['adx_strong'].values
feat['hour_x_dir']=feat['hour_utc'].values*feat['dir_short'].values
feat['year_x_adx']=feat['year_norm'].values*feat['adx_14'].values/100.0
feat['year_x_hurst']=feat['year_norm'].values*feat['hurst'].values
for tfl in ['5m','15m','1h','4h']:
    in_bull=df[f'in_bull_{tfl}'].values; in_bear=df[f'in_bear_{tfl}'].values
    dist_bull=df[f'dist_bull_{tfl}'].values; dist_bear=df[f'dist_bear_{tfl}'].values
    in_ifvg_b=df[f'in_ifvg_b_{tfl}'].values; in_ifvg_s=df[f'in_ifvg_s_{tfl}'].values
    brk_b=df[f'breaker_b_{tfl}'].values; brk_s=df[f'breaker_s_{tfl}'].values
    rej=df[f'rejection_{tfl}'].values
    feat[f'in_bull_fvg_{tfl}']=in_bull; feat[f'in_bear_fvg_{tfl}']=in_bear
    feat[f'dist_bull_fvg_{tfl}']=np.clip(dist_bull,0,9.9); feat[f'dist_bear_fvg_{tfl}']=np.clip(dist_bear,0,9.9)
    feat[f'fvg_aligned_{tfl}']=np.where(trade_dir==1,in_bull,in_bear)
    feat[f'in_ifvg_{tfl}']=np.maximum(in_ifvg_b,in_ifvg_s)
    feat[f'ifvg_aligned_{tfl}']=np.where(trade_dir==1,in_ifvg_s,in_ifvg_b)
    feat[f'breaker_aligned_{tfl}']=np.where(trade_dir==1,brk_b,brk_s)
    feat[f'rejection_{tfl}']=rej
feat['fvg_tf_confluence']=(feat['fvg_aligned_5m'].values+feat['fvg_aligned_15m'].values+feat['fvg_aligned_1h'].values+feat['fvg_aligned_4h'].values)
feat['htf_fvg_aligned']=np.maximum(feat['fvg_aligned_1h'].values,feat['fvg_aligned_4h'].values)
feat['ifvg_htf_aligned']=np.maximum(feat['ifvg_aligned_1h'].values,feat['ifvg_aligned_4h'].values)
feat['vix_proxy_5d']=np.array([_dr(d,'vix_proxy_5d',2.0) for d in date_arr])
feat['vix_proxy_20d']=np.array([_dr(d,'vix_proxy_20d',2.0) for d in date_arr])
feat['vol_regime']=np.array([_dr(d,'vol_regime',1.0) for d in date_arr])
feat['vol_high']=np.array([_dr(d,'vol_high',0.0) for d in date_arr]); feat['vol_low']=np.array([_dr(d,'vol_low',0.0) for d in date_arr])
feat['atr_trend']=np.array([_dr(d,'atr_trend',1.0) for d in date_arr])
feat['atr_expanding']=(feat['atr_trend'].values>1.15).astype(float); feat['atr_contracting']=(feat['atr_trend'].values<0.85).astype(float)
feat['vol_x_fvg_1h']=feat['vol_regime'].values*feat['fvg_aligned_1h'].values
feat['vol_x_htf_fvg']=feat['vol_regime'].values*feat['htf_fvg_aligned'].values
sweep_level=np.where(trade_dir==1,asia_lo,asia_hi)
dist_to_pdl=np.abs(sweep_level-p_lo_arr); dist_to_pdh=np.abs(sweep_level-p_hi_arr)
feat['equal_level_score']=clip(1.0-safe_norm(np.minimum(dist_to_pdl,dist_to_pdh),atr))
sweep_depth=np.where(trade_dir==1,sweep_level-cl,cl-sweep_level)
feat['sweep_depth_atr']=safe_norm(sweep_depth,atr); feat['deep_sweep']=(sweep_depth>atr*0.4).astype(float)
feat['shallow_sweep']=(sweep_depth<atr*0.1).astype(float); feat['sweep_wick_clean']=(feat['sweep_wick_atr'].values>0.4).astype(float)
feat['sweep_with_disp']=(feat['sweep_wick_clean'].values*feat['has_displacement'].values)
feat['sweep_quality_score']=(feat['equal_level_score'].values+feat['deep_sweep'].values+feat['sweep_wick_clean'].values+feat['fvg_aligned_15m'].values+feat['fvg_aligned_1h'].values).clip(0,5)/5.0
feat['rolling_5sess_wr']=np.array([roll5_map.get(ts,0.5) for ts in df['ts_str'].values])
feat['adx_10d_mean']=np.array([_dr(d,'adx_10d_mean',20.0) for d in date_arr])
feat['hurst_20d_mean']=np.array([_dr(d,'hurst_20d_mean',0.5) for d in date_arr])
feat['regime_trending']=(feat['adx_10d_mean'].values>22).astype(float)
feat['regime_hurst_trend']=(feat['hurst_20d_mean'].values>0.52).astype(float)
feat['recent_wr_high']=(feat['rolling_5sess_wr'].values>0.35).astype(float)
feat['recent_wr_low']=(feat['rolling_5sess_wr'].values<0.15).astype(float)
feat['regime_score']=(feat['regime_trending'].values+feat['regime_hurst_trend'].values+feat['vol_high'].values).clip(0,3)/3.0
feat['regime_x_htf_fvg']=feat['regime_score'].values*feat['htf_fvg_aligned'].values
feat['adx_x_sweep_quality']=feat['adx_10d_mean'].values/30.0*feat['sweep_quality_score'].values
REGIME_ORDER=['CONSOLIDATION','PRE_EXPANSION','EXPANSION','RETRACEMENT','DISTRIBUTION']
REGIME_ENC={r:i for i,r in enumerate(REGIME_ORDER)}
regime_arr=np.array([_regime_map.get(d,'UNKNOWN') for d in date_arr])
feat['regime_enc']=np.array([REGIME_ENC.get(r,-1) for r in regime_arr],dtype=float)
feat['regime_prob']=np.array([_regime_prob_map.get(d,0.5) for d in date_arr],dtype=float)
feat['is_pre_expansion']=(regime_arr=='PRE_EXPANSION').astype(float)
feat['is_expansion']=(regime_arr=='EXPANSION').astype(float)
feat['is_retracement']=(regime_arr=='RETRACEMENT').astype(float)
feat['is_consolidation']=(regime_arr=='CONSOLIDATION').astype(float)
feat['is_distribution']=(regime_arr=='DISTRIBUTION').astype(float)
feat['regime_prob_pre_exp']=np.array([_regime_probs_full.loc[d,'prob_PRE_EXPANSION'] if d in _regime_probs_full.index else 0.5 for d in date_arr],dtype=float)
if _OF_COLS:
    for c in _OF_COLS:
        if c in df.columns: feat[c]=df[c].values
feat.dropna(inplace=True)
X=feat.drop(columns=['trail','ts','year','_regime'],errors='ignore').astype(float) if 'trail' in feat.columns else feat.astype(float)
feat['trail']=df.loc[feat.index,'trail'].values if 'trail' not in feat.columns else feat['trail']
feat['ts']=df.loc[feat.index,'ts'].values if 'ts' not in feat.columns else feat['ts']
feat['year']=df.loc[feat.index,'year'].values if 'year' not in feat.columns else feat['year']
X=feat.drop(columns=['trail','ts','year'],errors='ignore').astype(float)
y=feat['trail'].values; ts_=pd.DatetimeIndex(feat['ts'].values); yr_=feat['year'].values
print(f"Dataset: {len(X):,} rows, {X.shape[1]} features  |  trail={y.mean()*100:.1f}%")

# ── STEP 4: Temporal split ────────────────────────────────────────────────────
print("\n[4/6] Temporal split ...")
train_mask=(ts_<VAL_START); val_mask=(ts_>=VAL_START)
X_tr,y_tr=X[train_mask],y[train_mask]; X_val,y_val=X[val_mask],y[val_mask]
yr_tr=yr_[train_mask]
sw_tr=np.array([YEAR_WEIGHTS.get(int(yr),1.0) for yr in yr_tr])
print(f"IS: {len(X_tr):,}  |  VAL: {len(X_val):,}")
print(f"IS trail: {y_tr.mean()*100:.1f}%  |  VAL trail: {y_val.mean()*100:.1f}%")

# ── STEP 5: Optuna cu GAP_PENALTY ────────────────────────────────────────────
print(f"\n[5/6] Optuna {OPTUNA_TRIALS} trials, GAP_PENALTY={GAP_PENALTY} ...")
_n=len(X_tr)
_max_d=3 if _n<2000 else (4 if _n<5000 else 6)
_mcw_lo=20 if _n<2000 else (10 if _n<5000 else 5)
_n_est=600 if _n<2000 else (1000 if _n<5000 else 2000)

def obj(trial):
    params={
        'n_estimators':     trial.suggest_int('n_estimators',200,_n_est),
        'max_depth':        trial.suggest_int('max_depth',2,_max_d),
        'learning_rate':    trial.suggest_float('learning_rate',0.003,0.08,log=True),
        'subsample':        trial.suggest_float('subsample',0.5,0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree',0.4,0.9),
        'min_child_weight': trial.suggest_int('min_child_weight',_mcw_lo,_mcw_lo*6),
        'gamma':            trial.suggest_float('gamma',0.0,3.0),
        'reg_alpha':        trial.suggest_float('reg_alpha',0.0,3.0),
        'reg_lambda':       trial.suggest_float('reg_lambda',0.5,6.0),
        'max_delta_step':   trial.suggest_int('max_delta_step',0,5),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight',3.0,15.0),
    }
    smote_r=trial.suggest_float('smote_ratio',0.20,0.55)
    try:
        from imblearn.over_sampling import BorderlineSMOTE
        sm=BorderlineSMOTE(sampling_strategy=smote_r,random_state=42,k_neighbors=5)
        Xs,ys=sm.fit_resample(X_tr,y_tr)
        sw_s=np.concatenate([sw_tr,np.ones(len(Xs)-len(X_tr))])
    except: Xs,ys,sw_s=X_tr,y_tr,sw_tr
    mdl=xgb.XGBClassifier(**params,use_label_encoder=False,eval_metric='logloss',
        random_state=42,n_jobs=-1,tree_method='hist',early_stopping_rounds=50)
    mdl.fit(Xs,ys,sample_weight=sw_s,eval_set=[(X_val,y_val)],verbose=False)
    val_auc=roc_auc_score(y_val,mdl.predict_proba(X_val)[:,1])
    is_auc =roc_auc_score(y_tr, mdl.predict_proba(X_tr)[:,1])   # IS real (non-SMOTE)
    gap=max(0,is_auc-val_auc-0.06)
    return val_auc - GAP_PENALTY*gap

study=optuna.create_study(direction='maximize')
study.optimize(obj,n_trials=OPTUNA_TRIALS,show_progress_bar=True,n_jobs=1)
best_p=study.best_params; best_smote=best_p.pop('smote_ratio')
print(f"Best trial: penalized_score={study.best_value:.4f}")

# Final model
try:
    from imblearn.over_sampling import BorderlineSMOTE
    sm_f=BorderlineSMOTE(sampling_strategy=best_smote,random_state=42,k_neighbors=5)
    Xsf,ysf=sm_f.fit_resample(X_tr,y_tr)
    sw_sf=np.concatenate([sw_tr,np.ones(len(Xsf)-len(X_tr))])
except: Xsf,ysf,sw_sf=X_tr,y_tr,sw_tr

final_m=xgb.XGBClassifier(**best_p,use_label_encoder=False,eval_metric='logloss',
    random_state=42,n_jobs=-1,tree_method='hist',early_stopping_rounds=50)
final_m.fit(Xsf,ysf,sample_weight=sw_sf,eval_set=[(X_val,y_val)],verbose=False)

is_auc  = roc_auc_score(y_tr, final_m.predict_proba(X_tr)[:,1])
oos_auc = roc_auc_score(y_val,final_m.predict_proba(X_val)[:,1])
gap     = is_auc - oos_auc
val_ll  = log_loss(y_val, final_m.predict_proba(X_val)[:,1])

print(f"\n{'='*50}")
print(f"  IS  AUC  : {is_auc:.4f}")
print(f"  OOS AUC  : {oos_auc:.4f}")
print(f"  GAP      : {gap:.4f}  (max tolerat: {MAX_GAP})")
print(f"  VAL LL   : {val_ll:.4f}  (max tolerat: {MAX_LL})")
print(f"{'='*50}")

# ── STEP 6: Accept / Reject ───────────────────────────────────────────────────
print(f"\n[6/6] Verdict ...")
REJECT = False
reasons = []
if gap > MAX_GAP:
    REJECT = True; reasons.append(f"GAP {gap:.4f} > {MAX_GAP}")
if val_ll > MAX_LL:
    REJECT = True; reasons.append(f"val_logloss {val_ll:.4f} > {MAX_LL} (worse than random)")
if oos_auc < 0.52:
    REJECT = True; reasons.append(f"OOS AUC {oos_auc:.4f} < 0.52 (no edge)")

if REJECT:
    print(f"\n❌  REJECTED — {' | '.join(reasons)}")
    print(f"    Fallback rămâne la mario_quality_v6_calibrated.pkl (ALL model)")
    print(f"    PKL NU a fost salvat.")
    sys.exit(0)

# Calibrare isotonică
_ir=IsotonicRegression(out_of_bounds='clip').fit(final_m.predict_proba(X_val)[:,1],y_val)
cal_m=_CalModel(final_m,_ir)
proba_cal=cal_m.predict_proba(X_val)[:,1]
auc_cal=roc_auc_score(y_val,proba_cal)
print(f"  AUC post-calibrare: {auc_cal:.4f}")

# Smoke test — verifică distribuția pe întreg val set (nu primele 5, care sunt majority class)
all_scores=cal_m.predict_proba(X_val)[:,1]
p5,p50,p95=np.percentile(all_scores,[5,50,95])
print(f"  Score distribution: p5={p5:.3f} | p50={p50:.3f} | p95={p95:.3f}")
# Colaps = p95 < 0.15 (modelul nu poate prezice nimic pozitiv) SAU p5 > 0.85 (totul e pozitiv)
collapsed=(p95 < 0.15 or p5 > 0.85)
if collapsed:
    print(f"❌  REJECTED — scores collapsed (p95={p95:.3f} < 0.15 sau p5={p5:.3f} > 0.85)")
    sys.exit(0)

with open(OUT_PKL,'wb') as f: pickle.dump(cal_m,f)
print(f"\n✅  ADĂUGAT → {OUT_PKL.name}")
print(f"   IS={is_auc:.4f}  OOS={oos_auc:.4f}  GAP={gap:.4f}  LL={val_ll:.4f}")
print(f"   quality_gate_live.py îl va detecta automat la următorul restart.")
