"""
train_sweep_unified.py — Unified Sweep Detection (LOM + NOM) per Regime
==========================================================================
Combină datele LOM și NOM (același pattern sweep+displacement, sesiuni diferite)
pentru a obține suficiente sample-uri pentru sub-modele per regim.

IS 2023-2024:
  LOM: ~348 setups | NOM: ~1358 setups → total ~1706
  Per regim: ~300-400 setups → viabil pentru XGBoost cu regularizare

Output:
  sweep_PRE_EXPANSION.pkl   ← activat când regime = PRE_EXPANSION
  sweep_EXPANSION.pkl
  sweep_RETRACEMENT.pkl
  sweep_ALL.pkl             ← fallback pentru regimuri rare

În checkers:
  lom_checker_v1.py / nom_checker_v1.py → load sweep_{regime}.pkl
  → score_sweep(feat_dict, regime) → float [0,1]
"""

import sqlite3, pickle, logging, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import BorderlineSMOTE
import xgboost as xgb
import optuna
from aladin_cal import _CalModel
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("SWEEP_UNIFIED")

DB   = Path(__file__).parent / "mario_trading.db"
OUT  = Path(__file__).parent  # salvăm în root Aladin

OPTUNA_TRIALS  = 60
TRAIN_YEARS    = [2023, 2024]
TEST_YEARS     = [2025, 2026]
YEAR_WEIGHTS   = {2023: 0.85, 2024: 1.00}
DECAY_HL       = 12   # half-life luni
TOP_N_FEATURES = 80
ACTIVE_REGIMES = ['PRE_EXPANSION', 'EXPANSION', 'RETRACEMENT', 'ALL']

# ── Calendar ─────────────────────────────────────────────────────────────────
_CAL_PATH = Path(__file__).parent / "data" / "economic_calendar.json"
try:
    _cal     = json.loads(_CAL_PATH.read_text())
    FOMC_DT  = set(_cal.get('fomc', []))
    NFP_DT   = set(_cal.get('nfp',  []))
    CPI_DT   = set(_cal.get('cpi',  []))
    PPI_DT   = set(_cal.get('ppi',  []))
    NEWS_DT  = FOMC_DT | NFP_DT | CPI_DT | PPI_DT
    log.info(f"Calendar: NFP={len(NFP_DT)}, FOMC={len(FOMC_DT)}")
except Exception as _e:
    log.warning(f"Calendar: {_e}")
    FOMC_DT=NFP_DT=CPI_DT=PPI_DT=NEWS_DT=set()

# ── Regime labels ─────────────────────────────────────────────────────────────
_REGIME_PATH = Path(__file__).parent / "data" / "regime_labels.parquet"
try:
    _rdf = pd.read_parquet(_REGIME_PATH)
    # For LOM use LON session, for NOM use NY session → merge both
    _rdf_lon = _rdf[_rdf['session']=='LON'][['date','regime','regime_prob']].rename(
        columns={'regime':'regime_LON','regime_prob':'prob_LON'})
    _rdf_ny  = _rdf[_rdf['session']=='NY'][['date','regime','regime_prob']].rename(
        columns={'regime':'regime_NY','regime_prob':'prob_NY'})
    _regime_map_lon = dict(zip(_rdf_lon['date'], _rdf_lon['regime_LON']))
    _regime_map_ny  = dict(zip(_rdf_ny['date'],  _rdf_ny['regime_NY']))
    log.info(f"Regime labels: LON={len(_regime_map_lon)}, NY={len(_regime_map_ny)} zile")
except Exception as _re:
    log.warning(f"Regime labels lipsă: {_re}")
    _regime_map_lon = {}; _regime_map_ny = {}

def _get_regime(date_str, session):
    if session == 'LON': return _regime_map_lon.get(date_str, 'UNKNOWN')
    return _regime_map_ny.get(date_str, 'UNKNOWN')

# ── Decay weighting ───────────────────────────────────────────────────────────
def compute_decay_weights(dates_series):
    lambda_ = np.log(2) / DECAY_HL
    today = pd.Timestamp.today()
    months_ago = ((today - pd.to_datetime(dates_series)).dt.days / 30.44).clip(0, 36)
    return np.exp(-lambda_ * months_ago).values

def sv(v, d=0.0):
    try: x=float(v); return x if np.isfinite(x) else d
    except: return d


# ════════════════════════════════════════════════════════════════════════════
# Extract LOM setups
# ════════════════════════════════════════════════════════════════════════════
LON_SESS_START_ET=400; LON_SESS_END_ET=700; PRE_LON_END_ET=359
ASIA_START_ET=0; ASIA_END_ET=359
LOM_MIN_SPIKE=5.0; LOM_MIN_DISP=4.0; LOM_TP=18.0; LOM_LABEL_WIN=60

def extract_lom(df, date_str):
    setups=[]
    pre_lon=df[df['hhmm']<=PRE_LON_END_ET]; lon_sess=df[df['hhmm'].between(LON_SESS_START_ET,LON_SESS_END_ET)]
    if len(pre_lon)<5 or len(lon_sess)<3: return setups
    pre_hi=float(pre_lon['high'].max()); pre_lo=float(pre_lon['low'].min()); pre_rng=pre_hi-pre_lo
    if pre_rng<3: return setups
    atr=float(df['atr_14'].replace(0,np.nan).dropna().iloc[-1]) if len(df)>0 else 10.0
    if atr<=0: atr=10.0
    asia_df=df[df['hhmm'].between(ASIA_START_ET,ASIA_END_ET)]
    asia_hi=float(asia_df['high'].max()) if len(asia_df)>0 else pre_hi
    asia_lo=float(asia_df['low'].min())  if len(asia_df)>0 else pre_lo
    asia_rng=asia_hi-asia_lo; asia_mid=(asia_hi+asia_lo)/2
    asia_close=float(asia_df['close'].iloc[-1]) if len(asia_df)>0 else asia_mid
    asia_dir=1 if asia_close>asia_mid else -1
    partial_thresh=pre_rng*0.50
    lon_reset=lon_sess.reset_index(drop=False)
    last_hhmm={'LONG':-999,'SHORT':-999}
    for i in range(1,len(lon_reset)-2):
        bar=lon_reset.iloc[i]; bar_hi=sv(bar['high']); bar_lo=sv(bar['low']); bar_hhmm=int(bar['hhmm'])
        for direction,spike_raw,is_valid in [
            ('SHORT',max(bar_hi-pre_hi,0),bar_hi-pre_hi>=LOM_MIN_SPIKE or (bar_hi>pre_hi and bar_hi-pre_hi>=partial_thresh)),
            ('LONG', max(pre_lo-bar_lo,0),pre_lo-bar_lo>=LOM_MIN_SPIKE or (bar_lo<pre_lo and pre_lo-bar_lo>=partial_thresh)),
        ]:
            if not is_valid or (bar_hhmm-last_hhmm[direction])<30: continue
            spike_mag=spike_raw
            after=lon_reset[lon_reset['hhmm'].between(bar_hhmm+1,bar_hhmm+45)]
            disp=None
            for _,ab in after.iterrows():
                ab_body=abs(sv(ab['close'])-sv(ab['open']))
                if direction=='SHORT' and sv(ab['close'])<sv(ab['open']) and ab_body>=LOM_MIN_DISP: disp=ab; break
                elif direction=='LONG' and sv(ab['close'])>sv(ab['open']) and ab_body>=LOM_MIN_DISP: disp=ab; break
            if disp is None: continue
            entry=sv(disp['close']); entry_hhmm=int(disp['hhmm']); dir_num=1 if direction=='LONG' else -1
            future=df[df['hhmm']>entry_hhmm].head(LOM_LABEL_WIN)
            if len(future)<3: continue
            if direction=='LONG': reached=float(future['high'].max())>=entry+LOM_TP; mfwd=float(future['high'].max()-entry)
            else: reached=float(future['low'].min())<=entry-LOM_TP; mfwd=float(entry-future['low'].min())
            label=1 if reached else 0
            r0=df.iloc[-1]; sbr=max(sv(bar['high']-bar['low']),0.01)
            if direction=='SHORT':
                ts_ci=1 if sv(bar['close'])<=pre_hi else 0
                wick=(sv(bar['high'])-max(sv(bar['close']),sv(bar['open'])))/atr
                ts_rej=(sv(bar['high'])-sv(bar['close']))/spike_mag if spike_mag>0 else 0
            else:
                ts_ci=1 if sv(bar['close'])>=pre_lo else 0
                wick=(min(sv(bar['close']),sv(bar['open']))-sv(bar['low']))/atr
                ts_rej=(sv(bar['close'])-sv(bar['low']))/spike_mag if spike_mag>0 else 0
            wick_pct=wick*atr/sbr; swc=1 if wick_pct>0.5 else 0; sda=spike_mag/atr
            sq=ts_ci*0.4+swc*0.3+(1 if sda>1.5 else 0)*0.2+0.1
            h4_mid=(sv(r0['h4_hi'])+sv(r0['h4_lo']))/2 if sv(r0['h4_hi'])>0 else 0
            h4_bias=1 if entry<h4_mid else (-1 if h4_mid>0 else 0)
            lw_hi=sv(r0['lw_hi']); lw_lo=sv(r0['lw_lo']); lw_rng=lw_hi-lw_lo
            weekly_prem=(entry-lw_lo)/lw_rng if lw_rng>0 else 0.5
            regime=_get_regime(date_str,'LON')
            feat={
                'session_enc':0,'spike_mag_atr':spike_mag/atr,'pre_rng_atr':pre_rng/atr,
                'ts_close_inside':ts_ci,'ts_rejection_str':ts_rej,'sweep_wick_atr':wick,
                'sweep_depth_atr':sda,'deep_sweep':1 if sda>1.5 else 0,'sweep_quality':sq,
                'disp_body_atr':abs(sv(disp['close'])-sv(disp['open']))/atr,
                'h4_bias':h4_bias,'h4_bias_aligned':1 if h4_bias==dir_num else 0,
                'weekly_premium_pct':weekly_prem,
                'weekly_prem_aligned':1 if (direction=='SHORT' and weekly_prem>0.5) or (direction=='LONG' and weekly_prem<0.5) else 0,
                'lw_range_atr':lw_rng/atr if atr>0 else 0,
                'dist_lw_hi':abs(entry-lw_hi)/atr,'dist_lw_lo':abs(entry-lw_lo)/atr,
                'dist_asia_hi_atr':abs(entry-asia_hi)/atr if asia_hi>0 else 0,
                'dist_asia_lo_atr':abs(entry-asia_lo)/atr if asia_lo>0 else 0,
                'asia_range_atr':asia_rng/atr,'asia_dir':float(asia_dir),
                'asia_dir_aligned':1 if asia_dir==dir_num else 0,
                'adx':sv(r0['adx_14']),'hurst':sv(r0['hurst'],0.5),
                'garch_vol':sv(r0['garch_vol']),'rvol':sv(r0['rvol'],1.0),
                'fisher_transform':sv(r0['fisher_transform']),
                'acf_lag1':sv(r0['acf_lag1']),'acf_lag5':sv(r0['acf_lag5']),
                'kalman_smooth':sv(r0['kalman_smooth']),'kalman_noise':sv(r0['kalman_noise']),
                'is_nfp_day':1 if date_str in NFP_DT else 0,
                'is_fomc_day':1 if date_str in FOMC_DT else 0,
                'is_news_day':1 if date_str in NEWS_DT else 0,
                'direction_enc':1 if direction=='SHORT' else 0,
                'day_of_week':sv(r0['day_of_week']),'month':sv(r0['month']),
                'is_thursday':1 if int(sv(r0['day_of_week']))==3 else 0,
                'is_friday':1 if int(sv(r0['day_of_week']))==4 else 0,
                'regime_enc':{'PRE_EXPANSION':1,'EXPANSION':2,'RETRACEMENT':3,'CONSOLIDATION':0,'DISTRIBUTION':4}.get(regime,-1),
                'is_pre_expansion':1 if regime=='PRE_EXPANSION' else 0,
                'is_expansion':1 if regime=='EXPANSION' else 0,
                'is_retracement':1 if regime=='RETRACEMENT' else 0,
                'dir_x_adx':dir_num*sv(r0['adx_14']),'dir_x_hurst':dir_num*sv(r0['hurst'],0.5),
                'vol_x_sweep':sv(r0['garch_vol'])*sq,'h4_x_weekly':float(1 if h4_bias==dir_num else 0)*float(1 if (direction=='SHORT' and weekly_prem>0.5) or (direction=='LONG' and weekly_prem<0.5) else 0),
                '_label':label,'_session':'LON','_date':date_str,'_regime':regime,
            }
            setups.append(feat); last_hhmm[direction]=bar_hhmm; break
    return setups


# ════════════════════════════════════════════════════════════════════════════
# Extract NOM setups
# ════════════════════════════════════════════════════════════════════════════
NY_SESS_START_ET=900; NY_SESS_END_ET=1300; PRE_NY_END_ET=859
LON_START_ET=400; LON_END_ET=630
NOM_MIN_SPIKE=5.0; NOM_MIN_DISP=4.0; NOM_TP=24.0; NOM_LABEL_WIN=60

def extract_nom(df, date_str):
    setups=[]
    pre_ny=df[df['hhmm']<=PRE_NY_END_ET]; ny_sess=df[df['hhmm'].between(NY_SESS_START_ET,NY_SESS_END_ET)]
    london=df[df['hhmm'].between(LON_START_ET,LON_END_ET)]
    if len(pre_ny)<20 or len(ny_sess)<5: return setups
    pre_hi=float(pre_ny['high'].max()); pre_lo=float(pre_ny['low'].min()); pre_rng=pre_hi-pre_lo
    if pre_rng<5: return setups
    atr=float(df['atr_14'].replace(0,np.nan).dropna().iloc[-1]) if len(df)>0 else 10.0
    if atr<=0: atr=10.0
    if len(london)>0:
        lon_hi=float(london['high'].max()); lon_lo=float(london['low'].min())
        lon_rng=lon_hi-lon_lo; lon_mid=(lon_hi+lon_lo)/2; lon_close=float(london['close'].iloc[-1])
    else:
        lon_hi=pre_hi; lon_lo=pre_lo; lon_rng=pre_rng; lon_mid=(pre_hi+pre_lo)/2; lon_close=lon_mid
    lon_dir=1 if lon_close>lon_mid else -1
    asia_df=df[df['hhmm'].between(0,359)]
    if len(asia_df)>0:
        asia_hi=float(asia_df['high'].max()); asia_lo=float(asia_df['low'].min())
        asia_close=float(asia_df['close'].iloc[-1]); asia_mid=(asia_hi+asia_lo)/2; asia_dir=1 if asia_close>asia_mid else -1
    else:
        asia_hi=pre_hi; asia_lo=pre_lo; asia_dir=0
    partial_thresh=pre_rng*0.50
    lon_reset=ny_sess.reset_index(drop=False); last_hhmm={'LONG':-999,'SHORT':-999}
    for i in range(1,len(lon_reset)-2):
        bar=lon_reset.iloc[i]; bar_hi=sv(bar['high']); bar_lo=sv(bar['low']); bar_hhmm=int(bar['hhmm'])
        for direction,spike_raw,is_valid in [
            ('SHORT',max(bar_hi-pre_hi,0),bar_hi-pre_hi>=NOM_MIN_SPIKE or (bar_hi>pre_hi and bar_hi-pre_hi>=partial_thresh)),
            ('LONG', max(pre_lo-bar_lo,0),pre_lo-bar_lo>=NOM_MIN_SPIKE or (bar_lo<pre_lo and pre_lo-bar_lo>=partial_thresh)),
        ]:
            if not is_valid or (bar_hhmm-last_hhmm[direction])<30: continue
            spike_mag=spike_raw
            after=lon_reset[lon_reset['hhmm'].between(bar_hhmm+1,bar_hhmm+45)]
            disp=None
            for _,ab in after.iterrows():
                ab_body=abs(sv(ab['close'])-sv(ab['open']))
                if direction=='SHORT' and sv(ab['close'])<sv(ab['open']) and ab_body>=NOM_MIN_DISP: disp=ab; break
                elif direction=='LONG' and sv(ab['close'])>sv(ab['open']) and ab_body>=NOM_MIN_DISP: disp=ab; break
            if disp is None: continue
            entry=sv(disp['close']); entry_hhmm=int(disp['hhmm']); dir_num=1 if direction=='LONG' else -1
            future=df[df['hhmm']>entry_hhmm].head(NOM_LABEL_WIN)
            if len(future)<3: continue
            if direction=='LONG': reached=float(future['high'].max())>=entry+NOM_TP; mfwd=float(future['high'].max()-entry)
            else: reached=float(future['low'].min())<=entry-NOM_TP; mfwd=float(entry-future['low'].min())
            label=1 if reached else 0
            r0=df.iloc[-1]; sbr=max(sv(bar['high']-bar['low']),0.01)
            if direction=='SHORT':
                ts_ci=1 if sv(bar['close'])<=pre_hi else 0
                wick=(sv(bar['high'])-max(sv(bar['close']),sv(bar['open'])))/atr
                ts_rej=(sv(bar['high'])-sv(bar['close']))/spike_mag if spike_mag>0 else 0
            else:
                ts_ci=1 if sv(bar['close'])>=pre_lo else 0
                wick=(min(sv(bar['close']),sv(bar['open']))-sv(bar['low']))/atr
                ts_rej=(sv(bar['close'])-sv(bar['low']))/spike_mag if spike_mag>0 else 0
            wick_pct=wick*atr/sbr; swc=1 if wick_pct>0.5 else 0; sda=spike_mag/atr
            sq=ts_ci*0.4+swc*0.3+(1 if sda>1.5 else 0)*0.2+0.1
            h4_mid=(sv(r0['h4_hi'])+sv(r0['h4_lo']))/2 if sv(r0['h4_hi'])>0 else 0
            h4_bias=1 if entry<h4_mid else (-1 if h4_mid>0 else 0)
            lw_hi=sv(r0['lw_hi']); lw_lo=sv(r0['lw_lo']); lw_rng=lw_hi-lw_lo
            weekly_prem=(entry-lw_lo)/lw_rng if lw_rng>0 else 0.5
            regime=_get_regime(date_str,'NY')
            triple_aligned=1 if (asia_dir==dir_num and lon_dir!=dir_num) else 0
            feat={
                'session_enc':1,'spike_mag_atr':spike_mag/atr,'pre_rng_atr':pre_rng/atr,
                'ts_close_inside':ts_ci,'ts_rejection_str':ts_rej,'sweep_wick_atr':wick,
                'sweep_depth_atr':sda,'deep_sweep':1 if sda>1.5 else 0,'sweep_quality':sq,
                'disp_body_atr':abs(sv(disp['close'])-sv(disp['open']))/atr,
                'h4_bias':h4_bias,'h4_bias_aligned':1 if h4_bias==dir_num else 0,
                'weekly_premium_pct':weekly_prem,
                'weekly_prem_aligned':1 if (direction=='SHORT' and weekly_prem>0.5) or (direction=='LONG' and weekly_prem<0.5) else 0,
                'lw_range_atr':lw_rng/atr if atr>0 else 0,
                'dist_lw_hi':abs(entry-lw_hi)/atr,'dist_lw_lo':abs(entry-lw_lo)/atr,
                'lon_range_atr':lon_rng/atr,'lon_dir':float(lon_dir),
                'lon_dir_aligned':1 if lon_dir==dir_num else 0,
                'dist_asia_hi_atr':abs(entry-asia_hi)/atr if len(asia_df)>0 else 0,
                'dist_asia_lo_atr':abs(entry-asia_lo)/atr if len(asia_df)>0 else 0,
                'asia_dir':float(asia_dir),'asia_dir_aligned':1 if asia_dir==dir_num else 0,
                'triple_sess_aligned':float(triple_aligned),
                'adx':sv(r0['adx_14']),'hurst':sv(r0['hurst'],0.5),
                'garch_vol':sv(r0['garch_vol']),'rvol':sv(r0['rvol'],1.0),
                'fisher_transform':sv(r0['fisher_transform']),
                'acf_lag1':sv(r0['acf_lag1']),'acf_lag5':sv(r0['acf_lag5']),
                'kalman_smooth':sv(r0['kalman_smooth']),'kalman_noise':sv(r0['kalman_noise']),
                'is_nfp_day':1 if date_str in NFP_DT else 0,
                'is_fomc_day':1 if date_str in FOMC_DT else 0,
                'is_news_day':1 if date_str in NEWS_DT else 0,
                'is_pre_nfp':1 if (date_str in NFP_DT and entry_hhmm<830) else 0,
                'is_post_nfp':1 if (date_str in NFP_DT and entry_hhmm>=830) else 0,
                'direction_enc':1 if direction=='SHORT' else 0,
                'day_of_week':sv(r0['day_of_week']),'month':sv(r0['month']),
                'is_thursday':1 if int(sv(r0['day_of_week']))==3 else 0,
                'is_friday':1 if int(sv(r0['day_of_week']))==4 else 0,
                'regime_enc':{'PRE_EXPANSION':1,'EXPANSION':2,'RETRACEMENT':3,'CONSOLIDATION':0,'DISTRIBUTION':4}.get(regime,-1),
                'is_pre_expansion':1 if regime=='PRE_EXPANSION' else 0,
                'is_expansion':1 if regime=='EXPANSION' else 0,
                'is_retracement':1 if regime=='RETRACEMENT' else 0,
                'dir_x_adx':dir_num*sv(r0['adx_14']),'dir_x_hurst':dir_num*sv(r0['hurst'],0.5),
                'vol_x_sweep':sv(r0['garch_vol'])*sq,'h4_x_weekly':float(1 if h4_bias==dir_num else 0)*float(1 if (direction=='SHORT' and weekly_prem>0.5) or (direction=='LONG' and weekly_prem<0.5) else 0),
                '_label':label,'_session':'NY','_date':date_str,'_regime':regime,
            }
            setups.append(feat); last_hhmm[direction]=bar_hhmm; break
    return setups


# ════════════════════════════════════════════════════════════════════════════
# Build dataset
# ════════════════════════════════════════════════════════════════════════════
def load_day(conn, date_str):
    df=pd.read_sql(f"""
        SELECT timestamp,open,high,low,close,volume,atr_14,
               adx_14,hurst,garch_vol,rvol,fisher_transform,acf_lag1,acf_lag5,
               kalman_smooth,kalman_noise,
               bar_delta,cum_delta,fvg_up,fvg_down,has_displacement,
               bar_buy_vol,bar_sell_vol,stacked_bull,stacked_bear,
               h4_hi,h4_lo,h1_hi,h1_lo,lw_hi,lw_lo,p_hi,p_lo,
               true_open,asia_hi,asia_lo,lon_hi,lon_lo,
               day_of_week,month
        FROM market_data WHERE date='{date_str}' ORDER BY timestamp""", conn)
    if len(df)<30: return None
    df['ts']=pd.to_datetime(df['timestamp'])
    df['hhmm']=df['ts'].dt.hour*100+df['ts'].dt.minute
    return df

def build_dataset(years):
    conn=sqlite3.connect(f'file:{DB}?mode=ro',uri=True,timeout=60)
    days=pd.read_sql(f"""SELECT DISTINCT date FROM market_data
        WHERE year IN ({','.join(map(str,years))}) AND day_of_week BETWEEN 1 AND 5
        ORDER BY date""",conn)['date'].tolist()
    all_setups=[]
    for d in days:
        df=load_day(conn,d)
        if df is None: continue
        all_setups.extend(extract_lom(df,d))
        all_setups.extend(extract_nom(df,d))
    conn.close()
    log.info(f"  {years}: {len(days)} zile → {len(all_setups)} setups (LOM+NOM)")
    return pd.DataFrame(all_setups)


# ════════════════════════════════════════════════════════════════════════════
# Training per regime
# ════════════════════════════════════════════════════════════════════════════
def train_regime_model(X_tr, y_tr, X_val, y_val, X_te, y_te, sw_tr, label='ALL'):
    """Optuna + SMOTE + CalibratedClassifierCV pentru un subset de regim."""
    if len(X_tr) < 60 or y_tr.sum() < 10:
        log.warning(f"  {label}: prea puțin data ({len(X_tr)} samples, {y_tr.sum()} positives) → skip")
        return None, 0, 0

    def objective(trial):
        params={
            'n_estimators':     trial.suggest_int('n_estimators',150,800),
            'max_depth':        trial.suggest_int('max_depth',2,4),
            'learning_rate':    trial.suggest_float('learning_rate',0.005,0.06,log=True),
            'subsample':        trial.suggest_float('subsample',0.5,0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree',0.4,0.80),
            'min_child_weight': trial.suggest_int('min_child_weight',15,60),
            'gamma':            trial.suggest_float('gamma',0.5,6.0),
            'reg_alpha':        trial.suggest_float('reg_alpha',0.5,6.0),
            'reg_lambda':       trial.suggest_float('reg_lambda',2.0,8.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight',3.0,12.0),
        }
        smote_r=trial.suggest_float('smote_ratio',0.10,0.40)
        try:
            sm=BorderlineSMOTE(sampling_strategy=smote_r,random_state=42,k_neighbors=min(5,y_tr.sum()-1))
            Xs,ys=sm.fit_resample(X_tr,y_tr)
            sws=np.concatenate([sw_tr,np.ones(len(Xs)-len(X_tr))])
        except: Xs,ys,sws=X_tr,y_tr,sw_tr
        m=xgb.XGBClassifier(**params,use_label_encoder=False,eval_metric='logloss',
                             random_state=42,n_jobs=-1,tree_method='hist',early_stopping_rounds=25)
        m.fit(Xs,ys,sample_weight=sws,eval_set=[(X_val,y_val)],verbose=False)
        if y_val.sum()==0 or y_val.sum()==len(y_val): return 0.5
        return roc_auc_score(y_val,m.predict_proba(X_val)[:,1])

    study=optuna.create_study(direction='maximize')
    study.optimize(objective,n_trials=OPTUNA_TRIALS,show_progress_bar=False,n_jobs=1)
    bp=study.best_params; smote_best=bp.pop('smote_ratio')
    log.info(f"  {label}: best val AUC={study.best_value:.4f}")

    try:
        sm=BorderlineSMOTE(sampling_strategy=smote_best,random_state=42,k_neighbors=min(5,y_tr.sum()-1))
        Xs,ys=sm.fit_resample(X_tr,y_tr)
        sws=np.concatenate([sw_tr,np.ones(len(Xs)-len(X_tr))])
    except: Xs,ys,sws=X_tr,y_tr,sw_tr

    base=xgb.XGBClassifier(**bp,use_label_encoder=False,eval_metric='logloss',
                            random_state=42,n_jobs=-1,tree_method='hist')
    base.fit(Xs,ys,sample_weight=sws,verbose=False)
    # sklearn 1.6+: cv='prefit' removed — manual isotonic calibration
    from sklearn.isotonic import IsotonicRegression as _IR
    _raw_val = base.predict_proba(X_val)[:, 1]
    _ir_cal  = _IR(out_of_bounds='clip').fit(_raw_val, y_val)

    cal = _CalModel(base, _ir_cal)

    is_auc =roc_auc_score(y_tr,cal.predict_proba(X_tr)[:,1])
    oos_auc=roc_auc_score(y_te,cal.predict_proba(X_te)[:,1]) if len(y_te)>20 and y_te.sum()>0 else 0
    log.info(f"  {label}: IS={is_auc:.4f} OOS={oos_auc:.4f} ({len(X_tr)} IS samples)")
    return cal, is_auc, oos_auc


def train_and_save():
    log.info("="*60)
    log.info("SWEEP UNIFIED — LOM+NOM per Regime")
    log.info("="*60)

    log.info(f"IS ({TRAIN_YEARS})...")
    df_tr=build_dataset(TRAIN_YEARS)
    log.info(f"OOS ({TEST_YEARS})...")
    df_te=build_dataset(TEST_YEARS)


    # ── Synthetic Order Flow features ─────────────────────────────────────
    _OF_PATH = Path(__file__).parent / "data" / "orderflow_features.parquet"
    if _OF_PATH.exists():
        _of = __import__('pandas').read_parquet(_OF_PATH)
        _of['date'] = _of['date'].astype(str)
        _OF_COLS_OF = [c for c in _of.columns if c not in ['session_id','date','session_type',
                      'session_open','session_close','session_high','session_low','total_vol']]
        _of_m = _of[['date','session_type'] + _OF_COLS_OF].rename(
            columns={'date':'_date','session_type':'_session'})
        df_tr = df_tr.merge(_of_m, on=['_date','_session'], how='left')
        df_te = df_te.merge(_of_m, on=['_date','_session'], how='left')
        for _c in _OF_COLS_OF:
            df_tr[_c] = df_tr[_c].fillna(0.0)
            df_te[_c] = df_te[_c].fillna(0.0)
        log.info(f"   Order flow: {len(_OF_COLS_OF)} features merged (LON+NY)")
    else:
        _OF_COLS_OF = []
    meta_cols=[c for c in df_tr.columns if c.startswith('_')]
    feature_cols=[c for c in df_tr.columns if c not in meta_cols]

    log.info(f"\nIS: {len(df_tr)} setups | LOM={len(df_tr[df_tr['_session']=='LON'])} NOM={len(df_tr[df_tr['_session']=='NY'])}")
    log.info(f"OOS: {len(df_te)} setups")
    log.info(f"Regimuri IS: {df_tr['_regime'].value_counts().to_dict()}")

    X_tr=df_tr[feature_cols].fillna(0); y_tr=df_tr['_label']
    X_te=df_te[feature_cols].fillna(0).reindex(columns=feature_cols,fill_value=0); y_te=df_te['_label']

    # Decay weights
    sw_tr=compute_decay_weights(df_tr['_date'])
    yr_weights=np.array([YEAR_WEIGHTS.get(int(d[:4]),1.0) for d in df_tr['_date']])
    sw_tr=sw_tr*yr_weights  # combine exponential decay × year weights

    # Feature selection (preliminary model on ALL data)
    log.info(f"\n▶  Feature selection (top {TOP_N_FEATURES}) ...")
    neg,pos=(y_tr==0).sum(),(y_tr==1).sum(); spw=neg/max(pos,1)
    _pre=xgb.XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,
        subsample=0.7,colsample_bytree=0.6,min_child_weight=20,gamma=1.5,
        reg_alpha=2.0,reg_lambda=4.0,scale_pos_weight=spw,
        random_state=42,n_jobs=-1,use_label_encoder=False,eval_metric='logloss',verbosity=0)
    _pre.fit(X_tr,y_tr,sample_weight=sw_tr,verbose=False)
    _imp=pd.Series(_pre.feature_importances_,index=feature_cols).sort_values(ascending=False)
    selected=_imp.head(TOP_N_FEATURES).index.tolist()
    log.info(f"   Top5: {selected[:5]}")
    X_tr=X_tr[selected]; X_te=X_te.reindex(columns=selected,fill_value=0)

    # Val split (temporal 80/20)
    val_cut=int(len(X_tr)*0.80)
    X_val=X_tr.iloc[val_cut:]; y_val=y_tr.iloc[val_cut:]
    X_tr2=X_tr.iloc[:val_cut]; y_tr2=y_tr.iloc[:val_cut]; sw2=sw_tr[:val_cut]

    regimes_is=df_tr['_regime'].values

    for regime_name in ACTIVE_REGIMES:
        log.info(f"\n{'='*40}\nRegim: {regime_name}\n{'='*40}")
        if regime_name=='ALL':
            mask=np.ones(len(X_tr2),dtype=bool)
        else:
            mask=(regimes_is[:val_cut]==regime_name)

        if mask.sum()<60:
            log.warning(f"  {regime_name}: {mask.sum()} samples IS → skip")
            continue

        model,is_auc,oos_auc=train_regime_model(
            X_tr2[mask],y_tr2.values[mask],X_val,y_val.values,X_te,y_te.values,sw2[mask],label=regime_name
        )
        if model is None: continue

        pkg={'model':model,'features':selected,'regime':regime_name,
             'is_auc':round(is_auc,4),'oos_auc':round(oos_auc,4),
             'n_features':len(selected),'train_years':TRAIN_YEARS,
             'version':'sweep_unified_v1','n_samples_is':int(mask.sum())}
        out_path=OUT/f'sweep_{regime_name}.pkl'
        with open(out_path,'wb') as f: pickle.dump(pkg,f)
        log.info(f"  ✅ Salvat: {out_path.name}")

        # WR thresholds
        for thr in [0.55,0.60,0.65,0.70]:
            te_proba=model.predict_proba(X_te)[:,1]
            mask_t=te_proba>=thr
            if mask_t.sum()>10:
                log.info(f"    WR@{thr}: {float(y_te.values[mask_t].mean()):.1%} ({mask_t.sum()} setups)")

    log.info("\n✅ SWEEP UNIFIED training complet.")


if __name__=="__main__":
    train_and_save()
