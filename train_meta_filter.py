"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  META-FILTER MODEL — Quality filter pe deasupra semnalelor mario_bot.json   ║
║                                                                              ║
║  STRATEGIA:                                                                  ║
║  1. Rulăm mario_bot.json (v1) pe datele de training (2015-2022)             ║
║  2. Simulăm fiecare trade predictat → eticheta = TRAIL_STOP (câștig real)  ║
║     vs non-TRAIL (BE + SL)                                                  ║
║  3. Antrenăm un XGBoost binar pe features suplimentare (order flow,         ║
║     volume, momentum) pentru a prezice calitatea tranzacției                 ║
║  4. OOS: aplicăm filtrul v1 + meta-filter → WR țintă 25-35%               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import sqlite3, json, pathlib, sys, warnings, time, gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from collections import defaultdict
warnings.filterwarnings("ignore")

DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(DIR))

PATH_DB       = DIR / "mario_trading.db"
V1_MODEL      = DIR / "mario_bot.json"
V1_FEATURES   = DIR / "mario_features.json"
OUT_MODEL     = DIR / "mario_meta_filter.json"
OUT_META      = DIR / "mario_meta_filter_features.json"

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_YEARS = list(range(2015, 2022))
VAL_YEARS   = [2022]
TEST_YEARS  = list(range(2023, 2026))
ALL_YEARS   = TRAIN_YEARS + VAL_YEARS + TEST_YEARS

V1_CONF_THRESHOLD = 0.55     # pragul v1 pentru semnale candidat (mai larg ca default)
CONTEXT_BARS  = 300

# RM params (identice cu backtest)
ATR_SL_MULT   = 0.8; MIN_SL_PTS = 6.0; MAX_SL_PTS = 14.0
BE_TRIGGER_R  = 0.5; TRAIL_START_R = 2.0; TRAIL_ATR_MULT = 0.5
TICK_VALUE    = 20.0

NY_S, NY_E = 16.0, 17.5

DIRECTION    = {1:'SHORT',2:'LONG',3:'SHORT',4:'LONG'}
SIGNAL_CLASSES = [1,2,3,4]

# Meta-features: adăugăm la cele 23 v1 features
META_FEATURES = [
    # Order Flow (din DB)
    'bar_delta_n', 'cum_delta_n', 'dom_ratio', 'of_big_balance',
    'of_doi', 'absorption_score', 'stacked_bull', 'stacked_bear',
    # Momentum & Volatility
    'momentum_5', 'momentum_15', 'slope_h1', 'h4_momentum',
    'vol_spike', 'rvol', 'adx_14', 'hurst', 'garch_vol',
    'fisher_transform', 'acf_lag1',
    # VP + SMC
    'dist_vah_atr', 'dist_val_atr', 'dist_poc_atr', 'dist_vwap_atr',
    'above_vah', 'below_val', 'inside_va',
    'fvg_up', 'fvg_down', 'has_displacement',
    # Candle quality
    'body_dir', 'wick_bias', 'range_atr_ratio', 'body_pct', 'close_strength',
    'sharp_reversal', 'rejection_candle',
    # Sweep quality
    'swept_above', 'swept_below', 'bars_since_sweep', 'reclaimed_after_sweep',
    'upper_wick_atr', 'lower_wick_atr',
]


# ── HELPERS ───────────────────────────────────────────────────────────────────
def safe_div(a, b, fill=0.0):
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.asarray(a,float) / np.where(np.asarray(b,float)!=0, np.asarray(b,float), np.nan)
    return np.where(np.isfinite(r), r, fill)
def clip5(x): return np.clip(np.asarray(x,float),-5,5)


def compute_v1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features originale v1 (23)"""
    ts = df['timestamp']; _td = (ts.dt.hour + ts.dt.minute/60.0).values
    cl = df['close'].values; hi = df['high'].values; lo = df['low'].values
    atr = np.where(df['atr_14'].values>0, df['atr_14'].values, 9.0).astype(float)
    pdh = pd.Series(df['p_hi'].values).ffill().values
    pdl = pd.Series(df['p_lo'].values).ffill().values
    asia_hi=pd.Series(df['asia_hi'].values).ffill().values
    asia_lo=pd.Series(df['asia_lo'].values).ffill().values
    lon_hi=pd.Series(df['lon_hi'].values).ffill().values
    lon_lo=pd.Series(df['lon_lo'].values).ffill().values
    h4_hi=pd.Series(df['h4_hi'].values).ffill().values
    h4_lo=pd.Series(df['h4_lo'].values).ffill().values
    lw_hi=pd.Series(df['lw_hi'].values).ffill().values
    lw_lo=pd.Series(df['lw_lo'].values).ffill().values
    h1_hi=pd.Series(df['h1_hi'].values).ffill().values
    h1_lo=pd.Series(df['h1_lo'].values).ffill().values

    out = {}
    out['dist_pdh'] = clip5(safe_div(pdh-cl,atr)).astype(np.float32)
    out['dist_pdl'] = clip5(safe_div(cl-pdl,atr)).astype(np.float32)
    out['atr_percentile'] = pd.Series(atr).rolling(100,min_periods=10).rank(pct=True).fillna(0.5).values.astype(np.float32)
    out['in_london_or']    = ((_td>=9.0)&(_td<9.5)).astype(np.int8)
    out['in_london_kz']    = ((_td>=9.5)&(_td<11.0)).astype(np.int8)
    out['in_london_close'] = ((_td>=11.0)&(_td<12.0)).astype(np.int8)
    out['in_pre_ny']       = ((_td>=15.0)&(_td<15.5)).astype(np.int8)
    out['in_ny_or']        = ((_td>=15.5)&(_td<16.0)).astype(np.int8)
    out['in_pre_ny_macro'] = ((_td>=15.75)&(_td<16.167)).astype(np.int8)
    out['in_ny_kz_core']   = ((_td>=16.0)&(_td<16.833)).astype(np.int8)
    out['in_ny_macro_1']   = ((_td>=16.833)&(_td<17.167)).astype(np.int8)
    out['in_ny_macro_2']   = ((_td>=17.167)&(_td<17.5)).astype(np.int8)
    out['in_any_macro']    = ((out['in_pre_ny_macro']==1)|(out['in_ny_macro_1']==1)|(out['in_ny_macro_2']==1)).astype(np.int8)
    out['mins_since_lon_open'] = np.clip((_td-9.0)*60,0,180).astype(np.float32)
    out['mins_since_ny_open']  = np.clip((_td-15.5)*60,0,120).astype(np.float32)

    dates = ts.dt.date
    dh = df.groupby(dates)['high'].cummax().values
    dl = df.groupby(dates)['low'].cummin().values
    out['range_day_score'] = (1.0/np.maximum(safe_div(dh-dl,atr,1.0),1.0)).astype(np.float32)

    out['liq_above_count'] = ((asia_hi>cl)+(pdh>cl)+(h4_hi>cl)+(lw_hi>cl)).astype(np.int8)
    out['liq_below_count'] = ((asia_lo<cl)+(pdl<cl)+(h4_lo<cl)+(lw_lo<cl)).astype(np.int8)
    out['broke_asia_hi'] = (hi>asia_hi).astype(np.int8); out['broke_asia_lo'] = (lo<asia_lo).astype(np.int8)
    out['broke_pdh'] = (hi>pdh).astype(np.int8);        out['broke_pdl'] = (lo<pdl).astype(np.int8)

    sw_up=((hi>asia_hi)+(hi>pdh)+(hi>lon_hi)+(hi>h4_hi)+(hi>h1_hi)).astype(int)
    sw_dn=((lo<asia_lo)+(lo<pdl)+(lo<lon_lo)+(lo<h4_lo)+(lo<h1_lo)).astype(int)
    out['sweep_direction'] = np.where(sw_up>sw_dn,1,np.where(sw_dn>sw_up,-1,0)).astype(np.int8)

    return pd.DataFrame(out, index=df.index)


def compute_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features suplimentare pentru meta-filter"""
    cl = df['close'].values.astype(float); hi = df['high'].values.astype(float)
    lo = df['low'].values.astype(float);   op = df['open'].values.astype(float)
    vol = df['volume'].values.astype(float)
    atr = np.where(df['atr_14'].values>0, df['atr_14'].values, 9.0).astype(float)
    vah = pd.Series(df['vah'].values).ffill().values.astype(float)
    val = pd.Series(df['val'].values).ffill().values.astype(float)
    poc = pd.Series(df['poc_level'].values).ffill().values.astype(float)
    a_hi=pd.Series(df['asia_hi'].values).ffill().values.astype(float)
    a_lo=pd.Series(df['asia_lo'].values).ffill().values.astype(float)
    pdh =pd.Series(df['p_hi'].values).ffill().values.astype(float)
    pdl =pd.Series(df['p_lo'].values).ffill().values.astype(float)
    h4h =pd.Series(df['h4_hi'].values).ffill().values.astype(float)
    h4l =pd.Series(df['h4_lo'].values).ffill().values.astype(float)

    out = {}
    # Order Flow
    bar_d = df['bar_delta'].fillna(0).values.astype(float)
    cum_d = df['cum_delta'].fillna(0).values.astype(float)
    vol_atr = vol*atr+1
    out['bar_delta_n']  = clip5(safe_div(bar_d, vol_atr)).astype(np.float32)
    cum_abs20 = pd.Series(np.abs(cum_d)).rolling(20).mean().fillna(1).values
    out['cum_delta_n']  = clip5(safe_div(cum_d, cum_abs20)).astype(np.float32)
    out['dom_ratio']    = df['dom_ratio'].fillna(1.0).values.astype(np.float32)
    out['of_big_balance']= df['of_big_balance'].fillna(0).values.astype(np.float32)
    out['of_doi']       = df['of_doi'].fillna(0).values.astype(np.float32)
    out['absorption_score']= df['absorption_score'].fillna(0).values.astype(np.float32)
    out['stacked_bull'] = df['stacked_bull'].fillna(0).values.astype(np.int8)
    out['stacked_bear'] = df['stacked_bear'].fillna(0).values.astype(np.int8)

    # Momentum
    cl_s = pd.Series(cl)
    out['momentum_5']  = clip5(safe_div((cl_s-cl_s.shift(5)).values, atr)).astype(np.float32)
    out['momentum_15'] = clip5(safe_div((cl_s-cl_s.shift(15)).values, atr)).astype(np.float32)
    out['slope_h1']    = clip5(safe_div((cl_s-cl_s.shift(60)).values, atr)).astype(np.float32)
    out['h4_momentum'] = clip5(safe_div((cl_s-cl_s.shift(240)).values,atr)).astype(np.float32)

    # Volatility
    atr_s = pd.Series(atr)
    out['vol_spike']   = (atr_s>atr_s.rolling(20).mean()*1.5).astype(np.int8).values
    out['rvol']        = df['rvol'].fillna(1.0).values.astype(np.float32)
    out['adx_14']      = df['adx_14'].fillna(20.0).values.astype(np.float32)
    out['hurst']       = df['hurst'].fillna(0.5).values.astype(np.float32)
    out['garch_vol']   = df['garch_vol'].fillna(0.0).values.astype(np.float32)
    out['fisher_transform'] = df['fisher_transform'].fillna(0.0).values.astype(np.float32)
    out['acf_lag1']    = df['acf_lag1'].fillna(0.0).values.astype(np.float32)

    # VP + SMC
    out['dist_vah_atr'] = clip5(safe_div(vah-cl,atr)).astype(np.float32)
    out['dist_val_atr'] = clip5(safe_div(cl-val,atr)).astype(np.float32)
    out['dist_poc_atr'] = clip5(safe_div(cl-poc,atr)).astype(np.float32)
    dv = df['dist_vwap'].fillna(0).values.astype(float)
    out['dist_vwap_atr']= clip5(safe_div(dv,atr)).astype(np.float32)
    out['above_vah']    = (cl>vah).astype(np.int8)
    out['below_val']    = (cl<val).astype(np.int8)
    out['inside_va']    = df['inside_va'].fillna(0).values.astype(np.int8)
    out['fvg_up']       = df['fvg_up'].fillna(0).values.astype(np.int8)
    out['fvg_down']     = df['fvg_down'].fillna(0).values.astype(np.int8)
    out['has_displacement']= df['has_displacement'].fillna(0).values.astype(np.int8)

    # Candle
    bar_range = np.maximum(hi-lo,0.01)
    uw = hi-np.maximum(cl,op); lw = np.minimum(cl,op)-lo
    body = np.abs(cl-op)
    out['body_dir']        = np.sign(cl-op).astype(np.int8)
    out['wick_bias']       = clip5(safe_div(uw-lw,bar_range)).astype(np.float32)
    out['range_atr_ratio'] = clip5(safe_div(bar_range,atr)).astype(np.float32)
    out['body_pct']        = clip5(safe_div(body,bar_range)).astype(np.float32)
    out['close_strength']  = safe_div(cl-lo,bar_range).astype(np.float32)
    out['sharp_reversal']  = ((out['range_atr_ratio']>1.5)&(out['close_strength']>0.7)).astype(np.int8)
    out['rejection_candle']= ((clip5(safe_div(np.maximum(uw,lw),bar_range))>0.6)&(body<0.4*bar_range)).astype(np.int8)
    out['upper_wick_atr']  = clip5(safe_div(uw,atr)).astype(np.float32)
    out['lower_wick_atr']  = clip5(safe_div(lw,atr)).astype(np.float32)

    # Sweep quality
    sw_up=((hi>a_hi)+(hi>pdh)+(hi>pd.Series(df['lon_hi'].values).ffill().values)+(hi>h4h)).astype(int)
    sw_dn=((lo<a_lo)+(lo<pdl)+(lo<pd.Series(df['lon_lo'].values).ffill().values)+(lo<h4l)).astype(int)
    out['swept_above'] = (sw_up>0).astype(np.int8)
    out['swept_below'] = (sw_dn>0).astype(np.int8)
    any_sw=((sw_up+sw_dn)>0).astype(int)
    bs=[]; cnt=99
    for v in any_sw: cnt=0 if v else cnt+1; bs.append(min(cnt,99))
    out['bars_since_sweep'] = np.array(bs,dtype=np.float32)
    sw_up10=pd.Series(sw_up).rolling(10,min_periods=1).max().values
    sw_dn10=pd.Series(sw_dn).rolling(10,min_periods=1).max().values
    out['reclaimed_after_sweep'] = (
        ((sw_up10>0)&(cl<a_hi)&(cl<pdh))|((sw_dn10>0)&(cl>a_lo)&(cl>pdl))
    ).astype(np.int8)

    return pd.DataFrame(out, index=df.index)


def simulate_trade_rm(df_raw, idx, direction, atr_val):
    """Simulare RM. Returneaza (r_mult, exit_reason)"""
    sl_pts = float(np.clip(ATR_SL_MULT*atr_val, MIN_SL_PTS, MAX_SL_PTS))
    ep = float(df_raw['close'].iloc[idx])
    ts_e = pd.to_datetime(df_raw['timestamp'].iloc[idx])
    end_td = NY_E

    sl = ep - sl_pts if direction=='LONG' else ep + sl_pts
    be_trig = ep + (BE_TRIGGER_R*sl_pts if direction=='LONG' else -BE_TRIGGER_R*sl_pts)
    tr_trig  = ep + (TRAIL_START_R*sl_pts if direction=='LONG' else -TRAIL_START_R*sl_pts)
    tr_sl = sl; be_hit = False; tr_hit = False

    for i in range(idx+1, min(idx+200, len(df_raw))):
        bh = float(df_raw['high'].iloc[i]); bl = float(df_raw['low'].iloc[i])
        bt = pd.to_datetime(df_raw['timestamp'].iloc[i])
        if (bt.hour + bt.minute/60.0) >= end_td + 0.5:
            break
        if direction=='LONG':
            if bh>=tr_trig: tr_hit=True
            if tr_hit: tr_sl=max(tr_sl, bh-TRAIL_ATR_MULT*atr_val)
            if bh>=be_trig and not be_hit: sl=ep; be_hit=True
            eff=max(sl,tr_sl) if tr_hit else sl
            if bl<=eff:
                reason = 'TRAIL_STOP' if tr_hit else ('BE_STOP' if be_hit else 'STOP_LOSS')
                return (eff-ep)/sl_pts, reason
        else:
            if bl<=tr_trig: tr_hit=True
            if tr_hit: tr_sl=min(tr_sl, bl+TRAIL_ATR_MULT*atr_val)
            if bl<=be_trig and not be_hit: sl=ep; be_hit=True
            eff=min(sl,tr_sl) if tr_hit else sl
            if bh>=eff:
                reason = 'TRAIL_STOP' if tr_hit else ('BE_STOP' if be_hit else 'STOP_LOSS')
                return (ep-eff)/sl_pts, reason

    last = float(df_raw['close'].iloc[min(idx+199, len(df_raw)-1)])
    pts = (last-ep) if direction=='LONG' else (ep-last)
    return pts/sl_pts, 'TIMEOUT'


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("═"*65)
    print("  META-FILTER — training pe semnale v1 (NY conf≥0.55)")
    print("═"*65)

    # Load v1 model
    v1 = xgb.XGBClassifier(); v1.load_model(str(V1_MODEL))
    v1_features = json.loads(V1_FEATURES.read_text())['features']
    print(f"  v1 features: {len(v1_features)}")

    COLS = """timestamp,open,high,low,close,volume,
               p_hi,p_lo,asia_hi,asia_lo,lon_hi,lon_lo,
               lw_hi,lw_lo,h4_hi,h4_lo,h1_hi,h1_lo,
               atr_14,vah,val,poc_level,inside_va,
               bar_delta,cum_delta,dom_ratio,of_big_balance,of_doi,
               absorption_score,stacked_bull,stacked_bear,
               rvol,adx_14,hurst,garch_vol,fisher_transform,acf_lag1,dist_vwap,
               fvg_up,fvg_down,has_displacement"""

    conn = sqlite3.connect(str(PATH_DB))
    prev_tail = None
    all_records = []

    for year in ALL_YEARS:
        print(f"\n📅 {year}...", end="", flush=True)
        df_yr = pd.read_sql(
            f"SELECT {COLS} FROM market_data WHERE year={year} ORDER BY timestamp", conn)
        if len(df_yr)==0: print(" skip"); continue

        if prev_tail is not None:
            combined = pd.concat([prev_tail, df_yr], ignore_index=True)
            si = len(prev_tail)
        else:
            combined = df_yr; si = 0

        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        v1f = compute_v1_features(combined)
        mf  = compute_meta_features(combined)

        td = (combined['timestamp'].dt.hour + combined['timestamp'].dt.minute/60.0).values
        in_ny = (td >= NY_S) & (td < NY_E)

        # v1 predictions
        Xv1 = v1f[v1_features].fillna(0).replace([np.inf,-np.inf],0).astype(np.float32)
        proba = v1.predict_proba(Xv1)
        pred = np.argmax(proba, axis=1); conf = proba.max(axis=1)

        # Signal mask (training: conf >= 0.55)
        sig_mask = np.isin(pred, SIGNAL_CLASSES) & (conf >= V1_CONF_THRESHOLD) & in_ny
        sig_mask[:si] = False   # nu training pe context bars
        sig_idxs = np.where(sig_mask)[0]

        records = []; last_exit = -1
        for idx in sig_idxs:
            if idx <= last_exit: continue
            pred_c = int(pred[idx]); direction = DIRECTION[pred_c]
            atr_val = float(combined['atr_14'].iloc[idx]) if combined['atr_14'].iloc[idx]>0 else 9.0
            r, reason = simulate_trade_rm(combined, idx, direction, atr_val)
            label = 1 if reason=='TRAIL_STOP' else 0

            # Colectam features meta pentru aceasta bara
            row_v1  = {f: float(v1f[f].iloc[idx]) for f in v1_features}
            row_meta= {f: float(mf[f].iloc[idx]) for f in META_FEATURES}
            row_meta.update(row_v1)
            row_meta['label']     = label
            row_meta['r_mult']    = r
            row_meta['exit_reason']= reason
            row_meta['pred_class'] = pred_c
            row_meta['confidence'] = float(conf[idx])
            row_meta['year']       = year
            row_meta['timestamp']  = str(combined['timestamp'].iloc[idx])
            records.append(row_meta)
            last_exit = idx

        all_records.extend(records)
        trail_n = sum(1 for r in records if r['label']==1)
        print(f" {len(records)} semnale, {trail_n} TRAIL ({100*trail_n/max(len(records),1):.0f}%)")
        prev_tail = df_yr.tail(CONTEXT_BARS).copy()
        del combined, v1f, mf, df_yr; gc.collect()

    conn.close()
    df = pd.DataFrame(all_records)
    df['label'] = df['label'].astype(int)
    print(f"\n  Total semnale: {len(df):,}")
    print(f"  TRAIL exits: {df['label'].sum():,} ({100*df['label'].mean():.1f}%)")
    print(f"  Non-TRAIL:   {(df['label']==0).sum():,}")

    all_features = v1_features + META_FEATURES

    df_train = df[df['year'].isin(TRAIN_YEARS)]
    df_val   = df[df['year'].isin(VAL_YEARS)]
    df_test  = df[df['year'].isin(TEST_YEARS)]
    print(f"\n  Split: train={len(df_train):,} val={len(df_val):,} test={len(df_test):,}")

    X_tr = df_train[all_features].fillna(0).values.astype(np.float32)
    y_tr = df_train['label'].values
    X_va = df_val[all_features].fillna(0).values.astype(np.float32)
    y_va = df_val['label'].values
    X_te = df_test[all_features].fillna(0).values.astype(np.float32)
    y_te = df_test['label'].values

    trail_rate = y_tr.mean()
    pos_weight = (1-trail_rate) / max(trail_rate, 0.01)
    print(f"\n  pos_weight (TRAIL/non-TRAIL): {pos_weight:.2f}")

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=5, learning_rate=0.05, n_estimators=500,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=10, gamma=2,
        reg_alpha=0.5, reg_lambda=2.0,
        scale_pos_weight=pos_weight,
        eval_metric='auc',
        early_stopping_rounds=30,
        n_jobs=-1, random_state=42, tree_method='hist',
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=50)

    # Evaluare
    print("\n" + "═"*65)
    for label, X, y, df_s in [
        ("VAL 2022", X_va, y_va, df_val),
        ("OOS 2023-2025", X_te, y_te, df_test)
    ]:
        proba_meta = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, proba_meta)
        print(f"\n  [{label}]  AUC: {auc:.4f}")
        print(f"  Total semnale v1: {len(y):,}  TRAIL rate: {y.mean()*100:.1f}%")

        # Analiza thresholds
        print(f"  {'Thr':<7} {'N':>5} {'Prec':>7} {'Recall':>8} {'T/zi':>7} {'WR_sim':>8}")
        days = df_s['timestamp'].apply(lambda x: x[:10]).nunique() if 'timestamp' in df_s.columns else 756
        for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            mask = proba_meta >= thr
            if mask.sum() < 10: continue
            y_sel = y[mask]
            prec   = y_sel.mean()   # fraction of selected that are TRAIL
            recall = y_sel.sum() / max(y.sum(), 1)
            n_per_day = mask.sum() / (days/3 if label=='OOS 2023-2025' else days)
            # Simulated WR: TRAIL=win, non-TRAIL = mix of BE(0R) + SL(-1R)
            # BE rate ≈ 60%, SL rate ≈ 40% among non-TRAIL
            wr_sim = prec * 1.0 + (1-prec) * 0.0  # trail=win, non-trail=0 (conservative)
            print(f"  thr≥{thr:.2f}  {mask.sum():>5}  {prec:>6.1%}  {recall:>7.1%}  {n_per_day:>7.2f}  {wr_sim:>7.1%}")

    model.save_model(str(OUT_MODEL))
    meta = dict(
        v1_model   = "mario_bot.json",
        v1_conf    = V1_CONF_THRESHOLD,
        all_features = all_features,
        v1_features  = v1_features,
        meta_features= META_FEATURES,
        train_years  = TRAIN_YEARS,
        val_years    = VAL_YEARS,
        test_years   = TEST_YEARS,
    )
    with open(str(OUT_META), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n  ✅ Meta-model salvat: {OUT_MODEL.name}")
    print(f"  ⏱️  Timp: {(time.time()-t0)/60:.1f} min")
    print("═"*65)


if __name__ == "__main__":
    main()
