"""
Backtest mario_bot_v2.json + simulare prop firm Lucid Trading
OOS: 2023-2025, NY+LON, conf >= 0.50 (SHORT_REV + LONG_REV)
RM: SL→BE la 0.5R, trail de la 2R
Prop firm: $50k, trailing DD $2k, 5 win days >$150, payout capped $2k
"""
import sys, sqlite3, json, pathlib, warnings, gc
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
warnings.filterwarnings("ignore")

DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(DIR))

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH    = DIR / "mario_bot_v2.json"
FEATURES_PATH = DIR / "mario_bot_v2_features.json"
PATH_DB       = DIR / "mario_trading.db"

CONF_THRESHOLD   = 0.50
SIGNAL_CLASSES   = [1, 2, 3, 4]   # SHORT_BRK, LONG_BRK, SHORT_REV, LONG_REV
DIRECTION        = {1:'SHORT', 2:'LONG', 3:'SHORT', 4:'LONG'}
REGIME_NAMES     = {0:'WAIT',1:'SHORT_BREAK',2:'LONG_BREAK',3:'SHORT_REV',4:'LONG_REV'}

# RM
ATR_SL_MULT  = 0.8
MIN_SL_PTS   = 6.0
MAX_SL_PTS   = 14.0
BE_TRIGGER_R = 0.5
TRAIL_START_R = 2.0
TRAIL_ATR_MULT = 0.5
TICK_VALUE   = 20.0   # $20/pt NQ Mini

# Entry windows (ore Romania)
LON_ENTRY_START, LON_ENTRY_END = 9.5,  11.0
NY_ENTRY_START,  NY_ENTRY_END  = 16.0, 17.5

# Prop firm
INITIAL_BALANCE     = 50_000.0
TRAILING_DD         = 2_000.0
EVAL_TARGET_PROFIT  = 3_000.0
FIRST_PAYOUT_PROFIT = 2_000.0
PAYOUT_BUFFER       = 1_000.0
WIN_DAY_MIN         = 150.0
WIN_DAYS_REQUIRED   = 5
PAYOUT_CAP          = 2_000.0   # max payout per tranzactie


# ── FEATURE COMPUTATION (identic cu train_mario_bot_v2.py) ───────────────────
def safe_div(a, b, fill=0.0):
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.asarray(a, float) / np.where(np.asarray(b, float) != 0,
                                             np.asarray(b, float), np.nan)
    return np.where(np.isfinite(r), r, fill)

def clip5(x): return np.clip(np.asarray(x, float), -5, 5)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    cl  = df['close'].values.astype(float)
    hi  = df['high'].values.astype(float)
    lo  = df['low'].values.astype(float)
    op  = df['open'].values.astype(float)
    vol = df['volume'].values.astype(float)
    atr = np.where(df['atr_14'].values > 0, df['atr_14'].values, 9.0).astype(float)

    pdh = pd.Series(df['p_hi'].values).ffill().values.astype(float)
    pdl = pd.Series(df['p_lo'].values).ffill().values.astype(float)
    a_hi= pd.Series(df['asia_hi'].values).ffill().values.astype(float)
    a_lo= pd.Series(df['asia_lo'].values).ffill().values.astype(float)
    l_hi= pd.Series(df['lon_hi'].values).ffill().values.astype(float)
    l_lo= pd.Series(df['lon_lo'].values).ffill().values.astype(float)
    h4h = pd.Series(df['h4_hi'].values).ffill().values.astype(float)
    h4l = pd.Series(df['h4_lo'].values).ffill().values.astype(float)
    vah = pd.Series(df['vah'].values).ffill().values.astype(float)
    val = pd.Series(df['val'].values).ffill().values.astype(float)
    poc = pd.Series(df['poc_level'].values).ffill().values.astype(float)
    lw_h= pd.Series(df['lw_hi'].values).ffill().values.astype(float)
    lw_l= pd.Series(df['lw_lo'].values).ffill().values.astype(float)

    ts = pd.to_datetime(df['timestamp'])
    td = (ts.dt.hour + ts.dt.minute / 60.0).values

    out = {}
    out['in_london_kz']    = ((td>=9.5)&(td<11.0)).astype(np.int8)
    out['in_london_or']    = ((td>=9.0)&(td<9.5)).astype(np.int8)
    out['in_london_close'] = ((td>=11.0)&(td<12.0)).astype(np.int8)
    out['in_pre_ny']       = ((td>=15.0)&(td<15.5)).astype(np.int8)
    out['in_ny_or']        = ((td>=15.5)&(td<16.0)).astype(np.int8)
    out['in_ny_kz_core']   = ((td>=16.0)&(td<16.833)).astype(np.int8)
    out['in_ny_macro_1']   = ((td>=16.833)&(td<17.167)).astype(np.int8)
    out['in_ny_macro_2']   = ((td>=17.167)&(td<17.5)).astype(np.int8)
    out['in_any_macro']    = ((out['in_ny_macro_1']==1)|(out['in_ny_macro_2']==1)).astype(np.int8)
    out['mins_since_lon_open'] = np.clip((td-9.0)*60,0,180).astype(np.float32)
    out['mins_since_ny_open']  = np.clip((td-15.5)*60,0,120).astype(np.float32)
    out['hour_sin'] = np.sin(2*np.pi*td/24).astype(np.float32)
    out['hour_cos'] = np.cos(2*np.pi*td/24).astype(np.float32)

    out['dist_pdh_atr']     = clip5(safe_div(pdh-cl,atr)).astype(np.float32)
    out['dist_pdl_atr']     = clip5(safe_div(cl-pdl,atr)).astype(np.float32)
    out['dist_asia_hi_atr'] = clip5(safe_div(a_hi-cl,atr)).astype(np.float32)
    out['dist_asia_lo_atr'] = clip5(safe_div(cl-a_lo,atr)).astype(np.float32)
    out['dist_lon_hi_atr']  = clip5(safe_div(l_hi-cl,atr)).astype(np.float32)
    out['dist_lon_lo_atr']  = clip5(safe_div(cl-l_lo,atr)).astype(np.float32)
    out['dist_vah_atr']     = clip5(safe_div(vah-cl,atr)).astype(np.float32)
    out['dist_val_atr']     = clip5(safe_div(cl-val,atr)).astype(np.float32)
    out['dist_poc_atr']     = clip5(safe_div(cl-poc,atr)).astype(np.float32)
    out['above_vah']        = (cl>vah).astype(np.int8)
    out['below_val']        = (cl<val).astype(np.int8)
    out['inside_va']        = df['inside_va'].fillna(0).values.astype(np.int8)

    out['broke_pdh']     = (hi>pdh).astype(np.int8)
    out['broke_pdl']     = (lo<pdl).astype(np.int8)
    out['broke_asia_hi'] = (hi>a_hi).astype(np.int8)
    out['broke_asia_lo'] = (lo<a_lo).astype(np.int8)
    out['broke_lon_hi']  = (hi>l_hi).astype(np.int8)
    out['broke_lon_lo']  = (lo<l_lo).astype(np.int8)
    out['broke_vah']     = (hi>vah).astype(np.int8)
    out['broke_val']     = (lo<val).astype(np.int8)

    sw_up = ((hi>a_hi)+(hi>pdh)+(hi>l_hi)+(hi>h4h)).astype(int)
    sw_dn = ((lo<a_lo)+(lo<pdl)+(lo<l_lo)+(lo<h4l)).astype(int)
    out['sweep_direction'] = np.where(sw_up>sw_dn,1,np.where(sw_dn>sw_up,-1,0)).astype(np.int8)
    out['swept_above']     = (sw_up>0).astype(np.int8)
    out['swept_below']     = (sw_dn>0).astype(np.int8)

    bar_range = np.maximum(hi-lo,0.01)
    uw = hi-np.maximum(cl,op); lw = np.minimum(cl,op)-lo
    out['sweep_wick_ratio']  = clip5(safe_div(np.maximum(uw,lw),bar_range)).astype(np.float32)
    out['upper_wick_atr']    = clip5(safe_div(uw,atr)).astype(np.float32)
    out['lower_wick_atr']    = clip5(safe_div(lw,atr)).astype(np.float32)

    any_sw = ((sw_up+sw_dn)>0).astype(int)
    bs=[]; cnt=99
    for v in any_sw:
        cnt=0 if v else cnt+1; bs.append(min(cnt,99))
    out['bars_since_sweep'] = np.array(bs,dtype=np.float32)
    sw_up10 = pd.Series(sw_up).rolling(10,min_periods=1).max().values
    sw_dn10 = pd.Series(sw_dn).rolling(10,min_periods=1).max().values
    out['reclaimed_after_sweep'] = (((sw_up10>0)&(cl<a_hi)&(cl<pdh))|((sw_dn10>0)&(cl>a_lo)&(cl>pdl))).astype(np.int8)

    body=np.abs(cl-op)
    out['body_dir']        = np.sign(cl-op).astype(np.int8)
    out['wick_bias']       = clip5(safe_div(uw-lw,bar_range)).astype(np.float32)
    out['range_atr_ratio'] = clip5(safe_div(bar_range,atr)).astype(np.float32)
    out['body_pct']        = clip5(safe_div(body,bar_range)).astype(np.float32)
    out['close_strength']  = safe_div(cl-lo,bar_range).astype(np.float32)
    out['sharp_reversal']  = ((out['range_atr_ratio']>1.5)&(out['close_strength']>0.7)).astype(np.int8)
    out['rejection_candle']= ((out['sweep_wick_ratio']>0.6)&(body<0.4*bar_range)).astype(np.int8)

    cl_s=pd.Series(cl)
    out['momentum_5']  = clip5(safe_div((cl_s-cl_s.shift(5)).values,atr)).astype(np.float32)
    out['momentum_15'] = clip5(safe_div((cl_s-cl_s.shift(15)).values,atr)).astype(np.float32)
    out['slope_h1']    = clip5(safe_div((cl_s-cl_s.shift(60)).values,atr)).astype(np.float32)
    out['h4_momentum'] = clip5(safe_div((cl_s-cl_s.shift(240)).values,atr)).astype(np.float32)

    atr_s=pd.Series(atr)
    out['atr_percentile'] = atr_s.rolling(100,min_periods=10).rank(pct=True).fillna(0.5).values.astype(np.float32)
    out['vol_spike']      = (atr_s>atr_s.rolling(20).mean()*1.5).astype(np.int8).values
    out['rvol']           = df['rvol'].fillna(1.0).values.astype(np.float32)
    out['adx_14']         = df['adx_14'].fillna(20.0).values.astype(np.float32)
    out['hurst']          = df['hurst'].fillna(0.5).values.astype(np.float32)
    out['garch_vol']      = df['garch_vol'].fillna(0.0).values.astype(np.float32)
    out['fisher_transform']= df['fisher_transform'].fillna(0.0).values.astype(np.float32)
    out['acf_lag1']       = df['acf_lag1'].fillna(0.0).values.astype(np.float32)

    out['liq_above'] = ((a_hi>cl)+(pdh>cl)+(h4h>cl)+(lw_h>cl)).astype(np.int8)
    out['liq_below'] = ((a_lo<cl)+(pdl<cl)+(h4l<cl)+(lw_l<cl)).astype(np.int8)

    dh=df.groupby(ts.dt.date)['high'].cummax().values
    dl=df.groupby(ts.dt.date)['low'].cummin().values
    out['range_day_score'] = (1.0/np.maximum(safe_div(dh-dl,atr,1.0),1.0)).astype(np.float32)

    bar_d=df['bar_delta'].fillna(0).values.astype(float)
    cum_d=df['cum_delta'].fillna(0).values.astype(float)
    vol_atr=vol*atr+1
    out['bar_delta_n']     = clip5(safe_div(bar_d,vol_atr)).astype(np.float32)
    cum_abs20=pd.Series(np.abs(cum_d)).rolling(20).mean().fillna(1).values
    out['cum_delta_n']     = clip5(safe_div(cum_d,cum_abs20)).astype(np.float32)
    out['dom_ratio']       = df['dom_ratio'].fillna(1.0).values.astype(np.float32)
    out['of_big_balance']  = df['of_big_balance'].fillna(0).values.astype(np.float32)
    out['of_doi']          = df['of_doi'].fillna(0).values.astype(np.float32)
    out['absorption_score']= df['absorption_score'].fillna(0).values.astype(np.float32)
    out['stacked_bull']    = df['stacked_bull'].fillna(0).values.astype(np.int8)
    out['stacked_bear']    = df['stacked_bear'].fillna(0).values.astype(np.int8)

    dv=df['dist_vwap'].fillna(0).values.astype(float)
    out['dist_vwap_atr']   = clip5(safe_div(dv,atr)).astype(np.float32)
    out['fvg_up']          = df['fvg_up'].fillna(0).values.astype(np.int8)
    out['fvg_down']        = df['fvg_down'].fillna(0).values.astype(np.int8)
    out['has_displacement']= df['has_displacement'].fillna(0).values.astype(np.int8)

    return pd.DataFrame(out, index=df.index)


# ── RM SIMULATION ─────────────────────────────────────────────────────────────
def simulate_trade(bars, entry_idx, direction, atr_entry):
    sl_pts = float(np.clip(ATR_SL_MULT * atr_entry, MIN_SL_PTS, MAX_SL_PTS))
    entry_px = float(bars['close'].iloc[entry_idx])
    if direction == 'LONG':
        sl = entry_px - sl_pts
        be_trig   = entry_px + BE_TRIGGER_R * sl_pts
        trail_trig= entry_px + TRAIL_START_R * sl_pts
        trail_sl  = sl; be_hit=False; trail_hit=False
        for i in range(entry_idx+1, len(bars)):
            bh = bars['high'].iloc[i]; bl = bars['low'].iloc[i]
            if bh >= trail_trig: trail_hit=True
            if trail_hit:
                trail_sl = max(trail_sl, bh - TRAIL_ATR_MULT*atr_entry)
            if bh >= be_trig and not be_hit:
                sl=entry_px; be_hit=True
            eff_sl = max(sl, trail_sl) if trail_hit else sl
            if bl <= eff_sl:
                pts=eff_sl-entry_px; r=pts/sl_pts
                reason='TRAIL' if trail_hit else ('BE' if be_hit else 'SL')
                return dict(r_mult=r,pts=pts,exit_reason=reason,sl_pts=sl_pts,entry_px=entry_px,exit_px=eff_sl)
    else:
        sl = entry_px + sl_pts
        be_trig   = entry_px - BE_TRIGGER_R * sl_pts
        trail_trig= entry_px - TRAIL_START_R * sl_pts
        trail_sl  = sl; be_hit=False; trail_hit=False
        for i in range(entry_idx+1, len(bars)):
            bh = bars['high'].iloc[i]; bl = bars['low'].iloc[i]
            if bl <= trail_trig: trail_hit=True
            if trail_hit:
                trail_sl = min(trail_sl, bl + TRAIL_ATR_MULT*atr_entry)
            if bl <= be_trig and not be_hit:
                sl=entry_px; be_hit=True
            eff_sl = min(sl, trail_sl) if trail_hit else sl
            if bh >= eff_sl:
                pts=entry_px-eff_sl; r=pts/sl_pts
                reason='TRAIL' if trail_hit else ('BE' if be_hit else 'SL')
                return dict(r_mult=r,pts=pts,exit_reason=reason,sl_pts=sl_pts,entry_px=entry_px,exit_px=eff_sl)

    last=float(bars['close'].iloc[-1])
    pts=(last-entry_px) if direction=='LONG' else (entry_px-last)
    return dict(r_mult=pts/sl_pts, pts=pts, exit_reason='TIMEOUT',
                sl_pts=sl_pts, entry_px=entry_px, exit_px=last)


# ── PROP FIRM SIMULATION ──────────────────────────────────────────────────────
def run_propfirm(df_trades):
    trades = df_trades.sort_values('timestamp').reset_index(drop=True)
    balance=INITIAL_BALANCE; peak=INITIAL_BALANCE
    floor=INITIAL_BALANCE-TRAILING_DD; floor_locked=False
    eval_passed=False; first_payout_done=False
    win_days_count=0; payout_eligible=False
    daily_pnl=defaultdict(float)
    total_payouts=0; total_withdrawn=0.0; n_blown=0
    attempts=[]; log=[]
    prev_date=None; run_trades=[]

    def reset():
        nonlocal balance,peak,floor,floor_locked,eval_passed
        nonlocal first_payout_done,win_days_count,payout_eligible,daily_pnl,prev_date,run_trades
        balance=INITIAL_BALANCE;peak=INITIAL_BALANCE;floor=INITIAL_BALANCE-TRAILING_DD
        floor_locked=False;eval_passed=False;first_payout_done=False
        win_days_count=0;payout_eligible=False;daily_pnl=defaultdict(float)
        prev_date=None;run_trades=[]
    reset()

    for _,t in trades.iterrows():
        pnl=float(t['pnl_usd']); ts=pd.to_datetime(t['timestamp']); date=ts.date()
        balance+=pnl; run_trades.append(t); daily_pnl[date]+=pnl

        if eval_passed and prev_date and date!=prev_date:
            if daily_pnl.get(prev_date,0)>WIN_DAY_MIN:
                win_days_count+=1
                if win_days_count>=WIN_DAYS_REQUIRED and not payout_eligible:
                    payout_eligible=True
                    log.append(f"  📅 ELIGIBIL  win_days={win_days_count}  bal=${balance:,.0f}  [{date}]")
        prev_date=date

        if balance>peak: peak=balance
        if not floor_locked:
            floor=min(peak-TRAILING_DD,INITIAL_BALANCE)
            if floor>=INITIAL_BALANCE: floor=INITIAL_BALANCE; floor_locked=True

        if not eval_passed and balance>=INITIAL_BALANCE+EVAL_TARGET_PROFIT:
            eval_passed=True
            log.append(f"  ✅ EVAL PASSED  bal=${balance:,.0f}  [{date}]")

        if eval_passed and payout_eligible:
            if not first_payout_done:
                if balance>=INITIAL_BALANCE+FIRST_PAYOUT_PROFIT:
                    payout=min(balance-(INITIAL_BALANCE+PAYOUT_BUFFER), PAYOUT_CAP)
                    if payout>0:
                        balance-=payout; total_withdrawn+=payout; total_payouts+=1
                        first_payout_done=True; floor=INITIAL_BALANCE; floor_locked=True
                        peak=max(peak,balance); win_days_count=0; payout_eligible=False
                        log.append(f"  💸 PAYOUT #{total_payouts:<3} +${payout:>7,.0f}  bal=${balance:,.0f}  [{date}]")
            else:
                if balance>INITIAL_BALANCE+PAYOUT_BUFFER:
                    payout=min(balance-(INITIAL_BALANCE+PAYOUT_BUFFER), PAYOUT_CAP)
                    balance-=payout; total_withdrawn+=payout; total_payouts+=1
                    floor=INITIAL_BALANCE; peak=max(peak,balance)
                    win_days_count=0; payout_eligible=False
                    log.append(f"  💸 PAYOUT #{total_payouts:<3} +${payout:>7,.0f}  bal=${balance:,.0f}  [{date}]")

        if balance<=floor:
            n_blown+=1; phase="FUNDED" if eval_passed else "EVAL"
            log.append(f"  💥 BLOWN #{n_blown}  {phase}  bal=${balance:,.0f}  floor=${floor:,.0f}  [{date}]")
            attempts.append(dict(run=n_blown,eval_passed=eval_passed,trades=len(run_trades),ts=date))
            reset()

    log.append(f"\n  🏁 FIN  bal=${balance:,.0f}  eval={'PASS' if eval_passed else 'FAIL'}  win_days={win_days_count}")
    return dict(n_blown=n_blown,total_payouts=total_payouts,total_withdrawn=total_withdrawn,
                attempts=attempts,log=log,final_balance=balance,eval_passed_final=eval_passed)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("═"*65)
    print("  BACKTEST + PROP FIRM — mario_bot_v2.json  (2023-2025)")
    print("═"*65)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    features = json.loads(FEATURES_PATH.read_text())['features']
    print(f"✓ Model loaded: {len(features)} features")

    # Procesează an cu an pentru memorie redusă
    COLS = """timestamp,open,high,low,close,volume,
               p_hi,p_lo,asia_hi,asia_lo,lon_hi,lon_lo,
               lw_hi,lw_lo,h4_hi,h4_lo,h1_hi,h1_lo,
               atr_14,vah,val,poc_level,inside_va,
               bar_delta,cum_delta,dom_ratio,of_big_balance,of_doi,
               absorption_score,stacked_bull,stacked_bear,
               rvol,adx_14,hurst,garch_vol,fisher_transform,acf_lag1,dist_vwap,
               fvg_up,fvg_down,has_displacement"""

    conn = sqlite3.connect(str(PATH_DB))
    # tail din 2022 ca context rolling
    ctx = pd.read_sql(f"SELECT {COLS} FROM market_data WHERE year=2022 ORDER BY timestamp", conn)
    ctx = ctx.tail(300).copy()

    feat_chunks = []; ohlc_chunks = []
    for yr in [2023, 2024, 2025]:
        df_yr = pd.read_sql(f"SELECT {COLS} FROM market_data WHERE year={yr} ORDER BY timestamp", conn)
        combined = pd.concat([ctx, df_yr], ignore_index=True)
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        fc = compute_features(combined)
        fc['timestamp'] = combined['timestamp'].values
        fc['atr_14']    = combined['atr_14'].values
        fc['high']      = combined['high'].values
        fc['low']       = combined['low'].values
        fc['close']     = combined['close'].values
        # păstrează doar rândurile din 2023/2024/2025 (nu contextul)
        fc = fc.iloc[len(ctx):].reset_index(drop=True)
        oc = df_yr.reset_index(drop=True)
        oc['timestamp'] = pd.to_datetime(oc['timestamp'])
        feat_chunks.append(fc); ohlc_chunks.append(oc)
        ctx = df_yr.tail(300).copy()
        del combined, df_yr, fc; gc.collect()
        print(f"   {yr}: OK")

    conn.close()
    feat_df = pd.concat(feat_chunks, ignore_index=True)
    df_ohlc = pd.concat(ohlc_chunks, ignore_index=True)
    del feat_chunks, ohlc_chunks; gc.collect()
    print(f"✓ OOS rows: {len(feat_df):,}")

    # Predicții
    X = feat_df[features].fillna(0).replace([np.inf,-np.inf],0).astype(np.float32)
    proba = model.predict_proba(X)
    pred  = np.argmax(proba, axis=1)
    conf  = proba.max(axis=1)

    td = (feat_df['timestamp'].dt.hour + feat_df['timestamp'].dt.minute/60.0).values
    in_lon = (td >= LON_ENTRY_START) & (td < LON_ENTRY_END)
    in_ny  = (td >= NY_ENTRY_START)  & (td < NY_ENTRY_END)
    in_entry = in_lon | in_ny
    has_signal = np.isin(pred, SIGNAL_CLASSES) & (conf >= CONF_THRESHOLD)
    entry_mask = in_entry & has_signal

    entry_indices = np.where(entry_mask)[0]
    print(f"✓ Semnale (conf>={CONF_THRESHOLD}): {len(entry_indices):,}")

    # Simulare trades
    trades = []
    last_exit = -1
    for idx in entry_indices:
        if idx <= last_exit: continue
        pred_c = int(pred[idx]); direction = DIRECTION[pred_c]
        atr_val = float(feat_df['atr_14'].iloc[idx]) if feat_df['atr_14'].iloc[idx] > 0 else 9.0
        ts_entry = feat_df['timestamp'].iloc[idx]
        sess = 'LON' if (td[idx]>=LON_ENTRY_START and td[idx]<LON_ENTRY_END) else 'NY'
        end_td = LON_ENTRY_END if sess=='LON' else NY_ENTRY_END

        future = df_ohlc.iloc[idx:idx+150].copy()
        ftd = (pd.to_datetime(future['timestamp']).dt.hour +
               pd.to_datetime(future['timestamp']).dt.minute/60.0).values
        future = future[ftd < end_td+0.5]
        if len(future) < 2: continue

        res = simulate_trade(future.reset_index(drop=True), 0, direction, atr_val)
        res.update(dict(
            timestamp=ts_entry, date=ts_entry.date(),
            session=sess, pred_class=pred_c,
            regime=REGIME_NAMES[pred_c], direction=direction,
            confidence=float(conf[idx]), atr_entry=atr_val,
        ))
        res['pnl_usd'] = res['r_mult'] * res['sl_pts'] * TICK_VALUE
        trades.append(res)
        last_exit = idx + res.get('bars_held', 0)

    df_t = pd.DataFrame(trades)
    df_t['win'] = (df_t['r_mult'] > 0).astype(int)
    df_t['year'] = pd.to_datetime(df_t['date']).dt.year
    df_t['month'] = pd.to_datetime(df_t['date']).dt.to_period('M')

    print("\n" + "─"*65)
    print("  REZULTATE BACKTEST")
    print("─"*65)
    n=len(df_t); days=df_t['date'].nunique()
    print(f"  Total trades:    {n:,}  ({days} zile)")
    print(f"  Trades/zi:       {n/days:.2f}")
    print(f"  Win rate:        {df_t['win'].mean()*100:.1f}%")
    print(f"  Avg R/trade:     {df_t['r_mult'].mean():+.3f}R")
    print(f"  P&L total:       ${df_t['pnl_usd'].sum():,.0f}")
    print(f"  P&L/zi medie:    ${df_t['pnl_usd'].sum()/days:,.0f}")

    print("\n  PER CLASĂ:")
    for cls in [1,2,3,4]:
        s=df_t[df_t['pred_class']==cls]
        if len(s)==0: continue
        print(f"    {REGIME_NAMES[cls]:<15} N={len(s):>4}  WR={s['win'].mean()*100:>4.1f}%  "
              f"AvgR={s['r_mult'].mean():>+.3f}  P&L=${s['pnl_usd'].sum():>9,.0f}")

    print("\n  PER SESIUNE:")
    for sess in ['LON','NY']:
        s=df_t[df_t['session']==sess]
        if len(s)==0: continue
        print(f"    {sess:<6} N={len(s):>4}  WR={s['win'].mean()*100:>4.1f}%  "
              f"AvgR={s['r_mult'].mean():>+.3f}  P&L=${s['pnl_usd'].sum():>9,.0f}")

    print("\n  PER AN:")
    for yr,g in df_t.groupby('year'):
        d=g['date'].nunique()
        print(f"    {yr}  N={len(g):>4}  WR={g['win'].mean()*100:>4.1f}%  "
              f"AvgR={g['r_mult'].mean():>+.3f}  P&L=${g['pnl_usd'].sum():>8,.0f}  "
              f"T/zi={len(g)/d:.1f}  P&L/zi=${g['pnl_usd'].sum()/d:,.0f}")

    print("\n  EXIT REASONS:")
    for r,g in df_t.groupby('exit_reason'):
        print(f"    {r:<10} N={len(g):>4}  AvgR={g['r_mult'].mean():>+.3f}  P&L=${g['pnl_usd'].sum():>9,.0f}")

    # Win day analysis
    daily_pnl = df_t.groupby('date')['pnl_usd'].sum()
    win_days_month = df_t.groupby('month').apply(
        lambda g: (g.groupby('date')['pnl_usd'].sum()>WIN_DAY_MIN).sum()
    )
    avg_win_days = win_days_month.mean()
    print(f"\n  Win days >$150/zi: medie {avg_win_days:.1f}/lună (trebuie 5 pentru payout)")
    print(f"  P&L mediu/zi: ${daily_pnl.mean():,.0f}  |  mediana: ${daily_pnl.median():,.0f}")

    # ── PROP FIRM SIM ──────────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  PROP FIRM SIMULATION — LUCID TRADING")
    print("═"*65)
    res = run_propfirm(df_t)
    for line in res['log']: print(line)

    attempts = res['attempts']
    n_total = len(attempts)+1
    blown_eval   = sum(1 for a in attempts if not a['eval_passed'])
    blown_funded = sum(1 for a in attempts if a['eval_passed'])

    print(f"\n  Total runs:        {n_total}")
    print(f"  Eval PASS rate:    {(n_total-blown_eval-blown_funded if res['eval_passed_final'] else n_total-blown_eval-blown_funded)/n_total*100:.0f}%")
    print(f"  Blown în EVAL:     {blown_eval}")
    print(f"  Blown pe FUNDED:   {blown_funded}")
    print(f"  Total payouturi:   {res['total_payouts']}")
    print(f"  Total retras:      ${res['total_withdrawn']:,.0f}")
    print(f"  Media/payout:      ${res['total_withdrawn']/max(res['total_payouts'],1):,.0f}")
    print(f"  Payouturi/lună:    {res['total_payouts']/36:.1f}")

    print("\n  BUSINESS CASE:")
    for cost in [150,250,350]:
        net = res['total_withdrawn'] - n_total*cost
        print(f"    Eval ${cost}/cont: net=${net:,.0f}  ROI={net/(n_total*cost)*100:+.0f}%")
    print("═"*65)

    df_t.to_csv(DIR/"backtest_mario_bot_v2_trades.csv", index=False)
    print(f"\n✅ Trades salvate: backtest_mario_bot_v2_trades.csv")


if __name__=="__main__":
    main()
