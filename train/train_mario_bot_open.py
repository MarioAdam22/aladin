"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MARIO BOT v2 — MODEL UNIFICAT (memory-efficient, an cu an)                ║
║  Combina features din: mario_bot + reversal_model + breakout               ║
║  TARGET: 5 clase — WAIT / SHORT_BREAK / LONG_BREAK / SHORT_REV / LONG_REV  ║
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
OUT_MODEL    = DIR / "mario_bot_open.json"
OUT_FEATURES = DIR / "mario_bot_open_features.json"
DATASET_CSV  = DIR / "mario_bot_open_dataset.csv"

# ── CONFIG ────────────────────────────────────────────────────────────────────
YEARS_ALL    = list(range(2015, 2026))
TRAIN_YEARS  = list(range(2015, 2022))   # 2015-2021 train
VAL_YEARS    = [2022]
TEST_YEARS   = list(range(2023, 2026))   # 2023-2025 OOS

# Entry windows — Eastern Time (UTC-5 winter)
# LON: 2:00-5:00 AM ET = 07:00-10:00 UTC
# NY:  8:00-11:00 AM ET = 13:00-16:00 UTC (NYSE open 9:30 AM ET = 14:30 UTC)
LON_START, LON_END = 7.0,  10.0
NY_START,  NY_END  = 13.0, 16.0
HORIZON            = 60
TP_ATR             = 2.0
CONTEXT_BARS       = 300   # bare extra din an precedent pt rolling features

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
    # Order Flow DB (100% coverage)
    'bar_delta_n','cum_delta_n','dom_ratio','of_big_balance',
    'of_doi','absorption_score','stacked_bull','stacked_bear',
    # VP + dist
    'dist_vwap_atr','fvg_up','fvg_down','has_displacement',
]


def safe_div(a, b, fill=0.0):
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.asarray(a, float) / np.where(np.asarray(b, float) != 0, np.asarray(b, float), np.nan)
    return np.where(np.isfinite(r), r, fill)

def clip5(x): return np.clip(np.asarray(x, float), -5, 5)


def compute_features_and_target(df: pd.DataFrame, start_idx: int = 0) -> pd.DataFrame:
    """
    Compute features + target. start_idx = prima bara care apartine anului curent
    (barele 0..start_idx-1 sunt context din an precedent).
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

    # Session windows — toate în UTC, convertite din ET
    # LON: 2-5 AM ET = 7-10 UTC
    # NY:  8-11 AM ET = 13-16 UTC, NYSE open = 9:30 AM ET = 14:30 UTC
    out['in_london_or']    = ((td >= 7.0)  & (td < 7.5)).astype(np.int8)   # London OR: 2:00-2:30 AM ET
    out['in_london_kz']    = ((td >= 7.5)  & (td < 10.0)).astype(np.int8)  # London KZ: 2:30-5:00 AM ET
    out['in_london_close'] = ((td >= 10.0) & (td < 11.0)).astype(np.int8)  # London close: 5-6 AM ET
    out['in_pre_ny']       = ((td >= 13.0) & (td < 13.5)).astype(np.int8)  # Pre-NY: 8:00-8:30 AM ET
    out['in_ny_or']        = ((td >= 13.5) & (td < 14.5)).astype(np.int8)  # NY pre-open: 8:30-9:30 AM ET
    out['in_ny_kz_core']   = ((td >= 14.5) & (td < 15.0)).astype(np.int8)  # NYSE first 30min: 9:30-10:00 AM ET
    out['in_ny_macro_1']   = ((td >= 15.0) & (td < 15.5)).astype(np.int8)  # NY Macro 1: 10:00-10:30 AM ET
    out['in_ny_macro_2']   = ((td >= 15.5) & (td < 16.0)).astype(np.int8)  # NY Macro 2: 10:30-11:00 AM ET
    out['in_any_macro']    = ((out['in_ny_macro_1']==1)|(out['in_ny_macro_2']==1)).astype(np.int8)
    out['mins_since_lon_open'] = np.clip((td-7.0)*60, 0, 180).astype(np.float32)   # de la 2 AM ET
    out['mins_since_ny_open']  = np.clip((td-14.5)*60, 0, 120).astype(np.float32)  # de la 9:30 AM ET (NYSE)
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

    # ── TARGET GENERATION ────────────────────────────────────────────────────
    # Post-OR bars în killzone
    in_kz = ((td >= 9.5) & (td <= 11.0)) | ((td >= 16.0) & (td <= 17.5))
    qualifying = np.where(in_kz)[0]

    target = np.zeros(n, dtype=np.int8)
    for idx in qualifying:
        if idx < start_idx:   # context bars — nu generăm target
            continue
        if idx + HORIZON >= n:
            continue
        a = atr[idx]
        if a <= 0: continue
        entry = cl[idx]
        tp = TP_ATR * a

        res_dir = res_bar = 0
        max_up = max_dn = 0.0
        for h in range(1, HORIZON+1):
            fi = idx + h
            if fi >= n: break
            up = hi[fi] - entry
            dn = entry - lo[fi]
            max_up = max(max_up, up)
            max_dn = max(max_dn, dn)
            if up >= tp and dn >= tp:
                res_dir = 1 if cl[fi] > entry else -1
                res_bar = h; break
            if up >= tp: res_dir = 1;  res_bar = h; break
            if dn >= tp: res_dir = -1; res_bar = h; break

        if res_dir == 0: continue
        if res_dir == 1:
            fut_lo = lo[idx+1:idx+res_bar+1]
            swept = (fut_lo.size > 0 and
                     ((not np.isnan(a_lo[idx]) and fut_lo.min() < a_lo[idx] - 0.2*a) or
                      (not np.isnan(pdl[idx])  and fut_lo.min() < pdl[idx] - 0.2*a)))
            if swept or max_dn >= 1.0*a:
                target[idx] = 4   # LONG_REV
            elif max_up >= 1.5*a:
                target[idx] = 2   # LONG_BREAK
        else:
            fut_hi = hi[idx+1:idx+res_bar+1]
            swept = (fut_hi.size > 0 and
                     ((not np.isnan(a_hi[idx]) and fut_hi.max() > a_hi[idx] + 0.2*a) or
                      (not np.isnan(pdh[idx])  and fut_hi.max() > pdh[idx] + 0.2*a)))
            if swept or max_up >= 1.0*a:
                target[idx] = 3   # SHORT_REV
            elif max_dn >= 1.5*a:
                target[idx] = 1   # SHORT_BREAK

    feat_df['target']    = target
    feat_df['timestamp'] = df['timestamp'].values
    feat_df['year']      = ts.dt.year.values

    # Returnează doar barele killzone cu start_idx (barele anului curent)
    kz_mask = in_kz & (np.arange(n) >= start_idx)
    return feat_df[kz_mask].copy()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("═"*65)
    print("  MARIO BOT v2 — TRAINING (memory-efficient)")
    print("═"*65)
    print(f"  Features: {len(FEATURES)}")

    conn = sqlite3.connect(str(PATH_DB))

    prev_tail = None      # ultimele CONTEXT_BARS bare din an precedent
    all_chunks = []       # dataset killzone acumulat

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

        # Combina cu tail din an precedent pentru rolling features
        if prev_tail is not None:
            df_combined = pd.concat([prev_tail, df_year], ignore_index=True)
            start_idx   = len(prev_tail)
        else:
            df_combined = df_year
            start_idx   = 0

        chunk = compute_features_and_target(df_combined, start_idx=start_idx)
        all_chunks.append(chunk)

        dist = chunk['target'].value_counts().sort_index()
        sig_count = (chunk['target'] != 0).sum()
        print(f"   ✓ {len(df_year):,} rows → {len(chunk):,} KZ bars, "
              f"signals={sig_count} "
              f"({', '.join(f'{REGIME_NAMES[k]}={v}' for k,v in dist.items() if k!=0)})")

        # Salvăm coada pentru contextul anului următor
        prev_tail = df_year.tail(CONTEXT_BARS).copy()
        del df_combined, df_year, chunk
        gc.collect()

    conn.close()

    print("\n🔗 Concatenez dataset...", flush=True)
    dataset = pd.concat(all_chunks, ignore_index=True)
    del all_chunks; gc.collect()

    # Salvare dataset (opcional, pentru debug)
    # dataset.to_csv(DATASET_CSV, index=False)
    print(f"   Dataset total: {len(dataset):,} rows")
    dist_total = dataset['target'].value_counts().sort_index()
    print("   Distribuție globală:")
    for k, v in dist_total.items():
        print(f"     {REGIME_NAMES[k]:<15}: {v:>7,}  ({v/len(dataset)*100:.2f}%)")

    # ── SPLIT ─────────────────────────────────────────────────────────────────
    train_mask = dataset['year'].isin(TRAIN_YEARS)
    val_mask   = dataset['year'].isin(VAL_YEARS)
    test_mask  = dataset['year'].isin(TEST_YEARS)

    X_train = dataset.loc[train_mask, FEATURES].fillna(0).replace([np.inf,-np.inf],0)
    y_train = dataset.loc[train_mask, 'target']
    X_val   = dataset.loc[val_mask,   FEATURES].fillna(0).replace([np.inf,-np.inf],0)
    y_val   = dataset.loc[val_mask,   'target']
    X_test  = dataset.loc[test_mask,  FEATURES].fillna(0).replace([np.inf,-np.inf],0)
    y_test  = dataset.loc[test_mask,  'target']

    print(f"\n📅 Split:")
    print(f"   Train (2015-2021): {len(X_train):,}  signals={(y_train!=0).sum():,}")
    print(f"   Val   (2022):      {len(X_val):,}  signals={(y_val!=0).sum():,}")
    print(f"   Test  (2023-2025): {len(X_test):,}  signals={(y_test!=0).sum():,}  ← OOS REAL")

    # ── UNDERSAMPLE WAIT ──────────────────────────────────────────────────────
    print("\n⚖️  Undersample WAIT 6:1...")
    n_sig   = (y_train != 0).sum()
    n_wait  = min(n_sig * 6, (y_train == 0).sum())
    rng = np.random.RandomState(42)
    wait_idx = rng.choice(y_train[y_train==0].index, size=int(n_wait), replace=False)
    sig_idx  = y_train[y_train!=0].index
    keep = np.sort(np.concatenate([wait_idx, sig_idx.values]))
    X_tr = X_train.loc[keep]
    y_tr = y_train.loc[keep]
    print(f"   Train după US: {len(X_tr):,}  (signals={n_sig:,})")

    # ── WEIGHTS ───────────────────────────────────────────────────────────────
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    sw = y_tr.map(dict(zip(np.unique(y_tr), cw))).values

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    print("\n🚀 Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=1200, max_depth=6, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        num_class=5, objective='multi:softprob',
        eval_metric='mlogloss', tree_method='hist',
        use_label_encoder=False, random_state=42,
        n_jobs=-1, early_stopping_rounds=50, verbosity=0,
    )
    model.fit(X_tr, y_tr, sample_weight=sw,
              eval_set=[(X_val.fillna(0), y_val)], verbose=200)
    print(f"   ✓ Best iteration: {model.best_iteration}")

    # ── EVALUATE OOS ──────────────────────────────────────────────────────────
    print("\n📊 EVALUARE OOS (2023-2025):")
    X_te = X_test.fillna(0).replace([np.inf,-np.inf],0)
    proba = model.predict_proba(X_te)
    pred  = np.argmax(proba, axis=1)
    y_te  = y_test.values

    acc = accuracy_score(y_te, pred)
    print(f"   Accuracy (all): {acc:.4f}")
    print()
    print(classification_report(y_te, pred,
          target_names=[REGIME_NAMES[i] for i in range(5)], zero_division=0))

    # Signal precision per confidence threshold
    print("🎯 PRECISION PER CONF THRESHOLD (semnale predicte):")
    for cls in range(1, 5):
        mask_pred = pred == cls
        if mask_pred.sum() == 0: continue
        conf_cls = proba[mask_pred, cls]
        for thr in [0.35, 0.40, 0.45, 0.50, 0.55]:
            m = conf_cls >= thr
            if m.sum() < 5: continue
            prec = (y_te[mask_pred][m] == cls).mean()
            n    = m.sum()
            print(f"   {REGIME_NAMES[cls]:<15} conf>={thr:.2f}: N={n:>4}  Prec={prec:.3f}")
        print()

    # ── SAVE ──────────────────────────────────────────────────────────────────
    model.save_model(str(OUT_MODEL))
    meta = dict(
        features=FEATURES, n_features=len(FEATURES),
        accuracy_oos=float(acc),
        trained_at=pd.Timestamp.now().isoformat(),
        train_years=TRAIN_YEARS, test_years=TEST_YEARS,
        rows_train=int(len(X_tr)), rows_test=int(len(X_te)),
        regime_names=REGIME_NAMES, conf_threshold=0.40,
        best_iteration=int(model.best_iteration),
    )
    OUT_FEATURES.write_text(json.dumps(meta, indent=2))

    print(f"\n✅ Model: {OUT_MODEL.name}")
    print(f"✅ Meta:  {OUT_FEATURES.name}")
    print(f"⏱️  Total: {time.time()-t0:.0f}s")
    print("═"*65)

    return model, proba, y_te, pred


if __name__ == "__main__":
    main()
