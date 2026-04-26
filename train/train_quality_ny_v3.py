"""
ALADIN — NY Quality Gate v3  (MTF ICT + VIX Proxy + Sweep Quality + Regime)
═════════════════════════════════════════════════════════════════════════════
Față de ny_v2 (133 features, AUC OOS ≈ 0.58), v3 adaugă:
  ✅ Multi-TF FVG (5m, 15m, 1h, 4h): bullish/bearish FVG, distanță, aliniat cu direcția
  ✅ Inversion FVG (pe fiecare TF): FVG mitigated → inversare rol S/R
  ✅ Breaker Blocks (pe fiecare TF): OB broken → acționează ca S/R invers
  ✅ Rejection Blocks (pe fiecare TF): candle cu wick mare la nivel cheie
  ✅ VIX proxy: realized vol 5d/20d din 1-min NQ data
  ✅ DXY proxy: market regime composite (ADX trend + ATR ratio)
  ✅ Sweep quality: level test count, equal lows/highs score, sweep wick quality
  ✅ Rolling regime: rolling 5-session WR, 10d ADX mean, 20d Hurst mean
  TARGET: AUC OOS → 0.68+
═════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import sqlite3, json as _json, warnings, pathlib, pickle
from datetime import date, timedelta
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
IS_START      = pd.Timestamp("2023-01-01")   # tăiem date vechi (pre-2023)
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

# ════════════════════════════════════════════════════════════════════════════
# NEWS CALENDAR (same as ny_v2)
# ════════════════════════════════════════════════════════════════════════════
# ── Economic Calendar (real dates din historical_news.csv) ────────────────
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
except Exception as _e:
    print(f"   ⚠️  Calendar load error: {_e}")
    FOMC_DATES = NFP_DATES = CPI_DATES = PPI_DATES = RETAIL_DATES = ISM_DATES = ANY_HIGH = set()

NEWS_DAYS = FOMC_DATES | NFP_DATES | CPI_DATES | PPI_DATES

def fomc_proximity(date_str):
    import pandas as _pd
    d = _pd.Timestamp(date_str).date()
    fomc_list = [_pd.Timestamp(x).date() for x in sorted(FOMC_DATES)]
    diffs = [abs((d - f).days) for f in fomc_list]
    return min(diffs) if diffs else 30


print("=" * 74)
print("train_quality_ny_v3.py — NY Quality Gate v3 (MTF ICT + VIX + Sweep + Regime)")
print("=" * 74)
print(f"\n   NFP: {len(NFP_DATES)} | FOMC: {len(FOMC_DATES)} | CPI: {len(CPI_DATES)}")

# ════════════════════════════════════════════════════════════════════════════
# FUNCȚIE UTILITARĂ: ICT features on a TF (shared with v6)
# ════════════════════════════════════════════════════════════════════════════

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
        if H[i-2] < L[i] and (L[i] - H[i-2]) > 0.5:
            bull_top[i] = L[i]; bull_bot[i] = H[i-2]
        if L[i-2] > H[i] and (L[i-2] - H[i]) > 0.5:
            bear_top[i] = L[i-2]; bear_bot[i] = H[i]

    in_bull   = np.zeros(n); in_bear   = np.zeros(n)
    dist_bull = np.full(n, 9.9); dist_bear = np.full(n, 9.9)
    in_ifvg_b = np.zeros(n); in_ifvg_s = np.zeros(n)
    breaker_b = np.zeros(n); breaker_s = np.zeros(n)
    rejection = np.zeros(n)

    active_bull = []; active_bear = []
    inv_bull_zones = []; inv_bear_zones = []
    bull_obs = []; bear_obs = []

    for i in range(n):
        c = C[i]; l = L[i]; h = H[i]; a = A[i]

        new_ab = []
        for top, bot, j in active_bull:
            if i - j > lookback: continue
            if l < bot: inv_bull_zones.append((top, bot, i))
            else: new_ab.append((top, bot, j))
        active_bull = new_ab

        new_ab2 = []
        for top, bot, j in active_bear:
            if i - j > lookback: continue
            if h > top: inv_bear_zones.append((top, bot, i))
            else: new_ab2.append((top, bot, j))
        active_bear = new_ab2

        if bull_top[i] > 0: active_bull.append((bull_top[i], bull_bot[i], i))
        if bear_top[i] > 0: active_bear.append((bear_top[i], bear_bot[i], i))

        if i >= 2:
            pb = C[i-1] - O[i-1]; pr = max(H[i-1] - L[i-1], 0.01)
            if pb > 0.55 * pr and pb > 1.0:
                bull_obs.append((C[i-1], O[i-1], i-1))
            if pb < -0.55 * pr and abs(pb) > 1.0:
                bear_obs.append((O[i-1], C[i-1], i-1))

        for top, bot, j in active_bull:
            if bot <= c <= top: in_bull[i] = 1.0
            d = min(abs(c-top), abs(c-bot)) / a
            dist_bull[i] = min(dist_bull[i], d)
        for top, bot, j in active_bear:
            if bot <= c <= top: in_bear[i] = 1.0
            d = min(abs(c-top), abs(c-bot)) / a
            dist_bear[i] = min(dist_bear[i], d)

        for top, bot, k in inv_bull_zones[-15:]:
            if i - k <= lookback * 2 and bot <= c <= top: in_ifvg_b[i] = 1.0
        for top, bot, k in inv_bear_zones[-15:]:
            if i - k <= lookback * 2 and bot <= c <= top: in_ifvg_s[i] = 1.0

        for top, bot, j in bull_obs[-20:]:
            if i - j <= lookback and c < min(bot, O[j]) - a * 0.05:
                if abs(c-top)/a < 0.8 or abs(c-bot)/a < 0.8: breaker_s[i] = 1.0
        for top, bot, j in bear_obs[-20:]:
            if i - j <= lookback and c > max(top, O[j]) + a * 0.05:
                if abs(c-top)/a < 0.8 or abs(c-bot)/a < 0.8: breaker_b[i] = 1.0

        if i >= 2:
            wu = H[i-1]-max(C[i-1],O[i-1]); wd = min(C[i-1],O[i-1])-L[i-1]
            bz = abs(C[i-1]-O[i-1])
            if wu > 2.5*max(bz,0.5) and wu > a*0.3:
                rt = H[i-1]; rb = max(C[i-1],O[i-1])
                if abs(c-rt)/a < 0.6 or abs(c-rb)/a < 0.6: rejection[i] = 1.0
            if wd > 2.5*max(bz,0.5) and wd > a*0.3:
                rt = min(C[i-1],O[i-1]); rb = L[i-1]
                if abs(c-rt)/a < 0.6 or abs(c-rb)/a < 0.6: rejection[i] = 1.0

    return pd.DataFrame({
        'in_bull': in_bull, 'in_bear': in_bear,
        'dist_bull': np.clip(dist_bull, 0, 9.9),
        'dist_bear': np.clip(dist_bear, 0, 9.9),
        'in_ifvg_b': in_ifvg_b, 'in_ifvg_s': in_ifvg_s,
        'breaker_b': breaker_b, 'breaker_s': breaker_s,
        'rejection': rejection,
    }, index=df_tf.index)


def compute_mtf_features(conn, setup_dates: list) -> pd.DataFrame:
    min_d = min(setup_dates); max_d = max(setup_dates)
    warmup_start = (pd.Timestamp(min_d) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    print(f"   Loading 1-min data: {warmup_start} → {max_d} ...")
    df1m = pd.read_sql(f"""
        SELECT timestamp, open, high, low, close, atr_14
        FROM market_data
        WHERE timestamp >= '{warmup_start} 00:00:00'
          AND timestamp <= '{max_d} 23:59:59'
        ORDER BY timestamp
    """, conn)
    df1m['ts'] = pd.to_datetime(df1m['timestamp'])
    df1m = df1m.set_index('ts')
    df1m.rename(columns={'atr_14': 'atr'}, inplace=True)
    df1m['atr'] = df1m['atr'].ffill().fillna(9.0)
    print(f"   1-min bars: {len(df1m):,}")

    all_features = pd.DataFrame(index=df1m.index)
    for tf_label, tf_rule, lookback in [
        ('5m', '5min', 25), ('15m', '15min', 20),
        ('1h', '1h', 20), ('4h', '4h', 15),
    ]:
        print(f"   Computing ICT on {tf_label} ...")
        df_tf = df1m.resample(tf_rule, label='left', closed='left').agg(
            open=('open','first'), high=('high','max'),
            low=('low','min'), close=('close','last'), atr=('atr','last')
        ).dropna(subset=['open'])
        df_tf['atr'] = df_tf['atr'].ffill().fillna(9.0)
        ict = compute_ict_on_tf(df_tf, lookback=lookback)
        ict_ffill = ict.reindex(df1m.index, method='ffill')
        for col in ict.columns:
            all_features[f'{col}_{tf_label}'] = ict_ffill[col]
        print(f"     {tf_label}: {len(df_tf):,} bars OK")

    all_features = all_features.fillna(0.0)
    all_features['ts_str'] = all_features.index.strftime('%Y-%m-%d %H:%M:%S')
    print(f"   MTF computed: {all_features.shape[1]-1} cols × {len(all_features):,} rows")
    return all_features


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: Load backtest CSV — NY session
# ════════════════════════════════════════════════════════════════════════════
print("\n▶  [1/7] Încărcare backtest CSV (NY session) ...")
df_csv = pd.read_csv(CSV_PATH)
df_ny  = df_csv[df_csv['session'] == 'NY'].copy()
df_ny['ts']          = pd.to_datetime(df_ny['timestamp'])
df_ny['trail']       = (df_ny['exit_reason'] == 'TRAIL').astype(int)
df_ny['hour_utc']    = df_ny['ts'].dt.hour
df_ny['day_of_week'] = df_ny['ts'].dt.dayofweek
df_ny['dir_short']   = (df_ny['direction'] == 'SHORT').astype(float)
df_ny['ts_str']      = df_ny['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_ny['date_str']    = df_ny['ts'].dt.strftime('%Y-%m-%d')
df_ny['year']        = df_ny['ts'].dt.year
df_ny = df_ny[df_ny['ts'] >= IS_START].copy()   # tăiem date pre-2023

print(f"   NY trades: {len(df_ny):,}  |  Trail: {df_ny['trail'].sum():,} ({df_ny['trail'].mean()*100:.1f}%)")
for yr in sorted(df_ny['year'].unique()):
    sub = df_ny[df_ny['year'] == yr]
    print(f"   {yr}: {len(sub):,} trades, trail={sub['trail'].mean()*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: Pre-compute session stats (SQL) — same as ny_v2
# ════════════════════════════════════════════════════════════════════════════
print("\n▶  [2/7] Pre-compute session stats (SQL) ...")
conn        = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=60)
setup_dates = sorted(df_ny['date_str'].unique())
dates_sql   = "','".join(setup_dates)

session_sql = f"""
SELECT
    date(timestamp)                                     AS date,
    MAX(CASE WHEN strftime('%H',timestamp) BETWEEN '07' AND '09' THEN high END)  AS lon_hi,
    MIN(CASE WHEN strftime('%H',timestamp) BETWEEN '07' AND '09' THEN low  END)  AS lon_lo,
    MAX(CASE WHEN strftime('%H',timestamp) = '09'         THEN close END)        AS lon_close,
    MAX(CASE WHEN strftime('%H:%M',timestamp) = '13:00'   THEN open  END)        AS ny_open,
    MAX(CASE WHEN strftime('%H:%M',timestamp) BETWEEN '13:00' AND '13:14' THEN high END) AS ny15_hi,
    MIN(CASE WHEN strftime('%H:%M',timestamp) BETWEEN '13:00' AND '13:14' THEN low  END) AS ny15_lo,
    AVG(CASE WHEN strftime('%H',timestamp) BETWEEN '07' AND '09' THEN atr_14 END) AS lon_avg_atr,
    MAX(CASE WHEN strftime('%H',timestamp) BETWEEN '00' AND '06' THEN high END)  AS asia_hi,
    MIN(CASE WHEN strftime('%H',timestamp) BETWEEN '00' AND '06' THEN low  END)  AS asia_lo
FROM market_data
WHERE date(timestamp) IN ('{dates_sql}')
GROUP BY date(timestamp)
"""
sess_df = pd.read_sql(session_sql, conn)
sess_df['date'] = sess_df['date'].astype(str)
sess_map = {row['date']: row for _, row in sess_df.iterrows()}
print(f"   Session stats: {len(sess_df):,} zile")

# Previous day trading map
all_td = pd.read_sql("SELECT DISTINCT date(timestamp) as date FROM market_data ORDER BY date", conn)['date'].tolist()
date_to_idx  = {d: i for i, d in enumerate(all_td)}
prev_day_map = {}
for d in setup_dates:
    idx = date_to_idx.get(d, -1)
    if idx > 0: prev_day_map[d] = all_td[idx - 1]

prev_days = list(set(prev_day_map.values()))
prev_ny_dir = {}
if prev_days:
    prev_sql = "','".join(prev_days)
    prev_ny  = pd.read_sql(f"""
        SELECT date(timestamp) as date,
               MAX(CASE WHEN strftime('%H:%M',timestamp) = '13:00' THEN open  END) as ny_o,
               MAX(CASE WHEN strftime('%H',timestamp) = '14'       THEN close END) as ny_c
        FROM market_data
        WHERE date(timestamp) IN ('{prev_sql}')
        GROUP BY date(timestamp)
    """, conn)
    for _, r in prev_ny.iterrows():
        if pd.notna(r['ny_o']) and pd.notna(r['ny_c']) and r['ny_o'] > 0:
            diff = float(r['ny_c']) - float(r['ny_o'])
            prev_ny_dir[r['date']] = 1 if diff > 1 else (-1 if diff < -1 else 0)

# Weekly profile (same as ny_v2)
all_needed     = list(set(setup_dates + prev_days))
all_needed_sql = "','".join(all_needed)
daily_sql = f"""
SELECT date(timestamp) as date,
       MAX(high) as day_hi, MIN(low) as day_lo,
       AVG(atr_14) as avg_atr, AVG(adx_14) as avg_adx, AVG(hurst) as avg_hurst
FROM market_data
WHERE date(timestamp) IN ('{all_needed_sql}')
GROUP BY date(timestamp)
"""
daily_df = pd.read_sql(daily_sql, conn)

# ════════════════════════════════════════════════════════════════════════════
# STEP 2b: MTF ICT features
# ════════════════════════════════════════════════════════════════════════════
print("\n▶  [2b/7] Pre-compute MTF ICT features ...")
mtf_features = compute_mtf_features(conn, setup_dates)

# ════════════════════════════════════════════════════════════════════════════
# STEP 2c: VIX proxy + rolling regime (daily stats)
# ════════════════════════════════════════════════════════════════════════════
print("\n▶  [2c/7] VIX proxy + rolling regime ...")

# Build daily regime from daily_df (already have avg_atr, avg_adx, avg_hurst)
daily_df['date']    = daily_df['date'].astype(str)
daily_df['date_dt'] = pd.to_datetime(daily_df['date'])
daily_df            = daily_df.sort_values('date').reset_index(drop=True)

# Also need daily_range: get from a broader query
dr_sql = f"""
SELECT date(timestamp) as date, (MAX(high)-MIN(low)) as daily_range
FROM market_data
WHERE date(timestamp) >= date('{setup_dates[0]}', '-30 days')
  AND date(timestamp) <= '{setup_dates[-1]}'
GROUP BY date(timestamp)
ORDER BY date
"""
dr_df = pd.read_sql(dr_sql, conn)
conn.close()

dr_df['date'] = dr_df['date'].astype(str)
daily_df = daily_df.merge(dr_df, on='date', how='left')
daily_df['avg_atr'] = daily_df['avg_atr'].ffill().fillna(9.0)
daily_df['daily_range'] = daily_df['daily_range'].fillna(daily_df['avg_atr'] * 2)
daily_df['range_atr_ratio']  = daily_df['daily_range'] / daily_df['avg_atr'].clip(lower=1)
daily_df['vix_proxy_5d']     = daily_df['range_atr_ratio'].rolling(5, min_periods=2).mean().shift(1)
daily_df['vix_proxy_20d']    = daily_df['range_atr_ratio'].rolling(20, min_periods=5).mean().shift(1)
daily_df['vol_regime']       = (daily_df['vix_proxy_5d'] /
                                daily_df['vix_proxy_20d'].clip(lower=0.5)).clip(upper=3)
daily_df['vol_high']         = (daily_df['vol_regime'] > 1.2).astype(float)
daily_df['vol_low']          = (daily_df['vol_regime'] < 0.8).astype(float)
daily_df['adx_10d_mean']     = daily_df['avg_adx'].rolling(10, min_periods=3).mean().shift(1)
daily_df['hurst_20d_mean']   = daily_df['avg_hurst'].rolling(20, min_periods=5).mean().shift(1)
daily_df['atr_5d']           = daily_df['avg_atr'].rolling(5, min_periods=2).mean().shift(1)
daily_df['atr_10d']          = daily_df['avg_atr'].rolling(10, min_periods=3).mean().shift(1)
daily_df['atr_trend']        = (daily_df['atr_5d'] / daily_df['atr_10d'].clip(lower=1)).clip(upper=3)
daily_df = daily_df.fillna(method='ffill').fillna(1.0)
daily_dict = {r['date']: r for _, r in daily_df.iterrows()}

# Weekly profile (identical to ny_v2)
daily_df2 = daily_df[daily_df['date'].isin(all_needed)].copy()
daily_df2['iso_year'] = daily_df2['date_dt'].dt.isocalendar().year.values
daily_df2['iso_week'] = daily_df2['date_dt'].dt.isocalendar().week.values
daily_df2['yw']       = daily_df2['iso_year'].astype(str) + '_' + daily_df2['iso_week'].astype(str)
daily_df2['dow_num']  = daily_df2['date_dt'].dt.dayofweek

mon_df2 = daily_df2[daily_df2['dow_num']==0][['yw','day_hi','day_lo']].rename(
    columns={'day_hi':'mon_hi','day_lo':'mon_lo'})
daily_df2 = daily_df2.merge(mon_df2, on='yw', how='left')
wk_ext2   = daily_df2.groupby('yw').agg(wk_hi=('day_hi','max'), wk_lo=('day_lo','min')).reset_index()
wk_ext2   = wk_ext2.sort_values('yw').reset_index(drop=True)
wk_ext2['prev_wk_hi'] = wk_ext2['wk_hi'].shift(1)
wk_ext2['prev_wk_lo'] = wk_ext2['wk_lo'].shift(1)
daily_df2 = daily_df2.merge(wk_ext2[['yw','prev_wk_hi','prev_wk_lo']], on='yw', how='left')
daily_df2['wk_hi_sofar']   = daily_df2.groupby('yw')['day_hi'].cummax()
daily_df2['wk_lo_sofar']   = daily_df2.groupby('yw')['day_lo'].cummin()
daily_df2['wk_range_sofar']= daily_df2['wk_hi_sofar'] - daily_df2['wk_lo_sofar']
daily_df2['atr_5d2']       = daily_df2['avg_atr'].rolling(5, min_periods=1).mean().shift(1)
daily_dict2 = {r['date']: r for _, r in daily_df2.iterrows()}

# Rolling 5-session WR from NY trades
df_ny_sorted = df_ny.sort_values('ts').copy()
df_ny_sorted['trail_roll5'] = (
    df_ny_sorted['trail'].rolling(5, min_periods=1).mean().shift(1).fillna(0.5)
)
roll5_map = dict(zip(df_ny_sorted['ts_str'], df_ny_sorted['trail_roll5']))

# ════════════════════════════════════════════════════════════════════════════
# STEP 3: JOIN cu market_data (entry bars)
# ════════════════════════════════════════════════════════════════════════════
print("\n▶  [3/7] JOIN cu market_data (DB) ...")
conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=60)

DB_COLS = [
    'timestamp','open','high','low','close','volume',
    'atr_14','asia_hi','asia_lo','p_hi','p_lo','true_open',
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

CHUNK    = 5000
db_parts = []
ts_list  = df_ny['ts_str'].tolist()
for i in range(0, len(ts_list), CHUNK):
    chunk = ts_list[i:i+CHUNK]
    ph    = ','.join(['?'] * len(chunk))
    q     = f"SELECT {cols_str} FROM market_data WHERE timestamp IN ({ph})"
    db_parts.append(pd.read_sql(q, conn, params=chunk))
conn.close()

db = pd.concat(db_parts, ignore_index=True)
db['ts_str'] = db['timestamp']
print(f"   DB rows joined: {len(db):,} / {len(df_ny):,} ({len(db)/len(df_ny)*100:.1f}%)")

df = df_ny.merge(db.drop(columns=['timestamp']), on='ts_str', how='inner')
df = df.merge(mtf_features.drop_duplicates('ts_str')[
    ['ts_str'] + [c for c in mtf_features.columns if c != 'ts_str']
], on='ts_str', how='left')
mtf_cols = [c for c in mtf_features.columns if c != 'ts_str']
for c in mtf_cols:
    df[c] = df[c].fillna(0.0)
print(f"   Post-merge: {len(df):,} trades, {len(mtf_cols)} MTF cols added")

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
    _before = len(df)
    df = df.merge(_of_merge, on='date_str', how='left')
    for _c in _OF_COLS:
        df[_c] = df[_c].fillna(0.0)
    print(f"   Order flow features: {len(_OF_COLS)} cols mergiate pe {len(df):,} trades")
else:
    print("   ⚠️  orderflow_features.parquet nu există — skip")
    _OF_COLS = []

# ════════════════════════════════════════════════════════════════════════════
# STEP 4: Feature Engineering NY v3
# ════════════════════════════════════════════════════════════════════════════
print("\n▶  [4/7] Feature engineering NY v3 ...")

cl   = df['close'].values.astype(float)
hi   = df['high'].values.astype(float)
lo   = df['low'].values.astype(float)
op   = df['open'].values.astype(float)
vol  = np.where(df['volume'].values > 0, df['volume'].values, 1).astype(float)
atr  = np.where(df['atr_14'].values > 0, df['atr_14'].values, 9.0).astype(float)

p_hi_arr = np.where(df['p_hi'].values > 0, df['p_hi'].values, cl)
p_lo_arr = np.where(df['p_lo'].values > 0, df['p_lo'].values, cl)
true_open= np.where(df['true_open'].values > 0, df['true_open'].values, cl)
h4h      = np.where(df['h4_hi'].values > 0, df['h4_hi'].values, cl)
h4l      = np.where(df['h4_lo'].values > 0, df['h4_lo'].values, cl)
h1h      = np.where(df['h1_hi'].values > 0, df['h1_hi'].values, cl)
h1l      = np.where(df['h1_lo'].values > 0, df['h1_lo'].values, cl)
poc_arr  = np.where(df['poc_level'].values > 0, df['poc_level'].values, cl)
vwap_arr = np.where(df['vwap'].values > 0, df['vwap'].values, cl)
lw_hi_arr= np.where(df['lw_hi'].values > 0, df['lw_hi'].values, cl)
lw_lo_arr= np.where(df['lw_lo'].values > 0, df['lw_lo'].values, cl)

date_arr = df['date_str'].values
_min_abs = df['ts'].dt.hour.values * 60 + df['ts'].dt.minute.values  # minute de la miezul nopții UTC

def _sess(d, key, fallback):
    r = sess_map.get(d)
    if r is None: return fallback
    v = r.get(key) if isinstance(r, dict) else getattr(r, key, fallback)
    return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else fallback

lon_hi_arr  = np.array([_sess(d, 'lon_hi',   cl[i]) for i, d in enumerate(date_arr)])
lon_lo_arr  = np.array([_sess(d, 'lon_lo',   cl[i]) for i, d in enumerate(date_arr)])
lon_cls_arr = np.array([_sess(d, 'lon_close',cl[i]) for i, d in enumerate(date_arr)])
ny_open_arr = np.array([_sess(d, 'ny_open',  cl[i]) for i, d in enumerate(date_arr)])
ny15hi_arr  = np.array([_sess(d, 'ny15_hi',  cl[i]) for i, d in enumerate(date_arr)])
ny15lo_arr  = np.array([_sess(d, 'ny15_lo',  cl[i]) for i, d in enumerate(date_arr)])
asia_hi_arr = np.where(df['asia_hi'].values > 0, df['asia_hi'].values,
                        np.array([_sess(d, 'asia_hi', cl[i]) for i, d in enumerate(date_arr)]))
asia_lo_arr = np.where(df['asia_lo'].values > 0, df['asia_lo'].values,
                        np.array([_sess(d, 'asia_lo', cl[i]) for i, d in enumerate(date_arr)]))

def _dd2(d, key, fallback=0.0):
    r = daily_dict2.get(d)
    if r is None: return fallback
    v = r[key] if isinstance(r, dict) else getattr(r, key, fallback)
    return float(v) if v is not None and pd.notna(v) else fallback

def _dr2(d, key, fallback=1.0):
    r = daily_dict.get(d)
    if r is None: return fallback
    v = r[key] if isinstance(r, dict) else getattr(r, key, fallback)
    return float(v) if v is not None and pd.notna(v) else fallback

prev_wk_hi_arr  = np.array([_dd2(d, 'prev_wk_hi', cl[i]) for i, d in enumerate(date_arr)])
prev_wk_lo_arr  = np.array([_dd2(d, 'prev_wk_lo', cl[i]) for i, d in enumerate(date_arr)])
mon_hi_arr      = np.array([_dd2(d, 'mon_hi', 0.0) for d in date_arr])
mon_lo_arr      = np.array([_dd2(d, 'mon_lo', 0.0) for d in date_arr])
wk_hi_sf_arr    = np.array([_dd2(d, 'wk_hi_sofar', cl[i]) for i, d in enumerate(date_arr)])
wk_lo_sf_arr    = np.array([_dd2(d, 'wk_lo_sofar', cl[i]) for i, d in enumerate(date_arr)])
wk_range_sf_arr = np.array([_dd2(d, 'wk_range_sofar', atr[i]) for i, d in enumerate(date_arr)])
atr_5d_arr      = np.array([_dd2(d, 'atr_5d2', atr[i]) for i, d in enumerate(date_arr)])
dow_num_arr     = np.array([_dd2(d, 'dow_num', 2.0) for d in date_arr])

is_fomc_arr   = np.array([d in FOMC_DATES   for d in date_arr], dtype=float)
is_nfp_arr    = np.array([d in NFP_DATES    for d in date_arr], dtype=float)
is_cpi_arr    = np.array([d in CPI_DATES    for d in date_arr], dtype=float)
is_ppi_arr    = np.array([d in PPI_DATES    for d in date_arr], dtype=float)
is_retail_arr = np.array([d in RETAIL_DATES for d in date_arr], dtype=float)
is_ism_arr    = np.array([d in ISM_DATES    for d in date_arr], dtype=float)
is_any_hi_arr = np.array([d in ANY_HIGH     for d in date_arr], dtype=float)
is_news_arr   = np.array([d in NEWS_DAYS    for d in date_arr], dtype=float)
fomc_prox     = np.array([fomc_proximity(d) for d in date_arr], dtype=float)
# Pre/post release: NFP/CPI/PPI la 8:30 AM ET = 13:30 UTC = 810 min
# FOMC la 14:00 ET = 19:00 UTC → toată fereastra NY quality gate (13:00-14:00 UTC) e pre-anunț
is_pre_nfp_arr  = np.array([(d in NFP_DATES   and _min_abs[i] < 810) for i,d in enumerate(date_arr)], dtype=float)
is_post_nfp_arr = np.array([(d in NFP_DATES   and _min_abs[i] >= 810) for i,d in enumerate(date_arr)], dtype=float)
is_pre_cpi_arr  = np.array([(d in CPI_DATES   and _min_abs[i] < 810) for i,d in enumerate(date_arr)], dtype=float)
is_post_cpi_arr = np.array([(d in CPI_DATES   and _min_abs[i] >= 810) for i,d in enumerate(date_arr)], dtype=float)
is_pre_ppi_arr  = np.array([(d in PPI_DATES   and _min_abs[i] < 810) for i,d in enumerate(date_arr)], dtype=float)
is_post_ppi_arr = np.array([(d in PPI_DATES   and _min_abs[i] >= 810) for i,d in enumerate(date_arr)], dtype=float)
is_fomc_wait_arr = is_fomc_arr.copy()  # FOMC la 19:00 UTC → toată fereastra NY e pre-anunț
prev_ny_push_arr = np.array([
    prev_ny_dir.get(prev_day_map.get(d, ''), 0) for d in date_arr], dtype=float)

trade_dir = np.where(df['dir_short'].values == 0, 1.0, -1.0)

def clip(x, c=CLIP):
    return np.clip(np.where(np.isfinite(x), x, 0.0), -c, c)
def safe_norm(num, denom, c=CLIP):
    return clip(np.where(np.abs(denom) > 0.01, num / denom, 0.0), c)

feat = pd.DataFrame()

# ── 1. TEMPORAL ─────────────────────────────────────────────────────────────
feat['dir_short']         = df['dir_short'].values
feat['hour_utc']          = df['hour_utc'].values.astype(float)
feat['min_in_ny']         = np.clip((df['hour_utc'].values - 13) * 60, 0, 120).astype(float)
feat['session_pct']       = feat['min_in_ny'].values / 120.0
feat['day_of_week']       = dow_num_arr
feat['is_monday']         = (dow_num_arr == 0).astype(float)
feat['is_tuesday']        = (dow_num_arr == 1).astype(float)
feat['is_wednesday']      = (dow_num_arr == 2).astype(float)
feat['is_thursday']       = (dow_num_arr == 3).astype(float)
feat['is_friday']         = (dow_num_arr == 4).astype(float)
feat['month']             = df['ts'].dt.month.values.astype(float)
feat['year_norm']         = (df['ts'].dt.year.values.astype(float) - 2023.0) / 2.0

# ── 2. NEWS CALENDAR ─────────────────────────────────────────────────────────
feat['is_fomc_day']       = is_fomc_arr
feat['is_nfp_day']        = is_nfp_arr
feat['is_cpi_day']        = is_cpi_arr
feat['is_ppi_day']        = is_ppi_arr
feat['is_retail_day']     = is_retail_arr
feat['is_ism_day']        = is_ism_arr
feat['is_any_high_day']   = is_any_hi_arr
feat['is_news_day']       = is_news_arr
feat['fomc_proximity']    = clip(fomc_prox / 14.0)
# Pre/post release timing (diferențiază contextul ÎNAINTE vs DUPĂ știre)
feat['is_pre_nfp']        = is_pre_nfp_arr
feat['is_post_nfp']       = is_post_nfp_arr
feat['is_pre_cpi']        = is_pre_cpi_arr
feat['is_post_cpi']       = is_post_cpi_arr
feat['is_pre_ppi']        = is_pre_ppi_arr
feat['is_post_ppi']       = is_post_ppi_arr
feat['is_fomc_wait']      = is_fomc_wait_arr  # tot sesiunea NY e pre-FOMC

# ── 3. NY OPEN DRIVE ─────────────────────────────────────────────────────────
ny15_range = ny15hi_arr - ny15lo_arr
ny15_up    = ny15hi_arr - ny_open_arr
ny15_dn    = ny_open_arr - ny15lo_arr
feat['ny_open_drive_bull']   = (ny15_up > ny15_dn + atr * 0.3).astype(float)
feat['ny_open_drive_bear']   = (ny15_dn > ny15_up + atr * 0.3).astype(float)
feat['ny_open_drive_neutral']= ((feat['ny_open_drive_bull'].values + feat['ny_open_drive_bear'].values) == 0).astype(float)
feat['ny15_range_atr']       = safe_norm(ny15_range, atr)
feat['drive_aligned_dir']    = np.where(df['dir_short'].values == 0,
                                         feat['ny_open_drive_bull'].values,
                                         feat['ny_open_drive_bear'].values)

# ── 4. GAP NY OPEN VS LON CLOSE ──────────────────────────────────────────────
gap_raw = ny_open_arr - lon_cls_arr
feat['gap_vs_lon_close_atr'] = safe_norm(gap_raw, atr)
feat['gap_up']               = (gap_raw > atr * 0.2).astype(float)
feat['gap_down']             = (gap_raw < -atr * 0.2).astype(float)
feat['gap_aligned']          = np.where(df['dir_short'].values == 0,
                                         feat['gap_up'].values, feat['gap_down'].values)

# ── 5. LONDON SESSION CONTEXT ─────────────────────────────────────────────────
valid_lon  = (lon_hi_arr > lon_lo_arr) & (lon_hi_arr > 0)
lon_range  = np.where(valid_lon, lon_hi_arr - lon_lo_arr, atr)
feat['dist_lon_hi_atr']   = safe_norm(cl - lon_hi_arr, atr)
feat['dist_lon_lo_atr']   = safe_norm(cl - lon_lo_arr, atr)
feat['lon_range_atr']     = clip(safe_norm(lon_range, atr), 20)
feat['swept_lon_hi']      = ((cl > lon_hi_arr) & valid_lon).astype(float)
feat['swept_lon_lo']      = ((cl < lon_lo_arr) & valid_lon).astype(float)
feat['lon_midpoint']      = safe_norm(cl - (lon_hi_arr + lon_lo_arr) / 2, atr)
feat['lon_range_narrow']  = (lon_range < atr).astype(float)
sweep_depth_ny = np.where(df['dir_short'].values == 0, cl - lon_lo_arr, lon_hi_arr - cl)
feat['sweep_depth_atr']   = safe_norm(sweep_depth_ny, atr)
feat['deep_sweep']        = (sweep_depth_ny > atr * 0.5).astype(float)
ny_in_lon = safe_norm(ny_open_arr - lon_lo_arr, lon_range)
feat['ny_open_in_lon']    = clip(ny_in_lon)
feat['ny_open_above_lon_mid'] = (ny_in_lon > 0.5).astype(float)
feat['ny_open_below_lon_mid'] = (ny_in_lon < 0.5).astype(float)
feat['lon_close_vs_mid']  = safe_norm(lon_cls_arr - (lon_hi_arr + lon_lo_arr) / 2, atr)
feat['lon_closed_weak']   = (np.abs(lon_cls_arr - (lon_hi_arr + lon_lo_arr) / 2) < atr * 0.3).astype(float)
# LON direction + range quality (features lipsă față de ny_v2)
lon_mid_arr                  = (lon_hi_arr + lon_lo_arr) / 2
feat['lon_dir_explicit']     = np.where(lon_cls_arr > lon_mid_arr, 1.0, -1.0)   # +1=bull, -1=bear
feat['lon_dir_aligned']      = (feat['lon_dir_explicit'].values == trade_dir).astype(float)
feat['lon_dir_opposite']     = (feat['lon_dir_explicit'].values != trade_dir).astype(float)
feat['lon_range_vs_atr5d']   = clip(safe_norm(lon_range, np.where(atr_5d_arr > 0, atr_5d_arr, atr)))
feat['lon_big_day']          = (lon_range > atr * 1.5).astype(float)
feat['lon_small_day']        = (lon_range < atr * 0.7).astype(float)
feat['ny_open_dist_lon_mid'] = safe_norm(ny_open_arr - lon_mid_arr, atr)  # signed: + = NY opens above LON mid

# ── 6. ASIA CONTEXT ──────────────────────────────────────────────────────────
valid_asia = (asia_hi_arr > asia_lo_arr) & (asia_hi_arr > 0)
feat['dist_asia_hi_atr']  = safe_norm(cl - asia_hi_arr, atr)
feat['dist_asia_lo_atr']  = safe_norm(cl - asia_lo_arr, atr)
feat['asia_range_atr']    = clip(safe_norm(asia_hi_arr - asia_lo_arr, atr), 20)
feat['swept_asia_hi']     = ((cl > asia_hi_arr) & valid_asia).astype(float)
feat['swept_asia_lo']     = ((cl < asia_lo_arr) & valid_asia).astype(float)

# Triple session alignment
asia_dir  = np.where(lon_hi_arr > (asia_hi_arr + asia_lo_arr) / 2, 1.0, -1.0)
lon_dir   = np.where(lon_cls_arr > (lon_hi_arr + lon_lo_arr) / 2, 1.0, -1.0)
feat['lon_vs_asia_aligned'] = (asia_dir == trade_dir).astype(float)
feat['triple_sess_aligned'] = ((asia_dir == trade_dir) & (lon_dir != trade_dir)).astype(float)

# ── 7. PREVIOUS NY PUSH ───────────────────────────────────────────────────────
feat['prev_ny_push_dir']  = prev_ny_push_arr
feat['prev_ny_aligned']   = (prev_ny_push_arr == trade_dir).astype(float)
feat['prev_ny_opposite']  = (prev_ny_push_arr == -trade_dir).astype(float)

# ── 8. PREVIOUS DAY CONTEXT ──────────────────────────────────────────────────
feat['dist_pdh_atr']      = safe_norm(df['dist_pdh'].values, atr)
feat['dist_pdl_atr']      = safe_norm(df['dist_pdl'].values, atr)
feat['above_true_open']   = (cl > true_open).astype(float)
feat['dist_true_open']    = safe_norm(cl - true_open, atr)
feat['dist_pdh_direct']   = safe_norm(cl - p_hi_arr, atr)
feat['dist_pdl_direct']   = safe_norm(cl - p_lo_arr, atr)

# ── 9. HTF BIAS ──────────────────────────────────────────────────────────────
h4_mid = (h4h + h4l) / 2; h1_mid = (h1h + h1l) / 2
feat['h4_bias']           = safe_norm(h4_mid - cl, atr)
feat['h1_bias']           = safe_norm(h1_mid - cl, atr)
feat['h4_h1_aligned']     = (np.sign(feat['h4_bias'].values) == np.sign(feat['h1_bias'].values)).astype(float)
feat['h4_bias_aligned']   = np.where(trade_dir == 1,
                                      (cl < h4_mid).astype(float), (cl > h4_mid).astype(float))
feat['h4_bullish_struct'] = (h4h > p_hi_arr * 0.999).astype(float)
feat['h4_bearish_struct'] = (h4l < p_lo_arr * 1.001).astype(float)
feat['h4_struct_aligned'] = np.where(trade_dir == 1,
                                      feat['h4_bullish_struct'].values, feat['h4_bearish_struct'].values)

# ── 10. WEEKLY PREMIUM / DISCOUNT ────────────────────────────────────────────
valid_pw    = (prev_wk_hi_arr > prev_wk_lo_arr) & (prev_wk_hi_arr > 0)
pw_range    = np.where(valid_pw, prev_wk_hi_arr - prev_wk_lo_arr, atr * 10)
wk_prem_pct = np.where(valid_pw, (cl - prev_wk_lo_arr) / pw_range - 0.5, 0.0)
feat['weekly_premium_pct']   = clip(wk_prem_pct)
feat['weekly_prem_direction']= np.sign(wk_prem_pct)
feat['in_weekly_premium']    = (wk_prem_pct > 0.1).astype(float)
feat['in_weekly_discount']   = (wk_prem_pct < -0.1).astype(float)
feat['weekly_prem_aligned']  = np.where(trade_dir == 1,
                                         feat['in_weekly_discount'].values, feat['in_weekly_premium'].values)
feat['dist_prev_wk_hi']      = safe_norm(cl - prev_wk_hi_arr, atr)
feat['dist_prev_wk_lo']      = safe_norm(cl - prev_wk_lo_arr, atr)
feat['lw_range_atr']         = clip(safe_norm(lw_hi_arr - lw_lo_arr, atr), 20)
feat['dist_lw_hi']           = safe_norm(cl - lw_hi_arr, atr)
feat['dist_lw_lo']           = safe_norm(cl - lw_lo_arr, atr)

# ── 11. WEEKLY PROFILE ────────────────────────────────────────────────────────
feat['week_range_so_far']    = clip(safe_norm(wk_range_sf_arr, atr), 20)
feat['atr_ratio_week']       = clip(safe_norm(atr, np.where(atr_5d_arr > 0, atr_5d_arr, atr)), 5)
feat['week_hi_taken']        = (cl > wk_hi_sf_arr * 0.998).astype(float)
feat['week_lo_taken']        = (cl < wk_lo_sf_arr * 1.002).astype(float)
mon_range_arr = np.where(mon_hi_arr > mon_lo_arr, mon_hi_arr - mon_lo_arr, 0.0)
feat['monday_range_pt']      = clip(safe_norm(mon_range_arr, atr), 10)
feat['monday_consol']        = (mon_range_arr < atr).astype(float)
feat['tuesday_rev_ctx']      = (feat['is_tuesday'].values *
                                (feat['week_hi_taken'].values + feat['week_lo_taken'].values).clip(0,1))
feat['wednesday_rev_ctx']    = feat['is_wednesday'].values * (
    np.abs(feat['weekly_premium_pct'].values.clip(-1,1)) > 0.3).astype(float)
feat['thursday_consol']      = feat['is_thursday'].values * feat['monday_consol'].values

# ── 12. VOLUME PROFILE ───────────────────────────────────────────────────────
feat['inside_va']            = df['inside_va'].fillna(0).values.astype(float)
feat['dist_poc_atr']         = safe_norm(df['dist_poc'].values, atr)
feat['dist_vwap_atr']        = safe_norm(df['dist_vwap'].values, atr)
feat['vah_dist']             = safe_norm(cl - df['vah'].fillna(0).values.astype(float), atr)
feat['val_dist']             = safe_norm(cl - df['val'].fillna(0).values.astype(float), atr)

# ── 13. ICT SIGNALS (1m) ─────────────────────────────────────────────────────
feat['has_displacement']     = df['has_displacement'].fillna(0).values.astype(float)
feat['fvg_up']               = df['fvg_up'].fillna(0).values.astype(float)
feat['fvg_down']             = df['fvg_down'].fillna(0).values.astype(float)
feat['is_smt_bearish']       = df['is_smt_bearish'].fillna(0).values.astype(float)
feat['is_smt_bullish']       = df['is_smt_bullish'].fillna(0).values.astype(float)
feat['smt_aligned']          = np.where(trade_dir == 1, feat['is_smt_bullish'].values, feat['is_smt_bearish'].values)
body = cl - op
feat['body_bear']            = (body < 0).astype(float)
feat['body_pct']             = clip(np.abs(body) / np.maximum(hi - lo, 0.01), 2)
feat['ob_proxy_bull']        = ((body > 0) & (df['has_displacement'].fillna(0).values > 0)).astype(float)
feat['ob_proxy_bear']        = ((body < 0) & (df['has_displacement'].fillna(0).values > 0)).astype(float)
feat['ob_aligned']           = np.where(trade_dir == 1, feat['ob_proxy_bull'].values, feat['ob_proxy_bear'].values)

# ── 14. MOMENTUM / REGIME (ny_v2) ─────────────────────────────────────────────
feat['hurst']                = df['hurst'].fillna(0.5).values.astype(float)
feat['adx_14']               = df['adx_14'].fillna(20).values.astype(float)
feat['adx_strong']           = (df['adx_14'].fillna(20).values > 25).astype(float)
feat['acf_lag1']             = df['acf_lag1'].fillna(0).values.astype(float)
feat['acf_lag5']             = df['acf_lag5'].fillna(0).values.astype(float)
feat['fisher_transform']     = df['fisher_transform'].fillna(0).values.astype(float)
feat['fisher_extreme']       = (np.abs(df['fisher_transform'].fillna(0).values) > 2.0).astype(float)
feat['fft_cycle']            = df['fft_cycle'].fillna(0).values.astype(float)
feat['kalman_smooth']        = df['kalman_smooth'].fillna(0).values.astype(float)
feat['kalman_noise']         = df['kalman_noise'].fillna(0).values.astype(float)
garch_raw                    = df['garch_vol'].fillna(0).values.astype(float)
feat['garch_vol_atr']        = clip(np.where(atr > 0, garch_raw * cl / atr, 1.0), 5)
feat['sample_entropy']       = df['sample_entropy'].fillna(2.0).values.astype(float)

pd_range_arr = np.maximum(p_hi_arr - p_lo_arr, 1.0)
day_hi_arr   = np.array([_dd2(d, 'day_hi', cl[i]) for i, d in enumerate(date_arr)])
day_lo_arr   = np.array([_dd2(d, 'day_lo', cl[i]) for i, d in enumerate(date_arr)])
day_range_arr= np.maximum(day_hi_arr - day_lo_arr, 1.0)
feat['inside_day']           = ((day_hi_arr < p_hi_arr) & (day_lo_arr > p_lo_arr)).astype(float)
feat['trend_day']            = (day_range_arr > 1.5 * atr).astype(float)
feat['range_day']            = ((day_range_arr > atr*0.5) & (day_range_arr < atr*1.5)).astype(float)

# ── 15. ORDERFLOW ────────────────────────────────────────────────────────────
bar_delta = df['bar_delta'].fillna(0).values.astype(float)
feat['bar_delta_norm']       = clip(bar_delta / np.maximum(vol, 1), 1)
feat['cum_delta_norm']       = clip(df['cum_delta'].fillna(0).values / np.maximum(vol, 1), 1)
feat['buy_sell_ratio']       = clip(df['bar_buy_vol'].fillna(0).values /
                                     np.maximum(df['bar_sell_vol'].fillna(0).values, 1), 5)
feat['absorption_score']     = df['absorption_score'].fillna(0).values.astype(float)
feat['stacked_bull']         = df['stacked_bull'].fillna(0).values.astype(float)
feat['stacked_bear']         = df['stacked_bear'].fillna(0).values.astype(float)
feat['of_doi']               = df['of_doi'].fillna(0).values.astype(float)

# ── 16. CANDLESTICK ──────────────────────────────────────────────────────────
wick_up   = hi - np.maximum(cl, op)
wick_down = np.minimum(cl, op) - lo
feat['sweep_wick_atr']       = safe_norm(np.maximum(wick_up, wick_down), atr)
feat['confidence']           = df['confidence'].values.astype(float)
feat['atr_entry']            = atr
feat['atr_vs_10d']           = clip(df.groupby(df['ts'].dt.date)['atr_14'].transform('mean').values /
                                     np.where(atr > 0, atr, 1), 3)

# ── 17. INTERACȚIUNI (ny_v2) ─────────────────────────────────────────────────
feat['dir_x_adx']            = feat['dir_short'].values * feat['adx_14'].values / 100.0
feat['dir_x_hurst']          = feat['dir_short'].values * feat['hurst'].values
feat['confidence_x_adx']     = feat['confidence'].values * feat['adx_strong'].values
feat['year_x_adx']           = feat['year_norm'].values * feat['adx_14'].values / 100.0
feat['year_x_hurst']         = feat['year_norm'].values * feat['hurst'].values
feat['news_x_dir']           = feat['is_news_day'].values * feat['dir_short'].values
feat['swept_lon_x_dir']      = np.where(df['dir_short'].values == 0,
                                          feat['swept_lon_lo'].values, feat['swept_lon_hi'].values)
feat['fomc_x_weekly_prem']   = feat['is_fomc_day'].values * feat['weekly_premium_pct'].values
feat['drive_x_swept']        = feat['drive_aligned_dir'].values * feat['swept_lon_x_dir'].values

# ══════════════════════════════════════════════════════
#  NEW SECTION A: MTF ICT FEATURES
# ══════════════════════════════════════════════════════
for tf_label in ['5m', '15m', '1h', '4h']:
    in_bull   = df[f'in_bull_{tf_label}'].values
    in_bear   = df[f'in_bear_{tf_label}'].values
    dist_bull = df[f'dist_bull_{tf_label}'].values
    dist_bear = df[f'dist_bear_{tf_label}'].values
    in_ifvg_b = df[f'in_ifvg_b_{tf_label}'].values
    in_ifvg_s = df[f'in_ifvg_s_{tf_label}'].values
    brk_b     = df[f'breaker_b_{tf_label}'].values
    brk_s     = df[f'breaker_s_{tf_label}'].values
    rej       = df[f'rejection_{tf_label}'].values

    feat[f'in_bull_fvg_{tf_label}']     = in_bull
    feat[f'in_bear_fvg_{tf_label}']     = in_bear
    feat[f'dist_bull_fvg_{tf_label}']   = np.clip(dist_bull, 0, 9.9)
    feat[f'dist_bear_fvg_{tf_label}']   = np.clip(dist_bear, 0, 9.9)
    feat[f'fvg_aligned_{tf_label}']     = np.where(trade_dir == 1, in_bull, in_bear)
    feat[f'in_ifvg_{tf_label}']         = np.maximum(in_ifvg_b, in_ifvg_s)
    feat[f'ifvg_aligned_{tf_label}']    = np.where(trade_dir == 1, in_ifvg_s, in_ifvg_b)
    feat[f'breaker_aligned_{tf_label}'] = np.where(trade_dir == 1, brk_b, brk_s)
    feat[f'rejection_{tf_label}']       = rej

feat['fvg_tf_confluence'] = (
    feat['fvg_aligned_5m'].values + feat['fvg_aligned_15m'].values +
    feat['fvg_aligned_1h'].values + feat['fvg_aligned_4h'].values
)
feat['htf_fvg_aligned']  = np.maximum(feat['fvg_aligned_1h'].values, feat['fvg_aligned_4h'].values)
feat['ifvg_htf_aligned'] = np.maximum(feat['ifvg_aligned_1h'].values, feat['ifvg_aligned_4h'].values)

# ══════════════════════════════════════════════════════
#  NEW SECTION B: VIX PROXY + DXY PROXY
# ══════════════════════════════════════════════════════
feat['vix_proxy_5d']    = np.array([_dr2(d, 'vix_proxy_5d', 2.0) for d in date_arr])
feat['vix_proxy_20d']   = np.array([_dr2(d, 'vix_proxy_20d', 2.0) for d in date_arr])
feat['vol_regime']      = np.array([_dr2(d, 'vol_regime', 1.0) for d in date_arr])
feat['vol_high']        = np.array([_dr2(d, 'vol_high', 0.0) for d in date_arr])
feat['vol_low']         = np.array([_dr2(d, 'vol_low', 0.0) for d in date_arr])
feat['atr_trend']       = np.array([_dr2(d, 'atr_trend', 1.0) for d in date_arr])
feat['atr_expanding']   = (feat['atr_trend'].values > 1.15).astype(float)
feat['atr_contracting'] = (feat['atr_trend'].values < 0.85).astype(float)
feat['vol_x_fvg_1h']   = feat['vol_regime'].values * feat['fvg_aligned_1h'].values
feat['vol_x_htf_fvg']  = feat['vol_regime'].values * feat['htf_fvg_aligned'].values
# NY-specific: vol × news interaction
feat['vol_x_fomc']     = feat['vol_regime'].values * feat['is_fomc_day'].values
feat['vol_x_nfp']      = feat['vol_regime'].values * feat['is_nfp_day'].values

# ══════════════════════════════════════════════════════
#  NEW SECTION C: SWEEP QUALITY FEATURES (LON hi/lo for NY)
# ══════════════════════════════════════════════════════
# NY sweeps LON hi/lo (not Asia like LON sweeps Asia hi/lo)
sweep_level_ny = np.where(trade_dir == 1, lon_lo_arr, lon_hi_arr)
dist_to_lon_lo = np.abs(sweep_level_ny - p_lo_arr)
dist_to_lon_hi = np.abs(sweep_level_ny - p_hi_arr)
feat['equal_level_score']    = clip(1.0 - safe_norm(
    np.minimum(dist_to_lon_lo, dist_to_lon_hi), atr))
feat['sweep_depth_lon_atr']  = safe_norm(
    np.where(trade_dir == 1, lon_lo_arr - cl, cl - lon_hi_arr), atr)
feat['deep_sweep_lon']       = (feat['sweep_depth_lon_atr'].values > 0.3).astype(float)
feat['shallow_sweep']        = (np.abs(feat['sweep_depth_lon_atr'].values) < 0.1).astype(float)
feat['sweep_wick_clean']     = (feat['sweep_wick_atr'].values > 0.4).astype(float)
feat['sweep_with_disp']      = feat['sweep_wick_clean'].values * feat['has_displacement'].values
feat['sweep_quality_score']  = (feat['equal_level_score'].values +
                                 feat['deep_sweep_lon'].values +
                                 feat['sweep_wick_clean'].values +
                                 feat['fvg_aligned_15m'].values +
                                 feat['fvg_aligned_1h'].values).clip(0, 5) / 5.0
# NY-specific sweep: did NY open inside or outside LON range?
feat['ny_open_outside_lon']  = ((ny_open_arr > lon_hi_arr) | (ny_open_arr < lon_lo_arr)).astype(float)

# ══════════════════════════════════════════════════════
#  NEW SECTION D: ROLLING REGIME
# ══════════════════════════════════════════════════════
feat['rolling_5sess_wr']  = np.array([roll5_map.get(ts_str, 0.5) for ts_str in df['ts_str'].values])
feat['adx_10d_mean']      = np.array([_dr2(d, 'adx_10d_mean', 20.0) for d in date_arr])
feat['hurst_20d_mean']    = np.array([_dr2(d, 'hurst_20d_mean', 0.5) for d in date_arr])
feat['regime_trending']   = (feat['adx_10d_mean'].values > 22).astype(float)
feat['regime_hurst_trend']= (feat['hurst_20d_mean'].values > 0.52).astype(float)
feat['recent_wr_high']    = (feat['rolling_5sess_wr'].values > 0.30).astype(float)
feat['recent_wr_low']     = (feat['rolling_5sess_wr'].values < 0.12).astype(float)
feat['regime_score']      = (feat['regime_trending'].values +
                              feat['regime_hurst_trend'].values +
                              feat['vol_high'].values).clip(0, 3) / 3.0
feat['regime_x_htf_fvg']  = feat['regime_score'].values * feat['htf_fvg_aligned'].values
feat['adx_x_sweep_quality']= feat['adx_10d_mean'].values / 30.0 * feat['sweep_quality_score'].values

# NY-specific interactions
feat['fomc_x_fvg_1h']    = feat['is_fomc_day'].values * feat['fvg_aligned_1h'].values
feat['drive_x_fvg_15m']  = feat['drive_aligned_dir'].values * feat['fvg_aligned_15m'].values
feat['triple_x_fvg_1h']  = feat['triple_sess_aligned'].values * feat['fvg_aligned_1h'].values
feat['news_x_vol']        = feat['is_news_day'].values * feat['vol_regime'].values

print(f"   Total features construite: {len(feat.columns)}")


# ── NY sub-session split ──────────────────────────────────────────────────
# _min_abs deja calculat mai sus (minute de la miezul nopții UTC)
feat['is_ny_open']        = ((_min_abs >= 780) & (_min_abs < 870)).astype(float)
feat['is_ny_afternoon']   = (_min_abs >= 870).astype(float)
feat['min_in_ny_open']    = np.clip(_min_abs - 780, 0, 90).astype(float)
feat['ny_open_early']     = ((_min_abs >= 780) & (_min_abs < 810)).astype(float)
feat['fomc_x_open']       = feat['is_fomc_day'].values * feat['is_ny_open'].values
feat['news_x_open']       = feat['is_news_day'].values * feat['is_ny_open'].values
feat['nfp_pre_x_open']    = feat['is_pre_nfp'].values * feat['is_ny_open'].values    # pre-NFP în fereastra NY open
feat['nfp_post_x_open']   = feat['is_post_nfp'].values * feat['is_ny_open'].values   # post-NFP în fereastra NY open
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

X    = feat.drop(columns=['trail', 'ts', 'year', '_regime']).astype(float)
y    = feat['trail'].values
ts_  = pd.DatetimeIndex(feat['ts'].values)
yr_  = feat['year'].values

print(f"   Dataset final: {len(X):,} rânduri, {X.shape[1]} features")
print(f"   Trail rate global: {y.mean()*100:.2f}%")
print(f"   Features noi vs ny_v2: {X.shape[1] - 133} features added")

# ════════════════════════════════════════════════════════════════════════════
# STEP 5: Temporal split + sample weights
# ════════════════════════════════════════════════════════════════════════════
print("\n▶  [5/7] Temporal split & sample weights ...")

train_mask = np.array(ts_ < VAL_START)
val_mask   = np.array(ts_ >= VAL_START)
X_tr_all, y_tr_all = X[train_mask], y[train_mask]
X_val,    y_val    = X[val_mask],   y[val_mask]
yr_tr              = yr_[train_mask]
sw_tr              = np.array([YEAR_WEIGHTS.get(int(yr), 1.0) for yr in yr_tr])

print(f"   Train: {len(X_tr_all):,} | Val (2025): {len(X_val):,}")
print(f"   Train trail: {y_tr_all.mean()*100:.1f}% | Val trail: {y_val.mean()*100:.1f}%")

if len(X_val) < 50:
    print("   ⚠️  Val mic → extindem la H2-2025.")
    VAL_START  = pd.Timestamp("2024-07-01")
    train_mask = np.array(ts_ < VAL_START)
    val_mask   = np.array(ts_ >= VAL_START)
    X_tr_all, y_tr_all = X[train_mask], y[train_mask]
    X_val,    y_val    = X[val_mask],   y[val_mask]
    yr_tr  = yr_[train_mask]
    sw_tr  = np.array([YEAR_WEIGHTS.get(int(yr), 1.0) for yr in yr_tr])
    print(f"   [ADJUSTED] Train: {len(X_tr_all):,} | Val: {len(X_val):,}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 6: Regime-aware Optuna tuning
# ════════════════════════════════════════════════════════════════════════════
print("\n▶  [6/7] Regime-aware Optuna tuning ({} trials per regime) ...".format(OPTUNA_TRIALS))

SAVE_DIR = DIR.parent
MODEL_NAME = 'mario_quality_ny_v3'
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
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, _n_est),
            'max_depth':        trial.suggest_int('max_depth', 2, _max_d),
            'learning_rate':    trial.suggest_float('learning_rate', 0.003, 0.08, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', _mcw_lo, _mcw_lo * 6),
            'gamma':            trial.suggest_float('gamma', 0.0, 3.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 3.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 6.0),
            'max_delta_step':   trial.suggest_int('max_delta_step', 0, 5),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 3.0, 15.0),
        }
        smote_ratio = trial.suggest_float('smote_ratio', 0.20, 0.55)
        try:
            sm = BorderlineSMOTE(sampling_strategy=smote_ratio, random_state=42, k_neighbors=5)
            X_sm, y_sm = sm.fit_resample(X_tr_r, y_tr_r)
            n_synth    = len(X_sm) - len(X_tr_r)
            sw_sm      = np.concatenate([sw_r, np.ones(n_synth)])
        except Exception:
            X_sm, y_sm = X_tr_r, y_tr_r; sw_sm = sw_r
        mdl = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss',
                               random_state=42, n_jobs=-1, tree_method='hist',
                               early_stopping_rounds=50)
        mdl.fit(X_sm, y_sm, sample_weight=sw_sm,
              eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction='maximize')
    study.optimize(obj_regime, n_trials=OPTUNA_TRIALS, show_progress_bar=False, n_jobs=1)
    best_p = study.best_params
    best_s = best_p.pop('smote_ratio')

    try:
        sm_f = BorderlineSMOTE(sampling_strategy=best_s, random_state=42, k_neighbors=5)
        X_sm_f, y_sm_f = sm_f.fit_resample(X_tr_r, y_tr_r)
        sw_sm_f = np.concatenate([sw_r, np.ones(len(X_sm_f) - len(X_tr_r))])
    except Exception:
        X_sm_f, y_sm_f = X_tr_r, y_tr_r; sw_sm_f = sw_r

    final_m = xgb.XGBClassifier(**best_p, use_label_encoder=False, eval_metric='logloss',
                                random_state=42, n_jobs=-1, tree_method='hist',
                                early_stopping_rounds=50)
    final_m.fit(X_sm_f, y_sm_f, sample_weight=sw_sm_f, eval_set=[(X_val, y_val)], verbose=False)

    is_auc = roc_auc_score(y_tr_r, final_m.predict_proba(X_tr_r)[:, 1])  # IS real pe training
    oos_auc = roc_auc_score(y_te, final_m.predict_proba(X_te)[:, 1]) if len(y_te) > 0 else 0
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
            _rpkl = DIR.parent / f"mario_quality_ny_v3_{regime_name}_calibrated.pkl"
            import pickle as _pkl
            with open(_rpkl,'wb') as _f: _pkl.dump(mdl, _f)
            print(f"   💾 {_rpkl.name}")

if 'ALL' not in regime_models:
    print("\n⚠️ Warning: No ALL model found")

# ════════════════════════════════════════════════════════════════════════════
# STEP 7: Calibrare model ALL + salvare
# ════════════════════════════════════════════════════════════════════════════
print("\n▶  [7/7] Calibrare model ALL + salvare ...")

if 'ALL' in regime_models:
    final_model = regime_models['ALL']
    print(f"   Using ALL-regime model")
else:
    print("   ERROR: No ALL model found!")
    raise ValueError("ALL regime model not trained")

print("\n   AUC per an:")
for yr in sorted(set(yr_)):
    mask = (yr_ == yr)
    if mask.sum() < 10: continue
    p_yr  = final_model.predict_proba(X[mask])[:, 1]
    auc_yr= roc_auc_score(y[mask], p_yr)
    tag   = "IN-SAMPLE" if (ts_[mask] < VAL_START).all() else \
            ("OOS" if (ts_[mask] >= VAL_START).all() else "PARTIAL")
    print(f"   {yr}: AUC={auc_yr:.4f}  n={mask.sum():,}  trail={y[mask].mean()*100:.1f}%  [{tag}]")

for h in [13, 14]:
    mask = (ts_.hour == h)
    if mask.sum() < 10: continue
    auc_h = roc_auc_score(y[mask], final_model.predict_proba(X[mask])[:,1])
    print(f"   h{h} UTC: AUC={auc_h:.4f}  n={mask.sum():,}  trail={y[mask].mean()*100:.1f}%")

proba_val = final_model.predict_proba(X_val)[:, 1]
auc_val   = roc_auc_score(y_val, proba_val)
print(f"\n   ✅ AUC OOS (2025) = {auc_val:.4f}")

print("   Calibrare isotonică ...")
# sklearn 1.6+: cv='prefit' removed — manual isotonic calibration
from sklearn.isotonic import IsotonicRegression as _IR
_raw_val = final_model.predict_proba(X_val)[:, 1]
_ir_cal  = _IR(out_of_bounds='clip').fit(_raw_val, y_val)

cal_model = _CalModel(final_model, _ir_cal)
proba_cal = cal_model.predict_proba(X_val)[:, 1]
auc_cal   = roc_auc_score(y_val, proba_cal)
print(f"   AUC post-calibrare: {auc_cal:.4f}")

fpr, tpr, thresholds = roc_curve(y_val, proba_cal)
best_thr = float(thresholds[np.argmax(tpr - fpr)])
print(f"   Optimal threshold (Youden J): {best_thr:.3f}")

print(f"\n   Threshold analysis (val Q4-2025):")
print(f"   {'THR':>6}  {'N':>6}  {'Trail%':>8}  {'EV@RR2.0':>10}  {'EV@RR2.5':>10}  {'EV@RR3.0':>10}")
SL = 240.0  # NY SL=12pt×$20=$240
MONTHS = 3
for thr in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
    mask = proba_cal >= thr
    n_t  = mask.sum()
    if n_t < 5:
        print(f"   {thr:>6.2f}  {n_t:>6}  {'—':>8}  {'—':>10}  {'—':>10}  {'—':>10}")
        continue
    y_sub  = y_val[mask]
    tr_pct = y_sub.mean()
    ev20   = tr_pct * 480  - (1 - tr_pct) * SL
    ev25   = tr_pct * 600  - (1 - tr_pct) * SL
    ev30   = tr_pct * 720  - (1 - tr_pct) * SL
    n_mo   = n_t / MONTHS
    print(f"   {thr:>6.2f}  {n_t:>6}  {tr_pct*100:>7.1f}%  {ev20:>+10.0f}  {ev25:>+10.0f}  {ev30:>+10.0f}   (~{n_mo:.1f}/luna)")

fi      = dict(zip(X.columns, final_model.feature_importances_))
fi_sort = sorted(fi.items(), key=lambda x: -x[1])
print(f"\n   Top 20 features:")
for name, imp in fi_sort[:20]:
    print(f"     {name}: {imp:.4f}")

# — Salvare —
model_path = DIR.parent / "mario_quality_ny_v3.json"
meta_path  = DIR.parent / "mario_quality_ny_v3_features.json"
cal_path   = DIR.parent / "mario_quality_ny_v3_calibrated.pkl"

final_model.save_model(str(model_path))
with open(cal_path, 'wb') as f:
    pickle.dump(cal_model, f)

import json
best_smote = 0.0   # defined inside train_one_regime_model; fallback
_bp_mdl = regime_models.get('ALL', None)
best_params = _bp_mdl.get_params() if _bp_mdl is not None else {}
meta = {
    "version":          "ny_v3",
    "session":          "NY",
    "active_hours_utc": [13, 14],
    "features":         X.columns.tolist(),
    "n_features":       X.shape[1],
    "auc_oos_2025":   round(auc_val, 4),
    "auc_calibrated":   round(auc_cal, 4),
    "best_threshold":   round(best_thr, 3),
    "best_params":      {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
                         for k, v in best_params.items()},
    "smote_ratio":      round(float(best_smote), 3),
    "year_weights":     YEAR_WEIGHTS,
    "top_features":     [n for n, _ in fi_sort[:10]],
    "new_vs_ny_v2":     "MTF FVG 5m/15m/1h/4h, Inversion FVG, Breaker, Rejection, VIX proxy, DXY proxy, Sweep Quality, Rolling Regime, NY-specific interactions",
    "train_period":     "2023-01-01 to 2024-12-31",
    "val_period":       "2025-01-01 to 2025-12-31",
    "n_train":          int(len(X_tr_all)),
    "n_val":            int(len(X_val)),
}
meta_path.write_text(json.dumps(meta, indent=2))

print(f"\n   ✅ Salvat: {model_path.name}")
print(f"   ✅ Salvat: {meta_path.name}")
print(f"   ✅ Salvat: {cal_path.name}")
print()
print("=" * 74)
print(f"  FINAL AUC OOS 2025  : {auc_val:.4f}")
print(f"  FINAL AUC CALIBRAT     : {auc_cal:.4f}")
print(f"  THRESHOLD OPTIM        : {best_thr:.3f}")
print(f"  FEATURES               : {X.shape[1]}")
print(f"  SESSION                : NY (h13-h14 UTC)")
print("=" * 74)
