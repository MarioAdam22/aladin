"""
train_scorer_v4_1.py — ICT Setup Scorer v4.1
=============================================
Îmbunătățiri față de v4:
  1. Sample weight decay by year (2020→0.4 ... 2024-2025→1.0)
  2. ICT Weekly Profile features (12 features noi)
  3. Walk-Forward extins: test years 2023, 2024, 2025
  4. Suport pentru CSV-uri opționale 2020-2022 (setups_2020.csv etc.)
  5. Memory-efficient: feature computation via SQL aggregation (nu load full 1m dataset)
"""

import sqlite3, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
DB      = Path('/Users/mario/Desktop/Aladin/mario_trading.db')
COND_DB = Path('/Users/mario/Desktop/Aladin/market_conditions.db')
CSV_V3  = Path('/Users/mario/Desktop/Aladin/setups/setups_v3.csv')
CSV_OR  = Path('/Users/mario/Desktop/Aladin/setups/setups_or_reversal.csv')
PKL_V41 = Path('/Users/mario/Desktop/Aladin/ict_setup_scorer_v4_1.pkl')

CSV_OLDER = [
    Path('/Users/mario/Desktop/Aladin/setups/setups_2020.csv'),
    Path('/Users/mario/Desktop/Aladin/setups/setups_2021.csv'),
    Path('/Users/mario/Desktop/Aladin/setups/setups_2022.csv'),
]

YEAR_WEIGHTS = {2020:0.40, 2021:0.55, 2022:0.70, 2023:0.85, 2024:1.00, 2025:1.00}

# ── Features v3 (51) ─────────────────────────────────────────────────────────
FEATURES_V3 = [
    'tp_atr_ratio','tp_reachable','tp_in_2atr','premium_discount','h4_aligned',
    'va_width_atr','sweep_extreme','full_setup','htf_aligned','in_first_15',
    'dist_val_atr','dist_vah_atr','pre_lon_pos','bar_range_atr','rr',
    'fvg_up','fvg_down','h4_bias','h1_bias','sl_atr_ratio',
    'vwap_aligned','dist_vwap_norm','dist_poc_atr','near_poc',
    'adx_14_val','adx_strong','adx_weak','hurst_val','hurst_trending',
    'is_monday','is_friday','body_with_trade','pre_lon_range_atr',
    'tp_at_pd_level','outside_va','early_lon','late_lon',
    'fisher_extreme','acf_trending',
    'sweep_dist_pny_close_atr','sweep_near_pny_close',
    'sweep_dist_plon_close_atr','sweep_near_plon_close',
    'sweep_dist_pdh_atr','sweep_dist_pdl_atr','sweep_near_pd',
    'in_open30','sweep_near_pd_open30',
    'prev_ny_push_dir','is_tuesday','is_thursday',
]

# ── Features Tier noi v4 (25) ─────────────────────────────────────────────────
FEATURES_NEW_V4 = [
    'weekly_premium_pct','weekly_prem_direction',
    'h4_structure','h4_struct_aligned',
    'daily_bias','daily_bias_aligned','h4_x_daily',
    'ob_bull_active','ob_bear_active','ob_aligned',
    'eq_highs_above','eq_lows_below','eq_sweep_aligned',
    'fvg_active_bull','fvg_active_bear','fvg_aligned',
    'day_type_trend','day_type_inside','day_type_range',
    'ny_open_in_lon','lon_range_pt','lon_range_narrow',
    'fomc_proximity','dow_mon','dow_fri',
]

# ── Features noi v4.1 — ICT Weekly Profile (17) ───────────────────────────────
FEATURES_NEW_V41 = [
    'dow_tue','dow_wed','dow_thu',
    'monday_range_pt','monday_was_consolidation',
    'weekly_hi_taken','weekly_lo_taken',
    'tuesday_reversal_ctx','wednesday_reversal_ctx','thursday_consol_ctx',
    'days_in_week','week_range_so_far',
    'session_pct_elapsed','prev_session_range_pt',
    'atr_ratio_week','sweep_depth_score','htf_weekly_bias_aligned',
]

# ── Features noi v4.2 — Cross-model pollination (kalman + regime + interactions)
FEATURES_NEW_V42 = [
    'garch_vol','kalman_smooth','kalman_noise','acf_lag5','rvol_bar',
    'dir_x_adx','dir_x_hurst','kalman_x_dir',
    'regime_enc','regime_is_pre','regime_is_exp',
]

FEATURES_V41 = FEATURES_V3 + FEATURES_NEW_V4 + FEATURES_NEW_V41 + FEATURES_NEW_V42

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1: Load training data
# ════════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("train_scorer_v4_1.py — ICT Setup Scorer v4.1")
print("=" * 60)
print("\nLoading training data...")

setups_v3 = pd.read_csv(CSV_V3)
or_df     = pd.read_csv(CSV_OR)

or_map = or_df[['date','ts','session','direction','entry_price','sl_price','sl_pt',
                'tp_price','tp_pt','rr','label','pnl_usd']].copy()
or_map['setup_type']  = 'OR_' + or_map['direction']
or_map['entry_hhmm']  = pd.to_datetime(or_map['ts']).dt.strftime('%H%M').astype(int)
or_map['disp_hhmm']   = or_map['entry_hhmm']
or_map['tp_level']    = 'or_high'
for col in ['score','asia_sweep','h4_sweep','mss_5m','trend_15m']:
    or_map[col] = 0

setups_all = pd.concat([setups_v3, or_map], ignore_index=True)

for csv_path in CSV_OLDER:
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        setups_all = pd.concat([df_old, setups_all], ignore_index=True)
        print(f"  ✓ {csv_path.name}: {len(df_old)} setups")
    else:
        print(f"  ○ {csv_path.name} lipsă")

setups_all['date_str'] = setups_all['date'].astype(str).str[:10]
setups_all['year']     = pd.to_datetime(setups_all['date_str']).dt.year
setups_all['entry_hhmm'] = pd.to_numeric(setups_all['entry_hhmm'], errors='coerce').fillna(1400).astype(int)

print(f"\nTotal: {len(setups_all)} setups | WR={setups_all['label'].mean()*100:.1f}%")
yd = setups_all.groupby('year').agg(n=('label','count'), wr=('label','mean'))
for y, r in yd.iterrows():
    print(f"  {y}: {int(r['n']):4d} setups, WR={r['wr']*100:.1f}%, weight={YEAR_WEIGHTS.get(y,1.0)}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2: Pre-compute session stats via SQL aggregation (memory-efficient)
# ════════════════════════════════════════════════════════════════════════════════
print("\nPre-computing session stats via SQL...")

setup_dates_list = sorted(setups_all['date_str'].unique())

con = sqlite3.connect(DB)

# Toate datele de trading (pt lookup prev day)
all_td = pd.read_sql("SELECT DISTINCT date FROM market_data ORDER BY date", con)['date'].tolist()
date_to_idx = {d: i for i, d in enumerate(all_td)}

# Prev trading day pentru fiecare setup date
prev_day_map = {}
for d in setup_dates_list:
    idx = date_to_idx.get(d, -1)
    if idx > 0:
        prev_day_map[d] = all_td[idx - 1]

prev_days = list(set(prev_day_map.values()))
all_needed = list(set(setup_dates_list + prev_days))

# ── Query 1: Entry bars (bara exactă la momentul entry) ─────────────────────
# Construim tabelul de (date, hour_min) pentru setup-urile noastre
print("  Entry bars...")
entry_times = setups_all[['date_str','entry_hhmm']].drop_duplicates()
entry_times['hm_str'] = entry_times['entry_hhmm'].apply(
    lambda x: f"{int(x)//100:02d}:{int(x)%100:02d}"
)
# Unique (date, hm) pairs
pairs = entry_times[['date_str','hm_str']].drop_duplicates()
# Load entry bars in batches of dates
dates_ql = "','".join(setup_dates_list)
unique_hm = "','".join(pairs['hm_str'].unique())
entry_bars_raw = pd.read_sql(f"""
    SELECT date, hour_min,
           open, high, low, close,
           p_hi, p_lo, h4_hi, h4_lo, h1_hi, h1_lo,
           poc_level, vah, val, fvg_up, fvg_down,
           adx_14, vwap, dist_vwap, atr_14,
           hurst, fisher_transform, acf_lag1, acf_lag5, body_size,
           garch_vol, kalman_smooth, kalman_noise, rvol
    FROM market_data
    WHERE date IN ('{dates_ql}')
      AND hour_min IN ('{unique_hm}')
    ORDER BY date, hour_min
""", con)
# Build dict: (date, hm) → row
entry_bar_dict = {}
for _, row in entry_bars_raw.iterrows():
    entry_bar_dict[(row['date'], row['hour_min'])] = row

# ── Query 2: Daily session stats (per-day aggregation) ───────────────────────
print("  Daily session stats...")
all_needed_sql = "','".join(all_needed)
daily_stats = pd.read_sql(f"""
    SELECT
        date,
        MAX(high)  as day_high,
        MIN(low)   as day_low,
        AVG(atr_14) as avg_atr,
        MAX(CASE WHEN hour_min BETWEEN '08:00' AND '11:59' THEN high END) as lon_high,
        MIN(CASE WHEN hour_min BETWEEN '08:00' AND '11:59' THEN low  END) as lon_low,
        MAX(CASE WHEN hour_min BETWEEN '13:30' AND '19:59' THEN high END) as ny_high,
        MIN(CASE WHEN hour_min BETWEEN '13:30' AND '19:59' THEN low  END) as ny_low,
        MAX(CASE WHEN hour_min BETWEEN '19:00' AND '20:00' THEN close END) as ny_close,
        MAX(CASE WHEN hour_min BETWEEN '11:00' AND '12:00' THEN close END) as lon_close,
        MAX(CASE WHEN hour_min = '13:30' THEN open END) as ny_open,
        MAX(CASE WHEN hour_min BETWEEN '13:30' AND '19:59' THEN close END) as ny_close2
    FROM market_data
    WHERE date IN ('{all_needed_sql}')
    GROUP BY date
""", con)
# Add row for avg_atr_5d (rolling)
daily_stats = daily_stats.sort_values('date').reset_index(drop=True)
daily_stats['atr_5d_mean'] = daily_stats['avg_atr'].rolling(5, min_periods=1).mean().shift(1)
daily_dict = {row['date']: row for _, row in daily_stats.iterrows()}

# ── Query 3: NY session push direction (prev day) ─────────────────────────────
print("  Prev NY push direction...")
prev_days_sql = "','".join(prev_days) if prev_days else "''"
ny_push = pd.read_sql(f"""
    SELECT date,
           MAX(CASE WHEN hour_min = '13:30' THEN open END) as ny_open,
           MAX(CASE WHEN hour_min BETWEEN '19:00' AND '20:00' THEN close END) as ny_close
    FROM market_data
    WHERE date IN ('{prev_days_sql}')
    GROUP BY date
""", con)
ny_push_dict = {}
for _, row in ny_push.iterrows():
    if row['ny_open'] and row['ny_close'] and not np.isnan(row['ny_open']):
        diff = float(row['ny_close']) - float(row['ny_open'])
        ny_push_dict[row['date']] = 1 if diff > 0 else (-1 if diff < 0 else 0)

# ── Query 4: Weekly profile — Monday range, week extremes ────────────────────
print("  Weekly stats...")
# Compute iso week for each date
daily_stats['date_dt'] = pd.to_datetime(daily_stats['date'])
daily_stats['iso_year'] = daily_stats['date_dt'].dt.isocalendar().year.values
daily_stats['iso_week'] = daily_stats['date_dt'].dt.isocalendar().week.values
daily_stats['yw'] = daily_stats['iso_year'].astype(str) + '_' + daily_stats['iso_week'].astype(str)
daily_stats['dow_num'] = daily_stats['date_dt'].dt.dayofweek

# Monday stats
mon_df = daily_stats[daily_stats['dow_num'] == 0][['yw','day_high','day_low','avg_atr']].rename(
    columns={'day_high':'mon_high','day_low':'mon_low','avg_atr':'mon_atr'}
)
daily_stats = daily_stats.merge(mon_df, on='yw', how='left')

# Prev week extremes
weekly_ext = daily_stats.groupby('yw').agg(wk_hi=('day_high','max'), wk_lo=('day_low','min')).reset_index()
weekly_ext_sorted = weekly_ext.sort_values('yw').reset_index(drop=True)
weekly_ext_sorted['prev_wk_hi'] = weekly_ext_sorted['wk_hi'].shift(1)
weekly_ext_sorted['prev_wk_lo'] = weekly_ext_sorted['wk_lo'].shift(1)
daily_stats = daily_stats.merge(weekly_ext_sorted[['yw','prev_wk_hi','prev_wk_lo']], on='yw', how='left')

# Week range so far (cumulative per week)
daily_stats = daily_stats.sort_values(['yw','date'])
daily_stats['wk_hi_sofar'] = daily_stats.groupby('yw')['day_high'].cummax()
daily_stats['wk_lo_sofar'] = daily_stats.groupby('yw')['day_low'].cummin()
daily_stats['week_range_sofar'] = daily_stats['wk_hi_sofar'] - daily_stats['wk_lo_sofar']

# Rebuild dict (now with weekly stats)
daily_dict = {row['date']: row for _, row in daily_stats.iterrows()}

# Prev day NY range (for prev_session_range_pt)
daily_stats['ny_range'] = daily_stats['ny_high'].fillna(0) - daily_stats['ny_low'].fillna(0)
ny_range_dict = dict(zip(daily_stats['date'], daily_stats['ny_range']))

con.close()
print(f"  Done. {len(daily_dict)} zile în cache.")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 3: Build features per setup
# ════════════════════════════════════════════════════════════════════════════════
def get_entry_bar(date_str, hhmm_int):
    hm = f"{hhmm_int//100:02d}:{hhmm_int%100:02d}"
    bar = entry_bar_dict.get((date_str, hm))
    if bar is None:
        # Fallback: cea mai apropiată bară disponibilă în ziua respectivă
        day_bars = entry_bars_raw[entry_bars_raw['date'] == date_str]
        if len(day_bars) == 0:
            return None
        # Găsim bara cu hour_min cel mai aproape de hm
        day_bars = day_bars.copy()
        day_bars['hm_int'] = day_bars['hour_min'].apply(lambda x: int(x.replace(':',''))//1 if x else 0)
        target = hhmm_int
        day_bars['diff'] = abs(day_bars['hm_int'] - target)
        bar = day_bars.sort_values('diff').iloc[0]
    return bar

# ── Regime map for scorer (PRE_EXP=1, EXP=2, RET=3, CONSOL=0, DIST=4) ────────
_REGIME_CSV_SCORER = Path('/Users/mario/Desktop/Aladin/data/regime_labels.csv')
_REGIME_ENC_SCORER = {'CONSOLIDATION': 0, 'PRE_EXPANSION': 1, 'EXPANSION': 2, 'RETRACEMENT': 3, 'DISTRIBUTION': 4}
try:
    _rl = pd.read_csv(_REGIME_CSV_SCORER)
    # Use LON session regime as the daily context
    _rl_day = _rl[_rl['session'] == 'LON'][['date', 'regime']].copy()
    _regime_map_scorer = {str(row['date']): _REGIME_ENC_SCORER.get(row['regime'], -1)
                          for _, row in _rl_day.iterrows()}
    print(f"Regime map scorer: {len(_regime_map_scorer)} dates")
except Exception as _re2:
    _regime_map_scorer = {}
    print(f"Regime labels lipsă pentru scorer: {_re2}")

def build_features(row):
    date_str = str(row['date'])[:10]
    hhmm     = int(row['entry_hhmm']) if pd.notna(row['entry_hhmm']) else 1400
    direction = str(row['direction'])
    dir_enc   = 1 if direction == 'LONG' else -1
    entry = float(row['entry_price'])
    sl    = float(row['sl_price'])
    tp    = float(row['tp_price'])
    sl_pt = float(row['sl_pt']) if pd.notna(row['sl_pt']) else abs(entry - sl)
    tp_pt = float(row['tp_pt']) if pd.notna(row['tp_pt']) else abs(entry - tp)
    rr    = float(row['rr'])    if pd.notna(row['rr'])    else (tp_pt / sl_pt if sl_pt > 0 else 1.5)

    bar = get_entry_bar(date_str, hhmm)
    ds  = daily_dict.get(date_str, {})
    prev_day = prev_day_map.get(date_str)
    prev_ds  = daily_dict.get(prev_day, {}) if prev_day else {}

    def _fv(src, key, default=0.0):
        try:
            v = src[key] if isinstance(src, dict) else getattr(src, key, default)
            v = float(v)
            return v if not (np.isnan(v) if isinstance(v, float) else False) else default
        except:
            return default

    # ATR
    atr = max(_fv(bar, 'atr_14', 20.0) if bar is not None else 20.0, 0.25)

    # Entry bar features
    if bar is not None:
        poc = _fv(bar, 'poc_level', entry)
        vah = _fv(bar, 'vah', entry + 20)
        val = _fv(bar, 'val', entry - 20)
        h4h = _fv(bar, 'h4_hi', entry + 10)
        h4l = _fv(bar, 'h4_lo', entry - 10)
        h1h = _fv(bar, 'h1_hi', entry + 5)
        h1l = _fv(bar, 'h1_lo', entry - 5)
        fvg_up_v  = _fv(bar, 'fvg_up',  0)
        fvg_dn_v  = _fv(bar, 'fvg_down', 0)
        adx  = _fv(bar, 'adx_14', 25)
        vwap = _fv(bar, 'vwap', entry)
        dist_vwap = _fv(bar, 'dist_vwap', 0)
        hurst  = _fv(bar, 'hurst', 0.52)
        fisher = _fv(bar, 'fisher_transform', 0)
        acf1   = _fv(bar, 'acf_lag1', 0)
        acf5   = _fv(bar, 'acf_lag5', 0)
        body_sz = _fv(bar, 'body_size', 0)
        bar_hi  = _fv(bar, 'high', entry + 2)
        bar_lo  = _fv(bar, 'low',  entry - 2)
        p_hi = _fv(bar, 'p_hi', entry + 30)
        p_lo = _fv(bar, 'p_lo', entry - 30)
        garch  = _fv(bar, 'garch_vol', 1.0)
        kalman = _fv(bar, 'kalman_smooth', 0.0)
        kalman_n = _fv(bar, 'kalman_noise', 0.0)
        rvol_b = _fv(bar, 'rvol', 1.0)
    else:
        poc=entry; vah=entry+20; val=entry-20
        h4h=entry+10; h4l=entry-10; h1h=entry+5; h1l=entry-5
        fvg_up_v=0; fvg_dn_v=0; adx=25; vwap=entry; dist_vwap=0
        hurst=0.52; fisher=0; acf1=0; acf5=0; body_sz=0
        bar_hi=entry+2; bar_lo=entry-2; p_hi=entry+30; p_lo=entry-30
        garch=1.0; kalman=0.0; kalman_n=0.0; rvol_b=1.0

    # Day stats
    day_hi = _fv(ds, 'day_high', entry + 50)
    day_lo = _fv(ds, 'day_low',  entry - 50)
    day_range = max(day_hi - day_lo, 1)

    # Pre-london range (from daily_stats lon session)
    pre_lon_hi  = _fv(ds, 'lon_high', entry + 20)
    pre_lon_lo  = _fv(ds, 'lon_low',  entry - 20)
    pre_lon_range = max(pre_lon_hi - pre_lon_lo, 1)

    # Prev session data
    prev_ny_close  = _fv(prev_ds, 'ny_close',  entry)
    prev_lon_close = _fv(prev_ds, 'lon_close', entry)
    prev_ny_push_dir = ny_push_dict.get(prev_day, 0) if prev_day else 0

    # ── V3 features ────────────────────────────────────────────────────────────
    va_width   = max(vah - val, 1)
    tp_atr_ratio  = tp_pt / atr
    sl_atr_ratio  = sl_pt / atr
    tp_reachable  = int(tp_pt <= 4 * atr)
    tp_in_2atr    = int(tp_pt <= 2 * atr)

    pd_range     = max(p_hi - p_lo, 1)
    prem_disc    = (entry - p_lo) / pd_range - 0.5

    h4_mid = (h4h + h4l) / 2; h1_mid = (h1h + h1l) / 2
    h4_bias = 1 if entry > h4_mid else (-1 if entry < h4_mid else 0)
    h1_bias = 1 if entry > h1_mid else (-1 if entry < h1_mid else 0)
    h4_aligned  = int(h4_bias == dir_enc)
    htf_aligned = int(h4_bias == dir_enc and h1_bias == dir_enc)

    va_width_atr  = va_width / atr
    dist_val_atr  = (entry - val) / atr
    dist_vah_atr  = (vah - entry) / atr
    outside_va    = int(entry < val or entry > vah)
    near_poc      = int(abs(entry - poc) <= 2 * atr)
    dist_poc_atr  = (entry - poc) / atr

    session = str(row.get('session', 'NY'))
    in_first_15 = int(hhmm % 100 < 15)
    early_lon   = int(session == 'LON' and hhmm < 830)
    late_lon    = int(session == 'LON' and hhmm >= 1000)
    in_open30   = int((session == 'LON' and 800 <= hhmm <= 830) or
                      (session == 'NY'  and 1330 <= hhmm <= 1400))

    sweep_extreme = int((entry - day_lo) / day_range < 0.05 or
                        (day_hi - entry) / day_range < 0.05)

    fvg_up_f = int(fvg_up_v > 0 and fvg_up_v < entry)
    fvg_dn_f = int(fvg_dn_v > 0 and fvg_dn_v > entry)
    has_disp  = 1
    full_setup = has_disp * min(fvg_up_f + fvg_dn_f, 1)

    pre_lon_range_atr = pre_lon_range / atr
    pre_lon_pos = (entry - pre_lon_lo) / pre_lon_range

    bar_range_atr = (bar_hi - bar_lo) / atr
    tp_level     = str(row.get('tp_level','')) if 'tp_level' in row else ''
    tp_at_pd_level = int(tp_level in ['pdh','pdl','p_hi','p_lo','lw_hi','lw_lo'])

    vwap_aligned   = int((dir_enc == 1 and entry < vwap) or (dir_enc == -1 and entry > vwap))
    dist_vwap_norm = dist_vwap / atr

    adx_strong     = int(adx >= 30); adx_weak = int(adx < 20)
    hurst_trending = int(hurst > 0.55)
    fisher_extreme = int(abs(fisher) > 1.5)
    acf_trending   = int(acf1 > 0.3)

    dow = pd.to_datetime(date_str).dayofweek
    is_monday  = int(dow == 0); is_friday   = int(dow == 4)
    is_tuesday = int(dow == 1); is_thursday = int(dow == 3)

    body_with_trade = int((dir_enc == 1 and body_sz > 0) or (dir_enc == -1 and body_sz < 0))

    dist_pny  = abs(entry - prev_ny_close)  / atr if atr > 0 else 0
    dist_plon = abs(entry - prev_lon_close) / atr if atr > 0 else 0
    sweep_near_pny  = int(dist_pny  <= 2.0)
    sweep_near_plon = int(abs(entry - prev_lon_close) <= 10)
    dist_pdh = abs(entry - p_hi) / atr; dist_pdl = abs(entry - p_lo) / atr
    sweep_near_pd = int(min(dist_pdh, dist_pdl) <= 1.5)
    sweep_near_pd_open30 = int(sweep_near_pd and in_open30)

    v3 = {
        'tp_atr_ratio': tp_atr_ratio, 'tp_reachable': tp_reachable, 'tp_in_2atr': tp_in_2atr,
        'premium_discount': prem_disc, 'h4_aligned': h4_aligned,
        'va_width_atr': va_width_atr, 'sweep_extreme': sweep_extreme,
        'full_setup': full_setup, 'htf_aligned': htf_aligned, 'in_first_15': in_first_15,
        'dist_val_atr': dist_val_atr, 'dist_vah_atr': dist_vah_atr, 'pre_lon_pos': pre_lon_pos,
        'bar_range_atr': bar_range_atr, 'rr': rr,
        'fvg_up': fvg_up_f, 'fvg_down': fvg_dn_f, 'h4_bias': h4_bias, 'h1_bias': h1_bias,
        'sl_atr_ratio': sl_atr_ratio, 'vwap_aligned': vwap_aligned, 'dist_vwap_norm': dist_vwap_norm,
        'dist_poc_atr': dist_poc_atr, 'near_poc': near_poc,
        'adx_14_val': adx, 'adx_strong': adx_strong, 'adx_weak': adx_weak,
        'hurst_val': hurst, 'hurst_trending': hurst_trending,
        'is_monday': is_monday, 'is_friday': is_friday,
        'body_with_trade': body_with_trade, 'pre_lon_range_atr': pre_lon_range_atr,
        'tp_at_pd_level': tp_at_pd_level, 'outside_va': outside_va,
        'early_lon': early_lon, 'late_lon': late_lon,
        'fisher_extreme': fisher_extreme, 'acf_trending': acf_trending,
        'sweep_dist_pny_close_atr': dist_pny,   'sweep_near_pny_close': sweep_near_pny,
        'sweep_dist_plon_close_atr': dist_plon, 'sweep_near_plon_close': sweep_near_plon,
        'sweep_dist_pdh_atr': dist_pdh, 'sweep_dist_pdl_atr': dist_pdl,
        'sweep_near_pd': sweep_near_pd, 'in_open30': in_open30,
        'sweep_near_pd_open30': sweep_near_pd_open30,
        'prev_ny_push_dir': prev_ny_push_dir,
        'is_tuesday': is_tuesday, 'is_thursday': is_thursday,
    }

    # ── Weekly Profile features (v4.1) ─────────────────────────────────────────
    dow_tue = int(dow == 1); dow_wed = int(dow == 2); dow_thu = int(dow == 3)

    mon_hi  = _fv(ds, 'mon_high', 0)
    mon_lo  = _fv(ds, 'mon_low',  0)
    mon_range_pt = (mon_hi - mon_lo) if mon_hi > 0 and mon_lo > 0 else 0.0
    monday_was_consol = int(0 < mon_range_pt < 40.0)

    prev_wk_hi = _fv(ds, 'prev_wk_hi', 0)
    prev_wk_lo = _fv(ds, 'prev_wk_lo', 0)
    weekly_hi_taken = int(prev_wk_hi > 0 and entry > prev_wk_hi)
    weekly_lo_taken = int(prev_wk_lo > 0 and entry < prev_wk_lo)

    tuesday_reversal_ctx = int(dow_tue and monday_was_consol and not weekly_hi_taken and not weekly_lo_taken)
    wednesday_reversal_ctx = int(dow_wed and monday_was_consol)
    thursday_consol_ctx = int(dow_thu and monday_was_consol)

    week_range_sofar = _fv(ds, 'week_range_sofar', 0)
    week_range_norm  = week_range_sofar / atr if atr > 0 else 0.0

    # Session % elapsed
    if session == 'LON':
        sess_start, sess_end = 800, 1230
    elif session == 'NY':
        sess_start, sess_end = 1330, 2000
    else:
        sess_start, sess_end = 800, 2000
    sess_total   = (sess_end//100*60 + sess_end%100) - (sess_start//100*60 + sess_start%100)
    sess_elapsed = (hhmm//100*60 + hhmm%100) - (sess_start//100*60 + sess_start%100)
    session_pct  = max(0.0, min(1.0, sess_elapsed / sess_total if sess_total > 0 else 0.5))

    # Prev session range
    prev_ny_range = float(ny_range_dict.get(prev_day, 80)) if prev_day else 80.0

    # ATR ratio week
    atr_5d = float(ds.get('atr_5d_mean', atr) or atr)
    atr_ratio_week = min(atr / atr_5d if atr_5d > 0 else 1.0, 3.0)

    # Sweep depth
    if dir_enc == 1:
        sweep_depth_score = max(0.0, min(5.0, (entry - day_lo) / atr if atr > 0 else 0))
    else:
        sweep_depth_score = max(0.0, min(5.0, (day_hi - entry) / atr if atr > 0 else 0))

    # HTF weekly bias aligned (using weekly premium direction proxy)
    # weekly_premium_pct from conditions (available per date)
    wpct = 0.5  # default; will be overridden by merged conditions in Step 4
    htf_weekly_bias_aligned = float(np.clip((0.5 - wpct) * dir_enc, -0.5, 0.5))

    wp = {
        'dow_tue': dow_tue, 'dow_wed': dow_wed, 'dow_thu': dow_thu,
        'monday_range_pt': min(mon_range_pt, 200.0),
        'monday_was_consolidation': monday_was_consol,
        'weekly_hi_taken': weekly_hi_taken, 'weekly_lo_taken': weekly_lo_taken,
        'tuesday_reversal_ctx': tuesday_reversal_ctx,
        'wednesday_reversal_ctx': wednesday_reversal_ctx,
        'thursday_consol_ctx': thursday_consol_ctx,
        'days_in_week': dow,
        'week_range_so_far': week_range_norm,
        'session_pct_elapsed': session_pct,
        'prev_session_range_pt': min(prev_ny_range, 500.0),
        'atr_ratio_week': atr_ratio_week,
        'sweep_depth_score': sweep_depth_score,
        'htf_weekly_bias_aligned': htf_weekly_bias_aligned,
    }

    # ── v4.2 cross-model features ─────────────────────────────────────────────
    # Regime from regime_labels.csv (loaded below at module level)
    _reg = _regime_map_scorer.get(date_str, -1)
    v42 = {
        'garch_vol':       garch,
        'kalman_smooth':   kalman,
        'kalman_noise':    kalman_n,
        'acf_lag5':        acf5,
        'rvol_bar':        rvol_b,
        'dir_x_adx':       dir_enc * adx / 100.0,
        'dir_x_hurst':     dir_enc * hurst,
        'kalman_x_dir':    dir_enc * kalman,
        'regime_enc':      float(_reg),
        'regime_is_pre':   1 if _reg == 1 else 0,
        'regime_is_exp':   1 if _reg == 2 else 0,
    }

    return {**v3, **wp, **v42}

print("Building features per setup...")
all_feats = []
for i, row in setups_all.iterrows():
    try:
        f = build_features(row)
    except Exception as e:
        f = {k: 0.0 for k in FEATURES_V3 + FEATURES_NEW_V41}
    all_feats.append(f)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(setups_all)}...", end='\r')
print(f"\n  Done. {len(all_feats)} rows")

df_feats = pd.DataFrame(all_feats)

# ════════════════════════════════════════════════════════════════════════════════
# STEP 4: Merge v4 conditions (2023-2025) + build v4 feature columns
# ════════════════════════════════════════════════════════════════════════════════
print("Merging conditions v4...")
con_c = sqlite3.connect(COND_DB)
cond  = pd.read_sql("SELECT * FROM conditions", con_c)
con_c.close()

merged_cond = setups_all[['date_str','direction']].merge(
    cond[['date','weekly_premium_pct','h4_structure','daily_bias',
          'ob_bull_active','ob_bear_active','eq_highs_above','eq_lows_below',
          'fvg_active_bull','fvg_active_bear','day_type',
          'ny_open_in_lon','lon_range','fomc_proximity','dow_enc']],
    left_on='date_str', right_on='date', how='left'
)

dir_enc_s = merged_cond['direction'].map({'LONG':1,'SHORT':-1}).fillna(0)

new_v4 = pd.DataFrame(index=setups_all.index)
new_v4['weekly_premium_pct']    = merged_cond['weekly_premium_pct'].fillna(0.5).values
new_v4['weekly_prem_direction'] = np.where(
    dir_enc_s == 1, 0.5 - merged_cond['weekly_premium_pct'].fillna(0.5),
    merged_cond['weekly_premium_pct'].fillna(0.5) - 0.5
).clip(-0.5, 0.5)
new_v4['h4_structure']      = merged_cond['h4_structure'].fillna(0).values
new_v4['h4_struct_aligned'] = (merged_cond['h4_structure'].fillna(0) * dir_enc_s).clip(-1,1).values
new_v4['daily_bias']        = merged_cond['daily_bias'].fillna(0).values
new_v4['daily_bias_aligned']= (merged_cond['daily_bias'].fillna(0) * dir_enc_s).clip(-1,1).values
new_v4['h4_x_daily']        = (merged_cond['h4_structure'].fillna(0) *
                                merged_cond['daily_bias'].fillna(0)).values
new_v4['ob_bull_active']    = merged_cond['ob_bull_active'].fillna(0).values
new_v4['ob_bear_active']    = merged_cond['ob_bear_active'].fillna(0).values
new_v4['ob_aligned']        = np.where(
    dir_enc_s == 1, merged_cond['ob_bull_active'].fillna(0),
    merged_cond['ob_bear_active'].fillna(0)).astype(float)
new_v4['eq_highs_above']    = merged_cond['eq_highs_above'].fillna(0).values
new_v4['eq_lows_below']     = merged_cond['eq_lows_below'].fillna(0).values
new_v4['eq_sweep_aligned']  = np.where(
    dir_enc_s == 1, merged_cond['eq_lows_below'].fillna(0),
    merged_cond['eq_highs_above'].fillna(0)).astype(float)
new_v4['fvg_active_bull']   = merged_cond['fvg_active_bull'].fillna(0).values
new_v4['fvg_active_bear']   = merged_cond['fvg_active_bear'].fillna(0).values
new_v4['fvg_aligned']       = np.where(
    dir_enc_s == 1, merged_cond['fvg_active_bull'].fillna(0),
    merged_cond['fvg_active_bear'].fillna(0)).astype(float)
new_v4['day_type_trend']    = (merged_cond['day_type'] == 'trend').astype(int).values
new_v4['day_type_inside']   = (merged_cond['day_type'] == 'inside').astype(int).values
new_v4['day_type_range']    = (merged_cond['day_type'] == 'range').astype(int).values
new_v4['ny_open_in_lon']    = merged_cond['ny_open_in_lon'].fillna(0).values
new_v4['lon_range_pt']      = merged_cond['lon_range'].fillna(80).values
new_v4['lon_range_narrow']  = (merged_cond['lon_range'].fillna(80) < 60).astype(int).values
new_v4['fomc_proximity']    = merged_cond['fomc_proximity'].fillna(0).values
new_v4['dow_mon']           = (merged_cond['dow_enc'] == 0).astype(int).values
new_v4['dow_fri']           = (merged_cond['dow_enc'] == 4).astype(int).values

# Update htf_weekly_bias_aligned cu valoarea reală din conditions
wpct_arr = merged_cond['weekly_premium_pct'].fillna(0.5).values
df_feats['htf_weekly_bias_aligned'] = np.clip(
    (0.5 - wpct_arr) * dir_enc_s.values, -0.5, 0.5
)

# ════════════════════════════════════════════════════════════════════════════════
# STEP 5: Assemble feature matrix
# ════════════════════════════════════════════════════════════════════════════════
X_v3  = df_feats[FEATURES_V3].fillna(0)
X_new4 = new_v4[FEATURES_NEW_V4].fillna(0).reset_index(drop=True)
X_wp   = df_feats[FEATURES_NEW_V41].fillna(0)

X_v41 = pd.concat([
    X_v3.reset_index(drop=True),
    X_new4,
    X_wp.reset_index(drop=True),
], axis=1)

y = setups_all['label'].values
sample_weights = setups_all['year'].map(YEAR_WEIGHTS).fillna(1.0).values

print(f"\nFeature matrix: {X_v41.shape}")
print(f"Label dist: {y.mean()*100:.1f}% positive")
print(f"Sample weights: {sample_weights.min():.2f} – {sample_weights.max():.2f}")

# ── Synthetic Order Flow features ─────────────────────────────────────────
_OF_PATH = Path(__file__).parent.parent / "data" / "orderflow_features.parquet"
if _OF_PATH.exists():
    _of = __import__('pandas').read_parquet(_OF_PATH)
    _of['date'] = _of['date'].astype(str)
    _OF_COLS = [c for c in _of.columns if c not in ['session_id','date','session_type',
                'session_open','session_close','session_high','session_low','total_vol']]
    _of_m = _of[['date','session_type'] + _OF_COLS].rename(
        columns={'date':'date_str','session_type':'session'})
    _of_joined = setups_all[['date_str','session']].merge(_of_m, on=['date_str','session'], how='left')
    _of_joined[_OF_COLS] = _of_joined[_OF_COLS].fillna(0.0)
    X_of = _of_joined[_OF_COLS].reset_index(drop=True)
    X_v41 = __import__('pandas').concat([X_v41.reset_index(drop=True), X_of], axis=1)
    print(f"   Order flow: {len(_OF_COLS)} features adaugate in X_v41")
else:
    _OF_COLS = []

missing = [f for f in FEATURES_V41 if f not in X_v41.columns]
if missing:
    print(f"  ⚠ Features lipsă: {missing}")
else:
    print(f"  ✓ Toate {len(FEATURES_V41)} features prezente")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 6: Walk-Forward CV (2023, 2024, 2025)
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Walk-Forward CV (2023, 2024, 2025) ───────────────────────")

wf_v41 = []; wf_v4b = []; wf_v3b = []

X_v4_combined = pd.concat([X_v3.reset_index(drop=True), X_new4], axis=1)

for test_year in [2023, 2024, 2025]:
    train_mask = (setups_all['year'] < test_year).values
    test_mask  = (setups_all['year'] == test_year).values

    if train_mask.sum() < 50 or test_mask.sum() < 20:
        print(f"  WF {test_year}: skip (train={train_mask.sum()}, test={test_mask.sum()})")
        continue

    y_tr = y[train_mask]; y_te = y[test_mask]
    sw_tr = sample_weights[train_mask]

    def fit_gbm(X_tr, y_tr, sw=None):
        gbm = GradientBoostingClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.7, min_samples_leaf=8, random_state=42
        )
        if sw is not None:
            gbm.fit(X_tr, y_tr, sample_weight=sw)
        else:
            gbm.fit(X_tr, y_tr)
        return gbm

    gbm41 = fit_gbm(X_v41.values[train_mask], y_tr, sw_tr)
    auc41 = roc_auc_score(y_te, gbm41.predict_proba(X_v41.values[test_mask])[:,1])

    gbm4  = fit_gbm(X_v4_combined.values[train_mask], y_tr)
    auc4  = roc_auc_score(y_te, gbm4.predict_proba(X_v4_combined.values[test_mask])[:,1])

    gbm3  = fit_gbm(X_v3.values[train_mask], y_tr)
    auc3  = roc_auc_score(y_te, gbm3.predict_proba(X_v3.values[test_mask])[:,1])

    wf_v41.append(auc41); wf_v4b.append(auc4); wf_v3b.append(auc3)
    print(f"  WF {test_year}: v3={auc3:.4f}  v4={auc4:.4f}  "
          f"v4.1={auc41:.4f}  Δ(v4→v4.1)={auc41-auc4:+.4f}  "
          f"(n_train={train_mask.sum()}, n_test={test_mask.sum()})")

if wf_v41:
    m41 = np.mean(wf_v41); m4 = np.mean(wf_v4b); m3 = np.mean(wf_v3b)
    print(f"\n  WF mean: v3={m3:.4f}  v4={m4:.4f}  v4.1={m41:.4f}  Δ={m41-m4:+.4f}")
else:
    m41=m4=m3=0.5

# ════════════════════════════════════════════════════════════════════════════════
# STEP 7: Train final model on ALL data
# ════════════════════════════════════════════════════════════════════════════════
print("\nTraining final v4.1 on all data...")
gbm_final = GradientBoostingClassifier(
    n_estimators=500, max_depth=3, learning_rate=0.025,
    subsample=0.7, min_samples_leaf=8, random_state=42
)
gbm_final.fit(X_v41, y, sample_weight=sample_weights)
probs_all = gbm_final.predict_proba(X_v41)[:,1]
setups_all['score_v41'] = probs_all

# Feature importance
print("\n── Top 25 Feature Importances ───────────────────────────────")
fi = pd.Series(gbm_final.feature_importances_, index=FEATURES_V41).sort_values(ascending=False)
for feat, imp in fi.head(25).items():
    tag = " ← WP" if feat in FEATURES_NEW_V41 else (" ← v4" if feat in FEATURES_NEW_V4 else "")
    print(f"  {feat:<35s}: {imp:.4f}{tag}")

print("\n── ICT Weekly Profile Importances ───────────────────────────")
wp_fi = fi[fi.index.isin(FEATURES_NEW_V41)].sort_values(ascending=False)
for feat, imp in wp_fi.items():
    print(f"  {feat:<35s}: {imp:.4f}")

# Score buckets
print("\n── Score Buckets ────────────────────────────────────────────")
setups_all['score_bucket'] = pd.cut(
    setups_all['score_v41'], bins=[0,.3,.4,.5,.6,.7,1.0],
    labels=['<30','30-40','40-50','50-60','60-70','>70']
)
bucket_stats = setups_all.groupby('score_bucket', observed=True).agg(
    n=('label','count'), wr=('label','mean')
).assign(wr_pct=lambda x: x['wr']*100)
print(bucket_stats[['n','wr_pct']].to_string())

hi = setups_all[setups_all['score_v41'] >= 0.60]
if len(hi):
    avg_rr = hi['rr'].mean() if 'rr' in hi.columns else 2.0
    wr = hi['label'].mean()
    ev = wr * avg_rr - (1 - wr)
    print(f"\n  Score>60: n={len(hi)}, WR={wr*100:.1f}%, RR={avg_rr:.2f}, EV={ev:+.3f}")

# Per zi
print("\n── WR per zi (setups_all) ───────────────────────────────────")
setups_all['dow_label'] = pd.to_datetime(setups_all['date_str']).dt.day_name()
dow_st = setups_all.groupby('dow_label').agg(
    n=('label','count'), wr=('label','mean'), score_avg=('score_v41','mean')
).assign(wr_pct=lambda x: x['wr']*100)
for d in ['Monday','Tuesday','Wednesday','Thursday','Friday']:
    if d in dow_st.index:
        r = dow_st.loc[d]
        print(f"  {d:<12s}: n={int(r['n']):4d}, WR={r['wr_pct']:.1f}%, score={r['score_avg']:.3f}")

# Save
payload = {
    'model':            gbm_final,
    'features':         FEATURES_V41,
    'features_v3':      FEATURES_V3,
    'features_new_v4':  FEATURES_NEW_V4,
    'features_new_v41': FEATURES_NEW_V41,
    'year_weights':     YEAR_WEIGHTS,
    'wf_auc_v41':       m41,
    'wf_auc_v4':        m4,
    'wf_auc_v3':        m3,
    'threshold_high':   0.60,
    'threshold_mid':    0.50,
    'n_setups':         len(setups_all),
    'n_features':       len(FEATURES_V41),
}
with open(PKL_V41, 'wb') as f:
    pickle.dump(payload, f)

print(f"\n{'='*60}")
print(f"Saved → {PKL_V41}")
print(f"WF AUC: v3={m3:.4f}  v4={m4:.4f}  v4.1={m41:.4f}  Δ={m41-m4:+.4f}")
print(f"Total: {len(FEATURES_V41)} features | {len(setups_all)} setups")
print(f"{'='*60}")
