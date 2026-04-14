"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          BACKTEST MARIO_BOT.JSON — Out-of-Sample Validation                 ║
║          5 clase: 0=WAIT 1=SHORT_BREAK 2=LONG_BREAK 3=SHORT_REV 4=LONG_REV ║
║          RM: SL→BE la 0.5R, trail de la 2R, ATR-based SL                   ║
║          $20/punct NQ Mini, risc 8-10pts (~$200/R)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sqlite3
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

DIR = pathlib.Path(__file__).parent

# ── CONFIG ────────────────────────────────────────────────────────────────────
PATH_DB         = DIR / "mario_trading.db"
MODEL_PATH      = DIR / "mario_bot.json"
FEATURES_PATH   = DIR / "mario_features.json"

BACKTEST_START  = "2023-01-01"
BACKTEST_END    = "2025-12-31"

# Risk Management
ATR_SL_MULT     = 0.8    # SL = 0.8 × ATR (aprox 8-10 pts pe NQ)
BE_TRIGGER_R    = 0.5    # SL → BE la +0.5R
TRAIL_START_R   = 2.0    # Trailing SL porneşte de la +2R
TRAIL_ATR_MULT  = 0.5    # Trail = 0.5 × ATR
MAX_SL_PTS      = 14.0   # SL maxim (clamp)
MIN_SL_PTS      = 6.0    # SL minim

TICK_VALUE      = 20.0   # $20/punct NQ Mini (1 contract)

# Confidence threshold per clasă
CONF_THRESHOLD  = 0.35   # P(clasă) ≥ 0.35 pentru trade

# Ferestre de intrare (ore Romania)
LON_ENTRY_START = 9.5    # 09:30 RO (post LON OR)
LON_ENTRY_END   = 11.0   # 11:00 RO
NY_ENTRY_START  = 16.0   # 16:00 RO (post NY OR)
NY_ENTRY_END    = 17.5   # 17:30 RO

REGIME_NAMES = {0: 'WAIT', 1: 'SHORT_BREAK', 2: 'LONG_BREAK', 3: 'SHORT_REV', 4: 'LONG_REV'}
SIGNAL_CLASSES = [1, 2, 3, 4]
DIRECTION = {1: 'SHORT', 2: 'LONG', 3: 'SHORT', 4: 'LONG'}


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    features = json.loads(FEATURES_PATH.read_text())['features']
    print(f"✓ Model loaded: {len(features)} features")
    return model, features


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(str(PATH_DB))
    query = f"""
        SELECT timestamp, open, high, low, close, volume,
               p_hi, p_lo, asia_hi, asia_lo, lon_hi, lon_lo,
               lw_hi, lw_lo, h4_hi, h4_lo, h1_hi, h1_lo,
               atr_14, vah, val, poc_level
        FROM market_data
        WHERE timestamp >= '{BACKTEST_START}'
          AND timestamp <= '{BACKTEST_END} 23:59:00'
        ORDER BY timestamp
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✓ Data: {len(df):,} rows ({BACKTEST_START} → {BACKTEST_END})")
    return df


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def safe_div(a, b, fill=0.0):
    b = np.asarray(b, dtype=float)
    b[b == 0] = np.nan
    result = np.asarray(a, dtype=float) / b
    result = np.where(np.isfinite(result), result, fill)
    return result


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ts   = df['timestamp']
    _td  = (ts.dt.hour + ts.dt.minute / 60.0).values
    cl   = df['close'].values
    hi   = df['high'].values
    lo   = df['low'].values
    pdh  = df['p_hi'].fillna(method='ffill').values
    pdl  = df['p_lo'].fillna(method='ffill').values

    # ── ATR ──────────────────────────────────────────────────────────────────
    atr = df['atr_14'].values
    # Fallback: compute from OHLC if missing
    if np.isnan(atr).mean() > 0.5:
        _tr = np.maximum(hi - lo,
              np.maximum(np.abs(hi - np.roll(cl, 1)),
                         np.abs(lo - np.roll(cl, 1))))
        _tr[0] = hi[0] - lo[0]
        atr = pd.Series(_tr).ewm(span=14, min_periods=1).mean().values
    atr = np.where(atr > 0, atr, 9.0)

    # ── dist_pdh, dist_pdl ────────────────────────────────────────────────────
    df['dist_pdh'] = np.clip(safe_div(pdh - cl, atr, 0), -5, 5)
    df['dist_pdl'] = np.clip(safe_div(cl - pdl, atr, 0), -5, 5)

    # ── atr_percentile ────────────────────────────────────────────────────────
    atr_s = pd.Series(atr)
    df['atr_percentile'] = atr_s.rolling(100, min_periods=10).rank(pct=True).fillna(0.5).values

    # ── Session time windows (ore Romania = timestamp) ────────────────────────
    df['in_london_or']     = ((_td >= 9.0)  & (_td < 9.5)).astype(np.int8)
    df['in_london_kz']     = ((_td >= 9.5)  & (_td < 11.0)).astype(np.int8)
    df['in_london_close']  = ((_td >= 11.0) & (_td < 12.0)).astype(np.int8)
    df['in_pre_ny']        = ((_td >= 15.0) & (_td < 15.5)).astype(np.int8)
    df['in_ny_or']         = ((_td >= 15.5) & (_td < 16.0)).astype(np.int8)
    df['in_pre_ny_macro']  = ((_td >= 15.75) & (_td < 16.167)).astype(np.int8)
    df['in_ny_kz_core']    = ((_td >= 16.0) & (_td < 16.833)).astype(np.int8)
    df['in_ny_macro_1']    = ((_td >= 16.833) & (_td < 17.167)).astype(np.int8)
    df['in_ny_macro_2']    = ((_td >= 17.167) & (_td < 17.5)).astype(np.int8)
    df['in_any_macro']     = (
        (df['in_pre_ny_macro'] == 1) |
        (df['in_ny_macro_1']   == 1) |
        (df['in_ny_macro_2']   == 1)
    ).astype(np.int8)

    _mins_lon = np.clip((_td - 9.0) * 60, 0, 180)
    _mins_ny  = np.clip((_td - 15.5) * 60, 0, 120)
    df['mins_since_lon_open'] = np.where(_td >= 9.0, _mins_lon, 0).astype(np.float32)
    df['mins_since_ny_open']  = np.where(_td >= 15.5, _mins_ny, 0).astype(np.float32)

    # ── range_day_score ───────────────────────────────────────────────────────
    df['_date'] = ts.dt.date
    day_grp = df.groupby('_date')
    df['_day_high'] = day_grp['high'].cummax().values
    df['_day_low']  = day_grp['low'].cummin().values
    df['_day_open'] = day_grp['open'].transform('first').values
    day_range  = df['_day_high'].values - df['_day_low'].values
    range_atr  = safe_div(day_range, atr, 1.0)
    df['range_day_score'] = (1.0 / np.maximum(range_atr, 1.0)).astype(np.float32)

    # ── liq_above_count, liq_below_count ─────────────────────────────────────
    asia_hi = df['asia_hi'].fillna(method='ffill').values
    asia_lo = df['asia_lo'].fillna(method='ffill').values
    h4_hi   = df['h4_hi'].fillna(method='ffill').values
    h4_lo   = df['h4_lo'].fillna(method='ffill').values
    lw_hi   = df['lw_hi'].fillna(method='ffill').values
    lw_lo   = df['lw_lo'].fillna(method='ffill').values

    df['liq_above_count'] = (
        (asia_hi > cl).astype(int) +
        (pdh     > cl).astype(int) +
        (h4_hi   > cl).astype(int) +
        (lw_hi   > cl).astype(int)
    )
    df['liq_below_count'] = (
        (asia_lo < cl).astype(int) +
        (pdl     < cl).astype(int) +
        (h4_lo   < cl).astype(int) +
        (lw_lo   < cl).astype(int)
    )

    # ── broke_asia_hi/lo, broke_pdh/pdl ───────────────────────────────────────
    df['broke_asia_hi'] = (hi > asia_hi).astype(int)
    df['broke_asia_lo'] = (lo < asia_lo).astype(int)
    df['broke_pdh']     = (hi > pdh).astype(int)
    df['broke_pdl']     = (lo < pdl).astype(int)

    # ── sweep_direction ───────────────────────────────────────────────────────
    lon_hi = df['lon_hi'].fillna(method='ffill').values
    lon_lo = df['lon_lo'].fillna(method='ffill').values
    h1_hi  = df['h1_hi'].fillna(method='ffill').values
    h1_lo  = df['h1_lo'].fillna(method='ffill').values

    swept_up = (
        (hi > asia_hi).astype(int) +
        (hi > pdh).astype(int) +
        (hi > lon_hi).astype(int) +
        (hi > h4_hi).astype(int) +
        (hi > h1_hi).astype(int)
    )
    swept_dn = (
        (lo < asia_lo).astype(int) +
        (lo < pdl).astype(int) +
        (lo < lon_lo).astype(int) +
        (lo < h4_lo).astype(int) +
        (lo < h1_lo).astype(int)
    )
    df['sweep_direction'] = np.where(
        swept_up > swept_dn, 1,
        np.where(swept_dn > swept_up, -1, 0)
    ).astype(int)

    # cleanup temp columns
    df = df.drop(columns=['_date', '_day_high', '_day_low', '_day_open'], errors='ignore')

    return df


# ── RISK MANAGEMENT SIMULATION ────────────────────────────────────────────────
def simulate_trade(bars: pd.DataFrame, entry_idx: int, direction: str, atr_entry: float):
    """
    Simulează un trade cu RM:
    - SL = ATR_SL_MULT × atr_entry (clampat MIN_SL_PTS..MAX_SL_PTS)
    - BE la 0.5R
    - Trail de la 2R cu TRAIL_ATR_MULT × atr_entry

    Returnează: dict cu r_mult, pts, bars_held, exit_reason
    """
    sl_pts = float(np.clip(ATR_SL_MULT * atr_entry, MIN_SL_PTS, MAX_SL_PTS))

    if direction == 'LONG':
        entry_px   = bars['close'].iloc[entry_idx]
        sl_price   = entry_px - sl_pts
        be_trigger = entry_px + BE_TRIGGER_R * sl_pts
        trail_trigger = entry_px + TRAIL_START_R * sl_pts
        trail_sl   = sl_price  # trailing SL starts at original
        be_hit     = False
        trail_hit  = False

        for i in range(entry_idx + 1, len(bars)):
            bar = bars.iloc[i]
            bar_high = bar['high']
            bar_low  = bar['low']

            # Trail update
            if bar_high >= trail_trigger:
                trail_hit = True
            if trail_hit:
                new_trail = bar_high - TRAIL_ATR_MULT * atr_entry
                trail_sl = max(trail_sl, new_trail)

            # BE update
            if bar_high >= be_trigger and not be_hit:
                sl_price = entry_px
                be_hit = True

            effective_sl = max(sl_price, trail_sl) if trail_hit else sl_price

            # Stop hit
            if bar_low <= effective_sl:
                exit_px = effective_sl
                pts = exit_px - entry_px
                r   = pts / sl_pts
                reason = 'TRAIL_STOP' if trail_hit else ('BE_STOP' if be_hit else 'STOP_LOSS')
                return dict(r_mult=r, pts=pts, bars_held=i - entry_idx, exit_reason=reason, entry_px=entry_px, exit_px=exit_px, sl_pts=sl_pts)

    else:  # SHORT
        entry_px   = bars['close'].iloc[entry_idx]
        sl_price   = entry_px + sl_pts
        be_trigger = entry_px - BE_TRIGGER_R * sl_pts
        trail_trigger = entry_px - TRAIL_START_R * sl_pts
        trail_sl   = sl_price
        be_hit     = False
        trail_hit  = False

        for i in range(entry_idx + 1, len(bars)):
            bar = bars.iloc[i]
            bar_high = bar['high']
            bar_low  = bar['low']

            # Trail update
            if bar_low <= trail_trigger:
                trail_hit = True
            if trail_hit:
                new_trail = bar_low + TRAIL_ATR_MULT * atr_entry
                trail_sl = min(trail_sl, new_trail)

            # BE update
            if bar_low <= be_trigger and not be_hit:
                sl_price = entry_px
                be_hit = True

            effective_sl = min(sl_price, trail_sl) if trail_hit else sl_price

            # Stop hit
            if bar_high >= effective_sl:
                exit_px = effective_sl
                pts = entry_px - exit_px
                r   = pts / sl_pts
                reason = 'TRAIL_STOP' if trail_hit else ('BE_STOP' if be_hit else 'STOP_LOSS')
                return dict(r_mult=r, pts=pts, bars_held=i - entry_idx, exit_reason=reason, entry_px=entry_px, exit_px=exit_px, sl_pts=sl_pts)

    # Timeout — nerealizat în fereastră
    last_px = bars['close'].iloc[-1]
    pts = (last_px - entry_px) if direction == 'LONG' else (entry_px - last_px)
    r   = pts / sl_pts
    return dict(r_mult=r, pts=pts, bars_held=len(bars) - entry_idx - 1, exit_reason='TIMEOUT', entry_px=entry_px, exit_px=last_px, sl_pts=sl_pts)


# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────
def run_backtest():
    print("\n" + "═"*65)
    print("  BACKTEST MARIO_BOT.JSON  —  Out-of-Sample Validation")
    print("═"*65)

    model, features = load_model()
    df_raw = load_data()
    df = compute_features(df_raw)

    # Verifică features
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"⚠️ Features lipsă (set la 0): {missing}")
        for f in missing:
            df[f] = 0.0

    print(f"✓ Features compute: {len(features)} ok")

    # ── Predicţii ──────────────────────────────────────────────────────────
    X = df[features].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
    proba = model.predict_proba(X)
    pred_class = np.argmax(proba, axis=1)
    pred_conf  = proba.max(axis=1)

    df['pred_class'] = pred_class
    df['pred_conf']  = pred_conf
    for c in range(5):
        df[f'p_cls{c}'] = proba[:, c]

    # ── Entry filter: killzone + confidence ────────────────────────────────
    td = (df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0).values

    in_entry_window = (
        ((td >= LON_ENTRY_START) & (td < LON_ENTRY_END)) |
        ((td >= NY_ENTRY_START)  & (td < NY_ENTRY_END))
    )
    has_signal = (df['pred_class'].isin(SIGNAL_CLASSES)) & (df['pred_conf'] >= CONF_THRESHOLD)
    entry_mask = in_entry_window & has_signal

    entry_indices = np.where(entry_mask)[0]
    print(f"✓ Semnale totale: {len(entry_indices):,}  (threshold={CONF_THRESHOLD})")

    # ── Simulare tranzacţii ────────────────────────────────────────────────
    trades = []
    last_exit_bar = -1  # anti-overlap: un trade o dată

    for idx in entry_indices:
        if idx <= last_exit_bar:
            continue  # trade activ, skip

        pred_c  = int(df['pred_class'].iloc[idx])
        conf    = float(df['pred_conf'].iloc[idx])
        direction = DIRECTION[pred_c]
        atr_val = float(df['atr_14'].iloc[idx])
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = 9.0
        ts_entry = df['timestamp'].iloc[idx]
        session  = 'LON' if (td[idx] >= LON_ENTRY_START and td[idx] < LON_ENTRY_END) else 'NY'

        # Bara de exit: până la fine sesiune (max 120 bare ~ 2h)
        end_session_td = LON_ENTRY_END if session == 'LON' else NY_ENTRY_END
        future = df.iloc[idx: idx + 150].copy()
        # Taie la fine sesiune
        future_td = (future['timestamp'].dt.hour + future['timestamp'].dt.minute / 60.0).values
        session_mask = future_td < end_session_td + 0.5
        if session_mask.sum() < 2:
            continue
        future = future[session_mask]

        result = simulate_trade(future.reset_index(drop=True), 0, direction, atr_val)
        result.update(dict(
            timestamp=ts_entry,
            session=session,
            pred_class=pred_c,
            regime=REGIME_NAMES[pred_c],
            direction=direction,
            confidence=conf,
            atr_entry=atr_val,
            date=ts_entry.date(),
        ))
        trades.append(result)
        last_exit_bar = idx + result['bars_held']

    if not trades:
        print("❌ Niciun trade generat!")
        return

    # ── Analiza rezultatelor ──────────────────────────────────────────────
    df_t = pd.DataFrame(trades)
    df_t['pnl_usd'] = df_t['r_mult'] * df_t['sl_pts'] * TICK_VALUE
    df_t['win'] = (df_t['r_mult'] > 0).astype(int)
    df_t['be']  = (df_t['exit_reason'] == 'BE_STOP').astype(int)

    total_trades = len(df_t)
    wins = df_t['win'].sum()
    wr   = wins / total_trades * 100
    avg_r = df_t['r_mult'].mean()
    total_pnl = df_t['pnl_usd'].sum()
    total_pts = df_t['pts'].sum()

    start_dt = pd.to_datetime(BACKTEST_START)
    end_dt   = pd.to_datetime(BACKTEST_END)
    trading_days = len(df_t['date'].unique())
    total_years  = (end_dt - start_dt).days / 365.25
    trades_per_day = total_trades / max(trading_days, 1)

    print("\n" + "─"*65)
    print("  REZULTATE GENERALE")
    print("─"*65)
    print(f"  Perioadă:          {BACKTEST_START} → {BACKTEST_END}  ({total_years:.1f} ani)")
    print(f"  Zile cu trades:    {trading_days}")
    print(f"  Total trades:      {total_trades:,}")
    print(f"  Trades/zi:         {trades_per_day:.2f}")
    print(f"  Win rate:          {wr:.1f}%")
    print(f"  Avg R/trade:       {avg_r:+.3f}R")
    print(f"  P&L total:         ${total_pnl:,.0f}")
    print(f"  Pts total:         {total_pts:.1f}")
    print(f"  BE exits:          {df_t['be'].sum()} ({df_t['be'].mean()*100:.1f}%)")

    # ─ Per clasă ────────────────────────────────────────────────────────────
    print("\n  PER CLASĂ:")
    print(f"  {'Clasă':<16} {'N':>5} {'WR%':>6} {'AvgR':>7} {'P&L':>10}")
    print("  " + "-"*50)
    for cls, name in REGIME_NAMES.items():
        if cls == 0:
            continue
        sub = df_t[df_t['pred_class'] == cls]
        if len(sub) == 0:
            print(f"  {name:<16} {'0':>5}")
            continue
        print(f"  {name:<16} {len(sub):>5} {sub['win'].mean()*100:>5.1f}% {sub['r_mult'].mean():>+6.3f}R  ${sub['pnl_usd'].sum():>9,.0f}")

    # ─ Per sesiune ──────────────────────────────────────────────────────────
    print("\n  PER SESIUNE:")
    print(f"  {'Sesiune':<10} {'N':>5} {'WR%':>6} {'AvgR':>7} {'P&L':>10}")
    print("  " + "-"*44)
    for sess in ['LON', 'NY']:
        sub = df_t[df_t['session'] == sess]
        if len(sub) == 0:
            print(f"  {sess:<10} {'0':>5}")
            continue
        print(f"  {sess:<10} {len(sub):>5} {sub['win'].mean()*100:>5.1f}% {sub['r_mult'].mean():>+6.3f}R  ${sub['pnl_usd'].sum():>9,.0f}")

    # ─ Per exit reason ────────────────────────────────────────────────────────
    print("\n  PER EXIT REASON:")
    for reason in df_t['exit_reason'].unique():
        sub = df_t[df_t['exit_reason'] == reason]
        print(f"  {reason:<20} N={len(sub):>4}  AvgR={sub['r_mult'].mean():+.3f}  P&L=${sub['pnl_usd'].sum():>9,.0f}")

    # ─ Per an ────────────────────────────────────────────────────────────────
    print("\n  PER AN:")
    print(f"  {'An':>5} {'N':>5} {'WR%':>6} {'AvgR':>7} {'P&L':>10} {'T/zi':>6}")
    print("  " + "-"*44)
    df_t['year'] = pd.to_datetime(df_t['date']).dt.year
    for yr, grp in df_t.groupby('year'):
        days_yr = len(grp['date'].unique())
        print(f"  {yr:>5} {len(grp):>5} {grp['win'].mean()*100:>5.1f}% {grp['r_mult'].mean():>+6.3f}R  ${grp['pnl_usd'].sum():>9,.0f}  {len(grp)/days_yr:>4.1f}")

    # ─ Confidence analysis ──────────────────────────────────────────────────
    print("\n  CONFIDENCE BUCKETS:")
    bins = [0.35, 0.45, 0.55, 0.65, 0.75, 1.01]
    labels = ['0.35-0.45','0.45-0.55','0.55-0.65','0.65-0.75','0.75+']
    df_t['conf_bucket'] = pd.cut(df_t['confidence'], bins=bins, labels=labels, right=False)
    for label, grp in df_t.groupby('conf_bucket', observed=True):
        if len(grp) == 0: continue
        print(f"  [{label}]  N={len(grp):>4}  WR={grp['win'].mean()*100:>4.1f}%  AvgR={grp['r_mult'].mean():>+.3f}  P&L=${grp['pnl_usd'].sum():>9,.0f}")

    # ─ Equity curve ─────────────────────────────────────────────────────────
    print("\n  EQUITY CURVE (monthly P&L):")
    df_t['month'] = pd.to_datetime(df_t['date']).dt.to_period('M')
    for m, grp in df_t.groupby('month'):
        pnl = grp['pnl_usd'].sum()
        bar = '█' * int(abs(pnl) / 200) if abs(pnl) >= 200 else ('▪' if pnl != 0 else ' ')
        sign = '+' if pnl >= 0 else ''
        print(f"  {m}  {sign}${pnl:>7,.0f}  {bar[:40]}")

    # ─ Salvare CSV ──────────────────────────────────────────────────────────
    out_csv = DIR / "backtest_mario_bot_trades.csv"
    df_t.to_csv(out_csv, index=False)
    print(f"\n✓ Trades salvate: {out_csv.name}")
    print("═"*65 + "\n")

    return df_t


if __name__ == "__main__":
    run_backtest()
