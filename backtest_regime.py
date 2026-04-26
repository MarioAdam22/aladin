"""
backtest_regime.py — Regime Classifier Backtest
================================================
Simulare prop firm Lucid Trading cu și fără regime filter.

Setup:
  - IS:  2022-01-01 → 2024-12-31
  - OOS: 2025-01-01 → 2026-04-08

Prop Firm Lucid:
  - Cont: $50,000
  - Trailing DD: $2,000
  - Payout la: profit ≥ $1,000 (equity ≥ $51,000)
  - Payout reset: cont revine la $50,000, DD floor resetat
  - 1 contract NQ = $20/punct
  - Max 2 trades/zi

Trade params (NOM/LOM style):
  - SL: 12 puncte
  - TP: 24 puncte (2R)
  - Entry: la displacement bar close în sesiunea activă

Compară 3 scenarii:
  A. FĂRĂ regime filter (toate sweep+disp semnale)
  B. CU regime filter (doar PRE_EXPANSION ≥ 0.65)
  C. REGIME NEGATIV: semnale în CONSOLIDATION (câte pierde sistemul)
"""

import sqlite3, joblib, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH       = Path(__file__).parent / "mario_trading.db"
MODEL_PATH    = Path(__file__).parent / "regime_classifier_v1.pkl"

IS_START  = "2022-01-01"
IS_END    = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END   = "2026-04-08"

# Prop firm
ACCOUNT_START  = 50_000.0
TRAILING_DD    = 2_000.0
PAYOUT_TARGET  = 1_000.0   # profit necesar pentru payout
NQ_PER_POINT   = 20.0
SL_PT          = 12.0
TP_PT          = 24.0
MAX_TRADES_DAY = 2
REGIME_THRESH  = 0.65       # prag minim pentru regime routing

# Sesiuni ET (DB e în ET)
LON_OPEN = ('04:00', '07:00')
NY_OPEN  = ('09:00', '11:30')

FEATURES = [
    'adx_14', 'hurst', 'garch_vol', 'kalman_smooth',
    'acf_lag1', 'acf_lag5', 'fisher_transform', 'sample_entropy',
    'inside_va', 'dist_vwap_atr', 'dist_poc_atr', 'dist_pdh_atr', 'dist_pdl_atr',
    'has_displacement', 'body_size_atr', 'rvol',
    'bar_delta_atr', 'cum_delta_20_atr',
    'delta_at_high_atr', 'delta_at_low_atr',
    'big_buy_count', 'big_sell_count',
    'imbalance_pct', 'dom_ratio',
    'hhmm_enc', 'is_session_open',
    'dist_sess_hi_atr', 'dist_sess_lo_atr',
    'h4_bias_atr', 'h1_bias_atr', 'above_true_open_atr',
    'day_of_week', 'month',
    'fvg_up', 'fvg_down',
    'pre_range_atr', 'sweep_dn_atr', 'sweep_up_atr',
]


# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    pkg = joblib.load(MODEL_PATH)
    return pkg['model'], pkg['label_encoder'], pkg['features'], pkg['regimes']


# ── Load data ─────────────────────────────────────────────────────────────────
def load_period(start, end):
    print(f"  Loading {start} → {end} ...")
    conn = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True, timeout=30,
                           check_same_thread=False)
    # Încarcă doar barele relevante: sesiuni + 60 bare forward per setup
    # → filtrăm la sesiuni + noapte pentru context
    every_n = 3  # sample every 3rd bar pentru memorie, dar păstrăm sesiunile
    df = pd.read_sql(f"""
        SELECT timestamp, date, hour_min, open, high, low, close, volume,
               adx_14, hurst, garch_vol, kalman_smooth,
               acf_lag1, acf_lag5, fisher_transform, sample_entropy,
               inside_va, dist_vwap, dist_poc, dist_pdh, dist_pdl,
               has_displacement, body_size, rvol,
               bar_delta, cum_delta,
               delta_at_high, delta_at_low, big_buy_count, big_sell_count,
               imbalance_pct, dom_ratio, fvg_up, fvg_down,
               atr_14, true_open, h4_hi, h4_lo, h1_hi, h1_lo,
               lon_hi, lon_lo, p_hi, p_lo,
               day_of_week, month
        FROM market_data
        WHERE date BETWEEN '{start}' AND '{end}'
          AND adx_14 > 0 AND atr_14 > 0
          AND (ROWID % {every_n} = 0
               OR hour_min BETWEEN '04:00' AND '12:00')
        ORDER BY timestamp
    """, conn)
    conn.close()
    print(f"  → {len(df):,} bare, {df['date'].nunique()} zile")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df):
    df = df.copy()
    atr = df['atr_14'].clip(lower=0.01)

    df['dist_vwap_atr']     = df['dist_vwap'].abs() / atr
    df['dist_poc_atr']      = df['dist_poc'].abs() / atr
    df['dist_pdh_atr']      = df['dist_pdh'].abs() / atr
    df['dist_pdl_atr']      = df['dist_pdl'].abs() / atr
    df['body_size_atr']     = df['body_size'] / atr
    df['bar_delta_atr']     = df['bar_delta'].abs() / df['volume'].clip(lower=1)
    df['delta_at_high_atr'] = df['delta_at_high'].abs() / atr
    df['delta_at_low_atr']  = df['delta_at_low'].abs() / atr
    df['cum_delta_20_atr']  = df['cum_delta'].rolling(20, min_periods=1).sum() / atr

    df['hhmm_enc'] = df['hour_min'].str.replace(':', '').astype(int)
    h = df['hhmm_enc']
    df['is_session_open'] = (
        ((h >= 400) & (h <= 700)) | ((h >= 900) & (h <= 1130))
    ).astype(int)

    df['sess_hi']        = df[['lon_hi', 'p_hi']].max(axis=1)
    sess_lo_vals         = df[['lon_lo', 'p_lo']].replace(0, np.nan).min(axis=1).fillna(df['close'] - atr * 5)
    df['sess_lo']        = sess_lo_vals
    df['dist_sess_hi_atr'] = (df['sess_hi'] - df['close']).abs() / atr
    df['dist_sess_lo_atr'] = (df['close'] - df['sess_lo']).abs() / atr

    df['h4_mid']            = (df['h4_hi'] + df['h4_lo']) / 2
    df['h1_mid']            = (df['h1_hi'] + df['h1_lo']) / 2
    df['h4_bias_atr']       = (df['close'] - df['h4_mid']) / atr
    df['h1_bias_atr']       = (df['close'] - df['h1_mid']) / atr
    df['above_true_open_atr'] = (df['close'] - df['true_open']) / atr

    df['pre_hi']    = df[['p_hi', 'lon_hi']].max(axis=1)
    pre_lo_vals     = df[['p_lo', 'lon_lo']].replace(0, np.nan).min(axis=1).fillna(df['close'] - atr * 10)
    df['pre_lo']    = pre_lo_vals
    df['pre_range'] = (df['pre_hi'] - df['pre_lo']).clip(lower=0.01)
    df['pre_range_atr'] = df['pre_range'] / atr
    df['sweep_dn_atr']  = (df['pre_lo'] - df['low']).clip(lower=0) / atr
    df['sweep_up_atr']  = (df['high'] - df['pre_hi']).clip(lower=0) / atr

    return df


# ── Predict regime pentru fiecare bară ────────────────────────────────────────
def predict_regimes(df, model, le, feat_cols):
    X = df[feat_cols].fillna(0).astype(float)
    probs     = model.predict_proba(X)
    pred_enc  = np.argmax(probs, axis=1)
    pred_prob = probs[np.arange(len(probs)), pred_enc]
    pred_cls  = le.classes_[pred_enc]

    regime_map = {0: 'CONSOLIDATION', 1: 'PRE_EXPANSION', 2: 'EXPANSION',
                  3: 'RETRACEMENT',   4: 'DISTRIBUTION'}
    df = df.copy()
    df['regime']      = [regime_map.get(c, 'UNKNOWN') for c in pred_cls]
    df['regime_prob'] = pred_prob
    return df


# ── Detectează setups sweep+displacement ─────────────────────────────────────
def find_setups(df):
    """
    Găsește bare cu sweep + displacement la session open.
    Returnează lista de setup-uri cu: date, bar_idx, direction, entry, atr
    """
    setups = []
    df_sess = df[df['is_session_open'] == 1].copy()

    for idx in df_sess.index:
        row = df.loc[idx]
        atr = float(row['atr_14'])
        if atr <= 0:
            continue

        sw_dn = float(row['sweep_dn_atr'])
        sw_up = float(row['sweep_up_atr'])
        has_d = int(row['has_displacement'])

        # Minim sweep 0.3×ATR + displacement confirmat
        if has_d == 0:
            continue
        if sw_dn < 0.3 and sw_up < 0.3:
            continue

        direction = 'LONG' if sw_dn >= sw_up else 'SHORT'
        entry = float(row['close'])

        setups.append({
            'date':       row['date'],
            'hour_min':   row['hour_min'],
            'bar_idx':    idx,
            'direction':  direction,
            'entry':      entry,
            'atr':        atr,
            'regime':     row['regime'],
            'regime_prob': float(row['regime_prob']),
        })

    return setups


# ── Simulare trade ────────────────────────────────────────────────────────────
def simulate_trade(df, setup):
    """
    Simulează un trade din momentul setup-ului.
    Returnează: pnl_pt, outcome ('TP'/'SL'/'OPEN')
    """
    idx       = setup['bar_idx']
    direction = setup['direction']
    entry     = setup['entry']
    sl_pt     = SL_PT
    tp_pt     = TP_PT

    sl = entry - sl_pt if direction == 'LONG' else entry + sl_pt
    tp = entry + tp_pt if direction == 'LONG' else entry - tp_pt

    # Caută în următoarele 60 bare
    future = df.loc[idx+1:idx+60]
    for _, frow in future.iterrows():
        hi = float(frow['high'])
        lo = float(frow['low'])
        if direction == 'LONG':
            if lo <= sl:
                return -sl_pt, 'SL'
            if hi >= tp:
                return tp_pt, 'TP'
        else:
            if hi >= sl:
                return -sl_pt, 'SL'
            if lo <= tp:
                return tp_pt, 'TP'

    # Nu a atins nici SL nici TP → close la ultimul preț disponibil
    if len(future) > 0:
        last_close = float(future.iloc[-1]['close'])
        pnl = (last_close - entry) * (1 if direction == 'LONG' else -1)
        return pnl, 'OPEN'
    return 0.0, 'OPEN'


# ── Prop firm simulation ──────────────────────────────────────────────────────
def prop_sim(trades_df):
    """
    Simulează prop firm Lucid cu trailing drawdown.
    Input: DataFrame cu coloanele [date, pnl_pt, outcome]
    """
    equity       = ACCOUNT_START
    peak_equity  = ACCOUNT_START
    dd_floor     = ACCOUNT_START - TRAILING_DD  # floor-ul curent
    payouts      = 0
    total_profit = 0.0
    blown        = False
    equity_curve = []
    payout_dates = []
    max_dd       = 0.0

    trades_per_day = {}

    for _, row in trades_df.iterrows():
        date = row['date']
        pnl_pt = float(row['pnl_pt'])
        pnl_usd = pnl_pt * NQ_PER_POINT

        # Max 2 trades/zi
        trades_per_day[date] = trades_per_day.get(date, 0) + 1
        if trades_per_day[date] > MAX_TRADES_DAY:
            continue

        equity += pnl_usd

        # Update trailing DD floor
        if equity > peak_equity:
            peak_equity = equity
            dd_floor    = peak_equity - TRAILING_DD

        # Check blown
        if equity <= dd_floor:
            blown = True
            equity_curve.append({'date': date, 'equity': equity, 'blown': True})
            break

        # Check payout
        if equity >= ACCOUNT_START + PAYOUT_TARGET:
            payouts     += 1
            profit_made  = equity - ACCOUNT_START
            total_profit += profit_made
            payout_dates.append({'date': date, 'profit': profit_made, 'payout_n': payouts})
            equity      = ACCOUNT_START
            peak_equity = ACCOUNT_START
            dd_floor    = ACCOUNT_START - TRAILING_DD

        dd_now = peak_equity - equity
        max_dd = max(max_dd, dd_now)
        equity_curve.append({'date': date, 'equity': equity, 'blown': False})

    return {
        'payouts':      payouts,
        'total_profit': round(total_profit, 0),
        'final_equity': round(equity, 0),
        'blown':        blown,
        'max_dd':       round(max_dd, 0),
        'payout_dates': payout_dates,
        'equity_curve': equity_curve,
    }


# ── Run backtest ──────────────────────────────────────────────────────────────
def run_backtest(df, label, regime_filter=None):
    """
    regime_filter:
      None        → toate semnalele (fără filtru)
      'PRE_EXPANSION' → doar PRE_EXPANSION ≥ REGIME_THRESH
      'NO_CONSOLIDATION' → exclude CONSOLIDATION
    """
    setups = find_setups(df)
    print(f"\n  {label}: {len(setups)} setups găsite")

    results = []
    for s in setups:
        # Aplică filtrul de regim
        if regime_filter == 'PRE_EXPANSION':
            if s['regime'] != 'PRE_EXPANSION' or s['regime_prob'] < REGIME_THRESH:
                continue
        elif regime_filter == 'NO_CONSOLIDATION':
            if s['regime'] == 'CONSOLIDATION' and s['regime_prob'] >= REGIME_THRESH:
                continue

        pnl_pt, outcome = simulate_trade(df, s)
        results.append({
            'date':    s['date'],
            'pnl_pt':  pnl_pt,
            'outcome': outcome,
            'regime':  s['regime'],
            'entry':   s['entry'],
            'direction': s['direction'],
        })

    if not results:
        print(f"  {label}: niciun trade după filtru")
        return None, None

    trades_df = pd.DataFrame(results)
    total     = len(trades_df)
    wins      = (trades_df['pnl_pt'] > 0).sum()
    losses    = (trades_df['pnl_pt'] < 0).sum()
    wr        = wins / total * 100
    avg_win   = trades_df[trades_df['pnl_pt'] > 0]['pnl_pt'].mean() if wins > 0 else 0
    avg_loss  = trades_df[trades_df['pnl_pt'] < 0]['pnl_pt'].mean() if losses > 0 else 0
    net_pt    = trades_df['pnl_pt'].sum()
    net_usd   = net_pt * NQ_PER_POINT

    print(f"  Trades: {total} | WR: {wr:.1f}% | Net: {net_pt:.0f}pt (${net_usd:,.0f})")
    print(f"  Avg Win: {avg_win:.1f}pt | Avg Loss: {avg_loss:.1f}pt | "
          f"Expectancy: {(wr/100*avg_win + (1-wr/100)*avg_loss):.2f}pt/trade")

    # Regime breakdown
    rg = trades_df.groupby('regime')['pnl_pt'].agg(['count','sum','mean'])
    rg['wr%'] = trades_df.groupby('regime').apply(
        lambda x: (x['pnl_pt']>0).mean()*100)
    print(f"\n  Breakdown pe regim:")
    print(rg.round(2).to_string())

    # Prop firm sim
    sim = prop_sim(trades_df.sort_values('date'))
    print(f"\n  🏦 PROP FIRM SIM:")
    print(f"     Payouts:       {sim['payouts']}")
    print(f"     Total profit:  ${sim['total_profit']:,.0f}")
    print(f"     Max DD:        ${sim['max_dd']:,.0f}")
    print(f"     Blown:         {'💥 DA' if sim['blown'] else '✅ NU'}")
    if sim['payout_dates']:
        print(f"     1st payout:    {sim['payout_dates'][0]['date']}")
        if len(sim['payout_dates']) >= 2:
            d1 = pd.to_datetime(sim['payout_dates'][0]['date'])
            d2 = pd.to_datetime(sim['payout_dates'][1]['date'])
            print(f"     2nd payout:    {sim['payout_dates'][1]['date']} "
                  f"({(d2-d1).days} zile după 1st)")

    return trades_df, sim


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("  REGIME CLASSIFIER — Backtest Prop Firm")
    print("=" * 70)

    model, le, feat_cols, regimes = load_model()

    for period_name, start, end in [
        ("IS  2022-2024", IS_START,  IS_END),
        ("OOS 2025-2026", OOS_START, OOS_END),
    ]:
        print(f"\n{'='*70}")
        print(f"  PERIOADĂ: {period_name}")
        print(f"{'='*70}")

        df = load_period(start, end)
        df = build_features(df)
        df = predict_regimes(df, model, le, feat_cols)

        # Distribuție regimuri în perioadă
        sess_df = df[df['is_session_open'] == 1]
        rg_dist = sess_df['regime'].value_counts()
        print(f"\n  Distribuție regim la session open:")
        for r, c in rg_dist.items():
            pct = 100 * c / len(sess_df)
            print(f"    {r:15s}: {c:6,} bare ({pct:.1f}%)")

        print(f"\n{'─'*70}")
        print(f"  SCENARIUL A — FĂRĂ filtru (toate semnalele)")
        print(f"{'─'*70}")
        trades_a, sim_a = run_backtest(df, "Fără filtru", regime_filter=None)

        print(f"\n{'─'*70}")
        print(f"  SCENARIUL B — CU regime filter (doar PRE_EXPANSION ≥ {REGIME_THRESH})")
        print(f"{'─'*70}")
        trades_b, sim_b = run_backtest(df, "PRE_EXPANSION filter",
                                       regime_filter='PRE_EXPANSION')

        print(f"\n{'─'*70}")
        print(f"  SCENARIUL C — Exclude CONSOLIDATION")
        print(f"{'─'*70}")
        trades_c, sim_c = run_backtest(df, "No-CONSOLIDATION filter",
                                       regime_filter='NO_CONSOLIDATION')

        # Comparație
        if trades_a is not None and trades_b is not None:
            wr_a = (trades_a['pnl_pt'] > 0).mean() * 100
            wr_b = (trades_b['pnl_pt'] > 0).mean() * 100
            net_a = trades_a['pnl_pt'].sum() * NQ_PER_POINT
            net_b = trades_b['pnl_pt'].sum() * NQ_PER_POINT
            print(f"\n  📊 COMPARAȚIE A vs B:")
            print(f"     WR:     {wr_a:.1f}% → {wr_b:.1f}% "
                  f"({'▲' if wr_b > wr_a else '▼'}{abs(wr_b-wr_a):.1f}%)")
            print(f"     Net $:  ${net_a:,.0f} → ${net_b:,.0f}")
            print(f"     Trades: {len(trades_a)} → {len(trades_b)} "
                  f"({len(trades_b)/len(trades_a)*100:.0f}% din total)")
            if sim_a and sim_b:
                print(f"     Payouts: {sim_a['payouts']} → {sim_b['payouts']}")
                print(f"     Blown:   {'DA' if sim_a['blown'] else 'NU'} → "
                      f"{'DA' if sim_b['blown'] else 'NU'}")

    print("\n✅ Backtest complet.")
