"""
ALADIN — Clustered Loss Analysis
Analizează UNDE și DE CE apar pierderile consecutive.
Folosește modelul deja antrenat + test set.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import os
import xgboost as xgb

DB_PATH   = "/Users/mario/Desktop/Aladin/mario_trading.db"
MODEL_PATH = "/Users/mario/Desktop/Aladin/mario_bot.json"
FEATURES_PATH = "/Users/mario/Desktop/Aladin/mario_features.json"

print("=" * 80)
print("🔍 CLUSTERED LOSS ANALYSIS — Unde apar pierderile consecutive?")
print("=" * 80)

# ── 1. Încarcă datele ──
print("\n📖 Încărcare date...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM market_data", conn)
conn.close()
print(f"   ✅ {len(df):,} bare încărcate")

# ── 2. Încarcă modelul și features ──
print("📦 Încărcare model...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

with open(FEATURES_PATH, 'r') as f:
    feat_meta = json.load(f)
features = feat_meta.get('features', [])
print(f"   ✅ Model încărcat cu {len(features)} features")

# ── 3. Recalculează features (identic cu train_mario_ai.py) ──
print("⚙️  Recalculare features...")

# Import funcțiile din train
import sys
sys.path.insert(0, "/Users/mario/Desktop/Aladin")
from train_mario_ai import add_reversal_features, add_amt_features, normalize_price_features

df = add_reversal_features(df)
df = add_amt_features(df)

# Features direcție
df['slope_h1']    = (df['close'] - df['close'].shift(60))  / (df['close'].shift(60).abs()  + 1e-8)
df['slope_h4']    = (df['close'] - df['close'].shift(240)) / (df['close'].shift(240).abs() + 1e-8)
df['momentum_15'] = (df['close'] - df['close'].shift(15))  / (df['close'].shift(15).abs()  + 1e-8)
df['body_dir']    = (df['close'] - df['open']) / (df['high'] - df['low']).clip(lower=1e-8)
df['wick_ratio']  = (df['high'] - df['low']) / (abs(df['close'] - df['open']) + 1e-8)
_body = (df['close'] - df['open']).abs()
_range = (df['high'] - df['low']).clip(lower=1e-8)
df['upper_wick']  = (df['high'] - df[['close', 'open']].max(axis=1)) / _range
df['lower_wick']  = (df[['close', 'open']].min(axis=1) - df['low']) / _range
df['wick_bias']   = df['upper_wick'] - df['lower_wick']
_log_ret = np.log(df['close'] / df['close'].shift(1))
df['realized_vol'] = _log_ret.rolling(20).std().fillna(0)
df['vol_of_vol']   = df['realized_vol'].rolling(20).std().fillna(0)
df['return_acf1']  = _log_ret.rolling(20).apply(
    lambda x: x.autocorr(lag=1) if len(x) >= 5 else 0, raw=False
).fillna(0)

# ATR + Target
_atr = df['atr_14'] if 'atr_14' in df.columns else (df['high'] - df['low']).rolling(14).mean()
_ATR_MULT = 1.0
_HORIZON  = 30
df['price_next'] = df['close'].shift(-_HORIZON)

# Target labeling (identic cu train)
df['target'] = 0
_move = df['price_next'] - df['close']
_atr_thresh = _atr * _ATR_MULT
df.loc[_move < -_atr_thresh, 'target'] = 1   # SHORT
df.loc[_move > _atr_thresh, 'target'] = 2    # LONG

# Killzone
if 'timestamp' in df.columns:
    _ts = pd.to_datetime(df['timestamp'])
    _time_decimal = _ts.dt.hour + _ts.dt.minute / 60.0
    _london = (_time_decimal >= 9.0) & (_time_decimal <= 11.0)
    _ny = (_time_decimal >= 15.5) & (_time_decimal <= 17.5)
    _in_killzone = _london | _ny
    df.loc[~_in_killzone, 'target'] = 0

df = df.dropna(subset=['price_next']).copy()

# ── 4. Test set (ultimele 20%) ──
split_val = int(len(df) * 0.80)
test_df = df.iloc[split_val:].copy()
print(f"   ✅ Test set: {len(test_df):,} bare")

# ── 5. Predicții ──
print("🤖 Rulare predicții pe test set...")
available = [f for f in features if f in test_df.columns]
X_test = test_df[available].ffill().fillna(0)
X_test = normalize_price_features(X_test)

y_proba = model.predict_proba(X_test)
y_pred  = np.argmax(y_proba, axis=1)

# Confidence filter
CONF_THRESHOLD = 0.45
for i in range(len(y_pred)):
    if y_pred[i] == 1 and y_proba[i, 1] < CONF_THRESHOLD:
        y_pred[i] = 0
    elif y_pred[i] == 2 and y_proba[i, 2] < CONF_THRESHOLD:
        y_pred[i] = 0

test_df = test_df.reset_index(drop=True)
test_df['pred'] = y_pred

# ── 6. Calculează PnL per trade ──
_returns = test_df['price_next'].values - test_df['close'].values
test_df['pnl'] = 0.0
_short = test_df['pred'] == 1
_long  = test_df['pred'] == 2
test_df.loc[_short, 'pnl'] = -_returns[_short]  # SHORT profit = -move
test_df.loc[_long, 'pnl']  =  _returns[_long]   # LONG profit = +move

# Doar trade-urile (non-WAIT)
trades = test_df[test_df['pred'] != 0].copy()
trades['is_loss'] = (trades['pnl'] < 0).astype(int)
trades['is_win']  = (trades['pnl'] > 0).astype(int)
print(f"   ✅ Total trades: {len(trades):,}")

# ── 7. Identifică losing streaks ──
print("\n" + "=" * 80)
print("📊 LOSING STREAKS ANALYSIS")
print("=" * 80)

streaks = []
cur_streak = 0
cur_streak_start = 0
cur_streak_pnl = 0

for idx, row in trades.iterrows():
    if row['pnl'] < 0:
        if cur_streak == 0:
            cur_streak_start = idx
        cur_streak += 1
        cur_streak_pnl += row['pnl']
    else:
        if cur_streak >= 5:  # doar streak-uri >= 5
            streaks.append({
                'start_idx': cur_streak_start,
                'length': cur_streak,
                'total_pnl': cur_streak_pnl,
                'avg_pnl': cur_streak_pnl / cur_streak,
            })
        cur_streak = 0
        cur_streak_pnl = 0

if cur_streak >= 5:
    streaks.append({
        'start_idx': cur_streak_start,
        'length': cur_streak,
        'total_pnl': cur_streak_pnl,
        'avg_pnl': cur_streak_pnl / cur_streak,
    })

print(f"\n   Streak-uri >= 5 pierderi: {len(streaks)}")
print(f"   Streak-uri >= 10: {sum(1 for s in streaks if s['length'] >= 10)}")
print(f"   Streak-uri >= 20: {sum(1 for s in streaks if s['length'] >= 20)}")
print(f"   Streak-uri >= 30: {sum(1 for s in streaks if s['length'] >= 30)}")

# ── 8. Analiză detaliată per streak ──
print("\n" + "=" * 80)
print("🔍 TOP 10 LOSING STREAKS — DETALII")
print("=" * 80)

# Sortăm după lungime (worst first)
streaks_sorted = sorted(streaks, key=lambda x: x['length'], reverse=True)

for i, s in enumerate(streaks_sorted[:10]):
    idx = s['start_idx']
    # Găsim informații despre contextul pieței
    streak_trades = trades.loc[idx:].head(s['length'])

    # Timestamp
    if 'timestamp' in test_df.columns:
        ts_start = test_df.loc[idx, 'timestamp'] if idx in test_df.index else 'N/A'
    else:
        ts_start = f"bar #{idx}"

    # Regim de piață
    atr_val = streak_trades['atr_14'].mean() if 'atr_14' in streak_trades.columns else 0
    vol_val = streak_trades['volume'].mean() if 'volume' in streak_trades.columns else 0

    # Realized vol (proxy pentru volatilitate)
    rv = streak_trades['realized_vol'].mean() if 'realized_vol' in streak_trades.columns else 0

    # Rotation factor (balance vs trend)
    rot = streak_trades['rotation_factor'].mean() if 'rotation_factor' in streak_trades.columns else 0

    # VA migration
    va_mig = streak_trades['va_migration'].mean() if 'va_migration' in streak_trades.columns else 0

    # Direcțiile trade-urilor
    n_short = (streak_trades['pred'] == 1).sum()
    n_long  = (streak_trades['pred'] == 2).sum()

    # Confidence medie
    streak_proba = y_proba[streak_trades.index - test_df.index[0]]
    conf_short = streak_proba[streak_trades['pred'].values == 1, 1].mean() if n_short > 0 else 0
    conf_long  = streak_proba[streak_trades['pred'].values == 2, 2].mean() if n_long > 0 else 0

    # Failed auction & excess
    fa = streak_trades['failed_auction'].mean() if 'failed_auction' in streak_trades.columns else 0
    exc = streak_trades['excess'].mean() if 'excess' in streak_trades.columns else 0

    # Inside VA
    iva = streak_trades['inside_va'].mean() if 'inside_va' in streak_trades.columns else 0

    # Initiative/Responsive
    ir = streak_trades['initiative_responsive'].mean() if 'initiative_responsive' in streak_trades.columns else 0

    print(f"\n   {'─' * 60}")
    print(f"   #{i+1} | Lungime: {s['length']} pierderi | PnL: {s['total_pnl']:,.1f} pts ({s['total_pnl']*20:,.0f}$)")
    print(f"   Start: {ts_start}")
    print(f"   Direcții: {n_short} SHORT + {n_long} LONG")
    print(f"   Conf medie: SHORT={conf_short:.2f} LONG={conf_long:.2f}")
    print(f"   📊 Context piață:")
    print(f"      ATR: {atr_val:.1f} | Volume: {vol_val:,.0f} | Realized Vol: {rv:.4f}")
    print(f"      Rotation: {rot:.2f} (0=trend, 1=range)")
    print(f"      VA Migration: {va_mig:+.3f}")
    print(f"      Inside VA: {iva:.1%}")
    print(f"      Failed Auction: {fa:+.2f} | Excess: {exc:+.2f}")
    print(f"      Initiative/Responsive: {ir:+.2f}")

# ── 9. Comparație: LOSING STREAKS vs WINNING STREAKS ──
print("\n" + "=" * 80)
print("📊 COMPARAȚIE: Condiții piață LOSS vs WIN")
print("=" * 80)

trades_loss = trades[trades['pnl'] < 0].copy()
trades_win  = trades[trades['pnl'] > 0].copy()

compare_features = ['atr_14', 'realized_vol', 'rotation_factor', 'va_migration',
                    'inside_va', 'failed_auction', 'excess', 'initiative_responsive',
                    'volume', 'body_dir', 'reversal_strength', 'swing_break',
                    'poor_high', 'poor_low', 'trend_exhaustion', 'adx_14']

print(f"\n   {'Feature':<25} {'Win Mean':>10} {'Loss Mean':>10} {'Diferență':>12}")
print(f"   {'─'*60}")
for feat in compare_features:
    if feat in trades.columns:
        w_mean = trades_win[feat].mean()
        l_mean = trades_loss[feat].mean()
        diff = w_mean - l_mean
        marker = " ⬅️" if abs(diff) > 0.01 else ""
        print(f"   {feat:<25} {w_mean:>10.4f} {l_mean:>10.4f} {diff:>+12.4f}{marker}")

# ── 10. Regime analysis pe losing streaks ──
print("\n" + "=" * 80)
print("📊 REGIME ANALYSIS — Unde pierde modelul cel mai mult?")
print("=" * 80)

# Clasificare regim
def classify_regime(row):
    rv = row.get('realized_vol', 0)
    adx = row.get('adx_14', 0)
    rot = row.get('rotation_factor', 0.5)

    if rv > 0.003:   # realized vol >0.3%
        return 'HIGH_VOL'
    elif adx > 25:
        return 'TREND'
    elif rot > 0.6:
        return 'RANGE'
    else:
        return 'NORMAL'

trades['regime'] = trades.apply(classify_regime, axis=1)

print(f"\n   {'Regime':<12} {'Total':>7} {'Wins':>7} {'Losses':>7} {'WR':>7} {'Avg PnL':>10} {'Total PnL':>12}")
print(f"   {'─'*70}")
for regime in ['TREND', 'RANGE', 'NORMAL', 'HIGH_VOL']:
    r = trades[trades['regime'] == regime]
    if len(r) == 0:
        continue
    wins = (r['pnl'] > 0).sum()
    losses = (r['pnl'] < 0).sum()
    wr = wins / len(r) * 100
    avg = r['pnl'].mean()
    total = r['pnl'].sum()
    marker = " ⚠️" if wr < 60 else " ✅" if wr > 70 else ""
    print(f"   {regime:<12} {len(r):>7,} {wins:>7,} {losses:>7,} {wr:>6.1f}% {avg:>+9.1f} {total:>+11,.0f}{marker}")

# ── 11. Killzone analysis ──
print("\n" + "=" * 80)
print("📊 KILLZONE ANALYSIS — Performanță per sesiune")
print("=" * 80)

if 'timestamp' in test_df.columns:
    trades['_ts'] = pd.to_datetime(test_df.loc[trades.index, 'timestamp'].values)
    trades['_hour'] = trades['_ts'].dt.hour + trades['_ts'].dt.minute / 60.0

    trades['killzone'] = 'OTHER'
    trades.loc[(trades['_hour'] >= 9.0) & (trades['_hour'] <= 11.0), 'killzone'] = 'LONDON'
    trades.loc[(trades['_hour'] >= 15.5) & (trades['_hour'] <= 17.5), 'killzone'] = 'NY'

    print(f"\n   {'Killzone':<12} {'Total':>7} {'Wins':>7} {'Losses':>7} {'WR':>7} {'Avg PnL':>10}")
    print(f"   {'─'*55}")
    for kz in ['LONDON', 'NY', 'OTHER']:
        r = trades[trades['killzone'] == kz]
        if len(r) == 0:
            continue
        wins = (r['pnl'] > 0).sum()
        losses = (r['pnl'] < 0).sum()
        wr = wins / len(r) * 100
        avg = r['pnl'].mean()
        print(f"   {kz:<12} {len(r):>7,} {wins:>7,} {losses:>7,} {wr:>6.1f}% {avg:>+9.1f}")

# ── 12. Confidence vs Outcome ──
print("\n" + "=" * 80)
print("📊 CONFIDENCE BUCKETS — Merge confidence-ul cu accuracy?")
print("=" * 80)

trades['confidence'] = 0.0
for idx in trades.index:
    p = y_pred[idx - test_df.index[0]]
    prob = y_proba[idx - test_df.index[0]]
    if p == 1:
        trades.loc[idx, 'confidence'] = prob[1]
    elif p == 2:
        trades.loc[idx, 'confidence'] = prob[2]

buckets = [(0.45, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.0)]
print(f"\n   {'Bucket':<15} {'Trades':>7} {'WR':>7} {'Avg PnL':>10} {'PF':>7}")
print(f"   {'─'*50}")
for lo, hi in buckets:
    b = trades[(trades['confidence'] >= lo) & (trades['confidence'] < hi)]
    if len(b) == 0:
        continue
    wr = (b['pnl'] > 0).sum() / len(b) * 100
    avg = b['pnl'].mean()
    gp = b['pnl'][b['pnl'] > 0].sum()
    gl = abs(b['pnl'][b['pnl'] < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    print(f"   {lo:.0%}-{hi:.0%}         {len(b):>7,} {wr:>6.1f}% {avg:>+9.1f} {pf:>6.2f}")

print("\n" + "=" * 80)
print("✅ ANALIZĂ COMPLETĂ")
print("=" * 80)
