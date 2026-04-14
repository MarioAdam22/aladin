
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║          ALADIN QUANTUM-ICT v5.0 — DATA PIPELINE                                        ║
║          build_ai_dataset.py  |  ICT + AMT + SMT  →  SQLite DB                         ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Pipeline:
  1. Incarcare CSV (QQQ + SPY) cu timestamp UTC → Europe/Bucharest
  2. Ierarhie HTF completă (Weekly, Monthly, Monday, Previous Day)
  3. Multi-Timeframe (H4, H1) prin resample
  4. Sessions (Asia, London) cu Midnight Open
  5. Auction Market Theory — POC + Value Area 70%
  6. ICT Core — FVG, Displacement, Power of 3
  7. Distance Features pentru XGBoost
  8. SMT Divergence (QQQ vs SPY swing-based)
  9. Salvare SQL cu indexuri optimizate

Fixes aplicate față de versiunea anterioară:
  - Data leakage fix: shift corect pe date, nu pe grup
  - Previous Day H/L corect cu reset_index + shift
  - 5 features noi: dist_poc, inside_va, dist_pdh, dist_pdl, atr_14
  - Index suplimentar pe date și hour_min
"""

import pandas as pd
import numpy as np
import sqlite3
import os

# =============================================================================
# CONFIG
# =============================================================================
PATH_QQQ = "/Users/mario/Desktop/Aladin/QQQ_24h_ICT_MASTER.csv"
PATH_SPY = "/Users/mario/Desktop/Aladin/SPY_24h_ICT_MASTER.csv"
PATH_DB  = "/Users/mario/Desktop/Aladin/mario_trading.db"


# =============================================================================
# HELPER — ADD TIMEFRAME LEVELS (H4, H1)
# =============================================================================
def add_tf(df_in: pd.DataFrame, resample_period: str, prefix: str) -> pd.DataFrame:
    """
    Resample la timeframe superior, shift(1) pentru a preveni lookahead bias,
    și merge înapoi pe timestamp.
    """
    df_idx = df_in.set_index('timestamp')
    temp   = df_idx.resample(resample_period).agg({'high': 'max', 'low': 'min'})
    temp   = temp.shift(1).rename(columns={'high': f'{prefix}_hi', 'low': f'{prefix}_lo'})
    temp   = temp.reset_index()
    # Reindex pentru fiecare bară: forward-fill din perioadele superioare
    merged = pd.merge_asof(
        df_in.sort_values('timestamp'),
        temp.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    return merged


# =============================================================================
# HELPER — SESSION RANGES (Asia, London)
# =============================================================================
def get_session(df: pd.DataFrame, start: str, end: str, prefix: str) -> pd.DataFrame:
    """Calculează High/Low per date pentru fereastra orară dată."""
    mask = (df['hour_min'] >= start) & (df['hour_min'] < end)
    sess = (
        df[mask]
        .groupby('date')
        .agg(**{f'{prefix}_hi': ('high', 'max'), f'{prefix}_lo': ('low', 'min')})
        .reset_index()
    )
    return sess


# =============================================================================
# HELPER — AMT (POC + Value Area 70%)
# =============================================================================
def get_amt_metrics(group: pd.DataFrame) -> pd.Series:
    """
    Point of Control = prețul cu cel mai mare volum.
    Value Area = prețurile care acoperă 70% din volumul zilei.
    """
    if group.empty or group['volume'].sum() == 0:
        return pd.Series([np.nan, np.nan, np.nan], index=['poc_level', 'vah', 'val'])

    vol_dist = (
        group.groupby('price_bin')['volume']
        .sum()
        .sort_values(ascending=False)
    )
    poc = vol_dist.idxmax()

    total_vol   = vol_dist.sum()
    target_vol  = total_vol * 0.70
    current_vol = vol_dist.iloc[0]
    v_bins      = [poc]

    for price, vol in vol_dist.iloc[1:].items():
        if current_vol >= target_vol:
            break
        current_vol += vol
        v_bins.append(price)

    return pd.Series(
        [poc, max(v_bins), min(v_bins)],
        index=['poc_level', 'vah', 'val']
    )


# =============================================================================
# CORE — PROCESS ONE SYMBOL
# =============================================================================
def process_symbol_data(file_path: str, symbol_name: str) -> pd.DataFrame | None:
    """
    Procesează un CSV brut (QQQ sau SPY) și calculează toate features ICT + AMT.
    Returnează DataFrame complet sau None dacă fișierul lipsește.
    """
    if not os.path.exists(file_path):
        print(f"❌ [{symbol_name}] Fișier negăsit: {file_path}")
        return None

    # ── Pasul 1: Incarcare & Timestamp ──────────────────────────────────────
    print(f"🚀 [{symbol_name}] Pasul 1: Încărcare și conversie timestamp...")
    df = pd.read_csv(file_path, sep=';')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Bucharest')

    df['date']        = df['timestamp'].dt.date
    df['hour_min']    = df['timestamp'].dt.strftime('%H:%M')
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['week_id']     = df['timestamp'].dt.isocalendar().week.astype(int)
    df['month']       = df['timestamp'].dt.month
    df['year']        = df['timestamp'].dt.year

    print(f"   ✅ {len(df):,} rânduri încărcate pentru {symbol_name}")

    # ── Pasul 2: HTF — Weekly High/Low (Săptămâna anterioară) ──────────────
    print(f"📅 [{symbol_name}] Pasul 2: Ierarhie HTF completă (W, M, D, Monday)...")

    # FIX: grupăm, calculăm, apoi shiftem date → nu grup (previne data leakage)
    lw = (
        df.groupby(['year', 'week_id'])
        .agg(lw_hi=('high', 'max'), lw_lo=('low', 'min'))
        .reset_index()
    )
    # Shift: fiecare săptămână primește valorile săptămânii ANTERIOARE
    lw = lw.sort_values(['year', 'week_id']).copy()
    lw['lw_hi'] = lw['lw_hi'].shift(1)
    lw['lw_lo'] = lw['lw_lo'].shift(1)
    df = df.merge(lw, on=['year', 'week_id'], how='left')

    # ── HTF — Monthly High/Low ───────────────────────────────────────────────
    lm = (
        df.groupby(['year', 'month'])
        .agg(lm_hi=('high', 'max'), lm_lo=('low', 'min'))
        .reset_index()
    )
    lm = lm.sort_values(['year', 'month']).copy()
    lm['lm_hi'] = lm['lm_hi'].shift(1)
    lm['lm_lo'] = lm['lm_lo'].shift(1)
    df = df.merge(lm, on=['year', 'month'], how='left')

    # ── Monday High/Low (range-ul săptămânii de luni) ───────────────────────
    monday_data = (
        df[df['day_of_week'] == 0]
        .groupby(['year', 'week_id'])
        .agg(m_hi=('high', 'max'), m_lo=('low', 'min'))
        .reset_index()
    )
    df = df.merge(monday_data, on=['year', 'week_id'], how='left')

    # ── Previous Day High/Low — FIX CORECT ──────────────────────────────────
    pd_hl = (
        df.groupby('date')
        .agg(p_hi=('high', 'max'), p_lo=('low', 'min'))
        .reset_index()
    )
    pd_hl = pd_hl.sort_values('date').copy()
    pd_hl['p_hi'] = pd_hl['p_hi'].shift(1)   # ziua anterioară, nu cea curentă
    pd_hl['p_lo'] = pd_hl['p_lo'].shift(1)
    df = df.merge(pd_hl, on='date', how='left')

    # ── Pasul 3: Multi-Timeframe (H4, H1) ───────────────────────────────────
    print(f"⏳ [{symbol_name}] Pasul 3: Multi-Timeframe (H4, H1)...")
    df = add_tf(df, '4h', 'h4')
    df = add_tf(df, '1h', 'h1')

    # ── Pasul 4: Midnight Open & Sessions ───────────────────────────────────
    print(f"⚓ [{symbol_name}] Pasul 4: True Open (07:00 RO) & Sessions...")
    day_open = (
        df[df['hour_min'] >= '07:00']
        .groupby('date')['open']
        .first()
        .rename('true_open')
        .reset_index()
    )
    df = df.merge(day_open, on='date', how='left')

    asia = get_session(df, '02:00', '07:00', 'asia')
    lon  = get_session(df, '09:00', '14:30', 'lon')
    df   = df.merge(asia, on='date', how='left')
    df   = df.merge(lon,  on='date', how='left')

    # ── Pasul 5: AMT — POC & Value Area ─────────────────────────────────────
    print(f"📊 [{symbol_name}] Pasul 5: Auction Market Theory (POC & Value Area 70%)...")
    df['price_bin'] = df['close'].round(2)
    amt_data = (
        df.groupby('date')
        .apply(get_amt_metrics, include_groups=False)
        .reset_index()
    )
    df = df.merge(amt_data, on='date', how='left')

    # ── Pasul 6: ICT Core ────────────────────────────────────────────────────
    print(f"🕯️ [{symbol_name}] Pasul 6: ICT Core (FVG, Displacement, Power of 3)...")

    # Fair Value Gap Bullish: low[i] > high[i-2]  (gap de nelichidate)
    df['fvg_up'] = (
        (df['low'] > df['high'].shift(2)) &
        (df['low'].shift(1) > df['high'].shift(2))
    ).astype(int)

    # Fair Value Gap Bearish
    df['fvg_down'] = (
        (df['high'] < df['low'].shift(2)) &
        (df['high'].shift(1) < df['low'].shift(2))
    ).astype(int)

    # Displacement (corp > 1.5x medie rolling 20)
    df['body_size']        = abs(df['close'] - df['open'])
    df['has_displacement'] = (
        df['body_size'] > df['body_size'].rolling(20).mean() * 1.5
    ).astype(int)

    # Power of 3 — price above/below Midnight Open
    df['is_above_open'] = (df['close'] > df['true_open'].fillna(df['open'])).astype(int)

    # ATR 14 (feature nou — important pentru sizing)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()

    # ── Pasul 7: Distance Features (noi — recomandate) ───────────────────────
    print(f"📐 [{symbol_name}] Pasul 7: Distance Features (dist_poc, inside_va, dist_pdh, dist_pdl)...")

    df['dist_poc']  = df['close'] - df['poc_level'].fillna(df['close'])
    df['inside_va'] = (
        (df['close'] >= df['val'].fillna(df['low'])) &
        (df['close'] <= df['vah'].fillna(df['high']))
    ).astype(int)
    df['dist_pdh']  = df['close'] - df['p_hi'].fillna(df['high'])
    df['dist_pdl']  = df['close'] - df['p_lo'].fillna(df['low'])

    # ── Forward-fill pentru stabilitate ─────────────────────────────────────
    cols_fill = [
        'lw_hi', 'lw_lo', 'lm_hi', 'lm_lo', 'm_hi', 'm_lo',
        'p_hi', 'p_lo', 'h4_hi', 'h4_lo', 'h1_hi', 'h1_lo',
        'asia_hi', 'asia_lo', 'lon_hi', 'lon_lo',
        'true_open', 'poc_level', 'vah', 'val',
    ]
    df[cols_fill] = df[cols_fill].ffill()

    print(f"   ✅ [{symbol_name}] Feature engineering complet — {len(df.columns)} coloane")
    return df


# =============================================================================
# MAIN — BUILD DUAL SQL (QQQ + SPY + SMT)
# =============================================================================
def build_to_sql_dual():
    """
    Procesează QQQ și SPY, calculează SMT Divergence și salvează în SQLite.
    """
    print("\n" + "=" * 80)
    print("🏗️  ALADIN DATA PIPELINE v5.0 — START")
    print("=" * 80)

    qqq_df = process_symbol_data(PATH_QQQ, "QQQ")
    spy_df = process_symbol_data(PATH_SPY, "SPY")

    if qqq_df is None or spy_df is None:
        print("❌ Pipeline oprit: unul sau ambele fișiere CSV lipsesc.")
        return

    # ── Pasul 8: SMT Divergence (QQQ vs SPY swing-based) ────────────────────
    print("\n🤝 Pasul 8: Calcul SMT Divergence (QQQ vs SPY)...")

    spy_trimmed = (
        spy_df[['timestamp', 'high', 'low', 'p_hi', 'p_lo']]
        .rename(columns={
            'high': 'spy_hi', 'low': 'spy_lo',
            'p_hi': 'spy_p_hi', 'p_lo': 'spy_p_lo'
        })
    )

    df_final = pd.merge(qqq_df, spy_trimmed, on='timestamp', how='inner')
    print(f"   ✅ Merge QQQ+SPY: {len(df_final):,} rânduri comune")

    # SMT Bearish: QQQ face High nou DAR SPY nu — divergență bearish
    df_final['is_smt_bearish'] = (
        (df_final['high'] > df_final['p_hi']) &
        (df_final['spy_hi'] < df_final['spy_p_hi'])
    ).astype(int)

    # SMT Bullish: QQQ face Low nou DAR SPY nu — divergență bullish
    df_final['is_smt_bullish'] = (
        (df_final['low'] < df_final['p_lo']) &
        (df_final['spy_lo'] > df_final['spy_p_lo'])
    ).astype(int)

    smt_bear_count = df_final['is_smt_bearish'].sum()
    smt_bull_count = df_final['is_smt_bullish'].sum()
    print(f"   📊 SMT Bearish signals: {smt_bear_count:,} | SMT Bullish: {smt_bull_count:,}")

    # ── Pasul 9: Salvare SQL ─────────────────────────────────────────────────
    print("\n🗄️  Pasul 9: Salvare în SQLite (sesiunea 07:00–22:00 RO)...")

    # Filtrăm la sesiunea operativă
    df_save = df_final[df_final['hour_min'] >= '07:00'].copy()

    # Convert timestamp la string pentru SQLite
    df_save['timestamp'] = df_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    spy_df_save = spy_df.copy()
    spy_df_save['timestamp'] = spy_df_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    conn = sqlite3.connect(PATH_DB)
    try:
        # Drop + recreate pentru date fresh
        df_save.to_sql('market_data', conn, if_exists='replace', index=False)
        spy_df_save.to_sql('spy_data', conn, if_exists='replace', index=False)

        # Indexuri pentru query rapid în engine
        conn.execute("DROP INDEX IF EXISTS idx_ts")
        conn.execute("DROP INDEX IF EXISTS idx_date")
        conn.execute("DROP INDEX IF EXISTS idx_hour")
        conn.execute("CREATE INDEX idx_ts   ON market_data (timestamp)")
        conn.execute("CREATE INDEX idx_date ON market_data (date)")
        conn.execute("CREATE INDEX idx_hour ON market_data (hour_min)")
        conn.commit()

        # Stats finale
        total_rows = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        min_ts     = conn.execute("SELECT MIN(timestamp) FROM market_data").fetchone()[0]
        max_ts     = conn.execute("SELECT MAX(timestamp) FROM market_data").fetchone()[0]
        smt_in_db  = conn.execute(
            "SELECT SUM(is_smt_bearish)+SUM(is_smt_bullish) FROM market_data"
        ).fetchone()[0]

        print(f"\n{'='*80}")
        print(f"✅ BAZA DE DATE ACTUALIZATĂ CU SUCCES!")
        print(f"   📁 Path   : {PATH_DB}")
        print(f"   📊 Rânduri: {total_rows:,}")
        print(f"   📅 Range  : {min_ts}  →  {max_ts}")
        print(f"   🤝 SMT signals total: {smt_in_db:,}")
        print(f"{'='*80}\n")

    except Exception as exc:
        print(f"❌ Eroare la salvare SQL: {exc}")
    finally:
        conn.close()


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    build_to_sql_dual()