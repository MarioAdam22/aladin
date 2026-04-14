"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║          ALADIN QUANTUM-ICT — NT8 NQ/ES DATA IMPORTER                                   ║
║          import_nt8_nq.py  |  NinjaTrader8 CSV → SQLite DB                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Utilizare:
  python3 import_nt8_nq.py --nq "NQ 03-26.Last.txt,NQ DEC25.Last.txt,..." \
                            --es "ES 06-26.Last.txt,ES DEC25.Last.txt,..." \
                            --db /Users/mario/Desktop/Aladin/mario_trading.db

Format fișier NT8 (fără header):
  YYYYMMDD HHMMSS;open;high;low;close;volume
  20251210 230200;26047.5;26047.5;26036.75;26036.75;2

Script-ul:
  1. Citește TOATE fișierele NQ și le concatenează cronologic
  2. Citește TOATE fișierele ES și le concatenează cronologic
  3. Calculează features identice cu build_ai_dataset.py (HTF, Sessions, AMT, ICT, SMT)
  4. Salvează în mario_trading.db:
     - tabel nq_data (principal, înlocuiește market_data pentru model)
     - tabel es_data  (corelație, ca SPY pentru QQQ)
  5. Actualizează market_data cu NQ (pentru compatibilitate mario_rag.py)
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import argparse
import glob
from pathlib import Path

# Advanced features (Hurst, GARCH, Kalman, ADX, VWAP, SampEn, Fisher, FFT, ACF)
try:
    import advanced_features as af
    _AF_OK = True
except ImportError:
    _AF_OK = False
    print("⚠️  advanced_features.py nu a fost găsit — features avansate dezactivate")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PATH_DB = "/Users/mario/Desktop/Aladin/mario_trading.db"

# Dacă nu sunt date ES disponibile, folism NQ ca proxy pentru SMT
TIMEZONE = "America/New_York"  # NT8 exportă în ET (Eastern Time)

# ─── CITIRE FIȘIER NT8 ────────────────────────────────────────────────────────
def read_nt8_file(file_path: str) -> pd.DataFrame:
    """
    Citește un fișier NT8 CSV (format YYYYMMDD HHMMSS;O;H;L;C;V).
    Nu are header. Returnează DataFrame cu coloane standard.
    """
    if not os.path.exists(file_path):
        print(f"  ⚠️  Fișier negăsit: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            file_path,
            sep=';',
            header=None,
            names=['ts_raw', 'open', 'high', 'low', 'close', 'volume'],
            dtype={'ts_raw': str, 'open': float, 'high': float,
                   'low': float, 'close': float, 'volume': float}
        )
        df = df.dropna(subset=['ts_raw'])
        df['timestamp'] = pd.to_datetime(df['ts_raw'], format='%Y%m%d %H%M%S', errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.drop(columns=['ts_raw'])
        df['volume'] = df['volume'].fillna(0).astype(int)
        rows = len(df)
        print(f"  ✅ {file_path} → {rows:,} bare | "
              f"{df['timestamp'].min()} → {df['timestamp'].max()}")
        return df
    except Exception as e:
        print(f"  ❌ Eroare la citire {file_path}: {e}")
        return pd.DataFrame()


def load_all_files(file_list: list) -> pd.DataFrame:
    """Concatenează mai multe fișiere NT8 și elimină duplicatele."""
    frames = []
    for f in file_list:
        f = f.strip()
        if not f:
            continue
        # Suport glob patterns
        matched = glob.glob(f)
        if matched:
            for m in matched:
                df = read_nt8_file(m)
                if not df.empty:
                    frames.append(df)
        else:
            df = read_nt8_file(f)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    print(f"  📊 Total combinat: {len(combined):,} bare | "
          f"{combined['timestamp'].min()} → {combined['timestamp'].max()}")
    return combined


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def add_tf(df_in: pd.DataFrame, resample_period: str, prefix: str) -> pd.DataFrame:
    """Adaugă High/Low la un timeframe superior (H4, H1) — no lookahead."""
    df_idx = df_in.set_index('timestamp')
    temp = df_idx.resample(resample_period).agg({'high': 'max', 'low': 'min'})
    temp = temp.shift(1).rename(columns={'high': f'{prefix}_hi', 'low': f'{prefix}_lo'})
    temp = temp.reset_index()
    merged = pd.merge_asof(
        df_in.sort_values('timestamp'),
        temp.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    return merged


def get_session(df: pd.DataFrame, start: str, end: str, prefix: str) -> pd.DataFrame:
    """High/Low per date pentru o fereastră de sesiune."""
    mask = (df['hour_min'] >= start) & (df['hour_min'] < end)
    sess = (
        df[mask]
        .groupby('date')
        .agg(**{f'{prefix}_hi': ('high', 'max'), f'{prefix}_lo': ('low', 'min')})
        .reset_index()
    )
    return sess


def get_amt_metrics(group: pd.DataFrame) -> pd.Series:
    """POC (Point of Control) + Value Area 70% pe zi."""
    if group.empty or group['volume'].sum() == 0:
        return pd.Series([np.nan, np.nan, np.nan], index=['poc_level', 'vah', 'val'])

    vol_dist = (
        group.groupby('price_bin')['volume']
        .sum()
        .sort_values(ascending=False)
    )
    poc = vol_dist.idxmax()
    total_vol = vol_dist.sum()
    target_vol = total_vol * 0.70
    current_vol = vol_dist.iloc[0]
    v_bins = [poc]

    for price, vol in vol_dist.iloc[1:].items():
        if current_vol >= target_vol:
            break
        current_vol += vol
        v_bins.append(price)

    return pd.Series(
        [poc, max(v_bins), min(v_bins)],
        index=['poc_level', 'vah', 'val']
    )


def compute_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Calculează toate features ICT + AMT pe un DataFrame de bare 1-min.
    Timestamps sunt naive (ET local).
    """
    print(f"\n  🔧 [{symbol}] Feature engineering...")

    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # ── Calendar ──────────────────────────────────────────────────────────────
    df['date']        = df['timestamp'].dt.date
    df['hour_min']    = df['timestamp'].dt.strftime('%H:%M')
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['week_id']     = df['timestamp'].dt.isocalendar().week.astype(int)
    df['month']       = df['timestamp'].dt.month
    df['year']        = df['timestamp'].dt.year

    # ── HTF: Weekly High/Low anterior ─────────────────────────────────────────
    lw = (
        df.groupby(['year', 'week_id'])
        .agg(lw_hi=('high', 'max'), lw_lo=('low', 'min'))
        .reset_index()
        .sort_values(['year', 'week_id'])
    )
    lw['lw_hi'] = lw['lw_hi'].shift(1)
    lw['lw_lo'] = lw['lw_lo'].shift(1)
    df = df.merge(lw, on=['year', 'week_id'], how='left')

    # ── HTF: Monthly High/Low anterior ────────────────────────────────────────
    lm = (
        df.groupby(['year', 'month'])
        .agg(lm_hi=('high', 'max'), lm_lo=('low', 'min'))
        .reset_index()
        .sort_values(['year', 'month'])
    )
    lm['lm_hi'] = lm['lm_hi'].shift(1)
    lm['lm_lo'] = lm['lm_lo'].shift(1)
    df = df.merge(lm, on=['year', 'month'], how='left')

    # ── Monday High/Low (range săptămână) ─────────────────────────────────────
    monday_data = (
        df[df['day_of_week'] == 0]
        .groupby(['year', 'week_id'])
        .agg(m_hi=('high', 'max'), m_lo=('low', 'min'))
        .reset_index()
    )
    df = df.merge(monday_data, on=['year', 'week_id'], how='left')

    # ── Previous Day High/Low ─────────────────────────────────────────────────
    # NQ futures: ziua "de tranzacționare" este 18:00 ET → 17:59 ET+1
    # Folosim date calendaristică simplă (suficientă pentru features)
    pd_hl = (
        df.groupby('date')
        .agg(p_hi=('high', 'max'), p_lo=('low', 'min'))
        .reset_index()
        .sort_values('date')
    )
    pd_hl['p_hi'] = pd_hl['p_hi'].shift(1)
    pd_hl['p_lo'] = pd_hl['p_lo'].shift(1)
    df = df.merge(pd_hl, on='date', how='left')

    # ── Multi-Timeframe (H4, H1) ──────────────────────────────────────────────
    print(f"    ⏳ Multi-Timeframe H4, H1...")
    df = add_tf(df, '4h', 'h4')
    df = add_tf(df, '1h', 'h1')

    # ── Sesiuni (ora ET) ──────────────────────────────────────────────────────
    # Asia: 18:00-00:00 ET (seara anterioară în futures)
    # London: 02:00-08:00 ET
    # NYOpen: 09:30 ET
    # Folosim "true_open" = prima bară de la 09:30 ET
    print(f"    ⚓ Sessions + True Open...")
    day_open = (
        df[df['hour_min'] >= '09:30']
        .groupby('date')['open']
        .first()
        .rename('true_open')
        .reset_index()
    )
    df = df.merge(day_open, on='date', how='left')

    # Asia session = 18:00 ET (seara) → folosim bare cu ora 18:00-23:59
    # Le grupăm pe data zilei URMĂTOARE (pentru că sesiunea Asia a nopții 18:00-23:59
    # aparține zilei de tranzacționare a zilei următoare)
    asia = get_session(df, '18:00', '24:00', 'asia_eve')  # seara
    asia2 = get_session(df, '00:00', '09:00', 'asia_morn')  # dimineața (00-09)

    # Simplificăm: Asia = 20:00-02:00, London = 02:00-08:00 (ore ET)
    asia_full = (
        df[(df['hour_min'] >= '20:00') | (df['hour_min'] < '02:00')]
        .groupby('date')
        .agg(asia_hi=('high', 'max'), asia_lo=('low', 'min'))
        .reset_index()
    )
    lon_full = (
        df[(df['hour_min'] >= '02:00') & (df['hour_min'] < '08:00')]
        .groupby('date')
        .agg(lon_hi=('high', 'max'), lon_lo=('low', 'min'))
        .reset_index()
    )
    df = df.merge(asia_full, on='date', how='left')
    df = df.merge(lon_full, on='date', how='left')

    # ── AMT: POC + Value Area 70% ─────────────────────────────────────────────
    print(f"    📊 AMT (POC, VAH, VAL)...")
    tick_size = 0.25  # NQ tick size (ES: 0.25)
    df['price_bin'] = (df['close'] / tick_size).round() * tick_size

    amt_data = (
        df.groupby('date')
        .apply(get_amt_metrics, include_groups=False)
        .reset_index()
    )
    df = df.merge(amt_data, on='date', how='left')

    # ── ICT Core ──────────────────────────────────────────────────────────────
    print(f"    🕯️  ICT (FVG, Displacement, Power of 3)...")

    # Fair Value Gap Bullish: low[i] > high[i-2]
    df['fvg_up'] = (
        (df['low'] > df['high'].shift(2)) &
        (df['low'].shift(1) > df['high'].shift(2))
    ).astype(int)

    # Fair Value Gap Bearish
    df['fvg_down'] = (
        (df['high'] < df['low'].shift(2)) &
        (df['high'].shift(1) < df['low'].shift(2))
    ).astype(int)

    # Displacement
    df['body_size']        = abs(df['close'] - df['open'])
    df['has_displacement'] = (
        df['body_size'] > df['body_size'].rolling(20).mean() * 1.5
    ).astype(int)

    # Power of 3 — preț față de True Open
    df['is_above_open'] = (df['close'] > df['true_open'].fillna(df['open'])).astype(int)

    # ATR 14
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()

    # ── Distance Features ─────────────────────────────────────────────────────
    print(f"    📐 Distance Features...")
    df['dist_poc']  = df['close'] - df['poc_level'].fillna(df['close'])
    df['inside_va'] = (
        (df['close'] >= df['val'].fillna(df['low'])) &
        (df['close'] <= df['vah'].fillna(df['high']))
    ).astype(int)
    df['dist_pdh']  = df['close'] - df['p_hi'].fillna(df['high'])
    df['dist_pdl']  = df['close'] - df['p_lo'].fillna(df['low'])

    # ── Forward-fill ──────────────────────────────────────────────────────────
    cols_fill = [
        'lw_hi', 'lw_lo', 'lm_hi', 'lm_lo', 'm_hi', 'm_lo',
        'p_hi', 'p_lo', 'h4_hi', 'h4_lo', 'h1_hi', 'h1_lo',
        'asia_hi', 'asia_lo', 'lon_hi', 'lon_lo',
        'true_open', 'poc_level', 'vah', 'val',
    ]
    df[cols_fill] = df[cols_fill].ffill()

    # ── Advanced Features (Hurst, GARCH, Kalman, ADX, VWAP, SampEn, Fisher, FFT, ACF) ──
    if _AF_OK:
        print(f"    🧮 [{symbol}] Advanced features (12 columns)...")
        df = af.compute_all_advanced(df)
    else:
        # Fill with zeros if module not available
        for col in ['hurst', 'garch_vol', 'kalman_smooth', 'kalman_noise', 'adx_14',
                     'vwap', 'dist_vwap', 'sample_entropy', 'fisher_transform',
                     'fft_cycle', 'acf_lag1', 'acf_lag5']:
            df[col] = 0.0

    # Timestamp → string pentru SQLite
    df['timestamp'] = df['timestamp'].astype(str)
    df['date']      = df['date'].astype(str)

    print(f"    ✅ [{symbol}] {len(df):,} bare, {len(df.columns)} coloane")
    return df


def add_smt_divergence(nq_df: pd.DataFrame, es_df: pd.DataFrame) -> pd.DataFrame:
    """
    SMT Divergence: NQ face High nou, ES nu → Bearish SMT.
                    NQ face Low nou, ES nu  → Bullish SMT.
    (analog cu QQQ vs SPY din pipeline-ul original)
    """
    if es_df.empty:
        print("  ⚠️  ES date lipsă — SMT va fi 0 pentru toate barele")
        nq_df['spy_hi']       = nq_df['high']
        nq_df['spy_lo']       = nq_df['low']
        nq_df['spy_p_hi']     = nq_df['p_hi']
        nq_df['spy_p_lo']     = nq_df['p_lo']
        nq_df['is_smt_bearish'] = 0
        nq_df['is_smt_bullish'] = 0
        return nq_df

    # Atenție: es_df are și ea coloana 'timestamp' (string) → trebuie eliminată
    # înainte de merge pentru a evita conflictul timestamp_x / timestamp_y
    es_trimmed = (
        es_df[['high', 'low', 'p_hi', 'p_lo']]
        .rename(columns={
            'high': 'spy_hi', 'low': 'spy_lo',
            'p_hi': 'spy_p_hi', 'p_lo': 'spy_p_lo'
        })
    )
    es_trimmed['timestamp_dt'] = pd.to_datetime(es_df['timestamp'])

    nq_df['timestamp_dt'] = pd.to_datetime(nq_df['timestamp'])

    merged = pd.merge_asof(
        nq_df.sort_values('timestamp_dt'),
        es_trimmed.sort_values('timestamp_dt'),
        on='timestamp_dt',
        direction='nearest',
        tolerance=pd.Timedelta('5min')
    )
    merged = merged.drop(columns=['timestamp_dt'])

    # Swing logic: NQ atinge un nou High dar ES nu → Bearish SMT
    look = 5
    merged['nq_new_high'] = (merged['high'] == merged['high'].rolling(look).max())
    merged['es_new_high'] = (merged['spy_hi'] == merged['spy_hi'].rolling(look).max())
    merged['nq_new_low']  = (merged['low'] == merged['low'].rolling(look).min())
    merged['es_new_low']  = (merged['spy_lo'] == merged['spy_lo'].rolling(look).min())

    merged['is_smt_bearish'] = (merged['nq_new_high'] & ~merged['es_new_high']).astype(int)
    merged['is_smt_bullish'] = (merged['nq_new_low']  & ~merged['es_new_low']).astype(int)

    merged = merged.drop(columns=['nq_new_high', 'es_new_high', 'nq_new_low', 'es_new_low'])
    return merged


def get_market_data_columns() -> list:
    """Returnează lista de coloane din tabelul market_data (ordinea exactă)."""
    return [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'date', 'hour_min', 'day_of_week', 'week_id', 'month', 'year',
        'lw_hi', 'lw_lo', 'lm_hi', 'lm_lo', 'm_hi', 'm_lo',
        'p_hi', 'p_lo', 'h4_hi', 'h4_lo', 'h1_hi', 'h1_lo',
        'true_open', 'asia_hi', 'asia_lo', 'lon_hi', 'lon_lo',
        'price_bin', 'poc_level', 'vah', 'val',
        'fvg_up', 'fvg_down', 'body_size', 'has_displacement', 'is_above_open',
        'atr_14', 'dist_poc', 'inside_va', 'dist_pdh', 'dist_pdl',
        'spy_hi', 'spy_lo', 'spy_p_hi', 'spy_p_lo',
        'is_smt_bearish', 'is_smt_bullish',
        # Advanced features (Tier 1 + Tier 2)
        'hurst', 'garch_vol', 'kalman_smooth', 'kalman_noise', 'adx_14',
        'vwap', 'dist_vwap', 'sample_entropy', 'fisher_transform',
        'fft_cycle', 'acf_lag1', 'acf_lag5',
    ]


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='NT8 NQ/ES → mario_trading.db')
    parser.add_argument('--nq',   required=True,  help='Fișiere NQ separate prin virgulă (sau glob)')
    parser.add_argument('--es',   default='',     help='Fișiere ES separate prin virgulă (sau glob)')
    parser.add_argument('--db',   default=PATH_DB, help='Path la mario_trading.db')
    parser.add_argument('--mode', default='replace',
                        choices=['replace', 'append'],
                        help='replace=șterge market_data și pune NQ; append=adaugă la ce există')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🏗️  ALADIN NT8 IMPORTER — NQ/ES → mario_trading.db")
    print("=" * 80)

    # ── 1. Citire fișiere ─────────────────────────────────────────────────────
    nq_files = [f.strip() for f in args.nq.split(',')]
    es_files = [f.strip() for f in args.es.split(',')] if args.es else []

    print("\n📁 Citire fișiere NQ...")
    nq_raw = load_all_files(nq_files)
    if nq_raw.empty:
        print("❌ Niciun fișier NQ valid găsit. Oprire.")
        sys.exit(1)

    print("\n📁 Citire fișiere ES...")
    es_raw = load_all_files(es_files) if es_files else pd.DataFrame()

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n🔧 Procesare NQ...")
    nq_df = compute_features(nq_raw, "NQ")

    es_df = pd.DataFrame()
    if not es_raw.empty:
        print("\n🔧 Procesare ES...")
        es_df = compute_features(es_raw, "ES")

    # ── 3. SMT Divergence ────────────────────────────────────────────────────
    print("\n🤝 SMT Divergence (NQ vs ES)...")
    nq_final = add_smt_divergence(nq_df, es_df)

    # ── 4. Selectare coloane finale ───────────────────────────────────────────
    all_cols = get_market_data_columns()
    # Asigurăm că toate coloanele există
    for col in all_cols:
        if col not in nq_final.columns:
            nq_final[col] = np.nan
    nq_final = nq_final[all_cols]

    # ── 5. Salvare în SQLite ──────────────────────────────────────────────────
    print(f"\n💾 Salvare în {args.db}...")
    conn = sqlite3.connect(args.db)

    # Salvăm tabelul nq_data (întotdeauna înlocuim tot)
    print("  📝 Scriere tabel nq_data...")
    nq_final.to_sql('nq_data', conn, if_exists='replace', index=False)
    conn.execute('CREATE INDEX IF NOT EXISTS idx_nq_ts ON nq_data(timestamp)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_nq_date ON nq_data(date, hour_min)')

    # ── Adaugă coloana source dacă nu există ─────────────────────────────────
    existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(market_data)").fetchall()]
    if "source" not in existing_cols:
        try:
            conn.execute("ALTER TABLE market_data ADD COLUMN source TEXT DEFAULT 'LEGACY'")
            print("  ✅ Coloana 'source' adăugată în market_data")
        except Exception:
            pass

    # Actualizare market_data pentru compatibilitate cu mario_rag.py
    nq_final['source'] = 'NT8_MANUAL'
    if args.mode == 'replace':
        print("  📝 Înlocuire market_data cu NQ (source=NT8_MANUAL)...")
        nq_final.to_sql('market_data', conn, if_exists='replace', index=False)
        conn.execute('CREATE INDEX IF NOT EXISTS idx_md_ts ON market_data(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_md_date ON market_data(date, hour_min)')
    else:
        # Append: înlocuim și BRIDGE_LIVE cu NT8_MANUAL pe același timestamp (calitate mai bună)
        existing_ts = pd.read_sql("SELECT DISTINCT timestamp FROM market_data WHERE source = 'NT8_MANUAL'", conn)
        new_rows = nq_final[~nq_final['timestamp'].isin(existing_ts['timestamp'])]
        # Suprascrie BRIDGE_LIVE cu NT8_MANUAL dacă timestamp-ul există
        bridge_ts = nq_final[nq_final['timestamp'].isin(
            pd.read_sql("SELECT timestamp FROM market_data WHERE source = 'BRIDGE_LIVE'", conn)['timestamp']
        )]
        if not bridge_ts.empty:
            conn.execute(
                f"DELETE FROM market_data WHERE source = 'BRIDGE_LIVE' AND timestamp IN ({','.join(['?']*len(bridge_ts))})",
                bridge_ts['timestamp'].tolist()
            )
            print(f"  🔄 Înlocuit {len(bridge_ts):,} bare BRIDGE_LIVE cu NT8_MANUAL (calitate mai bună)")
            new_rows = nq_final[~nq_final['timestamp'].isin(existing_ts['timestamp'])]
        print(f"  📝 Append market_data: {len(new_rows):,} bare noi (source=NT8_MANUAL)...")
        new_rows.to_sql('market_data', conn, if_exists='append', index=False)

    # Salvăm ES separat dacă avem date
    if not es_df.empty:
        es_cols = [c for c in all_cols if c not in ['spy_hi', 'spy_lo', 'spy_p_hi', 'spy_p_lo',
                                                       'is_smt_bearish', 'is_smt_bullish']]
        for col in es_cols:
            if col not in es_df.columns:
                es_df[col] = np.nan
        print("  📝 Scriere tabel es_data...")
        es_df[es_cols].to_sql('es_data', conn, if_exists='replace', index=False)

    conn.commit()
    conn.close()

    # ── 6. Statistici finale ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("✅ IMPORT COMPLET!")
    print(f"   NQ bare importate : {len(nq_final):,}")
    if not es_df.empty:
        print(f"   ES bare importate : {len(es_df):,}")
    ts_min = nq_final['timestamp'].min()
    ts_max = nq_final['timestamp'].max()
    print(f"   Interval          : {ts_min} → {ts_max}")
    print(f"   DB                : {args.db}")
    print("=" * 80)
    print("\n💡 Pasul următor: rulează retrain_model.py pentru a reantrera modelul pe NQ")


if __name__ == '__main__':
    main()
