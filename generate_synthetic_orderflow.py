"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN — generate_synthetic_orderflow.py                                   ║
║  Generează ~50 features sintetice de order flow din minute OHLCV            ║
║  Input:  data/NQ_continuous.parquet                                         ║
║  Output: data/orderflow_features.parquet                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

TOATE features sunt RELATIVE (normalized) — nu absolute — pentru a evita
data drift între 2016 și 2026.

Features generate:
  Tier 1 — Delta & CVD
    delta_bar, delta_ratio, cvd_session, cvd_zscore_20d, cvd_pct_20d,
    cvd_momentum, cvd_acceleration, cvd_divergence_flag

  Tier 2 — Footprint sintetic (distribuție volum în bară)
    buy_vol_est, sell_vol_est, buy_ratio, sell_ratio,
    footprint_imbalance, close_position (close relativ la high-low)

  Tier 3 — Absorption & Large trades
    absorption_score, absorption_zscore_20d, absorption_flag,
    large_trade_flag, large_trade_ratio, volume_thrust

  Tier 4 — Volume Profile sintetic (per sesiune)
    poc_dist, vah_dist, val_dist, vwap_dist, vwap_zscore,
    volume_at_poc_ratio, value_area_width

  Tier 5 — Market microstructure proxies
    amihud_illiquidity, amihud_zscore_20d,
    spread_proxy, spread_zscore_20d,
    price_impact, kyle_lambda_20d

  Tier 6 — Sesiune & Opening
    opening_drive_dir, opening_drive_strength,
    session_delta_cumul, session_delta_pct,
    stacked_imbalance_count, stacked_imbalance_dir

  Tier 7 — Order flow momentum
    delta_ema_fast, delta_ema_slow, delta_macd,
    cvd_roc_5, cvd_roc_10, cvd_trend_flag

Toate features sunt calculate per sesiune (LON = 03:00-11:00 ET,
NY = 09:30-16:00 ET) și normalize față de rolling windows pentru stationaritate.

Utilizare:
  python3 generate_synthetic_orderflow.py
  python3 generate_synthetic_orderflow.py --input data/NQ_continuous.parquet --out data/orderflow_features.parquet
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ALADIN_DIR = Path(__file__).parent
INPUT_FILE  = ALADIN_DIR / "data" / "NQ_continuous.parquet"
OUTPUT_FILE = ALADIN_DIR / "data" / "orderflow_features.parquet"

TIMEZONE    = "America/New_York"

# Sesiuni (ET)
SESSIONS = {
    "LON": ("03:00", "11:00"),
    "NY":  ("09:30", "16:00"),
}

# EWM half-life pentru normalizare decay-aware (Renaissance-style)
# Half-life 12 luni = 252 sesiuni trading
EWM_HL_SESSION = 252        # sesiuni
EWM_HL_BAR     = 252 * 390  # bare (252 sesiuni × 390 min/sesiune)
EWM_MIN_P      = 5          # min_periods sesiuni
EWM_MIN_P_BAR  = 100        # min_periods bare


# ─── TICK RULE + FOOTPRINT SINTETIC ──────────────────────────────────────────
def compute_bar_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 1 + Tier 2: Delta per bară și footprint sintetic.
    Tick Rule: buy_vol = vol * (close - low) / (high - low) — mai precis decât simplu close>open
    """
    df = df.copy()

    hl_range = df['high'] - df['low']
    hl_range = hl_range.replace(0, np.nan)

    # Footprint sintetic — distribuție volum bazată pe poziția close în range
    df['close_position'] = ((df['close'] - df['low']) / hl_range).clip(0, 1).fillna(0.5)
    df['buy_vol_est']    = df['volume'] * df['close_position']
    df['sell_vol_est']   = df['volume'] * (1 - df['close_position'])

    # Delta
    df['delta_bar']   = df['buy_vol_est'] - df['sell_vol_est']
    df['delta_ratio'] = (df['delta_bar'] / df['volume'].replace(0, np.nan)).fillna(0).clip(-1, 1)

    # Imbalance pe bară
    df['footprint_imbalance'] = df['delta_bar'] / df['volume'].replace(0, np.nan).fillna(0)

    # Buy/sell ratio
    total = df['volume'].replace(0, np.nan)
    df['buy_ratio']  = (df['buy_vol_est'] / total).fillna(0.5)
    df['sell_ratio'] = (df['sell_vol_est'] / total).fillna(0.5)

    return df


# ─── CVD SESIUNE ─────────────────────────────────────────────────────────────
def compute_session_cvd(df: pd.DataFrame, session_col: str = 'session_id') -> pd.DataFrame:
    """
    Tier 1: CVD cumulativ per sesiune + derivate.
    """
    df = df.copy()
    df['cvd_session'] = df.groupby(session_col)['delta_bar'].cumsum()

    # CVD final per sesiune (ultima valoare din sesiune)
    session_cvd_final = df.groupby(session_col)['cvd_session'].last().rename('cvd_final')
    df = df.merge(session_cvd_final, on=session_col, how='left')

    # CVD percentual față de volumul total al sesiunii
    session_vol = df.groupby(session_col)['volume'].sum().rename('session_total_vol')
    df = df.merge(session_vol, on=session_col, how='left')
    df['cvd_pct_session'] = df['cvd_final'] / df['session_total_vol'].replace(0, np.nan)

    return df


def compute_rolling_cvd_stats(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizează CVD final per sesiune față de rolling window.
    Input: DataFrame cu o linie per sesiune, cu cvd_final.
    """
    s = session_df.copy()

    # Z-score EWM decay-aware (halflife=252 sesiuni = 12 luni)
    roll_mean = s['cvd_final'].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).mean()
    roll_std  = s['cvd_final'].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).std().replace(0, np.nan)
    s['cvd_zscore_20d'] = ((s['cvd_final'] - roll_mean) / roll_std).fillna(0).clip(-4, 4)

    # Percentilă rolling 252 sesiuni (fereastră consistentă cu EWM)
    s['cvd_pct_20d'] = s['cvd_final'].rolling(EWM_HL_SESSION, min_periods=EWM_MIN_P).rank(pct=True).fillna(0.5)

    # Momentum și accelerație CVD (diferențe între sesiuni consecutive)
    s['cvd_momentum']     = s['cvd_final'].diff(1).fillna(0)
    s['cvd_acceleration'] = s['cvd_momentum'].diff(1).fillna(0)

    # Normalizăm momentum față de std
    mom_std = s['cvd_momentum'].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).std().replace(0, np.nan)
    s['cvd_momentum_z'] = (s['cvd_momentum'] / mom_std).fillna(0).clip(-4, 4)

    # CVD trend flag: 3 sesiuni consecutive în aceeași direcție
    s['cvd_trend_flag'] = (
        (s['cvd_final'] > s['cvd_final'].shift(1)) &
        (s['cvd_final'].shift(1) > s['cvd_final'].shift(2))
    ).astype(int) - (
        (s['cvd_final'] < s['cvd_final'].shift(1)) &
        (s['cvd_final'].shift(1) < s['cvd_final'].shift(2))
    ).astype(int)

    # EMA delta
    s['delta_ema_fast'] = s['cvd_final'].ewm(span=3, adjust=False).mean()
    s['delta_ema_slow'] = s['cvd_final'].ewm(span=8, adjust=False).mean()
    s['delta_macd']     = s['delta_ema_fast'] - s['delta_ema_slow']

    # ROC
    s['cvd_roc_5']  = s['cvd_final'].pct_change(5).fillna(0).clip(-5, 5)
    s['cvd_roc_10'] = s['cvd_final'].pct_change(10).fillna(0).clip(-5, 5)

    return s


# ─── ABSORPTION ──────────────────────────────────────────────────────────────
def compute_absorption(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 3: Absorption = volum mare dar range mic.
    absorption_score = volume / (high - low) — normalizat rolling.
    """
    df = df.copy()

    hl_range = (df['high'] - df['low']).replace(0, np.nan)
    df['absorption_raw'] = df['volume'] / hl_range

    # Z-score EWM bar-level (halflife=252 sesiuni × 390 bare)
    roll_mean = df['absorption_raw'].ewm(halflife=EWM_HL_BAR, min_periods=EWM_MIN_P_BAR).mean()
    roll_std  = df['absorption_raw'].ewm(halflife=EWM_HL_BAR, min_periods=EWM_MIN_P_BAR).std().replace(0, np.nan)
    df['absorption_zscore'] = ((df['absorption_raw'] - roll_mean) / roll_std).fillna(0).clip(-4, 4)

    # Flag absorption (zscore > 1.5 = bară cu volum mare dar range mic)
    df['absorption_flag'] = (df['absorption_zscore'] > 1.5).astype(int)

    return df


def compute_session_absorption(session_df: pd.DataFrame, bar_df: pd.DataFrame) -> pd.DataFrame:
    """Agregă absorption per sesiune."""
    # Absorption medie pe sesiune
    abs_session = bar_df.groupby('session_id').agg(
        absorption_score_mean=('absorption_zscore', 'mean'),
        absorption_score_max=('absorption_zscore', 'max'),
        absorption_flag_count=('absorption_flag', 'sum'),
    ).reset_index()

    s = session_df.merge(abs_session, on='session_id', how='left')

    # Normalizăm absorption_flag_count
    roll_mean = s['absorption_flag_count'].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).mean()
    roll_std  = s['absorption_flag_count'].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).std().replace(0, np.nan)
    s['absorption_zscore_20d'] = ((s['absorption_flag_count'] - roll_mean) / roll_std).fillna(0).clip(-4, 4)

    return s


# ─── LARGE TRADES ─────────────────────────────────────────────────────────────
def compute_large_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 3: Large trade proxy — bare cu volume > 2σ față de medie rolling.
    """
    df = df.copy()

    roll_mean = df['volume'].ewm(halflife=EWM_HL_BAR, min_periods=EWM_MIN_P_BAR).mean()
    roll_std  = df['volume'].ewm(halflife=EWM_HL_BAR, min_periods=EWM_MIN_P_BAR).std().replace(0, np.nan)
    df['volume_zscore'] = ((df['volume'] - roll_mean) / roll_std).fillna(0)

    df['large_trade_flag']  = (df['volume_zscore'] > 2.0).astype(int)
    df['large_trade_ratio'] = df['volume_zscore'].clip(0, 5) / 5.0  # normalize 0-1

    # Volume thrust: volum mare + delta în aceeași direcție
    df['volume_thrust'] = df['volume_zscore'] * df['delta_ratio'].abs()

    return df


# ─── VOLUME PROFILE SINTETIC ─────────────────────────────────────────────────
def compute_volume_profile(bar_df: pd.DataFrame, session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 4: VWAP, POC proxy, VAH/VAL per sesiune.
    """
    df = bar_df.copy()

    # VWAP exact per sesiune
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol']        = df['typical_price'] * df['volume']

    vwap_grp = df.groupby('session_id').agg(
        vwap=('tp_vol', 'sum'),
        total_vol=('volume', 'sum'),
        session_high=('high', 'max'),
        session_low=('low', 'min'),
        session_open=('open', 'first'),
        session_close=('close', 'last'),
    ).reset_index()
    vwap_grp['vwap'] = vwap_grp['vwap'] / vwap_grp['total_vol'].replace(0, np.nan)

    # POC proxy — preț unde s-a tranzacționat cel mai mult volum
    # Distribuim volumul fiecărei bare uniform pe 10 price buckets între high și low
    def compute_poc(group):
        price_vol = {}
        for _, row in group.iterrows():
            if row['high'] == row['low']:
                bucket = round(row['close'] * 4) / 4  # round to 0.25
                price_vol[bucket] = price_vol.get(bucket, 0) + row['volume']
            else:
                buckets = np.linspace(row['low'], row['high'], 10)
                vol_per_bucket = row['volume'] / 10
                for b in buckets:
                    bucket = round(b * 4) / 4
                    price_vol[bucket] = price_vol.get(bucket, 0) + vol_per_bucket
        if price_vol:
            return max(price_vol, key=price_vol.get)
        return np.nan

    # POC per sesiune (poate fi lent pe date mari — folosim aproximare rapidă)
    # Aproximare rapidă: prețul care apare cel mai mult în OHLCV weighted by volume
    poc_approx = df.groupby('session_id').apply(
        lambda g: (g['close'] * g['volume']).sum() / g['volume'].sum()
    ).rename('poc_approx').reset_index()

    vwap_grp = vwap_grp.merge(poc_approx, on='session_id', how='left')

    # Merge pe session_df
    s = session_df.merge(vwap_grp, on='session_id', how='left')

    # Value Area proxy (aproximare cu std ± 1σ în jurul VWAP = ~68% din volum)
    # VAH = VWAP + std(price) în sesiune, VAL = VWAP - std(price)
    price_std = df.groupby('session_id')['close'].std().rename('price_std_session').reset_index()
    s = s.merge(price_std, on='session_id', how='left')

    s['vah_approx'] = s['vwap'] + s['price_std_session']
    s['val_approx'] = s['vwap'] - s['price_std_session']

    # Distanțe față de close sesiune
    close_col = s['session_close'] if 'session_close' in s.columns else s.get('close_session', s['vwap'])
    s['vwap_dist']    = (close_col - s['vwap']) / s['vwap'].replace(0, np.nan)
    s['poc_dist']     = (close_col - s['poc_approx']) / s['poc_approx'].replace(0, np.nan)
    s['vah_dist']     = (s['vah_approx'] - close_col) / s['vwap'].replace(0, np.nan)
    s['val_dist']     = (close_col - s['val_approx']) / s['vwap'].replace(0, np.nan)
    s['value_area_width'] = (s['vah_approx'] - s['val_approx']) / s['vwap'].replace(0, np.nan)

    # VWAP z-score EWM decay-aware
    roll_mean = s['vwap_dist'].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).mean()
    roll_std  = s['vwap_dist'].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).std().replace(0, np.nan)
    s['vwap_zscore'] = ((s['vwap_dist'] - roll_mean) / roll_std).fillna(0).clip(-4, 4)

    return s


# ─── MICROSTRUCTURE PROXIES ───────────────────────────────────────────────────
def compute_microstructure(session_df: pd.DataFrame, bar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 5: Amihud illiquidity, spread proxy, Kyle's lambda.
    """
    df = bar_df.copy()

    # Return per bară (absolut)
    df['ret_abs'] = df['close'].pct_change().abs().fillna(0)

    # Amihud illiquidity = |return| / volume
    df['amihud_bar'] = df['ret_abs'] / df['volume'].replace(0, np.nan)

    # Spread proxy = (high - low) / close — bid-ask spread estimat
    df['spread_proxy_bar'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan)

    # Aggregate per sesiune
    micro = df.groupby('session_id').agg(
        amihud_mean=('amihud_bar', 'mean'),
        amihud_max=('amihud_bar', 'max'),
        spread_proxy_mean=('spread_proxy_bar', 'mean'),
        spread_proxy_max=('spread_proxy_bar', 'max'),
    ).reset_index()

    s = session_df.merge(micro, on='session_id', how='left')

    # Z-score EWM decay-aware
    for col in ['amihud_mean', 'spread_proxy_mean']:
        roll_mean = s[col].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).mean()
        roll_std  = s[col].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).std().replace(0, np.nan)
        s[f'{col}_zscore'] = ((s[col] - roll_mean) / roll_std).fillna(0).clip(-4, 4)

    # Kyle's lambda proxy: Δprice / Δvolume (price impact)
    s['kyle_lambda'] = s.get('session_close', s['vwap'] if 'vwap' in s.columns else pd.Series(dtype=float))
    if 'session_close' in s.columns and 'session_open' in s.columns:
        price_move = (s['session_close'] - s['session_open']).abs()
        s['kyle_lambda'] = price_move / s['total_vol'].replace(0, np.nan)
        roll_mean = s['kyle_lambda'].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).mean()
        roll_std  = s['kyle_lambda'].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).std().replace(0, np.nan)
        s['kyle_lambda_zscore'] = ((s['kyle_lambda'] - roll_mean) / roll_std).fillna(0).clip(-4, 4)

    return s


# ─── OPENING DRIVE ────────────────────────────────────────────────────────────
def compute_opening_drive(bar_df: pd.DataFrame, session_df: pd.DataFrame,
                           open_minutes: int = 30) -> pd.DataFrame:
    """
    Tier 6: Opening drive — direcție și forță în primele N minute ale sesiunii.
    """
    df = bar_df.copy()

    # Bare din primele open_minutes minute ale sesiunii
    df['bar_num'] = df.groupby('session_id').cumcount()
    opening_bars = df[df['bar_num'] < open_minutes]

    opening = opening_bars.groupby('session_id').agg(
        open_start=('open', 'first'),
        open_end=('close', 'last'),
        open_high=('high', 'max'),
        open_low=('low', 'min'),
        open_volume=('volume', 'sum'),
        open_delta=('delta_bar', 'sum'),
    ).reset_index()

    opening['opening_drive_raw'] = opening['open_end'] - opening['open_start']
    opening['opening_range']     = opening['open_high'] - opening['open_low']
    opening['opening_drive_dir'] = np.sign(opening['opening_drive_raw'])

    # Strength: mișcare ca % din range
    opening['opening_drive_strength'] = (
        opening['opening_drive_raw'].abs() /
        opening['opening_range'].replace(0, np.nan)
    ).clip(0, 1).fillna(0.5)

    # Delta în opening
    opening['opening_delta_ratio'] = (
        opening['open_delta'] / opening['open_volume'].replace(0, np.nan)
    ).fillna(0).clip(-1, 1)

    # Normalizăm opening_drive_raw rolling
    roll_mean = opening['opening_drive_raw'].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).mean()
    roll_std  = opening['opening_drive_raw'].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).std().replace(0, np.nan)
    opening['opening_drive_zscore'] = (
        (opening['opening_drive_raw'] - roll_mean) / roll_std
    ).fillna(0).clip(-4, 4)

    s = session_df.merge(
        opening[['session_id', 'opening_drive_dir', 'opening_drive_strength',
                  'opening_drive_zscore', 'opening_delta_ratio', 'opening_range']],
        on='session_id', how='left'
    )
    return s


# ─── STACKED IMBALANCES ───────────────────────────────────────────────────────
def compute_stacked_imbalances(bar_df: pd.DataFrame, session_df: pd.DataFrame,
                                threshold: float = 0.3) -> pd.DataFrame:
    """
    Tier 6: Bare consecutive cu delta în aceeași direcție și |delta_ratio| > threshold.
    """
    df = bar_df.copy()

    # Direcție delta per bară
    df['delta_dir'] = np.sign(df['delta_ratio'])
    df['is_imbalance'] = (df['delta_ratio'].abs() > threshold).astype(int)
    df['imbalance_signed'] = df['delta_dir'] * df['is_imbalance']

    # Stacked: contorizăm run-uri consecutive de aceeași direcție
    df['run_id']   = (df['delta_dir'] != df['delta_dir'].shift(1)).cumsum()
    run_lengths    = df.groupby('run_id')['is_imbalance'].transform('sum')
    df['stack_len'] = run_lengths * df['is_imbalance']

    # Aggregate per sesiune
    stacked = df.groupby('session_id').agg(
        stacked_imbalance_max=('stack_len', 'max'),
        stacked_imbalance_count=('is_imbalance', 'sum'),
        stacked_imbalance_dir=('imbalance_signed', 'sum'),
    ).reset_index()

    # Normalizăm
    stacked['stacked_imbalance_dir'] = np.sign(stacked['stacked_imbalance_dir'])
    roll_mean = stacked['stacked_imbalance_count'].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).mean()
    roll_std  = stacked['stacked_imbalance_count'].ewm(halflife=EWM_HL_SESSION, min_periods=EWM_MIN_P).std().replace(0, np.nan)
    stacked['stacked_zscore'] = (
        (stacked['stacked_imbalance_count'] - roll_mean) / roll_std
    ).fillna(0).clip(-4, 4)

    s = session_df.merge(
        stacked[['session_id', 'stacked_imbalance_max', 'stacked_imbalance_count',
                  'stacked_imbalance_dir', 'stacked_zscore']],
        on='session_id', how='left'
    )
    return s


# ─── CVD DIVERGENCE ──────────────────────────────────────────────────────────
def compute_cvd_divergence(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 1: Price face high nou dar CVD nu → divergență bearish.
    Price face low nou dar CVD nu → divergență bullish.
    """
    s = session_df.copy()

    if 'session_close' not in s.columns or 'cvd_final' not in s.columns:
        return s

    price_high_5 = s['session_close'].rolling(5, min_periods=3).max()
    cvd_high_5   = s['cvd_final'].rolling(5, min_periods=3).max()
    price_low_5  = s['session_close'].rolling(5, min_periods=3).min()
    cvd_low_5    = s['cvd_final'].rolling(5, min_periods=3).min()

    # Divergență bearish: preț nou high, CVD NU nou high
    s['cvd_bearish_div'] = (
        (s['session_close'] >= price_high_5 * 0.998) &
        (s['cvd_final'] < cvd_high_5 * 0.998)
    ).astype(int)

    # Divergență bullish: preț nou low, CVD NU nou low
    s['cvd_bullish_div'] = (
        (s['session_close'] <= price_low_5 * 1.002) &
        (s['cvd_final'] > cvd_low_5 * 1.002)
    ).astype(int)

    return s


# ─── SESSION BUILDER ─────────────────────────────────────────────────────────
def build_sessions(df: pd.DataFrame, session_type: str = "NY") -> tuple:
    """
    Împarte datele în sesiuni și returnează (bar_df, session_df).
    """
    start_time, end_time = SESSIONS[session_type]
    df = df.copy()

    # Convertim la ET
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df['timestamp_et'] = df['timestamp'].dt.tz_convert(TIMEZONE)

    df['date']     = df['timestamp_et'].dt.date
    df['time_str'] = df['timestamp_et'].dt.strftime('%H:%M')

    # Filtrăm orele sesiunii
    mask = (df['time_str'] >= start_time) & (df['time_str'] < end_time)
    session_bars = df[mask].copy()

    if session_bars.empty:
        raise ValueError(f"Nu s-au găsit bare pentru sesiunea {session_type}")

    # Session ID unic = date
    session_bars['session_id'] = session_bars['date'].astype(str)

    # Session summary (una per zi)
    session_summary = session_bars.groupby('session_id').agg(
        date=('date', 'first'),
        session_open=('open', 'first'),
        session_close=('close', 'last'),
        session_high=('high', 'max'),
        session_low=('low', 'min'),
        total_vol=('volume', 'sum'),
    ).reset_index()

    return session_bars, session_summary


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────
def generate_features(input_path: Path, output_path: Path, session_type: str = "NY",
                       verbose: bool = True) -> pd.DataFrame:
    """Pipeline complet de generare features."""

    if verbose:
        print(f"\n📊 Citesc {input_path}...")

    df = pd.read_parquet(input_path)

    if verbose:
        print(f"   {len(df):,} bare | {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(f"\n⚙️  Calculez features sintetice de order flow ({session_type})...\n")

    # 1. Bar-level features (delta, footprint, absorption, large trades)
    if verbose: print("  [1/7] Delta & Footprint sintetic...")
    df = compute_bar_delta(df)

    if verbose: print("  [2/7] Absorption & Large trades...")
    df = compute_absorption(df)
    df = compute_large_trades(df)

    # 2. Split în sesiuni
    if verbose: print(f"  [3/7] Building sesiuni {session_type}...")
    bar_df, session_df = build_sessions(df, session_type)

    # 3. CVD per sesiune
    if verbose: print("  [4/7] CVD sesiune + rolling stats...")
    bar_df = compute_session_cvd(bar_df, 'session_id')

    # CVD final per sesiune → merge în session_df
    cvd_final = bar_df.groupby('session_id').agg(
        cvd_final=('cvd_session', 'last'),
        cvd_pct_session=('cvd_pct_session', 'last'),
    ).reset_index()
    session_df = session_df.merge(cvd_final, on='session_id', how='left')
    session_df = compute_rolling_cvd_stats(session_df)

    # 4. Volume Profile
    if verbose: print("  [5/7] Volume Profile (VWAP, POC, VAH/VAL)...")
    session_df = compute_volume_profile(bar_df, session_df)

    # 5. Microstructure
    if verbose: print("  [6/7] Market Microstructure (Amihud, Kyle, Spread)...")
    session_df = compute_microstructure(session_df, bar_df)

    # 6. Opening drive, stacked imbalances, absorption session
    if verbose: print("  [7/7] Opening drive, Stacked imbalances, Absorption...")
    session_df = compute_opening_drive(bar_df, session_df)
    session_df = compute_stacked_imbalances(bar_df, session_df)
    session_df = compute_session_absorption(session_df, bar_df)
    session_df = compute_cvd_divergence(session_df)

    # Adăugăm session_type
    session_df['session_type'] = session_type

    # ─── Lista finală de features ─────────────────────────────────────────────
    feature_cols = [
        'session_id', 'date', 'session_type',
        # CVD & Delta
        'cvd_final', 'cvd_zscore_20d', 'cvd_pct_20d', 'cvd_pct_session',
        'cvd_momentum', 'cvd_momentum_z', 'cvd_acceleration',
        'cvd_roc_5', 'cvd_roc_10', 'cvd_trend_flag',
        'delta_ema_fast', 'delta_ema_slow', 'delta_macd',
        'cvd_bearish_div', 'cvd_bullish_div',
        # Absorption
        'absorption_score_mean', 'absorption_score_max',
        'absorption_flag_count', 'absorption_zscore_20d',
        # Large trades
        # (bar-level, aggregate per sesiune separat dacă e nevoie)
        # Volume Profile
        'vwap_dist', 'vwap_zscore', 'poc_dist',
        'vah_dist', 'val_dist', 'value_area_width',
        # Microstructure
        'amihud_mean', 'amihud_mean_zscore',
        'spread_proxy_mean', 'spread_proxy_mean_zscore',
        # Opening drive
        'opening_drive_dir', 'opening_drive_strength',
        'opening_drive_zscore', 'opening_delta_ratio', 'opening_range',
        # Stacked imbalances
        'stacked_imbalance_max', 'stacked_imbalance_count',
        'stacked_imbalance_dir', 'stacked_zscore',
        # Volume
        'total_vol',
        # Sesiune OHLC
        'session_open', 'session_close', 'session_high', 'session_low',
    ]

    # Păstrăm doar coloanele existente
    available = [c for c in feature_cols if c in session_df.columns]
    result = session_df[available].copy()

    # Fill NA cu 0 pentru features numerice
    num_cols = result.select_dtypes(include=[np.number]).columns
    result[num_cols] = result[num_cols].fillna(0)

    if verbose:
        print(f"\n✅ Features generate:")
        print(f"   Sesiuni: {len(result):,}")
        print(f"   Features: {len(available) - 3} (fără session_id, date, session_type)")
        print(f"   Interval: {result['date'].min()} → {result['date'].max()}")
        print(f"\n   Features list:")
        for col in sorted(available):
            if col not in ['session_id', 'date', 'session_type']:
                print(f"     ✓ {col}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic order flow features')
    parser.add_argument('--input', type=str, default=str(INPUT_FILE),
                        help='Input parquet (NQ_continuous)')
    parser.add_argument('--out', type=str, default=str(OUTPUT_FILE),
                        help='Output parquet path')
    parser.add_argument('--session', type=str, default='both',
                        choices=['NY', 'LON', 'both'],
                        help='Sesiunea de calculat (default: both)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.out)

    print("=" * 65)
    print("  ALADIN — Synthetic Order Flow Generator")
    print("=" * 65)

    if not input_path.exists():
        print(f"❌ Input nu există: {input_path}")
        print(f"   Rulează mai întâi: python3 stitch_continuous_nq.py")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sessions = ['NY', 'LON'] if args.session == 'both' else [args.session]
    all_results = []

    for session_type in sessions:
        print(f"\n{'='*40}")
        print(f"  Sesiunea: {session_type}")
        print(f"{'='*40}")
        try:
            result = generate_features(input_path, output_path, session_type,
                                        verbose=not args.quiet)
            all_results.append(result)
        except Exception as e:
            print(f"  ⚠️  Eroare sesiunea {session_type}: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("❌ Nu s-au generat features")
        return 1

    final = pd.concat(all_results, ignore_index=True)
    final = final.sort_values(['date', 'session_type']).reset_index(drop=True)

    final.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n💾 Salvat: {output_path} ({size_mb:.1f} MB)")
    print(f"   {len(final):,} sesiuni × {len(final.columns)} coloane")

    return 0


if __name__ == '__main__':
    exit(main())
