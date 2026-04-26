"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  ALADIN — Advanced Features Module                                                       ║
║  advanced_features.py  |  Tier 1 + Tier 2 Indicators                                   ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Tier 1 (Critical):
  1. Hurst Exponent          — trending vs mean-reverting regime detection
  2. GARCH(1,1)              — forward-looking volatility prediction
  3. Kalman Filter           — adaptive price smoothing + noise score
  4. ADX                     — trend strength indicator
  5. VWAP                    — volume-weighted average price (intraday anchor)
  6. Fractional Kelly        — optimal position sizing

Tier 2 (Enhancement):
  7. Sample Entropy (SampEn) — market complexity / predictability
  8. Sortino Ratio           — downside-only risk-adjusted return
  9. Calmar Ratio            — return / max drawdown
 10. Ehlers Fisher Transform — normalized oscillator turning points
 11. FFT Cycle Detection     — dominant cycle period extraction
 12. Autocorrelation (ACF)   — serial correlation in returns

Usage:
  import advanced_features as af
  df = af.compute_all_advanced(df)           # Add all features to DataFrame
  af.hurst_exponent(series, window=100)      # Individual function
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from numba import njit            
from joblib import Parallel, delayed 
from tqdm import tqdm             

# =============================================================================
# 1. HURST EXPONENT — Regime Detection
# =============================================================================
# H > 0.5 → trending (momentum works)
# H < 0.5 → mean-reverting (fade works)
# H ≈ 0.5 → random walk (no edge)

def hurst_exponent(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Rolling Hurst exponent via R/S (Rescaled Range) analysis.
    Uses log(R/S) / log(n) over sub-windows.
    """
    def _hurst_rs(x):
        if len(x) < 20:
            return 0.5
        n = len(x)
        mean_x = np.mean(x)
        y = np.cumsum(x - mean_x)
        r = np.max(y) - np.min(y)
        s = np.std(x, ddof=1)
        if s < 1e-10:
            return 0.5
        rs = r / s
        if rs < 1e-10:
            return 0.5
        return np.log(rs) / np.log(n)

    returns = series.pct_change().fillna(0)
    result = returns.rolling(window, min_periods=20).apply(_hurst_rs, raw=True)
    return result.fillna(0.5)


# =============================================================================
# 2. GARCH(1,1) — Volatility Prediction
# =============================================================================
# σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
# Simplified iterative version (no MLE fitting — too slow for rolling)
# Uses standard GARCH(1,1) params: α=0.1, β=0.85, ω=1-α-β=0.05

def garch_volatility(series: pd.Series, alpha: float = 0.10, beta: float = 0.85,
                     window: int = 50) -> pd.Series:
    """
    GARCH(1,1) conditional volatility with fixed parameters.
    Returns predicted volatility (σ) for next period.
    Faster than MLE fitting, still captures vol clustering.
    """
    omega = 1.0 - alpha - beta
    returns = series.pct_change().fillna(0).values
    n = len(returns)
    sigma2 = np.full(n, np.nan)

    # Initialize with rolling variance of first `window` returns
    if n < window:
        return pd.Series(np.full(n, 0.0), index=series.index)

    init_var = np.var(returns[:window])
    if init_var < 1e-16:
        init_var = 1e-8
    sigma2[:window] = init_var

    # GARCH recursion
    for t in range(window, n):
        eps2 = returns[t - 1] ** 2
        sigma2[t] = omega * init_var + alpha * eps2 + beta * sigma2[t - 1]
        # Clamp to avoid explosion
        sigma2[t] = min(max(sigma2[t], 1e-16), 1.0)

    result = np.sqrt(np.where(np.isnan(sigma2), 0, sigma2))
    return pd.Series(result, index=series.index)


# =============================================================================
# 3. KALMAN FILTER — Adaptive Smoothing + Noise Score
# =============================================================================
# Simple 1D Kalman: state = price, measurement = observed price
# Returns: smoothed price + noise score (|raw - smoothed| / ATR)

def kalman_filter(series: pd.Series, process_noise: float = 0.01,
                  measurement_noise: float = 0.1) -> Tuple[pd.Series, pd.Series]:
    """
    1D Kalman filter for price smoothing.
    Returns (smoothed_price, kalman_noise_score).
    noise_score = abs(raw - smoothed) — higher = more noisy market.
    """
    values = series.values.astype(float)
    n = len(values)
    smoothed = np.zeros(n)
    noise = np.zeros(n)

    # Initialize
    x_est = values[0] if not np.isnan(values[0]) else 0.0
    p_est = 1.0
    Q = process_noise    # Process noise covariance
    R = measurement_noise  # Measurement noise covariance

    for t in range(n):
        if np.isnan(values[t]):
            smoothed[t] = x_est
            noise[t] = 0.0
            continue

        # Predict
        x_pred = x_est
        p_pred = p_est + Q

        # Update (Kalman gain)
        K = p_pred / (p_pred + R)
        x_est = x_pred + K * (values[t] - x_pred)
        p_est = (1 - K) * p_pred

        smoothed[t] = x_est
        noise[t] = abs(values[t] - x_est)

    return (pd.Series(smoothed, index=series.index),
            pd.Series(noise, index=series.index))


# =============================================================================
# 4. ADX — Average Directional Index (Trend Strength)
# =============================================================================
# ADX > 25 = trending, ADX < 20 = chop/range
# Uses Wilder smoothing (EMA with alpha = 1/period)

def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """
    ADX (Average Directional Index) — trend strength [0, 100].
    ADX > 25 → strong trend, ADX < 20 → range/chop.
    """
    # +DM / -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0),
                        index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0),
                         index=high.index)

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Wilder smoothing (EMA with alpha=1/period)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.clip(lower=1e-8)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.clip(lower=1e-8)

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-8)
    adx_val = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx_val.fillna(0)


# =============================================================================
# 5. VWAP — Volume Weighted Average Price
# =============================================================================
# Intraday anchor: VWAP = Σ(Price × Volume) / Σ(Volume)
# Resets daily. Distance from VWAP = mean-reversion signal.

def vwap(close: pd.Series, high: pd.Series, low: pd.Series,
         volume: pd.Series, dates: pd.Series) -> pd.Series:
    """
    VWAP resetting daily. Uses typical price = (H+L+C)/3.
    Returns VWAP series aligned to input index.
    """
    typical = (high + low + close) / 3.0
    tp_vol = typical * volume

    # Group by date, cumulative sum
    cum_tpv = tp_vol.groupby(dates).cumsum()
    cum_vol = volume.groupby(dates).cumsum()

    vwap_vals = cum_tpv / cum_vol.clip(lower=1)
    return vwap_vals.fillna(close)


# =============================================================================
# 6. FRACTIONAL KELLY — Optimal Position Sizing
# =============================================================================
# Full Kelly: f* = (b*p - q) / b  where p=win_rate, q=1-p, b=avg_win/avg_loss
# Fractional Kelly: f_frac = fraction * f*  (typically 25-50%)

def fractional_kelly(win_rate: float, avg_win: float, avg_loss: float,
                     fraction: float = 0.25) -> dict:
    """
    Fractional Kelly criterion for position sizing.
    Returns dict with full_kelly, fractional_kelly, and sizing recommendation.

    Args:
        win_rate: 0-1 probability of winning
        avg_win: average win amount (positive)
        avg_loss: average loss amount (positive, will be treated as positive)
        fraction: Kelly fraction (default 0.25 = quarter Kelly, conservative)
    """
    avg_loss = abs(avg_loss) if avg_loss != 0 else 1.0
    avg_win = abs(avg_win) if avg_win != 0 else 0.0
    p = max(0.0, min(1.0, win_rate))
    q = 1.0 - p

    if avg_loss < 1e-8:
        return {"full_kelly": 0.0, "fractional_kelly": 0.0,
                "risk_pct": 0.0, "edge": 0.0, "recommendation": "NO EDGE"}

    b = avg_win / avg_loss  # odds ratio
    full_kelly = (b * p - q) / b if b > 0 else 0.0
    full_kelly = max(0.0, full_kelly)  # Never negative (no edge = 0)

    frac_kelly = full_kelly * fraction
    edge = p * avg_win - q * avg_loss

    if full_kelly <= 0:
        rec = "NO EDGE — do not trade"
    elif frac_kelly < 0.01:
        rec = "Minimal edge — minimum size"
    elif frac_kelly < 0.05:
        rec = "Small edge — conservative sizing"
    elif frac_kelly < 0.15:
        rec = "Good edge — standard sizing"
    else:
        rec = "Strong edge — max allowed sizing"

    return {
        "full_kelly":       round(full_kelly * 100, 2),      # percentage
        "fractional_kelly": round(frac_kelly * 100, 2),      # percentage
        "risk_pct":         round(min(frac_kelly, 0.10) * 100, 2),  # capped at 10%
        "edge":             round(edge, 2),
        "odds_ratio":       round(b, 2),
        "recommendation":   rec,
    }


# =============================================================================
# 7. SAMPLE ENTROPY (SampEn) — Market Complexity (M4 TURBO OPTIMIZED)
# =============================================================================
# Low SampEn → predictable/structured market (good for pattern-based trading)
# High SampEn → chaotic/random market (reduce position size)

@njit(fastmath=True)
def _numba_sampen_core(x, m, r):
    """Nucleu matematic compilat care rulează la viteză de C++"""
    n = len(x)
    if n < m + 2: return 0.0
    
    def _count_matches(template_len):
        count = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                match = True
                for k in range(template_len):
                    if abs(x[i + k] - x[j + k]) > r:
                        match = False
                        break
                if match: count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0: return 0.0
    return -np.log(A / B) if A > 0 else 0.0

def sample_entropy(series: pd.Series, m: int = 2, r_mult: float = 0.2,
                   window: int = 100) -> pd.Series:
    """
    Rolling Sample Entropy - Versiunea paralelizată pentru 10 nuclee.
    """
    returns = series.pct_change().fillna(0).values
    stds = pd.Series(returns).rolling(window).std().values
    n = len(returns)
    
    # Funcția care va fi trimisă către fiecare nucleu al procesorului M4
    def _worker(i):
        if i < window or stds[i] < 1e-10: 
            return 0.0
        # Trimitem bucata de date către motorul Numba
        return _numba_sampen_core(returns[i-window:i], m, r_mult * stds[i])

    # n_jobs=-1 forțează folosirea tuturor celor 10 nuclee simultan
    results = Parallel(n_jobs=-1)(
        delayed(_worker)(i) for i in tqdm(range(n), desc="      [M4] SampEn Progress")
    )
    return pd.Series(results, index=series.index)


# =============================================================================
# 8. SORTINO RATIO — Downside Risk-Adjusted Return
# =============================================================================
# Like Sharpe but only penalizes downside volatility

def sortino_ratio(pnl_list: list, annualize: float = 252.0) -> float:
    """
    Sortino ratio from list of PnL values.
    Only uses negative returns for denominator.
    """
    if len(pnl_list) < 2:
        return 0.0

    arr = np.array(pnl_list, dtype=float)
    mean_ret = np.mean(arr)
    downside = arr[arr < 0]

    if len(downside) < 2:
        return 999.0 if mean_ret > 0 else 0.0

    downside_std = np.std(downside, ddof=1)
    if downside_std < 1e-10:
        return 999.0 if mean_ret > 0 else 0.0

    return round(mean_ret / downside_std * np.sqrt(annualize), 2)


# =============================================================================
# 9. SHARPE RATIO — Classic Risk-Adjusted Return
# =============================================================================
# Penalizează ATÂT volatilitatea pozitivă cât și cea negativă.
# Diferență față de Sortino: Sortino e mai corect pentru trading (ne interesează
# doar riscul downside), dar Sharpe e universal recunoscut — Instagram, brokers,
# prop firms îl folosesc cel mai des.
# Interpretare: < 0 = rău, 0-1 = slab, 1-2 = decent, > 2 = bun, > 3 = excepțional

def sharpe_ratio(pnl_list: list, annualize: float = 252.0) -> float:
    """
    Sharpe ratio from list of PnL values (trade-by-trade).
    annualize=252 pentru daily P&L; 1.0 dacă dai lista brută (nu anualizezi).
    Returnează 0.0 dacă insuficiente date sau deviație zero.
    """
    if len(pnl_list) < 2:
        return 0.0
    arr = np.array(pnl_list, dtype=float)
    mean_ret = np.mean(arr)
    std_ret  = np.std(arr, ddof=1)
    if std_ret < 1e-10:
        return 999.0 if mean_ret > 0 else 0.0
    return round(mean_ret / std_ret * np.sqrt(annualize), 2)


# =============================================================================
# 10. CALMAR RATIO — Return / Max Drawdown
# =============================================================================

def calmar_ratio(total_pnl: float, max_drawdown_usd: float) -> float:
    """
    Calmar ratio = annualized return / max drawdown.
    Simplified: total PnL / max drawdown (absolute).
    """
    if max_drawdown_usd < 1e-2:
        return 999.0 if total_pnl > 0 else 0.0
    return round(total_pnl / max_drawdown_usd, 2)


# =============================================================================
# 10. EHLERS FISHER TRANSFORM — Normalized Oscillator
# =============================================================================
# Transforms any oscillator to have nearly Gaussian distribution
# Sharp turning points → clear entry/exit signals

def ehlers_fisher_transform(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Ehlers Fisher Transform.
    Normalizes price to [-1, 1] range then applies Fisher transform.
    Result: near-Gaussian with sharp turning points.
    """
    # Normalize to [-1, 1] using rolling min/max
    roll_max = series.rolling(period, min_periods=1).max()
    roll_min = series.rolling(period, min_periods=1).min()
    rng = (roll_max - roll_min).clip(lower=1e-8)
    normalized = 2.0 * (series - roll_min) / rng - 1.0
    # Clip to avoid infinity in atanh
    normalized = normalized.clip(-0.999, 0.999)

    # Fisher transform: F = 0.5 * ln((1+x)/(1-x)) = atanh(x)
    # Apply EMA smoothing first
    smoothed = normalized.ewm(span=5, adjust=False).mean().clip(-0.999, 0.999)
    fisher = 0.5 * np.log((1.0 + smoothed) / (1.0 - smoothed))

    return fisher.fillna(0)


# =============================================================================
# 11. FFT CYCLE DETECTION — Dominant Cycle Period
# =============================================================================
# Extract dominant cycle length from price data
# Useful for timing entries (are we at cycle peak or trough?)

def fft_dominant_cycle(series: pd.Series, window: int = 128) -> pd.Series:
    """
    Rolling FFT to detect dominant cycle period.
    Returns the dominant cycle length (in bars) for each point.
    Window should be power of 2 for FFT efficiency.
    """
    def _dominant_period(x):
        x = np.asarray(x, dtype=float)
        # Detrend (remove linear trend)
        n = len(x)
        if n < 16:
            return 0.0
        trend = np.linspace(x[0], x[-1], n)
        detrended = x - trend

        # FFT
        fft_vals = np.fft.rfft(detrended)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n)

        # Ignore DC component (freq=0) and very low frequencies
        min_period = 5    # minimum 5-bar cycle
        max_period = n // 2  # max half the window
        valid_mask = (freqs > 0) & (freqs <= 1.0 / min_period)
        if not np.any(valid_mask):
            return 0.0

        power_valid = power[valid_mask]
        freqs_valid = freqs[valid_mask]

        # Dominant frequency → period
        dominant_idx = np.argmax(power_valid)
        dominant_freq = freqs_valid[dominant_idx]
        if dominant_freq < 1e-10:
            return 0.0
        period = 1.0 / dominant_freq

        return min(period, max_period)

    result = series.rolling(window, min_periods=32).apply(_dominant_period, raw=True)
    return result.fillna(0)


# =============================================================================
# 12. AUTOCORRELATION (ACF) — Serial Correlation
# =============================================================================
# Positive ACF → momentum regime (trend-following works)
# Negative ACF → mean-reversion regime (fade works)
# ACF ≈ 0 → random walk (no edge from past returns)

def rolling_acf(series: pd.Series, lag: int = 1, window: int = 50) -> pd.Series:
    """
    Rolling autocorrelation at specified lag.
    Measures serial correlation in returns.
    """
    returns = series.pct_change().fillna(0)
    result = returns.rolling(window, min_periods=10).apply(
        lambda x: pd.Series(x).autocorr(lag=lag) if len(x) >= lag + 2 else 0.0,
        raw=False
    )
    return result.fillna(0)


# =============================================================================
# MASTER FUNCTION — Compute All Advanced Features
# =============================================================================

# Feature column names exported by this module
ADVANCED_FEATURE_COLS = [
    'hurst',                # Hurst exponent (0-1)
    'garch_vol',            # GARCH(1,1) predicted volatility
    'kalman_smooth',        # Kalman filtered price
    'kalman_noise',         # |raw - kalman| noise score
    'adx_14',              # ADX trend strength (0-100)
    'vwap',                # Volume-weighted average price
    'dist_vwap',           # Distance from VWAP (close - vwap)
    'sample_entropy',      # SampEn complexity score
    'fisher_transform',    # Ehlers Fisher normalized oscillator
    'fft_cycle',           # Dominant cycle period (bars)
    'acf_lag1',            # Autocorrelation lag-1
    'acf_lag5',            # Autocorrelation lag-5
]


def compute_all_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all Tier 1 + Tier 2 features and add as columns.
    Requires: close, high, low, volume, date columns.
    Returns: df with ADVANCED_FEATURE_COLS added.
    """
    df = df.copy()
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(1, index=df.index)

    # Determine date column for VWAP reset
    if 'date' in df.columns:
        dates = df['date']
    elif 'timestamp' in df.columns:
        dates = pd.to_datetime(df['timestamp']).dt.date
    else:
        dates = pd.Series(0, index=df.index)  # No reset — treat as single day

    print("    🧮 Advanced Features: Hurst Exponent...")
    df['hurst'] = hurst_exponent(close, window=100)

    print("    🧮 Advanced Features: GARCH(1,1)...")
    df['garch_vol'] = garch_volatility(close, alpha=0.10, beta=0.85, window=50)

    print("    🧮 Advanced Features: Kalman Filter...")
    df['kalman_smooth'], df['kalman_noise'] = kalman_filter(close)

    print("    🧮 Advanced Features: ADX(14)...")
    df['adx_14'] = adx(high, low, close, period=14)

    print("    🧮 Advanced Features: VWAP...")
    df['vwap'] = vwap(close, high, low, volume, dates)
    df['dist_vwap'] = close - df['vwap']

    print("    🧮 Advanced Features: Sample Entropy...")
    df['sample_entropy'] = sample_entropy(close, m=2, r_mult=0.2, window=100)

    print("    🧮 Advanced Features: Ehlers Fisher Transform...")
    df['fisher_transform'] = ehlers_fisher_transform(close, period=10)

    print("    🧮 Advanced Features: FFT Cycle Detection...")
    df['fft_cycle'] = fft_dominant_cycle(close, window=128)

    print("    🧮 Advanced Features: Autocorrelation...")
    df['acf_lag1'] = rolling_acf(close, lag=1, window=50)
    df['acf_lag5'] = rolling_acf(close, lag=5, window=50)

    print("    ✅ Advanced Features: 12 coloane adăugate")
    return df


# =============================================================================
# LIVE MODE — Compute from small window (last N bars)
# =============================================================================

def compute_live_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute advanced features for live inference from a small DataFrame
    (typically last 100-200 bars from SQLite).
    Same as compute_all_advanced but without progress prints.
    """
    df = df.copy()
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(1, index=df.index)

    if 'date' in df.columns:
        dates = df['date']
    elif 'timestamp' in df.columns:
        dates = pd.to_datetime(df['timestamp']).dt.date
    else:
        dates = pd.Series(0, index=df.index)

    df['hurst'] = hurst_exponent(close, window=min(100, len(df) - 1))
    df['garch_vol'] = garch_volatility(close, window=min(50, len(df) - 1))
    df['kalman_smooth'], df['kalman_noise'] = kalman_filter(close)
    df['adx_14'] = adx(high, low, close, period=14)
    df['vwap'] = vwap(close, high, low, volume, dates)
    df['dist_vwap'] = close - df['vwap']
    # SampEn is expensive — use smaller window for live
    df['sample_entropy'] = sample_entropy(close, m=2, r_mult=0.2, window=min(50, len(df) - 1))
    df['fisher_transform'] = ehlers_fisher_transform(close, period=10)
    df['fft_cycle'] = fft_dominant_cycle(close, window=min(128, len(df)))
    df['acf_lag1'] = rolling_acf(close, lag=1, window=min(50, len(df) - 1))
    df['acf_lag5'] = rolling_acf(close, lag=5, window=min(50, len(df) - 1))

    return df
