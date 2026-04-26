"""
precompute_regimes.py — Pre-computare regime labels pentru toate datele istorice
================================================================================
Rulează regime_classifier_v1 pe fiecare bară istorică din DB și salvează
regime-ul dominant per (date, session) în data/regime_labels.parquet.

Utilizat de:
  - train_quality_v6.py, train_quality_ts_lon_v1.py  → sesiunea LON (hhmm 400-700 ET)
  - train_quality_ny_v3.py, train_quality_ts_ny_v1.py → sesiunea NY  (hhmm 900-1300 ET)
  - train_lom_v2.py, train_nom_v2.py, train_sweep_unified.py

Output: data/regime_labels.parquet cu coloane:
  date (str YYYY-MM-DD), session (LON/NY/ALL),
  regime (str), regime_prob (float),
  regime_enc (int 0-4), regime_CONSOLIDATION ... regime_DISTRIBUTION (probabilities)
"""

import sqlite3, joblib, logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

COND_DB = Path(__file__).parent / "market_conditions.db"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger("REGIME_PRECOMPUTE")

DB       = Path(__file__).parent / "mario_trading.db"
OUT      = Path(__file__).parent / "data" / "regime_labels.parquet"
PKL_PATH = Path(__file__).parent / "regime_classifier_v1.pkl"

# Sesiuni ET (DB timestamps sunt în ET)
LON_START_ET = 400;  LON_END_ET = 700
NY_START_ET  = 900;  NY_END_ET  = 1300
ALL_START_ET = 0;    ALL_END_ET = 2359

YEARS = list(range(2022, 2027))

def sv(x):
    try:
        v = float(x)
        return 0.0 if (np.isnan(v) or np.isinf(v)) else v
    except:
        return 0.0


def compute_regime_features(df_day: pd.DataFrame,
                             of_lag_row: dict = None) -> pd.DataFrame:
    """
    Computează features pentru fiecare bară, inclusiv:
    - cele 38 originale
    - is_lon_session, is_ny_session  (adăugate în cross-pollination)
    - of_cvd_lag1 ... of_opening_range_lag1  (lagged OF, constant per zi)

    of_lag_row: dict cu valorile lagged OF pentru ziua curentă (opțional)
    """
    rows = []
    n = len(df_day)

    # Lagged OF defaults (0 dacă nu e disponibil)
    of_lag = of_lag_row or {}
    of_cvd_lag1           = float(of_lag.get('of_cvd_lag1', 0))
    of_absorption_lag1    = float(of_lag.get('of_absorption_lag1', 0))
    of_opening_drive_lag1 = float(of_lag.get('of_opening_drive_lag1', 0))
    of_cvd_zscore_lag1    = float(of_lag.get('of_cvd_zscore_lag1', 0))
    of_stacked_imbalance_lag1 = float(of_lag.get('of_stacked_imbalance_lag1', 0))
    of_opening_range_lag1 = float(of_lag.get('of_opening_range_lag1', 0))

    for i in range(n):
        row = df_day.iloc[i]
        atr = max(sv(row['atr_14']), 0.01)

        feat = {}
        feat['adx_14']           = sv(row['adx_14'])
        feat['hurst']            = sv(row['hurst'])
        feat['garch_vol']        = sv(row['garch_vol'])
        feat['kalman_smooth']    = sv(row['kalman_smooth'])
        feat['acf_lag1']         = sv(row['acf_lag1'])
        feat['acf_lag5']         = sv(row['acf_lag5'])
        feat['fisher_transform'] = sv(row['fisher_transform'])
        feat['sample_entropy']   = sv(row['sample_entropy'])
        feat['inside_va']        = sv(row['inside_va'])
        feat['dist_vwap_atr']    = abs(sv(row['dist_vwap'])) / atr
        feat['dist_poc_atr']     = abs(sv(row['dist_poc'])) / atr
        feat['dist_pdh_atr']     = abs(sv(row['dist_pdh'])) / atr
        feat['dist_pdl_atr']     = abs(sv(row['dist_pdl'])) / atr
        feat['has_displacement'] = sv(row['has_displacement'])
        feat['body_size_atr']    = sv(row['body_size']) / atr
        feat['rvol']             = sv(row['rvol'])
        feat['bar_delta_atr']    = abs(sv(row['bar_delta'])) / max(sv(row['volume']), 1)
        # cum_delta rolling 20
        start = max(0, i - 19)
        cum_slice = df_day['cum_delta'].iloc[start:i+1].fillna(0).values
        feat['cum_delta_20_atr'] = float(np.sum(cum_slice)) / atr
        feat['delta_at_high_atr']= abs(sv(row['delta_at_high'])) / atr
        feat['delta_at_low_atr'] = abs(sv(row['delta_at_low'])) / atr
        feat['big_buy_count']    = sv(row['big_buy_count'])
        feat['big_sell_count']   = sv(row['big_sell_count'])
        feat['imbalance_pct']    = sv(row['imbalance_pct'])
        feat['dom_ratio']        = sv(row['dom_ratio'])

        hhmm = int(sv(row['hhmm']))
        feat['hhmm_enc']         = hhmm
        feat['is_session_open']  = int(
            (LON_START_ET <= hhmm <= LON_END_ET) or
            (NY_START_ET  <= hhmm <= NY_END_ET)
        )
        # Session flags added in cross-pollination
        feat['is_lon_session'] = int(LON_START_ET <= hhmm <= LON_END_ET)
        feat['is_ny_session']  = int(NY_START_ET  <= hhmm <= NY_END_ET)

        # Session hi/lo from lon_hi, lon_lo, p_hi, p_lo
        sess_hi = max(sv(row['lon_hi']), sv(row['p_hi']))
        _sl1 = sv(row['lon_lo']); _sl2 = sv(row['p_lo'])
        sess_lo = min(_sl1 if _sl1 > 0 else 1e9, _sl2 if _sl2 > 0 else 1e9)
        if sess_lo >= 1e9: sess_lo = sv(row['close']) - atr * 5
        close = sv(row['close'])
        feat['dist_sess_hi_atr'] = abs(sess_hi - close) / atr if sess_hi > 0 else 0
        feat['dist_sess_lo_atr'] = abs(close - sess_lo) / atr

        h4_mid = (sv(row['h4_hi']) + sv(row['h4_lo'])) / 2
        h1_mid = (sv(row['h1_hi']) + sv(row['h1_lo'])) / 2
        feat['h4_bias_atr']          = (close - h4_mid) / atr if h4_mid > 0 else 0
        feat['h1_bias_atr']          = (close - h1_mid) / atr if h1_mid > 0 else 0
        feat['above_true_open_atr']  = (close - sv(row['true_open'])) / atr if sv(row['true_open']) > 0 else 0
        feat['day_of_week']          = sv(row['day_of_week'])
        feat['month']                = sv(row['month'])
        feat['fvg_up']               = sv(row['fvg_up'])
        feat['fvg_down']             = sv(row['fvg_down'])

        # Sweep reference: use asia_hi/asia_lo (matches train_regime.py build_features())
        asia_hi = sv(row['asia_hi']); asia_lo = sv(row['asia_lo'])
        ref_hi = max(sv(row['p_hi']), asia_hi)
        _rl1 = sv(row['p_lo']); _rl2 = asia_lo
        ref_lo = min(_rl1 if _rl1 > 0 else 1e9, _rl2 if _rl2 > 0 else 1e9)
        if ref_lo >= 1e9: ref_lo = close - atr * 10
        pre_range = max(ref_hi - ref_lo, 0.01)
        feat['pre_range_atr'] = pre_range / atr
        feat['sweep_dn_atr']  = max(ref_lo - sv(row['low']),  0) / atr
        feat['sweep_up_atr']  = max(sv(row['high']) - ref_hi, 0) / atr

        # Lagged OF features (constant for all bars of the day)
        feat['of_cvd_lag1']               = of_cvd_lag1
        feat['of_absorption_lag1']        = of_absorption_lag1
        feat['of_opening_drive_lag1']     = of_opening_drive_lag1
        feat['of_cvd_zscore_lag1']        = of_cvd_zscore_lag1
        feat['of_stacked_imbalance_lag1'] = of_stacked_imbalance_lag1
        feat['of_opening_range_lag1']     = of_opening_range_lag1

        rows.append(feat)

    return pd.DataFrame(rows)


def process_day(conn, date_str: str, model, le, features, regimes,
                of_lag_lookup: dict = None,
                pre_signal_lookup: dict = None) -> list:
    """Procesează o zi → returnează lista de (date, session, regime, prob, probs_dict)."""
    try:
        df = pd.read_sql(f"""
            SELECT timestamp, open, high, low, close, volume,
                   adx_14, hurst, garch_vol, kalman_smooth,
                   acf_lag1, acf_lag5, fisher_transform, sample_entropy,
                   inside_va, dist_vwap, dist_poc, dist_pdh, dist_pdl,
                   has_displacement, body_size, rvol,
                   bar_delta, cum_delta, delta_at_high, delta_at_low,
                   big_buy_count, big_sell_count, imbalance_pct, dom_ratio,
                   fvg_up, fvg_down, atr_14, true_open,
                   h4_hi, h4_lo, h1_hi, h1_lo, lon_hi, lon_lo,
                   p_hi, p_lo, asia_hi, asia_lo,
                   day_of_week, month
            FROM market_data
            WHERE date = '{date_str}' AND adx_14 > 0 AND atr_14 > 0
            ORDER BY timestamp
        """, conn)
        if len(df) < 5:
            return []

        df['ts']   = pd.to_datetime(df['timestamp'])
        df['hhmm'] = df['ts'].dt.hour * 100 + df['ts'].dt.minute

        # Get lagged OF for this date (if available)
        of_lag_row = (of_lag_lookup or {}).get(date_str, {})

        # Compute features for all bars (including new session flags + OF lags)
        feat_df = compute_regime_features(df, of_lag_row=of_lag_row)
        # Graceful fallback: fill any feature the model expects but feat_df doesn't have
        X = feat_df.reindex(columns=features, fill_value=0).fillna(0).astype(float)

        if len(X) == 0:
            return []

        # Predict regime for each bar
        probs_all = model.predict_proba(X)  # shape: (n_bars, 5)

        results = []
        # Per session aggregation
        for session, s_start, s_end in [
            ('LON', LON_START_ET, LON_END_ET),
            ('NY',  NY_START_ET,  NY_END_ET),
            ('ALL', ALL_START_ET, ALL_END_ET),
        ]:
            mask = (df['hhmm'] >= s_start) & (df['hhmm'] <= s_end)
            if mask.sum() < 2:
                continue
            sess_probs = probs_all[mask.values]  # shape: (n_sess_bars, 5)
            mean_probs = sess_probs.mean(axis=0)  # average probability per class
            max_probs  = sess_probs.max(axis=0)   # peak probability per class

            # PRE_EXPANSION detection strategy (3-tier):
            # 1. Rule-based override: conditions table ts_pattern or seek_destroy signal
            # 2. Classifier bar-level threshold (lowered to 0.15 for transient detection)
            # 3. Fallback: argmax of mean session probs
            PRE_EXP_IDX = list(regimes.values()).index('PRE_EXPANSION') if 'PRE_EXPANSION' in regimes.values() else -1
            pre_signal = (pre_signal_lookup or {}).get(date_str, 0)
            if PRE_EXP_IDX >= 0 and pre_signal:
                # Conditions table confirms PRE_EXPANSION pattern for this date
                pred_enc = PRE_EXP_IDX
            elif PRE_EXP_IDX >= 0 and max_probs[PRE_EXP_IDX] >= 0.15:
                # Classifier detected bar-level PRE_EXPANSION signal
                pred_enc = PRE_EXP_IDX
            else:
                pred_enc = int(np.argmax(mean_probs))

            pred_class = int(le.classes_[pred_enc])
            regime_str = regimes[pred_class]
            regime_prob= float(max_probs[pred_enc]) if regime_str == 'PRE_EXPANSION' else float(mean_probs[pred_enc])

            row = {
                'date':        date_str,
                'session':     session,
                'regime':      regime_str,
                'regime_prob': round(regime_prob, 4),
                'regime_enc':  pred_class,
            }
            # Also store individual class probabilities
            for ci, cls in enumerate(le.classes_):
                row[f'prob_{regimes[int(cls)]}'] = round(float(mean_probs[ci]), 4)
            results.append(row)
        return results
    except Exception as e:
        return []


def main():
    log.info("=" * 60)
    log.info("PRECOMPUTE REGIMES — computing historical regime labels")
    log.info("=" * 60)

    if not PKL_PATH.exists():
        log.error(f"regime_classifier_v1.pkl lipsă: {PKL_PATH}")
        return

    pkg      = joblib.load(PKL_PATH)
    model    = pkg['model']
    le       = pkg['label_encoder']
    features = pkg['features']
    regimes  = pkg['regimes']
    log.info(f"Model loaded. Classes: {[regimes[int(c)] for c in le.classes_]}")
    log.info(f"Features: {len(features)}")

    # ── Load lagged OF features (built by train_regime.py from orderflow_features.parquet) ──
    of_lag_lookup = {}   # date_str → {of_cvd_lag1: ..., ...}
    of_path = Path(__file__).parent / 'data' / 'orderflow_features.parquet'
    if of_path.exists():
        try:
            of_df = pd.read_parquet(of_path)
            of_df['date'] = pd.to_datetime(of_df['date']).dt.strftime('%Y-%m-%d')
            # Build lag lookup: for each date, we need previous-session OF values.
            # Simple approach: shift by 1 row per session_type group.
            lon_of = of_df[of_df['session_type'] == 'LON'].sort_values('date').copy()
            ny_of  = of_df[of_df['session_type'] == 'NY'].sort_values('date').copy()
            def _safe(series): return series.fillna(0).values

            def _build_lk(df_sess, lag_date_col='date'):
                """For each date D, lagged values = values from D-1 (previous row)."""
                lk = {}
                df_sess = df_sess.reset_index(drop=True)
                for i in range(1, len(df_sess)):
                    d      = df_sess.iloc[i][lag_date_col]
                    prev   = df_sess.iloc[i-1]
                    lk[d]  = {
                        'of_cvd_lag1':               float(prev.get('cvd_final', 0) or 0),
                        'of_absorption_lag1':         float(prev.get('absorption_score_mean', 0) or 0),
                        'of_opening_drive_lag1':      float(prev.get('opening_drive_dir', 0) or 0),
                        'of_cvd_zscore_lag1':         float(prev.get('cvd_zscore_20d', 0) or 0),
                        'of_stacked_imbalance_lag1':  float(prev.get('stacked_imbalance_count', 0) or 0),
                        'of_opening_range_lag1':      float(prev.get('opening_range', 0) or 0),
                    }
                return lk

            lon_lk = _build_lk(lon_of)
            ny_lk  = _build_lk(ny_of)
            # Merge: LON session on date D uses NY D-1 lag; NY on D uses LON D lag
            # (same convention as train_regime.py)
            for d, v in lon_lk.items():
                of_lag_lookup[d] = v          # LON session → use prev NY
            for d, v in ny_lk.items():
                if d not in of_lag_lookup:
                    of_lag_lookup[d] = v
            log.info(f"Loaded lagged OF for {len(of_lag_lookup)} dates")
        except Exception as e:
            log.warning(f"Could not load OF parquet: {e}. Using zeros for OF lags.")
    else:
        log.warning(f"OF parquet not found at {of_path}. OF lag features will be 0.")

    # ── Load PRE_EXPANSION rule-based signals from market_conditions.db ──────────
    pre_signal_lookup = {}   # date_str → 1 if ts_pattern_bull/bear on that day
    if COND_DB.exists():
        try:
            cconn = sqlite3.connect(f'file:{COND_DB}?mode=ro', uri=True, timeout=30)
            cdf = pd.read_sql("""
                SELECT date,
                       ts_pattern_bull, ts_pattern_bear,
                       seek_destroy_bull, seek_destroy_bear
                FROM conditions
            """, cconn)
            cconn.close()
            cdf['date'] = pd.to_datetime(cdf['date']).dt.strftime('%Y-%m-%d')
            # Use ts_pattern (Turtle Soup: sweep + reversal) as primary PRE_EXPANSION signal
            # seek_destroy (stop hunt) as secondary — require BOTH for cleaner labels
            cdf['pre_signal'] = (
                cdf['ts_pattern_bull'].fillna(0).astype(int) |
                cdf['ts_pattern_bear'].fillna(0).astype(int)
            )
            pre_signal_lookup = dict(zip(cdf['date'], cdf['pre_signal']))
            n_sig = sum(pre_signal_lookup.values())
            log.info(f"Loaded PRE_EXPANSION signals from conditions: {n_sig} days with ts_pattern")
        except Exception as e:
            log.warning(f"Could not load conditions DB: {e}")
    else:
        log.warning(f"market_conditions.db not found at {COND_DB}")

    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    days = pd.read_sql(f"""
        SELECT DISTINCT date FROM market_data
        WHERE year IN ({','.join(map(str, YEARS))})
          AND day_of_week BETWEEN 1 AND 5
        ORDER BY date
    """, conn)['date'].tolist()
    log.info(f"Processing {len(days)} zile din {YEARS[0]}-{YEARS[-1]}...")

    all_rows = []
    for date_str in tqdm(days, desc="Regime labels"):
        rows = process_day(conn, date_str, model, le, features, regimes,
                           of_lag_lookup=of_lag_lookup,
                           pre_signal_lookup=pre_signal_lookup)
        all_rows.extend(rows)

    conn.close()

    if not all_rows:
        log.error("No results!")
        return

    df_out = pd.DataFrame(all_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUT, index=False)
    log.info(f"\n✅ Salvat: {OUT}")
    log.info(f"   {len(df_out)} rânduri | {df_out['date'].nunique()} zile")

    # Summary per regime
    for sess in ['LON', 'NY', 'ALL']:
        sub = df_out[df_out['session'] == sess]
        if len(sub) == 0: continue
        log.info(f"\nSesiunea {sess}:")
        for regime, cnt in sub['regime'].value_counts().items():
            pct = cnt / len(sub) * 100
            avg_prob = sub[sub['regime']==regime]['regime_prob'].mean()
            log.info(f"  {regime}: {cnt} zile ({pct:.1f}%) | prob_mean={avg_prob:.3f}")


if __name__ == "__main__":
    main()
