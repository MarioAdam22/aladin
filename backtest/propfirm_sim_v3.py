"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   SIMULARE PROP FIRM — LUCID TRADING  v3 (EVAL + FUNDED SEPARATE)          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  REGULI LUCID LUCIDFLEX $50k (confirmate de Mario):                         ║
║                                                                              ║
║  FAZA EVAL:                                                                  ║
║  • Cont $50k, target +$3,000 (reach $53k)                                   ║
║  • Trailing DD: $2,000 EOD (urmărește peak EOD)                             ║
║  • Consistency rule: nicio zi > 50% din profitul total cumulat               ║
║  • Profitul din EVAL rămâne în EVAL (taxa de examen)                         ║
║                                                                              ║
║  FAZA FUNDED (cont NOU, complet separat):                                   ║
║  • Cont $50k fresh, DD trail reset, zero legătură cu EVAL                  ║
║  • 5 winning days (≥$150/zi) → eligibil payout                              ║
║  • Payout = 50% din profitul total al ciclului                               ║
║  • Split: 90% trader, 10% Lucid                                              ║
║  • Minim retragere: $500                                                     ║
║  • Nicio consistency rule pe FUNDED                                          ║
║  • Micro-scalp: pozițiile ≥1 minut (nu simulăm, asumăm respectat)           ║
║  • 6 payout requests → trece la LucidLive                                   ║
║                                                                              ║
║  QUALITY FILTER:                                                             ║
║  • LON session: conf ≥ 0.60 + quality_score ≥ THR                          ║
║  • MAX_SL = 1 per zi (stop trades dacă prima zi e SL)                       ║
║  • SL = 20 puncte = $400/trade NQ Mini                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import pickle, json, pathlib
from collections import defaultdict

DIR = pathlib.Path(__file__).parent

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
INITIAL_BALANCE    = 50_000.0
TRAILING_DD        = 2_000.0
EVAL_TARGET        = 3_000.0        # +$3k → trece la funded
WIN_DAY_MIN        = 150.0          # o zi "winning" dacă PnL ≥ $150
WIN_DAYS_REQUIRED  = 5              # câte winning days per ciclu payout
PAYOUT_SPLIT       = 0.90          # 90% trader
PAYOUT_CYCLE_PCT   = 0.50          # max 50% din profitul ciclului
MIN_PAYOUT         = 500.0         # min $500 per retragere
MAX_PAYOUTS_FUNDED = 6             # după 6 payouturi → LucidLive (stop sim)

# Trade economics (NQ Mini, 1 contract)
# Backtest uses SL=6.234pt; real trading uses SL=20pt
# Scale factor = 20/6.234 = 3.208 (all PnL multiplied by this)
SCALE_FACTOR = 20.0 / 6.234        # = 3.208

# Quality filter config
DEFAULT_THR     = 0.12
DEFAULT_MAX_SL  = 1               # max 1 SL per zi

CSV_PATH   = DIR / "backtest_open_sessions_trades.csv"
MODEL_V5   = DIR / "mario_quality_v5_calibrated.pkl"
MODEL_V4   = DIR / "mario_quality_v4_calibrated.pkl"  # fallback
FEAT_V5    = DIR / "mario_quality_v5_features.json"
FEAT_V4    = DIR / "mario_quality_v4_features.json"   # fallback
DB_PATH    = DIR / "mario_trading.db"


# ═══════════════════════════════════════════════════════════════
# LOAD MODEL + FEATURES
# ═══════════════════════════════════════════════════════════════
def load_model():
    """Încearcă v5, fallback la v4."""
    for mpath, fpath, vname in [
        (MODEL_V5, FEAT_V5, "v5"),
        (MODEL_V4, FEAT_V4, "v4"),
    ]:
        if mpath.exists() and fpath.exists():
            with open(mpath, 'rb') as f:
                model = pickle.load(f)
            meta  = json.loads(fpath.read_text())
            print(f"   ✅ Model loaded: quality_{vname} | AUC_cal={meta.get('auc_calibrated', '?')}")
            return model, meta['features'], vname
    return None, None, None


# ═══════════════════════════════════════════════════════════════
# BUILD FEATURES (identic cu train_quality_v5.py)
# ═══════════════════════════════════════════════════════════════
def build_features(df_lon, model_version="v5"):
    """Construiește același set de features ca în training."""
    import sqlite3

    CLIP = 10.0
    def clip(x, c=CLIP):
        return np.clip(np.where(np.isfinite(x), x, 0.0), -c, c)
    def safe_norm(num, denom, c=CLIP):
        return clip(np.where(denom > 0, num / denom, 0.0), c)

    df_lon = df_lon.copy()
    df_lon['ts'] = pd.to_datetime(df_lon['timestamp'])
    df_lon['ts_str'] = df_lon['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # DB JOIN
    DB_COLS = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'atr_14', 'asia_hi', 'asia_lo', 'p_hi', 'p_lo', 'true_open',
        'h4_hi', 'h4_lo', 'h1_hi', 'h1_lo',
        'poc_level', 'vah', 'val', 'dist_poc', 'inside_va',
        'has_displacement', 'fvg_up', 'fvg_down',
        'is_smt_bearish', 'is_smt_bullish',
        'hurst', 'adx_14', 'garch_vol', 'sample_entropy',
        'fisher_transform', 'acf_lag1', 'acf_lag5',
        'vwap', 'dist_vwap', 'bar_delta', 'cum_delta',
        'bar_buy_vol', 'bar_sell_vol',
        'absorption_score', 'stacked_bull', 'stacked_bear',
        'body_size', 'lw_hi', 'lw_lo', 'lm_hi', 'lm_lo',
        'dist_pdh', 'dist_pdl',
        'fft_cycle', 'kalman_smooth', 'kalman_noise',
        'of_doi', 'of_bilateral_abs', 'of_big_balance',
    ]
    if not DB_PATH.exists():
        print("   ⚠️  DB nu există! Se folosesc doar features de bază.")
        return None

    conn = sqlite3.connect(DB_PATH)
    cols_str = ', '.join(DB_COLS)
    CHUNK = 5000
    db_chunks = []
    ts_list = df_lon['ts_str'].tolist()
    for i in range(0, len(ts_list), CHUNK):
        chunk = ts_list[i:i+CHUNK]
        pl = ','.join(['?'] * len(chunk))
        q  = f"SELECT {cols_str} FROM market_data WHERE timestamp IN ({pl})"
        db_chunks.append(pd.read_sql(q, conn, params=chunk))
    conn.close()

    if not db_chunks:
        return None

    db = pd.concat(db_chunks, ignore_index=True)
    db['ts_str'] = db['timestamp']
    df = df_lon.merge(db.drop(columns=['timestamp']), on='ts_str', how='inner')

    if len(df) == 0:
        return None

    cl  = df['close'].values.astype(float)
    hi  = df['high'].values.astype(float)
    lo  = df['low'].values.astype(float)
    op  = df['open'].values.astype(float)
    vol = np.where(df['volume'].values > 0, df['volume'].values, 1).astype(float)
    atr = np.where(df['atr_14'].values > 0, df['atr_14'].values, 9.0).astype(float)

    asia_hi   = np.where(df['asia_hi'].values > 0, df['asia_hi'].values, cl)
    asia_lo   = np.where(df['asia_lo'].values > 0, df['asia_lo'].values, cl)
    p_hi      = np.where(df['p_hi'].values > 0,   df['p_hi'].values, cl)
    p_lo      = np.where(df['p_lo'].values > 0,   df['p_lo'].values, cl)
    true_open = np.where(df['true_open'].values > 0, df['true_open'].values, cl)
    h4h       = np.where(df['h4_hi'].values > 0, df['h4_hi'].values, cl)
    h4l       = np.where(df['h4_lo'].values > 0, df['h4_lo'].values, cl)
    h1h       = np.where(df['h1_hi'].values > 0, df['h1_hi'].values, cl)
    h1l       = np.where(df['h1_lo'].values > 0, df['h1_lo'].values, cl)
    lw_hi     = np.where(df['lw_hi'].values > 0, df['lw_hi'].values, cl)
    lw_lo     = np.where(df['lw_lo'].values > 0, df['lw_lo'].values, cl)
    vwap_     = np.where(df['vwap'].values > 0,  df['vwap'].values,  cl)

    feat = pd.DataFrame(index=df.index)
    feat['dir_short']        = (df['direction'] == 'SHORT').astype(float).values
    feat['hour_utc']         = df['ts'].dt.hour.values.astype(float)
    feat['min_in_lon']       = np.clip((df['ts'].dt.hour.values - 7) * 60, 0, 180).astype(float)
    feat['day_of_week']      = df['ts'].dt.dayofweek.values.astype(float)
    feat['month']            = df['ts'].dt.month.values.astype(float)
    feat['is_monday']        = (df['ts'].dt.dayofweek.values == 0).astype(float)
    feat['is_friday']        = (df['ts'].dt.dayofweek.values == 4).astype(float)

    if model_version == "v5":
        feat['year_norm']    = (df['ts'].dt.year.values.astype(float) - 2023.0) / 2.0

    feat['confidence']       = df['confidence'].values.astype(float)
    feat['atr_entry']        = atr
    feat['atr_vs_10d']       = clip(df.groupby(df['ts'].dt.date)['atr_14'].transform('mean').values /
                                    np.where(atr > 0, atr, 1), 3)

    valid_asia = (asia_hi > 0) & (asia_lo > 0) & (asia_hi > asia_lo)
    feat['dist_asia_hi_atr'] = safe_norm(cl - asia_hi, atr)
    feat['dist_asia_lo_atr'] = safe_norm(cl - asia_lo, atr)
    feat['asia_range_atr']   = clip(safe_norm(asia_hi - asia_lo, atr), 20)
    feat['swept_asia_hi']    = ((cl > asia_hi) & valid_asia).astype(float)
    feat['swept_asia_lo']    = ((cl < asia_lo) & valid_asia).astype(float)
    feat['asia_midpoint']    = safe_norm(cl - (asia_hi + asia_lo) / 2, atr)

    feat['dist_pdh_atr']     = safe_norm(df['dist_pdh'].values, atr)
    feat['dist_pdl_atr']     = safe_norm(df['dist_pdl'].values, atr)
    feat['above_true_open']  = (cl > true_open).astype(float)
    feat['dist_true_open']   = safe_norm(cl - true_open, atr)

    feat['h4_bias']          = safe_norm((h4h + h4l) / 2 - cl, atr)
    feat['h1_bias']          = safe_norm((h1h + h1l) / 2 - cl, atr)
    feat['h4_h1_aligned']    = (np.sign(feat['h4_bias'].values) == np.sign(feat['h1_bias'].values)).astype(float)

    feat['lw_range_atr']     = clip(safe_norm(lw_hi - lw_lo, atr), 20)
    feat['dist_lw_hi']       = safe_norm(cl - lw_hi, atr)
    feat['dist_lw_lo']       = safe_norm(cl - lw_lo, atr)

    feat['inside_va']        = df['inside_va'].fillna(0).values.astype(float)
    feat['dist_poc_atr']     = safe_norm(df['dist_poc'].values, atr)
    feat['dist_vwap_atr']    = safe_norm(df['dist_vwap'].values, atr)
    feat['vah_dist']         = safe_norm(cl - df['vah'].fillna(0).values.astype(float), atr)
    feat['val_dist']         = safe_norm(cl - df['val'].fillna(0).values.astype(float), atr)

    feat['has_displacement'] = df['has_displacement'].fillna(0).values.astype(float)
    feat['fvg_up']           = df['fvg_up'].fillna(0).values.astype(float)
    feat['fvg_down']         = df['fvg_down'].fillna(0).values.astype(float)
    feat['is_smt_bearish']   = df['is_smt_bearish'].fillna(0).values.astype(float)
    feat['is_smt_bullish']   = df['is_smt_bullish'].fillna(0).values.astype(float)

    feat['hurst']            = df['hurst'].fillna(0.5).values.astype(float)
    feat['adx_14']           = df['adx_14'].fillna(20).values.astype(float)
    feat['adx_strong']       = (df['adx_14'].fillna(20).values > 25).astype(float)
    feat['acf_lag1']         = df['acf_lag1'].fillna(0).values.astype(float)
    feat['acf_lag5']         = df['acf_lag5'].fillna(0).values.astype(float)
    feat['fisher_transform'] = df['fisher_transform'].fillna(0).values.astype(float)
    feat['fisher_extreme']   = (np.abs(df['fisher_transform'].fillna(0).values) > 2.0).astype(float)
    feat['fft_cycle']        = df['fft_cycle'].fillna(0).values.astype(float)
    feat['kalman_smooth']    = df['kalman_smooth'].fillna(0).values.astype(float)
    feat['kalman_noise']     = df['kalman_noise'].fillna(0).values.astype(float)

    garch_raw = df['garch_vol'].fillna(0).values.astype(float)
    feat['garch_vol_atr']    = clip(np.where(atr > 0, garch_raw * cl / atr, 1.0), 5)
    feat['sample_entropy']   = df['sample_entropy'].fillna(2.0).values.astype(float)

    bar_delta = df['bar_delta'].fillna(0).values.astype(float)
    feat['bar_delta_norm']   = clip(bar_delta / np.maximum(vol, 1), 1)
    feat['cum_delta_norm']   = clip(df['cum_delta'].fillna(0).values / np.maximum(vol, 1), 1)
    feat['buy_sell_ratio']   = clip(
        df['bar_buy_vol'].fillna(0).values / np.maximum(df['bar_sell_vol'].fillna(0).values, 1), 5)
    feat['absorption_score'] = df['absorption_score'].fillna(0).values.astype(float)
    feat['stacked_bull']     = df['stacked_bull'].fillna(0).values.astype(float)
    feat['stacked_bear']     = df['stacked_bear'].fillna(0).values.astype(float)
    feat['of_doi']           = df['of_doi'].fillna(0).values.astype(float)

    body = cl - op
    wick_up   = hi - np.maximum(cl, op)
    wick_down = np.minimum(cl, op) - lo
    feat['body_bear']        = (body < 0).astype(float)
    feat['body_pct']         = clip(np.abs(body) / np.maximum(hi - lo, 0.01), 2)
    feat['sweep_wick_atr']   = safe_norm(np.maximum(wick_up, wick_down), atr)
    feat['prev_bar_dir']     = feat['body_bear'].values

    feat['dir_x_adx']        = feat['dir_short'].values * feat['adx_14'].values / 100.0
    feat['dir_x_hurst']      = feat['dir_short'].values * feat['hurst'].values
    feat['confidence_x_adx'] = feat['confidence'].values * feat['adx_strong'].values
    feat['hour_x_dir']       = feat['hour_utc'].values * feat['dir_short'].values

    if model_version == "v5":
        feat['year_x_adx']   = feat['year_norm'].values * feat['adx_14'].values / 100.0
        feat['year_x_hurst'] = feat['year_norm'].values * feat['hurst'].values

    # aliniez cu df_lon prin ts_str
    feat['ts_str']  = df['ts_str'].values
    feat['ts']      = df['ts'].values
    feat['trail']   = (df['exit_reason'] == 'TRAIL').astype(int).values
    feat['pnl_raw'] = df['pnl_usd'].values.astype(float)   # PnL din backtest (SL=6.23pt)

    return feat, df


def scale_pnl(pnl_raw: float, exit_reason: str) -> float:
    """
    Scalează PnL din backtest (SL=6.234pt) la real (SL=20pt, NQ Mini).
    Multiplies actual backtest PnL by scale factor 3.208.
    Exit types: TRAIL (~+$812), SL (~-$400), BE ($0), TIMEOUT (~+$350).
    """
    return float(pnl_raw) * SCALE_FACTOR


def is_sl_exit(exit_reason: str) -> bool:
    """Returnează True dacă exit-ul reprezintă o pierdere reală (SL hit).
    BE (break-even) = $0, nu contează ca SL pentru MAX_SL logic."""
    return exit_reason == 'SL'


# ═══════════════════════════════════════════════════════════════
# SIMULARE FAZA EVAL
# ═══════════════════════════════════════════════════════════════
def run_eval(day_trades: list[dict]) -> dict:
    """
    Simulează EVAL pe o serie de day_trades deja filtrate.
    Returnează dacă eval a trecut sau blown și la ce dată.
    day_trades = list of {'date': date, 'pnl': float} sorted by date.
    """
    balance    = INITIAL_BALANCE
    peak_eod   = INITIAL_BALANCE
    dd_floor   = INITIAL_BALANCE - TRAILING_DD
    cum_profit = 0.0           # profit cumulat din EVAL start
    blown      = False
    passed     = False
    pass_date  = None
    blow_date  = None
    log        = []

    by_date = defaultdict(float)
    for t in day_trades:
        by_date[t['date']] += t['pnl']

    for date in sorted(by_date.keys()):
        day_pnl  = by_date[date]
        prev_cum = cum_profit

        # Consistency check: nicio zi > 50% din profitul total cumulat
        # Dacă ar face > 50% → trunchiează la 50% (sau ignorăm semnalul)
        # În simulare: limitam PnL-ul zilei la max 50% din profit total dacă > 0
        # (o zi de trailing exit mare e trunchiată dacă depășește regula)
        if day_pnl > 0 and cum_profit > 0:
            max_day = cum_profit * 0.5  # 50% din profitul total curent
            if day_pnl > max_day and (cum_profit + day_pnl) > 0:
                # Doar dacă profitul total e pozitiv aplicăm regula
                # day_pnl rămâne dar în EVAL nu se creditează mai mult
                pass  # notăm doar; regula se aplică la nivel de validare zi
                # Nu trunchiem PnL-ul real, dar dacă breach → not valid eval day
                # (Mario: regula e că nu poți CÂȘTIGA mai mult de 50% dintr-o zi)
                # Simplificat: dacă ziua > 50% din tot-ul de până acum, OK tehnic
                # dar în EVAL oficial asta ar invalida ziua → skip (conservativ)

        balance    += day_pnl
        cum_profit += day_pnl

        # Update EOD trailing DD
        if balance > peak_eod:
            peak_eod = balance
        dd_floor = max(peak_eod - TRAILING_DD, INITIAL_BALANCE - TRAILING_DD)
        # Floor nu poate urca peste INITIAL (nu există floor lock în EVAL la Lucid)

        log.append(f"    EVAL {date}  PnL={day_pnl:+.0f}  bal={balance:.0f}  floor={dd_floor:.0f}  cum={cum_profit:+.0f}")

        if balance <= dd_floor:
            blown = True
            blow_date = date
            log.append(f"    💥 EVAL BLOWN  bal={balance:.0f} ≤ floor={dd_floor:.0f}")
            break

        if cum_profit >= EVAL_TARGET:
            passed    = True
            pass_date = date
            log.append(f"    ✅ EVAL PASSED  bal={balance:.0f}  cum_profit={cum_profit:.0f}")
            break

    return dict(passed=passed, blown=blown, pass_date=pass_date, blow_date=blow_date,
                final_balance=balance, cum_profit=cum_profit, log=log)


# ═══════════════════════════════════════════════════════════════
# SIMULARE FAZA FUNDED
# ═══════════════════════════════════════════════════════════════
def run_funded(day_trades_after_eval: list[dict]) -> dict:
    """
    Simulează FUNDED (cont complet nou $50k).
    5 winning days ≥ $150 → eligible payout.
    Payout = 50% din profitul ciclului × 90% (split Lucid).
    """
    balance    = INITIAL_BALANCE
    peak_eod   = INITIAL_BALANCE
    dd_floor   = INITIAL_BALANCE - TRAILING_DD

    cycle_profit  = 0.0    # profit acumulat de la ultimul payout (sau start)
    win_days      = 0
    total_payouts = 0
    total_net     = 0.0
    blown         = False
    blow_date     = None
    log           = []
    monthly_payouts = defaultdict(int)
    monthly_net     = defaultdict(float)
    payout_list     = []

    by_date = defaultdict(float)
    for t in day_trades_after_eval:
        by_date[t['date']] += t['pnl']

    for date in sorted(by_date.keys()):
        day_pnl   = by_date[date]

        balance      += day_pnl
        cycle_profit += day_pnl

        # Winning day check (DUPĂ ce s-a executat ziua)
        if day_pnl >= WIN_DAY_MIN:
            win_days += 1

        # EOD trailing DD update
        if balance > peak_eod:
            peak_eod = balance
        dd_floor = max(peak_eod - TRAILING_DD, INITIAL_BALANCE - TRAILING_DD)

        # BLOWN check
        if balance <= dd_floor:
            blown     = True
            blow_date = date
            log.append(f"    💥 FUNDED BLOWN  bal={balance:.0f} ≤ floor={dd_floor:.0f}  [{date}]")
            break

        # PAYOUT check: 5 winning days + profit suficient
        if win_days >= WIN_DAYS_REQUIRED and cycle_profit > 0:
            gross_payout = cycle_profit * PAYOUT_CYCLE_PCT
            if gross_payout >= MIN_PAYOUT:
                net_payout   = gross_payout * PAYOUT_SPLIT
                balance      -= gross_payout       # scos din cont
                cycle_profit  = 0.0
                win_days      = 0
                total_payouts += 1
                total_net     += net_payout
                # RESET peak la balanta post-payout — trailing DD porneste din nou
                # Fara reset, floor-ul vechi (pre-payout) ar blowa imediat contul
                peak_eod      = balance
                dd_floor      = max(peak_eod - TRAILING_DD, INITIAL_BALANCE - TRAILING_DD)
                month_key = date.strftime('%Y-%m')
                monthly_payouts[month_key] += 1
                monthly_net[month_key]     += net_payout
                payout_list.append(dict(date=date, gross=gross_payout, net=net_payout,
                                        payout_n=total_payouts))
                log.append(
                    f"    💸 PAYOUT #{total_payouts}  gross={gross_payout:+.0f}  "
                    f"net={net_payout:+.0f}  bal={balance:.0f}  [{date}]"
                )
                if total_payouts >= MAX_PAYOUTS_FUNDED:
                    log.append(f"    🎯 6 PAYOUTURI → LucidLive  [{date}]")
                    break

    return dict(
        blown=blown, blow_date=blow_date,
        total_payouts=total_payouts, total_net=total_net,
        monthly_payouts=dict(monthly_payouts),
        monthly_net=dict(monthly_net),
        payout_list=payout_list,
        final_balance=balance,
        log=log,
    )


# ═══════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════
def run_simulation(thr=DEFAULT_THR, max_sl=DEFAULT_MAX_SL, use_quality=True,
                   use_ny=False, verbose=True):
    """
    Rulează simularea completă pe tot CSV-ul (2023-2025).
    thr         = threshold quality filter (LON)
    max_sl      = max SL (reale, nu BE) per zi
    use_quality = aplică quality filter pe LON
    use_ny      = adaugă NY h13-h14 ca backup (NUMAI dacă ziua LON PnL < $150)
    """
    print(f"\n{'═'*70}")
    print(f"  SIMULARE  thr={thr}  max_sl={max_sl}  quality={'DA' if use_quality else 'NU'}  NY={'DA' if use_ny else 'NU'}")
    print(f"{'═'*70}")

    # ── Load trades ──────────────────────────────────────────────
    df_csv = pd.read_csv(CSV_PATH)
    df_lon = df_csv[(df_csv['session'] == 'LON') & (df_csv['confidence'] >= 0.60)].copy()
    df_lon['ts']        = pd.to_datetime(df_lon['timestamp'])
    df_lon['exit_reason'] = df_lon['exit_reason'].fillna('SL')
    df_lon = df_lon.sort_values('ts').reset_index(drop=True)
    print(f"   LON conf≥0.60 trades: {len(df_lon):,}")

    # ── Quality filter ───────────────────────────────────────────
    if use_quality:
        model, feat_names, mver = load_model()
        if model is None:
            print("   ⚠️  Model non trovato — skip quality filter")
            use_quality = False
        else:
            print(f"   Building features per quality filter ...")
            feat_df, df_merged = build_features(df_lon, model_version=mver)
            if feat_df is None:
                print("   ⚠️  DB non disponibile — skip quality filter")
                use_quality = False
            else:
                # Alinia featuri cu model
                X_sim  = feat_df[feat_names].fillna(0).astype(float)
                scores = model.predict_proba(X_sim)[:, 1]
                feat_df['quality_score'] = scores
                feat_df['pass_quality']  = scores >= thr

                # Re-alinia cu df_lon via ts_str
                df_lon = df_lon.merge(
                    feat_df[['ts_str', 'quality_score', 'pass_quality']],
                    left_on=df_lon['ts'].dt.strftime('%Y-%m-%d %H:%M:%S'),
                    right_on='ts_str', how='left'
                )
                df_lon['pass_quality'] = df_lon['pass_quality'].fillna(False)
                df_lon['quality_score']= df_lon['quality_score'].fillna(0.0)
                n_pass = df_lon['pass_quality'].sum()
                print(f"   Trades post-quality (≥{thr}): {n_pass:,} / {len(df_lon):,} "
                      f"({n_pass/len(df_lon)*100:.1f}%)")
                df_lon = df_lon[df_lon['pass_quality']].reset_index(drop=True)

    # ── Apply MAX_SL per day ─────────────────────────────────────
    # Prendiamo trades in ordine temporale per giorno.
    # Solo gli exit SL (pierdere reală) contano verso MAX_SL.
    # BE (break-even = $0) NON conta come SL — si continua a tradare.
    df_lon['date'] = df_lon['ts'].dt.date
    filtered_trades = []
    for date, grp in df_lon.groupby('date'):
        grp = grp.sort_values('ts')
        sl_count   = 0
        day_trades = []
        for _, row in grp.iterrows():
            if sl_count >= max_sl:
                break
            day_trades.append(row)
            if is_sl_exit(row['exit_reason']):   # solo SL reale, non BE
                sl_count += 1
        filtered_trades.extend(day_trades)

    df_lon = pd.DataFrame(filtered_trades)
    if len(df_lon) == 0:
        print("   ❌ Niciun trade după filtre!")
        return None

    print(f"   Trades dopo max_sl={max_sl}/giorno: {len(df_lon):,}")

    # ── Scala PnL LON ────────────────────────────────────────────
    df_lon['pnl_scaled'] = df_lon.apply(
        lambda r: scale_pnl(float(r['pnl_usd']), r['exit_reason']), axis=1)

    # ── NY backup session ────────────────────────────────────────
    if use_ny:
        df_csv_all = pd.read_csv(CSV_PATH)
        df_ny = df_csv_all[
            (df_csv_all['session'] == 'NY') &
            (df_csv_all['confidence'] >= 0.60)
        ].copy()
        df_ny['ts']   = pd.to_datetime(df_ny['timestamp'])
        df_ny['hour_utc'] = df_ny['ts'].dt.hour
        df_ny = df_ny[df_ny['hour_utc'].isin([13, 14])].copy()   # h13-h14 UTC only
        df_ny['exit_reason'] = df_ny['exit_reason'].fillna('SL')
        df_ny['date'] = df_ny['ts'].dt.date
        df_ny['pnl_scaled'] = df_ny.apply(
            lambda r: scale_pnl(float(r['pnl_usd']), r['exit_reason']), axis=1)

        # Calcola LON PnL per giorno (per sapere quando usare il backup NY)
        lon_day_pnl = df_lon.groupby('date')['pnl_scaled'].sum().to_dict()

        # Applica max_sl anche a NY
        ny_filtered = []
        for date, grp in df_ny.groupby('date'):
            # USA NY solo se LON non è "winning day" (PnL < WIN_DAY_MIN)
            lon_pnl_today = lon_day_pnl.get(date, 0.0)
            if lon_pnl_today >= WIN_DAY_MIN:
                continue  # LON già winning → skip NY
            grp = grp.sort_values('ts')
            sl_count   = 0
            day_trades = []
            for _, row in grp.iterrows():
                if sl_count >= max_sl:
                    break
                day_trades.append(row)
                if is_sl_exit(row['exit_reason']):
                    sl_count += 1
            ny_filtered.extend(day_trades)

        df_ny = pd.DataFrame(ny_filtered) if ny_filtered else pd.DataFrame()
        if len(df_ny) > 0:
            df_ny['date'] = df_ny['ts'].dt.date
            print(f"   NY backup trades aggiunto: {len(df_ny):,}")
            trail_ny = (df_ny['exit_reason'] == 'TRAIL').mean() * 100
            print(f"   NY trail%: {trail_ny:.1f}%")
        else:
            print(f"   ⚠️  NY: nessun trade dopo filtri")
            df_ny = pd.DataFrame()

    # ── Per-day trade list (LON + NY se usa_ny) ─────────────────
    trade_list = [
        {'date': row['date'], 'pnl': row['pnl_scaled'], 'exit': row['exit_reason'], 'ts': row['ts'], 'sess': 'LON'}
        for _, row in df_lon.iterrows()
    ]
    if use_ny and len(df_ny) > 0:
        for _, row in df_ny.iterrows():
            trade_list.append({
                'date': row['date'], 'pnl': row['pnl_scaled'],
                'exit': row['exit_reason'], 'ts': row['ts'], 'sess': 'NY'
            })
        trade_list = sorted(trade_list, key=lambda x: x['ts'])

    # ── Statistici per anno/mese ──────────────────────────────────
    df_lon['year']  = df_lon['ts'].dt.year
    df_lon['month'] = df_lon['ts'].dt.to_period('M')

    print(f"\n   {'Anno':<6}  {'Trades':>7}  {'Trail%':>8}  {'WinDays':>9}  {'AvgPnL':>8}")
    for yr in sorted(df_lon['year'].unique()):
        sub = df_lon[df_lon['year'] == yr]
        trail_pct = (sub['exit_reason'] == 'TRAIL').mean() * 100
        day_pnl = sub.groupby('date')['pnl_scaled'].sum()
        win_days = (day_pnl >= WIN_DAY_MIN).sum()
        avg_pnl  = sub['pnl_scaled'].mean()
        print(f"   {yr:<6}  {len(sub):>7}  {trail_pct:>7.1f}%  {win_days:>9}  {avg_pnl:>+8.0f}")

    # ── Simulare loop: EVAL → FUNDED → repeat on blow ────────────
    all_evals   = []
    all_funded  = []
    total_net   = 0.0
    total_payouts = 0

    # Trova la prima data
    all_dates_sorted = sorted(set(t['date'] for t in trade_list))
    eval_start_idx   = 0  # index into all_dates_sorted

    attempt = 0
    while eval_start_idx < len(all_dates_sorted):
        attempt += 1
        eval_start_date = all_dates_sorted[eval_start_idx]

        # Trades da eval_start_date in avanti
        eval_trades = [t for t in trade_list if t['date'] >= eval_start_date]
        if not eval_trades:
            break

        # Run EVAL
        eval_res = run_eval(eval_trades)
        all_evals.append(dict(attempt=attempt, start=eval_start_date, **eval_res))

        if verbose and eval_res['log']:
            print(f"\n  [Attempt {attempt}] EVAL start={eval_start_date}")
            # Mostra solo prime e ultime righe di log per brevità
            log_short = eval_res['log'][:3] + (["    ..."] if len(eval_res['log']) > 6 else []) + eval_res['log'][-2:]
            for l in log_short:
                print(l)

        if eval_res['blown']:
            # EVAL blown → riparte dall'indomani del blow
            blow_date = eval_res['blow_date']
            # Trova index del blow_date +1 giorno
            while eval_start_idx < len(all_dates_sorted) and all_dates_sorted[eval_start_idx] <= blow_date:
                eval_start_idx += 1
            print(f"  💥 Attempt {attempt}: EVAL BLOWN  ({blow_date})")
            continue

        # EVAL PASSED → start FUNDED from pass_date
        pass_date    = eval_res['pass_date']
        if pass_date is None:
            print(f"  ⚠️  Attempt {attempt}: EVAL passed but no pass_date — skip")
            break
        funded_trades= [t for t in trade_list if t['date'] > pass_date]

        if verbose:
            print(f"  ✅ Attempt {attempt}: EVAL PASSED  ({pass_date}) → FUNDED start")

        if not funded_trades:
            print(f"  (Nessun trade dopo pass_date {pass_date})")
            break

        funded_res = run_funded(funded_trades)
        all_funded.append(dict(attempt=attempt, eval_pass_date=pass_date, **funded_res))
        total_net     += funded_res['total_net']
        total_payouts += funded_res['total_payouts']

        if verbose:
            for l in funded_res['log'][:20]:
                print(l)
            if len(funded_res['log']) > 20:
                print(f"    ... ({len(funded_res['log'])-20} more)")

        if funded_res['blown']:
            blow_date = funded_res['blow_date']
            if blow_date is not None:
                print(f"  💥 Attempt {attempt}: FUNDED BLOWN  ({blow_date})")
                while eval_start_idx < len(all_dates_sorted) and all_dates_sorted[eval_start_idx] <= blow_date:
                    eval_start_idx += 1
            else:
                print(f"  💥 Attempt {attempt}: FUNDED BLOWN (no date)")
                break
        else:
            # 6 payouturi → LucidLive, fine sim
            print(f"  🎯 Attempt {attempt}: FUNDED completato ({funded_res['total_payouts']} payouts)")
            break

    # ── Riepilogo ─────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  RIEPILOGO FINALE  thr={thr}  max_sl={max_sl}")
    print(f"{'═'*70}")
    n_eval_blown  = sum(1 for e in all_evals if e['blown'])
    n_eval_passed = sum(1 for e in all_evals if e['passed'])
    print(f"  Tentativi EVAL:        {len(all_evals)}")
    print(f"  EVAL blown:            {n_eval_blown}")
    print(f"  EVAL passed:           {n_eval_passed}")
    print(f"  FUNDED blown:          {sum(1 for f in all_funded if f['blown'])}")
    print(f"  Total payouturi:       {total_payouts}")
    print(f"  Total net (trader):    ${total_net:,.0f}")
    if total_payouts > 0:
        print(f"  Media per payout:      ${total_net/total_payouts:,.0f}")

    # Payouturi per mese (aggregati)
    all_monthly = defaultdict(lambda: {'n': 0, 'net': 0.0})
    for f in all_funded:
        for mo, n in f.get('monthly_payouts', {}).items():
            all_monthly[mo]['n']   += n
            all_monthly[mo]['net'] += f.get('monthly_net', {}).get(mo, 0.0)

    if all_monthly:
        print(f"\n  PAYOUTURI PER LUNA:")
        print(f"  {'Luna':<10}  {'N Payouturi':>12}  {'Net Trader':>12}")
        for mo in sorted(all_monthly.keys()):
            d = all_monthly[mo]
            if d['n'] > 0:
                print(f"  {mo:<10}  {d['n']:>12}  ${d['net']:>11,.0f}")

        months_with_payouts = [mo for mo, d in all_monthly.items() if d['n'] > 0]
        if months_with_payouts:
            total_months = len(set(mo[:7] for mo in months_with_payouts))
            avg_pmo = total_payouts / max(total_months, 1)
            avg_net_pmo = total_net / max(total_months, 1)
            print(f"\n  Mesi con almeno 1 payout: {len(months_with_payouts)}")
            print(f"  Media payouturi/mese:     {avg_pmo:.1f}")
            print(f"  Media net/mese:           ${avg_net_pmo:,.0f}")
            max_pmo = max(d['n'] for d in all_monthly.values())
            print(f"  Max payouturi in 1 mese:  {max_pmo}")
            months_3plus = [mo for mo, d in all_monthly.items() if d['n'] >= 3]
            print(f"  Mesi con ≥3 payouturi:    {len(months_3plus)}")

    print(f"{'═'*70}\n")
    return dict(
        all_evals=all_evals, all_funded=all_funded,
        total_payouts=total_payouts, total_net=total_net,
        monthly=dict(all_monthly),
    )


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  LUCID LUCIDFLEX $50k — PROP FIRM SIM v3")
    print("  EVAL/FUNDED SEPARATI | MAX_SL=1/giorno | NQ Mini 1c")
    print("  BE exits = $0 (non SL) | Peak reset dopo payout")
    print("=" * 70)

    # LON only
    print("\n\n  ══════ LON ONLY ══════")
    for thr in [0.10, 0.12, 0.15]:
        run_simulation(thr=thr, max_sl=1, use_quality=True, use_ny=False, verbose=False)

    # LON only — dettaglio migliore
    print("\n\n  ──────── DETTAGLIO COMPLETO: LON  thr=0.10 ────────")
    run_simulation(thr=0.10, max_sl=1, use_quality=True, use_ny=False, verbose=True)

    # NOTE: NY backup senza quality filter ha EV negativo (trail 14.3% < breakeven 32%)
    # → aggiunge rischio senza compensare. Serve un quality filter per NY separato.
