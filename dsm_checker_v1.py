"""
DSM (Double Sweep Model) Checker v1
=====================================
Detectează continuation setups DUPĂ un sweep within-range + displacement.
Activ pe sesiunile London și NY.

OOS (2025-2026, XGB v2, 30-min cooldown): AUC=0.709 | WR=79.8%@0.65 (326 setups) | WR=87.1%@0.70 (163 setups)
"""

import pickle, sqlite3, logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd

logger = logging.getLogger("ALADIN.DSM")

# ── UTC gate windows ──────────────────────────────────────────────────────────
SESSIONS = {
    'LON': {'utc_st': 830,  'utc_en': 1030,  # London active: 08:30-10:30 UTC
            'pre_st_et': 100, 'pre_en_et': 359,
            'sess_st_et': 400, 'sess_en_et': 630, 'enc': 0},
    'NY':  {'utc_st': 1400, 'utc_en': 1700,  # NY active: 14:00-17:00 UTC (extins)
            'pre_st_et': 400, 'pre_en_et': 929,
            'sess_st_et': 930, 'sess_en_et': 1200, 'enc': 1},
}

MIN_SWEEP_PT = 4.0
MIN_DISP_PT  = 3.0
MODEL_PATH   = Path(__file__).parent / "dsm_model_v1.pkl"
_ET_H        = 4   # UTC-4 = EDT

_DSM_MODEL = None

def _load_dsm_model():
    global _DSM_MODEL
    if _DSM_MODEL is not None:
        return _DSM_MODEL
    if not MODEL_PATH.exists():
        logger.warning(f"DSM model lipsă: {MODEL_PATH}")
        return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            _DSM_MODEL = pickle.load(f)
        logger.info(f"✅ DSM loaded: OOS AUC={_DSM_MODEL.get('oos_auc',0):.3f} | WR={_DSM_MODEL.get('wr_oos',0):.1%}@{_DSM_MODEL.get('threshold',0.65):.2f}")
        return _DSM_MODEL
    except Exception as e:
        logger.error(f"DSM load error: {e}")
        return None


def sv(v, d=0.0):
    try: x = float(v); return x if np.isfinite(x) else d
    except: return d


def check_dsm_setup(db_path: str, now_utc: datetime = None) -> dict | None:
    """
    Verifică dacă există un setup DSM (Double Sweep continuation) la momentul curent.
    Returnează dict cu {direction, entry, sl, tp, rr, dsm_score, session} sau None.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    hhmm = now_utc.hour * 100 + now_utc.minute

    # Detectăm sesiunea activă
    active_sess = None
    for sname, scfg in SESSIONS.items():
        if scfg['utc_st'] <= hhmm <= scfg['utc_en']:
            active_sess = sname
            sess_cfg = scfg
            break
    if active_sess is None:
        return None

    pkg = _load_dsm_model()
    if pkg is None:
        return None

    now_et = now_utc - timedelta(hours=_ET_H)
    today  = now_et.date()

    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True,
                               timeout=30, check_same_thread=False)
        day_df = pd.read_sql(f"""
            SELECT timestamp, open, high, low, close, volume,
                   atr_14, bar_delta, cum_delta, fvg_up, fvg_down, has_displacement,
                   body_size, adx_14, hurst, dist_poc, inside_va, dist_vwap,
                   delta_at_high, delta_at_low, big_buy_count, big_sell_count,
                   absorption_score, stacked_bull, stacked_bear, of_doi, of_big_balance,
                   bar_buy_vol, bar_sell_vol, garch_vol, kalman_smooth,
                   fisher_transform, acf_lag1, acf_lag5, rvol,
                   vah, val, poc_level, p_hi, p_lo, lw_hi, lw_lo,
                   h4_hi, h4_lo, h1_hi, h1_lo, true_open,
                   asia_hi, asia_lo, day_of_week, month
            FROM market_data
            WHERE date = '{today}'
              AND hour_min BETWEEN '00:00' AND '{now_et.strftime("%H:%M")}'
            ORDER BY timestamp
        """, conn)
        conn.close()
    except Exception as e:
        logger.error(f"DSM DB error: {e}")
        return None

    if len(day_df) < 20:
        return None

    day_df['ts']   = pd.to_datetime(day_df['timestamp'])
    day_df['hhmm'] = day_df['ts'].dt.hour * 100 + day_df['ts'].dt.minute

    pre  = day_df[day_df['hhmm'].between(sess_cfg['pre_st_et'],  sess_cfg['pre_en_et'])]
    sess = day_df[day_df['hhmm'].between(sess_cfg['sess_st_et'], sess_cfg['sess_en_et'])]

    if len(pre) < 5 or len(sess) < 3:
        logger.debug(f"DSM {active_sess}: date insuficiente pre={len(pre)} sess={len(sess)}")
        return None

    pre_hi = float(pre['high'].max())
    pre_lo = float(pre['low'].min())
    pre_rng = pre_hi - pre_lo
    if pre_rng < 8:
        return None

    atr = float(day_df['atr_14'].replace(0, np.nan).dropna().iloc[-1]) if len(day_df) > 0 else 10.0
    if atr <= 0: atr = 10.0

    # ── Caută cel mai recent sweep + displacement în sesiune ──────────────────
    best_setup = None
    best_score = 0.0
    r0 = day_df.iloc[-1]
    pre_last = pre.iloc[-1] if len(pre) > 0 else r0

    sess_reset = sess.reset_index(drop=False)
    for i in range(2, len(sess_reset)):
        bar    = sess_reset.iloc[i]
        bar_hi = sv(bar['high'])
        bar_lo = sv(bar['low'])
        bar_cl = sv(bar['close'])
        bar_op = sv(bar['open'])

        sweep_dn = pre_lo - bar_lo
        sweep_up = bar_hi - pre_hi
        partial_dn = pre_lo + 0.3 * pre_rng - bar_lo
        partial_up = bar_hi - (pre_hi - 0.3 * pre_rng)

        for direction, s_val, p_val in [('LONG', sweep_dn, partial_dn), ('SHORT', sweep_up, partial_up)]:
            is_sweep = (s_val >= MIN_SWEEP_PT) or (p_val >= MIN_SWEEP_PT)
            if not is_sweep:
                continue

            # Cauta displacement dupa sweep
            after = sess_reset.iloc[i+1:min(i+6, len(sess_reset))]
            disp_bar = None
            for _, ab in after.iterrows():
                ab_body = abs(sv(ab['close']) - sv(ab['open']))
                if direction == 'LONG' and sv(ab['close']) > sv(ab['open']) and ab_body >= MIN_DISP_PT:
                    disp_bar = ab; break
                elif direction == 'SHORT' and sv(ab['close']) < sv(ab['open']) and ab_body >= MIN_DISP_PT:
                    disp_bar = ab; break

            if disp_bar is None:
                continue

            entry_price = sv(disp_bar['close'])
            entry_hhmm  = int(disp_bar['hhmm'])
            dir_num = 1 if direction == 'LONG' else -1

            # ── Features ──────────────────────────────────────────────────────
            feat = {
                'session_enc':      sess_cfg['enc'],
                'entry_hhmm':       entry_hhmm,
                'day_of_week':      sv(r0['day_of_week']),
                'month':            sv(r0['month']),
                'sweep_mag_atr':    (s_val if s_val > 0 else p_val) / atr,
                'sweep_is_outside': 1 if s_val >= MIN_SWEEP_PT else 0,
                'partial_depth':    p_val / pre_rng if pre_rng > 0 else 0,
                'pre_range_atr':    pre_rng / atr,
                'bar_range_atr':    (bar_hi - bar_lo) / atr,
                'sweep_wick_pct':   abs(bar_lo - min(sv(bar['open']),sv(bar['close']))) / (bar_hi-bar_lo+0.01) if direction=='LONG' else abs(bar_hi - max(sv(bar['open']),sv(bar['close']))) / (bar_hi-bar_lo+0.01),
                'spike_close_inside': 1 if (direction=='LONG' and bar_cl>=pre_lo) or (direction=='SHORT' and bar_cl<=pre_hi) else 0,
                'disp_body_atr':    abs(sv(disp_bar['close'])-sv(disp_bar['open'])) / atr,
                'disp_has_flag':    sv(disp_bar['has_displacement']),
                'disp_imbalance':   abs(sv(disp_bar['bar_buy_vol'])-sv(disp_bar['bar_sell_vol'])) / max(sv(disp_bar['volume']),1),
                'h4_bias':          dir_num * (entry_price-(sv(r0['h4_hi'])+sv(r0['h4_lo']))/2) / atr if sv(r0['h4_hi'])>0 else 0,
                'h1_bias':          dir_num * (entry_price-(sv(r0['h1_hi'])+sv(r0['h1_lo']))/2) / atr if sv(r0['h1_hi'])>0 else 0,
                'above_true_open':  dir_num * (entry_price-sv(r0['true_open'])) / atr if sv(r0['true_open'])>0 else 0,
                'dist_prev_wk_hi':  (sv(r0['lw_hi'])-entry_price)/atr if sv(r0['lw_hi'])>0 else 0,
                'dist_prev_wk_lo':  (entry_price-sv(r0['lw_lo']))/atr if sv(r0['lw_lo'])>0 else 0,
                'dist_prev_day_hi': (sv(r0['p_hi'])-entry_price)/atr if sv(r0['p_hi'])>0 else 0,
                'dist_prev_day_lo': (entry_price-sv(r0['p_lo']))/atr if sv(r0['p_lo'])>0 else 0,
                'dist_poc_atr':     dir_num*(sv(r0['poc_level'])-entry_price)/atr if sv(r0['poc_level'])>0 else 0,
                'dist_vah_atr':     (sv(r0['vah'])-entry_price)/atr if sv(r0['vah'])>0 else 0,
                'dist_val_atr':     (entry_price-sv(r0['val']))/atr if sv(r0['val'])>0 else 0,
                'inside_va':        sv(disp_bar['inside_va']),
                'dist_vwap_atr':    dir_num*sv(disp_bar['dist_vwap'])/atr,
                'cum_delta_norm':   sv(disp_bar['cum_delta'])/max(sv(disp_bar['volume']),1),
                'bar_delta_norm':   sv(disp_bar['bar_delta'])/max(sv(disp_bar['volume']),1)*dir_num,
                'of_doi':           sv(r0['of_doi'])*dir_num,
                'of_big_balance':   sv(r0['of_big_balance'])*dir_num,
                'absorption_score': sv(r0['absorption_score']),
                'stacked_bull':     sv(r0['stacked_bull']) if direction=='LONG' else sv(r0['stacked_bear']),
                'big_buy_sell_ratio': (sv(disp_bar['big_buy_count'])-sv(disp_bar['big_sell_count']))/max(sv(disp_bar['big_buy_count'])+sv(disp_bar['big_sell_count']),1)*dir_num,
                'atr_norm':         atr/10.0,
                'garch_vol':        sv(r0['garch_vol']),
                'adx_14':           sv(r0['adx_14']),
                'hurst':            sv(r0['hurst']),
                'rvol':             sv(r0['rvol'],1.0),
                'pre_close_vs_mid': (sv(pre_last['close'])-(pre_hi+pre_lo)/2)/pre_rng if pre_rng>0 else 0,
                'pre_delta_trend':  sv(pre_last['cum_delta'])*dir_num/max(abs(sv(pre_last['cum_delta'])),1),
                'fvg_aligned':      sv(disp_bar['fvg_up']) if direction=='LONG' else sv(disp_bar['fvg_down']),
                'fisher_transform': sv(r0['fisher_transform'])*dir_num,
                'acf_lag1':         sv(r0['acf_lag1']),
                'dist_asia_hi_atr': (sv(r0['asia_hi'])-entry_price)/atr if sv(r0['asia_hi'])>0 else 0,
                'dist_asia_lo_atr': (entry_price-sv(r0['asia_lo']))/atr if sv(r0['asia_lo'])>0 else 0,
            }

            # ── Score ML ──────────────────────────────────────────────────────
            feats_model = pkg['features']
            row_df = pd.DataFrame([{f: feat.get(f, 0.0) for f in feats_model}]).fillna(0).astype(float)
            score  = float(pkg['model'].predict_proba(row_df)[0, 1])
            threshold = pkg.get('threshold', 0.65)

            if score < threshold:
                continue

            if score > best_score:
                best_score = score
                # SL/TP: SL 12pt, TP 24pt (2R) pentru DSM
                sl_pt = 12.0
                tp_pt = 24.0
                sl = entry_price - sl_pt if direction == 'LONG' else entry_price + sl_pt
                tp = entry_price + tp_pt if direction == 'LONG' else entry_price - tp_pt
                best_setup = {
                    'direction':  direction,
                    'setup_type': f'DSM_{active_sess}',
                    'session':    active_sess,
                    'entry':      round(entry_price, 2),
                    'sl':         round(sl, 2),
                    'sl_pt':      sl_pt,
                    'tp':         round(tp, 2),
                    'tp_pt':      tp_pt,
                    'rr':         round(tp_pt / sl_pt, 2),
                    'dsm_score':  round(score, 4),
                    'ml_score':   round(score, 4),
                    'action':     f"{'LONG' if direction=='LONG' else 'SHORT'}_DSM",
                    'message':    f"🌊 DSM_{active_sess}_{direction} | sweep+disp | score={score:.3f} | entry={entry_price:.1f} sl={sl:.1f} tp={tp:.1f}",
                }

    if best_setup is None:
        return None

    # ── Staleness check: dacă prețul curent e deja >1.5×ATR față de entry → setup expirat ──
    # (sweep la 09:30, entry la 09:32, dar checker rulează la 10:43 → nu mai intrăm în premium)
    try:
        last_bar = sess.iloc[-1]
        current_close = float(last_bar['close']) if 'close' in last_bar else 0.0
        if current_close > 0 and atr > 0:
            dist_from_entry = current_close - best_setup['entry']
            if best_setup['direction'] == 'LONG' and dist_from_entry > 1.5 * atr:
                logger.info(f"DSM: setup expirat — preț curent {current_close:.1f} este {dist_from_entry:.1f}pt "
                            f"({dist_from_entry/atr:.1f}×ATR) peste entry {best_setup['entry']:.1f} → SKIP")
                return None
            elif best_setup['direction'] == 'SHORT' and dist_from_entry < -1.5 * atr:
                logger.info(f"DSM: setup expirat — preț curent {current_close:.1f} este {abs(dist_from_entry):.1f}pt "
                            f"({abs(dist_from_entry)/atr:.1f}×ATR) sub entry {best_setup['entry']:.1f} → SKIP")
                return None
    except Exception:
        pass

    logger.info(best_setup['message'])
    return best_setup
