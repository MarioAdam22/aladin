"""
Aladin ORH 10-Year Backtest
═══════════════════════════
Testează strategia Opening Range Breakout pe toate zilele London + NY
din market_data, folosind EXACT risk management-ul din bridge_api.py:
  • SL inițial 20 pts (scalping default), scalat cu ATR ratio
  • 0.5R milestone → SL la BE
  • 0.85R milestone → SL la +0.5R profit + TRAILING ATR ACTIV
  • 1R/1.5R/2R/2.5R/3R → trail progresiv (ATR × 0.70/0.62/0.55/0.60/0.65)
  • Max hold 45 min per trade
  • Entry: primul close peste ORH (LONG) sau sub ORL (SHORT) post-OR

Killzones (Romania time):
  London Open: 09:00-11:00  (OR = 09:00-09:30)
  NY    Open: 15:30-17:30   (OR = 15:30-16:00)

Output: stats per an + total 11 ani
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

DB = Path(__file__).parent / "mario_trading.db"

# ── Risk Management params ────────────────────────────────────────────────
SL_BASE_PTS   = 20.0       # scalping_london/ny default
ATR_REF       = 9.0        # ATR normal NQ (pentru scaling SL)
MILESTONE_05R = 0.50       # → BE (protejare capital)

# ── "LET IT RUN" RM — trailing larg, lasă mișcarea să se dezvolte ────────
# Filozofie: NQ face mișcări de 30-60 pts după OR breakout valid.
# Nu trailăm agresiv devreme — lasăm tradeul să respire.
# 0.5R → SL la BE. 1R → SL la +0.5R (MILESTONE_1R). Trail pornește la 2R.
# La 3R strângem la 1.2×ATR, la 4R la 1.0×ATR, la 5R la 0.8×ATR.
TRAIL_START_R  = 2.0       # trailing pornește abia la 2R (era 0.85R)
TRAIL_ATR_PCT  = {2.0: 1.5, 3.0: 1.2, 4.0: 1.0, 5.0: 0.8}
TRAIL_RISK_PCT = {2.0: 0.60, 3.0: 0.50, 4.0: 0.40, 5.0: 0.30}
MAX_HOLD_MIN   = 120       # 2 ore — lasă mișcarea să se dezvolte (era 45)

# ── Milestone 1R → SL la +0.5R ────────────────────────────────────────────
# True  = la 1R SL urcă la entry + 0.5R (worst case după 1R = +0.5R)
# False = SL rămâne la BE până la 2R (50% BE exits, mai mult spațiu)
# ATENȚIE: NQ face pullback-uri normale de 10-15 pts (1R→0.5R) înainte să continue.
# Cu True, tăiem winners mari care intră temporar sub +0.5R. Recomandat: False.
MILESTONE_1R_LOCK = False  # dezactivat — losers rămân BE, winners continuă

# ── PARTIAL EXITS (mirror mario_rag.py / bridge_api.py exact) ───────────────
USE_PARTIAL_EXITS = True
PARTIAL_1R_FRAC   = 0.50   # exit 50% at +1R, move remainder to BE
PARTIAL_2R_FRAC   = 0.25   # exit 25% more at +2R, trail remaining 25%

# ── v14 ENTRY FILTERS (configurable via env / code) ──
NY_ONLY               = False   # True = skip LON killzone
SWEEP_DEPTH_MIN_ATR   = 0.0     # require sweep depth ≥ X × ATR (0 = off)
REQUIRE_FVG_INTACT    = False   # require fvg_up/fvg_down intact at entry
REQUIRE_MSS           = False   # require MSS/CHoCH confirmation
BREAK_ONLY            = False   # True = skip REV classes (3,4) — BREAK directional only

# Killzone config — ore CET (UTC+1 iarna, UTC+2 vara) = ora DB
# LON Open Romania = 10:30 → CET 09:30. OR = 09:30-10:00 CET = 10:30-11:00 Romania
# NY  Open Romania = 16:30 → CET 15:30. OR = 15:30-16:00 CET = 16:30-17:00 Romania
KZ = {
    "LON": (9.50, 11.50, 30),   # 09:30-11:30 CET = 10:30-12:30 Romania
    "NY":  (15.50, 17.50, 30),  # 15:30-17:30 CET = 16:30-18:30 Romania
}


def load_bars_for_year(year: int) -> pd.DataFrame:
    """Încarcă barele pentru un singur an (reduce RAM: 3.9M → ~50MB/an cu float64).
    Include Dec an-1 (context structural ATR/VP) și Ian an+1 (finalizare trades Dec).
    """
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        f"SELECT timestamp, open, high, low, close, volume, atr_14,"
        f"       val, vah, poc_level,"
        f"       h1_lo, h1_hi, h4_lo, h4_hi,"
        f"       lw_lo, lw_hi "
        f"FROM market_data "
        f"WHERE timestamp >= '{year-1}-12-01' AND timestamp < '{year+1}-02-01' "
        f"ORDER BY timestamp",
        conn,
    )
    conn.close()
    # Păstrăm float64 pentru OHLCV+ATR (float32 introduce erori ~0.03 pts la ~20000)
    for c in ['open','high','low','close','volume','atr_14',
              'val','vah','poc_level','h1_lo','h1_hi','h4_lo','h4_hi','lw_lo','lw_hi']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['ts'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['ts']).reset_index(drop=True)
    df['date'] = df['ts'].dt.date
    df['hour_dec'] = df['ts'].dt.hour + df['ts'].dt.minute / 60.0
    df['atr_14'] = df['atr_14'].fillna(ATR_REF).replace(0, float(ATR_REF))
    for col in ['val','vah','poc_level','h1_lo','h1_hi','h4_lo','h4_hi','lw_lo','lw_hi']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


def load_bars() -> pd.DataFrame:
    """DEPRECATED — folosit doar dacă RAM permite (3.9M rows). Preferă load_bars_for_year."""
    print(f"📂 Loading market_data from {DB}...")
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, atr_14,"
        "       val, vah, poc_level,"
        "       h1_lo, h1_hi, h4_lo, h4_hi,"
        "       lw_lo, lw_hi "
        "FROM market_data ORDER BY timestamp",
        conn,
    )
    conn.close()
    float_cols = ['open','high','low','close','atr_14',
                  'val','vah','poc_level','h1_lo','h1_hi','h4_lo','h4_hi','lw_lo','lw_hi']
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    df['ts'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['ts']).reset_index(drop=True)
    df['date'] = df['ts'].dt.date
    df['hour_dec'] = df['ts'].dt.hour + df['ts'].dt.minute / 60.0
    df['atr_14'] = df['atr_14'].fillna(ATR_REF).replace(0, float(ATR_REF))
    # Fill structural level nulls cu 0 (filtrate la calcul SL)
    for col in ['val','vah','poc_level','h1_lo','h1_hi','h4_lo','h4_hi','lw_lo','lw_hi']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    print(f"   ✓ {len(df):,} bars | {df['ts'].min()} → {df['ts'].max()}")
    return df


def compute_trail_pts(atr_now: float, risk: float, r_level: float) -> float:
    """Trail point distance at given R level.
    'Let it run' RM: trail larg la 2R (1.5×ATR), strânge progresiv la 3-4-5R.
    """
    pct_risk = TRAIL_RISK_PCT.get(r_level, 0.30)
    pct_atr  = TRAIL_ATR_PCT.get(r_level, 0.80)
    return max(round(risk * pct_risk, 2), round(atr_now * pct_atr, 1))


def compute_structural_sl(direction: str, entry_px: float, atr_now: float,
                           h1_lo: float, h1_hi: float,
                           h4_lo: float, h4_hi: float,
                           asia_lo: float, asia_hi: float,
                           val: float, vah: float,
                           lw_lo: float, lw_hi: float) -> tuple:
    """
    SL structural — exact logica din mario_rag.py compute_sl_tp():
      LONG: SL = max candidat structural sub close (h1_lo, h4_lo, asia_lo, val, lw_lo)
      SHORT: SL = min candidat structural deasupra close (h1_hi, h4_hi, asia_hi, vah, lw_hi)
    SL_MIN = max(8, ATR×0.55) | SL_MAX = min(20, ATR×1.20) | fallback = ATR×0.85

    Returns: (sl_price, risk_pts)
    """
    _HARD_CAP = 20.0
    SL_MIN     = max(8.0,        round(atr_now * 0.55, 1))
    SL_DEFAULT = min(_HARD_CAP - 1, max(SL_MIN + 2.0, round(atr_now * 0.85, 1)))
    SL_MAX     = min(_HARD_CAP,  round(atr_now * 1.20, 1))
    SL_MAX     = max(SL_MAX, SL_MIN + 1)
    SL_DEFAULT = min(SL_DEFAULT, SL_MAX)

    if direction == "LONG":
        candidates = [x for x in [h1_lo, h4_lo, asia_lo, val, lw_lo]
                      if 0 < x < entry_px - SL_MIN]
        if candidates:
            sl_px = max(candidates)          # cel mai aproape de close
            dist  = entry_px - sl_px
            if dist < SL_MIN: sl_px = entry_px - SL_MIN
            if dist > SL_MAX: sl_px = entry_px - SL_MAX
        else:
            sl_px = entry_px - SL_DEFAULT
    else:  # SHORT
        candidates = [x for x in [h1_hi, h4_hi, asia_hi, vah, lw_hi]
                      if x > entry_px + SL_MIN]
        if candidates:
            sl_px = min(candidates)          # cel mai aproape de close
            dist  = sl_px - entry_px
            if dist < SL_MIN: sl_px = entry_px + SL_MIN
            if dist > SL_MAX: sl_px = entry_px + SL_MAX
        else:
            sl_px = entry_px + SL_DEFAULT

    risk = abs(entry_px - sl_px)
    risk = max(SL_MIN, min(risk, SL_MAX))    # clamp final de siguranță
    # Recalculăm sl_px din risk clamped (evitare floating point drift)
    sl_px = entry_px - risk if direction == "LONG" else entry_px + risk
    return round(sl_px, 2), round(risk, 2)


def simulate_trade_partial(bars: pd.DataFrame, entry_idx: int, direction: str,
                           entry_px: float, atr_now: float,
                           sl_explicit: float = None, risk_explicit: float = None) -> dict:
    """
    RM exact mirror mario_rag.py:
      • 0.5R  → SL la BE
      • 0.85R → SL la +0.5R profit (lock)
      • 1R    → exit 50% partial; trailing neagresiv pornit (TRAIL[1.0])
      • 1.5R  → trailing mai agresiv (TRAIL[1.5])
      • 2R    → exit 25% partial; lock SL la +1R; trailing ft agresiv (TRAIL[2.0])
      • 2.5R  → trailing ft agresiv (TRAIL[2.5])
      • 3R    → trailing ft agresiv (TRAIL[3.0])
    Worst case după TP1R: +0.0R (BE). Worst case după TP2R: +0.5R weighted.

    sl_explicit / risk_explicit: pentru v14 ICT SL la ORL/ORH (override ATR default)
    """
    if sl_explicit is not None and risk_explicit is not None:
        sl   = sl_explicit
        risk = risk_explicit
    else:
        atr_ratio = atr_now / ATR_REF
        risk = SL_BASE_PTS * atr_ratio
        risk = max(10.0, min(risk, 50.0))
        sl = entry_px - risk if direction == "LONG" else entry_px + risk
    trail_r = 0.0
    mae = 0.0; mfe = 0.0

    # Partial tracking
    remaining_frac = 1.0
    realized_pts_weighted = 0.0
    hit_05R  = False
    hit_085R = False
    hit_1R   = False
    hit_2R   = False
    reason_parts = []

    max_bars = MAX_HOLD_MIN
    end_idx = min(entry_idx + max_bars, len(bars) - 1)

    for j in range(entry_idx + 1, end_idx + 1):
        bar = bars.iloc[j]
        h, l, c = bar['high'], bar['low'], bar['close']
        bar_atr = bar['atr_14'] if bar['atr_14'] > 0 else atr_now

        if direction == "LONG":
            fav_extreme = h - entry_px
            adv_extreme = entry_px - l
        else:
            fav_extreme = entry_px - l
            adv_extreme = h - entry_px
        mfe = max(mfe, fav_extreme)
        mae = max(mae, adv_extreme)

        # R progress (intrabar peak, optimist pentru milestones)
        if direction == "LONG":
            r_peak = (h - entry_px) / risk
        else:
            r_peak = (entry_px - l) / risk

        # ── Milestone SL moves (nu ies din pozitie, doar muta SL) ──────────

        # 0.5R → SL la BE
        if not hit_05R and r_peak >= 0.50:
            hit_05R = True
            be_px = entry_px
            if (direction == "LONG" and be_px > sl) or (direction == "SHORT" and be_px < sl):
                sl = be_px
            trail_r = 0.5

        # 0.85R → SL la +0.5R profit
        if not hit_085R and r_peak >= 0.85:
            hit_085R = True
            lock_px = entry_px + 0.5 * risk if direction == "LONG" else entry_px - 0.5 * risk
            if (direction == "LONG" and lock_px > sl) or (direction == "SHORT" and lock_px < sl):
                sl = lock_px
            trail_r = 0.85

        # ── Partial exits ───────────────────────────────────────────────────

        # +1R partial (50%)
        if not hit_1R and r_peak >= 1.0 and remaining_frac > 0:
            hit_1R = True
            tp1_px = entry_px + risk if direction == "LONG" else entry_px - risk
            pts_leg = (tp1_px - entry_px) if direction == "LONG" else (entry_px - tp1_px)
            realized_pts_weighted += pts_leg * PARTIAL_1R_FRAC
            remaining_frac -= PARTIAL_1R_FRAC
            # SL cel putin la BE pentru rest
            be_px = entry_px
            if (direction == "LONG" and be_px > sl) or (direction == "SHORT" and be_px < sl):
                sl = be_px
            trail_r = 1.0
            reason_parts.append("TP1R")

        # +2R partial (25%)
        if hit_1R and not hit_2R and r_peak >= 2.0 and remaining_frac > 0:
            hit_2R = True
            tp2_px = entry_px + 2*risk if direction == "LONG" else entry_px - 2*risk
            pts_leg = (tp2_px - entry_px) if direction == "LONG" else (entry_px - tp2_px)
            realized_pts_weighted += pts_leg * PARTIAL_2R_FRAC
            remaining_frac -= PARTIAL_2R_FRAC
            # Lock SL la +1R pentru rest
            lock1r = entry_px + risk if direction == "LONG" else entry_px - risk
            if (direction == "LONG" and lock1r > sl) or (direction == "SHORT" and lock1r < sl):
                sl = lock1r
            trail_r = 2.0
            reason_parts.append("TP2R")

        if remaining_frac <= 0.001:
            total_pts = realized_pts_weighted
            return {
                'exit_idx': j,
                'exit_px': entry_px + total_pts if direction == "LONG" else entry_px - total_pts,
                'pts': total_pts, 'r_mult': total_pts / risk,
                'reason': '+'.join(reason_parts) or 'PARTIAL',
                'mae': mae, 'mfe': mfe, 'risk': risk,
                'bars_held': j - entry_idx,
            }

        # ── SL check pe rest ────────────────────────────────────────────────
        if direction == "LONG":
            if l <= sl:
                exit_px = sl
                pts_leg = exit_px - entry_px
                total_pts = realized_pts_weighted + pts_leg * remaining_frac
                return {
                    'exit_idx': j, 'exit_px': exit_px,
                    'pts': total_pts, 'r_mult': total_pts / risk,
                    'reason': ('+'.join(reason_parts) + '+' if reason_parts else '') + _reason_from_trail(trail_r),
                    'mae': mae, 'mfe': mfe, 'risk': risk,
                    'bars_held': j - entry_idx,
                }
        else:
            if h >= sl:
                exit_px = sl
                pts_leg = entry_px - exit_px
                total_pts = realized_pts_weighted + pts_leg * remaining_frac
                return {
                    'exit_idx': j, 'exit_px': exit_px,
                    'pts': total_pts, 'r_mult': total_pts / risk,
                    'reason': ('+'.join(reason_parts) + '+' if reason_parts else '') + _reason_from_trail(trail_r),
                    'mae': mae, 'mfe': mfe, 'risk': risk,
                    'bars_held': j - entry_idx,
                }

        # ── Trailing ATR progresiv (pornit după TP1R) ───────────────────────
        # 1R-1.5R: neagresiv | 1.5R-2R: mai agresiv | 2R+: ft agresiv
        if hit_1R:
            for r_lvl in (3.0, 2.5, 2.0, 1.5, 1.0):
                if r_peak >= r_lvl:
                    trail_pts = compute_trail_pts(bar_atr, risk, r_lvl)
                    if direction == "LONG":
                        new_trail = c - trail_pts
                        if new_trail > sl:
                            sl = new_trail
                    else:
                        new_trail = c + trail_pts
                        if new_trail < sl:
                            sl = new_trail
                    trail_r = r_lvl
                    break

    # MAX HOLD — inchide rest la close-ul ultimului bar
    final_bar = bars.iloc[end_idx]
    exit_px = final_bar['close']
    pts_leg = (exit_px - entry_px) if direction == "LONG" else (entry_px - exit_px)
    total_pts = realized_pts_weighted + pts_leg * remaining_frac
    return {
        'exit_idx': end_idx, 'exit_px': exit_px,
        'pts': total_pts, 'r_mult': total_pts / risk,
        'reason': ('+'.join(reason_parts) + '+' if reason_parts else '') + 'MAX_HOLD',
        'mae': mae, 'mfe': mfe, 'risk': risk,
        'bars_held': end_idx - entry_idx,
    }


def simulate_trade(bars: pd.DataFrame, entry_idx: int, direction: str,
                    entry_px: float, atr_now: float,
                    sl_explicit: float = None, risk_explicit: float = None) -> dict:
    """
    Simulează un trade cu exact risk management-ul din bridge_api.py.
    Dacă USE_PARTIAL_EXITS=True delegăm la simulate_trade_partial (50/25/25).
    sl_explicit/risk_explicit: override structural SL (v14 VP-based)
    """
    if USE_PARTIAL_EXITS:
        return simulate_trade_partial(bars, entry_idx, direction, entry_px, atr_now,
                                      sl_explicit=sl_explicit, risk_explicit=risk_explicit)

    # ── "Let it run" RM ──────────────────────────────────────────────────────
    # SL structural dacă e explicit (v14 ICT SL), altfel ATR-based
    if sl_explicit is not None and risk_explicit is not None:
        risk = risk_explicit
        sl   = sl_explicit
    else:
        atr_ratio = atr_now / ATR_REF
        risk = SL_BASE_PTS * atr_ratio
        risk = max(10.0, min(risk, 50.0))
        sl = entry_px - risk if direction == "LONG" else entry_px + risk

    trail_r      = 0.0
    milestone_05 = False   # 0.5R → BE
    milestone_1r = False   # 1R  → +0.5R (dacă MILESTONE_1R_LOCK=True)
    trail_active = False   # trailing pornit abia la TRAIL_START_R

    mae = 0.0
    mfe = 0.0

    max_bars = MAX_HOLD_MIN
    end_idx = min(entry_idx + max_bars, len(bars) - 1)

    for j in range(entry_idx + 1, end_idx + 1):
        bar = bars.iloc[j]
        h, l, c = bar['high'], bar['low'], bar['close']
        bar_atr = bar['atr_14'] if bar['atr_14'] > 0 else atr_now

        # Update MFE/MAE
        if direction == "LONG":
            fav_extreme = h - entry_px
            adv_extreme = entry_px - l
        else:
            fav_extreme = entry_px - l
            adv_extreme = h - entry_px
        mfe = max(mfe, fav_extreme)
        mae = max(mae, adv_extreme)

        # === SL hit check (intrabar, pessimist: SL e hit înainte de milestone) ===
        # Pentru long: dacă low ≤ sl → SL hit
        # Pentru short: dacă high ≥ sl → SL hit
        if direction == "LONG":
            if l <= sl:
                exit_px = sl
                pts = exit_px - entry_px
                return {
                    'exit_idx': j, 'exit_px': exit_px, 'pts': pts,
                    'r_mult': pts / risk, 'reason': _reason_from_trail(trail_r),
                    'mae': mae, 'mfe': mfe, 'risk': risk,
                    'bars_held': j - entry_idx,
                }
        else:  # SHORT
            if h >= sl:
                exit_px = sl
                pts = entry_px - exit_px
                return {
                    'exit_idx': j, 'exit_px': exit_px, 'pts': pts,
                    'r_mult': pts / risk, 'reason': _reason_from_trail(trail_r),
                    'mae': mae, 'mfe': mfe, 'risk': risk,
                    'bars_held': j - entry_idx,
                }

        # === Milestone checks ===
        if direction == "LONG":
            prog = (h - entry_px) / risk   # R peak pe bara curentă
        else:
            prog = (entry_px - l) / risk

        # 0.5R → SL la BE
        if not milestone_05 and prog >= MILESTONE_05R:
            milestone_05 = True
            be_px = entry_px
            if (direction == "LONG" and be_px > sl) or (direction == "SHORT" and be_px < sl):
                sl = be_px
            trail_r = 0.5

        # 1R → SL la +0.5R (opțional, reduce BE exits)
        # Logică: trade-ul a dovedit putere la 1R → garantăm minim +0.5R
        # Winners mari (3R+): SL la +0.5R e oricum depăşit de trailing → fără impact
        if MILESTONE_1R_LOCK and not milestone_1r and prog >= 1.0:
            milestone_1r = True
            lock_05r = entry_px + 0.5 * risk if direction == "LONG" else entry_px - 0.5 * risk
            if (direction == "LONG" and lock_05r > sl) or (direction == "SHORT" and lock_05r < sl):
                sl = lock_05r
            trail_r = 1.0

        # TRAIL_START_R (2R) → pornește trailing larg
        # Între 0.5R și 2R: SL la +0.5R (sau BE dacă MILESTONE_1R_LOCK=False)
        if not trail_active and prog >= TRAIL_START_R:
            trail_active = True

        # === Trailing larg activ după 2R =====================================
        if trail_active:
            r_now = prog
            # ordine descrescătoare — aplicăm cel mai strâns trail disponibil
            for r_lvl in sorted(TRAIL_ATR_PCT.keys(), reverse=True):
                if r_now >= r_lvl:
                    trail_pts = compute_trail_pts(bar_atr, risk, r_lvl)
                    if direction == "LONG":
                        new_trail = c - trail_pts
                        if new_trail > sl:
                            sl = new_trail
                    else:
                        new_trail = c + trail_pts
                        if new_trail < sl:
                            sl = new_trail
                    trail_r = r_lvl
                    break

    # === MAX HOLD → close la market (close ultimei bare) ===
    final_bar = bars.iloc[end_idx]
    exit_px = final_bar['close']
    pts = (exit_px - entry_px) if direction == "LONG" else (entry_px - exit_px)
    return {
        'exit_idx': end_idx, 'exit_px': exit_px, 'pts': pts,
        'r_mult': pts / risk, 'reason': 'MAX_HOLD',
        'mae': mae, 'mfe': mfe, 'risk': risk,
        'bars_held': end_idx - entry_idx,
    }


def _reason_from_trail(trail_r: float) -> str:
    if trail_r == 0:   return 'SL_INITIAL'
    if trail_r == 0.5: return 'BE'
    if trail_r == 1.0: return 'LOCK_05R'   # 1R milestone: SL la +0.5R
    return f'TRAIL_{trail_r}R'


def backtest_orh(bars: pd.DataFrame) -> pd.DataFrame:
    """Iterează pe fiecare zi × killzone, detectează OR și primul breakout."""
    trades = []

    bars_by_date = bars.groupby('date')
    total_days = len(bars_by_date)
    print(f"📅 Total zile în DB: {total_days}")

    processed = 0
    for date, day_df in bars_by_date:
        processed += 1
        if processed % 500 == 0:
            print(f"   ... {processed}/{total_days} zile procesate, {len(trades)} trades")

        for kz_name, (kz_start, kz_end, or_minutes) in KZ.items():
            or_end = kz_start + or_minutes / 60.0

            kz_mask = (day_df['hour_dec'] >= kz_start) & (day_df['hour_dec'] <= kz_end)
            kz_bars = day_df[kz_mask]
            if len(kz_bars) < or_minutes + 5:
                continue  # zi cu date incomplete

            or_mask = kz_bars['hour_dec'] < or_end
            or_bars = kz_bars[or_mask]
            post_or_bars = kz_bars[~or_mask]

            if len(or_bars) < 10 or len(post_or_bars) < 5:
                continue

            orh = or_bars['high'].max()
            orl = or_bars['low'].min()

            # Detectăm primul breakout (close confirm)
            entry_long_idx = None
            entry_short_idx = None
            for idx, row in post_or_bars.iterrows():
                if entry_long_idx is None and row['close'] > orh:
                    entry_long_idx = idx
                if entry_short_idx is None and row['close'] < orl:
                    entry_short_idx = idx
                if entry_long_idx is not None or entry_short_idx is not None:
                    # Tranzacționăm PRIMUL breakout (indiferent de direcție)
                    break

            # Alegem primul breakout care s-a întâmplat
            entry_idx = None
            direction = None
            if entry_long_idx is not None and entry_short_idx is not None:
                if entry_long_idx < entry_short_idx:
                    entry_idx, direction = entry_long_idx, "LONG"
                else:
                    entry_idx, direction = entry_short_idx, "SHORT"
            elif entry_long_idx is not None:
                entry_idx, direction = entry_long_idx, "LONG"
            elif entry_short_idx is not None:
                entry_idx, direction = entry_short_idx, "SHORT"

            if entry_idx is None:
                continue

            entry_bar = bars.loc[entry_idx]
            entry_px = entry_bar['close']
            atr_now = entry_bar['atr_14'] if entry_bar['atr_14'] > 0 else ATR_REF

            # Găsim poziția în array-ul global
            entry_pos = bars.index.get_loc(entry_idx)
            result = simulate_trade(bars, entry_pos, direction, entry_px, atr_now)

            trades.append({
                'date': date,
                'killzone': kz_name,
                'direction': direction,
                'entry_ts': entry_bar['ts'],
                'entry_px': entry_px,
                'orh': orh, 'orl': orl, 'or_width': orh - orl,
                'atr': atr_now,
                **result,
            })

    return pd.DataFrame(trades)


def report(trades_df: pd.DataFrame):
    if len(trades_df) == 0:
        print("❌ Niciun trade generat"); return

    print("\n" + "═" * 80)
    print("📊 BACKTEST ORH 10-YEAR — REZULTATE")
    print("═" * 80)

    trades_df['year'] = pd.to_datetime(trades_df['entry_ts']).dt.year
    trades_df['win'] = trades_df['pts'] > 0
    trades_df['pnl_usd'] = trades_df['pts'] * 20.0  # NQ $20/pt

    # Total
    tot = len(trades_df)
    wr = trades_df['win'].mean() * 100
    avg_pts = trades_df['pts'].mean()
    med_pts = trades_df['pts'].median()
    total_pts = trades_df['pts'].sum()
    win_pts = trades_df[trades_df['win']]['pts'].mean()
    loss_pts = trades_df[~trades_df['win']]['pts'].mean()

    # Max consecutive losses (global)
    cl_max, cl_cur = 0, 0
    for w in trades_df['win'].values:
        if not w:
            cl_cur += 1; cl_max = max(cl_max, cl_cur)
        else:
            cl_cur = 0

    # Max drawdown $ (cumulative)
    trades_df_sorted = trades_df.sort_values('entry_ts').reset_index(drop=True)
    cum = trades_df_sorted['pnl_usd'].cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min()

    print(f"\n🎯 TOTAL ({trades_df['year'].min()}-{trades_df['year'].max()})")
    print(f"   Trades:            {tot:,}")
    print(f"   Win Rate:          {wr:.1f}%")
    print(f"   Avg Win:           {win_pts:+.1f} pts")
    print(f"   Avg Loss:          {loss_pts:+.1f} pts")
    print(f"   Avg pts/trade:     {avg_pts:+.2f} pts  (median {med_pts:+.2f})")
    print(f"   Total pts:         {total_pts:+,.0f} pts")
    print(f"   Total P&L:         ${total_pts * 20:+,.0f}")
    print(f"   Max Consec Loss:   {cl_max}")
    print(f"   Max Drawdown:      ${max_dd:+,.0f}")
    if loss_pts != 0 and win_pts is not None:
        exp = wr/100 * win_pts + (1-wr/100) * loss_pts
        pf = (win_pts * trades_df['win'].sum()) / abs(loss_pts * (~trades_df['win']).sum())
        print(f"   Expectancy:        {exp:+.2f} pts/trade")
        print(f"   Profit Factor:     {pf:.2f}")

    # Per killzone
    print(f"\n🌍 PER KILLZONE:")
    for kz in ['LON', 'NY']:
        sub = trades_df[trades_df['killzone'] == kz]
        if len(sub):
            print(f"   {kz}: {len(sub):,} trades | WR {sub['win'].mean()*100:.1f}% | "
                  f"avg {sub['pts'].mean():+.2f} pts | total ${sub['pts'].sum()*20:+,.0f}")

    # Per direcție
    print(f"\n↕️  PER DIRECȚIE:")
    for d in ['LONG', 'SHORT']:
        sub = trades_df[trades_df['direction'] == d]
        if len(sub):
            print(f"   {d}:  {len(sub):,} | WR {sub['win'].mean()*100:.1f}% | "
                  f"avg {sub['pts'].mean():+.2f} pts")

    # Per exit reason
    print(f"\n🚪 EXIT REASONS:")
    for r, sub in trades_df.groupby('reason'):
        print(f"   {r:15s}: {len(sub):5,} ({len(sub)/tot*100:4.1f}%) | "
              f"avg {sub['pts'].mean():+6.2f} pts")

    # Per an
    print(f"\n📅 PER AN:")
    print(f"   {'Year':6s} {'Trades':>7s} {'WR':>6s} {'AvgPts':>8s} {'TotPts':>10s} {'MaxCL':>6s} {'MaxDD':>10s}")
    for y, sub in trades_df.groupby('year'):
        sub = sub.sort_values('entry_ts').reset_index(drop=True)
        sw = sub['win'].mean() * 100
        cl_y_max, cl_y_cur = 0, 0
        for w in sub['win'].values:
            if not w:
                cl_y_cur += 1; cl_y_max = max(cl_y_max, cl_y_cur)
            else: cl_y_cur = 0
        cum_y = sub['pnl_usd'].cumsum()
        dd_y = (cum_y - cum_y.cummax()).min()
        print(f"   {y}   {len(sub):>7,} {sw:>5.1f}% {sub['pts'].mean():>+7.2f}  "
              f"{sub['pts'].sum():>+9,.0f} {cl_y_max:>6d} ${dd_y:>+8,.0f}")

    # MFE/MAE distribution pe trades winning
    wins = trades_df[trades_df['win']]
    if len(wins):
        print(f"\n🔥 MFE DISTRIBUTION (winning trades):")
        for pct in [50, 75, 90, 95, 99]:
            v = np.percentile(wins['mfe'], pct)
            print(f"   P{pct}: {v:6.1f} pts")
        big = (wins['mfe'] > 50).sum()
        huge = (wins['mfe'] > 100).sum()
        print(f"   Trades cu MFE > 50 pts:  {big:,} ({big/len(wins)*100:.1f}% din wins)")
        print(f"   Trades cu MFE > 100 pts: {huge:,} ({huge/len(wins)*100:.1f}% din wins)")

    print("═" * 80)


# =============================================================================
# v13 REGIME-AWARE BACKTEST — adds model filter + bridge layers
# =============================================================================
# Layers on top of naked ORH:
#   1. v13 9-class regime filter (predict on entry bar; skip WAIT / wrong direction)
#   2. Adaptive score_min (CL0→0.50, CL1→0.65, CL2→0.80)
#   3. CL=3 hard stop → cooldown 180s (≈3 bars)
#   4. Daily cap: 3 trades per killzone (LON + NY = 6/day max)
#   5. Post-SL cooldown per killzone (3 min)
# =============================================================================
def _load_v13_model(kz: str = None):
    """
    Încarcă XGBoost model + feature meta.
    kz='LON' → mario_bot_lon.json (dacă există), fallback mario_bot.json
    kz='NY'  → mario_bot_ny.json  (dacă există), fallback mario_bot.json
    kz=None  → mario_bot.json (model unificat)
    """
    import os, json
    import xgboost as xgb
    _base = Path(__file__).parent
    # Caută model specializat pe killzone
    _candidates = []
    if kz == 'LON':
        _candidates = [(_base / "mario_bot_lon.json", _base / "mario_features_lon.json")]
    elif kz == 'NY':
        _candidates = [(_base / "mario_bot_ny.json",  _base / "mario_features_ny.json")]
    _candidates.append((_base / "mario_bot.json", _base / "mario_features.json"))

    for _model_path, _meta_path in _candidates:
        if not _model_path.exists() or not _meta_path.exists():
            continue
        try:
            m = xgb.XGBClassifier()
            m.load_model(str(_model_path))
            with open(_meta_path) as f:
                meta = json.load(f)
            feats = meta.get('features', [])
            thr = float(meta.get('conf_threshold', 0.50))
            print(f"   ✅ Model loaded: {_model_path.name} ({len(feats)} features, thr={thr:.2%})")
            return m, feats, thr
        except Exception as e:
            print(f"   ⚠️ Model load failed ({_model_path.name}): {e}")
    return None, None, None


def backtest_orh_v13(bars: pd.DataFrame, enable_v13: bool = True) -> pd.DataFrame:
    """ORH backtest with v13 regime filter + bridge CL/adaptive/cooldown layers."""
    try:
        from aladin_v13 import (
            add_weekly_features, add_daytype_features, add_sweep_features,
            REGIME_NAMES,
        )
        # v14: 5-class map (1=SHORT_BREAK, 2=LONG_BREAK, 3=SHORT_REV, 4=LONG_REV)
        # Păstrăm și 9-class fallback pentru modele vechi.
        V13_DIR = {0: 0, 1: -1, 2: +1, 3: -1, 4: +1,
                   5: -1, 6: +1, 7: -1, 8: +1}  # legacy 9-class aliases
    except ImportError:
        print("   ⚠️ aladin_v13 indisponibil — fallback naked ORH")
        return backtest_orh(bars)

    # Încearcă să încarce modele specializate per killzone (LON + NY separate)
    # Dacă nu există → fallback la modelul unificat
    models = {}   # {'LON': (model, feats, thr), 'NY': (model, feats, thr)}
    model_unified, feats_unified, thr_unified = (None, None, 0.50)

    if enable_v13:
        _lon_model, _lon_feats, _lon_thr = _load_v13_model('LON')
        _ny_model,  _ny_feats,  _ny_thr  = _load_v13_model('NY')
        if _lon_model is not None:
            models['LON'] = (_lon_model, _lon_feats, _lon_thr)
        if _ny_model is not None:
            models['NY'] = (_ny_model, _ny_feats, _ny_thr)
        if not models:
            # Niciun model specializat — încearcă modelul unificat
            model_unified, feats_unified, thr_unified = _load_v13_model(None)
            if model_unified is None:
                print("   ⚠️ Niciun model găsit — rulez naked ORH")
                return backtest_orh(bars)
            models['LON'] = (model_unified, feats_unified, thr_unified)
            models['NY']  = (model_unified, feats_unified, thr_unified)
            print("   ℹ️ Modele specializate LON/NY absente — folosesc model unificat pentru ambele")
        conf_thr_base = min(t for _, _, t in models.values())

    # Pre-compute v13 + ORH + session time features ONE TIME pe întreg dataset-ul
    print("🧮 v13: computing ORH + session time + weekly + daytype + sweep features...")
    df = bars.copy()
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from train_mario_ai import add_orh_features as _add_orh, \
                                   add_session_time_features as _add_stime
        df = _add_orh(df)
        df = _add_stime(df)
        print(f"   ✅ ORH + session time features adăugate")
    except Exception as _orh_err:
        print(f"   ⚠️ Feature compute warning: {_orh_err}")
    try:
        df = add_weekly_features(df)
        df = add_daytype_features(df)
        df = add_sweep_features(df)
    except Exception as e:
        print(f"   ⚠️ v13 feature compute failed: {e}")
        return backtest_orh(bars)

    # ── FIX BUG #1: flag-uri CUMULATIVE per zi pentru sweep ──
    print("   🔧 building cumulative sweep flags per session...")
    for _col in ['broke_asia_hi', 'broke_asia_lo', 'broke_lon_hi', 'broke_lon_lo',
                 'broke_pdh', 'broke_pdl']:
        if _col in df.columns:
            df[f'cum_{_col}'] = df.groupby('date')[_col].cummax().astype(int)

    # ── BATCH PREDICT per killzone (folosește modelul specializat dacă există) ──
    all_proba_per_kz = {}   # {'LON': (proba_array, pos_lookup), 'NY': (...)}
    pos_lookup_global = {orig_idx: i for i, orig_idx in enumerate(df.index)}

    for _kz_name, (_kz_model, _kz_feats, _kz_thr) in models.items():
        print(f"🧮 Batch predict {_kz_name}: {len(df):,} bare × {len(_kz_feats)} features...")
        import time as _t
        _t0 = _t.time()
        _X = df.reindex(columns=_kz_feats, fill_value=0).fillna(0)
        _proba = _kz_model.predict_proba(_X)
        all_proba_per_kz[_kz_name] = (_proba, pos_lookup_global)
        print(f"   ✅ {_kz_name}: {_proba.shape[0]:,} × {_proba.shape[1]} clase în {_t.time()-_t0:.1f}s")

    # Fallback compatibility — all_proba pentru diagnostic
    _first_kz = list(all_proba_per_kz.keys())[0]
    all_proba    = all_proba_per_kz[_first_kz][0]
    pos_lookup   = pos_lookup_global
    model_feats  = models[_first_kz][1]

    # ── DIAGNOSTIC: distribuție argmax DOAR pe bare post_orh ──
    try:
        _po_mask = (df.get('post_orh', pd.Series(0, index=df.index)).values == 1)
        _po_count = int(_po_mask.sum())
        if _po_count > 0:
            _proba_po = all_proba[_po_mask]
            _argmax = _proba_po.argmax(axis=1)
            _maxconf = _proba_po.max(axis=1)
            print(f"   📊 Diagnostic post_orh ({_po_count:,} bare):")
            for _c in range(_proba_po.shape[1]):
                _n_c = int((_argmax == _c).sum())
                if _n_c > 0:
                    _mean_conf = float(_maxconf[_argmax == _c].mean())
                    _pct_50 = int(((_argmax == _c) & (_maxconf >= 0.50)).sum())
                    _pct_40 = int(((_argmax == _c) & (_maxconf >= 0.40)).sum())
                    _pct_30 = int(((_argmax == _c) & (_maxconf >= 0.30)).sum())
                    print(f"      class {_c}: {_n_c:>8,} ({100*_n_c/_po_count:5.2f}%) | "
                          f"mean_conf={_mean_conf:.3f} | ≥0.30: {_pct_30:,} | ≥0.40: {_pct_40:,} | ≥0.50: {_pct_50:,}")
    except Exception as _dx:
        print(f"   ⚠️ diagnostic failed: {_dx}")

    trades = []
    # Per-killzone state (reset daily)
    state = {}  # date → {kz: {'cl': 0, 'cooldown_until_ts': None, 'trades_today': 0}}

    bars_by_date = df.groupby('date')
    total_days = len(bars_by_date)
    print(f"📅 v13 backtest: {total_days} zile")

    processed = 0
    for date, day_df in bars_by_date:
        processed += 1
        if processed % 500 == 0:
            print(f"   ... {processed}/{total_days} zile, {len(trades)} trades (v13)")

        state_day = {kz: {'cl': 0, 'cooldown_until_ts': None, 'trades_today': 0} for kz in KZ}

        for kz_name, (kz_start, kz_end, or_minutes) in KZ.items():
            if NY_ONLY and kz_name != "NY":
                continue
            or_end = kz_start + or_minutes / 60.0
            kz_mask = (day_df['hour_dec'] >= kz_start) & (day_df['hour_dec'] <= kz_end)
            kz_bars = day_df[kz_mask]
            if len(kz_bars) < or_minutes + 5:
                continue
            or_bars = kz_bars[kz_bars['hour_dec'] < or_end]
            post_or_bars = kz_bars[kz_bars['hour_dec'] >= or_end]
            if len(or_bars) < 10 or len(post_or_bars) < 5:
                continue

            orh = or_bars['high'].max()
            orl = or_bars['low'].min()

            st = state_day[kz_name]

            # v13 FILTERS (quality > quantity) — QA'd:
            # FIX #2: NU commitem la primul breakout; încercăm bare până găsim una care trece
            # FIX #3: WHITELIST extins (adăugăm 1,2 EXPANSION cu thr separat)
            # FIX #4: HIGH_THR=conf_thr_base (floor 0.50) — CL escalation rămâne
            # Sweep: cumulative flags per zi (broke-ul s-a întâmplat ÎNAINTE de breakout)
            # Per-class thresholds calibrate pe diagnostic (vezi output batch predict):
            #  - class 6 (REV_LOW→LONG) e semnalul real: 5,919 bare ≥0.50, mean 0.408
            #  - class 5 (REV_HIGH→SHORT) rar dar curat: 151 bare ≥0.40
            #  - class 1 (SHORT_EXP) noisy: mean 0.333 → cerem conf mare (0.45)
            #  - class 2 (LONG_EXP) rar dar e prima clasă LONG după 6
            #  - class 3,4,7,8 skip (model nu le prezice)
            # Clase cu sweep requirement (reversal only)
            # Selectează modelul și proba array pentru killzone-ul curent
            _kz_proba, _kz_pos_lookup = all_proba_per_kz.get(
                kz_name, all_proba_per_kz.get(_first_kz, (all_proba, pos_lookup_global))
            )
            _kz_model_thr = models.get(kz_name, models[_first_kz])[2]

            # v14: 5-class model (1=SHORT_BREAK, 2=LONG_BREAK, 3=SHORT_REV, 4=LONG_REV).
            # Auto-detect pe baza shape-ului de predict.
            _n_proba_classes = _kz_proba.shape[1] if _kz_proba is not None else 5
            if _n_proba_classes == 5:
                CLASS_THR = {
                    1: 0.40,   # SHORT_BREAK
                    2: 0.40,   # LONG_BREAK
                    3: 0.45,   # SHORT_REV — cere sweep high
                    4: 0.45,   # LONG_REV — cere sweep low
                }
                CLASS_NEED_SWEEP = {3, 4}
                _REV_LONG_CLS = {4}
                _REV_SHORT_CLS = {3}
            else:
                # Legacy 9-class fallback
                CLASS_THR = {
                    2: 0.40,   # LONG_EXPANSION
                    5: 0.45,   # REVERSAL_HIGH (SHORT)
                    6: 0.50,   # REVERSAL_LOW (LONG)
                }
                CLASS_NEED_SWEEP = {5, 6}
                _REV_LONG_CLS = {6}
                _REV_SHORT_CLS = {5}

            _dbg = {'no_break': 0, 'no_pos': 0, 'not_in_thr': 0, 'break_only': 0,
                    'dir_mismatch': 0, 'conf_low': 0, 'passed': 0}

            for idx, row in post_or_bars.iterrows():
                if st['trades_today'] >= 3:
                    break
                if st['cooldown_until_ts'] is not None and row['ts'] < st['cooldown_until_ts']:
                    continue

                # Detect breakout direction — close în afara OR
                breakout_dir = None
                if row['close'] > orh:
                    breakout_dir = "LONG"
                elif row['close'] < orl:
                    breakout_dir = "SHORT"
                if breakout_dir is None:
                    _dbg['no_break'] += 1
                    continue

                # ── Regime prediction ──
                _p_pos = _kz_pos_lookup.get(idx) if _kz_proba is not None else None
                if _p_pos is None:
                    _dbg['no_pos'] += 1
                    continue
                _bar = df.iloc[_p_pos]

                try:
                    proba = _kz_proba[_p_pos]
                    regime_cls = int(np.argmax(proba))
                    regime_conf = float(proba[regime_cls])
                    pred_dir = V13_DIR.get(regime_cls, 0)
                except Exception:
                    continue

                # Filtre per clasă
                if regime_cls not in CLASS_THR:
                    _dbg['not_in_thr'] += 1
                    continue
                if BREAK_ONLY and regime_cls in {3, 4}:
                    _dbg['break_only'] += 1
                    continue
                base_thr = CLASS_THR[regime_cls]

                # CL escalation
                if st['cl'] == 0:
                    thr = base_thr
                elif st['cl'] == 1:
                    thr = min(base_thr + 0.10, 0.85)
                else:
                    thr = min(base_thr + 0.20, 0.95)

                if pred_dir == 0:
                    _dbg['not_in_thr'] += 1
                    continue

                # Direcția trade-ului (din model)
                trade_dir = "LONG" if pred_dir == +1 else "SHORT"

                # Direction alignment:
                # BREAK (cls 1,2): urmăm breakout-ul → trade_dir == breakout_dir
                # REV   (cls 3,4): fadem fake breakout → trade_dir OPUS breakout_dir
                _is_rev = regime_cls in {3, 4, 5, 6, 7, 8}
                if _is_rev:
                    _opp = {"LONG": "SHORT", "SHORT": "LONG"}
                    if trade_dir != _opp.get(breakout_dir, ""):
                        _dbg['dir_mismatch'] += 1
                        continue
                else:
                    if trade_dir != breakout_dir:
                        _dbg['dir_mismatch'] += 1
                        continue

                if regime_conf < thr:
                    _dbg['conf_low'] += 1
                    continue

                _dbg['passed'] += 1

                entry_px = row['close']
                atr_now = row['atr_14'] if row['atr_14'] > 0 else ATR_REF
                entry_pos = bars.index.get_loc(idx)
                result = simulate_trade(bars, entry_pos, trade_dir, entry_px, atr_now)

                trades.append({
                    'date': date, 'killzone': kz_name, 'direction': trade_dir,
                    'entry_ts': row['ts'], 'entry_px': entry_px,
                    'orh': orh, 'orl': orl, 'or_width': orh - orl, 'atr': atr_now,
                    'regime_cls': regime_cls, 'regime_conf': regime_conf,
                    **result,
                })

                st['trades_today'] += 1

                # Update CL + cooldown per bridge rules
                pnl_pts = result.get('pts', 0.0)
                if pnl_pts < 0:
                    st['cl'] += 1
                    # Post-SL cooldown 3 min
                    st['cooldown_until_ts'] = row['ts'] + pd.Timedelta(minutes=3)
                    if st['cl'] >= 3:
                        # CL=3 hard stop restul zilei pe killzone
                        st['trades_today'] = 3
                        break
                else:
                    st['cl'] = 0

    # Debug summary
    print(f"\n🔍 DEBUG FILTER SUMMARY:")
    print(f"   no_break (inside OR):  {_dbg.get('no_break','?')}")
    print(f"   no_pos (idx missing):  {_dbg.get('no_pos','?')}")
    print(f"   not_in_CLASS_THR:      {_dbg.get('not_in_thr','?')}")
    print(f"   dir_mismatch:          {_dbg.get('dir_mismatch','?')}")
    print(f"   conf_low:              {_dbg.get('conf_low','?')}")
    print(f"   ✅ passed all filters: {_dbg.get('passed','?')}")

    return pd.DataFrame(trades)


# =============================================================================
# v14 BINARY BREAKOUT MODEL BACKTEST
# =============================================================================

def _load_breakout_model(kz: str):
    """
    Încarcă modelul breakout (v14.5 regresie sau v14 binary — autodetect din meta).
    Returnează: (model, feats, model_type, min_expected_r)
      model_type = "regression" | "binary"
      min_expected_r = threshold (R) sub care nu intrăm în trade
    """
    import xgboost as xgb, json
    _base = Path(__file__).parent
    _model_path = _base / f"mario_bot_breakout_{kz.lower()}.json"
    _meta_path  = _base / f"mario_bot_breakout_{kz.lower()}_meta.json"

    if not _model_path.exists():
        print(f"   ⚠️ Model breakout {kz} nu există: {_model_path}")
        print(f"   💡 Rulează: python3 train_breakout_model.py {kz.lower()}")
        return None, None, None, None

    try:
        meta = {}
        feats = []
        min_r = 0.20   # default threshold regresie

        if _meta_path.exists():
            with open(_meta_path) as f:
                meta = json.load(f)
            feats = meta.get('features', [])
            min_r = float(meta.get('min_expected_r', 0.20))

        model_type = meta.get('model_type', 'binary_breakout')
        # Regresie = model care prezice o valoare continuă (R)
        # Binar = model care prezice probabilitate (mfe≥threshold sau REAL/FAKE)
        is_regression = ('regression' in model_type) and ('binary' not in model_type)

        if is_regression:
            m = xgb.XGBRegressor()
            m.load_model(str(_model_path))
            print(f"   ✅ Breakout REGRESSOR {kz}: {_model_path.name} | "
                  f"MAE={meta.get('mae_test','?')} | R²={meta.get('r2_test','?')} | "
                  f"min_r={min_r} | {len(feats)} features")
        else:
            m = xgb.XGBClassifier()
            m.load_model(str(_model_path))
            label_info = meta.get('label', '')
            print(f"   ✅ Breakout CLASSIFIER {kz}: {_model_path.name} | "
                  f"AUC={meta.get('auc_test', meta.get('auc','?'))} | "
                  f"label={label_info[:40]} | thr={min_r} | {len(feats)} features")

        return m, feats, model_type, min_r

    except Exception as e:
        print(f"   ❌ Load failed {_model_path}: {e}")
        return None, None, None, None


def _brk_features_at_moment(
    df_global: pd.DataFrame,
    date, kz: dict, kz_name: str,
    or_bars: pd.DataFrame, post_or: pd.DataFrame,
    breakout_idx: int, breakout_dir: str,
    orh: float, orl: float, or_width: float, atr_ref: float
) -> dict:
    """
    Calculează features la momentul breakout-ului (adaptat din train_breakout_model.py).
    """
    brk = post_or.iloc[breakout_idx]

    # ── Candle structure ──
    bar_range = max(brk['high'] - brk['low'], 0.1)
    body      = abs(brk['close'] - brk['open'])
    body_pct  = body / bar_range
    up_wick   = (brk['high'] - max(brk['open'], brk['close'])) / bar_range
    lo_wick   = (min(brk['open'], brk['close']) - brk['low'])  / bar_range
    cl_str    = (brk['close'] - brk['low']) / bar_range
    is_strong = int(cl_str >= 0.75) if breakout_dir == "LONG" else int(cl_str <= 0.25)
    bar_range_atr = bar_range / atr_ref if atr_ref > 0 else 0.0

    # ── Volume ──
    or_vol_avg  = max(or_bars['volume'].mean(), 1.0)
    vol_vs_or   = float(np.clip(brk['volume'] / or_vol_avg, 0, 20))

    # Volume vs 20 bare anterioare în global df
    brk_ts = brk['ts']
    df_idx  = df_global[df_global['ts'] == brk_ts].index
    if len(df_idx) > 0:
        ai = df_idx[0]
        lb = df_global.loc[max(0, ai-20):ai-1, 'volume']
        vol_20avg = float(lb.mean()) if len(lb) > 0 else or_vol_avg
    else:
        vol_20avg = or_vol_avg
    vol_vs_20 = float(np.clip(brk['volume'] / max(vol_20avg, 1.0), 0, 20))

    # ── OR context ──
    or_width_atr = or_width / atr_ref if atr_ref > 0 else 0.0
    or_mid       = (orh + orl) / 2.0

    # ── Timing ──
    or_end_min = int(kz['or_end'] * 60)
    brk_min    = int(brk['minute_of_day']) if 'minute_of_day' in brk.index else (
        brk['ts'].hour * 60 + brk['ts'].minute)
    min_after  = max(0, brk_min - or_end_min)
    if   min_after <  5: time_bin = 0
    elif min_after < 15: time_bin = 1
    elif min_after < 30: time_bin = 2
    elif min_after < 45: time_bin = 3
    else:                time_bin = 4

    # ── Prior liquidity ──
    prev_days = df_global[df_global['date'] < date]
    if len(prev_days) > 0:
        prev_day = prev_days[prev_days['date'] == prev_days['date'].max()]
        pdh = float(prev_day['high'].max()) if len(prev_day) > 0 else np.nan
        pdl = float(prev_day['low'].min())  if len(prev_day) > 0 else np.nan
    else:
        pdh = pdl = np.nan

    today_bars = df_global[df_global['date'] == date]
    kz_start = kz.get('start', 9.0)
    asia_bars = today_bars[today_bars['hour_dec'].between(2.0, kz_start - 0.01)]
    asia_hi = float(asia_bars['high'].max()) if len(asia_bars) > 0 else np.nan
    asia_lo = float(asia_bars['low'].min())  if len(asia_bars) > 0 else np.nan

    or_hi_all = float(or_bars['high'].max())
    or_lo_all = float(or_bars['low'].min())

    broke_pdh = int(not np.isnan(pdh) and or_hi_all > pdh)
    broke_pdl = int(not np.isnan(pdl) and or_lo_all < pdl)
    broke_ahi = int(not np.isnan(asia_hi) and or_hi_all > asia_hi)
    broke_alo = int(not np.isnan(asia_lo) and or_lo_all < asia_lo)
    or_swept_above = int(not np.isnan(pdh) and or_hi_all > pdh and breakout_dir == "SHORT")
    or_swept_below = int(not np.isnan(pdl) and or_lo_all < pdl and breakout_dir == "LONG")

    # ── Momentum pre-OR ──
    or_open_px  = float(or_bars.iloc[0]['close'])
    or_close_px = float(or_bars.iloc[-1]['close'])
    momentum_pre = float(np.clip((or_close_px - or_open_px) / atr_ref if atr_ref > 0 else 0.0, -5, 5))

    # ── H1/H4 trend ──
    slope_h1 = slope_h4 = h1_trend = h4_trend = 0.0
    _brk_row = brk
    if 'h1_hi' in _brk_row.index and 'h1_lo' in _brk_row.index:
        h1_mid     = (_brk_row['h1_hi'] + _brk_row['h1_lo']) / 2.0 if _brk_row.get('h1_hi', 0) > 0 else 0.0
        or_h1_mid  = (or_bars.iloc[-1].get('h1_hi', 0) + or_bars.iloc[-1].get('h1_lo', 0)) / 2.0
        slope_h1   = float(np.clip((h1_mid - or_h1_mid) / atr_ref if atr_ref > 0 and h1_mid > 0 else 0.0, -5, 5))
        if breakout_dir == "LONG":
            h1_trend = 1 if slope_h1 > 0.1 else (0 if slope_h1 > -0.1 else -1)
        else:
            h1_trend = 1 if slope_h1 < -0.1 else (0 if slope_h1 < 0.1 else -1)
    if 'h4_hi' in _brk_row.index and 'h4_lo' in _brk_row.index:
        h4_mid     = (_brk_row['h4_hi'] + _brk_row['h4_lo']) / 2.0 if _brk_row.get('h4_hi', 0) > 0 else 0.0
        or_h4_mid  = (or_bars.iloc[-1].get('h4_hi', 0) + or_bars.iloc[-1].get('h4_lo', 0)) / 2.0
        slope_h4   = float(np.clip((h4_mid - or_h4_mid) / atr_ref if atr_ref > 0 and h4_mid > 0 else 0.0, -5, 5))
        if breakout_dir == "LONG":
            h4_trend = 1 if slope_h4 > 0.05 else (0 if slope_h4 > -0.05 else -1)
        else:
            h4_trend = 1 if slope_h4 < -0.05 else (0 if slope_h4 < 0.05 else -1)

    # ── ATR context ──
    atr_10d = float(df_global['atr_14'].rolling(10*390, min_periods=100).mean().reindex([df_idx[0] if len(df_idx) > 0 else df_global.index[-1]]).fillna(atr_ref).iloc[0]) if len(df_idx) > 0 else atr_ref
    atr_vs_10d = float(np.clip(atr_ref / max(atr_10d, 1.0), 0.3, 3.0))

    # ── Distanța de breakout ──
    if breakout_dir == "LONG":
        dist_long  = float(np.clip((brk['close'] - orh) / atr_ref if atr_ref > 0 else 0.0, 0, 10))
        dist_short = 0.0
    else:
        dist_long  = 0.0
        dist_short = float(np.clip((orl - brk['close']) / atr_ref if atr_ref > 0 else 0.0, 0, 10))

    # ── OR candle count și close vs mid ──
    or_candle_count = len(or_bars)
    last_or_close   = float(or_bars.iloc[-1]['close'])
    or_close_vs_mid = float(np.clip((last_or_close - or_mid) / or_width if or_width > 0 else 0.0, -1, 1))

    return {
        "body_pct":           round(body_pct, 4),
        "upper_wick_pct":     round(up_wick, 4),
        "lower_wick_pct":     round(lo_wick, 4),
        "close_strength":     round(cl_str, 4),
        "is_strong_close":    is_strong,
        "bar_range_atr":      round(bar_range_atr, 4),
        "vol_vs_or_avg":      round(vol_vs_or, 4),
        "vol_vs_20bar_avg":   round(vol_vs_20, 4),
        "or_width_atr":       round(or_width_atr, 4),
        "or_width_pts":       round(or_width, 2),
        "min_after_or":       min_after,
        "min_after_or_bin":   time_bin,
        "broke_prev_day_hi":  broke_pdh,
        "broke_prev_day_lo":  broke_pdl,
        "broke_asia_hi":      broke_ahi,
        "broke_asia_lo":      broke_alo,
        "or_swept_above":     or_swept_above,
        "or_swept_below":     or_swept_below,
        "momentum_pre_or":    round(momentum_pre, 4),
        "h1_trend_vs_break":  int(h1_trend),
        "h4_trend_vs_break":  int(h4_trend),
        "slope_h1_norm":      round(slope_h1, 4),
        "slope_h4_norm":      round(slope_h4, 4),
        "atr_abs":            round(atr_ref, 2),
        "atr_vs_10d_avg":     round(atr_vs_10d, 4),
        "dist_above_orh_atr": round(dist_long, 4),
        "dist_below_orl_atr": round(dist_short, 4),
        "or_candle_count":    or_candle_count,
        "or_close_vs_mid":    round(or_close_vs_mid, 4),
    }


# Feature list — EXACT mirror train_breakout_model.BREAKOUT_FEATURES (31 features)
_BREAKOUT_FEATURES = [
    "body_pct", "upper_wick_pct", "lower_wick_pct", "close_strength",
    "is_strong_close", "bar_range_atr",
    "vol_vs_or_avg",
    "or_width_atr", "or_width_pts",
    "min_after_or", "min_after_or_bin",
    "broke_pdh", "broke_pdl",
    "broke_asia_hi", "broke_asia_lo",
    "or_swept_above", "or_swept_below",
    "momentum_pre_or",
    "h1_trend_vs_break", "h4_trend_vs_break",
    "atr_abs", "atr_vs_10d_avg",
    "dist_above_orh_atr", "dist_below_orl_atr",
    "or_candle_count", "or_close_vs_mid",
    # VP / structural (v14.1)
    "inside_va", "dist_poc_atr", "sl_dist_atr", "above_vah", "below_val",
]

# Killzone config pentru v14 (or_end needed for timing calc)
_KZ_V14 = {
    "LON": {"start": 9.0, "end": 11.0, "or_end": 9.5},
    "NY":  {"start": 15.5, "end": 17.5, "or_end": 16.0},
}


def backtest_v14(bars: pd.DataFrame, conf_threshold: float = 0.20) -> pd.DataFrame:
    """
    v14.5 Regression Breakout Backtest — autodetect regression vs binary model.
    Regression: iau trade dacă predicted_R >= conf_threshold (default 0.20R)
    Binary:     iau trade dacă proba_REAL >= conf_threshold (default 0.50)
    """
    print("\n🔬 v14.5 REGRESSION BREAKOUT BACKTEST")
    print(f"   Threshold: {conf_threshold} | LON + NY independente per zi")

    # ── Încarcă modele (autodetect regression vs binary) ─────────────────────
    models = {}
    for kz_name in ["LON", "NY"]:
        m, feats, model_type, min_r = _load_breakout_model(kz_name)
        if m is not None:
            _is_reg = model_type and 'regression' in model_type
            # Threshold: pentru regresie din meta, pentru binary din conf_threshold param
            _thr = min_r if _is_reg else conf_threshold
            models[kz_name] = (m, feats or _BREAKOUT_FEATURES, _is_reg, _thr)
        else:
            print(f"   ⚠️ {kz_name}: model lipsă — killzone sărit")
    if not models:
        print("   ❌ Niciun model. Rulează: python3 train_breakout_model.py")
        return pd.DataFrame()

    # ── Coloane auxiliare ───────────────────────────────────────────────────
    if 'hour_dec' not in bars.columns:
        bars['hour_dec'] = bars['ts'].dt.hour + bars['ts'].dt.minute / 60.0
    if 'minute_of_day' not in bars.columns:
        bars['minute_of_day'] = bars['ts'].dt.hour * 60 + bars['ts'].dt.minute
    for _c in ['h1_hi','h1_lo','h4_hi','h4_lo']:
        if _c not in bars.columns: bars[_c] = 0.0

    # ── Precompute daily stats O SINGURĂ DATĂ (PDH/PDL/Asia/ATR10d) ─────────
    print("   ⚙️  Precomputing daily stats...")
    _dg = bars.groupby('date', sort=True).agg(
        day_hi  = ('high',  'max'),
        day_lo  = ('low',   'min'),
        avg_atr = ('atr_14','mean'),
    ).reset_index()
    _dg['pdh']     = _dg['day_hi'].shift(1)
    _dg['pdl']     = _dg['day_lo'].shift(1)
    _dg['atr_10d'] = _dg['avg_atr'].rolling(10, min_periods=3).mean()

    # Asia hi/lo per zi (bare 02:00-08:00)
    _asia = bars[bars['hour_dec'].between(2.0, 7.99)].groupby('date').agg(
        asia_hi=('high','max'), asia_lo=('low','min')).reset_index()
    _dg = _dg.merge(_asia, on='date', how='left')
    _dg['asia_hi'] = _dg['asia_hi'].fillna(0)
    _dg['asia_lo'] = _dg['asia_lo'].fillna(0)

    # Dict rapid date → stats
    _daily = {r['date']: r for _, r in _dg.iterrows()}
    print(f"   ✅ {len(_daily)} zile cu stats precomputed")

    # ── Indexare rapidă ts→pos ───────────────────────────────────────────────
    _ts_to_pos = {ts: i for i, ts in enumerate(bars['ts'])}

    trades = []
    _dbg   = {"no_brk": 0, "low_conf": 0, "passed": 0}

    groups     = list(bars.groupby('date', sort=True))
    total_days = len(groups)
    print(f"   📅 {total_days} zile | procesez LON + NY per zi...\n")

    for i, (date, day_df) in enumerate(groups):
        if i % 500 == 0:
            print(f"   {i}/{total_days} zile | {len(trades)} trades", end="\r")

        ds   = _daily.get(date, {})
        pdh  = float(ds.get('pdh',  np.nan) or np.nan)
        pdl  = float(ds.get('pdl',  np.nan) or np.nan)
        ahi  = float(ds.get('asia_hi', 0) or 0)
        alo  = float(ds.get('asia_lo', 0) or 0)
        a10  = float(ds.get('atr_10d', 9.0) or 9.0)

        for kz_name, (model, feats, is_regression, thr) in models.items():
            kz = _KZ_V14[kz_name]

            kz_bars = day_df[(day_df['hour_dec'] >= kz['start']) &
                             (day_df['hour_dec'] <= kz['end'])]
            if len(kz_bars) < 15: continue

            or_bars = kz_bars[kz_bars['hour_dec'] < kz['or_end']]
            post_or = kz_bars[kz_bars['hour_dec'] >= kz['or_end']].reset_index(drop=True)
            if len(or_bars) < 5 or len(post_or) < 3: continue

            orh      = float(or_bars['high'].max())
            orl      = float(or_bars['low'].min())
            or_width = orh - orl
            atr_ref  = float(or_bars['atr_14'].median())
            if or_width < 1.0 or atr_ref < 1.0: continue

            # Primul breakout (close confirm)
            brk_idx = brk_dir = None
            for j, row in post_or.iterrows():
                if row['close'] > orh:   brk_idx=j; brk_dir="LONG";  break
                elif row['close'] < orl: brk_idx=j; brk_dir="SHORT"; break
            if brk_idx is None:
                _dbg['no_brk'] += 1; continue

            brk = post_or.iloc[brk_idx]

            # ── Features rapide (fără lookup global) ──────────────────────
            rng    = max(float(brk['high']-brk['low']), 0.1)
            body   = abs(float(brk['close']-brk['open']))
            cl_str = (float(brk['close'])-float(brk['low']))/rng
            is_str = int(cl_str>=0.75) if brk_dir=="LONG" else int(cl_str<=0.25)
            or_vol = max(float(or_bars['volume'].mean()), 1.0)
            or_mid = (orh+orl)/2.0
            min_after = max(0, int(brk['minute_of_day'])-int(kz['or_end']*60))
            tb = 0 if min_after<5 else(1 if min_after<15 else(2 if min_after<30 else(3 if min_after<45 else 4)))

            or_hi=float(or_bars['high'].max()); or_lo=float(or_bars['low'].min())
            b_pdh=int(not np.isnan(pdh) and or_hi>pdh)
            b_pdl=int(not np.isnan(pdl) and or_lo<pdl)
            b_ahi=int(ahi>0 and or_hi>ahi)
            b_alo=int(alo>0 and or_lo<alo)
            mom=float(np.clip((float(or_bars.iloc[-1]['close'])-float(or_bars.iloc[0]['close']))/atr_ref,-5,5))

            h1h=float(brk.get('h1_hi',0) or 0); h1l=float(brk.get('h1_lo',0) or 0); h1t=0
            if h1h>0 and h1l>0:
                oh=float(or_bars.iloc[-1].get('h1_hi',0) or 0)+float(or_bars.iloc[-1].get('h1_lo',0) or 0)
                sl1=((h1h+h1l)/2-oh/2)/atr_ref if atr_ref>0 else 0
                h1t=1 if(brk_dir=="LONG" and sl1>0.1)or(brk_dir=="SHORT" and sl1<-0.1) else 0

            h4h=float(brk.get('h4_hi',0) or 0); h4l=float(brk.get('h4_lo',0) or 0); h4t=0
            if h4h>0 and h4l>0:
                oh4=float(or_bars.iloc[-1].get('h4_hi',0) or 0)+float(or_bars.iloc[-1].get('h4_lo',0) or 0)
                sl4=((h4h+h4l)/2-oh4/2)/atr_ref if atr_ref>0 else 0
                h4t=1 if(brk_dir=="LONG" and sl4>0.05)or(brk_dir=="SHORT" and sl4<-0.05) else 0

            last_or = float(or_bars.iloc[-1]['close'])

            # VP levels (prev session)
            val_px  = float(brk.get('val', 0) or 0)
            vah_px  = float(brk.get('vah', 0) or 0)
            poc_px  = float(brk.get('poc_level', 0) or 0)

            # entry_px definit ACUM — folosit în feat_vec VP și în SL structural
            entry_px = float(brk['close'])
            atr_now  = float(brk['atr_14']) if float(brk['atr_14']) > 0 else ATR_REF

            # SL structural pt feat_vec — folosim atr_ref (ca în training, OR median ATR)
            _sl_tmp, _risk_tmp = compute_structural_sl(
                brk_dir, entry_px, atr_ref,
                h1l, h1h, h4l, h4h, alo, ahi, val_px, vah_px,
                float(brk.get('lw_lo',0) or 0), float(brk.get('lw_hi',0) or 0))

            # ── Features v14.5 noi ────────────────────────────────────────────
            _thresh_app = orh - 0.3*or_width if brk_dir=="LONG" else orl + 0.3*or_width
            last10 = or_bars.tail(10)
            if brk_dir == "LONG":
                approach_count = int((last10['high'] >= _thresh_app).sum())
            else:
                approach_count = int((last10['low']  <= _thresh_app).sum())

            _or_vols = or_bars['volume'].values.astype(float)
            if len(_or_vols) >= 4:
                _x = np.arange(len(_or_vols))
                _coeff = float(np.polyfit(_x, _or_vols, 1)[0])
                vol_trend = float(np.clip(_coeff / max(np.mean(_or_vols), 1.0), -5.0, 5.0))
            else:
                vol_trend = 0.0

            _last5 = or_bars.tail(5)
            _l5r   = float(_last5['high'].max() - _last5['low'].min())
            range_compression = float(np.clip(_l5r / max(or_width, 0.1), 0.1, 2.0))

            feat_vec = [
                body/rng,
                (float(brk['high'])-max(float(brk['open']),float(brk['close'])))/rng,
                (min(float(brk['open']),float(brk['close']))-float(brk['low']))/rng,
                cl_str, is_str,
                min(rng/atr_ref,10.0),
                min(float(brk['volume'])/or_vol,20.0),
                min(or_width/atr_ref,20.0), or_width,
                min_after, tb,
                b_pdh, b_pdl, b_ahi, b_alo,
                int(b_pdh and brk_dir=="SHORT"),
                int(b_pdl and brk_dir=="LONG"),
                mom, h1t, h4t,
                atr_ref,
                float(np.clip(atr_ref/max(a10,1.0),0.3,3.0)),
                float(np.clip((entry_px-orh)/atr_ref if brk_dir=="LONG" else 0,0,10)),
                float(np.clip((orl-entry_px)/atr_ref if brk_dir=="SHORT" else 0,0,10)),
                len(or_bars),
                float(np.clip((last_or-or_mid)/or_width if or_width>0 else 0,-1,1)),
                # VP features (v14.1)
                int(val_px > 0 and vah_px > 0 and val_px <= entry_px <= vah_px),
                float(np.clip((entry_px - poc_px)/atr_ref if poc_px > 0 else 0, -10, 10)),
                float(np.clip(_risk_tmp / max(atr_ref, 0.1), 0.5, 5.0)),
                int(brk_dir=="LONG"  and vah_px > 0 and entry_px > vah_px),
                int(brk_dir=="SHORT" and val_px > 0 and entry_px < val_px),
                # v14.5 noi
                approach_count,
                vol_trend,
                range_compression,
            ]

            # Aliniază la feature list model (dict pentru compatibilitate cu orice ordine)
            feat_dict = dict(zip(_BREAKOUT_FEATURES + [
                'or_approach_count','vol_trend_in_or','range_compression'], feat_vec))
            X = np.array([[feat_dict.get(f, 0.0) for f in feats]])

            # ── Predict: regresie sau binary ──────────────────────────────────
            if is_regression:
                predicted_r = float(model.predict(X)[0])
                real_conf   = predicted_r   # pentru logging
                passes      = predicted_r >= thr
            else:
                proba     = model.predict_proba(X)[0]
                real_conf = float(proba[1]) if len(proba) > 1 else 0.0
                passes    = real_conf >= thr

            if not passes:
                _dbg['low_conf'] += 1; continue

            _dbg['passed'] += 1

            # entry_px și atr_now deja definite mai sus (înainte de feat_vec)
            brk_ts   = brk['ts']
            entry_pos = _ts_to_pos.get(brk_ts)
            if entry_pos is None: continue

            # ── SL structural pentru TRADE — folosim atr_now (brk bar ATR) ──
            # _sl_tmp/_risk_tmp din feat_vec foloseau atr_ref (OR median) pt feature
            # Recalculăm cu atr_now pentru execuție reală (mai precis)
            sl_px, risk_pts = compute_structural_sl(
                direction = brk_dir,
                entry_px  = entry_px,
                atr_now   = atr_now,
                h1_lo  = h1l,
                h1_hi  = h1h,
                h4_lo  = h4l,
                h4_hi  = h4h,
                asia_lo= alo,
                asia_hi= ahi,
                val    = val_px,
                vah    = vah_px,
                lw_lo  = float(brk.get('lw_lo', 0) or 0),
                lw_hi  = float(brk.get('lw_hi', 0) or 0),
            )

            result = simulate_trade(bars, entry_pos, brk_dir, entry_px, atr_now,
                                    sl_explicit=sl_px, risk_explicit=risk_pts)

            trades.append({
                'date': date, 'killzone': kz_name, 'direction': brk_dir,
                'entry_ts': brk_ts, 'entry_px': entry_px,
                'orh': round(orh,2), 'orl': round(orl,2),
                'or_width': round(or_width,2), 'atr': round(atr_now,2),
                'real_conf': round(real_conf,4), 'min_after_or': min_after,
                **result,
            })

    print(f"\n   📊 DEBUG v14:")
    print(f"      no_breakout:    {_dbg['no_brk']:,}")
    print(f"      low_conf (<{conf_threshold:.0%}): {_dbg['low_conf']:,}")
    print(f"      ✅ passed:      {_dbg['passed']:,}")
    _td = len(trades); _td_days = max(_td/max(total_days,1)*252,0)
    print(f"      trades/an est: {_td_days:.0f}")

    return pd.DataFrame(trades)


if __name__ == "__main__":
    import sys, os
    # Env toggles
    if os.getenv("NO_PARTIAL") == "1":
        USE_PARTIAL_EXITS = False
    if os.getenv("NY_ONLY") == "1":
        NY_ONLY = True
    if os.getenv("SWEEP_DEPTH") is not None:
        SWEEP_DEPTH_MIN_ATR = float(os.getenv("SWEEP_DEPTH"))
    if os.getenv("REQUIRE_FVG") == "1":
        REQUIRE_FVG_INTACT = True
    if os.getenv("REQUIRE_MSS") == "1":
        REQUIRE_MSS = True
    if os.getenv("BREAK_ONLY") == "1":
        BREAK_ONLY = True
    print(f"⚙️  USE_PARTIAL_EXITS={USE_PARTIAL_EXITS} | NY_ONLY={NY_ONLY} | BREAK_ONLY={BREAK_ONLY} | "
          f"SWEEP_DEPTH_MIN_ATR={SWEEP_DEPTH_MIN_ATR} | REQUIRE_FVG={REQUIRE_FVG_INTACT} | REQUIRE_MSS={REQUIRE_MSS}")

    mode = sys.argv[1] if len(sys.argv) > 1 else "naked"

    # ── Procesare an cu an (evitare OOM pentru 3.9M bare) ─────────────────────
    # Fiecare an are ~350K bare. Procesăm separat și combinăm trades la final.
    import sqlite3 as _sq3
    _conn = _sq3.connect(DB)
    _yrs  = [r[0] for r in _conn.execute(
        "SELECT DISTINCT CAST(strftime('%Y', timestamp) AS INT) FROM market_data ORDER BY 1"
    ).fetchall()]
    _conn.close()
    print(f"📅 Ani disponibili: {_yrs[0]}–{_yrs[-1]} ({len(_yrs)} ani)")

    all_trades = []
    for yr in _yrs:
        print(f"\n── Procesez {yr} ───────────────────────────────────────")
        bars = load_bars_for_year(yr)
        print(f"   ✓ {len(bars):,} bare | {bars['ts'].min()} → {bars['ts'].max()}")

        if mode == "v14":
            thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.60
            yr_trades = backtest_v14(bars, conf_threshold=thr)
        elif mode == "v13":
            yr_trades = backtest_orh_v13(bars, enable_v13=True)
        else:
            yr_trades = backtest_orh(bars)

        # Păstrăm DOAR tradele din anul curent (Dec prev / Ian next excluse)
        if len(yr_trades):
            yr_trades['_entry_year'] = pd.to_datetime(yr_trades['entry_ts']).dt.year
            yr_trades = yr_trades[yr_trades['_entry_year'] == yr].drop(columns=['_entry_year'])
            all_trades.append(yr_trades)
            print(f"   ✓ {len(yr_trades):,} trades în {yr}")

        del bars  # eliberăm RAM înainte de următorul an

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    if mode == "v14":
        out_name = f"backtest_orh_v14_thr{int(float(sys.argv[2]) * 100) if len(sys.argv) > 2 else 60}_trades.csv"
    elif mode == "v13":
        out_name = "backtest_orh_10y_v13_trades.csv"
    else:
        out_name = "backtest_orh_10y_trades.csv"

    print(f"\n✓ {len(trades):,} trade-uri simulate (total)")
    out = Path(__file__).parent / out_name
    trades.to_csv(out, index=False)
    print(f"💾 Salvat: {out}")
    report(trades)
