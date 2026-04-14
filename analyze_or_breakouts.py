"""
OR Breakout Statistical Analysis
=================================
Pentru fiecare zi LON + NY:
  1. Calculează OR (ORH, ORL) din primele 30 min
  2. Găsește primul breakout (close > ORH sau close < ORL)
  3. Clasifică: REAL (nu revine în OR) vs FAKE (revine în OR)
  4. Statistici: minut, expansiune, condiții, win rate per oră

Scopul: înțelege CÂND și CUM se întâmplă breakout-urile reale
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

DB = Path(__file__).parent / "mario_trading.db"

KZ = {
    "LON": {"start": 9.0,  "end": 12.0, "or_end": 9.5},   # OR = 09:00-09:30 RO
    "NY":  {"start": 15.5, "end": 17.5, "or_end": 16.0},  # OR = 15:30-16:00 RO
}

def load_bars():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume, atr_14 "
        "FROM market_data ORDER BY timestamp", conn)
    conn.close()
    df['ts'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['ts'].dt.date
    df['hour_dec'] = df['ts'].dt.hour + df['ts'].dt.minute / 60.0
    df['minute_of_day'] = df['ts'].dt.hour * 60 + df['ts'].dt.minute
    df['atr_14'] = df['atr_14'].fillna(9.0).replace(0, 9.0)
    print(f"✅ {len(df):,} bars | {df['ts'].min()} → {df['ts'].max()}")
    return df

def analyze():
    df = load_bars()
    results = []

    bars_by_date = df.groupby('date')
    total_days = len(bars_by_date)
    print(f"📅 Analizez {total_days} zile...\n")

    for date, day_df in bars_by_date:
        day_df = day_df.reset_index(drop=True)

        for kz_name, kz in KZ.items():
            kz_bars = day_df[(day_df['hour_dec'] >= kz['start']) &
                             (day_df['hour_dec'] <= kz['end'])]
            if len(kz_bars) < 15:
                continue

            or_bars = kz_bars[kz_bars['hour_dec'] < kz['or_end']]
            post_or = kz_bars[kz_bars['hour_dec'] >= kz['or_end']].reset_index(drop=True)

            if len(or_bars) < 5 or len(post_or) < 3:
                continue

            orh = or_bars['high'].max()
            orl = or_bars['low'].min()
            or_width = orh - orl
            or_mid   = (orh + orl) / 2
            atr_ref  = or_bars['atr_14'].median()

            if or_width < 1.0 or atr_ref < 1.0:
                continue

            # ── Găsim primul breakout ──
            breakout_idx = None
            breakout_dir = None
            for i, row in post_or.iterrows():
                if row['close'] > orh:
                    breakout_idx = i
                    breakout_dir = "LONG"
                    break
                elif row['close'] < orl:
                    breakout_idx = i
                    breakout_dir = "SHORT"
                    break

            if breakout_idx is None:
                # Nu a existat breakout în sesiune
                results.append({
                    'date': date, 'killzone': kz_name,
                    'had_breakout': False, 'direction': None,
                    'is_real': None, 'breakout_type': 'NO_BREAK',
                })
                continue

            breakout_bar = post_or.iloc[breakout_idx]
            breakout_ts  = breakout_bar['ts']
            breakout_px  = breakout_bar['close']
            breakout_min = int(breakout_bar['minute_of_day'])

            # Minutul relativ față de start OR
            or_start_min = int(kz['or_end'] * 60)
            min_after_or = breakout_min - or_start_min

            # ── Clasificare REAL vs FAKE ──
            # REAL = prețul nu mai revine în OR (nu mai trece de orh dacă SHORT, orl dacă LONG)
            post_breakout = post_or.iloc[breakout_idx + 1:]
            returned_to_or = False
            max_expansion = 0.0
            session_end_px = breakout_px

            for _, bar in post_breakout.iterrows():
                if breakout_dir == "LONG":
                    if bar['low'] < orl:   # revenit complet în OR
                        returned_to_or = True
                        break
                    exp = bar['high'] - orh
                else:  # SHORT
                    if bar['high'] > orh:  # revenit complet în OR
                        returned_to_or = True
                        break
                    exp = orl - bar['low']
                max_expansion = max(max_expansion, exp)
                session_end_px = bar['close']

            is_real = not returned_to_or

            # Expansiunea în R (raportată la ATR)
            expansion_atr = max_expansion / atr_ref if atr_ref > 0 else 0
            expansion_pts = max_expansion

            # Câte bare până la 1R expansiune?
            r1_target = atr_ref  # 1R = 1 ATR (cam 9-10 pts NQ)
            bars_to_1r = None
            for bi, bar in enumerate(post_breakout.itertuples()):
                if breakout_dir == "LONG":
                    exp = bar.high - orh
                else:
                    exp = orl - bar.low
                if exp >= r1_target:
                    bars_to_1r = bi + 1
                    break

            # OR stats
            results.append({
                'date':           str(date),
                'killzone':       kz_name,
                'had_breakout':   True,
                'direction':      breakout_dir,
                'is_real':        is_real,
                'breakout_type':  'REAL' if is_real else 'FAKE',
                'breakout_min':   breakout_min,
                'min_after_or':   min_after_or,
                'or_width':       round(or_width, 2),
                'or_width_atr':   round(or_width / atr_ref, 3),
                'atr':            round(atr_ref, 2),
                'max_expansion_pts': round(expansion_pts, 2),
                'max_expansion_atr': round(expansion_atr, 3),
                'bars_to_1r':     bars_to_1r,
                'orh':            round(orh, 2),
                'orl':            round(orl, 2),
            })

    return pd.DataFrame(results)


def report(df: pd.DataFrame):
    print("\n" + "═" * 70)
    print("📊 OR BREAKOUT ANALYSIS")
    print("═" * 70)

    total_sessions = len(df)
    had_break = df[df['had_breakout'] == True]
    no_break  = df[df['had_breakout'] == False]

    print(f"\n📌 OVERVIEW GENERAL:")
    print(f"   Total sesiuni analizate: {total_sessions:,}")
    print(f"   Sesiuni cu breakout:     {len(had_break):,} ({100*len(had_break)/total_sessions:.1f}%)")
    print(f"   Sesiuni FĂRĂ breakout:   {len(no_break):,} ({100*len(no_break)/total_sessions:.1f}%)")

    real = had_break[had_break['is_real'] == True]
    fake = had_break[had_break['is_real'] == False]
    print(f"\n   Din sesiunile cu breakout:")
    print(f"   ✅ REAL (nu revine în OR): {len(real):,} ({100*len(real)/len(had_break):.1f}%)")
    print(f"   ❌ FAKE (revine în OR):    {len(fake):,} ({100*len(fake)/len(had_break):.1f}%)")

    for kz in ['LON', 'NY']:
        sub = had_break[had_break['killzone'] == kz]
        if len(sub) == 0:
            continue
        r = sub[sub['is_real'] == True]
        f = sub[sub['is_real'] == False]
        print(f"\n{'─'*50}")
        print(f"🕐 {kz} BREAKOUT STATS ({len(sub):,} sesiuni cu breakout)")
        print(f"   REAL: {len(r):,} ({100*len(r)/len(sub):.1f}%)  |  FAKE: {len(f):,} ({100*len(f)/len(sub):.1f}%)")

        # Directie split
        for d in ['LONG', 'SHORT']:
            ds = sub[sub['direction'] == d]
            dr = ds[ds['is_real'] == True]
            if len(ds) > 0:
                print(f"   {d}: {len(ds):,} total | {len(dr):,} real ({100*len(dr)/len(ds):.1f}%)")

        # Minutul breakout-ului
        print(f"\n   ⏰ CÂND se produce breakout-ul (minute după OR end):")
        real_sub = sub[sub['is_real'] == True]
        fake_sub = sub[sub['is_real'] == False]
        for label, s in [("REAL", real_sub), ("FAKE", fake_sub)]:
            if len(s) > 5:
                bins = [0, 5, 10, 15, 20, 30, 45, 60, 999]
                labels = ["0-5m","5-10m","10-15m","15-20m","20-30m","30-45m","45-60m","60m+"]
                s2 = s.copy()
                s2['bin'] = pd.cut(s2['min_after_or'], bins=bins, labels=labels, right=False)
                dist = s2['bin'].value_counts().sort_index()
                pcts = (dist / len(s) * 100).round(1)
                row = " | ".join([f"{l}:{p:.0f}%" for l, p in zip(pcts.index, pcts.values) if p > 0])
                print(f"   {label:4s}: {row}")

        # Expansiunea
        print(f"\n   📏 EXPANSIUNE maximă după breakout (pts):")
        for label, s in [("REAL", real_sub), ("FAKE", fake_sub)]:
            if len(s) > 5:
                print(f"   {label}: "
                      f"P25={s['max_expansion_pts'].quantile(0.25):.1f} | "
                      f"P50={s['max_expansion_pts'].quantile(0.50):.1f} | "
                      f"P75={s['max_expansion_pts'].quantile(0.75):.1f} | "
                      f"P90={s['max_expansion_pts'].quantile(0.90):.1f} | "
                      f"P99={s['max_expansion_pts'].quantile(0.99):.1f} pts")

        # OR width
        print(f"\n   📐 OR WIDTH (pts) — influențează calitatea breakout?")
        print(f"   REAL OR width: {real_sub['or_width'].median():.1f} pts median (ATR ratio: {real_sub['or_width_atr'].median():.2f}x)")
        print(f"   FAKE OR width: {fake_sub['or_width'].median():.1f} pts median (ATR ratio: {fake_sub['or_width_atr'].median():.2f}x)")

        # Bare până la 1R
        r_with_1r = real_sub.dropna(subset=['bars_to_1r'])
        if len(r_with_1r) > 10:
            pct_1r = 100 * len(r_with_1r) / len(real_sub)
            print(f"\n   🎯 Câte REAL breakout-uri ajung la 1R ({real_sub['atr'].median():.0f} pts):")
            print(f"   {len(r_with_1r):,} din {len(real_sub):,} ({pct_1r:.1f}%)")
            print(f"   Timp median la 1R: {r_with_1r['bars_to_1r'].median():.0f} bare (~{r_with_1r['bars_to_1r'].median():.0f} min)")

    # Per an
    print(f"\n{'─'*50}")
    print(f"📅 REAL BREAKOUT RATE PE AN:")
    had_break['year'] = pd.to_datetime(had_break['date']).dt.year
    for yr, yg in had_break.groupby('year'):
        r_yr = yg[yg['is_real'] == True]
        pct = 100 * len(r_yr) / len(yg)
        lon_r = yg[(yg['killzone']=='LON') & (yg['is_real']==True)]
        ny_r  = yg[(yg['killzone']=='NY')  & (yg['is_real']==True)]
        print(f"   {yr}: {pct:.1f}% real  |  LON: {len(lon_r):3d}  NY: {len(ny_r):3d}  total real: {len(r_yr):3d}")

    print("\n" + "═" * 70)
    return df


if __name__ == "__main__":
    results = analyze()
    results.to_csv(Path(__file__).parent / "or_breakout_analysis.csv", index=False)
    print(f"💾 Salvat: or_breakout_analysis.csv ({len(results):,} sesiuni)")
    report(results)
