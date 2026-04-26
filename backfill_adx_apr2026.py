"""
backfill_adx_apr2026.py
=======================
Backfillează adx_14 + hurst + garch_vol + kalman_smooth + kalman_noise + vwap +
dist_vwap + sample_entropy + fisher_transform + fft_cycle + acf_lag1 + acf_lag5
pentru toate barele din mario_trading.db unde adx_14 = 0 și date > 2026-04-08.

Root cause: NT8 importer a adăugat bare noi (Apr 9 - Apr 22 2026) fără a rula
advanced_features.compute_all_advanced(). Precompute_regimes filtrează cu
`adx_14 > 0`, deci aceste date lipsesc din regime_labels.parquet.

Fix: calculăm indicatorii pe fereastra de context suficientă (Jan 2026 → Apr 22)
și actualizăm DB-ul.
"""

import sys, pathlib, sqlite3, logging
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/backfill_adx.log')
    ]
)
log = logging.getLogger("BACKFILL_ADX")

DB_PATH     = pathlib.Path(__file__).parent / "mario_trading.db"
CONTEXT_START = "2025-10-01"  # context generos pentru rolling (Hurst are nevoie de 100+ bare)
FILL_FROM     = "2026-04-09"  # prima dată care trebuie backfillată

ADVANCED_COLS = [
    'hurst', 'garch_vol', 'kalman_smooth', 'kalman_noise', 'adx_14',
    'vwap', 'dist_vwap', 'sample_entropy', 'fisher_transform',
    'fft_cycle', 'acf_lag1', 'acf_lag5'
]

def main():
    log.info("=" * 70)
    log.info("BACKFILL ADX Apr 2026 → mario_trading.db")
    log.info("=" * 70)

    try:
        import advanced_features as af
        log.info("✅ advanced_features importat")
    except ImportError as e:
        log.error(f"❌ Nu pot importa advanced_features: {e}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))

    # ── 1. Încărcăm fereastra de context pentru rolling ──────────────────
    log.info(f"Citim market_data din {CONTEXT_START} → prezent ...")
    df = pd.read_sql(f"""
        SELECT rowid, timestamp, open, high, low, close, volume, date,
               adx_14, atr_14
        FROM market_data
        WHERE date >= '{CONTEXT_START}'
        ORDER BY timestamp
    """, conn)

    log.info(f"  Rânduri totale în fereastră: {len(df):,}")

    # ── 2. Compute advanced features pe toată fereastra ──────────────────
    log.info("Calculăm advanced features (Hurst, GARCH, ADX, VWAP ...) ...")
    df_feat = af.compute_all_advanced(df.copy())
    log.info("✅ Compute_all_advanced terminat")

    # ── 3. Filtrăm doar barele de actualizat (adx_14=0, date >= FILL_FROM) ──
    mask_update = (df['date'] >= FILL_FROM) & (df['adx_14'] == 0)
    n_to_update = mask_update.sum()
    log.info(f"Bare de actualizat (adx=0, date>={FILL_FROM}): {n_to_update:,}")

    if n_to_update == 0:
        log.info("Nimic de actualizat — totul e deja computat.")
        conn.close()
        return

    rows_update = df[mask_update].copy()
    rows_feat   = df_feat[mask_update].copy()

    # ── 4. UPDATE batch ───────────────────────────────────────────────────
    log.info("Scriem în DB ...")
    cur = conn.cursor()
    updated = 0

    for i in range(len(rows_update)):
        rowid = int(rows_update.iloc[i]['rowid'])
        set_parts = []
        values    = []
        for col in ADVANCED_COLS:
            val = rows_feat.iloc[i][col] if col in rows_feat.columns else 0.0
            if pd.isna(val) or np.isinf(val):
                val = 0.0
            set_parts.append(f"{col} = ?")
            values.append(float(val))

        values.append(rowid)
        sql = f"UPDATE market_data SET {', '.join(set_parts)} WHERE rowid = ?"
        cur.execute(sql, values)
        updated += 1

        if updated % 500 == 0:
            conn.commit()
            log.info(f"  ... {updated}/{n_to_update} actualizate")

    conn.commit()
    log.info(f"✅ {updated:,} bare actualizate în mario_trading.db")

    # ── 5. Verificare ─────────────────────────────────────────────────────
    check = pd.read_sql(f"""
        SELECT date, COUNT(*) as bars,
               SUM(CASE WHEN adx_14 > 0 THEN 1 ELSE 0 END) as adx_ok
        FROM market_data
        WHERE date >= '{FILL_FROM}'
        GROUP BY date ORDER BY date
    """, conn)
    log.info("Status după update:")
    log.info("\n" + check.to_string())

    conn.close()
    log.info("\n✅ backfill_adx_apr2026.py COMPLET")
    log.info("Rulează precompute_regimes.py pentru a extinde regime_labels.parquet până la Apr 22")


if __name__ == "__main__":
    main()
