"""
train_quantum.py — Antrenare circuite quantum Aladin
=====================================================
Rulează: python3 train_quantum.py

Ce face:
  1. Încarcă jurnalul de trade-uri (aladin_trade_journal.csv + mario_trading.db)
  2. Mapează coloanele la formatul așteptat de train_main_circuit
  3. Antrenează MAIN_WEIGHTS (circuitul principal 6 qubiți) — salvează în aladin_main_weights.npy
  4. Antrenează noise_filter (circuitul de filtru 4 qubiți) — salvează în aladin_noise_weights.npy
  5. Actualizează rl_weights.json: activează quantum cu weight inițial 0.05

Output:
  - aladin_main_weights.npy   (MAIN_WEIGHTS antrenate)
  - aladin_noise_weights.npy  (noise filter antrenat)
  - train_quantum_report.txt  (raport loss per epocă)
"""

import os, sys, json, logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# ── Setup path ──────────────────────────────────────────────────────────────
ALADIN_DIR = Path(__file__).parent
sys.path.insert(0, str(ALADIN_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [QUANTUM] %(levelname)s — %(message)s",
)
log = logging.getLogger("quantum_train")

# ── Constante ────────────────────────────────────────────────────────────────
JOURNAL_PATH     = ALADIN_DIR / "aladin_trade_journal.csv"
DB_PATH          = ALADIN_DIR / "mario_trading.db"
MAIN_WEIGHTS_PATH = ALADIN_DIR / "aladin_main_weights.npy"
RL_WEIGHTS_PATH  = ALADIN_DIR / "rl_weights.json"
REPORT_PATH      = ALADIN_DIR / "train_quantum_report.txt"


# ── 1. ÎNCARCĂ ȘI PREPROCESEAZĂ DATE ────────────────────────────────────────
def load_training_data() -> pd.DataFrame:
    """
    Încarcă trade-urile cu result WIN/LOSS și mapează coloanele
    la formatul așteptat de train_main_circuit.
    """
    frames = []

    # A) Din CSV jurnal
    if JOURNAL_PATH.exists():
        df_csv = pd.read_csv(JOURNAL_PATH, low_memory=False)
        log.info(f"CSV jurnal: {len(df_csv)} rânduri")

        # Mapare coloane
        if "hybrid_score" in df_csv.columns and "score" not in df_csv.columns:
            df_csv["score"] = df_csv["hybrid_score"]
        if "regime" in df_csv.columns and "has_displacement" not in df_csv.columns:
            df_csv["has_displacement"] = df_csv["regime"].str.contains(
                "TREND|BREAK|DISP", case=False, na=False
            ).astype(int)

        # Normalizare result
        df_csv["result"] = df_csv["result"].fillna("").str.upper().str.strip()

        # Filtrăm doar trade-urile executate cu result cunoscut
        df_trades = df_csv[df_csv["result"].isin(["WIN", "LOSS", "TP", "SL", "PROFIT"])].copy()
        df_trades["result"] = df_trades["result"].map(
            {"WIN": "WIN", "PROFIT": "WIN", "TP": "WIN", "LOSS": "LOSS", "SL": "LOSS"}
        )
        frames.append(df_trades)
        log.info(f"  → {len(df_trades)} trade-uri cu result (WIN={len(df_trades[df_trades.result=='WIN'])} LOSS={len(df_trades[df_trades.result=='LOSS'])})")

    # B) Din DB market_data (fallback și date suplimentare)
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        # Luăm coloanele relevante din market_data
        df_db = pd.read_sql_query("""
            SELECT atr_14, poc_level, inside_va, has_displacement,
                   is_smt_bullish, is_smt_bearish, fvg_up, fvg_down,
                   body_size, dist_poc
            FROM market_data
            WHERE atr_14 > 0
            ORDER BY ROWID DESC
            LIMIT 3000
        """, conn)
        conn.close()
        log.info(f"DB market_data: {len(df_db)} bare pentru context features")
    except Exception as e:
        log.warning(f"DB skip: {e}")
        df_db = pd.DataFrame()

    if not frames:
        log.error("Nu există date pentru antrenare!")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)

    # Completăm features lipsă cu defaults rezonabile
    defaults = {
        "score":          50.0,
        "poc_level":      0.5,
        "inside_va":      0.5,
        "has_displacement": 0,
        "atr_14":         10.0,
        "smt":            False,
        "fvg":            False,
        "is_smt_bullish": 0,
        "is_smt_bearish": 0,
        "fvg_up":         0,
        "fvg_down":       0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    # Dacă poc_level e preț (>1), normalizăm la 0-1 față de medie
    if df["poc_level"].mean() > 1:
        _mid = df["poc_level"].median()
        df["poc_level"] = (df["poc_level"] / _mid).clip(0, 2) / 2.0

    # Normalizăm score la 0-1 dacă e pe 0-100
    if df["score"].max() > 1:
        df["score"] = df["score"] / 100.0

    log.info(f"Dataset final: {len(df)} rânduri | WIN={len(df[df.result=='WIN'])} LOSS={len(df[df.result=='LOSS'])}")
    return df


# ── 2. ANTRENARE ─────────────────────────────────────────────────────────────
def run_training():
    log.info("=" * 60)
    log.info("ALADIN QUANTUM TRAINING")
    log.info("=" * 60)

    df = load_training_data()

    # Import mario_rag DUPĂ ce am verificat datele (evităm crash la import)
    log.info("Import mario_rag (poate dura 10-20s la prima rulare)...")
    try:
        import mario_rag as mr
        log.info("mario_rag importat OK")
    except Exception as e:
        log.error(f"mario_rag import fail: {e}")
        sys.exit(1)

    report_lines = [
        f"QUANTUM TRAINING REPORT — {datetime.now(timezone.utc).isoformat()}",
        f"Dataset: {len(df)} trade-uri (WIN={len(df[df.result=='WIN'])} LOSS={len(df[df.result=='LOSS'])})",
        "=" * 60,
    ]

    # ── A) MAIN CIRCUIT (6 qubiți) ──────────────────────────────────────────
    log.info("\n⚛️  Antrenare MAIN CIRCUIT (6 qubiți, 50 epoci)...")
    losses_main = mr.train_main_circuit(journal_df=df, epochs=50)

    if losses_main:
        log.info(f"✅ MAIN CIRCUIT antrenat | Loss inițial: {losses_main[0]:.6f} → final: {losses_main[-1]:.6f}")
        report_lines.append(f"\nMAIN CIRCUIT:")
        report_lines.append(f"  Loss inițial: {losses_main[0]:.6f}")
        report_lines.append(f"  Loss final:   {losses_main[-1]:.6f}")
        report_lines.append(f"  Reducere:     {(1 - losses_main[-1]/losses_main[0])*100:.1f}%")
        report_lines.append(f"  Epochs:       {len(losses_main)}")
        report_lines.append(f"  Losses: {[round(l, 5) for l in losses_main[::10]]}")  # fiecare a 10-a

        # ── B) NOISE FILTER (4 qubiți) ──────────────────────────────────────
        log.info("\n⚛️  Antrenare NOISE FILTER (4 qubiți, 30 epoci)...")
        try:
            losses_noise = mr.train_noise_filter(df)
            if losses_noise:
                log.info(f"✅ NOISE FILTER antrenat | Loss: {losses_noise[0]:.6f} → {losses_noise[-1]:.6f}")
                report_lines.append(f"\nNOISE FILTER:")
                report_lines.append(f"  Loss inițial: {losses_noise[0]:.6f}")
                report_lines.append(f"  Loss final:   {losses_noise[-1]:.6f}")
        except Exception as e:
            log.warning(f"Noise filter skip: {e}")
            report_lines.append(f"\nNOISE FILTER: skip ({e})")

        # ── C) Activăm quantum în rl_weights.json ───────────────────────────
        log.info("\n📊 Activăm quantum în rl_weights.json...")
        try:
            with open(RL_WEIGHTS_PATH) as f:
                rl_data = json.load(f)
            w = rl_data.get("weights", {})
            if w.get("quantum", 0) == 0.0:
                # Activăm cu weight mic — RL decide dacă merită mai mult
                q_weight = 0.05
                # Redistribuim din orderflow (cel mai mare)
                w["orderflow"] = round(w.get("orderflow", 0.45) - q_weight, 4)
                w["quantum"]   = q_weight
                # Renormalizăm
                _sum = sum(w.values())
                w = {k: round(v / _sum, 4) for k, v in w.items()}
                rl_data["weights"]    = w
                rl_data["updated_at"] = datetime.now(timezone.utc).isoformat()
                rl_data["note"]       = rl_data.get("note", "") + " | quantum activat după antrenare"
                with open(RL_WEIGHTS_PATH, "w") as f:
                    json.dump(rl_data, f, indent=2)
                log.info(f"✅ quantum activat: {q_weight} (OF redus la {w['orderflow']})")
                report_lines.append(f"\nRL WEIGHTS actualizate: quantum={q_weight} OF={w['orderflow']}")
        except Exception as e:
            log.warning(f"RL weights update skip: {e}")

    else:
        log.error("❌ MAIN CIRCUIT nu s-a antrenat (date insuficiente sau eroare)")
        report_lines.append("\nMAIN CIRCUIT: FAIL")

    # ── Salvează raport ──────────────────────────────────────────────────────
    report_text = "\n".join(report_lines)
    REPORT_PATH.write_text(report_text)
    log.info(f"\n📄 Raport salvat: {REPORT_PATH}")
    print("\n" + report_text)


if __name__ == "__main__":
    run_training()
