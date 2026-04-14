"""
╔══════════════════════════════════════════════════════════════════╗
║  ALADIN — Supabase Client  (UPDATE #1)                          ║
║  supabase_client.py                                              ║
╚══════════════════════════════════════════════════════════════════╝

Furnizează:
  - get_client()          → supabase.Client (singleton)
  - log_signal()          → inserează un semnal Aladin în `signals`
  - log_trade()           → inserează un trade în `trades`
  - get_recent_signals()  → ultimele N semnale din `signals`
  - update_trade_result() → actualizează PnL după închidere trade

Tabele Supabase necesare (rulează SQL din create_tables.sql):
  - signals  : fiecare apel aladin_engine() loghează un semnal
  - trades   : trade-uri automate / manuale cu rezultat PnL

Credențiale: configurate direct mai jos (sau via env vars).
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Încarcă .env din același director ca supabase_client.py
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv opțional

# =============================================================================
# CREDENȚIALE SUPABASE
# =============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# =============================================================================
# SINGLETON CLIENT
# =============================================================================
_supabase_client = None

def get_client():
    """Returnează clientul Supabase (singleton). Fallback graceful dacă librăria lipsește."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    try:
        from supabase import create_client, Client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Supabase conectat: %s", SUPABASE_URL)
        return _supabase_client
    except ImportError:
        logger.warning("⚠️  supabase-py nu e instalat. Rulează: pip3 install supabase --break-system-packages")
        return None
    except Exception as e:
        logger.error("❌ Supabase connect error: %s", e)
        return None


# =============================================================================
# LOG SEMNAL ALADIN
# =============================================================================
def log_signal(
    symbol:          str,
    direction:       str,           # "LONG" / "SHORT" / "WAIT"
    score_pct:       float,
    ai_score:        float,
    verdict:         str,
    ict_component:   float = 0.0,
    q_component:     float = 0.0,
    sentiment_score: float = 0.5,
    sentiment_mult:  float = 1.0,
    vix_mult:        float = 1.0,
    macro_mult:      float = 1.0,
    regime:          str   = "unknown",
    killzone:        str   = "",
    live_mode:       bool  = False,
    raw_score:       float = 0.0,
    extra:           Optional[dict] = None,
) -> Optional[dict]:
    """
    Inserează un semnal în tabela `signals`.
    Returnează row inserat sau None la eroare.
    """
    client = get_client()
    if client is None:
        return None

    row = {
        "ts":               datetime.now(timezone.utc).isoformat(),
        "symbol":           symbol,
        "direction":        direction,
        "score_pct":        round(score_pct, 2),
        "ai_score":         round(ai_score, 2),
        "verdict":          verdict[:200] if verdict else "",
        "ict_component":    round(ict_component, 4),
        "q_component":      round(q_component, 4),
        "sentiment_score":  round(sentiment_score, 4),
        "sentiment_mult":   round(sentiment_mult, 4),
        "vix_mult":         round(vix_mult, 4),
        "macro_mult":       round(macro_mult, 4),
        "regime":           regime,
        "killzone":         killzone,
        "live_mode":        live_mode,
        "raw_score":        round(raw_score, 4),
    }
    if extra and isinstance(extra, dict):
        # Serializam extra ca JSON string dacă există
        import json
        row["extra"] = json.dumps(extra)[:500]

    try:
        res = client.table("signals").insert(row).execute()
        logger.info("📡 Supabase signal logged: %s %s %.1f%%", symbol, direction, score_pct)
        return res.data[0] if res.data else row
    except Exception as e:
        logger.warning("⚠️  Supabase log_signal error: %s", e)
        return None


# =============================================================================
# LOG TRADE (autotrade / manual)
# =============================================================================
def log_trade(
    symbol:       str,
    direction:    str,          # "LONG" / "SHORT"
    score_pct:    float,
    ai_score:     float,
    entry_price:  float,
    sl_price:     float,
    tp_price:     float,
    qty:          float = 1.0,
    risk_usd:     float = 0.0,
    live_mode:    bool  = False,
    note:         str   = "",
) -> Optional[dict]:
    """
    Inserează un trade deschis în tabela `trades`.
    Returnează row inserat (cu id) sau None.
    """
    client = get_client()
    if client is None:
        return None

    row = {
        "ts_open":      datetime.now(timezone.utc).isoformat(),
        "symbol":       symbol,
        "direction":    direction,
        "score_pct":    round(score_pct, 2),
        "ai_score":     round(ai_score, 2),
        "entry_price":  entry_price,
        "sl_price":     sl_price,
        "tp_price":     tp_price,
        "qty":          qty,
        "risk_usd":     round(risk_usd, 2),
        "live_mode":    live_mode,
        "status":       "OPEN",
        "note":         note[:200] if note else "",
        "pnl":          0.0,
    }

    try:
        res = client.table("trades").insert(row).execute()
        logger.info("📈 Supabase trade logged: %s %s @ %.2f", symbol, direction, entry_price)
        return res.data[0] if res.data else row
    except Exception as e:
        logger.warning("⚠️  Supabase log_trade error: %s", e)
        return None


# =============================================================================
# UPDATE REZULTAT TRADE (la închidere)
# =============================================================================
def update_trade_result(
    trade_id:    int,
    exit_price:  float,
    pnl:         float,
    status:      str = "CLOSED",   # "CLOSED" / "SL_HIT" / "TP_HIT" / "MANUAL"
) -> Optional[dict]:
    """Actualizează un trade existent cu prețul de ieșire și PnL."""
    client = get_client()
    if client is None:
        return None
    try:
        res = (
            client.table("trades")
            .update({
                "ts_close":  datetime.now(timezone.utc).isoformat(),
                "exit_price": exit_price,
                "pnl":        round(pnl, 2),
                "status":     status,
            })
            .eq("id", trade_id)
            .execute()
        )
        logger.info("✅ Supabase trade #%d updated: %s PnL=%.2f", trade_id, status, pnl)
        return res.data[0] if res.data else None
    except Exception as e:
        logger.warning("⚠️  Supabase update_trade_result error: %s", e)
        return None


# =============================================================================
# QUERY — ultimele N semnale
# =============================================================================
def get_recent_signals(symbol: str = "NQ", limit: int = 20) -> list:
    """Returnează ultimele `limit` semnale pentru `symbol` din Supabase."""
    client = get_client()
    if client is None:
        return []
    try:
        res = (
            client.table("signals")
            .select("*")
            .eq("symbol", symbol)
            .order("ts", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.warning("⚠️  Supabase get_recent_signals error: %s", e)
        return []


# =============================================================================
# QUERY — statistici trade history
# =============================================================================
def get_trade_stats(symbol: str = "NQ") -> dict:
    """
    Returnează statistici agregate din tabela `trades`:
    total_trades, wins, losses, win_rate, total_pnl, avg_pnl
    """
    client = get_client()
    if client is None:
        return {}
    try:
        res = (
            client.table("trades")
            .select("pnl, status, direction")
            .eq("symbol", symbol)
            .neq("status", "OPEN")
            .execute()
        )
        rows = res.data or []
        if not rows:
            return {"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0}

        pnls  = [r["pnl"] for r in rows]
        wins  = sum(1 for p in pnls if p > 0)
        total = len(pnls)
        return {
            "total_trades": total,
            "wins":         wins,
            "losses":       total - wins,
            "win_rate":     round(wins / total * 100, 1) if total else 0.0,
            "total_pnl":    round(sum(pnls), 2),
            "avg_pnl":      round(sum(pnls) / total, 2) if total else 0.0,
        }
    except Exception as e:
        logger.warning("⚠️  Supabase get_trade_stats error: %s", e)
        return {}


# =============================================================================
# TEST STANDALONE
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n🔌 Test conexiune Supabase...")
    c = get_client()
    if c:
        print("✅ Conectat!")
        # Test log semnal
        r = log_signal(
            symbol="NQ", direction="LONG", score_pct=67.5, ai_score=31.2,
            verdict="TEST SIGNAL — Supabase integration check",
            regime="bullish", killzone="NY Open", live_mode=False,
        )
        print(f"📡 Signal logged: {r}")
    else:
        print("❌ Conexiune eșuată — instalează supabase-py")
