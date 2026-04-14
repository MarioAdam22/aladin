"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN QUANTUM-ICT — FastAPI Endpoint                                      ║
║  api.py  |  Update #20 — REST API pentru semnale Aladin                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Endpoints:
  POST /signal?timestamp=YYYY-MM-DD HH:MM:SS  → semnal Aladin complet
  GET  /health                                  → status sistem
  GET  /signal/latest                           → ultimul semnal calculat
  GET  /metrics/backtest                        → metrici backtest cached

Rulare: uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import os
import json
import logging

# Lazy import pentru mario_rag (evită loading lent la startup)
_aladin_engine = None

def get_engine():
    global _aladin_engine
    if _aladin_engine is None:
        from mario_rag import aladin_engine
        _aladin_engine = aladin_engine
    return _aladin_engine

app = FastAPI(
    title       = "Aladin Quantum-ICT API",
    description = "REST API pentru sistemul de trading algorithmic Aladin. Returnează semnale ICT + ML + Quantum.",
    version     = "6.8",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS — permite acces din browser/React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

logger = logging.getLogger("aladin-api")
_last_signal: dict = {}

# =============================================================================
# SCHEMAS
# =============================================================================
class SignalRequest(BaseModel):
    timestamp: str  # format: "2025-01-15 09:30:00"
    balance: float = 10000.0

class SignalResponse(BaseModel):
    timestamp:   str
    verdict:     str
    score:       float
    score_pct:   float
    direction:   str
    conviction:  str
    regime:      str
    killzone:    Optional[str]
    risk_usd:    Optional[float]
    sl:          Optional[float]
    tp:          Optional[float]
    ict_signals: int
    ai_prob:     Optional[float]
    quantum_edge: Optional[float]
    error:       Optional[str] = None

class HealthResponse(BaseModel):
    status:    str
    version:   str
    timestamp: str
    db_ok:     bool
    model_ok:  bool

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """
    Update #20: Health check endpoint.
    Pinguit de UptimeRobot la fiecare 5 minute (Update #24).
    """
    db_path    = "/Users/mario/Desktop/Aladin/mario_trading.db"
    model_path = "/Users/mario/Desktop/Aladin/mario_bot.json"
    return HealthResponse(
        status    = "ok",
        version   = "6.8",
        timestamp = datetime.now().isoformat(),
        db_ok     = os.path.exists(db_path),
        model_ok  = os.path.exists(model_path),
    )


@app.post("/signal", tags=["Signals"])
def get_signal(
    timestamp: str = Query(..., description="Format: YYYY-MM-DD HH:MM:SS", example="2025-01-15 09:30:00"),
    balance:   float = Query(10000.0, description="Capital disponibil în USD"),
):
    """
    Update #20: Calculează semnalul Aladin pentru un timestamp dat.
    Returnează verdict complet: ICT + ML + Quantum + Orderflow.
    """
    global _last_signal
    try:
        engine = get_engine()
        result = engine(timestamp, balance=balance)

        if not result:
            raise HTTPException(status_code=500, detail="Engine a returnat None")

        signal = {
            "timestamp":    timestamp,
            "verdict":      result.get("verdict", "N/A"),
            "score":        round(float(result.get("score", 0)), 4),
            "score_pct":    round(float(result.get("score", 0)) * 100, 2),
            "direction":    result.get("trade_direction", "LONG"),
            "conviction":   result.get("conviction", "LOW"),
            "regime":       result.get("regime", "UNKNOWN"),
            "killzone":     result.get("killzone"),
            "risk_usd":     result.get("risk", {}).get("risk_usd"),
            "sl":           result.get("risk", {}).get("sl"),
            "tp":           result.get("risk", {}).get("tp"),
            "ict_signals":  int(result.get("ict_signals", 0)),
            "ai_prob":      result.get("ai_prob"),
            "quantum_edge": result.get("quantum_edge"),
            "close":        result.get("close"),
            "narrative":    result.get("narrative", ""),
            "error":        None,
        }
        _last_signal = signal
        return signal

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signal/latest", tags=["Signals"])
def get_latest_signal():
    """Returnează ultimul semnal calculat (cached)."""
    if not _last_signal:
        raise HTTPException(status_code=404, detail="Niciun semnal calculat încă. Apelează POST /signal mai întâi.")
    return _last_signal


@app.get("/signal/now", tags=["Signals"])
def get_signal_now(balance: float = Query(10000.0)):
    """Calculează semnalul pentru momentul curent."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:00")
    return get_signal(timestamp=ts, balance=balance)


@app.get("/system/info", tags=["System"])
def system_info():
    """Returnează informații despre sistem și modelele disponibile."""
    paths = {
        "db":         "/Users/mario/Desktop/Aladin/mario_trading.db",
        "model_xgb":  "/Users/mario/Desktop/Aladin/mario_bot.json",
        "model_lgbm": "/Users/mario/Desktop/Aladin/mario_bot_lgbm.pkl",
        "model_ens":  "/Users/mario/Desktop/Aladin/mario_bot_rf.pkl",
        "feat_imp":   "/Users/mario/Desktop/Aladin/mario_bot_feat_imp.csv",
    }
    return {
        "version": "6.8",
        "models":  {k: os.path.exists(v) for k, v in paths.items()},
        "paths":   {k: v for k, v in paths.items() if os.path.exists(v)},
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# TRADES & EQUITY ENDPOINTS
# =============================================================================

JOURNAL_PATH = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"

# ── Schema pentru adăugare trade manual ──────────────────────────────────────
class TradeEntry(BaseModel):
    timestamp:     str   = ""          # "2026-03-14 09:30:00"
    direction:     str   = "LONG"      # LONG | SHORT
    entry_price:   float = 0.0
    exit_price:    float = 0.0
    pnl:           float = 0.0
    result:        str   = "WIN"       # WIN | LOSS | BREAKEVEN
    hybrid_score:  float = 0.0
    killzone:      str   = ""
    risk_usd:      float = 100.0
    rr:            str   = "3:1"
    stop_loss:     float = 0.0
    take_profit:   float = 0.0
    notes:         str   = ""

@app.post("/trades/add", tags=["Journal"])
def add_trade(trade: TradeEntry):
    """Adaugă o tranzacție manuală în journal CSV."""
    import pandas as pd
    from datetime import datetime
    try:
        row = trade.dict()
        if not row["timestamp"]:
            row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row["logged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row["source"] = "manual"
        df_new = pd.DataFrame([row])
        if not os.path.exists(JOURNAL_PATH):
            df_new.to_csv(JOURNAL_PATH, index=False)
        else:
            df_new.to_csv(JOURNAL_PATH, mode="a", header=False, index=False)
        return {"ok": True, "message": f"Trade adăugat: {row['direction']} {row['result']} ${row['pnl']:.2f}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trades/import", tags=["Journal"])
async def import_trades_csv(file: UploadFile = File(...)):
    """Import trades dintr-un CSV. Coloane recunoscute automat."""
    import pandas as pd, io
    from datetime import datetime
    REQUIRED = ["timestamp", "direction", "pnl", "result"]
    try:
        contents = await file.read()
        df_import = pd.read_csv(io.StringIO(contents.decode("utf-8", errors="replace")))
        df_import.columns = [c.strip().lower().replace(" ", "_") for c in df_import.columns]

        # Mapări alternative de coloane
        col_map = {
            "date": "timestamp", "time": "timestamp", "datetime": "timestamp",
            "side": "direction", "type": "direction",
            "profit": "pnl", "p&l": "pnl", "pl": "pnl", "net_pnl": "pnl",
            "outcome": "result", "win": "result", "status": "result",
        }
        df_import.rename(columns={k: v for k, v in col_map.items() if k in df_import.columns}, inplace=True)

        # Normalizare coloane lipsă
        for col in REQUIRED:
            if col not in df_import.columns:
                df_import[col] = "" if col != "pnl" else 0.0

        # Normalizare result → WIN/LOSS
        if "result" in df_import.columns:
            def norm_result(v):
                v = str(v).upper()
                if v in ("1", "TRUE", "WIN", "W", "PROFIT", "YES"): return "WIN"
                if v in ("0", "FALSE", "LOSS", "L", "LOSS", "NO"):  return "LOSS"
                return v
            df_import["result"] = df_import["result"].apply(norm_result)

        df_import["logged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_import["source"]    = "import"

        if not os.path.exists(JOURNAL_PATH):
            df_import.to_csv(JOURNAL_PATH, index=False)
        else:
            # Append fără header
            df_import.to_csv(JOURNAL_PATH, mode="a", header=False, index=False)

        return {"ok": True, "imported": len(df_import), "columns_found": list(df_import.columns)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/trades/{idx}", tags=["Journal"])
def update_trade(idx: int, payload: dict):
    """Update câmpuri pe o tranzacție (setup_grade, mistake, notes etc.)."""
    import pandas as pd
    try:
        if not os.path.exists(JOURNAL_PATH):
            raise HTTPException(status_code=404, detail="Journal gol")
        df = pd.read_csv(JOURNAL_PATH)
        real_idx = len(df) - 1 - idx
        if real_idx < 0 or real_idx >= len(df):
            raise HTTPException(status_code=404, detail="Index invalid")
        for key, val in payload.items():
            df.at[real_idx, key] = val
        df.to_csv(JOURNAL_PATH, index=False)
        return {"ok": True, "updated": payload}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/journal/export", tags=["Journal"])
def export_journal(source: str = Query("ALL")):
    """Descarcă jurnalul ca CSV."""
    import pandas as pd
    from fastapi.responses import StreamingResponse
    import io
    try:
        if not os.path.exists(JOURNAL_PATH):
            return StreamingResponse(io.StringIO(""), media_type="text/csv",
                                     headers={"Content-Disposition": "attachment; filename=journal.csv"})
        df = pd.read_csv(JOURNAL_PATH)
        if source != "ALL" and "source" in df.columns:
            if source == "REAL":
                df = df[df["source"].isin(["manual", "import"]) | (~df["source"].isin(["backtest"]))]
            elif source == "BACKTEST":
                df = df[df["source"] == "backtest"]
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=aladin_journal.csv"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/journal/analytics", tags=["Journal"])
def journal_analytics():
    """Analitics complet: monthly stats, setup grade breakdown, mistake breakdown, equity reala."""
    import pandas as pd, numpy as np
    try:
        if not os.path.exists(JOURNAL_PATH):
            return {"monthly": [], "by_grade": [], "by_mistake": [], "equity": [], "labels": [],
                    "best_setup": "—", "worst_mistake": "—", "improvement_trend": []}

        df = pd.read_csv(JOURNAL_PATH)
        # Filtrăm doar tranzacțiile reale cu PnL
        real_df = df[df.get("source", pd.Series(["bot"]*len(df))).isin(["manual","import","bot"]) &
                     df["pnl"].notna() & (df["pnl"] != 0)].copy() if "source" in df.columns else \
                  df[df["pnl"].notna() & (df["pnl"] != 0)].copy()

        # Equity reala (cumsum PnL)
        initial = 10000.0
        if not real_df.empty and "pnl" in real_df.columns:
            eq_vals = [initial] + list(initial + real_df["pnl"].astype(float).cumsum())
            ts_col  = "timestamp" if "timestamp" in real_df.columns else ("date" if "date" in real_df.columns else None)
            eq_labels = list(real_df[ts_col].astype(str)) if ts_col else list(range(len(real_df)))
            eq_labels = ["Start"] + eq_labels
        else:
            eq_vals, eq_labels = [initial], ["Start"]

        # Monthly stats
        monthly = []
        if not real_df.empty:
            ts_col = "timestamp" if "timestamp" in real_df.columns else ("date" if "date" in real_df.columns else None)
            if ts_col:
                real_df["_dt"] = pd.to_datetime(real_df[ts_col], errors="coerce")
                real_df["_month"] = real_df["_dt"].dt.to_period("M").astype(str)
                for month, grp in real_df.groupby("_month"):
                    wins   = int((grp.get("result", pd.Series()) == "WIN").sum())
                    losses = int((grp.get("result", pd.Series()) == "LOSS").sum())
                    total  = wins + losses
                    monthly.append({
                        "month":    month,
                        "trades":   total,
                        "wins":     wins,
                        "losses":   losses,
                        "win_rate": round(wins/total*100, 1) if total else 0,
                        "pnl":      round(float(grp["pnl"].sum()), 2),
                    })

        # By setup grade (A+/A/B/C)
        by_grade = []
        if "setup_grade" in real_df.columns:
            for grade, grp in real_df.groupby("setup_grade"):
                if pd.isna(grade) or grade == "": continue
                wins = int((grp.get("result", pd.Series()) == "WIN").sum())
                tot  = len(grp)
                by_grade.append({
                    "grade":    str(grade),
                    "trades":   tot,
                    "win_rate": round(wins/tot*100, 1) if tot else 0,
                    "avg_pnl":  round(float(grp["pnl"].mean()), 2),
                    "total_pnl":round(float(grp["pnl"].sum()), 2),
                })
            by_grade.sort(key=lambda x: ["A+","A","B","C"].index(x["grade"]) if x["grade"] in ["A+","A","B","C"] else 99)

        # By mistake
        by_mistake = []
        if "mistake" in real_df.columns:
            for mistake, grp in real_df.groupby("mistake"):
                if pd.isna(mistake) or mistake in ("", "Nicio greșeală"): continue
                tot  = len(grp)
                by_mistake.append({
                    "mistake":   str(mistake),
                    "count":     tot,
                    "avg_pnl":   round(float(grp["pnl"].mean()), 2),
                    "total_loss":round(float(grp["pnl"].sum()), 2),
                })
            by_mistake.sort(key=lambda x: x["total_loss"])  # worst first

        # Calendar (zilnic)
        calendar = []
        if not real_df.empty and "_dt" in real_df.columns:
            real_df["_day"] = real_df["_dt"].dt.strftime("%Y-%m-%d")
            for day, grp in real_df.groupby("_day"):
                calendar.append({"date": day, "pnl": round(float(grp["pnl"].sum()), 2), "trades": len(grp)})

        # Improvement trend (monthly avg PnL last 6 months)
        improvement = [{"month": m["month"], "pnl": m["pnl"], "win_rate": m["win_rate"]}
                       for m in monthly[-6:]]

        # Best setup, worst mistake
        best_setup    = max(by_grade,    key=lambda x: x["win_rate"])["grade"]    if by_grade    else "—"
        worst_mistake = min(by_mistake,  key=lambda x: x["total_loss"])["mistake"] if by_mistake else "—"

        return {
            "monthly":          monthly,
            "by_grade":         by_grade,
            "by_mistake":       by_mistake,
            "equity":           eq_vals,
            "labels":           eq_labels,
            "calendar":         calendar,
            "improvement_trend":improvement,
            "best_setup":       best_setup,
            "worst_mistake":    worst_mistake,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/trades/{idx}", tags=["Journal"])
def delete_trade(idx: int):
    """Șterge o tranzacție din journal după index (0 = ultima)."""
    import pandas as pd
    try:
        if not os.path.exists(JOURNAL_PATH):
            raise HTTPException(status_code=404, detail="Journal gol")
        df = pd.read_csv(JOURNAL_PATH)
        real_idx = len(df) - 1 - idx   # idx 0 = ultima linie
        if real_idx < 0 or real_idx >= len(df):
            raise HTTPException(status_code=404, detail="Index invalid")
        df.drop(index=real_idx, inplace=True)
        df.to_csv(JOURNAL_PATH, index=False)
        return {"ok": True, "deleted_at": real_idx, "remaining": len(df)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trades", tags=["Journal"])
def get_trades(limit: int = Query(100, description="Număr maxim de tranzacții")):
    """Returnează ultimele tranzacții din journal CSV."""
    import pandas as pd
    import math
    if not os.path.exists(JOURNAL_PATH):
        return {"trades": [], "total": 0}
    try:
        df = pd.read_csv(JOURNAL_PATH)
        total = len(df)
        df = df.tail(limit).iloc[::-1]  # ultimele N, ordine inversă

        # Înlocuiește NaN/Inf cu None (null JSON valid) — fix pentru serializare
        def clean_val(v):
            if v is None: return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
            return v

        trades = [
            {k: clean_val(v) for k, v in row.items()}
            for row in df.to_dict(orient="records")
        ]
        return {"trades": trades, "total": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/equity", tags=["Journal"])
def get_equity(max_points: int = 300):
    """Returnează istoricul soldului (equity curve) din journal. Downsample la max_points."""
    import pandas as pd
    journal_path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
    if not os.path.exists(journal_path):
        return {"equity": [], "labels": []}
    try:
        df = pd.read_csv(journal_path)
        if "balance_after" not in df.columns and "balance" in df.columns:
            df["balance_after"] = df["balance"]
        bal_col = "balance_after" if "balance_after" in df.columns else df.columns[-1]
        equity  = df[bal_col].dropna().tolist()
        labels  = df["date"].tolist() if "date" in df.columns else list(range(len(equity)))
        # Downsample pentru performanță vizuală
        if len(equity) > max_points:
            step = max(1, len(equity) // max_points)
            indices = list(range(0, len(equity), step))
            if indices[-1] != len(equity) - 1:
                indices.append(len(equity) - 1)
            equity = [equity[i] for i in indices]
            labels = [labels[i] for i in indices]
        return {"equity": equity, "labels": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["Journal"])
def get_stats():
    """Returnează statistici agregate: win rate, profit factor, drawdown."""
    import pandas as pd
    journal_path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
    if not os.path.exists(journal_path):
        return {"win_rate": 0, "total_trades": 0, "profit_factor": 0, "balance": 10000}
    try:
        df = pd.read_csv(journal_path)
        total  = len(df)
        if total == 0:
            return {"win_rate": 0, "total_trades": 0, "profit_factor": 0, "balance": 10000}

        won_col = "won" if "won" in df.columns else ("result" if "result" in df.columns else None)
        win_rate = 0.0
        if won_col:
            wins = df[won_col].sum() if df[won_col].dtype == bool else (df[won_col] == "WIN").sum()
            win_rate = round(wins / total * 100, 1)

        pnl_col = "pnl" if "pnl" in df.columns else ("profit" if "profit" in df.columns else None)
        profit_factor = 0.0
        total_pnl     = 0.0
        if pnl_col:
            gains  = df[df[pnl_col] > 0][pnl_col].sum()
            losses = abs(df[df[pnl_col] < 0][pnl_col].sum())
            profit_factor = round(gains / losses, 2) if losses > 0 else float("inf")
            total_pnl     = round(df[pnl_col].sum(), 2)

        bal_col = "balance_after" if "balance_after" in df.columns else ("balance" if "balance" in df.columns else None)
        balance = round(float(df[bal_col].iloc[-1]), 2) if bal_col else 10000.0

        return {
            "win_rate":      win_rate,
            "total_trades":  total,
            "profit_factor": profit_factor,
            "total_pnl":     total_pnl,
            "balance":       balance,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/stats", tags=["AI Model"])
def get_model_stats():
    """Returnează statistici model AI: accuracy, features, importanță, confusion matrix."""
    import pandas as pd
    feat_path    = "/Users/mario/Desktop/Aladin/mario_features.json"
    feat_imp_path= "/Users/mario/Desktop/Aladin/mario_bot_feat_imp.csv"
    result = {"features": [], "accuracy": 0, "top_features": [], "feat_importance": [], "confusion_matrix": None}
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            meta = json.load(f)
        result.update({
            "features":          meta.get("features", []),
            "n_features":        meta.get("n_features", 0),
            "accuracy":          meta.get("accuracy", 0),
            "train_accuracy":    meta.get("train_accuracy", 0),
            "top_features":      meta.get("top_features", []),
            "trained_at":        meta.get("trained_at", ""),
            "rows_trained":      meta.get("rows_trained", 0),
            "confusion_matrix":  meta.get("confusion_matrix", None),
            "class_report":      meta.get("class_report", None),
        })
    if os.path.exists(feat_imp_path):
        df = pd.read_csv(feat_imp_path)
        result["feat_importance"] = df.head(15).to_dict(orient="records")
    return result


@app.get("/model/bias", tags=["AI Model"])
def get_model_bias():
    """Analizează bias-ul modelului: câte semnale LONG vs SHORT în ultimele 100 de semnale."""
    journal_path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
    if not os.path.exists(journal_path):
        return {"long_pct": 0, "short_pct": 0, "total": 0, "bias": "N/A"}
    try:
        import pandas as pd
        df = pd.read_csv(journal_path).tail(100)
        dir_col = "direction" if "direction" in df.columns else None
        if not dir_col:
            return {"long_pct": 0, "short_pct": 0, "total": 0, "bias": "N/A"}
        total = len(df)
        longs  = int((df[dir_col].str.upper() == "LONG").sum())
        shorts = int((df[dir_col].str.upper() == "SHORT").sum())
        long_pct  = round(longs  / total * 100, 1) if total else 0
        short_pct = round(shorts / total * 100, 1) if total else 0
        bias = "LONG" if long_pct > 65 else ("SHORT" if short_pct > 65 else "Echilibrat")
        return {"long_pct": long_pct, "short_pct": short_pct, "total": total, "bias": bias}
    except Exception as e:
        return {"long_pct": 0, "short_pct": 0, "total": 0, "bias": "N/A", "error": str(e)}


# =============================================================================
# BACKTEST — Async job system cu progress tracking
# =============================================================================
import threading, uuid as _uuid

_bt_jobs: dict = {}   # job_id → {status, progress, equity, result, error}

class _MockProgress:
    """Mock Streamlit progress_bar/status_text — scrie progresul în job dict."""
    def __init__(self, job_id: str = ""):
        self.job_id = job_id
    def progress(self, val):
        if self.job_id and self.job_id in _bt_jobs:
            pct = int(val * 100) if isinstance(val, float) and val <= 1.0 else int(val)
            _bt_jobs[self.job_id]["progress"] = min(pct, 99)
    def text(self, msg):
        if self.job_id and self.job_id in _bt_jobs:
            _bt_jobs[self.job_id]["status_text"] = str(msg)[:120]


def _build_result(df, initial_balance: float):
    """Construiește dict-ul de răspuns din DataFrame-ul de backtest."""
    import pandas as pd
    if df is None or (hasattr(df, "empty") and df.empty):
        return {"trades": [], "equity": [], "stats": {}}

    trades_df = df[df["action"] == "TRADE"].copy() if "action" in df.columns else df.copy()

    total    = len(trades_df)
    wins     = int((trades_df["result"] == "WIN").sum()) if total else 0
    losses   = total - wins
    win_rate = round(wins / total * 100, 2) if total else 0
    gross_p  = float(trades_df.loc[trades_df["result"] == "WIN",  "pnl"].sum()) if total else 0
    gross_l  = abs(float(trades_df.loc[trades_df["result"] == "LOSS", "pnl"].sum())) if total else 0
    pf       = round(gross_p / gross_l, 2) if gross_l > 0 else 0
    net_pnl  = float(trades_df["pnl"].sum()) if total else 0
    final_bal= float(df["balance"].iloc[-1]) if not df.empty else initial_balance

    # Equity curve — toate rândurile cu coloana balance
    equity = []
    for _, row in df.iterrows():
        try:
            bal = float(row["balance"])
            ts  = str(row.get("timestamp", row.get("date", "")))[:10]
            equity.append({"ts": ts, "balance": bal})
        except Exception:
            pass

    # ── Streak calculation ────────────────────────────────────────────────────
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    if not trades_df.empty:
        for r in trades_df["result"].tolist():
            if r == "WIN":
                cur_win += 1; cur_loss = 0
                if cur_win > max_win_streak: max_win_streak = cur_win
            elif r == "LOSS":
                cur_loss += 1; cur_win = 0
                if cur_loss > max_loss_streak: max_loss_streak = cur_loss

    # ── Time-of-Day stats ─────────────────────────────────────────────────────
    time_of_day = []
    if "time" in trades_df.columns and not trades_df.empty:
        try:
            tod = trades_df.groupby("time")["pnl"].agg(["sum", "count"]).reset_index()
            time_of_day = [
                {"time": str(r["time"]), "pnl": round(float(r["sum"]), 2), "trades": int(r["count"])}
                for _, r in tod.iterrows()
            ]
        except Exception:
            pass

    # ── Day-of-Week stats ─────────────────────────────────────────────────────
    day_of_week = []
    if "date" in trades_df.columns and not trades_df.empty:
        try:
            _tdf = trades_df.copy()
            _tdf["_dow"] = pd.to_datetime(_tdf["date"], errors="coerce").dt.day_name()
            dow = _tdf.groupby("_dow")["pnl"].agg(["sum", "count"]).reset_index()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            day_of_week = sorted(
                [{"day": str(r["_dow"]), "pnl": round(float(r["sum"]), 2), "trades": int(r["count"])}
                 for _, r in dow.iterrows() if r["_dow"] in day_order],
                key=lambda x: day_order.index(x["day"])
            )
        except Exception:
            pass

    # ── Max Drawdown (real, pe equity curve) ─────────────────────────────────
    max_dd = 0.0
    try:
        eq_vals = [e["balance"] for e in equity]
        if eq_vals:
            peak = eq_vals[0]
            for v in eq_vals:
                if v > peak: peak = v
                dd = (peak - v) / peak * 100 if peak > 0 else 0
                if dd > max_dd: max_dd = dd
        max_dd = round(max_dd, 2)
    except Exception:
        pass

    # ── Monte Carlo (200 simulări bootstrap cu replacement) ───────────────────
    monte_carlo = {}
    try:
        import numpy as np
        if not trades_df.empty and "pnl" in trades_df.columns and total >= 10:
            pnl_arr = trades_df["pnl"].values.astype(float)
            n_sims  = 200
            rng     = np.random.default_rng(42)
            sim_finals  = []
            sim_max_dds = []
            for _ in range(n_sims):
                # Bootstrap: sample CU replacement → fiecare simulare e un "univers alternativ"
                sampled  = rng.choice(pnl_arr, size=len(pnl_arr), replace=True)
                _mc_eq   = initial_balance + np.cumsum(sampled)
                _mc_peak = np.maximum.accumulate(_mc_eq)
                _mc_dd   = (_mc_eq - _mc_peak) / np.where(_mc_peak > 0, _mc_peak, 1) * 100
                sim_finals.append(float(_mc_eq[-1]))
                sim_max_dds.append(float(_mc_dd.min()))
            sf  = np.array(sim_finals)
            sdd = np.array(sim_max_dds)
            monte_carlo = {
                "median_final":  round(float(np.median(sf)), 2),
                "prob_profit":   round(float(np.mean(sf > initial_balance) * 100), 1),
                "p5_final":      round(float(np.percentile(sf, 5)), 2),
                "p95_final":     round(float(np.percentile(sf, 95)), 2),
                "worst_case":    round(float(np.min(sf)), 2),
                "best_case":     round(float(np.max(sf)), 2),
                "median_max_dd": round(float(np.median(sdd)), 2),
                "worst_max_dd":  round(float(np.min(sdd)), 2),
            }
    except Exception:
        pass

    # ── Trades list (max 300) ─────────────────────────────────────────────────
    trades_list = []
    for _, row in trades_df.head(300).iterrows():
        trades_list.append({
            "id":        int(row.get("trade_id", 0)),
            "date":      str(row.get("date", "")),
            "time":      str(row.get("time", "")),
            "direction": str(row.get("direction", "-")),
            "result":    str(row.get("result", "-")),
            "pnl":       float(row.get("pnl", 0)),
            "score":     float(row.get("score", 0)),
            "balance":   float(row.get("balance", initial_balance)),
            "risk_usd":  float(row.get("risk_usd", 0)),
            "killzone":  str(row.get("killzone", "-")),
        })

    # ── Auto-save backtest trades în journal ──────────────────────────────────
    try:
        import pandas as _pd
        from datetime import datetime as _dt

        # Salvăm DOAR tranzacțiile reale (WIN sau LOSS), nu SKIP / "-"
        journal_rows = []
        for t in trades_list:
            if t.get("result") not in ("WIN", "LOSS"):
                continue
            ts = f"{t.get('date','')} {t.get('time','')}".strip()
            journal_rows.append({
                "timestamp":    ts,
                "direction":    t.get("direction", "-"),
                "entry_price":  "",
                "exit_price":   "",
                "pnl":          float(t.get("pnl", 0)),
                "result":       t.get("result", "-"),
                "hybrid_score": float(t.get("score", 0)),
                "killzone":     t.get("killzone", "-"),
                "risk_usd":     float(t.get("risk_usd", 0)),
                "rr":           "",
                "stop_loss":    "",
                "take_profit":  "",
                "notes":        "",
                "setup_grade":  "",
                "mistake":      "",
                "logged_at":    _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source":       "backtest",
            })

        if not journal_rows:
            import logging as _log
            _log.warning("[Journal] Backtest: nicio tranzacție WIN/LOSS de salvat")
        else:
            _df_j = _pd.DataFrame(journal_rows)

            if not os.path.exists(JOURNAL_PATH):
                _df_j.to_csv(JOURNAL_PATH, index=False)
            else:
                _df_existing = _pd.read_csv(JOURNAL_PATH)

                # Șterge vechile rânduri de backtest și înlocuiește cu cele noi
                if "source" in _df_existing.columns:
                    _df_existing = _df_existing[_df_existing["source"] != "backtest"]

                _df_merged = _pd.concat([_df_existing, _df_j], ignore_index=True)
                _df_merged.to_csv(JOURNAL_PATH, index=False)

            import logging as _log
            _log.info(f"[Journal] Salvate {len(journal_rows)} tranzacții backtest în {JOURNAL_PATH}")

    except Exception as _je:
        import logging as _log
        _log.error(f"[Journal] Eroare la salvarea backtestului în journal: {_je}")

    return {
        "trades": trades_list,
        "equity": equity,
        "stats": {
            "total_trades":    total,
            "total_wins":      wins,
            "total_losses":    losses,
            "win_rate":        win_rate,
            "profit_factor":   pf,
            "net_pnl":         round(net_pnl, 2),
            "final_balance":   round(final_bal, 2),
            "return_pct":      round((final_bal - initial_balance) / initial_balance * 100, 2),
            "gross_profit":    round(gross_p, 2),
            "gross_loss":      round(gross_l, 2),
            "max_drawdown":    max_dd,
            "avg_win":         round(gross_p / wins, 2) if wins else 0,
            "avg_loss":        round(-gross_l / losses, 2) if losses else 0,
            "best_trade":      float(trades_df["pnl"].max()) if total else 0,
            "worst_trade":     float(trades_df["pnl"].min()) if total else 0,
            "max_win_streak":  max_win_streak,
            "max_loss_streak": max_loss_streak,
            "time_of_day":     time_of_day,
            "day_of_week":     day_of_week,
            "monte_carlo":     monte_carlo,
        },
    }


@app.post("/backtest/start", tags=["Backtest"])
def start_backtest(
    start_date:       str   = Query("2024-01-01"),
    end_date:         str   = Query("2024-12-31"),
    initial_balance:  float = Query(10000.0),
    risk_per_trade:   float = Query(1.0),
    rr_ratio:         float = Query(2.0),
    score_threshold:  float = Query(0.65),
    entry_times_str:  str   = Query("09:30,10:00,10:30,14:00,14:30"),
    max_trades_day:   int   = Query(3),
    walk_forward:     bool  = Query(False),
):
    """Pornește backtestul async și returnează un job_id. Sondează /backtest/status/{job_id}."""
    import sys, importlib
    sys.path.insert(0, "/Users/mario/Desktop/Aladin")

    entry_times = [t.strip() for t in entry_times_str.split(",") if t.strip()]

    # Walk-forward: rulează doar pe ultimele 30% din intervalul de date (out-of-sample)
    actual_start  = start_date
    in_sample_end = None
    if walk_forward:
        try:
            s = datetime.strptime(start_date, "%Y-%m-%d")
            e = datetime.strptime(end_date,   "%Y-%m-%d")
            from datetime import timedelta
            split_dt     = s + timedelta(days=int((e - s).days * 0.7))
            in_sample_end = split_dt.strftime("%Y-%m-%d")
            actual_start  = in_sample_end
        except Exception:
            pass

    job_id = _uuid.uuid4().hex[:10]
    _bt_jobs[job_id] = {
        "status": "running", "progress": 0,
        "status_text": "Se inițializează...", "result": None, "error": None,
    }

    def _run():
        try:
            dash = importlib.import_module("DASHBOARD")
            importlib.reload(dash)
            prog = _MockProgress(job_id)
            df = dash.run_backtest(
                start_date      = actual_start,
                end_date        = end_date,
                entry_times     = entry_times,
                max_trades_day  = max_trades_day,
                score_threshold = score_threshold,
                initial_balance = initial_balance,
                risk_per_trade  = risk_per_trade,
                rr_ratio        = rr_ratio,
                win_base        = 0.5,
                progress_bar    = prog,
                status_text     = prog,
            )
            result = _build_result(df, initial_balance)
            result["walk_forward"]  = walk_forward
            result["in_sample_end"] = in_sample_end
            result["_params"] = {
                "start_date":      start_date,
                "end_date":        end_date,
                "initial_balance": initial_balance,
                "risk_per_trade":  risk_per_trade,
                "rr_ratio":        rr_ratio,
                "score_threshold": score_threshold,
            }
            _bt_jobs[job_id]["result"]   = result
            _bt_jobs[job_id]["status"]   = "done"
            _bt_jobs[job_id]["progress"] = 100

            # ── Persistăm ultimul backtest pe disc ────────────────────────────
            try:
                import json as _json
                _last_bt_path = os.path.join(os.path.dirname(JOURNAL_PATH), "last_backtest.json")
                _save = dict(result)
                _save["equity"] = result.get("equity", [])[:500]
                with open(_last_bt_path, "w") as _f:
                    _json.dump(_save, _f)
            except Exception as _pe:
                import logging as _log
                _log.warning(f"[Backtest] Nu am putut salva last_backtest.json: {_pe}")

        except Exception as exc:
            _bt_jobs[job_id]["status"] = "error"
            _bt_jobs[job_id]["error"]  = str(exc)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@app.get("/backtest/last", tags=["Backtest"])
def get_last_backtest():
    """Returnează ultimul backtest salvat pe disc (persistă între sesiuni)."""
    import json as _json
    _last_bt_path = os.path.join(os.path.dirname(JOURNAL_PATH), "last_backtest.json")
    if not os.path.exists(_last_bt_path):
        return {"found": False}
    try:
        with open(_last_bt_path) as _f:
            data = _json.load(_f)
        return {"found": True, "result": data}
    except Exception as e:
        return {"found": False, "error": str(e)}


@app.get("/backtest/status/{job_id}", tags=["Backtest"])
def get_backtest_status(job_id: str):
    """Returnează starea curentă a unui job de backtest."""
    if job_id not in _bt_jobs:
        raise HTTPException(status_code=404, detail="Job negăsit")
    job = _bt_jobs[job_id]
    return {
        "status":      job["status"],           # "running" | "done" | "error"
        "progress":    job.get("progress", 0),  # 0–100
        "status_text": job.get("status_text", ""),
        "error":       job.get("error"),
        "result":      job["result"] if job["status"] == "done" else None,
    }


# Backward-compat — păstrăm /backtest/run dar cu bug-ul capital → initial_balance rezolvat
@app.post("/backtest/run", tags=["Backtest"])
def run_backtest_api(
    start_date:      str   = Query("2024-01-01"),
    end_date:        str   = Query("2024-12-31"),
    initial_balance: float = Query(10000.0),
    risk_per_trade:  float = Query(1.0),
    rr_ratio:        float = Query(2.0),
    score_threshold: float = Query(0.65),
):
    """Rulează backtestul sincron (fallback). Preferă /backtest/start pentru async."""
    import sys, importlib
    sys.path.insert(0, "/Users/mario/Desktop/Aladin")
    try:
        dash = importlib.import_module("DASHBOARD")
        importlib.reload(dash)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nu pot importa DASHBOARD.py: {e}")
    try:
        prog = _MockProgress()
        df = dash.run_backtest(
            start_date=start_date, end_date=end_date,
            entry_times=["09:30","10:00","10:30","14:00","14:30"],
            max_trades_day=3, score_threshold=score_threshold,
            initial_balance=initial_balance, risk_per_trade=risk_per_trade,
            rr_ratio=rr_ratio, win_base=0.5,
            progress_bar=prog, status_text=prog,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _build_result(df, initial_balance)


# =============================================================================
# DAY ANALYSIS
# =============================================================================

@app.get("/day-analysis", tags=["Dashboard"])
def day_analysis(date: str = Query(..., description="YYYY-MM-DD")):
    """Returnează toate semnalele/tranzacțiile dintr-o zi specifică din journal."""
    import pandas as pd, math

    def clean(v):
        if v is None: return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
        return v

    try:
        if not os.path.exists(JOURNAL_PATH):
            return {"date": date, "trades": [], "summary": {}}

        df = pd.read_csv(JOURNAL_PATH)

        # Detectăm coloana de timestamp
        ts_col = None
        for c in ["timestamp", "date", "entry_time", "logged_at"]:
            if c in df.columns:
                ts_col = c
                break
        if ts_col is None:
            return {"date": date, "trades": [], "summary": {"error": "Nicio coloană de dată găsită"}}

        df["_date"] = pd.to_datetime(df[ts_col], errors="coerce").dt.strftime("%Y-%m-%d")
        day_df = df[df["_date"] == date].copy()
        day_df = day_df.drop(columns=["_date"])

        if day_df.empty:
            return {"date": date, "trades": [], "summary": {"message": f"Nicio înregistrare pentru {date}"}}

        trades = [{ k: clean(v) for k, v in row.items() } for row in day_df.to_dict(orient="records")]

        # Summary pentru ziua respectivă
        pnl_col = "pnl" if "pnl" in day_df.columns else None
        res_col = "result" if "result" in day_df.columns else None

        total_pnl = round(float(day_df[pnl_col].fillna(0).sum()), 2) if pnl_col else 0
        wins      = int((day_df[res_col] == "WIN").sum())  if res_col else 0
        losses    = int((day_df[res_col] == "LOSS").sum()) if res_col else 0
        total_trades = wins + losses

        # Scor mediu
        score_col = "hybrid_score" if "hybrid_score" in day_df.columns else ("score" if "score" in day_df.columns else None)
        avg_score = round(float(day_df[score_col].fillna(0).mean()), 3) if score_col else 0

        # Killzone dominantă
        kz_col = "killzone" if "killzone" in day_df.columns else None
        top_kz = day_df[kz_col].mode()[0] if kz_col and not day_df[kz_col].dropna().empty else "—"

        summary = {
            "total_records": len(day_df),
            "total_trades":  total_trades,
            "wins":          wins,
            "losses":        losses,
            "win_rate":      round(wins / total_trades * 100, 1) if total_trades else 0,
            "total_pnl":     total_pnl,
            "avg_score":     avg_score,
            "top_killzone":  str(top_kz),
            "result":        "✅ Zi profitabilă" if total_pnl > 0 else ("❌ Zi în pierdere" if total_pnl < 0 else "⚡ Breakeven"),
        }

        return {"date": date, "trades": trades, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MARKET DATA, FEAR & GREED, ECONOMIC CALENDAR, QUICK NOTES
# =============================================================================
import requests as _requests

NOTES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_notes.json")

# ── Market Data (Yahoo Finance) ──────────────────────────────────────────────
@app.get("/market-data", tags=["Live"])
def market_data():
    """Returnează prețurile live pentru DXY, Gold, BTC, SPY."""
    symbols = {"DXY": "DX-Y.NYB", "Gold": "GC=F", "BTC": "BTC-USD", "SPY": "SPY"}
    result = {}
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    for name, sym in symbols.items():
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1m&range=1d"
            r = _requests.get(url, headers=headers, timeout=6)
            meta = r.json()["chart"]["result"][0]["meta"]
            price = float(meta.get("regularMarketPrice") or meta.get("chartPreviousClose", 0))
            prev  = float(meta.get("previousClose") or meta.get("chartPreviousClose", price))
            chg   = round((price - prev) / max(abs(prev), 0.01) * 100, 2) if prev else 0
            result[name] = {"price": round(price, 4), "prev": round(prev, 4), "change_pct": chg, "symbol": sym}
        except Exception as e:
            logger.warning(f"market_data {name}: {e}")
            result[name] = {"price": 0, "prev": 0, "change_pct": 0, "symbol": sym}
    return result

# ── Fear & Greed Index ────────────────────────────────────────────────────────
@app.get("/fear-greed", tags=["Live"])
def fear_greed():
    """Returnează Fear & Greed Index (alternative.me)."""
    try:
        r = _requests.get("https://api.alternative.me/fng/?limit=7", timeout=6)
        return r.json()
    except Exception as e:
        logger.warning(f"fear_greed: {e}")
        return {"data": [{"value": "50", "value_classification": "Neutral", "timestamp": "0"}]}

# ── Economic Calendar ─────────────────────────────────────────────────────────
_CAL_CACHE: dict = {"ts": 0.0, "events": [], "all_events": []}

def _fetch_cal_events(week_url: str):
    """Fetch + parse events from a ForexFactory week JSON URL."""
    r = _requests.get(week_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    all_events = r.json()
    parsed = []
    for ev in all_events:
        if ev.get("country") in ("USD", "EUR", "GBP"):
            raw_date = ev.get("date", "")
            ev_date  = raw_date[:10] if len(raw_date) >= 10 else ""
            ev_time  = raw_date[11:16] if len(raw_date) >= 16 else "—"
            parsed.append({
                "date":     ev_date,
                "time":     ev_time,
                "country":  ev.get("country", ""),
                "event":    ev.get("title", ""),
                "impact":   ev.get("impact", ""),
                "prev":     ev.get("previous", ""),
                "forecast": ev.get("forecast", ""),
            })
    return sorted(parsed, key=lambda x: (x["date"], x["time"]))

@app.get("/economic-calendar", tags=["Live"])
def economic_calendar():
    """Returnează evenimentele economice ale săptămânii (ForexFactory) cu cache 30 min."""
    import time as _time
    global _CAL_CACHE
    now = _time.time()
    # Return cache if fresh (< 30 min)
    if now - _CAL_CACHE["ts"] < 1800 and _CAL_CACHE["events"]:
        return {"events": _CAL_CACHE["events"], "cached": True}
    try:
        events = _fetch_cal_events("https://nfs.faireconomy.media/ff_calendar_thisweek.json")
        _CAL_CACHE = {"ts": now, "events": events, "all_events": events}
        today = datetime.now().strftime("%Y-%m-%d")
        relevant = [e for e in events if e["date"] >= today]
        return {"events": relevant[:25]}
    except Exception as e:
        logger.warning(f"economic_calendar fetch failed: {e}")
        # Return stale cache if available
        if _CAL_CACHE["events"]:
            logger.info("Returning stale calendar cache")
            return {"events": _CAL_CACHE["events"], "stale": True}
        return {"events": [], "error": str(e)}

@app.get("/economic-calendar/week", tags=["Live"])
def economic_calendar_week(date: str = ""):
    """Returnează toate evenimentele pentru săptămâna care conține `date` (YYYY-MM-DD)."""
    import time as _time
    try:
        from datetime import date as _date, timedelta
        if date:
            d = _date.fromisoformat(date)
        else:
            d = _date.today()
        # Compute Mon of that week
        mon = d - timedelta(days=d.weekday())
        week_str = mon.strftime("%Y-%m-%d")
        # ForexFactory URL pattern for specific weeks
        # Try thisweek first if it matches, else nextweek
        today_mon = _date.today() - timedelta(days=_date.today().weekday())
        if mon == today_mon:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        elif mon == today_mon + timedelta(weeks=1):
            url = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"
        else:
            # For past weeks, return from cache if available
            events = _CAL_CACHE.get("all_events", [])
            week_events = [e for e in events if e["date"] >= week_str and e["date"] < (mon + timedelta(days=7)).strftime("%Y-%m-%d")]
            return {"events": week_events, "week_start": week_str}
        events = _fetch_cal_events(url)
        week_events = [e for e in events if e["date"] >= week_str and e["date"] < (mon + timedelta(days=7)).strftime("%Y-%m-%d")]
        return {"events": week_events, "week_start": week_str}
    except Exception as e:
        logger.warning(f"economic_calendar_week: {e}")
        return {"events": [], "error": str(e)}

# ── Quick Notes ───────────────────────────────────────────────────────────────
@app.get("/notes", tags=["Notes"])
def get_notes():
    if not os.path.exists(NOTES_PATH):
        return {"notes": []}
    try:
        with open(NOTES_PATH) as f:
            return json.load(f)
    except:
        return {"notes": []}

@app.post("/notes", tags=["Notes"])
def add_note(body: dict):
    notes = []
    if os.path.exists(NOTES_PATH):
        try:
            with open(NOTES_PATH) as f:
                notes = json.load(f).get("notes", [])
        except:
            pass
    notes.insert(0, {"text": str(body.get("text", ""))[:500], "ts": datetime.now().strftime("%Y-%m-%d %H:%M")})
    notes = notes[:30]
    with open(NOTES_PATH, "w") as f:
        json.dump({"notes": notes}, f, ensure_ascii=False)
    return {"ok": True}

@app.delete("/notes/{idx}", tags=["Notes"])
def delete_note(idx: int):
    if not os.path.exists(NOTES_PATH):
        raise HTTPException(404, "No notes")
    with open(NOTES_PATH) as f:
        data = json.load(f)
    notes = data.get("notes", [])
    if idx < 0 or idx >= len(notes):
        raise HTTPException(404, "Index out of range")
    notes.pop(idx)
    with open(NOTES_PATH, "w") as f:
        json.dump({"notes": notes}, f, ensure_ascii=False)
    return {"ok": True}

# =============================================================================
# CACHE CLEAR ENDPOINT
# =============================================================================
@app.post("/cache/clear", tags=["System"])
def clear_cache():
    """Șterge toate cache-urile în memorie."""
    global _CAL_CACHE
    _CAL_CACHE.clear()
    return {"ok": True, "cleared": ["economic_calendar"]}

# =============================================================================
# MONTE CARLO SIMULATION ENDPOINT
# =============================================================================
@app.get("/backtest/montecarlo", tags=["Backtest"])
def monte_carlo(n_sims: int = 200, n_trades: int = 100):
    """Simulează N scenarii Monte Carlo bazate pe trade-urile istorice."""
    import pandas as pd
    import numpy as np
    import random
    journal_path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
    try:
        df = pd.read_csv(journal_path, low_memory=False)
        trades = df[df['pnl'].notna() & df['result'].notna()].copy()
        pnls = pd.to_numeric(trades['pnl'], errors='coerce').dropna().tolist()
        if len(pnls) < 10:
            return {"error": "Insuficiente trade-uri"}

        results = []
        for _ in range(min(n_sims, 500)):
            sample = random.choices(pnls, k=min(n_trades, len(pnls)))
            equity = [0]
            for p in sample:
                equity.append(equity[-1] + p)
            results.append(equity)

        # Statistici
        finals = [r[-1] for r in results]
        finals.sort()

        return {
            "simulations": results[:50],  # max 50 pentru UI
            "percentile_10": round(finals[int(len(finals)*0.10)], 2),
            "percentile_50": round(finals[int(len(finals)*0.50)], 2),
            "percentile_90": round(finals[int(len(finals)*0.90)], 2),
            "prob_positive": round(sum(1 for f in finals if f > 0) / len(finals) * 100, 1),
            "n_sims": len(results),
            "n_trades": n_trades,
        }
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# DAY-OF-WEEK STATS ENDPOINT
# =============================================================================
@app.get("/stats/dow", tags=["Analytics"])
def stats_dow():
    """Win rate și PnL mediu per zi a săptămânii."""
    import pandas as pd
    journal_path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
    try:
        df = pd.read_csv(journal_path, low_memory=False)
        trades = df[df['pnl'].notna() & df['result'].notna()].copy()
        trades['pnl'] = pd.to_numeric(trades['pnl'], errors='coerce')
        trades['ts'] = pd.to_datetime(trades['timestamp'], errors='coerce')
        trades['dow'] = trades['ts'].dt.day_name()
        trades['win'] = trades['result'].str.upper() == 'WIN'

        days_order = ['Monday','Tuesday','Wednesday','Thursday','Friday']
        result = []
        for d in days_order:
            sub = trades[trades['dow'] == d]
            if len(sub) == 0:
                result.append({"day": d, "n": 0, "wr": 0, "avg_pnl": 0, "total_pnl": 0})
            else:
                result.append({
                    "day": d,
                    "n": len(sub),
                    "wr": round(sub['win'].mean() * 100, 1),
                    "avg_pnl": round(float(sub['pnl'].mean()), 2),
                    "total_pnl": round(float(sub['pnl'].sum()), 2),
                })
        return {"dow": result}
    except Exception as e:
        return {"error": str(e), "dow": []}

# =============================================================================
# HOUR STATS ENDPOINT
# =============================================================================
@app.get("/stats/hour", tags=["Analytics"])
def stats_hour():
    """Win rate și PnL mediu per oră."""
    import pandas as pd
    journal_path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
    try:
        df = pd.read_csv(journal_path, low_memory=False)
        trades = df[df['pnl'].notna() & df['result'].notna()].copy()
        trades['pnl'] = pd.to_numeric(trades['pnl'], errors='coerce')
        trades['ts'] = pd.to_datetime(trades['timestamp'], errors='coerce')
        trades['hour'] = trades['ts'].dt.hour
        trades['win'] = trades['result'].str.upper() == 'WIN'

        result = []
        for h in range(9, 20):
            sub = trades[trades['hour'] == h]
            result.append({
                "hour": h,
                "n": len(sub),
                "wr": round(float(sub['win'].mean() * 100), 1) if len(sub) > 0 else 0,
                "avg_pnl": round(float(sub['pnl'].mean()), 2) if len(sub) > 0 else 0,
                "total_pnl": round(float(sub['pnl'].sum()), 2) if len(sub) > 0 else 0,
            })
        return {"hours": result}
    except Exception as e:
        return {"error": str(e), "hours": []}

# =============================================================================
# SYSTEM ERROR LOGS ENDPOINT
# =============================================================================
import traceback as _tb
_error_log: list = []

def log_error(context: str, exc: Exception):
    """Helper to log errors globally."""
    _error_log.append({
        "ts": datetime.now().isoformat(),
        "context": context,
        "error": str(exc),
        "traceback": _tb.format_exc()[-500:],
    })
    _error_log[:] = _error_log[-50:]  # keep last 50

@app.get("/system/errors", tags=["System"])
def get_error_logs():
    return {"errors": list(reversed(_error_log)), "count": len(_error_log)}

@app.post("/system/errors/clear", tags=["System"])
def clear_error_logs():
    _error_log.clear()
    return {"ok": True}

# =============================================================================
# TOP WINNING PATTERNS ENDPOINT
# =============================================================================
@app.get("/stats/patterns", tags=["Analytics"])
def top_patterns():
    """Top 5 combinații (killzone + direction + score_bucket) cu cel mai bun WR."""
    import pandas as pd
    journal_path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
    try:
        df = pd.read_csv(journal_path, low_memory=False)
        trades = df[df['pnl'].notna() & df['result'].notna()].copy()
        trades['pnl'] = pd.to_numeric(trades['pnl'], errors='coerce')
        trades['score'] = pd.to_numeric(trades.get('hybrid_score', trades.get('score', 0)), errors='coerce').fillna(0)
        trades['win'] = trades['result'].str.upper() == 'WIN'
        trades['score_bucket'] = pd.cut(trades['score'], bins=[0,40,55,70,101], labels=['<40','40-55','55-70','>70'])
        trades['direction'] = trades['direction'].fillna('?')
        trades['killzone'] = trades['killzone'].fillna('Unknown')

        groups = trades.groupby(['killzone','direction','score_bucket'], observed=True)
        results = []
        for name, grp in groups:
            if len(grp) < 3: continue
            results.append({
                "pattern": f"{name[0]} | {name[1]} | Score {name[2]}",
                "n": len(grp),
                "wr": round(float(grp['win'].mean()*100), 1),
                "avg_pnl": round(float(grp['pnl'].mean()), 2),
                "total_pnl": round(float(grp['pnl'].sum()), 2),
            })
        results.sort(key=lambda x: x['wr'], reverse=True)
        return {"patterns": results[:10]}
    except Exception as e:
        return {"error": str(e), "patterns": []}

# =============================================================================
# WAVE 3: Error Logging
# =============================================================================
_error_log: list = []

def _log_error(context: str, error: str):
    """Helper to log errors."""
    from datetime import datetime
    _error_log.append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "context": context,
        "error": error,
    })
    if len(_error_log) > 200:
        _error_log.pop(0)

# =============================================================================
# WAVE 3: Cache Management
# =============================================================================
_CAL_CACHE = {}

# =============================================================================
# ENDPOINT 1: SHAP Values
# =============================================================================
@app.get("/model/shap", tags=["AI Model"])
def get_shap_values():
    """Calculează SHAP values pentru ultimele 50 de predicții."""
    import pandas as pd, numpy as np, json, os
    model_path   = "/Users/mario/Desktop/Aladin/mario_xgb_model.json"
    journal_path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
    feat_path    = "/Users/mario/Desktop/Aladin/mario_features.json"

    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        return {"shap_values": [], "features": [], "error": "Model sau features lipsă"}

    try:
        import shap, xgboost as xgb

        with open(feat_path) as f:
            meta = json.load(f)
        features = meta.get("features", [])

        model = xgb.XGBClassifier()
        model.load_model(model_path)

        if os.path.exists(journal_path):
            df = pd.read_csv(journal_path, low_memory=False).tail(50)
            # Selectează doar coloanele care există și sunt features
            available = [f for f in features if f in df.columns]
            if len(available) < 3:
                return {"shap_values": [], "features": [], "error": "Prea puține features disponibile în jurnal"}
            X = df[available].apply(pd.to_numeric, errors='coerce').fillna(0)
        else:
            return {"shap_values": [], "features": [], "error": "Jurnal lipsă"}

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)

        # Dacă multiclass, ia clasa 1 (LONG/WIN)
        if isinstance(shap_vals, list):
            sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            sv = shap_vals

        # Mean absolute SHAP per feature
        mean_shap = np.abs(sv).mean(axis=0)
        result = sorted(
            [{"feature": f, "importance": round(float(v), 4)}
             for f, v in zip(available, mean_shap)],
            key=lambda x: x["importance"], reverse=True
        )[:15]

        return {"shap_values": result, "n_samples": len(X), "features": available[:15]}
    except Exception as e:
        return {"shap_values": [], "features": [], "error": str(e)}

# =============================================================================
# ENDPOINT 2: Correlations Matrix
# =============================================================================
@app.get("/stats/correlations", tags=["Stats"])
def get_correlations():
    """Corelații între variabilele numerice din jurnal."""
    import pandas as pd, numpy as np, os
    journal_path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
    if not os.path.exists(journal_path):
        return {"matrix": [], "labels": []}
    try:
        df = pd.read_csv(journal_path, low_memory=False)
        trades = df[df['pnl'].notna()].copy()
        num_cols = ['hybrid_score', 'pnl', 'risk_usd', 'rr', 'position_size']
        available = [c for c in num_cols if c in trades.columns]
        if len(available) < 2:
            return {"matrix": [], "labels": []}
        sub = trades[available].apply(pd.to_numeric, errors='coerce').dropna()
        corr = sub.corr().round(3)
        return {
            "matrix": corr.values.tolist(),
            "labels": available,
            "n": len(sub)
        }
    except Exception as e:
        return {"matrix": [], "labels": [], "error": str(e)}

# =============================================================================
# ENDPOINT 3: Anomalies Detection
# =============================================================================
@app.get("/stats/anomalies", tags=["Stats"])
def get_anomalies():
    """Detectează tranzacții anormale folosind Isolation Forest."""
    import pandas as pd, numpy as np, os
    journal_path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
    if not os.path.exists(journal_path):
        return {"anomalies": []}
    try:
        from sklearn.ensemble import IsolationForest
        df = pd.read_csv(journal_path, low_memory=False)
        trades = df[df['pnl'].notna() & df['result'].notna()].copy()
        trades['pnl'] = pd.to_numeric(trades['pnl'], errors='coerce')
        trades['score'] = pd.to_numeric(trades['hybrid_score'], errors='coerce')
        trades = trades.dropna(subset=['pnl', 'score'])

        if len(trades) < 10:
            return {"anomalies": []}

        X = trades[['pnl', 'score']].values
        clf = IsolationForest(contamination=0.1, random_state=42)
        preds = clf.fit_predict(X)

        anomalies = []
        for i, (pred, (_, row)) in enumerate(zip(preds, trades.iterrows())):
            if pred == -1:
                anomalies.append({
                    "timestamp": str(row.get('timestamp', '')),
                    "pnl": round(float(row['pnl']), 2),
                    "score": round(float(row['score']), 1),
                    "direction": str(row.get('direction', '')),
                    "reason": "PnL sau scor neobișnuit față de media istorică"
                })

        return {"anomalies": anomalies[-20:], "total": len(anomalies)}
    except Exception as e:
        return {"anomalies": [], "error": str(e)}

# =============================================================================
# ENDPOINT 4: News Sentiment
# =============================================================================
@app.get("/news/sentiment", tags=["Market"])
def get_news_sentiment():
    """Analizează sentimentul știrilor economice cu VADER."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

        events = _fetch_cal_events()
        if not events:
            return {"sentiment": "NEUTRU", "score": 0, "events": []}

        scores = []
        analyzed = []
        for ev in events[:10]:
            title = ev.get("title") or ev.get("event") or ev.get("name") or ""
            if not title:
                continue
            vs = analyzer.polarity_scores(title)
            scores.append(vs['compound'])
            analyzed.append({
                "title": title,
                "score": round(vs['compound'], 3),
                "sentiment": "POZITIV" if vs['compound'] > 0.05 else ("NEGATIV" if vs['compound'] < -0.05 else "NEUTRU")
            })

        avg = sum(scores) / len(scores) if scores else 0
        overall = "POZITIV" if avg > 0.05 else ("NEGATIV" if avg < -0.05 else "NEUTRU")

        return {
            "sentiment": overall,
            "score": round(avg, 3),
            "events": analyzed
        }
    except Exception as e:
        return {"sentiment": "NEUTRU", "score": 0, "error": str(e)}

# =============================================================================
# ENDPOINT 5: Support/Resistance levels (folosind pandas-ta)
# =============================================================================
@app.get("/market/levels", tags=["Market"])
def get_support_resistance():
    """Calculează nivele de suport/rezistență din datele QQQ."""
    import pandas as pd, numpy as np, os
    data_path = "/Users/mario/Desktop/Aladin/QQQ_10y_minutes.csv"
    if not os.path.exists(data_path):
        return {"supports": [], "resistances": [], "current": None}
    try:
        df = pd.read_csv(data_path, low_memory=False).tail(500)

        # Detectare coloane
        close_col = next((c for c in df.columns if 'close' in c.lower()), None)
        high_col  = next((c for c in df.columns if 'high' in c.lower()), None)
        low_col   = next((c for c in df.columns if 'low' in c.lower()), None)

        if not all([close_col, high_col, low_col]):
            return {"supports": [], "resistances": [], "current": None}

        closes = pd.to_numeric(df[close_col], errors='coerce').dropna().values
        highs  = pd.to_numeric(df[high_col],  errors='coerce').dropna().values
        lows   = pd.to_numeric(df[low_col],   errors='coerce').dropna().values

        current = float(closes[-1])

        # Pivot points simpli (swing highs/lows)
        def find_pivots(arr, n=5):
            pivots = []
            for i in range(n, len(arr)-n):
                if all(arr[i] >= arr[i-j] and arr[i] >= arr[i+j] for j in range(1, n+1)):
                    pivots.append(round(float(arr[i]), 2))
                elif all(arr[i] <= arr[i-j] and arr[i] <= arr[i+j] for j in range(1, n+1)):
                    pivots.append(round(float(arr[i]), 2))
            return pivots

        swing_highs = sorted([h for h in find_pivots(highs) if h > current], reverse=False)[:5]
        swing_lows  = sorted([l for l in find_pivots(lows)  if l < current], reverse=True)[:5]

        return {
            "current": current,
            "resistances": swing_highs[:3],
            "supports": swing_lows[:3],
        }
    except Exception as e:
        return {"supports": [], "resistances": [], "current": None, "error": str(e)}

# =============================================================================
# ENDPOINT 6: Optimization Engine (Optuna)
# =============================================================================
@app.post("/backtest/optimize", tags=["Backtest"])
def optimize_params(body: dict = None):
    """Optimizează parametrii strategiei folosind Optuna."""
    import optuna, pandas as pd, numpy as np, os
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    journal_path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
    if not os.path.exists(journal_path):
        return {"error": "Jurnal lipsă", "best_params": {}}

    try:
        df = pd.read_csv(journal_path, low_memory=False)
        trades = df[df['pnl'].notna() & df['result'].notna()].copy()
        trades['pnl']   = pd.to_numeric(trades['pnl'], errors='coerce')
        trades['score'] = pd.to_numeric(trades['hybrid_score'], errors='coerce')
        trades = trades.dropna(subset=['pnl', 'score'])

        if len(trades) < 10:
            return {"error": "Prea puține tranzacții", "best_params": {}}

        def objective(trial):
            score_min = trial.suggest_float("score_min", 40, 75)
            rr_min    = trial.suggest_float("rr_min", 1.0, 4.0)

            filtered = trades[trades['score'] >= score_min]
            if 'rr' in trades.columns:
                filtered = filtered[pd.to_numeric(filtered['rr'], errors='coerce').fillna(0) >= rr_min]

            if len(filtered) < 5:
                return -999

            total_pnl = filtered['pnl'].sum()
            wr = (filtered['pnl'] > 0).mean()
            return float(total_pnl * wr)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50, timeout=10)

        best = study.best_params
        best_value = round(study.best_value, 2)

        return {
            "best_params": {
                "score_min": round(best.get("score_min", 55), 1),
                "rr_min":    round(best.get("rr_min", 2.0), 2),
            },
            "best_score": best_value,
            "n_trials": len(study.trials),
            "message": f"Scor optim: {best.get('score_min', 55):.1f}%, R:R minim: {best.get('rr_min', 2):.2f}"
        }
    except Exception as e:
        return {"error": str(e), "best_params": {}}

# =============================================================================
# ENDPOINT 7: Compare Strategies
# =============================================================================
@app.get("/backtest/compare", tags=["Backtest"])
def compare_strategies():
    """Compară strategia 'toate trade-urile' vs 'trade-uri filtrate score>=55'."""
    import pandas as pd, os
    journal_path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
    if not os.path.exists(journal_path):
        return {"strategies": []}
    try:
        df = pd.read_csv(journal_path, low_memory=False)
        trades = df[df['pnl'].notna() & df['result'].notna()].copy()
        trades['pnl']   = pd.to_numeric(trades['pnl'], errors='coerce')
        trades['score'] = pd.to_numeric(trades['hybrid_score'], errors='coerce')
        trades['win']   = trades['result'].astype(str).str.upper() == 'WIN'

        def stats(subset, name):
            if len(subset) == 0:
                return {"name": name, "n": 0, "pnl": 0, "wr": 0, "avg_pnl": 0, "equity": []}
            wins   = subset[subset['win']]
            losses = subset[~subset['win']]
            avg_win  = wins['pnl'].mean()   if len(wins)   > 0 else 0
            avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
            equity = subset['pnl'].cumsum().tolist()
            # Downsample equity
            step = max(1, len(equity)//50)
            equity = equity[::step]
            return {
                "name":     name,
                "n":        len(subset),
                "pnl":      round(float(subset['pnl'].sum()), 2),
                "wr":       round(float(subset['win'].mean() * 100), 1),
                "avg_pnl":  round(float(subset['pnl'].mean()), 2),
                "avg_win":  round(float(avg_win), 2),
                "avg_loss": round(float(avg_loss), 2),
                "equity":   [round(v, 2) for v in equity],
            }

        all_trades     = stats(trades, "Toate trade-urile")
        filtered_55    = stats(trades[trades['score'] >= 55], "Score ≥ 55")
        filtered_60    = stats(trades[trades['score'] >= 60], "Score ≥ 60")

        return {"strategies": [all_trades, filtered_55, filtered_60]}
    except Exception as e:
        return {"strategies": [], "error": str(e)}

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

# =============================================================================
# TELEGRAM TOGGLE  (appended)
# =============================================================================
_telegram_enabled: bool = True

@app.post("/telegram/toggle", tags=["Notifications"])
def telegram_toggle(enabled: bool = True):
    """Activează / dezactivează notificările Telegram."""
    global _telegram_enabled
    _telegram_enabled = enabled
    try:
        import telegram_alerts as ta
        if hasattr(ta, "set_enabled"):
            ta.set_enabled(enabled)
    except Exception:
        pass
    return {"ok": True, "telegram_enabled": _telegram_enabled}

@app.get("/telegram/status", tags=["Notifications"])
def telegram_status():
    return {"telegram_enabled": _telegram_enabled}

# =============================================================================
# FLASH NEWS — titluri recente din economic calendar cache  (appended)
# =============================================================================
@app.get("/news/flash", tags=["Market"])
def flash_news():
    """Returnează ultimele știri (titluri) din calendar-ul economic cached."""
    events = _fetch_cal_events()
    headlines = []
    for ev in events[:20]:
        title = ev.get("title") or ev.get("event") or ev.get("name") or ""
        impact = ev.get("impact") or ev.get("importance") or ""
        currency = ev.get("currency") or ev.get("country") or ""
        time_str = ev.get("time") or ev.get("date") or ""
        if title:
            headlines.append({
                "title": title,
                "impact": str(impact).upper(),
                "currency": currency,
                "time": time_str,
            })
    return {"headlines": headlines}

# =============================================================================
# WAVE 3: Top Patterns Endpoint
# =============================================================================
@app.get("/stats/patterns", tags=["Stats"])
def get_patterns():
    """Top combinații direction+killzone+score_bucket cu cel mai bun win rate."""
    journal_path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
    if not os.path.exists(journal_path):
        return {"patterns": []}
    try:
        import pandas as pd
        df = pd.read_csv(journal_path, low_memory=False)
        trades = df[df['pnl'].notna() & df['result'].notna()].copy()
        trades['win'] = trades['result'].astype(str).str.upper() == 'WIN'
        trades['pnl'] = pd.to_numeric(trades['pnl'], errors='coerce')
        trades['score'] = pd.to_numeric(trades['hybrid_score'], errors='coerce')
        trades['score_bucket'] = pd.cut(trades['score'], bins=[0,40,55,70,101], labels=['0-40','40-55','55-70','70+'])
        trades['direction'] = trades['direction'].fillna('—')
        trades['kz'] = trades['killzone'].fillna('Unknown')

        patterns = []
        for (direction, kz, sbucket), group in trades.groupby(['direction','kz','score_bucket']):
            if len(group) < 3:
                continue
            wr = round(group['win'].mean() * 100, 1)
            avg_pnl = round(group['pnl'].mean(), 2)
            patterns.append({
                "pattern": f"{direction} | {kz} | Score {sbucket}",
                "n": len(group),
                "wr": wr,
                "avg_pnl": avg_pnl,
            })

        patterns.sort(key=lambda x: x['wr'], reverse=True)
        return {"patterns": patterns[:10]}
    except Exception as e:
        _log_error("get_patterns", str(e))
        return {"patterns": [], "error": str(e)}

# =============================================================================
# WAVE 3: Error Logs Endpoints
# =============================================================================
@app.get("/system/errors", tags=["System"])
def get_errors():
    """Return last 50 logged errors."""
    return {"errors": _error_log[-50:]}

@app.post("/system/errors/clear", tags=["System"])
def clear_errors():
    """Clear error log."""
    global _error_log
    _error_log = []
    return {"ok": True}

# =============================================================================
# WAVE 3: Cache Clear Endpoint
# =============================================================================
@app.post("/cache/clear", tags=["System"])
def clear_cache():
    """Clear internal cache."""
    global _CAL_CACHE
    _CAL_CACHE = {}
    return {"ok": True, "message": "Cache cleared"}
