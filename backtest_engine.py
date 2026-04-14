"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN BACKTEST ENGINE v1.0                                                 ║
║  Replay bar-cu-bar pe date reale (yfinance) + semnale Aladin pe OHLCV       ║
║                                                                              ║
║  Filtre active în backtest (nu necesită orderflow live):                     ║
║    • Bar Structure Quality (body/range ratio)                                ║
║    • Price Momentum Streak (4 bare consecutive)                              ║
║    • Volume Climax (2× average)                                              ║
║    • Liquidity Sweep (fake breakout + reversare)                             ║
║    • MSS / CHoCH (Market Structure Shift)                                    ║
║    • Equal Highs / Equal Lows                                                ║
║    • HTF Bias H4/H1 (calculat din barele disponibile)                        ║
║    • ATR-relative SL/TP                                                      ║
║                                                                              ║
║  Filtre inactive în backtest (necesită DOM/orderflow live):                  ║
║    • E1 DOM Bid/Ask ratio                                                    ║
║    • E5 POC Drift / E6 DOM Ratio Trend / E7 Delta Divergence                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("backtest_engine")

# ── Symbole suportate ─────────────────────────────────────────────────────────
SYMBOL_MAP = {
    "NQ":  "NQ=F",    # NASDAQ-100 E-mini
    "ES":  "ES=F",    # S&P 500 E-mini
    "MNQ": "MNQ=F",   # Micro NASDAQ
    "MES": "MES=F",   # Micro S&P
    "RTY": "RTY=F",   # Russell 2000 E-mini
    "YM":  "YM=F",    # Dow Jones E-mini
}

# Valoare per punct per contract
POINT_VALUE = {
    "NQ": 20.0, "MNQ": 2.0,
    "ES": 50.0, "MES": 5.0,
    "RTY": 50.0, "YM": 5.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# STATE — sesiunea curentă de backtest
# ═══════════════════════════════════════════════════════════════════════════════
class BacktestSession:
    def __init__(self):
        self.reset()

    def reset(self):
        self.df: Optional[pd.DataFrame] = None   # toate barele încărcate
        self.current_idx: int = 0                # index bar curent
        self.symbol: str = "NQ"
        self.timeframe: str = "5m"
        self.balance: float = 10000.0
        self.peak_balance: float = 10000.0

        # Poziție curentă (manual sau aladin)
        self.position: Optional[Dict] = None     # {dir, entry, sl, tp, size, source}

        # Jurnal trades
        self.trades: List[Dict] = []
        self.aladin_trades: List[Dict] = []      # ce ar fi dat Aladin

        # Semnale Aladin pre-calculate per bar
        self.signals: List[Optional[Dict]] = []  # parallel cu df rows


# Instanță globală — resetată la fiecare sesiune nouă
_bt = BacktestSession()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — yfinance
# ═══════════════════════════════════════════════════════════════════════════════
def load_data(symbol: str = "NQ", timeframe: str = "5m",
              days: int = 7) -> Tuple[bool, str]:
    """
    Descarcă date OHLCV din yfinance și inițializează sesiunea de backtest.
    Returnează (success, message).
    """
    global _bt
    try:
        import yfinance as yf

        ticker_sym = SYMBOL_MAP.get(symbol.upper(), f"{symbol}=F")

        # yfinance: interval 1m max 7 zile, 5m max 60 zile, 15m max 60 zile
        period_map = {"1m": f"{min(days, 7)}d", "5m": f"{min(days, 59)}d",
                      "15m": f"{min(days, 59)}d", "1h": f"{min(days, 730)}d",
                      "1d": f"{min(days, 3650)}d"}
        period = period_map.get(timeframe, f"{days}d")

        t = yf.Ticker(ticker_sym)
        df = t.history(period=period, interval=timeframe, auto_adjust=True)

        if df.empty:
            return False, f"Nu s-au găsit date pentru {ticker_sym}"

        # Curăță și redenumește coloanele
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.dropna()
        df = df[df["volume"] > 0]

        # Adaugă indicatori tehnici de bază
        df = _add_indicators(df)

        _bt.reset()
        _bt.df = df.reset_index()
        _bt.df.rename(columns={"index": "ts", "Datetime": "ts"}, inplace=True)
        if "Datetime" in _bt.df.columns:
            _bt.df.rename(columns={"Datetime": "ts"}, inplace=True)
        # Asigură că prima coloană e timestamp
        if _bt.df.columns[0] != "ts":
            ts_col = [c for c in _bt.df.columns if "time" in c.lower() or c == "ts"]
            if ts_col:
                _bt.df.rename(columns={ts_col[0]: "ts"}, inplace=True)

        _bt.symbol = symbol.upper()
        _bt.timeframe = timeframe
        _bt.current_idx = 50   # Start după 50 bare (pentru indicatori)
        _bt.signals = [None] * len(_bt.df)

        # Pre-calculăm semnalele Aladin pentru toate barele
        _precompute_signals()

        n = len(_bt.df)
        return True, f"✅ {n} bare {timeframe} {symbol} încărcate. Start la bara {_bt.current_idx}/{n}"

    except Exception as e:
        logger.error(f"load_data error: {e}", exc_info=True)
        return False, f"Eroare: {e}"


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adaugă ATR(14), volum mediu, și alte calcule de bază."""
    # True Range
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["prev_close"]),
                   abs(df["low"] - df["prev_close"]))
    )
    df["atr14"] = df["tr"].rolling(14).mean()
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    df.drop(columns=["prev_close", "tr"], inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL ENGINE — scoring OHLCV pentru fiecare bară
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_signal_at(idx: int) -> Optional[Dict]:
    """
    Calculează semnalul Aladin la bara idx.
    Returnează None dacă nu e semnal clar, sau {direction, score, sl, tp, entry}.
    """
    df = _bt.df
    if idx < 50 or idx >= len(df):
        return None

    window = df.iloc[max(0, idx-50):idx+1]   # 50 bare context + curentă
    bar = df.iloc[idx]

    try:
        o, h, l, c = float(bar.open), float(bar.high), float(bar.low), float(bar.close)
        vol  = float(bar.volume)
        atr  = float(bar.atr14) if not math.isnan(bar.atr14) else (h - l)
        vol_avg = float(bar.vol_avg20) if not math.isnan(bar.vol_avg20) else vol

        # ── DIRECȚIE BIAS — din HTF aproximativ (H4 = ultimele 48 bare de 5m) ──
        h4_bars = df.iloc[max(0, idx-48):idx]
        h1_bars = df.iloc[max(0, idx-12):idx]
        h4_bullish = (h4_bars["close"].iloc[-1] > h4_bars["close"].iloc[0]
                      if len(h4_bars) >= 2 else True)
        h1_bullish = (h1_bars["close"].iloc[-1] > h1_bars["close"].iloc[0]
                      if len(h1_bars) >= 2 else True)
        htf_long = h4_bullish and h1_bullish
        htf_short = not h4_bullish and not h1_bullish
        if not htf_long and not htf_short:
            return None   # HTF ambiguu — nu tranzacționăm

        trade_direction = "LONG" if htf_long else "SHORT"
        score = 0.50   # scor de bază

        bars5 = window.iloc[-6:-1]   # 5 bare anterioare (excl. curentă)

        # ── E2. BAR STRUCTURE QUALITY ─────────────────────────────────────────
        body = abs(c - o)
        rng  = h - l if h != l else 0.0001
        body_ratio = body / rng
        if body_ratio >= 0.65:
            bullish_bar = c > o
            if (bullish_bar and trade_direction == "LONG") or (not bullish_bar and trade_direction == "SHORT"):
                score += 0.05
            else:
                score -= 0.04
        elif body_ratio <= 0.20:
            score -= 0.04   # doji = incertitudine

        # ── E3. MOMENTUM STREAK — 4 bare consecutive aliniate ────────────────
        if len(bars5) >= 4:
            closes = list(bars5["close"].iloc[-4:])
            streak_bull = all(closes[i] < closes[i+1] for i in range(3))
            streak_bear = all(closes[i] > closes[i+1] for i in range(3))
            if streak_bull and trade_direction == "LONG":
                score += 0.06
            elif streak_bear and trade_direction == "SHORT":
                score += 0.06
            elif streak_bull and trade_direction == "SHORT":
                score -= 0.06
            elif streak_bear and trade_direction == "LONG":
                score -= 0.06

        # ── E4. VOLUME CLIMAX ────────────────────────────────────────────────
        if vol_avg > 0 and vol >= vol_avg * 2.0:
            bullish_bar = c > o
            if (bullish_bar and trade_direction == "LONG") or (not bullish_bar and trade_direction == "SHORT"):
                score += 0.08
            else:
                score -= 0.07

        # ── E8. LIQUIDITY SWEEP ───────────────────────────────────────────────
        if len(bars5) >= 5:
            prev_hi = float(bars5["high"].max())
            prev_lo = float(bars5["low"].min())
            if h > prev_hi and c < prev_hi:   # bearish sweep
                if trade_direction == "SHORT":
                    score += 0.09
                else:
                    score -= 0.09
            elif l < prev_lo and c > prev_lo:  # bullish sweep
                if trade_direction == "LONG":
                    score += 0.09
                else:
                    score -= 0.09

        # ── E9. MSS / CHoCH ───────────────────────────────────────────────────
        if len(window) >= 9:
            c_arr    = list(window["close"].iloc[-9:])
            h_arr    = list(window["high"].iloc[-9:])
            l_arr    = list(window["low"].iloc[-9:])
            rec_hi   = max(h_arr[-5:-1])
            rec_lo   = min(l_arr[-5:-1])
            trend_dn = c_arr[-2] < c_arr[-5]
            trend_up = c_arr[-2] > c_arr[-5]
            if trend_dn and c > rec_hi and trade_direction == "LONG":
                score += 0.07
            elif trend_up and c < rec_lo and trade_direction == "SHORT":
                score += 0.07
            elif trend_dn and c > rec_hi and trade_direction == "SHORT":
                score -= 0.06
            elif trend_up and c < rec_lo and trade_direction == "LONG":
                score -= 0.06

        # ── E10. EQUAL HIGHS / LOWS ───────────────────────────────────────────
        if len(bars5) >= 5:
            tol = atr * 0.10   # 10% din ATR ca toleranță
            h6  = list(bars5["high"])
            l6  = list(bars5["low"])
            max_hi = max(h6)
            min_lo = min(l6)
            eq_hi  = sum(1 for hv in h6 if abs(hv - max_hi) <= tol)
            eq_lo  = sum(1 for lv in l6 if abs(lv - min_lo) <= tol)
            if eq_hi >= 2 and h >= max_hi - tol:
                if trade_direction == "SHORT":
                    score += 0.05
                else:
                    score -= 0.05
            elif eq_lo >= 2 and l <= min_lo + tol:
                if trade_direction == "LONG":
                    score += 0.05
                else:
                    score -= 0.05

        # ── SCOR FINAL ────────────────────────────────────────────────────────
        score = max(0.0, min(score, 1.0))
        score_pct = round(score * 100, 1)

        # Prag minim: 62% pentru semnal valid
        if score_pct < 62.0:
            return None

        # ── SL / TP bazat pe ATR ─────────────────────────────────────────────
        rr     = 2.0
        sl_pts = round(atr * 1.2, 2)
        tp_pts = round(sl_pts * rr, 2)

        if trade_direction == "LONG":
            sl_px = round(c - sl_pts, 2)
            tp_px = round(c + tp_pts, 2)
        else:
            sl_px = round(c + sl_pts, 2)
            tp_px = round(c - tp_pts, 2)

        return {
            "direction": trade_direction,
            "score":     score_pct,
            "entry":     round(c, 2),
            "sl":        sl_px,
            "tp":        tp_px,
            "sl_pts":    round(sl_pts, 2),
            "tp_pts":    round(tp_pts, 2),
            "atr":       round(atr, 2),
            "bar_idx":   idx,
        }

    except Exception as e:
        logger.debug(f"Signal error at idx {idx}: {e}")
        return None


def _precompute_signals():
    """Pre-calculează semnalele Aladin pentru toate barele (rapid)."""
    if _bt.df is None:
        return
    n = len(_bt.df)
    _bt.signals = [None] * n
    for i in range(50, n):
        _bt.signals[i] = _compute_signal_at(i)
    total = sum(1 for s in _bt.signals if s is not None)
    logger.info(f"Pre-compute semnale: {total}/{n} bare cu semnal")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP — avansează o bară
# ═══════════════════════════════════════════════════════════════════════════════
def step_bar(n_bars: int = 1) -> Dict:
    """
    Avansează n_bars bare înainte.
    Returnează starea curentă: bara, semnalul Aladin, poziția activă, P&L.
    """
    if _bt.df is None:
        return {"error": "Nu există date încărcate. Apasă Load Data mai întâi."}

    # Avansăm
    _bt.current_idx = min(_bt.current_idx + n_bars, len(_bt.df) - 1)
    idx = _bt.current_idx

    bar = _bt.df.iloc[idx]
    ts  = str(bar.get("ts", bar.name) if "ts" in bar.index else bar.name)

    # Verificăm dacă poziția activă a atins SL/TP
    if _bt.position:
        _check_position_exit(idx)

    # Semnal Aladin la bara curentă
    signal = _bt.signals[idx]

    # Construim bara pentru grafic (ultimele 200 bare)
    start_chart = max(0, idx - 199)
    chart_bars = []
    for i in range(start_chart, idx + 1):
        r = _bt.df.iloc[i]
        ts_i = str(r.get("ts", i) if "ts" in r.index else i)
        sig_i = _bt.signals[i]
        chart_bars.append({
            "ts":    ts_i,
            "o":     round(float(r.open), 2),
            "h":     round(float(r.high), 2),
            "l":     round(float(r.low), 2),
            "c":     round(float(r.close), 2),
            "v":     int(r.volume),
            "signal": sig_i["direction"][0] if sig_i else None,   # "B"/"S"
            "score":  sig_i["score"] if sig_i else None,
        })

    pnl_total = sum(t.get("pnl", 0) for t in _bt.trades)

    return {
        "idx":          idx,
        "total_bars":   len(_bt.df),
        "ts":           ts,
        "bar": {
            "o": round(float(bar.open), 2),
            "h": round(float(bar.high), 2),
            "l": round(float(bar.low), 2),
            "c": round(float(bar.close), 2),
            "v": int(bar.volume),
            "atr": round(float(bar.atr14), 2) if not math.isnan(bar.atr14) else 0,
        },
        "signal":       signal,       # Aladin signal (dacă există)
        "position":     _bt.position, # poziție activă curentă
        "trades":       _bt.trades[-10:],  # ultimele 10 trades
        "pnl_total":    round(pnl_total, 2),
        "balance":      round(_bt.balance, 2),
        "chart_bars":   chart_bars,
        "done":         idx >= len(_bt.df) - 1,
    }


def _check_position_exit(idx: int):
    """Verifică dacă bara curentă a atins SL sau TP."""
    if not _bt.position:
        return
    pos = _bt.position
    bar = _bt.df.iloc[idx]
    h, l = float(bar.high), float(bar.low)
    dir_ = pos["direction"]
    entry = pos["entry"]
    sl = pos["sl"]
    tp = pos["tp"]
    size = pos.get("size", 1)
    pv   = POINT_VALUE.get(_bt.symbol, 20.0)

    hit_sl = (dir_ == "LONG" and l <= sl) or (dir_ == "SHORT" and h >= sl)
    hit_tp = (dir_ == "LONG" and h >= tp) or (dir_ == "SHORT" and l <= tp)

    if hit_tp or hit_sl:
        exit_px = tp if hit_tp else sl
        pnl_pts = (exit_px - entry) if dir_ == "LONG" else (entry - exit_px)
        pnl_usd = round(pnl_pts * pv * size, 2)
        _bt.balance += pnl_usd
        _bt.peak_balance = max(_bt.peak_balance, _bt.balance)

        trade_rec = {
            "dir":    dir_,
            "entry":  entry,
            "exit":   exit_px,
            "sl":     sl,
            "tp":     tp,
            "result": "WIN" if hit_tp else "LOSS",
            "pnl":    pnl_usd,
            "pnl_pts": round(pnl_pts, 2),
            "source": pos.get("source", "manual"),
            "bar_in": pos.get("bar_in", idx),
            "bar_out": idx,
        }
        _bt.trades.append(trade_rec)
        if pos.get("source") == "aladin":
            _bt.aladin_trades.append(trade_rec)
        _bt.position = None


# ═══════════════════════════════════════════════════════════════════════════════
# PLACE TRADE — manual trade
# ═══════════════════════════════════════════════════════════════════════════════
def place_trade(direction: str, source: str = "manual", size: int = 1) -> Dict:
    """
    Deschide o poziție la prețul de close al barei curente.
    source: "manual" | "aladin"
    """
    if _bt.df is None:
        return {"ok": False, "error": "Nu există date"}
    if _bt.position:
        return {"ok": False, "error": "Există deja o poziție deschisă. Închide-o mai întâi."}

    idx = _bt.current_idx
    bar = _bt.df.iloc[idx]
    c   = float(bar.close)
    atr = float(bar.atr14) if not math.isnan(bar.atr14) else (float(bar.high) - float(bar.low))

    sl_pts = round(atr * 1.2, 2)
    tp_pts = round(sl_pts * 2.0, 2)
    sl = round(c - sl_pts, 2) if direction == "LONG" else round(c + sl_pts, 2)
    tp = round(c + tp_pts, 2) if direction == "LONG" else round(c - tp_pts, 2)

    _bt.position = {
        "direction": direction,
        "entry":  c,
        "sl":     sl,
        "tp":     tp,
        "size":   size,
        "source": source,
        "bar_in": idx,
    }
    return {"ok": True, "position": _bt.position}


def close_position() -> Dict:
    """Închide poziția la prețul de close al barei curente."""
    if not _bt.position:
        return {"ok": False, "error": "Nu există poziție deschisă"}

    idx = _bt.current_idx
    bar = _bt.df.iloc[idx]
    c   = float(bar.close)
    pos = _bt.position
    pv  = POINT_VALUE.get(_bt.symbol, 20.0)

    pnl_pts = (c - pos["entry"]) if pos["direction"] == "LONG" else (pos["entry"] - c)
    pnl_usd = round(pnl_pts * pv * pos.get("size", 1), 2)
    _bt.balance += pnl_usd
    _bt.peak_balance = max(_bt.peak_balance, _bt.balance)

    trade_rec = {
        "dir":    pos["direction"],
        "entry":  pos["entry"],
        "exit":   round(c, 2),
        "sl":     pos["sl"],
        "tp":     pos["tp"],
        "result": "WIN" if pnl_usd > 0 else "LOSS",
        "pnl":    pnl_usd,
        "pnl_pts": round(pnl_pts, 2),
        "source": pos.get("source", "manual"),
        "bar_in": pos.get("bar_in", idx),
        "bar_out": idx,
    }
    _bt.trades.append(trade_rec)
    _bt.position = None
    return {"ok": True, "trade": trade_rec, "pnl": pnl_usd, "balance": round(_bt.balance, 2)}


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY — statistici finale
# ═══════════════════════════════════════════════════════════════════════════════
def get_summary() -> Dict:
    """Returnează statisticile complete ale sesiunii de backtest."""
    trades = _bt.trades
    if not trades:
        return {"trades": 0, "message": "Nu există trades înregistrate"}

    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    pnl    = sum(t["pnl"] for t in trades)
    avg_win  = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    pf = abs(avg_win * len(wins)) / abs(avg_loss * len(losses)) if losses and avg_loss != 0 else 99.0

    # Max drawdown
    running = 10000.0
    peak = 10000.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x.get("bar_out", 0)):
        running += t["pnl"]
        peak     = max(peak, running)
        dd       = (peak - running) / peak * 100 if peak > 0 else 0
        max_dd   = max(max_dd, dd)

    manual = [t for t in trades if t.get("source") == "manual"]
    aladin = [t for t in trades if t.get("source") == "aladin"]

    def stats(lst):
        if not lst: return {}
        w = [t for t in lst if t["result"] == "WIN"]
        l = [t for t in lst if t["result"] == "LOSS"]
        return {
            "trades":   len(lst),
            "wins":     len(w),
            "losses":   len(l),
            "win_rate": round(len(w) / len(lst) * 100, 1) if lst else 0,
            "pnl":      round(sum(t["pnl"] for t in lst), 2),
            "avg_win":  round(sum(t["pnl"] for t in w) / len(w), 2) if w else 0,
            "avg_loss": round(sum(t["pnl"] for t in l) / len(l), 2) if l else 0,
        }

    return {
        "total_trades":    len(trades),
        "wins":            len(wins),
        "losses":          len(losses),
        "win_rate":        round(len(wins) / len(trades) * 100, 1),
        "pnl_total":       round(pnl, 2),
        "avg_win":         round(avg_win, 2),
        "avg_loss":        round(avg_loss, 2),
        "profit_factor":   round(pf, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "balance":         round(_bt.balance, 2),
        "manual":          stats(manual),
        "aladin":          stats(aladin),
        "trade_log":       trades,
    }


def get_state() -> Dict:
    """Status curent al sesiunii de backtest."""
    if _bt.df is None:
        return {"loaded": False}
    return {
        "loaded":      True,
        "symbol":      _bt.symbol,
        "timeframe":   _bt.timeframe,
        "total_bars":  len(_bt.df),
        "current_idx": _bt.current_idx,
        "position":    _bt.position,
        "balance":     round(_bt.balance, 2),
        "trades":      len(_bt.trades),
        "pnl":         round(sum(t.get("pnl", 0) for t in _bt.trades), 2),
    }


def reset_session() -> Dict:
    """Reset complet — păstrează datele, resetează trades și poziția."""
    _bt.balance     = 10000.0
    _bt.peak_balance= 10000.0
    _bt.position    = None
    _bt.trades      = []
    _bt.aladin_trades = []
    _bt.current_idx = 50
    return {"ok": True, "message": "Sesiune resetată. Datele sunt păstrate."}
