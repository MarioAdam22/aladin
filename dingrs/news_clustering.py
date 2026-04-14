"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN — NEWS CONVERSATION ENGINE v1.0                                     ║
║  Agentul Fundamental: "Conversația Știrilor"                                ║
║                                                                              ║
║  Logică:                                                                     ║
║    1. Încarcă calendarul economic (CSV local sau GitHub + cache)             ║
║    2. Grupează știrile în ferestre de 30 minute (clustering temporal)        ║
║    3. Calculează deviația: (Actual - Forecast) / |Forecast|                 ║
║    4. Detectează corelații între tipuri de știri (CPI+Core CPI, NFP+Unemp.) ║
║    5. Calculează scorul cumulat al "conversației" pentru fiecare fereastră  ║
║    6. Returnează impact_score [0,1] și context pentru Agentul Executiv      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("news_clustering")

# ─── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR    = Path(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE   = CACHE_DIR / "aladin_news_cache.parquet"
CSV_BACKUP   = CACHE_DIR / "economic_calendar.csv"

WINDOW_MIN   = 30    # Fereastră de grupare știri (minute)
LOOKBACK_MIN = 120   # Cât timp în urmă verificăm știri relevante
LOOKAHEAD_MIN = 60   # Cât timp în față verificăm știri iminente

# ─── Greutăți Impact Știri ────────────────────────────────────────────────────
IMPACT_WEIGHTS = {"High": 1.0, "Medium": 0.5, "Low": 0.2, "": 0.1}

# ─── Corelații Cunoscute (știri care "vorbesc" între ele) ─────────────────────
# Dacă apar împreună în aceeași fereastră → impact cumulat multiplicat
CORRELATED_GROUPS = [
    # Inflație
    {"CPI", "Core CPI", "PPI", "Core PPI", "PCE", "Core PCE"},
    # Piața muncii
    {"Non-Farm Payrolls", "NFP", "Unemployment Rate", "Jobless Claims", "ADP"},
    # Fed / Dobânzi
    {"Fed Funds Rate", "FOMC", "Fed Minutes", "Federal Reserve"},
    # GDP / Creștere
    {"GDP", "Retail Sales", "Industrial Production", "ISM Manufacturing"},
    # Piața imobiliară
    {"Housing Starts", "Building Permits", "Existing Home Sales", "New Home Sales"},
]

# ─── Surse Calendar Economic ──────────────────────────────────────────────────
# Sursa 1: ForexFactory calendar (JSON, 2010-prezent, cel mai complet)
# → descarcă manual cu: curl -L <url> -o ~/Desktop/Aladin/data/historical_news.json
FOREXFACTORY_LOCAL = CACHE_DIR / "data" / "historical_news.json"

# Sursa 2: GitHub CSV backup (auto-descărcat)
CALENDAR_SOURCES = [
    "https://raw.githubusercontent.com/mdeverna/economic_calendar/master/data/calendar_raw.csv",
    "https://cdn.jsdelivr.net/gh/mdeverna/economic_calendar@master/data/calendar_raw.csv",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. ÎNCĂRCARE DATE
# ══════════════════════════════════════════════════════════════════════════════

def load_calendar(force_refresh: bool = False) -> pd.DataFrame:
    """
    Încarcă calendarul economic. Ordinea priorităților:
      1. ForexFactory JSON local (~/Desktop/Aladin/data/historical_news.json)
         → cel mai complet, 2010-prezent, descărcat manual cu curl
      2. Cache Parquet local (dacă < 6 ore vechime)
      3. GitHub CSV (auto-descărcat, mai puține date istorice)
      4. CSV backup local
    """
    # ── Prioritate 1: ForexFactory JSON local (dacă există) ──────────────────
    if FOREXFACTORY_LOCAL.exists():
        try:
            df = _load_forexfactory_json(FOREXFACTORY_LOCAL)
            if df is not None and len(df) > 100:
                log.info(f"🏆 ForexFactory JSON: {len(df):,} știri din {FOREXFACTORY_LOCAL.name}")
                # Salvează cache actualizat
                try:
                    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(CACHE_FILE, index=False)
                except Exception:
                    pass
                return df
        except Exception as e:
            log.warning(f"ForexFactory JSON load error: {e}")

    # ── Prioritate 2: Cache Parquet valid ─────────────────────────────────────
    if not force_refresh and CACHE_FILE.exists():
        age_hours = (datetime.now().timestamp() - CACHE_FILE.stat().st_mtime) / 3600
        if age_hours < 6:
            try:
                df = pd.read_parquet(CACHE_FILE)
                log.info(f"📂 Calendar din cache: {len(df):,} știri (vârstă: {age_hours:.1f}h)")
                return df
            except Exception:
                pass

    # ── Prioritate 3: Descarcă din GitHub ─────────────────────────────────────
    df = None
    for url in CALENDAR_SOURCES:
        try:
            log.info(f"🌐 Descarcă calendar din: {url[:60]}...")
            df = pd.read_csv(url, low_memory=False)
            log.info(f"✅ {len(df):,} știri descărcate")
            break
        except Exception as e:
            log.warning(f"   Sursă eșuată: {e}")

    # ── Prioritate 4: CSV backup local ────────────────────────────────────────
    if df is None and CSV_BACKUP.exists():
        df = pd.read_csv(CSV_BACKUP, low_memory=False)
        log.warning(f"⚠️  Fallback la CSV local: {len(df):,} știri")

    if df is None:
        log.error("❌ Nu s-a putut încărca calendarul. Returnez date goale.")
        return pd.DataFrame(columns=["datetime", "event", "currency", "impact",
                                     "actual", "forecast", "previous"])

    df = _normalize_calendar(df)

    # Salvează cache
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_FILE, index=False)
        df.to_csv(CSV_BACKUP, index=False)
        log.info(f"💾 Calendar salvat în cache: {CACHE_FILE}")
    except Exception:
        pass

    return df


def _load_forexfactory_json(path: Path) -> Optional[pd.DataFrame]:
    """
    Parsează formatul JSON de la ForexFactory calendar (coder-no-name/forexfactory-calendar).

    Structura JSON așteptată (array de obiecte):
    [
      {
        "date": "Jan 01, 2024",
        "time": "8:30am",
        "currency": "USD",
        "impact": "High",           // "High" / "Medium" / "Low" / "Non-Economic"
        "event": "Non-Farm Payrolls",
        "actual": "256K",
        "forecast": "160K",
        "previous": "212K"
      }, ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        # Uneori e wrapat în {"data": [...]}
        if isinstance(raw, dict):
            raw = raw.get("data", raw.get("events", raw.get("calendar", [])))

    if not raw:
        return None

    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        # Parsare dată + oră
        date_str = str(item.get("date", "") or item.get("Date", ""))
        time_str = str(item.get("time", "") or item.get("Time", ""))
        try:
            if time_str.lower() in ("all day", "tentative", ""):
                dt_str = date_str
            else:
                dt_str = f"{date_str} {time_str}"
            dt = pd.to_datetime(dt_str, errors="coerce", utc=False)
        except Exception:
            dt = pd.NaT

        rows.append({
            "datetime": dt,
            "event":    str(item.get("event",    item.get("Event",    item.get("title", "")))),
            "currency": str(item.get("currency", item.get("Currency", item.get("country", "USD")))),
            "impact":   str(item.get("impact",   item.get("Impact",   item.get("importance", "")))),
            "actual":   str(item.get("actual",   item.get("Actual",   ""))),
            "forecast": str(item.get("forecast", item.get("Forecast", ""))),
            "previous": str(item.get("previous", item.get("Previous", ""))),
        })

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Filtrează USD + curăță impactul
    df = df[df["currency"].str.upper().isin(["USD", "US", ""])]
    df["impact"] = df["impact"].str.replace("Non-Economic", "Low").str.replace("Holiday", "Low")

    return df.sort_values("datetime").reset_index(drop=True)


def _normalize_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizează coloanele indiferent de sursa CSV."""
    # Mapare coloane posibile → standardizate
    col_map = {
        # Timp
        "date": "date", "time": "time", "datetime": "datetime",
        "Date": "date", "Time": "time", "DateTime": "datetime",
        "timestamp": "datetime",
        # Eveniment
        "event": "event", "Event": "event", "title": "event", "name": "event",
        "indicator": "event",
        # Monedă / Țară
        "currency": "currency", "Currency": "currency",
        "country": "currency", "Country": "currency",
        # Impact
        "impact": "impact", "Impact": "impact",
        "importance": "impact", "Importance": "impact",
        # Valori
        "actual": "actual", "Actual": "actual",
        "forecast": "forecast", "Forecast": "forecast", "consensus": "forecast",
        "previous": "previous", "Previous": "previous", "prior": "previous",
    }

    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})

    # Construiește coloana datetime
    if "datetime" not in df.columns:
        if "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                errors="coerce",
                utc=False,
            )
        elif "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["datetime"] = pd.NaT

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)

    # Asigură coloanele necesare
    for col in ["event", "currency", "impact", "actual", "forecast", "previous"]:
        if col not in df.columns:
            df[col] = ""

    # Filtrează doar USD (FX relevanți pentru NQ/ES)
    if "currency" in df.columns:
        df = df[df["currency"].astype(str).str.upper().isin(["USD", "US", ""])]

    # Filtrează rânduri fără dată
    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. CALCULUL DEVIAȚIEI
# ══════════════════════════════════════════════════════════════════════════════

def compute_deviation(actual: any, forecast: any) -> float:
    """
    Calculează deviația normalizată: (Actual - Forecast) / |Forecast|
    Returnează 0.0 dacă datele lipsesc.
    Semn: pozitiv = better-than-expected, negativ = worse-than-expected
    """
    try:
        a = _parse_number(str(actual))
        f = _parse_number(str(forecast))
        if a is None or f is None:
            return 0.0
        if abs(f) < 1e-10:
            return 0.5 if a > 0 else -0.5
        return float(np.clip((a - f) / abs(f), -3.0, 3.0))
    except Exception:
        return 0.0


def _parse_number(s: str) -> Optional[float]:
    """Parsează string numeric: '3.5%', '-12K', '1.2M' → float"""
    s = s.strip()
    if s in ("", "N/A", "None", "nan", "-", "?"):
        return None
    multiplier = 1.0
    if s.endswith("K") or s.endswith("k"):
        multiplier = 1_000; s = s[:-1]
    elif s.endswith("M") or s.endswith("m"):
        multiplier = 1_000_000; s = s[:-1]
    elif s.endswith("B") or s.endswith("b"):
        multiplier = 1_000_000_000; s = s[:-1]
    s = s.replace("%", "").replace(",", "").strip()
    try:
        return float(s) * multiplier
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. DETECTARE CORELAȚII
# ══════════════════════════════════════════════════════════════════════════════

def get_correlation_multiplier(events_in_window: List[str]) -> float:
    """
    Dacă 2+ știri din același grup corelat apar în fereastră → impact amplificat.
    Maximum multiplicator: 2.5x (3+ știri corelate).
    """
    best_mult = 1.0
    for group in CORRELATED_GROUPS:
        matches = sum(
            1 for ev in events_in_window
            if any(kw.lower() in ev.lower() for kw in group)
        )
        if matches >= 3:
            best_mult = max(best_mult, 2.5)
        elif matches == 2:
            best_mult = max(best_mult, 1.7)

    return best_mult


def get_event_category(event_name: str) -> str:
    """Categorisează o știre pentru context semantic."""
    e = event_name.lower()
    if any(k in e for k in ["cpi", "ppi", "pce", "inflation", "price index"]):
        return "INFLATION"
    if any(k in e for k in ["nfp", "payroll", "unemployment", "jobless", "adp"]):
        return "LABOR"
    if any(k in e for k in ["fed", "fomc", "interest rate", "funds rate"]):
        return "MONETARY"
    if any(k in e for k in ["gdp", "retail", "industrial", "ism", "pmi"]):
        return "GROWTH"
    if any(k in e for k in ["housing", "home", "building", "construction"]):
        return "HOUSING"
    if any(k in e for k in ["trade", "deficit", "export", "import"]):
        return "TRADE"
    return "OTHER"


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLUSTERE TEMPORALE (30-minute windows)
# ══════════════════════════════════════════════════════════════════════════════

def build_time_windows(df: pd.DataFrame, window_minutes: int = WINDOW_MIN) -> List[Dict]:
    """
    Grupează știrile în ferestre temporale de N minute.
    Returnează lista de clustere cu scorul calculat.
    """
    if df.empty:
        return []

    df = df.copy()
    df["window_start"] = df["datetime"].dt.floor(f"{window_minutes}min")

    clusters = []
    for window_start, group in df.groupby("window_start"):
        cluster = _analyze_cluster(group, window_start)
        if cluster:
            clusters.append(cluster)

    clusters.sort(key=lambda x: x["window_start"])
    return clusters


def _analyze_cluster(group: pd.DataFrame, window_start) -> Optional[Dict]:
    """Analizează un cluster de știri dintr-o fereastră temporală."""
    if len(group) == 0:
        return None

    events_list = []
    total_raw_impact = 0.0
    total_deviation  = 0.0
    categories_seen  = set()
    has_data         = False

    for _, row in group.iterrows():
        event_name = str(row.get("event", "Unknown"))
        impact_str = str(row.get("impact", "Low"))
        actual     = row.get("actual", "")
        forecast   = row.get("forecast", "")
        previous   = row.get("previous", "")

        impact_w   = IMPACT_WEIGHTS.get(impact_str, 0.1)
        deviation  = compute_deviation(actual, forecast)
        category   = get_event_category(event_name)

        # Dacă avem Actual, știrea e publicată (nu e viitoare)
        is_released = bool(actual and str(actual) not in ("", "N/A", "nan", "-"))
        if is_released:
            has_data = True

        total_raw_impact += impact_w
        total_deviation  += deviation * impact_w  # Deviație ponderată cu importanța
        categories_seen.add(category)

        events_list.append({
            "event":      event_name,
            "impact":     impact_str,
            "impact_w":   round(impact_w, 2),
            "actual":     str(actual),
            "forecast":   str(forecast),
            "previous":   str(previous),
            "deviation":  round(deviation, 3),
            "category":   category,
            "released":   is_released,
        })

    # Corelare multiplier
    event_names = [e["event"] for e in events_list]
    corr_mult   = get_correlation_multiplier(event_names)

    # Impact score final [0, 1]
    # Formula: raw_impact × correlation × |normalized deviation| → clampat la [0,1]
    n = max(len(events_list), 1)
    avg_impact = total_raw_impact / n
    avg_dev    = abs(total_deviation / n) if has_data else 0.0
    avg_dev    = min(avg_dev, 1.0)  # Cap la 1.0

    # Score: impact structural + deviation confirmată de date
    raw_score     = avg_impact * (0.4 + 0.6 * avg_dev) * corr_mult
    impact_score  = float(np.clip(raw_score, 0.0, 1.0))

    # Sentiment direcțional: pozitiv = bullish USD (bearish NQ short-term)
    direction_sentiment = np.sign(total_deviation)  # +1=beat, -1=miss, 0=neutral

    # Alerta dacă scor > 0.6 (eveniment cu impact mare)
    alert_level = "BLACKOUT" if impact_score > 0.8 else \
                  "HIGH"     if impact_score > 0.6 else \
                  "MEDIUM"   if impact_score > 0.35 else "LOW"

    return {
        "window_start":        window_start.isoformat(),
        "window_end":          (window_start + timedelta(minutes=WINDOW_MIN)).isoformat(),
        "events":              events_list,
        "n_events":            len(events_list),
        "categories":          list(categories_seen),
        "corr_multiplier":     round(corr_mult, 2),
        "avg_deviation":       round(total_deviation / n, 3),
        "impact_score":        round(impact_score, 4),
        "direction_sentiment": int(direction_sentiment),
        "has_released_data":   has_data,
        "alert_level":         alert_level,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. QUERY PRINCIPAL — ce se întâmplă acum?
# ══════════════════════════════════════════════════════════════════════════════

class NewsConversationEngine:
    """
    Motorul principal pentru Agentul Fundamental.
    Menține calendarul în memorie și răspunde la query-uri în timp real.
    """

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._clusters: List[Dict]        = []
        self._last_refresh: float         = 0.0

    def load(self, force_refresh: bool = False):
        """Încarcă sau reîncarcă calendarul."""
        self._df       = load_calendar(force_refresh)
        self._clusters = build_time_windows(self._df)
        self._last_refresh = datetime.now().timestamp()
        log.info(f"✅ NewsEngine: {len(self._df):,} știri, {len(self._clusters)} clustere")

    def _ensure_loaded(self):
        """Auto-reload la 6 ore."""
        age = datetime.now().timestamp() - self._last_refresh
        if self._df is None or age > 21600:
            self.load()

    def get_current_context(self, ts: Optional[str] = None) -> Dict:
        """
        Returnează contextul complet de știri pentru un timestamp dat.
        Dacă ts=None, folosește ora curentă.

        Returnează:
          - active_cluster:    clusterul activ acum (dacă există)
          - upcoming_cluster:  următorul cluster în 60 min
          - past_cluster:      ultimul cluster publicat
          - impact_score:      scorul maxim din clustere relevante [0,1]
          - trading_advice:    recomandare: TRADE / CAUTION / BLACKOUT
          - summary:           text descriptiv pentru log
        """
        self._ensure_loaded()

        now = pd.Timestamp(ts) if ts else pd.Timestamp.now()

        window_start = now.floor(f"{WINDOW_MIN}min")
        window_end   = window_start + timedelta(minutes=WINDOW_MIN)

        # Clustere relevante
        active_cluster   = None
        upcoming_cluster = None
        past_cluster     = None

        for cluster in self._clusters:
            cstart = pd.Timestamp(cluster["window_start"])
            cend   = pd.Timestamp(cluster["window_end"])

            if cstart <= now < cend:
                active_cluster = cluster
            elif now <= cstart <= now + timedelta(minutes=LOOKAHEAD_MIN):
                if upcoming_cluster is None:
                    upcoming_cluster = cluster
            elif cend <= now and (now - cend).seconds <= LOOKBACK_MIN * 60:
                past_cluster = cluster

        # Impact score maxim din clustere relevante
        max_score = max(
            [c["impact_score"] for c in [active_cluster, upcoming_cluster] if c],
            default=0.0
        )

        # Sfat de trading
        if active_cluster and active_cluster["alert_level"] in ("BLACKOUT", "HIGH"):
            trading_advice = "BLACKOUT"
        elif upcoming_cluster and upcoming_cluster["impact_score"] > 0.6 and \
             (pd.Timestamp(upcoming_cluster["window_start"]) - now).seconds < 900:  # 15 min
            trading_advice = "CAUTION"
        elif max_score > 0.35:
            trading_advice = "CAUTION"
        else:
            trading_advice = "TRADE"

        # Sentiment direcțional al știrilor recente
        sentiment = 0
        if past_cluster and past_cluster["has_released_data"]:
            sentiment = past_cluster["direction_sentiment"]

        # Summary text
        parts = []
        if active_cluster:
            names = [e["event"] for e in active_cluster["events"][:2]]
            parts.append(f"ACTIV: {', '.join(names)} (impact={active_cluster['impact_score']:.2f})")
        if upcoming_cluster:
            mins_away = int((pd.Timestamp(upcoming_cluster["window_start"]) - now).seconds / 60)
            names = [e["event"] for e in upcoming_cluster["events"][:2]]
            parts.append(f"ÎN {mins_away}min: {', '.join(names)} (impact={upcoming_cluster['impact_score']:.2f})")
        if not parts:
            parts.append("Nicio știre cu impact major în fereastră")

        return {
            "timestamp":         now.isoformat(),
            "trading_advice":    trading_advice,
            "impact_score":      round(max_score, 4),
            "direction_sentiment": int(sentiment),
            "active_cluster":    active_cluster,
            "upcoming_cluster":  upcoming_cluster,
            "past_cluster":      past_cluster,
            "summary":           " | ".join(parts),
        }

    def get_news_score(self, ts: Optional[str] = None) -> float:
        """
        Returnează scorul simplu [0,1] pentru utilizare în formula Aladin.
        0 = liniște, 1 = blackout complet.
        """
        ctx = self.get_current_context(ts)
        return ctx["impact_score"]

    def get_trading_multiplier(self, ts: Optional[str] = None) -> Tuple[float, str]:
        """
        Returnează multiplier pentru scorul de trading:
          - BLACKOUT → 0.0 (nu tranzacționa)
          - CAUTION  → 0.5 (reduce sizing)
          - TRADE    → 1.0 (normal)
        """
        ctx = self.get_current_context(ts)
        advice = ctx["trading_advice"]
        mult = {"BLACKOUT": 0.0, "CAUTION": 0.5, "TRADE": 1.0}.get(advice, 1.0)
        return mult, ctx["summary"]

    def get_upcoming_events(self, hours_ahead: float = 24) -> List[Dict]:
        """Returnează lista de știri în următoarele N ore."""
        self._ensure_loaded()
        now     = pd.Timestamp.now()
        cutoff  = now + timedelta(hours=hours_ahead)
        results = []
        for cluster in self._clusters:
            cstart = pd.Timestamp(cluster["window_start"])
            if now <= cstart <= cutoff:
                results.append(cluster)
        return results

    def get_daily_risk_profile(self, date: Optional[str] = None) -> Dict:
        """
        Profilul de risc al unei zile întregi.
        Util pentru a decide dimineața dacă ziua e tradeabilă.
        """
        self._ensure_loaded()
        target_date = pd.Timestamp(date).date() if date else datetime.now().date()

        day_clusters = [
            c for c in self._clusters
            if pd.Timestamp(c["window_start"]).date() == target_date
        ]

        if not day_clusters:
            return {"date": str(target_date), "risk_level": "LOW", "score": 0.0, "events": []}

        max_score = max(c["impact_score"] for c in day_clusters)
        total_high = sum(1 for c in day_clusters if c["alert_level"] in ("HIGH", "BLACKOUT"))
        risk_level = "BLACKOUT" if max_score > 0.8 else \
                     "HIGH"     if max_score > 0.6 else \
                     "MEDIUM"   if max_score > 0.3 else "LOW"

        all_events = []
        for c in day_clusters:
            for e in c["events"]:
                all_events.append({
                    "time":   c["window_start"][11:16],
                    "event":  e["event"],
                    "impact": e["impact"],
                    "score":  c["impact_score"],
                })

        return {
            "date":       str(target_date),
            "risk_level": risk_level,
            "score":      round(max_score, 4),
            "n_high_windows": total_high,
            "events":     all_events,
        }


# ─── Singleton global ─────────────────────────────────────────────────────────
_news_engine: Optional[NewsConversationEngine] = None

def get_news_engine() -> NewsConversationEngine:
    """Returnează instanța singleton a motorului de știri."""
    global _news_engine
    if _news_engine is None:
        _news_engine = NewsConversationEngine()
        try:
            _news_engine.load()
        except Exception as e:
            log.warning(f"News engine load error: {e}")
    return _news_engine


# ══════════════════════════════════════════════════════════════════════════════
# CLI / TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    engine = NewsConversationEngine()
    engine.load()

    print("\n" + "=" * 70)
    print("🗞️  ALADIN NEWS CONVERSATION ENGINE — Test")
    print("=" * 70)

    # Context curent
    ctx = engine.get_current_context()
    print(f"\n📍 Timestamp: {ctx['timestamp'][:19]}")
    print(f"📊 Impact Score: {ctx['impact_score']:.2%}")
    print(f"🚦 Trading Advice: {ctx['trading_advice']}")
    print(f"📝 Summary: {ctx['summary']}")

    if ctx["active_cluster"]:
        print(f"\n⚡ CLUSTER ACTIV:")
        for e in ctx["active_cluster"]["events"]:
            print(f"   {e['event']:35s} Impact={e['impact']:6s} Dev={e['deviation']:+.3f}")

    # Profilul zilei
    print(f"\n📅 PROFIL ZI:")
    profile = engine.get_daily_risk_profile()
    print(f"   Risk: {profile['risk_level']}  Score: {profile['score']:.2%}  High windows: {profile['n_high_windows']}")

    # Upcoming events (24h)
    upcoming = engine.get_upcoming_events(hours_ahead=8)
    print(f"\n⏰ UPCOMING ({len(upcoming)} clustere în 8h):")
    for c in upcoming[:5]:
        print(f"   {c['window_start'][11:16]}  Score={c['impact_score']:.2f}  Alert={c['alert_level']}")
        for e in c["events"]:
            print(f"      → {e['event']} ({e['impact']})")

    # Multiplier de trading
    mult, msg = engine.get_trading_multiplier()
    print(f"\n🎯 Trading Multiplier: {mult:.2f}  ({msg})")
    print("=" * 70)
