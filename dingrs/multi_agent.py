"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN — MULTI-AGENT ORCHESTRATOR v1.0                                     ║
║  Optimizat pentru Apple Silicon M4                                           ║
║                                                                              ║
║  3 Agenți în paralel:                                                        ║
║    • TechnicalAgent  — prețul, OrderFlow, AI (mario_rag)                    ║
║    • FundamentalAgent — "conversația" știrilor (news_clustering)             ║
║    • ExecutiveAgent   — decizia finală (consensul celorlalți doi)            ║
║                                                                              ║
║  Arhitectura:                                                                 ║
║    asyncio.gather() → agenți în paralel pe loop async                       ║
║    ProcessPoolExecutor → CPU-bound tasks pe core-uri M4 separate            ║
║    LatencyTracker → monitorizare timp răspuns per agent                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("multi_agent")

# ─── Config M4 ────────────────────────────────────────────────────────────────
# Apple M4: 10 core (4 perf + 6 eff) — folosim 3 procese pentru cele 3 sarcini CPU-bound
M4_WORKERS    = min(os.cpu_count() or 4, 4)   # Max 4 procese CPU-bound
TIMEOUT_S     = 2.0    # Max 2s per agent (latency SLA)
CONSENSUS_MIN = 2      # Minimum 2 agenți de acord pentru execuție


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketSnapshot:
    """Snapshot piață livrat de bridge_api.py la fiecare analiză."""
    symbol:         str
    timestamp:      str
    close:          float
    high:           float
    low:            float
    open:           float
    volume:         float
    cum_delta:      float = 0.0
    imbalance_pct:  float = 0.0
    vwap:           float = 0.0
    poc:            float = 0.0
    vah:            float = 0.0
    val:            float = 0.0
    bid_ask_ratio:  float = 1.0
    bar_history:    List[Dict] = field(default_factory=list)  # ultimele N bare


@dataclass
class AgentResult:
    """Rezultatul unui agent: scor, direcție, context."""
    agent_name:     str
    score:          float          # [0, 1] — convicție
    direction:      str            # "LONG" / "SHORT" / "WAIT"
    confidence:     float          # [0, 1] — cât de sigur e agentul
    latency_ms:     float = 0.0
    context:        Dict  = field(default_factory=dict)
    error:          Optional[str] = None


@dataclass
class ExecutiveDecision:
    """Decizia finală a Agentului Executiv."""
    timestamp:      str
    action:         str            # "LONG" / "SHORT" / "WAIT" / "BLACKOUT"
    final_score:    float          # [0, 100] — scor final combinat
    confidence:     float          # [0, 1]
    consensus:      int            # Câți agenți sunt de acord
    reasoning:      str
    technical:      AgentResult
    fundamental:    AgentResult
    latency_total_ms: float = 0.0
    execute:        bool = False   # True dacă trebuie trimisă comandă NT8


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1: TECHNICAL AGENT (mario_rag + OrderFlow)
# ══════════════════════════════════════════════════════════════════════════════

def _technical_agent_sync(snapshot: MarketSnapshot) -> AgentResult:
    """
    Rulat în ProcessPool — CPU-bound (XGBoost + Quantum).
    Apelează mario_rag.aladin_engine() cu datele din snapshot.
    """
    t0 = time.perf_counter()
    try:
        import sys, os
        _dir = os.path.dirname(os.path.abspath(__file__))
        if _dir not in sys.path:
            sys.path.insert(0, _dir)

        import pandas as pd
        import numpy as np
        import mario_rag

        # Construiește DataFrame din bar_history
        if len(snapshot.bar_history) < 5:
            return AgentResult(
                agent_name="Technical",
                score=0.0, direction="WAIT", confidence=0.0,
                latency_ms=0.0,
                context={"reason": "insufficient_bars"},
            )

        rows = snapshot.bar_history
        df = pd.DataFrame([{
            "open":           r.get("price", {}).get("open",  snapshot.open),
            "high":           r.get("price", {}).get("high",  snapshot.high),
            "low":            r.get("price", {}).get("low",   snapshot.low),
            "close":          r.get("price", {}).get("close", snapshot.close),
            "volume":         r.get("price", {}).get("volume", snapshot.volume),
            "poc_level":      r.get("volume_profile", {}).get("poc", snapshot.poc),
            "vah":            r.get("volume_profile", {}).get("vah", snapshot.vah),
            "val":            r.get("volume_profile", {}).get("val", snapshot.val),
            "true_open":      rows[0].get("price", {}).get("open", snapshot.open),
            "dist_poc":       r.get("price", {}).get("close", 0) - r.get("volume_profile", {}).get("poc", 0),
            "inside_va": int(
                r.get("volume_profile", {}).get("val", 0) <=
                r.get("price", {}).get("close", 0) <=
                r.get("volume_profile", {}).get("vah", 0)
            ),
            "has_displacement": int(abs(r.get("orderflow", {}).get("imbalance_pct", 0)) > 60),
            "is_above_open": int(r.get("price", {}).get("close", 0) > rows[0].get("price", {}).get("open", 0)),
            # Placeholder features
            "lm_hi":0,"lm_lo":0,"lw_hi":0,"lw_lo":0,
            "m_hi":0,"m_lo":0,"p_hi":0,"p_lo":0,
            "h4_hi":0,"h4_lo":0,"h1_hi":0,"h1_lo":0,
            "asia_hi":0,"asia_lo":0,"lon_hi":0,"lon_lo":0,
            "fvg_up":0,"fvg_down":0,
            "is_smt_bearish":0,"is_smt_bullish":0,
            "dist_pdh":0,"dist_pdl":0,"atr_14":0,
        } for r in rows])

        result = mario_rag.aladin_engine(df=df, target_ts=snapshot.timestamp)

        if not result:
            return AgentResult(agent_name="Technical", score=0.0, direction="WAIT", confidence=0.0)

        score     = float(result.get("score", 0)) / 100.0
        verdict   = str(result.get("verdict", ""))
        direction = str(result.get("trade_direction", "LONG"))
        ai_score  = float(result.get("ai_score", 0)) / 100.0

        # Confidence basată pe consistența semnalelor
        confidence = float(result.get("ai_score", 0)) / 100.0 * 0.5 + score * 0.5

        return AgentResult(
            agent_name  = "Technical",
            score       = score,
            direction   = direction,
            confidence  = round(confidence, 4),
            latency_ms  = round((time.perf_counter() - t0) * 1000, 1),
            context     = {
                "verdict":       verdict[:60],
                "ai_score":      ai_score,
                "quantum_score": result.get("quantum_score", 0),
                "ict_signals":   result.get("ict_signals", 0),
                "cum_delta":     snapshot.cum_delta,
                "imbalance":     snapshot.imbalance_pct,
                "poc":           snapshot.poc,
                "vwap":          snapshot.vwap,
            },
        )

    except Exception as e:
        return AgentResult(
            agent_name="Technical",
            score=0.0, direction="WAIT", confidence=0.0,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            error=str(e)[:100],
        )


def _fundamental_agent_sync(snapshot: MarketSnapshot) -> AgentResult:
    """
    Rulat în ThreadPool — I/O-bound (news calendar).
    Apelează news_clustering.get_news_engine().
    """
    t0 = time.perf_counter()
    try:
        import sys, os
        _dir = os.path.dirname(os.path.abspath(__file__))
        if _dir not in sys.path:
            sys.path.insert(0, _dir)

        from news_clustering import get_news_engine

        engine = get_news_engine()
        ctx    = engine.get_current_context(snapshot.timestamp)
        mult, msg = engine.get_trading_multiplier(snapshot.timestamp)

        impact = ctx["impact_score"]
        advice = ctx["trading_advice"]
        sentiment = ctx["direction_sentiment"]  # +1=bullish USD, -1=bearish USD

        # Traduce sentimentul știrilor în direcție pentru NQ:
        # USD bullish (beat expectations) → tendință bearish NQ
        # USD bearish (miss expectations) → tendință bullish NQ
        if advice == "BLACKOUT":
            direction = "WAIT"
            score     = 0.0
            confidence = 0.9   # Sigur că nu trebuie tranzacționat
        elif advice == "CAUTION":
            direction = "WAIT"
            score     = 0.3
            confidence = 0.6
        else:
            # TRADE: știrile nu blochează
            if sentiment > 0:   # USD beat → NQ bearish
                direction = "SHORT"
                score     = 0.5 + impact * 0.3
            elif sentiment < 0:  # USD miss → NQ bullish
                direction = "LONG"
                score     = 0.5 + impact * 0.3
            else:
                direction = "WAIT"
                score     = 0.5
            confidence = 0.4 + impact * 0.4

        return AgentResult(
            agent_name  = "Fundamental",
            score       = round(score, 4),
            direction   = direction,
            confidence  = round(confidence, 4),
            latency_ms  = round((time.perf_counter() - t0) * 1000, 1),
            context     = {
                "advice":     advice,
                "impact":     impact,
                "sentiment":  sentiment,
                "mult":       mult,
                "summary":    msg[:80],
                "active":     bool(ctx.get("active_cluster")),
                "upcoming":   bool(ctx.get("upcoming_cluster")),
            },
        )

    except Exception as e:
        return AgentResult(
            agent_name="Fundamental",
            score=0.5, direction="WAIT", confidence=0.2,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            error=str(e)[:100],
        )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3: EXECUTIVE AGENT (decizia finală)
# ══════════════════════════════════════════════════════════════════════════════

def executive_agent(
    technical:    AgentResult,
    fundamental:  AgentResult,
    snapshot:     MarketSnapshot,
) -> ExecutiveDecision:
    """
    Agentul Executiv combină semnalele și ia decizia finală.

    Logică:
      1. BLACKOUT news → WAIT indiferent de technical
      2. Consens direcțional (ambii agenți de acord) → execuție cu confidence crescut
      3. Technical dominant (fundamental neutru) → urmează technical cu reducere sizing
      4. Conflict major → WAIT

    Ponderi Executive:
      Technical:    60%  (AI + Quantum + ICT + OrderFlow)
      Fundamental:  40%  (news impact multiplier)
    """
    ts = datetime.now(timezone.utc).isoformat()

    # ── 1. Blackout news → stop ───────────────────────────────────────────────
    if fundamental.context.get("advice") == "BLACKOUT":
        return ExecutiveDecision(
            timestamp    = ts,
            action       = "BLACKOUT",
            final_score  = 0.0,
            confidence   = 0.95,
            consensus    = 0,
            reasoning    = f"🚨 NEWS BLACKOUT — {fundamental.context.get('summary','')[:60]}",
            technical    = technical,
            fundamental  = fundamental,
            execute      = False,
        )

    # ── 2. Scor final ponderat ────────────────────────────────────────────────
    news_mult  = fundamental.context.get("mult", 1.0)
    tech_score = technical.score
    fund_score = fundamental.score if fundamental.direction != "WAIT" else 0.5

    # Scor combinat
    raw_combined = (0.60 * tech_score + 0.40 * fund_score) * news_mult
    final_score  = round(min(raw_combined * 100, 100), 2)

    # ── 3. Direcție prin consens ──────────────────────────────────────────────
    tech_dir = technical.direction
    fund_dir = fundamental.direction

    # Fundamental neutru → urmează technical
    if fund_dir == "WAIT":
        action    = tech_dir
        consensus = 1
    elif tech_dir == fund_dir:
        # Acord total → confidence crescut
        action    = tech_dir
        consensus = 2
    else:
        # Conflict → WAIT dacă scorul technical nu e copleșitor
        if tech_score > 0.75 and technical.confidence > 0.65:
            action    = tech_dir   # Technical dominant → override fundamental
            consensus = 1
        else:
            action    = "WAIT"
            consensus = 0

    # ── 4. Gate finală ────────────────────────────────────────────────────────
    # Nu executăm dacă:
    #   - Scorul final < 55
    #   - Technical score < 0.50
    #   - Action == WAIT
    should_execute = (
        final_score >= 55 and
        tech_score  >= 0.50 and
        action not in ("WAIT", "BLACKOUT")
    )

    # ── 5. Confidence combinat ────────────────────────────────────────────────
    if consensus == 2:
        confidence = (technical.confidence + fundamental.confidence) / 2 * 1.2
    elif consensus == 1:
        confidence = technical.confidence * 0.8
    else:
        confidence = min(technical.confidence, fundamental.confidence) * 0.5
    confidence = round(min(confidence, 1.0), 4)

    # ── 6. Reasoning text ─────────────────────────────────────────────────────
    parts = [
        f"Technical={tech_score:.0%}({tech_dir}) Fundamental={fund_score:.0%}({fund_dir})",
        f"News={fundamental.context.get('advice','?')} mult={news_mult:.2f}",
        f"Consens={consensus}/2 Score={final_score:.1f}%",
    ]
    if not should_execute:
        parts.append("→ WAIT (prag insuficient)")

    return ExecutiveDecision(
        timestamp     = ts,
        action        = action if should_execute else "WAIT",
        final_score   = final_score,
        confidence    = confidence,
        consensus     = consensus,
        reasoning     = " | ".join(parts),
        technical     = technical,
        fundamental   = fundamental,
        execute       = should_execute,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — rulează toți 3 agenții în paralel
# ══════════════════════════════════════════════════════════════════════════════

class AladinOrchestrator:
    """
    Orchestratorul principal Aladin.
    Folosește asyncio.gather() pentru paralelism I/O și
    ProcessPoolExecutor pentru CPU-bound (XGBoost/Quantum).
    """

    def __init__(self):
        self._thread_pool   = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aladin_fund")
        self._process_pool  = ProcessPoolExecutor(max_workers=M4_WORKERS)
        self._decision_log: List[ExecutiveDecision] = []
        self._call_count:   int = 0

        log.info(f"🤖 AladinOrchestrator pornit (M4_WORKERS={M4_WORKERS})")

    async def analyze(self, snapshot: MarketSnapshot) -> ExecutiveDecision:
        """
        Rulează toți agenții în paralel și returnează decizia executivă.
        Latency SLA: < 2000ms total.
        """
        t0 = time.perf_counter()
        self._call_count += 1

        loop = asyncio.get_event_loop()

        try:
            # ── Paralel: Technical (process pool) + Fundamental (thread pool) ──
            tech_future = loop.run_in_executor(
                self._process_pool,
                _technical_agent_sync,
                snapshot,
            )
            fund_future = loop.run_in_executor(
                self._thread_pool,
                _fundamental_agent_sync,
                snapshot,
            )

            # Așteptăm ambii cu timeout
            try:
                tech_result, fund_result = await asyncio.wait_for(
                    asyncio.gather(tech_future, fund_future, return_exceptions=True),
                    timeout=TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                log.warning(f"⏰ Agent timeout după {TIMEOUT_S}s")
                tech_result = AgentResult("Technical", 0.0, "WAIT", 0.0, error="timeout")
                fund_result = AgentResult("Fundamental", 0.5, "WAIT", 0.2, error="timeout")

            # Handle exceptions returnate de gather (return_exceptions=True)
            if isinstance(tech_result, Exception):
                log.warning(f"Technical agent error: {tech_result}")
                tech_result = AgentResult("Technical", 0.0, "WAIT", 0.0, error=str(tech_result)[:50])
            if isinstance(fund_result, Exception):
                log.warning(f"Fundamental agent error: {fund_result}")
                fund_result = AgentResult("Fundamental", 0.5, "WAIT", 0.2, error=str(fund_result)[:50])

            # ── Executive decision ────────────────────────────────────────────
            decision = executive_agent(tech_result, fund_result, snapshot)
            decision.latency_total_ms = round((time.perf_counter() - t0) * 1000, 1)

            # Log
            log.info(
                f"[#{self._call_count}] {decision.action:5s} "
                f"score={decision.final_score:.1f}% "
                f"conf={decision.confidence:.2f} "
                f"consensus={decision.consensus}/2 "
                f"exec={decision.execute} "
                f"lat={decision.latency_total_ms:.0f}ms"
            )
            if tech_result.error:
                log.debug(f"  Tech error: {tech_result.error}")
            if fund_result.error:
                log.debug(f"  Fund error: {fund_result.error}")

            # Stochează în log (ultimele 100)
            self._decision_log.append(decision)
            if len(self._decision_log) > 100:
                self._decision_log.pop(0)

            return decision

        except Exception as e:
            log.error(f"Orchestrator error: {e}")
            # Safe fallback
            return ExecutiveDecision(
                timestamp    = datetime.now(timezone.utc).isoformat(),
                action       = "WAIT",
                final_score  = 0.0,
                confidence   = 0.0,
                consensus    = 0,
                reasoning    = f"Orchestrator error: {str(e)[:50]}",
                technical    = AgentResult("Technical",   0.0, "WAIT", 0.0),
                fundamental  = AgentResult("Fundamental", 0.5, "WAIT", 0.0),
                execute      = False,
            )

    def get_decision_history(self, last_n: int = 20) -> List[Dict]:
        """Returnează ultimele N decizii pentru dashboard."""
        return [
            {
                "ts":         d.timestamp[:19],
                "action":     d.action,
                "score":      d.final_score,
                "confidence": d.confidence,
                "consensus":  d.consensus,
                "execute":    d.execute,
                "latency_ms": d.latency_total_ms,
                "tech":       d.technical.direction,
                "fund":       d.fundamental.direction,
            }
            for d in self._decision_log[-last_n:]
        ]

    def get_performance_stats(self) -> Dict:
        """Statistici de performanță ale orchestratorului."""
        if not self._decision_log:
            return {"calls": 0}

        latencies    = [d.latency_total_ms for d in self._decision_log]
        execute_rate = sum(1 for d in self._decision_log if d.execute) / len(self._decision_log)
        actions      = [d.action for d in self._decision_log]

        return {
            "calls":          self._call_count,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "max_latency_ms": round(max(latencies), 1),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies)*0.95)], 1),
            "execute_rate":   round(execute_rate, 3),
            "long_pct":       round(actions.count("LONG")  / len(actions) * 100, 1),
            "short_pct":      round(actions.count("SHORT") / len(actions) * 100, 1),
            "wait_pct":       round(actions.count("WAIT")  / len(actions) * 100, 1),
        }

    def shutdown(self):
        """Oprire curată a pool-urilor."""
        self._thread_pool.shutdown(wait=False)
        self._process_pool.shutdown(wait=False)
        log.info("Orchestrator oprit.")


# ─── Singleton global ─────────────────────────────────────────────────────────
_orchestrator: Optional[AladinOrchestrator] = None

def get_orchestrator() -> AladinOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AladinOrchestrator()
    return _orchestrator


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRARE CU BRIDGE API
# ══════════════════════════════════════════════════════════════════════════════

async def analyze_from_bridge(nt8_data, bar_history: list) -> Optional[ExecutiveDecision]:
    """
    Convenience wrapper — apelat din bridge_api.py în background task.
    Convertește NT8Data (Pydantic) → MarketSnapshot și rulează orkestrarea.
    """
    try:
        snapshot = MarketSnapshot(
            symbol        = nt8_data.symbol,
            timestamp     = nt8_data.timestamp,
            close         = nt8_data.price.close,
            high          = nt8_data.price.high,
            low           = nt8_data.price.low,
            open          = nt8_data.price.open,
            volume        = nt8_data.price.volume,
            cum_delta     = nt8_data.orderflow.cum_delta,
            imbalance_pct = nt8_data.orderflow.imbalance_pct,
            vwap          = nt8_data.orderflow.vwap,
            poc           = nt8_data.volume_profile.poc,
            vah           = nt8_data.volume_profile.vah,
            val           = nt8_data.volume_profile.val,
            bid_ask_ratio = nt8_data.dom_liquidity.bid_ask_ratio,
            bar_history   = bar_history,
        )
        orch = get_orchestrator()
        return await orch.analyze(snapshot)
    except Exception as e:
        log.warning(f"analyze_from_bridge error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CLI TEST — python3 multi_agent.py
# ══════════════════════════════════════════════════════════════════════════════

async def _demo():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    print("\n" + "=" * 70)
    print("🤖 ALADIN MULTI-AGENT ORCHESTRATOR — Demo")
    print("=" * 70)

    # Snapshot simulat
    snapshot = MarketSnapshot(
        symbol        = "NQ",
        timestamp     = datetime.now(timezone.utc).isoformat(),
        close         = 21500.0,
        high          = 21520.0,
        low           = 21480.0,
        open          = 21490.0,
        volume        = 15000,
        cum_delta     = 2500.0,
        imbalance_pct = 65.0,
        vwap          = 21495.0,
        poc           = 21488.0,
        vah           = 21510.0,
        val           = 21470.0,
        bid_ask_ratio = 1.15,
        bar_history   = [],  # Fără history → Technical va returna WAIT
    )

    orch = get_orchestrator()

    print("\n⏳ Rulând cei 3 agenți în paralel...")
    t0 = time.perf_counter()
    decision = await orch.analyze(snapshot)
    elapsed  = (time.perf_counter() - t0) * 1000

    print(f"\n{'─'*70}")
    print(f"⚡ DECIZIE FINALĂ:  {decision.action}")
    print(f"   Score:          {decision.final_score:.1f}%")
    print(f"   Confidence:     {decision.confidence:.2%}")
    print(f"   Consens:        {decision.consensus}/2")
    print(f"   Execute:        {decision.execute}")
    print(f"   Latency Total:  {elapsed:.0f}ms")
    print(f"   Reasoning:      {decision.reasoning}")
    print(f"\n   Technical:  dir={decision.technical.direction}  score={decision.technical.score:.2f}  lat={decision.technical.latency_ms:.0f}ms")
    print(f"   Fundamental: dir={decision.fundamental.direction}  score={decision.fundamental.score:.2f}  lat={decision.fundamental.latency_ms:.0f}ms")

    if decision.technical.error:
        print(f"   ⚠️  Tech error:  {decision.technical.error}")
    if decision.fundamental.error:
        print(f"   ⚠️  Fund error:  {decision.fundamental.error}")

    print(f"\n   Fundamental context: {decision.fundamental.context}")
    print("=" * 70)

    orch.shutdown()


if __name__ == "__main__":
    asyncio.run(_demo())
