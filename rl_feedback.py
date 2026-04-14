"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN — Reinforcement Learning Feedback Loop  (UPDATE #11)                ║
║  rl_feedback.py                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Sistem: Multi-Armed Bandit cu Experience Replay
─────────────────────────────────────────────────
Idee: după fiecare trade câștigat/pierdut, ajustează weights-urile celor 7
componente din formula de scoring:
  AI (XGBoost) | ICT | Quantum | RelStrength | Orderflow | Sentiment | VolumeProfile

Algoritm:
  1. La fiecare trade închis, citim scorul fiecărei componente la momentul intrării
  2. Dacă trade = WIN  → creștem ușor weights-urile componentelor cu scor mare
  3. Dacă trade = LOSS → scădem ușor weights-urile componentelor cu scor mare
  4. Normalizăm weights să sumeze 1.0 cu min 0.05 / max 0.50 per componentă
  5. Salvăm în rl_weights.json — mario_rag.py citește la fiecare apel

Hyperparametri:
  ALPHA     = 0.05   (learning rate — cât de mult se modifică la fiecare trade)
  GAMMA     = 0.95   (decay — trecuturi recente contează mai mult)
  MIN_W     = 0.05   (minimum weight per componentă)
  MAX_W     = 0.50   (maximum weight per componentă)
  N_HISTORY = 200    (nr. maxim trades în replay buffer)
"""

import os
import json
import logging
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURARE
# =============================================================================
RL_WEIGHTS_PATH  = "/Users/mario/Desktop/Aladin/rl_weights.json"
RL_HISTORY_PATH  = "/Users/mario/Desktop/Aladin/rl_history.json"
JOURNAL_PATH     = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"

ALPHA     = 0.05    # learning rate
GAMMA     = 0.95    # discount factor pentru trades mai vechi
MIN_W     = 0.02    # min weight per componentă (absolut) — 0.05 forța quantum la 0.05 chiar dacă dezactivat
MAX_W     = 0.60    # max weight per componentă — OF poate merge până la 0.60 dacă domină
# Fix v7.4: floor RL adaptive — normal 50% din default, deep cut 25% pentru loss streaks
# Exemplu: ICT default=0.25 → normal floor=0.125, deep floor=0.0625 pentru loss streaks
RL_FLOOR_PCT = 0.50   # 50% din default_weight = floor normal (was 60%)
RL_FLOOR_DEEP = 0.25  # 25% din default_weight = deep floor dacă loss_streak > 5
N_HISTORY = 200       # replay buffer size

# Componentele score-ului (ordinea contează — identică cu mario_rag.py)
COMPONENTS = ["ai", "ict", "quantum", "rel_strength", "orderflow", "sentiment", "volume_profile"]

# Weights default (sum = 1.0) — v4: volume_profile componentă separată (split din orderflow)
# Actualizate 08-04-2026: v4 — volume_profile adăugat ca a 7-a componentă
# orderflow 0.45→0.40 (-0.05), volume_profile 0.00→0.05 (nou)
DEFAULT_WEIGHTS = {
    "ai":             0.20,   # redus: model antrenat pe piețe diferite, bias bullish
    "ict":            0.25,   # neschimbat: structural, reliable
    "quantum":        0.00,   # dezactivat: weights neantrenate = zgomot
    "rel_strength":   0.05,   # neschimbat
    "orderflow":      0.40,   # ușor redus: 0.45→0.40 (cedat 0.05 la volume_profile)
    "sentiment":      0.05,   # neschimbat
    "volume_profile": 0.05,   # nou: RVOL + delta exhaustion + profile shape + LVN/HVN + prev POC
}


# =============================================================================
# LOAD / SAVE WEIGHTS
# =============================================================================
def load_weights() -> dict:
    """Încarcă weights din fișier sau returnează defaults."""
    try:
        if os.path.exists(RL_WEIGHTS_PATH):
            with open(RL_WEIGHTS_PATH, "r") as f:
                data = json.load(f)
            weights = data.get("weights", DEFAULT_WEIGHTS.copy())
            # Validare: toate componentele trebuie să fie prezente
            for c in COMPONENTS:
                if c not in weights:
                    weights[c] = DEFAULT_WEIGHTS[c]
            return weights
    except Exception as e:
        logger.debug(f"RL load_weights error: {e}")
    return DEFAULT_WEIGHTS.copy()


def save_weights(weights: dict, meta: dict = None):
    """Salvează weights + metadata în JSON."""
    try:
        data = {
            "weights":    weights,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "sum":        round(sum(weights.values()), 6),
            "meta":       meta or {},
        }
        with open(RL_WEIGHTS_PATH, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"💾 RL weights salvate: {json.dumps({k: round(v,3) for k,v in weights.items()})}")
    except Exception as e:
        logger.warning(f"RL save_weights error: {e}")


# =============================================================================
# NORMALIZE WEIGHTS
# =============================================================================
def normalize_weights(weights: dict, loss_streak: int = 0) -> dict:
    """
    Normalizează weights să sumeze 1.0 cu clip [MIN_W, MAX_W].
    Algoritmul iterează până convergă (max 20 iterații).

    Fix v7.4: Accept loss_streak parameter pentru adaptive floor (deep cuts dacă loss_streak > 5).
    """
    w = {k: float(v) for k, v in weights.items()}
    # Fix v7.4: Adaptive floor — deeper cuts allowed after consecutive losses
    floor_pct = RL_FLOOR_DEEP if loss_streak > 5 else RL_FLOOR_PCT

    for _ in range(20):
        # Clip la [MIN_W, MAX_W] + floor RL adaptive (protecție anti-degradare)
        w = {k: max(max(MIN_W, DEFAULT_WEIGHTS.get(k, MIN_W) * floor_pct), min(MAX_W, v)) for k, v in w.items()}
        total = sum(w.values())
        if abs(total - 1.0) < 1e-6:
            break
        # Rescale
        w = {k: v / total for k, v in w.items()}

    return {k: round(v, 6) for k, v in w.items()}


# =============================================================================
# UPDATE WEIGHTS DIN UN TRADE
# =============================================================================
def update_from_trade(
    component_scores: dict,   # {'ai': 0.72, 'ict': 0.60, ...} — scorurile la intrare
    trade_result: str,        # "WIN" sau "LOSS"
    pnl: float = 0.0,         # PnL în USD (pentru magnitude)
    current_weights: dict = None,
) -> dict:
    """
    Actualizează weights pe baza unui trade.

    Logica:
    - WIN:  componentele cu scor mare (>0.5) primesc boost proporțional
    - LOSS: componentele cu scor mare (>0.5) sunt penalizate proporțional
    - Magnitude ajustată de |pnl| — trade mare = update mai puternic

    Fix v7.4: Skip RL update for noise-level trades (not statistically significant)
    """
    # Fix v7.4: Skip RL update for noise-level trades (not statistically significant)
    if pnl is not None and abs(pnl) < 50:  # $50 = typical NQ noise on 1 contract
        logger.debug(f"RL skip: PnL ${pnl:.0f} < $50 noise threshold")
        return current_weights if current_weights else load_weights()

    weights = current_weights.copy() if current_weights else load_weights()

    is_win    = trade_result.upper() in ("WIN", "TP_HIT", "PROFIT")
    sign      = +1.0 if is_win else -1.0

    # Scaling bazat pe PnL (normalizat la 0.5–1.5x)
    pnl_scale = min(max(abs(pnl) / 200.0, 0.5), 1.5) if pnl else 1.0

    effective_alpha = ALPHA * pnl_scale

    for comp in COMPONENTS:
        comp_score = float(component_scores.get(comp, 0.5))
        # Ajustare: componentele cu scor mare contribuie mai mult la update
        contribution = (comp_score - 0.5) * 2.0   # mapare [0,1] → [-1,+1]
        delta = sign * contribution * effective_alpha
        weights[comp] = weights.get(comp, DEFAULT_WEIGHTS[comp]) + delta

    return normalize_weights(weights)


# =============================================================================
# EXPERIENCE REPLAY — re-antrenează pe ultimele N trades
# =============================================================================
def experience_replay(n: int = 50) -> dict:
    """
    Re-calculează weights optim din ultimele N trades din history.
    Mai robust decât update-ul incremental singur.
    """
    history = load_history()
    if len(history) < 5:
        logger.info("RL: prea puține date pentru replay (min 5 trades)")
        return load_weights()

    recent = history[-n:]   # ultimele N trades
    weights = DEFAULT_WEIGHTS.copy()

    for i, trade in enumerate(recent):
        # Fix v7.4: Discount formula ensures newest trades have weight 1.0 (i=n-1 → GAMMA^0=1.0)
        # i=0 (oldest) gets GAMMA^(n-1), i=n-1 (newest) gets GAMMA^0=1.0
        discount = GAMMA ** (len(recent) - i - 1)
        comp_scores = trade.get("component_scores", {})
        result      = trade.get("result", "LOSS")
        pnl         = float(trade.get("pnl", 0))

        is_win = result.upper() in ("WIN", "TP_HIT", "PROFIT")
        sign   = +1.0 if is_win else -1.0

        for comp in COMPONENTS:
            comp_score   = float(comp_scores.get(comp, 0.5))
            contribution = (comp_score - 0.5) * 2.0
            delta        = sign * contribution * ALPHA * discount
            weights[comp] = weights.get(comp, DEFAULT_WEIGHTS[comp]) + delta

    normalized = normalize_weights(weights)
    logger.info(f"🔄 RL Experience Replay ({len(recent)} trades): {json.dumps({k: round(v,3) for k,v in normalized.items()})}")
    return normalized


# =============================================================================
# HISTORY (replay buffer)
# =============================================================================
def load_history() -> list:
    try:
        if os.path.exists(RL_HISTORY_PATH):
            with open(RL_HISTORY_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_to_history(trade_data: dict):
    """Adaugă un trade în replay buffer (max N_HISTORY)."""
    history = load_history()
    history.append({**trade_data, "ts": datetime.now(timezone.utc).isoformat()})
    if len(history) > N_HISTORY:
        history = history[-N_HISTORY:]
    try:
        with open(RL_HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning(f"RL save_history error: {e}")


# =============================================================================
# FUNCȚIA PRINCIPALĂ — apelată din bridge_api după fiecare trade închis
# =============================================================================
def on_trade_closed(
    component_scores: dict,
    result: str,
    pnl: float = 0.0,
    score_pct: float = 0.0,
    direction: str = "",
) -> dict:
    """
    Apelată de bridge_api.py după fiecare trade cu rezultat cunoscut.
    Returnează noile weights.

    Args:
        component_scores: {'ai': 0.72, 'ict': 0.60, 'quantum': 0.45, ...}
        result: "WIN" / "LOSS" / "TP_HIT" / "SL_HIT"
        pnl: PnL în USD
        score_pct: scorul total la intrare (0-100)
        direction: "LONG" / "SHORT"
    """
    # Salvează în history
    save_to_history({
        "component_scores": component_scores,
        "result":    result,
        "pnl":       pnl,
        "score_pct": score_pct,
        "direction": direction,
    })

    history = load_history()
    n_trades = len(history)

    # Primele 10 trades: update incremental simplu
    # Peste 10 trades: experience replay la fiecare 5 trades, altfel incremental
    if n_trades >= 10 and n_trades % 5 == 0:
        new_weights = experience_replay(n=min(n_trades, 50))
        method = "replay"
    else:
        current = load_weights()
        new_weights = update_from_trade(
            component_scores=component_scores,
            trade_result=result,
            pnl=pnl,
            current_weights=current,
        )
        method = "incremental"

    # Calculează drift față de defaults
    drift = {k: round(new_weights[k] - DEFAULT_WEIGHTS[k], 4) for k in COMPONENTS}

    save_weights(new_weights, meta={
        "method":    method,
        "n_trades":  n_trades,
        "last_result": result,
        "last_pnl":  pnl,
        "drift_from_default": drift,
    })

    logger.info(
        f"🧠 RL Update ({method}, trade #{n_trades}): "
        f"result={result} pnl=${pnl:+.0f} | "
        f"AI={new_weights['ai']:.3f} ICT={new_weights['ict']:.3f} "
        f"Q={new_weights['quantum']:.3f} Sent={new_weights['sentiment']:.3f}"
    )

    return new_weights


# =============================================================================
# STATISTICI RL
# =============================================================================
def get_rl_stats() -> dict:
    """Returnează statistici despre starea curentă a RL."""
    weights = load_weights()
    history = load_history()

    wins   = sum(1 for t in history if t.get("result","").upper() in ("WIN","TP_HIT","PROFIT"))
    losses = sum(1 for t in history if t.get("result","").upper() in ("LOSS","SL_HIT"))
    total  = len(history)

    # Cel mai performant component (weight crescut față de default)
    best_comp  = max(COMPONENTS, key=lambda c: weights.get(c,0) - DEFAULT_WEIGHTS.get(c,0))
    worst_comp = min(COMPONENTS, key=lambda c: weights.get(c,0) - DEFAULT_WEIGHTS.get(c,0))

    return {
        "weights":      weights,
        "default":      DEFAULT_WEIGHTS,
        "drift":        {k: round(weights.get(k,0) - DEFAULT_WEIGHTS[k], 4) for k in COMPONENTS},
        "n_trades":     total,
        "wins":         wins,
        "losses":       losses,
        "win_rate":     round(wins / total * 100, 1) if total else 0.0,
        "best_component":  best_comp,
        "worst_component": worst_comp,
        "updated_at":   datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# RESET
# =============================================================================
def reset_weights():
    """Resetează weights la default și golește history."""
    save_weights(DEFAULT_WEIGHTS.copy(), meta={"reset": True})
    try:
        if os.path.exists(RL_HISTORY_PATH):
            os.remove(RL_HISTORY_PATH)
    except Exception:
        pass
    logger.info("🔄 RL weights resetate la default")


# =============================================================================
# TEST STANDALONE
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n🧠 Test RL Feedback Loop")
    print("=" * 50)

    # Simulează 20 trades
    import random
    rng = random.Random(42)

    for i in range(20):
        scores = {c: rng.uniform(0.3, 0.9) for c in COMPONENTS}
        # AI + ICT buni → WIN mai des
        is_win = (scores["ai"] + scores["ict"]) / 2 > 0.6
        result = "WIN" if is_win else "LOSS"
        pnl    = rng.uniform(100, 500) if is_win else -rng.uniform(50, 200)

        new_w = on_trade_closed(scores, result, pnl, score_pct=rng.uniform(55, 80))
        print(f"  Trade #{i+1}: {result} ${pnl:+.0f} → AI={new_w['ai']:.3f} ICT={new_w['ict']:.3f}")

    print("\n📊 Final RL Stats:")
    stats = get_rl_stats()
    print(f"  Win Rate: {stats['win_rate']}% ({stats['wins']}W/{stats['losses']}L)")
    print(f"  Best component: {stats['best_component']}")
    print(f"  Weights: {json.dumps({k: round(v,3) for k,v in stats['weights'].items()})}")
    print(f"  Drift: {json.dumps(stats['drift'])}")
