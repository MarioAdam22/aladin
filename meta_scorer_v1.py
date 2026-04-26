"""
meta_scorer_v1.py — Meta-Scorer ALADIN v1
==========================================
Agregă semnale de la toate modelele într-un scor final (0-1).

Formula:
  final_score = signal_score × regime_mult × rag_agreement × rag_score_norm

Unde:
  signal_score   = ML score din checker-ul activ (nom_score, lom_score etc.)
                   sau RR normalizat pentru ICT/TS
  regime_mult    = multiplicator calibrat per (regim × checker) — din backtest
  rag_agreement  = 1.0 dacă RAG confirmă direcția, 0.85 neutru, 0.55 contrar
  rag_score_norm = rag_score / 100, clamped [0, 1]

Decizie:
  final_score >= META_THRESHOLD → TRADE
  final_score <  META_THRESHOLD → WAIT

Multiplieri calibrați empiric pe backtest_regime.py:
  PRE_EXPANSION + NOM/LOM = cel mai mare edge   → 1.20 / 1.15
  EXPANSION     + ICT/TS  = trend continuation  → 1.15 / 1.10
  RETRACEMENT   + ICT/TS  = pullback entry      → 1.20 / 1.15
  DISTRIBUTION  + TS      = reversal dominant   → 1.20
  CONSOLIDATION  orice    = penalizare severă   → 0.50-0.60
"""

import logging

logger = logging.getLogger("ALADIN.META")

# ── Threshold global ──────────────────────────────────────────────────────────
META_THRESHOLD = 0.42   # sub asta → WAIT

# ── Multiplicatori calibrați (regim × checker) ────────────────────────────────
# Calibrare: backtest OOS 2024-2026 (backtest_regime.py scenariile A/B/C)
# PRE_EXPANSION B scenario saved $20k vs no filter → NOM/LOM boosted
# CONSOLIDATION all scenarios bad → penalizare
REGIME_MULTS: dict[tuple[str, str], float] = {
    # PRE_EXPANSION — sweep manipulation → NOM/LOM cel mai bun
    ('PRE_EXPANSION', 'NOM'):         1.20,
    ('PRE_EXPANSION', 'LOM'):         1.15,
    ('PRE_EXPANSION', 'DSM'):         0.90,
    ('PRE_EXPANSION', 'ICT'):         0.80,
    ('PRE_EXPANSION', 'TS'):          0.75,

    # EXPANSION — HTF trend activ → ICT/TS preferați
    ('EXPANSION', 'ICT'):             1.15,
    ('EXPANSION', 'TS'):              1.10,
    ('EXPANSION', 'DSM'):             1.00,
    ('EXPANSION', 'NOM'):             0.70,
    ('EXPANSION', 'LOM'):             0.70,

    # RETRACEMENT — pullback în expansion → ICT clasic + TS
    ('RETRACEMENT', 'ICT'):           1.20,
    ('RETRACEMENT', 'TS'):            1.15,
    ('RETRACEMENT', 'DSM'):           1.10,
    ('RETRACEMENT', 'NOM'):           0.80,
    ('RETRACEMENT', 'LOM'):           0.80,

    # DISTRIBUTION — extreme VWAP, fading → TS reversal dominant
    ('DISTRIBUTION', 'TS'):           1.20,
    ('DISTRIBUTION', 'ICT'):          1.05,
    ('DISTRIBUTION', 'DSM'):          0.90,
    ('DISTRIBUTION', 'NOM'):          0.70,
    ('DISTRIBUTION', 'LOM'):          0.70,

    # CONSOLIDATION — niciunul nu are edge → penalizare
    ('CONSOLIDATION', 'ICT'):         0.60,
    ('CONSOLIDATION', 'TS'):          0.60,
    ('CONSOLIDATION', 'DSM'):         0.50,
    ('CONSOLIDATION', 'NOM'):         0.50,
    ('CONSOLIDATION', 'LOM'):         0.50,

    # UNKNOWN / prob scăzut → neutru
    ('UNKNOWN', 'ICT'):               1.00,
    ('UNKNOWN', 'TS'):                1.00,
    ('UNKNOWN', 'DSM'):               1.00,
    ('UNKNOWN', 'NOM'):               1.00,
    ('UNKNOWN', 'LOM'):               1.00,
}


def _infer_checker(signal: dict) -> str:
    """Deduce checker type din signal dict."""
    # checker_name injectat de gate_verdict() → prioritate
    cn = str(signal.get('checker_name', '')).upper()
    if cn in ('NOM', 'LOM', 'DSM', 'ICT', 'TS'):
        return cn
    # Fallback: din setup_type
    st = str(signal.get('setup_type', '')).upper()
    if 'NOM' in st:   return 'NOM'
    if 'LOM' in st:   return 'LOM'
    if 'DSM' in st:   return 'DSM'
    if 'TS'  in st:   return 'TS'
    return 'ICT'


def _get_signal_score(signal: dict, checker: str) -> float:
    """
    Extrage normalized ML score (0-1) din semnal.
    NOM/LOM/DSM → ml_score direct (deja 0-1)
    ICT/TS       → RR normalizat [1.5, 4.0] → [0.35, 0.80]
    """
    if checker in ('NOM', 'LOM', 'DSM'):
        return float(signal.get('ml_score', 0.5))
    # ICT / TS: RR norm
    rr = float(signal.get('rr', 1.5))
    return min(0.90, max(0.20, (rr - 1.0) / 3.5 * 0.60 + 0.30))


def compute_meta_score(
    signal:           dict,
    checker_name:     str   = None,
    regime:           str   = 'UNKNOWN',
    regime_prob:      float = 0.0,
    rag_direction:    str   = 'NEUTRAL',
    rag_score:        float = 50.0,
    signal_direction: str   = None,
) -> tuple[float, dict]:
    """
    Calculează meta-score final pentru un semnal.

    Args:
        signal:           dict returnat de checker (NOM/LOM/DSM/ICT/TS)
        checker_name:     'NOM', 'LOM', 'DSM', 'ICT', 'TS' (dedus dacă None)
        regime:           regimul curent din classify_regime()
        regime_prob:      probabilitatea regimului (0-1)
        rag_direction:    direcția din mario_rag ('LONG'/'SHORT'/'NEUTRAL')
        rag_score:        scorul mario_rag (0-100)
        signal_direction: direcția semnalului (dedusă din signal dacă None)

    Returnează:
        (final_score: float, breakdown: dict)
        breakdown include decizia 'TRADE'/'WAIT' și componentele individuale
    """
    checker = checker_name or _infer_checker(signal)

    # ── 1. Signal score (0-1) ─────────────────────────────────────────────────
    sig_sc = _get_signal_score(signal, checker)

    # ── 2. Regime multiplier ──────────────────────────────────────────────────
    regime_key  = regime if regime else 'UNKNOWN'
    regime_mult = REGIME_MULTS.get(
        (regime_key, checker),
        REGIME_MULTS.get(('UNKNOWN', checker), 1.0)
    )
    # Dacă prob regim e sub 0.50, interpolăm spre 1.0 (incertitudine → neutru)
    if regime_prob < 0.50:
        alpha       = regime_prob / 0.50          # 0.0-1.0
        regime_mult = (1.0 - alpha) * 1.0 + alpha * regime_mult

    # ── 3. RAG agreement ──────────────────────────────────────────────────────
    sig_dir = signal_direction or str(signal.get('direction', '')).upper()
    rag_dir = str(rag_direction).upper().strip()

    if not rag_dir or rag_dir == 'NEUTRAL':
        rag_agr = 0.85                              # neutral → ușoară penalizare
    elif sig_dir:
        if (sig_dir == 'LONG'  and rag_dir in ('LONG',  'BUY'))  or \
           (sig_dir == 'SHORT' and rag_dir in ('SHORT', 'SELL')):
            rag_agr = 1.00                          # confirmat → bonus maxim
        else:
            rag_agr = 0.55                          # contrar → penalizare dar nu blocare
    else:
        rag_agr = 0.85

    # ── 4. RAG score normalization ─────────────────────────────────────────────
    # rag_score vine 0-100 din mario_rag
    rag_norm = float(rag_score) / 100.0
    rag_norm = max(0.10, min(1.0, rag_norm))        # clamp, nu mai mic de 0.10

    # ── Formula finală ────────────────────────────────────────────────────────
    final = sig_sc * regime_mult * rag_agr * rag_norm
    final = round(min(1.0, max(0.0, final)), 4)

    decision = 'TRADE' if final >= META_THRESHOLD else 'WAIT'

    breakdown = {
        'checker':       checker,
        'signal_score':  round(sig_sc, 4),
        'regime':        regime_key,
        'regime_prob':   round(regime_prob, 3),
        'regime_mult':   round(regime_mult, 4),
        'rag_direction': rag_dir,
        'rag_agreement': round(rag_agr, 3),
        'rag_score':     round(rag_score, 2),
        'rag_norm':      round(rag_norm, 4),
        'final_score':   final,
        'threshold':     META_THRESHOLD,
        'decision':      decision,
    }

    logger.info(
        f"🧮 MetaScore [{checker}|{regime_key}(p={regime_prob:.2f})] "
        f"sig={sig_sc:.3f} × mult={regime_mult:.3f} × "
        f"rag_agr={rag_agr:.2f} × rag_norm={rag_norm:.3f} = {final:.4f} "
        f"→ {'✅ TRADE' if decision == 'TRADE' else '⏳ WAIT'}"
    )
    return final, breakdown


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Exemple rapide
    test_cases = [
        dict(signal={'setup_type': 'NY_NOM', 'ml_score': 0.78, 'direction': 'LONG'},
             regime='PRE_EXPANSION', regime_prob=0.88,
             rag_direction='LONG', rag_score=72,
             label="NOM LONG în PRE_EXPANSION, RAG confirmă"),
        dict(signal={'setup_type': 'LON_LOM', 'ml_score': 0.70, 'direction': 'SHORT'},
             regime='PRE_EXPANSION', regime_prob=0.75,
             rag_direction='LONG', rag_score=65,
             label="LOM SHORT în PRE_EXPANSION, RAG CONTRARAR"),
        dict(signal={'setup_type': 'NY_SHORT', 'rr': 2.0, 'direction': 'SHORT'},
             regime='EXPANSION', regime_prob=0.82,
             rag_direction='SHORT', rag_score=80,
             label="ICT SHORT în EXPANSION, RAG confirmă"),
        dict(signal={'setup_type': 'LON_LONG', 'rr': 1.5, 'direction': 'LONG'},
             regime='CONSOLIDATION', regime_prob=0.85,
             rag_direction='LONG', rag_score=55,
             label="ICT LONG în CONSOLIDATION → ar trebui blocat"),
    ]
    print(f"\n{'='*70}")
    print(f"  META-SCORER v1 — Test Cases  (threshold={META_THRESHOLD})")
    print(f"{'='*70}")
    for tc in test_cases:
        sc, bd = compute_meta_score(
            signal=tc['signal'], regime=tc['regime'], regime_prob=tc['regime_prob'],
            rag_direction=tc['rag_direction'], rag_score=tc['rag_score']
        )
        emoji = '✅' if bd['decision'] == 'TRADE' else '⏳'
        print(f"\n  {tc['label']}")
        print(f"    {emoji} {bd['decision']} — final={sc:.4f} "
              f"(sig={bd['signal_score']:.3f} × mult={bd['regime_mult']:.3f} "
              f"× rag_agr={bd['rag_agreement']:.2f} × rag_norm={bd['rag_norm']:.3f})")
    print()
