"""
sweep_scorer.py — Runtime Sweep Model Scorer
=============================================
Încarcă lazy modelele sweep_PRE_EXPANSION / EXPANSION / RETRACEMENT / ALL
și scorează un setup dat sweep_feats dict (primit din lom_checker / nom_checker).

Usage:
    from sweep_scorer import score_sweep, score_sweep_gated, get_sweep_threshold
    prob = score_sweep(sig['sweep_feats'], sig.get('regime_str', 'ALL'))
    # returns float [0, 1]  (0.5 dacă model nu e disponibil)

    # Gated: returnează 0.5 neutru dacă prob < threshold per regim
    prob_gated = score_sweep_gated(sig['sweep_feats'], sig.get('regime_str', 'ALL'))

    # Threshold recomandat per regim (OOS 2025, threshold sweep analiză)
    thr = get_sweep_threshold('EXPANSION')   # → 0.52
"""

import sys
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger("ALADIN.SweepScorer")
BASE   = Path(__file__).parent

# ── Ensure Aladin dir is on sys.path so _CalModel (aladin_cal) unpickles OK ──
_BASE_STR = str(BASE)
if _BASE_STR not in sys.path:
    sys.path.insert(0, _BASE_STR)

# ── Lazy cache per regim ────────────────────────────────────────────────────────
_SWEEP_MODELS = {}   # key = regime_str ('PRE_EXPANSION' | 'EXPANSION' | 'RETRACEMENT' | 'ALL')

SUPPORTED_REGIMES = ['PRE_EXPANSION', 'EXPANSION', 'RETRACEMENT', 'ALL']

MIN_OOS_AUC = 0.58   # modele sub acest prag sunt ignorate → fallback la ALL

# ── Per-regime thresholds (OOS 2025 sweep analysis) ────────────────────────
# Basis: threshold la care ensemble dă n_signals ≈ old și hit_rate superior.
# La prob < threshold, predicția nu adaugă edge → returnăm 0.5 neutru.
SWEEP_THRESHOLDS: dict = {
    'ALL':           0.55,   # n=207 hr=0.691 ev12=222R (vs old n=211 hr=0.673)
    'PRE_EXPANSION': 0.55,   # n=94  hr=0.638 ev12=86R  (vs old n=96  hr=0.583)
    'EXPANSION':     0.52,   # n=55  hr=0.764 ev12=71R  (old mort >0.57)
    'RETRACEMENT':   0.52,   # n=27  hr=0.667 ev12=27R  (old mort >0.52)
    'CONSOLIDATION': 0.52,   # n=188 hr=0.574 ev12=136R (vs old n=182 hr=0.571)
}


def _load_sweep(regime: str):
    """Încarcă modelul sweep pentru regimul dat. Singleton per regim.
    Prioritate candidați (în ordine):
      1. sweep_{REGIME}_ensemble.pkl  — nou: ALL data + regime boost (cel mai bun)
      2. sweep_{REGIME}.pkl           — vechi: trained doar pe regim (fallback)
      3. sweep_ALL_ensemble.pkl       — fallback universal
    Dacă OOS AUC < MIN_OOS_AUC, modelul e sărit și se încearcă următorul candidat.
    """
    if regime in _SWEEP_MODELS:
        return _SWEEP_MODELS[regime]

    # Candidați în ordinea preferinței
    candidates = []
    if regime == 'ALL':
        candidates = [BASE / 'sweep_ALL_ensemble.pkl', BASE / 'sweep_ALL.pkl']
    else:
        candidates = [
            BASE / f'sweep_{regime}_ensemble.pkl',   # nou — ALL data + boost
            BASE / f'sweep_{regime}.pkl',             # vechi — per-regim pur
        ]

    pkg = None
    for p in candidates:
        if not p.exists():
            continue
        try:
            loaded = pickle.load(open(p, 'rb'))
            oos = loaded.get('oos_auc', 0)
            if oos < MIN_OOS_AUC and regime != 'ALL':
                logger.warning(
                    f"⚠️  Sweep [{regime}] OOS={oos:.3f} < {MIN_OOS_AUC} → ignorat, fallback ALL"
                )
                continue
            pkg = loaded
            ptype = loaded.get('type', 'single')
            logger.info(
                f"✅ Sweep [{regime}] loaded from {p.name}: "
                f"type={ptype} OOS={oos:.3f} | {loaded.get('n_features', len(loaded.get('features', [])))} feats"
            )
            break
        except Exception as e:
            logger.warning(f"Sweep [{regime}] load error {p.name}: {e}")

    _SWEEP_MODELS[regime] = pkg
    return pkg


def _resolve_regime(regime_str: str) -> str:
    """
    Întoarce regimul corect pentru care există model.
    CONSOLIDATION / DISTRIBUTION / UNKNOWN / OTHER → fallback la 'ALL'
    """
    if regime_str in ('PRE_EXPANSION', 'EXPANSION', 'RETRACEMENT'):
        return regime_str
    return 'ALL'


def _predict_pkg(pkg: dict, row: pd.DataFrame) -> float:
    """Prediction dispatcher: handles both single models and ensembles."""
    if pkg.get('type') == 'ensemble':
        models  = pkg['models']
        weights = pkg.get('weights')  # None = equal weight
        preds   = [float(m.predict_proba(row)[0, 1]) for m in models]
        if weights is not None:
            return float(np.average(preds, weights=weights))
        return float(np.mean(preds))
    else:
        return float(pkg['model'].predict_proba(row)[0, 1])


def score_sweep(sweep_feats: dict, regime_str: str = 'ALL') -> float:
    """
    Scorează un setup cu modelul sweep corespunzător regimului curent.

    Args:
        sweep_feats: dict cu features compute în lom_checker / nom_checker
        regime_str:  regimul curent ('PRE_EXPANSION', 'EXPANSION', 'RETRACEMENT', altceva='ALL')

    Returns:
        float [0, 1] — probabilitate win conform sweep model
        0.5 dacă model indisponibil sau regim UNKNOWN (neutru, nu penalizează)
    """
    if not sweep_feats:
        return 0.5

    # UNKNOWN regime: model predictions are inverted (AUC≈0.20 on OOS data).
    # Return 0.5 (no edge) to avoid harmful trades on UNKNOWN setups.
    if regime_str == 'UNKNOWN':
        logger.debug("Sweep: UNKNOWN regime → returning 0.5 (no edge)")
        return 0.5

    target = _resolve_regime(regime_str)
    pkg = _load_sweep(target)

    # Fallback la ALL dacă modelul specific nu e disponibil
    if pkg is None and target != 'ALL':
        logger.debug(f"Sweep [{target}] indisponibil → fallback ALL")
        pkg = _load_sweep('ALL')

    if pkg is None:
        return 0.5

    try:
        features = pkg.get('features', [])
        if not features:
            return 0.5

        row  = pd.DataFrame([sweep_feats]).reindex(columns=features, fill_value=0).fillna(0)
        prob = _predict_pkg(pkg, row)
        return float(np.clip(prob, 0.0, 1.0))

    except Exception as e:
        logger.debug(f"Sweep score error [{target}]: {e}")
        return 0.5


def get_sweep_threshold(regime_str: str) -> float:
    """
    Returnează threshold-ul recomandat per regim (din analiza OOS 2025).
    Folosit în score_sweep_gated() și în orice gate extern.
    """
    return SWEEP_THRESHOLDS.get(regime_str, SWEEP_THRESHOLDS['ALL'])


def score_sweep_gated(sweep_feats: dict, regime_str: str = 'ALL') -> float:
    """
    Ca score_sweep(), dar aplică threshold per regim:
      - prob >= threshold  → returnează prob (edge real)
      - prob <  threshold  → returnează 0.5 (neutru, nu penalizează composite)

    Folosit în ict_gate_v3._apply_stacking() pentru a evita că predicțiile
    sub prag să tragă în jos scorul composite și să blocheze semnale bune.

    Args:
        sweep_feats: dict cu features
        regime_str:  regimul curent

    Returns:
        float [0, 1]
    """
    prob = score_sweep(sweep_feats, regime_str)
    thr  = get_sweep_threshold(regime_str)
    if prob < thr:
        logger.debug(
            f"Sweep [{regime_str}] prob={prob:.3f} < thr={thr} → 0.5 neutru"
        )
        return 0.5
    return prob


def preload_all():
    """Pre-încarcă toate modelele sweep la startup (opțional)."""
    for r in SUPPORTED_REGIMES:
        _load_sweep(r)
    loaded = [r for r, m in _SWEEP_MODELS.items() if m is not None]
    logger.info(f"Sweep models pre-loaded: {loaded}")
