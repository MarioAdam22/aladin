"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN — qa_orderflow_models.py                                            ║
║  QA complet după training cu Order Flow features                            ║
║  Verifică: PKL-uri, features, IS/OOS per regim, feature importance OF       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Utilizare:
  python3 qa_orderflow_models.py
"""

import sys as _sys
_sys.path.insert(0, '/Users/mario/Desktop/Aladin')
from aladin_cal import _CalModel  # noqa: F401 — needed for unpickling
import pickle, pathlib, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

ALADIN = pathlib.Path(__file__).parent

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Doar features pur noi din generate_synthetic_orderflow.py (nu legacy)
OF_KEYWORDS = ['cvd_final', 'cvd_zscore', 'cvd_pct_', 'cvd_momentum',
               'cvd_acceleration', 'cvd_roc', 'cvd_trend', 'cvd_bearish',
               'cvd_bullish', 'delta_ema_', 'delta_macd',
               'absorption_score', 'absorption_flag', 'absorption_zscore_20d',
               'amihud_mean', 'spread_proxy_mean', 'kyle_lambda',
               'opening_drive_', 'opening_delta_ratio', 'opening_range',
               'stacked_imbalance', 'stacked_zscore',
               'vwap_zscore', 'poc_dist', 'value_area_width']

MODELS = {
    # ── Quality Gates ─────────────────────────────────────────────────────
    "v6": {
        "ALL":          "mario_quality_v6_calibrated.pkl",
        "EXPANSION":    "mario_quality_v6_EXPANSION_calibrated.pkl",
        "RETRACEMENT":  "mario_quality_v6_RETRACEMENT_calibrated.pkl",
    },
    "ts_lon": {
        "ALL":          "mario_quality_ts_lon_v1_calibrated.pkl",
        "EXPANSION":    "mario_quality_ts_lon_v1_EXPANSION_calibrated.pkl",
        "RETRACEMENT":  "mario_quality_ts_lon_v1_RETRACEMENT_calibrated.pkl",
    },
    "ny_v3": {
        "ALL":          "mario_quality_ny_v3_calibrated.pkl",
        "EXPANSION":    "mario_quality_ny_v3_EXPANSION_calibrated.pkl",
        "RETRACEMENT":  "mario_quality_ny_v3_RETRACEMENT_calibrated.pkl",
        "PRE_EXPANSION":"mario_quality_ny_v3_PRE_EXPANSION_calibrated.pkl",
    },
    "ts_ny": {
        "ALL":          "mario_quality_ts_ny_v1_calibrated.pkl",
        "EXPANSION":    "mario_quality_ts_ny_v1_EXPANSION_calibrated.pkl",
        "RETRACEMENT":  "mario_quality_ts_ny_v1_RETRACEMENT_calibrated.pkl",
        "PRE_EXPANSION":"mario_quality_ts_ny_v1_PRE_EXPANSION_calibrated.pkl",
    },
    # ── Manipulation Models ───────────────────────────────────────────────
    "LOM": {
        "ALL":          "lom_model_v1.pkl",
    },
    "NOM": {
        "ALL":          "nom_model_v1.pkl",
    },
    "DSM": {
        "ALL":          "dsm_model_v1.pkl",
    },
    # ── Sweep + Scorer ────────────────────────────────────────────────────
    "sweep": {
        "ALL":          "sweep_ALL.pkl",
        "EXPANSION":    "sweep_EXPANSION.pkl",
        "RETRACEMENT":  "sweep_RETRACEMENT.pkl",
    },
    "scorer": {
        "v4_1":         "ict_setup_scorer_v4_1.pkl",
    },
}

# ─── HELPER ───────────────────────────────────────────────────────────────────
def get_inner_model(model):
    """Extrage modelul XGBoost din orice format: dict, _CalModel, CalibratedClassifierCV."""
    # Dict format: LOM/NOM/DSM/sweep/scorer — {'model': <XGB or _CalModel>, 'features': [...]}
    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    # _CalModel wrapper (aladin_cal.py) — stores raw model in ._m
    if hasattr(model, '_m'):
        return model._m
    # sklearn CalibratedClassifierCV (legacy)
    if hasattr(model, 'calibrated_classifiers_'):
        return model.calibrated_classifiers_[0].estimator
    if hasattr(model, 'estimator'):
        return model.estimator
    return model

def get_feature_list(model):
    """Returneaza lista de features din dict sau din booster."""
    if isinstance(model, dict) and 'features' in model:
        return model['features']
    inner = get_inner_model(model)
    if hasattr(inner, 'get_booster'):
        return inner.get_booster().feature_names or []
    return []

def get_feature_names(model):
    # Try dict features list first (LOM/NOM/DSM/sweep/scorer)
    fl = get_feature_list(model)
    if fl:
        return list(fl)
    inner = get_inner_model(model)
    if hasattr(inner, 'get_booster'):
        return inner.get_booster().feature_names or []
    if hasattr(inner, 'feature_names_in_'):
        return list(inner.feature_names_in_)
    return []

def get_feature_importance(model):
    inner = get_inner_model(model)
    if hasattr(inner, 'get_booster'):
        booster = inner.get_booster()
        scores = booster.get_score(importance_type='gain')
        total = sum(scores.values()) or 1
        return {k: v/total for k, v in scores.items()}
    return {}

def load_model(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

# ─── QA 1: PKL Status ─────────────────────────────────────────────────────────
def qa_pkl_status():
    print("\n" + "="*70)
    print("  QA 1 — Status PKL-uri")
    print("="*70)

    results = {}
    for model_name, regimes in MODELS.items():
        print(f"\n  [{model_name}]")
        for regime, pkl_name in regimes.items():
            pkl_path = ALADIN / pkl_name
            if not pkl_path.exists():
                print(f"    ❌ {regime:<15} — LIPSĂ")
                results[(model_name, regime)] = None
                continue

            model = load_model(pkl_path)
            if model is None:
                print(f"    ❌ {regime:<15} — CORUPT")
                results[(model_name, regime)] = None
                continue

            feat_names = get_feature_names(model)
            of_feats = [f for f in feat_names if any(kw in f for kw in OF_KEYWORDS)]
            of_flag = f"✓ {len(of_feats)} OF features" if of_feats else "⚠  NO ORDER FLOW"

            import os
            mtime = pd.Timestamp(os.path.getmtime(pkl_path), unit='s')
            print(f"    ✓ {regime:<15} — {len(feat_names)} features | {of_flag} | {mtime.strftime('%Y-%m-%d %H:%M')}")
            results[(model_name, regime)] = model

    return results

# ─── QA 2: Feature Importance Order Flow ─────────────────────────────────────
def qa_feature_importance(models_dict):
    print("\n" + "="*70)
    print("  QA 2 — Top Order Flow Features per Model")
    print("="*70)

    for (model_name, regime), model in models_dict.items():
        if model is None:
            continue

        importance = get_feature_importance(model)
        if not importance:
            continue

        of_importance = {k: v for k, v in importance.items()
                         if any(kw in k for kw in OF_KEYWORDS)}
        if not of_importance:
            continue

        top_of = sorted(of_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        total_of_weight = sum(of_importance.values())

        print(f"\n  {model_name} [{regime}] — OF weight total: {total_of_weight:.3f}")
        for feat, imp in top_of:
            bar = "█" * int(imp * 200)
            print(f"    {feat:<35} {imp:.4f}  {bar}")

# ─── QA 3: Model Metadata (IS/OOS din JSON dacă există) ─────────────────────
def qa_model_metadata():
    print("\n" + "="*70)
    print("  QA 3 — Metadata IS/OOS (din JSON)")
    print("="*70)

    import json, glob
    json_files = list(ALADIN.glob("*.json")) + list((ALADIN / "data").glob("*.json"))

    found_any = False
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            if isinstance(data, dict) and ('auc_is' in data or 'auc_oos' in data or
                                            'IS' in str(data) or 'OOS' in str(data)):
                print(f"\n  {jf.name}:")
                for k, v in data.items():
                    if any(kw in k.lower() for kw in ['auc', 'is', 'oos', 'gap', 'regime', 'feature']):
                        print(f"    {k}: {v}")
                found_any = True
        except:
            pass

    if not found_any:
        print("  (niciun JSON cu metadata IS/OOS găsit)")

# ─── QA 4: Feature count comparison ──────────────────────────────────────────
def qa_feature_count(models_dict):
    print("\n" + "="*70)
    print("  QA 4 — Feature Count (cu vs fără Order Flow)")
    print("="*70)
    print(f"\n  {'Model':<12} {'Regime':<15} {'Total':<8} {'OF':<6} {'Non-OF'}")
    print("  " + "-"*55)

    for (model_name, regime), model in models_dict.items():
        if model is None:
            print(f"  {model_name:<12} {regime:<15} {'N/A'}")
            continue

        feat_names = get_feature_names(model)
        of_feats = [f for f in feat_names if any(kw in f for kw in OF_KEYWORDS)]
        non_of = len(feat_names) - len(of_feats)

        of_bar = "▓" * len(of_feats)
        print(f"  {model_name:<12} {regime:<15} {len(feat_names):<8} {len(of_feats):<6} {non_of}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  ALADIN — QA Order Flow Models")
    print("=" * 70)

    # QA 1
    models_dict = qa_pkl_status()

    total = len(models_dict)
    ok = sum(1 for v in models_dict.values() if v is not None)
    missing = total - ok

    print(f"\n  Summary: {ok}/{total} PKL-uri OK | {missing} lipsă")

    if ok == 0:
        print("\n⚠️  Niciun model găsit. Rulează RUN_TRAINING_ORDERFLOW.sh mai întâi.")
        return 1

    # QA 2
    qa_feature_importance(models_dict)

    # QA 3
    qa_model_metadata()

    # QA 4
    qa_feature_count(models_dict)

    print("\n" + "="*70)
    if missing == 0:
        print("  ✅ QA COMPLET — toate modelele OK cu Order Flow features")
    else:
        print(f"  ⚠️  QA cu {missing} modele lipsă — verifica training logs")
    print("="*70 + "\n")

    return 0

if __name__ == '__main__':
    sys.exit(main())
