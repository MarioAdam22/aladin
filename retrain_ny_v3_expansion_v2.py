"""
retrain_ny_v3_expansion_v2.py
=============================
Retrains ONLY ny_v3 EXPANSION with GAP_PENALTY=4.0 (+ ALL for script integrity).
Saves: mario_quality_ny_v3_EXPANSION_v2_calibrated.pkl
       mario_quality_ny_v3_ALL_v2_calibrated.pkl   (not used live)
       mario_quality_ny_v3_v2_calibrated.pkl        (not used live)

GAP_PENALTY: penalizes IS-OOS gap > 0.06 in every Optuna trial → prevents overfit.
"""

import sys, pathlib, re, logging
sys.path.insert(0, str(pathlib.Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger("RETRAIN_NYV3_EXP_V2")

SRC_SCRIPT = pathlib.Path(__file__).parent / "train" / "train_quality_ny_v3.py"
GAP_PENALTY = 4.0
TARGET_REGIME = "EXPANSION"

log.info(f"=== Retrain ny_v3 {TARGET_REGIME} v2 cu GAP_PENALTY={GAP_PENALTY} ===")
log.info(f"Source: {SRC_SCRIPT}")

src = SRC_SCRIPT.read_text()

# ── PATCH 1: Add GAP_PENALTY constant after OPTUNA_TRIALS ──────────────────
if "GAP_PENALTY" not in src:
    src = src.replace(
        "OPTUNA_TRIALS = 40",
        f"OPTUNA_TRIALS = 60\nGAP_PENALTY  = {GAP_PENALTY}  # retrain _v2: penalizare IS-OOS gap"
    )
    log.info("✅ GAP_PENALTY injectat")

# ── PATCH 2: Filter ACTIVE_REGIMES to [TARGET_REGIME, 'ALL'] ───────────────
src = re.sub(
    r"ACTIVE_REGIMES\s*=\s*\[.*?\]",
    f"ACTIVE_REGIMES = ['{TARGET_REGIME}', 'ALL']",
    src
)
log.info(f"✅ ACTIVE_REGIMES → ['{TARGET_REGIME}', 'ALL']")

# ── PATCH 3: Add GAP_PENALTY to Optuna obj_regime ──────────────────────────
# First try exact match
OLD_RETURN = (
    "        mdl.fit(X_sm, y_sm, sample_weight=sw_sm,\n"
    "              eval_set=[(X_val, y_val)], verbose=False)\n"
    "        return roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])"
)
NEW_RETURN = (
    "        mdl.fit(X_sm, y_sm, sample_weight=sw_sm,\n"
    "              eval_set=[(X_val, y_val)], verbose=False)\n"
    "        val_auc_t = roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])\n"
    "        # GAP_PENALTY v2: penalize IS-OOS gap > 0.06\n"
    "        try:\n"
    "            is_auc_t = roc_auc_score(y_tr_r, mdl.predict_proba(X_tr_r)[:, 1])\n"
    "            gap_t = max(0, is_auc_t - val_auc_t - 0.06)\n"
    "            return val_auc_t - GAP_PENALTY * gap_t\n"
    "        except Exception:\n"
    "            return val_auc_t"
)
if OLD_RETURN in src:
    src = src.replace(OLD_RETURN, NEW_RETURN)
    log.info("✅ GAP_PENALTY injectat în obj_regime")
else:
    # Try with 4-space indent variant
    OLD_RETURN_4 = (
        "        mdl.fit(X_sm, y_sm, sample_weight=sw_sm,\n"
        "                eval_set=[(X_val, y_val)], verbose=False)\n"
        "        return roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])"
    )
    NEW_RETURN_4 = (
        "        mdl.fit(X_sm, y_sm, sample_weight=sw_sm,\n"
        "                eval_set=[(X_val, y_val)], verbose=False)\n"
        "        val_auc_t = roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])\n"
        "        try:\n"
        "            is_auc_t = roc_auc_score(y_tr_r, mdl.predict_proba(X_tr_r)[:, 1])\n"
        "            gap_t = max(0, is_auc_t - val_auc_t - 0.06)\n"
        "            return val_auc_t - GAP_PENALTY * gap_t\n"
        "        except Exception:\n"
        "            return val_auc_t"
    )
    if OLD_RETURN_4 in src:
        src = src.replace(OLD_RETURN_4, NEW_RETURN_4)
        log.info("✅ GAP_PENALTY injectat (indent variant)")
    else:
        log.warning("⚠️ Nu am găsit exact return pattern — verific snippet:")
        # Find the return line in obj_regime
        idx = src.find("def obj_regime")
        if idx >= 0:
            snippet = src[idx:idx+800]
            log.warning(f"obj_regime snippet:\n{snippet}")

# ── PATCH 4: Rename PKL saves → _v2 ──────────────────────────────────────
src = src.replace(
    'f"mario_quality_ny_v3_{regime_name}_calibrated.pkl"',
    'f"mario_quality_ny_v3_{regime_name}_v2_calibrated.pkl"'
)
src = src.replace(
    '"mario_quality_ny_v3_calibrated.pkl"',
    '"mario_quality_ny_v3_v2_calibrated.pkl"'
)
src = src.replace(
    '"mario_quality_ny_v3_model.json"',
    '"mario_quality_ny_v3_v2_model.json"'
)
src = src.replace(
    '"mario_quality_ny_v3_meta.json"',
    '"mario_quality_ny_v3_v2_meta.json"'
)
log.info("✅ PKL names → _v2")

# ── EXECUTE ────────────────────────────────────────────────────────────────
log.info(f"\n{'='*60}")
log.info(f"Lansez training ny_v3 {TARGET_REGIME} v2 ...")
log.info(f"{'='*60}\n")

exec_globals = {'__file__': str(SRC_SCRIPT), '__name__': '__main__'}
exec(compile(src, str(SRC_SCRIPT), 'exec'), exec_globals)

log.info(f"\n✅ retrain_ny_v3_expansion_v2.py COMPLET")
