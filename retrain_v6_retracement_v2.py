"""
retrain_v6_retracement_v2.py
============================
Retrains ONLY v6 RETRACEMENT with GAP_PENALTY=4.0 (+ ALL for script integrity).
Saves: mario_quality_v6_RETRACEMENT_v2_calibrated.pkl
       mario_quality_v6_ALL_v2_calibrated.pkl   (not used live)
       mario_quality_v6_v2_calibrated.pkl       (not used live)

Approach: patches train_quality_v6.py in-memory, executes modified version.
GAP_PENALTY: penalizes IS-OOS gap > 0.06 in every Optuna trial → prevents overfit.
"""

import sys, pathlib, re, logging
sys.path.insert(0, str(pathlib.Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger("RETRAIN_V6_RET_V2")

SRC_SCRIPT = pathlib.Path(__file__).parent / "train" / "train_quality_v6.py"
GAP_PENALTY = 4.0
TARGET_REGIME = "RETRACEMENT"

log.info(f"=== Retrain v6 {TARGET_REGIME} v2 cu GAP_PENALTY={GAP_PENALTY} ===")
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
    log.warning("⚠️ N-am găsit obj_regime return exact — încerc fallback regex")
    src = re.sub(
        r"(        mdl\.fit\(X_sm, y_sm, sample_weight=sw_sm,\s*\n\s*eval_set=\[\(X_val, y_val\)\], verbose=False\)\s*\n\s*)return roc_auc_score\(y_val, mdl\.predict_proba\(X_val\)\[:, 1\]\)",
        NEW_RETURN,
        src
    )

# ── PATCH 4: Rename PKL saves → _v2 ──────────────────────────────────────
src = src.replace(
    f'"mario_quality_v6_{{}}_calibrated.pkl"'.replace("{}", "{regime_name}"),
    f'"mario_quality_v6_{{regime_name}}_v2_calibrated.pkl"'
)
# Safer f-string aware replacement
src = src.replace(
    'f"mario_quality_v6_{regime_name}_calibrated.pkl"',
    'f"mario_quality_v6_{regime_name}_v2_calibrated.pkl"'
)
src = src.replace(
    '"mario_quality_v6_calibrated.pkl"',
    '"mario_quality_v6_v2_calibrated.pkl"'
)
src = src.replace(
    '"mario_quality_v6_model.json"',
    '"mario_quality_v6_v2_model.json"'
)
src = src.replace(
    '"mario_quality_v6_meta.json"',
    '"mario_quality_v6_v2_meta.json"'
)
log.info("✅ PKL names → _v2")

# ── EXECUTE ────────────────────────────────────────────────────────────────
log.info(f"\n{'='*60}")
log.info(f"Lansez training v6 {TARGET_REGIME} v2 ...")
log.info(f"{'='*60}\n")

exec_globals = {'__file__': str(SRC_SCRIPT), '__name__': '__main__'}
exec(compile(src, str(SRC_SCRIPT), 'exec'), exec_globals)

log.info(f"\n✅ retrain_v6_retracement_v2.py COMPLET")
