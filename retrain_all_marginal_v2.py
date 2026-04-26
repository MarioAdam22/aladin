"""
retrain_all_marginal_v2.py — Task B
=====================================
GAP_PENALTY=4.0 retrain pentru toate modelele MARGINAL rămase.

Modele:
  v6         : EXPANSION (gap 0.143)
  ts_lon_v1  : EXPANSION (gap 0.145), RETRACEMENT (gap 0.128)
  ts_ny_v1   : EXPANSION (gap 0.142), PRE_EXPANSION (gap 0.137), RETRACEMENT (gap 0.107)
  ny_v3      : PRE_EXPANSION (gap 0.124), RETRACEMENT (gap 0.130)

Saves: mario_quality_{model}_{regime}_v2_calibrated.pkl
Updates quality_gate_live.py via _v2 preference (already patched).
"""

import sys, pathlib, re, logging, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger("RETRAIN_MARGINAL_V2")

DIR = pathlib.Path(__file__).parent
GAP_PENALTY = 4.0

JOBS = [
    # (source_script, target_regimes, model_prefix, obj_return_old, use_space_indent)
    (
        "train/train_quality_v6.py",
        ["EXPANSION", "ALL"],
        "mario_quality_v6",
        "v6_EXPANSION",
    ),
    (
        "train/train_quality_ts_lon_v1.py",
        ["EXPANSION", "RETRACEMENT", "ALL"],
        "mario_quality_ts_lon_v1",
        "ts_lon_EXPRET",
    ),
    (
        "train/train_quality_ts_ny_v1.py",
        ["PRE_EXPANSION", "EXPANSION", "RETRACEMENT", "ALL"],
        "mario_quality_ts_ny_v1",
        "ts_ny_PREXPRET",
    ),
    (
        "train/train_quality_ny_v3.py",
        ["PRE_EXPANSION", "RETRACEMENT", "ALL"],
        "mario_quality_ny_v3",
        "ny_v3_PRERET",
    ),
]

# Return patterns for each source script (different indentation styles)
RETURN_PATTERNS = {
    "train/train_quality_v6.py": (
        "        mdl.fit(X_sm, y_sm, sample_weight=sw_sm,\n"
        "              eval_set=[(X_val, y_val)], verbose=False)\n"
        "        return roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])",
        "        mdl.fit(X_sm, y_sm, sample_weight=sw_sm,\n"
        "              eval_set=[(X_val, y_val)], verbose=False)\n"
        "        val_auc_t = roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])\n"
        "        try:\n"
        "            is_auc_t = roc_auc_score(y_tr_r, mdl.predict_proba(X_tr_r)[:, 1])\n"
        "            gap_t = max(0, is_auc_t - val_auc_t - 0.06)\n"
        "            return val_auc_t - GAP_PENALTY * gap_t\n"
        "        except Exception:\n"
        "            return val_auc_t",
    ),
    "train/train_quality_ny_v3.py": (
        "        mdl.fit(X_sm, y_sm, sample_weight=sw_sm,\n"
        "              eval_set=[(X_val, y_val)], verbose=False)\n"
        "        return roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])",
        "        mdl.fit(X_sm, y_sm, sample_weight=sw_sm,\n"
        "              eval_set=[(X_val, y_val)], verbose=False)\n"
        "        val_auc_t = roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])\n"
        "        try:\n"
        "            is_auc_t = roc_auc_score(y_tr_r, mdl.predict_proba(X_tr_r)[:, 1])\n"
        "            gap_t = max(0, is_auc_t - val_auc_t - 0.06)\n"
        "            return val_auc_t - GAP_PENALTY * gap_t\n"
        "        except Exception:\n"
        "            return val_auc_t",
    ),
    # ts_lon and ts_ny have compact indentation
    "train/train_quality_ts_lon_v1.py": (
        "        mdl.fit(X_sm,y_sm,sample_weight=sw_sm,eval_set=[(X_val,y_val)],verbose=False)\n"
        "        return roc_auc_score(y_val,mdl.predict_proba(X_val)[:,1])",
        "        mdl.fit(X_sm,y_sm,sample_weight=sw_sm,eval_set=[(X_val,y_val)],verbose=False)\n"
        "        val_auc_t = roc_auc_score(y_val,mdl.predict_proba(X_val)[:,1])\n"
        "        try:\n"
        "            is_auc_t = roc_auc_score(y_tr_r,mdl.predict_proba(X_tr_r)[:,1])\n"
        "            gap_t = max(0, is_auc_t - val_auc_t - 0.06)\n"
        "            return val_auc_t - GAP_PENALTY * gap_t\n"
        "        except Exception:\n"
        "            return val_auc_t",
    ),
    "train/train_quality_ts_ny_v1.py": (
        "        mdl.fit(X_sm,y_sm,sample_weight=sw_sm,eval_set=[(X_val,y_val)],verbose=False)\n"
        "        return roc_auc_score(y_val,mdl.predict_proba(X_val)[:,1])",
        "        mdl.fit(X_sm,y_sm,sample_weight=sw_sm,eval_set=[(X_val,y_val)],verbose=False)\n"
        "        val_auc_t = roc_auc_score(y_val,mdl.predict_proba(X_val)[:,1])\n"
        "        try:\n"
        "            is_auc_t = roc_auc_score(y_tr_r,mdl.predict_proba(X_tr_r)[:,1])\n"
        "            gap_t = max(0, is_auc_t - val_auc_t - 0.06)\n"
        "            return val_auc_t - GAP_PENALTY * gap_t\n"
        "        except Exception:\n"
        "            return val_auc_t",
    ),
}

# PKL prefixes for rename
PKL_PREFIX_MAP = {
    "train/train_quality_v6.py":        ("mario_quality_v6", "mario_quality_v6_v2"),
    "train/train_quality_ts_lon_v1.py": ("mario_quality_ts_lon_v1", "mario_quality_ts_lon_v1_v2"),
    "train/train_quality_ts_ny_v1.py":  ("mario_quality_ts_ny_v1", "mario_quality_ts_ny_v1_v2"),
    "train/train_quality_ny_v3.py":     ("mario_quality_ny_v3", "mario_quality_ny_v3_v2"),
}

def patch_and_run(src_rel, regimes, job_id):
    src_path = DIR / src_rel
    log.info(f"\n{'='*60}")
    log.info(f"[{job_id}] Patching {src_rel} → regimes={regimes}")
    log.info(f"{'='*60}")

    src = src_path.read_text()

    # Patch 1: GAP_PENALTY
    if "GAP_PENALTY" not in src:
        src = src.replace("OPTUNA_TRIALS = 40",
                          f"OPTUNA_TRIALS = 60\nGAP_PENALTY  = {GAP_PENALTY}")
    log.info("  ✅ GAP_PENALTY injected")

    # Patch 2: ACTIVE_REGIMES
    src = re.sub(r"ACTIVE_REGIMES\s*=\s*\[.*?\]",
                 f"ACTIVE_REGIMES = {regimes}", src)
    log.info(f"  ✅ ACTIVE_REGIMES → {regimes}")

    # Patch 3: obj_regime return
    old_ret, new_ret = RETURN_PATTERNS[src_rel]
    if old_ret in src:
        src = src.replace(old_ret, new_ret)
        log.info("  ✅ GAP_PENALTY logic injected into obj_regime")
    else:
        log.warning(f"  ⚠️  Return pattern not found for {src_rel} — skipping GAP_PENALTY injection")

    # Patch 4: PKL names → _v2
    old_pfx, new_pfx = PKL_PREFIX_MAP[src_rel]
    for suffix in ["_calibrated.pkl", "_model.json", "_meta.json", "_features.json"]:
        # f-string form: f"mario_quality_X_{regime_name}_calibrated.pkl"
        src = src.replace(f'f"{old_pfx}_{{regime_name}}{suffix}"',
                          f'f"{new_pfx}_{{regime_name}}{suffix}"')
        # fixed string form: "mario_quality_X_calibrated.pkl"
        src = src.replace(f'"{old_pfx}{suffix}"', f'"{new_pfx}{suffix}"')
        # pathlib form
        src = src.replace(f'"{old_pfx}_calibrated.pkl"', f'"{new_pfx}_calibrated.pkl"')
    log.info(f"  ✅ PKL names: {old_pfx} → {new_pfx}")

    # Verify compile
    try:
        compile(src, src_rel, 'exec')
        log.info("  ✅ Compile OK")
    except SyntaxError as e:
        log.error(f"  ❌ Compile error: {e}")
        return

    # Execute
    exec_globals = {'__file__': str(src_path), '__name__': '__main__'}
    exec(compile(src, str(src_path), 'exec'), exec_globals)
    log.info(f"  ✅ [{job_id}] DONE")


# Run all jobs
for src_rel, regimes, _prefix, job_id in JOBS:
    try:
        patch_and_run(src_rel, regimes, job_id)
    except Exception as e:
        log.error(f"[{job_id}] FAILED: {e}", exc_info=True)

log.info("\n=== retrain_all_marginal_v2.py COMPLETE ===")
