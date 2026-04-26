"""
retrain_v2b.py — GAP_PENALTY=2.5 fix pentru modelele over-regularizate din v2
===============================================================================
Problema: retrain_all_marginal_v2.py a folosit GAP_PENALTY=4.0 → prea agresiv
pentru 5 modele (ts_lon/EXPANSION, ts_ny/PRE_EXP+EXP+RET, ny_v3/RETRACEMENT).

Soluție: re-antrenare cu GAP_PENALTY=2.5 + naming corect pentru quality_gate_live.py:
  {prefix}_{regime}_v2_calibrated.pkl   ← format așteptat de quality_gate_live.py

Modele retrenate:
  ts_lon  : EXPANSION, ALL
  ts_ny   : PRE_EXPANSION, EXPANSION, RETRACEMENT, ALL
  ny_v3   : RETRACEMENT, ALL

Saves:
  mario_quality_ts_lon_v1_EXPANSION_v2_calibrated.pkl
  mario_quality_ts_ny_v1_PRE_EXPANSION_v2_calibrated.pkl
  mario_quality_ts_ny_v1_EXPANSION_v2_calibrated.pkl
  mario_quality_ts_ny_v1_RETRACEMENT_v2_calibrated.pkl
  mario_quality_ny_v3_RETRACEMENT_v2_calibrated.pkl
"""

import sys, pathlib, re, logging
sys.path.insert(0, str(pathlib.Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger("RETRAIN_V2B")

DIR          = pathlib.Path(__file__).parent
GAP_PENALTY  = 2.5

# ── Jobs: (source_script, ACTIVE_REGIMES, job_id) ────────────────────────────
JOBS = [
    (
        "train/train_quality_ts_lon_v1.py",
        ["EXPANSION", "ALL"],
        "ts_lon_v2b",
    ),
    (
        "train/train_quality_ts_ny_v1.py",
        ["PRE_EXPANSION", "EXPANSION", "RETRACEMENT", "ALL"],
        "ts_ny_v2b",
    ),
    (
        "train/train_quality_ny_v3.py",
        ["RETRACEMENT", "ALL"],
        "ny_v3_v2b",
    ),
]

# ── Original return pattern (no GAP_PENALTY) per source script ────────────────
RETURN_PATTERNS = {
    "train/train_quality_ts_lon_v1.py": (
        "        mdl.fit(X_sm,y_sm,sample_weight=sw_sm,eval_set=[(X_val,y_val)],verbose=False)\n"
        "        return roc_auc_score(y_val,mdl.predict_proba(X_val)[:,1])",
        "        mdl.fit(X_sm,y_sm,sample_weight=sw_sm,eval_set=[(X_val,y_val)],verbose=False)\n"
        "        val_auc_t = roc_auc_score(y_val,mdl.predict_proba(X_val)[:,1])\n"
        "        try:\n"
        "            is_auc_t = roc_auc_score(y_tr_r,mdl.predict_proba(X_tr_r)[:,1])\n"
        "            gap_t = max(0, is_auc_t - val_auc_t - 0.06)\n"
        f"            return val_auc_t - {GAP_PENALTY} * gap_t\n"
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
        f"            return val_auc_t - {GAP_PENALTY} * gap_t\n"
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
        f"            return val_auc_t - {GAP_PENALTY} * gap_t\n"
        "        except Exception:\n"
        "            return val_auc_t",
    ),
}

# ── PKL renames: regime-specific only (correct naming for quality_gate_live.py) ─
# quality_gate_live.py expects: {prefix}_{regime}_v2_calibrated.pkl
# Original:  f"mario_quality_{model}_{regime_name}_calibrated.pkl"
# New:       f"mario_quality_{model}_{regime_name}_v2_calibrated.pkl"
PKL_REGIME_NAMES = {
    "train/train_quality_ts_lon_v1.py": "mario_quality_ts_lon_v1",
    "train/train_quality_ts_ny_v1.py":  "mario_quality_ts_ny_v1",
    "train/train_quality_ny_v3.py":     "mario_quality_ny_v3",
}


def patch_and_run(src_rel, regimes, job_id):
    src_path = DIR / src_rel
    log.info(f"\n{'='*60}")
    log.info(f"[{job_id}] Patching {src_rel} → regimes={regimes} GAP_PENALTY={GAP_PENALTY}")
    log.info(f"{'='*60}")

    src = src_path.read_text()

    # Patch 1: GAP_PENALTY constant (inject if not present)
    if "GAP_PENALTY" not in src:
        src = src.replace("OPTUNA_TRIALS = 40",
                          f"OPTUNA_TRIALS = 60\nGAP_PENALTY  = {GAP_PENALTY}")
    log.info(f"  ✅ GAP_PENALTY = {GAP_PENALTY}")

    # Patch 2: ACTIVE_REGIMES
    src = re.sub(r"ACTIVE_REGIMES\s*=\s*\[.*?\]",
                 f"ACTIVE_REGIMES = {regimes}", src)
    log.info(f"  ✅ ACTIVE_REGIMES → {regimes}")

    # Patch 3: obj_regime return → GAP_PENALTY logic
    old_ret, new_ret = RETURN_PATTERNS[src_rel]
    if old_ret in src:
        src = src.replace(old_ret, new_ret)
        log.info("  ✅ GAP_PENALTY logic injected into obj_regime")
    else:
        log.warning(f"  ⚠️  Return pattern not found — skipping GAP_PENALTY injection")

    # Patch 4: PKL regime-specific naming → {regime}_v2_calibrated.pkl
    # (CORRECT format expected by quality_gate_live.py)
    pfx = PKL_REGIME_NAMES[src_rel]
    old_pkl = f'f"{pfx}_{{regime_name}}_calibrated.pkl"'
    new_pkl = f'f"{pfx}_{{regime_name}}_v2_calibrated.pkl"'
    if old_pkl in src:
        src = src.replace(old_pkl, new_pkl)
        log.info(f"  ✅ PKL regime names: {pfx}_{{regime}}_calibrated → {pfx}_{{regime}}_v2_calibrated")
    else:
        log.warning(f"  ⚠️  PKL f-string pattern not found for {src_rel}")

    # Also handle alternative f-string forms without spaces
    # e.g. f'{pfx}_{regime_name}_calibrated.pkl'
    for q in ['"', "'"]:
        old2 = f"f{q}{pfx}_{{regime_name}}_calibrated.pkl{q}"
        new2 = f"f{q}{pfx}_{{regime_name}}_v2_calibrated.pkl{q}"
        if old2 in src:
            src = src.replace(old2, new2)
            log.info(f"  ✅ PKL alt form patched ({q})")

    # Verify compile
    try:
        compile(src, src_rel, 'exec')
        log.info("  ✅ Compile OK")
    except SyntaxError as e:
        log.error(f"  ❌ Compile error: {e}")
        return False

    # Execute
    exec_globals = {'__file__': str(src_path), '__name__': '__main__'}
    exec(compile(src, str(src_path), 'exec'), exec_globals)
    log.info(f"  ✅ [{job_id}] DONE")
    return True


# ── Run all jobs ──────────────────────────────────────────────────────────────
log.info(f"=== retrain_v2b.py START (GAP_PENALTY={GAP_PENALTY}) ===")
for src_rel, regimes, job_id in JOBS:
    try:
        ok = patch_and_run(src_rel, regimes, job_id)
        if not ok:
            log.error(f"[{job_id}] FAILED to patch/compile")
    except Exception as e:
        log.error(f"[{job_id}] EXCEPTION: {e}", exc_info=True)

log.info("\n=== retrain_v2b.py COMPLETE ===")
