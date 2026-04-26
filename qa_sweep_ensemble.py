"""
qa_sweep_ensemble.py — Task C: Sweep Ensemble QA
=================================================
Replay 2025 signals din sweep_dataset through:
  - OLD models:     sweep_{REGIME}.pkl (single model)
  - NEW ensemble:   sweep_{REGIME}_ensemble.pkl (3-seed ensemble)

Output per regime:
  n_signals (score>=0.65), hit_rate@0.65, vs_old (Δhit_rate), auc_old, auc_new

Saves: ~/Desktop/Aladin/qa/sweep_ensemble_qa.md
Log:   /tmp/sweep_ensemble_qa.log
"""

import sys, pathlib, json, warnings, logging
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler('/tmp/sweep_ensemble_qa.log', mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("SWEEP_QA")

BASE    = pathlib.Path(__file__).parent
QA_DIR  = BASE / "qa"
OUT_MD  = QA_DIR / "sweep_ensemble_qa.md"

DATASET_PATH = BASE / "sweep_dataset_2022_2023_2024_2025_v3.parquet"
THRESHOLD    = 0.65
OOS_YEAR     = 2025   # OOS evaluation year

REGIMES      = ['ALL', 'PRE_EXPANSION', 'EXPANSION', 'RETRACEMENT', 'CONSOLIDATION']

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(path):
    """Load PKL safely; returns None if not found."""
    if not path.exists():
        log.warning(f"  ⚠️  Not found: {path.name}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        log.warning(f"  ⚠️  Load error {path.name}: {e}")
        return None


def _align_features(X, feats):
    """Reindex X to exactly the features list, filling missing cols with 0."""
    Xf = pd.DataFrame(index=X.index)
    for f in feats:
        Xf[f] = X[f].fillna(0) if f in X.columns else 0.0
    return Xf.astype(float)


def score_old(pkg, X):
    """Score with old single-model PKL dict (has 'model' and 'features' keys)."""
    feats = pkg.get('features', [])
    mdl   = pkg.get('model')
    if mdl is None or not feats:
        return np.full(len(X), 0.5)
    Xf = _align_features(X, feats)
    try:
        return mdl.predict_proba(Xf)[:, 1]
    except Exception as e:
        log.warning(f"  score_old error: {e}")
        return np.full(len(X), 0.5)


def score_ensemble(pkg, X):
    """Score with new ensemble PKL dict (has 'models', 'weights', 'features')."""
    feats   = pkg.get('features', [])
    models  = pkg.get('models', [])
    raw_w   = pkg.get('weights', None)
    weights = np.atleast_1d(np.array(raw_w)) if raw_w is not None else np.ones(max(len(models),1)) / max(len(models),1)
    if not models or not feats:
        return np.full(len(X), 0.5)
    Xf = _align_features(X, feats)
    preds = []
    for m in models:
        try:
            preds.append(m.predict_proba(Xf)[:, 1])
        except Exception as e:
            log.warning(f"  ensemble model error: {e}")
            preds.append(np.full(len(Xf), 0.5))
    if not preds:
        return np.full(len(X), 0.5)
    # Trim weights to match number of successful predictions
    w = weights[:len(preds)]
    w = w / w.sum() if w.sum() > 0 else np.full(len(preds), 1.0/len(preds))
    return np.average(np.array(preds), axis=0, weights=w)


def compute_auc(y, scores):
    """ROC-AUC; returns nan if only one class."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2 or len(y) < 10:
        return float('nan')
    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return float('nan')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("  SWEEP ENSEMBLE QA — Replay 2025 signals")
    log.info("=" * 65)

    # Load dataset
    log.info(f"\n[1] Loading dataset: {DATASET_PATH.name}")
    if not DATASET_PATH.exists():
        log.error(f"  ❌ Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_parquet(DATASET_PATH)
    log.info(f"  Total: {len(df):,} rows | cols: {len(df.columns)}")

    # Filter to OOS year
    df['_dt'] = pd.to_datetime(df['_date'])
    oos_mask  = df['_dt'].dt.year >= OOS_YEAR
    df_oos    = df[oos_mask].copy().reset_index(drop=True)
    df_is     = df[~oos_mask].copy().reset_index(drop=True)
    log.info(f"  OOS ({OOS_YEAR}+): {len(df_oos):,} rows | IS: {len(df_is):,} rows")

    if df_oos.empty:
        log.error("  ❌ No OOS data found!")
        sys.exit(1)

    # Ensure QA dir
    QA_DIR.mkdir(exist_ok=True)

    # ── Per-regime scoring ────────────────────────────────────────────────────
    rows = []

    for regime in REGIMES:
        log.info(f"\n[Regime: {regime}]")

        # Load old and new models
        old_path = BASE / f"sweep_{regime}.pkl"
        ens_path = BASE / f"sweep_{regime}_ensemble.pkl"
        pkg_old  = load_model(old_path)
        pkg_ens  = load_model(ens_path)

        if pkg_old is None and pkg_ens is None:
            log.warning(f"  Both models missing for {regime} — skip")
            continue

        # Filter to regime subset (for per-regime stats)
        # _regime col has market regime labels; ALL uses everything
        if regime == 'ALL':
            mask = np.ones(len(df_oos), dtype=bool)
        else:
            mask = df_oos['_regime'] == regime

        sub = df_oos[mask].copy()
        if len(sub) < 5:
            log.warning(f"  Too few OOS samples ({len(sub)}) for {regime} — skip")
            continue

        y = sub['_label'].values.astype(int)
        log.info(f"  OOS samples: {len(sub):,} | win rate: {y.mean():.3f}")

        # Score old
        scores_old = score_old(pkg_old, sub) if pkg_old else np.full(len(sub), 0.5)
        # Score ensemble
        scores_ens = score_ensemble(pkg_ens, sub) if pkg_ens else np.full(len(sub), 0.5)

        # Signals at threshold
        sig_old_mask = scores_old >= THRESHOLD
        sig_ens_mask = scores_ens >= THRESHOLD
        n_old = sig_old_mask.sum()
        n_ens = sig_ens_mask.sum()

        # Hit rates
        hr_old = float(y[sig_old_mask].mean()) if n_old > 0 else float('nan')
        hr_ens = float(y[sig_ens_mask].mean()) if n_ens > 0 else float('nan')
        delta_hr = hr_ens - hr_old if not (np.isnan(hr_old) or np.isnan(hr_ens)) else float('nan')

        # AUC
        auc_old = compute_auc(y, scores_old)
        auc_ens = compute_auc(y, scores_ens)

        # Score distribution
        p50_old = float(np.percentile(scores_old, 50))
        p75_old = float(np.percentile(scores_old, 75))
        p50_ens = float(np.percentile(scores_ens, 50))
        p75_ens = float(np.percentile(scores_ens, 75))

        # Log
        log.info(f"  OLD model  : n_signals={n_old} | hit_rate={hr_old:.3f} | AUC={auc_old:.4f} | "
                 f"OOS AUC stored={pkg_old.get('oos_auc', 'N/A') if pkg_old else 'N/A'}")
        log.info(f"  ENS model  : n_signals={n_ens} | hit_rate={hr_ens:.3f} | AUC={auc_ens:.4f} | "
                 f"OOS AUC stored={pkg_ens.get('oos_auc', 'N/A') if pkg_ens else 'N/A'}")
        log.info(f"  Δ hit_rate : {delta_hr:+.3f} | score median: {p50_old:.3f}→{p50_ens:.3f}")

        rows.append({
            'regime':       regime,
            'n_oos':        len(sub),
            'base_wr':      round(float(y.mean()), 3),
            # OLD
            'n_sig_old':    int(n_old),
            'hr_old':       round(hr_old, 3) if not np.isnan(hr_old) else 'N/A',
            'auc_old':      round(auc_old, 4) if not np.isnan(auc_old) else 'N/A',
            'stored_auc_old': round(pkg_old.get('oos_auc', float('nan')), 4) if pkg_old else 'N/A',
            # NEW ensemble
            'n_sig_ens':    int(n_ens),
            'hr_ens':       round(hr_ens, 3) if not np.isnan(hr_ens) else 'N/A',
            'auc_ens':      round(auc_ens, 4) if not np.isnan(auc_ens) else 'N/A',
            'stored_auc_ens': round(pkg_ens.get('oos_auc', float('nan')), 4) if pkg_ens else 'N/A',
            # Delta
            'delta_hr':     f"{delta_hr:+.3f}" if not np.isnan(delta_hr) else 'N/A',
            'delta_n_sig':  int(n_ens) - int(n_old),
            'score_median_old': round(p50_old, 3),
            'score_median_ens': round(p50_ens, 3),
            'score_p75_old':    round(p75_old, 3),
            'score_p75_ens':    round(p75_ens, 3),
        })

    # ── Also score by session (LON vs NY) for ALL regime ─────────────────────
    session_rows = []
    pkg_all_old = load_model(BASE / "sweep_ALL.pkl")
    pkg_all_ens = load_model(BASE / "sweep_ALL_ensemble.pkl")

    if pkg_all_old and pkg_all_ens and '_session' in df_oos.columns:
        for sess in ['LON', 'NY']:
            sub_s = df_oos[df_oos['_session'] == sess]
            if len(sub_s) < 5:
                continue
            y_s = sub_s['_label'].values.astype(int)
            s_old = score_old(pkg_all_old, sub_s)
            s_ens = score_ensemble(pkg_all_ens, sub_s)
            sig_o = s_old >= THRESHOLD
            sig_e = s_ens >= THRESHOLD
            hr_o = float(y_s[sig_o].mean()) if sig_o.sum() > 0 else float('nan')
            hr_e = float(y_s[sig_e].mean()) if sig_e.sum() > 0 else float('nan')
            session_rows.append({
                'session':   sess,
                'n_oos':     len(sub_s),
                'base_wr':   round(float(y_s.mean()), 3),
                'n_sig_old': int(sig_o.sum()),
                'hr_old':    round(hr_o, 3) if not np.isnan(hr_o) else 'N/A',
                'n_sig_ens': int(sig_e.sum()),
                'hr_ens':    round(hr_e, 3) if not np.isnan(hr_e) else 'N/A',
                'delta_hr':  f"{hr_e-hr_o:+.3f}" if not (np.isnan(hr_o) or np.isnan(hr_e)) else 'N/A',
                'auc_old':   round(compute_auc(y_s, s_old), 4),
                'auc_ens':   round(compute_auc(y_s, s_ens), 4),
            })
            log.info(f"  Session {sess}: n={len(sub_s)} | old n_sig={sig_o.sum()} hr={hr_o:.3f} | "
                     f"ens n_sig={sig_e.sum()} hr={hr_e:.3f}")

    # ── Write Markdown report ─────────────────────────────────────────────────
    log.info(f"\n[Writing report] → {OUT_MD}")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(OUT_MD, 'w') as f:
        f.write(f"# Sweep Ensemble QA — {now_str}\n\n")
        f.write(f"**OOS Period:** {OOS_YEAR}+  |  **Threshold:** {THRESHOLD}  "
                f"|  **Dataset:** {DATASET_PATH.name}\n\n")
        f.write(f"**Old:** `sweep_REGIME.pkl` (single model)  \n")
        f.write(f"**New:** `sweep_REGIME_ensemble.pkl` (3-seed ensemble)  \n\n")

        # ── Per-regime table ──────────────────────────────────────────────────
        f.write("## Per-Regime Results\n\n")
        header = (
            "| Regime | n_oos | base_wr | "
            "n_sig_old | hr_old | auc_old | "
            "n_sig_ens | hr_ens | auc_ens | "
            "Δhit_rate | Δn_signals |\n"
        )
        sep = (
            "|--------|-------|---------|"
            "-----------|--------|---------|"
            "-----------|--------|---------|"
            "-----------|------------|\n"
        )
        f.write(header)
        f.write(sep)
        for r in rows:
            f.write(
                f"| {r['regime']:<15} | {r['n_oos']:>5} | {r['base_wr']:>7} | "
                f"{r['n_sig_old']:>9} | {str(r['hr_old']):>6} | {str(r['auc_old']):>7} | "
                f"{r['n_sig_ens']:>9} | {str(r['hr_ens']):>6} | {str(r['auc_ens']):>7} | "
                f"{str(r['delta_hr']):>9} | {r['delta_n_sig']:>10} |\n"
            )
        f.write("\n")

        # ── Score distribution table ──────────────────────────────────────────
        f.write("## Score Distribution (median / p75)\n\n")
        f.write("| Regime | old_p50 | old_p75 | ens_p50 | ens_p75 |\n")
        f.write("|--------|---------|---------|---------|----------|\n")
        for r in rows:
            f.write(
                f"| {r['regime']:<15} | {r['score_median_old']:>7} | {r['score_p75_old']:>7} | "
                f"{r['score_median_ens']:>7} | {r['score_p75_ens']:>8} |\n"
            )
        f.write("\n")

        # ── Session breakdown ─────────────────────────────────────────────────
        if session_rows:
            f.write("## Session Breakdown (ALL regime model)\n\n")
            f.write("| Session | n_oos | base_wr | n_sig_old | hr_old | n_sig_ens | hr_ens | Δhit_rate | auc_old | auc_ens |\n")
            f.write("|---------|-------|---------|-----------|--------|-----------|--------|-----------|---------|----------|\n")
            for r in session_rows:
                f.write(
                    f"| {r['session']:<7} | {r['n_oos']:>5} | {r['base_wr']:>7} | "
                    f"{r['n_sig_old']:>9} | {str(r['hr_old']):>6} | "
                    f"{r['n_sig_ens']:>9} | {str(r['hr_ens']):>6} | "
                    f"{str(r['delta_hr']):>9} | {r['auc_old']:>7} | {r['auc_ens']:>8} |\n"
                )
            f.write("\n")

        # ── Summary ──────────────────────────────────────────────────────────
        f.write("## Summary\n\n")
        if rows:
            all_row = next((r for r in rows if r['regime'] == 'ALL'), None)
            if all_row:
                f.write(f"**ALL regime (full OOS):**  \n")
                f.write(f"- Old model: {all_row['n_sig_old']} signals @ {THRESHOLD}, "
                        f"hit_rate = {all_row['hr_old']}, AUC = {all_row['auc_old']}  \n")
                f.write(f"- Ensemble:  {all_row['n_sig_ens']} signals @ {THRESHOLD}, "
                        f"hit_rate = {all_row['hr_ens']}, AUC = {all_row['auc_ens']}  \n")
                f.write(f"- Δ hit_rate = **{all_row['delta_hr']}** | "
                        f"Δ signals = {all_row['delta_n_sig']:+d}  \n\n")

            # Find best improvement
            valid = [(r['regime'], float(r['delta_hr'])) for r in rows
                     if isinstance(r['delta_hr'], str) and r['delta_hr'] != 'N/A']
            if valid:
                valid.sort(key=lambda x: x[1], reverse=True)
                best_r, best_d = valid[0]
                worst_r, worst_d = valid[-1]
                f.write(f"**Best improvement:** {best_r} (+{best_d:.3f} hit_rate)  \n")
                f.write(f"**Worst:** {worst_r} ({worst_d:+.3f} hit_rate)  \n\n")

        f.write(f"*Generated: {now_str}*\n")

    log.info(f"✅  Report written → {OUT_MD}")

    # Print summary to stdout
    print("\n" + "=" * 65)
    print("SWEEP ENSEMBLE QA RESULTS")
    print("=" * 65)
    header_fmt = f"{'Regime':<16} {'n_oos':>5} {'n_sig_old':>9} {'hr_old':>7} {'n_sig_ens':>9} {'hr_ens':>7} {'Δhr':>7} {'auc_old':>8} {'auc_ens':>8}"
    print(header_fmt)
    print("-" * 90)
    for r in rows:
        print(f"{r['regime']:<16} {r['n_oos']:>5} {r['n_sig_old']:>9} "
              f"{str(r['hr_old']):>7} {r['n_sig_ens']:>9} {str(r['hr_ens']):>7} "
              f"{str(r['delta_hr']):>7} {str(r['auc_old']):>8} {str(r['auc_ens']):>8}")

    log.info("\n=== qa_sweep_ensemble.py COMPLETE ===")


if __name__ == '__main__':
    main()
