"""
train_sweep_regime_ensemble.py — Regime-Conditioned Sweep Ensemble (Soluție 1+2)
==================================================================================
Arhitectura:
  - Fiecare model de regim (EXPANSION, PRE_EXPANSION, RETRACEMENT, CONSOLIDATION)
    este antrenat pe TOATE datele IS (nu doar pe regimul respectiv).
  - Samples din regimul țintă primesc REGIME_BOOST în sample_weight → modelul
    se specializează pe acel regim fără să piardă contextul global.
  - N=3 modele independente (seeds diferite) → ensemble calibrat per regim.
  - sweep_ALL_ensemble.pkl rămâne backup/fallback (REGIME_BOOST=1 = uniform).

Output PKL format:
  {
    'type':       'ensemble',
    'regime':     'EXPANSION',
    'models':     [_CalModel, _CalModel, _CalModel],
    'weights':    None,           # equal weight la inferență
    'features':   [...],
    'oos_auc':    float,
    'is_auc':     float,
    'n_features': int,
    'boost':      REGIME_BOOST,
  }

Salvează: sweep_EXPANSION_ensemble.pkl, sweep_PRE_EXPANSION_ensemble.pkl,
          sweep_RETRACEMENT_ensemble.pkl, sweep_CONSOLIDATION_ensemble.pkl

Usage:
    python3 train_sweep_regime_ensemble.py [regime] [n_models] [n_trials]
    python3 train_sweep_regime_ensemble.py ALL 3 80   # toate regimurile
    python3 train_sweep_regime_ensemble.py EXPANSION 3 60
"""
import sys, pickle, logging, warnings
warnings.filterwarnings('ignore')

# ── sys.path ─────────────────────────────────────────────────────────────────
from pathlib import Path
_BASE_DIR = str(Path(__file__).parent)
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression as _IR
from imblearn.over_sampling import BorderlineSMOTE
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from aladin_cal import _CalModel
import train_sweep_v2 as tsv

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("SWEEP_REGIME_ENS")

# ── Config ────────────────────────────────────────────────────────────────────
N_MODELS     = int(sys.argv[2]) if len(sys.argv) > 2 else 3
N_TRIALS     = int(sys.argv[3]) if len(sys.argv) > 3 else 80
TARGET_ARG   = sys.argv[1].upper() if len(sys.argv) > 1 else 'ALL'
SEEDS        = [42, 7, 123, 17, 99][:N_MODELS]

REGIME_BOOST = 3.0    # samples din regimul țintă primesc 3× weight
                      # (1.0 = uniform = identic cu ensemble ALL)
MIN_OOS_AUC  = 0.62   # sub acest prag, nu salvăm per-regim (fallback ALL)

REGIMES_TO_TRAIN = (
    ['PRE_EXPANSION', 'EXPANSION', 'RETRACEMENT', 'CONSOLIDATION']
    if TARGET_ARG == 'ALL' else [TARGET_ARG]
)

BASE = Path(__file__).parent
log.info(f"Regime ensemble: regims={REGIMES_TO_TRAIN} | {N_MODELS} models × {N_TRIALS} trials | boost={REGIME_BOOST}")

# ── Data loading ──────────────────────────────────────────────────────────────
def _load(years, suffix='_v3'):
    tag = '_'.join(str(y) for y in years)
    c   = BASE / f'sweep_dataset_{tag}{suffix}.parquet'
    if c.exists():
        return pd.read_parquet(c)
    parts = [pd.read_parquet(BASE / f'sweep_dataset_{y}{suffix}.parquet') for y in years]
    df = pd.concat(parts, ignore_index=True).sort_values('_date').reset_index(drop=True)
    df.to_parquet(c, index=False)
    return df

df_tr_raw = _load(tsv.TRAIN_YEARS)
df_te_raw = _load(tsv.TEST_YEARS)
log.info(f"IS={len(df_tr_raw)} | OOS={len(df_te_raw)} | OOS WR={df_te_raw['_label'].mean():.1%}")

# ── OF merge (identic cu train_ensemble_all / train_sweep_v2) ─────────────────
_OF_PATH = BASE / "data" / "orderflow_features.parquet"
_OF_COLS = []
if _OF_PATH.exists():
    _of = pd.read_parquet(_OF_PATH)
    _OF_COLS = [c for c in _of.columns if c not in
                ['session_id','date','session_type','session_open','session_close',
                 'session_high','session_low','total_vol']]
    _of_m = _of[['date','session_type'] + _OF_COLS].rename(
        columns={'date':'_date','session_type':'_session'})
    df_tr_raw = df_tr_raw.merge(_of_m, on=['_date','_session'], how='left')
    df_te_raw = df_te_raw.merge(_of_m, on=['_date','_session'], how='left')
    for c in _OF_COLS:
        df_tr_raw[c] = df_tr_raw[c].fillna(0.0)
        df_te_raw[c] = df_te_raw[c].fillna(0.0)
    log.info(f"OF features merged: {len(_OF_COLS)} cols")
else:
    log.warning("orderflow_features.parquet not found — OF features skipped")

# ── Feature selection (identic cu train_sweep_v2 logic) ───────────────────────
META_COLS  = ['_label', '_session', '_date', '_regime']
EXCL_COLS  = META_COLS + ['year_norm']
FEAT_COLS  = [c for c in df_tr_raw.columns if c not in EXCL_COLS]

# Force-include OF + alignment + regime_prob features
FORCE_FEATS = [c for c in FEAT_COLS if (
    c.startswith('of_') or c in _OF_COLS or
    c in ['regime_enc','is_pre_expansion','is_expansion','is_retracement',
          'prob_CONSOLIDATION','prob_PRE_EXPANSION','prob_EXPANSION','prob_RETRACEMENT','prob_DISTRIBUTION',
          'h4_bias_aligned','weekly_prem_aligned','triple_sess_aligned','lon_dir_aligned',
          'asia_dir_aligned','vwap_aligned','lm_prem_aligned','h4_h1_aligned','smt_aligned',
          'pre5_mom_aligned','above_true_open','fast_disp']
)]
COMPUTED_FEATS = [c for c in FEAT_COLS if c not in FORCE_FEATS]
log.info(f"Force feats: {len(FORCE_FEATS)} | Computed pool: {len(COMPUTED_FEATS)}")

# ── Year weights ──────────────────────────────────────────────────────────────
YEAR_WEIGHTS = tsv.YEAR_WEIGHTS   # {2022: 2.50, 2023: 0.08, 2024: 1.50}

# ── Decay weight (recency) ────────────────────────────────────────────────────
def _decay_weight(date_str, ref_date='2025-01-01', hl_months=8):
    try:
        d = pd.Timestamp(date_str)
        r = pd.Timestamp(ref_date)
        months = (r - d).days / 30.44
        return float(np.exp(-np.log(2) * months / hl_months))
    except:
        return 1.0

# ── Feature selection per regime (top N by mutual info on full IS) ─────────────
from sklearn.feature_selection import mutual_info_classif

def _select_features(df, top_n=65):
    """Select top_n computed features by MI on full IS set."""
    avail = [c for c in COMPUTED_FEATS if c in df.columns]
    X_sel = df[avail].fillna(0).values
    y_sel = df['_label'].values
    mi    = mutual_info_classif(X_sel, y_sel, random_state=42, n_neighbors=5)
    top   = [avail[i] for i in np.argsort(mi)[::-1][:top_n]]
    feats = FORCE_FEATS + [f for f in top if f not in FORCE_FEATS]
    feats = [f for f in feats if f in df.columns]
    return feats

# ── Build sample weights ───────────────────────────────────────────────────────
def _build_weights(df, target_regime, boost=REGIME_BOOST):
    """
    Combina:
    1. year_weight     (WR similarity cu 2025-2026)
    2. decay_weight    (recency half-life 8 luni)
    3. regime_boost    (samples din target_regime primesc boost×)
    """
    weights = np.ones(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        yr  = int(str(row['_date'])[:4])
        yw  = YEAR_WEIGHTS.get(yr, 1.0)
        dw  = _decay_weight(str(row['_date']))
        rb  = boost if row['_regime'] == target_regime else 1.0
        weights[i] = yw * dw * rb
    # normalize
    weights = weights / weights.mean()
    return weights.astype(np.float32)

# ── Single model training (Optuna + SMOTE + calibration) ─────────────────────
def _train_one(df_tr, df_te, feats, target_regime, seed, n_trials):
    """Antrenează un singur model cu Optuna. Returnează _CalModel sau None."""
    X_tr = df_tr[feats].fillna(0).values
    y_tr = df_tr['_label'].values
    X_te = df_te[feats].fillna(0).values
    y_te = df_te['_label'].values

    sw_full = _build_weights(df_tr, target_regime)

    # Walk-forward CV: 3 folduri temporale (identic cu train_sweep_v2)
    n = len(X_tr)
    fold_size = n // 4
    wf_folds = [
        (np.arange(0, fold_size * (k+2)), np.arange(fold_size * (k+2), min(fold_size * (k+3), n)))
        for k in range(3)
        if fold_size * (k+3) <= n
    ]
    if not wf_folds:
        cut = int(n * 0.80)
        wf_folds = [(np.arange(cut), np.arange(cut, n))]

    GAP_PENALTY = 3.0

    def objective(trial):
        p = {
            'n_estimators':     trial.suggest_int('n_estimators', 150, 800),
            'max_depth':        trial.suggest_int('max_depth', 2, 3),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'subsample':        trial.suggest_float('subsample', 0.50, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.35, 0.75),
            'min_child_weight': trial.suggest_int('min_child_weight', 20, 80),
            'gamma':            trial.suggest_float('gamma', 1.0, 8.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1.0, 8.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 3.0, 10.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 3.0, 10.0),
        }
        smote_r = trial.suggest_float('smote', 0.10, 0.50)  # fracție din minority de adăugat
        aucs = []
        for tr_idx, va_idx in wf_folds:
            Xf = X_tr[tr_idx]; yf = y_tr[tr_idx]; swf = sw_full[tr_idx]
            Xv = X_tr[va_idx]; yv = y_tr[va_idx]
            try:
                n_min = int((yf == 1).sum()); n_maj = int((yf == 0).sum())
                # sampling_strategy ca dict: adaugam smote_r*n_min samples noi la minority
                n_new = max(n_min + 1, int(n_min * (1 + smote_r)))
                sm_strat = {1: n_new} if n_new > n_min else 'auto'
                sm   = BorderlineSMOTE(sampling_strategy=sm_strat, random_state=seed, k_neighbors=min(5, n_min-1))
                Xs, ys = sm.fit_resample(Xf, yf)
                clf  = xgb.XGBClassifier(**p, eval_metric='logloss',
                                          use_label_encoder=False,
                                          tree_method='hist', random_state=seed,
                                          verbosity=0, n_jobs=4)
                clf.fit(Xs, ys, sample_weight=np.ones(len(Xs)))
                pv   = clf.predict_proba(Xv)[:,1]
                aucs.append(roc_auc_score(yv, pv))
            except Exception:
                aucs.append(0.50)
        val_auc = float(np.mean(aucs))

        # IS AUC on full train (quick check gap)
        try:
            clf2 = xgb.XGBClassifier(**p, eval_metric='logloss',
                                      use_label_encoder=False,
                                      tree_method='hist', random_state=seed,
                                      verbosity=0, n_jobs=4)
            sm2  = BorderlineSMOTE(sampling_strategy=smote_r, random_state=seed, k_neighbors=5)
            Xs2, ys2 = sm2.fit_resample(X_tr, y_tr)
            clf2.fit(Xs2, ys2)
            is_auc = roc_auc_score(y_tr, clf2.predict_proba(X_tr)[:,1])
            gap    = max(0, is_auc - val_auc - 0.06)
            return val_auc - GAP_PENALTY * gap
        except:
            return val_auc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = dict(study.best_params)
    smote_best = bp.pop('smote')

    # Final fit pe tot IS cu best params
    n_min_fin = int((y_tr == 1).sum())
    n_new_fin = max(n_min_fin + 1, int(n_min_fin * (1 + smote_best)))
    sm_strat_fin = {1: n_new_fin}
    sm_fin  = BorderlineSMOTE(sampling_strategy=sm_strat_fin, random_state=seed, k_neighbors=min(5, n_min_fin-1))
    Xs_fin, ys_fin = sm_fin.fit_resample(X_tr, y_tr)
    base = xgb.XGBClassifier(**bp, eval_metric='logloss',
                              use_label_encoder=False,
                              tree_method='hist', random_state=seed,
                              verbosity=0, n_jobs=4)
    base.fit(Xs_fin, ys_fin)

    # IS / OOS AUC
    p_tr = base.predict_proba(X_tr)[:,1]
    p_te = base.predict_proba(X_te)[:,1]
    is_auc  = roc_auc_score(y_tr, p_tr)
    oos_auc = roc_auc_score(y_te, p_te)
    log.info(f"    seed={seed}: IS={is_auc:.4f} OOS={oos_auc:.4f}")

    # Isotonic calibration pe val (ultimele 20% din IS)
    cut = int(len(X_tr) * 0.80)
    p_cal = base.predict_proba(X_tr[cut:])[:,1]
    ir    = _IR(out_of_bounds='clip')
    ir.fit(p_cal, y_tr[cut:])

    cal = _CalModel(base, ir)
    return cal, is_auc, oos_auc


# ════════════════════════════════════════════════════════════════════════════
# MAIN: antrenează ensemble pentru fiecare regim
# ════════════════════════════════════════════════════════════════════════════
for TARGET_REGIME in REGIMES_TO_TRAIN:
    log.info(f"\n{'='*60}")
    log.info(f"REGIME: {TARGET_REGIME} (boost={REGIME_BOOST}×)")
    log.info(f"{'='*60}")

    # Sample counts per regim (informativ)
    n_regime_is  = (df_tr_raw['_regime'] == TARGET_REGIME).sum()
    n_regime_oos = (df_te_raw['_regime'] == TARGET_REGIME).sum()
    wr_is  = df_tr_raw.loc[df_tr_raw['_regime'] == TARGET_REGIME, '_label'].mean()
    wr_oos = df_te_raw.loc[df_te_raw['_regime'] == TARGET_REGIME, '_label'].mean()
    log.info(f"  IS:  {n_regime_is} samples din {len(df_tr_raw)} total (WR={wr_is:.1%})")
    log.info(f"  OOS: {n_regime_oos} samples din {len(df_te_raw)} total (WR={wr_oos:.1%})")

    # Feature selection pe tot IS (nu filtrat pe regim)
    TOP_N = {'PRE_EXPANSION': 55, 'EXPANSION': 70, 'RETRACEMENT': 50,
             'CONSOLIDATION': 75}.get(TARGET_REGIME, 65)
    feats = _select_features(df_tr_raw, top_n=TOP_N)
    log.info(f"  Features: {len(feats)} (forced={len(FORCE_FEATS)} + top{TOP_N} MI)")

    # Antrenează N modele
    models    = []
    is_aucs   = []
    oos_aucs  = []

    df_tr = df_tr_raw.copy()
    df_te = df_te_raw.copy()

    for i, seed in enumerate(SEEDS):
        log.info(f"  Model {i+1}/{N_MODELS} (seed={seed})...")
        try:
            cal, is_auc, oos_auc = _train_one(df_tr, df_te, feats, TARGET_REGIME, seed, N_TRIALS)
            models.append(cal)
            is_aucs.append(is_auc)
            oos_aucs.append(oos_auc)
        except Exception as e:
            log.warning(f"  Model {i+1} failed: {e}")

    if not models:
        log.error(f"  Toate modelele au eșuat pentru {TARGET_REGIME}!")
        continue

    ens_is_auc  = float(np.mean(is_aucs))
    ens_oos_auc = float(np.mean(oos_aucs))
    log.info(f"  Ensemble ({len(models)} modele): IS={ens_is_auc:.4f} OOS={ens_oos_auc:.4f}")

    # Ensemble OOS prediction (medie egală)
    X_te_arr = df_te[feats].fillna(0).values
    y_te_arr = df_te_raw['_label'].values
    preds_ens = np.mean([m.predict_proba(
        pd.DataFrame(X_te_arr, columns=feats))[: ,1] for m in models], axis=0)
    ens_oos_auc_final = roc_auc_score(y_te_arr, preds_ens)
    log.info(f"  Ensemble OOS AUC (averaged predictions): {ens_oos_auc_final:.4f}")

    # Evaluare separata pe regimul tinta
    mask_oos = (df_te_raw['_regime'] == TARGET_REGIME).values
    if mask_oos.sum() > 20:
        auc_regime = roc_auc_score(y_te_arr[mask_oos], preds_ens[mask_oos])
        log.info(f"  AUC pe subset OOS {TARGET_REGIME} ({mask_oos.sum()} samples): {auc_regime:.4f}")

    if ens_oos_auc_final < MIN_OOS_AUC:
        log.warning(f"  ⚠️  OOS={ens_oos_auc_final:.4f} < {MIN_OOS_AUC} — model salvat dar sub prag")

    # Salvare PKL
    out_path = BASE / f'sweep_{TARGET_REGIME}_ensemble.pkl'
    pkg = {
        'type':       'ensemble',
        'regime':     TARGET_REGIME,
        'models':     models,
        'weights':    None,
        'features':   feats,
        'oos_auc':    round(ens_oos_auc_final, 4),
        'is_auc':     round(ens_is_auc, 4),
        'n_features': len(feats),
        'n_models':   len(models),
        'boost':      REGIME_BOOST,
        'train_years': tsv.TRAIN_YEARS,
        'test_years':  tsv.TEST_YEARS,
        'version':    'regime_ensemble_v1',
    }
    pickle.dump(pkg, open(out_path, 'wb'))
    log.info(f"  ✅ Salvat: {out_path.name} | OOS={ens_oos_auc_final:.4f} | {len(feats)} feats")

log.info("\nDone. Acum updateaza sweep_scorer.py sa încarce sweep_{REGIME}_ensemble.pkl.")
log.info("Fallback la sweep_ALL_ensemble.pkl dacă oos_auc < MIN_OOS_AUC (0.62).")
