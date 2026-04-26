"""
train_ensemble_all.py — Ensemble of N independent ALL models (different Optuna seeds).
Averages calibrated OOS probabilities → reduces variance floor, pushes AUC higher.

v3: correct OF merge, default TPE (no multivariate), store cal objects directly.
Usage:
  python3 train_ensemble_all.py [n_models] [n_trials]
  Defaults: 3 models, 80 trials
"""
import sys, pickle, logging, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/sessions/dreamy-youthful-turing/mnt/Aladin')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression as _IR
from imblearn.over_sampling import BorderlineSMOTE
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from aladin_cal import _CalModel
import train_sweep_v2 as tsv

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("ENSEMBLE_ALL")

N_MODELS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 80
SEEDS    = [42, 7, 123, 17, 99, 55, 13][:N_MODELS]

BASE = Path('/sessions/dreamy-youthful-turing/mnt/Aladin')
log.info(f"Ensemble: {N_MODELS} models × {N_TRIALS} trials (seeds={SEEDS})")

# ─── Data loading ─────────────────────────────────────────────────────────────
def _load(years, suffix='_v3'):
    tag = '_'.join(str(y) for y in years)
    c   = BASE / f'sweep_dataset_{tag}{suffix}.parquet'
    if c.exists():
        return pd.read_parquet(c)
    parts = [pd.read_parquet(BASE / f'sweep_dataset_{y}{suffix}.parquet') for y in years]
    df = pd.concat(parts, ignore_index=True).sort_values('_date').reset_index(drop=True)
    df.to_parquet(c, index=False)
    return df

df_tr = _load(tsv.TRAIN_YEARS)
df_te = _load(tsv.TEST_YEARS)
log.info(f"IS={len(df_tr)} rows | OOS={len(df_te)} rows WR={df_te['_label'].mean():.1%}")

# ─── OF features (correct merge, same as train_sweep_v2) ─────────────────────
_OF_PATH = BASE / "data" / "orderflow_features.parquet"
_OF_COLS = []
if _OF_PATH.exists():
    _of = pd.read_parquet(_OF_PATH)
    _OF_COLS = [c for c in _of.columns if c not in
                ['session_id','date','session_type','session_open','session_close',
                 'session_high','session_low','total_vol']]
    _of_m = _of[['date','session_type'] + _OF_COLS].rename(
        columns={'date': '_date', 'session_type': '_session'})
    df_tr = df_tr.merge(_of_m, on=['_date','_session'], how='left')
    df_te = df_te.merge(_of_m, on=['_date','_session'], how='left')
    for c in _OF_COLS:
        df_tr[c] = df_tr[c].fillna(0.0)
        df_te[c] = df_te[c].fillna(0.0)
    log.info(f"OF features: {len(_OF_COLS)}")

# ─── Alignment features ───────────────────────────────────────────────────────
for df_ in [df_tr, df_te]:
    if 'opening_drive_dir' in df_.columns:
        df_['od_aligned'] = (
            ((df_['direction_enc'] == 1) & (df_['opening_drive_dir'] < 0)) |
            ((df_['direction_enc'] == 0) & (df_['opening_drive_dir'] > 0))
        ).astype(float)
    if 'stacked_imbalance_dir' in df_.columns:
        df_['stacked_imb_aligned'] = (
            ((df_['direction_enc'] == 1) & (df_['stacked_imbalance_dir'] < 0)) |
            ((df_['direction_enc'] == 0) & (df_['stacked_imbalance_dir'] > 0))
        ).astype(float)
    if 'cvd_trend_flag' in df_.columns:
        df_['cvd_aligned'] = (
            ((df_['direction_enc'] == 1) & (df_['cvd_trend_flag'] < 0)) |
            ((df_['direction_enc'] == 0) & (df_['cvd_trend_flag'] > 0))
        ).astype(float)

# ─── Regime prob features ─────────────────────────────────────────────────────
_RPROB_COLS = ['regime_prob', 'prob_CONSOLIDATION', 'prob_PRE_EXPANSION',
               'prob_EXPANSION', 'prob_RETRACEMENT', 'prob_DISTRIBUTION']
try:
    _rprob_df = pd.read_parquet(BASE / "data" / "regime_labels.parquet")[
        ['date', 'session'] + _RPROB_COLS
    ].rename(columns={'date': '_date', 'session': '_session'})
    df_tr = df_tr.merge(_rprob_df, on=['_date', '_session'], how='left')
    df_te = df_te.merge(_rprob_df, on=['_date', '_session'], how='left')
    for c in _RPROB_COLS:
        df_tr[c] = df_tr[c].fillna(0.5)
        df_te[c] = df_te[c].fillna(0.5)
    log.info(f"Regime prob merged")
except Exception as e:
    log.warning(f"Regime prob merge failed: {e}")

meta_cols    = [c for c in df_tr.columns if c.startswith('_')]
feature_cols = [c for c in df_tr.columns if c not in meta_cols]

# ─── Force-include ────────────────────────────────────────────────────────────
_OF_FORCE_COLS = [c for c in _OF_COLS if c in feature_cols]
_ALIGN_FORCE   = [c for c in ['od_aligned', 'stacked_imb_aligned', 'cvd_aligned'] if c in feature_cols]
_RPROB_FORCE   = [c for c in _RPROB_COLS if c in feature_cols]
_OF_ALL_FORCE  = list(dict.fromkeys(_OF_FORCE_COLS + _ALIGN_FORCE + _RPROB_FORCE))
log.info(f"Force-include: {len(_OF_ALL_FORCE)} features")

# ─── Weights + feature selection ─────────────────────────────────────────────
_decay_raw  = tsv.compute_decay_weights(df_tr['_date'])
yr_w_global = np.array([tsv.YEAR_WEIGHTS.get(int(d[:4]), 1.0) for d in df_tr['_date']])
sw_global   = (_decay_raw * yr_w_global)
sw_global  /= sw_global.mean()

X_tr_full = df_tr[feature_cols].fillna(0)
y_tr_full = df_tr['_label'].values
X_te_full = df_te[feature_cols].fillna(0).reindex(columns=feature_cols, fill_value=0)
y_te_full = df_te['_label'].values

_comp_cols = [c for c in feature_cols if c not in set(_OF_ALL_FORCE)]
log.info(f"Feature selection: top {tsv.TOP_N_FEATURES} from {len(_comp_cols)} computed cols...")
neg, pos = (y_tr_full == 0).sum(), (y_tr_full == 1).sum()
_pre = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                          subsample=0.7, colsample_bytree=0.6, min_child_weight=20,
                          gamma=2.0, reg_alpha=2.0, reg_lambda=5.0,
                          scale_pos_weight=neg / max(pos, 1),
                          random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0)
_pre.fit(X_tr_full[_comp_cols], y_tr_full, sample_weight=sw_global, verbose=False)
imp = pd.Series(_pre.feature_importances_, index=_comp_cols).sort_values(ascending=False)
selected = _OF_ALL_FORCE + imp.head(tsv.TOP_N_FEATURES).index.tolist()
log.info(f"Feature pool: {len(selected)} total")

X_tr = X_tr_full[selected]
X_te = X_te_full.reindex(columns=selected, fill_value=0)

# ─── Val split: year-2024 boundary ───────────────────────────────────────────
val_cut = int((df_tr['_date'].str[:4].astype(int) < 2024).sum())
log.info(f"Val split: {val_cut} IS rows (2022-2023), {len(X_tr)-val_cut} val rows (2024)")

X_val = X_tr.iloc[val_cut:]
y_val = y_tr_full[val_cut:]
X_tr2 = X_tr.iloc[:val_cut]
y_tr2 = y_tr_full[:val_cut]

_ryw = tsv.REGIME_YEAR_WEIGHTS['ALL']
_yr_w_r = np.array([_ryw.get(int(d[:4]), 1.0) for d in df_tr['_date'][:val_cut]])
sw2 = (_decay_raw[:val_cut] * _yr_w_r)
sw2 /= sw2.mean()

# Pre-compute WR adjustment for val (constant across trials)
_oos_wr = float(y_te_full.mean())
_val_wr = float(y_val.mean())
if 0.1 < _val_wr < 0.9 and abs(_val_wr - _oos_wr) > 0.03:
    _wadj = np.where(y_val == 1, _oos_wr / _val_wr,
                     (1 - _oos_wr) / max(1 - _val_wr, 1e-6))
else:
    _wadj = None
log.info(f"OOS WR={_oos_wr:.1%} val WR={_val_wr:.1%} → WR-adj={'yes' if _wadj is not None else 'no'}")

GAP_PENALTY = tsv.GAP_PENALTY

# ─── Per-seed training ────────────────────────────────────────────────────────
all_cal_models = []
all_oos_probas = []
all_val_probas = []
all_val_aucs   = []

for seed_idx, seed in enumerate(SEEDS):
    log.info(f"\n{'='*50}\nModel {seed_idx+1}/{N_MODELS} (seed={seed})\n{'='*50}")

    _seed = seed  # capture for closure

    def objective(trial, _s=_seed):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 60, 200),
            'max_depth':        trial.suggest_int('max_depth', 2, 3),
            'learning_rate':    trial.suggest_float('lr', 0.008, 0.05, log=True),
            'subsample':        trial.suggest_float('sub', 0.45, 0.80),
            'colsample_bytree': trial.suggest_float('col', 0.35, 0.70),
            'min_child_weight': trial.suggest_int('mcw', 30, 100),
            'gamma':            trial.suggest_float('gamma', 3.0, 15.0),
            'reg_alpha':        trial.suggest_float('alpha', 1.0, 8.0),
            'reg_lambda':       trial.suggest_float('lambda', 3.0, 12.0),
            'scale_pos_weight': trial.suggest_float('spw', 2.0, 8.0),
        }
        smote_r = trial.suggest_float('smote_r', 0.55, 0.90)
        try:
            sm = BorderlineSMOTE(sampling_strategy=smote_r, random_state=_s,
                                 k_neighbors=min(5, int(y_tr2.sum()) - 1))
            Xs, ys = sm.fit_resample(X_tr2, y_tr2)
            sws = np.concatenate([sw2, np.ones(len(Xs) - len(X_tr2))])
        except:
            Xs, ys, sws = X_tr2, y_tr2, sw2

        m = xgb.XGBClassifier(**params, eval_metric='logloss',
                               random_state=_s, n_jobs=-1,
                               tree_method='hist', verbosity=0)
        m.fit(Xs, ys, sample_weight=sws, verbose=False)

        val_proba = m.predict_proba(X_val)[:, 1]
        if _wadj is not None:
            val_auc = roc_auc_score(y_val, val_proba, sample_weight=_wadj)
        else:
            val_auc = roc_auc_score(y_val, val_proba)

        is_auc = roc_auc_score(ys, m.predict_proba(Xs)[:, 1])
        gap_penalty = GAP_PENALTY * max(0, is_auc - val_auc - 0.06)
        return val_auc - gap_penalty

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed)  # default non-multivariate
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False, n_jobs=1)

    bp = dict(study.best_params)
    smote_best = bp.pop('smote_r')
    log.info(f"  seed={seed}: best_val={study.best_value:.4f} n_est={bp['n_estimators']}")

    # Final fit — store cal directly, no re-fit needed
    try:
        sm = BorderlineSMOTE(sampling_strategy=smote_best, random_state=seed,
                             k_neighbors=min(5, int(y_tr2.sum()) - 1))
        Xs, ys = sm.fit_resample(X_tr2, y_tr2)
        sws = np.concatenate([sw2, np.ones(len(Xs) - len(X_tr2))])
    except:
        Xs, ys, sws = X_tr2, y_tr2, sw2

    base = xgb.XGBClassifier(**bp, eval_metric='logloss',
                              random_state=seed, n_jobs=-1, tree_method='hist', verbosity=0)
    base.fit(Xs, ys, sample_weight=sws, verbose=False)

    raw_val = base.predict_proba(X_val)[:, 1]
    ir_cal  = _IR(out_of_bounds='clip').fit(raw_val, y_val)
    cal     = _CalModel(base, ir_cal)

    all_cal_models.append(cal)
    oos_p = cal.predict_proba(X_te)[:, 1]
    val_p = cal.predict_proba(X_val)[:, 1]
    all_oos_probas.append(oos_p)
    all_val_probas.append(val_p)

    single_oos = roc_auc_score(y_te_full, oos_p)
    single_val = roc_auc_score(y_val, val_p)
    all_val_aucs.append(single_val)
    is_auc = roc_auc_score(y_tr2, cal.predict_proba(X_tr2)[:, 1])
    log.info(f"  seed={seed}: IS={is_auc:.4f} OOS={single_oos:.4f} gap={is_auc-single_oos:.3f}")

# ─── Ensemble ─────────────────────────────────────────────────────────────────
oos_stack = np.stack(all_oos_probas, axis=1)
val_stack = np.stack(all_val_probas, axis=1)

ens_oos_eq  = oos_stack.mean(axis=1)
ens_val_eq  = val_stack.mean(axis=1)
auc_oos_eq  = roc_auc_score(y_te_full, ens_oos_eq)
auc_val_eq  = roc_auc_score(y_val, ens_val_eq)

val_aucs_arr = np.array(all_val_aucs)
weights = val_aucs_arr / val_aucs_arr.sum()
ens_oos_wt  = (oos_stack * weights).sum(axis=1)
auc_oos_wt  = roc_auc_score(y_te_full, ens_oos_wt)

best_ens = ens_oos_eq if auc_oos_eq >= auc_oos_wt else ens_oos_wt
best_auc = max(auc_oos_eq, auc_oos_wt)
best_type = 'equal' if auc_oos_eq >= auc_oos_wt else 'val_weighted'

log.info(f"\n{'='*60}")
log.info(f"ENSEMBLE ({N_MODELS} models × {N_TRIALS} trials) RESULTS:")
log.info(f"  Equal-weight:  OOS={auc_oos_eq:.4f}")
log.info(f"  Val-weighted:  OOS={auc_oos_wt:.4f}  (w={np.round(weights,3)})")
log.info(f"  Best: {best_type} → OOS={best_auc:.4f}")
log.info(f"  Individual OOSs: {[round(roc_auc_score(y_te_full, p), 4) for p in all_oos_probas]}")
for thr in [0.50, 0.55, 0.60]:
    m_t = best_ens >= thr
    if m_t.sum() > 5:
        log.info(f"  WR@{thr}: {float(y_te_full[m_t].mean()):.1%} ({int(m_t.sum())} setups)")
log.info(f"{'='*60}")

# ─── Save ─────────────────────────────────────────────────────────────────────
pkg = {
    'type':           'ensemble',
    'ensemble_type':  best_type,
    'n_models':       N_MODELS,
    'n_trials':       N_TRIALS,
    'seeds':          SEEDS,
    'features':       selected,
    'regime':         'ALL',
    'oos_auc':        round(best_auc, 4),
    'val_auc':        round(auc_val_eq, 4),
    'train_years':    tsv.TRAIN_YEARS,
    'version':        'ensemble_v3',
    'models':         all_cal_models,          # list of _CalModel objects
    'weights':        weights.tolist() if best_type == 'val_weighted' else None,
    'n_samples_is':   len(X_tr2),
}

out_path = BASE / 'sweep_ALL_ensemble.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(pkg, f)
log.info(f"\n✅ Saved: {out_path.name} | OOS={best_auc:.4f} ({best_type}) | {N_MODELS} models")
