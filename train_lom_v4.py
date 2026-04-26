"""
train_lom_v4.py — London Open Manipulation Ensemble v4
=======================================================
Fix pentru overfit sever din v1/v3 (IS=0.974 vs OOS=0.677):

  ✅ Ensemble N=3 modele independente (seeds 42, 7, 123)
  ✅ IsotonicRegression calibration → _CalModel (identic cu sweep)
  ✅ TOP_N_FEATURES = 35 (de la 75 — regula N/10 cu 348 samples)
  ✅ GAP_PENALTY = 4.0 în Optuna (mai agresiv decât sweep datorită sample-ului mic)
  ✅ 3 walk-forward folds (de la 1)
  ✅ SMOTE ca fracție din minority (nu ratio absolut — fix bug v3)
  ✅ Regularizare mai strictă: min_child_weight 30-120, gamma 2-12
  ✅ Adaugă 2025 în training (suntem în 2026)
  ✅ Salvează în format ensemble PKL (compatibil cu sweep_scorer dispatch)

Output: lom_model_v4.pkl
Checker: schimbă MODEL_PATH în lom_checker_v1.py → lom_model_v4.pkl

Usage:
    python3 train_lom_v4.py [n_models] [n_trials]
    Defaults: 3 modele, 80 trials
"""
import sys, pickle, logging, warnings
warnings.filterwarnings('ignore')

from pathlib import Path
_BASE = str(Path(__file__).parent)
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression as _IR
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import BorderlineSMOTE
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from aladin_cal import _CalModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("LOM_V4")

# ── Config ────────────────────────────────────────────────────────────────────
N_MODELS     = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_TRIALS     = int(sys.argv[2]) if len(sys.argv) > 2 else 80
SEEDS        = [42, 7, 123][:N_MODELS]
OUT          = Path(_BASE) / "lom_model_v4.pkl"

# Include 2025 în training — suntem în 2026, 2025 nu mai e "viitor"
# Dar dăm greutate mai mică ca să nu domine (2025 e bullish, diferit de 2022-2024)
TRAIN_YEARS  = [2022, 2023, 2024, 2025]
TEST_YEARS   = [2026]           # OOS: doar 2026 (4 luni)
# Fallback: dacă 2026 are prea puține date, folosim și 2025 ca OOS
FALLBACK_OOS = [2025, 2026]

YEAR_WEIGHTS = {2019: 0.40, 2020: 0.50, 2021: 0.60,
                2022: 0.90, 2023: 0.40, 2024: 1.00, 2025: 0.70}

TOP_N        = 35    # max features — regula N/10 cu ~350-450 IS samples
GAP_PENALTY  = 4.0   # mai agresiv decât sweep (3.0) datorită sample mic
MIN_OOS_AUC  = 0.62

log.info(f"LOM v4 Ensemble: {N_MODELS} modele × {N_TRIALS} trials | TOP_N={TOP_N} | GAP={GAP_PENALTY}")

# ── Import build_dataset din v3 (reutilizăm extracția de features) ────────────
log.info("Importând build_dataset din train_lom_v3 ...")
import train_lom_v3 as _v3

# Suprascrie TRAIN/TEST YEARS în v3 înainte de build_dataset
_v3.TRAIN_YEARS = TRAIN_YEARS
_v3.TEST_YEARS  = TEST_YEARS

# ── Build datasets ─────────────────────────────────────────────────────────────
log.info(f"Extragere IS ({TRAIN_YEARS}) ...")
regime_pkg    = _v3.load_regime_classifier()
of_lag_lookup = _v3.load_of_lag_features()
df_tr = _v3.build_dataset(TRAIN_YEARS, regime_pkg=regime_pkg, of_lag_lookup=of_lag_lookup)

log.info(f"Extragere OOS ({TEST_YEARS}) ...")
df_te = _v3.build_dataset(TEST_YEARS, regime_pkg=regime_pkg, of_lag_lookup=of_lag_lookup)

# Dacă OOS are sub 30 samples, folosim fallback
if len(df_te) < 30:
    log.warning(f"OOS prea mic ({len(df_te)} samples) → fallback la {FALLBACK_OOS}")
    df_te = _v3.build_dataset(FALLBACK_OOS, regime_pkg=regime_pkg, of_lag_lookup=of_lag_lookup)

log.info(f"IS={len(df_tr)} samples | OOS={len(df_te)} samples")

if len(df_tr) < 50:
    log.error("Prea puține IS samples — verifică DB-ul.")
    sys.exit(1)

# ── Feature selection ─────────────────────────────────────────────────────────
META_COLS = [c for c in df_tr.columns if c.startswith('_') or
             c in ['date','session','direction','entry_time','entry_ts','exit_ts']]
y_tr_all  = df_tr['_label'].values
y_te_all  = df_te['_label'].values if len(df_te) > 0 else np.array([])

FEAT_COLS = [c for c in df_tr.columns if c not in META_COLS + ['_label']]
FEAT_COLS = [c for c in FEAT_COLS if df_tr[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

# Force include: regime + OF + alignment (nu tăia din astea)
FORCE = [c for c in FEAT_COLS if any(x in c for x in [
    'regime_enc', 'is_pre_expansion', 'is_expansion', 'is_retracement',
    'of_', 'h4_bias_aligned', 'h4_h1_aligned', 'weekly_prem_aligned',
    'triple_sess_aligned', 'vwap_aligned', 'smt_aligned', 'asia_dir_aligned',
    'fvg_tf_confluence', 'htf_fvg_aligned',
])]
COMPUTED = [c for c in FEAT_COLS if c not in FORCE]

log.info(f"Total features: {len(FEAT_COLS)} | Force: {len(FORCE)} | Computed pool: {len(COMPUTED)}")

# Mutual info feature selection pe computed
X_sel = df_tr[COMPUTED].fillna(0).values
mi    = mutual_info_classif(X_sel, y_tr_all, random_state=42, n_neighbors=5)
top_computed = [COMPUTED[i] for i in np.argsort(mi)[::-1][:max(1, TOP_N - len(FORCE))]]
FEATURES = FORCE + [f for f in top_computed if f not in FORCE]
FEATURES = [f for f in FEATURES if f in df_tr.columns]

log.info(f"Features selectate: {len(FEATURES)} (force={len(FORCE)} + top_MI={len(top_computed)})")

X_tr = df_tr[FEATURES].fillna(0).values
X_te = df_te[FEATURES].fillna(0).values if len(df_te) > 0 else np.zeros((0, len(FEATURES)))
y_tr = y_tr_all
y_te = y_te_all

# ── Sample weights (year decay) ───────────────────────────────────────────────
def _sw(df):
    w = np.ones(len(df))
    for i, row in enumerate(df['_date'].values):
        yr = int(str(row)[:4])
        w[i] = YEAR_WEIGHTS.get(yr, 0.70)
    return (w / w.mean()).astype(np.float32)

sw_tr = _sw(df_tr)

# ── Walk-forward folds (3 folduri temporale) ───────────────────────────────────
n = len(X_tr)
fs = n // 4
wf_folds = [
    (np.arange(0, fs * (k+2)), np.arange(fs * (k+2), min(fs * (k+3), n)))
    for k in range(3) if fs * (k+3) <= n and fs * (k+2) >= 20
]
if not wf_folds:
    cut = int(n * 0.80)
    wf_folds = [(np.arange(cut), np.arange(cut, n))]
log.info(f"Walk-forward folds: {len(wf_folds)} | fold sizes: {[len(v) for _,v in wf_folds]}")

# ── Single model training ──────────────────────────────────────────────────────
def _train_one(seed):
    def objective(trial):
        p = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
            'max_depth':        trial.suggest_int('max_depth', 2, 3),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'subsample':        trial.suggest_float('subsample', 0.45, 0.80),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.30, 0.70),
            'min_child_weight': trial.suggest_int('min_child_weight', 30, 120),
            'gamma':            trial.suggest_float('gamma', 2.0, 12.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 2.0, 12.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 4.0, 15.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2.0, 10.0),
        }
        smote_r = trial.suggest_float('smote', 0.10, 0.60)  # fracție din minority de adăugat

        val_aucs = []
        for tr_idx, va_idx in wf_folds:
            Xf = X_tr[tr_idx]; yf = y_tr[tr_idx]; swf = sw_tr[tr_idx]
            Xv = X_tr[va_idx]; yv = y_tr[va_idx]
            if yv.sum() < 3 or (len(yv) - yv.sum()) < 3:
                continue
            try:
                n_min = int((yf == 1).sum())
                n_new = max(n_min + 1, int(n_min * (1 + smote_r)))
                sm = BorderlineSMOTE(sampling_strategy={1: n_new},
                                     random_state=seed, k_neighbors=min(5, n_min - 1))
                Xs, ys = sm.fit_resample(Xf, yf)
                sws = np.concatenate([swf, np.ones(len(Xs) - len(Xf))])
            except Exception:
                Xs, ys, sws = Xf, yf, swf

            clf = xgb.XGBClassifier(**p, eval_metric='logloss', tree_method='hist',
                                     random_state=seed, verbosity=0, n_jobs=4)
            clf.fit(Xs, ys, sample_weight=sws)
            pv = clf.predict_proba(Xv)[:, 1]
            if yv.sum() > 0 and yv.sum() < len(yv):
                val_aucs.append(roc_auc_score(yv, pv))

        if not val_aucs:
            return 0.5
        val_auc = float(np.mean(val_aucs))

        # GAP_PENALTY: penalizăm IS-OOS gap > 0.06
        try:
            sm2 = BorderlineSMOTE(sampling_strategy={1: int((y_tr == 1).sum() * (1 + smote_r))},
                                   random_state=seed, k_neighbors=min(5, int((y_tr==1).sum())-1))
            Xs2, ys2 = sm2.fit_resample(X_tr, y_tr)
            clf2 = xgb.XGBClassifier(**p, eval_metric='logloss', tree_method='hist',
                                      random_state=seed, verbosity=0, n_jobs=4)
            clf2.fit(Xs2, ys2)
            is_auc = roc_auc_score(y_tr, clf2.predict_proba(X_tr)[:, 1])
            gap = max(0, is_auc - val_auc - 0.06)
            return val_auc - GAP_PENALTY * gap
        except Exception:
            return val_auc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    bp = dict(study.best_params)
    smote_best = bp.pop('smote')
    log.info(f"  seed={seed}: best_val={study.best_value:.4f}")

    # Final fit pe tot IS
    n_min_fin = int((y_tr == 1).sum())
    n_new_fin = max(n_min_fin + 1, int(n_min_fin * (1 + smote_best)))
    try:
        sm_fin = BorderlineSMOTE(sampling_strategy={1: n_new_fin},
                                  random_state=seed, k_neighbors=min(5, n_min_fin - 1))
        Xs_fin, ys_fin = sm_fin.fit_resample(X_tr, y_tr)
        sws_fin = np.concatenate([sw_tr, np.ones(len(Xs_fin) - len(X_tr))])
    except Exception:
        Xs_fin, ys_fin, sws_fin = X_tr, y_tr, sw_tr

    base = xgb.XGBClassifier(**bp, eval_metric='logloss', tree_method='hist',
                               random_state=seed, verbosity=0, n_jobs=4)
    base.fit(Xs_fin, ys_fin, sample_weight=sws_fin)

    p_tr  = base.predict_proba(X_tr)[:, 1]
    is_auc = roc_auc_score(y_tr, p_tr)
    oos_auc = roc_auc_score(y_te, base.predict_proba(X_te)[:, 1]) if len(y_te) > 10 else 0.0
    log.info(f"  seed={seed}: IS={is_auc:.4f} OOS={oos_auc:.4f}")

    # Isotonic calibration pe ultimele 20% din IS
    cut = int(len(X_tr) * 0.80)
    ir  = _IR(out_of_bounds='clip')
    ir.fit(base.predict_proba(X_tr[cut:])[:, 1], y_tr[cut:])
    cal = _CalModel(base, ir)
    return cal, is_auc, oos_auc


# ── Antrenează N modele ────────────────────────────────────────────────────────
log.info(f"\nIS WR={y_tr.mean():.1%} | OOS WR={y_te.mean():.1%} (n={len(y_te)})")
models, is_aucs, oos_aucs = [], [], []

for i, seed in enumerate(SEEDS):
    log.info(f"\nModel {i+1}/{N_MODELS} (seed={seed}) ...")
    try:
        cal, ia, oa = _train_one(seed)
        models.append(cal)
        is_aucs.append(ia)
        oos_aucs.append(oa)
    except Exception as e:
        log.warning(f"  Model {i+1} failed: {e}")

if not models:
    log.error("Toate modelele au eșuat!")
    sys.exit(1)

# ── Ensemble OOS AUC ──────────────────────────────────────────────────────────
df_te_feat = pd.DataFrame(X_te, columns=FEATURES)
preds_ens  = np.mean([m.predict_proba(df_te_feat)[:, 1] for m in models], axis=0)
ens_oos    = roc_auc_score(y_te, preds_ens) if len(y_te) > 10 else 0.0
ens_is     = float(np.mean(is_aucs))
ens_oos_avg = float(np.mean(oos_aucs))

log.info(f"\n{'='*50}")
log.info(f"Ensemble ({len(models)} modele):")
log.info(f"  IS AUC (mean): {ens_is:.4f}")
log.info(f"  OOS AUC (averaged probabilities): {ens_oos:.4f}")
log.info(f"  OOS AUC (mean individual): {ens_oos_avg:.4f}")

if ens_oos < MIN_OOS_AUC:
    log.warning(f"  ⚠️  OOS={ens_oos:.4f} < {MIN_OOS_AUC} — model salvat dar sub prag")

# ── Salvare ───────────────────────────────────────────────────────────────────
pkg = {
    'type':        'ensemble',
    'models':      models,
    'weights':     None,
    'features':    FEATURES,
    'oos_auc':     round(ens_oos, 4),
    'is_auc':      round(ens_is, 4),
    'n_features':  len(FEATURES),
    'n_models':    len(models),
    'train_years': TRAIN_YEARS,
    'test_years':  TEST_YEARS,
    'gap_penalty': GAP_PENALTY,
    'top_n':       TOP_N,
    'version':     'lom_v4_ensemble',
}
pickle.dump(pkg, open(OUT, 'wb'))
log.info(f"\n✅ Salvat: {OUT.name} | IS={ens_is:.4f} OOS={ens_oos:.4f} | {len(FEATURES)} feats")
log.info(f"   Schimbă MODEL_PATH în lom_checker_v1.py → 'lom_model_v4.pkl'")
