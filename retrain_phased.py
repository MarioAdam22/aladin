"""
retrain_phased.py — Phased retraining with dataset caching
============================================================
Usage:
  python3 retrain_phased.py build nom   → build NOM dataset → cache
  python3 retrain_phased.py build lom   → build LOM dataset → cache
  python3 retrain_phased.py train nom   → load cache → Optuna 25 trials → save PKL
  python3 retrain_phased.py train lom   → load cache → Optuna 25 trials → save PKL
  python3 retrain_phased.py train dsm   → build + train DSM (smaller dataset)
  python3 retrain_phased.py train sweep → build + train sweep_unified

Caches stored in: ./data/cache/
"""

import sys, pickle, logging, json
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("PHASED")

CACHE_DIR = Path(__file__).parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DB  = Path(__file__).parent / "mario_trading.db"
DIR = Path(__file__).parent

# ── Optuna fast settings ──────────────────────────────────────────────────────
N_TRIALS = 10          # fast trials — no SMOTE in objective, scale_pos_weight only
N_WF_FOLDS = 0        # skip walk-forward, use simple IS/OOS split (much faster)

# ══════════════════════════════════════════════════════════════════════════════
# NOM
# ══════════════════════════════════════════════════════════════════════════════

def build_nom(split='is'):
    """Build NOM dataset and save to parquet cache. split='is'|'oos'"""
    log.info(f"=== BUILD NOM DATASET ({split.upper()}) ===")
    sys.path.insert(0, str(DIR))
    import train_nom_v3 as m

    regime_pkg  = m.load_regime_classifier()
    of_lookup   = m.load_of_lag_features()

    if split == 'is':
        log.info("Building IS dataset [2022-2024]...")
        df = m.build_dataset([2022, 2023, 2024], regime_pkg=regime_pkg, of_lag_lookup=of_lookup)
        out = CACHE_DIR / "nom_is.parquet"
    else:
        log.info("Building OOS dataset [2025-2026]...")
        df = m.build_dataset([2025, 2026], regime_pkg=regime_pkg, of_lag_lookup=of_lookup)
        out = CACHE_DIR / "nom_oos.parquet"

    df.to_parquet(out, index=False)
    log.info(f"Saved: {len(df)} rows → {out}")


def train_nom():
    """Load NOM cache and train with Optuna (fast mode)."""
    log.info("=== TRAIN NOM (fast, 25 trials) ===")
    import xgboost as xgb, optuna
    from sklearn.metrics import roc_auc_score
    from sklearn.calibration import CalibratedClassifierCV
    from imblearn.over_sampling import BorderlineSMOTE
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_tr = CACHE_DIR / "nom_is.parquet"
    out_te = CACHE_DIR / "nom_oos.parquet"
    if not out_tr.exists():
        log.error("Cache missing — run: python3 retrain_phased.py build nom"); return

    df_tr = pd.read_parquet(out_tr)
    df_te = pd.read_parquet(out_te) if out_te.exists() else pd.DataFrame()
    log.info(f"IS: {len(df_tr)} rows | OOS: {len(df_te)} rows")

    META = [c for c in df_tr.columns if c.startswith('_')]
    feature_cols = [c for c in df_tr.columns if not c.startswith('_')]

    X_tr = df_tr[feature_cols].fillna(0)
    y_tr = df_tr['_label']
    yr_  = df_tr['_date'].apply(lambda d: int(str(d)[:4]))
    YEAR_WEIGHTS = {2022: 0.75, 2023: 0.90, 2024: 1.00}
    sw_  = np.array([YEAR_WEIGHTS.get(int(y), 1.0) for y in yr_])

    X_te = df_te[feature_cols].fillna(0).reindex(columns=feature_cols, fill_value=0) if len(df_te) > 0 else pd.DataFrame(columns=feature_cols)
    y_te = df_te['_label'] if len(df_te) > 0 else pd.Series(dtype=int)

    # ── Feature selection (fast pre-screen) ──────────────────────────────────
    log.info("Feature selection...")
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / max(pos, 1)
    pre = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             subsample=0.7, colsample_bytree=0.6, min_child_weight=25,
                             gamma=1.5, reg_alpha=2.0, reg_lambda=4.0,
                             scale_pos_weight=spw, random_state=42, n_jobs=-1,
                             eval_metric='logloss', verbosity=0)
    pre.fit(X_tr, y_tr, sample_weight=sw_, verbose=False)
    imp = pd.Series(pre.feature_importances_, index=feature_cols).sort_values(ascending=False)

    must_keep = [c for c in [
        'regime_enc','regime_is_pre','regime_is_exp','regime_is_ret','regime_aligned',
        'of_cvd_lag1','of_abs_lag1','of_od_lag1','of_lon_cvd_x_dir','of_regime_x_sweep',
        'equal_lo_count','rolling_5sess_wr','fisher_transform','fomc_proximity',
        'drive_x_sweep','ts_sweep_pct_lon','adx_10d_mean','disp_body','kalman_smooth',
        'garch_vol','atr_vs_5d','dir_x_hurst','vol_x_sweep',
        'asia_dir_x_h4','h4_x_weekly','spike_vs_asia_hi','spike_vs_asia_lo',
        'dist_lw_hi','dist_lw_lo','pre_rng_vs_lw','sweep_vs_lw_rng',
        'is_thursday','is_friday','pre_rng_atr','asia_dir_explicit',
    ] if c in feature_cols]
    selected = list(dict.fromkeys(must_keep + imp.head(75).index.tolist()))
    log.info(f"Selected {len(selected)} features | top5: {imp.head(5).index.tolist()}")
    X_tr = X_tr[selected]
    X_te = X_te.reindex(columns=selected, fill_value=0)

    # ── Optuna (fast — no SMOTE in trials, use scale_pos_weight) ─────────────
    split = int(len(X_tr) * 0.80)
    X_val_o, y_val_o = X_tr.iloc[split:].values, y_tr.iloc[split:].values
    X_tr_o,  y_tr_o  = X_tr.iloc[:split].values, y_tr.iloc[:split].values
    sw_o = sw_[:split]
    neg_o = (y_tr_o == 0).sum(); pos_o = (y_tr_o == 1).sum()
    spw_o = neg_o / max(pos_o, 1)

    def objective(trial):
        p = {
            'n_estimators':     trial.suggest_int('n_estimators', 150, 350),
            'max_depth':        trial.suggest_int('max_depth', 2, 4),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'subsample':        trial.suggest_float('subsample', 0.55, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.40, 0.75),
            'min_child_weight': trial.suggest_int('min_child_weight', 20, 60),
            'gamma':            trial.suggest_float('gamma', 1.0, 6.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1.0, 6.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 3.0, 8.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', max(spw_o * 0.6, 2.0), spw_o * 1.5),
        }
        clf = xgb.XGBClassifier(**p, objective='binary:logistic', eval_metric='auc',
                                 tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)
        clf.fit(X_tr_o, y_tr_o, sample_weight=sw_o, verbose=False)
        pred = clf.predict_proba(X_val_o)[:, 1]
        return roc_auc_score(y_val_o, pred) if len(np.unique(y_val_o)) > 1 else 0.5

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    bp = study.best_params
    log.info(f"Best val AUC={study.best_value:.4f} | params={bp}")

    # ── Final model (with SMOTE) ──────────────────────────────────────────────
    try:
        sm = BorderlineSMOTE(sampling_strategy=0.25, random_state=42,
                             k_neighbors=min(5, int(y_tr.sum()) - 1))
        Xf, yf = sm.fit_resample(X_tr.values, y_tr.values)
    except Exception:
        Xf, yf = X_tr.values, y_tr.values

    final_clf = xgb.XGBClassifier(**bp, objective='binary:logistic', eval_metric='auc',
                                   tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)
    final_clf.fit(Xf, yf, verbose=False)
    cal = CalibratedClassifierCV(final_clf, cv='prefit', method='isotonic')
    cal.fit(X_tr.values, y_tr.values)

    is_auc  = roc_auc_score(y_tr, cal.predict_proba(X_tr.values)[:, 1])
    oos_auc = roc_auc_score(y_te, cal.predict_proba(X_te.values)[:, 1]) if len(y_te) > 0 and len(np.unique(y_te)) > 1 else 0.0
    log.info(f"IS={is_auc:.4f}  OOS={oos_auc:.4f}")

    # ── Quantile models ───────────────────────────────────────────────────────
    log.info("Training quantile models...")
    q_models = {}
    mfe_col  = '_max_fwd' if '_max_fwd' in df_tr.columns else None
    if mfe_col:
        y_mfe = df_tr[mfe_col].fillna(0).values
        for alpha in [0.10, 0.25, 0.50, 0.75, 0.90]:
            qm = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
                                   n_estimators=300, max_depth=3, learning_rate=0.05,
                                   subsample=0.7, colsample_bytree=0.6,
                                   reg_alpha=2.0, reg_lambda=4.0,
                                   random_state=42, n_jobs=-1, verbosity=0)
            qm.fit(X_tr.values, y_mfe, sample_weight=sw_, verbose=False)
            q_models[alpha] = qm
    log.info(f"Quantile models: {len(q_models)}")

    # ── Survival model ────────────────────────────────────────────────────────
    log.info("Training survival model...")
    surv_model = None
    tte_col = '_time_to_event' if '_time_to_event' in df_tr.columns else None
    if tte_col:
        y_tte = np.log1p(df_tr[tte_col].fillna(60).values)
        sm_reg = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                   subsample=0.7, colsample_bytree=0.6,
                                   reg_alpha=2.0, reg_lambda=4.0,
                                   random_state=42, n_jobs=-1, verbosity=0)
        sm_reg.fit(X_tr.values, y_tte, sample_weight=sw_, verbose=False)
        surv_model = sm_reg
    log.info("Survival model done")

    # ── Save PKL ──────────────────────────────────────────────────────────────
    OUT = DIR / "nom_model_v3.pkl"
    old_auc = 0.0
    if OUT.exists():
        try:
            old_pkg = pickle.load(open(OUT, 'rb'))
            old_auc = old_pkg.get('oos_auc', 0.0)
        except Exception:
            pass

    if oos_auc >= old_auc - 0.005:
        pkg = {
            'model': cal, 'quantile_models': q_models, 'survival_model': surv_model,
            'features': list(selected), 'is_auc': round(is_auc, 4),
            'oos_auc': round(oos_auc, 4), 'n_features': len(selected),
            'train_years': [2022, 2023, 2024], 'test_years': [2025, 2026],
            'version': 'v3_cross_pollinated_fast', 'tp_mult': 2.0, 'label_window': 60,
            'quantile_alphas': [0.10, 0.25, 0.50, 0.75, 0.90],
            'has_regime_enc': True, 'has_lagged_of': True,
        }
        with open(OUT, 'wb') as f:
            pickle.dump(pkg, f)
        log.info(f"💾 Saved {OUT} | IS={is_auc:.4f} OOS={oos_auc:.4f} (prev={old_auc:.4f})")
    else:
        log.warning(f"OOS regression {oos_auc:.4f} < {old_auc:.4f} - 0.005 → kept old model")

    imp2 = pd.Series(final_clf.feature_importances_, index=selected).sort_values(ascending=False)
    log.info(f"Top 15:\n{imp2.head(15).to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# LOM
# ══════════════════════════════════════════════════════════════════════════════

def build_lom(split='is'):
    """Build LOM dataset and save to parquet cache. split='is'|'oos'"""
    log.info(f"=== BUILD LOM DATASET ({split.upper()}) ===")
    sys.path.insert(0, str(DIR))
    import train_lom_v3 as m

    regime_pkg = m.load_regime_classifier()
    of_lookup  = m.load_of_lag_features()

    if split == 'is':
        log.info("Building IS dataset [2021-2024]...")
        df = m.build_dataset([2021, 2022, 2023, 2024], regime_pkg=regime_pkg, of_lag_lookup=of_lookup)
        out = CACHE_DIR / "lom_is.parquet"
    else:
        log.info("Building OOS dataset [2025-2026]...")
        df = m.build_dataset([2025, 2026], regime_pkg=regime_pkg, of_lag_lookup=of_lookup)
        out = CACHE_DIR / "lom_oos.parquet"

    df.to_parquet(out, index=False)
    log.info(f"Saved: {len(df)} rows → {out}")


def train_lom():
    """Load LOM cache and train with Optuna (fast mode)."""
    log.info("=== TRAIN LOM (fast, 25 trials) ===")
    import xgboost as xgb, optuna
    from sklearn.metrics import roc_auc_score
    from sklearn.calibration import CalibratedClassifierCV
    from imblearn.over_sampling import BorderlineSMOTE
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_tr = CACHE_DIR / "lom_is.parquet"
    out_te = CACHE_DIR / "lom_oos.parquet"
    if not out_tr.exists():
        log.error("Cache missing — run: python3 retrain_phased.py build lom"); return

    df_tr = pd.read_parquet(out_tr)
    df_te = pd.read_parquet(out_te) if out_te.exists() else pd.DataFrame()
    log.info(f"IS: {len(df_tr)} rows | OOS: {len(df_te)} rows")

    feature_cols = [c for c in df_tr.columns if not c.startswith('_')]
    X_tr = df_tr[feature_cols].fillna(0)
    y_tr = df_tr['_label']
    yr_  = df_tr['_date'].apply(lambda d: int(str(d)[:4]))
    YEAR_WEIGHTS = {2021: 0.70, 2022: 0.80, 2023: 0.90, 2024: 1.00}
    sw_  = np.array([YEAR_WEIGHTS.get(int(y), 1.0) for y in yr_])
    X_te = df_te[feature_cols].fillna(0).reindex(columns=feature_cols, fill_value=0) if len(df_te) > 0 else pd.DataFrame(columns=feature_cols)
    y_te = df_te['_label'] if len(df_te) > 0 else pd.Series(dtype=int)

    # Feature selection
    log.info("Feature selection...")
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / max(pos, 1)
    pre = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             subsample=0.7, colsample_bytree=0.6, min_child_weight=25,
                             gamma=1.5, reg_alpha=2.0, reg_lambda=4.0,
                             scale_pos_weight=spw, random_state=42, n_jobs=-1,
                             eval_metric='logloss', verbosity=0)
    pre.fit(X_tr, y_tr, sample_weight=sw_, verbose=False)
    imp = pd.Series(pre.feature_importances_, index=feature_cols).sort_values(ascending=False)

    must_keep = [c for c in [
        'regime_enc','regime_is_pre','regime_is_exp','regime_is_ret','regime_aligned',
        'of_cvd_lag1','of_abs_lag1','of_od_lag1','of_cvd_regime',
        'garch_vol','atr_vs_5d','pre_rng_atr','asia_dir_explicit','asia_range_vs_atr5d',
        'dir_x_hurst','vol_x_sweep','asia_dir_x_h4','h4_x_weekly',
        'spike_vs_asia_hi','spike_vs_asia_lo','dist_lw_hi','dist_lw_lo',
        'value_area_width','dist_asia_hi_atr','dist_asia_lo_atr',
        'pre_rng_vs_lw','sweep_vs_lw_rng','is_thursday','is_friday',
        'disp_wick_ratio','kalman_smooth',
    ] if c in feature_cols]
    selected = list(dict.fromkeys(must_keep + imp.head(75).index.tolist()))
    log.info(f"Selected {len(selected)} features | top5: {imp.head(5).index.tolist()}")
    X_tr = X_tr[selected]
    X_te = X_te.reindex(columns=selected, fill_value=0)

    # Optuna (fast — no SMOTE in trials)
    split = int(len(X_tr) * 0.80)
    X_val_o, y_val_o = X_tr.iloc[split:].values, y_tr.iloc[split:].values
    X_tr_o,  y_tr_o  = X_tr.iloc[:split].values, y_tr.iloc[:split].values
    sw_o = sw_[:split]
    neg_o = (y_tr_o == 0).sum(); pos_o = (y_tr_o == 1).sum()
    spw_o = neg_o / max(pos_o, 1)

    def objective(trial):
        p = {
            'n_estimators':     trial.suggest_int('n_estimators', 150, 350),
            'max_depth':        trial.suggest_int('max_depth', 2, 4),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'subsample':        trial.suggest_float('subsample', 0.55, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.40, 0.75),
            'min_child_weight': trial.suggest_int('min_child_weight', 20, 60),
            'gamma':            trial.suggest_float('gamma', 1.0, 6.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1.0, 6.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 3.0, 8.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', max(spw_o * 0.6, 2.0), spw_o * 1.5),
        }
        clf = xgb.XGBClassifier(**p, objective='binary:logistic', eval_metric='auc',
                                 tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)
        clf.fit(X_tr_o, y_tr_o, sample_weight=sw_o, verbose=False)
        pred = clf.predict_proba(X_val_o)[:, 1]
        return roc_auc_score(y_val_o, pred) if len(np.unique(y_val_o)) > 1 else 0.5

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    bp = study.best_params
    log.info(f"Best val AUC={study.best_value:.4f} | params={bp}")

    # Final model (with SMOTE)
    try:
        sm = BorderlineSMOTE(sampling_strategy=0.25, random_state=42,
                             k_neighbors=min(5, int(y_tr.sum()) - 1))
        Xf, yf = sm.fit_resample(X_tr.values, y_tr.values)
    except Exception:
        Xf, yf = X_tr.values, y_tr.values

    final_clf = xgb.XGBClassifier(**bp, objective='binary:logistic', eval_metric='auc',
                                   tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)
    final_clf.fit(Xf, yf, verbose=False)
    cal = CalibratedClassifierCV(final_clf, cv='prefit', method='isotonic')
    cal.fit(X_tr.values, y_tr.values)

    is_auc  = roc_auc_score(y_tr, cal.predict_proba(X_tr.values)[:, 1])
    oos_auc = roc_auc_score(y_te, cal.predict_proba(X_te.values)[:, 1]) if len(y_te) > 0 and len(np.unique(y_te)) > 1 else 0.0
    log.info(f"IS={is_auc:.4f}  OOS={oos_auc:.4f}")

    # Quantile
    log.info("Training quantile models...")
    q_models = {}
    if '_max_fwd' in df_tr.columns:
        y_mfe = df_tr['_max_fwd'].fillna(0).values
        for alpha in [0.10, 0.25, 0.50, 0.75, 0.90]:
            qm = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
                                   n_estimators=300, max_depth=3, learning_rate=0.05,
                                   subsample=0.7, colsample_bytree=0.6,
                                   reg_alpha=2.0, reg_lambda=4.0,
                                   random_state=42, n_jobs=-1, verbosity=0)
            qm.fit(X_tr.values, y_mfe, sample_weight=sw_, verbose=False)
            q_models[alpha] = qm

    # Survival
    surv_model = None
    if '_time_to_event' in df_tr.columns:
        y_tte = np.log1p(df_tr['_time_to_event'].fillna(60).values)
        sm_reg = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                   subsample=0.7, colsample_bytree=0.6,
                                   reg_alpha=2.0, reg_lambda=4.0,
                                   random_state=42, n_jobs=-1, verbosity=0)
        sm_reg.fit(X_tr.values, y_tte, sample_weight=sw_, verbose=False)
        surv_model = sm_reg

    # Save
    OUT = DIR / "lom_model_v3.pkl"
    old_auc = 0.0
    if OUT.exists():
        try:
            old_pkg = pickle.load(open(OUT, 'rb'))
            old_auc = old_pkg.get('oos_auc', 0.0)
        except Exception:
            pass

    if oos_auc >= old_auc - 0.005:
        pkg = {
            'model': cal, 'quantile_models': q_models, 'survival_model': surv_model,
            'features': list(selected), 'is_auc': round(is_auc, 4),
            'oos_auc': round(oos_auc, 4), 'n_features': len(selected),
            'train_years': [2021, 2022, 2023, 2024], 'test_years': [2025, 2026],
            'version': 'v3_cross_pollinated_fast', 'tp_mult': 2.0, 'label_window': 60,
            'quantile_alphas': [0.10, 0.25, 0.50, 0.75, 0.90],
            'has_regime_enc': True, 'has_lagged_of': True,
        }
        with open(OUT, 'wb') as f:
            pickle.dump(pkg, f)
        log.info(f"💾 Saved {OUT} | IS={is_auc:.4f} OOS={oos_auc:.4f} (prev={old_auc:.4f})")
    else:
        log.warning(f"OOS regression {oos_auc:.4f} < {old_auc:.4f} - 0.005 → kept old model")

    imp2 = pd.Series(final_clf.feature_importances_, index=selected).sort_values(ascending=False)
    log.info(f"Top 15:\n{imp2.head(15).to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# DSM (smaller dataset, can build+train in one call)
# ══════════════════════════════════════════════════════════════════════════════

def train_dsm():
    log.info("=== TRAIN DSM (build + train in one call) ===")
    import subprocess
    result = subprocess.run(['python3', str(DIR / 'train_dsm.py')],
                            capture_output=True, text=True, cwd=str(DIR))
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)

    phase  = sys.argv[1].lower()                    # build | train
    target = sys.argv[2].lower()                    # nom | lom | dsm | sweep
    split  = sys.argv[3].lower() if len(sys.argv) > 3 else 'is'   # is | oos

    if phase == 'build' and target == 'nom':
        build_nom(split)
    elif phase == 'train' and target == 'nom':
        train_nom()
    elif phase == 'build' and target == 'lom':
        build_lom(split)
    elif phase == 'train' and target == 'lom':
        train_lom()
    elif phase == 'train' and target == 'dsm':
        train_dsm()
    else:
        print(f"Unknown phase/target: {phase} {target}")
        sys.exit(1)
