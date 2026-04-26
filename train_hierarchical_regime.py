"""
train_hierarchical_regime.py — Multi-Scale Probabilistic Regime
================================================================
Implements THREE parallel regime scales as described in the architecture:

  macro_regime  → weekly  (BULL / BEAR / CHOPPY — 3 states, slow transitions)
  meso_regime   → daily   (5 states: CONSOLIDATION/PRE_EXPANSION/EXPANSION/
                            RETRACEMENT/DISTRIBUTION, conditioned on macro)
  micro_regime  → intraday (same 5 states, updated per bar via Bayesian posterior)

Output PKL: hierarchical_regime_v1.pkl
Output parquet: data/hierarchical_regime_labels.parquet

Usage:
    python3 train_hierarchical_regime.py

Integration in checkers:
    from train_hierarchical_regime import predict_hierarchical_regime
    result = predict_hierarchical_regime(pkg, bar_features)
    # → {'macro': 'BULL', 'macro_prob': 0.82,
    #    'meso': 'PRE_EXPANSION', 'meso_prob': 0.61, 'meso_enc': 1,
    #    'macro_enc': 0, 'combined_enc': 5}
"""

import sqlite3
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

BASE   = Path(__file__).parent
DB     = BASE / "mario_trading.db"
OUT    = BASE / "hierarchical_regime_v1.pkl"
OUT_LABELS = BASE / "data" / "hierarchical_regime_labels.parquet"

# ── State definitions ─────────────────────────────────────────────────────────
MACRO_STATES = ['BULL', 'BEAR', 'CHOPPY']   # 3 weekly states
MESO_STATES  = ['CONSOLIDATION', 'PRE_EXPANSION', 'EXPANSION',
                'RETRACEMENT', 'DISTRIBUTION']   # 5 daily states

# ── Feature sets ──────────────────────────────────────────────────────────────
MACRO_FEATURES = [
    'hurst_w', 'adx_w', 'range_pct_w', 'trend_slope_w',
    'vol_w', 'above_open_pct_w', 'up_days_w',
]
MESO_FEATURES = [
    'adx_14', 'hurst', 'garch_vol', 'kalman_smooth', 'acf_lag1',
    'fisher_transform', 'sample_entropy', 'has_displacement',
    'dist_vwap_atr', 'dist_poc_atr', 'dist_pdh_atr', 'dist_pdl_atr',
    'h4_bias_atr', 'h1_bias_atr', 'above_true_open_atr',
    'sweep_dn_atr', 'sweep_up_atr', 'pre_range_atr',
    'fvg_up', 'fvg_down', 'rvol',
]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MACRO REGIME — Weekly HMM-inspired XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def build_weekly_features(conn) -> pd.DataFrame:
    """Aggregate 1-min bars to daily → weekly features for macro regime.
    Uses market_data (3.9M bars) since nq_data only has recent rows."""
    sql = """
    SELECT date,
           year,
           MIN(open)  AS open,
           MAX(high)  AS high,
           MIN(low)   AS low,
           MAX(close) AS close,
           AVG(atr_14)      AS atr_14,
           AVG(hurst)       AS hurst,
           AVG(garch_vol)   AS garch_vol,
           AVG(adx_14)      AS adx_14,
           MAX(true_open)   AS true_open,
           MAX(has_displacement) AS has_displacement
    FROM market_data
    WHERE year >= 2021 AND adx_14 > 0 AND atr_14 > 0
      AND day_of_week BETWEEN 1 AND 5
    GROUP BY date
    ORDER BY date
    """
    df = pd.read_sql(sql, conn)
    # Compute derived columns
    atr_d = df['atr_14'].clip(lower=1)
    df['above_true_open_atr'] = (df['close'] - df['true_open'].fillna(df['close'])) / atr_d
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.to_period('W')

    weekly = df.groupby('week').agg(
        date_start=('date', 'first'),
        date_end=('date', 'last'),
        open=('open', 'first'),
        close=('close', 'last'),
        high=('high', 'max'),
        low=('low', 'min'),
        adx_mean=('adx_14', 'mean'),
        hurst_mean=('hurst', 'mean'),
        garch_mean=('garch_vol', 'mean'),
        atr_mean=('atr_14', 'mean'),
        disp_count=('has_displacement', 'sum'),
        n_days=('date', 'count'),
    ).reset_index()

    atr = weekly['atr_mean'].clip(lower=1)
    weekly['hurst_w']       = weekly['hurst_mean']
    weekly['adx_w']         = weekly['adx_mean']
    weekly['range_pct_w']   = (weekly['high'] - weekly['low']) / weekly['close'].clip(lower=1) * 100
    weekly['trend_slope_w'] = (weekly['close'] - weekly['open']) / atr
    weekly['vol_w']         = weekly['garch_mean']
    weekly['above_open_pct_w'] = ((weekly['close'] > weekly['open']).astype(float))
    weekly['up_days_w']     = weekly['disp_count'] / weekly['n_days'].clip(lower=1)

    # Rule-based macro label (used as training target)
    cond_bull  = (weekly['trend_slope_w'] > 0.5) & (weekly['adx_w'] > 22) & (weekly['hurst_w'] > 0.52)
    cond_bear  = (weekly['trend_slope_w'] < -0.5) & (weekly['adx_w'] > 22) & (weekly['hurst_w'] > 0.52)
    weekly['macro_label'] = 'CHOPPY'
    weekly.loc[cond_bull, 'macro_label'] = 'BULL'
    weekly.loc[cond_bear, 'macro_label'] = 'BEAR'

    return weekly


def train_macro_classifier(weekly: pd.DataFrame):
    """Train XGBoost macro-regime classifier on weekly features."""
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import balanced_accuracy_score

    feat_cols = [c for c in MACRO_FEATURES if c in weekly.columns]
    X = weekly[feat_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(weekly['macro_label'])

    # Temporal split: last 20% as OOS
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    oos_pred = clf.predict(X_te)
    oos_acc  = balanced_accuracy_score(y_te, oos_pred)
    print(f"  Macro classifier OOS balanced_acc: {oos_acc:.3f}")

    return clf, le, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MESO REGIME — Daily XGBoost conditioned on macro
# ══════════════════════════════════════════════════════════════════════════════

def build_daily_features(conn) -> pd.DataFrame:
    """Aggregate 1-min bars from market_data to daily features for meso training.
    nq_data only has 5 days (2026-04-16..21) so we use market_data instead."""
    sql = """
    SELECT date, year,
           AVG(atr_14)          AS atr_14,
           AVG(hurst)           AS hurst,
           AVG(garch_vol)       AS garch_vol,
           AVG(kalman_smooth)   AS kalman_smooth,
           AVG(acf_lag1)        AS acf_lag1,
           AVG(fisher_transform)AS fisher_transform,
           AVG(sample_entropy)  AS sample_entropy,
           MAX(has_displacement)AS has_displacement,
           AVG(dist_vwap)       AS dist_vwap,
           AVG(dist_poc)        AS dist_poc,
           AVG(dist_pdh)        AS dist_pdh,
           AVG(dist_pdl)        AS dist_pdl,
           MAX(h4_hi)           AS h4_hi,
           MIN(h4_lo)           AS h4_lo,
           MAX(h1_hi)           AS h1_hi,
           MIN(h1_lo)           AS h1_lo,
           MAX(true_open)       AS true_open,
           MAX(close)           AS close,
           MAX(fvg_up)          AS fvg_up,
           MAX(fvg_down)        AS fvg_down,
           AVG(adx_14)          AS adx_14,
           MAX(p_hi)            AS p_hi,
           MIN(p_lo)            AS p_lo,
           MAX(asia_hi)         AS asia_hi,
           MIN(asia_lo)         AS asia_lo,
           AVG(rvol)            AS rvol
    FROM market_data
    WHERE year >= 2021 AND adx_14 > 0 AND atr_14 > 0
      AND day_of_week BETWEEN 1 AND 5
    GROUP BY date
    ORDER BY date
    """
    df = pd.read_sql(sql, conn)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    atr = df['atr_14'].clip(lower=1)
    df['dist_vwap_atr']       = df['dist_vwap'].fillna(0).abs() / atr
    df['dist_poc_atr']        = df['dist_poc'].fillna(0).abs()  / atr
    df['dist_pdh_atr']        = df['dist_pdh'].fillna(0).abs()  / atr
    df['dist_pdl_atr']        = df['dist_pdl'].fillna(0).abs()  / atr
    df['h4_mid']              = (df['h4_hi'].fillna(df['close']) + df['h4_lo'].fillna(df['close'])) / 2
    df['h1_mid']              = (df['h1_hi'].fillna(df['close']) + df['h1_lo'].fillna(df['close'])) / 2
    df['h4_bias_atr']         = (df['close'] - df['h4_mid']) / atr
    df['h1_bias_atr']         = (df['close'] - df['h1_mid']) / atr
    df['above_true_open_atr'] = (df['close'] - df['true_open'].fillna(df['close'])) / atr
    df['pre_range_atr']       = (df['p_hi'].fillna(0) - df['p_lo'].fillna(0)).clip(lower=0) / atr
    df['sweep_up_atr']        = 0.0   # daily bars: no intraday sweep columns
    df['sweep_dn_atr']        = 0.0
    # rvol: garch_vol relative to 20-day mean (fallback if DB rvol is null)
    if df['rvol'].isna().all() or (df['rvol'] == 0).all():
        df['rvol'] = df['garch_vol'] / df['garch_vol'].rolling(20, min_periods=5).mean().fillna(df['garch_vol'])

    return df


def load_meso_labels(conn) -> pd.DataFrame:
    """Load existing regime_labels.parquet as meso training targets."""
    labels_path = BASE / 'data' / 'regime_labels.parquet'
    if labels_path.exists():
        labels = pd.read_parquet(labels_path)
        # Use LON session as primary daily regime
        lon = labels[labels['session'] == 'LON'][['date', 'regime']].copy()
        lon['date'] = pd.to_datetime(lon['date']).dt.strftime('%Y-%m-%d')
        return lon.rename(columns={'regime': 'meso_label'})
    # Fallback: rule-based
    return None


def train_meso_classifier(daily: pd.DataFrame, labels: pd.DataFrame,
                           macro_clf, macro_le, macro_feat_cols,
                           weekly: pd.DataFrame):
    """Train meso classifier with macro context injected as features."""
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import balanced_accuracy_score

    # Merge with meso labels
    df = daily.merge(labels, on='date', how='inner')
    if len(df) < 50:
        print("  ⚠️  Not enough labeled days for meso training")
        return None, None, None

    # Add macro context: map each daily date to its weekly macro regime
    weekly['date_end'] = pd.to_datetime(weekly['date_end']).dt.strftime('%Y-%m-%d')

    # Build date → macro_enc mapping
    macro_map = {}
    for _, wrow in weekly.iterrows():
        # Predict macro for this week
        w_feat = np.array([[wrow.get(f, 0) for f in macro_feat_cols]])
        macro_enc = int(macro_clf.predict(w_feat)[0])
        macro_probs = macro_clf.predict_proba(w_feat)[0]
        # Assign to all days in this week
        d_start = str(wrow['date_start'])[:10]
        d_end   = str(wrow['date_end'])[:10]
        dates_in_week = pd.date_range(d_start, d_end).strftime('%Y-%m-%d').tolist()
        for d in dates_in_week:
            macro_map[d] = {
                'macro_enc': macro_enc,
                **{f'macro_p_{s.lower()}': float(macro_probs[i])
                   for i, s in enumerate(macro_le.classes_)},
            }

    # Add macro features to daily
    for col in ['macro_enc', 'macro_p_bull', 'macro_p_bear', 'macro_p_choppy']:
        df[col] = df['date'].map(lambda d: macro_map.get(d, {}).get(col, 0))

    feat_cols = [c for c in MESO_FEATURES if c in df.columns]
    feat_cols += ['macro_enc', 'macro_p_bull', 'macro_p_bear', 'macro_p_choppy']

    X = df[feat_cols].fillna(0).values
    le = LabelEncoder()
    le.fit(MESO_STATES)
    # Ensure unknown labels are mapped to CONSOLIDATION
    df['meso_label'] = df['meso_label'].where(df['meso_label'].isin(MESO_STATES), 'CONSOLIDATION')
    y = le.transform(df['meso_label'])

    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    clf = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    oos_pred = clf.predict(X_te)
    oos_acc  = balanced_accuracy_score(y_te, oos_pred)
    print(f"  Meso classifier OOS balanced_acc: {oos_acc:.3f}")

    return clf, le, feat_cols, macro_map


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MICRO REGIME — Online Bayesian Intraday Updater
#     (see bayesian_regime_updater.py for the full intraday component)
#     Here we embed the prior computation: prior = meso posterior at session open
# ══════════════════════════════════════════════════════════════════════════════

def compute_micro_priors(meso_clf, meso_le, meso_feat_cols,
                         daily: pd.DataFrame) -> dict:
    """
    For each date, compute P(meso_regime) as the micro prior at session open.
    This is the starting distribution before any intraday bars are observed.
    Returns dict: date → np.array of shape (5,) probabilities in MESO_STATES order.
    """
    date_priors = {}
    feat_cols_present = [c for c in meso_feat_cols if c in daily.columns]
    for _, row in daily.iterrows():
        x = np.array([[row.get(c, 0) for c in feat_cols_present]])
        probs = meso_clf.predict_proba(x)[0]
        # Align to MESO_STATES order
        aligned = np.zeros(len(MESO_STATES))
        for i, cls in enumerate(meso_le.classes_):
            j = MESO_STATES.index(cls) if cls in MESO_STATES else 0
            aligned[j] = probs[i]
        date_priors[str(row['date'])[:10]] = aligned
    return date_priors


# ══════════════════════════════════════════════════════════════════════════════
# 4.  COMBINED LABEL: macro_enc × 5 + meso_enc  → 15 combined states
# ══════════════════════════════════════════════════════════════════════════════

def build_combined_labels(daily: pd.DataFrame, labels: pd.DataFrame,
                          macro_map: dict, meso_clf, meso_le, meso_feat_cols) -> pd.DataFrame:
    """Produce the full hierarchical_regime_labels.parquet."""
    df = daily.copy()
    feat_cols_present = [c for c in meso_feat_cols if c in df.columns]

    records = []
    for _, row in df.iterrows():
        d = str(row['date'])[:10]
        macro_info = macro_map.get(d, {'macro_enc': 0})
        macro_enc  = macro_info.get('macro_enc', 0)
        macro_name = MACRO_STATES[macro_enc] if macro_enc < len(MACRO_STATES) else 'CHOPPY'

        # Meso prediction
        x = np.array([[row.get(c, 0) for c in feat_cols_present]])
        # Add macro features
        meso_x_extra = np.array([[
            macro_enc,
            macro_info.get('macro_p_bull', 0),
            macro_info.get('macro_p_bear', 0),
            macro_info.get('macro_p_choppy', 0),
        ]])
        x_full = np.hstack([x, meso_x_extra])
        meso_probs = meso_clf.predict_proba(x_full)[0]
        meso_enc   = int(meso_clf.predict(x_full)[0])
        meso_name  = meso_le.inverse_transform([meso_enc])[0]

        # Prior for micro (= meso posterior)
        aligned_probs = {f'prob_{s}': 0.0 for s in MESO_STATES}
        for i, cls in enumerate(meso_le.classes_):
            aligned_probs[f'prob_{cls}'] = float(meso_probs[i])

        records.append({
            'date':         d,
            'macro_regime': macro_name,
            'macro_enc':    macro_enc,
            **{f'macro_p_{s.lower()}': macro_info.get(f'macro_p_{s.lower()}', 0)
               for s in MACRO_STATES},
            'meso_regime':  meso_name,
            'meso_enc':     meso_enc,
            **aligned_probs,
            'combined_enc': macro_enc * len(MESO_STATES) + meso_enc,
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPER  (used by checkers at runtime)
# ══════════════════════════════════════════════════════════════════════════════

def predict_hierarchical_regime(pkg: dict, bar_features: dict) -> dict:
    """
    Given a bar_features dict (same keys as MESO_FEATURES + weekly aggregates),
    return full hierarchical regime prediction.

    bar_features should include:
        adx_14, hurst, garch_vol, kalman_smooth, acf_lag1,
        fisher_transform, sample_entropy, has_displacement,
        dist_vwap_atr, dist_poc_atr, h4_bias_atr, h1_bias_atr,
        above_true_open_atr, sweep_dn_atr, sweep_up_atr, pre_range_atr,
        fvg_up, fvg_down, rvol,
        hurst_w, adx_w, range_pct_w, trend_slope_w, vol_w,
        above_open_pct_w, up_days_w  (weekly aggregates from last 5 days)
    """
    macro_clf   = pkg['macro_clf']
    macro_le    = pkg['macro_le']
    macro_feats = pkg['macro_feat_cols']
    meso_clf    = pkg['meso_clf']
    meso_le     = pkg['meso_le']
    meso_feats  = pkg['meso_feat_cols']

    def sv(k): return float(bar_features.get(k, 0) or 0)

    # Macro
    x_macro = np.array([[sv(f) for f in macro_feats]])
    macro_enc   = int(macro_clf.predict(x_macro)[0])
    macro_probs = macro_clf.predict_proba(x_macro)[0]
    macro_name  = macro_le.inverse_transform([macro_enc])[0]

    # Meso (with macro context)
    x_meso_base = np.array([[sv(f) for f in meso_feats if f in [
        'adx_14','hurst','garch_vol','kalman_smooth','acf_lag1',
        'fisher_transform','sample_entropy','has_displacement',
        'dist_vwap_atr','dist_poc_atr','dist_pdh_atr','dist_pdl_atr',
        'h4_bias_atr','h1_bias_atr','above_true_open_atr',
        'sweep_dn_atr','sweep_up_atr','pre_range_atr',
        'fvg_up','fvg_down','rvol',
    ]]])
    x_macro_ctx = np.array([[
        float(macro_enc),
        float(macro_probs[list(macro_le.classes_).index('BULL')] if 'BULL' in macro_le.classes_ else 0),
        float(macro_probs[list(macro_le.classes_).index('BEAR')] if 'BEAR' in macro_le.classes_ else 0),
        float(macro_probs[list(macro_le.classes_).index('CHOPPY')] if 'CHOPPY' in macro_le.classes_ else 0),
    ]])
    x_meso = np.hstack([x_meso_base, x_macro_ctx])
    meso_enc   = int(meso_clf.predict(x_meso)[0])
    meso_probs = meso_clf.predict_proba(x_meso)[0]
    meso_name  = meso_le.inverse_transform([meso_enc])[0]

    # Micro prior = meso posterior (Bayesian updater takes over from here)
    micro_prior = {cls: float(meso_probs[i]) for i, cls in enumerate(meso_le.classes_)}

    return {
        'macro_regime':    macro_name,
        'macro_enc':       macro_enc,
        'macro_prob':      float(macro_probs[macro_enc]),
        'macro_probs':     {cls: float(p) for cls, p in
                            zip(macro_le.classes_, macro_probs)},
        'meso_regime':     meso_name,
        'meso_enc':        meso_enc,
        'meso_prob':       float(meso_probs[meso_enc]),
        'meso_probs':      {cls: float(p) for cls, p in
                            zip(meso_le.classes_, meso_probs)},
        'micro_prior':     micro_prior,
        'combined_enc':    macro_enc * len(MESO_STATES) + meso_enc,
        'regime_enc':      meso_enc,   # backward compat with existing checkers
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def train_and_save():
    print("=" * 65)
    print("  HIERARCHICAL REGIME — macro/meso/micro")
    print("=" * 65)

    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)

    # 1. Macro
    print("\n[1/4] Building weekly features ...")
    weekly = build_weekly_features(conn)
    print(f"  {len(weekly)} weeks | macro dist: {weekly['macro_label'].value_counts().to_dict()}")

    print("\n[2/4] Training macro classifier ...")
    macro_clf, macro_le, macro_feat_cols = train_macro_classifier(weekly)

    # 2. Meso
    print("\n[3/4] Building daily features + meso classifier ...")
    daily  = build_daily_features(conn)
    labels = load_meso_labels(conn)

    if labels is not None:
        result = train_meso_classifier(daily, labels, macro_clf, macro_le,
                                       macro_feat_cols, weekly)
        if result[0] is not None:
            meso_clf, meso_le, meso_feat_cols, macro_map = result
        else:
            print("  ⚠️  Fallback: using regime_classifier_v1 as meso")
            meso_clf = meso_le = meso_feat_cols = macro_map = None
    else:
        meso_clf = meso_le = meso_feat_cols = macro_map = None

    conn.close()

    if meso_clf is None:
        print("  ⚠️  Meso training failed — PKL not saved")
        return

    # 3. Combined labels
    print("\n[4/4] Building combined label parquet ...")
    OUT_LABELS.parent.mkdir(exist_ok=True)
    combined = build_combined_labels(daily, labels, macro_map,
                                     meso_clf, meso_le, meso_feat_cols)
    combined.to_parquet(OUT_LABELS, index=False)
    print(f"  ✅ Saved {len(combined)} rows → {OUT_LABELS.name}")
    print(f"  Macro dist: {combined['macro_regime'].value_counts().to_dict()}")
    print(f"  Meso dist:  {combined['meso_regime'].value_counts().to_dict()}")

    # Save PKL
    pkg = {
        'macro_clf':      macro_clf,
        'macro_le':       macro_le,
        'macro_feat_cols':macro_feat_cols,
        'meso_clf':       meso_clf,
        'meso_le':        meso_le,
        'meso_feat_cols': meso_feat_cols,
        'macro_states':   MACRO_STATES,
        'meso_states':    MESO_STATES,
        'macro_map':      macro_map,
        'version':        'hierarchical_v1',
    }
    with open(OUT, 'wb') as f:
        pickle.dump(pkg, f)
    print(f"\n  ✅ Saved PKL → {OUT.name}")


if __name__ == '__main__':
    train_and_save()
