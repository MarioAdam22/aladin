"""
bayesian_regime_updater.py — Online Bayesian Intraday Regime Updating
======================================================================
Implements P(regime | bar_1...bar_n) updated with each new 1-min bar.

Architecture:
    Prior   = meso posterior from hierarchical_regime_v1 (computed at session open)
    Update  = Bayesian: P(regime|bar_t) ∝ P(bar_t|regime) × P(regime|bar_t-1)
    Likelihood = P(bar_t|regime) from pre-fitted Gaussian per regime per feature

Usage in checker (intraday, per new bar):
    from bayesian_regime_updater import BayesianRegimeUpdater

    # Initialize once per session:
    updater = BayesianRegimeUpdater(pkg)
    updater.reset(prior_probs)   # prior_probs from hierarchical_regime_v1

    # Each new bar:
    bar_features = {'adx_14': 28.3, 'hurst': 0.48, ...}
    result = updater.update(bar_features)
    # → {'regime': 'PRE_EXPANSION', 'prob': 0.71, 'entropy': 0.83,
    #    'probs': {'CONSOLIDATION':0.12, 'PRE_EXPANSION':0.71, ...}}

Training:
    python3 bayesian_regime_updater.py
    → Saves bayesian_regime_updater_v1.pkl
"""

import sqlite3
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

BASE   = Path(__file__).parent
DB     = BASE / "mario_trading.db"
OUT    = BASE / "bayesian_regime_updater_v1.pkl"

MESO_STATES = ['CONSOLIDATION', 'PRE_EXPANSION', 'EXPANSION',
               'RETRACEMENT', 'DISTRIBUTION']

# Features used for likelihood computation (available on 1-min bars)
LIKELIHOOD_FEATURES = [
    'adx_14', 'hurst', 'garch_vol', 'fisher_transform',
    'dist_vwap_atr', 'h4_bias_atr', 'above_true_open_atr',
    'sweep_dn_atr', 'sweep_up_atr',
]

# Transition matrix: how likely is regime to change bar-to-bar?
# Diagonal = persistence, off-diagonal = transition probability
# Based on domain knowledge: regimes are sticky intraday
TRANSITION_MATRIX = np.array([
    # CONS   PRE    EXP    RET    DIST
    [0.920, 0.040, 0.020, 0.015, 0.005],  # from CONSOLIDATION
    [0.100, 0.820, 0.060, 0.015, 0.005],  # from PRE_EXPANSION
    [0.030, 0.020, 0.890, 0.055, 0.005],  # from EXPANSION
    [0.050, 0.020, 0.040, 0.880, 0.010],  # from RETRACEMENT
    [0.100, 0.050, 0.050, 0.050, 0.750],  # from DISTRIBUTION
])


# ══════════════════════════════════════════════════════════════════════════════
# Gaussian likelihood model per regime
# ══════════════════════════════════════════════════════════════════════════════

class GaussianLikelihoodModel:
    """
    For each regime and each feature, stores (mean, std) of that feature
    conditional on the regime. Used to compute P(bar_features | regime).
    Trained from labeled daily bars.
    """

    def __init__(self):
        self.params  = {}   # regime → {feature → (mean, std)}
        self.features = []

    def fit(self, df: pd.DataFrame, label_col: str, features: list):
        """Fit Gaussian parameters from labeled daily data."""
        self.features = features
        for regime in MESO_STATES:
            sub = df[df[label_col] == regime][features].fillna(0)
            if len(sub) < 5:
                # Fallback: use global stats
                sub = df[features].fillna(0)
            self.params[regime] = {
                feat: (float(sub[feat].mean()), max(float(sub[feat].std()), 0.01))
                for feat in features
            }
        return self

    def log_likelihood(self, regime: str, bar_features: dict) -> float:
        """Compute log P(bar_features | regime) assuming feature independence."""
        if regime not in self.params:
            return 0.0
        log_lik = 0.0
        for feat in self.features:
            val = float(bar_features.get(feat, 0) or 0)
            mu, sigma = self.params[regime][feat]
            # log N(val; mu, sigma)
            log_lik += norm.logpdf(val, loc=mu, scale=sigma)
        return log_lik

    def likelihoods(self, bar_features: dict) -> np.ndarray:
        """Return likelihood array for all regimes (log space → exp)."""
        log_liks = np.array([
            self.log_likelihood(r, bar_features) for r in MESO_STATES
        ])
        # Normalize to avoid numerical issues
        log_liks -= log_liks.max()
        liks = np.exp(log_liks)
        return liks


# ══════════════════════════════════════════════════════════════════════════════
# Online Bayesian Updater (used at runtime)
# ══════════════════════════════════════════════════════════════════════════════

class BayesianRegimeUpdater:
    """
    Stateful intraday Bayesian regime updater.
    Call reset() at session open, update() for each new bar.
    """

    def __init__(self, pkg: dict):
        self.likelihood_model = pkg['likelihood_model']
        self.transition_matrix = pkg.get('transition_matrix', TRANSITION_MATRIX)
        self.states = MESO_STATES
        self.posterior = None
        self.n_updates = 0

    def reset(self, prior_probs: dict = None):
        """
        Initialize posterior at session open.
        prior_probs: dict {regime: probability} from hierarchical_regime_v1.
        If None, uses uniform prior.
        """
        if prior_probs is None:
            self.posterior = np.ones(len(self.states)) / len(self.states)
        else:
            self.posterior = np.array([
                float(prior_probs.get(s, 1.0 / len(self.states)))
                for s in self.states
            ])
            self.posterior /= self.posterior.sum()
        self.n_updates = 0

    def update(self, bar_features: dict) -> dict:
        """
        Bayesian update: incorporate new bar evidence.

        P(regime_t | bar_1..bar_t) ∝ P(bar_t | regime_t)
                                     × Σ P(regime_t | regime_t-1) × P(regime_t-1 | bar_1..bar_t-1)

        Returns current posterior as dict.
        """
        if self.posterior is None:
            self.reset()

        # 1. Predict step: apply transition matrix
        predicted = self.transition_matrix.T @ self.posterior

        # 2. Update step: multiply by likelihood
        likelihoods = self.likelihood_model.likelihoods(bar_features)
        updated = predicted * likelihoods

        # 3. Normalize
        total = updated.sum()
        if total < 1e-10:
            updated = predicted.copy()  # fallback: ignore this bar
        else:
            updated /= total

        self.posterior = updated
        self.n_updates += 1

        best_idx  = int(np.argmax(updated))
        best_prob = float(updated[best_idx])
        entropy   = float(-np.sum(updated * np.log(updated + 1e-10)) / np.log(len(self.states)))

        return {
            'regime':      self.states[best_idx],
            'regime_enc':  best_idx,
            'prob':        best_prob,
            'entropy':     entropy,        # 0=certain, 1=uniform
            'n_updates':   self.n_updates,
            'probs':       {s: float(updated[i]) for i, s in enumerate(self.states)},
        }

    def current_state(self) -> dict:
        """Return current posterior without updating."""
        if self.posterior is None:
            return {'regime': 'CONSOLIDATION', 'prob': 0.2, 'entropy': 1.0}
        best_idx  = int(np.argmax(self.posterior))
        best_prob = float(self.posterior[best_idx])
        entropy   = float(-np.sum(self.posterior * np.log(self.posterior + 1e-10))
                         / np.log(len(self.states)))
        return {
            'regime':    self.states[best_idx],
            'regime_enc': best_idx,
            'prob':      best_prob,
            'entropy':   entropy,
            'n_updates': self.n_updates,
            'probs':     {s: float(self.posterior[i]) for i, s in enumerate(self.states)},
        }


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def load_training_data(conn) -> pd.DataFrame:
    """Load daily bars aggregated from market_data (nq_data only has 5 days of 2026)."""
    sql = """
    SELECT date,
           AVG(adx_14)           AS adx_14,
           AVG(hurst)            AS hurst,
           AVG(garch_vol)        AS garch_vol,
           AVG(fisher_transform) AS fisher_transform,
           AVG(dist_vwap)        AS dist_vwap,
           AVG(dist_poc)         AS dist_poc,
           AVG(dist_pdh)         AS dist_pdh,
           AVG(dist_pdl)         AS dist_pdl,
           MAX(h4_hi)            AS h4_hi,
           MIN(h4_lo)            AS h4_lo,
           MAX(h1_hi)            AS h1_hi,
           MIN(h1_lo)            AS h1_lo,
           MAX(true_open)        AS true_open,
           MAX(close)            AS close,
           AVG(atr_14)           AS atr_14,
           MAX(fvg_up)           AS fvg_up,
           MAX(fvg_down)         AS fvg_down,
           MAX(p_hi)             AS p_hi,
           MIN(p_lo)             AS p_lo,
           MAX(asia_hi)          AS asia_hi,
           MIN(asia_lo)          AS asia_lo
    FROM market_data
    WHERE year >= 2021 AND adx_14 > 0 AND atr_14 > 0
      AND day_of_week BETWEEN 1 AND 5
    GROUP BY date
    ORDER BY date
    """
    df = pd.read_sql(sql, conn)
    atr = df['atr_14'].clip(lower=1)
    df['dist_vwap_atr']       = df['dist_vwap'].fillna(0).abs() / atr
    df['dist_poc_atr']        = df['dist_poc'].fillna(0).abs()  / atr
    df['h4_mid']              = (df['h4_hi'].fillna(df['close']) + df['h4_lo'].fillna(df['close'])) / 2
    df['h1_mid']              = (df['h1_hi'].fillna(df['close']) + df['h1_lo'].fillna(df['close'])) / 2
    df['h4_bias_atr']         = (df['close'] - df['h4_mid']) / atr
    df['h1_bias_atr']         = (df['close'] - df['h1_mid']) / atr
    df['above_true_open_atr'] = (df['close'] - df['true_open'].fillna(df['close'])) / atr
    # Sweep: use asia_hi/asia_lo reference (no intraday sweep on daily bars → set 0)
    df['sweep_up_atr']        = 0.0
    df['sweep_dn_atr']        = 0.0
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return df


def train_and_save():
    print("=" * 65)
    print("  BAYESIAN REGIME UPDATER — Likelihood Model Training")
    print("=" * 65)

    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    daily = load_training_data(conn)
    conn.close()

    # Load regime labels
    labels_path = BASE / 'data' / 'regime_labels.parquet'
    if not labels_path.exists():
        print("  ⚠️  regime_labels.parquet not found — run train_regime.py first")
        return

    labels = pd.read_parquet(labels_path)
    lon_labels = labels[labels['session'] == 'LON'][['date', 'regime']].copy()
    lon_labels['date'] = pd.to_datetime(lon_labels['date']).dt.strftime('%Y-%m-%d')

    df = daily.merge(lon_labels, on='date', how='inner')
    print(f"  Training data: {len(df)} days")
    print(f"  Regime dist: {df['regime'].value_counts().to_dict()}")

    feat_cols = [c for c in LIKELIHOOD_FEATURES if c in df.columns]
    print(f"  Likelihood features: {feat_cols}")

    # Fit Gaussian likelihood model
    likelihood_model = GaussianLikelihoodModel()
    likelihood_model.fit(df, 'regime', feat_cols)

    # Validate: for each day, init with uniform prior, update with daily features,
    # check if posterior matches label
    correct = 0
    for _, row in df.iterrows():
        bar_feats = {f: row.get(f, 0) for f in feat_cols}
        updater = BayesianRegimeUpdater({
            'likelihood_model': likelihood_model,
            'transition_matrix': TRANSITION_MATRIX,
        })
        updater.reset()
        # Single update (1 bar)
        result = updater.update(bar_feats)
        if result['regime'] == row['regime']:
            correct += 1

    acc = correct / len(df)
    print(f"\n  Single-bar accuracy (uniform prior): {acc:.3f}")
    print(f"  (Expected ~0.35-0.55 — better than chance with good prior)")

    # Print likelihood params summary
    print("\n  Likelihood params (mean ± std per regime):")
    for regime in MESO_STATES:
        if regime in likelihood_model.params:
            p = likelihood_model.params[regime]
            adx_mu, adx_s = p.get('adx_14', (0, 1))
            hurst_mu, hurst_s = p.get('hurst', (0.5, 0.1))
            print(f"    {regime:<18} adx={adx_mu:.1f}±{adx_s:.1f}  "
                  f"hurst={hurst_mu:.2f}±{hurst_s:.2f}")

    pkg = {
        'likelihood_model':  likelihood_model,
        'transition_matrix': TRANSITION_MATRIX,
        'likelihood_features': feat_cols,
        'meso_states':       MESO_STATES,
        'single_bar_acc':    acc,
        'n_training_days':   len(df),
        'version':           'bayesian_v1',
    }

    with open(OUT, 'wb') as f:
        pickle.dump(pkg, f)
    print(f"\n  ✅ Saved → {OUT.name}")


def load_bayesian_updater(path=None) -> BayesianRegimeUpdater:
    """Load the updater from PKL. Use in checkers."""
    p = Path(path) if path else OUT
    if not p.exists():
        return None
    with open(p, 'rb') as f:
        pkg = pickle.load(f)
    return BayesianRegimeUpdater(pkg)


if __name__ == '__main__':
    train_and_save()
