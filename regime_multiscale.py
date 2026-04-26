"""
regime_multiscale.py — Multi-Scale Regime Classifier v1
=========================================================
Three-layer regime system:

  1. HierarchicalHMM
     - Macro-level: 3 states (bull / bear / chop) learned from daily OHLCV features
       using Gaussian HMM (hmmlearn).
     - Micro-level: per macro-state transition matrices → different 5-state micro
       HMMs conditioned on macro context.
     - Outputs: macro_state (0-2), micro_state (0-4), micro_proba[5]

  2. OnlineBayesianRegime
     - Intraday posterior updates using Dirichlet–Multinomial conjugate model.
     - Prior = yesterday's regime distribution.
     - Updates with each new bar's soft regime evidence.
     - Outputs: bayesian_posterior[5] — probability vector over 5 regimes

  3. LatentRegimeEncoder (sklearn MLP autoencoder — replaces VAE when no PyTorch)
     - Learns 8-dim continuous latent regime space from daily feature sequences.
     - Encoder: daily_features[d] → latent[8]
     - Decoder: latent[8] → reconstruction (unsupervised)
     - Outputs: latent[8] — continuous regime embedding

  RegimeMultiscale = wrapper that calls all 3 and returns a combined feature vector
  for downstream models (LOM v3, NOM v3, stacking).

Usage:
    from regime_multiscale import RegimeMultiscale
    rms = RegimeMultiscale.load('regime_multiscale_v1.pkl')
    features = rms.predict_features(date_str, bar_features)  # → dict

Training:
    python regime_multiscale.py
"""

import sqlite3, warnings, joblib, json, logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("REGIME_MS")

DB      = Path(__file__).parent / "mario_trading.db"
OUT_PKL = Path(__file__).parent / "regime_multiscale_v1.pkl"

MACRO_STATES = 3   # bull / bear / chop
MICRO_STATES = 5   # CONSOLIDATION / PRE_EXPANSION / EXPANSION / RETRACEMENT / DISTRIBUTION
LATENT_DIM   = 8   # latent regime space dimensionality

TRAIN_START  = "2018-01-01"
TRAIN_END    = "2024-12-31"
OOS_START    = "2025-01-01"
OOS_END      = "2026-04-22"

DAILY_FEATURES = [
    'avg_atr', 'daily_range', 'range_atr_ratio',
    'avg_adx', 'avg_hurst', 'avg_rvol',
    'avg_body', 'avg_delta_abs',
    'close_vs_open',         # daily directional bias
    'high_low_ratio',        # intraday range relative to close
    'vwap_dist_mean',        # mean dist from VWAP
    'cum_delta_eod',         # end-of-day cum_delta sign
]

INTRADAY_FEATURES = [
    'adx_14', 'hurst', 'garch_vol',
    'dist_vwap_atr', 'dist_poc_atr',
    'bar_delta_atr', 'rvol',
    'is_session_open',
    'sweep_dn_atr', 'sweep_up_atr',
    'has_displacement',
]


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_daily_features(conn, start, end):
    """
    Returns DataFrame with one row per trading day:
    avg_atr, daily_range, range_atr_ratio, avg_adx, avg_hurst, etc.
    """
    log.info(f"  Loading daily features {start} → {end} ...")
    q = f"""
        SELECT
            date(timestamp) as date,
            AVG(atr_14)     as avg_atr,
            MAX(high)-MIN(low) as daily_range,
            AVG(adx_14)     as avg_adx,
            AVG(hurst)      as avg_hurst,
            AVG(rvol)       as avg_rvol,
            AVG(body_size)  as avg_body,
            AVG(ABS(bar_delta)) as avg_delta_abs,
            (MAX(close)-MIN(open))/NULLIF(AVG(atr_14),0) as close_vs_open,
            (MAX(high)-MIN(low))/NULLIF(AVG(close),0)    as high_low_ratio,
            AVG(dist_vwap)  as vwap_dist_mean,
            SUM(bar_delta)  as cum_delta_eod
        FROM market_data
        WHERE date BETWEEN '{start}' AND '{end}'
          AND atr_14 > 0 AND adx_14 > 0
        GROUP BY date(timestamp)
        ORDER BY date
    """
    df = pd.read_sql(q, conn)
    df['date'] = df['date'].astype(str)
    df['range_atr_ratio'] = df['daily_range'] / df['avg_atr'].clip(lower=0.01)
    df = df.ffill().fillna(0.0)
    log.info(f"  → {len(df)} days")
    return df


def load_intraday_sample(conn, start, end, max_rows=200_000):
    """Sample of 1-min bars for intraday regime model training."""
    log.info(f"  Loading intraday sample {start} → {end} ...")
    total = pd.read_sql(
        f"SELECT COUNT(*) as n FROM market_data WHERE date BETWEEN '{start}' AND '{end}' AND adx_14>0",
        conn).iloc[0, 0]
    every_n = max(1, int(total / max_rows))
    q = f"""
        SELECT date(timestamp) as date, hour_min,
               atr_14, adx_14, hurst, garch_vol,
               dist_vwap, dist_poc, bar_delta, rvol,
               body_size, has_displacement,
               lon_hi, lon_lo, p_hi, p_lo, high, low, close
        FROM market_data
        WHERE date BETWEEN '{start}' AND '{end}'
          AND adx_14 > 0 AND atr_14 > 0
          AND (ROWID % {every_n} = 0)
        ORDER BY date, hour_min
        LIMIT {max_rows}
    """
    df = pd.read_sql(q, conn)
    df['date'] = df['date'].astype(str)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. Hierarchical HMM
# ══════════════════════════════════════════════════════════════════════════════

class HierarchicalHMM:
    """
    Two-level HMM:
    - Macro HMM: 3 states (bull/bear/chop) trained on daily features
    - Micro HMMs: one per macro state, 5 states each, trained on intraday features
      within the corresponding macro regime

    Training:
        fit_macro(daily_df)   — trains macro 3-state HMM
        fit_micro(intraday_df, daily_macro_states) — trains 3 micro HMMs

    Inference:
        predict(date_str, intraday_features) → (macro_state, micro_state, micro_proba[5])
    """

    def __init__(self, n_macro=MACRO_STATES, n_micro=MICRO_STATES):
        self.n_macro = n_macro
        self.n_micro = n_micro
        self.macro_hmm = None
        self.micro_hmms = {}   # macro_state → GaussianHMM
        self.macro_scaler = StandardScaler()
        self.micro_scaler = StandardScaler()
        self.macro_feature_cols = DAILY_FEATURES
        self.micro_feature_cols = INTRADAY_FEATURES
        self.date_macro_cache = {}   # date → macro_state (inference cache)

    def _prep_daily(self, df):
        cols = [c for c in self.macro_feature_cols if c in df.columns]
        X = df[cols].fillna(0).values.astype(float)
        return X, cols

    def _prep_intraday(self, df):
        atr = df['atr_14'].clip(lower=0.01)
        df = df.copy()
        df['dist_vwap_atr'] = df['dist_vwap'].abs() / atr
        df['dist_poc_atr']  = df['dist_poc'].abs() / atr
        df['bar_delta_atr'] = df['bar_delta'].abs() / df['rvol'].clip(lower=0.01)
        h = df['hour_min'].str.replace(':', '').astype(int)
        df['is_session_open'] = (
            ((h >= 400) & (h <= 700)) | ((h >= 900) & (h <= 1130))
        ).astype(float)
        sess_hi = df[['lon_hi', 'p_hi']].max(axis=1)
        sess_lo = df[['lon_lo', 'p_lo']].min(axis=1)
        df['sweep_dn_atr'] = ((sess_lo - df['low']).clip(lower=0) / atr)
        df['sweep_up_atr'] = ((df['high'] - sess_hi).clip(lower=0) / atr)
        cols = [c for c in self.micro_feature_cols if c in df.columns]
        X = df[cols].fillna(0).values.astype(float)
        return X, cols

    def fit_macro(self, daily_df):
        from hmmlearn.hmm import GaussianHMM
        log.info(f"  [HierHMM] Fitting macro HMM ({self.n_macro} states) on {len(daily_df)} days ...")
        X_raw, cols = self._prep_daily(daily_df)
        self.macro_feature_cols = cols
        X = self.macro_scaler.fit_transform(X_raw)
        self.macro_hmm = GaussianHMM(
            n_components=self.n_macro, covariance_type='diag',
            n_iter=200, random_state=42, verbose=False,
            min_covar=1e-3,
        )
        self.macro_hmm.fit(X)
        states = self.macro_hmm.predict(X)
        # Assign semantic labels: sort macro states by mean(avg_adx + range_atr_ratio)
        # highest activity → bull/bear, lowest → chop
        adx_col = cols.index('avg_adx') if 'avg_adx' in cols else 0
        mean_by_state = [X_raw[states == s, adx_col].mean() if (states == s).sum() > 0 else 0.0
                         for s in range(self.n_macro)]
        self.macro_label_map = {s: int(np.argsort(mean_by_state)[s]) for s in range(self.n_macro)}
        log.info(f"  [HierHMM] Macro state distribution: "
                 f"{ {s: int((states==s).sum()) for s in range(self.n_macro)} }")
        # Cache state → date
        dates = daily_df['date'].values
        for i, d in enumerate(dates):
            self.date_macro_cache[str(d)] = int(states[i])
        return states

    def fit_micro(self, intraday_df, daily_macro_states, daily_dates):
        from hmmlearn.hmm import GaussianHMM
        # Build date → macro_state lookup
        d2m = {str(d): int(s) for d, s in zip(daily_dates, daily_macro_states)}
        intraday_df = intraday_df.copy()
        intraday_df['macro_state'] = intraday_df['date'].map(d2m).fillna(-1).astype(int)

        X_all_raw, micro_cols = self._prep_intraday(intraday_df)
        self.micro_feature_cols = micro_cols
        X_all = self.micro_scaler.fit_transform(X_all_raw)

        for ms in range(self.n_macro):
            mask = intraday_df['macro_state'].values == ms
            X_ms = X_all[mask]
            n_ms = X_ms.shape[0]
            if n_ms < self.n_micro * 20:
                log.warning(f"  [HierHMM] macro={ms}: only {n_ms} samples — using global model")
                X_ms = X_all

            log.info(f"  [HierHMM] Fitting micro HMM macro={ms} ({self.n_micro} states) on {len(X_ms)} bars ...")
            hmm = GaussianHMM(
                n_components=self.n_micro, covariance_type='diag',
                n_iter=150, random_state=42, verbose=False, min_covar=1e-3,
            )
            hmm.fit(X_ms)
            self.micro_hmms[ms] = hmm

    def predict(self, date_str, intraday_features_dict):
        """
        date_str: 'YYYY-MM-DD'
        intraday_features_dict: dict of feature_name → value for current bar

        Returns: (macro_state, micro_state, micro_proba[5])
        """
        # Macro state (use cache if available, else predict)
        macro_state = self.date_macro_cache.get(date_str, 0)

        hmm = self.micro_hmms.get(macro_state, self.micro_hmms.get(0))
        if hmm is None:
            return macro_state, 0, np.ones(self.n_micro) / self.n_micro

        x = np.array([[intraday_features_dict.get(c, 0.0) for c in self.micro_feature_cols]])
        x_sc = self.micro_scaler.transform(x)
        micro_state = int(hmm.predict(x_sc)[0])
        # Posterior probability over micro states
        try:
            log_prob, fwd = hmm.score_samples(x_sc)
            micro_proba = np.exp(fwd[0] - np.max(fwd[0]))
            micro_proba /= micro_proba.sum()
        except Exception:
            micro_proba = np.eye(self.n_micro)[micro_state]

        return macro_state, micro_state, micro_proba


# ══════════════════════════════════════════════════════════════════════════════
# 2. Online Bayesian Regime Updater
# ══════════════════════════════════════════════════════════════════════════════

class OnlineBayesianRegime:
    """
    Dirichlet–Multinomial online Bayesian model for intraday regime posterior.

    Prior: uniform or previous-day regime distribution.
    Update: each new bar provides "soft evidence" for each regime class.

    The evidence mapping uses a small matrix E[feature_bin, regime] that is
    learned from training data.

    Usage:
        obs = OnlineBayesianRegime.load_or_train(daily_df, intraday_df)
        prior = obs.get_prior(date_str)          # α vector from yesterday
        posterior = obs.update(prior, bar_obs)   # after seeing a bar
    """

    def __init__(self, n_regimes=MICRO_STATES):
        self.n_regimes = n_regimes
        self.alpha_prior = np.ones(n_regimes)  # uniform Dirichlet prior
        # Evidence matrix: how each "observation bin" maps to regime probabilities
        # Rows = discretized observation bins (adx_level × session_open), Cols = regimes
        # Learned from empirical regime label distribution per bin
        self.evidence_matrix = None    # shape (n_bins, n_regimes)
        self.bin_names = []
        self.daily_terminal_states = {}  # date → terminal posterior (α vector)

    def _make_bins(self, df_intraday):
        """
        Creates discrete observation bins from key bar features:
        - adx_level: 0=low(<18), 1=mid(18-25), 2=high(>25)
        - session: 0=other, 1=LON, 2=NY
        Returns array of bin indices per bar.
        """
        atr = df_intraday['atr_14'].clip(lower=0.01)
        adx = df_intraday['adx_14'].values
        h = df_intraday['hour_min'].str.replace(':', '').astype(int)
        sweep = (((df_intraday[['lon_lo', 'p_lo']].min(axis=1) - df_intraday['low']).clip(lower=0) /
                   atr > 0.3) |
                  ((df_intraday['high'] - df_intraday[['lon_hi', 'p_hi']].max(axis=1)).clip(lower=0) /
                   atr > 0.3)).astype(int).values

        adx_bin  = np.where(adx < 18, 0, np.where(adx < 25, 1, 2))
        sess_bin = np.where((h >= 400) & (h <= 700), 1,
                   np.where((h >= 900) & (h <= 1130), 2, 0))
        has_d = df_intraday['has_displacement'].fillna(0).values.astype(int)

        # Combined bin: 3 adx × 3 sess × 2 sweep × 2 disp = 36 bins
        bins = adx_bin * 12 + sess_bin * 4 + sweep * 2 + has_d
        self.n_bins = 36
        return bins

    def fit(self, intraday_df, regime_labels):
        """
        regime_labels: array of int regime labels (0-4) per bar, same length as intraday_df.
        Learns evidence matrix from empirical P(bin | regime).
        """
        log.info(f"  [BayesRegime] Fitting on {len(intraday_df)} bars ...")
        bins = self._make_bins(intraday_df)
        # Build evidence matrix: E[bin, regime] = P(regime | bin) from training
        E = np.zeros((self.n_bins, self.n_regimes)) + 0.1   # Laplace smoothing
        for b in range(self.n_bins):
            mask = bins == b
            if mask.sum() > 0:
                lab = regime_labels[mask]
                for r in range(self.n_regimes):
                    E[b, r] += (lab == r).sum()
        # Normalize rows
        E = E / E.sum(axis=1, keepdims=True)
        self.evidence_matrix = E
        log.info(f"  [BayesRegime] Evidence matrix: {E.shape}")

    def get_prior(self, date_str, default_uniform=True):
        """
        Returns the α (Dirichlet concentration) vector for the start of `date_str`.
        Uses yesterday's terminal state if available, else uniform.
        """
        if date_str in self.daily_terminal_states:
            return self.daily_terminal_states[date_str].copy()
        if default_uniform:
            return np.ones(self.n_regimes) * 2.0   # weak uniform prior
        return self.alpha_prior.copy()

    def update(self, alpha, bar_obs_dict, df_single_bar=None):
        """
        Bayesian update: α_new = α + likelihood(observation).
        bar_obs_dict: dict with adx_14, hour_min, has_displacement, etc.
        Returns updated α and normalized posterior probabilities.
        """
        if self.evidence_matrix is None:
            return alpha, alpha / alpha.sum()

        # Compute bin for this bar
        adx = bar_obs_dict.get('adx_14', 20.0)
        hm  = bar_obs_dict.get('hhmm', bar_obs_dict.get('hour_min_int', 1000))
        sw  = bar_obs_dict.get('sweep_dn_atr', 0.0) + bar_obs_dict.get('sweep_up_atr', 0.0)
        hd  = int(bar_obs_dict.get('has_displacement', 0))
        adx_bin  = 0 if adx < 18 else (1 if adx < 25 else 2)
        sess_bin = 1 if 400 <= hm <= 700 else (2 if 900 <= hm <= 1130 else 0)
        sw_bin   = 1 if sw > 0.3 else 0
        b = adx_bin * 12 + sess_bin * 4 + sw_bin * 2 + hd
        b = min(b, self.n_bins - 1)

        # Likelihood from evidence matrix
        likelihood = self.evidence_matrix[b]
        alpha_new = alpha + likelihood
        posterior = alpha_new / alpha_new.sum()
        return alpha_new, posterior

    def set_terminal_state(self, date_str, alpha):
        """Call at end of trading day to save state for next day's prior."""
        self.daily_terminal_states[date_str] = alpha.copy()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Latent Regime Encoder (sklearn autoencoder, replaces VAE)
# ══════════════════════════════════════════════════════════════════════════════

class LatentRegimeEncoder:
    """
    MLP Autoencoder that learns a compressed latent representation of
    daily regime conditions. The bottleneck layer = latent_dim = 8.

    Architecture:
        Encoder: daily_features[n_feat] → 32 → 16 → latent[8]
        Decoder: latent[8] → 16 → 32 → daily_features[n_feat]

    Implemented as two MLPRegressors (encoder + decoder) trained jointly
    by training a combined model that reconstructs input.
    For inference, we extract the encoder portion.
    """

    def __init__(self, latent_dim=LATENT_DIM):
        self.latent_dim = latent_dim
        self.scaler = StandardScaler()
        # Combined autoencoder: in → latent (bottleneck) → out
        # hidden_layer_sizes: [32, 16, latent_dim, 16, 32]
        self._autoencoder = None
        self._encoder = None
        self.feature_cols = DAILY_FEATURES
        self.date_latent_cache = {}   # date → latent[8]

    def fit(self, daily_df):
        log.info(f"  [LatentEnc] Fitting autoencoder on {len(daily_df)} days ...")
        cols = [c for c in self.feature_cols if c in daily_df.columns]
        self.feature_cols = cols
        X_raw = daily_df[cols].fillna(0).values.astype(float)
        X = self.scaler.fit_transform(X_raw)
        n_feat = X.shape[1]

        # Train autoencoder as one big MLP that maps X → X
        # Bottleneck is the 4th hidden layer (latent_dim neurons)
        hidden = (64, 32, 16, self.latent_dim, 16, 32, 64)
        self._autoencoder = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation='tanh', solver='adam',
            max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, verbose=False,
        )
        self._autoencoder.fit(X, X)

        # Encoder: separate MLP trained to reproduce bottleneck activations
        # We extract latent representations by passing X through autoencoder's
        # first 4 layers manually
        log.info(f"  [LatentEnc] Extracting latent vectors ...")
        latent_vectors = self._encode_batch(X)

        # Train a lightweight encoder MLP: X → latent (supervised on extracted latent)
        self._encoder = MLPRegressor(
            hidden_layer_sizes=(32, 16, self.latent_dim),
            activation='tanh', solver='adam',
            max_iter=300, random_state=42,
            verbose=False,
        )
        self._encoder.fit(X, latent_vectors)

        # Cache all training dates
        dates = daily_df['date'].values
        for i, d in enumerate(dates):
            self.date_latent_cache[str(d)] = latent_vectors[i].tolist()

        log.info(f"  [LatentEnc] Done. Latent dim={self.latent_dim}, "
                 f"reconstruction error={self._autoencoder.loss_:.4f}")

    def _encode_batch(self, X_scaled):
        """Extract activations from the bottleneck layer of the autoencoder."""
        # Manually propagate through layers up to bottleneck
        # autoencoder architecture: hidden_layer_sizes = (64, 32, 16, latent_dim, ...)
        # Bottleneck is index 3 (0-based) in hidden_layer_sizes
        m = self._autoencoder
        h = X_scaled
        bottleneck_idx = 3  # layer index of latent_dim neurons
        for i, (W, b) in enumerate(zip(m.coefs_, m.intercepts_)):
            h = h @ W + b
            if i == bottleneck_idx:
                break
            # activation (tanh for all hidden, linear for output)
            h = np.tanh(h)
        return h  # latent representations, shape (n_samples, latent_dim)

    def encode(self, date_str, daily_features_dict=None):
        """
        Returns latent vector for a given date.
        If date is in cache, returns cached value.
        Otherwise uses encoder MLP with provided features dict.
        """
        if date_str in self.date_latent_cache:
            return np.array(self.date_latent_cache[date_str])
        if daily_features_dict is None or self._encoder is None:
            return np.zeros(self.latent_dim)
        x = np.array([[daily_features_dict.get(c, 0.0) for c in self.feature_cols]])
        x_sc = self.scaler.transform(x)
        return self._encoder.predict(x_sc)[0]

    def encode_batch(self, daily_df):
        """Encode all dates in daily_df, returns array (n_days, latent_dim)."""
        cols = [c for c in self.feature_cols if c in daily_df.columns]
        X_raw = daily_df[cols].fillna(0).values.astype(float)
        X = self.scaler.transform(X_raw)
        return self._encoder.predict(X)


# ══════════════════════════════════════════════════════════════════════════════
# Combined RegimeMultiscale
# ══════════════════════════════════════════════════════════════════════════════

class RegimeMultiscale:
    """
    Combined multi-scale regime system.
    Provides a single predict_features() call that returns a dict of
    all regime features to be concatenated with model inputs.

    Feature dict keys:
        macro_state         — int 0-2 (HMM macro)
        micro_state         — int 0-4 (HMM micro)
        micro_proba_0..4    — float (micro state probabilities)
        bayes_posterior_0..4— float (Bayesian posterior per regime)
        latent_0..7         — float (continuous regime embedding)
        macro_is_bull       — 0/1
        macro_is_bear       — 0/1
        macro_is_chop       — 0/1
    """

    def __init__(self):
        self.hmm   = HierarchicalHMM()
        self.bayes = OnlineBayesianRegime()
        self.enc   = LatentRegimeEncoder()
        self.daily_df = None
        self._daily_feature_cache = {}   # date → dict

    def fit(self, conn, train_start=TRAIN_START, train_end=TRAIN_END):
        log.info("=" * 65)
        log.info("  REGIME MULTISCALE — Training")
        log.info("=" * 65)

        # 1. Daily features
        log.info("\n[1] Loading daily features ...")
        daily_df = load_daily_features(conn, train_start, train_end)
        self.daily_df = daily_df

        # Cache daily feature dicts for fast lookup
        for _, row in daily_df.iterrows():
            self.daily_df_cache = {
                r['date']: {c: float(r[c]) for c in DAILY_FEATURES if c in daily_df.columns}
                for _, r in daily_df.iterrows()
            }

        # 2. Latent encoder
        log.info("\n[2] Training Latent Regime Encoder ...")
        self.enc.fit(daily_df)

        # 3. Macro HMM
        log.info("\n[3] Training Macro HMM ...")
        macro_states = self.hmm.fit_macro(daily_df)

        # 4. Intraday sample for micro HMM + Bayesian model
        log.info("\n[4] Loading intraday sample ...")
        intraday_df = load_intraday_sample(conn, train_start, train_end)

        log.info("\n[5] Training Micro HMMs ...")
        self.hmm.fit_micro(intraday_df, macro_states, daily_df['date'].values)

        # 6. Bayesian model
        log.info("\n[6] Training Online Bayesian Regime ...")
        # Use regime labels from existing regime_classifier_v1 if available
        regime_pkl = Path(__file__).parent / "regime_classifier_v1.pkl"
        if regime_pkl.exists():
            import joblib as jl2
            pkg = jl2.load(regime_pkl)
            rc_model = pkg['model']
            rc_features = pkg['features']
            rc_le = pkg['label_encoder']
            log.info("  Using regime_classifier_v1 labels for Bayesian training ...")
            # Build feature matrix for intraday_df
            from train_regime import build_features as _bf
            intraday_feat = _bf(intraday_df)
            avail_feat = [c for c in rc_features if c in intraday_feat.columns]
            X_intra = intraday_feat[avail_feat].fillna(0).astype(float)
            if len(avail_feat) == len(rc_features):
                bayes_labels = rc_le.inverse_transform(rc_model.predict(X_intra))
            else:
                log.warning(f"  Feature mismatch ({len(avail_feat)}/{len(rc_features)}), using micro HMM labels")
                bayes_labels = np.array([
                    self.hmm.predict(
                        str(intraday_df.iloc[i]['date']),
                        {c: float(intraday_df.iloc[i][c]) for c in self.hmm.micro_feature_cols if c in intraday_df.columns}
                    )[1] for i in range(0, len(intraday_df), 100)
                ])[:len(intraday_df)]
        else:
            log.warning("  regime_classifier_v1.pkl not found — using micro HMM labels")
            bayes_labels = np.zeros(len(intraday_df), dtype=int)

        self.bayes.fit(intraday_df, bayes_labels)
        log.info("\nTraining complete.")

    def predict_features(self, date_str, bar_features_dict, bayesian_alpha=None):
        """
        Returns a flat dict of regime features for a single bar.
        bar_features_dict: dict with adx_14, hour_min, has_displacement, etc.
        bayesian_alpha: current session's α vector (pass None to start fresh).
        """
        # HMM
        macro_state, micro_state, micro_proba = self.hmm.predict(date_str, bar_features_dict)

        # Bayesian update
        if bayesian_alpha is None:
            bayesian_alpha = self.bayes.get_prior(date_str)
        bayes_alpha_new, bayes_posterior = self.bayes.update(bayesian_alpha, bar_features_dict)

        # Latent encoding
        daily_feat = getattr(self, 'daily_df_cache', {}).get(date_str)
        latent = self.enc.encode(date_str, daily_feat)

        features = {
            'macro_state':    macro_state,
            'micro_state':    micro_state,
            'macro_is_bull':  int(macro_state == 2),
            'macro_is_bear':  int(macro_state == 0),
            'macro_is_chop':  int(macro_state == 1),
        }
        for i, p in enumerate(micro_proba):
            features[f'micro_proba_{i}'] = float(p)
        for i, p in enumerate(bayes_posterior):
            features[f'bayes_posterior_{i}'] = float(p)
        for i, v in enumerate(latent):
            features[f'latent_{i}'] = float(v)

        return features, bayes_alpha_new

    def get_date_features(self, date_str):
        """Get pre-computed features for a full date (from cache)."""
        return {
            'macro_state': self.hmm.date_macro_cache.get(date_str, 0),
            'latent':      self.enc.encode(date_str),
        }

    def save(self, path=OUT_PKL):
        joblib.dump(self, path)
        log.info(f"✅ Saved RegimeMultiscale → {path}")

    @classmethod
    def load(cls, path=OUT_PKL):
        return joblib.load(path)


# ══════════════════════════════════════════════════════════════════════════════
# Training entry point
# ══════════════════════════════════════════════════════════════════════════════

def train():
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    rms = RegimeMultiscale()
    rms.fit(conn)
    conn.close()
    rms.save()
    log.info("Done.")


if __name__ == '__main__':
    train()
