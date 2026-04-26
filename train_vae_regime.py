"""
train_vae_regime.py — VAE Latent Space Regime (Continuous Regime Representation)
==================================================================================
Trains a Variational Autoencoder on daily market feature sequences.
Instead of hard regime labels, days are represented as points in a
continuous 8-dimensional latent space where similar market conditions
cluster together.

Architecture:
    Encoder: [21 features] → Dense(64) → Dense(32) → z_mean(8), z_logvar(8)
    Decoder: z(8) → Dense(32) → Dense(64) → [21 features]
    Loss: MSE reconstruction + KL divergence (β-VAE with β=0.5)

Output:
    vae_regime_v1.pkl  — trained VAE + scaler + regime cluster centers
    data/vae_latent_coords.parquet — date → 8-dim latent coordinates

Usage:
    from train_vae_regime import encode_bar, load_vae_pkg
    pkg  = load_vae_pkg()
    z    = encode_bar(pkg, bar_features)   # → np.array(8,)
    info = get_continuous_regime(pkg, z)
    # → {'z': [...8 floats...],
    #    'nearest_regime': 'EXPANSION',
    #    'regime_distances': {'EXPANSION': 0.31, 'PRE_EXPANSION': 0.84, ...},
    #    'regime_enc_soft': 2,
    #    'certainty': 0.71}   # 1=very close to cluster center, 0=boundary

Run:
    python3 train_vae_regime.py
"""

import numpy as np
import pandas as pd
import pickle
import sqlite3
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).parent
DB   = BASE / "mario_trading.db"
OUT  = BASE / "vae_regime_v1.pkl"
OUT_LATENT = BASE / "data" / "vae_latent_coords.parquet"

LATENT_DIM = 8
MESO_STATES = ['CONSOLIDATION', 'PRE_EXPANSION', 'EXPANSION',
               'RETRACEMENT', 'DISTRIBUTION']

VAE_FEATURES = [
    'adx_14', 'hurst', 'garch_vol', 'kalman_smooth', 'acf_lag1',
    'fisher_transform', 'sample_entropy', 'has_displacement',
    'dist_vwap_atr', 'dist_poc_atr', 'h4_bias_atr', 'h1_bias_atr',
    'above_true_open_atr', 'sweep_dn_atr', 'sweep_up_atr',
    'pre_range_atr', 'fvg_up', 'fvg_down', 'rvol',
    'atr_expanding', 'body_bear_pct',
]


# ══════════════════════════════════════════════════════════════════════════════
# VAE implemented in pure NumPy (no PyTorch dependency)
# ══════════════════════════════════════════════════════════════════════════════

class VAE:
    """
    Minimal VAE with 2-layer encoder/decoder, trained with Adam + reparameterization.
    Pure NumPy implementation — no external deep-learning dependencies.
    """

    def __init__(self, input_dim: int, latent_dim: int = 8,
                 hidden_dim: int = 32, beta: float = 0.5):
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta       = beta
        self._init_weights()

    def _init_weights(self):
        d, h, z = self.input_dim, self.hidden_dim, self.latent_dim
        scale = 0.1
        rng   = np.random.default_rng(42)

        # Encoder
        self.We1 = rng.normal(0, scale, (d, h))
        self.be1 = np.zeros(h)
        self.We2 = rng.normal(0, scale, (h, h))
        self.be2 = np.zeros(h)
        self.Wmu = rng.normal(0, scale, (h, z))
        self.bmu = np.zeros(z)
        self.Wlv = rng.normal(0, scale, (h, z))
        self.blv = np.zeros(z)

        # Decoder
        self.Wd1 = rng.normal(0, scale, (z, h))
        self.bd1 = np.zeros(h)
        self.Wd2 = rng.normal(0, scale, (h, h))
        self.bd2 = np.zeros(h)
        self.Wout = rng.normal(0, scale, (h, d))
        self.bout = np.zeros(d)

        # Adam moments
        self._init_adam()

    def _init_adam(self):
        self._adam_t = 0
        self._adam_m = {}
        self._adam_v = {}
        for name in ['We1','be1','We2','be2','Wmu','bmu','Wlv','blv',
                     'Wd1','bd1','Wd2','bd2','Wout','bout']:
            p = getattr(self, name)
            self._adam_m[name] = np.zeros_like(p)
            self._adam_v[name] = np.zeros_like(p)

    @staticmethod
    def _relu(x):   return np.maximum(0, x)
    @staticmethod
    def _relu_d(x): return (x > 0).astype(float)

    def encode(self, X: np.ndarray):
        """X: (batch, input_dim) → z_mean, z_logvar"""
        h1  = self._relu(X @ self.We1 + self.be1)
        h2  = self._relu(h1 @ self.We2 + self.be2)
        mu  = h2 @ self.Wmu + self.bmu
        lv  = np.clip(h2 @ self.Wlv + self.blv, -4, 4)
        return mu, lv, h1, h2

    def reparameterize(self, mu, lv, training=True):
        if training:
            eps = np.random.randn(*mu.shape)
            return mu + eps * np.exp(0.5 * lv)
        return mu

    def decode(self, z: np.ndarray):
        """z: (batch, latent_dim) → x_recon"""
        h1 = self._relu(z  @ self.Wd1 + self.bd1)
        h2 = self._relu(h1 @ self.Wd2 + self.bd2)
        xr = h2 @ self.Wout + self.bout
        return xr, h1, h2

    def loss(self, X: np.ndarray):
        mu, lv, _, _ = self.encode(X)
        z    = self.reparameterize(mu, lv)
        xr, _, _ = self.decode(z)
        recon = np.mean((X - xr) ** 2)
        kl    = -0.5 * np.mean(1 + lv - mu**2 - np.exp(lv))
        return recon + self.beta * kl, recon, kl

    def _adam_update(self, name, grad, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self._adam_t += 1
        m = self._adam_m[name]
        v = self._adam_v[name]
        m[:] = b1 * m + (1 - b1) * grad
        v[:] = b2 * v + (1 - b2) * grad**2
        m_hat = m / (1 - b1**self._adam_t)
        v_hat = v / (1 - b2**self._adam_t)
        return lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X: np.ndarray, epochs: int = 300, batch_size: int = 64,
            lr: float = 1e-3, verbose: bool = True):
        """Train VAE with mini-batch Adam."""
        n = len(X)
        losses = []

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            epoch_loss = []

            for start in range(0, n, batch_size):
                xb = X[idx[start:start+batch_size]]

                # Forward
                h1e = self._relu(xb @ self.We1 + self.be1)
                h2e = self._relu(h1e @ self.We2 + self.be2)
                mu  = h2e @ self.Wmu + self.bmu
                lv  = np.clip(h2e @ self.Wlv + self.blv, -4, 4)
                eps_  = np.random.randn(*mu.shape)
                z   = mu + eps_ * np.exp(0.5 * lv)
                h1d = self._relu(z  @ self.Wd1 + self.bd1)
                h2d = self._relu(h1d @ self.Wd2 + self.bd2)
                xr  = h2d @ self.Wout + self.bout

                # Loss
                diff  = xr - xb
                recon = np.mean(diff**2)
                kl    = -0.5 * np.mean(1 + lv - mu**2 - np.exp(lv))
                loss  = recon + self.beta * kl
                epoch_loss.append(loss)

                # Backward (simplified gradient, chain rule)
                bs = len(xb)

                # Decoder gradients
                d_xr   = 2 * diff / (bs * self.input_dim)
                d_h2d  = d_xr @ self.Wout.T * self._relu_d(h2d)
                d_h1d  = d_h2d @ self.Wd2.T * self._relu_d(h1d)
                d_z    = d_h1d @ self.Wd1.T

                # KL gradients for mu and lv
                d_mu_kl = (mu) / bs                            # dKL/dmu
                d_lv_kl = (-0.5 * (1 - np.exp(lv))) / bs      # dKL/dlv

                # Encoder gradients (through z = mu + eps*exp(0.5*lv))
                d_mu  = d_z + self.beta * d_mu_kl
                d_lv  = d_z * eps_ * 0.5 * np.exp(0.5 * lv) + self.beta * d_lv_kl
                d_h2e = (d_mu @ self.Wmu.T + d_lv @ self.Wlv.T) * self._relu_d(h2e)
                d_h1e = d_h2e @ self.We2.T * self._relu_d(h1e)

                # Weight gradients
                grads = {
                    'Wout': h2d.T @ d_xr,       'bout': d_xr.sum(0),
                    'Wd2':  h1d.T @ d_h2d,       'bd2':  d_h2d.sum(0),
                    'Wd1':  z.T   @ d_h1d,       'bd1':  d_h1d.sum(0),
                    'Wmu':  h2e.T @ d_mu,         'bmu':  d_mu.sum(0),
                    'Wlv':  h2e.T @ d_lv,         'blv':  d_lv.sum(0),
                    'We2':  h1e.T @ d_h2e,        'be2':  d_h2e.sum(0),
                    'We1':  xb.T  @ d_h1e,        'be1':  d_h1e.sum(0),
                }
                for name, grad in grads.items():
                    update = self._adam_update(name, grad, lr=lr)
                    setattr(self, name, getattr(self, name) - update)

            epoch_loss_mean = float(np.mean(epoch_loss))
            losses.append(epoch_loss_mean)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1:>3}/{epochs}  loss={epoch_loss_mean:.4f}")

        return losses

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode X to latent mean vectors (no sampling)."""
        mu, _, _, _ = self.encode(X)
        return mu


# ══════════════════════════════════════════════════════════════════════════════
# Cluster centers: map latent space back to regime names
# ══════════════════════════════════════════════════════════════════════════════

def fit_cluster_centers(Z: np.ndarray, labels: np.ndarray,
                        label_names: list) -> dict:
    """
    Compute cluster center in latent space for each regime.
    Returns {regime: z_center (np.array)}
    """
    centers = {}
    for i, name in enumerate(label_names):
        mask = labels == i
        if mask.sum() > 0:
            centers[name] = Z[mask].mean(axis=0)
        else:
            centers[name] = np.zeros(Z.shape[1])
    return centers


def get_continuous_regime(pkg: dict, z: np.ndarray) -> dict:
    """
    Given a latent vector z, find the nearest regime cluster center.
    Returns continuous soft assignment.
    """
    centers  = pkg['cluster_centers']   # {regime: np.array(latent_dim)}
    dists    = {r: float(np.linalg.norm(z - c)) for r, c in centers.items()}
    nearest  = min(dists, key=dists.get)
    nearest_enc = MESO_STATES.index(nearest) if nearest in MESO_STATES else 0

    # Soft assignment via inverse distance weighting
    inv_d   = {r: 1.0 / (d + 1e-6) for r, d in dists.items()}
    total   = sum(inv_d.values())
    soft_p  = {r: v / total for r, v in inv_d.items()}

    # Certainty: how close to nearest vs average distance
    min_d   = dists[nearest]
    avg_d   = np.mean(list(dists.values()))
    certainty = float(max(0, 1 - min_d / (avg_d + 1e-6)))

    return {
        'z':                 z.tolist(),
        'nearest_regime':    nearest,
        'regime_enc_soft':   nearest_enc,
        'regime_distances':  dists,
        'regime_probs':      soft_p,
        'certainty':         certainty,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Encoder helper for runtime
# ══════════════════════════════════════════════════════════════════════════════

def encode_bar(pkg: dict, bar_features: dict) -> np.ndarray:
    """Encode a single bar's features to latent space."""
    vae     = pkg['vae']
    scaler  = pkg['scaler']
    feats   = pkg['features']
    x = np.array([[float(bar_features.get(f, 0) or 0) for f in feats]])
    x_scaled = scaler.transform(x)
    z = vae.transform(x_scaled)
    return z[0]


def load_vae_pkg(path=None):
    p = Path(path) if path else OUT
    if not p.exists():
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_training_data(conn) -> pd.DataFrame:
    """Aggregate 1-min bars from market_data to daily (nq_data only has 5 days of 2026)."""
    sql = """
    SELECT date,
           AVG(atr_14)           AS atr_14,
           AVG(hurst)            AS hurst,
           AVG(garch_vol)        AS garch_vol,
           AVG(kalman_smooth)    AS kalman_smooth,
           AVG(acf_lag1)         AS acf_lag1,
           AVG(fisher_transform) AS fisher_transform,
           AVG(sample_entropy)   AS sample_entropy,
           MAX(has_displacement) AS has_displacement,
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
           MAX(fvg_up)           AS fvg_up,
           MAX(fvg_down)         AS fvg_down,
           AVG(adx_14)           AS adx_14,
           MAX(p_hi)             AS p_hi,
           MIN(p_lo)             AS p_lo,
           MAX(asia_hi)          AS asia_hi,
           MIN(asia_lo)          AS asia_lo
    FROM market_data
    WHERE year >= 2021 AND adx_14 > 0 AND atr_14 > 0
      AND day_of_week BETWEEN 1 AND 5
    GROUP BY date ORDER BY date
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
    df['sweep_up_atr']        = 0.0   # no intraday sweep on daily aggregates
    df['sweep_dn_atr']        = 0.0
    df['pre_range_atr']       = (df['p_hi'].fillna(0) - df['p_lo'].fillna(0)).clip(lower=0) / atr
    df['rvol']                = df['garch_vol'] / df['garch_vol'].rolling(20, min_periods=5).mean().fillna(df['garch_vol'])
    df['atr_expanding']       = (df['atr_14'] > df['atr_14'].rolling(5).mean()).astype(float)
    df['body_bear_pct']       = (df['close'] < df['true_open'].fillna(df['close'])).astype(float)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def train_and_save():
    print("=" * 65)
    print("  VAE REGIME — Continuous Latent Space")
    print(f"  Latent dim: {LATENT_DIM} | Features: {len(VAE_FEATURES)}")
    print("=" * 65)

    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    daily = load_training_data(conn)
    conn.close()

    feat_cols = [c for c in VAE_FEATURES if c in daily.columns]
    print(f"\n  Features available: {len(feat_cols)}/{len(VAE_FEATURES)}")

    X_raw = daily[feat_cols].fillna(0).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    print(f"  Training samples: {len(X)}")

    # Train VAE
    print("\n  Training VAE ...")
    vae = VAE(input_dim=len(feat_cols), latent_dim=LATENT_DIM,
              hidden_dim=32, beta=0.5)
    vae.fit(X, epochs=300, batch_size=64, lr=5e-4, verbose=True)

    # Encode all training data
    Z = vae.transform(X)
    print(f"\n  Latent space: {Z.shape}  "
          f"mean_norm={np.linalg.norm(Z, axis=1).mean():.2f}")

    # Load regime labels for cluster center computation
    labels_path = BASE / 'data' / 'regime_labels.parquet'
    cluster_centers = {}
    silhouette = None

    if labels_path.exists():
        labels = pd.read_parquet(labels_path)
        lon_labels = labels[labels['session'] == 'LON'][['date', 'regime']].copy()
        lon_labels['date'] = pd.to_datetime(lon_labels['date']).dt.strftime('%Y-%m-%d')
        daily_with_regime = daily.merge(lon_labels, on='date', how='inner')

        if len(daily_with_regime) > 0:
            Z_labeled = vae.transform(scaler.transform(
                daily_with_regime[feat_cols].fillna(0).values
            ))
            from sklearn.preprocessing import LabelEncoder
            le_regime = LabelEncoder()
            le_regime.fit(MESO_STATES)
            regime_vals = daily_with_regime['regime'].where(
                daily_with_regime['regime'].isin(MESO_STATES), 'CONSOLIDATION'
            )
            y_labels = le_regime.transform(regime_vals)

            cluster_centers = fit_cluster_centers(Z_labeled, y_labels, le_regime.classes_.tolist())

            # Silhouette score: how well separated are regime clusters?
            if len(np.unique(y_labels)) > 1:
                silhouette = silhouette_score(Z_labeled, y_labels, sample_size=min(500, len(Z_labeled)))
                print(f"\n  Silhouette score (regime separation in latent space): {silhouette:.3f}")
                print(f"  (>0.3 = good separation, >0.5 = excellent)")

            print(f"\n  Cluster centers computed for {len(cluster_centers)} regimes")
            for r, c in cluster_centers.items():
                print(f"    {r:<18} center_norm={np.linalg.norm(c):.2f}")

    # Save latent coordinates
    OUT_LATENT.parent.mkdir(exist_ok=True)
    latent_df = pd.DataFrame(Z, columns=[f'z_{i}' for i in range(LATENT_DIM)])
    latent_df.insert(0, 'date', daily['date'].values)
    latent_df.to_parquet(OUT_LATENT, index=False)
    print(f"\n  ✅ Saved latent coords → {OUT_LATENT.name}")

    # Reconstruction quality
    X_recon = vae.decode(Z)[0]
    recon_err = np.mean((X - X_recon) ** 2)
    print(f"  Reconstruction MSE: {recon_err:.4f}")

    pkg = {
        'vae':             vae,
        'scaler':          scaler,
        'features':        feat_cols,
        'latent_dim':      LATENT_DIM,
        'cluster_centers': cluster_centers,
        'meso_states':     MESO_STATES,
        'silhouette':      silhouette,
        'recon_mse':       float(recon_err),
        'n_training':      len(X),
        'version':         'vae_v1',
    }
    with open(OUT, 'wb') as f:
        pickle.dump(pkg, f)
    print(f"  ✅ Saved PKL → {OUT.name}")


if __name__ == '__main__':
    train_and_save()
