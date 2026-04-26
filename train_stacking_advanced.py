"""
train_stacking_advanced.py — Advanced Stacking System v1
==========================================================
Three-tier meta-learning on top of LOM/NOM v3 predictions:

  1. UncertaintyWrapper
     - MC Dropout simulation: runs N=30 bootstrap resamples of the prediction
       (XGBoost has no native dropout, we simulate via feature subsets + data subsets)
     - Conformal prediction: uses calibration set residuals to build CI bounds
     - Output: mean_prob, ci_lower, ci_upper, uncertainty = ci_upper - ci_lower

  2. TemporalAttentionMeta
     - Attention over last N=25 trades (rolling window)
     - Learns to dynamically reweight LOM/NOM/regime signals based on which
       model performed best in recent trades
     - Implemented as a learned linear attention + learned decay (no transformer)
     - Output: attention_prob = weighted combination of model signals

  3. RLMetaPolicy (REINFORCE)
     - State:  [lom_prob, nom_prob, regime_enc_norm, uncertainty, recent_wr_5, recent_wr_10]
     - Actions: 0=skip, 1=standard, 2=oversize
     - Reward:  risk-adjusted PnL based on label + probabilities
     - Policy:  2-layer MLP with softmax output, trained via REINFORCE
     - Output:  action_proba[3], recommended_action

  StackingPipeline: combines all 3 layers into a final trade decision
  Input: one row per trade (lom_prob, nom_prob, regime_enc, label, max_fwd, mae)
  Output: final_prob, action, sizing_multiplier

Usage:
    python train_stacking_advanced.py
    → Loads lom_model_v3.pkl / nom_model_v3.pkl (or v1 fallback)
    → Trains meta-layers
    → Saves stacking_advanced_v1.pkl

Inference (from aladin_cal.py or live):
    from train_stacking_advanced import StackingPipeline
    stack = StackingPipeline.load('stacking_advanced_v1.pkl')
    result = stack.predict(lom_prob=0.72, nom_prob=0.68, regime_enc=2,
                           recent_wr=[1,0,1,1,0,0,1,0,1,1])
    → {action: 'standard', sizing: 1.0, final_prob: 0.71, uncertainty: 0.08}
"""

import pickle, logging, warnings, json, importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("STACKING_ADV")

BASE_DIR = Path(__file__).parent
# v4 models preferred (lower overfit, better OOS); fallback to v3 then v1
LOM_PKL  = next((BASE_DIR / f"lom_model_{v}.pkl" for v in ("v4", "v3", "v1")
                 if (BASE_DIR / f"lom_model_{v}.pkl").exists()), BASE_DIR / "lom_model_v1.pkl")
NOM_PKL  = next((BASE_DIR / f"nom_model_{v}.pkl" for v in ("v4", "v3", "v1")
                 if (BASE_DIR / f"nom_model_{v}.pkl").exists()), BASE_DIR / "nom_model_v1.pkl")
OUT_PKL  = BASE_DIR / "stacking_advanced_v1.pkl"

# Fallbacks if v4/v3 not available
LOM_FALLBACK = BASE_DIR / "lom_model_v1.pkl"
NOM_FALLBACK = BASE_DIR / "nom_model_v1.pkl"

log.info(f"Stacking base models: LOM={LOM_PKL.name}, NOM={NOM_PKL.name}")

N_MC_SAMPLES    = 30     # MC bootstrap samples for uncertainty
ATT_WINDOW      = 25     # temporal attention window (recent trades)
RL_EPOCHS       = 200    # REINFORCE training epochs
RL_LR           = 0.01   # policy gradient learning rate
RL_GAMMA        = 0.95   # reward discount
RL_ENTROPY_COEF = 0.01   # entropy regularization


# ══════════════════════════════════════════════════════════════════════════════
# 1. Uncertainty Wrapper (MC Bootstrap + Conformal)
# ══════════════════════════════════════════════════════════════════════════════

class UncertaintyWrapper:
    """
    Wraps a fitted XGBoost model and produces uncertainty-quantified predictions.

    Two approaches:
    a) MC Bootstrap: retrain N lightweight models on bootstrap subsamples
       of training data → distribution of predictions → CI
    b) Conformal: use calibration set residuals to build valid coverage intervals

    We use (a) for training-time CI and (b) for inference-time CI.
    """

    def __init__(self, n_samples=N_MC_SAMPLES):
        self.n_samples = n_samples
        self.conformal_quantiles = None   # fitted from calibration set
        self.base_model = None

    def fit_conformal(self, model, X_cal, y_cal):
        """
        Fit conformal prediction using calibration set.
        Nonconformity score = |y - p_hat| for regression-like bound.
        """
        self.base_model = model
        p_cal = model.predict_proba(X_cal)[:, 1]
        # Nonconformity scores
        scores = np.abs(p_cal - y_cal.astype(float))
        # Target coverage levels
        self.conformal_quantiles = {
            0.90: float(np.quantile(scores, 0.90)),
            0.80: float(np.quantile(scores, 0.80)),
            0.95: float(np.quantile(scores, 0.95)),
        }
        log.info(f"  Conformal quantiles: {self.conformal_quantiles}")

    def predict_with_uncertainty(self, model, X):
        """
        Returns (mean_prob, ci_lower, ci_upper, uncertainty) arrays.
        Uses conformal bounds if fitted, else simple Laplace smoothing.
        """
        p = model.predict_proba(X)[:, 1]
        if self.conformal_quantiles:
            half_width = self.conformal_quantiles.get(0.80, 0.1)
            ci_lower = np.clip(p - half_width, 0, 1)
            ci_upper = np.clip(p + half_width, 0, 1)
        else:
            # Laplace bound: wider for extreme probs (less confident at edges)
            uncertainty_base = 2 * p * (1 - p)  # 0 at edges, 0.5 at 0.5
            half_width = 0.05 + 0.15 * uncertainty_base
            ci_lower = np.clip(p - half_width, 0, 1)
            ci_upper = np.clip(p + half_width, 0, 1)

        uncertainty = ci_upper - ci_lower
        return p, ci_lower, ci_upper, uncertainty

    def mc_predict(self, model, X, X_train, y_train, n_samples=None):
        """
        MC Bootstrap: resample training data N times, fit lightweight models,
        collect prediction variance as proxy for epistemic uncertainty.
        Uses a very fast shallow XGB for speed.
        """
        import xgboost as xgb
        n = n_samples or self.n_samples
        preds = []
        n_tr = len(X_train)
        for i in range(n):
            idx = np.random.choice(n_tr, size=int(n_tr * 0.8), replace=True)
            Xs = X_train[idx]; ys = y_train[idx]
            if ys.sum() == 0 or ys.sum() == len(ys): continue
            m = xgb.XGBClassifier(
                n_estimators=50, max_depth=2, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.5,
                scale_pos_weight=(ys==0).sum()/max((ys==1).sum(),1),
                eval_metric='logloss', random_state=i, n_jobs=1, verbosity=0,
                tree_method='hist',
            )
            m.fit(Xs, ys, verbose=False)
            preds.append(m.predict_proba(X)[:, 1])

        if not preds:
            p = model.predict_proba(X)[:, 1]
            return p, p*0.9, p*1.1, np.ones(len(p))*0.1

        preds = np.array(preds)  # (n_samples, n_test)
        mean_pred = preds.mean(axis=0)
        ci_lower  = np.percentile(preds, 10, axis=0)
        ci_upper  = np.percentile(preds, 90, axis=0)
        uncertainty = ci_upper - ci_lower
        return mean_pred, ci_lower, ci_upper, uncertainty


# ══════════════════════════════════════════════════════════════════════════════
# 2. Temporal Attention Meta-Learner
# ══════════════════════════════════════════════════════════════════════════════

class TemporalAttentionMeta:
    """
    Learns dynamic model weights based on recent N-trade performance.

    At each trade t, we observe the performance of LOM/NOM/regime in the
    last ATT_WINDOW trades and reweight them:

    weight_lom[t]   = softmax(learned_decay * recent_lom_accuracy)
    weight_nom[t]   = softmax(learned_decay * recent_nom_accuracy)
    weight_regime[t]= softmax(learned_decay * recent_regime_accuracy)

    Final: attention_prob = weighted sum of lom_prob, nom_prob, base_prob

    Trained by minimizing log-loss of attention_prob against y.
    """

    def __init__(self, window=ATT_WINDOW):
        self.window = window
        # Learned attention weights [w_lom, w_nom, w_base, w_recent_wr]
        self.weights = np.array([0.35, 0.35, 0.20, 0.10])
        self.decay   = 0.85    # exponential decay for recency weighting
        self.scaler  = StandardScaler()
        self.is_fitted = False

    def _build_attention_features(self, probs_lom, probs_nom, regime_enc,
                                   y_true=None, recent_wr_window=None):
        """
        Build attention feature matrix: (n_trades, 4)
        Features: [lom_prob, nom_prob, base_prob, rolling_wr_context]
        """
        n = len(probs_lom)
        base_prob = (probs_lom + probs_nom) / 2

        # Rolling accuracy: how well did each model predict the last W trades?
        if y_true is not None:
            lom_acc_roll  = pd.Series((probs_lom > 0.5).astype(int) == y_true).rolling(self.window, min_periods=1).mean().fillna(0.5).values
            nom_acc_roll  = pd.Series((probs_nom > 0.5).astype(int) == y_true).rolling(self.window, min_periods=1).mean().fillna(0.5).values
            recent_wr     = pd.Series(y_true.astype(float)).rolling(self.window, min_periods=1).mean().fillna(0.5).values
        else:
            lom_acc_roll  = np.ones(n) * 0.5
            nom_acc_roll  = np.ones(n) * 0.5
            recent_wr     = np.ones(n) * 0.5 if recent_wr_window is None else np.array(recent_wr_window)

        # Attention context features
        X = np.column_stack([
            probs_lom, probs_nom, base_prob,
            regime_enc / 4.0,          # normalize regime to 0-1
            lom_acc_roll, nom_acc_roll,
            recent_wr,
            np.abs(probs_lom - probs_nom),   # model disagreement
            np.maximum(probs_lom, probs_nom), # best model
        ])
        return X

    def fit(self, probs_lom, probs_nom, regime_enc, y_true):
        """
        Train attention weights to minimize log-loss of combined prediction.
        Uses logistic regression as the mixing layer.
        """
        log.info(f"  [TemporalAtt] Fitting on {len(y_true)} trades ...")
        X = self._build_attention_features(probs_lom, probs_nom, regime_enc, y_true)
        X_sc = self.scaler.fit_transform(X)

        # Train logistic regression meta-learner
        self.meta_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        self.meta_lr.fit(X_sc, y_true)
        preds = self.meta_lr.predict_proba(X_sc)[:, 1]
        train_auc = roc_auc_score(y_true, preds)
        log.info(f"  [TemporalAtt] Train AUC: {train_auc:.4f}")
        self.is_fitted = True

    def predict(self, probs_lom, probs_nom, regime_enc, recent_wr_window=None):
        """Returns attention-weighted probability for each trade."""
        if not self.is_fitted:
            return (probs_lom + probs_nom) / 2
        X = self._build_attention_features(
            np.atleast_1d(probs_lom), np.atleast_1d(probs_nom),
            np.atleast_1d(regime_enc), recent_wr_window=recent_wr_window)
        X_sc = self.scaler.transform(X)
        return self.meta_lr.predict_proba(X_sc)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# 3. RL Meta-Policy (REINFORCE with numpy MLP)
# ══════════════════════════════════════════════════════════════════════════════

class RLMetaPolicy:
    """
    Lightweight REINFORCE policy for trade sizing decisions.

    State:  [lom_prob, nom_prob, regime_norm, uncertainty, recent_wr_5, recent_wr_10] = 6 dims
    Actions: 0=skip, 1=standard, 2=oversize

    Architecture: 2-layer MLP
      Input(6) → Hidden(16, tanh) → Hidden(8, tanh) → Output(3, softmax)

    Training: REINFORCE with entropy regularization
    Reward:
      skip:     0.0 (neutral)
      standard: label * 1.0 - (1-label) * 1.0  (win=+1, loss=-1)
      oversize: label * 2.0 - (1-label) * 2.0  (win=+2, loss=-2, higher variance)

    Risk adjustment: lower reward for high uncertainty
    """

    def __init__(self, state_dim=6, n_actions=3, hidden1=16, hidden2=8):
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Initialize MLP weights (Xavier)
        scale1 = np.sqrt(2.0 / (state_dim + hidden1))
        scale2 = np.sqrt(2.0 / (hidden1 + hidden2))
        scale3 = np.sqrt(2.0 / (hidden2 + n_actions))

        self.W1 = np.random.randn(state_dim, hidden1) * scale1
        self.b1 = np.zeros(hidden1)
        self.W2 = np.random.randn(hidden1, hidden2) * scale2
        self.b2 = np.zeros(hidden2)
        self.W3 = np.random.randn(hidden2, n_actions) * scale3
        self.b3 = np.zeros(n_actions)
        self.is_trained = False

    def _forward(self, x):
        """Forward pass: x (n, state_dim) → proba (n, n_actions)"""
        h1 = np.tanh(x @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        # Softmax
        logits -= logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(logits)
        proba = exp_l / exp_l.sum(axis=-1, keepdims=True)
        return proba, (h1, h2, logits)

    def _compute_reward(self, action, label, prob, uncertainty):
        """Compute risk-adjusted reward for a given action."""
        win = int(label)
        unc_penalty = uncertainty * 0.5   # penalize high uncertainty

        if action == 0:   # skip
            return 0.0
        elif action == 1: # standard
            base_reward = 1.0 if win else -1.0
        else:             # oversize
            base_reward = 2.0 if win else -2.0

        # Risk adjustment: scale down reward for uncertain predictions
        confidence = max(prob - 0.5, 0) * 2   # 0 at prob=0.5, 1 at prob=1.0
        return base_reward * (1.0 - unc_penalty) * (0.5 + 0.5 * confidence)

    def _reinforce_update(self, states, actions, rewards, proba_cache, lr):
        """One REINFORCE update step using policy gradient."""
        n = len(states)
        if n == 0: return

        # Compute discounted returns
        returns = np.zeros(n)
        G = 0.0
        for t in range(n-1, -1, -1):
            G = rewards[t] + RL_GAMMA * G
            returns[t] = G

        # Normalize returns
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Accumulate gradients
        dW1 = np.zeros_like(self.W1); db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2); db2 = np.zeros_like(self.b2)
        dW3 = np.zeros_like(self.W3); db3 = np.zeros_like(self.b3)

        for t in range(n):
            x = states[t:t+1]
            a = actions[t]
            G_t = returns[t]
            proba, (h1, h2, logits) = self._forward(x)
            p = proba[0]

            # Policy gradient: d/dθ log π(a|s) * G_t
            # d logπ(a|s) = one_hot(a) - π(a|s)
            grad_logits = np.zeros(self.n_actions)
            grad_logits[a] = 1.0
            grad_logits -= p
            # Entropy bonus
            entropy_grad = -(np.log(p + 1e-8) + 1)
            grad_logits += RL_ENTROPY_COEF * entropy_grad

            grad_out = grad_logits * G_t  # (n_actions,)

            # Backprop through W3
            dW3 += h2[0, :, np.newaxis] * grad_out[np.newaxis, :]
            db3 += grad_out

            # Through W2
            dh2 = grad_out @ self.W3.T * (1 - h2[0] ** 2)  # tanh deriv
            dW2 += h1[0, :, np.newaxis] * dh2[np.newaxis, :]
            db2 += dh2

            # Through W1
            dh1 = dh2 @ self.W2.T * (1 - h1[0] ** 2)
            dW1 += x[0, :, np.newaxis] * dh1[np.newaxis, :]
            db1 += dh1

        # Apply gradients
        self.W1 += lr * dW1 / n; self.b1 += lr * db1 / n
        self.W2 += lr * dW2 / n; self.b2 += lr * db2 / n
        self.W3 += lr * dW3 / n; self.b3 += lr * db3 / n

    def train(self, states, labels, probs, uncertainties, epochs=RL_EPOCHS, lr=RL_LR):
        """Train the RL policy on historical trades."""
        log.info(f"  [RLPolicy] Training {epochs} epochs on {len(states)} trades ...")
        n = len(states)
        best_reward = -np.inf
        best_weights = None

        for epoch in range(epochs):
            # Shuffle for stochastic training
            idx = np.random.permutation(n)
            s = states[idx]; l = labels[idx]; p = probs[idx]; u = uncertainties[idx]

            # Sample actions from current policy
            proba_arr, _ = self._forward(s)
            actions = np.array([np.random.choice(self.n_actions, p=proba_arr[i])
                                 for i in range(n)])

            # Compute rewards
            rewards = np.array([self._compute_reward(actions[i], l[i], p[i], u[i])
                                 for i in range(n)])

            # REINFORCE update
            self._reinforce_update(s, actions, rewards, proba_arr, lr)

            if epoch % 20 == 0:
                total_reward = rewards.sum()
                skip_rate = (actions == 0).mean()
                oversize_rate = (actions == 2).mean()
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_weights = (self.W1.copy(), self.b1.copy(),
                                    self.W2.copy(), self.b2.copy(),
                                    self.W3.copy(), self.b3.copy())
                if epoch % 40 == 0:
                    log.info(f"    epoch={epoch:3d}  reward={total_reward:.2f}  "
                             f"skip={skip_rate:.1%}  oversize={oversize_rate:.1%}")

        # Restore best
        if best_weights:
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = best_weights
        self.is_trained = True
        log.info(f"  [RLPolicy] Training done. Best reward: {best_reward:.2f}")

    def predict(self, state):
        """Returns (action_proba[3], recommended_action)."""
        s = np.atleast_2d(state)
        proba, _ = self._forward(s)
        action = int(np.argmax(proba[0]))
        return proba[0], action

    def predict_batch(self, states):
        """Returns (proba_array (n,3), actions (n,))."""
        proba, _ = self._forward(np.atleast_2d(states))
        actions = proba.argmax(axis=1)
        return proba, actions


# ══════════════════════════════════════════════════════════════════════════════
# Combined Stacking Pipeline
# ══════════════════════════════════════════════════════════════════════════════

ACTION_NAMES   = {0: 'skip', 1: 'standard', 2: 'oversize'}
SIZING_MULT    = {0: 0.0,    1: 1.0,        2: 1.5}

class StackingPipeline:
    """
    Full stacking pipeline: UncertaintyWrapper + TemporalAttentionMeta + RLMetaPolicy.

    Input (per trade):
        lom_prob:    float       — LOM model win probability
        nom_prob:    float       — NOM model win probability
        regime_enc:  int 0-4     — current regime
        recent_wr:   list        — recent win/loss history [1,0,1,1,...]

    Output:
        final_prob:      float       — combined probability
        action:          str         — 'skip'/'standard'/'oversize'
        sizing:          float       — position size multiplier (0, 1.0, 1.5)
        uncertainty:     float       — CI width (0-1)
        attention_prob:  float       — temporal attention prediction
        rl_action_proba: array(3)    — RL policy distribution
    """

    def __init__(self):
        self.uncertainty = UncertaintyWrapper()
        self.attention   = TemporalAttentionMeta()
        self.rl_policy   = RLMetaPolicy()
        self.state_scaler = StandardScaler()
        self.is_fitted   = False
        self.version     = 'v1_uncertainty_attention_rl'
        self.metadata    = {}

    def _build_rl_states(self, lom_probs, nom_probs, regime_encs, uncertainties, y=None):
        """Build state matrix for RL policy."""
        n = len(lom_probs)
        if y is not None:
            wr_5  = pd.Series(y.astype(float)).rolling(5,  min_periods=1).mean().fillna(0.5).values
            wr_10 = pd.Series(y.astype(float)).rolling(10, min_periods=1).mean().fillna(0.5).values
        else:
            wr_5  = np.ones(n) * 0.5
            wr_10 = np.ones(n) * 0.5

        states = np.column_stack([
            lom_probs,
            nom_probs,
            regime_encs / 4.0,    # normalize to 0-1
            uncertainties,
            wr_5,
            wr_10,
        ])
        return states

    def fit(self, df_trades):
        """
        Train all meta-layers on historical trade data.

        df_trades must have columns:
            lom_prob, nom_prob, regime_enc, label (_label),
            max_fwd (_max_fwd), mae (_mae_raw or mae)
        """
        log.info("=" * 65)
        log.info("  ADVANCED STACKING — Training")
        log.info("=" * 65)
        n = len(df_trades)
        log.info(f"  Trades: {n}")

        lom_probs  = df_trades['lom_prob'].values.astype(float)
        nom_probs  = df_trades['nom_prob'].values.astype(float)
        regime_enc = df_trades['regime_enc'].values.astype(float)
        y          = df_trades['label'].values.astype(int)

        # 1. Uncertainty (conformal on last 20% as calibration set)
        log.info("\n[1] Uncertainty Wrapper (conformal) ...")
        cal_n = max(int(n * 0.20), 30)
        # Use LOM prob directly as base; conformal on combined
        base_prob = (lom_probs + nom_probs) / 2
        cal_idx = np.arange(n - cal_n, n)
        tr_idx  = np.arange(0, n - cal_n)

        # Fit conformal using calibration set residuals
        cal_probs = base_prob[cal_idx]
        cal_y     = y[cal_idx]
        scores    = np.abs(cal_probs - cal_y.astype(float))
        self.uncertainty.conformal_quantiles = {
            0.80: float(np.quantile(scores, 0.80)),
            0.90: float(np.quantile(scores, 0.90)),
            0.95: float(np.quantile(scores, 0.95)),
        }
        log.info(f"  Conformal q80={self.uncertainty.conformal_quantiles[0.80]:.3f}")

        # Compute uncertainties for full set
        half_width = self.uncertainty.conformal_quantiles[0.80]
        uncertainties = np.clip(2 * half_width * np.ones(n), 0.01, 0.50)
        # Adjust: wider CI for probs close to 0.5, narrower for extremes
        uncertainty_adj = uncertainties * (1.0 + np.abs(base_prob - 0.5) * (-0.5))
        uncertainties = np.clip(uncertainty_adj, 0.01, 0.50)

        # 2. Temporal Attention
        log.info("\n[2] Temporal Attention Meta-Learner ...")
        self.attention.fit(lom_probs, nom_probs, regime_enc, y)
        att_probs = self.attention.predict(lom_probs, nom_probs, regime_enc)

        # 3. RL Policy
        log.info("\n[3] RL Meta-Policy (REINFORCE) ...")
        states = self._build_rl_states(lom_probs, nom_probs, regime_enc, uncertainties, y)
        states_sc = self.state_scaler.fit_transform(states)
        self.rl_policy.train(states_sc, y, base_prob, uncertainties)

        # 4. Final calibration layer: Isotonic regression on combined signal
        log.info("\n[4] Final Isotonic Calibration ...")
        combined = 0.4 * lom_probs + 0.4 * nom_probs + 0.2 * att_probs
        rl_proba, rl_actions = self.rl_policy.predict_batch(states_sc)
        # Add RL weight: prob that action=2 (oversize) gets slightly higher prob
        rl_weight = rl_proba[:, 2] * 0.05   # +5% for oversize action
        combined_final = np.clip(combined + rl_weight, 0, 1)

        self.final_isotonic = IsotonicRegression(out_of_bounds='clip')
        self.final_isotonic.fit(combined_final, y)

        # Evaluation
        final_cal = self.final_isotonic.predict(combined_final)
        train_auc = roc_auc_score(y, final_cal)
        log.info(f"\n  Training AUC (combined): {train_auc:.4f}")

        # Store metadata
        self.metadata = {
            'n_trades': n,
            'train_auc': round(train_auc, 4),
            'conformal_q80': self.uncertainty.conformal_quantiles[0.80],
            'rl_action_dist': {
                'skip':     float(rl_proba[:,0].mean()),
                'standard': float(rl_proba[:,1].mean()),
                'oversize': float(rl_proba[:,2].mean()),
            },
        }
        log.info(f"  RL action distribution: {self.metadata['rl_action_dist']}")
        self.is_fitted = True

    def predict(self, lom_prob, nom_prob, regime_enc,
                recent_wr=None, uncertainty_override=None):
        """
        Single-trade prediction.
        recent_wr: list of recent win/loss labels (last 10 is enough)
        """
        if not self.is_fitted:
            p = (lom_prob + nom_prob) / 2
            return {'final_prob': p, 'action': 'standard', 'sizing': 1.0,
                    'uncertainty': 0.1, 'attention_prob': p, 'rl_action_proba': np.array([0.1, 0.8, 0.1])}

        # Compute uncertainty
        if uncertainty_override is not None:
            uncertainty = uncertainty_override
        else:
            half_width = self.uncertainty.conformal_quantiles.get(0.80, 0.1)
            uncertainty = 2 * half_width

        # Temporal attention
        wr_5  = float(np.mean(recent_wr[-5:])) if recent_wr and len(recent_wr) >= 5 else 0.5
        wr_10 = float(np.mean(recent_wr[-10:])) if recent_wr and len(recent_wr) >= 10 else 0.5
        att_prob = float(self.attention.predict(
            np.array([lom_prob]), np.array([nom_prob]), np.array([regime_enc]),
            recent_wr_window=[wr_5]
        )[0])

        # RL state
        state = np.array([[lom_prob, nom_prob, regime_enc/4.0, uncertainty, wr_5, wr_10]])
        state_sc = self.state_scaler.transform(state)
        rl_proba, rl_action = self.rl_policy.predict(state_sc[0])

        # Combined final prob
        combined = 0.4 * lom_prob + 0.4 * nom_prob + 0.2 * att_prob
        rl_weight = rl_proba[2] * 0.05
        combined_final = np.clip(combined + rl_weight, 0, 1)
        final_prob = float(self.final_isotonic.predict([combined_final])[0])

        action_name = ACTION_NAMES[rl_action]
        sizing = SIZING_MULT[rl_action]

        return {
            'final_prob':      final_prob,
            'action':          action_name,
            'sizing':          sizing,
            'uncertainty':     float(uncertainty),
            'attention_prob':  float(att_prob),
            'rl_action_proba': rl_proba,
            'lom_prob':        lom_prob,
            'nom_prob':        nom_prob,
        }

    def predict_batch(self, df):
        """Batch predict on DataFrame with lom_prob, nom_prob, regime_enc columns."""
        results = []
        recent_wr = []
        for _, row in df.iterrows():
            r = self.predict(
                float(row['lom_prob']), float(row['nom_prob']),
                float(row.get('regime_enc', 2)),
                recent_wr=recent_wr[-10:] if recent_wr else None,
            )
            results.append(r)
            if 'label' in row.index:
                recent_wr.append(int(row['label']))
        return pd.DataFrame(results)

    def save(self, path=OUT_PKL):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        log.info(f"✅ Saved StackingPipeline → {path}")

    @classmethod
    def load(cls, path=OUT_PKL):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Data preparation from PKL models
# ══════════════════════════════════════════════════════════════════════════════

def load_base_model_predictions(model_pkl, dataset_pkl_path=None, is_lom=True):
    """
    Load a LOM/NOM v3 PKL and generate predictions on its full training dataset.

    Strategy (in priority order):
    1. Call build_dataset() from the corresponding v3 training script — this gives
       the complete feature-rich DataFrame WITH _label, _date, etc.
    2. Fall back to loading the saved dataset PKLs and attaching labels from the
       backtest CSV by positional alignment (last resort, less reliable).

    Returns DataFrame with lom_prob/nom_prob, regime_enc, label, date columns.
    """
    if not model_pkl.exists():
        log.warning(f"  {model_pkl} not found"); return None
    with open(model_pkl, 'rb') as f:
        pkg = pickle.load(f)

    model    = pkg['model']
    features = pkg['features']
    prefix   = 'lom' if is_lom else 'nom'

    # ── Strategy 0: load pre-cached predictions parquet (fast path) ─────────────
    cache_path = BASE_DIR / "data" / f"{prefix}_preds_cache.parquet"
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if f"{prefix}_prob" in cached.columns and 'label' in cached.columns and len(cached) > 50:
                log.info(f"  [{prefix}] Loaded cached predictions: {len(cached)} rows")
                return cached
        except Exception as e:
            log.warning(f"  [{prefix}] Cache load failed: {e}")

    # ── Strategy 1: call build_dataset() from the v3 training script ──────────
    df = None
    try:
        import sys, importlib
        script_name = f'train_{prefix}_v3'
        spec = importlib.util.spec_from_file_location(
            script_name, BASE_DIR / f"{script_name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[script_name] = mod
        spec.loader.exec_module(mod)

        all_years = [2023, 2024, 2025, 2026]
        log.info(f"  [{prefix}] Calling build_dataset({all_years}) ...")
        df = mod.build_dataset(all_years)
        log.info(f"  [{prefix}] build_dataset → {len(df)} rows, "
                 f"cols_with_meta={len([c for c in df.columns if c.startswith('_')])}")
    except Exception as e:
        log.warning(f"  [{prefix}] build_dataset() failed: {e}")
        df = None

    # ── Strategy 2: PKL + backtest CSV positional alignment ───────────────────
    if df is None or '_label' not in df.columns:
        log.info(f"  [{prefix}] Falling back to PKL + backtest CSV alignment ...")
        try:
            # Load both train and test PKLs combined
            pkls = []
            for suffix in ['train', 'test']:
                p = BASE_DIR / f"{prefix}_dataset_{suffix}.pkl"
                if p.exists():
                    pkls.append(pd.read_pickle(p))
            if not pkls:
                log.warning(f"  [{prefix}] No dataset PKLs found"); return None
            df_feat = pd.concat(pkls, ignore_index=True)

            # Load backtest CSV labels for this session
            csv_path = BASE_DIR / "backtest" / "backtest_bridge_v3.csv"
            bt = pd.read_csv(csv_path)
            session = 'LON' if is_lom else 'NY'
            bt_sess = bt[bt['session'] == session].reset_index(drop=True)
            n = min(len(df_feat), len(bt_sess))
            log.info(f"  [{prefix}] PKL rows={len(df_feat)}, CSV rows={len(bt_sess)}, "
                     f"using min={n} (positional alignment)")
            df_feat = df_feat.iloc[:n].copy()
            df_feat['_label'] = bt_sess['label'].values[:n]
            df_feat['_date']  = bt_sess['date'].values[:n]
            df = df_feat
        except Exception as e2:
            log.warning(f"  [{prefix}] Fallback also failed: {e2}"); return None

    # ── Generate predictions ───────────────────────────────────────────────────
    if df is None or len(df) == 0:
        log.warning(f"  [{prefix}] No data available"); return None

    feat_cols = [c for c in features if c in df.columns]
    if len(feat_cols) < len(features) * 0.5:
        log.warning(f"  [{prefix}] Only {len(feat_cols)}/{len(features)} features found — "
                    f"predictions may be degraded")

    X = df[feat_cols].reindex(columns=features, fill_value=0).fillna(0)
    probs = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        f"{prefix}_prob":  probs,
        'regime_enc': df.get('regime_enc', pd.Series(2, index=df.index)).fillna(2).values,
        'label':      df['_label'].values,
        'max_fwd':    df.get('_max_fwd', pd.Series(0.0, index=df.index)).fillna(0).values,
        'mae':        df.get('_mae_raw', pd.Series(0.0, index=df.index)).fillna(0).values,
        'date':       df['_date'].values,
    })
    log.info(f"  [{prefix}] predictions: {len(result)} trades, "
             f"mean_prob={probs.mean():.3f}, WR={result['label'].mean():.1%}")
    return result


def prepare_training_data():
    """Combine LOM + NOM predictions for stacking training."""
    log.info("[prep] Loading LOM predictions ...")
    lom_preds = load_base_model_predictions(
        LOM_PKL if LOM_PKL.exists() else LOM_FALLBACK, is_lom=True)
    log.info("[prep] Loading NOM predictions ...")
    nom_preds = load_base_model_predictions(
        NOM_PKL if NOM_PKL.exists() else NOM_FALLBACK, is_lom=False)

    if lom_preds is None and nom_preds is None:
        log.error("No base model predictions available"); return None

    # Combine: use both if available, else just one
    if lom_preds is not None and nom_preds is not None:
        # Merge on date (different sessions so concat)
        lom_preds['nom_prob'] = lom_preds['lom_prob']  # cross-fill for attention
        nom_preds['lom_prob'] = nom_preds['nom_prob']
        df_all = pd.concat([lom_preds, nom_preds], ignore_index=True).sort_values('date')
    elif lom_preds is not None:
        lom_preds['nom_prob'] = lom_preds['lom_prob']
        df_all = lom_preds
    else:
        nom_preds['lom_prob'] = nom_preds['nom_prob']
        df_all = nom_preds

    df_all = df_all.dropna(subset=['lom_prob', 'nom_prob', 'label'])
    log.info(f"[prep] Combined: {len(df_all)} trades, WR={df_all['label'].mean():.1%}")
    return df_all


def train_stacking():
    log.info("=" * 65)
    log.info("  ADVANCED STACKING v1 — Training")
    log.info("=" * 65)

    df_all = prepare_training_data()
    if df_all is None or len(df_all) < 50:
        log.error("Insufficient data for stacking"); return

    pipeline = StackingPipeline()
    pipeline.fit(df_all)
    pipeline.save()

    # Quick evaluation
    results = pipeline.predict_batch(df_all)
    if 'final_prob' in results.columns and len(df_all) > 20:
        y_true = df_all['label'].values.astype(int)
        final_auc = roc_auc_score(y_true, results['final_prob'].values)
        log.info(f"\n  Final AUC (stacked): {final_auc:.4f}")

        skip_rate = (results['action'] == 'skip').mean()
        os_rate   = (results['action'] == 'oversize').mean()
        log.info(f"  Action distribution: skip={skip_rate:.1%}, oversize={os_rate:.1%}")

    log.info("\nDone.")


if __name__ == '__main__':
    np.random.seed(42)
    train_stacking()
