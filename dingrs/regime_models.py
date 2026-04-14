"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN — Regime-Specific Models                                            ║
║  regime_models.py  |  Update #52 — Model separat per regim de piață       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Detectează regimul (TRENDING/RANGING/VOLATILE) și aplică modelul corespunzător.
Antrenare separată per regim din train_mario_ai.py cu flag --regime.

Utilizare:
  from regime_models import RegimeRouter
  router = RegimeRouter()
  proba = router.predict_proba(X, regime="TRENDING UP")
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger("aladin-regime")

MODEL_DIR = "/Users/mario/Desktop"

REGIME_MODEL_MAP = {
    "TRENDING UP":   "mario_bot_regime_trend.pkl",
    "TRENDING DOWN": "mario_bot_regime_trend.pkl",   # același model trend, directional
    "RANGING":       "mario_bot_regime_range.pkl",
    "VOLATILE":      "mario_bot_regime_volatile.pkl",
    "UNKNOWN":       "mario_bot.json",  # fallback la XGBoost principal
}


class RegimeRouter:
    """
    Update #52: Router de modele bazat pe regim de piață.
    Menține un model per regim, cu fallback la XGBoost principal.
    """

    def __init__(self):
        self._models = {}
        self._load_models()

    def _load_models(self):
        """Încarcă modelele disponibile."""
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost nu este instalat")
            return

        for regime, filename in REGIME_MODEL_MAP.items():
            path = os.path.join(MODEL_DIR, filename)
            if not os.path.exists(path):
                continue
            try:
                if filename.endswith('.json'):
                    model = xgb.XGBClassifier()
                    model.load_model(path)
                    self._models[regime] = model
                elif filename.endswith('.pkl'):
                    with open(path, 'rb') as f:
                        self._models[regime] = pickle.load(f)
                logger.debug(f"✅ Regime model încărcat: {regime} → {filename}")
            except Exception as e:
                logger.debug(f"⚠️ Regime model {regime} nedisponibil: {e}")

    def predict_proba(self, X: pd.DataFrame, regime: str = "UNKNOWN") -> np.ndarray:
        """
        Returnează probabilitățile pentru regimul dat.
        Fallback la UNKNOWN (XGBoost principal) dacă modelul specific lipsește.
        """
        # Normalizăm regimul
        regime_key = regime.upper() if regime else "UNKNOWN"
        if "TRENDING" in regime_key:
            regime_key = "TRENDING UP" if "UP" in regime_key else "TRENDING DOWN"
        elif "RANGING" in regime_key or "RANGE" in regime_key:
            regime_key = "RANGING"
        elif "VOLATILE" in regime_key:
            regime_key = "VOLATILE"

        model = self._models.get(regime_key) or self._models.get("UNKNOWN")

        if model is None:
            logger.debug("Niciun model regime disponibil — returnez [0.5, 0.25, 0.25]")
            return np.array([[0.5, 0.25, 0.25]])

        try:
            proba = model.predict_proba(X)
            logger.debug(f"   Regime {regime_key}: proba={proba[-1]}")
            return proba
        except Exception as e:
            logger.warning(f"Regime predict error ({regime_key}): {e}")
            return np.array([[0.5, 0.25, 0.25]])

    def is_regime_model_available(self, regime: str) -> bool:
        """Verifică dacă există un model specific pentru regimul dat."""
        return regime in self._models and regime != "UNKNOWN"

    @property
    def available_regimes(self) -> list:
        return list(self._models.keys())


def train_regime_models():
    """
    Antrenează modele separate per regim din datele din mario_trading.db.
    Apelat din train_mario_ai.py cu: train_regime_models()
    """
    try:
        import sqlite3
        import xgboost as xgb
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return

    # Simulare import din train_mario_ai.py
    FEATURES_BASE = []
    FEATURES_EXTRA = []
    PATH_DB = "/Users/mario/Desktop/Aladin/mario_trading.db"

    def normalize_price_features(df):
        """Dummy normalization"""
        return df

    def generate_atr_target(df, horizon=60, atr_multiplier=2.0):
        """Dummy target generation"""
        return np.zeros(len(df), dtype=int)

    def add_sniper_conditions(df):
        """Dummy sniper conditions"""
        df['sweep_h'] = 0
        df['sweep_l'] = 0
        df['has_displacement'] = 0
        df['poc_level'] = df['close']
        return df

    print("\n🎯 ANTRENARE REGIME-SPECIFIC MODELS...")

    if not os.path.exists(PATH_DB):
        print(f"❌ DB lipsă: {PATH_DB}")
        return

    conn = sqlite3.connect(PATH_DB)
    try:
        df = pd.read_sql_query("SELECT * FROM market_data", conn)
    except Exception as e:
        print(f"❌ Eroare citire DB: {e}")
        conn.close()
        return
    finally:
        conn.close()

    if df.empty:
        print("❌ Nu sunt date în market_data")
        return

    # Calculează ATR pentru clasificare regim
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    slope_h4 = (df['close'] - df['close'].shift(240)) / (df['close'].shift(240).abs() + 1e-8)
    atr_pct_rank = atr.rolling(1000, min_periods=100).rank(pct=True)

    # Clasificare regim: TRENDING / RANGING / VOLATILE
    df['regime_label'] = 'RANGING'
    df.loc[abs(slope_h4) > 0.005, 'regime_label'] = 'TRENDING'
    df.loc[atr_pct_rank > 0.80, 'regime_label'] = 'VOLATILE'

    # Features
    all_feats = FEATURES_BASE + FEATURES_EXTRA
    df['slope_h1'] = (df['close'] - df['close'].shift(60)) / (df['close'].shift(60).abs() + 1e-8)
    df['slope_h4'] = slope_h4
    df['momentum_15'] = (df['close'] - df['close'].shift(15)) / (df['close'].shift(15).abs() + 1e-8)
    df['body_dir'] = np.sign(df['close'] - df['open'])
    df['wick_ratio'] = (df['high'] - df['low']) / (abs(df['close'] - df['open']) + 1e-8)

    df = add_sniper_conditions(df)
    df['target_atr'] = generate_atr_target(df, horizon=60, atr_multiplier=2.0)

    short_cond = (
        ((df['sweep_h'].rolling(10, min_periods=1).max() == 1) | (df['has_displacement'] == 1)) &
        (df['close'] < df['poc_level'].fillna(df['close'])) & (df['target_atr'] == 1)
    )
    long_cond = (
        ((df['sweep_l'].rolling(10, min_periods=1).max() == 1) | (df['has_displacement'] == 1)) &
        (df['close'] > df['poc_level'].fillna(df['close'])) & (df['target_atr'] == 2)
    )

    df['target'] = 0
    df.loc[short_cond, 'target'] = 1
    df.loc[long_cond, 'target'] = 2

    for regime_label, filename in [
        ('TRENDING', 'mario_bot_regime_trend.pkl'),
        ('RANGING', 'mario_bot_regime_range.pkl'),
        ('VOLATILE', 'mario_bot_regime_volatile.pkl'),
    ]:
        regime_df = df[df['regime_label'] == regime_label].copy()
        if len(regime_df) < 1000:
            print(f"   ⚠️ {regime_label}: prea puține date ({len(regime_df)}) — skip")
            continue

        available = [f for f in all_feats if f in regime_df.columns]
        if not available:
            available = ['slope_h1', 'slope_h4', 'momentum_15', 'body_dir', 'wick_ratio']

        X = normalize_price_features(regime_df[available].ffill().fillna(0))
        y = regime_df['target'].astype(int)

        split = int(len(X) * 0.8)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        try:
            model = xgb.XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.02,
                objective='multi:softprob', num_class=3,
                tree_method='hist', random_state=42, verbosity=0,
            )
            model.fit(X_tr, y_tr)

            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_te, np.argmax(model.predict_proba(X_te), axis=1))
            print(f"   ✅ {regime_label}: {len(regime_df):,} bare | acc={acc:.3f}")

            save_path = os.path.join(MODEL_DIR, filename)
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"      Salvat: {save_path}")
        except Exception as e:
            print(f"   ❌ {regime_label}: Eroare antrenare: {e}")


if __name__ == '__main__':
    train_regime_models()
