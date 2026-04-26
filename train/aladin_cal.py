"""Shared calibration class — importable by all training and inference scripts.

Defined here at module level so pickle can find it as ``aladin_cal._CalModel``
regardless of which script does the loading.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression as _IR


class _CalModel:
    """Manual isotonic calibration wrapper (replaces CalibratedClassifierCV cv='prefit',
    removed in sklearn 1.6)."""

    classes_ = np.array([0, 1])

    def __init__(self, m, ir):
        self._m, self._ir = m, ir

    def predict_proba(self, X):
        r = self._m.predict_proba(X)[:, 1]
        c = np.clip(self._ir.predict(r), 0, 1)
        return np.c_[1 - c, c]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
