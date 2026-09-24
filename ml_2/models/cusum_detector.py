"""CUSUM-style sequential change-point/anomaly detector -- reframes "is an authorized person here" as
"has the live stream changed relative to its OWN continuously-updated baseline" instead of a static
per-window classifier. The baseline adapts slowly (EMA) toward whatever it's currently seeing, so slow
legitimate drift (temperature, furniture, hardware aging) doesn't accumulate into false alarms the way
a fixed per-day baseline would -- this is the mechanism, not just a rebrand of the empty-room
calibration already used elsewhere.

Stateful and sequential BY DESIGN: `score_session` must be called with windows from one session in
their natural time order (not shuffled), since the running baseline/cumulative statistic only mean
anything as a function of what came immediately before.
"""
from __future__ import annotations

import numpy as np


class CusumDetector:
    def __init__(self, alpha: float = 0.02, k: float = 0.5, h: float = 8.0):
        self.alpha = alpha  # EMA rate the running baseline adapts toward newly observed windows
        self.k = k          # CUSUM slack/allowance -- deviations below this don't accumulate
        self.h = h          # decision threshold on the cumulative statistic
        self.mean_: np.ndarray | None = None
        self.var_: np.ndarray | None = None

    def fit(self, X_calib: np.ndarray) -> "CusumDetector":
        """`X_calib`: (n_windows, n_subcarriers) authorized-only per-subcarrier amplitude means, used
        only to set the INITIAL baseline -- the baseline then adapts online during scoring."""
        self.mean_ = X_calib.mean(axis=0)
        self.var_ = X_calib.var(axis=0) + 1e-6
        return self

    def score_session(self, X_session: np.ndarray) -> np.ndarray:
        """X_session: (n_windows_in_order, n_subcarriers) for ONE session, in time order. Returns a
        genuine-ness score per window (higher = more like the adapting baseline, i.e. more
        authorized-like) -- the CUSUM statistic itself, negated."""
        mean = self.mean_.copy()
        g_pos = 0.0
        scores = np.empty(len(X_session))
        for i, x in enumerate(X_session):
            z = (x - mean) / np.sqrt(self.var_)
            deviation = float(np.mean(z ** 2))
            g_pos = max(0.0, g_pos + deviation - self.k)
            scores[i] = -g_pos
            mean = (1 - self.alpha) * mean + self.alpha * x
            if g_pos > self.h:
                g_pos = 0.0  # standard CUSUM restart after a flagged change point
        return scores
