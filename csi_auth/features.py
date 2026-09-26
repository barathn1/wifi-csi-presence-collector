"""Handcrafted per-window statistics -- input to the classical (SVM/RandomForest) branch.

Per subcarrier, per channel (amplitude, phase): mean, std, skew, excess-kurtosis over the window's time
axis, plus RSSI mean/std. A compact statistical summary, independent implementation of the same general
idea used elsewhere in CSI-identification literature (e.g. the "statgram" concept).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def _moments(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = arr.mean(axis=0)
    diff = arr - mean
    std = np.sqrt((diff * diff).mean(axis=0))
    skew = (diff ** 3).mean(axis=0) / (std ** 3 + EPS)
    kurt = (diff ** 4).mean(axis=0) / (std ** 4 + EPS) - 3.0
    return mean, std, skew, kurt


def window_features(amplitude: np.ndarray, phase: np.ndarray, rssi: np.ndarray) -> np.ndarray:
    """amplitude/phase: (window_packets, n_subcarriers), rssi: (window_packets,) -> 1D feature vector."""
    parts = []
    for arr in (amplitude, phase):
        parts.extend(_moments(arr))
    parts.append(np.array([rssi.mean(), rssi.std()]))
    feats = np.concatenate(parts).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0)
