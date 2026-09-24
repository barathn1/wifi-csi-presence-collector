"""Handcrafted per-subcarrier statistical features (mean/std/skew/kurtosis, amplitude+phase) for the
classical (SVM/GBM) models -- kept per-subcarrier throughout, never pooled/averaged across subcarriers,
per this project's core signal assumption (a given subcarrier reacts differently depending on who's
present).

Vectorized per-session (not a Python loop per window): every window in a session is gathered in one
fancy-indexing call and its moments computed as one batched numpy reduction, instead of iterating
window-by-window -- window-by-window was the actual bottleneck on a ~178k-window dataset (many minutes
-> a few seconds).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def feature_names(n_subcarriers: int) -> list[str]:
    names = []
    for channel in ("amp", "phase"):
        for stat in ("mean", "std", "skew", "kurt"):
            names += [f"{channel}_{stat}_sc{i}" for i in range(n_subcarriers)]
    names += ["rssi_mean", "rssi_std"]
    return names


def _batched_moments(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """windows: (n_windows, window_packets, n_subcarriers) -> 4x (n_windows, n_subcarriers)."""
    mean = windows.mean(axis=1)
    diff = windows - mean[:, None, :]
    std = np.sqrt((diff * diff).mean(axis=1))
    skew = (diff ** 3).mean(axis=1) / (std ** 3 + EPS)
    kurt = (diff ** 4).mean(axis=1) / (std ** 4 + EPS) - 3.0
    return mean, std, skew, kurt


def _session_feature_block(amplitude: np.ndarray, phase: np.ndarray, rssi: np.ndarray,
                            starts: np.ndarray, window_packets: int) -> np.ndarray:
    idx = starts[:, None] + np.arange(window_packets)[None, :]  # (n_windows, window_packets)
    amp_windows = amplitude[idx]     # (n_windows, window_packets, n_subcarriers)
    phase_windows = phase[idx]
    rssi_windows = rssi[idx]         # (n_windows, window_packets)

    parts = []
    for windows in (amp_windows, phase_windows):
        parts.extend(_batched_moments(windows))
    parts.append(rssi_windows.mean(axis=1, keepdims=True))
    parts.append(rssi_windows.std(axis=1, keepdims=True))
    features = np.concatenate(parts, axis=1).astype(np.float32)  # (n_windows, feature_dim)
    return np.nan_to_num(features, nan=0.0)


def build_feature_matrix(window_index: pd.DataFrame, calibration=None) -> np.ndarray:
    n = len(window_index)
    out: np.ndarray | None = None
    for cache_path, group in window_index.groupby("cache_path", sort=False):
        with np.load(cache_path, mmap_mode="r") as d:
            amplitude, phase, rssi = np.array(d["amplitude"]), np.array(d["phase"]), np.array(d["rssi"])
        window_packets = int(group["end"].iloc[0] - group["start"].iloc[0])
        starts = group["start"].values.astype(np.int64)

        if calibration is not None:
            # calibration is per-(date, receiver_mac), constant within one cache_path's group -- apply
            # once to the whole session array rather than per-window, since it's the same transform.
            amplitude, phase = calibration(amplitude, phase, group.iloc[0])

        block = _session_feature_block(amplitude, phase, rssi, starts, window_packets)
        if out is None:
            out = np.empty((n, block.shape[1]), dtype=np.float32)
        out[group.index.values] = block
    return out
