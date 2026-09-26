"""arXiv:2507.12854's CSI noise-reduction recipe: a windowed Hampel filter for outlier removal,
followed by a Butterworth low-pass filter for smoothing.

Restored: the original module backing every `from ml.data_pipeline.paper_2507_12854_preprocessing
import ...` call site in this repo (bilstm_ch6_pipeline.py, ablation_preprocessing.py, walk_bilstm_
pipeline.py, walk_bilstm_sept21_pipeline.py) is missing from disk -- rebuilt from those call sites'
own documented parameters (Hampel window=15 packets / n_sigmas=3 MAD units; Butterworth order=5 /
cutoff=10Hz low-pass), not reverse-engineered guesswork.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


def hampel_filter(x: np.ndarray, window: int = 15, n_sigmas: float = 3.0) -> np.ndarray:
    """Per-column (subcarrier) windowed Hampel filter along axis 0 (time): flags points more than
    `n_sigmas` scaled median-absolute-deviations from their local rolling median and replaces them
    with that median. Every packet is kept -- output has the same shape/length as `x`, never drops a
    row. Vectorized via pandas rolling (center=True, min_periods=1) rather than a per-timestep Python
    loop -- see `ablation_preprocessing.remove_outliers_moving_iqr`'s docstring for the measured cost
    of the naive version at this project's data volume (tens of thousands of packets per session)."""
    df = pd.DataFrame(x)
    roll = df.rolling(window=window, center=True, min_periods=1)
    med = roll.median()
    mad = (df - med).abs().rolling(window=window, center=True, min_periods=1).median()
    threshold = n_sigmas * 1.4826 * mad
    is_outlier = (df - med).abs() > threshold
    return df.where(~is_outlier, med).to_numpy(dtype=x.dtype)


def butterworth_lowpass(x: np.ndarray, fs: float, cutoff: float = 10.0, order: int = 5) -> np.ndarray:
    """Zero-phase low-pass Butterworth filter along axis 0 (time), applied independently per column
    (subcarrier). `fs`: the signal's own native packet rate (Hz) -- never a fixed constant, since it
    varies session to session (see time_resample.py)."""
    nyquist = fs / 2.0
    normal_cutoff = min(cutoff / nyquist, 0.99)
    sos = butter(order, normal_cutoff, btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=0).astype(x.dtype)
