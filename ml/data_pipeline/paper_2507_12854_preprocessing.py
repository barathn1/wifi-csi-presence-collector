"""Preprocessing pipeline replicating arXiv:2507.12854 (Avola et al., "Transformer-Based Person
Identification via Wi-Fi CSI Amplitude and Phase Perturbations", 99.82% on 6 subjects), reconstructed
from the paper's text (no source code available) via the following extracted steps, applied in order:

1. Temporal mean reduction: average every two consecutive time samples, halving temporal resolution.
2. Hampel filter: MAD-based sliding-window outlier removal (window=15, beta/n_sigmas=3), replacing
   flagged outliers with an exponential-smoothing estimate (alpha=0.8) rather than the local median.
3. Low-pass Butterworth filter: 5th order, cutoff=10 Hz (applied at the post-reduction rate).
4. Phase calibration: per-packet linear-trend removal across the subcarrier axis, using only the two
   endpoint subcarriers' phase to estimate slope/intercept -- removes the CFO/SFO-induced linear ramp.

Interpretive choices NOT specified in the extracted text (the paper's own source isn't available to
check against):
- Steps 1-3 are applied to amplitude directly, and to phase UNWRAPPED ALONG THE TIME AXIS first (per
  subcarrier) -- plain averaging/filtering of wrapped phase across time would create artifacts at
  every +-pi crossing.
- Step 4 (endpoint linear-fit) unwraps along the SUBCARRIER axis per packet (a different axis, for a
  different purpose: removing a per-packet linear ramp, not smoothing over time).
- The Hampel replacement's "exponential smoothing over previous values" is implemented as a running
  EWMA that is itself updated only with clean (non-outlier) values, so outliers can't contaminate it.

ONLY the paper's model architecture (BranchEncoder / DualBranchTransformer in models/) was already an
exact match in this repo before this file existed -- this module is the missing preprocessing half.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def temporal_mean_reduce(x: np.ndarray) -> np.ndarray:
    """Average consecutive time-sample pairs along axis 0, halving temporal resolution (paper Eq. 5)."""
    n = x.shape[0] - (x.shape[0] % 2)
    return (x[:n:2] + x[1:n:2]) / 2.0


def hampel_filter(x: np.ndarray, window: int = 15, n_sigmas: float = 3.0, alpha: float = 0.8) -> np.ndarray:
    """Per-subcarrier Hampel outlier removal along axis 0 (time), MAD-based sliding window, replacing
    flagged points with a clean-values-only EWMA (paper Eq. 6-8)."""
    x = x.copy()
    n_time = x.shape[0]
    half = window // 2
    ewma = x[0].copy()
    for t in range(n_time):
        lo, hi = max(0, t - half), min(n_time, t + half + 1)
        segment = x[lo:hi]
        med = np.median(segment, axis=0)
        mad = np.median(np.abs(segment - med), axis=0) * 1.4826  # MAD -> Gaussian-equivalent sigma
        is_outlier = np.abs(x[t] - med) > n_sigmas * mad
        clean_val = np.where(is_outlier, ewma, x[t])
        ewma = alpha * ewma + (1 - alpha) * clean_val
        x[t] = np.where(is_outlier, ewma, x[t])
    return x


def butterworth_lowpass(x: np.ndarray, fs: float, cutoff: float = 10.0, order: int = 5) -> np.ndarray:
    """5th-order low-pass Butterworth, filtfilt (zero-phase) along axis 0 (time)."""
    nyq = fs / 2.0
    b, a = butter(order, min(cutoff / nyq, 0.99), btype="low")
    return filtfilt(b, a, x, axis=0)


def calibrate_phase_endpoints(phase: np.ndarray) -> np.ndarray:
    """Per-packet: remove the linear CFO/SFO trend estimated from the two endpoint subcarriers' phase
    (paper Eq. 9-11). Unwraps along the subcarrier axis first (axis=1) since raw phase is wrapped."""
    n_sub = phase.shape[1]
    unwrapped = np.unwrap(phase, axis=1)
    idx = np.arange(n_sub)
    slope = (unwrapped[:, -1] - unwrapped[:, 0]) / (n_sub - 1)
    intercept = unwrapped[:, 0]
    trend = slope[:, None] * idx[None, :] + intercept[:, None]
    return unwrapped - trend


def preprocess_session(amplitude: np.ndarray, phase: np.ndarray, rate_hz: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Full pipeline for one session's (n_packets, n_subcarriers) amplitude/phase. Returns
    (amp_processed, phase_processed, effective_rate_hz)."""
    phase_unwrapped_time = np.unwrap(phase, axis=0)

    amp_reduced = temporal_mean_reduce(amplitude)
    phase_reduced = temporal_mean_reduce(phase_unwrapped_time)
    effective_rate = rate_hz / 2.0

    amp_hampel = hampel_filter(amp_reduced)
    phase_hampel = hampel_filter(phase_reduced)

    amp_smooth = butterworth_lowpass(amp_hampel, fs=effective_rate)
    phase_smooth = butterworth_lowpass(phase_hampel, fs=effective_rate)

    phase_calibrated = calibrate_phase_endpoints(phase_smooth)
    return amp_smooth.astype(np.float32), phase_calibrated.astype(np.float32), effective_rate


def window_1s_50pct(n_packets: int, rate_hz: float) -> tuple[int, list[int]]:
    """1-second windows, 50% overlap (paper: "100 packets (1 second)... 50% overlap"), scaled to
    THIS data's own effective rate rather than the paper's literal 100-packet figure (which is
    specific to their 100 Hz setup)."""
    win = max(1, round(rate_hz * 1.0))
    stride = max(1, win // 2)
    starts = list(range(0, n_packets - win + 1, stride))
    return win, starts
