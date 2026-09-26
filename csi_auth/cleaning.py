"""Outlier/spike removal on decoded CSI amplitude, applied per session right after decode.py, before
windowing/features.

Method: a Hampel filter, per subcarrier, over the packet-index axis -- a rolling median absolute
deviation (MAD) spike detector, standard for sensor time series where isolated corrupted/glitched
samples need removing without smoothing the real signal the way a low-pass filter would. A sample is
flagged as a spike if it deviates from its local rolling median by more than `n_sigmas` robust standard
deviations (1.4826 * MAD is the normal-distribution-consistent scale estimate for MAD); spikes are
replaced with that local median rather than dropped, so every session keeps its original packet count
and windowing/timing logic downstream needs no changes.

Only applied to amplitude (not phase): phase is circular (wraps at +-pi), and a naive median/MAD test
misfires right at the wrap boundary -- fixing that properly is a separate piece of work, out of scope
here since amplitude is the primary channel this pipeline (and most CSI-identification literature)
relies on.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

DEFAULT_WINDOW = 11   # packets; must be odd for scipy.medfilt
DEFAULT_N_SIGMAS = 4.0


def hampel_filter_amplitude(amplitude: np.ndarray, window: int = DEFAULT_WINDOW,
                             n_sigmas: float = DEFAULT_N_SIGMAS) -> tuple[np.ndarray, float]:
    """amplitude: (n_packets, n_subcarriers). Returns (cleaned_amplitude, fraction_flagged_as_spikes)."""
    if window % 2 == 0:
        window += 1
    if amplitude.shape[0] < window:
        return amplitude.copy(), 0.0

    cleaned = amplitude.copy()
    n_flagged = 0
    for col in range(amplitude.shape[1]):
        series = amplitude[:, col]
        med = medfilt(series, kernel_size=window)
        mad = medfilt(np.abs(series - med), kernel_size=window)
        threshold = np.maximum(n_sigmas * 1.4826 * mad, 1e-6)  # floor avoids flagging a flat-zero column
        spike = np.abs(series - med) > threshold
        cleaned[spike, col] = med[spike]
        n_flagged += int(spike.sum())

    frac_flagged = n_flagged / amplitude.size
    return cleaned, frac_flagged
