"""Best-effort reconstruction of Wi-Gait (Computer Networks 2023, doi:10.1016/j.comnet.2023.109751) --
NOT a verified replication. Both the ScienceDirect and ACM pages are paywalled with no accessible
preprint, so this is built from the concepts confirmed across abstracts/citations for Wi-Gait and the
closely related GaitSense/GaitID line of work, not the paper's actual formulas:

1. CSI amplitude sanitization (outlier removal) -- reuses the Hampel filter already built for the
   arXiv:2507.12854 replication (paper_2507_12854_preprocessing.hampel_filter).
2. Torso-vs-limb motion separation: gait research in this area consistently separates a low-frequency
   torso-dominant component from a higher-frequency limb-swing component (the "red curve = torso,
   white curves = residual limb energy" framing cited for Wi-Gait). Implemented here as two Butterworth
   bands: torso ~0-2 Hz, limb-swing ~2-8 Hz -- band edges are a reasonable guess for human gait cadence,
   not values taken from the paper (unavailable).
3. Gait-cycle segmentation: peaks in the torso-band signal (mean amplitude across subcarriers) mark
   cycle boundaries, matching the general "segment by gait cycle" idea common to this literature.
4. Each cycle is resampled to a fixed length (cycles have different durations at different walking
   paces) so cycles can be batched.

If the real paper's sanitization/segmentation/band choices differ from this, results here should be
read as "does the GENERAL gait-cycle idea transfer to this dataset", not "does Wi-Gait itself work".
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from ml.data_pipeline.paper_2507_12854_preprocessing import hampel_filter

TORSO_BAND = (0.1, 2.0)   # Hz
LIMB_BAND = (2.0, 8.0)    # Hz
CYCLE_LEN = 32            # resampled samples per gait cycle
MIN_CYCLE_S = 0.5         # floor on cycle length, human gait cadence


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    lo, hi = max(low / nyq, 1e-4), min(high / nyq, 0.99)
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, x, axis=0)


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, min(cutoff / nyq, 0.99), btype="low")
    return filtfilt(b, a, x, axis=0)


def resample_cycle(segment: np.ndarray, target_len: int = CYCLE_LEN) -> np.ndarray:
    """segment: (n_samples, n_sub) -> (target_len, n_sub), linear interpolation per subcarrier."""
    n = segment.shape[0]
    if n == target_len:
        return segment
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    return np.stack([np.interp(x_new, x_old, segment[:, s]) for s in range(segment.shape[1])], axis=1)


def extract_gait_cycles(amplitude: np.ndarray, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """amplitude: (n_packets, n_sub) raw decoded amplitude for one (walking) session.
    Returns (torso_cycles, limb_cycles), each (n_cycles, CYCLE_LEN, n_sub)."""
    sanitized = hampel_filter(amplitude)
    torso = lowpass(sanitized, rate_hz, TORSO_BAND[1])
    limb = bandpass(sanitized, rate_hz, *LIMB_BAND)

    torso_signal = torso.mean(axis=1)
    min_distance = max(1, int(MIN_CYCLE_S * rate_hz))
    peaks, _ = find_peaks(torso_signal, distance=min_distance, prominence=torso_signal.std() * 0.3)

    torso_cycles, limb_cycles = [], []
    for i in range(len(peaks) - 1):
        lo, hi = peaks[i], peaks[i + 1]
        if hi - lo < 3:
            continue
        torso_cycles.append(resample_cycle(torso[lo:hi]))
        limb_cycles.append(resample_cycle(limb[lo:hi]))

    if not torso_cycles:
        return np.empty((0, CYCLE_LEN, amplitude.shape[1])), np.empty((0, CYCLE_LEN, amplitude.shape[1]))
    return np.stack(torso_cycles).astype(np.float32), np.stack(limb_cycles).astype(np.float32)
