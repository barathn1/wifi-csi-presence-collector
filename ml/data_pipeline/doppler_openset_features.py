"""Frequency-domain (Doppler-like) alternative to the time-domain amplitude features used everywhere
else in the walk_bilstm pipeline, motivated by the fact that raw time-domain CSI amplitude mixes gait
with room geometry/multipath (which is person-INDEPENDENT and dominates the signal at this feature
resolution -- see walk_bilstm_pipeline.py's and the various run_walk_bilstm_*openset.py scripts'
docstrings for the extensive evidence this is why open-set separation has failed every other way
tried). Human gait shows up as energy in a narrow, characteristic low-frequency band (~0.5-3Hz: leg
swing, arm swing, torso bounce) riding on top of the room's much slower/static reflections -- a
short-time Fourier transform isolates that band directly instead of hoping a BiLSTM learns to ignore
the room from raw amplitude.

Each already-built [400, 256] window (400 samples = 4s at the pipeline's fixed 100Hz interpolation
grid, 256 = 128 amplitude + 128 phase) is reduced to a [n_frames, n_freq_bins] "Doppler profile": STFT
each of the 128 amplitude subcarriers, keep only the low-frequency bins gait actually occupies, sum
magnitude across all 128 subcarriers (one combined velocity-profile time series, not per-subcarrier --
matches the physical picture: every subcarrier sees the same moving body, just with different fading,
so summing pools SNR instead of picking one arbitrary "best" subcarrier the way time-domain top-30
variance selection does).
"""
from __future__ import annotations

import numpy as np

STFT_WINDOW = 64   # ~0.64s at the pipeline's 100Hz grid
STFT_HOP = 8       # ~0.08s
MAX_FREQ_HZ = 5.0  # gait fundamental + a couple harmonics; well above typical stride rate (~1-2Hz)
SAMPLE_RATE_HZ = 100.0  # FINAL_LEN=400 samples / WINDOW_SEC=4.0s, fixed by walk_bilstm_pipeline


def _stft_magnitude(x: np.ndarray, window: int, hop: int) -> np.ndarray:
    """x: (n_samples,). Returns (n_freq_bins, n_frames) magnitude spectrogram, Hann-windowed,
    no external deps (numpy rfft only)."""
    n_samples = len(x)
    n_frames = max(1, (n_samples - window) // hop + 1)
    win = np.hanning(window)
    frames = np.stack([x[i * hop: i * hop + window] * win for i in range(n_frames)])  # (n_frames, window)
    spec = np.fft.rfft(frames, axis=1)  # (n_frames, window//2+1)
    return np.abs(spec).T  # (n_freq_bins, n_frames)


def doppler_profile(amp_window: np.ndarray, max_freq_hz: float = MAX_FREQ_HZ,
                     stft_window: int = STFT_WINDOW, stft_hop: int = STFT_HOP,
                     sample_rate_hz: float = SAMPLE_RATE_HZ) -> np.ndarray:
    """amp_window: (400, 128) -- one window's amplitude channels (pre-topk-selection, pre-zscore).
    Returns (n_frames, n_freq_bins_kept): summed-across-subcarriers STFT magnitude, low-frequency
    bins only, transposed to (time, freq) to match every other model here's (seq_len, n_features)
    input convention."""
    n_bins_total = stft_window // 2 + 1
    freq_res_hz = sample_rate_hz / stft_window
    n_bins_keep = max(1, int(np.ceil(max_freq_hz / freq_res_hz)) + 1)
    n_bins_keep = min(n_bins_keep, n_bins_total)

    summed = None
    for sub in range(amp_window.shape[1]):
        mag = _stft_magnitude(amp_window[:, sub], stft_window, stft_hop)[:n_bins_keep]  # (n_bins_keep, n_frames)
        summed = mag if summed is None else summed + mag
    return summed.T.astype(np.float32)  # (n_frames, n_bins_keep)


def build_doppler_dataset(windows_all: np.ndarray) -> np.ndarray:
    """windows_all: (N, 400, 256) raw pipeline windows (128 amp + 128 phase, pre-selection).
    Returns (N, n_frames, n_bins_keep) Doppler profiles, amplitude channels only."""
    amp = windows_all[:, :, :128]
    profiles = [doppler_profile(amp[i]) for i in range(len(amp))]
    return np.stack(profiles)


def per_window_zscore_doppler(profiles: np.ndarray) -> np.ndarray:
    """Same per-window, per-channel standardization idea as walk_bilstm_pipeline.per_window_zscore,
    applied to Doppler profiles instead of raw amplitude/phase."""
    mu = profiles.mean(axis=1, keepdims=True)
    std = profiles.std(axis=1, keepdims=True) + 1e-6
    return ((profiles - mu) / std).astype(np.float32)
