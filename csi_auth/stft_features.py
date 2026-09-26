"""Frequency-domain (Doppler/micro-motion) features -- never tried anywhere else in this pipeline,
which has been entirely time-domain so far (raw amplitude sequences, or time-domain stats). Motivated
directly by FINDINGS.md's own conclusion: identity in this data lives in gait DYNAMICS, and the
physical mechanism for that is Doppler shift from stride/limb-swing motion -- inherently a frequency-
domain signature (stride rate ~1-2 Hz, limb-swing harmonics reaching into the ~10-15 Hz range per WiFi
sensing literature), not something time-domain mean/std/skew/kurtosis was ever going to isolate
cleanly.

The one thing that makes this tricky: capture rate varies 200-500 Hz across sessions in this dataset
(see FINDINGS.md), so a raw per-bin FFT of a fixed-packet-count window has a DIFFERENT bin-to-Hz
mapping depending on which session it came from -- comparing raw bins across sessions would be
comparing different physical frequencies without realizing it. Fixed here by resampling each window
onto a uniform time grid spanning its own ACTUAL duration (from device_time_us, not assumed), then
interpolating the resulting spectrum onto one fixed target Hz grid shared by every window regardless
of source session's capture rate.
"""
from __future__ import annotations

import numpy as np

# Gait/limb-motion Doppler content for a walking human falls roughly in this band -- see module
# docstring. 0.5 Hz steps -> 39 features, deliberately compact (a handful of real physical bins, not
# hundreds of raw FFT bins) so a discriminative model doesn't need much data to find structure in it.
TARGET_FREQ_GRID_HZ = np.arange(0.5, 20.0, 0.5)


def window_doppler_spectrum(amplitude: np.ndarray, elapsed_s_window: np.ndarray) -> np.ndarray:
    """amplitude: (window_packets, n_subcarriers), already cleaned+masked. elapsed_s_window:
    (window_packets,) actual elapsed seconds per packet in this window (monotonic, from
    device_time_us -- NOT assumed evenly spaced, real packet arrival always jitters). Returns one
    compact motion-frequency spectrum (len(TARGET_FREQ_GRID_HZ),), averaged across subcarriers (so
    per-subcarrier noise cancels, leaving the shared motion-induced Doppler content), on the fixed Hz
    grid above regardless of this window's native capture rate."""
    n_packets = amplitude.shape[0]
    duration = elapsed_s_window[-1] - elapsed_s_window[0]
    if duration <= 0:
        return np.zeros(len(TARGET_FREQ_GRID_HZ), dtype=np.float32)

    uniform_t = np.linspace(elapsed_s_window[0], elapsed_s_window[-1], n_packets)
    native_rate = (n_packets - 1) / duration

    spectra = np.empty((amplitude.shape[1], n_packets // 2 + 1), dtype=np.float64)
    for sc in range(amplitude.shape[1]):
        resampled = np.interp(uniform_t, elapsed_s_window, amplitude[:, sc])
        resampled = resampled - resampled.mean()
        spectra[sc] = np.abs(np.fft.rfft(resampled))
    mean_spectrum = spectra.mean(axis=0)

    native_freqs = np.fft.rfftfreq(n_packets, d=1.0 / native_rate)
    grid = np.interp(TARGET_FREQ_GRID_HZ, native_freqs, mean_spectrum)
    return grid.astype(np.float32)
