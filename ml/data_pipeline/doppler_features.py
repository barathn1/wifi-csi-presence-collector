"""Doppler/velocity-domain features -- see ml/reports/day2_next_steps.md item 9 (the "bigger lift, most
physically sound" option) and [[project-day2-cross-channel-root-cause]] in memory. Doppler shift from
body motion barely changes between 2402MHz and 2472MHz (~3% of carrier frequency), unlike raw
per-subcarrier amplitude which is tied to the exact (mismatched) frequency probed on each day. Targets
motion-FREQUENCY content (gait, movement) rather than the invariant_features.py module's static
distribution-shape stats, which tested near-chance for this task.

Needs a LONGER segment than the base ~1s/200-packet window (a gait cycle's fundamental is ~1-3Hz, too
slow to resolve well in 1s) -- built on the same 10-30s segments as evaluation/segment_aggregation.py's
consecutive-window grouping, but reads straight from the per-session decode cache rather than
regrouping windowed slices, so it isn't tied to the 200-packet/50%-overlap windowing choice at all.

Caveat: assumes roughly uniform packet timing within a segment (uses the session's average rate, not
per-packet device_time_us) -- fine for a first pass given CSI packet rate is fairly stable within one
session (see windowing.py's docstring: 162-263 Hz observed, mean ~217 Hz), but a real refinement would
resample onto a uniform time grid using the exact timestamps before the Welch PSD.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

EPS = 1e-6
DOPPLER_BANDS_HZ = [(0.3, 1), (1, 3), (3, 6), (6, 15), (15, 30)]


def doppler_band_powers(amplitude: np.ndarray, sample_rate_hz: float,
                         bands: list[tuple[float, float]] = DOPPLER_BANDS_HZ) -> np.ndarray:
    """amplitude: (T, n_sub) over a long (multi-second) segment -> (len(bands),) normalized band-power
    fractions of the subcarrier-averaged Doppler power spectrum. No dependence on which subcarrier
    index held which value -- only on the TEMPORAL frequency content of the motion, averaged across
    subcarriers."""
    detrended = amplitude - amplitude.mean(axis=0, keepdims=True)
    nperseg = min(256, len(detrended))
    if nperseg < 8:
        return np.zeros(len(bands), dtype=np.float32)
    freqs, psd = signal.welch(detrended, fs=sample_rate_hz, axis=0, nperseg=nperseg)
    avg_psd = psd.mean(axis=1)  # (n_freq,) -- averaged across subcarriers
    total_power = avg_psd.sum() + EPS
    band_powers = np.array([avg_psd[(freqs >= lo) & (freqs < hi)].sum() / total_power for lo, hi in bands])
    return band_powers.astype(np.float32)


def doppler_feature_names(bands: list[tuple[float, float]] = DOPPLER_BANDS_HZ) -> list[str]:
    names = []
    for channel in ("amp", "phase"):
        names += [f"{channel}_doppler_{lo}-{hi}Hz" for lo, hi in bands]
    return names


def extract_doppler_segment_features(amplitude: np.ndarray, phase: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """amplitude/phase: (T, n_sub) over a long segment -> concatenated amp+phase Doppler band powers."""
    amp_bands = doppler_band_powers(amplitude, sample_rate_hz)
    phase_bands = doppler_band_powers(phase, sample_rate_hz)
    return np.nan_to_num(np.concatenate([amp_bands, phase_bands]), nan=0.0)


def iter_long_segments(manifest_row, cache_path, segment_packets: int = 3000, calibration=None):
    """Non-overlapping segment_packets-length chunks straight from one session's decode cache --
    bypasses the window_index/windowing.py's 200-packet windowing entirely, since Doppler features
    need a much longer horizon. Yields (amplitude_segment, phase_segment, avg_rate_hz)."""
    with np.load(cache_path) as d:
        amplitude, phase, device_time_us = d["amplitude"], d["phase"], d["device_time_us"]
    n = amplitude.shape[0]
    for start in range(0, n - segment_packets + 1, segment_packets):
        end = start + segment_packets
        span_s = (device_time_us[end - 1] - device_time_us[start]) / 1e6
        rate_hz = segment_packets / span_s if span_s > 0 else 200.0
        amp_seg, phase_seg = amplitude[start:end], phase[start:end]
        if calibration is not None:
            amp_seg, phase_seg = calibration(amp_seg, phase_seg, manifest_row)
        yield amp_seg, phase_seg, rate_hz
