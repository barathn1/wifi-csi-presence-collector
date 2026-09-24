"""Resample the TIME axis of decoded CSI onto a fixed real-world sample rate, so a window of N packets
represents the same real-world DURATION regardless of which session (and therefore which native capture
rate) it came from. Companion to csi_resample.py, which does the analogous thing for the subcarrier axis.

Real packets arrive at an uneven, session-dependent rate driven by real-time network conditions
(contention/congestion/stimulus traffic at capture time -- NOT channel or bandwidth, both of which can be
identical across sessions while the rate still differs). This project's Day3-channel-6 collection split
almost perfectly by rate between authorized (146-222Hz) and unauthorized (226-268Hz) training sessions,
purely because of when each block happened to be recorded -- see
[[project-day3-ch6-packet-rate-confound]]. Fixed-packet-count windowing then silently correlates window
DURATION with whatever label happened to be recorded at that pace: a non-biometric shortcut a model can
learn instead of real gait/reflection signal. Resampling every session onto the same fixed rate BEFORE
windowing closes that gap -- "200 packets" then means the same real seconds everywhere.

The target rate is NOT a hardcoded constant -- it's derived from whatever dataset is actually being
processed (`compute_target_rate_hz`, the minimum native rate across that dataset's sessions, same
never-fabricate-resolution-that-was-never-captured principle csi_resample.py applies to the subcarrier
axis, but computed fresh each time rather than baked in). A hardcoded ceiling would permanently throw away
resolution the moment better hardware/collection captures faster -- e.g. if a future collection's slowest
session is 380Hz instead of today's 130.8Hz, re-deriving the target rate picks ~372Hz, not whatever number
happened to be right for THIS dataset.

Downsampling uses BIN-AVERAGING, not point interpolation: every original packet falling in a given output
bin contributes to that bin's mean (amplitude arithmetic mean, phase via circular/unit-circle mean, same
wrap-safe trick csi_resample.py uses). A naive np.interp-based resample would only ever look at the two
nearest neighboring packets per output sample and silently discard every packet in between -- exactly the
"ignore most of the data from faster sessions" failure mode this is meant to avoid. Only bins that end up
with zero packets in them (a gap from packet drops wider than one bin) fall back to interpolating between
neighboring non-empty bins.
"""
from __future__ import annotations

import numpy as np

DEFAULT_SAFETY_MARGIN = 0.98  # shave a hair off the observed minimum so float/timing jitter in the
# slowest session never causes it to be accidentally upsampled by a few samples


def compute_target_rate_hz(native_rates_hz: list[float], safety_margin: float = DEFAULT_SAFETY_MARGIN) -> float:
    """Derive the common resample target from the ACTUAL sessions in play -- never a fixed constant, so
    it rises automatically if a future collection's slowest session is faster than today's."""
    assert native_rates_hz, "need at least one session's native rate to derive a target"
    return min(native_rates_hz) * safety_margin


def fix_clock_reset(device_time_us: np.ndarray) -> np.ndarray:
    """Boolean mask selecting the longest contiguous monotonically-increasing run of `device_time_us`.
    `device_time_us` is time-since-boot, not time-since-session-start -- a receiver reboot mid-session
    (seen on several 2026-09-21/22 sessions, the first collection with 3 receivers sharing a power/USB
    setup) makes it jump sharply backward at the reset point, then climb again from near-zero. That
    single discontinuity breaks BOTH the native-rate estimate (a naive last-minus-first duration goes
    negative) and `resample_time_axis`'s bin-index math (which assumes strictly increasing time) if left
    in. Dropping the shorter side of the split is a conservative fix -- some real packets are discarded,
    but nothing is fabricated and no cross-reset time-ordering corruption reaches downstream code.
    A no-op (mask all True) when `device_time_us` is already monotonic (every session before
    2026-09-21)."""
    diffs = np.diff(device_time_us.astype(np.int64))
    if (diffs >= 0).all():
        return np.ones(len(device_time_us), dtype=bool)
    reset_points = np.flatnonzero(diffs < 0) + 1  # index of the first sample AFTER each reset
    boundaries = [0] + reset_points.tolist() + [len(device_time_us)]
    seg_lengths = np.diff(boundaries)
    best = int(np.argmax(seg_lengths))
    mask = np.zeros(len(device_time_us), dtype=bool)
    mask[boundaries[best]:boundaries[best + 1]] = True
    return mask


def session_native_rate_hz(device_time_us: np.ndarray) -> float:
    """Packets/second actually captured in this session, straight from its own timestamps (after
    dropping any clock-reset artifact -- see fix_clock_reset)."""
    mask = fix_clock_reset(device_time_us)
    dt = device_time_us[mask]
    duration_s = (dt[-1] - dt[0]) / 1e6
    return (len(dt) - 1) / duration_s


def resample_time_axis(
    amplitude: np.ndarray, phase: np.ndarray, device_time_us: np.ndarray, rssi: np.ndarray,
    target_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """amplitude/phase: (n_packets, n_subcarriers), rssi/device_time_us: (n_packets,). Returns all four
    resampled onto a uniform grid at target_rate_hz via bin-averaging (every packet contributes to
    exactly one output bin's mean -- see module docstring for why this beats point interpolation for
    downsampling). `target_rate_hz` must be explicitly computed by the caller (e.g. via
    `compute_target_rate_hz` over the actual dataset in play) -- no silent hardcoded default, so a stale
    rate from a different dataset can never be reused by accident."""
    t = (device_time_us - device_time_us[0]).astype(np.float64) / 1e6  # seconds since session start
    bin_width = 1.0 / target_rate_hz
    n_new = max(2, int(t[-1] / bin_width) + 1)
    bin_idx = np.minimum((t / bin_width).astype(np.int64), n_new - 1)

    n_sub = amplitude.shape[1]
    counts = np.bincount(bin_idx, minlength=n_new).astype(np.float64)
    counts_safe = np.maximum(counts, 1)  # avoid /0 for empty bins; those get overwritten by interp below

    amp_sums = np.zeros((n_new, n_sub), dtype=np.float64)
    cos_sums = np.zeros((n_new, n_sub), dtype=np.float64)
    sin_sums = np.zeros((n_new, n_sub), dtype=np.float64)
    rssi_sums = np.zeros(n_new, dtype=np.float64)
    np.add.at(amp_sums, bin_idx, amplitude)
    np.add.at(cos_sums, bin_idx, np.cos(phase))
    np.add.at(sin_sums, bin_idx, np.sin(phase))
    np.add.at(rssi_sums, bin_idx, rssi)

    amp_new = (amp_sums / counts_safe[:, None]).astype(np.float32)
    cos_mean = cos_sums / counts_safe[:, None]
    sin_mean = sin_sums / counts_safe[:, None]
    phase_new = np.arctan2(sin_mean, cos_mean).astype(np.float32)
    rssi_new = (rssi_sums / counts_safe).astype(np.float32)

    empty = counts == 0
    if empty.any():
        # a gap wider than one bin (e.g. a run of dropped packets) -- fill from the nearest non-empty
        # bins on either side rather than leaving a fabricated zero/average-of-nothing value.
        filled_idx = np.flatnonzero(~empty)
        t_new = (np.arange(n_new) + 0.5) * bin_width
        for k in range(n_sub):
            amp_new[empty, k] = np.interp(t_new[empty], t_new[filled_idx], amp_new[filled_idx, k])
            cos_i = np.interp(t_new[empty], t_new[filled_idx], np.cos(phase_new[filled_idx, k]))
            sin_i = np.interp(t_new[empty], t_new[filled_idx], np.sin(phase_new[filled_idx, k]))
            phase_new[empty, k] = np.arctan2(sin_i, cos_i)
        rssi_new[empty] = np.interp(t_new[empty], t_new[filled_idx], rssi_new[filled_idx])

    device_time_us_new = (device_time_us[0] + ((np.arange(n_new) + 0.5) * bin_width * 1e6)).astype(np.int64)
    return amp_new, phase_new, rssi_new, device_time_us_new
