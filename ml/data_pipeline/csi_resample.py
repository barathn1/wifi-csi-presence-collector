"""Resample the subcarrier axis of decoded CSI so sessions captured under different PHY frame formats
(different subcarrier counts) can share one fixed-shape model input.

Day 1 sessions were captured almost entirely on a 40MHz-bonded channel (cwb=1, 186 subcarriers); Day 2
sessions negotiated a 20MHz channel instead (cwb=0, 128 subcarriers) -- confirmed via the per-packet
sig_mode/mcs/cwb/channel_primary/channel_secondary fields in samples.npz, not a firmware/config change
(config_snapshot is identical). TARGET_SUBCARRIERS is set to 128 (the smaller of the two) so every
session is downsampled, never upsampled -- upsampling would fabricate frequency resolution Day 2 never
actually captured.

Amplitude interpolates directly (a smooth magnitude curve across subcarrier index). Phase is wrapped
(-pi, pi], so linear interpolation across a wrap boundary produces spurious jumps; interpolating the
unit-circle (cos, sin) components and recombining via atan2 avoids that.
"""
from __future__ import annotations

import numpy as np

TARGET_SUBCARRIERS = 128


def linear_interp_matrix(src_n: int, target_n: int) -> np.ndarray:
    """(target_n, src_n) matrix M such that `arr @ M.T` linearly interpolates each row from src_n to
    target_n samples over a shared normalized [0, 1] grid -- avoids a per-row np.interp Python loop."""
    if src_n == target_n:
        return np.eye(src_n, dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, src_n)
    tgt_x = np.linspace(0.0, 1.0, target_n)
    M = np.zeros((target_n, src_n), dtype=np.float32)
    idx_hi = np.searchsorted(src_x, tgt_x, side="left").clip(1, src_n - 1)
    idx_lo = idx_hi - 1
    x_lo, x_hi = src_x[idx_lo], src_x[idx_hi]
    w_hi = np.where(x_hi > x_lo, (tgt_x - x_lo) / (x_hi - x_lo), 0.0)
    M[np.arange(target_n), idx_lo] = 1.0 - w_hi
    M[np.arange(target_n), idx_hi] += w_hi
    return M


def resample_amplitude(amplitude: np.ndarray, target_n: int = TARGET_SUBCARRIERS) -> np.ndarray:
    """amplitude: (n_packets, src_n) -> (n_packets, target_n)."""
    M = linear_interp_matrix(amplitude.shape[1], target_n)
    return (amplitude.astype(np.float32) @ M.T).astype(np.float32)


def resample_phase(phase: np.ndarray, target_n: int = TARGET_SUBCARRIERS) -> np.ndarray:
    """phase: (n_packets, src_n) radians -> (n_packets, target_n), wrap-safe via unit-circle interp."""
    M = linear_interp_matrix(phase.shape[1], target_n)
    cos_r = np.cos(phase).astype(np.float32) @ M.T
    sin_r = np.sin(phase).astype(np.float32) @ M.T
    return np.arctan2(sin_r, cos_r).astype(np.float32)
