"""Causal, streaming analog of ml.data_pipeline.time_resample.resample_time_axis -- lets
ml/inference/live_infer.py feed live packets (arriving at an uneven native rate) through a
checkpoint trained with mode="resampled_timenorm", which expects windows built from a FIXED
target_rate_hz grid rather than raw packet counts (see time_resample.py's module docstring for
why: fixed-packet-count windows would otherwise conflate window duration with capture rate).

Bins are averaged the same way as the batch version (amplitude arithmetic mean, phase via
circular/unit-circle mean). One deliberate difference: the batch version fills a fully-empty bin
(a packet-drop gap wider than one bin) by INTERPOLATING between its nearest non-empty neighbors on
either side; a live stream has no "later" neighbor yet, so empty bins here are forward-filled from
the last emitted bin instead -- the causal equivalent.
"""
from __future__ import annotations

import numpy as np


class LiveTimeResampler:
    def __init__(self, target_rate_hz: float):
        self.bin_width_s = 1.0 / target_rate_hz
        self._t0_us: int | None = None
        self._bin_idx: int | None = None
        self._amp_sum: np.ndarray | None = None
        self._cos_sum: np.ndarray | None = None
        self._sin_sum: np.ndarray | None = None
        self._count = 0
        self._last: tuple[np.ndarray, np.ndarray] | None = None

    def _open_bin(self, amplitude: np.ndarray, phase: np.ndarray) -> None:
        self._amp_sum = amplitude.astype(np.float64)
        self._cos_sum = np.cos(phase).astype(np.float64)
        self._sin_sum = np.sin(phase).astype(np.float64)
        self._count = 1

    def _accumulate(self, amplitude: np.ndarray, phase: np.ndarray) -> None:
        self._amp_sum += amplitude
        self._cos_sum += np.cos(phase)
        self._sin_sum += np.sin(phase)
        self._count += 1

    def _close_bin(self) -> tuple[np.ndarray, np.ndarray]:
        amp = (self._amp_sum / self._count).astype(np.float32)
        phase = np.arctan2(self._sin_sum / self._count, self._cos_sum / self._count).astype(np.float32)
        self._last = (amp, phase)
        return self._last

    def push(self, amplitude: np.ndarray, phase: np.ndarray, device_time_us: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Feed one native-rate decoded packet. Returns zero or more resampled (amplitude, phase)
        packets: empty while the packet still belongs to the currently-open bin, one in the common
        case where it closes exactly the bin it lands past, or more than one if a packet-drop gap
        closed several bins at once (each extra one forward-filled from the last emitted bin)."""
        if self._t0_us is None:
            self._t0_us = device_time_us
            self._bin_idx = 0
            self._open_bin(amplitude, phase)
            return []

        t_s = (device_time_us - self._t0_us) / 1e6
        bin_idx = int(t_s / self.bin_width_s)
        if bin_idx <= self._bin_idx:  # still the open bin (or harmless clock jitter) -- keep accumulating
            self._accumulate(amplitude, phase)
            return []

        emitted = [self._close_bin()]
        self._bin_idx += 1
        while self._bin_idx < bin_idx:
            emitted.append(self._last)  # forward-fill an empty (packet-drop) bin
            self._bin_idx += 1
        self._open_bin(amplitude, phase)
        return emitted
