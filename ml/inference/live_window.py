"""A real sliding/rolling window over a live packet stream -- nothing like this existed anywhere in
this repo before (ml/data_pipeline/windowing.py is entirely file/batch-based: it operates on an
already-fully-recorded samples.npz, and even the live browser viewer's JS only keeps a scrolling
*display* buffer, not a real fixed-length window for computation).

Matches windowing.py's own window_packets=200/stride_packets=100 (50% overlap) convention exactly, so
a live window is the same shape a checkpoint was trained on.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class RollingWindower:
    def __init__(self, window_packets: int = 200, stride_packets: int = 100):
        self.window_packets = window_packets
        self.stride_packets = stride_packets
        self._amp: deque = deque(maxlen=window_packets)
        self._phase: deque = deque(maxlen=window_packets)
        self._n_pushed = 0

    def push(self, amplitude: np.ndarray, phase: np.ndarray) -> bool:
        """Feed one decoded packet. Returns True when a full window is ready to read via get_window().
        Ready exactly when `windowing.py::build_window_index`'s batch convention
        (`range(0, n - window_packets + 1, stride_packets)`) would also emit a window -- i.e. the
        FIRST window is ready as soon as the buffer first fills (not after one extra stride's delay,
        which an earlier version of this method mistakenly did)."""
        self._amp.append(amplitude)
        self._phase.append(phase)
        self._n_pushed += 1
        if self._n_pushed < self.window_packets:
            return False
        return (self._n_pushed - self.window_packets) % self.stride_packets == 0

    def get_window(self) -> tuple[np.ndarray, np.ndarray]:
        """(window_packets, n_subcarriers) amplitude, phase -- call only after push() returns True."""
        return np.stack(self._amp), np.stack(self._phase)
