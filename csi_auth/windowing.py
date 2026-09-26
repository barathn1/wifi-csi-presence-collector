"""Fixed-length, overlapping windows over one session's decoded CSI."""
from __future__ import annotations

import numpy as np

WINDOW_PACKETS = 200   # ~1s at this hardware's typical ~200Hz capture rate
STRIDE_PACKETS = 100   # 50% overlap


def make_windows(amplitude: np.ndarray, phase: np.ndarray, rssi: np.ndarray,
                  window_packets: int = WINDOW_PACKETS, stride_packets: int = STRIDE_PACKETS):
    """Yields (start, end) slices; caller indexes amplitude/phase/rssi with them."""
    n = amplitude.shape[0]
    for start in range(0, n - window_packets + 1, stride_packets):
        yield start, start + window_packets
