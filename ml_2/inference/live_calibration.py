"""Live equivalent of ml_2/data/calibration.py::compute_baseline -- same per-subcarrier mean/std math,
but fit from packets collected live during a short "stand outside the room" phase at the start of a
live-inference session, instead of pre-recorded `none` sessions off disk. Reuses the Baseline dataclass
and EPS directly so ml_2.data.calibration.apply_baseline needs zero changes to work here.
"""
from __future__ import annotations

import numpy as np

from ml_2.data.calibration import EPS, Baseline


def compute_live_baseline(amplitude: np.ndarray, phase: np.ndarray, label: str = "live") -> Baseline:
    """amplitude/phase: (n_packets, n_subcarriers), collected during the live calibration period."""
    amp_mean, amp_var = amplitude.mean(axis=0), amplitude.var(axis=0)
    phase_mean, phase_var = phase.mean(axis=0), phase.var(axis=0)
    return Baseline(
        label=label,
        amp_mean=amp_mean.astype(np.float32), amp_std=np.sqrt(np.maximum(amp_var, EPS)).astype(np.float32),
        phase_mean=phase_mean.astype(np.float32), phase_std=np.sqrt(np.maximum(phase_var, EPS)).astype(np.float32),
    )
