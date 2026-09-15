"""Live equivalent of ml/data_pipeline/calibration.py::compute_day_baseline -- same math, different
data source. compute_day_baseline reads pre-recorded `none`-labeled sessions off disk for a given date;
this computes the identical DayBaseline from packets collected live during a short "stand outside the
room" calibration phase at the start of a live-inference session. Reuses the DayBaseline dataclass and
EPS constant directly so ml.data_pipeline.calibration.apply_variant_a needs zero changes to work here.
"""
from __future__ import annotations

import numpy as np

from ml.data_pipeline.calibration import EPS, DayBaseline


def compute_live_baseline(amplitude: np.ndarray, phase: np.ndarray, label: str = "live") -> DayBaseline:
    """amplitude/phase: (n_packets, n_subcarriers), collected during the live calibration period."""
    amp_mean, amp_var = amplitude.mean(axis=0), amplitude.var(axis=0)
    phase_mean, phase_var = phase.mean(axis=0), phase.var(axis=0)
    return DayBaseline(
        date=label, source_sessions=[], n_packets=amplitude.shape[0],
        amp_mean=amp_mean.astype(np.float32), amp_std=np.sqrt(np.maximum(amp_var, EPS)).astype(np.float32),
        phase_mean=phase_mean.astype(np.float32), phase_std=np.sqrt(np.maximum(phase_var, EPS)).astype(np.float32),
    )
