"""Per-day, per-subcarrier empty-room self-calibration -- standardize a day's windows against that
SAME day's own `none` (empty-room) session stats, per (receiver_mac) since different receivers have
different hardware gain/noise floors. Per-subcarrier vectors, never a single global scalar -- the core
signal this whole project relies on ("subcarrier k reacts differently per person") would be washed out
by a global normalization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml_2.data.windowing import cache_row

EPS = 1e-6


@dataclass
class Baseline:
    label: str
    amp_mean: np.ndarray
    amp_std: np.ndarray
    phase_mean: np.ndarray
    phase_std: np.ndarray


def compute_baseline(session_rows: pd.DataFrame, label: str) -> Baseline:
    amp_sum = amp_sumsq = phase_sum = phase_sumsq = None
    n = 0
    for _, row in session_rows.iterrows():
        path = cache_row(row)
        if path is None:
            continue
        with np.load(path) as d:
            amp, phase = d["amplitude"], d["phase"]
        if amp_sum is None:
            amp_sum, amp_sumsq = amp.sum(0), (amp.astype(np.float64) ** 2).sum(0)
            phase_sum, phase_sumsq = phase.sum(0), (phase.astype(np.float64) ** 2).sum(0)
        else:
            amp_sum += amp.sum(0)
            amp_sumsq += (amp.astype(np.float64) ** 2).sum(0)
            phase_sum += phase.sum(0)
            phase_sumsq += (phase.astype(np.float64) ** 2).sum(0)
        n += amp.shape[0]
    if n == 0:
        raise ValueError(f"no packets found to compute baseline '{label}'")
    amp_mean = amp_sum / n
    amp_std = np.sqrt(np.maximum(amp_sumsq / n - amp_mean ** 2, EPS))
    phase_mean = phase_sum / n
    phase_std = np.sqrt(np.maximum(phase_sumsq / n - phase_mean ** 2, EPS))
    return Baseline(label=label, amp_mean=amp_mean.astype(np.float32), amp_std=amp_std.astype(np.float32),
                     phase_mean=phase_mean.astype(np.float32), phase_std=phase_std.astype(np.float32))


def apply_baseline(amplitude: np.ndarray, phase: np.ndarray, baseline: Baseline) -> tuple[np.ndarray, np.ndarray]:
    return (amplitude - baseline.amp_mean) / (baseline.amp_std + EPS), \
        (phase - baseline.phase_mean) / (baseline.phase_std + EPS)


def per_receiver_day_baselines(manifest: pd.DataFrame) -> dict[tuple[str, str], Baseline]:
    """One baseline per (date, receiver_mac), fit from that date+receiver's `none` sessions."""
    baselines = {}
    none_rows = manifest[manifest["label"] == "none"]
    for (date, mac), group in none_rows.groupby(["date", "receiver_mac"]):
        try:
            baselines[(date, mac)] = compute_baseline(group, label=f"{date}_{mac}")
        except ValueError as exc:
            print(f"  no baseline for {date}/{mac}: {exc}")
    return baselines


def make_calibration_fn(baselines: dict[tuple[str, str], Baseline]):
    """A single Baseline for sessions where per-(date,receiver) doesn't exist (falls back to a global
    baseline), returned as a (amp, phase, row) -> (amp, phase) callable for windowing.iter_windows."""
    fallback = compute_fallback(baselines)

    def calibration(amplitude: np.ndarray, phase: np.ndarray, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        baseline = baselines.get((row["date"], row["receiver_mac"]), fallback)
        return apply_baseline(amplitude, phase, baseline)

    return calibration


def compute_fallback(baselines: dict[tuple[str, str], Baseline]) -> Baseline:
    if not baselines:
        raise ValueError("no baselines at all -- need at least one date/receiver with a `none` session")
    any_baseline = next(iter(baselines.values()))
    amp_means = np.stack([b.amp_mean for b in baselines.values()])
    amp_stds = np.stack([b.amp_std for b in baselines.values()])
    phase_means = np.stack([b.phase_mean for b in baselines.values()])
    phase_stds = np.stack([b.phase_std for b in baselines.values()])
    return Baseline(label="fallback_mean_of_all", amp_mean=amp_means.mean(0), amp_std=amp_stds.mean(0),
                     phase_mean=phase_means.mean(0), phase_std=phase_stds.mean(0))
