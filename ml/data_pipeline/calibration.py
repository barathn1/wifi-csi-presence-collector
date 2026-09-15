"""Empty-room baseline calibration -- the user's cross-day normalization idea, built as two variants.

- **Variant A (self-calibration, OpenCSI-style)**: standardize a day's occupied windows against that
  same day's own `none` (empty-room) session statistics. Each day is independently zero-mean/unit-var
  relative to its own quiet-period baseline. This is the direct analog of OpenCSI's Z-score approach
  (arXiv:2607.26665), which reported F1 0.99 vs 0.87 for cross-room/cross-hardware transfer on ESP32 --
  scoped to binary occupancy there, being tested here for person-ID/auth for the first time.
- **Variant B (cross-day re-referencing)**: map *today's* empty-room-relative signal onto a *reference
  day's* empty-room statistics, not just to zero-mean/unit-var in isolation:

      calibrated = (raw - mu_today_empty) * (sigma_ref_empty / sigma_today_empty) + mu_ref_empty

  This is the literal mechanism behind "calibrate with today's empty room, then normalize to the
  previous day's baseline" -- it preserves the reference day's original scale/offset as the fixed
  target frame instead of collapsing every day to the same 0/1 moments. No paper validates this exact
  claim for person-ID (see RESEARCH_NOTES.md section 12) -- this is the requested "give it a try", not
  a result borrowed from the literature.

Real cross-day validation needs Day 2 data. Until then, `day1_proxy_baselines()` below picks two Day-1
`none` sessions far apart in time to exercise the full Variant B code path today and get an early, weak
signal -- NOT proof of cross-day benefit, just a mechanical/sanity check ahead of real data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.data_pipeline.windowing import cache_session

EPS = 1e-6


@dataclass
class DayBaseline:
    date: str
    source_sessions: list[str]
    n_packets: int
    amp_mean: np.ndarray   # (n_subcarriers,)
    amp_std: np.ndarray
    phase_mean: np.ndarray
    phase_std: np.ndarray


def compute_day_baseline(manifest: pd.DataFrame, date: str, mode: str = "resampled") -> DayBaseline:
    """Fit a baseline from ALL of a date's `none` sessions (not just windowed subsets -- uses every
    cached packet for a tighter per-subcarrier mean/std estimate). `mode` must match whatever window
    index this baseline will be applied to ("resampled" for cross-day work, "native" otherwise)."""
    none_rows = manifest[
        manifest["session_dir"].str.contains(f"/{date}/", regex=False) & (manifest["label"] == "none")
    ]
    if none_rows.empty:
        raise ValueError(f"no `none` (empty-room) sessions found for date {date}")

    amp_sum = amp_sumsq = phase_sum = phase_sumsq = None
    n = 0
    sources = []
    for _, row in none_rows.iterrows():
        cache_path = cache_session(row["session_dir"], mode)
        if cache_path is None:
            continue
        with np.load(cache_path) as d:
            amp, phase = d["amplitude"], d["phase"]
        sources.append(row["session_dir"])
        if amp_sum is None:
            amp_sum, amp_sumsq = amp.sum(0), (amp.astype(np.float64) ** 2).sum(0)
            phase_sum, phase_sumsq = phase.sum(0), (phase.astype(np.float64) ** 2).sum(0)
        else:
            amp_sum += amp.sum(0)
            amp_sumsq += (amp.astype(np.float64) ** 2).sum(0)
            phase_sum += phase.sum(0)
            phase_sumsq += (phase.astype(np.float64) ** 2).sum(0)
        n += amp.shape[0]

    amp_mean = amp_sum / n
    amp_std = np.sqrt(np.maximum(amp_sumsq / n - amp_mean ** 2, EPS))
    phase_mean = phase_sum / n
    phase_std = np.sqrt(np.maximum(phase_sumsq / n - phase_mean ** 2, EPS))

    return DayBaseline(
        date=date, source_sessions=sources, n_packets=n,
        amp_mean=amp_mean.astype(np.float32), amp_std=amp_std.astype(np.float32),
        phase_mean=phase_mean.astype(np.float32), phase_std=phase_std.astype(np.float32),
    )


def compute_day_rssi_baseline(manifest: pd.DataFrame, date: str, mode: str = "resampled") -> tuple[float, float]:
    """Same idea as compute_day_baseline but for the single scalar RSSI field, from that date's `none`
    sessions -- RSSI carries some class signal (see day2_next_steps.md item 7) but drifts across days
    on its own, so it needs the same per-day self-calibration before being fused into a model."""
    none_rows = manifest[
        manifest["session_dir"].str.contains(f"/{date}/", regex=False) & (manifest["label"] == "none")
    ]
    if none_rows.empty:
        raise ValueError(f"no `none` (empty-room) sessions found for date {date}")
    values = []
    for _, row in none_rows.iterrows():
        cache_path = cache_session(row["session_dir"], mode)
        if cache_path is None:
            continue
        with np.load(cache_path) as d:
            values.append(d["rssi"])
    all_rssi = np.concatenate(values)
    return float(all_rssi.mean()), float(all_rssi.std())


def apply_variant_a(amplitude: np.ndarray, phase: np.ndarray, baseline: DayBaseline) -> tuple[np.ndarray, np.ndarray]:
    """Self-calibration: standardize against the SAME day's empty-room baseline."""
    amp_z = (amplitude - baseline.amp_mean) / (baseline.amp_std + EPS)
    phase_z = (phase - baseline.phase_mean) / (baseline.phase_std + EPS)
    return amp_z, phase_z


def apply_variant_b(
    amplitude: np.ndarray, phase: np.ndarray, baseline_today: DayBaseline, baseline_ref: DayBaseline,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-day re-referencing: map today's signal onto the reference day's empty-room frame."""
    amp_ref = (amplitude - baseline_today.amp_mean) * (baseline_ref.amp_std / (baseline_today.amp_std + EPS)) \
        + baseline_ref.amp_mean
    phase_ref = (phase - baseline_today.phase_mean) * (baseline_ref.phase_std / (baseline_today.phase_std + EPS)) \
        + baseline_ref.phase_mean
    return amp_ref, phase_ref


def apply_l1_gain_norm(amplitude: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Variant C: per-packet L1 gain normalization, `H_k / mean_k(|H_k|)` -- divide each packet's
    amplitude by its OWN mean magnitude across subcarriers, no empty-room baseline needed at all.

    Motivation (see ml/reports/day2_next_steps.md item 3): a 2026 cross-device CSI paper found ESP32
    per-subcarrier hardware profiles stable across days/channels (>0.9999 correlation) -- most of the
    Day1/Day2 drift measured in cross_day_drift.py is propagation/AGC, not hardware. Per-day Z-score
    (Variant A/B) recenters against a specific day's empty-room stats; L1 gain norm instead removes
    each packet's own AGC-driven overall gain, independent of any other session -- a different,
    complementary correction, not a replacement for A/B (can be composed: L1-normalize first, then
    still apply A/B on the normalized signal if desired)."""
    gain = amplitude.mean(axis=-1, keepdims=True)
    return amplitude / (gain + EPS), phase


def day1_proxy_baselines(manifest: pd.DataFrame, date: str, mode: str = "resampled"):
    """Pick the two Day-1 `none` sessions furthest apart in time as stand-ins for 'today' vs 'a prior
    day's' baseline -- a same-day mechanical check of the Variant B code path, superseded now that real
    Day 1 vs Day 2 baselines exist (see run_day2_sweep.py's Stage 0)."""
    none_rows = manifest[
        manifest["session_dir"].str.contains(f"/{date}/", regex=False) & (manifest["label"] == "none")
    ].sort_values("start_ts")
    if len(none_rows) < 2:
        raise ValueError(f"need >=2 `none` sessions on {date} for a proxy baseline pair, found {len(none_rows)}")

    earliest, latest = none_rows.iloc[0], none_rows.iloc[-1]
    baseline_early = compute_day_baseline_from_sessions([earliest["session_dir"]], date + "_early", mode)
    baseline_late = compute_day_baseline_from_sessions([latest["session_dir"]], date + "_late", mode)
    return baseline_early, baseline_late


def compute_day_baseline_from_sessions(session_dirs: list[str], label: str, mode: str = "resampled",
                                        target_rate_hz: float | None = None) -> DayBaseline:
    """`target_rate_hz` is required for (and only for) mode="resampled_timenorm" -- so the empty-room
    baseline is computed over the SAME time-normalized representation the windows it calibrates use,
    rather than mixing a raw-rate baseline with time-normalized windows."""
    amp_sum = amp_sumsq = phase_sum = phase_sumsq = None
    n = 0
    for session_dir in session_dirs:
        cache_path = cache_session(session_dir, mode, target_rate_hz=target_rate_hz)
        with np.load(cache_path) as d:
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
    amp_mean = amp_sum / n
    amp_std = np.sqrt(np.maximum(amp_sumsq / n - amp_mean ** 2, EPS))
    phase_mean = phase_sum / n
    phase_std = np.sqrt(np.maximum(phase_sumsq / n - phase_mean ** 2, EPS))
    return DayBaseline(
        date=label, source_sessions=session_dirs, n_packets=n,
        amp_mean=amp_mean.astype(np.float32), amp_std=amp_std.astype(np.float32),
        phase_mean=phase_mean.astype(np.float32), phase_std=phase_std.astype(np.float32),
    )


if __name__ == "__main__":
    from ml.data_pipeline.windowing import load_manifest

    manifest = load_manifest()
    date = manifest["session_dir"].iloc[0].split("/")[1]

    baseline = compute_day_baseline(manifest, date)
    print(f"Variant A baseline for {date}: {baseline.n_packets} packets from {baseline.source_sessions}")
    print(f"  amp_mean range: [{baseline.amp_mean.min():.2f}, {baseline.amp_mean.max():.2f}]")
    print(f"  amp_std  range: [{baseline.amp_std.min():.2f}, {baseline.amp_std.max():.2f}]")

    early, late = day1_proxy_baselines(manifest, date)
    print(f"\nDay-1 proxy baselines (Variant B mechanical check):")
    print(f"  early='{early.date}' ({early.source_sessions}), late='{late.date}' ({late.source_sessions})")
    print(f"  amp_mean delta (early vs late), mean abs: {np.abs(early.amp_mean - late.amp_mean).mean():.3f}")
    print(f"  amp_std  delta (early vs late), mean abs: {np.abs(early.amp_std - late.amp_std).mean():.3f}")
