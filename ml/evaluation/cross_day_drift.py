"""Stage 0 EDA (see plan): does the empty-room-based cross-day calibration (Variant B) actually shrink
the real cross-day gap, or is it not worth the sweep budget? Answered directly from data, not assumed.

Day 1 and Day 2's `none` (empty-room) baselines are genuinely different days now (previously only a
same-day proxy existed, see calibration.py's day1_proxy_baselines). This script:
1. Reports the RAW empty-room drift itself (Day1 `none` vs Day2 `none`, per-subcarrier amp/phase mean
   and std) -- how different are the two days' quiet-room baselines to begin with.
2. For each occupied class (authorized, unauthorized), reports the RAW cross-day gap (Day1's mean
   amplitude/phase for that class vs Day2's), then the gap AFTER applying Variant B to re-reference
   Day2's class mean onto Day1's empty-room frame -- exploits linearity: since apply_variant_b is an
   affine transform, applying it to the class MEAN gives the same result as applying it to every
   packet then re-averaging (mean commutes with an affine map), so no per-packet pass is needed here.

A meaningful shrink in step 2's gap after calibration is the empirical case FOR spending sweep budget
on calibB; a negligible shrink is the case against it.
"""
from __future__ import annotations

import numpy as np

from ml.data_pipeline.calibration import apply_variant_b, compute_day_baseline, compute_day_baseline_from_sessions
from ml.data_pipeline.windowing import load_manifest

MODE = "resampled"


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def main() -> None:
    manifest = load_manifest()
    dates = sorted(manifest["session_dir"].str.split("/").str[1].unique())
    assert len(dates) == 2, f"expected exactly 2 dates, found {dates}"
    day1, day2 = dates  # day1 = earlier date = the reference day

    none_base = {d: compute_day_baseline(manifest, d, mode=MODE) for d in dates}

    print(f"=== raw empty-room drift: {day1} vs {day2} ===")
    print(f"  amp_mean  |diff| mean={_mean_abs(none_base[day1].amp_mean, none_base[day2].amp_mean):.4f} "
          f"max={np.abs(none_base[day1].amp_mean - none_base[day2].amp_mean).max():.4f}")
    print(f"  amp_std   |diff| mean={_mean_abs(none_base[day1].amp_std, none_base[day2].amp_std):.4f}")
    print(f"  phase_mean|diff| mean={_mean_abs(none_base[day1].phase_mean, none_base[day2].phase_mean):.4f} rad")
    print(f"  phase_std |diff| mean={_mean_abs(none_base[day1].phase_std, none_base[day2].phase_std):.4f} rad")
    print(f"  (from {none_base[day1].n_packets} vs {none_base[day2].n_packets} empty-room packets)")

    for label in ("authorized", "unauthorized"):
        sessions = {
            d: manifest.loc[
                manifest["session_dir"].str.contains(f"/{d}/", regex=False) & (manifest["label"] == label),
                "session_dir",
            ].tolist()
            for d in dates
        }
        class_base = {d: compute_day_baseline_from_sessions(sessions[d], f"{d}_{label}", mode=MODE) for d in dates}

        raw_gap_amp = _mean_abs(class_base[day1].amp_mean, class_base[day2].amp_mean)
        raw_gap_phase = _mean_abs(class_base[day1].phase_mean, class_base[day2].phase_mean)

        calib_amp_mean, calib_phase_mean = apply_variant_b(
            class_base[day2].amp_mean.reshape(1, -1), class_base[day2].phase_mean.reshape(1, -1),
            baseline_today=none_base[day2], baseline_ref=none_base[day1],
        )
        calib_gap_amp = _mean_abs(class_base[day1].amp_mean, calib_amp_mean.ravel())
        calib_gap_phase = _mean_abs(class_base[day1].phase_mean, calib_phase_mean.ravel())

        print(f"\n=== {label}: cross-day gap, {day1} vs {day2} ({len(sessions[day1])}+{len(sessions[day2])} sessions) ===")
        print(f"  amplitude:  raw gap={raw_gap_amp:.4f}  after Variant B={calib_gap_amp:.4f}  "
              f"({'shrank' if calib_gap_amp < raw_gap_amp else 'GREW'} "
              f"{100 * (1 - calib_gap_amp / max(raw_gap_amp, 1e-9)):.1f}%)")
        print(f"  phase:      raw gap={raw_gap_phase:.4f}  after Variant B={calib_gap_phase:.4f}  "
              f"({'shrank' if calib_gap_phase < raw_gap_phase else 'GREW'} "
              f"{100 * (1 - calib_gap_phase / max(raw_gap_phase, 1e-9)):.1f}%)")


if __name__ == "__main__":
    main()
