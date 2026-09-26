"""Cross-day (train Day1 -> test Day2) taskB_identity, WITHOUT cropping any subcarriers -- the
alternative to run_taskB_day_to_day.py's crop-to-128 approach, per explicit request.

Day 1 is native 186 subcarriers, Day 2 is native 128 (see run_taskB_day_to_day.py's docstring for
the HT40/HT20 finding). Cropping Day 1 down to 128 throws away 58 measured channels; instead this:

1. Keeps Day 1 (train) at its full native 186 width, untouched.
2. Applies Variant B cross-day re-referencing (data_pipeline/calibration.py) to Day 2 (test): maps
   Day 2's 128 MEASURED channels onto Day 1's empty-room reference scale for those same channel
   indices (`calibrated = (raw - mu_day2_empty) * (sigma_day1_empty / sigma_day2_empty) + mu_day1_empty`).
   This assumes Day 1's first 128 native subcarriers are the same physical frequency bins as Day 2's
   128 (HT40 plausibly = HT20's primary channel + extra secondary-channel subcarriers appended) --
   not proven, flagged as an assumption.
3. Pads Day 2's remaining 58 channels (which Day 2's hardware never measured, in EITHER mode) with
   Day 1's own empty-room mean for those channels -- the Variant-B-consistent "no information, assume
   at today's baseline" fill, not an arbitrary zero. Nothing measured is discarded; only channels Day 2
   never captured are filled with a neutral value.

Both days end up at a shared 186-wide representation with zero cropping of anything Day 2 actually
recorded, so the existing 186-subcarrier model zoo can be reused directly.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.windowing import load_manifest
from ml.data_pipeline.splits import day_disjoint_split, assert_no_group_leakage
from ml.data_pipeline.tasks import taskB_identity
from ml.data_pipeline.calibration import DayBaseline, apply_variant_b
from ml.models.baselines import make_baseline, make_svm
from ml.models.cnn1d import CNN1DDualBranch
from ml.models.lstm import LSTMDualBranch, BiLSTMDualBranch
from ml.models.transformer_whofi import WhoFiTransformer
from ml.training.train import log_rows
from ml.training.run_taskB_day_to_day import run_classical, run_deep, WINDOW_PACKETS, STRIDE_PACKETS
from ml.training.run_taskB_same_day import cache_session_native
from ml.training.run_taskB_day2_recalibrated import compute_native_baseline

N_SUB_DAY1 = 186
N_SUB_DAY2 = 128


def build_all_days_native_window_index() -> pd.DataFrame:
    """Like run_taskB_day_to_day.build_taskB_window_index, but each session is cached at its OWN
    native subcarrier width (via cache_session_native) instead of a fixed crop -- so Day 1 rows carry
    186-wide packets and Day 2 rows carry 128-wide packets straight out of the cache. The calibration
    function applied downstream is what reconciles the widths, not the caching step."""
    manifest = load_manifest()
    auth = manifest[manifest["label"] == "authorized"].reset_index(drop=True)

    rows = []
    for _, row in auth.iterrows():
        cache_path = cache_session_native(row["session_dir"])
        with np.load(cache_path, mmap_mode="r") as d:
            n = d["amplitude"].shape[0]
            device_time_us = d["device_time_us"]
            if n < WINDOW_PACKETS:
                print(f"  skipping {row['session_dir']}: only {n} packets < window size {WINDOW_PACKETS}")
                continue
            date = row["session_dir"].split("/")[1]
            for start in range(0, n - WINDOW_PACKETS + 1, STRIDE_PACKETS):
                end = start + WINDOW_PACKETS
                rows.append({
                    "cache_path": str(cache_path), "start": start, "end": end,
                    "session_dir": row["session_dir"], "label": row["label"],
                    "person_id": row["person_id"], "motion": row.get("motion", ""),
                    "date": date, "window_start_time_us": int(device_time_us[start]),
                })
    window_index = pd.DataFrame(rows)
    mask, _ = taskB_identity(window_index)
    assert mask.all(), "build_all_days_native_window_index should only ever contain authorized rows"
    return window_index


def main() -> None:
    manifest = load_manifest()
    dates = sorted(manifest.loc[manifest["label"] == "authorized", "session_dir"]
                    .str.split("/").str[1].unique())
    train_date, test_date = dates[0], dates[1]
    print(f"train date (native {N_SUB_DAY1} sub) = {train_date}, test date (native {N_SUB_DAY2} sub) = {test_date}")

    def none_sessions(day: str) -> list[str]:
        return sorted(manifest.loc[
            manifest["session_dir"].str.contains(f"/{day}/", regex=False) & (manifest["label"] == "none"),
            "session_dir",
        ])

    print(f"\ncomputing {train_date} empty-room baseline ({N_SUB_DAY1} sub)...")
    baseline_day1 = compute_native_baseline(none_sessions(train_date), train_date)
    print(f"  {baseline_day1.n_packets} packets, {baseline_day1.amp_mean.shape[0]} subcarriers")

    print(f"computing {test_date} empty-room baseline ({N_SUB_DAY2} sub)...")
    baseline_day2 = compute_native_baseline(none_sessions(test_date), test_date)
    print(f"  {baseline_day2.n_packets} packets, {baseline_day2.amp_mean.shape[0]} subcarriers")

    baseline_day1_common = DayBaseline(
        date=train_date + "_common128", source_sessions=baseline_day1.source_sessions,
        n_packets=baseline_day1.n_packets,
        amp_mean=baseline_day1.amp_mean[:N_SUB_DAY2], amp_std=baseline_day1.amp_std[:N_SUB_DAY2],
        phase_mean=baseline_day1.phase_mean[:N_SUB_DAY2], phase_std=baseline_day1.phase_std[:N_SUB_DAY2],
    )
    pad_amp_mean = baseline_day1.amp_mean[N_SUB_DAY2:]      # (58,) -- Variant-B-consistent neutral fill
    pad_phase_mean = baseline_day1.phase_mean[N_SUB_DAY2:]

    def calib_fn(amp: np.ndarray, phase: np.ndarray, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        if row["date"] == train_date:
            return amp, phase  # already native 186, in Day1's own frame by definition
        # Day 2: re-reference the 128 measured channels onto Day1's empty-room scale, then pad
        amp_ref, phase_ref = apply_variant_b(amp, phase, baseline_day2, baseline_day1_common)
        n = amp.shape[0]
        pad_amp = np.tile(pad_amp_mean, (n, 1))
        pad_phase = np.tile(pad_phase_mean, (n, 1))
        return np.concatenate([amp_ref, pad_amp], axis=1), np.concatenate([phase_ref, pad_phase], axis=1)

    print("\nbuilding cross-day (native-width, no crop) taskB window index...")
    window_index = build_all_days_native_window_index()
    print(f"total windows: {len(window_index)}")
    print(window_index.groupby(["date", "person_id"]).size())

    day_split = day_disjoint_split(window_index)
    if day_split is None:
        raise RuntimeError("need >1 date for day-disjoint split")
    split_test_date, train_idx, test_idx = day_split
    assert split_test_date == test_date
    assert_no_group_leakage(window_index, train_idx, test_idx, "session_dir")
    print(f"train={train_date} ({len(train_idx)} windows), test={test_date} ({len(test_idx)} windows)")

    results = []
    results.append(run_classical(window_index, train_idx, test_idx, "random_forest", make_baseline, calibration=calib_fn))
    results.append(run_classical(window_index, train_idx, test_idx, "svm", make_svm, calibration=calib_fn))
    results.append(run_deep(window_index, train_idx, test_idx, "cnn1d", lambda n, c: CNN1DDualBranch(n, c), N_SUB_DAY1, calib_fn))
    results.append(run_deep(window_index, train_idx, test_idx, "lstm", lambda n, c: LSTMDualBranch(n, c), N_SUB_DAY1, calib_fn))
    results.append(run_deep(window_index, train_idx, test_idx, "bilstm", lambda n, c: BiLSTMDualBranch(n, c), N_SUB_DAY1, calib_fn))
    results.append(run_deep(window_index, train_idx, test_idx, "transformer_whofi", lambda n, c: WhoFiTransformer(n, c), N_SUB_DAY1, calib_fn))

    log_path = REPO_ROOT / "ml/evaluation/results/experiment_log.csv"
    crop_baseline = {}
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        crop_rows = log_df[log_df["stage"] == "taskB_day_to_day"]
        for _, row in crop_rows.iterrows():
            crop_baseline[row["model"]] = row["accuracy"]

    print("\n=== cross-day (train Day1 -> test Day2), crop-128 vs Variant-B no-crop ===")
    print(f"{'model':<20}{'crop128 acc':>14}{'no-crop acc':>14}{'no-crop anjali_recall':>24}")
    for r in results:
        base = crop_baseline.get(r["model"])
        base_str = f"{base*100:.1f}%" if base is not None else "n/a"
        print(f"{r['model']:<20}{base_str:>14}{r['accuracy']*100:>13.1f}%{r['anjali_recall']*100:>23.1f}%")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "taskB_day_to_day_variantB_nocrop",
        "task": "taskB_identity", "preprocessing": "variantB_nocrop", "model": r["model"],
        "split_type": "day_disjoint", "fold": 0, "accuracy": r["accuracy"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": f"anjali_recall={r['anjali_recall']:.4f}; train={train_date} test={test_date}",
    } for r in results])
    print("\nlogged to", log_path)


if __name__ == "__main__":
    main()
