"""Day 2 (2026-09-10) is 128 native subcarriers throughout (occupied AND empty-room sessions) --
so unlike the Day1<->Day2 cross-day comparison, no cropping is needed here. This applies Variant-A
self-calibration (data_pipeline/calibration.py: z-score each occupied window against Day 2's OWN
empty-room ('none') baseline, per subcarrier) to the Day 2 anjali-vs-barath windows, then reruns the
identical session-disjoint 5-fold pooled-CV identification sweep from run_taskB_same_day.py, so the
calibrated numbers are directly comparable to the uncalibrated Day 2 baseline already on record in
experiment_log.csv (stage=taskB_same_day).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.windowing import load_manifest
from ml.data_pipeline.splits import session_disjoint_kfold, assert_no_group_leakage
from ml.data_pipeline.calibration import DayBaseline, apply_variant_a
from ml.models.baselines import make_baseline, make_svm
from ml.models.cnn1d import CNN1DDualBranch
from ml.models.lstm import LSTMDualBranch, BiLSTMDualBranch
from ml.models.transformer_whofi import WhoFiTransformer
from ml.training.train import log_rows
from ml.training.run_taskB_day_to_day import run_classical, run_deep
from ml.training.run_taskB_same_day import cache_session_native, build_day_window_index, MODEL_SPECS, N_SPLITS, SEED

DAY = "2026-09-10"
EPS = 1e-6


def compute_native_baseline(session_dirs: list[str], label: str) -> DayBaseline:
    amp_sum = amp_sumsq = phase_sum = phase_sumsq = None
    n = 0
    for sd in session_dirs:
        cache_path = cache_session_native(sd)
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


def main() -> None:
    manifest = load_manifest()
    none_sessions = sorted(manifest.loc[
        manifest["session_dir"].str.contains(f"/{DAY}/", regex=False) & (manifest["label"] == "none"),
        "session_dir",
    ])
    print(f"computing Day 2 empty-room baseline from {len(none_sessions)} 'none' sessions...")
    baseline = compute_native_baseline(none_sessions, DAY)
    print(f"  {baseline.n_packets} packets, {baseline.amp_mean.shape[0]} subcarriers, "
          f"amp_mean range=[{baseline.amp_mean.min():.2f}, {baseline.amp_mean.max():.2f}], "
          f"amp_std range=[{baseline.amp_std.min():.2f}, {baseline.amp_std.max():.2f}]")

    def calib_fn(amp, phase, row):
        return apply_variant_a(amp, phase, baseline)

    print(f"\n=== {DAY}: building same-day (native, Variant-A recalibrated) taskB window index ===")
    window_index, n_sub = build_day_window_index(DAY)
    print(f"total windows: {len(window_index)}, n_subcarriers={n_sub}")

    totals = {name: {"n_correct": 0, "n_test": 0, "n_anjali_correct": 0, "n_anjali_total": 0}
              for name, _, _ in MODEL_SPECS}

    for fold, (train_idx, test_idx) in enumerate(session_disjoint_kfold(window_index, n_splits=N_SPLITS, seed=SEED)):
        assert_no_group_leakage(window_index, train_idx, test_idx, "session_dir")
        test_classes = window_index.iloc[test_idx]["person_id"].value_counts().to_dict()
        print(f"  fold {fold}: train={len(train_idx)} test={len(test_idx)} test_classes={test_classes}")
        for name, kind, factory in MODEL_SPECS:
            if kind == "classical":
                r = run_classical(window_index, train_idx, test_idx, name, factory, calibration=calib_fn)
            else:
                r = run_deep(window_index, train_idx, test_idx, name, factory, n_sub, calibration=calib_fn)
            t = totals[name]
            t["n_correct"] += r["n_correct"]
            t["n_test"] += r["n_test"]
            t["n_anjali_correct"] += r["n_anjali_correct"]
            t["n_anjali_total"] += r["n_anjali_total"]

    results = []
    for name, _, _ in MODEL_SPECS:
        t = totals[name]
        results.append({
            "model": name,
            "accuracy": t["n_correct"] / t["n_test"],
            "anjali_recall": t["n_anjali_correct"] / t["n_anjali_total"] if t["n_anjali_total"] else float("nan"),
            "n_test": t["n_test"],
        })

    log_path = REPO_ROOT / "ml/evaluation/results/experiment_log.csv"
    baseline_rows = {}
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        same_day = log_df[(log_df["stage"] == "taskB_same_day") & log_df["notes"].str.contains(f"day={DAY}", na=False)]
        for _, row in same_day.iterrows():
            baseline_rows[row["model"]] = row["accuracy"]

    print(f"\n=== {DAY}: Variant-A self-calibration vs uncalibrated (session-disjoint 5-fold pooled CV) ===")
    print(f"{'model':<20}{'uncalibrated acc':>18}{'calibrated acc':>18}{'calibrated anjali_recall':>26}")
    for r in results:
        base = baseline_rows.get(r["model"])
        base_str = f"{base*100:.1f}%" if base is not None else "n/a"
        print(f"{r['model']:<20}{base_str:>18}{r['accuracy']*100:>17.1f}%{r['anjali_recall']*100:>25.1f}%")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "taskB_day2_recalibrated",
        "task": "taskB_identity", "preprocessing": "variantA_selfcalib", "model": r["model"],
        "split_type": "session_disjoint_5fold_pooled", "fold": "all", "accuracy": r["accuracy"],
        "n_train": "", "n_test": r["n_test"],
        "notes": f"anjali_recall={r['anjali_recall']:.4f}; day={DAY}",
    } for r in results])
    print("\nlogged to", log_path)


if __name__ == "__main__":
    main()
