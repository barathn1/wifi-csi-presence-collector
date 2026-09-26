"""TF-Mamba approximation (models/tfmamba_approx.py) on Day 3 (2026-09-15, channel 6/20MHz, native
128 subcarriers -- same capture mode as Day 2, see run_taskB_day3_evaluation.py), on the paper's
preprocessing/windowing recipe (build_person_windows / day_window_size, reused unchanged from
run_paper_2507_12854_replication.py and run_paper_2507_12854_cross_day.py), so these numbers sit
directly next to the existing tfmamba_approx_same_day / tfmamba_approx_cross_day entries in
experiment_log.csv for Day1/Day2.

1. Same-day on Day 3 alone (temporal 70/10/20 split per person, matches run_tfmamba_replication.py).
2. Cross-day: train on Day1+Day2 pooled -> test on Day3 (walking only -- the larger, more informative
   class; standing dropped to keep this run bounded, see below). Each day is z-scored against its OWN
   paper-preprocessed empty-room baseline (Variant-A style), then Day1 (186 sub, native) and Day2
   (128 sub, zero-padded to 186) are pooled for train; Day3 (128 sub) is zero-padded to 186 for test.

A first attempt at this ran all 5 combinations (same-day pooled/standing/walking + cross-day pooled
standing/walking) at EPOCHS=30 with un-subsampled windows and was killed after 4+ hours with the
process still stuck in the FIRST (smallest) run, ~9.5GB resident -- almost certainly swapping, since
MinimalMambaBlock's scan is a Python for-loop over T (mamba_block.py), and train_classifier prints
nothing until all epochs finish, so there was no way to see progress. This version bounds the risk:
window count is capped (subsample_windows) so arrays fit comfortably in memory, EPOCHS is cut to 6,
and scope is cut to the 2 most informative runs.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.windowing import load_manifest
from ml.models.tfmamba_approx import TFMambaApprox
from ml.training.train import train_classifier, log_rows
from ml.training.run_paper_2507_12854_replication import (
    build_person_windows, day_window_size, temporal_split, ArrayWindowDataset, SEED,
)
from ml.training.run_paper_2507_12854_cross_day import compute_baseline, zscore, pad_to, N_SUB_DAY1

DAY1, DAY2, DAY3 = "2026-09-09", "2026-09-10", "2026-09-15"
EPOCHS = 6           # cut from the paper-recipe default of 30 -- see module docstring
MAX_WINDOWS_PER_GROUP = 3000  # cap per (day, person) group to bound memory/runtime


def subsample_windows(amp: np.ndarray, phase: np.ndarray, cap: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    if len(amp) <= cap:
        return amp, phase
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(amp), size=cap, replace=False))
    return amp[idx], phase[idx]


def anjali_recall_of(model, test_ds, classes) -> float:
    model.eval()
    anjali_idx = classes.index("anjali")
    anjali_correct, anjali_total = 0, 0
    with torch.no_grad():
        for amp, phase, label in DataLoader(test_ds, batch_size=64):
            mask = label == anjali_idx
            if not mask.any():
                continue
            pred = model(amp, phase).argmax(dim=-1)
            anjali_correct += (pred[mask] == label[mask]).sum().item()
            anjali_total += mask.sum().item()
    return anjali_correct / anjali_total if anjali_total else float("nan")


def run_day3_same_day(motion_filter: str | None = None) -> dict:
    manifest = load_manifest()
    day_auth = manifest[(manifest["label"] == "authorized")
                         & (manifest["session_dir"].str.contains(f"/{DAY3}/", regex=False))]
    if motion_filter is not None:
        day_auth = day_auth[day_auth["motion"] == motion_filter]
    win_samples = day_window_size(manifest, DAY3)
    tag = motion_filter or "standing+walking"
    print(f"\n=== Day3 same-day ({tag}): TF-Mamba approx, window={win_samples} samples ===")

    per_person = {}
    for person in sorted(day_auth["person_id"].unique()):
        sessions = day_auth.loc[day_auth["person_id"] == person, "session_dir"].tolist()
        amp, phase = build_person_windows(sessions, win_samples)
        n_before = len(amp)
        amp, phase = subsample_windows(amp, phase, MAX_WINDOWS_PER_GROUP)
        print(f"  {person}: {len(sessions)} sessions -> {n_before} windows"
              + (f" (subsampled to {len(amp)})" if len(amp) < n_before else ""), flush=True)
        per_person[person] = (amp, phase)

    classes = sorted(per_person.keys())
    train_amp, train_phase, train_y = [], [], []
    test_amp, test_phase, test_y = [], [], []
    for person in classes:
        amp, phase = per_person[person]
        train_idx, test_idx = temporal_split(len(amp))
        train_amp.append(amp[train_idx]); train_phase.append(phase[train_idx])
        train_y.append(np.full(len(train_idx), classes.index(person)))
        test_amp.append(amp[test_idx]); test_phase.append(phase[test_idx])
        test_y.append(np.full(len(test_idx), classes.index(person)))

    train_ds = ArrayWindowDataset(np.concatenate(train_amp), np.concatenate(train_phase), np.concatenate(train_y))
    test_ds = ArrayWindowDataset(np.concatenate(test_amp), np.concatenate(test_phase), np.concatenate(test_y))
    n_sub = train_ds.amp.shape[-1]
    mb = (train_ds.amp.nbytes + train_ds.phase.nbytes + test_ds.amp.nbytes + test_ds.phase.nbytes) / 1e6
    print(f"  train={len(train_ds)}, test={len(test_ds)}, n_subcarriers={n_sub}, arrays~{mb:.0f}MB", flush=True)

    torch.manual_seed(SEED)
    model = TFMambaApprox(n_sub, len(classes))
    print(f"  training {EPOCHS} epochs...", flush=True)
    result = train_classifier(model, train_ds, test_ds, epochs=EPOCHS, seed=SEED)
    anjali_recall = anjali_recall_of(model, test_ds, classes)
    print(f"  done: accuracy={result['accuracy']*100:.1f}% anjali_recall={anjali_recall*100:.1f}%", flush=True)

    return {"split": "same_day", "day": DAY3, "motion": tag, "accuracy": result["accuracy"],
            "anjali_recall": anjali_recall, "n_train": len(train_ds), "n_test": len(test_ds)}


def run_cross_day_pooled(motion: str) -> dict | None:
    manifest = load_manifest()

    def auth(day):
        m = manifest[(manifest["label"] == "authorized") & (manifest["motion"] == motion)
                      & manifest["session_dir"].str.contains(f"/{day}/", regex=False)]
        return m

    def none_sessions(day):
        return sorted(manifest.loc[
            manifest["session_dir"].str.contains(f"/{day}/", regex=False) & (manifest["label"] == "none"),
            "session_dir",
        ])

    day1_auth, day2_auth, day3_auth = auth(DAY1), auth(DAY2), auth(DAY3)
    if day1_auth.empty or day2_auth.empty or day3_auth.empty:
        print(f"  cross-day pooled ({motion}): skipped, missing data for one of the three days")
        return None

    # Day1 and Day2 must share ONE window length to be pooled into a single train array
    # (concatenate requires matching shapes) -- day_window_size returns a day-specific value
    # derived from that day's own packet rate, so pick the smaller of the two for both. Baseline
    # mean/std are computed over flattened samples regardless of window chunk size, so recomputing
    # each day's baseline at win_train instead of its "native" day_window_size value doesn't change
    # the statistics, only how the packets happen to be chunked.
    win_train = min(day_window_size(manifest, DAY1), day_window_size(manifest, DAY2))
    win3 = day_window_size(manifest, DAY3)
    baseline1 = compute_baseline(none_sessions(DAY1), win_train)
    baseline2 = compute_baseline(none_sessions(DAY2), win_train)
    baseline3 = compute_baseline(none_sessions(DAY3), win3)

    classes = sorted(set(day1_auth["person_id"]) | set(day2_auth["person_id"]) | set(day3_auth["person_id"]))
    print(f"\n=== cross-day pooled ({motion}): Day1+Day2(win={win_train}) -> Day3(win={win3}) ===")

    train_amp, train_phase, train_y = [], [], []
    for day_auth, win, baseline, day in [(day1_auth, win_train, baseline1, DAY1), (day2_auth, win_train, baseline2, DAY2)]:
        for person in classes:
            sessions = day_auth.loc[day_auth["person_id"] == person, "session_dir"].tolist()
            if not sessions:
                continue
            amp, phase = build_person_windows(sessions, win)
            n_before = len(amp)
            amp, phase = subsample_windows(amp, phase, MAX_WINDOWS_PER_GROUP)
            amp_z, phase_z = zscore(amp, phase, baseline)
            amp_z, phase_z = pad_to(amp_z, phase_z, N_SUB_DAY1)
            train_amp.append(amp_z); train_phase.append(phase_z)
            train_y.append(np.full(len(amp_z), classes.index(person)))
            print(f"  {day} {person}: {len(sessions)} sessions -> {n_before} windows"
                  + (f" (subsampled to {len(amp_z)})" if len(amp_z) < n_before else ""), flush=True)

    test_amp, test_phase, test_y = [], [], []
    for person in classes:
        sessions = day3_auth.loc[day3_auth["person_id"] == person, "session_dir"].tolist()
        if not sessions:
            continue
        amp, phase = build_person_windows(sessions, win3)
        n_before = len(amp)
        amp, phase = subsample_windows(amp, phase, MAX_WINDOWS_PER_GROUP)
        amp_z, phase_z = zscore(amp, phase, baseline3)
        amp_z, phase_z = pad_to(amp_z, phase_z, N_SUB_DAY1)
        test_amp.append(amp_z); test_phase.append(phase_z)
        test_y.append(np.full(len(amp_z), classes.index(person)))
        print(f"  {DAY3} {person}: {len(sessions)} sessions -> {n_before} windows"
              + (f" (subsampled to {len(amp_z)})" if len(amp_z) < n_before else ""), flush=True)

    train_ds = ArrayWindowDataset(np.concatenate(train_amp).astype(np.float32),
                                   np.concatenate(train_phase).astype(np.float32), np.concatenate(train_y))
    test_ds = ArrayWindowDataset(np.concatenate(test_amp).astype(np.float32),
                                  np.concatenate(test_phase).astype(np.float32), np.concatenate(test_y))
    mb = (train_ds.amp.nbytes + train_ds.phase.nbytes + test_ds.amp.nbytes + test_ds.phase.nbytes) / 1e6
    print(f"  train={len(train_ds)}, test={len(test_ds)}, n_subcarriers={N_SUB_DAY1}, arrays~{mb:.0f}MB", flush=True)
    print(f"  train class balance: {np.bincount(train_ds.labels)}, test class balance: {np.bincount(test_ds.labels)}", flush=True)

    torch.manual_seed(SEED)
    model = TFMambaApprox(N_SUB_DAY1, len(classes))
    print(f"  training {EPOCHS} epochs...", flush=True)
    result = train_classifier(model, train_ds, test_ds, epochs=EPOCHS, seed=SEED)
    anjali_recall = anjali_recall_of(model, test_ds, classes)
    print(f"  done: accuracy={result['accuracy']*100:.1f}% anjali_recall={anjali_recall*100:.1f}%", flush=True)

    return {"split": "cross_day_pooled", "day": f"{DAY1}+{DAY2}->{DAY3}", "motion": motion,
            "accuracy": result["accuracy"], "anjali_recall": anjali_recall,
            "n_train": len(train_ds), "n_test": len(test_ds)}


def main() -> None:
    # Scope cut to 2 runs (same-day pooled, cross-day pooled on walking only -- the larger class)
    # after a first attempt at all 5 combinations got stuck for 4+ hours; see module docstring.
    results = []
    results.append(run_day3_same_day())
    cross = run_cross_day_pooled("walking")
    if cross:
        results.append(cross)

    print("\n=== TF-Mamba approximation, Day 3 evaluation ===")
    print(f"{'split':<18}{'day':<28}{'motion':<18}{'accuracy':>10}{'anjali_recall':>16}")
    for r in results:
        print(f"{r['split']:<18}{r['day']:<28}{r['motion']:<18}{r['accuracy']*100:>9.1f}%{r['anjali_recall']*100:>15.1f}%")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": f"tfmamba_approx_day3_{r['split']}",
        "task": "taskB_identity",
        "preprocessing": "paper_hampel_butterworth_phasecal" if r["split"] == "same_day" else "paper_recipe_variantA_zeropad",
        "model": "tfmamba_approx",
        "split_type": "temporal_70_10_20_per_person" if r["split"] == "same_day" else "day_disjoint_per_motion",
        "fold": 0, "accuracy": r["accuracy"], "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": f"anjali_recall={r['anjali_recall']:.4f}; day={r['day']}; motion={r['motion']}; NOT a verified TF-Mamba replication (IEEE Xplore inaccessible)",
    } for r in results])
    print("\nlogged to", REPO_ROOT / "ml/evaluation/results/experiment_log.csv")


if __name__ == "__main__":
    main()
