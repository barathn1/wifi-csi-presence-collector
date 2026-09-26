"""Cross-day (train Day1 -> test Day2) version of run_paper_2507_12854_replication.py: the paper's
exact preprocessing (Hampel + Butterworth + phase calibration + temporal mean reduction) and its
DualBranchTransformer architecture, now with Variant-A-style self-calibration added as the
normalization step to bridge the two days, since the paper itself never crosses days.

Per the same-day replication's finding (pooling standing+walking wrecks accuracy even within one
day), this evaluates cross-day PER MOTION (standing-only, walking-only) rather than pooled -- pooling
would confound two already-hard problems (motion-mixing failure + day-shift failure) into one number.

Normalization: after the full paper preprocessing, z-score each day's amplitude/phase against that
SAME day's own paper-preprocessed empty-room ('none') baseline (Variant A style, computed on the
identically-preprocessed representation, not raw). This makes the two days comparable without forcing
either onto the other's raw scale (Variant B), and without picking an arbitrary reference frame.

Day 1 is native 186 subcarriers, Day 2 is native 128 -- no cropping (per explicit prior instruction):
Day 2's z-scored 128 channels are padded with zeros for the 58 channels Day 1 has and Day 2 doesn't.
Zero is the principled neutral fill here (not an arbitrary constant) because in z-scored space, 0
means exactly "at baseline" -- unlike padding raw amplitude, which has no such natural neutral value.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.windowing import load_manifest
from ml.models.transformer_dualbranch import DualBranchTransformer
from ml.training.train import train_classifier, log_rows
from ml.training.run_taskB_same_day import cache_session_native
from ml.training.run_paper_2507_12854_replication import (
    build_person_windows, temporal_split, ArrayWindowDataset, EPOCHS, SEED,
)

N_SUB_DAY1 = 186
N_SUB_DAY2 = 128


def day_window_size(sessions: pd.DataFrame) -> int:
    avg_rate = (sessions["sample_count"].astype(float) / sessions["duration_s"].astype(float)).mean()
    return max(1, round(avg_rate / 2.0))


def compute_baseline(session_dirs: list[str], win_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean/std of the paper-preprocessed amplitude/phase, pooled across every window from these
    (empty-room) sessions -- i.e. Variant-A's baseline, but on the fully-preprocessed representation."""
    amp, phase = build_person_windows(session_dirs, win_samples)  # (n_windows, win, n_sub)
    flat_amp = amp.reshape(-1, amp.shape[-1])
    flat_phase = phase.reshape(-1, phase.shape[-1])
    return (flat_amp.mean(0), flat_amp.std(0) + 1e-6,
            flat_phase.mean(0), flat_phase.std(0) + 1e-6)


def zscore(amp: np.ndarray, phase: np.ndarray, baseline) -> tuple[np.ndarray, np.ndarray]:
    amp_mean, amp_std, phase_mean, phase_std = baseline
    return (amp - amp_mean) / amp_std, (phase - phase_mean) / phase_std


def pad_to(amp: np.ndarray, phase: np.ndarray, n_target: int) -> tuple[np.ndarray, np.ndarray]:
    n_pad = n_target - amp.shape[-1]
    if n_pad <= 0:
        return amp, phase
    pad_shape = amp.shape[:-1] + (n_pad,)
    return (np.concatenate([amp, np.zeros(pad_shape, dtype=amp.dtype)], axis=-1),
            np.concatenate([phase, np.zeros(pad_shape, dtype=phase.dtype)], axis=-1))


def run_motion(motion: str, win1: int, win2: int, baseline1, baseline2) -> dict:
    manifest = load_manifest()
    train_date, test_date = "2026-09-09", "2026-09-10"

    day1_auth = manifest[(manifest["label"] == "authorized") & (manifest["motion"] == motion)
                          & manifest["session_dir"].str.contains(f"/{train_date}/", regex=False)]
    day2_auth = manifest[(manifest["label"] == "authorized") & (manifest["motion"] == motion)
                          & manifest["session_dir"].str.contains(f"/{test_date}/", regex=False)]

    print(f"\n=== motion={motion}: {train_date} win={win1} ({N_SUB_DAY1} sub) -> "
          f"{test_date} win={win2} ({N_SUB_DAY2} sub) ===")

    classes = sorted(set(day1_auth["person_id"]) | set(day2_auth["person_id"]))
    train_amp, train_phase, train_y = [], [], []
    for person in classes:
        sessions = day1_auth.loc[day1_auth["person_id"] == person, "session_dir"].tolist()
        if not sessions:
            continue
        amp, phase = build_person_windows(sessions, win1)
        amp_z, phase_z = zscore(amp, phase, baseline1)
        train_amp.append(amp_z); train_phase.append(phase_z)
        train_y.append(np.full(len(amp_z), classes.index(person)))
        print(f"  {train_date} {person}: {len(sessions)} sessions -> {len(amp_z)} windows")

    test_amp, test_phase, test_y = [], [], []
    for person in classes:
        sessions = day2_auth.loc[day2_auth["person_id"] == person, "session_dir"].tolist()
        if not sessions:
            continue
        amp, phase = build_person_windows(sessions, win2)
        amp_z, phase_z = zscore(amp, phase, baseline2)
        amp_z, phase_z = pad_to(amp_z, phase_z, N_SUB_DAY1)
        test_amp.append(amp_z); test_phase.append(phase_z)
        test_y.append(np.full(len(amp_z), classes.index(person)))
        print(f"  {test_date} {person}: {len(sessions)} sessions -> {len(amp_z)} windows")

    train_ds = ArrayWindowDataset(np.concatenate(train_amp).astype(np.float32),
                                   np.concatenate(train_phase).astype(np.float32), np.concatenate(train_y))
    test_ds = ArrayWindowDataset(np.concatenate(test_amp).astype(np.float32),
                                  np.concatenate(test_phase).astype(np.float32), np.concatenate(test_y))
    print(f"  train={len(train_ds)} windows (win={win1}), test={len(test_ds)} windows (win={win2}), "
          f"n_subcarriers={N_SUB_DAY1}")
    print(f"  train class balance: {np.bincount(train_ds.labels)}, test class balance: {np.bincount(test_ds.labels)}")

    torch.manual_seed(SEED)
    model = DualBranchTransformer(N_SUB_DAY1, len(classes))
    result = train_classifier(model, train_ds, test_ds, epochs=EPOCHS, seed=SEED)

    anjali_idx = classes.index("anjali")
    model.eval()
    from torch.utils.data import DataLoader
    anjali_correct, anjali_total = 0, 0
    with torch.no_grad():
        for amp, phase, label in DataLoader(test_ds, batch_size=64):
            mask = label == anjali_idx
            if not mask.any():
                continue
            pred = model(amp, phase).argmax(dim=-1)
            anjali_correct += (pred[mask] == label[mask]).sum().item()
            anjali_total += mask.sum().item()
    anjali_recall = anjali_correct / anjali_total if anjali_total else float("nan")

    return {"motion": motion, "accuracy": result["accuracy"], "anjali_recall": anjali_recall,
            "n_train": len(train_ds), "n_test": len(test_ds)}


def main() -> None:
    manifest = load_manifest()
    train_date, test_date = "2026-09-09", "2026-09-10"
    day1_auth_all = manifest[(manifest["label"] == "authorized")
                              & manifest["session_dir"].str.contains(f"/{train_date}/", regex=False)]
    day2_auth_all = manifest[(manifest["label"] == "authorized")
                              & manifest["session_dir"].str.contains(f"/{test_date}/", regex=False)]
    day1_none = manifest[(manifest["label"] == "none")
                          & manifest["session_dir"].str.contains(f"/{train_date}/", regex=False)]
    day2_none = manifest[(manifest["label"] == "none")
                          & manifest["session_dir"].str.contains(f"/{test_date}/", regex=False)]

    win1 = day_window_size(day1_auth_all)
    win2 = day_window_size(day2_auth_all)
    print(f"computing {train_date} empty-room baseline (paper-preprocessed, {len(day1_none)} sessions, win={win1})...")
    baseline1 = compute_baseline(sorted(day1_none["session_dir"]), win1)
    print(f"computing {test_date} empty-room baseline (paper-preprocessed, {len(day2_none)} sessions, win={win2})...")
    baseline2 = compute_baseline(sorted(day2_none["session_dir"]), win2)

    results = [run_motion(m, win1, win2, baseline1, baseline2) for m in ("standing", "walking")]

    print("\n=== arXiv:2507.12854 recipe, cross-day (train Day1 -> test Day2), Variant-A self-calib + zero-pad ===")
    print(f"{'motion':<12}{'accuracy':>10}{'anjali_recall':>16}{'n_train':>10}{'n_test':>10}")
    for r in results:
        print(f"{r['motion']:<12}{r['accuracy']*100:>9.1f}%{r['anjali_recall']*100:>15.1f}%{r['n_train']:>10}{r['n_test']:>10}")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "paper_2507_12854_cross_day",
        "task": "taskB_identity", "preprocessing": "paper_recipe_variantA_zeropad", "model": "transformer_dualbranch",
        "split_type": "day_disjoint_per_motion", "fold": 0, "accuracy": r["accuracy"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": f"anjali_recall={r['anjali_recall']:.4f}; motion={r['motion']}; train=2026-09-09 test=2026-09-10",
    } for r in results])
    print("\nlogged to", REPO_ROOT / "ml/evaluation/results/experiment_log.csv")


if __name__ == "__main__":
    main()
