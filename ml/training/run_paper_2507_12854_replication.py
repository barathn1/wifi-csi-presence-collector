"""Replicates arXiv:2507.12854's exact recipe -- preprocessing (paper_2507_12854_preprocessing.py),
1s/50%-overlap windowing, the paper's own temporal 70/10/20 split methodology, and its exact
DualBranchTransformer architecture (already in models/transformer_dualbranch.py, matches the paper's
d_model=32/4 heads/d_ff=64/dropout=0.2) -- on THIS project's anjali-vs-barath data, pooling standing +
walking motion the way the paper pools its 6 orientations (motion is this dataset's analog of their
orientation variable: same subject, same session, different physical configuration).

Run separately per day: Day 1 sessions are native 186 subcarriers, Day 2 are native 128 (see
run_taskB_day_to_day.py's docstring for the HT40/HT20 finding) -- this is a same-day replication,
not a cross-day one, matching the paper's own single-session setup (they never cross days either).

Deviation from the paper, stated plainly: no validation-based early stopping (their "max 50 epochs,
patience 10") -- this trains a fixed 30 epochs instead, and the middle 10% of each person's timeline
(their validation slice) is held out of both train and test rather than used for early stopping.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.windowing import load_manifest
from ml.data_pipeline.paper_2507_12854_preprocessing import preprocess_session
from ml.models.transformer_dualbranch import DualBranchTransformer
from ml.training.train import train_classifier, log_rows
from ml.training.run_taskB_same_day import cache_session_native

EPOCHS = 30
SEED = 0
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10  # held out of both train and test, matching the paper's split proportions


class ArrayWindowDataset(Dataset):
    def __init__(self, amp: np.ndarray, phase: np.ndarray, labels: np.ndarray):
        self.amp, self.phase, self.labels = amp, phase, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.amp[i]), torch.from_numpy(self.phase[i]), int(self.labels[i])


def build_person_windows(session_dirs: list[str], win_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate one person's sessions (in session_dir sort order = chronological, both motions
    pooled) into windows, IN ORDER, so a temporal train/val/test split is meaningful. `win_samples` is
    fixed at the DAY level (not recomputed per session) -- each session's real packet rate jitters
    slightly, and windows must all be the same length to stack into one array."""
    amp_windows, phase_windows = [], []
    stride = max(1, win_samples // 2)
    for sd in sorted(session_dirs):
        cache_path = cache_session_native(sd)
        with np.load(cache_path) as d:
            amp_raw, phase_raw, t = d["amplitude"], d["phase"], d["device_time_us"]
        rate_hz = len(t) / ((t[-1] - t[0]) / 1e6)
        amp_p, phase_p, eff_rate = preprocess_session(amp_raw, phase_raw, rate_hz)
        for s in range(0, len(amp_p) - win_samples + 1, stride):
            amp_windows.append(amp_p[s:s + win_samples])
            phase_windows.append(phase_p[s:s + win_samples])
    return np.stack(amp_windows), np.stack(phase_windows)


def day_window_size(manifest: pd.DataFrame, day: str) -> int:
    """Canonical 1-second window size (in samples) for a day, from that day's average native packet
    rate, halved by the paper's temporal-mean-reduction step."""
    day_auth = manifest[(manifest["label"] == "authorized")
                         & (manifest["session_dir"].str.contains(f"/{day}/", regex=False))]
    avg_rate = (day_auth["sample_count"].astype(float) / day_auth["duration_s"].astype(float)).mean()
    return max(1, round(avg_rate / 2.0))


def temporal_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_idx = np.arange(0, n_train)
    test_idx = np.arange(n_train + n_val, n)
    return train_idx, test_idx


def run_day(day: str, motion_filter: str | None = None) -> dict:
    manifest = load_manifest()
    day_auth = manifest[(manifest["label"] == "authorized")
                         & (manifest["session_dir"].str.contains(f"/{day}/", regex=False))]
    if motion_filter is not None:
        day_auth = day_auth[day_auth["motion"] == motion_filter]
    win_samples = day_window_size(manifest, day)
    tag = motion_filter or "standing+walking"
    print(f"\n=== {day} ({tag}): preprocessing + windowing (paper recipe), window={win_samples} samples ===")

    per_person = {}
    for person in sorted(day_auth["person_id"].unique()):
        sessions = day_auth.loc[day_auth["person_id"] == person, "session_dir"].tolist()
        motions = day_auth.loc[day_auth["person_id"] == person, "motion"].tolist()
        print(f"  {person}: {len(sessions)} sessions, motions={sorted(set(motions))}")
        amp, phase = build_person_windows(sessions, win_samples)
        print(f"    -> {len(amp)} windows, shape={amp.shape[1:]}")
        per_person[person] = (amp, phase)

    train_amp, train_phase, train_y = [], [], []
    test_amp, test_phase, test_y = [], [], []
    classes = sorted(per_person.keys())
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
    print(f"  train={len(train_ds)} windows, test={len(test_ds)} windows, n_subcarriers={n_sub}")
    print(f"  train class balance: {np.bincount(train_ds.labels)}, test class balance: {np.bincount(test_ds.labels)}")

    torch.manual_seed(SEED)
    model = DualBranchTransformer(n_sub, len(classes))
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

    return {"day": day, "motion": motion_filter or "standing+walking", "accuracy": result["accuracy"],
            "anjali_recall": anjali_recall, "n_train": len(train_ds), "n_test": len(test_ds)}


def main() -> None:
    days = ["2026-09-09", "2026-09-10"]
    results = [run_day(day) for day in days]
    results += [run_day(day, motion_filter=m) for day in days for m in ("standing", "walking")]

    print("\n=== arXiv:2507.12854 replication (paper preprocessing + windowing + split, same-day) ===")
    print(f"{'day':<12}{'motion':<18}{'accuracy':>10}{'anjali_recall':>16}{'n_train':>10}{'n_test':>10}")
    for r in results:
        print(f"{r['day']:<12}{r['motion']:<18}{r['accuracy']*100:>9.1f}%{r['anjali_recall']*100:>15.1f}%{r['n_train']:>10}{r['n_test']:>10}")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "paper_2507_12854_replication",
        "task": "taskB_identity", "preprocessing": "paper_hampel_butterworth_phasecal", "model": "transformer_dualbranch",
        "split_type": "temporal_70_10_20_per_person", "fold": 0, "accuracy": r["accuracy"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": f"anjali_recall={r['anjali_recall']:.4f}; day={r['day']}; motion={r['motion']}",
    } for r in results])
    print("\nlogged to", REPO_ROOT / "ml/evaluation/results/experiment_log.csv")


if __name__ == "__main__":
    main()
