"""Cross-day (train Day1 -> test Day2) evaluation of the Wi-Gait approximation (models/wigait_approx.py
+ data_pipeline/wigait_approx_preprocessing.py) -- walking-only (gait needs motion), on the same
anjali-vs-barath task used throughout this investigation.

Normalization: z-score each day's gait-cycle waveforms against that SAME day's own cycle population
(mean/std per subcarrier, computed once from Day 1's cycles and once from Day 2's) -- there's no
empty-room analog here since gait cycles only exist during walking, so this normalizes against the
walking data itself rather than a `none`-session baseline. Day 2 (128 native subcarriers) is then
zero-padded to Day 1's 186 -- zero is the neutral value in z-scored space, per the same reasoning used
in run_paper_2507_12854_cross_day.py.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.windowing import load_manifest
from ml.data_pipeline.wigait_approx_preprocessing import extract_gait_cycles, CYCLE_LEN
from ml.models.wigait_approx import WiGaitApprox
from ml.training.train import train_classifier, log_rows
from ml.training.run_taskB_same_day import cache_session_native

EPOCHS = 30
SEED = 0
N_SUB_DAY1 = 186
N_SUB_TARGET = 186


class GaitCycleDataset(Dataset):
    def __init__(self, torso: np.ndarray, limb: np.ndarray, labels: np.ndarray):
        self.torso, self.limb, self.labels = torso, limb, labels  # (N, CYCLE_LEN, n_sub)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        torso_t = torch.from_numpy(self.torso[i]).transpose(0, 1)  # (n_sub, CYCLE_LEN)
        limb_t = torch.from_numpy(self.limb[i]).transpose(0, 1)
        return torso_t, limb_t, int(self.labels[i])


def person_cycles(sessions: list[str]) -> tuple[np.ndarray, np.ndarray]:
    torso_all, limb_all = [], []
    for sd in sorted(sessions):
        cache_path = cache_session_native(sd)
        with np.load(cache_path) as d:
            amp, t = d["amplitude"], d["device_time_us"]
        rate_hz = len(t) / ((t[-1] - t[0]) / 1e6)
        torso, limb = extract_gait_cycles(amp, rate_hz)
        if len(torso):
            torso_all.append(torso); limb_all.append(limb)
    if not torso_all:
        return np.empty((0, CYCLE_LEN, 0)), np.empty((0, CYCLE_LEN, 0))
    return np.concatenate(torso_all), np.concatenate(limb_all)


def zscore_and_pad(torso: np.ndarray, limb: np.ndarray, n_target: int) -> tuple[np.ndarray, np.ndarray]:
    t_mean, t_std = torso.mean(axis=(0, 1)), torso.std(axis=(0, 1)) + 1e-6
    l_mean, l_std = limb.mean(axis=(0, 1)), limb.std(axis=(0, 1)) + 1e-6
    torso_z = (torso - t_mean) / t_std
    limb_z = (limb - l_mean) / l_std
    n_pad = n_target - torso_z.shape[-1]
    if n_pad > 0:
        pad_shape = torso_z.shape[:-1] + (n_pad,)
        torso_z = np.concatenate([torso_z, np.zeros(pad_shape, dtype=np.float32)], axis=-1)
        limb_z = np.concatenate([limb_z, np.zeros(pad_shape, dtype=np.float32)], axis=-1)
    return torso_z.astype(np.float32), limb_z.astype(np.float32)


def main() -> None:
    manifest = load_manifest()
    train_date, test_date = "2026-09-09", "2026-09-10"

    day1 = manifest[(manifest["label"] == "authorized") & (manifest["motion"] == "walking")
                     & manifest["session_dir"].str.contains(f"/{train_date}/", regex=False)]
    day2 = manifest[(manifest["label"] == "authorized") & (manifest["motion"] == "walking")
                     & manifest["session_dir"].str.contains(f"/{test_date}/", regex=False)]

    classes = sorted(set(day1["person_id"]) | set(day2["person_id"]))
    print(f"classes: {classes}")

    train_torso, train_limb, train_y = [], [], []
    for person in classes:
        sessions = day1.loc[day1["person_id"] == person, "session_dir"].tolist()
        torso, limb = person_cycles(sessions)
        torso, limb = zscore_and_pad(torso, limb, N_SUB_TARGET)
        print(f"  {train_date} {person}: {len(sessions)} sessions -> {len(torso)} gait cycles")
        train_torso.append(torso); train_limb.append(limb)
        train_y.append(np.full(len(torso), classes.index(person)))

    test_torso, test_limb, test_y = [], [], []
    for person in classes:
        sessions = day2.loc[day2["person_id"] == person, "session_dir"].tolist()
        torso, limb = person_cycles(sessions)
        torso, limb = zscore_and_pad(torso, limb, N_SUB_TARGET)
        print(f"  {test_date} {person}: {len(sessions)} sessions -> {len(torso)} gait cycles")
        test_torso.append(torso); test_limb.append(limb)
        test_y.append(np.full(len(torso), classes.index(person)))

    train_ds = GaitCycleDataset(np.concatenate(train_torso), np.concatenate(train_limb), np.concatenate(train_y))
    test_ds = GaitCycleDataset(np.concatenate(test_torso), np.concatenate(test_limb), np.concatenate(test_y))
    print(f"train={len(train_ds)} cycles, test={len(test_ds)} cycles, n_subcarriers={N_SUB_TARGET}")
    print(f"train class balance: {np.bincount(train_ds.labels)}, test class balance: {np.bincount(test_ds.labels)}")

    torch.manual_seed(SEED)
    model = WiGaitApprox(N_SUB_TARGET, len(classes))
    result = train_classifier(model, train_ds, test_ds, epochs=EPOCHS, seed=SEED)

    anjali_idx = classes.index("anjali")
    model.eval()
    anjali_correct, anjali_total = 0, 0
    with torch.no_grad():
        for torso, limb, label in DataLoader(test_ds, batch_size=64):
            mask = label == anjali_idx
            if not mask.any():
                continue
            pred = model(torso, limb).argmax(dim=-1)
            anjali_correct += (pred[mask] == label[mask]).sum().item()
            anjali_total += mask.sum().item()
    anjali_recall = anjali_correct / anjali_total if anjali_total else float("nan")

    print(f"\n=== Wi-Gait approximation, cross-day (train {train_date} -> test {test_date}), walking-only ===")
    print(f"accuracy={result['accuracy']*100:.1f}% anjali_recall={anjali_recall*100:.1f}% "
          f"n_train={len(train_ds)} n_test={len(test_ds)}")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "wigait_approx_cross_day",
        "task": "taskB_identity", "preprocessing": "wigait_approx_zscore_zeropad", "model": "wigait_approx",
        "split_type": "day_disjoint_walking_only", "fold": 0, "accuracy": result["accuracy"],
        "n_train": len(train_ds), "n_test": len(test_ds),
        "notes": f"anjali_recall={anjali_recall:.4f}; train={train_date} test={test_date}; NOT a verified Wi-Gait replication (paywalled)",
    }])
    print("\nlogged to", REPO_ROOT / "ml/evaluation/results/experiment_log.csv")


if __name__ == "__main__":
    main()
