"""Same-day comparison: TF-Mamba approximation (models/tfmamba_approx.py) vs. this repo's WhoFi replica
(models/transformer_dualbranch.py), on IDENTICAL data/preprocessing/split -- reuses every helper from
run_paper_2507_12854_replication.py (the WhoFi-paper replication script) unchanged, only swapping the
model class, so the two numbers are a fair apples-to-apples comparison on this dataset.
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
    build_person_windows, day_window_size, temporal_split, ArrayWindowDataset, EPOCHS, SEED,
)


def run_day(day: str, motion_filter: str | None = None) -> dict:
    manifest = load_manifest()
    day_auth = manifest[(manifest["label"] == "authorized")
                         & (manifest["session_dir"].str.contains(f"/{day}/", regex=False))]
    if motion_filter is not None:
        day_auth = day_auth[day_auth["motion"] == motion_filter]
    win_samples = day_window_size(manifest, day)
    tag = motion_filter or "standing+walking"
    print(f"\n=== {day} ({tag}): TF-Mamba approx, window={win_samples} samples ===")

    per_person = {}
    for person in sorted(day_auth["person_id"].unique()):
        sessions = day_auth.loc[day_auth["person_id"] == person, "session_dir"].tolist()
        amp, phase = build_person_windows(sessions, win_samples)
        print(f"  {person}: {len(sessions)} sessions -> {len(amp)} windows")
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
    print(f"  train={len(train_ds)}, test={len(test_ds)}, n_subcarriers={n_sub}")

    torch.manual_seed(SEED)
    model = TFMambaApprox(n_sub, len(classes))
    result = train_classifier(model, train_ds, test_ds, epochs=EPOCHS, seed=SEED)

    anjali_idx = classes.index("anjali")
    model.eval()
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

    return {"day": day, "motion": tag, "accuracy": result["accuracy"], "anjali_recall": anjali_recall,
            "n_train": len(train_ds), "n_test": len(test_ds)}


def main() -> None:
    days = ["2026-09-09", "2026-09-10"]
    results = [run_day(day) for day in days]
    results += [run_day(day, motion_filter=m) for day in days for m in ("standing", "walking")]

    print("\n=== TF-Mamba approximation, same-day (vs WhoFi replica -- see run_paper_2507_12854_replication.py) ===")
    print(f"{'day':<12}{'motion':<18}{'accuracy':>10}{'anjali_recall':>16}")
    for r in results:
        print(f"{r['day']:<12}{r['motion']:<18}{r['accuracy']*100:>9.1f}%{r['anjali_recall']*100:>15.1f}%")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "tfmamba_approx_same_day",
        "task": "taskB_identity", "preprocessing": "paper_hampel_butterworth_phasecal", "model": "tfmamba_approx",
        "split_type": "temporal_70_10_20_per_person", "fold": 0, "accuracy": r["accuracy"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": f"anjali_recall={r['anjali_recall']:.4f}; day={r['day']}; motion={r['motion']}; NOT a verified TF-Mamba replication (IEEE Xplore inaccessible)",
    } for r in results])
    print("\nlogged to", REPO_ROOT / "ml/evaluation/results/experiment_log.csv")


if __name__ == "__main__":
    main()
