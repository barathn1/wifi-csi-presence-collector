"""Cross-day comparison: TF-Mamba approximation vs. the WhoFi replica, on IDENTICAL data/preprocessing/
normalization/split -- reuses every helper from run_paper_2507_12854_cross_day.py unchanged, only
swapping the model class.
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
from ml.training.run_paper_2507_12854_replication import build_person_windows, EPOCHS, SEED, ArrayWindowDataset
from ml.training.run_paper_2507_12854_cross_day import (
    day_window_size, compute_baseline, zscore, pad_to, N_SUB_DAY1,
)


def run_motion(motion: str, win1: int, win2: int, baseline1, baseline2) -> dict:
    manifest = load_manifest()
    train_date, test_date = "2026-09-09", "2026-09-10"

    day1_auth = manifest[(manifest["label"] == "authorized") & (manifest["motion"] == motion)
                          & manifest["session_dir"].str.contains(f"/{train_date}/", regex=False)]
    day2_auth = manifest[(manifest["label"] == "authorized") & (manifest["motion"] == motion)
                          & manifest["session_dir"].str.contains(f"/{test_date}/", regex=False)]

    print(f"\n=== TF-Mamba, motion={motion}: {train_date} win={win1} -> {test_date} win={win2} ===")
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

    train_ds = ArrayWindowDataset(np.concatenate(train_amp).astype(np.float32),
                                   np.concatenate(train_phase).astype(np.float32), np.concatenate(train_y))
    test_ds = ArrayWindowDataset(np.concatenate(test_amp).astype(np.float32),
                                  np.concatenate(test_phase).astype(np.float32), np.concatenate(test_y))
    print(f"  train={len(train_ds)}, test={len(test_ds)}, n_subcarriers={N_SUB_DAY1}")

    torch.manual_seed(SEED)
    model = TFMambaApprox(N_SUB_DAY1, len(classes))
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
    print(f"computing {train_date} baseline (win={win1})...")
    baseline1 = compute_baseline(sorted(day1_none["session_dir"]), win1)
    print(f"computing {test_date} baseline (win={win2})...")
    baseline2 = compute_baseline(sorted(day2_none["session_dir"]), win2)

    results = [run_motion(m, win1, win2, baseline1, baseline2) for m in ("standing", "walking")]

    print("\n=== TF-Mamba approximation, cross-day (vs WhoFi replica -- run_paper_2507_12854_cross_day.py) ===")
    print(f"{'motion':<12}{'accuracy':>10}{'anjali_recall':>16}")
    for r in results:
        print(f"{r['motion']:<12}{r['accuracy']*100:>9.1f}%{r['anjali_recall']*100:>15.1f}%")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "tfmamba_approx_cross_day",
        "task": "taskB_identity", "preprocessing": "paper_recipe_variantA_zeropad", "model": "tfmamba_approx",
        "split_type": "day_disjoint_per_motion", "fold": 0, "accuracy": r["accuracy"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": f"anjali_recall={r['anjali_recall']:.4f}; motion={r['motion']}; NOT a verified TF-Mamba replication",
    } for r in results])
    print("\nlogged to", REPO_ROOT / "ml/evaluation/results/experiment_log.csv")


if __name__ == "__main__":
    main()
