"""Day-to-day (same-day) and cross-day identity accuracy for taskB_identity (anjali vs barath) on
a single-branch, AMPLITUDE-ONLY BiLSTM (`ml/models/lstm.py::BiLSTMAmplitudeOnly`), fed
[window_packets x 128] amplitude matrices from the consistent channel-6-only preprocessing in
`ml/data_pipeline/bilstm_ch6_pipeline.py` (validate packet metadata -> keep one consistent CSI
packet config -> parse IQ -> amplitude -> fixed 128 subcarriers -> Hampel outlier removal ->
Butterworth smoothing -> per-window z-score normalization -> common time-normalized rate -> fixed
200-packet windows). Channel-11 session on 2026-09-15 is excluded by construction. Phase is still
cached by the pipeline (a side effect of the denoise step) but never read here -- this model has no
phase branch at all.

Uses a local amplitude-only train/eval loop (`train_amp_only`) instead of
`ml.training.train.train_classifier`, since that shared trainer's `model(amp, phase)` call
signature assumes every model in this repo's zoo takes both branches -- true for everything else
in `ml/models/`, not for this one.

Reports three things, not just one number, per this project's "evaluate honestly" convention:
1. Same-day: session-disjoint k-fold WITHIN each date (upper bound -- same router/room/day).
2. Cross-day (pairwise): train on one date's full data, test on another's (the real question).
3. Leave-one-day-out: train on the other two dates pooled, test on the held-out one (uses more
   training data than a single pairwise run, closer to how this would actually be deployed).

Every split result is averaged over `--seeds` random seeds (default 3) -- this project's own
trainer docstring (ml/training/train.py) found seed variance on data this size can be as large as
the effect being measured, so a single-seed number is not trustworthy on its own.

    python3 -m ml.training.run_bilstm_ch6_day_to_day
    python3 -m ml.training.run_bilstm_ch6_day_to_day --epochs 10 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import itertools
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data_pipeline.bilstm_ch6_pipeline import (
    TARGET_DATES,
    build_ch6_manifest,
    build_or_load_window_index,
    compute_dataset_target_rate_hz,
    window_zscore,
)
from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import assert_no_group_leakage, session_disjoint_kfold
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.models.lstm import BiLSTMAmplitudeOnly
from ml.training.train import log_rows

CLASSES = ["anjali", "barath"]
LOG_PATH = REPO_ROOT / "ml/evaluation/results/bilstm_ch6_day_to_day_log.csv"


def make_dataset(window_index: pd.DataFrame) -> CsiWindowDataset:
    return CsiWindowDataset(window_index, "taskB_identity", calibration=window_zscore, classes=CLASSES)


def train_amp_only(model: nn.Module, train_ds: CsiWindowDataset, test_ds: CsiWindowDataset,
                    epochs: int, seed: int, batch_size: int = 64, lr: float = 1e-3) -> float:
    """Same training procedure as ml.training.train.train_classifier, but calls model(amp) only --
    phase is loaded by CsiWindowDataset (every window's cache has it) and simply discarded here."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    for _ in range(epochs):
        model.train()
        for amp, _phase, label in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(amp), label)
            loss.backward()
            opt.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for amp, _phase, label in test_loader:
            pred = model(amp).argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += len(label)
    return correct / max(total, 1)


def run_once(train_ds: CsiWindowDataset, test_ds: CsiWindowDataset, epochs: int, seed: int) -> float:
    model = BiLSTMAmplitudeOnly(n_subcarriers=128, n_classes=len(CLASSES))
    return train_amp_only(model, train_ds, test_ds, epochs=epochs, seed=seed)


def multi_seed_accuracy(train_ds: CsiWindowDataset, test_ds: CsiWindowDataset, epochs: int,
                         seeds: list[int]) -> tuple[float, float, list[float]]:
    accs = [run_once(train_ds, test_ds, epochs, seed) for seed in seeds]
    return float(np.mean(accs)), float(np.std(accs)), accs


def same_day(window_index: pd.DataFrame, epochs: int, seeds: list[int], n_splits: int) -> list[dict]:
    print("\n=== SAME-DAY (session-disjoint k-fold within each date) ===")
    rows = []
    for date in TARGET_DATES:
        day_index = window_index[window_index["date"] == date].reset_index(drop=True)
        if day_index.empty:
            continue
        full_ds = make_dataset(day_index)
        n_sessions = full_ds.index["session_dir"].nunique()
        splits = min(n_splits, n_sessions)
        fold_accs = []
        for fold, (train_idx, test_idx) in enumerate(session_disjoint_kfold(full_ds.index, n_splits=splits)):
            assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")
            train_ds = full_ds.subset_by_index_rows(train_idx)
            test_ds = full_ds.subset_by_index_rows(test_idx)
            mean_acc, std_acc, accs = multi_seed_accuracy(train_ds, test_ds, epochs, seeds)
            fold_accs.append(mean_acc)
            print(f"  {date} fold {fold}: acc={mean_acc*100:.1f}% (+/-{std_acc*100:.1f}, seeds={accs}) "
                  f"n_train={len(train_ds)} n_test={len(test_ds)}")
            rows.append({"kind": "same_day", "train": date, "test": date, "fold": fold,
                         "accuracy_mean": mean_acc, "accuracy_std": std_acc,
                         "n_train": len(train_ds), "n_test": len(test_ds)})
        print(f"  {date} MEAN across {len(fold_accs)} folds: {np.mean(fold_accs)*100:.1f}%")
    return rows


def cross_day_pairwise(window_index: pd.DataFrame, epochs: int, seeds: list[int]) -> list[dict]:
    print("\n=== CROSS-DAY (pairwise: train one date, test another) ===")
    rows = []
    for train_date, test_date in itertools.permutations(TARGET_DATES, 2):
        train_index = window_index[window_index["date"] == train_date].reset_index(drop=True)
        test_index = window_index[window_index["date"] == test_date].reset_index(drop=True)
        if train_index.empty or test_index.empty:
            continue
        train_ds = make_dataset(train_index)
        test_ds = make_dataset(test_index)
        mean_acc, std_acc, accs = multi_seed_accuracy(train_ds, test_ds, epochs, seeds)
        print(f"  train={train_date} test={test_date}: acc={mean_acc*100:.1f}% (+/-{std_acc*100:.1f}, "
              f"seeds={accs}) n_train={len(train_ds)} n_test={len(test_ds)}")
        rows.append({"kind": "cross_day_pairwise", "train": train_date, "test": test_date, "fold": 0,
                     "accuracy_mean": mean_acc, "accuracy_std": std_acc,
                     "n_train": len(train_ds), "n_test": len(test_ds)})
    return rows


def leave_one_day_out(window_index: pd.DataFrame, epochs: int, seeds: list[int]) -> list[dict]:
    print("\n=== LEAVE-ONE-DAY-OUT (train on the other two dates pooled, test on the held-out one) ===")
    rows = []
    for held_out in TARGET_DATES:
        train_dates = [d for d in TARGET_DATES if d != held_out]
        train_index = window_index[window_index["date"].isin(train_dates)].reset_index(drop=True)
        test_index = window_index[window_index["date"] == held_out].reset_index(drop=True)
        train_ds = make_dataset(train_index)
        test_ds = make_dataset(test_index)
        mean_acc, std_acc, accs = multi_seed_accuracy(train_ds, test_ds, epochs, seeds)
        print(f"  train={train_dates} test={held_out}: acc={mean_acc*100:.1f}% (+/-{std_acc*100:.1f}, "
              f"seeds={accs}) n_train={len(train_ds)} n_test={len(test_ds)}")
        rows.append({"kind": "leave_one_day_out", "train": "+".join(train_dates), "test": held_out, "fold": 0,
                     "accuracy_mean": mean_acc, "accuracy_std": std_acc,
                     "n_train": len(train_ds), "n_test": len(test_ds)})
    return rows


def main(epochs: int, seeds: list[int], n_splits: int) -> None:
    manifest = build_ch6_manifest()
    print(f"channel-6 sessions across {TARGET_DATES}: {len(manifest)}")
    print(manifest.groupby(["date", "person_id"]).size())

    target_rate_hz = compute_dataset_target_rate_hz(manifest)
    cache_csv = REPO_ROOT / (
        f"ml/data_pipeline/cache/window_index_bilstm_ch6_{target_rate_hz:.2f}hz_n{len(manifest)}.csv"
    )
    window_index = build_or_load_window_index(cache_csv, manifest, target_rate_hz)
    print(f"total windows: {len(window_index)}")

    all_rows = []
    all_rows += same_day(window_index, epochs, seeds, n_splits)
    all_rows += cross_day_pairwise(window_index, epochs, seeds)
    all_rows += leave_one_day_out(window_index, epochs, seeds)

    print("\n=== SUMMARY ===")
    print(f"{'kind':<20}{'train':<24}{'test':<14}{'accuracy':>10}")
    summary_rows = []
    for kind in ("same_day", "cross_day_pairwise", "leave_one_day_out"):
        sub = [r for r in all_rows if r["kind"] == kind]
        if kind == "same_day":
            for date in TARGET_DATES:
                date_rows = [r for r in sub if r["train"] == date]
                if not date_rows:
                    continue
                mean_acc = np.mean([r["accuracy_mean"] for r in date_rows])
                summary_rows.append((kind, date, date, mean_acc))
        else:
            for r in sub:
                summary_rows.append((kind, r["train"], r["test"], r["accuracy_mean"]))
    for kind, train, test, acc in summary_rows:
        print(f"{kind:<20}{train:<24}{test:<14}{acc*100:>9.1f}%")

    log_rows_out = [{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "bilstm_ch6", "task": "taskB_identity",
        "preprocessing": "metadatavalidation+hampel+butterworth+timenorm+windowzscore", "model": "bilstm_amplitude_only",
        "split_type": r["kind"], "fold": r.get("fold", 0), "accuracy": r["accuracy_mean"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": f"train={r['train']} test={r['test']} std={r['accuracy_std']:.4f} epochs={epochs} seeds={seeds}",
    } for r in all_rows]
    log_rows(log_rows_out)
    print(f"\nlogged {len(log_rows_out)} rows to {REPO_ROOT / 'ml/evaluation/results/experiment_log.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()
    main(epochs=args.epochs, seeds=args.seeds, n_splits=args.n_splits)
