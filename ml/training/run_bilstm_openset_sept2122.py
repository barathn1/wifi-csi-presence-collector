"""BiLSTM follow-up to `run_svm_openset_sept2122.py`: same 2026-09-21/22 channel-6, 3-receiver,
auth (anjali/barath) vs. non_auth (everyone else, including empty-room `none`) open-set question,
same Hampel+Butterworth+time-resampled preprocessing and window index
(`ml.data_pipeline.sept2122_ch6_openset_pipeline`) -- only the model changes, from a linear SVM over
handcrafted per-window statistics to `ml.models.lstm.BiLSTMAmplitudeOnly` reading the raw amplitude
sequence directly (this project's existing channel-6 identity architecture, matching
`run_bilstm_ch6_day_to_day.py`'s single-branch amplitude-only setup, just re-targeted from
taskB_identity (anjali vs barath) to taskD_auth_vs_nonauth (auth vs everyone else) and from
2026-09-15/16/17 to 2026-09-21/22).

The SVM result (see `run_svm_openset_sept2122.py`'s docstring/log) was cross-day AUROC ~0.50-0.56 --
chance level -- with same-day performance itself inconsistent across the two dates (2026-09-22 alone:
AUROC up to 0.94; 2026-09-21 alone: chance). The question here is whether a sequence model that reads
the raw per-packet amplitude trajectory (gait-timing information the per-window mean/std/skew/
kurtosis features collapse away) does any better, not whether it can fix a day-specific confound if
that's what's actually driving the SVM's asymmetry -- same-day-per-date numbers below will show
whether that asymmetry persists here too.

Reports the same three things `run_bilstm_ch6_day_to_day.py` does, per this project's "evaluate
honestly" convention, each averaged over `--seeds` seeds (this project's own trainer docstring found
seed variance on data this size can be as large as the effect being measured):
1. Same-day: session-disjoint k-fold within each date.
2. Cross-day pairwise, both directions (the real, already-open-set-by-construction question, since
   the unauthorized identities on the 21st and 22nd don't overlap at all).

CPU-only, no GPU assumed (matches this project's existing training scripts) -- expect this to run for
a while (same-day 5-fold x2 dates + cross-day pairwise x2, each x `--seeds` seeds x `--epochs` epochs
over tens of thousands of 200-packet windows). Designed to be left running unattended.

    python -m ml.training.run_bilstm_openset_sept2122
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.sept2122_ch6_openset_pipeline import (
    TARGET_DATES,
    build_manifest,
    build_or_load_window_index,
    compute_dataset_target_rate_hz,
)
from ml.data_pipeline.splits import assert_no_group_leakage, stratified_session_disjoint_kfold
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.evaluation.metrics import compute_auroc, compute_eer
from ml.models.lstm import BiLSTMAmplitudeOnly

LOG_PATH = REPO_ROOT / "ml/evaluation/results/bilstm_openset_sept2122_log.csv"
LOG_FIELDS = [
    "timestamp", "epochs", "split_type", "train", "test", "fold", "seed", "accuracy", "auroc", "eer",
    "far_unauthorized", "far_none", "frr", "recall_anjali", "recall_barath", "n_train", "n_test",
]
EPS = 1e-6


def per_window_zscore(amp: np.ndarray, phase: np.ndarray, row) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each window's own amplitude (per subcarrier, over the window's time axis) -- removes
    per-window/per-session absolute-magnitude differences (distance/attenuation/AGC) so the model reads
    shape, not scale. Phase passed through untouched (this model has no phase branch)."""
    amp_n = (amp - amp.mean(axis=0)) / (amp.std(axis=0) + EPS)
    return amp_n.astype(np.float32), phase


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = build_manifest()
    print(f"channel-6 (session,receiver) rows across {TARGET_DATES}: {len(manifest)}")
    target_rate_hz = compute_dataset_target_rate_hz(manifest)
    window_index = build_or_load_window_index(manifest, target_rate_hz)
    print(f"total windows: {len(window_index)}")
    print(window_index.groupby(["date", "label"]).size())
    return manifest, window_index


def make_ds(window_index: pd.DataFrame) -> CsiWindowDataset:
    return CsiWindowDataset(window_index, "taskD_auth_vs_nonauth", calibration=per_window_zscore, classes=[0, 1])


def train_and_eval(train_ds: CsiWindowDataset, test_ds: CsiWindowDataset, epochs: int, seed: int,
                    batch_size: int = 64, lr: float = 1e-3, verbose: bool = False) -> dict:
    torch.manual_seed(seed)
    model = BiLSTMAmplitudeOnly(n_subcarriers=128, n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    for epoch in range(epochs):
        model.train()
        epoch_loss, grad_norm_sum, n_batches = 0.0, 0.0, 0
        for amp, _phase, label in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(amp), label)
            loss.backward()
            if verbose:
                grad_norm_sum += sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if verbose:
            print(f"    epoch {epoch}: mean_loss={epoch_loss/n_batches:.4f} mean_grad_norm={grad_norm_sum/n_batches:.4f}")

    model.eval()
    probs_list, true_list = [], []
    with torch.no_grad():
        for amp, _phase, label in test_loader:
            probs_list.append(torch.softmax(model(amp), dim=-1).numpy())
            true_list.append(label.numpy())
    probs = np.concatenate(probs_list)
    true = np.concatenate(true_list)

    scores = probs[:, 1]  # P(auth)
    pred = probs.argmax(axis=-1)
    accuracy = float((pred == true).mean())
    auroc = compute_auroc(true, scores)
    eer, _ = compute_eer(true, scores)

    raw_label = test_ds.index["label"].values
    mask_unauth = raw_label == "unauthorized"
    far_unauthorized = float(pred[mask_unauth].mean()) if mask_unauth.any() else float("nan")
    mask_none = raw_label == "none"
    far_none = float(pred[mask_none].mean()) if mask_none.any() else float("nan")
    mask_auth = true == 1
    frr = float((pred[mask_auth] == 0).mean()) if mask_auth.any() else float("nan")

    def _recall(person: str) -> float:
        mask = test_ds.index["person_id"].values == person
        return float((pred[mask] == 1).mean()) if mask.any() else float("nan")

    return {
        "accuracy": accuracy, "auroc": auroc, "eer": eer, "far_unauthorized": far_unauthorized,
        "far_none": far_none, "frr": frr, "recall_anjali": _recall("anjali"), "recall_barath": _recall("barath"),
        "n_train": len(train_ds), "n_test": len(test_ds),
    }


def _print_and_row(m: dict, seed: int, split_type: str, train: str, test: str, fold: int, epochs: int) -> dict:
    print(f"    seed={seed} acc={m['accuracy']*100:.1f}% auroc={m['auroc']:.3f} eer={m['eer']:.3f} "
          f"far_unauth={m['far_unauthorized']*100:.1f}% far_none={m['far_none']*100:.1f}% frr={m['frr']*100:.1f}% "
          f"n_train={m['n_train']} n_test={m['n_test']}")
    return {**m, "seed": seed, "split_type": split_type, "train": train, "test": test, "fold": fold, "epochs": epochs}


def same_day(window_index: pd.DataFrame, epochs: int, seeds: list[int], n_splits: int) -> list[dict]:
    print("\n=== SAME-DAY (session-disjoint k-fold within each date) ===")
    rows = []
    for date in TARGET_DATES:
        day_index = window_index[window_index["date"] == date].reset_index(drop=True)
        if day_index.empty:
            continue
        full_ds = make_ds(day_index)
        n_sessions = full_ds.index["session_dir"].nunique()
        splits = min(n_splits, n_sessions)
        fold_accs = []
        for fold, (train_idx, test_idx) in enumerate(stratified_session_disjoint_kfold(full_ds.index, n_splits=splits)):
            assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")
            train_ds = full_ds.subset_by_index_rows(train_idx)
            test_ds = full_ds.subset_by_index_rows(test_idx)
            print(f"  {date} fold {fold} (n_train={len(train_ds)} n_test={len(test_ds)}):")
            fold_rows = [_print_and_row(train_and_eval(train_ds, test_ds, epochs, seed), seed, "same_day", date, date, fold, epochs)
                         for seed in seeds]
            log_rows(fold_rows)  # flush after every fold -- a long unattended run can be killed mid-way
            rows += fold_rows
            fold_accs.append(np.mean([r["accuracy"] for r in fold_rows]))
        print(f"  {date} MEAN accuracy across {len(fold_accs)} folds: {np.mean(fold_accs)*100:.1f}%")
    return rows


def cross_day_pairwise(window_index: pd.DataFrame, epochs: int, seeds: list[int]) -> list[dict]:
    print("\n=== CROSS-DAY (pairwise: train one date, test the other) ===")
    rows = []
    for train_date, test_date in [(TARGET_DATES[0], TARGET_DATES[1]), (TARGET_DATES[1], TARGET_DATES[0])]:
        train_index = window_index[window_index["date"] == train_date].reset_index(drop=True)
        test_index = window_index[window_index["date"] == test_date].reset_index(drop=True)
        if train_index.empty or test_index.empty:
            continue
        train_ds = make_ds(train_index)
        test_ds = make_ds(test_index)
        print(f"  train={train_date} test={test_date} (n_train={len(train_ds)} n_test={len(test_ds)}):")
        pair_rows = []
        for seed in seeds:
            m = train_and_eval(train_ds, test_ds, epochs, seed)
            pair_rows.append(_print_and_row(m, seed, "cross_day_pairwise", train_date, test_date, 0, epochs))
        log_rows(pair_rows)  # flush after every train/test pair -- see same_day's comment
        rows += pair_rows
    return rows


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = [{**{k: r.get(k, "") for k in LOG_FIELDS}, "timestamp": datetime.now(timezone.utc).isoformat()}
           for r in rows]
    df = pd.DataFrame(out, columns=LOG_FIELDS)
    df.to_csv(LOG_PATH, mode="a" if LOG_PATH.exists() else "w", header=not LOG_PATH.exists(), index=False)


def main(epochs: int, seeds: list[int], n_splits: int) -> None:
    _, window_index = build_dataset()

    all_rows = []
    all_rows += same_day(window_index, epochs, seeds, n_splits)
    all_rows += cross_day_pairwise(window_index, epochs, seeds)
    # each fold/pair is already flushed to LOG_PATH as it completes (see same_day/cross_day_pairwise) --
    # not logged again here, so a killed run leaves only the fold/pair in progress unrecorded.

    print("\n=== SUMMARY (mean over seeds) ===")
    df = pd.DataFrame(all_rows)
    print(f"{'split':<20}{'train':<14}{'test':<14}{'fold':>5}{'acc':>8}{'auroc':>8}{'eer':>8}"
          f"{'far_unauth':>12}{'far_none':>10}")
    group_cols = ["split_type", "train", "test", "fold"]
    for keys, g in df.groupby(group_cols, sort=False):
        split_type, train, test, fold = keys
        print(f"{split_type:<20}{train:<14}{test:<14}{fold:>5}{g['accuracy'].mean()*100:>7.1f}%"
              f"{g['auroc'].mean():>8.3f}{g['eer'].mean():>8.3f}"
              f"{g['far_unauthorized'].mean()*100:>11.1f}%{g['far_none'].mean()*100:>9.1f}%")
    print(f"\nlogged {len(all_rows)} rows to {LOG_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()
    main(epochs=args.epochs, seeds=args.seeds, n_splits=args.n_splits)
