"""Leave-One-Day-Out cross-day identity accuracy for the walking-only BiLSTM (anjali vs barath),
2026-09-15/16/17, channel 6. Pipeline: `ml/data_pipeline/walk_bilstm_pipeline.py` (metadata-validated
decode -> phase sanitize -> per-subcarrier Hampel -> Butterworth 20Hz lowpass -> time-based 4s/50%-
overlap windows interpolated to 400 points) -> per-fold top-30-variance subcarrier selection + per-
window z-score (fit on the two TRAINING days only, frozen and applied unchanged to the held-out day,
so no cross-day statistic ever leaks into the fold it's evaluated on) -> `BiLSTMTriplet`
(stacked BiLSTM -> 32-d embedding -> softmax head), trained with combined cross-entropy +
semi-hard triplet loss on the embedding.

3 folds (train on the other two dates pooled, test on the held-out one); reported accuracy is the
average of all 3, not any single train/test pair.

    python -m ml.training.run_walk_bilstm_loo_day
    python -m ml.training.run_walk_bilstm_loo_day --epochs 15 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.walk_bilstm_pipeline import (
    TARGET_DATES,
    apply_topk,
    build_dataset,
    build_walk_manifest,
    per_window_zscore,
    select_topk_variance,
)
from ml.models.bilstm_triplet import BiLSTMTriplet
from ml.training.losses import triplet_semihard_loss
from ml.training.train import log_rows

CLASSES = ["anjali", "barath"]
NOISE_STD = 0.01
TIME_SHIFT_MAX = 20         # +/- timesteps (of the 400-point interpolated grid), not raw packets
SUBCARRIER_DROPOUT_P = 0.10
TRIPLET_WEIGHT = 1.0
CHECKPOINT_DIR = REPO_ROOT / "ml/checkpoints"


class WalkWindowDataset(Dataset):
    """In-memory dataset over already-reduced+normalized [N, 400, 60] windows. `augment=True` applies
    the spec's three single-antenna augmentations (Gaussian noise, random time shift, subcarrier
    dropout) fresh on every __getitem__ call -- training data only, never the test fold."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray, augment: bool):
        self.windows = torch.from_numpy(windows.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.windows[idx].clone()
        if self.augment:
            x = x + torch.randn_like(x) * NOISE_STD
            shift = int(torch.randint(-TIME_SHIFT_MAX, TIME_SHIFT_MAX + 1, (1,)))
            if shift:
                x = torch.roll(x, shifts=shift, dims=0)
            n_features = x.shape[1]
            n_drop = max(1, int(round(SUBCARRIER_DROPOUT_P * n_features)))
            drop_idx = torch.randperm(n_features)[:n_drop]
            x[:, drop_idx] = 0.0
        return x, self.labels[idx]


def prepare_fold(windows_all: np.ndarray, meta, held_out_date: str, k: int = 30):
    train_mask = (meta["date"] != held_out_date).values
    test_mask = (meta["date"] == held_out_date).values
    train_raw, test_raw = windows_all[train_mask], windows_all[test_mask]

    idx = select_topk_variance(train_raw, k=k)            # fit on TRAIN days only
    train_reduced = per_window_zscore(apply_topk(train_raw, idx))
    test_reduced = per_window_zscore(apply_topk(test_raw, idx))  # frozen idx, applied to held-out day

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    train_labels = meta.loc[train_mask, "person_id"].map(class_to_idx).values
    test_labels = meta.loc[test_mask, "person_id"].map(class_to_idx).values
    return train_reduced, train_labels, test_reduced, test_labels


def _accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            _, logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / max(total, 1)


def train_one_fold(train_reduced, train_labels, test_reduced, test_labels, epochs: int, seed: int,
                    batch_size: int = 32, lr: float = 1e-3, verbose: bool = False
                    ) -> tuple[float, float, nn.Module]:
    """Returns (train_accuracy, test_accuracy, trained_model) -- train accuracy (on a no-augmentation
    pass over the training windows) is reported alongside test accuracy so a low test score can be
    told apart from "didn't learn the training data at all" (an optimization problem) vs. "learned
    training data but doesn't generalize across days" (the actual cross-day question this fold is
    meant to answer). The trained model is returned (not just its accuracy) so
    `run_walk_bilstm_openset.py` can reuse the exact same fold's model/embeddings for open-set
    false-accept evaluation against unauthorized people, instead of retraining."""
    torch.manual_seed(seed)
    n_features = train_reduced.shape[2]
    model = BiLSTMTriplet(n_features=n_features, num_persons=len(CLASSES))

    train_ds = WalkWindowDataset(train_reduced, train_labels, augment=True)
    train_eval_ds = WalkWindowDataset(train_reduced, train_labels, augment=False)
    test_ds = WalkWindowDataset(test_reduced, test_labels, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            embedding, logits = model(x)
            loss = ce_loss(logits, y)
            if len(torch.unique(y)) > 1:  # triplet loss needs at least 2 classes present in the batch
                loss = loss + TRIPLET_WEIGHT * triplet_semihard_loss(embedding, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
            train_acc = _accuracy(model, train_eval_loader)
            print(f"    epoch {epoch+1}/{epochs}: loss={total_loss/n_batches:.4f} train_acc={train_acc*100:.1f}%")

    train_acc = _accuracy(model, train_eval_loader)
    test_acc = _accuracy(model, test_loader)
    return train_acc, test_acc, model


def leave_one_day_out(windows_all: np.ndarray, meta, epochs: int, seeds: list[int], verbose: bool) -> list[dict]:
    print("\n=== LEAVE-ONE-DAY-OUT (walking, channel 6, top-30-variance BiLSTM+triplet) ===")
    rows = []
    for held_out in TARGET_DATES:
        train_reduced, train_labels, test_reduced, test_labels = prepare_fold(windows_all, meta, held_out)
        train_dates = [d for d in TARGET_DATES if d != held_out]
        print(f"  train={train_dates} test={held_out} (n_train={len(train_labels)} n_test={len(test_labels)})")
        results = [train_one_fold(train_reduced, train_labels, test_reduced, test_labels, epochs, seed, verbose=verbose)
                   for seed in seeds]
        train_accs, test_accs, _models = zip(*results)
        mean_train, mean_test, std_test = float(np.mean(train_accs)), float(np.mean(test_accs)), float(np.std(test_accs))
        print(f"    train_acc={mean_train*100:.1f}% test_acc={mean_test*100:.1f}% (+/-{std_test*100:.1f}, "
              f"seeds test={list(test_accs)})")
        rows.append({"kind": "leave_one_day_out", "train": "+".join(train_dates), "test": held_out,
                     "accuracy_mean": mean_test, "accuracy_std": std_test, "train_accuracy_mean": mean_train,
                     "n_train": len(train_labels), "n_test": len(test_labels)})
    return rows


def main(epochs: int, seeds: list[int], verbose: bool) -> None:
    manifest = build_walk_manifest()
    print(f"walking channel-6 sessions across {TARGET_DATES}: {len(manifest)}")
    print(manifest.groupby(["date", "person_id"]).size())

    windows_all, meta = build_dataset(manifest)
    print(f"windows_all shape: {windows_all.shape}")

    rows = leave_one_day_out(windows_all, meta, epochs, seeds, verbose)

    fold_accs = [r["accuracy_mean"] for r in rows]
    fold_train_accs = [r["train_accuracy_mean"] for r in rows]
    print(f"\n=== SUMMARY: mean train_acc={np.mean(fold_train_accs)*100:.1f}% mean test_acc across "
          f"{len(fold_accs)} folds = {np.mean(fold_accs)*100:.1f}% "
          f"(per-fold test: {[f'{a*100:.1f}%' for a in fold_accs]}) ===")

    log_rows_out = [{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "walk_bilstm_triplet",
        "task": "taskB_identity_walking_only",
        "preprocessing": "hampel+butterworth20hz+timewindow4s+topk30variance+windowzscore",
        "model": "bilstm_triplet", "split_type": r["kind"], "fold": 0, "accuracy": r["accuracy_mean"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": f"train={r['train']} test={r['test']} std={r['accuracy_std']:.4f} "
                 f"train_acc={r['train_accuracy_mean']:.4f} epochs={epochs} seeds={seeds}",
    } for r in rows]
    log_rows(log_rows_out)
    print(f"logged {len(log_rows_out)} rows to {REPO_ROOT / 'ml/evaluation/results/experiment_log.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--verbose", action="store_true", help="print per-epoch loss/train-accuracy")
    args = p.parse_args()
    main(epochs=args.epochs, seeds=args.seeds, verbose=args.verbose)
