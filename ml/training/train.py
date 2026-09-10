"""Generic supervised trainer for the transformer/CNN/LSTM model zoo (everything with a
`forward(amplitude, phase) -> logits` interface). CPU-only, so this uses a single session-disjoint
80/20 split rather than full 5-fold CV for the deep models -- the classical RandomForest baseline
(run_stage1.py) already gives the full-CV honest number cheaply; the deep-model sweep here prioritizes
covering every architecture within a reasonable wall-clock budget over exhaustive CV on each one.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import assert_no_group_leakage, session_disjoint_kfold
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.data_pipeline.windowing import load_manifest

LOG_PATH = REPO_ROOT / "ml/evaluation/results/experiment_log.csv"
DEVICE = torch.device("cpu")


def train_classifier(model: nn.Module, train_ds: CsiWindowDataset, test_ds: CsiWindowDataset,
                      epochs: int = 6, batch_size: int = 64, lr: float = 1e-3, seed: int = 0) -> dict:
    """seed fixes weight init + minibatch shuffling -- added after discovering that re-training the
    SAME model on the SAME fold gave wildly different results (taskD fold 1: AUROC 0.757 vs 0.698,
    unauthorized-false-accept 26% vs 57%, across two otherwise-identical runs). On a dataset this small
    with only a handful of epochs, seed variance can be as large as the effect being measured -- so
    every architecture comparison in this codebase needs to control for it, not just the train/test
    split. Comparing architectures still requires averaging over several seeds, not just fixing one."""
    torch.manual_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for amp, phase, label in train_loader:
            amp, phase, label = amp.to(DEVICE), phase.to(DEVICE), label.to(DEVICE)
            opt.zero_grad()
            logits = model(amp, phase)
            loss = loss_fn(logits, label)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        history.append(total_loss / max(n_batches, 1))

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for amp, phase, label in test_loader:
            amp, phase, label = amp.to(DEVICE), phase.to(DEVICE), label.to(DEVICE)
            pred = model(amp, phase).argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += len(label)

    return {"accuracy": correct / max(total, 1), "train_loss_curve": history,
            "n_train": len(train_ds), "n_test": len(test_ds)}


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    fieldnames = ["timestamp", "stage", "task", "preprocessing", "model", "split_type", "fold",
                  "accuracy", "eer", "auroc", "n_train", "n_test", "notes"]
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_model_on_task(model_factory, model_name: str, task_name: str, n_subcarriers: int = 186,
                       epochs: int = 6) -> dict:
    manifest = load_manifest()
    index_path = REPO_ROOT / "ml/data_pipeline/cache/window_index_w200_s100.csv"
    window_index = pd.read_csv(index_path)

    full_ds = CsiWindowDataset(window_index, task_name)
    n_classes = len(full_ds.classes)

    # single session-disjoint 80/20 split (see module docstring for why not full k-fold here)
    train_idx, test_idx = next(session_disjoint_kfold(full_ds.index, n_splits=5))
    assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")

    train_ds = full_ds.subset_by_index_rows(train_idx)
    test_ds = full_ds.subset_by_index_rows(test_idx)

    model = model_factory(n_subcarriers, n_classes)
    result = train_classifier(model, train_ds, test_ds, epochs=epochs)
    result.update({"task": task_name, "model": model_name, "n_classes": n_classes, "classes": full_ds.classes})
    return result


if __name__ == "__main__":
    import argparse

    from ml.models.transformer_crossattn import CrossAttentionTransformer
    from ml.models.transformer_dualbranch import DualBranchTransformer

    p = argparse.ArgumentParser()
    p.add_argument("--task", default="task0_presence")
    p.add_argument("--model", default="dualbranch", choices=["dualbranch", "crossattn"])
    p.add_argument("--epochs", type=int, default=6)
    args = p.parse_args()

    factories = {
        "dualbranch": lambda n_sub, n_cls: DualBranchTransformer(n_sub, n_cls),
        "crossattn": lambda n_sub, n_cls: CrossAttentionTransformer(n_sub, n_cls),
    }
    result = run_model_on_task(factories[args.model], args.model, args.task, epochs=args.epochs)
    print(f"{args.model} on {args.task}: accuracy={result['accuracy']:.4f} "
          f"(n_train={result['n_train']}, n_test={result['n_test']}, classes={result['classes']})")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": 2, "task": args.task,
        "preprocessing": "raw", "model": args.model, "split_type": "session_disjoint_single_split",
        "fold": 0, "accuracy": result["accuracy"], "n_train": result["n_train"], "n_test": result["n_test"],
    }])
