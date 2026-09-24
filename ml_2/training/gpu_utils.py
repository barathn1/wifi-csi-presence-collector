"""GPU-aware drop-in for ml.training.train.train_classifier (which hardcodes `torch.device("cpu")` --
a deliberate prior constraint per that module's own docstring, now lifted). Also used to retrain the
existing WhoFi/dualbranch/crossattn/receiver-fusion models on GPU without editing ml/ itself.

Every function here takes an optional `device` override (defaults to module-level DEVICE) so
`run_folds_parallel` can run several folds concurrently, each pinned to a different physical GPU --
these models are tiny (10k-40k params), so a single fold barely saturates one GPU; running 2 folds at
once on the 2 available L4s roughly halves wall-clock instead of leaving the second GPU idle.
"""
from __future__ import annotations

import concurrent.futures

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()
FOLD_DEVICES = [torch.device(f"cuda:{i}") for i in range(N_GPUS)] if N_GPUS > 0 else [DEVICE]
# leave headroom for normal machine use rather than claiming every core: DataLoader workers are
# capped per-fold so N_PARALLEL_FOLDS folds running at once (one per GPU) don't oversubscribe.
import os
_CPU_COUNT = os.cpu_count() or 4
WORKERS_PER_FOLD = max(1, min(4, (_CPU_COUNT - 4) // max(1, len(FOLD_DEVICES))))


def train_classifier_gpu(model: nn.Module, train_ds, test_ds, epochs: int = 6, batch_size: int = 64,
                          lr: float = 1e-3, seed: int = 0, sample_weights: np.ndarray | None = None,
                          device: torch.device | None = None) -> dict:
    """Same contract as ml.training.train.train_classifier (model.forward(amp, phase) -> logits),
    just running on `device` (default DEVICE) instead of a hardcoded CPU."""
    device = device or DEVICE
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    if sample_weights is not None:
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double),
                                         num_samples=len(train_ds), replacement=True,
                                         generator=torch.Generator().manual_seed(seed))
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                   num_workers=WORKERS_PER_FOLD, pin_memory=True, multiprocessing_context="fork")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=WORKERS_PER_FOLD, pin_memory=True, multiprocessing_context="fork")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=WORKERS_PER_FOLD, pin_memory=True, multiprocessing_context="fork")

    history = []
    for _ in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for amp, phase, label in train_loader:
            amp, phase, label = amp.to(device), phase.to(device), label.to(device)
            opt.zero_grad()
            loss = loss_fn(model(amp, phase), label)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        history.append(total_loss / max(n_batches, 1))

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for amp, phase, label in test_loader:
            amp, phase, label = amp.to(device), phase.to(device), label.to(device)
            pred = model(amp, phase).argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += len(label)
    return {"accuracy": correct / max(total, 1), "train_loss_curve": history,
            "n_train": len(train_ds), "n_test": len(test_ds)}


@torch.no_grad()
def neural_logits_gpu(model: nn.Module, ds, device: torch.device | None = None) -> tuple[np.ndarray, np.ndarray]:
    device = device or DEVICE
    model.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=WORKERS_PER_FOLD, pin_memory=True, multiprocessing_context="fork")
    all_logits, all_y = [], []
    for amp, phase, y in loader:
        logits = model(amp.to(device), phase.to(device))
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(np.asarray(y))
    return np.concatenate(all_logits), np.concatenate(all_y)


def binary_eval_gpu(model: nn.Module, test_ds, device: torch.device | None = None) -> dict:
    """Window-level pooled metrics PLUS session/day-level breakdown (see
    ml_2.data.metrics.session_level_metrics's docstring for why the pooled number alone is not
    trustworthy on this dataset -- overlapping windows within a session are highly correlated)."""
    from ml_2.data.metrics import compute_auroc, compute_eer, format_session_breakdown, session_level_metrics

    logits, y_true = neural_logits_gpu(model, test_ds, device=device)
    proba = torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    pred = logits.argmax(axis=-1)
    eer, _ = compute_eer(y_true, proba)
    auroc = compute_auroc(y_true, proba)
    out = {"accuracy": float((pred == y_true).mean()), "eer": eer, "auroc": auroc}

    session_dirs = test_ds.index["session_dir"].values
    dates = test_ds.index["date"].values if "date" in test_ds.index.columns else None
    sm = session_level_metrics(y_true, proba, pred, session_dirs, dates)
    out["session_accuracy"] = sm["session_accuracy"]
    out["session_auroc"] = sm["session_auroc"]
    out["n_sessions"] = sm["n_sessions"]
    out["_session_breakdown_str"] = format_session_breakdown(sm)
    out["_per_session_table"] = sm["per_session_table"]
    original_labels = test_ds.index["label"].values
    for neg in ("unauthorized", "none"):
        m = original_labels == neg
        if m.sum() > 0:
            out[f"false_accept_{neg}"] = float((pred[m] == 1).mean())
    return out


def run_folds_parallel(fold_items: list, fold_fn) -> list:
    """Runs `fold_fn(item, device)` for every item in `fold_items`, distributed round-robin across
    FOLD_DEVICES (all visible GPUs, or [DEVICE] if 0-1 GPUs) using a thread pool sized to the number
    of devices. Threads, not processes: each fold's Python-level loop is thin (a handful of tensor
    ops per batch), and PyTorch releases the GIL during the actual CUDA kernel launches/compute, so
    N threads pinned to N distinct devices run genuinely concurrently without the CUDA-context/
    pickling complications multiprocessing would add. Falls back to plain sequential execution when
    there's only one device (no thread-pool overhead for nothing)."""
    if len(FOLD_DEVICES) <= 1:
        return [fold_fn(item, FOLD_DEVICES[0]) for item in fold_items]
    results = [None] * len(fold_items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(FOLD_DEVICES)) as pool:
        futures = {pool.submit(fold_fn, item, FOLD_DEVICES[i % len(FOLD_DEVICES)]): i
                   for i, item in enumerate(fold_items)}
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    return results
