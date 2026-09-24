"""Contrastive self-supervised pretraining (ml_2/models/contrastive_pretrain.py) on EVERY window,
labels ignored, then a frozen-backbone linear probe trained per fold on just that fold's labeled data.
The point: attack label scarcity by learning a general embedding from cheap unlabeled data first.

    python3 -m ml_2.training.train_contrastive --pretrain-epochs 10 --probe-epochs 10
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_2.data.decode import REPO_ROOT
from ml_2.data.splits import assert_no_group_leakage, leave_one_day_out, leave_one_unauthorized_person_out
from ml_2.data.torch_dataset import CsiWindowDataset
from ml_2.data.metrics import compute_auroc, compute_eer, save_session_breakdown_csv
from ml_2.models.backbone import DualBranchEmbedding
from ml_2.models.contrastive_pretrain import LinearProbe, contrastive_pretrain_epoch
from ml_2.training.common_data import Channel6Dataset, N_SUBCARRIERS, load_channel6_dataset
from ml_2.training.gpu_utils import DEVICE

LOG_PATH = REPO_ROOT / "ml_2/evaluation/results/contrastive_log.csv"
SESSION_CSV_PATH = REPO_ROOT / "ml_2/evaluation/results/session_breakdown.csv"
LOG_FIELDNAMES = ["timestamp", "split_type", "held_out", "accuracy", "eer", "auroc",
                   "session_accuracy", "session_auroc", "n_sessions",
                   "false_accept_unauthorized", "false_accept_none", "n_train", "n_test", "notes"]


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def pretrain_backbone(full_ds, epochs: int, seed: int, batch_size: int = 64, lr: float = 1e-3) -> DualBranchEmbedding:
    torch.manual_seed(seed)
    backbone = DualBranchEmbedding(n_subcarriers=N_SUBCARRIERS).to(DEVICE)
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, multiprocessing_context="fork")
    for epoch in range(epochs):
        loss = contrastive_pretrain_epoch(backbone, loader, opt, DEVICE)
        print(f"  pretrain epoch {epoch}: nt_xent_loss={loss:.4f}")
    return backbone


@torch.no_grad()
def embed_all(backbone, ds, batch_size: int = 64) -> tuple[torch.Tensor, np.ndarray]:
    backbone.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, multiprocessing_context="fork")
    embs, labels = [], []
    for amp, phase, label in loader:
        embs.append(backbone(amp.to(DEVICE), phase.to(DEVICE)).cpu())
        labels.append(np.asarray(label))
    return torch.cat(embs), np.concatenate(labels)


def train_probe(train_emb: torch.Tensor, train_labels: np.ndarray, n_classes: int, epochs: int, lr: float = 1e-2):
    probe = LinearProbe(embed_dim=train_emb.shape[1], n_classes=n_classes).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    X, y = train_emb.to(DEVICE), torch.as_tensor(train_labels, dtype=torch.long, device=DEVICE)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(probe(X), y)
        loss.backward()
        opt.step()
    return probe


def run_fold(backbone, full_ds, train_idx: np.ndarray, test_idx: np.ndarray, probe_epochs: int) -> dict:
    train_ds, test_ds = full_ds.subset_by_index_rows(train_idx), full_ds.subset_by_index_rows(test_idx)
    train_emb, train_labels = embed_all(backbone, train_ds)
    test_emb, _ = embed_all(backbone, test_ds)
    probe = train_probe(train_emb, train_labels, n_classes=len(full_ds.classes), epochs=probe_epochs)

    with torch.no_grad():
        logits = probe(test_emb.to(DEVICE)).cpu()
    proba = torch.softmax(logits, dim=-1)[:, 1].numpy()
    pred = logits.argmax(dim=-1).numpy()
    y_true = (full_ds.index.iloc[test_idx]["label"].values == "authorized").astype(int)
    eer, _ = compute_eer(y_true, proba)
    auroc = compute_auroc(y_true, proba)
    out = {"accuracy": float((pred == y_true).mean()), "eer": eer, "auroc": auroc,
           "n_train": len(train_idx), "n_test": len(test_idx)}
    test_rows = full_ds.index.iloc[test_idx]
    labels = test_rows["label"].values
    for neg in ("unauthorized", "none"):
        m = labels == neg
        if m.sum() > 0:
            out[f"false_accept_{neg}"] = float((pred[m] == 1).mean())

    from ml_2.data.metrics import format_session_breakdown, session_level_metrics
    sm = session_level_metrics(y_true, proba, pred, test_rows["session_dir"].values, test_rows["date"].values)
    out["session_accuracy"], out["session_auroc"], out["n_sessions"] = \
        sm["session_accuracy"], sm["session_auroc"], sm["n_sessions"]
    out["_session_breakdown_str"] = format_session_breakdown(sm)
    out["_per_session_table"] = sm["per_session_table"]
    return out


def main(pretrain_epochs: int, probe_epochs: int, seed: int) -> None:
    print(f"training on device: {DEVICE}")
    dataset: Channel6Dataset = load_channel6_dataset()
    full_ds = CsiWindowDataset(dataset.window_index, "auth_vs_nonauth", calibration=dataset.calibration)

    print(f"\n=== contrastive pretraining on all {len(full_ds)} windows (unlabeled) ===")
    backbone = pretrain_backbone(full_ds, pretrain_epochs, seed)

    print("\n=== contrastive+probe open-set: leave-one-unauthorized-person-out ===")
    fold_metrics = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(dataset.window_index, include_none=True):
        assert_no_group_leakage(dataset.window_index, train_idx, test_idx)
        m = run_fold(backbone, full_ds, train_idx, test_idx, probe_epochs)
        m.update({"split_type": "open_set_loo_stranger", "held_out": held_out})
        fold_metrics.append(m)
        print(f"  held out '{held_out}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f}")
        print(m.pop("_session_breakdown_str", ""))
        per_session_table = m.pop("_per_session_table", None)
        if per_session_table is not None:
            save_session_breakdown_csv(per_session_table, SESSION_CSV_PATH, "contrastive_pretrain_probe",
                                        "open_set_loo_stranger", held_out)
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "notes": "leave-one-unauthorized-person-out, frozen contrastive backbone + linear probe"}])
    print(f"  MEAN window_auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}  "
          f"MEAN session_auroc={np.nanmean([m['session_auroc'] for m in fold_metrics]):.3f}")

    print("\n=== contrastive+probe cross-day: leave-one-day-out ===")
    fold_metrics = []
    for held_out_date, train_idx, test_idx in leave_one_day_out(dataset.window_index):
        test_labels = dataset.window_index.iloc[test_idx]["label"]
        if (test_labels == "authorized").sum() == 0 or len(test_labels.unique()) < 2:
            print(f"  skipping {held_out_date}: degenerate class balance")
            continue
        m = run_fold(backbone, full_ds, train_idx, test_idx, probe_epochs)
        m.update({"split_type": "cross_day_loo", "held_out": held_out_date})
        fold_metrics.append(m)
        print(f"  held out day '{held_out_date}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f}")
        print(m.pop("_session_breakdown_str", ""))
        per_session_table = m.pop("_per_session_table", None)
        if per_session_table is not None:
            save_session_breakdown_csv(per_session_table, SESSION_CSV_PATH, "contrastive_pretrain_probe",
                                        "cross_day_loo", held_out_date)
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "notes": "leave-one-day-out, frozen contrastive backbone + linear probe"}])
    print(f"  MEAN window_auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}  "
          f"MEAN session_auroc={np.nanmean([m['session_auroc'] for m in fold_metrics]):.3f}")
    print(f"\nDone. Results -> {LOG_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--probe-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(pretrain_epochs=args.pretrain_epochs, probe_epochs=args.probe_epochs, seed=args.seed)
